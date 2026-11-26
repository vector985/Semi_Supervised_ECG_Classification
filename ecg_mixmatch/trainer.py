"""Iteration-based training with validation checkpoint selection."""

import json
import logging
from time import perf_counter

import numpy as np
import torch

from .data import repeat_batches
from .evaluation import evaluate, synchronize
from .mixmatch import augment_views, compute_losses, guess_labels, mix_batches

logger = logging.getLogger(__name__)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_scheduler(optimizer, args):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.iterations, eta_min=min(1e-7, args.learning_rate)
        )
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=20, factor=0.5,
            min_lr=min(1e-6, args.learning_rate),
        )
    if args.iterations < 3:
        return None
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, total_steps=args.iterations,
        pct_start=max(0.05, 2 / args.iterations),
    )


def train(model, labeled_loader, unlabeled_loader, val_loader, device, args, model_config):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(optimizer, args)
    rng = np.random.default_rng(args.seed)
    labeled_batches = repeat_batches(labeled_loader)
    unlabeled_batches = repeat_batches(unlabeled_loader)
    non_blocking = device.type == "cuda"
    best_score = float("-inf")
    stale = 0
    history = {"training": [], "validation": []}
    training_seconds = 0.0
    synchronize(device)
    started = perf_counter()

    for step in range(1, args.iterations + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        signals, labels = next(labeled_batches)
        (unlabeled,) = next(unlabeled_batches)
        signals = signals.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        views = augment_views(unlabeled.to(device, non_blocking=non_blocking), args.views)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            targets = guess_labels(model, views, args.temperature)
        # Keep cached inference casts out of the gradient-tracked forward.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            mixed, mixed_targets = mix_batches(
                signals, labels, views, targets, alpha=args.alpha, rng=rng
            )
            logits = model(mixed)
        supervised, unsupervised = compute_losses(logits, mixed_targets, len(labels))
        unsupervised_weight = args.lambda_u * min(step / max(args.warmup, 1), 1.0)
        loss = supervised + unsupervised_weight * unsupervised
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()
        if scheduler is not None and args.scheduler != "plateau":
            scheduler.step()

        should_validate = step % args.eval_interval == 0 or step == args.iterations
        if step == 1 or step % args.log_interval == 0 or should_validate:
            sup, unsup, total = torch.stack((supervised, unsupervised, loss)).detach().cpu().tolist()
            if not np.isfinite(total):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            history["training"].append({
                "step": step, "supervised_loss": sup, "unsupervised_loss": unsup,
                "loss": total, "learning_rate": learning_rate,
                "unsupervised_weight": unsupervised_weight,
            })
            logger.info("Step %d/%d, loss %.4f", step, args.iterations, total)

        if not should_validate:
            continue
        synchronize(device)
        training_seconds += perf_counter() - started
        validation = evaluate(model, val_loader, device, amp=args.amp)
        score = 0.7 * validation["macro_f1"] + 0.3 * validation["accuracy"]
        history["validation"].append({"step": step, "score": score, **validation})
        logger.info("Validation at step %d, macro-F1 %.4f, score %.4f",
                    step, validation["macro_f1"], score)
        if args.scheduler == "plateau":
            scheduler.step(validation["macro_f1"])
        if score > best_score:
            best_score, best_step, best_validation = score, step, validation
            stale = 0
            state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            temporary = args.output / "best_model.tmp"
            torch.save({"model_state": state, "model_config": model_config}, temporary)
            temporary.replace(args.output / "best_model.pth")
        else:
            stale += 1
        save_json(args.output / "history.json", history)
        if stale >= args.patience:
            break
        started = perf_counter()

    return {
        "steps": step,
        "best_step": best_step,
        "best_score": best_score,
        "best_validation": best_validation,
        "training_seconds": training_seconds,
        "steps_per_second": step / training_seconds,
    }
