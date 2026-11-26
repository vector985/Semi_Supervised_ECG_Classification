"""Command-line training and checkpoint evaluation."""

import argparse
import json
import logging
import math
from pathlib import Path

import torch

from . import CLASS_NAMES
from .data import load_splits, make_loader, split_training
from .evaluation import evaluate
from .model import ECGCNN
from .trainer import save_json, train

logger = logging.getLogger(__name__)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def make_parser():
    parser = argparse.ArgumentParser(description="Train or evaluate an ECG MixMatch model.")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="Train and save the best validation checkpoint")
    assess = commands.add_parser("evaluate", help="Evaluate a saved checkpoint")
    for command in (fit, assess):
        command.add_argument(
            "--data", type=Path, required=True,
            help="NPZ with ecgstrain/labelstrain, ecgsval/labelsval, and ecgstest/labelstest",
        )
        command.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
        command.add_argument("--workers", type=int, default=0)
        command.add_argument("--cpu-threads", type=positive_int)
        command.add_argument("--eval-batch-size", type=positive_int, default=256)
        command.add_argument("--trusted-pickle", action="store_true", help="Allow trusted object arrays")
        command.add_argument("--amp", action="store_true", help="Enable BF16 CUDA autocast")
    fit.add_argument("--output", type=Path, required=True, help="New directory for run outputs")
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--iterations", type=positive_int, default=10_000)
    fit.add_argument("--log-interval", type=positive_int, default=100)
    fit.add_argument("--eval-interval", type=positive_int, default=500)
    fit.add_argument("--patience", type=positive_int, default=10)
    fit.add_argument("--scheduler", choices=("one_cycle", "cosine", "plateau"), default="one_cycle")
    fit.add_argument("--learning-rate", type=float, default=1e-3)
    fit.add_argument("--weight-decay", type=float, default=1e-4)
    fit.add_argument("--batch-size", type=positive_int, default=32)
    fit.add_argument("--labeled-ratio", type=float, default=0.2)
    fit.add_argument("--views", type=positive_int, default=2)
    fit.add_argument("--alpha", type=float, default=0.75)
    fit.add_argument("--lambda-u", type=float, default=1.0)
    fit.add_argument("--temperature", type=float, default=0.5)
    fit.add_argument("--warmup", type=int, default=500)
    fit.add_argument("--base-channels", type=positive_int, default=32)
    fit.add_argument("--gru-hidden", type=positive_int, default=128)
    fit.add_argument("--dropout", type=float, default=0.5)
    assess.add_argument("--checkpoint", type=Path, required=True)
    assess.add_argument("--split", choices=("val", "test"), default="test")
    return parser


def validate_args(parser, args):
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    if args.command != "train":
        return
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.batch_size < 2 or args.base_channels % 8:
        parser.error("--batch-size must be at least 2 and --base-channels a multiple of 8")
    if not 0 < args.labeled_ratio < 1 or not 0 <= args.dropout < 1:
        parser.error("--labeled-ratio must be in (0,1) and --dropout in [0,1)")
    for name in ("warmup", "alpha", "lambda_u", "weight_decay", "learning_rate", "temperature"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            option = name.replace("_", "-")
            parser.error(f"--{option} must be finite and nonnegative")
    if args.learning_rate == 0 or args.temperature == 0:
        parser.error("--learning-rate and --temperature must be positive")


def resolve_device(parser, args):
    available = {"cpu": True, "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    name = args.device
    if name == "auto":
        name = next(kind for kind in ("cuda", "mps", "cpu") if available[kind])
    if not available[name]:
        parser.error(f"{name} is unavailable in this PyTorch environment")
    if args.amp and (name != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("--amp requires CUDA with BF16 support")
    return torch.device(name)


def train_command(args, splits, device):
    torch.manual_seed(args.seed)
    labeled, unlabeled = split_training(
        splits["train"], labeled_ratio=args.labeled_ratio, seed=args.seed
    )
    common = {"workers": args.workers, "device": device}
    sizes = {"labeled": min(args.batch_size, len(labeled)), "unlabeled": min(args.batch_size, len(unlabeled))}
    labeled_loader = make_loader(labeled, sizes["labeled"], shuffle=True, seed=args.seed, **common)
    unlabeled_loader = make_loader(
        unlabeled, sizes["unlabeled"], shuffle=True, seed=args.seed + 1, **common
    )
    validation = make_loader(splits["val"], args.eval_batch_size, **common)
    model_config = {
        "base_channels": args.base_channels, "gru_hidden": args.gru_hidden, "dropout": args.dropout,
    }
    model = ECGCNN(**model_config).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    args.output.mkdir(parents=True, exist_ok=False)
    logger.info("Device %s, model parameters %d, labeled %d, unlabeled %d",
                device, parameters, len(labeled), len(unlabeled))
    summary = train(model, labeled_loader, unlabeled_loader, validation, device, args, model_config)
    summary.update({
        "class_names": CLASS_NAMES,
        "parameters": parameters,
        "device": str(device),
        "pytorch_version": str(torch.__version__),
        "cpu_threads": torch.get_num_threads(),
        "effective_batch_sizes": sizes,
        "split_sizes": {name: len(dataset) for name, dataset in splits.items()},
        "training_labels": {"labeled": len(labeled), "unlabeled": len(unlabeled)},
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    })
    save_json(args.output / "results_summary.json", summary)
    return summary


def evaluate_command(args, splits, device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "model_state" in checkpoint:
        model = ECGCNN(**checkpoint["model_config"])
        state = checkpoint["model_state"]
    else:
        model, state = ECGCNN(), checkpoint
    model.load_state_dict(state)
    model.to(device)
    loader = make_loader(
        splits[args.split], args.eval_batch_size, workers=args.workers, device=device
    )
    return {
        "split": args.split,
        "class_names": CLASS_NAMES,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "batch_size": args.eval_batch_size,
        "amp": args.amp,
        **evaluate(model, loader, device, amp=args.amp),
    }


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    device = resolve_device(parser, args)
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        names = ("train", "val") if args.command == "train" else (args.split,)
        splits = load_splits(args.data, names, trusted_pickle=args.trusted_pickle)
        if args.command == "train":
            result = train_command(args, splits, device)
        else:
            result = evaluate_command(args, splits, device)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
