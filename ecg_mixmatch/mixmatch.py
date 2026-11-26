"""ECG augmentations, pseudo-label guessing, and MixMatch losses."""

import torch
from torch.nn import functional as F

from . import CLASS_NAMES


def augment_views(signals, views):
    """Return fresh weak or strong perturbations in (batch, view, time) order."""
    batch, length = signals.shape
    count = batch * views
    device = signals.device
    time = torch.arange(length, device=device)
    strong = torch.rand(count, 1, device=device) < 0.5
    amplitude = torch.where(strong, 0.20, 0.02)
    scale = 1 + (2 * torch.rand(count, 1, device=device) - 1) * amplitude
    bias = 0.4 * torch.rand(count, 1, device=device) - 0.2
    noise = torch.rand(count, 1, device=device)
    noise_std = torch.where(strong, 0.08 + 0.17 * noise, 0.005 + 0.015 * noise)
    x = signals.repeat_interleave(views, dim=0).mul_(scale)
    x.add_(torch.randn_like(x).mul_(noise_std)).add_(torch.where(strong, bias, 0.0))

    shifts = torch.randint(-3, 4, (count, 1), device=device)
    shifts = torch.where(strong, 0, shifts)
    x = x.gather(1, (time - shifts).remainder(length))
    max_mask = int(length * 0.4)
    if max_mask:
        span = torch.randint(1, max_mask + 1, (count, 1), device=device)
        start = (torch.rand(count, 1, device=device) * (length - span + 1)).long()
        mask = strong & (torch.rand(count, 1, device=device) < 0.5)
        x.masked_fill_(mask & (time >= start) & (time < start + span), 0)
    return x.reshape(batch, views, length)


@torch.no_grad()
def guess_labels(model, views, temperature):
    """Average predicted probabilities across views and sharpen in log space."""
    was_training = model.training
    model.eval()
    try:
        logits = model(views.flatten(0, 1)).float().reshape(*views.shape[:2], -1)
    finally:
        model.train(was_training)
    probabilities = logits.softmax(dim=-1).mean(dim=1)
    logs = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    return ((logs - logs.amax(dim=-1, keepdim=True)) / temperature).softmax(dim=-1)


def mix_batches(labeled, labels, views, pseudo_labels, *, alpha, rng):
    """Keep labeled anchors first and pick partners from both groups."""
    labeled_targets = F.one_hot(labels, num_classes=len(CLASS_NAMES)).float()
    x = torch.cat((labeled, views.flatten(0, 1)), dim=0)
    targets = torch.cat(
        (labeled_targets, pseudo_labels.repeat_interleave(views.shape[1], dim=0)), dim=0
    )
    weight = float(rng.beta(alpha, alpha)) if alpha else 1.0
    partner_weight = min(weight, 1.0 - weight)
    partners = torch.randperm(len(x), device=x.device)
    return x.lerp(x[partners], partner_weight), targets.lerp(targets[partners], partner_weight)


def compute_losses(logits, targets, labeled_count):
    logits = logits.float()
    labeled = logits[:labeled_count].log_softmax(dim=-1)
    supervised = -(targets[:labeled_count] * labeled).sum(dim=-1).mean()
    unlabeled = logits[labeled_count:].softmax(dim=-1)
    unsupervised = F.mse_loss(unlabeled, targets[labeled_count:])
    return supervised, unsupervised
