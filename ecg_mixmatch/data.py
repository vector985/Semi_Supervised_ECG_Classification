"""Read preprocessed data and sample labeled and unlabeled training splits."""

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from . import CLASS_NAMES


def label_indices(labels):
    if labels.ndim == 2 and labels.shape[1] > 1:
        labels = labels.astype(np.float64)
        if not np.isfinite(labels).all():
            raise ValueError("One-hot label rows must contain finite numbers")
        return np.where(labels.any(axis=1), labels.argmax(axis=1), -1)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.ravel()
    if labels.ndim != 1:
        raise ValueError("Expected a label vector or a one-hot matrix")
    if np.issubdtype(labels.dtype, np.number):
        return labels

    names = np.char.strip(labels.astype(str))
    indices = np.full(len(names), -1, dtype=np.int64)
    for idx, name in enumerate(CLASS_NAMES):
        indices[(names == name) | (names == str(idx))] = idx
    return indices


def load_splits(path, names, *, trusted_pickle=False):
    splits = {}
    archive = np.load(path, allow_pickle=trusted_pickle)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Expected an NPZ archive")
    with archive:
        for name in names:
            signal_key, label_key = f"ecgs{name}", f"labels{name}"
            if signal_key not in archive.files or label_key not in archive.files:
                raise ValueError(f"{path} must contain {signal_key} and {label_key}")
            try:
                signals, raw_labels = archive[signal_key], archive[label_key]
            except ValueError as exc:
                if "allow_pickle=False" in str(exc):
                    raise ValueError("This NPZ contains object arrays. Use --trusted-pickle "
                                     "only for an archive you trust.") from exc
                raise
            labels = label_indices(raw_labels)
            keep = np.isin(labels, range(len(CLASS_NAMES)))
            if signals.ndim < 1 or len(signals) != len(labels) or not keep.any():
                raise ValueError(f"Split {name} needs aligned ECGs and N/A/O labels")
            if not keep.all():
                signals = signals[keep]
            if signals.dtype == object and signals.ndim == 1:
                signals = np.stack(signals)
            if signals.ndim == 3 and signals.shape[-1] == 1:
                signals = signals.squeeze(-1)
            elif signals.ndim == 3 and signals.shape[1] == 1:
                signals = signals.squeeze(1)
            if signals.ndim != 2 or signals.shape[1] < 2:
                raise ValueError(f"Split {name} needs fixed-length single-lead ECGs")
            signals = np.ascontiguousarray(signals, dtype=np.float32)
            if not np.isfinite(signals).all():
                raise ValueError(f"Split {name} contains non-finite ECG samples")
            splits[name] = TensorDataset(
                torch.from_numpy(signals), torch.from_numpy(labels[keep].astype(np.int64))
            )
    return splits


def split_training(dataset, *, labeled_ratio, seed):
    signals, labels = dataset.tensors
    if labels.unique().numel() != len(CLASS_NAMES):
        raise ValueError("Training requires examples of all N/A/O classes")
    labeled, unlabeled = train_test_split(
        np.arange(len(labels)), train_size=int(len(labels) * labeled_ratio),
        stratify=labels.numpy(), random_state=seed,
    )
    if labels[labeled].unique().numel() != len(CLASS_NAMES):
        raise ValueError("Increase --labeled-ratio so every class is represented")
    return Subset(dataset, labeled.tolist()), Subset(TensorDataset(signals), unlabeled.tolist())


def make_loader(dataset, batch_size, *, device, workers=0, shuffle=False, seed=42):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
        pin_memory=device.type == "cuda", num_workers=workers, persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def repeat_batches(loader):
    if len(loader) == 0:
        raise ValueError("Cannot repeat a loader with no batches")
    while True:
        yield from loader
