"""Evaluate rhythm predictions with a fixed class order."""

from time import perf_counter

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch

from . import CLASS_NAMES


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def evaluate(model, loader, device, *, amp=False):
    model.eval()
    predicted, actual = [], []
    synchronize(device)
    started = perf_counter()
    for signals, labels in loader:
        signals = signals.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            predicted.append(model(signals).argmax(dim=-1))
        actual.append(labels)
    predicted = torch.cat(predicted).cpu().numpy()
    actual = torch.cat(actual).numpy()
    elapsed = perf_counter() - started
    class_ids = list(range(len(CLASS_NAMES)))
    report = classification_report(
        actual, predicted, labels=class_ids, target_names=CLASS_NAMES,
        zero_division=0, output_dict=True,
    )
    macro = report["macro avg"]
    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": macro["f1-score"],
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "per_class": {name: report[name] for name in CLASS_NAMES},
        "confusion_matrix": confusion_matrix(actual, predicted, labels=class_ids).tolist(),
        "samples": len(actual),
        "inference_seconds": elapsed,
        "samples_per_second": len(actual) / elapsed,
    }
