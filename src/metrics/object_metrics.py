"""Lesion-level detection metrics."""

from __future__ import annotations
import numpy as np
import torch
from scipy import ndimage


def object_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    small_threshold: int = 64,
    channelwise: bool = False,
) -> dict[str, float]:
    """Compute lesion detection metrics per semantic class or region channel."""

    pred, truth = prediction.detach().cpu().numpy(), target.detach().cpu().numpy()
    detected = total = small_total = small_detected = predicted_total = predicted_matched = 0
    for p, t in zip(pred, truth):
        if channelwise:
            pairs = [(pc > 0, tc > 0) for pc, tc in zip(p, t)]
        else:
            classes = sorted(set(np.unique(p).tolist()) | set(np.unique(t).tolist()) - {0})
            classes = [value for value in classes if value != 0]
            pairs = [(p == value, t == value) for value in classes]
        for class_prediction, class_target in pairs:
            labels, count = ndimage.label(class_target)
            for index in range(1, count + 1):
                obj = labels == index
                hit = bool(class_prediction[obj].any())
                size = int(obj.sum())
                total += 1
                detected += hit
                if size < small_threshold:
                    small_total += 1
                    small_detected += hit
            predicted_labels, predicted_count = ndimage.label(class_prediction)
            predicted_total += predicted_count
            for index in range(1, predicted_count + 1):
                predicted_matched += bool(class_target[predicted_labels == index].any())
    recall = detected / max(total, 1)
    precision=predicted_matched/max(predicted_total,1);f1=2*precision*recall/max(precision+recall,1e-8)
    return {"object_recall": recall, "object_f1": f1, "small_component_recall": small_detected / max(small_total, 1),
            "false_negative_lesion_rate": 1 - recall, "object_count": float(total),
            "small_component_count": float(small_total), "small_component_threshold": float(small_threshold)}
