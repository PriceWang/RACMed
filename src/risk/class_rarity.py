"""Class-frequency based pixel and sample risk."""

from __future__ import annotations

from collections.abc import Mapping
import torch


def class_rarity_score(
    targets: torch.Tensor, class_frequency: Mapping[int, float] | None = None
) -> torch.Tensor:
    """Map labels to inverse normalized frequency; infer frequencies per batch if absent."""
    labels = targets.long()
    if labels.ndim >= 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if class_frequency is None:
        values, counts = labels.unique(return_counts=True)
        total = counts.sum().item()
        frequencies = {int(v): float(c.item() / total) for v, c in zip(values, counts)}
    else:
        frequencies = {int(k): float(v) for k, v in class_frequency.items()}
    if not frequencies:
        return torch.zeros_like(labels, dtype=torch.float32)
    maximum = max(frequencies.values())
    minimum = min(frequencies.values())
    scale = max(maximum - minimum, 1e-8)
    out = torch.zeros_like(labels, dtype=torch.float32)
    for cls in labels.unique():
        frequency = frequencies.get(int(cls), maximum)
        out[labels == cls] = (maximum - frequency) / scale if maximum > minimum else 0.0
    return out
