"""Shape-based task semantics shared by losses, risk maps, and metrics."""

from __future__ import annotations

import torch


def is_multilabel_target(logits: torch.Tensor, targets: torch.Tensor) -> bool:
    """Return whether channels represent independent binary labels/regions."""

    return (
        targets.ndim == logits.ndim
        and targets.shape[1] == logits.shape[1]
    )
