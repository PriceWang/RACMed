"""Predictive uncertainty functions."""

from __future__ import annotations

import math
import torch


def probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert binary or multiclass logits into an explicit class distribution."""
    logits = logits.float()
    if logits.shape[1] == 1:
        p = logits.sigmoid()
        return torch.cat((1 - p, p), dim=1)
    return logits.softmax(dim=1)


def normalized_entropy(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy normalized to [0, 1], excluding the class dimension."""
    p = probabilities(logits).clamp_min(eps)
    entropy = -(p * p.log()).sum(dim=1) / math.log(p.shape[1])
    return entropy.clamp(0, 1)


def low_confidence_margin(logits: torch.Tensor) -> torch.Tensor:
    """Risk from the inverse gap between the two most likely classes."""
    p = probabilities(logits)
    top = p.topk(min(2, p.shape[1]), dim=1).values
    margin = top[:, 0] if top.shape[1] == 1 else top[:, 0] - top[:, 1]
    return (1 - margin).clamp(0, 1)
