"""Teacher-compressed probability disagreement."""

from __future__ import annotations

import math
import torch
from .uncertainty import probabilities


def probability_disagreement(
    compressed_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    method: str = "js",
    eps: float = 1e-8,
    multilabel: bool = False,
) -> torch.Tensor:
    """Compute bounded JS, normalized KL, or L1 distance over classes."""
    if multilabel:
        ps,qs=compressed_logits.float().sigmoid(),teacher_logits.float().sigmoid();p=torch.stack((1-ps,ps),dim=2).clamp_min(eps);q=torch.stack((1-qs,qs),dim=2).clamp_min(eps);class_dim=2
    else:
        p = probabilities(compressed_logits.float()).clamp_min(eps);q = probabilities(teacher_logits.float()).clamp_min(eps);class_dim=1
    if p.shape != q.shape:
        raise ValueError(f"Teacher/compressed probability shapes differ: {p.shape} vs {q.shape}")
    if method == "js":
        m = (p + q) / 2
        value = ((p * (p.log() - m.log())).sum(class_dim) + (q * (q.log() - m.log())).sum(class_dim)) / (2 * math.log(2))
    elif method == "kl":
        value = (q * (q.log() - p.log())).sum(class_dim)
        value = 1 - torch.exp(-value)
    elif method == "l1":
        value = (p - q).abs().sum(class_dim) / 2
    else:
        raise ValueError(f"Unknown disagreement method: {method}")
    if multilabel:value=value.mean(1)
    return value.clamp(0, 1)
