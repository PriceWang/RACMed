"""Differentiable calibration objectives."""

from __future__ import annotations
import torch
import torch.nn.functional as F


def brier_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim == logits.ndim:
        return F.mse_loss(logits.sigmoid(), targets.float())
    if logits.shape[1] == 1:
        return F.mse_loss(logits.sigmoid().squeeze(1), targets.float())
    probs = logits.softmax(1)
    truth = F.one_hot(targets.long(), logits.shape[1]).movedim(-1, 1).float()
    return (probs - truth).pow(2).sum(1).mean()
