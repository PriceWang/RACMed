"""Task-aware loss and metric objectives for layer-risk profiling."""

from __future__ import annotations

from collections.abc import Callable

import torch

from src.losses.supervised import supervised_loss
from src.utils.task_semantics import is_multilabel_target


def profile_task_metric(
    logits: torch.Tensor,
    targets: torch.Tensor,
    task_type: str,
    num_classes: int,
) -> torch.Tensor:
    """Return a fast task metric suitable for repeated perturbation forwards."""

    if task_type == "classification":
        if logits.ndim == 2 and logits.shape[1] == 1:
            prediction = logits[:, 0].sigmoid().ge(0.5)
            truth = targets.reshape(-1).bool()
            return prediction.eq(truth).float().mean()
        if targets.ndim == logits.ndim:
            return logits.sigmoid().ge(0.5).eq(targets.bool()).float().mean()
        return logits.argmax(1).eq(targets.reshape(-1).long()).float().mean()

    if is_multilabel_target(logits, targets):
        prediction = logits.sigmoid().ge(0.5).float()
        truth = targets.float()
        dimensions = tuple(range(2, logits.ndim))
        intersection = (prediction * truth).sum(dimensions)
        denominator = prediction.sum(dimensions) + truth.sum(dimensions)
        valid = denominator > 0
        dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
        return dice[valid].mean() if valid.any() else logits.new_tensor(1.0)

    truth = targets.long()
    if truth.ndim == logits.ndim and truth.shape[1] == 1:
        truth = truth[:, 0]
    if logits.shape[1] == 1:
        prediction = logits[:, 0].sigmoid().ge(0.5).long()
        class_indices = [1]
    else:
        prediction = logits.argmax(1)
        class_indices = list(range(1, max(int(num_classes), logits.shape[1])))
    scores = []
    for class_index in class_indices:
        predicted = prediction.eq(class_index)
        expected = truth.eq(class_index)
        denominator = predicted.sum() + expected.sum()
        if denominator > 0:
            scores.append((2 * (predicted & expected).sum().float() + 1e-6) / (denominator.float() + 1e-6))
    return torch.stack(scores).mean() if scores else logits.new_tensor(1.0)


def build_profile_objectives(
    task_type: str,
    num_classes: int,
    loss_config: dict | None = None,
) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Build the supervised loss and task metric used by the profiler."""

    supported = {"focal_gamma", "dice_include_background", "dice_squared", "dice_batch"}
    options = {key: value for key, value in (loss_config or {}).items() if key in supported}

    def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return supervised_loss(logits, targets, task_type, **options)

    def metric_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return profile_task_metric(logits, targets, task_type, num_classes)

    return loss_fn, metric_fn
