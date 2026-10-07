"""Task-aware supervised objectives."""

from __future__ import annotations
import torch
import torch.nn.functional as F
from src.utils.task_semantics import is_multilabel_target


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-5,
    include_background: bool = False,
    squared_pred: bool = False,
    batch: bool = False,
) -> torch.Tensor:
    """Soft Dice with MONAI-compatible background, squared-pred and batch modes."""
    multilabel=is_multilabel_target(logits,targets)
    if targets.ndim == logits.ndim and targets.shape[1] == 1:
        targets = targets[:, 0]
    if multilabel:
        probs=logits.sigmoid();truth=targets.float()
    elif logits.shape[1] == 1:
        probs = logits.sigmoid()
        truth = targets.float().unsqueeze(1)
    else:
        probs = logits.softmax(1)
        truth = F.one_hot(targets.long(), logits.shape[1]).movedim(-1, 1).float()
        if logits.shape[1] > 1 and not include_background:
            probs, truth = probs[:, 1:], truth[:, 1:]
    dims = tuple(range(2, logits.ndim))
    if batch:
        dims = (0, *dims)
    denominator_probs = probs.square() if squared_pred else probs
    denominator_truth = truth.square() if squared_pred else truth
    score = (2 * (probs * truth).sum(dims) + eps) / (denominator_probs.sum(dims) + denominator_truth.sum(dims) + eps)
    return 1 - score.mean()


def supervised_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    task_type: str,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
    dice_include_background: bool = False,
    dice_squared: bool = False,
    dice_batch: bool = False,
    pixel_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """CE/BCE for classification and DiceCE (optionally focal CE) for segmentation."""
    if task_type == "classification":
        if targets.ndim == logits.ndim or logits.shape[1] == 1:
            loss=F.binary_cross_entropy_with_logits(logits.squeeze(1), targets.float(),reduction="none")
        else:
            loss=F.cross_entropy(logits, targets.long(), weight=class_weights,reduction="none")
        if pixel_weight is None:return loss.mean()
        weight=pixel_weight.to(loss)
        while weight.ndim<loss.ndim:weight=weight.unsqueeze(-1)
        weight=weight.expand_as(loss)
        return (loss*weight).sum()/weight.sum().clamp_min(1e-8)
    if is_multilabel_target(logits,targets):
        ce=F.binary_cross_entropy_with_logits(logits,targets.float(),reduction="none")
        if focal_gamma>0:
            probability=torch.exp(-ce)
            ce=((1-probability)**focal_gamma)*ce
        if pixel_weight is not None:
            weight=pixel_weight.to(ce)
            if weight.ndim==ce.ndim-1:weight=weight.unsqueeze(1)
            ce=(ce*weight).sum()/weight.expand_as(ce).sum().clamp_min(1e-8)
        else:ce=ce.mean()
        return ce+dice_loss(
            logits,targets,include_background=True,squared_pred=dice_squared,batch=dice_batch
        )
    target = targets[:, 0] if targets.ndim == logits.ndim and targets.shape[1] == 1 else targets
    if logits.shape[1] == 1:
        ce = F.binary_cross_entropy_with_logits(logits[:, 0], target.float(), reduction="none")
    else:
        ce = F.cross_entropy(logits, target.long(), weight=class_weights, reduction="none")
    if focal_gamma > 0:
        ce = ((1 - torch.exp(-ce)) ** focal_gamma) * ce
    if pixel_weight is not None:
        weight=pixel_weight.to(ce)
        ce=(ce*weight).sum()/weight.sum().clamp_min(1e-8)
    else:ce=ce.mean()
    return ce + dice_loss(
        logits,
        target,
        include_background=dice_include_background,
        squared_pred=dice_squared,
        batch=dice_batch,
    )
