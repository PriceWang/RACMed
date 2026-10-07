"""Differentiable boundary preservation loss."""

import torch
import torch.nn.functional as F
from src.risk.boundary_map import boundary_risk_map, soft_prediction_boundary
from src.utils.task_semantics import is_multilabel_target


def boundary_loss(logits: torch.Tensor, targets: torch.Tensor, boundary_target: torch.Tensor | None = None) -> torch.Tensor:
    multilabel=is_multilabel_target(logits,targets)
    probs = logits.sigmoid() if logits.shape[1] == 1 or multilabel else logits.softmax(1)
    prediction = soft_prediction_boundary(probs).clamp(1e-6, 1 - 1e-6)
    if boundary_target is not None:
        truth=(boundary_target.to(prediction.device)>0.75).float()
    elif multilabel:
        truth=torch.stack([boundary_risk_map(targets[:,channel]) for channel in range(targets.shape[1])]).amax(0)
        truth=(truth>0.75).float()
    else:
        truth = (boundary_risk_map(targets) > 0.75).float()
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction, truth = prediction.float(), truth.float()
        bce = F.binary_cross_entropy(prediction, truth)
        dice = 1 - (2 * (prediction * truth).sum() + 1) / (prediction.sum() + truth.sum() + 1)
    return bce + dice
