"""Calibration metrics and reliability diagram bins."""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def _adaptive_edges(values:torch.Tensor,n_bins:int,max_samples:int=1_000_000) -> torch.Tensor:
    """Estimate adaptive-bin edges from a deterministic bounded voxel sample."""

    flat=values.reshape(-1)
    if flat.numel()>max_samples:
        stride=math.ceil(flat.numel()/max_samples)
        flat=flat[::stride]
    return torch.quantile(flat,torch.linspace(0,1,n_bins+1,device=flat.device))


def _bounded_pair(first:torch.Tensor,second:torch.Tensor,max_samples:int=1_000_000) -> tuple[torch.Tensor,torch.Tensor]:
    """Take the same deterministic, evenly spaced sample from dense tensors."""

    left=first.reshape(-1);right=second.reshape(-1)
    if left.numel()>max_samples:
        stride=math.ceil(left.numel()/max_samples)
        left=left[::stride];right=right[::stride]
    return left,right


def _confidence(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.shape[1] == 1:
        p = logits.sigmoid().squeeze(1)
        return torch.maximum(p, 1 - p), (p >= .5).long()
    return logits.softmax(1).max(1)


def reliability_diagram_data(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15, adaptive: bool = False) -> list[dict[str, float]]:
    conf, pred = _confidence(logits)
    correct = pred.eq(targets.long()).float()
    conf,correct=_bounded_pair(conf,correct)
    edges = _adaptive_edges(conf,n_bins) if adaptive else torch.linspace(0, 1, n_bins + 1, device=conf.device)
    rows = []
    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1] if i < n_bins - 1 else conf <= 1)
        rows.append({"lower": float(edges[i]), "upper": float(edges[i + 1]),
                     "count": int(mask.sum()), "confidence": float(conf[mask].mean()) if mask.any() else 0.0,
                     "accuracy": float(correct[mask].mean()) if mask.any() else 0.0})
    return rows


def expected_calibration_error(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15, adaptive: bool = False) -> float:
    if targets.ndim==logits.ndim:
        probs=logits.sigmoid();confidence=torch.maximum(probs,1-probs).flatten();correct=(probs.ge(.5)==targets.bool()).float().flatten()
        confidence,correct=_bounded_pair(confidence,correct)
        edges=_adaptive_edges(confidence,n_bins) if adaptive else torch.linspace(0,1,n_bins+1,device=logits.device);ece=0.
        for i in range(n_bins):
            mask=(confidence>edges[i])&(confidence<=edges[i+1])
            if mask.any():ece+=float(mask.float().mean()*torch.abs(correct[mask].mean()-confidence[mask].mean()))
        return ece
    rows=reliability_diagram_data(logits,targets,n_bins,adaptive)
    total=max(sum(row["count"] for row in rows),1)
    return float(sum(row["count"] / total * abs(row["accuracy"] - row["confidence"])
                     for row in rows))


def brier_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if targets.ndim==logits.ndim:return float(F.mse_loss(logits.sigmoid(),targets.float()))
    if logits.shape[1] == 1:
        return float(F.mse_loss(logits.sigmoid().squeeze(1), targets.float()))
    truth = F.one_hot(targets.long(), logits.shape[1]).movedim(-1, 1).float()
    return float((logits.softmax(1) - truth).pow(2).sum(1).mean())


def negative_log_likelihood(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if targets.ndim==logits.ndim:return float(F.binary_cross_entropy_with_logits(logits,targets.float()))
    if logits.shape[1] == 1:
        return float(F.binary_cross_entropy_with_logits(logits.squeeze(1), targets.float()))
    return float(F.cross_entropy(logits, targets.long()))
