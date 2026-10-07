"""Boundary overlap metrics."""

import torch
from src.risk.boundary_map import boundary_risk_map


def boundary_f1(prediction: torch.Tensor, target: torch.Tensor, tolerance: float = .5) -> float:
    pb = boundary_risk_map(prediction) > tolerance
    tb = boundary_risk_map(target) > tolerance
    tp = (pb & tb).sum().float(); precision = tp / pb.sum().clamp_min(1); recall = tp / tb.sum().clamp_min(1)
    return float(2 * precision * recall / (precision + recall).clamp_min(1e-8))
