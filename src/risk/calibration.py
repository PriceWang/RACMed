"""Validation-only calibration for teacher-derived risk estimates."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _calibration_nll(
    logits: torch.Tensor, targets: torch.Tensor, temperature: torch.Tensor
) -> torch.Tensor:
    scaled = logits / temperature
    if scaled.ndim == 1 or scaled.shape[1] == 1:
        return F.binary_cross_entropy_with_logits(
            scaled.reshape(-1), targets.float().reshape(-1)
        )
    if targets.ndim == scaled.ndim and targets.shape == scaled.shape:
        return F.binary_cross_entropy_with_logits(scaled, targets.float())
    target = targets[:, 0] if targets.ndim == scaled.ndim else targets
    return F.cross_entropy(scaled, target.long())


def fit_logit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    minimum: float = 0.05,
    maximum: float = 20.0,
) -> float:
    """Fit one positive temperature on validation logits by minimizing NLL.

    Degenerate or non-improving fits safely return ``1.0`` so calibration can
    never make the validation NLL worse.
    """

    values = logits.detach().float().cpu()
    labels = targets.detach().cpu()
    if values.numel() == 0 or labels.numel() == 0:
        return 1.0
    lower, upper = float(minimum), float(maximum)
    if not 0 < lower <= 1.0 <= upper:
        raise ValueError("Temperature bounds must satisfy 0 < minimum <= 1 <= maximum.")
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.2, max_iter=80, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(lower, upper)
        loss = _calibration_nll(values, labels, temperature)
        loss.backward()
        return loss

    before = _calibration_nll(values, labels, torch.tensor(1.0))
    try:
        optimizer.step(closure)
    except (RuntimeError, ValueError):
        return 1.0
    temperature = float(log_temperature.detach().exp().clamp(lower, upper))
    after = _calibration_nll(values, labels, torch.tensor(temperature))
    if not torch.isfinite(after) or after > before + 1e-7:
        return 1.0
    return temperature


def fit_binary_platt_scaling(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[float, float]:
    """Fit a positive affine binary-logit calibration without changing ranking."""

    values=logits.detach().float().cpu().reshape(-1)
    labels=targets.detach().float().cpu().reshape(-1)
    if values.numel()==0 or values.numel()!=labels.numel():return 1.0,0.0
    log_scale=torch.zeros((),requires_grad=True);bias=torch.zeros((),requires_grad=True)
    optimizer=torch.optim.LBFGS([log_scale,bias],lr=.2,max_iter=80,line_search_fn="strong_wolfe")
    def closure()->torch.Tensor:
        optimizer.zero_grad();scale=log_scale.exp().clamp(.01,100.)
        loss=F.binary_cross_entropy_with_logits(values*scale+bias,labels);loss.backward();return loss
    before=F.binary_cross_entropy_with_logits(values,labels)
    try:optimizer.step(closure)
    except (RuntimeError,ValueError):return 1.0,0.0
    scale=float(log_scale.detach().exp().clamp(.01,100.));offset=float(bias.detach())
    after=F.binary_cross_entropy_with_logits(values*scale+offset,labels)
    if not torch.isfinite(after) or after>before+1e-7:return 1.0,0.0
    return scale,offset
