"""Logit knowledge-distillation losses."""

from __future__ import annotations
import torch
import torch.nn.functional as F


def softened_kl(
    compressed_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
    weight: torch.Tensor | None = None,
    normalize_weight: bool = True,
) -> torch.Tensor:
    """Temperature-scaled KL with optional sample/spatial weights."""
    t = float(temperature)
    if compressed_logits.shape[1] == 1:
        sp = torch.cat((-compressed_logits, compressed_logits), 1)
        tp = torch.cat((-teacher_logits, teacher_logits), 1)
    else:
        sp, tp = compressed_logits, teacher_logits
    pointwise = F.kl_div(F.log_softmax(sp / t, 1), F.softmax(tp.detach() / t, 1), reduction="none").sum(1) * (t * t)
    if weight is None:
        return pointwise.mean()
    while weight.ndim < pointwise.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(pointwise)
    weighted=pointwise*weight
    return weighted.sum()/weight.sum().clamp_min(1e-8) if normalize_weight else weighted.mean()


def softened_bernoulli_kl(compressed_logits:torch.Tensor,teacher_logits:torch.Tensor,temperature:float=4.,weight:torch.Tensor|None=None,normalize_weight:bool=True)->torch.Tensor:
    """Independent-label temperature KD for multi-label classification."""
    # Compute in FP32 with log-sigmoid identities. Directly clamping FP16
    # probabilities at 1e-8 is ineffective (it rounds to 0/1) and yields NaNs
    # once a compressed model becomes confident under AMP.
    t=float(temperature);student=compressed_logits.float()/t;teacher=teacher_logits.detach().float()/t
    q=teacher.sigmoid();log_q=F.logsigmoid(teacher);log_one_minus_q=F.logsigmoid(-teacher)
    log_p=F.logsigmoid(student);log_one_minus_p=F.logsigmoid(-student)
    point=(q*(log_q-log_p)+(1-q)*(log_one_minus_q-log_one_minus_p)).mean(1)*(t*t)
    if weight is None:return point.mean()
    weighted=point*weight
    return weighted.sum()/weight.sum().clamp_min(1e-8) if normalize_weight else weighted.mean()
