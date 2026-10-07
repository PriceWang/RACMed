"""Composite risk-gated RACMed distillation objective."""

from __future__ import annotations
from typing import Any
import torch
from torch import nn

from .boundary_losses import boundary_loss
from .calibration_losses import brier_loss
from .feature_kd import feature_distillation
from .kd_losses import softened_kl,softened_bernoulli_kl
from .relation_kd import relation_distillation
from .sparsity_losses import sparsity_loss
from .supervised import supervised_loss


class RACMedLoss(nn.Module):
    """Risk-aware combination of prediction, evidence, reliability and compression losses."""

    def __init__(self, temperature: float = 4.0, lambda_global: float = 1.0,
                 lambda_local: float = 1.0, lambda_boundary: float = .2,
                 lambda_relation: float = .1, lambda_calib: float = .1,
                 lambda_sparse: float = 1e-5, focal_gamma: float = 0.0,
                 dice_include_background: bool = False, dice_squared: bool = False,
                 dice_batch: bool = False, local_feature_mode: str = "l2",
                 risk_supervision_strength: float = 0.0,
                 risk_supervision_foreground_only: bool = False,
                 component_supervision_strength: float = 0.0,
                 local_logit_weight: float = 0.0,
                 teacher_correctness_gating: bool = False,
                 risk_gating_enabled: bool = True) -> None:
        super().__init__()
        self.temperature = temperature
        self.weights = dict(global_kd=lambda_global, local_kd=lambda_local, boundary=lambda_boundary,
                            relation=lambda_relation, calib=lambda_calib, sparse=lambda_sparse)
        self.focal_gamma = focal_gamma
        self.local_feature_mode=str(local_feature_mode)
        self.risk_supervision_strength=float(risk_supervision_strength)
        self.risk_supervision_foreground_only=bool(risk_supervision_foreground_only)
        self.component_supervision_strength=float(component_supervision_strength)
        self.local_logit_weight=float(local_logit_weight)
        self.teacher_correctness_gating=bool(teacher_correctness_gating)
        self.risk_gating_enabled=bool(risk_gating_enabled)
        self.supervised_kwargs = {
            "focal_gamma": focal_gamma,
            "dice_include_background": dice_include_background,
            "dice_squared": dice_squared,
            "dice_batch": dice_batch,
        }

    def forward(self, compressed_outputs: dict[str, Any], teacher_outputs: dict[str, Any],
                targets: torch.Tensor, images: torch.Tensor | None,
                risk_outputs: dict[str, Any], task_type: str,
                model: nn.Module | None = None) -> dict[str, torch.Tensor]:
        del images
        compressed, teacher = compressed_outputs["logits"], teacher_outputs["logits"]
        risk = risk_outputs["risk_map"].detach()
        entropy = risk_outputs["teacher_entropy"].detach()
        teacher_trust = 1 - entropy
        if self.teacher_correctness_gating:
            if targets.ndim == teacher.ndim and targets.shape[1] == teacher.shape[1]:
                correctness = teacher.detach().gt(0).eq(targets.gt(0.5)).float().mean(1)
            elif teacher.shape[1] == 1:
                target = targets[:, 0] if targets.ndim == teacher.ndim else targets
                correctness = teacher.detach()[:, 0].gt(0).eq(target.gt(0.5)).float()
            else:
                target = targets[:, 0] if targets.ndim == teacher.ndim else targets
                correctness = teacher.detach().argmax(1).eq(target.long()).float()
            teacher_trust = teacher_trust * correctness
        reliable = (1 - risk) * teacher_trust if self.risk_gating_enabled else teacher_trust
        high_risk_reliable = risk * teacher_trust if self.risk_gating_enabled else teacher_trust
        risk_supervision_strength=self.risk_supervision_strength if self.risk_gating_enabled else 0.0
        focus_weight=torch.ones_like(risk) if risk_supervision_strength>0 or (self.component_supervision_strength>0 and task_type.startswith("segmentation")) else None
        if risk_supervision_strength>0:
            focus=((risk-.5)*2).clamp(0,1)
            if self.risk_supervision_foreground_only and task_type.startswith("segmentation"):
                foreground=targets.float().amax(1) if targets.ndim==compressed.ndim else (targets[:,0] if targets.ndim==compressed.ndim and targets.shape[1]==1 else targets).gt(0).float()
                focus=focus*foreground
            focus_weight=focus_weight+risk_supervision_strength*focus
        if self.component_supervision_strength>0 and task_type.startswith("segmentation"):
            component_weight=risk_outputs.get("small_component_weight",torch.ones_like(risk)).detach().to(risk)
            focus_weight=focus_weight+self.component_supervision_strength*(component_weight-1).clamp_min(0)
        sup = supervised_loss(compressed, targets, task_type, pixel_weight=focus_weight, **self.supervised_kwargs)
        global_kd = softened_bernoulli_kl(compressed,teacher,self.temperature,reliable,normalize_weight=False) if targets.ndim==compressed.ndim and targets.shape[1]==compressed.shape[1] else softened_kl(compressed, teacher, self.temperature,reliable,normalize_weight=False)
        if self.weights["local_kd"] > 0:
            local_terms = []
            common = set(compressed_outputs.get("features", {})) & set(teacher_outputs.get("features", {}))
            for key in sorted(common):
                local_terms.append(feature_distillation(compressed_outputs["features"][key], teacher_outputs["features"][key], high_risk_reliable,self.local_feature_mode))
            local = torch.stack(local_terms).mean() if local_terms else compressed.sum() * 0
            if self.local_logit_weight>0:
                local_logits=softened_bernoulli_kl(compressed,teacher,self.temperature,high_risk_reliable) if targets.ndim==compressed.ndim and targets.shape[1]==compressed.shape[1] else softened_kl(compressed,teacher,self.temperature,high_risk_reliable)
                local=local+self.local_logit_weight*local_logits
        else:
            local = compressed.sum() * 0
        boundary = boundary_loss(compressed, targets, risk_outputs.get("boundary_score")) if self.weights["boundary"] > 0 and task_type.startswith("segmentation") else compressed.sum() * 0
        relation = relation_distillation(compressed_outputs.get("features", {}).get("bottleneck", compressed),
                                         teacher_outputs.get("features", {}).get("bottleneck", teacher), None if task_type=="classification" else targets) if self.weights["relation"] > 0 else compressed.sum() * 0
        calib = brier_loss(compressed, targets)
        sparse = sparsity_loss(model).to(compressed.device) if model is not None else compressed.sum() * 0
        values = {"sup": sup, "global_kd": global_kd, "local_kd": local, "boundary": boundary,
                  "relation": relation, "calib": calib, "sparse": sparse}
        total = sup + sum(self.weights[key] * values[key] for key in self.weights)
        return {"total": total, **values}
