"""Composite region- and sample-level RACMed risk maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch

from .boundary_map import boundary_risk_map
from .class_rarity import class_rarity_score
from .disagreement import probability_disagreement
from .small_component import small_component_maps
from .uncertainty import low_confidence_margin, normalized_entropy
from src.utils.task_semantics import is_multilabel_target


@dataclass
class RiskMapBuilder:
    """Build interpretable risk factors and their bounded composite."""

    task_type: str = "segmentation2d"
    alpha_boundary: float = 2.0
    alpha_small: float = 2.0
    alpha_entropy: float = 1.0
    alpha_disagreement: float = 1.0
    alpha_class: float = 1.0
    alpha_margin: float = 1.0
    small_component_threshold: int = 64
    max_component_weight: float = 16.0
    disagreement_method: str = "js"
    high_risk_threshold: float = 0.7
    teacher_temperature: float = 1.0

    def __call__(
        self,
        compressed_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        image: torch.Tensor | None = None,
        class_frequency: dict[int, float] | None = None,
        static_outputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        del image
        temperature=max(float(self.teacher_temperature),1e-6)
        calibrated_teacher=teacher_logits.float()/temperature
        multilabel=is_multilabel_target(teacher_logits,targets)
        if multilabel:
            p=calibrated_teacher.sigmoid().clamp(1e-7,1-1e-7);entropy=(-(p*p.log()+(1-p)*(1-p).log())/torch.log(torch.tensor(2.,device=p.device))).mean(1)
            disagreement=probability_disagreement(compressed_logits,calibrated_teacher,self.disagreement_method,multilabel=True)
            reduction_dims=tuple(range(2,targets.ndim))
            if class_frequency is None:
                frequencies=targets.float().mean(0).view(1,-1).expand(targets.shape[0],-1) if self.task_type=="classification" else targets.float().mean(reduction_dims)
            else:
                frequencies=torch.tensor([class_frequency.get(i,1.) for i in range(targets.shape[1])],device=targets.device).view(1,-1).expand(targets.shape[0],-1)
            maximum=frequencies.amax(1,keepdim=True);minimum=frequencies.amin(1,keepdim=True)
            inverse=(maximum-frequencies)/(maximum-minimum).clamp_min(1e-8)
            weights=inverse.view(targets.shape[0],targets.shape[1],*([1]*(targets.ndim-2)))
            weighted=targets.float()*weights
            rarity=weighted.sum(1)/targets.float().sum(1).clamp_min(1) if self.task_type=="classification" else weighted.amax(1)
        else:
            entropy = normalized_entropy(calibrated_teacher);disagreement = probability_disagreement(compressed_logits, calibrated_teacher, self.disagreement_method)
            if self.task_type.startswith("segmentation") and class_frequency is None:
                rarity=torch.stack([class_rarity_score(sample.unsqueeze(0))[0] for sample in targets])
            else:
                rarity = class_rarity_score(targets, class_frequency)
        if static_outputs is not None:
            if self.task_type == "classification":raise ValueError("Static risk factors are only valid for segmentation targets.")
            rarity=static_outputs["class_rarity"].to(device=teacher_logits.device,dtype=torch.float32)
        if self.task_type == "classification":
            margin = (1-(calibrated_teacher.sigmoid()-.5).abs()*2).mean(1) if multilabel else low_confidence_margin(calibrated_teacher)
            raw = self.alpha_entropy * entropy + self.alpha_disagreement * disagreement + self.alpha_class * rarity + self.alpha_margin * margin
            risk = torch.sigmoid(raw)
            zeros = torch.zeros_like(risk)
            components: dict[str, Any] = {"components": 0, "small_components": 0}
            boundary, small = zeros, zeros
            component_weight=torch.ones_like(risk)
        else:
            if static_outputs is not None:
                boundary=static_outputs["boundary_score"].to(device=teacher_logits.device,dtype=torch.float32)
                small=static_outputs["small_component_score"].to(device=teacher_logits.device,dtype=torch.float32)
                component_weight=static_outputs.get("small_component_weight",torch.ones_like(small)).to(device=teacher_logits.device,dtype=torch.float32)
                components={"components":0,"small_components":0}
            elif multilabel:
                boundaries=[];small_maps=[];component_weights=[];component_rows=[]
                for channel in range(targets.shape[1]):
                    channel_target=targets[:,channel]
                    boundaries.append(boundary_risk_map(channel_target))
                    channel_small,channel_weight,channel_stats=small_component_maps(channel_target,self.small_component_threshold,self.max_component_weight)
                    small_maps.append(channel_small);component_weights.append(channel_weight);component_rows.append(channel_stats)
                boundary=torch.stack(boundaries).amax(0);small=torch.stack(small_maps).amax(0)
                component_weight=torch.stack(component_weights).amax(0)
                sizes=[size for row in component_rows for size in row["component_sizes"]]
                components={"components":sum(row["components"] for row in component_rows),"small_components":sum(row["small_components"] for row in component_rows),"component_sizes":sizes}
                components["small_component_fraction"]=components["small_components"]/max(components["components"],1)
            else:
                boundary = boundary_risk_map(targets)
                small,component_weight,components=small_component_maps(targets,self.small_component_threshold,self.max_component_weight)
            raw = self.alpha_boundary * boundary + self.alpha_small * small + self.alpha_entropy * entropy + self.alpha_disagreement * disagreement + self.alpha_class * rarity
            risk = torch.sigmoid(raw)
        summary = {
            "mean_risk": float(risk.mean().detach()),
            "high_risk_ratio": float((risk >= self.high_risk_threshold).float().mean().detach()),
            "small_lesion_ratio": float((small > 0).float().mean().detach()),
            "boundary_ratio": float((boundary > 0.5).float().mean().detach()),
            "high_entropy_ratio": float((entropy > 0.7).float().mean().detach()),
            "high_disagreement_ratio": float((disagreement > 0.3).float().mean().detach()),
            **{k: v for k, v in components.items() if k != "component_sizes"},
        }
        return {"risk_map": risk, "boundary_score": boundary, "small_component_score": small,
                "small_component_weight":component_weight,
                "teacher_entropy": entropy, "disagreement": disagreement, "class_rarity": rarity,
                "risk_summary": summary, "object_statistics": components}
