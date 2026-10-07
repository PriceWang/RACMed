"""TorchXRayVision model wrappers for chest X-ray teacher models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .paths import resolve_model_path


def _weight_filename(models_module, weights: str) -> str:
    if weights not in models_module.model_urls:
        valid = sorted(k for k in models_module.model_urls if k.startswith("densenet"))
        raise ValueError(f"Unsupported TorchXRayVision weights '{weights}'. Valid DenseNet weights: {valid}")
    return os.path.basename(models_module.model_urls[weights]["weights_url"])


def _pathology_index(targets: Sequence[str], pathology: str) -> int:
    normalized = [str(item).strip().lower() for item in targets]
    key = pathology.strip().lower()
    if key not in normalized:
        raise ValueError(f"Pathology '{pathology}' not found in TorchXRayVision targets: {list(targets)}")
    return normalized.index(key)


class TorchXRayVisionAdapter(nn.Module):
    """Adapt a TorchXRayVision multi-label model to RACMed's output contract."""

    source = "torchxrayvision"

    def __init__(
        self,
        weights: str = "densenet121-res224-rsna",
        cache_dir: str | Path | None = None,
        pathology: str = "Pneumonia",
        num_classes: int = 2,
        in_channels: int = 1,
        require_local_weights: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        try:
            import torchxrayvision.models as xrv_models
        except ImportError as exc:
            raise ImportError("Install torchxrayvision to use source=torchxrayvision") from exc

        if in_channels != 1:
            raise ValueError("TorchXRayVision chest X-ray models expect single-channel inputs.")
        if num_classes not in (1, 2):
            raise ValueError("TorchXRayVision pathology adapter supports num_classes=1 or 2.")

        cache = resolve_model_path(cache_dir, "torchxrayvision")
        expected = cache / _weight_filename(xrv_models, weights)
        if require_local_weights and not expected.exists():
            raise FileNotFoundError(
                f"TorchXRayVision weight file is missing: {expected.name}. "
                "Place it under the configured cache_dir or set require_local_weights=false to allow torchxrayvision download."
            )

        self.backbone = xrv_models.DenseNet(weights=weights, cache_dir=str(cache), apply_sigmoid=False)
        self.pathology = pathology
        self.pathology_index = _pathology_index(getattr(self.backbone, "targets", []), pathology)
        self.num_classes = num_classes
        self.normalize = normalize
        self.input_resolution = int(getattr(self.backbone, "input_resolution", 224))

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        if x.shape[-2:] != (self.input_resolution, self.input_resolution):
            x = F.interpolate(
                x,
                size=(self.input_resolution, self.input_resolution),
                mode="bilinear",
                align_corners=False,
            )
        if self.normalize:
            x = x.clamp(0, 1) * 2048.0 - 1024.0
        return x

    def forward(self, x: torch.Tensor, return_features: bool = True) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        prepared = self._prepare(x)
        # XRV restores per-pathology operating thresholds with its checkpoints.
        # Its public ``forward`` therefore applies sigmoid and ``op_norm`` even
        # when ``apply_sigmoid=False``.  AUC is unchanged by that monotone
        # transform, but treating the returned probabilities as logits corrupts
        # entropy gating and KD.  Use the official feature/classifier path to
        # retain the exact backbone while exposing genuine raw logits.
        representation = self.backbone.features2(prepared)
        raw = self.backbone.classifier(representation)
        positive = raw[:, self.pathology_index : self.pathology_index + 1]
        logits = positive if self.num_classes == 1 else torch.cat([torch.zeros_like(positive), positive], dim=1)
        features = {"bottleneck": representation} if return_features else {}
        return {"logits": logits, "features": features}


def create_torchxrayvision_model(cfg: dict, num_classes: int, in_channels: int) -> TorchXRayVisionAdapter:
    """Factory helper for ``source=torchxrayvision``."""

    return TorchXRayVisionAdapter(
        weights=cfg.get("weights", cfg.get("name", "densenet121-res224-rsna")),
        cache_dir=cfg.get("cache_dir") or cfg.get("local_pretrained_path"),
        pathology=cfg.get("pathology", "Pneumonia"),
        num_classes=num_classes,
        in_channels=in_channels,
        require_local_weights=bool(cfg.get("require_local_weights", True)),
        normalize=bool(cfg.get("xrv_normalize", True)),
    )
