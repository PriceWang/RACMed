"""Adapters for MONAI model-zoo bundles stored under ``external_models``."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .paths import resolve_model_path


class BraTSSegResNetBundleAdapter(nn.Module):
    """Expose the MONAI BraTS bundle's native TC/WT/ET sigmoid logits."""

    source = "monai_bundle"

    def __init__(
        self,
        model_dir: str | Path | None = None,
        num_classes: int = 3,
        in_channels: int = 4,
        require_local_weights: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 4:
            raise ValueError("The BraTS MONAI bundle expects four MRI modalities.")
        if num_classes != 3:
            raise ValueError("The BraTS MONAI bundle has exactly three region outputs: TC, WT, and ET.")
        try:
            from monai.networks.nets import SegResNet
        except ImportError as exc:  # pragma: no cover - monai is a project dependency
            raise ImportError("Install monai to use source=monai_bundle") from exc

        self.backbone = SegResNet(
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,
        )
        self.num_classes = int(num_classes)
        directory = resolve_model_path(model_dir, "brats_mri_segmentation")
        checkpoint = directory / "models" / "model.pt"
        if require_local_weights and not checkpoint.exists():
            raise FileNotFoundError(
                f"BraTS MONAI bundle checkpoint is missing: {checkpoint}. "
                "Place the bundle under the configured model root or set require_local_weights=false."
            )
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            result = self.backbone.load_state_dict(state.get("model", state), strict=False)
            if len(result.missing_keys) > 0:
                raise RuntimeError(f"BraTS bundle load left missing keys: {result.missing_keys[:8]}")

    def forward(self, x: torch.Tensor, return_features: bool = True) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        logits = self.backbone(x)
        features = {"bottleneck": logits} if return_features else {}
        return {"logits": logits, "features": features}


def create_monai_bundle_model(cfg: dict, num_classes: int, in_channels: int) -> nn.Module:
    """Create a supported local MONAI bundle teacher."""

    name = str(cfg.get("name", "brats_segresnet")).lower()
    if name in {"brats_segresnet", "brats_mri_segmentation", "segresnet_brats"}:
        return BraTSSegResNetBundleAdapter(
            model_dir=cfg.get("model_dir") or cfg.get("local_pretrained_path"),
            num_classes=int(cfg.get("num_classes", num_classes)),
            in_channels=int(cfg.get("in_channels", in_channels)),
            require_local_weights=bool(cfg.get("require_local_weights", True)),
        )
    raise ValueError(f"Unsupported MONAI bundle model: {name}")
