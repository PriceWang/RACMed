"""Model factory for validated external teachers and compressed models."""

from __future__ import annotations
import logging
from collections.abc import Mapping
from pathlib import Path
import torch

from .classification import CompactClassifier
from .segmentation2d import CompactUNet2D
from .segmentation3d import CompactUNet3D

FALLBACK_MESSAGE="Pretrained weights unavailable. Falling back to random initialization."


def _state_dict_from_checkpoint(checkpoint: object) -> Mapping[str, torch.Tensor]:
    """Extract a tensor state dict from common PyTorch checkpoint formats."""

    if isinstance(checkpoint, Mapping):
        for key in ("model", "state_dict", "net", "network"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
        return checkpoint
    raise TypeError(f"Unsupported checkpoint object type: {type(checkpoint)!r}")


def _strip_prefix(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key.removeprefix(prefix): value for key, value in state.items()}


def _add_prefix(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {f"{prefix}{key}": value for key, value in state.items()}


def _matching_key_count(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> int:
    model_state = model.state_dict()
    return sum(
        1
        for key, value in state.items()
        if key in model_state and hasattr(value, "shape") and tuple(model_state[key].shape) == tuple(value.shape)
    )


def _load_checkpoint_flexible(model: torch.nn.Module, path: Path) -> None:
    """Load a checkpoint and fail loudly when no compatible tensor is loaded."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw_state = dict(_state_dict_from_checkpoint(checkpoint))
    candidates: list[dict[str, torch.Tensor]] = [raw_state]
    if any(key.startswith("module.") for key in raw_state):
        candidates.append(_strip_prefix(raw_state, "module."))
    if hasattr(model, "backbone"):
        candidates.extend([_add_prefix(state, "backbone.") for state in list(candidates)])
    best = max(candidates, key=lambda state: _matching_key_count(model, state))
    loaded = _matching_key_count(model, best)
    if loaded == 0:
        sample = list(raw_state)[:8]
        raise RuntimeError(f"No compatible tensors were loaded from {path}. Sample checkpoint keys: {sample}")
    result = model.load_state_dict(best, strict=False)
    logging.info(
        "Loaded %d tensors from %s with %d missing and %d unexpected keys.",
        loaded,
        path,
        len(result.missing_keys),
        len(result.unexpected_keys),
    )


def create_model(config: dict, role: str | None = None):
    """Create a model and load an optional local checkpoint with safe pretrained fallback."""
    base=config.get("model",config); cfg={**base,**(config.get(role,{}) if role else {})}
    task=cfg.get("task_type",config.get("task_type","classification")); source=cfg.get("source","custom"); name=cfg.get("name","compact")
    inc=int(cfg.get("in_channels",base.get("in_channels",1))); classes=int(cfg.get("num_classes",base.get("num_classes",3))); pretrained=bool(cfg.get("pretrained",False))
    try:
        if source=="timm":
            if task != "classification":
                raise ValueError("The validated timm path is classification-only.")
            from .timm_wrappers import create_timm_model
            model=create_timm_model(name,pretrained,inc,classes)
        elif source=="torchxrayvision":
            from .torchxrayvision_wrappers import create_torchxrayvision_model
            model=create_torchxrayvision_model(cfg,classes,inc)
        elif source=="medsam":
            from .medsam_wrappers import create_medsam_model
            model=create_medsam_model(cfg,classes,inc)
        elif source=="monai_bundle":
            from .monai_bundle_wrappers import create_monai_bundle_model
            model=create_monai_bundle_model(cfg, classes, inc)
        elif source=="custom":
            if task=="classification": model=CompactClassifier(inc,classes,int(cfg.get("width",12)),int(cfg.get("spatial_dims",2)))
            elif task=="segmentation3d":
                if name not in {"compact_unet3d", "compact_unet"}:
                    raise ValueError(f"Unsupported custom 3D segmentation model: {name}. Use compact_unet3d.")
                model=CompactUNet3D(inc,classes,tuple(cfg.get("channels",[4,8])),str(cfg.get("norm","batch")))
            else:
                if name not in {"compact_unet2d", "compact_unet"}:
                    raise ValueError(f"Unsupported custom 2D segmentation model: {name}. Use compact_unet2d.")
                model=CompactUNet2D(
                    inc,
                    classes,
                    tuple(cfg.get("channels",[8,16])),
                    prompt_aware=bool(cfg.get("prompt_aware",False)),
                    prompt_channels=cfg.get("prompt_channels"),
                    norm=str(cfg.get("norm","batch")),
                )
        else: raise ValueError(f"Unsupported model source: {source}")
    except Exception as exc:
        if bool(cfg.get("require_local_weights", False)):
            raise
        if not pretrained: raise
        logging.warning("%s Cause: %s",FALLBACK_MESSAGE,exc)
        retry={**config}; retry[role or "model"]={**cfg,"pretrained":False}
        return create_model(retry,role)
    checkpoint=cfg.get("checkpoint") or cfg.get("local_pretrained_path")
    if checkpoint:
        path=Path(checkpoint)
        if not path.exists(): raise FileNotFoundError(f"Model checkpoint not found: {path}")
        _load_checkpoint_flexible(model,path)
    return model
