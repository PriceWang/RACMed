"""Shared model invocation for image-only and prompt-aware networks."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


def _accepts_prompts(model: nn.Module) -> bool:
    base = model.module if hasattr(model, "module") else model
    return bool(getattr(base, "accepts_prompts", False))


def forward_model(
    model: nn.Module,
    images: torch.Tensor,
    batch: Mapping[str, object] | None = None,
    return_features: bool = True,
) -> dict:
    """Invoke a model while forwarding protocol inputs only when supported."""

    if _accepts_prompts(model):
        if batch is None or "boxes" not in batch:
            raise ValueError("Prompt-aware model requires batch['boxes'].")
        boxes = batch["boxes"]
        valid = batch.get("box_valid")
        if not isinstance(boxes, torch.Tensor):
            raise TypeError("batch['boxes'] must be a tensor")
        asynchronous=images.device.type=="cuda"
        boxes = boxes.to(images.device,non_blocking=asynchronous)
        valid = valid.to(images.device,non_blocking=asynchronous) if isinstance(valid, torch.Tensor) else valid
        return model(images, boxes=boxes, box_valid=valid, return_features=return_features)
    return model(images, return_features=return_features)
