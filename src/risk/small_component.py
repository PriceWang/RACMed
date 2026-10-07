"""Connected-component small-lesion scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy import ndimage


def small_component_maps(
    targets: torch.Tensor, threshold: int = 64, max_weight: float = 16.0
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return small-component risk and bounded inverse-size supervision weights."""

    if max_weight < 1:
        raise ValueError("max_weight must be at least 1.0.")
    device = targets.device
    if targets.ndim >= 4 and targets.shape[1] == 1:
        targets = targets[:, 0]
    array = targets.detach().cpu().numpy().astype(np.int64)
    output = np.zeros_like(array, dtype=np.float32)
    weight = np.ones_like(array, dtype=np.float32)
    sizes: list[int] = []
    small = 0
    structure = ndimage.generate_binary_structure(array.ndim - 1, 1)
    for b, sample in enumerate(array):
        for cls in np.unique(sample):
            if cls == 0:
                continue
            components, count = ndimage.label(sample == cls, structure=structure)
            for index in range(1, count + 1):
                region = components == index
                size = int(region.sum())
                sizes.append(size)
                if size < threshold:
                    small += 1
                    output[b][region] = max(output[b][region].max(initial=0), 1.0 - size / max(threshold, 1) * 0.5)
                    normalized = min(float(max_weight), float(threshold) / max(size, 1))
                    weight[b][region] = np.maximum(weight[b][region], normalized)
    stats: dict[str, Any] = {
        "components": len(sizes),
        "small_components": small,
        "component_sizes": sizes,
        "small_component_fraction": small / max(len(sizes), 1),
    }
    return torch.from_numpy(output).to(device), torch.from_numpy(weight).to(device), stats


def small_component_risk(
    targets: torch.Tensor, threshold: int = 64
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Mark foreground components below ``threshold`` pixels/voxels as risky."""

    risk, _, stats = small_component_maps(targets, threshold)
    return risk, stats
