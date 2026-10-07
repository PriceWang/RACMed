"""Distance-transform boundary risk maps for 2D and 3D labels."""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


def _labels(targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim >= 4 and targets.shape[1] == 1:
        targets = targets[:, 0]
    return targets.long()


def boundary_risk_map(targets: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """Return an exponentially decaying score around all class boundaries."""
    device = targets.device
    labels = _labels(targets).detach().cpu().numpy()
    scores: list[np.ndarray] = []
    structure = ndimage.generate_binary_structure(labels.ndim - 1, 1)
    for sample in labels:
        boundary = np.zeros_like(sample, dtype=bool)
        for cls in np.unique(sample):
            region = sample == cls
            if not region.any():
                continue
            eroded = ndimage.binary_erosion(region, structure=structure, border_value=0)
            boundary |= region ^ eroded
        distance = ndimage.distance_transform_edt(~boundary)
        scores.append(np.exp(-distance / max(float(sigma), 1e-6)).astype(np.float32))
    return torch.from_numpy(np.stack(scores)).to(device=device)


def soft_prediction_boundary(probabilities: torch.Tensor) -> torch.Tensor:
    """Compute a differentiable morphological gradient of class probabilities."""
    dims = probabilities.ndim - 2
    if dims == 2:
        high = torch.nn.functional.max_pool2d(probabilities, 3, 1, 1)
        low = -torch.nn.functional.max_pool2d(-probabilities, 3, 1, 1)
    elif dims == 3:
        high = torch.nn.functional.max_pool3d(probabilities, 3, 1, 1)
        low = -torch.nn.functional.max_pool3d(-probabilities, 3, 1, 1)
    else:
        raise ValueError(f"Expected 2D/3D probabilities, got {probabilities.shape}")
    return (high - low).amax(dim=1).clamp(0, 1)
