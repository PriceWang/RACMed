"""Bounding-box prompts derived from segmentation masks."""

from __future__ import annotations

import torch


def boxes_from_mask(
    mask: torch.Tensor,
    num_foreground_classes: int,
    perturb: int = 0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one ``xyxy`` box and validity flag per foreground class.

    Coordinates use the original mask coordinate system. Each valid side is
    independently expanded by an integer sampled from ``[0, perturb]`` and
    clipped to the image bounds, matching MedSAM's simulated-box protocol.
    """

    if mask.ndim != 2:
        raise ValueError(f"Prompt boxes require a 2D mask, got shape {tuple(mask.shape)}")
    if num_foreground_classes < 1:
        raise ValueError("num_foreground_classes must be positive")
    height, width = mask.shape
    boxes = torch.zeros(num_foreground_classes, 4, dtype=torch.float32, device=mask.device)
    valid = torch.zeros(num_foreground_classes, dtype=torch.bool, device=mask.device)
    max_perturb = max(int(perturb), 0)
    for index, class_id in enumerate(range(1, num_foreground_classes + 1)):
        points = torch.nonzero(mask == class_id, as_tuple=False)
        if points.numel() == 0:
            continue
        y0, x0 = points.amin(0).tolist()
        y1, x1 = points.amax(0).tolist()
        if max_perturb:
            offsets = torch.randint(0, max_perturb + 1, (4,), generator=generator).tolist()
        else:
            offsets = [0, 0, 0, 0]
        coordinates=[
            max(int(x0) - offsets[0], 0),
            max(int(y0) - offsets[1], 0),
            min(int(x1) + offsets[2], width - 1),
            min(int(y1) + offsets[3], height - 1),
        ]
        boxes[index] = torch.tensor(
            coordinates,
            dtype=torch.float32,
            device=mask.device,
        )
        valid[index] = True
    return boxes, valid
