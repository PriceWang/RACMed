"""Deterministic synthetic medical images for offline pipeline checks."""

from __future__ import annotations
from typing import Any
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from .prompt_boxes import boxes_from_mask


class SyntheticMedicalDataset(Dataset):
    """Generate organs, small lesions, vessels, blurred boundaries, and labels."""

    def __init__(self, length: int = 16, task_type: str = "segmentation2d",
                 image_size: list[int] | tuple[int, ...] = (64, 64),
                 in_channels: int = 1, num_classes: int = 3, seed: int = 0,
                 prompt_channels: int | None = None) -> None:
        self.length, self.task_type = int(length), task_type
        self.image_size, self.in_channels, self.num_classes = tuple(image_size), in_channels, num_classes
        self.seed = seed
        self.prompt_channels = int(prompt_channels or max(num_classes - 1, 1))
        if task_type == "segmentation3d" and len(self.image_size) == 2:
            self.image_size = (16, *self.image_size)

    def __len__(self) -> int:
        return self.length

    def _sample(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + index)
        shape = self.image_size
        grid = np.ogrid[tuple(slice(0, n) for n in shape)]
        center = np.array(shape) / 2 + rng.uniform(-.05, .05, len(shape)) * np.array(shape)
        radius = np.array(shape) * .28
        organ = sum(((g-center[d])/radius[d])**2 for d, g in enumerate(grid)) <= 1
        mask = np.zeros(shape, np.int64); mask[organ] = 1
        lesion_center = center + np.array(shape) * .12
        lesion_radius = max(2, int(min(shape) * .07))
        lesion = sum((g-lesion_center[d])**2 for d, g in enumerate(grid)) <= lesion_radius**2
        mask[lesion] = min(2, self.num_classes - 1)
        # One-pixel/voxel curved vessel-like foreground evidence.
        if len(shape) == 2:
            y = np.arange(shape[0]); x = (shape[1] * .25 + np.sin(y/5) * shape[1]*.08).astype(int)
            valid = (x >= 0) & (x < shape[1]); mask[y[valid], x[valid]] = min(2, self.num_classes-1)
        else:
            z = np.arange(shape[0]); y = (shape[1] * .25 + np.sin(z/2) * 2).astype(int)
            x = np.full_like(z, shape[2]//3); mask[z, np.clip(y,0,shape[1]-1), x] = min(2,self.num_classes-1)
        intensity = .15 + .55*(mask == 1) + .8*(mask == 2)
        intensity = ndimage.gaussian_filter(intensity.astype(np.float32), sigma=.9)
        intensity += rng.normal(0, .06, shape).astype(np.float32)
        channels = np.stack([intensity * (1 + .05*c) for c in range(self.in_channels)])
        return channels.astype(np.float32), mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, mask = self._sample(index)
        if self.task_type == "classification":
            target = torch.tensor(index % self.num_classes, dtype=torch.long)
        else:
            target = torch.from_numpy(mask)
        sample = {"image": torch.from_numpy(image), "target": target, "id": f"synthetic_{index:04d}"}
        if self.task_type == "segmentation2d":
            boxes, valid = boxes_from_mask(target, self.prompt_channels)
            sample.update({"boxes": boxes, "box_valid": valid})
        return sample
