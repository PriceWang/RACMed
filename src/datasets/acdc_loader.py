"""ACDC raw-format adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .volume_utils import (
    read_medical_volume,
    scalar_volume_to_dhw,
    zscore_nonzero,
)
from .prompt_boxes import boxes_from_mask


class ACDCSliceDataset(Dataset):
    """Expose labeled ACDC ED/ES volumes as foreground 2D prompted slices."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: Sequence[int] | None = None,
        split_counts: tuple[int, int, int] | None = None,
        box_perturb: int = 0,
        seed: int = 0,
        split_policy: str = "official_holdout",
        target_mode: str = "classes",
        region_class_ids: Sequence[int] = (1,2,3),
    ) -> None:
        root = Path(root)
        self.image_size = tuple(int(value) for value in image_size) if image_size else None
        self.box_perturb = int(box_perturb)
        self.seed = int(seed)
        self.target_mode=str(target_mode)
        if self.target_mode not in {"classes","regions"}:raise ValueError("ACDC target_mode must be classes or regions.")
        self.region_class_ids=tuple(int(value) for value in region_class_ids)
        self._volume_cache:dict[str,tuple[torch.Tensor,torch.Tensor]]={}
        if not self.region_class_ids or any(value not in {1,2,3} for value in self.region_class_ids):raise ValueError("ACDC region_class_ids must select labels 1, 2, or 3.")
        training = sorted(path for path in (root / "training").glob("patient*") if path.is_dir())
        testing = sorted(path for path in (root / "testing").glob("patient*") if path.is_dir())
        if split_policy == "all_labeled":
            patient_dirs = training + testing
        elif split_counts is not None:
            if sum(split_counts) != len(training):
                raise ValueError("ACDC split_counts must match the number of training patients")
            boundaries = (split_counts[0], split_counts[0] + split_counts[1])
            patient_dirs = {
                "train": training[: boundaries[0]],
                "val": training[boundaries[0] : boundaries[1]],
                "test": training[boundaries[1] :],
            }[split]
        elif split == "test" and testing:
            patient_dirs = testing
        else:
            cut = max(1, round(len(training) * 0.8))
            patient_dirs = training[:cut] if split == "train" else training[cut:]
        self.samples: list[dict[str, Any]] = []
        for patient_dir in patient_dirs:
            for label_path in sorted(patient_dir.glob("*_gt.nii.gz")):
                image_path = patient_dir / label_path.name.replace("_gt.nii.gz", ".nii.gz")
                if not image_path.exists():
                    continue
                label_volume = scalar_volume_to_dhw(read_medical_volume(label_path))
                for slice_index in np.flatnonzero((label_volume > 0).reshape(label_volume.shape[0], -1).any(1)):
                    self.samples.append(
                        {
                            "image": image_path,
                            "label": label_path,
                            "slice": int(slice_index),
                            "id": f"{image_path.name.removesuffix('.nii.gz')}_slice{slice_index:03d}",
                        }
                    )
        if not self.samples:
            raise FileNotFoundError("No foreground ACDC slices were found")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        cache_key=str(sample["image"])
        if cache_key not in self._volume_cache:
            image_volume=torch.from_numpy(scalar_volume_to_dhw(read_medical_volume(sample["image"]))).unsqueeze(0)
            image_volume=zscore_nonzero(image_volume)
            label_volume=torch.from_numpy(scalar_volume_to_dhw(read_medical_volume(sample["label"]))).long()
            self._volume_cache[cache_key]=(image_volume,label_volume)
        image_volume,label_volume=self._volume_cache[cache_key]
        image = image_volume[:, sample["slice"]]
        target = label_volume[sample["slice"]]
        foreground=image!=0
        if foreground.any():
            low=image[foreground].amin();high=image[foreground].amax()
            image=torch.where(foreground,(image-low)/(high-low).clamp_min(1e-8),torch.zeros_like(image))
        if self.image_size is not None:
            image = F.interpolate(
                image.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False
            ).squeeze(0)
            target = F.interpolate(
                target[None, None].float(), size=self.image_size, mode="nearest"
            ).squeeze(0).squeeze(0).long()
        generator = torch.Generator().manual_seed(self.seed + index)
        all_boxes,all_valid = boxes_from_mask(target,3,self.box_perturb,generator)
        selection=torch.tensor([class_id-1 for class_id in self.region_class_ids])
        boxes,valid=all_boxes[selection],all_valid[selection]
        output_target=torch.stack([target==class_id for class_id in self.region_class_ids]).float() if self.target_mode=="regions" else target
        return {
            "image": image.float(),
            "target": output_target,
            "boxes": boxes,
            "box_valid": valid,
            "id": sample["id"],
            "group_id": sample["image"].name.removesuffix(".nii.gz"),
        }
