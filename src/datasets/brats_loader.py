"""BraTS 2018 raw multimodal adapter used by the validated 3D task."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from .volume_utils import (
    read_medical_volume,
    resize_3d_image,
    resize_3d_label,
    zscore_nonzero,
)


class BraTSDataset(Dataset):
    """Load the BraTS 2018 HGG/LGG cases with native nested-region targets."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        in_channels: int = 4,
        image_size: Sequence[int] | None = None,
        dataset_version: str = "2018",
        target_mode: str = "classes",
        split_counts: tuple[int, int, int] | None = None,
        split_seed: int = 123,
        crop_size: Sequence[int] | None = None,
        crop_training: bool = False,
        crop_positive_probability: float = 0.8,
        crop_region_weights: Sequence[float] | None = None,
    ) -> None:
        root = Path(root)
        self.in_channels = in_channels
        self.image_size = list(image_size) if image_size is not None else None
        self.dataset_version = str(dataset_version)
        if self.dataset_version != "2018":
            raise ValueError("The retained BraTS adapter supports dataset_version=2018 only.")
        self.target_mode = str(target_mode)
        self.split=str(split)
        self.crop_size=tuple(int(value) for value in crop_size) if crop_size is not None else None
        self.crop_training=bool(crop_training)
        self.crop_positive_probability=float(crop_positive_probability)
        self.crop_region_weights=tuple(float(value) for value in crop_region_weights) if crop_region_weights is not None else None
        case_dirs = self._find_2018_cases(root)
        self.samples = self._split_2018(case_dirs, split, split_counts, split_seed)
        self.case_ids = [path.name for path in self.samples]
        if not self.samples:
            raise FileNotFoundError("No labeled BraTS cases were found for the requested split")

    @staticmethod
    def _find_2018_cases(root: Path) -> list[Path]:
        training = root / "MICCAI_BraTS_2018_Data_Training"
        candidates = sorted([*training.glob("HGG/Brats18_*"), *training.glob("LGG/Brats18_*")])
        cases = [path for path in candidates if (path / f"{path.name}_seg.nii").exists()]
        if not cases:
            raise FileNotFoundError("No labeled BraTS 2018 HGG/LGG cases were found")
        return cases

    @staticmethod
    def _split_2018(
        cases: list[Path],
        split: str,
        split_counts: tuple[int, int, int] | None,
        seed: int,
    ) -> list[Path]:
        counts = split_counts or (200, 42, 43)
        if sum(counts) != len(cases):
            raise ValueError(f"BraTS split counts {counts} do not match {len(cases)} labeled cases")
        rng = np.random.RandomState(int(seed))
        train_cases, remainder = train_test_split(cases, train_size=counts[0], random_state=rng)
        val_cases, test_cases = train_test_split(remainder, train_size=counts[1], random_state=rng)
        partitions = {"train": train_cases, "val": val_cases, "test": test_cases}
        if split not in partitions:
            raise ValueError(f"Unsupported split: {split}")
        return list(partitions[split])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        case_dir = self.samples[index]
        case_id = case_dir.name
        images = [
            torch.from_numpy(
                np.asarray(read_medical_volume(case_dir / f"{case_id}_{modality}.nii"))
            )
            for modality in ("t1ce", "t1", "t2", "flair")[: self.in_channels]
        ]
        image = resize_3d_image(zscore_nonzero(torch.stack(images, dim=0).float()), self.image_size)
        label = torch.from_numpy(
            np.asarray(read_medical_volume(case_dir / f"{case_id}_seg.nii"))
        ).long()
        label = resize_3d_label(label, self.image_size)
        if self.target_mode == "regions":
            target = torch.stack(
                [
                    (label == 1) | (label == 4),
                    label > 0,
                    label == 4,
                ],
                dim=0,
            ).float()
        else:
            target = label
        if self.split=="train" and self.crop_training and self.crop_size is not None:
            image,target=self._training_crop(image,target)
        return {"image": image, "target": target, "id": case_id}

    def _training_crop(self,image:torch.Tensor,target:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        spatial=image.shape[1:]
        padding=[]
        for current,required in reversed(list(zip(spatial,self.crop_size))):
            padding.extend((0,max(required-current,0)))
        if any(padding):
            image=F.pad(image,padding);target=F.pad(target,padding)
        spatial=image.shape[1:]
        foreground=target.bool().any(0) if target.ndim==image.ndim else target>0
        if foreground.any() and torch.rand(())<self.crop_positive_probability:
            selected=foreground
            if self.crop_region_weights is not None and target.ndim==image.ndim:
                if len(self.crop_region_weights)!=target.shape[0]:
                    raise ValueError(f"Expected {target.shape[0]} crop region weights, got {len(self.crop_region_weights)}")
                available=torch.tensor([bool(channel.any()) for channel in target],dtype=torch.bool)
                weights=torch.tensor(self.crop_region_weights,dtype=torch.float32)*available
                if weights.sum()>0:selected=target[torch.multinomial(weights/weights.sum(),1).item()].bool()
            coordinates=selected.nonzero()
            center=coordinates[torch.randint(len(coordinates),(1,)).item()]
        else:
            center=torch.tensor([torch.randint(size,(1,)).item() for size in spatial])
        starts=[max(0,min(int(c)-size//2,current-size)) for c,current,size in zip(center,spatial,self.crop_size)]
        slices=tuple(slice(start,start+size) for start,size in zip(starts,self.crop_size))
        return image[(slice(None),*slices)],target[(slice(None),*slices)]
