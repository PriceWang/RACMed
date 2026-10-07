"""Generic paired-image 2D segmentation dataset."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from .volume_utils import split_indices
from .prompt_boxes import boxes_from_mask

EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.npy")


def _files(folder: Path) -> list[Path]:
    return sorted({p for pattern in EXTENSIONS for p in folder.glob(pattern)})


def _read(path: Path) -> np.ndarray:
    return np.load(path) if path.suffix.lower() == ".npy" else np.asarray(Image.open(path))


class Generic2DSegmentationDataset(Dataset):
    """Read images and basename-matched masks from the documented directory contract."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        in_channels: int = 1,
        num_classes: int | None = None,
        image_size: list[int] | tuple[int, int] | None = None,
        prompt_boxes: bool = False,
        box_perturb: int = 0,
        seed: int = 0,
        split_policy: str = "default",
        binary_mask_threshold: float | None = None,
    ) -> None:
        root = Path(root)
        base = root / split if (root / split / "images").exists() else root
        self.images = _files(base / "images")
        masks = _files(base / "masks"); by_stem = {p.stem: p for p in masks}
        self.pairs = [(p, by_stem[p.stem]) for p in self.images if p.stem in by_stem]
        if base == root and self.pairs and split_policy != "all_labeled":
            indices = set(split_indices(len(self.pairs), split, val_fraction=0.15, test_fraction=0.15))
            self.pairs = [pair for idx, pair in enumerate(self.pairs) if idx in indices]
        if not self.pairs: raise FileNotFoundError(f"No paired 2D data under {base}/images and {base}/masks")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.prompt_boxes = bool(prompt_boxes)
        self.box_perturb = int(box_perturb)
        self.seed = int(seed)
        self.binary_mask_threshold = binary_mask_threshold

    def __len__(self): return len(self.pairs)

    def __getitem__(self, index):
        ip, mp = self.pairs[index]; image, mask = _read(ip), _read(mp)
        if image.ndim == 2: image = image[..., None]
        image = image[..., :self.in_channels]
        if image.shape[-1] < self.in_channels: image = np.repeat(image, self.in_channels, -1)
        image = image.astype(np.float32); image = image / max(float(image.max()), 1.)
        if mask.ndim == 3: mask = mask[..., 0]
        mask = _normalize_mask(mask, self.num_classes, self.binary_mask_threshold)
        image_tensor = torch.from_numpy(image.transpose(2,0,1).copy())
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        if self.image_size is not None:
            target = tuple(int(v) for v in self.image_size)
            image_tensor = F.interpolate(image_tensor.unsqueeze(0).float(), size=target, mode="bilinear", align_corners=False).squeeze(0)
            mask_tensor = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0).float(), size=target, mode="nearest").squeeze(0).squeeze(0).long()
        result = {"image": image_tensor, "target": mask_tensor, "id": ip.stem}
        if self.prompt_boxes:
            generator = torch.Generator().manual_seed(self.seed + index)
            boxes, valid = boxes_from_mask(
                mask_tensor,
                max(int(self.num_classes or int(mask_tensor.max()) + 1) - 1, 1),
                self.box_perturb,
                generator,
            )
            result.update({"boxes": boxes, "box_valid": valid})
        return result


def _normalize_mask(
    mask: np.ndarray,
    num_classes: int | None,
    binary_mask_threshold: float | None = None,
) -> np.ndarray:
    """Normalize labels while rejecting near-black lossy JPEG artifacts."""

    mask = np.asarray(mask)
    if num_classes == 2 and mask.max(initial=0) > 1:
        threshold = binary_mask_threshold
        if threshold is None:
            maximum = float(mask.max(initial=0))
            threshold = 0.5 if maximum <= 1.0 else maximum / 2.0
        return (mask > float(threshold)).astype(np.int64)
    if num_classes == 2 and np.issubdtype(mask.dtype, np.floating):
        threshold = 0.5 if binary_mask_threshold is None else float(binary_mask_threshold)
        return (mask > threshold).astype(np.int64)
    return mask.astype(np.int64)
