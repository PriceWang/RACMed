"""Shared 3D medical volume loading and tensor-shaping helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def read_medical_volume(path: str | Path) -> np.ndarray:
    """Read NIfTI, MHA/NRRD, or NPY as a NumPy array."""

    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.name.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("nibabel is required for NIfTI data") from exc
        image=nib.load(str(path),mmap=True)
        return np.asanyarray(image.dataobj)
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError("SimpleITK is required for MHA/NRRD data") from exc
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def scalar_volume_to_dhw(array: np.ndarray) -> np.ndarray:
    """Convert a scalar 3D medical array to ``[D, H, W]`` convention."""

    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D scalar volume, got shape {array.shape}")
    return np.moveaxis(array, -1, 0)


def resize_3d_image(image: torch.Tensor, image_size: Sequence[int] | None) -> torch.Tensor:
    """Resize ``[C, D, H, W]`` image tensor with trilinear interpolation."""

    if image_size is None:
        return image
    target = tuple(int(v) for v in image_size)
    return F.interpolate(
        image.unsqueeze(0).float(),
        size=target,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)


def resize_3d_label(label: torch.Tensor, image_size: Sequence[int] | None) -> torch.Tensor:
    """Resize ``[D, H, W]`` label tensor with nearest-neighbor interpolation."""

    if image_size is None:
        return label
    target = tuple(int(v) for v in image_size)
    resized = F.interpolate(
        label.unsqueeze(0).unsqueeze(0).float(),
        size=target,
        mode="nearest",
    ).squeeze(0).squeeze(0)
    return resized.long()


def zscore_nonzero(image: torch.Tensor) -> torch.Tensor:
    """Per-channel z-score using non-zero voxels when available."""

    out = image.float().clone()
    for channel in range(out.shape[0]):
        values = out[channel]
        mask = values != 0
        reference = values[mask] if mask.any() else values.flatten()
        if mask.any():
            normalized=torch.zeros_like(values)
            normalized[mask]=(values[mask]-reference.mean())/reference.std().clamp_min(1e-6)
            out[channel]=normalized
        else:
            out[channel]=(values-reference.mean())/reference.std().clamp_min(1e-6)
    return out


def split_indices(n_items: int, split: str, val_fraction: float = 0.15, test_fraction: float = 0.15) -> range:
    """Deterministic train/val/test index ranges for patient-level splitting."""

    if n_items <= 0:
        return range(0, 0)
    n_test = max(1, round(n_items * test_fraction)) if n_items >= 3 else 0
    n_val = max(1, round(n_items * val_fraction)) if n_items >= 2 else 0
    n_train = max(n_items - n_val - n_test, 1)
    if split == "train":
        return range(0, min(n_train, n_items))
    if split == "val":
        return range(min(n_train, n_items), min(n_train + n_val, n_items))
    if split == "test":
        start = min(n_train + n_val, n_items)
        return range(start, n_items)
    raise ValueError(f"Unsupported split: {split}")
