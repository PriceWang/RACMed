"""Class/region prototype relation distillation."""

from __future__ import annotations
import torch
import torch.nn.functional as F


def _resize_region(region: torch.Tensor, spatial_shape: tuple[int, ...]) -> torch.Tensor:
    """Resize a binary region while preserving small positive components."""

    mask = region.unsqueeze(1).float()
    if tuple(mask.shape[2:]) == spatial_shape:
        return mask
    if all(output <= source for output, source in zip(spatial_shape, mask.shape[2:])):
        pool = F.adaptive_max_pool2d if len(spatial_shape) == 2 else F.adaptive_max_pool3d
        return pool(mask, spatial_shape)
    return F.interpolate(mask, size=spatial_shape, mode="nearest")


def _prototype(feature: torch.Tensor, region: torch.Tensor) -> torch.Tensor | None:
    mask = _resize_region(region, tuple(feature.shape[2:])).to(dtype=feature.dtype)
    count = mask.sum()
    if count.item() == 0:
        return None
    reduce_dims = (0, *range(2, feature.ndim))
    return (feature * mask).sum(reduce_dims) / count


def relation_distillation(compressed: torch.Tensor, teacher: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
    """Match cosine-similarity matrices of valid class prototypes."""
    s, t = compressed, teacher.detach()
    if s.ndim == 2 or targets is None:
        sp, tp = s.flatten(1), t.flatten(1)
    else:
        sp_list, tp_list = [], []
        if targets.ndim==s.ndim and targets.shape[1]>1:
            masks=targets.bool().unbind(1)
        else:
            target = targets[:, 0] if targets.ndim == s.ndim and targets.shape[1] == 1 else targets
            masks=((target==cls) for cls in target.unique())
        for region in masks:
            if not region.any():continue
            student_prototype = _prototype(s, region)
            teacher_prototype = _prototype(t, region)
            if student_prototype is None or teacher_prototype is None:
                continue
            sp_list.append(student_prototype)
            tp_list.append(teacher_prototype)
        if not sp_list:return s.sum()*0
        sp, tp = torch.stack(sp_list), torch.stack(tp_list)
    if sp.shape[0] < 2:
        return s.sum() * 0
    sp, tp = F.normalize(sp, dim=-1), F.normalize(tp, dim=-1)
    return F.mse_loss(sp @ sp.transpose(0, 1), tp @ tp.transpose(0, 1))
