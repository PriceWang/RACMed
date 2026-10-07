"""Feature alignment for heterogeneous teacher and compressed networks."""

from __future__ import annotations
import torch
import torch.nn.functional as F


def align_feature(compressed: torch.Tensor, teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Align spatial and channel dimensions without persistent trainable state."""
    if compressed.ndim != teacher.ndim:
        # Classification backbones commonly expose a pooled [B, C] feature on
        # one side and a terminal [B, C, 1, 1] activation on the other. Pooling
        # the spatial representation preserves the per-channel attention vector
        # and avoids inventing spatial support that the pooled teacher lacks.
        if compressed.ndim > 2:
            compressed = compressed.flatten(2).mean(-1)
        if teacher.ndim > 2:
            teacher = teacher.flatten(2).mean(-1)
    if compressed.ndim >= 3 and compressed.shape[2:] != teacher.shape[2:]:
        mode = {3: "linear", 4: "bilinear", 5: "trilinear"}.get(compressed.ndim)
        if mode is None:
            compressed = compressed.flatten(2).mean(-1)
            teacher = teacher.flatten(2).mean(-1)
        else:
            compressed = F.interpolate(compressed, size=teacher.shape[2:], mode=mode, align_corners=False)
    if compressed.ndim>1 and compressed.shape[1]!=teacher.shape[1]:
        moved=compressed.movedim(1,-1);shape=moved.shape;resized=F.interpolate(moved.reshape(-1,1,shape[-1]),size=teacher.shape[1],mode="nearest");compressed=resized.reshape(*shape[:-1],teacher.shape[1]).movedim(-1,1)
    return compressed, teacher.detach()


def feature_distillation(
    compressed: torch.Tensor, teacher: torch.Tensor, weight: torch.Tensor | None = None, mode: str = "l2"
) -> torch.Tensor:
    if mode == "attention":
        if compressed.ndim < 4 or teacher.ndim != compressed.ndim:
            raise ValueError("Attention feature KD expects matching 2D/3D feature ranks.")
        student_energy=compressed.float().square().mean(1).sqrt()
        teacher_energy=teacher.detach().float().square().mean(1).sqrt()
        if student_energy.shape[1:]!=teacher_energy.shape[1:]:
            interpolation="bilinear" if student_energy.ndim==3 else "trilinear"
            student_energy=F.interpolate(student_energy.unsqueeze(1),size=teacher_energy.shape[1:],mode=interpolation,align_corners=False)[:,0]
        dims=tuple(range(1,student_energy.ndim))
        student_energy=student_energy/student_energy.mean(dims,keepdim=True).clamp_min(1e-6)
        teacher_energy=teacher_energy/teacher_energy.mean(dims,keepdim=True).clamp_min(1e-6)
        loss=(student_energy-teacher_energy).square()
        if weight is not None:
            interpolation="bilinear" if loss.ndim==3 else "trilinear"
            local_weight=F.interpolate(weight.float().unsqueeze(1),size=loss.shape[1:],mode=interpolation,align_corners=False)[:,0]
            return (loss*local_weight).sum()/local_weight.sum().clamp_min(1e-8)
        return loss.mean()
    s, t = align_feature(compressed, teacher)
    if mode == "cosine":
        loss = 1 - F.cosine_similarity(s, t, dim=1)
    else:
        loss = (s - t).pow(2).mean(1)
    if weight is not None and weight.ndim == 1:
        per_sample=loss.flatten(1).mean(1) if loss.ndim>1 else loss
        return (per_sample*weight).sum()/weight.sum().clamp_min(1e-8)
    if weight is not None and loss.ndim >= 3:
        interp_mode = "bilinear" if loss.ndim == 3 else "trilinear"
        w = F.interpolate(weight.unsqueeze(1), size=loss.shape[1:], mode=interp_mode, align_corners=False)[:, 0]
        return (loss * w).sum() / w.sum().clamp_min(1e-8)
    if weight is not None and loss.ndim == 1:
        return (loss * weight).sum() / weight.sum().clamp_min(1e-8)
    return loss.mean()
