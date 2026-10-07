"""Attention-transfer baseline."""

import torch
import torch.nn.functional as F
from .feature_kd import align_feature


def attention_transfer(compressed: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    # Attention transfer compares spatial energy, which is independent of the
    # feature-channel count. Reduce channels before spatial alignment: full
    # 5-D feature interpolation is unnecessarily expensive for 3-D models.
    if compressed.ndim == teacher.ndim and compressed.ndim >= 3:
        sa = compressed.float().pow(2).mean(1, keepdim=True)
        ta = teacher.detach().float().pow(2).mean(1, keepdim=True)
        if sa.shape[2:] != ta.shape[2:]:
            mode = {3: "linear", 4: "bilinear", 5: "trilinear"}[sa.ndim]
            sa = F.interpolate(sa, size=ta.shape[2:], mode=mode, align_corners=False)
        sa, ta = sa.flatten(1), ta.flatten(1)
    else:
        s, t = align_feature(compressed, teacher)
        if s.ndim == 2:
            # Pooled classification features carry attention over channels.
            sa, ta = s.float().pow(2), t.float().pow(2)
        else:
            sa = s.float().pow(2).mean(1).flatten(1)
            ta = t.float().pow(2).mean(1).flatten(1)
    return F.mse_loss(F.normalize(sa, dim=1), F.normalize(ta, dim=1))
