"""Compact classification models and output adapters."""

from __future__ import annotations
import torch
from torch import nn


class CompactClassifier(nn.Module):
    """Compact CNN classifier used as the compressed RACMed model."""

    def __init__(self, in_channels: int = 1, num_classes: int = 3, width: int = 12, spatial_dims: int = 2) -> None:
        super().__init__()
        conv=nn.Conv2d if spatial_dims==2 else nn.Conv3d;norm=nn.BatchNorm2d if spatial_dims==2 else nn.BatchNorm3d;pool=nn.MaxPool2d if spatial_dims==2 else nn.MaxPool3d;avg=nn.AdaptiveAvgPool2d if spatial_dims==2 else nn.AdaptiveAvgPool3d
        self.stage1=nn.Sequential(conv(in_channels,width,3,padding=1),norm(width),nn.ReLU(),pool(2))
        self.stage2=nn.Sequential(conv(width,width*2,3,padding=1),norm(width*2),nn.ReLU(),avg(1))
        self.head=nn.Linear(width*2,num_classes)
    def forward(self,x,return_features: bool=True):
        a=self.stage1(x); b=self.stage2(a); logits=self.head(b.flatten(1))
        return {"logits":logits,"features":{"stage1":a,"bottleneck":b} if return_features else {}}


class ClassificationAdapter(nn.Module):
    """Adapt a tensor-returning classifier to RACMed's dictionary contract."""
    def __init__(self,backbone: nn.Module): super().__init__(); self.backbone=backbone
    def forward(self,x,return_features: bool=True):
        logits=self.backbone(x)
        if hasattr(logits,"logits"): logits=logits.logits
        return {"logits":logits,"features":{"bottleneck":logits.unsqueeze(-1).unsqueeze(-1)} if return_features else {}}
