"""Compact 3D segmentation model used as the compressed RACMed target."""
from .segmentation2d import CompactUNet2D


class CompactUNet3D(CompactUNet2D):
    """Compact 3D U-Net for volumetric segmentation experiments."""

    def __init__(self,in_channels=1,num_classes=3,channels=(4,8),norm="batch"):
        super().__init__(in_channels,num_classes,channels,spatial_dims=3,norm=norm)
