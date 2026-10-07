"""timm classifier wrapper."""

from .classification import ClassificationAdapter


def create_timm_model(name,pretrained,in_channels,num_classes):
    import timm
    return ClassificationAdapter(timm.create_model(name,pretrained=pretrained,in_chans=in_channels,num_classes=num_classes))
