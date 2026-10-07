"""Compact U-Net segmentation models with feature dictionaries."""

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def _normalization(dims:int,channels:int,name:str):
    kind=str(name).lower()
    if kind=="batch":return (nn.BatchNorm2d if dims==2 else nn.BatchNorm3d)(channels)
    if kind=="instance":return (nn.InstanceNorm2d if dims==2 else nn.InstanceNorm3d)(channels,affine=True)
    if kind=="group":
        groups=min(8,channels)
        while channels%groups:groups-=1
        return nn.GroupNorm(groups,channels)
    raise ValueError(f"Unsupported normalization: {name}")


def _block(dims,in_c,out_c,norm="batch"):
    conv=nn.Conv2d if dims==2 else nn.Conv3d
    return nn.Sequential(conv(in_c,out_c,3,padding=1),_normalization(dims,out_c,norm),nn.ReLU(),conv(out_c,out_c,3,padding=1),_normalization(dims,out_c,norm),nn.ReLU())


def _fixed_trilinear_upsample2x(x:torch.Tensor) -> torch.Tensor:
    """Exact separable 2x trilinear resize without ``upsample_trilinear3d``.

    This is used only by numerically checked fixed-shape deployment export
    paths; it also avoids unsupported native trilinear lowering in Core ML.
    Edge replication and the 0.75/0.25 weights match ``align_corners=False``.
    """

    previous=torch.cat((x[:,:,0:1],x[:,:,:-1]),dim=2)
    following=torch.cat((x[:,:,1:],x[:,:,-1:]),dim=2)
    even=0.25*previous+0.75*x
    odd=0.75*x+0.25*following
    x=torch.stack((even,odd),dim=3).flatten(2,3)

    previous=torch.cat((x[:,:,:,0:1],x[:,:,:,:-1]),dim=3)
    following=torch.cat((x[:,:,:,1:],x[:,:,:,-1:]),dim=3)
    even=0.25*previous+0.75*x
    odd=0.75*x+0.25*following
    x=torch.stack((even,odd),dim=4).flatten(3,4)

    previous=torch.cat((x[:,:,:,:,0:1],x[:,:,:,:,:-1]),dim=4)
    following=torch.cat((x[:,:,:,:,1:],x[:,:,:,:,-1:]),dim=4)
    even=0.25*previous+0.75*x
    odd=0.75*x+0.25*following
    return torch.stack((even,odd),dim=5).flatten(4,5)


def rasterize_box_prompts(x,boxes,box_valid=None,prompt_channels=None):
    """Convert ``[B,P,4]`` XYXY boxes to tensor prompt maps."""
    channels=int(prompt_channels if prompt_channels is not None else boxes.shape[1])
    if boxes.ndim != 3 or boxes.shape[-1] != 4 or boxes.shape[1] != channels:
        raise ValueError(f"Expected boxes [B,{channels},4], got {tuple(boxes.shape)}")
    valid=torch.ones(boxes.shape[:2],device=boxes.device,dtype=torch.bool) if box_valid is None else box_valid.bool()
    maps=x.new_zeros((x.shape[0],channels,*x.shape[-2:]))
    height,width=x.shape[-2:]
    for batch_index in range(x.shape[0]):
        for prompt_index in range(channels):
            if not bool(valid[batch_index,prompt_index]):continue
            x0,y0,x1,y1=boxes[batch_index,prompt_index]
            left=max(0,min(width-1,int(torch.floor(x0).item())))
            top=max(0,min(height-1,int(torch.floor(y0).item())))
            right=max(left+1,min(width,int(torch.ceil(x1).item())+1))
            bottom=max(top+1,min(height,int(torch.ceil(y1).item())+1))
            maps[batch_index,prompt_index,top:bottom,left:right]=1
    return maps


class CompactUNet2D(nn.Module):
    """Compact U-Net used as the compressed RACMed model for 2D/3D inputs."""

    def __init__(self,in_channels=1,num_classes=3,channels=(8,16),spatial_dims=2,prompt_aware=False,prompt_channels=None,norm="batch"):
        super().__init__(); self.spatial_dims=spatial_dims
        self.accepts_prompts=bool(prompt_aware)
        if self.accepts_prompts and spatial_dims != 2:
            raise ValueError("Box-prompt channels are supported for 2D segmentation only.")
        self.prompt_channels=(int(prompt_channels) if prompt_channels is not None else max(int(num_classes)-1,1)) if self.accepts_prompts else 0
        pool=nn.MaxPool2d if spatial_dims==2 else nn.MaxPool3d; conv=nn.Conv2d if spatial_dims==2 else nn.Conv3d
        channels=tuple(channels)
        if len(channels)<2: raise ValueError("CompactUNet requires at least two channel stages.")
        self.pool=pool(2)
        encoder_channels=(in_channels+self.prompt_channels,*channels[:-1])
        self.encoders=nn.ModuleList([_block(spatial_dims,src,dst,norm) for src,dst in zip(encoder_channels,channels)])
        decoder_specs=list(zip(reversed(channels[1:]),reversed(channels[:-1])))
        self.decoders=nn.ModuleList([_block(spatial_dims,low+skip,skip,norm) for low,skip in decoder_specs])
        self.head=conv(channels[0],num_classes,1)
    def _prompt_maps(self,x,boxes,box_valid=None):
        return rasterize_box_prompts(x,boxes,box_valid,self.prompt_channels)
    def forward(self,x,return_features: bool=True,boxes=None,box_valid=None):
        if self.accepts_prompts:
            if boxes is None:raise ValueError("Prompt-aware CompactUNet2D requires boxes.")
            return self.forward_with_prompt_maps(x,self._prompt_maps(x,boxes,box_valid),return_features)
        return self._forward_tensor(x,return_features)
    def forward_with_prompt_maps(self,x,prompt_maps,return_features: bool=True):
        """Forward with pre-rasterized prompt maps for tensor-only deployment."""
        if not self.accepts_prompts:raise ValueError("This model was not configured for prompt inputs.")
        if prompt_maps.shape[:2]!=(x.shape[0],self.prompt_channels) or prompt_maps.shape[2:]!=x.shape[2:]:
            raise ValueError(f"Expected prompt maps [B,{self.prompt_channels},H,W], got {tuple(prompt_maps.shape)}")
        return self._forward_tensor(torch.cat((x,prompt_maps.to(x)),dim=1),return_features)
    def _forward_tensor(self,x,return_features: bool=True):
        mode="bilinear" if self.spatial_dims==2 else "trilinear"; stages=[]; h=x
        for index,encoder in enumerate(self.encoders):
            if index>0:h=self.pool(h)
            h=encoder(h);stages.append(h)
        b=stages[-1];d=b;decoder_feats=[]
        for decoder,skip in zip(self.decoders,reversed(stages[:-1])):
            # Core ML's TorchScript frontend cannot reliably lower the
            # ``aten::Int`` nodes emitted by ``size=skip.shape[2:]`` on recent
            # PyTorch releases. Deployment export may temporarily select an
            # equivalent fixed 2x resize for validated, divisible input sizes.
            # Training and evaluation keep the dynamic path. ONNX/Core ML
            # export may select this path only after an exact output check.
            deployment_resize=getattr(self,"_fixed_deployment_upsample",False)
            if deployment_resize:
                up=(
                    _fixed_trilinear_upsample2x(d)
                    if self.spatial_dims==3 and deployment_resize=="separable"
                    else F.interpolate(d,scale_factor=2.0,mode=mode,align_corners=False)
                )
            else:
                up=F.interpolate(d,size=skip.shape[2:],mode=mode,align_corners=False)
            d=decoder(torch.cat((skip,up),1));decoder_feats.append(d)
        logits=self.head(d)
        feats={f"stage{i+1}":value for i,value in enumerate(stages)}|{"bottleneck":b} if return_features else {}
        if return_features:
            feats.update({f"decoder{i+1}":value for i,value in enumerate(decoder_feats)})
        return {"logits":logits,"features":feats}
