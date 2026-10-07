"""MedSAM teacher wrapper for prompt-based 2D medical segmentation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .paths import resolve_model_path


class OfficialMedSAMAdapter(nn.Module):
    """Run the authors' SAM-format MedSAM checkpoint with official preprocessing."""

    source="medsam"
    accepts_prompts=True

    def __init__(self,checkpoint_file:str|Path,num_classes:int=2,in_channels:int=3,model_type:str="vit_b",independent_regions:bool=False) -> None:
        super().__init__()
        if num_classes<1:raise ValueError("MedSAM num_classes must be positive.")
        if in_channels not in {1,3}:raise ValueError("Official MedSAM expects grayscale or RGB images.")
        checkpoint=resolve_model_path(checkpoint_file)
        if not checkpoint.exists():raise FileNotFoundError(f"Official MedSAM checkpoint is missing: {checkpoint.name}")
        try:
            from segment_anything import sam_model_registry
        except ImportError as exc:
            raise ImportError("Install segment-anything to use the official MedSAM checkpoint.") from exc
        if model_type not in sam_model_registry:raise ValueError(f"Unknown SAM model type: {model_type}")
        self.backbone=sam_model_registry[model_type](checkpoint=str(checkpoint))
        self.num_classes=int(num_classes)
        self.independent_regions=bool(independent_regions)

    def _prepare_image(self,x:torch.Tensor) -> torch.Tensor:
        if x.shape[1]==1:x=x.repeat(1,3,1,1)
        elif x.shape[1]!=3:raise ValueError("Official MedSAM expects 1 or 3 image channels.")
        return F.interpolate(x.float().clamp(0,1),size=(1024,1024),mode="bicubic",align_corners=False)

    def forward(self,x:torch.Tensor,return_features:bool=True,boxes:torch.Tensor|None=None,box_valid:torch.Tensor|None=None) -> dict:
        original_size=x.shape[-2:];batch=x.shape[0];prompt_count=self.num_classes if self.independent_regions else max(self.num_classes-1,1)
        if boxes is None:
            if prompt_count!=1:raise ValueError("Multiclass MedSAM requires one box per foreground class.")
            boxes=torch.tensor([0.,0.,float(original_size[1]-1),float(original_size[0]-1)],device=x.device,dtype=x.dtype).view(1,1,4).repeat(batch,1,1)
        if boxes.shape!=(batch,prompt_count,4):raise ValueError(f"Expected boxes [{batch},{prompt_count},4], got {tuple(boxes.shape)}")
        pixels=self._prepare_image(x);embeddings=self.backbone.image_encoder(pixels)
        scaled=boxes.to(device=x.device,dtype=pixels.dtype).clone()
        scaled[...,0::2]*=1024/original_size[1];scaled[...,1::2]*=1024/original_size[0]
        masks=[]
        for batch_index in range(batch):
            sparse,dense=self.backbone.prompt_encoder(points=None,boxes=scaled[batch_index],masks=None)
            low_resolution,_=self.backbone.mask_decoder(
                image_embeddings=embeddings[batch_index:batch_index+1],
                image_pe=self.backbone.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            masks.append(low_resolution)
        low_resolution=torch.cat(masks,dim=0)
        foreground=F.interpolate(low_resolution,size=original_size,mode="bilinear",align_corners=False)
        foreground=foreground[:,0].reshape(batch,prompt_count,*original_size)
        if box_valid is not None:foreground=foreground.masked_fill(~box_valid.to(x.device).bool()[...,None,None],-20.)
        logits=foreground if self.independent_regions or self.num_classes==1 else torch.cat((-foreground.amax(1,keepdim=True),foreground),1)
        return {"logits":logits,"features":{"bottleneck":embeddings} if return_features else {}}


def create_medsam_model(cfg: dict, num_classes: int, in_channels: int) -> nn.Module:
    """Create the official SAM-format MedSAM teacher used by RACMed."""
    backend = str(cfg.get("backend", "official_sam"))
    if backend != "official_sam":
        raise ValueError("RACMed's validated MedSAM backend is official_sam.")
    checkpoint=cfg.get("checkpoint_file")
    if checkpoint is None:
        directory=resolve_model_path(cfg.get("model_dir") or cfg.get("local_pretrained_path"),"medsam-vit-base")
        checkpoint=directory/"medsam_vit_b.pth"
    return OfficialMedSAMAdapter(checkpoint,int(cfg.get("num_classes",num_classes)),int(cfg.get("in_channels",in_channels)),str(cfg.get("model_type","vit_b")),bool(cfg.get("independent_regions",False)))
