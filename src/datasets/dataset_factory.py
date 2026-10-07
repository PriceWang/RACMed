"""Dataset factory for the validated RACMed experiment families."""

from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from .acdc_loader import ACDCSliceDataset
from .brats_loader import BraTSDataset
from .generic_2d_segmentation import Generic2DSegmentationDataset
from .rsna_loader import RSNAPneumoniaDataset
from .paths import resolve_dataset_root
from .synthetic_debug_dataset import SyntheticMedicalDataset


def create_dataset(config: dict, split: str, debug: bool = False):
    dcfg=config.get("dataset",{}); task=config.get("task_type",config.get("model",{}).get("task_type","segmentation2d"))
    if debug or dcfg.get("name")=="synthetic":
        length=(8 if split=="train" else 4) if debug else int(dcfg.get(f"{split}_length",8 if split=="train" else 4))
        image_size=([8,24,24] if task=="segmentation3d" else [32,32]) if debug else dcfg.get("image_size",[32,32])
        compressed = config.get("compressed_model", {})
        classes = int(dcfg.get("num_classes",config.get("model",{}).get("num_classes",3)))
        prompts = compressed.get("prompt_channels", max(classes - 1, 1))
        return SyntheticMedicalDataset(length,task,image_size,int(dcfg.get("in_channels",config.get("model",{}).get("in_channels",1))),classes,int(config.get("seed",0))+{"train":0,"val":1000,"test":2000}[split],int(prompts))
    kind=dcfg.get("type",task)
    name=str(dcfg.get("name",kind)).lower()
    root=resolve_dataset_root(dcfg.get("root","data"), name)
    in_channels=dcfg.get("in_channels",1)
    image_size=dcfg.get("image_size")
    num_classes=dcfg.get("num_classes")
    if kind in {"rsna", "rsna_pneumonia"} or name in {"rsna", "rsna_pneumonia"}:
        return RSNAPneumoniaDataset(
            root,
            split,
            int(dcfg.get("split_seed", config.get("seed", 0))),
            image_size or [224, 224],
            dcfg.get("views", ["PA", "AP"]),
            unit_scale=bool(dcfg.get("unit_scale",False)),
            preload=bool(dcfg.get("preload",False)),
            preload_cache_dir=dcfg.get("preload_cache_dir"),
            preload_device=str(dcfg.get("preload_device","auto")),
            preload_workers=dcfg.get("preload_workers",config.get("training",{}).get("num_workers")),
            preload_batch_size=int(dcfg.get("preload_batch_size",64)),
        )
    if kind=="segmentation2d":
        if name in {"acdc", "acdc_slices", "acdc_medsam"}:
            return ACDCSliceDataset(
                root,
                split,
                image_size,
                tuple(dcfg["split_counts"]) if dcfg.get("split_counts") else None,
                int(dcfg.get(f"{split}_box_perturb", dcfg.get("box_perturb", 0))),
                int(config.get("seed", 0)),
                str(dcfg.get("split_policy", "official_holdout")),
                str(dcfg.get("target_mode","classes")),
                tuple(dcfg.get("region_class_ids",[1,2,3])),
            )
        return Generic2DSegmentationDataset(
            root,
            split,
            in_channels,
            num_classes,
            image_size,
            bool(dcfg.get("prompt_boxes", False)),
            int(dcfg.get(f"{split}_box_perturb", dcfg.get("box_perturb", 0))),
            int(config.get("seed", 0)),
            str(dcfg.get("split_policy", "default")),
            dcfg.get("binary_mask_threshold"),
        )
    if kind=="segmentation3d":
        if name == "brats2018":
            return BraTSDataset(
                root,
                split,
                in_channels,
                image_size,
                dataset_version=str(dcfg.get("dataset_version", "2018")),
                target_mode=str(dcfg.get("target_mode", "classes")),
                split_counts=tuple(dcfg["split_counts"]) if dcfg.get("split_counts") else None,
                split_seed=int(dcfg.get("split_seed", 123)),
                crop_size=dcfg.get("crop_size"),
                crop_training=bool(dcfg.get("crop_training",False)),
                crop_positive_probability=float(dcfg.get("crop_positive_probability",0.8)),
                crop_region_weights=dcfg.get("crop_region_weights"),
            )
        raise ValueError(f"Unsupported validated 3D dataset: {name}. Use brats2018.")
    raise ValueError(f"Unsupported dataset type: {kind}")


def create_dataloaders(config: dict, debug: bool = False) -> dict:
    t=config.get("training",{}); batch=int(t.get("batch_size",4)); workers=int(t.get("num_workers",0))
    collate_fn=None
    cuda_pipeline=config.get("device","auto")!="cpu" and torch.cuda.is_available()
    pin_memory=bool(t.get("pin_memory",cuda_pipeline)) and cuda_pipeline
    loader_options={
        "batch_size":batch,
        "num_workers":workers,
        "pin_memory":pin_memory,
        "persistent_workers":workers>0,
    }
    if workers>0:loader_options["prefetch_factor"]=max(int(t.get("prefetch_factor",2)),1)
    return {
        split:DataLoader(
            create_dataset(config,split,debug),
            shuffle=split=="train",
            collate_fn=collate_fn if split=="train" else None,
            **loader_options,
        )
        for split in ("train","val","test")
    }
