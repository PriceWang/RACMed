"""Official RSNA Pneumonia DICOM adapter using TorchXRayVision preprocessing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from src.pipeline.progress import pipeline_logger
from src.utils.progress import progress_bar


def _read_dicom_pixels(path: Path) -> np.ndarray:
    """Decode one DICOM on CPU; resize/normalization is intentionally deferred."""

    import pydicom

    return np.asarray(pydicom.dcmread(path).pixel_array)


class RSNAPneumoniaDataset(Dataset):
    """Load official RSNA DICOM images with patient-grouped deterministic splits."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        seed: int = 0,
        image_size: Sequence[int] = (224, 224),
        views: Sequence[str] = ("PA", "AP"),
        test_fraction: float = 0.2,
        validation_fraction_within_train: float = 0.2,
        unit_scale: bool = False,
        preload: bool = False,
        preload_cache_dir: str | Path | None = None,
        preload_device: str = "auto",
        preload_workers: int | None = None,
        preload_batch_size: int = 64,
    ) -> None:
        root = Path(root)
        image_dir = root / "stage_2_train_images"
        labels = root / "stage_2_train_labels.csv"
        if not image_dir.exists() or not labels.exists():
            raise FileNotFoundError(
                "RSNA requires stage_2_train_images/ and stage_2_train_labels.csv under dataset.root"
            )
        try:
            import torchxrayvision as xrv
            from torchvision.transforms import Compose
        except ImportError as exc:
            raise ImportError("Install torchxrayvision and torchvision to use the RSNA adapter") from exc

        size = tuple(int(value) for value in image_size)
        if len(size) != 2 or size[0] != size[1]:
            raise ValueError("TorchXRayVision RSNA preprocessing requires a square image_size")
        transforms = Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(size[0])])
        header_csv = root / "dicom_headers.csv"
        kwargs = {
            "imgpath": str(image_dir),
            "csvpath": str(labels),
            "views": list(views),
            "transform": transforms,
            "extension": ".dcm",
        }
        if header_csv.exists():
            kwargs["dicomcsvpath"] = str(header_csv)
        self.dataset = xrv.datasets.RSNA_Pneumonia_Dataset(**kwargs)
        pathology_names = [str(item).strip().lower() for item in self.dataset.pathologies]
        self.pathology_index = pathology_names.index("pneumonia")

        indices = np.arange(len(self.dataset))
        groups = self.dataset.csv["patientid"].astype(str).to_numpy()
        outer = GroupShuffleSplit(n_splits=1, test_size=float(test_fraction), random_state=int(seed))
        train_pool, test_indices = next(outer.split(indices, groups=groups))
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=float(validation_fraction_within_train),
            random_state=int(seed) + 1,
        )
        train_relative, val_relative = next(
            inner.split(train_pool, groups=groups[train_pool])
        )
        split_indices = {
            "train": train_pool[train_relative],
            "val": train_pool[val_relative],
            "test": test_indices,
        }
        if split not in split_indices:
            raise ValueError(f"Unsupported split: {split}")
        self.indices = np.asarray(split_indices[split], dtype=np.int64)
        self.patient_ids = [str(self.dataset.csv.iloc[index]["patientid"]) for index in self.indices]
        self.unit_scale=bool(unit_scale)
        self._image_dir=image_dir
        requested_device=str(preload_device).lower()
        if requested_device not in {"auto","cpu","cuda"}:
            raise ValueError("RSNA preload_device must be auto, cpu, or cuda")
        use_cuda=requested_device!="cpu" and torch.cuda.is_available()
        if requested_device=="cuda" and not use_cuda:
            pipeline_logger().warning("RSNA_CACHE | requested_device=cuda unavailable; falling back to cpu")
        self.preload_device=torch.device("cuda" if use_cuda else "cpu")
        default_workers=min(16,max((os.cpu_count() or 1)//2,1))
        self.preload_workers=max(int(preload_workers if preload_workers is not None else default_workers),0)
        self.preload_batch_size=max(int(preload_batch_size),1)
        self._preloaded_images:list[torch.Tensor]|torch.Tensor|None=None
        if preload:
            scale_name="unit" if self.unit_scale else "xrv"
            cache_name=f"{split}_seed{int(seed)}_{size[0]}px_{scale_name}.pt"
            cache_path=Path(preload_cache_dir)/cache_name if preload_cache_dir else None
            cache_loaded=False
            if cache_path is not None and cache_path.exists():
                pipeline_logger().info(
                    "RSNA_CACHE | split=%s status=loading_cached images=%d",
                    split,
                    len(self.indices),
                )
                try:
                    cached=torch.load(cache_path,map_location="cpu",weights_only=True,mmap=True)
                    if len(cached)!=len(self.indices):
                        raise ValueError(f"RSNA preload cache length mismatch for split {split}")
                    self._preloaded_images=cached
                    cache_loaded=True
                    pipeline_logger().info("RSNA_CACHE | split=%s status=ready images=%d",split,len(self.indices))
                except (OSError,RuntimeError,EOFError,ValueError) as exc:
                    pipeline_logger().warning(
                        "RSNA_CACHE | split=%s status=invalid_cache error=%s; rebuilding",
                        split,
                        type(exc).__name__,
                    )
            if not cache_loaded:
                pipeline_logger().info(
                    "RSNA_CACHE | split=%s status=building images=%d preprocess_device=%s decode_workers=%d batch_size=%d",
                    split,
                    len(self.indices),
                    self.preload_device,
                    self.preload_workers,
                    self.preload_batch_size,
                )
                self._preloaded_images=self._build_preload_cache(split,size[0])
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True,exist_ok=True)
                    temporary=cache_path.with_suffix(cache_path.suffix+".tmp")
                    torch.save(self._preloaded_images,temporary)
                    temporary.replace(cache_path)
                pipeline_logger().info("RSNA_CACHE | split=%s status=ready images=%d",split,len(self.indices))

    def _build_preload_cache(self,split:str,size:int) -> torch.Tensor:
        """Decode DICOMs concurrently and batch XRV-equivalent preprocessing on GPU."""

        total=len(self.indices)
        output=torch.empty((total,1,size,size),dtype=torch.float16)
        log_interval=max(1,total//10)
        starts=range(0,total,self.preload_batch_size)
        executor=ThreadPoolExecutor(max_workers=self.preload_workers) if self.preload_workers>1 else None
        try:
            batches=progress_bar(
                starts,
                desc=f"RSNA {split} GPU cache" if self.preload_device.type=="cuda" else f"RSNA {split} CPU cache",
                total=math.ceil(total/self.preload_batch_size),
                unit="batch",
                leave=True,
            )
            next_log=log_interval
            for start in batches:
                end=min(start+self.preload_batch_size,total)
                paths=[
                    self._image_dir/f"{self.dataset.csv.iloc[int(source_index)]['patientid']}.dcm"
                    for source_index in self.indices[start:end]
                ]
                decoded=list(executor.map(_read_dicom_pixels,paths)) if executor is not None else [_read_dicom_pixels(path) for path in paths]
                output[start:end]=self._preprocess_decoded_batch(decoded,size)
                batches.set_postfix(images=f"{end}/{total}",device=self.preload_device.type)
                if end>=next_log or end==total:
                    pipeline_logger().info("RSNA_CACHE_PROGRESS | split=%s image=%d/%d",split,end,total)
                    while next_log<=end:next_log+=log_interval
        finally:
            if executor is not None:executor.shutdown(wait=True,cancel_futures=True)
        return output

    def _preprocess_decoded_batch(self,decoded:list[np.ndarray],size:int) -> torch.Tensor:
        """Apply XRV center crop, normalization and antialiased resize in batches."""

        result=torch.empty((len(decoded),1,size,size),dtype=torch.float16)
        groups:dict[tuple[int,int],list[tuple[int,torch.Tensor]]]={}
        for index,array in enumerate(decoded):
            if array.ndim>2:array=array[:,:,0]
            if array.ndim!=2:raise ValueError(f"Expected 2D RSNA pixels, got shape {array.shape}")
            if float(array.max())>255:raise ValueError(f"RSNA image maximum {float(array.max())} exceeds XRV bound 255")
            height,width=array.shape;crop=min(height,width);top=height//2-crop//2;left=width//2-crop//2
            cropped=np.ascontiguousarray(array[top:top+crop,left:left+crop],dtype=np.float32)
            groups.setdefault(cropped.shape,[]).append((index,torch.from_numpy(cropped).unsqueeze(0)))
        for rows in groups.values():
            indices=[index for index,_ in rows]
            batch=torch.stack([value for _,value in rows])
            if self.preload_device.type=="cuda":batch=batch.pin_memory()
            batch=batch.to(self.preload_device,non_blocking=self.preload_device.type=="cuda")
            batch=(2.0*(batch/255.0)-1.0)*1024.0
            batch=self._xrv_resize(batch,size)
            if self.unit_scale:batch=(batch+1024.0)/2048.0
            result[indices]=batch.to(device="cpu",dtype=torch.float16)
        return result

    @staticmethod
    def _xrv_resize(batch:torch.Tensor,size:int) -> torch.Tensor:
        """GPU/CPU equivalent of XRayResizer's skimage order-1 resize."""

        factor=batch.shape[-1]/float(size)
        if factor>1.0:
            sigma=(factor-1.0)/2.0
            radius=int(4.0*sigma+0.5)
            if radius>0:
                coordinates=torch.arange(-radius,radius+1,device=batch.device,dtype=batch.dtype)
                kernel=torch.exp(-0.5*(coordinates/sigma).square());kernel/=kernel.sum()
                batch=F.conv2d(batch,kernel.view(1,1,1,-1),padding=(0,radius))
                batch=F.conv2d(batch,kernel.view(1,1,-1,1),padding=(radius,0))
        return F.interpolate(batch,size=(size,size),mode="bilinear",align_corners=False)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        source_index = int(self.indices[index])
        if self._preloaded_images is None:
            sample = self.dataset[source_index]
            image = torch.as_tensor(np.asarray(sample["img"]).copy(), dtype=torch.float32)
            if self.unit_scale:image=(image+1024.0)/2048.0
        else:
            image=self._preloaded_images[index].float()
        target = torch.tensor(int(self.dataset.labels[source_index,self.pathology_index]), dtype=torch.long)
        return {"image": image, "target": target, "id": self.patient_ids[index]}
