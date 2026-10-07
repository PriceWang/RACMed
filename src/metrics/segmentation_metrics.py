"""2D/3D segmentation overlap, surface, boundary, and object metrics."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from scipy import ndimage
from .boundary_metrics import boundary_f1
from .object_metrics import object_metrics
from .calibration_metrics import brier_score, expected_calibration_error, negative_log_likelihood
from src.utils.task_semantics import is_multilabel_target


def _surface_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not a.any() and not b.any(): return np.array([0.])
    if not a.any() or not b.any(): return np.array([float(max(a.shape))])
    sa = a ^ ndimage.binary_erosion(a); sb = b ^ ndimage.binary_erosion(b)
    return np.concatenate((ndimage.distance_transform_edt(~sb)[sa], ndimage.distance_transform_edt(~sa)[sb]))


def segmentation_reliability_metrics(
    logits:torch.Tensor,
    targets:torch.Tensor,
    threshold:float|list[float]|tuple[float,...],
    small_component_threshold:int=64,
) -> dict[str,float]:
    """Compute only metrics needed for validation-time reliability selection."""

    multilabel=is_multilabel_target(logits,targets)
    if multilabel:
        values=[float(threshold)]*logits.shape[1] if isinstance(threshold,(int,float)) else [float(value) for value in threshold]
        threshold_tensor=logits.new_tensor(values).view(1,-1,*([1]*(logits.ndim-2)))
        pred=logits.sigmoid()>=threshold_tensor;target=targets.bool()
        pairs=[(pred[:,index],target[:,index]) for index in range(logits.shape[1])]
        object_pred=pred.long();object_target=target.long()
        boundary_pred=pred.any(1).long();boundary_target=target.any(1).long()
    else:
        value=float(threshold[0] if isinstance(threshold,(list,tuple)) else threshold)
        pred=(logits.softmax(1)[:,1]>=value).long()
        target=targets[:,0].long() if targets.ndim==logits.ndim else targets.long()
        pairs=[(pred.bool(),target.eq(1))]
        object_pred=pred;object_target=target
        boundary_pred=pred;boundary_target=target
    dice=[];precision=[]
    for prediction,truth in pairs:
        tp=(prediction&truth).sum().float();fp=(prediction&~truth).sum().float();fn=(~prediction&truth).sum().float()
        dice.append(float((2*tp+1e-6)/(2*tp+fp+fn+1e-6)))
        precision.append(float(tp/(tp+fp).clamp_min(1)))
    objects=object_metrics(object_pred,object_target,small_component_threshold,channelwise=multilabel)
    return {
        "dice":float(np.mean(dice)),
        "precision":float(np.mean(precision)),
        "boundary_f1":boundary_f1(boundary_pred,boundary_target),
        "small_component_recall":float(objects["small_component_recall"]),
        "false_negative_lesion_rate":float(objects["false_negative_lesion_rate"]),
    }


def segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None = None,
    threshold: float | list[float] | tuple[float, ...] | None = None,
    small_component_threshold: int = 64,
    include_surface: bool = True,
) -> dict[str, object]:
    multilabel=is_multilabel_target(logits,targets)
    binary_softmax=not multilabel and logits.shape[1]==2 and threshold is not None
    if multilabel:
        if threshold is None:
            thresholds=[0.5]*logits.shape[1]
        elif isinstance(threshold,(int,float)):
            thresholds=[float(threshold)]*logits.shape[1]
        else:
            thresholds=[float(value) for value in threshold]
            if len(thresholds)!=logits.shape[1]:
                raise ValueError(f"Expected {logits.shape[1]} segmentation thresholds, got {len(thresholds)}")
        threshold_tensor=logits.new_tensor(thresholds).view(1,-1,*([1]*(logits.ndim-2)))
        pred=logits.sigmoid()>=threshold_tensor;target=targets.bool();classes=range(logits.shape[1])
    else:
        if binary_softmax:
            values=[float(threshold)] if isinstance(threshold,(int,float)) else [float(value) for value in threshold]
            if len(values)!=1:raise ValueError(f"Expected one foreground threshold, got {len(values)}")
            thresholds=values
            pred=(logits.softmax(1)[:,1]>=thresholds[0]).long()
        else:
            pred = (logits.sigmoid()[:, 0] >= .5).long() if logits.shape[1] == 1 else logits.argmax(1)
        target = targets[:, 0] if targets.ndim == logits.ndim and targets.shape[1] == 1 else targets.long()
        classes = range(1, num_classes or logits.shape[1]) if (num_classes or logits.shape[1]) > 1 else range(1)
    pairs=[(pred[:,cls],target[:,cls]) if multilabel else (pred==cls,target==cls) for cls in classes]
    surface_distances=[]
    if include_surface:
        def surface_for_pair(pair):
            p,t=pair
            distances=[_surface_distances(pi.cpu().numpy(),ti.cpu().numpy()) for pi,ti in zip(p,t)]
            return np.concatenate(distances)
        if len(pairs)>1:
            with ThreadPoolExecutor(max_workers=min(len(pairs),4)) as executor:
                surface_distances=list(executor.map(surface_for_pair,pairs))
        else:
            surface_distances=[surface_for_pair(pairs[0])]
    rows = []
    for index,(p,t) in enumerate(pairs):
        tp = (p & t).sum().float(); fp = (p & ~t).sum().float(); fn = (~p & t).sum().float(); tn = (~p & ~t).sum().float()
        dice = float((2*tp+1e-6)/(2*tp+fp+fn+1e-6)); iou = float((tp+1e-6)/(tp+fp+fn+1e-6))
        row={"dice": dice, "iou": iou, "precision": float(tp/(tp+fp).clamp_min(1)),
             "recall": float(tp/(tp+fn).clamp_min(1)), "sensitivity": float(tp/(tp+fn).clamp_min(1)),
             "specificity": float(tn/(tn+fp).clamp_min(1))}
        if include_surface:
            merged=surface_distances[index]
            row.update({"hd95":float(np.percentile(merged,95)),"assd":float(merged.mean())})
        rows.append(row)
    mean = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
    object_pred=pred.long()
    object_target=target.long()
    mean.update({"worst_class_dice": min(r["dice"] for r in rows), "class_wise": rows,
                 "boundary_f1": boundary_f1(pred.any(1).long() if multilabel else pred, target.any(1).long() if multilabel else target),
                 **object_metrics(object_pred, object_target, small_component_threshold, channelwise=multilabel),
                 "ece": expected_calibration_error(logits.float(), target),
                 "adaptive_ece": expected_calibration_error(logits.float(), target, adaptive=True),
                 "brier": brier_score(logits.float(), target),
                 "nll": negative_log_likelihood(logits.float(), target)})
    if multilabel or binary_softmax:
        mean["thresholds"]=thresholds
    per_volume = []
    overlap_pred=object_pred.any(1).long() if multilabel else object_pred
    overlap_target=object_target.any(1).long() if multilabel else object_target
    for p, t in zip(overlap_pred, overlap_target):
        inter = ((p > 0) & (t > 0)).sum().float(); denom = (p > 0).sum() + (t > 0).sum()
        per_volume.append(float((2*inter+1e-6)/(denom+1e-6)))
    mean["worst_case_per_volume_dice"] = min(per_volume)
    return mean
