"""Threshold-independent lesion detection curves for 2D and 3D segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np
import torch
from scipy import ndimage


def foreground_probabilities(
    logits:torch.Tensor,targets:torch.Tensor,
) -> tuple[np.ndarray,np.ndarray]:
    """Convert binary-softmax, multiclass, or multilabel tensors to C binary maps."""

    values=logits.detach().float().cpu();truth=targets.detach().cpu()
    multilabel=truth.ndim==values.ndim and truth.shape[1]==values.shape[1]
    if multilabel:
        probabilities=values.sigmoid();binary=truth.bool()
    else:
        labels=truth[:,0] if truth.ndim==values.ndim and truth.shape[1]==1 else truth
        if values.shape[1]==1:
            probabilities=values.sigmoid();binary=labels.bool().unsqueeze(1)
        else:
            probabilities=values.softmax(1)[:,1:]
            binary=torch.stack([labels.eq(index) for index in range(1,values.shape[1])],dim=1)
    return probabilities[0].numpy(),binary[0].numpy().astype(bool,copy=False)


def _component_counts(
    prediction:np.ndarray,truth:np.ndarray,small_threshold:int,min_iou:float,
) -> tuple[int,int,int,int,int]:
    """One-to-one connected-component matches using a minimum IoU."""

    structure=ndimage.generate_binary_structure(truth.ndim,1)
    target_labels,target_count=ndimage.label(truth,structure)
    target_sizes=np.bincount(target_labels.ravel(),minlength=target_count+1)[1:]
    small_ids={index+1 for index,size in enumerate(target_sizes) if int(size)<small_threshold}
    return _component_counts_precomputed(
        prediction,target_labels,target_count,target_sizes,small_ids,min_iou,structure,
    )


def _component_counts_precomputed(
    prediction:np.ndarray,target_labels:np.ndarray,target_count:int,
    target_sizes:np.ndarray,small_ids:set[int],min_iou:float,
    structure:np.ndarray,
) -> tuple[int,int,int,int,int]:
    """Match prediction components against one pre-labeled ground truth."""

    prediction_labels,prediction_count=ndimage.label(prediction,structure)
    prediction_sizes=np.bincount(prediction_labels.ravel(),minlength=prediction_count+1)[1:]
    matched_target:set[int]=set();matched_prediction:set[int]=set()
    overlap=(target_labels>0)&(prediction_labels>0)
    candidates=[]
    if overlap.any():
        encoded=(
            target_labels[overlap].astype(np.int64)*(prediction_count+1)
            + prediction_labels[overlap].astype(np.int64)
        )
        pair_ids,intersections=np.unique(encoded,return_counts=True)
        for pair_id,intersection in zip(pair_ids,intersections):
            target_id=int(pair_id//(prediction_count+1));prediction_id=int(pair_id%(prediction_count+1))
            union=int(target_sizes[target_id-1]+prediction_sizes[prediction_id-1]-intersection)
            iou=float(intersection/max(union,1))
            if iou>=min_iou:candidates.append((iou,target_id,prediction_id))
    for _,target_id,prediction_id in sorted(candidates,reverse=True):
        if target_id in matched_target or prediction_id in matched_prediction:continue
        matched_target.add(target_id);matched_prediction.add(prediction_id)
    return (
        len(matched_target),target_count,
        len(matched_target&small_ids),len(small_ids),
        prediction_count-len(matched_prediction),
    )


def _monotonic_points(rows:list[dict],metric:str,max_fp:float) -> tuple[np.ndarray,np.ndarray]:
    grouped:dict[float,float]={0.0:0.0}
    for row in rows:
        x=max(float(row["false_positives_per_case"]),0.0)
        if x>max_fp:continue
        grouped[x]=max(grouped.get(x,0.0),float(row[metric]))
    x=np.asarray(sorted(grouped),dtype=np.float64)
    y=np.maximum.accumulate(np.asarray([grouped[value] for value in x],dtype=np.float64))
    if x[-1]<max_fp:
        x=np.append(x,max_fp);y=np.append(y,y[-1])
    return x,y


def _recall_at(rows:list[dict],metric:str,target_fp:float) -> float:
    eligible=[float(row[metric]) for row in rows if float(row["false_positives_per_case"])<=target_fp]
    return max(eligible,default=0.0)


@dataclass
class LesionOperatingCurve:
    """Accumulate FROC counts across cases without retaining prediction volumes."""

    thresholds:tuple[float,...]
    small_threshold:int=64
    min_iou:float=0.10
    max_fp_per_case:float=4.0
    workers:int=1

    def __post_init__(self) -> None:
        values=sorted({float(value) for value in self.thresholds},reverse=True)
        if not values:raise ValueError("Lesion operating curve requires thresholds.")
        if not 0<self.min_iou<=1:raise ValueError("min_iou must be in (0, 1].")
        if self.workers<1:raise ValueError("workers must be positive.")
        self.thresholds=tuple(values);count=len(values)
        self.detected=np.zeros(count,dtype=np.int64);self.total=np.zeros(count,dtype=np.int64)
        self.small_detected=np.zeros(count,dtype=np.int64);self.small_total=np.zeros(count,dtype=np.int64)
        self.false_positives=np.zeros(count,dtype=np.int64);self.cases=0

    def update(self,logits:torch.Tensor,targets:torch.Tensor) -> None:
        probabilities,truth=foreground_probabilities(logits,targets)
        if probabilities.shape!=truth.shape:raise ValueError(f"Probability/target mismatch: {probabilities.shape} vs {truth.shape}")
        self.cases+=1
        component_info=[]
        for channel in range(probabilities.shape[0]):
            structure=ndimage.generate_binary_structure(truth[channel].ndim,1)
            labels,count=ndimage.label(truth[channel],structure)
            sizes=np.bincount(labels.ravel(),minlength=count+1)[1:]
            small_ids={index+1 for index,size in enumerate(sizes) if int(size)<self.small_threshold}
            component_info.append((labels,count,sizes,small_ids,structure))
        jobs=[
            (threshold_index,channel,threshold)
            for threshold_index,threshold in enumerate(self.thresholds)
            for channel in range(probabilities.shape[0])
        ]
        def evaluate(job):
            threshold_index,channel,threshold=job
            labels,count,sizes,small_ids,structure=component_info[channel]
            counts=_component_counts_precomputed(
                probabilities[channel]>=threshold,labels,count,sizes,small_ids,
                self.min_iou,structure,
            )
            return threshold_index,counts
        if self.workers>1 and len(jobs)>1:
            with ThreadPoolExecutor(max_workers=min(self.workers,len(jobs))) as executor:
                results=executor.map(evaluate,jobs)
                for threshold_index,counts in results:self._accumulate(threshold_index,counts)
        else:
            for job in jobs:
                threshold_index,counts=evaluate(job);self._accumulate(threshold_index,counts)

    def _accumulate(self,threshold_index:int,counts:tuple[int,int,int,int,int]) -> None:
        detected,total,small_detected,small_total,false_positives=counts
        self.detected[threshold_index]+=detected;self.total[threshold_index]+=total
        self.small_detected[threshold_index]+=small_detected;self.small_total[threshold_index]+=small_total
        self.false_positives[threshold_index]+=false_positives

    def compute(
        self,fp_targets:Iterable[float]=(0.25,0.5,1.0,2.0,4.0),
        auc_fp_limits:Iterable[float]|None=None,
    ) -> tuple[dict,list[dict]]:
        if self.cases<1:raise ValueError("Lesion operating curve received zero cases.")
        rows=[]
        for index,threshold in enumerate(self.thresholds):
            rows.append({
                "threshold":threshold,
                "lesion_recall":float(self.detected[index]/max(self.total[index],1)),
                "small_lesion_recall":float(self.small_detected[index]/max(self.small_total[index],1)),
                "false_positives_per_case":float(self.false_positives[index]/self.cases),
                "detected_lesions":int(self.detected[index]),"ground_truth_lesions":int(self.total[index]),
                "detected_small_lesions":int(self.small_detected[index]),"ground_truth_small_lesions":int(self.small_total[index]),
            })
        summary={
            "evaluated_cases":self.cases,
            "ground_truth_lesions":int(self.total[0]),
            "ground_truth_small_lesions":int(self.small_total[0]),
            "component_match_min_iou":self.min_iou,
            "small_component_threshold":self.small_threshold,
            "max_fp_per_case":self.max_fp_per_case,
        }
        limits=sorted({float(value) for value in (auc_fp_limits or (self.max_fp_per_case,))})
        if not limits or any(value<=0 or value>self.max_fp_per_case for value in limits):
            raise ValueError("FROC AUC limits must be in (0, max_fp_per_case].")
        for limit in limits:
            label=str(limit).rstrip("0").rstrip(".").replace(".","p")
            x,y=_monotonic_points(rows,"lesion_recall",limit)
            sx,sy=_monotonic_points(rows,"small_lesion_recall",limit)
            summary[f"froc_auc_0_{label}"]=float(np.trapezoid(y,x)/limit)
            summary[f"small_froc_auc_0_{label}"]=float(np.trapezoid(sy,sx)/limit)
        primary_label=str(limits[-1]).rstrip("0").rstrip(".").replace(".","p")
        summary["primary_froc_auc_key"]=f"froc_auc_0_{primary_label}"
        for target in fp_targets:
            label=str(float(target)).rstrip("0").rstrip(".").replace(".","p")
            summary[f"lesion_recall_at_{label}_fp_per_case"]=_recall_at(rows,"lesion_recall",float(target))
            summary[f"small_lesion_recall_at_{label}_fp_per_case"]=_recall_at(rows,"small_lesion_recall",float(target))
        return summary,rows
