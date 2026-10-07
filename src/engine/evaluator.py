"""Task-aware model evaluation with per-case records."""

from __future__ import annotations
from itertools import islice
import numpy as np
import torch
import torch.nn.functional as F
from src.metrics.classification_metrics import classification_metrics
from src.metrics.segmentation_metrics import segmentation_metrics,segmentation_reliability_metrics
from src.models.forward import forward_model
from src.utils.progress import progress_bar


class LogitBiasCalibration(torch.nn.Module):
    """Add fixed validation-selected class biases to dictionary model logits."""
    def __init__(self,model:torch.nn.Module,bias:list[float]|tuple[float,...]):
        super().__init__();self.model=model
        self.register_buffer("bias",torch.tensor(bias,dtype=torch.float32))
    def forward(self,x,return_features=True,**kwargs):
        output=self.model(x,return_features=return_features,**kwargs)
        shape=(1,-1,*([1]*(output["logits"].ndim-2)))
        return {**output,"logits":output["logits"]+self.bias.to(output["logits"]).view(shape)}


@torch.inference_mode()
def calibrate_background_logit_bias(
    model,loader,device,task_type:str,num_classes:int,
    candidates:list[float]|tuple[float,...],objective:str="dice",
    max_batches:int|None=None,sliding_window:dict|None=None,
    reliability_weights:dict[str,float]|None=None,
    small_component_threshold:int=64,
)->float:
    """Select one background-class bias on validation data only."""
    values=[float(value) for value in candidates]
    if not values:raise ValueError("Background-bias calibration requires candidates.")
    weights={"dice":.65,"boundary_f1":.10,"precision":.05,"small_component_recall":.10,"false_negative_lesion_rate":.10}
    weights.update(reliability_weights or {})
    scores=[]
    for value in progress_bar(values,desc="Background-bias calibration",total=len(values),unit="candidate",leave=True):
        calibrated=LogitBiasCalibration(model,[value,*([0.]*(num_classes-1))]).to(device)
        metrics,_=evaluate(calibrated,loader,task_type,device,num_classes,max_batches=max_batches,sliding_window=sliding_window,small_component_threshold=small_component_threshold)
        if objective=="dice":score=float(metrics["dice"])
        elif objective=="reliability":
            components={"dice":metrics["dice"],"boundary_f1":metrics["boundary_f1"],"precision":metrics["precision"],"small_component_recall":metrics["small_component_recall"],"false_negative_lesion_rate":1-float(metrics["false_negative_lesion_rate"])}
            denominator=sum(max(float(weights.get(key,0.)),0.) for key in components)
            score=sum(max(float(weights.get(key,0.)),0.)*float(component) for key,component in components.items())/max(denominator,1e-8)
        else:raise ValueError(f"Unknown background-bias objective: {objective}")
        scores.append(score)
    return values[int(np.argmax(scores))]


@torch.inference_mode()
def calibrate_segmentation_thresholds(
    model,
    loader,
    device,
    candidates: list[float] | tuple[float, ...] | None = None,
    max_batches: int | None = None,
    sliding_window: dict | None = None,
    objective: str = "dice",
    reliability_weights: dict[str, float] | None = None,
    max_dice_drop: float | None = None,
    small_component_threshold: int = 64,
) -> list[float]:
    """Select validation thresholds by pooled Dice or a reliability composite."""

    grid = [float(value) for value in (candidates or np.linspace(0.3, 0.8, 11))]
    if not grid:
        raise ValueError("Segmentation threshold calibration requires at least one candidate.")
    objective=str(objective).lower()
    if objective not in {"dice","reliability"}:raise ValueError(f"Unknown threshold objective: {objective}")
    weights={"dice":.55,"boundary_f1":.10,"precision":.05,"small_component_recall":.15,"false_negative_lesion_rate":.15}
    weights.update(reliability_weights or {})
    model.eval();true_positive=None;false_positive=None;false_negative=None;reliability_total=torch.zeros(len(grid),dtype=torch.float64);reliability_dice=torch.zeros(len(grid),dtype=torch.float64);reliability_batches=0;reliability_classes=1
    calibration_total=min(len(loader),int(max_batches)) if max_batches is not None else len(loader)
    calibration_source=islice(loader,calibration_total) if max_batches is not None else loader
    for step,batch in enumerate(progress_bar(calibration_source,desc="Threshold calibration",total=calibration_total,unit="batch",leave=True)):
        asynchronous=device.type=="cuda"
        x=batch["image"].to(device,non_blocking=asynchronous);target=batch["target"].to(device,non_blocking=asynchronous)
        with torch.autocast(device_type=device.type,enabled=asynchronous):
            if sliding_window:
                from monai.inferers import sliding_window_inference

                logits=sliding_window_inference(
                    x,
                    roi_size=tuple(sliding_window.get("roi_size",[96,96,96])),
                    sw_batch_size=int(sliding_window.get("sw_batch_size",1)),
                    predictor=lambda window:model(window,return_features=False)["logits"],
                    overlap=float(sliding_window.get("overlap",.25)),
                )
            else:
                logits=forward_model(model,x,batch,return_features=False)["logits"]
        multilabel=target.ndim==logits.ndim and target.shape[1]==logits.shape[1]
        binary_softmax=logits.shape[1]==2 and (
            target.ndim==logits.ndim-1 or (target.ndim==logits.ndim and target.shape[1]==1)
        )
        if not multilabel and not binary_softmax:
            raise ValueError("Threshold calibration requires binary-softmax or multilabel segmentation targets.")
        if true_positive is None:
            calibrated_classes=logits.shape[1] if multilabel else 1
            shape=(calibrated_classes,len(grid));true_positive=torch.zeros(shape,dtype=torch.float64);false_positive=torch.zeros_like(true_positive);false_negative=torch.zeros_like(true_positive)
        if binary_softmax:
            probability=logits.float().softmax(1)[:,1:2]
            binary_target=target[:,0] if target.ndim==logits.ndim else target
            truth=binary_target.eq(1).unsqueeze(1)
        else:
            probability=logits.float().sigmoid();truth=target.bool()
        if objective=="reliability":
            reliability_classes=logits.shape[1] if multilabel else 1
            metric_logits=logits.detach().cpu();metric_target=target.detach().cpu();metric_small_threshold=small_component_threshold
            if logits.ndim==5:
                metric_logits=F.interpolate(metric_logits,scale_factor=.5,mode="trilinear",align_corners=False)
                metric_target=F.interpolate(metric_target.float(),size=metric_logits.shape[2:],mode="nearest").to(target.dtype)
                metric_small_threshold=max(1,int(round(small_component_threshold/8)))
            for threshold_index,threshold in enumerate(grid):
                threshold_values=[threshold]*reliability_classes
                metrics=segmentation_reliability_metrics(metric_logits,metric_target,threshold_values,metric_small_threshold)
                values={"dice":metrics["dice"],"boundary_f1":metrics["boundary_f1"],"precision":metrics["precision"],"small_component_recall":metrics["small_component_recall"],"false_negative_lesion_rate":1-float(metrics["false_negative_lesion_rate"])}
                denominator=sum(max(float(weights.get(key,0.)),0.) for key in values)
                reliability_total[threshold_index]+=sum(max(float(weights.get(key,0.)),0.)*float(value) for key,value in values.items())/max(denominator,1e-8)
                reliability_dice[threshold_index]+=float(metrics["dice"])
            reliability_batches+=1
        for class_index in range(probability.shape[1]):
            class_truth=truth[:,class_index]
            for threshold_index,threshold in enumerate(grid):
                prediction=probability[:,class_index]>=threshold
                true_positive[class_index,threshold_index]+=(prediction & class_truth).sum().cpu()
                false_positive[class_index,threshold_index]+=(prediction & ~class_truth).sum().cpu()
                false_negative[class_index,threshold_index]+=(~prediction & class_truth).sum().cpu()
    if true_positive is None:
        raise ValueError("Threshold calibration received zero validation batches.")
    if objective=="reliability":
        if reliability_batches==0:raise ValueError("Reliability threshold calibration received zero validation batches.")
        scores=reliability_total.clone()
        if max_dice_drop is not None:
            baseline_index=min(range(len(grid)),key=lambda index:abs(grid[index]-.5))
            minimum_dice=reliability_dice[baseline_index]/reliability_batches-max(float(max_dice_drop),0.)
            scores[reliability_dice/reliability_batches<minimum_dice]=-torch.inf
        return [grid[int(scores.argmax())]]*reliability_classes
    dice=(2*true_positive+1e-6)/(2*true_positive+false_positive+false_negative+1e-6)
    return [grid[int(index)] for index in dice.argmax(1)]


def _mean_metric_dict(rows: list[dict]) -> dict:
    """Average segmentation metric dictionaries from variable-shape volumes."""

    numeric = {
        key: float(sum(float(row[key]) for row in rows) / len(rows))
        for key, value in rows[0].items()
        if isinstance(value, (int, float))
    }
    if "class_wise" in rows[0]:
        class_count = len(rows[0]["class_wise"])
        numeric["class_wise"] = [
            {
                key: float(sum(float(row["class_wise"][idx][key]) for row in rows) / len(rows))
                for key, value in rows[0]["class_wise"][idx].items()
                if isinstance(value, (int, float))
            }
            for idx in range(class_count)
        ]
        numeric["class_wise_median"] = [
            {
                key: float(np.median([float(row["class_wise"][idx][key]) for row in rows]))
                for key,value in rows[0]["class_wise"][idx].items()
                if isinstance(value,(int,float))
            }
            for idx in range(class_count)
        ]
    if "dice" in rows[0]:
        numeric["median_dice"]=float(np.median([float(row["dice"]) for row in rows]))
    return numeric


@torch.inference_mode()
def evaluate(
    model,
    loader,
    task_type,
    device,
    num_classes,
    max_batches: int | None = None,
    threshold: float | None = None,
    threshold_strategy: str | None = None,
    sliding_window: dict | None = None,
    progress_label: str | None = None,
    small_component_threshold: int = 64,
):
    model.eval();logits=[];targets=[];ids=[]
    segmentation_rows=[];per_case=[];current_group=None;group_logits=[];group_targets=[]

    def flush_group():
        nonlocal group_logits,group_targets,current_group
        if not group_logits:return
        if group_logits[0].ndim==3:
            pred=torch.stack(group_logits,dim=1).unsqueeze(0)
            target_stack_dim=1 if group_targets[0].ndim==group_logits[0].ndim and group_targets[0].shape[0]==group_logits[0].shape[0] else 0
            truth=torch.stack(group_targets,dim=target_stack_dim).unsqueeze(0)
        else:
            pred=group_logits[0].unsqueeze(0);truth=group_targets[0].unsqueeze(0)
        metrics_i=segmentation_metrics(pred,truth,num_classes,threshold=threshold,small_component_threshold=small_component_threshold)
        segmentation_rows.append(metrics_i)
        case_row={"case_id":current_group,"score":float(metrics_i["worst_case_per_volume_dice"])}
        for class_index,class_metrics in enumerate(metrics_i.get("class_wise",[])):
            case_row[f"class_{class_index}_dice"]=float(class_metrics["dice"])
        per_case.append(case_row)
        group_logits=[];group_targets=[]

    total_batches=min(len(loader),int(max_batches)) if max_batches is not None else len(loader)
    progress_interval=max(1,total_batches//10) if total_batches else 1
    source=islice(loader,total_batches) if max_batches is not None else loader
    batches=progress_bar(
        source,
        desc=f"Evaluate {progress_label}",
        total=total_batches,
        unit="batch",
        leave=True,
    ) if progress_label else source
    for step,batch in enumerate(batches):
        asynchronous=device.type=="cuda"
        x=batch["image"].to(device,non_blocking=asynchronous); y=batch["target"].to(device,non_blocking=asynchronous)
        with torch.autocast(device_type=device.type,enabled=asynchronous):
            if sliding_window and task_type.startswith("segmentation"):
                from monai.inferers import sliding_window_inference

                def _predict(window):
                    return model(window, return_features=False)["logits"]

                out = {
                    "logits": sliding_window_inference(
                        x,
                        roi_size=tuple(sliding_window.get("roi_size", [96, 96, 96])),
                        sw_batch_size=int(sliding_window.get("sw_batch_size", 1)),
                        predictor=_predict,
                        overlap=float(sliding_window.get("overlap", 0.25)),
                    )
                }
            else:
                out=forward_model(model,x,batch,return_features=False)
        if task_type.startswith("segmentation"):
            batch_logits=out["logits"].cpu();batch_targets=y.cpu()
            group_ids=batch.get("group_id",batch["id"])
            for index,group_id in enumerate(group_ids):
                group_id=str(group_id)
                if current_group is not None and group_id!=current_group:flush_group()
                current_group=group_id;group_logits.append(batch_logits[index]);group_targets.append(batch_targets[index])
        else:
            logits.append(out["logits"].cpu());targets.append(y.cpu());ids.extend(batch["id"])
        completed=step+1
        if progress_label and (completed % progress_interval == 0 or completed == total_batches):
            from src.pipeline.progress import pipeline_logger

            pipeline_logger().info("EVAL_PROGRESS | label=%s batch=%d/%d",progress_label,completed,total_batches)
    if task_type.startswith("segmentation"):
        flush_group()
        if not segmentation_rows:raise ValueError("Evaluation received zero batches; max_batches must be positive when set.")
        metrics=_mean_metric_dict(segmentation_rows)
        if "thresholds" in segmentation_rows[0]:metrics["thresholds"]=segmentation_rows[0]["thresholds"]
        return metrics,per_case
    if not logits: raise ValueError("Evaluation received zero batches; max_batches must be positive when set.")
    pred=torch.cat(logits);truth=torch.cat(targets)
    metrics=classification_metrics(pred,truth,threshold=threshold,threshold_strategy=threshold_strategy) if task_type=="classification" else segmentation_metrics(pred,truth,num_classes)
    effective_threshold = float(metrics.get("threshold", threshold if threshold is not None else 0.5))
    per_case=[]
    for index,case_id in enumerate(ids):
        if task_type=="classification" and truth.ndim>1:score=float((pred[index].sigmoid().ge(.5)==truth[index].bool()).float().mean())
        elif task_type=="classification" and pred.shape[1]==1:score=float((pred[index,0].sigmoid()>=effective_threshold)==truth[index].bool())
        elif task_type=="classification":score=float(pred[index].argmax()==truth[index])
        else:
            score=float(segmentation_metrics(pred[index:index+1],truth[index:index+1],num_classes)["worst_case_per_volume_dice"])
        per_case.append({"case_id":case_id,"score":score})
    return metrics,per_case
