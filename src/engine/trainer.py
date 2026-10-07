"""Unified supervised, baseline KD, and RACMed training engine."""

from __future__ import annotations
import json, logging
from pathlib import Path
import hashlib
from itertools import islice
import pandas as pd
import torch
from torch.optim import Adam, AdamW, SGD
import yaml

from src.datasets.dataset_factory import create_dataloaders
from src.losses.kd_losses import softened_kl,softened_bernoulli_kl
from src.losses.attention_kd import attention_transfer
from src.losses.relation_kd import relation_distillation
from src.losses.racmed_loss import RACMedLoss
from src.losses.supervised import supervised_loss
from src.models.model_factory import create_model
from src.models.forward import forward_model
from src.risk.risk_map import RiskMapBuilder
from src.risk.calibration import fit_binary_platt_scaling,fit_logit_temperature
from src.utils.method_naming import public_method_name
from src.utils.progress import progress_bar
from .checkpoint import load_checkpoint, save_checkpoint
from .evaluator import LogitBiasCalibration,calibrate_background_logit_bias,calibrate_segmentation_thresholds,evaluate
from .logger import configure_logger
from .reproducibility import save_provenance, seed_everything


def _jsonable(value):
    if isinstance(value,dict):return {k:_jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_jsonable(v) for v in value]
    try:
        if pd.isna(value):return None
    except (TypeError,ValueError):
        return value
    return value


def set_model_training_mode(model:torch.nn.Module,freeze_batch_norm:bool=False)->None:
    """Enter train mode while optionally preserving pretrained BatchNorm statistics."""
    model.train()
    if freeze_batch_norm:
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


class BinaryPlattCalibration(torch.nn.Module):
    """Apply a validation-fitted positive affine transform to binary logits."""
    def __init__(self,model:torch.nn.Module,scale:float,bias:float)->None:
        super().__init__();self.model=model
        self.register_buffer("scale",torch.tensor(float(scale)))
        self.register_buffer("bias",torch.tensor(float(bias)))
    def forward(self,x:torch.Tensor,return_features:bool=True,**kwargs):
        output=self.model(x,return_features=return_features,**kwargs)
        return {**output,"logits":output["logits"]*self.scale+self.bias}


def model_selection_score(metrics:dict,task_type:str,training_config:dict) -> float:
    """Return the configured validation objective without consulting test data."""
    requested=str(training_config.get("save_best_by","")).removeprefix("val_")
    if requested in {"reliability_score","risk_adjusted_score"}:
        weights={"dice":0.7,"boundary_f1":0.1,"precision":0.1,"small_component_recall":0.05,"false_negative_lesion_rate":0.05}
        weights.update(training_config.get("reliability_weights",{}))
        if task_type=="classification":
            classification={"auc":0.4,"balanced_accuracy":0.3,"worst_class_sensitivity":0.2,"ece":0.1}
            classification.update(training_config.get("reliability_weights",{}));weights=classification
            values={"auc":metrics.get("auc",0.),"balanced_accuracy":metrics.get("balanced_accuracy",0.),"worst_class_sensitivity":metrics.get("worst_class_sensitivity",0.),"ece":1-float(metrics.get("ece",1.))}
        else:
            values={"dice":metrics.get("dice",0.),"boundary_f1":metrics.get("boundary_f1",0.),"precision":metrics.get("precision",0.),"small_component_recall":metrics.get("small_component_recall",0.),"false_negative_lesion_rate":1-float(metrics.get("false_negative_lesion_rate",1.))}
        total=sum(max(float(weights.get(key,0.)),0.) for key in values)
        return sum(max(float(weights.get(key,0.)),0.)*float(value) for key,value in values.items())/max(total,1e-8)
    if requested and requested in metrics:return float(metrics[requested])
    return float(metrics["accuracy"] if task_type=="classification" else metrics["dice"])


def is_model_selection_improvement(
    score:float,
    best:float,
    minimum_delta:float=0.0,
    primary_score:float|None=None,
    reference_primary:float|None=None,
    max_primary_drop:float|None=None,
    best_primary_seen:float|None=None,
    selected_primary:float|None=None,
)->bool:
    """Apply reliability improvement and optional primary-performance safety gates."""
    primary_reference=best_primary_seen if best_primary_seen is not None else reference_primary
    if max_primary_drop is not None and primary_score is not None and primary_reference is not None:
        minimum_primary=float(primary_reference)-max(float(max_primary_drop),0.0)
        if float(primary_score)<minimum_primary:
            return False
        if selected_primary is not None and float(selected_primary)<minimum_primary:
            return True
    return float(score)>float(best)+max(float(minimum_delta),0.0)


class Trainer:
    """Own the train/validation/model-selection/final-test lifecycle."""
    def __init__(self,config:dict,mode:str="racmed",debug:bool=False,kd_type:str="vanilla"):
        if mode=="kd" and kd_type not in {"vanilla","attention","relation"}:
            raise ValueError(f"Unsupported retained KD baseline: {kd_type}")
        self.config=config;self.mode=mode;self.debug=debug;self.kd_type=kd_type
        configured_label=config.get("reporting",{}).get("method_label")
        self.method_name=public_method_name(configured_label,kd_type) if configured_label else public_method_name(mode,kd_type)
        self.output=Path(config.get("output_dir","outputs/debug" if debug else "outputs/experiment")); self.output.mkdir(parents=True,exist_ok=True)
        for folder in ("checkpoints","logs/tensorboard","predictions","visualizations/risk_maps","visualizations/boundary_errors","visualizations/disagreement_maps","compression","deployment"): (self.output/folder).mkdir(parents=True,exist_ok=True)
        self.logger=configure_logger(self.output); seed_everything(int(config.get("seed",0)))
        (self.output/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False));save_provenance(self.output)
        self.device=torch.device("cuda" if torch.cuda.is_available() and config.get("device","auto")!="cpu" else "cpu")
        self.task=config.get("task_type",config.get("model",{}).get("task_type","segmentation2d"));self.classes=int(config.get("model",{}).get("num_classes",3))
        self.loaders=create_dataloaders(config,debug);role="teacher" if mode=="teacher" else "compressed_model"
        self.model=create_model(config,role).to(self.device)
        self.teacher=None
        if mode in ("racmed","kd"):
            self.teacher=create_model(config,"teacher").to(self.device).eval()
            for parameter in self.teacher.parameters():parameter.requires_grad_(False)
        t=config.get("training",{});lr=float(t.get("lr",1e-3));wd=float(t.get("weight_decay",1e-5))
        optimizer_name=str(t.get("optimizer","adamw")).lower()
        if optimizer_name=="sgd":
            self.optimizer=SGD(self.model.parameters(),lr=lr,momentum=.9,weight_decay=wd)
        elif optimizer_name=="adam":
            self.optimizer=Adam(self.model.parameters(),lr=lr,weight_decay=wd)
        else:
            self.optimizer=AdamW(self.model.parameters(),lr=lr,weight_decay=wd)
        self.amp_enabled=bool(t.get("amp",False) and self.device.type=="cuda")
        train_loader=self.loaders["train"]
        self.logger.info(
            "runtime device=%s amp=%s pin_memory=%s workers=%d prefetch_factor=%s",
            self.device,
            self.amp_enabled,
            train_loader.pin_memory,
            train_loader.num_workers,
            getattr(train_loader,"prefetch_factor",None),
        )
        self.scaler=torch.amp.GradScaler(self.device.type,enabled=self.amp_enabled,init_scale=float(t.get("amp_init_scale",65536.0)))
        self.optimizer_steps=0
        self.gradient_clip_norm=float(t.get("gradient_clip_norm",0.0))
        self.freeze_batch_norm=bool(t.get("freeze_batch_norm",False))
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=max(int(t.get("epochs",1)),1)) if t.get("scheduler","cosine")=="cosine" else None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer=SummaryWriter(self.output/"logs/tensorboard")
        except ImportError:self.writer=None
        loss_cfg=config.get("loss",{})
        self.kd_weight=float(loss_cfg.get("baseline_kd_weight",loss_cfg.get("lambda_global",1.0)))
        self.teacher_features_required=(
            (mode=="racmed" and (float(loss_cfg.get("lambda_local",1.0))>0 or float(loss_cfg.get("lambda_relation",0.1))>0))
            or (mode=="kd" and kd_type in {"attention","relation"})
        )
        self.cache_teacher_outputs=bool(t.get("cache_teacher_outputs",False))
        self.teacher_cache:dict[str,dict[str,object]]={}
        cache_root=t.get("teacher_cache_dir")
        self.teacher_cache_dir=(Path(cache_root)/f"seed{int(config.get('seed',0))}" if cache_root else None)
        if self.teacher_cache_dir is not None:self.teacher_cache_dir.mkdir(parents=True,exist_ok=True)
        self.cache_static_risk=bool(t.get("cache_static_risk",False))
        self.static_risk_cache:dict[str,dict[str,torch.Tensor]]={}
        self.supervised_kwargs={k:loss_cfg[k] for k in ("focal_gamma","dice_include_background","dice_squared","dice_batch") if k in loss_cfg}
        self.criterion=RACMedLoss(**{k:v for k,v in loss_cfg.items() if k in {"temperature","lambda_global","lambda_local","lambda_boundary","lambda_relation","lambda_calib","lambda_sparse","focal_gamma","dice_include_background","dice_squared","dice_batch","local_feature_mode","risk_supervision_strength","risk_supervision_foreground_only","component_supervision_strength","local_logit_weight","teacher_correctness_gating","risk_gating_enabled"}})
        risk_cfg=dict(config.get("risk",{}))
        temperature_setting=risk_cfg.get("teacher_temperature",1.0)
        max_temperature_batches=risk_cfg.pop("teacher_temperature_max_batches",None)
        if isinstance(temperature_setting,str) and temperature_setting.lower()=="auto":
            if mode=="racmed":
                if self.task!="classification" or self.teacher is None:
                    raise ValueError("Automatic teacher temperature requires RACMed classification training.")
                risk_cfg["teacher_temperature"]=self._fit_teacher_risk_temperature(max_temperature_batches)
            else:
                risk_cfg["teacher_temperature"]=1.0
        else:
            risk_cfg["teacher_temperature"]=float(temperature_setting)
        self.risk_builder=RiskMapBuilder(task_type=self.task,**{k:v for k,v in risk_cfg.items() if hasattr(RiskMapBuilder,k)})
        self.classification_threshold=None

    @torch.inference_mode()
    def _fit_teacher_risk_temperature(self,max_batches:int|None=None)->float:
        """Fit teacher confidence on validation data without consulting test samples."""
        logits=[];targets=[]
        calibration_total=min(len(self.loaders["val"]),int(max_batches)) if max_batches is not None else len(self.loaders["val"])
        calibration_source=islice(self.loaders["val"],calibration_total) if max_batches is not None else self.loaders["val"]
        for batch in progress_bar(calibration_source,desc="Teacher temperature calibration",total=calibration_total,unit="batch",leave=True):
            image=batch["image"].to(self.device,non_blocking=self.device.type=="cuda")
            with torch.autocast(device_type=self.device.type,enabled=self.amp_enabled):
                output=self._teacher_forward(image,batch)
            logits.append(output["logits"].detach().float().cpu())
            targets.append(batch["target"].detach().cpu())
        if not logits:
            self.logger.warning("Teacher risk calibration received zero validation batches; using temperature 1.0.")
            return 1.0
        temperature=fit_logit_temperature(torch.cat(logits),torch.cat(targets))
        payload={"split":"validation","temperature":temperature,"batches":len(logits)}
        (self.output/"teacher_risk_calibration.json").write_text(json.dumps(payload,indent=2))
        self.logger.info("validation-fitted teacher risk temperature=%.4f",temperature)
        return temperature

    def _teacher_forward(self,x:torch.Tensor,batch:dict) -> dict:
        """Run or reuse fixed-teacher logits for deterministic samples/prompts."""
        ids=[str(value) for value in batch.get("id",[])]
        persistent_cache=getattr(self,"teacher_cache_dir",None)
        if self.cache_teacher_outputs and persistent_cache is not None:
            for case_id in ids:
                if case_id in self.teacher_cache:continue
                stem=hashlib.sha1(case_id.encode()).hexdigest()
                path=persistent_cache/f"{stem}.pt" if self.teacher_features_required else persistent_cache/f"{stem}.logits.pt"
                fallback=persistent_cache/f"{stem}.pt"
                if not path.exists():path=fallback
                if not path.exists():continue
                entry=torch.load(path,map_location="cpu",weights_only=True)
                if self.teacher_features_required and not entry.get("features"):continue
                if not self.teacher_features_required:entry={"logits":entry["logits"],"features":{}}
                self.teacher_cache[case_id]=entry
        if self.cache_teacher_outputs and ids and all(case_id in self.teacher_cache for case_id in ids):
            entries=[self.teacher_cache[case_id] for case_id in ids]
            logits=torch.stack([entry["logits"] for entry in entries]).to(device=x.device,dtype=x.dtype)
            features={}
            if self.teacher_features_required:
                for key in entries[0]["features"]:
                    features[key]=torch.stack([entry["features"][key] for entry in entries]).to(device=x.device,dtype=x.dtype)
            return {"logits":logits,"features":features}
        with torch.no_grad():
            output=forward_model(self.teacher,x,batch,self.teacher_features_required)
        if self.cache_teacher_outputs and ids:
            cpu_logits=output["logits"].detach().to(device="cpu",dtype=torch.float16)
            cpu_features={key:tensor.detach().to(device="cpu",dtype=torch.float16) for key,tensor in output.get("features",{}).items()}
            for index,case_id in enumerate(ids):
                features={key:tensor[index].clone() for key,tensor in cpu_features.items()}
                entry={"logits":cpu_logits[index].clone(),"features":features}
                self.teacher_cache[case_id]=entry
                if persistent_cache is not None:
                    stem=hashlib.sha1(case_id.encode()).hexdigest();torch.save(entry,persistent_cache/f"{stem}.pt")
                    torch.save({"logits":entry["logits"],"features":{}},persistent_cache/f"{stem}.logits.pt")
            if not self.teacher_features_required:output["features"]={}
        return output

    def _risk_forward(self,output:dict,teacher:dict,y:torch.Tensor,x:torch.Tensor,batch:dict) -> dict:
        """Reuse target-only risk factors while recomputing prediction-dependent risk."""
        ids=[str(value) for value in batch.get("id",[])]
        if self.cache_static_risk and ids and all(case_id in self.static_risk_cache for case_id in ids):
            static={key:torch.stack([self.static_risk_cache[case_id][key] for case_id in ids]).to(self.device,non_blocking=self.device.type=="cuda") for key in ("boundary_score","small_component_score","small_component_weight","class_rarity")}
            return self.risk_builder(output["logits"].detach(),teacher["logits"],y,x,static_outputs=static)
        risk=self.risk_builder(output["logits"].detach(),teacher["logits"],y,x)
        if self.cache_static_risk and ids:
            cached_keys=("boundary_score","small_component_score","small_component_weight","class_rarity")
            cpu_risk={key:risk[key].detach().to(device="cpu",dtype=torch.float16) for key in cached_keys}
            for index,case_id in enumerate(ids):
                self.static_risk_cache[case_id]={key:value[index].clone() for key,value in cpu_risk.items()}
        return risk

    def _epoch(self,epoch:int|None=None,total_epochs:int|None=None):
        set_model_training_mode(self.model,self.freeze_batch_norm);totals={};accum=int(self.config.get("training",{}).get("gradient_accumulation_steps",1));self.optimizer.zero_grad(set_to_none=True)
        amp=self.amp_enabled
        max_train_batches=self.config.get("training",{}).get("max_train_batches")
        train_limit=int(max_train_batches) if max_train_batches is not None else len(self.loaders["train"])
        progress_interval=max(1,train_limit//10)
        seen_batches=0
        batches=progress_bar(
            islice(self.loaders["train"],train_limit),
            desc=f"Train epoch {('?' if epoch is None else epoch+1)}/{('?' if total_epochs is None else total_epochs)}",
            total=train_limit,
            unit="batch",
            leave=False,
        )
        for step,batch in enumerate(batches):
            asynchronous=self.device.type=="cuda"
            x,y=batch["image"].to(self.device,non_blocking=asynchronous),batch["target"].to(self.device,non_blocking=asynchronous)
            with torch.autocast(device_type=self.device.type,enabled=amp):
                output=forward_model(self.model,x,batch,True)
                if self.mode=="racmed":
                    teacher=self._teacher_forward(x,batch)
                    risk=self._risk_forward(output,teacher,y,x,batch)
                    losses=self.criterion(output,teacher,y,x,risk,self.task,self.model)
                elif self.mode=="kd":
                    teacher=self._teacher_forward(x,batch)
                    sup=supervised_loss(output["logits"],y,self.task,**self.supervised_kwargs)
                    sf=output["features"].get("bottleneck",output["logits"]);tf=teacher["features"].get("bottleneck",teacher["logits"])
                    if self.kd_type=="attention":kd=attention_transfer(sf,tf)
                    elif self.kd_type=="relation":kd=relation_distillation(sf,tf,None if self.task=="classification" else y)
                    else:kd=softened_bernoulli_kl(output["logits"],teacher["logits"]) if y.ndim==output["logits"].ndim else softened_kl(output["logits"],teacher["logits"])
                    losses={"total":sup+self.kd_weight*kd,"sup":sup,"global_kd":kd}
                else:
                    sup=supervised_loss(output["logits"],y,self.task,**self.supervised_kwargs);losses={"total":sup,"sup":sup}
                loss=losses["total"]/accum
            self.scaler.scale(loss).backward()
            if (step+1)%accum==0 or step+1==train_limit or step+1==len(self.loaders["train"]):
                if self.gradient_clip_norm>0:
                    self.scaler.unscale_(self.optimizer);torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.gradient_clip_norm,error_if_nonfinite=False)
                scale_before=self.scaler.get_scale()
                self.scaler.step(self.optimizer);self.scaler.update()
                if not self.amp_enabled or self.scaler.get_scale()>=scale_before:self.optimizer_steps+=1
                self.optimizer.zero_grad(set_to_none=True)
            for key,value in losses.items():totals[key]=totals.get(key,0.)+float(value.detach())
            seen_batches+=1
            batches.set_postfix(loss=f"{totals['total']/seen_batches:.4f}")
            if seen_batches % progress_interval == 0 or seen_batches == train_limit:
                self.logger.info(
                    "train_progress epoch=%s/%s batch=%d/%d loss=%.4f",
                    "?" if epoch is None else epoch + 1,
                    "?" if total_epochs is None else total_epochs,
                    seen_batches,
                    train_limit,
                    totals["total"] / seen_batches,
                )
        if seen_batches==0: raise ValueError("Training received zero batches; max_train_batches must be positive when set.")
        return {key:value/max(seen_batches,1) for key,value in totals.items()}

    def fit(self):
        t=self.config.get("training",{});epochs=1 if self.debug else int(t.get("epochs",100));patience=int(t.get("early_stopping_patience",30));validation_interval=max(int(t.get("validation_interval",1)),1);best=-float("inf");bad=0;history=[]
        resume=t.get("resume")
        if resume:load_checkpoint(resume,self.model,self.optimizer)
        max_eval_batches=t.get("max_val_batches",t.get("max_eval_batches"))
        eval_cfg=self.config.get("evaluation",{})
        small_component_threshold=int(eval_cfg.get("small_component_threshold",self.config.get("risk",{}).get("small_component_threshold",64)))
        threshold_strategy=eval_cfg.get("threshold_strategy") if self.task=="classification" else None
        minimum_delta=float(t.get("model_selection_min_delta",0.0))
        primary_metric=str(t.get("model_selection_primary_metric","accuracy" if self.task=="classification" else "dice"))
        max_primary_drop=t.get("model_selection_max_primary_drop")
        reference_primary=None;best_primary_seen=None;selected_primary=None
        if bool(t.get("evaluate_initial_checkpoint",False)):
            metrics,_=evaluate(self.model,self.loaders["val"],self.task,self.device,self.classes,max_batches=max_eval_batches,threshold_strategy=threshold_strategy,sliding_window=eval_cfg.get("sliding_window"),progress_label="validation:initial",small_component_threshold=small_component_threshold)
            score=model_selection_score(metrics,self.task,t);best=score
            reference_primary=float(metrics[primary_metric]) if primary_metric in metrics else None
            best_primary_seen=reference_primary;selected_primary=reference_primary
            self.classification_threshold=metrics.get("threshold",self.classification_threshold)
            save_checkpoint(self.output/"checkpoints/best.pt",self.model,self.optimizer,-1,metrics)
            history.append({"epoch":-1,"val_model_selection_score":score,**{f"val_{key}":value for key,value in metrics.items() if isinstance(value,(int,float))}})
            self.logger.info("epoch=-1 initial_checkpoint val_score=%.4f",score)
        for epoch in range(epochs):
            steps_before=self.optimizer_steps;losses=self._epoch(epoch,epochs);should_validate=epoch%validation_interval==0 or epoch==epochs-1
            metrics={};score=None
            if should_validate:
                metrics,_=evaluate(self.model,self.loaders["val"],self.task,self.device,self.classes,max_batches=max_eval_batches,threshold_strategy=threshold_strategy,sliding_window=eval_cfg.get("sliding_window"),progress_label=f"validation:epoch_{epoch+1}",small_component_threshold=small_component_threshold)
                score=model_selection_score(metrics,self.task,t)
            row={"epoch":epoch,**({"val_model_selection_score":score} if score is not None else {}),**{f"train_{k}":v for k,v in losses.items()},**{f"val_{k}":v for k,v in metrics.items() if isinstance(v,(int,float))}};history.append(row)
            save_checkpoint(self.output/"checkpoints/last.pt",self.model,self.optimizer,epoch,metrics)
            if score is not None:
                primary_score=float(metrics[primary_metric]) if primary_metric in metrics else None
                if primary_score is not None:
                    best_primary_seen=primary_score if best_primary_seen is None else max(best_primary_seen,primary_score)
                if is_model_selection_improvement(
                    score,best,minimum_delta,
                    primary_score=primary_score,
                    reference_primary=reference_primary,
                    max_primary_drop=max_primary_drop,
                    best_primary_seen=best_primary_seen,
                    selected_primary=selected_primary,
                ):
                    best=score;bad=0;selected_primary=primary_score;self.classification_threshold=metrics.get("threshold",self.classification_threshold);save_checkpoint(self.output/"checkpoints/best.pt",self.model,self.optimizer,epoch,metrics)
                else:bad+=1
                self.logger.info("epoch=%d loss=%.4f val_score=%.4f",epoch,losses["total"],score)
            else:self.logger.info("epoch=%d loss=%.4f validation=skipped",epoch,losses["total"])
            if self.scheduler and self.optimizer_steps>steps_before:self.scheduler.step()
            if self.writer:
                for key,value in losses.items():self.writer.add_scalar(f"train/{key}",value,epoch)
                if score is not None:self.writer.add_scalar("validation/model_selection_score",score,epoch)
            if epoch==0 and self.mode=="racmed":self._save_risk_visualizations()
            if should_validate and bad>=patience:break
        if self.writer:self.writer.close()
        pd.DataFrame(history).to_csv(self.output/"training_history.csv",index=False)
        load_checkpoint(self.output/"checkpoints/best.pt",self.model)
        return history

    @torch.inference_mode()
    def _save_risk_visualizations(self):
        """Persist a representative validation risk decomposition without touching test data."""
        if self.task=="classification" or self.task=="segmentation3d":
            return
        import matplotlib;matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        batch=next(iter(self.loaders["val"]));asynchronous=self.device.type=="cuda";x,y=batch["image"].to(self.device,non_blocking=asynchronous),batch["target"].to(self.device,non_blocking=asynchronous);compressed=forward_model(self.model,x,batch,True);teacher=self._teacher_forward(x,batch);risk=self.risk_builder(compressed["logits"],teacher["logits"],y,x)
        def _display_slice(tensor):
            array=tensor.detach().cpu().numpy()
            if array.ndim==3:
                return array[array.shape[0]//2]
            return array
        image=_display_slice(x[0].mean(0));names=("risk_map","boundary_score","disagreement")
        for name in names:
            value=_display_slice(risk[name][0]);fig,axes=plt.subplots(1,2,figsize=(6,3));axes[0].imshow(image,cmap="gray");axes[0].axis("off");axes[1].imshow(value,cmap="magma",vmin=0,vmax=1);axes[1].axis("off");fig.tight_layout();folder={"risk_map":"risk_maps","boundary_score":"boundary_errors","disagreement":"disagreement_maps"}[name];fig.savefig(self.output/f"visualizations/{folder}/epoch000.png",dpi=160);plt.close(fig)

    def test(self):
        training=self.config.get("training",{})
        max_eval_batches=training.get("max_test_batches",training.get("max_eval_batches"))
        evaluation=self.config.get("evaluation",{});threshold=self.classification_threshold;model=self.model
        if self.task.startswith("segmentation") and evaluation.get("fixed_segmentation_threshold") is not None:
            threshold=evaluation["fixed_segmentation_threshold"]
        small_component_threshold=int(evaluation.get("small_component_threshold",self.config.get("risk",{}).get("small_component_threshold",64)))
        fixed_classification=evaluation.get("fixed_classification_calibration")
        if self.task=="classification" and fixed_classification is not None:
            model=BinaryPlattCalibration(
                self.model,
                float(fixed_classification.get("scale",1.0)),
                float(fixed_classification.get("bias",0.0)),
            ).to(self.device)
            threshold=float(fixed_classification.get("threshold",threshold))
        elif self.task=="classification" and evaluation.get("classification_logit_calibration",False):
            logits=[];targets=[];max_calibration_batches=training.get("max_val_batches",training.get("max_eval_batches"))
            self.model.eval()
            calibration_total=min(len(self.loaders["val"]),int(max_calibration_batches)) if max_calibration_batches is not None else len(self.loaders["val"])
            calibration_source=islice(self.loaders["val"],calibration_total) if max_calibration_batches is not None else self.loaders["val"]
            for batch in progress_bar(calibration_source,desc="Classification calibration",total=calibration_total,unit="batch",leave=True):
                image=batch["image"].to(self.device,non_blocking=self.device.type=="cuda")
                with torch.autocast(device_type=self.device.type,enabled=self.amp_enabled):output=forward_model(self.model,image,batch,False)
                logits.append(output["logits"].detach().float().cpu());targets.append(batch["target"].detach().cpu())
            if not logits:raise ValueError("Classification calibration received zero validation batches.")
            values=torch.cat(logits);labels=torch.cat(targets)
            if values.ndim!=2 or values.shape[1]!=1:raise ValueError("Platt calibration currently requires one binary logit.")
            scale,bias=fit_binary_platt_scaling(values,labels);model=BinaryPlattCalibration(self.model,scale,bias).to(self.device)
            validation_metrics,_=evaluate(model,self.loaders["val"],self.task,self.device,self.classes,max_batches=max_calibration_batches,threshold_strategy=evaluation.get("threshold_strategy"),progress_label="validation:calibration")
            threshold=validation_metrics.get("threshold",threshold)
            (self.output/"classification_calibration.json").write_text(json.dumps({"split":"validation","scale":scale,"bias":bias,"threshold":threshold},indent=2))
        if self.task.startswith("segmentation") and evaluation.get("background_logit_bias_calibration",False):
            background_bias=calibrate_background_logit_bias(
                self.model,self.loaders["val"],self.device,self.task,self.classes,
                candidates=evaluation.get("background_logit_bias_candidates",[0.0]),
                objective=evaluation.get("background_logit_bias_objective","dice"),
                max_batches=training.get("max_val_batches",training.get("max_eval_batches")),
                sliding_window=evaluation.get("sliding_window"),
                reliability_weights=evaluation.get("background_logit_bias_weights",training.get("reliability_weights")),
                small_component_threshold=small_component_threshold,
            )
            model=LogitBiasCalibration(self.model,[background_bias,*([0.]*(self.classes-1))]).to(self.device)
            (self.output/"segmentation_bias_calibration.json").write_text(json.dumps({"split":"validation","background_bias":background_bias},indent=2))
        if self.task.startswith("segmentation") and evaluation.get("segmentation_threshold_calibration",False) and evaluation.get("fixed_segmentation_threshold") is None:
            threshold=calibrate_segmentation_thresholds(
                model,
                self.loaders["val"],
                self.device,
                candidates=evaluation.get("threshold_candidates"),
                max_batches=training.get("max_val_batches",training.get("max_eval_batches")),
                sliding_window=evaluation.get("sliding_window"),
                objective=evaluation.get("segmentation_threshold_objective","dice"),
                reliability_weights=evaluation.get("segmentation_threshold_weights",training.get("reliability_weights")),
                max_dice_drop=evaluation.get("segmentation_threshold_max_dice_drop"),
                small_component_threshold=small_component_threshold,
            )
            (self.output/"segmentation_threshold_calibration.json").write_text(json.dumps({
                "split":"validation",
                "threshold":threshold,
                "objective":evaluation.get("segmentation_threshold_objective","dice"),
                "max_dice_drop":evaluation.get("segmentation_threshold_max_dice_drop"),
            },indent=2))
        metrics,cases=evaluate(model,self.loaders["test"],self.task,self.device,self.classes,max_batches=max_eval_batches,threshold=threshold,sliding_window=evaluation.get("sliding_window"),progress_label="final_test",small_component_threshold=small_component_threshold)
        (self.output/"metrics.json").write_text(json.dumps(_jsonable({"test":metrics,"mode":self.mode,"method":self.method_name,"kd_type":self.kd_type if self.mode=="kd" else None}),indent=2))
        pd.DataFrame(cases).to_csv(self.output/"per_case_metrics.csv",index=False)
        flat={k:v for k,v in metrics.items() if isinstance(v,(int,float))};pd.DataFrame([{**flat,"method":self.method_name,"mode":self.mode,"kd_type":self.kd_type if self.mode=="kd" else None,"task":self.task,"seed":self.config.get("seed",0),"score":flat.get("accuracy",flat.get("dice",0.))}]).to_csv(self.output/"summary.csv",index=False)
        return metrics


def run_training(config:dict,mode:str="racmed",debug:bool=False,kd_type:str="vanilla"):
    trainer=Trainer(config,mode,debug,kd_type);history=trainer.fit();test=trainer.test();return {"history":history,"test":test,"output_dir":str(trainer.output)}
