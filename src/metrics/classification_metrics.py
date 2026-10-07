"""Classification discrimination and reliability metrics."""

from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from .calibration_metrics import brier_score, expected_calibration_error, negative_log_likelihood


def _youden_threshold(prob: np.ndarray, y: np.ndarray) -> float:
    candidates = np.unique(prob)
    if candidates.size == 0:
        return 0.5
    best_threshold = float(candidates[0])
    best_score = -float("inf")
    for threshold in candidates:
        pred = prob >= threshold
        cm = confusion_matrix(y, pred.astype(int), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        score = sensitivity + specificity - 1
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float | None = None,
    threshold_strategy: str | None = None,
) -> dict[str, object]:
    if targets.ndim==logits.ndim:
        p=logits.sigmoid().detach().cpu().numpy();y=targets.detach().cpu().numpy().astype(int);pred=(p>=.5).astype(int)
        sensitivity=[];specificity=[]
        for i in range(y.shape[1]):
            cm=confusion_matrix(y[:,i],pred[:,i],labels=[0,1]);tn,fp,fn,tp=cm.ravel();sensitivity.append(tp/max(tp+fn,1));specificity.append(tn/max(tn+fp,1))
        try:auc=roc_auc_score(y,p,average="macro")
        except ValueError:auc=float("nan")
        return {"accuracy":float((pred==y).all(1).mean()),"balanced_accuracy":float((np.mean(sensitivity)+np.mean(specificity))/2),"auc":float(auc),"macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),"precision":float(precision_score(y,pred,average="macro",zero_division=0)),"recall":float(recall_score(y,pred,average="macro",zero_division=0)),"sensitivity":float(np.mean(sensitivity)),"specificity":float(np.mean(specificity)),"worst_class_sensitivity":float(np.min(sensitivity)),"class_wise":{str(i):{"sensitivity":sensitivity[i],"specificity":specificity[i]} for i in range(y.shape[1])},"nll":negative_log_likelihood(logits,targets),"brier":brier_score(logits,targets),"ece":expected_calibration_error(logits,targets)}
    if logits.shape[1] == 1:
        prob = logits.sigmoid().squeeze(1).detach().cpu().numpy()
        y = targets.detach().cpu().numpy().astype(int)
        if threshold_strategy == "youden":
            threshold = _youden_threshold(prob, y)
        threshold = 0.5 if threshold is None else float(threshold)
        pred = (prob >= threshold).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = np.array([tn / max(tn + fp, 1), tp / max(tp + fn, 1)], dtype=float)
        specificity = np.array([tp / max(tp + fn, 1), tn / max(tn + fp, 1)], dtype=float)
        try:
            auc = roc_auc_score(y, prob)
        except ValueError:
            auc = float("nan")
        classwise = {str(i): {"sensitivity": float(sensitivity[i]), "specificity": float(specificity[i])} for i in [0, 1]}
        return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "auc": float(auc), "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "precision": float(precision_score(y, pred, average="macro", zero_division=0)),
                "recall": float(recall_score(y, pred, average="macro", zero_division=0)),
                "sensitivity": float(sensitivity.mean()), "specificity": float(specificity.mean()),
                "worst_class_sensitivity": float(sensitivity.min()), "class_wise": classwise,
                "nll": negative_log_likelihood(logits, targets), "brier": brier_score(logits, targets),
                "ece": expected_calibration_error(logits, targets), "threshold": threshold}
    probs = logits.softmax(1) if logits.shape[1] > 1 else torch.cat((1-logits.sigmoid(), logits.sigmoid()), 1)
    y, p = targets.detach().cpu().numpy(), probs.detach().cpu().numpy()
    pred = p.argmax(1); labels = np.arange(p.shape[1])
    cm = confusion_matrix(y, pred, labels=labels)
    tp = np.diag(cm); fn = cm.sum(1) - tp; fp = cm.sum(0) - tp; tn = cm.sum() - tp - fn - fp
    sensitivity = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=tp+fn>0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=tn+fp>0)
    try:
        auc = roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else roc_auc_score(y, p, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    classwise = {str(i): {"sensitivity": float(sensitivity[i]), "specificity": float(specificity[i])} for i in labels}
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "auc": float(auc), "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "precision": float(precision_score(y, pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y, pred, average="macro", zero_division=0)),
            "sensitivity": float(sensitivity.mean()), "specificity": float(specificity.mean()),
            "worst_class_sensitivity": float(sensitivity.min()), "class_wise": classwise,
            "nll": negative_log_likelihood(logits, targets), "brier": brier_score(logits, targets),
            "ece": expected_calibration_error(logits, targets)}
