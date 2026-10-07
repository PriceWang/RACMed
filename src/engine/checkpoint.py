"""Atomic model training-state checkpoints."""

from pathlib import Path
import torch


def save_checkpoint(path,model,optimizer=None,epoch=0,metrics=None,serialize_model:bool=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
    payload={"model":model.state_dict(),"optimizer":optimizer.state_dict() if optimizer else None,"epoch":epoch,"metrics":metrics or {}}
    if serialize_model:payload.update({"model":None,"model_object":model,"structural_model":True})
    torch.save(payload,tmp);tmp.replace(path)


def load_checkpoint(path,model,optimizer=None):
    state=torch.load(path,map_location="cpu",weights_only=False)
    if state.get("structural_model"):
        raise ValueError("Structural checkpoints must be restored with load_model_checkpoint().")
    model.load_state_dict(state["model"])
    if optimizer and state.get("optimizer"):optimizer.load_state_dict(state["optimizer"])
    return state


def load_model_checkpoint(path,model):
    """Restore a regular state dict or return a serialized structural model."""

    state=torch.load(path,map_location="cpu",weights_only=False)
    restored=state.get("model_object")
    if restored is None:
        model.load_state_dict(state["model"]);restored=model
    return restored,state
