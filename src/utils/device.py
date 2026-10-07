"""Runtime device selection."""
import torch


def get_device(preference="auto"):return torch.device("cuda" if preference!="cpu" and torch.cuda.is_available() else "cpu")
