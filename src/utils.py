from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import get_worker_info
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def seed_data_worker(worker_id: int) -> None:
    info = get_worker_info()
    if info is None:
        return
    worker_seed = int(info.seed % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    if hasattr(info.dataset, "set_worker_seed"):
        info.dataset.set_worker_seed(worker_seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(state: dict, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_num_params(num_params: int) -> float:
    return num_params / 1_000_000.0


def softmax_iou(confusion_matrix: np.ndarray) -> float:
    intersection = np.diag(confusion_matrix)
    union = (
        confusion_matrix.sum(axis=1)
        + confusion_matrix.sum(axis=0)
        - intersection
    )
    iou = intersection / np.clip(union, a_min=1.0, a_max=None)
    return float(np.mean(iou))


def per_class_iou(confusion_matrix: np.ndarray) -> np.ndarray:
    intersection = np.diag(confusion_matrix).astype(np.float64)
    union = (
        confusion_matrix.sum(axis=1).astype(np.float64)
        + confusion_matrix.sum(axis=0).astype(np.float64)
        - intersection
    )
    return intersection / np.clip(union, a_min=1.0, a_max=None)


def overall_accuracy(confusion_matrix: np.ndarray) -> float:
    correct = float(np.diag(confusion_matrix).sum())
    total = float(confusion_matrix.sum())
    return correct / max(total, 1.0)


def confusion_matrix_from_predictions(
    preds: np.ndarray, targets: np.ndarray, num_classes: int
) -> np.ndarray:
    mask = (targets >= 0) & (targets < num_classes)
    labels = num_classes * targets[mask].astype(int) + preds[mask].astype(int)
    counts = np.bincount(labels, minlength=num_classes**2)
    return counts.reshape(num_classes, num_classes)


def env_or_default(name: str, default: str) -> str:
    return os.environ.get(name, default)
