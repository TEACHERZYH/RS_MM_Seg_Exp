from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import confusion_matrix_from_predictions, softmax_iou


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    train_cfg = config["train"]
    if train_cfg["optimizer"].lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=train_cfg["lr"],
            momentum=0.9,
            weight_decay=train_cfg["weight_decay"],
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )


def train_one_epoch(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    model.train()
    teacher_model.eval()

    meters = defaultdict(float)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device)
        aux = batch["aux"].to(device)
        mask = batch["mask"].to(device)
        aux_available = batch["aux_available"].to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            teacher_out = teacher_model(image, aux, torch.ones_like(aux_available))
        student_out = model(image, aux, aux_available)
        losses = criterion(student_out, teacher_out, mask)
        losses["total"].backward()
        optimizer.step()

        preds = student_out["logits"].argmax(dim=1).detach().cpu().numpy()
        targets = mask.detach().cpu().numpy()
        confusion += confusion_matrix_from_predictions(preds, targets, num_classes)

        for key, value in losses.items():
            meters[key] += float(value.item())

    num_batches = max(len(loader), 1)
    summary = {key: value / num_batches for key, value in meters.items()}
    summary["miou"] = softmax_iou(confusion)
    return summary


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, float]:
    model.eval()
    meters = defaultdict(float)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in tqdm(loader, desc="eval", leave=False):
        image = batch["image"].to(device)
        aux = batch["aux"].to(device)
        mask = batch["mask"].to(device)
        aux_available = batch["aux_available"].to(device)

        outputs = model(image, aux, aux_available)
        losses = criterion(outputs, None, mask)
        preds = outputs["logits"].argmax(dim=1).cpu().numpy()
        targets = mask.cpu().numpy()
        confusion += confusion_matrix_from_predictions(preds, targets, num_classes)

        for key, value in losses.items():
            meters[key] += float(value.item())

    num_batches = max(len(loader), 1)
    summary = {key: value / num_batches for key, value in meters.items()}
    summary["miou"] = softmax_iou(confusion)
    return summary
