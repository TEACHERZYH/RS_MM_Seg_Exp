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
    teacher_model: torch.nn.Module | None,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    distill_weight: float = 1.0,
    grad_clip_norm: float = 0.0,
    ema_decay: float = 0.0,
) -> dict[str, float]:
    model.train()
    if teacher_model is not None:
        teacher_model.eval()

    meters = defaultdict(float)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device)
        aux = batch["aux"].to(device)
        mask = batch["mask"].to(device)
        aux_available = batch["aux_available"].to(device)
        main_available = batch.get("main_available", torch.ones_like(aux_available)).to(device)

        optimizer.zero_grad()
        teacher_out = None
        if teacher_model is not None and distill_weight > 0.0:
            teacher_image = batch.get("teacher_image", image).to(device)
            teacher_aux = batch.get("teacher_aux", aux).to(device)
            with torch.no_grad():
                teacher_out = teacher_model(teacher_image, teacher_aux, torch.ones_like(aux_available))
        student_out = model(image, aux, aux_available, main_available)
        losses = criterion(student_out, teacher_out, mask, distill_weight=distill_weight)
        losses["total"].backward()
        if grad_clip_norm and grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        if teacher_model is not None and ema_decay > 0.0:
            update_ema_model(teacher_model, model, ema_decay)

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
def update_ema_model(
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    decay: float,
) -> None:
    teacher_state = teacher_model.state_dict()
    student_state = student_model.state_dict()
    for name, teacher_value in teacher_state.items():
        student_value = student_state[name].detach()
        if teacher_value.dtype.is_floating_point:
            teacher_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
        else:
            teacher_value.copy_(student_value)


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
        main_available = batch.get("main_available", torch.ones_like(aux_available)).to(device)

        outputs = model(image, aux, aux_available, main_available)
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
