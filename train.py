from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.isprs_dataset import ISPRSMultimodalDataset
from src.engine import build_optimizer, evaluate, train_one_epoch
from src.losses import SegmentationCriterion
from src.models.qalf_net import QALFNet
from src.utils import ensure_dir, format_num_params, load_config, resolve_device, save_checkpoint, set_seed


def build_dataset(config: dict, split_name: str, training: bool) -> ISPRSMultimodalDataset:
    ds = config["dataset"]
    split_file = str(Path(ds["split_dir"]) / ds[split_name])
    return ISPRSMultimodalDataset(
        root_dir=ds["root_dir"],
        split_file=split_file,
        image_dir=ds["image_dir"],
        aux_dir=ds["aux_dir"],
        mask_dir=ds["mask_dir"],
        image_suffix=ds["image_suffix"],
        aux_suffix=ds["aux_suffix"],
        mask_suffix=ds["mask_suffix"],
        input_size=ds["input_size"],
        missing_prob=ds["missing_prob"] if training else 0.0,
        degradation_prob=ds["degradation_prob"] if training else 0.0,
        normalize_aux=ds["normalize_aux"],
        training=training,
        augment=ds.get("augment", False),
        hflip_prob=ds.get("hflip_prob", 0.0),
        vflip_prob=ds.get("vflip_prob", 0.0),
        rotate90_prob=ds.get("rotate90_prob", 0.0),
        color_jitter=ds.get("color_jitter", 0.0),
        missing_target=ds.get("missing_target", "aux"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])
    device = resolve_device(config["train"]["device"])
    output_dir = ensure_dir(config["experiment"]["output_dir"])
    log_path = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
    )
    logger = logging.getLogger("train")

    train_dataset = build_dataset(config, "train_split", training=True)
    val_dataset = build_dataset(config, "val_split", training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config["eval"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = QALFNet(**config["model"]).to(device)
    teacher_model = QALFNet(**config["model"]).to(device)
    teacher_model.load_state_dict(model.state_dict())

    criterion = SegmentationCriterion(
        num_classes=config["dataset"]["num_classes"],
        ce_weight=config["loss"]["ce_weight"],
        dice_weight=config["loss"]["dice_weight"],
        feat_weight=config["loss"]["feat_weight"],
        pred_weight=config["loss"]["pred_weight"],
    ).to(device)
    optimizer = build_optimizer(model, config)

    use_distill = (config["loss"].get("feat_weight", 0.0) > 0.0) or (config["loss"].get("pred_weight", 0.0) > 0.0)
    teacher_model = None
    if use_distill:
        teacher_model = QALFNet(**config["model"]).to(device)
        teacher_model.load_state_dict(model.state_dict())
        for param in teacher_model.parameters():
            param.requires_grad = False

    best_miou = -1.0
    start_epoch = 1
    resume_path = config["train"].get("resume")
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if use_distill and teacher_model is not None:
            teacher_model.load_state_dict(model.state_dict())
        resume_optimizer = config["train"].get("resume_optimizer", True)
        if resume_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        best_miou = checkpoint.get("best_miou", checkpoint.get("val_metrics", {}).get("miou", best_miou))
        loaded_epoch = int(checkpoint.get("epoch", 0))
        if resume_optimizer:
            start_epoch = loaded_epoch + 1
        if teacher_model is not None:
            teacher_model.load_state_dict(model.state_dict())
        if resume_optimizer:
            logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)
        else:
            logger.info("Loaded weights from %s at source epoch %d; fine-tuning from epoch 1", resume_path, loaded_epoch)
        save_checkpoint(
            {
                "epoch": loaded_epoch if resume_optimizer else 0,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "val_metrics": checkpoint.get("val_metrics", {}),
                "best_miou": best_miou,
            },
            output_dir / "best_model.pt",
        )

    logger.info(
        "Trainable params: %.2fM",
        format_num_params(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    )

    distill_start_epoch = int(config["train"].get("distill_start_epoch", 0))
    ema_decay = float(config["train"].get("ema_decay", 0.99 if use_distill else 0.0))
    grad_clip_norm = float(config["train"].get("grad_clip_norm", 0.0))
    early_stopping_patience = int(config["train"].get("early_stopping_patience", 0))
    early_stopping_min_delta = float(config["train"].get("early_stopping_min_delta", 0.0))
    stale_epochs = 0

    for epoch in range(start_epoch, config["train"]["epochs"] + 1):
        distill_weight = 1.0 if use_distill and epoch > distill_start_epoch else 0.0

        train_metrics = train_one_epoch(
            model,
            teacher_model,
            train_loader,
            criterion,
            optimizer,
            device,
            config["dataset"]["num_classes"],
            distill_weight=distill_weight,
            grad_clip_norm=grad_clip_norm,
            ema_decay=ema_decay,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            config["dataset"]["num_classes"],
        )

        logger.info(
            (
                "Epoch %03d | train_loss=%.4f ce=%.4f dice=%.4f feat=%.4f pred=%.4f "
                "train_miou=%.4f | val_loss=%.4f val_miou=%.4f distill_weight=%.2f"
            ),
            epoch,
            train_metrics["total"],
            train_metrics.get("ce", 0.0),
            train_metrics.get("dice", 0.0),
            train_metrics.get("feat", 0.0),
            train_metrics.get("pred", 0.0),
            train_metrics["miou"],
            val_metrics["total"],
            val_metrics["miou"],
            distill_weight,
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "val_metrics": val_metrics,
            "best_miou": best_miou,
        }
        save_checkpoint(checkpoint, output_dir / "last_model.pt")

        if val_metrics["miou"] > best_miou + early_stopping_min_delta:
            best_miou = val_metrics["miou"]
            stale_epochs = 0
            checkpoint["best_miou"] = best_miou
            save_checkpoint(checkpoint, output_dir / "best_model.pt")
        else:
            stale_epochs += 1
            if early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
                logger.info("Early stopping after %d stale epochs.", stale_epochs)
                break

    logger.info("Best validation mIoU: %.4f", best_miou)


if __name__ == "__main__":
    main()
