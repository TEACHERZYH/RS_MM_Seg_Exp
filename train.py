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
    )
    optimizer = build_optimizer(model, config)

    best_miou = -1.0
    logger.info(
        "Trainable params: %.2fM",
        format_num_params(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    )

    for epoch in range(1, config["train"]["epochs"] + 1):
        teacher_model.load_state_dict(model.state_dict())

        train_metrics = train_one_epoch(
            model,
            teacher_model,
            train_loader,
            criterion,
            optimizer,
            device,
            config["dataset"]["num_classes"],
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            config["dataset"]["num_classes"],
        )

        logger.info(
            "Epoch %03d | train_loss=%.4f train_miou=%.4f | val_loss=%.4f val_miou=%.4f",
            epoch,
            train_metrics["total"],
            train_metrics["miou"],
            val_metrics["total"],
            val_metrics["miou"],
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "val_metrics": val_metrics,
        }
        save_checkpoint(checkpoint, output_dir / "last_model.pt")

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            save_checkpoint(checkpoint, output_dir / "best_model.pt")

    logger.info("Best validation mIoU: %.4f", best_miou)


if __name__ == "__main__":
    main()
