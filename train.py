from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from src.datasets.isprs_dataset import ISPRSMultimodalDataset
from src.engine import build_optimizer, evaluate, train_one_epoch
from src.functional_entropy import BlockRandomProbeScheduler, DimensionNormalizedFunctionalEntropy
from src.losses import SegmentationCriterion
from src.models.qalf_net import QALFNet
from src.utils import (
    ensure_dir,
    format_num_params,
    load_config,
    resolve_device,
    save_checkpoint,
    seed_data_worker,
    set_seed,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_dataset(
    config: dict,
    split_name: str,
    training: bool,
    augmentation_seed: int = 0,
    corruption_seed: int = 0,
) -> ISPRSMultimodalDataset:
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
        degradation_noise_std=ds.get("degradation_noise_std", 12.0),
        degradation_blur_kernel=ds.get("degradation_blur_kernel", 5),
        degradation_lowres_scale=ds.get("degradation_lowres_scale", 4),
        degradation_mask_min_fraction=ds.get("degradation_mask_min_fraction", 0.125),
        degradation_mask_max_fraction=ds.get("degradation_mask_max_fraction", 1.0 / 3.0),
        degradation_mask_position=ds.get("degradation_mask_position", "legacy_half"),
        normalize_aux=ds["normalize_aux"],
        training=training,
        augment=ds.get("augment", False),
        hflip_prob=ds.get("hflip_prob", 0.0),
        vflip_prob=ds.get("vflip_prob", 0.0),
        rotate90_prob=ds.get("rotate90_prob", 0.0),
        color_jitter=ds.get("color_jitter", 0.0),
        missing_target=ds.get("missing_target", "aux"),
        augmentation_seed=augmentation_seed,
        corruption_seed=corruption_seed,
    )


def main() -> None:
    wall_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_id = int(config["experiment"]["seed"])
    rng = config["experiment"].get("rng", {})
    init_seed = int(rng.get("init_seed", seed_id))
    sampler_seed = int(rng.get("sampler_seed", seed_id + 1))
    worker_seed = int(rng.get("worker_seed", seed_id + 2))
    augmentation_seed = int(rng.get("augmentation_seed", seed_id + 3))
    corruption_seed = int(rng.get("corruption_seed", seed_id + 4))
    deterministic = bool(config["train"].get("deterministic_algorithms", False))
    set_seed(init_seed, deterministic=deterministic)
    device = resolve_device(config["train"]["device"])
    output_dir = ensure_dir(config["experiment"]["output_dir"])
    completion_path = output_dir / "training_complete.json"
    completion_path.unlink(missing_ok=True)
    log_path = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
    )
    logger = logging.getLogger("train")

    train_dataset = build_dataset(
        config,
        "train_split",
        training=True,
        augmentation_seed=augmentation_seed,
        corruption_seed=corruption_seed,
    )
    val_dataset = build_dataset(config, "val_split", training=False)

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(sampler_seed)
    worker_generator = torch.Generator()
    worker_generator.manual_seed(worker_seed)
    train_sampler = RandomSampler(train_dataset, generator=sampler_generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        sampler=train_sampler,
        num_workers=config["train"]["num_workers"],
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_data_worker,
        generator=worker_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config["eval"]["num_workers"],
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_data_worker,
    )

    logger.info(
        "RNG seed_id=%d init=%d sampler=%d worker=%d augmentation=%d corruption=%d deterministic=%s",
        seed_id,
        init_seed,
        sampler_seed,
        worker_seed,
        augmentation_seed,
        corruption_seed,
        deterministic,
    )

    model = QALFNet(**config["model"]).to(device)

    criterion = SegmentationCriterion(
        num_classes=config["dataset"]["num_classes"],
        ce_weight=config["loss"]["ce_weight"],
        dice_weight=config["loss"]["dice_weight"],
        feat_weight=config["loss"]["feat_weight"],
        pred_weight=config["loss"]["pred_weight"],
    ).to(device)
    optimizer = build_optimizer(model, config)

    entropy_cfg = config["loss"].get("functional_entropy", {})
    functional_entropy = None
    functional_entropy_probe_scheduler = None
    entropy_warmup_epochs = int(entropy_cfg.get("warmup_epochs", 0))
    entropy_probe_size = int(entropy_cfg.get("probe_size", 64))
    entropy_probe_batch_size = int(entropy_cfg.get("probe_batch_size", 1))
    entropy_probe_interval = int(entropy_cfg.get("probe_interval", 4))
    entropy_active_loss_scale = int(entropy_cfg.get("active_loss_scale", entropy_probe_interval))
    if bool(entropy_cfg.get("enabled", False)):
        if config["model"].get("fusion_mode") != "dynamic_gated":
            raise ValueError("DN-MFE upgrade configs require fusion_mode='dynamic_gated'")
        if not bool(config["model"].get("second_order_compatible_activations", False)):
            raise ValueError("DN-MFE requires second_order_compatible_activations=true")
        functional_entropy = DimensionNormalizedFunctionalEntropy(
            prediction_weight=float(entropy_cfg.get("prediction_weight", 0.30)),
            feature_weight=float(entropy_cfg.get("feature_weight", 0.02)),
            feature_scales=entropy_cfg.get("feature_scales", [0, 1, 2, 3]),
            eps=float(entropy_cfg.get("eps", 1e-6)),
            stability_offset=float(entropy_cfg.get("stability_offset", 0.05)),
        ).to(device)
        probe_seed = int(entropy_cfg.get("probe_seed", seed_id * 100 + 66))
        if probe_seed != seed_id * 100 + 66:
            raise ValueError("DN-MFE probe_seed must follow the frozen seed_id*100+66 formula")
        functional_entropy_probe_scheduler = BlockRandomProbeScheduler(
            seed=probe_seed,
            interval=entropy_probe_interval,
        )

    use_distill = (config["loss"].get("feat_weight", 0.0) > 0.0) or (config["loss"].get("pred_weight", 0.0) > 0.0)
    teacher_model = None
    if use_distill:
        teacher_model = QALFNet(**config["model"]).to(device)
        teacher_model.load_state_dict(model.state_dict())
        for param in teacher_model.parameters():
            param.requires_grad = False

    best_miou = -1.0
    start_epoch = 1
    training_global_step = 0
    checkpoint = None
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
            training_global_step = int(checkpoint.get("training_global_step", loaded_epoch * len(train_loader)))
            if functional_entropy_probe_scheduler is not None:
                scheduler_state = checkpoint.get("functional_entropy_probe_scheduler")
                if scheduler_state is None:
                    raise RuntimeError("DN-MFE optimizer resume requires checkpointed probe RNG state")
                functional_entropy_probe_scheduler.load_state_dict(scheduler_state)
        if teacher_model is not None:
            teacher_model.load_state_dict(model.state_dict())
        if resume_optimizer:
            logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)
        else:
            best_miou = -1.0
            logger.info(
                "Loaded weights from %s at source epoch %d; fine-tuning from epoch 1 with reset best state",
                resume_path,
                loaded_epoch,
            )

    logger.info(
        "Trainable params: %.2fM",
        format_num_params(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    )

    distill_start_epoch = int(config["train"].get("distill_start_epoch", 0))
    ema_decay = float(config["train"].get("ema_decay", 0.99 if use_distill else 0.0))
    grad_clip_norm = float(config["train"].get("grad_clip_norm", 0.0))
    if functional_entropy is not None and (
        entropy_probe_size <= 0 or entropy_probe_batch_size <= 0 or entropy_probe_interval <= 0
    ):
        raise ValueError("DN-MFE probe_size, probe_batch_size, and probe_interval must be positive")
    if functional_entropy is not None and entropy_active_loss_scale != entropy_probe_interval:
        raise ValueError("DN-MFE active_loss_scale must equal probe_interval for unbiased sparse sampling")
    early_stopping_patience = int(config["train"].get("early_stopping_patience", 0))
    early_stopping_min_delta = float(config["train"].get("early_stopping_min_delta", 0.0))
    recovery_interval = int(config["train"].get("recovery_interval_epochs", 0))
    if functional_entropy is not None and early_stopping_patience != 0:
        raise ValueError("DN-MFE claim-bearing runs require early_stopping_patience=0")
    stale_epochs = 0
    completed_epoch = start_epoch - 1
    termination_reason = "max_epochs"

    for epoch in range(start_epoch, config["train"]["epochs"] + 1):
        completed_epoch = epoch
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
            functional_entropy=functional_entropy,
            epoch=epoch,
            functional_entropy_warmup_epochs=entropy_warmup_epochs,
            functional_entropy_probe_size=entropy_probe_size,
            functional_entropy_probe_batch_size=entropy_probe_batch_size,
            functional_entropy_probe_interval=entropy_probe_interval,
            functional_entropy_probe_scheduler=functional_entropy_probe_scheduler,
            global_step_offset=training_global_step,
        )
        training_global_step += len(train_loader)
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
                "fe=%.4f fe_pred=%.4f fe_feat=%.4f "
                "train_miou=%.4f | val_loss=%.4f val_miou=%.4f distill_weight=%.2f"
            ),
            epoch,
            train_metrics["total"],
            train_metrics.get("ce", 0.0),
            train_metrics.get("dice", 0.0),
            train_metrics.get("feat", 0.0),
            train_metrics.get("pred", 0.0),
            train_metrics.get("fe_total", 0.0),
            train_metrics.get("fe_pred", 0.0),
            train_metrics.get("fe_feat", 0.0),
            train_metrics["miou"],
            val_metrics["total"],
            val_metrics["miou"],
            distill_weight,
        )

        improved = val_metrics["miou"] > best_miou + early_stopping_min_delta
        if improved:
            best_miou = val_metrics["miou"]
            stale_epochs = 0
        else:
            stale_epochs += 1

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "val_metrics": val_metrics,
            "best_miou": best_miou,
            "training_global_step": training_global_step,
        }
        if functional_entropy_probe_scheduler is not None:
            checkpoint["functional_entropy_probe_scheduler"] = functional_entropy_probe_scheduler.state_dict()
            trace_path = output_dir / "dn_mfe_probe_trace.json"
            trace_payload = {
                "schema": "qalf-dn-mfe-probe-trace-v1",
                "probe_seed": functional_entropy_probe_scheduler.seed,
                "probe_interval": functional_entropy_probe_scheduler.interval,
                "training_global_step": training_global_step,
                "trace": functional_entropy_probe_scheduler.trace,
            }
            trace_temporary = trace_path.with_suffix(".json.tmp")
            trace_temporary.write_text(json.dumps(trace_payload, indent=2) + "\n", encoding="utf-8")
            trace_temporary.replace(trace_path)
        save_checkpoint(checkpoint, output_dir / "last_model.pt")
        if recovery_interval > 0 and epoch % recovery_interval == 0:
            save_checkpoint(checkpoint, output_dir / "recovery_model.pt")

        if improved:
            save_checkpoint(checkpoint, output_dir / "best_model.pt")
        elif early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
            logger.info("Early stopping after %d stale epochs.", stale_epochs)
            termination_reason = "early_stopping"
            break

    logger.info("Best validation mIoU: %.4f", best_miou)
    requested_epochs = int(config["train"]["epochs"])
    last_model_path = output_dir / "last_model.pt"
    completion = {
        "schema": "qalf-training-completion-v1",
        "status": "pass" if completed_epoch == requested_epochs else "incomplete",
        "termination_reason": termination_reason,
        "completed_epoch": completed_epoch,
        "requested_epochs": requested_epochs,
        "config": args.config,
        "config_sha256": sha256_file(Path(args.config)),
        "last_model_sha256": sha256_file(last_model_path) if last_model_path.exists() else None,
        "training_global_step": training_global_step,
        "dn_mfe_probe_trace_sha256": (
            sha256_file(output_dir / "dn_mfe_probe_trace.json")
            if (output_dir / "dn_mfe_probe_trace.json").exists()
            else None
        ),
        "wall_seconds": time.perf_counter() - wall_start,
    }
    temporary = completion_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    temporary.replace(completion_path)


if __name__ == "__main__":
    main()
