from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from src.datasets.isprs_dataset import ISPRSMultimodalDataset
from src.engine import evaluate
from src.losses import SegmentationCriterion
from src.models.qalf_net import QALFNet
from src.utils import load_checkpoint, load_config, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test_split")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config["train"]["device"])

    ds = config["dataset"]
    split_file = str(Path(ds["split_dir"]) / ds[args.split])
    dataset = ISPRSMultimodalDataset(
        root_dir=ds["root_dir"],
        split_file=split_file,
        image_dir=ds["image_dir"],
        aux_dir=ds["aux_dir"],
        mask_dir=ds["mask_dir"],
        image_suffix=ds["image_suffix"],
        aux_suffix=ds["aux_suffix"],
        mask_suffix=ds["mask_suffix"],
        input_size=ds["input_size"],
        missing_prob=0.0,
        degradation_prob=0.0,
        normalize_aux=ds["normalize_aux"],
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config["eval"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = QALFNet(**config["model"]).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device.type)
    model.load_state_dict(ckpt["model"])

    criterion = SegmentationCriterion(
        num_classes=config["dataset"]["num_classes"],
        ce_weight=config["loss"]["ce_weight"],
        dice_weight=config["loss"]["dice_weight"],
        feat_weight=config["loss"]["feat_weight"],
        pred_weight=config["loss"]["pred_weight"],
    )

    metrics = evaluate(model, loader, criterion, device, config["dataset"]["num_classes"])
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
