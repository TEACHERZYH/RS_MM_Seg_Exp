#!/usr/bin/env python3
"""Audit and evaluate the 12 checkpoint-reusing availability-masked routes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_eval_protocol import Scenario, build_dataset
from src.models.qalf_net import QALFNet
from src.utils import load_checkpoint, load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="configs/rebuild_20260714/derived_evaluation_manifest.csv"
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sync-target", default="")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def audit_rows(rows: list[dict[str, str]]) -> None:
    parent_rows = read_rows(ROOT / "configs/rebuild_20260714/config_manifest.csv")
    if len(parent_rows) != 85:
        raise RuntimeError("Derived routes require the frozen 85-row parent manifest")
    parent = {row["stage_id"]: row for row in parent_rows}
    expected_sources = {
        f"N4-{index:03d}" for index in (10, 11, 13, 14, 16, 17, 28, 29, 31, 32, 34, 35)
    }
    if (
        {row["source_stage_id"] for row in rows} != expected_sources
        or len({row["route_id"] for row in rows}) != 12
        or len({row["output_dir"] for row in rows}) != 12
    ):
        raise RuntimeError("Derived route IDs, outputs, or frozen source stages drifted")
    for row in rows:
        source = parent.get(row["source_stage_id"])
        config = ROOT / row["config"]
        if source is None:
            raise RuntimeError(f"Derived route has no parent stage: {row['route_id']}")
        if (
            source["phase"] != "n4"
            or source["method"] != "fixed_late"
            or source["checkpoint_for_eval"] != row["source_checkpoint"]
            or source["config_sha256"] != row["source_config_sha256"]
            or any(source[field] != row[field] for field in ("dataset", "regime", "seed_id"))
            or row["source_fusion_mode"] != "fixed_average"
            or row["target_fusion_mode"] != "availability_masked_average"
            or not config.is_file()
            or sha256_file(config) != row["config_sha256"]
        ):
            raise RuntimeError(f"Derived route provenance drift: {row['route_id']}")
        source_config = load_config(str(ROOT / source["config"]))
        derived_config = load_config(str(config))
        source_model = dict(source_config["model"])
        derived_model = dict(derived_config["model"])
        source_mode = source_model.pop("fusion_mode")
        derived_mode = derived_model.pop("fusion_mode")
        if (
            source_mode != "fixed_average"
            or derived_mode != "availability_masked_average"
            or source_model != derived_model
        ):
            raise RuntimeError(f"Derived route changed more than the masked fusion rule: {row['route_id']}")


def equivalence_check(row: dict[str, str]) -> float:
    derived_config = load_config(str(ROOT / row["config"]))
    source_config_path = None
    parent_rows = read_rows(ROOT / "configs/rebuild_20260714/config_manifest.csv")
    for source in parent_rows:
        if source["stage_id"] == row["source_stage_id"]:
            source_config_path = ROOT / source["config"]
            break
    if source_config_path is None:
        raise RuntimeError(f"Derived route has no parent stage: {row['route_id']}")
    source_config = load_config(str(source_config_path))
    if source_config["model"]["fusion_mode"] != "fixed_average":
        raise RuntimeError(f"Derived source route is not Fixed Late: {row['route_id']}")
    device = resolve_device(derived_config["train"]["device"])
    source_model = QALFNet(**source_config["model"]).to(device).eval()
    derived_model = QALFNet(**derived_config["model"]).to(device).eval()
    checkpoint = load_checkpoint(str(ROOT / row["source_checkpoint"]), map_location=device.type)
    source_model.load_state_dict(checkpoint["model"], strict=True)
    derived_model.load_state_dict(checkpoint["model"], strict=True)
    dataset = build_dataset(
        derived_config,
        row["eval_split"],
        Scenario("full", 0.0, 0.0, False, False),
        0,
        None,
    )
    batch = next(iter(DataLoader(dataset, batch_size=min(2, len(dataset)), shuffle=False, num_workers=0)))
    image = batch["image"].to(device)
    aux = batch["aux"].to(device)
    available = torch.ones(image.shape[0], device=device)
    with torch.no_grad():
        source_logits = source_model(image, aux, available, available)["logits"]
        derived_logits = derived_model(image, aux, available, available)["logits"]
    return float((source_logits - derived_logits).abs().max().item())


def main() -> None:
    args = parse_args()
    manifest = ROOT / args.manifest
    rows = read_rows(manifest)
    if len(rows) != 12:
        raise RuntimeError(f"Derived evaluation manifest must contain 12 rows, found {len(rows)}")
    audit_rows(rows)
    print(json.dumps({"status": "audit_pass", "routes": len(rows), "execute": args.execute}))
    if not args.execute:
        return
    if not args.sync_target:
        raise RuntimeError("Formal derived-route evaluation requires an off-host sync target")

    equivalence_rows = []
    for row in rows:
        checkpoint = ROOT / row["source_checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        error = equivalence_check(row)
        tolerance = float(row["equivalence_tolerance"])
        if error > tolerance:
            raise RuntimeError(f"Derived all-available equivalence failed: {row['route_id']}={error}")
        corruption = (
            ROOT
            / f"outputs/rebuild_20260714/protocol/corruption_manifest_{row['dataset']}_n4_{row['eval_split']}_v1.csv"
        )
        if not corruption.is_file():
            raise FileNotFoundError(corruption)
        output = ROOT / row["output_dir"] / f"eval_protocol_{row['eval_split']}_manifest_v1"
        command = [
            args.python,
            "scripts/run_eval_protocol.py",
            "--config",
            row["config"],
            "--checkpoint",
            row["source_checkpoint"],
            "--split",
            row["eval_split"],
            "--trials",
            "3",
            "--include-missing-primary",
            "--corruption-manifest",
            corruption.relative_to(ROOT).as_posix(),
            "--output-dir",
            output.relative_to(ROOT).as_posix(),
        ]
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            raise RuntimeError(f"Derived evaluation failed: {row['route_id']}")
        equivalence_rows.append(
            {
                "route_id": row["route_id"],
                "source_stage_id": row["source_stage_id"],
                "source_checkpoint_sha256": sha256_file(checkpoint),
                "all_available_logits_max_abs": error,
                "tolerance": tolerance,
                "status": "pass",
            }
        )
    report = {
        "schema": "qalf-derived-evaluation-gate-v1",
        "status": "pass",
        "manifest_sha256": sha256_file(manifest),
        "routes": equivalence_rows,
    }
    output = ROOT / "outputs/rebuild_20260714/protocol/derived_evaluation_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for source in ("outputs/rebuild_20260714/derived", "outputs/rebuild_20260714/protocol"):
        subprocess.run(
            [
                args.python,
                "scripts/sync_rebuild_outputs.py",
                "--source",
                source,
                "--target",
                f"{args.sync_target.rstrip('/')}/{source}",
            ],
            cwd=ROOT,
            check=True,
        )
    print(json.dumps({"status": "pass", "routes": len(rows)}))


if __name__ == "__main__":
    main()
