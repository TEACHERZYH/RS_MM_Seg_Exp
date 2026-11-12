#!/usr/bin/env python3
"""Validate the fair matrix and export commands without running experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs/minimal_claim_m2_r2_20261102_fair"
CORE_SCENARIOS = {"full", "missing_aux", "degraded", "missing_aux_and_degraded"}
TRIAL_POLICY = "full=1|missing_aux=1|degraded=3|combined=3"


def resolved(path: str) -> Path:
    return (ROOT / path).resolve()


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_plan(matrix: Path) -> dict:
    stages = rows(matrix / "training_stage_manifest.csv")
    known = {}
    training_outputs = set()
    commands = []
    lock = json.loads((matrix / "split_lock.json").read_text(encoding="utf-8"))
    for row in stages:
        path = ROOT / row["config"]
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest().upper() != row["config_sha256"].upper():
            raise ValueError(f"Config hash mismatch: {path}")
        if row["stage_id"] in known or (row["resume_stage_id"] and row["resume_stage_id"] not in known):
            raise ValueError(f"Duplicate or unordered training dependency: {row['stage_id']}")
        for field, value in (("epochs", config["train"]["epochs"]),
                             ("fusion_mode", config["model"]["fusion_mode"]),
                             ("stage_id", config["experiment"]["stage_id"]),
                             ("dataset", config["dataset"]["name"]),
                             ("method", config["experiment"]["route"]),
                             ("regime", config["experiment"]["regime"]),
                             ("seed_id", config["experiment"]["seed"]),
                             ("evidence_version", config["experiment"]["evidence_version"]),
                             ("output_dir", config["experiment"]["output_dir"]),
                             ("resume", config["train"].get("resume", ""))):
            if str(row[field]) != str(value):
                raise ValueError(f"Stage/config mismatch: {row['stage_id']}/{field}")
        output = resolved(row["output_dir"])
        if output in training_outputs:
            raise ValueError(f"Duplicate training output directory: {row['stage_id']}")
        if resolved(row["checkpoint_for_eval"]) != output / "last_model.pt":
            raise ValueError(f"Training checkpoint endpoint mismatch: {row['stage_id']}")
        if row["resume_stage_id"]:
            parent = known[row["resume_stage_id"]]
            if not row["resume"] or resolved(row["resume"]) != resolved(parent["checkpoint_for_eval"]):
                raise ValueError(f"Parent checkpoint mismatch: {row['stage_id']}")
            for field in ("dataset", "seed_id"):
                if row[field] != parent[field]:
                    raise ValueError(f"Parent identity mismatch: {row['stage_id']}/{field}")
        elif row["resume"]:
            raise ValueError(f"Resume path has no parent stage: {row['stage_id']}")
        training_outputs.add(output)
        known[row["stage_id"]] = row
        commands.append(dict(kind="train", id=row["stage_id"],
                             requires_stage=row["resume_stage_id"] or None,
                             argv=[sys.executable, "train.py", "--config", row["config"]]))
    expected_counts = {"main": 24, "control": 4, "holdout": 24}
    result_directories = set()
    for scope in ("main", "control", "holdout"):
        evaluations = rows(matrix / f"{scope}_evaluation_manifest.csv")
        if len(evaluations) != expected_counts[scope]:
            raise ValueError(f"Unexpected {scope} evaluation count")
        evaluation_ids = set()
        evaluation_runs = set()
        for row in evaluations:
            if row["scope"] != scope:
                raise ValueError(f"Evaluation scope mismatch: {scope}/{row['eval_id']}")
            if row["eval_id"] in evaluation_ids:
                raise ValueError(f"Duplicate evaluation ID: {scope}/{row['eval_id']}")
            evaluation_ids.add(row["eval_id"])
            source = known.get(row["source_stage_id"])
            if source is None:
                raise ValueError(f"Unknown evaluation source stage: {row['source_stage_id']}")
            for field in ("dataset", "method", "regime", "seed_id", "evidence_version"):
                if row[field] != source[field]:
                    raise ValueError(f"Evaluation stage identity mismatch: {scope}/{row['eval_id']}/{field}")
            source_phase = "main" if scope == "holdout" else scope
            if source["phase"] != source_phase:
                raise ValueError(f"Evaluation source phase mismatch: {scope}/{row['eval_id']}")
            if resolved(row["config"]) != resolved(source["config"]):
                raise ValueError(f"Evaluation source config mismatch: {scope}/{row['eval_id']}")
            if resolved(row["checkpoint"]) != resolved(source["checkpoint_for_eval"]):
                raise ValueError(f"Evaluation checkpoint mismatch: {scope}/{row['eval_id']}")
            expected_split = "holdout_split" if scope == "holdout" else "val_split"
            if row["eval_split"] != expected_split:
                raise ValueError(f"Evaluation split policy mismatch: {scope}/{row['eval_id']}")
            scenario_names = row["scenarios"].split("|")
            if set(scenario_names) != CORE_SCENARIOS or len(scenario_names) != 4:
                raise ValueError(f"Evaluation scenario policy mismatch: {scope}/{row['eval_id']}")
            if row.get("trial_policy") != TRIAL_POLICY or row.get("expected_summaries") != "4":
                raise ValueError(f"Evaluation trial policy mismatch: {scope}/{row['eval_id']}")
            run_key = (row["source_stage_id"], row["eval_split"])
            result_dir = resolved(row["result_dir"])
            if run_key in evaluation_runs or result_dir in result_directories:
                raise ValueError(f"Duplicate evaluation run or result directory: {scope}/{row['eval_id']}")
            evaluation_runs.add(run_key)
            result_directories.add(result_dir)
            config_path = ROOT / row["config"]
            if hashlib.sha256(config_path.read_bytes()).hexdigest().upper() != row["config_sha256"].upper():
                raise ValueError(f"Evaluation config hash mismatch: {row['eval_id']}")
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if row["eval_split"] not in config["dataset"]:
                raise ValueError(f"Unknown split key: {row['eval_split']}")
            split = "internal_holdout" if scope == "holdout" else "validation"
            expected = lock["datasets"][row["dataset"]]["split_sha256"][split]
            if expected.upper() != row["split_sha256"].upper():
                raise ValueError(f"Split hash mismatch: {scope}/{row['eval_id']}")
            commands.append(dict(kind=scope, id=row["eval_id"],
                                 requires_stage=row["source_stage_id"],
                                 requires_separate_holdout_authorization=scope == "holdout",
                                 argv=[sys.executable, "scripts/run_eval_protocol.py", "--config", row["config"],
                                       "--checkpoint", row["checkpoint"], "--split", row["eval_split"],
                                       "--expected-split-sha256", row["split_sha256"],
                                       "--require-training-complete",
                                       "--trials", "3", "--single-clean-trial", "--output-dir", row["result_dir"]]))
    return dict(status="validated_commands_only", executed=False, working_directory=str(ROOT),
                training_stages=len(stages), commands=commands)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(args.matrix.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(dict(status=plan["status"], commands=len(plan["commands"]), executed=False)))


if __name__ == "__main__":
    main()
