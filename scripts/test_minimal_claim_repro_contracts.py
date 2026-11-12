#!/usr/bin/env python3
"""Contract checks for the corrected public QALF reproduction matrix."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import summarize_minimal_claim_results as agg
from test_seed_statistics import make_evidence_fixture


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    matrix = repo / "configs/minimal_claim_m2_r2_20261102_fair"
    main_rows = read_rows(matrix / "main_evaluation_manifest.csv")
    holdout_rows = read_rows(matrix / "holdout_evaluation_manifest.csv")
    if len(main_rows) != 24 or len(holdout_rows) != 24:
        raise AssertionError("Corrected matrix must contain 24 main and 24 holdout rows")
    required = {"eval_split", "split_sha256", "result_dir"}
    if not required.issubset(main_rows[0]):
        raise AssertionError("Main manifest is missing split/result binding columns")
    if any(row["eval_split"] != "val_split" for row in main_rows):
        raise AssertionError("Main rows must bind val_split")
    if any(row["eval_split"] != "holdout_split" for row in holdout_rows):
        raise AssertionError("Holdout rows must bind holdout_split")

    fixed_configs = sorted((matrix / "main").glob("*fixed_late*.yaml"))
    if len(fixed_configs) != 18:
        raise AssertionError(f"Expected 18 fixed-late configs, found {len(fixed_configs)}")
    if any("fusion_mode: availability_masked_average" not in p.read_text(encoding="utf-8") for p in fixed_configs):
        raise AssertionError("All corrected fixed-late configs must use availability masking")
    stage_rows = read_rows(matrix / "training_stage_manifest.csv")
    fixed_stage_rows = [row for row in stage_rows if row["method"] == "fixed_late"]
    if len(fixed_stage_rows) != 18 or any(row["fusion_mode"] != "availability_masked_average" for row in fixed_stage_rows):
        raise AssertionError("Training manifest is not synchronized with corrected fixed-late configs")
    if any("qalf_minimal_m2_20260715" in row["resume"] for row in stage_rows):
        raise AssertionError("Training manifest still contains legacy resume paths")
    epochs = {
        (row["dataset"], row["regime"]): int(row["epochs"])
        for row in stage_rows
        if row["method"] in {"fixed_late", "qalf"}
    }
    if epochs.get(("vaihingen", "base_clean")) != 45 or epochs.get(("potsdam", "base_clean")) != 35:
        raise AssertionError("Dataset-specific base budgets are not bound")
    if epochs.get(("vaihingen", "robust")) != 28 or epochs.get(("potsdam", "robust")) != 13:
        raise AssertionError("Dataset-specific robust budgets are not bound")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        result_dir = root / "result"
        result_dir.mkdir()
        (result_dir / "eval_protocol_summary.csv").write_text(
            "scenario,miou_6class,miou_6class_std,oa,oa_std\n"
            "full,0.700000,0.010000,0.900000,0.005000\n"
            "missing_aux,0.600000,0.020000,0.800000,0.006000\n",
            encoding="utf-8",
        )
        old_root = agg.ROOT
        agg.ROOT = root
        try:
            identity, _ = make_evidence_fixture(result_dir)
            rows = [
                {
                    "eval_id": "T-001",
                    "scope": "main",
                    "dataset": "vaihingen",
                    "method": "qalf",
                    "regime": "robust",
                    "seed_id": "101",
                    "scenarios": "full|missing_aux",
                    "eval_split": "val_split",
                    "split_sha256": "ABC",
                    "result_dir": "result",
                    **identity,
                }
            ]
            output, complete = agg.collect(rows)
        finally:
            agg.ROOT = old_root
        if len(output) != 2 or complete[0]["status"] != "complete":
            raise AssertionError("Synthetic current-schema aggregation failed")
    print('{"status":"pass","main_rows":24,"holdout_rows":24,"fixed_configs":18}')


if __name__ == "__main__":
    main()
