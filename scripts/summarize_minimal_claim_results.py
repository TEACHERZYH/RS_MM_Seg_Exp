#!/usr/bin/env python3
"""Summarize the current 24-row QALF evaluation manifest.

This schema is deliberately separate from the historical rebuild aggregator.
It consumes the CSV emitted by ``run_eval_protocol.py`` and keeps the
configuration, checkpoint, result directory, evaluation split, and split hash
bound together for every row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "configs/minimal_claim_m2_r2_20261102_fair/main_evaluation_manifest.csv"
DEFAULT_OUTPUT = "outputs/qalf_repro_fair_20261102/statistics"
REQUIRED_MANIFEST_COLUMNS = {
    "eval_id",
    "scope",
    "dataset",
    "method",
    "regime",
    "seed_id",
    "config",
    "config_sha256",
    "checkpoint",
    "scenarios",
    "eval_split",
    "split_sha256",
    "result_dir",
}
REQUIRED_RESULT_COLUMNS = {"scenario", "miou_6class", "miou_6class_std", "oa", "oa_std"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def result_path(row: dict[str, str]) -> Path:
    result_dir = Path(row["result_dir"])
    if not result_dir.is_absolute():
        result_dir = ROOT / result_dir
    return result_dir / "eval_protocol_summary.csv"


def collect(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output_rows: list[dict[str, object]] = []
    completeness: list[dict[str, object]] = []
    for row in rows:
        expected = [item for item in row["scenarios"].split("|") if item]
        path = result_path(row)
        base = {
            "eval_id": row["eval_id"],
            "scope": row["scope"],
            "dataset": row["dataset"],
            "method": row["method"],
            "regime": row["regime"],
            "seed_id": row["seed_id"],
            "eval_split": row["eval_split"],
            "split_sha256": row["split_sha256"],
            "result_path": str(path),
        }
        if not path.is_file():
            completeness.append({**base, "status": "missing", "scenario_count": 0})
            continue
        result_rows = read_csv(path)
        fields = set(result_rows[0]) if result_rows else set()
        missing_fields = sorted(REQUIRED_RESULT_COLUMNS - fields)
        observed = [item.get("scenario", "") for item in result_rows]
        status = "complete"
        reason = ""
        if missing_fields:
            status = "invalid_schema"
            reason = f"missing result columns: {','.join(missing_fields)}"
        elif set(observed) != set(expected) or len(observed) != len(expected):
            status = "incomplete_or_duplicate_scenarios"
            reason = f"expected={expected} observed={observed}"
        completeness.append({
            **base,
            "status": status,
            "scenario_count": len(observed),
            "reason": reason,
        })
        if status != "complete":
            continue
        for item in result_rows:
            output_rows.append({
                **base,
                "scenario": item["scenario"],
                "miou_6class": item["miou_6class"],
                "miou_6class_std": item["miou_6class_std"],
                "oa": item["oa"],
                "oa_std": item["oa_std"],
            })
    return output_rows, completeness


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest_path = (ROOT / args.manifest).resolve()
    rows = read_csv(manifest_path)
    if not rows:
        raise RuntimeError(f"Manifest has no rows: {manifest_path}")
    missing_columns = sorted(REQUIRED_MANIFEST_COLUMNS - set(rows[0]))
    if missing_columns:
        raise RuntimeError(f"Manifest schema is incomplete: {missing_columns}")
    output_rows, completeness = collect(rows)
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "minimal_claim_evaluation_rows.csv", output_rows)
    write_csv(output_dir / "minimal_claim_evaluation_completeness.csv", completeness)
    incomplete = [item for item in completeness if item["status"] != "complete"]
    payload = {
        "schema": "qalf-minimal-claim-aggregation-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_rows": len(rows),
        "complete_rows": len(rows) - len(incomplete),
        "incomplete_rows": len(incomplete),
        "result_rows": len(output_rows),
        "status": "pass" if not incomplete else "incomplete",
    }
    (output_dir / "minimal_claim_evaluation_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(f"Evaluation outputs are incomplete for {len(incomplete)} manifest row(s)")


if __name__ == "__main__":
    main()
