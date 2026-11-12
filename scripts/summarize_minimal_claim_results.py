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
import math
import statistics
from collections import defaultdict
from pathlib import Path
from reproduction_evidence import validate_completion
import yaml


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
    "trial_policy",
}
REQUIRED_RESULT_COLUMNS = {"scenario", "miou_6class", "miou_6class_std", "oa", "oa_std"}
TRIAL_POLICY = "full=1|missing_aux=1|degraded=3|combined=3"


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


def validate_manifest_rows(rows: list[dict[str, str]], manifest_path: Path) -> None:
    """Check manuscript identities against configurations and the frozen split lock."""
    lock_path = manifest_path.parent / "split_lock.json"
    lock = None
    identities = set()
    for row in rows:
        identity = tuple(row[field] for field in ("scope", "dataset", "method", "regime", "seed_id", "eval_split"))
        if identity in identities:
            raise ValueError(f"Duplicate manifest run identity: {identity}")
        identities.add(identity)
        if row["scope"] not in {"main", "control", "holdout"}:
            continue
        if lock is None:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
        key = "internal_holdout" if row["scope"] == "holdout" else "validation"
        if row["split_sha256"].upper() != lock["datasets"][row["dataset"]]["split_sha256"][key].upper():
            raise ValueError(f"Manifest split differs from split_lock: {identity}")
        config_path = ROOT / row["config"]
        if sha256_file(config_path) != row["config_sha256"].upper():
            raise ValueError(f"Manifest config hash mismatch: {identity}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        expected = dict(dataset=config["dataset"]["name"], method=config["experiment"]["route"],
                        regime=config["experiment"]["regime"], seed_id=config["experiment"]["seed"],
                        source_stage_id=config["experiment"]["stage_id"])
        if any(str(row.get(field, "")) != str(value) for field, value in expected.items()):
            raise ValueError(f"Manifest/config identity mismatch: {identity}")


def provenance_error(row: dict[str, str], path: Path) -> str:
    """Reject a CSV unless its metadata binds the requested run and artifacts."""
    try:
        metadata = json.loads((path.parent / "eval_protocol_metadata.json").read_text(encoding="utf-8"))
        for field, expected in (("config_sha256", row["config_sha256"]),
                                ("split_sha256", row["split_sha256"]),
                                ("summary_csv_sha256", sha256_file(path))):
            if str(metadata.get(field, "")).upper() != expected.upper():
                return f"{field} mismatch"
        if metadata.get("split") != row["eval_split"]:
            return "eval_split mismatch"
        if row.get("trial_policy") != TRIAL_POLICY:
            return "unsupported or missing trial_policy"
        if metadata.get("trials") != 3 or metadata.get("single_clean_trial") is not True:
            return "corruption trial policy mismatch"
        if metadata.get("corruption_manifest") or metadata.get("corruption_manifest_sha256"):
            return "unexpected external corruption manifest"
        for field in ("config", "checkpoint"):
            requested = (ROOT / row[field]).resolve()
            actual = (ROOT / metadata[field]).resolve()
            if actual != requested:
                return f"{field} path mismatch"
            if sha256_file(requested) != str(metadata.get(f"{field}_sha256", "")).upper():
                return f"{field} file hash mismatch"
        contract = metadata.get("checkpoint_contract", {})
        if contract.get("status") != "pass":
            return "checkpoint contract was not validated"
        completion_path = (ROOT / row["checkpoint"]).parent / "training_complete.json"
        if sha256_file(completion_path) != contract.get("training_complete_sha256"):
            return "training completion receipt hash mismatch"
        validate_completion(completion_path, row["config_sha256"], metadata["checkpoint_sha256"], contract.get("epoch"))
        for filename, hash_key in (("eval_protocol_results.json", "results_json_sha256"),
                                   ("eval_protocol_classwise.csv", "classwise_csv_sha256"),
                                   ("eval_protocol_confusion.json", "confusion_json_sha256")):
            if sha256_file(path.parent / filename) != str(metadata.get(hash_key, "")).upper():
                return f"{filename} hash mismatch"
        results = json.loads((path.parent / "eval_protocol_results.json").read_text(encoding="utf-8"))
        expected_scenarios = row["scenarios"].split("|")
        if set(results) != set(expected_scenarios) or set(metadata.get("scenarios", [])) != set(expected_scenarios):
            return "result scenarios mismatch"
        counts = dict(full=1, missing_aux=1, degraded=3, missing_aux_and_degraded=3)
        for scenario, result in results.items():
            trials = result["trials"]
            if sorted(t["trial"] for t in trials) != list(range(counts[scenario])):
                return f"actual trial identity/count mismatch: {scenario}"
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return f"missing or invalid provenance: {exc}"
    return ""


def seed_statistics(rows: list[dict[str, object]]) -> tuple[list[dict], list[dict]]:
    """Separate training-seed uncertainty from within-checkpoint corruption trials."""
    groups = defaultdict(dict)
    split_hashes = {}
    for row in rows:
        split_key = (row["dataset"], row["eval_split"])
        split_hash = str(row["split_sha256"]).upper()
        if split_hashes.setdefault(split_key, split_hash) != split_hash:
            raise ValueError(f"Different split hashes cannot be pooled: {split_key}")
        key = tuple(row[k] for k in ("scope", "eval_split", "dataset", "method", "regime", "scenario"))
        seed = str(row["seed_id"])
        if seed in groups[key]:
            raise ValueError(f"Duplicate seed in result group: {key}/{seed}")
        value = str(row["miou_6class"])
        groups[key][seed] = None if value == "not_estimable" else float(value)
    summaries = []
    for key, values in sorted(groups.items()):
        numeric = [v for v in values.values() if v is not None]
        complete = set(values) == {"101", "202", "303"} and len(numeric) == 3
        summaries.append(dict(zip(("scope", "eval_split", "dataset", "method", "regime", "scenario"), key),
                              n_training_seeds=len(values), status="complete" if complete else "not_estimable",
                              miou_mean=statistics.mean(numeric) if complete else "not_estimable",
                              miou_sample_sd=statistics.stdev(numeric) if complete else "not_estimable"))
    contrasts = []
    for key, candidate in sorted(groups.items()):
        scope, split, dataset, method, regime, scenario = key
        comparisons = []
        if method == "qalf" and regime == "robust":
            comparisons.append(("qalf_minus_fixed_robust", (scope, split, dataset, "fixed_late", "robust", scenario)))
        if regime == "robust":
            comparisons.append(("robust_minus_clean_continuation", (scope, split, dataset, method, "clean_continuation", scenario)))
        for label, control_key in comparisons:
            control = groups.get(control_key, {})
            valid = set(candidate) == set(control) == {"101", "202", "303"}
            valid = valid and all(v is not None for v in [*candidate.values(), *control.values()])
            diffs = [candidate[s] - control[s] for s in ("101", "202", "303")] if valid else []
            mean = statistics.mean(diffs) if valid else None
            half = 4.302652729911275 * statistics.stdev(diffs) / math.sqrt(3) if valid else None
            contrasts.append(dict(scope=scope, eval_split=split, dataset=dataset, method=method,
                                  scenario=scenario, contrast=label,
                                  status="complete" if valid else "not_estimable",
                                  mean_difference=mean if valid else "not_estimable",
                                  ci95_low=mean-half if valid else "not_estimable",
                                  ci95_high=mean+half if valid else "not_estimable",
                                  positive_seeds=sum(v > 0 for v in diffs) if valid else "not_estimable",
                                  n_training_seeds=3 if valid else 0))
    return summaries, contrasts


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
            completeness.append({**base, "status": "missing", "scenario_count": 0, "reason": "output absent"})
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
        if status == "complete":
            for item in result_rows:
                for metric in REQUIRED_RESULT_COLUMNS - {"scenario"}:
                    value = item[metric]
                    if value == "not_estimable" and metric.startswith("miou"):
                        continue
                    try:
                        number = float(value)
                        if not math.isfinite(number) or not 0 <= number <= 1:
                            raise ValueError(value)
                    except (ValueError, TypeError):
                        status, reason = "invalid_metric", f"{metric}={value}"
            if status == "complete":
                reason = provenance_error(row, path)
                if reason:
                    status = "invalid_provenance"
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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
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
    validate_manifest_rows(rows, manifest_path)
    output_rows, completeness = collect(rows)
    summaries, contrasts = seed_statistics(output_rows)
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "minimal_claim_evaluation_rows.csv", output_rows)
    write_csv(output_dir / "minimal_claim_evaluation_completeness.csv", completeness)
    write_csv(output_dir / "training_seed_summary.csv", summaries)
    write_csv(output_dir / "paired_seed_contrasts.csv", contrasts)
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
