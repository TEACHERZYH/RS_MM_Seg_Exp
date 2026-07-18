#!/usr/bin/env python3
"""Validate and summarize all manifest-backed QALF rebuild evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_SCENARIOS = {"full", "missing_aux", "degraded", "missing_aux_and_degraded"}
EXPECTED_SCENARIOS = CORE_SCENARIOS | {"missing_primary"}
METRICS = ("miou_mean", "miou_5class_mean", "oa_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/rebuild_20260714/config_manifest.csv")
    parser.add_argument("--output-dir", default="outputs/rebuild_20260714/statistics")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def evaluation_path(row: dict[str, str]) -> Path:
    return ROOT / row["output_dir"] / f"eval_protocol_{row['eval_split']}_manifest_v1/eval_protocol_summary.csv"


def collect(manifest_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    results = []
    completeness = []
    for row in manifest_rows:
        path = evaluation_path(row)
        if not path.exists():
            completeness.append(
                {"stage_id": row["stage_id"], "phase": row["phase"], "status": "missing", "path": str(path)}
            )
            continue
        eval_rows = read_csv(path)
        scenarios = {item["scenario"] for item in eval_rows}
        scenario_counts = {
            scenario: sum(item["scenario"] == scenario for item in eval_rows) for scenario in scenarios
        }
        status = (
            "complete"
            if scenarios == EXPECTED_SCENARIOS and all(count == 1 for count in scenario_counts.values())
            else "incomplete_or_duplicate_scenarios"
        )
        completeness.append(
            {
                "stage_id": row["stage_id"],
                "phase": row["phase"],
                "status": status,
                "path": str(path),
                "scenario_count": len(scenarios),
                "scenario_counts": json.dumps(scenario_counts, sort_keys=True),
            }
        )
        for item in eval_rows:
            results.append(
                {
                    "stage_id": row["stage_id"],
                    "phase": row["phase"],
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "regime": row["regime"],
                    "seed_id": row["seed_id"],
                    "scenario": item["scenario"],
                    **{metric: item[metric] for metric in METRICS},
                    "source_csv": str(path),
                }
            )
    return results, completeness


def collect_derived(
    manifest_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    results = []
    completeness = []
    for row in manifest_rows:
        path = evaluation_path(row)
        if not path.is_file():
            completeness.append(
                {"stage_id": row["route_id"], "phase": "derived", "status": "missing", "path": str(path)}
            )
            continue
        eval_rows = read_csv(path)
        scenarios = {item["scenario"] for item in eval_rows}
        status = "complete" if scenarios == EXPECTED_SCENARIOS and len(eval_rows) == 5 else "incomplete_or_duplicate_scenarios"
        completeness.append(
            {
                "stage_id": row["route_id"],
                "phase": "derived",
                "status": status,
                "path": str(path),
                "scenario_count": len(scenarios),
            }
        )
        for item in eval_rows:
            results.append(
                {
                    "stage_id": row["route_id"],
                    "phase": "derived",
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "regime": row["regime"],
                    "seed_id": row["seed_id"],
                    "scenario": item["scenario"],
                    **{metric: item[metric] for metric in METRICS},
                    "source_csv": str(path),
                }
            )
    return results, completeness


def n4_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["phase"] == "n4" and row["scenario"] in CORE_SCENARIOS:
            groups[(row["dataset"], row["method"], row["regime"], row["scenario"])].append(row)
    summary = []
    for (dataset, method, regime, scenario), group in sorted(groups.items()):
        seeds = sorted({row["seed_id"] for row in group}, key=int)
        if seeds != ["101", "202", "303"]:
            raise RuntimeError(f"N4 seed mismatch: {dataset}/{method}/{regime}/{scenario}: {seeds}")
        item: dict[str, object] = {
            "dataset": dataset,
            "method": method,
            "regime": regime,
            "scenario": scenario,
            "n_training_seeds": 3,
            "seeds": ";".join(seeds),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_training_seed_mean"] = f"{statistics.mean(values):.6f}"
            item[f"{metric}_training_seed_sd"] = f"{statistics.stdev(values):.6f}"
        summary.append(item)
    if len(summary) != 2 * 2 * 3 * 4:
        raise RuntimeError(f"Expected 48 N4 summary groups, found {len(summary)}")
    return summary


def n4_contrasts(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    lookup = {
        (row["dataset"], row["method"], row["regime"], row["seed_id"], row["scenario"]): row
        for row in rows
        if row["phase"] == "n4" and row["scenario"] in CORE_SCENARIOS
    }
    contrasts = []
    for dataset in ("potsdam", "vaihingen"):
        for scenario in sorted(CORE_SCENARIOS):
            for seed in ("101", "202", "303"):
                q_base = lookup[(dataset, "qalf", "clean", seed, scenario)]
                q_clean_control = lookup[(dataset, "qalf", "clean_continuation", seed, scenario)]
                q_robust = lookup[(dataset, "qalf", "robust", seed, scenario)]
                f_base = lookup[(dataset, "fixed_late", "clean", seed, scenario)]
                f_clean_control = lookup[(dataset, "fixed_late", "clean_continuation", seed, scenario)]
                f_robust = lookup[(dataset, "fixed_late", "robust", seed, scenario)]
                contrasts.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "seed_id": seed,
                        "qalf_robust_minus_clean_continuation_miou": f"{float(q_robust['miou_mean']) - float(q_clean_control['miou_mean']):.6f}",
                        "fixed_robust_minus_clean_continuation_miou": f"{float(f_robust['miou_mean']) - float(f_clean_control['miou_mean']):.6f}",
                        "qalf_clean_continuation_minus_base_clean_miou": f"{float(q_clean_control['miou_mean']) - float(q_base['miou_mean']):.6f}",
                        "fixed_clean_continuation_minus_base_clean_miou": f"{float(f_clean_control['miou_mean']) - float(f_base['miou_mean']):.6f}",
                        "qalf_robust_minus_fixed_robust_miou": f"{float(q_robust['miou_mean']) - float(f_robust['miou_mean']):.6f}",
                    }
                )
    return contrasts


def endpoint_decisions(contrasts: list[dict[str, object]]) -> dict:
    per_dataset = {}
    for dataset in ("potsdam", "vaihingen"):
        selected = [
            row for row in contrasts if row["dataset"] == dataset and row["scenario"] == "missing_aux_and_degraded"
        ]
        q_delta = [float(row["qalf_robust_minus_clean_continuation_miou"]) for row in selected]
        f_delta = [float(row["fixed_robust_minus_clean_continuation_miou"]) for row in selected]
        q_budget = [float(row["qalf_clean_continuation_minus_base_clean_miou"]) for row in selected]
        f_budget = [float(row["fixed_clean_continuation_minus_base_clean_miou"]) for row in selected]
        architecture = [float(row["qalf_robust_minus_fixed_robust_miou"]) for row in selected]
        d_robust = statistics.mean([(q + f) / 2.0 for q, f in zip(q_delta, f_delta)])
        d_arch = statistics.mean(architecture)
        d_arch_sd = statistics.stdev(architecture)
        stress_guardrail = {}
        for label, field in (
            ("qalf", "qalf_robust_minus_clean_continuation_miou"),
            ("fixed_late", "fixed_robust_minus_clean_continuation_miou"),
        ):
            stress_deltas = []
            full_deltas = []
            for seed in ("101", "202", "303"):
                seed_rows = [
                    row
                    for row in contrasts
                    if row["dataset"] == dataset and row["seed_id"] == seed
                ]
                stress_deltas.append(
                    statistics.mean(
                        float(row[field])
                        for row in seed_rows
                        if row["scenario"] in {"missing_aux", "degraded", "missing_aux_and_degraded"}
                    )
                )
                full_deltas.append(
                    float(next(row[field] for row in seed_rows if row["scenario"] == "full"))
                )
            stress_guardrail[label] = {
                "stress_mean_delta": statistics.mean(stress_deltas),
                "positive_stress_seed_count": sum(value > 0.0 for value in stress_deltas),
                "full_mean_delta": statistics.mean(full_deltas),
                "pass": (
                    statistics.mean(stress_deltas) > 0.0
                    and sum(value > 0.0 for value in stress_deltas) >= 2
                    and statistics.mean(full_deltas) >= -0.005
                ),
                "seed_stress_deltas": stress_deltas,
                "seed_full_deltas": full_deltas,
            }
        per_dataset[dataset] = {
            "D_robust": d_robust,
            "QALF_robust_effect_mean": statistics.mean(q_delta),
            "Fixed_robust_effect_mean": statistics.mean(f_delta),
            "QALF_positive_robust_seed_count": sum(value > 0.0 for value in q_delta),
            "Fixed_positive_robust_seed_count": sum(value > 0.0 for value in f_delta),
            "QALF_extra_budget_effect_mean": statistics.mean(q_budget),
            "Fixed_extra_budget_effect_mean": statistics.mean(f_budget),
            "D_arch": d_arch,
            "D_arch_seed_difference_sd": d_arch_sd,
            "robust_driver_dataset_pass": (
                statistics.mean(q_delta) > 0.0
                and statistics.mean(f_delta) > 0.0
                and sum(value > 0.0 for value in q_delta) >= 2
                and sum(value > 0.0 for value in f_delta) >= 2
                and d_robust > abs(d_arch)
            ),
            "architecture_advantage_dataset_pass": (
                all(value > 0.0 for value in architecture) and d_arch > d_arch_sd
            ),
            "stress_state_guardrail": stress_guardrail,
            "stress_state_guardrail_pass": all(item["pass"] for item in stress_guardrail.values()),
        }
    robust_driver = all(item["robust_driver_dataset_pass"] for item in per_dataset.values())
    stress_guardrail_pass = all(item["stress_state_guardrail_pass"] for item in per_dataset.values())
    return {
        "schema": "qalf-rebuild-primary-endpoint-decision-v2",
        "scenario": "missing_aux_and_degraded",
        "per_dataset": per_dataset,
        "robust_training_main_driver_claim_supported": robust_driver,
        "stress_state_generalization_guardrail_pass": stress_guardrail_pass,
        "robust_training_claim_scope": (
            "evaluated_missing_and_degraded_states"
            if robust_driver and stress_guardrail_pass
            else "combined_state_only"
            if robust_driver
            else "method_or_dataset_specific_only"
        ),
        "consistent_architecture_advantage_claim_supported": all(
            item["architecture_advantage_dataset_pass"] for item in per_dataset.values()
        ),
        "availability_mechanism_claim": "pending_n7_gate_diagnostics",
    }


def n5_confirmation(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict]:
    selected = [row for row in rows if row["phase"] == "n5" and row["scenario"] in CORE_SCENARIOS]
    lookup = {
        (row["dataset"], row["method"], row["regime"], row["scenario"]): float(row["miou_mean"])
        for row in selected
    }
    if len(selected) != 56 or len(lookup) != 56:
        raise RuntimeError(f"Expected 56 unique N5 core-scenario rows, found {len(selected)}/{len(lookup)}")
    contrasts = []
    decisions = {}
    stress_scenarios = ("missing_aux", "degraded", "missing_aux_and_degraded")
    for dataset, seed_id in (("vaihingen", "254"), ("potsdam", "264")):
        if {row["seed_id"] for row in selected if row["dataset"] == dataset} != {seed_id}:
            raise RuntimeError(f"N5 confirmatory seed drift: {dataset}")
        for scenario in sorted(CORE_SCENARIOS):
            q_robust = lookup[(dataset, "qalf", "robust", scenario)]
            q_control = lookup[(dataset, "qalf", "clean_continuation", scenario)]
            f_robust = lookup[(dataset, "fixed_late", "robust", scenario)]
            f_control = lookup[(dataset, "fixed_late", "clean_continuation", scenario)]
            contrasts.append(
                {
                    "dataset": dataset,
                    "seed_id": seed_id,
                    "scenario": scenario,
                    "qalf_robust_minus_clean_continuation_miou": q_robust - q_control,
                    "fixed_robust_minus_clean_continuation_miou": f_robust - f_control,
                    "qalf_robust_minus_fixed_robust_miou": q_robust - f_robust,
                    "qalf_robust_minus_optical_clean_miou": (
                        q_robust - lookup[(dataset, "main_only", "clean", scenario)]
                    ),
                }
            )
        dataset_rows = [row for row in contrasts if row["dataset"] == dataset]
        combined = next(row for row in dataset_rows if row["scenario"] == "missing_aux_and_degraded")
        d_robust = statistics.mean(
            [
                float(combined["qalf_robust_minus_clean_continuation_miou"]),
                float(combined["fixed_robust_minus_clean_continuation_miou"]),
            ]
        )
        d_arch = float(combined["qalf_robust_minus_fixed_robust_miou"])
        q_stress = statistics.mean(
            float(row["qalf_robust_minus_clean_continuation_miou"])
            for row in dataset_rows
            if row["scenario"] in stress_scenarios
        )
        f_stress = statistics.mean(
            float(row["fixed_robust_minus_clean_continuation_miou"])
            for row in dataset_rows
            if row["scenario"] in stress_scenarios
        )
        full = next(row for row in dataset_rows if row["scenario"] == "full")
        decisions[dataset] = {
            "seed_id": seed_id,
            "combined_D_robust": d_robust,
            "combined_D_arch": d_arch,
            "combined_directional_confirmation": (
                float(combined["qalf_robust_minus_clean_continuation_miou"]) > 0.0
                and float(combined["fixed_robust_minus_clean_continuation_miou"]) > 0.0
                and d_robust > abs(d_arch)
            ),
            "stress_state_directional_confirmation": (
                q_stress > 0.0
                and f_stress > 0.0
                and float(full["qalf_robust_minus_clean_continuation_miou"]) >= -0.005
                and float(full["fixed_robust_minus_clean_continuation_miou"]) >= -0.005
            ),
            "qalf_stress_mean_delta": q_stress,
            "fixed_stress_mean_delta": f_stress,
            "qalf_minus_optical_combined": combined["qalf_robust_minus_optical_clean_miou"],
        }
    report = {
        "schema": "qalf-n5-internal-holdout-confirmation-v1",
        "status": "descriptive_confirmation_complete",
        "evidence_boundary": "internal_result_blind_holdout",
        "official_test_server": False,
        "inference": "single_preregistered_seed_per_dataset_no_significance_inference",
        "per_dataset": decisions,
        "combined_directionally_confirmed_both_datasets": all(
            item["combined_directional_confirmation"] for item in decisions.values()
        ),
        "stress_state_directionally_confirmed_both_datasets": all(
            item["stress_state_directional_confirmation"] for item in decisions.values()
        ),
    }
    return contrasts, report


def quality_weighted_ablation(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    lookup = {
        (row["phase"], row["dataset"], row["method"], row["regime"], row["seed_id"], row["scenario"]): row
        for row in rows
        if row["scenario"] in CORE_SCENARIOS
    }
    output = []
    for dataset in ("potsdam", "vaihingen"):
        for seed in ("101", "202", "303"):
            for scenario in sorted(CORE_SCENARIOS):
                quality_base = lookup[("n6", dataset, "quality_weighted", "clean", seed, scenario)]
                quality_control = lookup[
                    ("n6", dataset, "quality_weighted", "clean_continuation", seed, scenario)
                ]
                quality_robust = lookup[("n6", dataset, "quality_weighted", "robust", seed, scenario)]
                qalf_robust = lookup[("n4", dataset, "qalf", "robust", seed, scenario)]
                output.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "seed_id": seed,
                        "quality_robust_minus_clean_continuation_miou": (
                            f"{float(quality_robust['miou_mean']) - float(quality_control['miou_mean']):.6f}"
                        ),
                        "quality_clean_continuation_minus_base_clean_miou": (
                            f"{float(quality_control['miou_mean']) - float(quality_base['miou_mean']):.6f}"
                        ),
                        "qalf_robust_minus_quality_robust_miou": (
                            f"{float(qalf_robust['miou_mean']) - float(quality_robust['miou_mean']):.6f}"
                        ),
                        "inference_role": "seed_matched_component_link",
                    }
                )
    if len(output) != 24:
        raise RuntimeError(f"Expected 24 quality-weighted ablation rows, found {len(output)}")
    return output


def component_performance_gate(rows: list[dict[str, str]]) -> dict:
    lookup = {
        (row["dataset"], row["method"], row["regime"], row["seed_id"], row["scenario"]): float(
            row["miou_mean"]
        )
        for row in rows
        if row["scenario"] in CORE_SCENARIOS
    }
    links = (
        ("masked_minus_fixed", "availability_masked_fixed", "fixed_late"),
        ("quality_minus_masked", "quality_weighted", "availability_masked_fixed"),
        ("qalf_minus_quality", "qalf", "quality_weighted"),
    )
    report = {}
    stress_scenarios = ("missing_aux", "degraded", "missing_aux_and_degraded")
    for link, candidate_method, control_method in links:
        datasets = {}
        for dataset in ("potsdam", "vaihingen"):
            seed_rows = []
            stress_deltas = []
            full_deltas = []
            for seed in ("101", "202", "303"):
                candidate_stress = statistics.mean(
                    lookup[(dataset, candidate_method, "robust", seed, scenario)]
                    for scenario in stress_scenarios
                )
                control_stress = statistics.mean(
                    lookup[(dataset, control_method, "robust", seed, scenario)]
                    for scenario in stress_scenarios
                )
                stress_delta = candidate_stress - control_stress
                full_delta = (
                    lookup[(dataset, candidate_method, "robust", seed, "full")]
                    - lookup[(dataset, control_method, "robust", seed, "full")]
                )
                stress_deltas.append(stress_delta)
                full_deltas.append(full_delta)
                seed_rows.append(
                    {
                        "seed_id": seed,
                        "stress_mean_delta": stress_delta,
                        "full_delta": full_delta,
                    }
                )
            stress_sd = statistics.stdev(stress_deltas)
            dataset_pass = (
                sum(value > 0.0 for value in stress_deltas) >= 2
                and statistics.mean(stress_deltas) > stress_sd
                and statistics.mean(full_deltas) >= -0.005
            )
            datasets[dataset] = {
                "positive_seed_count": sum(value > 0.0 for value in stress_deltas),
                "mean_stress_delta": statistics.mean(stress_deltas),
                "sample_sd_stress_delta": stress_sd,
                "mean_full_delta": statistics.mean(full_deltas),
                "pass": dataset_pass,
                "seeds": seed_rows,
            }
        report[link] = {
            "mechanism_evidence": "separate_gate_required",
            "performance_pass": all(item["pass"] for item in datasets.values()),
            "datasets": datasets,
        }
    optical_datasets = {}
    for dataset in ("potsdam", "vaihingen"):
        seed = "101"
        stress_delta = statistics.mean(
            lookup[(dataset, "qalf", "robust", seed, scenario)]
            - lookup[(dataset, "main_only", "robust", seed, scenario)]
            for scenario in stress_scenarios
        )
        full_delta = (
            lookup[(dataset, "qalf", "robust", seed, "full")]
            - lookup[(dataset, "main_only", "robust", seed, "full")]
        )
        optical_datasets[dataset] = {
            "seed_id": seed,
            "stress_mean_delta": stress_delta,
            "full_delta": full_delta,
            "pass": False,
            "reason": "single_seed_fallback_reference_cannot_satisfy_three_seed_consistency_gate",
        }
    report["qalf_minus_optical_fallback"] = {
        "mechanism_evidence": "not_applicable",
        "performance_pass": False,
        "inference_role": "descriptive_single_seed_fallback_attribution_only",
        "datasets": optical_datasets,
    }
    return {
        "schema": "qalf-component-performance-gate-v1",
        "criterion": "per dataset >=2/3 positive stress deltas, mean > sample SD, mean full delta >= -0.50 pp",
        "links": report,
    }


def main() -> None:
    args = parse_args()
    manifest_path = (ROOT / args.manifest).resolve()
    manifest_rows = read_csv(manifest_path)
    if len(manifest_rows) != 85:
        raise RuntimeError(f"Expected 85 manifest rows, found {len(manifest_rows)}")
    rows, completeness = collect(manifest_rows)
    derived_manifest_path = ROOT / "configs/rebuild_20260714/derived_evaluation_manifest.csv"
    derived_manifest = read_csv(derived_manifest_path)
    if len(derived_manifest) != 12:
        raise RuntimeError(f"Expected 12 derived routes, found {len(derived_manifest)}")
    derived_rows, derived_completeness = collect_derived(derived_manifest)
    rows.extend(derived_rows)
    completeness.extend(derived_completeness)
    missing = [row for row in completeness if row["status"] != "complete"]
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "rebuild_evaluation_completeness.csv", completeness)
    (output / "rebuild_evaluation_completeness.json").write_text(
        json.dumps(
            {
                "manifest_stages": 85,
                "derived_routes": 12,
                "manifest_sha256": sha256_file(manifest_path),
                "derived_manifest_sha256": sha256_file(derived_manifest_path),
                "complete_routes": 97 - len(missing),
                "incomplete_routes": len(missing),
                "status": "pass" if not missing else "incomplete",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if missing:
        print(json.dumps({"status": "incomplete", "missing_or_incomplete_stages": len(missing)}))
        if not args.allow_incomplete:
            raise RuntimeError(f"Rebuild evaluation is incomplete for {len(missing)} stage(s)")
        return

    all_gate_path = ROOT / "outputs/rebuild_20260714/protocol/all_all_gate_report.json"
    if not all_gate_path.is_file():
        raise FileNotFoundError(f"Passing all-phase gate is required before aggregation: {all_gate_path}")
    all_gate = json.loads(all_gate_path.read_text(encoding="utf-8"))
    derived_gate_path = ROOT / "outputs/rebuild_20260714/protocol/derived_evaluation_gate.json"
    derived_gate = json.loads(derived_gate_path.read_text(encoding="utf-8"))
    if (
        all_gate.get("status") != "pass"
        or all_gate.get("selected_stages") != 85
        or all_gate.get("manifest_sha256") != sha256_file(manifest_path)
        or derived_gate.get("status") != "pass"
        or derived_gate.get("manifest_sha256") != sha256_file(derived_manifest_path)
        or len(derived_gate.get("routes", [])) != 12
    ):
        raise RuntimeError("All-phase gate does not match the complete formal manifest")

    write_csv(output / "rebuild_all_eval_rows.csv", rows)
    summary = n4_summary(rows)
    contrasts = n4_contrasts(rows)
    decisions = endpoint_decisions(contrasts)
    quality_ablation = quality_weighted_ablation(rows)
    component_gate = component_performance_gate(rows)
    n5_contrasts, n5_report = n5_confirmation(rows)
    write_csv(output / "n4_training_seed_summary.csv", summary)
    write_csv(output / "n4_seed_contrasts.csv", contrasts)
    write_csv(output / "quality_weighted_ablation.csv", quality_ablation)
    write_csv(output / "n5_internal_holdout_contrasts.csv", n5_contrasts)
    (output / "component_performance_gate.json").write_text(
        json.dumps(component_gate, indent=2), encoding="utf-8"
    )
    (output / "primary_endpoint_decisions.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    (output / "n5_internal_holdout_confirmation.json").write_text(
        json.dumps(n5_report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "evaluation_rows": len(rows),
                "robust_driver_supported": decisions["robust_training_main_driver_claim_supported"],
                "architecture_advantage_supported": decisions["consistent_architecture_advantage_claim_supported"],
            }
        )
    )


if __name__ == "__main__":
    main()
