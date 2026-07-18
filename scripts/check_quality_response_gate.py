#!/usr/bin/env python3
"""Apply the preregistered two-dataset degradation-response wording gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EXPECTED_TARGETS = {"main", "aux"}
EXPECTED_CORRUPTIONS = {"noise", "blur", "mask", "lowres"}
EXPECTED_SCALES = {0, 1, 2, 3}
EXPECTED_SEVERITIES = set(range(6))
EXPECTED_TRIALS = set(range(3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", action="append", required=True, help="dataset=long.csv")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def ranks(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    result = np.empty(len(array), dtype=np.float64)
    position = 0
    while position < len(array):
        end = position + 1
        while end < len(array) and array[order[end]] == array[order[position]]:
            end += 1
        result[order[position:end]] = (position + end - 1) / 2.0 + 1.0
        position = end
    return result


def spearman(x: list[float], y: list[float]) -> float:
    rx = ranks(x)
    ry = ranks(y)
    if float(rx.std()) == 0.0 or float(ry.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def parse_inputs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"dataset CSV must use dataset=path: {value}")
        dataset, path = value.split("=", 1)
        if dataset in result:
            raise ValueError(f"Duplicate dataset input: {dataset}")
        result[dataset] = Path(path)
    if set(result) != {"vaihingen", "potsdam"}:
        raise ValueError("Quality-response gate requires Vaihingen and Potsdam")
    return result


def main() -> None:
    args = parse_args()
    inputs = parse_inputs(args.dataset_csv)
    curve_rows = []
    dataset_reports = {}
    for dataset, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["scenario"] == "targeted_corruption"]
        indexed = {
            (
                row["sample_id"],
                row["target"],
                row["corruption"],
                int(row["scale"]),
                int(row["severity"]),
                int(row["trial"]),
            ): row
            for row in rows
        }
        if len(indexed) != len(rows):
            raise RuntimeError(f"Quality-response input contains duplicate frozen points: {dataset}")
        sample_ids = sorted({row["sample_id"] for row in rows})
        targets = sorted({row["target"] for row in rows})
        corruptions = sorted({row["corruption"] for row in rows})
        scales = sorted({int(row["scale"]) for row in rows})
        if (
            not sample_ids
            or set(targets) != EXPECTED_TARGETS
            or set(corruptions) != EXPECTED_CORRUPTIONS
            or set(scales) != EXPECTED_SCALES
            or {int(row["severity"]) for row in rows} != EXPECTED_SEVERITIES
            or any(
                int(row["trial"]) not in ({0} if int(row["severity"]) == 0 else EXPECTED_TRIALS)
                for row in rows
            )
        ):
            raise RuntimeError(f"Quality-response input violates the frozen factorial grid: {dataset}")
        for row in rows:
            values = (
                float(row["quality_main"]),
                float(row["quality_aux"]),
                float(row["gate_main"]),
                float(row["gate_aux"]),
            )
            if not all(np.isfinite(value) for value in values):
                raise RuntimeError(f"Quality-response input contains a non-finite diagnostic: {dataset}")
        expected_points = len(sample_ids) * 2 * 4 * 4 * (1 + 5 * 3)
        if len(rows) != expected_points:
            raise RuntimeError(
                f"Quality-response point grid is incomplete for {dataset}: {len(rows)} != {expected_points}"
            )
        expected_curves = len(sample_ids) * 2 * 4 * 4 * 3
        built_curves = 0
        for sample_id in sample_ids:
            for target in targets:
                for corruption in corruptions:
                    for scale in scales:
                        for trial in range(3):
                            ordered = []
                            for severity in range(6):
                                key = (
                                    sample_id,
                                    target,
                                    corruption,
                                    scale,
                                    severity,
                                    0 if severity == 0 else trial,
                                )
                                if key not in indexed:
                                    raise RuntimeError(
                                        f"Quality-response curve lacks a frozen point: {dataset}/{key}"
                                    )
                                ordered.append(indexed[key])
                            built_curves += 1
                            severities = list(range(6))
                            quality_field = "quality_main" if target == "main" else "quality_aux"
                            gate_field = "gate_main" if target == "main" else "gate_aux"
                            curve_rows.append(
                                {
                                    "dataset": dataset,
                                    "sample_id": sample_id,
                                    "target": target,
                                    "corruption": corruption,
                                    "scale": scale,
                                    "trial": trial,
                                    "q_rho": spearman(
                                        severities, [float(row[quality_field]) for row in ordered]
                                    ),
                                    "g_rho": spearman(
                                        severities, [float(row[gate_field]) for row in ordered]
                                    ),
                                }
                            )
        if built_curves != expected_curves:
            raise RuntimeError(
                f"Incomplete quality-response curve grid for {dataset}: {built_curves} != {expected_curves}"
            )
        selected = [row for row in curve_rows if row["dataset"] == dataset]
        family_scale_groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
        for row in selected:
            family_scale_groups[(row["target"], row["corruption"], int(row["scale"]))].append(row)
        family_scale = []
        for (target, corruption, scale), group in sorted(family_scale_groups.items()):
            family_scale.append(
                {
                    "target": target,
                    "corruption": corruption,
                    "scale": scale,
                    "q_median_rho": float(np.median([row["q_rho"] for row in group])),
                    "g_median_rho": float(np.median([row["g_rho"] for row in group])),
                }
            )
        q_global = float(np.median([row["q_rho"] for row in selected]))
        g_global = float(np.median([row["g_rho"] for row in selected]))
        q_negative_fraction = sum(row["q_median_rho"] < 0.0 for row in family_scale) / len(family_scale)
        g_negative_fraction = sum(row["g_median_rho"] < 0.0 for row in family_scale) / len(family_scale)
        dataset_reports[dataset] = {
            "curves": len(selected),
            "family_scale_groups": len(family_scale),
            "q_global_median_rho": q_global,
            "g_global_median_rho": g_global,
            "q_negative_family_scale_fraction": q_negative_fraction,
            "g_negative_family_scale_fraction": g_negative_fraction,
            "pass": (
                q_global < 0.0
                and g_global < 0.0
                and q_negative_fraction >= 0.75
                and g_negative_fraction >= 0.75
            ),
            "family_scale": family_scale,
        }
    gate_pass = all(report["pass"] for report in dataset_reports.values())
    payload = {
        "schema": "qalf-quality-response-gate-v1",
        "status": "pass" if gate_pass else "wording_downgrade",
        "datasets": dataset_reports,
        "claim_if_pass": "degradation-responsive reliability/quality weighting",
        "claim_if_fail": "availability-aware fusion with exported forward diagnostics",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    long_path = output.with_name("quality_response_curve_correlations.csv")
    with long_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(curve_rows)
    print(json.dumps({"status": payload["status"], "curves": len(curve_rows)}))


if __name__ == "__main__":
    main()
