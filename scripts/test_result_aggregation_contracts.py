#!/usr/bin/env python3
"""Synthetic tests for primary guardrails and result-blind N5 aggregation."""

from __future__ import annotations

import copy
import json

from summarize_full_rebuild_results import CORE_SCENARIOS, endpoint_decisions, n5_confirmation


def n4_contrasts() -> list[dict[str, object]]:
    rows = []
    for dataset in ("vaihingen", "potsdam"):
        for seed in ("101", "202", "303"):
            for scenario in CORE_SCENARIOS:
                rows.append(
                    {
                        "dataset": dataset,
                        "seed_id": seed,
                        "scenario": scenario,
                        "qalf_robust_minus_clean_continuation_miou": -0.004 if scenario == "full" else 0.030,
                        "fixed_robust_minus_clean_continuation_miou": -0.003 if scenario == "full" else 0.020,
                        "qalf_clean_continuation_minus_base_clean_miou": 0.001,
                        "fixed_clean_continuation_minus_base_clean_miou": 0.001,
                        "qalf_robust_minus_fixed_robust_miou": 0.005,
                    }
                )
    return rows


def n5_rows() -> list[dict[str, str]]:
    rows = []
    for dataset, seed in (("vaihingen", "254"), ("potsdam", "264")):
        for scenario in CORE_SCENARIOS:
            stress = scenario != "full"
            values = {
                ("main_only", "clean"): 0.50,
                ("fixed_late", "clean"): 0.55,
                ("fixed_late", "clean_continuation"): 0.56,
                ("fixed_late", "robust"): 0.60 if stress else 0.559,
                ("qalf", "clean"): 0.56,
                ("qalf", "clean_continuation"): 0.57,
                ("qalf", "robust"): 0.62 if stress else 0.569,
            }
            for (method, regime), value in values.items():
                rows.append(
                    {
                        "phase": "n5",
                        "dataset": dataset,
                        "method": method,
                        "regime": regime,
                        "seed_id": seed,
                        "scenario": scenario,
                        "miou_mean": str(value),
                    }
                )
    return rows


def main() -> None:
    decision = endpoint_decisions(n4_contrasts())
    if (
        not decision["robust_training_main_driver_claim_supported"]
        or not decision["stress_state_generalization_guardrail_pass"]
        or decision["robust_training_claim_scope"] != "evaluated_missing_and_degraded_states"
    ):
        raise AssertionError("Valid primary endpoint and stress guardrail did not pass")

    failed = copy.deepcopy(n4_contrasts())
    for row in failed:
        if row["scenario"] == "full":
            row["qalf_robust_minus_clean_continuation_miou"] = -0.006
    failed_decision = endpoint_decisions(failed)
    if failed_decision["stress_state_generalization_guardrail_pass"]:
        raise AssertionError("Full-input loss beyond 0.50 pp did not block broad wording")
    if failed_decision["robust_training_claim_scope"] != "combined_state_only":
        raise AssertionError("Failed stress guardrail did not preserve the bounded combined-only claim")

    contrasts, holdout = n5_confirmation(n5_rows())
    if len(contrasts) != 8 or not holdout["combined_directionally_confirmed_both_datasets"]:
        raise AssertionError("N5 result-blind directional confirmation was not summarized")
    if holdout["official_test_server"] or "no_significance" not in holdout["inference"]:
        raise AssertionError("N5 aggregation crossed its internal single-seed evidence boundary")
    print(
        json.dumps(
            {
                "status": "pass",
                "primary_guardrail_cases": 2,
                "n5_contrast_rows": len(contrasts),
            }
        )
    )


if __name__ == "__main__":
    main()
