"""Read-only plan validation: reject manifest drift before emitting commands."""
from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

if __package__:
    from . import plan_reproduction as planner
else:
    import plan_reproduction as planner


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.matrix = Path(self.temp.name)
        for path in planner.DEFAULT_MATRIX.iterdir():
            if path.suffix in {".csv", ".json"}:
                shutil.copyfile(path, self.matrix / path.name)

    def change(self, manifest, mutate):
        path = self.matrix / manifest
        values = planner.rows(path)
        columns = list(values[0])
        mutate(values)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(values)

    def rejected(self, manifest, mutate, message):
        self.change(manifest, mutate)
        with self.assertRaisesRegex(ValueError, message):
            planner.build_plan(self.matrix)

    def test_current_matrix_exports_commands_only(self):
        plan = planner.build_plan(self.matrix)
        self.assertFalse(plan["executed"])
        self.assertEqual(plan["training_stages"], 43)
        self.assertEqual(len(plan["commands"]), 95)
        holdouts = [item for item in plan["commands"] if item["kind"] == "holdout"]
        self.assertEqual(len(holdouts), 24)
        self.assertTrue(all(item["requires_separate_holdout_authorization"] for item in holdouts))
        self.assertTrue(all("--expected-split-sha256" in item["argv"] for item in holdouts))
        self.assertTrue(all("--require-training-complete" in item["argv"]
                            for item in plan["commands"] if item["kind"] != "train"))

    def test_parent_id_must_match_the_resume_checkpoint(self):
        self.rejected("training_stage_manifest.csv",
                      lambda values: values[2].update(resume_stage_id="M3-002"), "Parent checkpoint mismatch")

    def test_resume_cannot_be_orphaned(self):
        self.rejected("training_stage_manifest.csv",
                      lambda values: values[2].update(resume_stage_id=""), "Resume path has no parent stage")

    def test_stage_seed_must_match_yaml(self):
        self.rejected("training_stage_manifest.csv",
                      lambda values: values[0].update(seed_id="202"), "Stage/config mismatch.*seed_id")

    def test_stage_output_cannot_name_a_different_endpoint(self):
        self.rejected("training_stage_manifest.csv",
                      lambda values: values[0].update(checkpoint_for_eval="outputs/wrong.pt"), "endpoint mismatch")

    def test_evaluation_seed_must_match_source_stage(self):
        self.rejected("main_evaluation_manifest.csv",
                      lambda values: values[0].update(seed_id="202"), "stage identity mismatch.*seed_id")

    def test_evaluation_method_must_match_source_stage(self):
        self.rejected("main_evaluation_manifest.csv",
                      lambda values: values[0].update(method="qalf"), "stage identity mismatch.*method")

    def test_evaluation_dataset_must_match_source_stage(self):
        self.rejected("main_evaluation_manifest.csv",
                      lambda values: values[0].update(dataset="potsdam"), "stage identity mismatch.*dataset")

    def test_evaluation_regime_must_match_source_stage(self):
        self.rejected("main_evaluation_manifest.csv",
                      lambda values: values[0].update(regime="robust"), "stage identity mismatch.*regime")

    def test_evaluation_checkpoint_must_match_source_endpoint(self):
        self.rejected("main_evaluation_manifest.csv",
                      lambda values: values[0].update(checkpoint=values[2]["checkpoint"]), "Evaluation checkpoint mismatch")

    def test_evaluation_config_must_match_source_stage(self):
        def mutate(values):
            values[0]["config"] = values[2]["config"]
            values[0]["config_sha256"] = values[2]["config_sha256"]
        self.rejected("main_evaluation_manifest.csv", mutate, "Evaluation source config mismatch")

    def test_evaluation_source_stage_must_exist(self):
        self.rejected("main_evaluation_manifest.csv",
                      lambda values: values[0].update(source_stage_id="MISSING"), "Unknown evaluation source stage")

    def test_evaluation_id_must_be_unique_within_scope(self):
        self.rejected("main_evaluation_manifest.csv",
                      lambda values: values[1].update(eval_id=values[0]["eval_id"]), "Duplicate evaluation ID")

    def test_duplicate_run_cannot_be_given_another_id(self):
        def mutate(values):
            identity = values[1]["eval_id"]
            values[1] = dict(values[0], eval_id=identity)
        self.rejected("main_evaluation_manifest.csv", mutate, "Duplicate evaluation run or result directory")

    def test_result_directory_must_be_unique(self):
        self.rejected("main_evaluation_manifest.csv",
                      lambda values: values[1].update(result_dir=values[0]["result_dir"]), "Duplicate evaluation run or result directory")

    def test_declared_scope_must_match_manifest_scope(self):
        self.rejected("main_evaluation_manifest.csv",
                      lambda values: values[0].update(scope="holdout"), "Evaluation scope mismatch")

    def test_existing_train_split_is_not_a_valid_holdout_alias(self):
        self.rejected("holdout_evaluation_manifest.csv",
                      lambda values: values[0].update(eval_split="train_split"), "Evaluation split policy mismatch")

    def test_trial_policy_must_not_be_blank_or_unknown(self):
        self.rejected("main_evaluation_manifest.csv",
                      lambda values: values[0].update(trial_policy=""), "Evaluation trial policy mismatch")

    def test_missing_trial_policy_column_is_rejected(self):
        path = self.matrix / "main_evaluation_manifest.csv"
        values = planner.rows(path)
        for item in values:
            item.pop("trial_policy")
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
        with self.assertRaisesRegex(ValueError, "Evaluation trial policy mismatch"):
            planner.build_plan(self.matrix)

    def test_scenario_count_cannot_be_preserved_with_duplicate_scenario(self):
        self.rejected("main_evaluation_manifest.csv",
                      lambda values: values[0].update(scenarios="full|missing_aux|degraded|degraded"), "Evaluation scenario policy mismatch")


if __name__ == "__main__":
    unittest.main()
