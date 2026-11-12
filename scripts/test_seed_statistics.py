"""Regression tests for training-seed uncertainty and invalid evidence."""
import json
import tempfile
import unittest
from pathlib import Path

import summarize_minimal_claim_results as agg
from reproduction_evidence import validate_checkpoint


def make_evidence_fixture(base, scenarios="full|missing_aux"):
    config, checkpoint = base / "config.yaml", base / "last_model.pt"
    config.write_text("synthetic fixture", encoding="utf-8")
    checkpoint.write_bytes(b"synthetic checkpoint identity fixture")
    row = dict(config=str(config), config_sha256=agg.sha256_file(config), checkpoint=str(checkpoint),
               split_sha256="ABC", eval_split="val_split", trial_policy=agg.TRIAL_POLICY, scenarios=scenarios)
    meta = dict(config=str(config), checkpoint=str(checkpoint), split="val_split", split_sha256="ABC",
                config_sha256=agg.sha256_file(config), checkpoint_sha256=agg.sha256_file(checkpoint),
                summary_csv_sha256=agg.sha256_file(base / "eval_protocol_summary.csv"),
                trials=3, single_clean_trial=True, scenarios=scenarios.split("|"))
    counts = dict(full=1, missing_aux=1, degraded=3, missing_aux_and_degraded=3)
    results = {s: dict(trials=[dict(trial=i) for i in range(counts[s])]) for s in scenarios.split("|")}
    for name, content, key in (("eval_protocol_results.json", json.dumps(results), "results_json_sha256"),
                               ("eval_protocol_classwise.csv", "fixture", "classwise_csv_sha256"),
                               ("eval_protocol_confusion.json", "{}", "confusion_json_sha256")):
        (base / name).write_text(content, encoding="utf-8")
        meta[key] = agg.sha256_file(base / name)
    receipt = dict(status="pass", termination_reason="max_epochs", completed_epoch=1, requested_epochs=1,
                   config_sha256=meta["config_sha256"], last_model_sha256=meta["checkpoint_sha256"])
    (base / "training_complete.json").write_text(json.dumps(receipt), encoding="utf-8")
    meta["checkpoint_contract"] = dict(status="pass", epoch=1,
                                       training_complete_sha256=agg.sha256_file(base / "training_complete.json"))
    (base / "eval_protocol_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return row, meta


class SeedTests(unittest.TestCase):
    def records(self):
        result = []
        for seed, delta in zip((101, 202, 303), (-0.02, 0.01, -0.005)):
            for method, value in (("fixed_late", 0.7), ("qalf", 0.7 + delta)):
                result.append(dict(scope="main", eval_split="val_split", dataset="test", method=method,
                                   regime="robust", scenario="full", seed_id=str(seed), miou_6class=str(value), split_sha256="ABC"))
        return result

    def test_paired_interval_and_signs(self):
        summaries, contrasts = agg.seed_statistics(self.records())
        contrast = next(c for c in contrasts if c["contrast"] == "qalf_minus_fixed_robust")
        self.assertAlmostEqual(contrast["mean_difference"], -0.005)
        self.assertEqual(contrast["positive_seeds"], 1)
        self.assertLess(contrast["ci95_low"], 0)
        self.assertGreater(contrast["ci95_high"], 0)
        self.assertAlmostEqual(next(s for s in summaries if s["method"] == "fixed_late")["miou_sample_sd"], 0)

    def test_missing_and_non_estimable_seeds(self):
        for records in (self.records()[:-1], self.records()):
            if len(records) == 6:
                records[-1]["miou_6class"] = "not_estimable"
            _, contrasts = agg.seed_statistics(records)
            self.assertEqual(contrasts[0]["status"], "not_estimable")

    def test_duplicate_seed_is_rejected(self):
        records = self.records()
        with self.assertRaises(ValueError):
            agg.seed_statistics(records + [records[0]])

    def test_missing_and_tampered_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            csv = base / "eval_protocol_summary.csv"
            csv.write_text("fixture", encoding="utf-8")
            row = dict(config="", config_sha256="", checkpoint="", split_sha256="ABC", eval_split="val_split")
            self.assertTrue(agg.provenance_error(row, csv))
            row, meta = make_evidence_fixture(base)
            self.assertEqual(agg.provenance_error(row, csv), "")
            csv.write_text("changed", encoding="utf-8")
            self.assertIn("summary_csv_sha256 mismatch", agg.provenance_error(row, csv))

    def test_split_hash_mixing_is_rejected(self):
        rows = self.records()
        rows[-1]["split_sha256"] = "DIFFERENT"
        with self.assertRaises(ValueError):
            agg.seed_statistics(rows)

    def test_trial_policy_and_payload_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            csv = base / "eval_protocol_summary.csv"
            csv.write_text("fixture", encoding="utf-8")
            row, meta = make_evidence_fixture(base)
            self.assertIn("trial_policy", agg.provenance_error({**row, "trial_policy": "typo"}, csv))
            results_path = base / "eval_protocol_results.json"
            result = json.loads(results_path.read_text())
            result["full"]["trials"].append(dict(trial=1))
            results_path.write_text(json.dumps(result))
            meta["results_json_sha256"] = agg.sha256_file(results_path)
            (base / "eval_protocol_metadata.json").write_text(json.dumps(meta))
            self.assertIn("actual trial identity/count mismatch", agg.provenance_error(row, csv))

    def test_renamed_wrong_checkpoint_and_epoch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last_model.pt"
            path.write_bytes(b"fixture")
            config = dict(experiment=dict(seed=101), train=dict(epochs=28))
            for checkpoint in (dict(config=dict(experiment=dict(seed=303), train=dict(epochs=28)), epoch=28),
                               dict(config=config, epoch=27)):
                with self.assertRaises(ValueError):
                    validate_checkpoint(checkpoint, config, path, "ABC")

    def test_manifest_seed_split_and_duplicate_identity(self):
        manifest = agg.ROOT / agg.DEFAULT_MANIFEST
        row = agg.read_csv(manifest)[0]
        agg.validate_manifest_rows([row], manifest)
        for changed in ({**row, "seed_id": "303"}, {**row, "split_sha256": "BAD"}):
            with self.assertRaises(ValueError):
                agg.validate_manifest_rows([changed], manifest)
        with self.assertRaises(ValueError):
            agg.validate_manifest_rows([row, row], manifest)


if __name__ == "__main__":
    unittest.main()
