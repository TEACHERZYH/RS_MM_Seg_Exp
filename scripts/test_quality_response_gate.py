#!/usr/bin/env python3
"""Synthetic pass, downgrade, and malformed-grid tests for the N7 wording gate."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "scenario",
    "target",
    "corruption",
    "severity",
    "trial",
    "sample_id",
    "scale",
    "quality_main",
    "quality_aux",
    "gate_main",
    "gate_aux",
]


def rows(direction: float = -1.0) -> list[dict[str, object]]:
    result = []
    for target in ("main", "aux"):
        for corruption in ("noise", "blur", "mask", "lowres"):
            for scale in range(4):
                for severity in range(6):
                    trials = (0,) if severity == 0 else range(3)
                    for trial in trials:
                        value = 0.5 + direction * 0.05 * severity
                        result.append(
                            {
                                "scenario": "targeted_corruption",
                                "target": target,
                                "corruption": corruption,
                                "severity": severity,
                                "trial": trial,
                                "sample_id": "sample_001",
                                "scale": scale,
                                "quality_main": value,
                                "quality_aux": value,
                                "gate_main": value,
                                "gate_aux": value,
                            }
                        )
    return result


def write(path: Path, values: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def run(inputs: tuple[Path, Path], output: Path, expected: int = 0) -> subprocess.CompletedProcess:
    command = [sys.executable, "scripts/check_quality_response_gate.py"]
    for dataset, path in zip(("vaihingen", "potsdam"), inputs):
        command.extend(["--dataset-csv", f"{dataset}={path}"])
    command.extend(["--output", str(output)])
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != expected:
        raise RuntimeError(f"Quality-response test returned {result.returncode}: {result.stderr[-400:]}")
    return result


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="qalf_quality_gate_") as temporary:
        root = Path(temporary)
        vaihingen = root / "vaihingen.csv"
        potsdam = root / "potsdam.csv"
        output = root / "quality.json"
        write(vaihingen, rows(-1.0))
        write(potsdam, rows(-1.0))
        run((vaihingen, potsdam), output)
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload["status"] != "pass" or any(
            report["curves"] != 96 for report in payload["datasets"].values()
        ):
            raise AssertionError("Valid negative-response grid did not pass")

        write(potsdam, rows(1.0))
        run((vaihingen, potsdam), output)
        if json.loads(output.read_text(encoding="utf-8"))["status"] != "wording_downgrade":
            raise AssertionError("Non-responsive dataset did not downgrade the wording")

        malformed = rows(-1.0)
        malformed.append(dict(malformed[0]))
        write(potsdam, malformed)
        run((vaihingen, potsdam), output, expected=1)

        malformed = [row for row in rows(-1.0) if int(row["scale"]) != 3]
        write(potsdam, malformed)
        run((vaihingen, potsdam), output, expected=1)
    print(json.dumps({"status": "pass", "valid_curves": 192, "malformed_cases_rejected": 2}))


if __name__ == "__main__":
    main()
