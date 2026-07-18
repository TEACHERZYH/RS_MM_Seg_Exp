#!/usr/bin/env python3
"""Run full-file integrity and class-map QA for a prepared ISPRS root."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_MAP = {
    0: "impervious_surface",
    1: "building",
    2: "low_vegetation",
    3: "tree",
    4: "car",
    5: "clutter_background",
}
IGNORE_INDEX = 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-files", type=int, default=0, help="Smoke-only cap; 0 scans every file.")
    parser.add_argument("--expected-size", type=int, default=512)
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def split_for_id(sample_id: str, split_sets: dict[str, set[str]]) -> str:
    matches = [name for name, ids in split_sets.items() if sample_id in ids]
    return matches[0] if len(matches) == 1 else "invalid"


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    modality_dirs = {name: root / name for name in ("images", "dsm", "masks")}
    file_maps = {
        name: {path.stem: path for path in sorted(directory.glob("*.png"))}
        for name, directory in modality_dirs.items()
    }
    id_sets = {name: set(files) for name, files in file_maps.items()}
    if len({frozenset(ids) for ids in id_sets.values()}) != 1:
        raise RuntimeError({name: len(ids) for name, ids in id_sets.items()})

    split_ids = {
        name: read_ids(root / "splits" / f"{name}.txt")
        for name in ("train", "val", "test")
    }
    split_sets = {name: set(ids) for name, ids in split_ids.items()}
    if any(split_sets[a] & split_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("Split overlap detected")
    if set().union(*split_sets.values()) != id_sets["images"]:
        raise RuntimeError("Split union does not equal the prepared file-id set")

    all_ids = sorted(id_sets["images"])
    scan_ids = all_ids[: args.max_files] if args.max_files > 0 else all_ids
    class_counts: Counter[int] = Counter()
    content_locations: dict[str, dict[str, list[str]]] = {
        name: defaultdict(list) for name in modality_dirs
    }
    rows: list[dict[str, object]] = []
    failures: list[str] = []

    for sample_id in scan_ids:
        split = split_for_id(sample_id, split_sets)
        row: dict[str, object] = {"sample_id": sample_id, "split": split}
        for modality, files in file_maps.items():
            path = files[sample_id]
            try:
                with Image.open(path) as image:
                    array = np.asarray(image)
                height, width = array.shape[:2]
                row[f"{modality}_shape"] = "x".join(str(value) for value in array.shape)
                row[f"{modality}_dtype"] = str(array.dtype)
                row[f"{modality}_min"] = int(array.min())
                row[f"{modality}_max"] = int(array.max())
                digest = sha256_file(path)
                row[f"{modality}_sha256"] = digest
                content_locations[modality][digest].append(f"{split}:{sample_id}")
                if (height, width) != (args.expected_size, args.expected_size):
                    failures.append(f"size:{modality}:{sample_id}:{height}x{width}")
                if array.dtype != np.uint8:
                    failures.append(f"dtype:{modality}:{sample_id}:{array.dtype}")
                if modality == "images" and (array.ndim != 3 or array.shape[2] != 3):
                    failures.append(f"channels:{modality}:{sample_id}:{array.shape}")
                if modality in {"dsm", "masks"} and array.ndim != 2:
                    failures.append(f"channels:{modality}:{sample_id}:{array.shape}")
                if modality == "masks":
                    values, counts = np.unique(array, return_counts=True)
                    class_counts.update({int(value): int(count) for value, count in zip(values, counts)})
            except Exception as exc:
                failures.append(f"decode:{modality}:{sample_id}:{type(exc).__name__}")
        rows.append(row)

    allowed_classes = set(CLASS_MAP) | {IGNORE_INDEX}
    unexpected_classes = sorted(set(class_counts) - allowed_classes)
    missing_classes = sorted(set(CLASS_MAP) - set(class_counts))
    if unexpected_classes:
        failures.append(f"unexpected_mask_classes:{unexpected_classes}")
    if args.max_files == 0 and missing_classes:
        failures.append(f"missing_mask_classes:{missing_classes}")

    duplicate_summary = {}
    duplicate_rows = []
    cross_split_image_duplicates = 0
    for modality, groups in content_locations.items():
        duplicates = {digest: ids for digest, ids in groups.items() if len(ids) > 1}
        cross_split = [
            ids for ids in duplicates.values() if len({item.split(":", 1)[0] for item in ids}) > 1
        ]
        duplicate_summary[modality] = {
            "duplicate_hash_groups": len(duplicates),
            "cross_split_duplicate_hash_groups": len(cross_split),
        }
        for digest, ids in sorted(duplicates.items()):
            duplicate_rows.append(
                {
                    "modality": modality,
                    "sha256": digest,
                    "locations": ";".join(sorted(ids)),
                    "location_count": len(ids),
                    "cross_split": len({item.split(":", 1)[0] for item in ids}) > 1,
                }
            )
        if modality == "images":
            cross_split_image_duplicates = len(cross_split)
    if cross_split_image_duplicates:
        failures.append(f"cross_split_image_duplicates:{cross_split_image_duplicates}")

    split_sha256 = {name: sha256_file(root / "splits" / f"{name}.txt") for name in split_ids}
    file_set_sha256 = {
        name: hashlib.sha256("\n".join(sorted(files)).encode("utf-8")).hexdigest().upper()
        for name, files in file_maps.items()
    }
    dataset_manifest = {
        "schema": "qalf-dataset-manifest-v1",
        "root": str(root),
        "expected_size": args.expected_size,
        "file_counts": {name: len(files) for name, files in file_maps.items()},
        "file_set_sha256": file_set_sha256,
        "split_counts": {name: len(ids) for name, ids in split_ids.items()},
        "split_sha256": split_sha256,
    }
    class_map_manifest = {
        "schema": "qalf-class-map-v1",
        "class_map": CLASS_MAP,
        "ignore_index": IGNORE_INDEX,
        "class_pixel_counts": dict(sorted(class_counts.items())),
    }
    (output / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")
    (output / "class_map_manifest.json").write_text(
        json.dumps(class_map_manifest, indent=2), encoding="utf-8"
    )
    duplicate_path = output / "duplicate_report.csv"
    with duplicate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["modality", "sha256", "locations", "location_count", "cross_split"]
        )
        writer.writeheader()
        writer.writerows(duplicate_rows)

    report = {
        "schema": "qalf-dataset-audit-v1",
        "root": str(root),
        "formal_full_scan": args.max_files == 0,
        "file_counts": {name: len(files) for name, files in file_maps.items()},
        "split_counts": {name: len(ids) for name, ids in split_ids.items()},
        "split_sha256": split_sha256,
        "file_set_sha256": file_set_sha256,
        "scanned_ids": len(scan_ids),
        "class_map": CLASS_MAP,
        "ignore_index": IGNORE_INDEX,
        "class_pixel_counts": dict(sorted(class_counts.items())),
        "duplicate_summary": duplicate_summary,
        "artifacts": {
            "dataset_manifest": "dataset_manifest.json",
            "class_map_manifest": "class_map_manifest.json",
            "duplicate_report": "duplicate_report.csv",
            "file_audit": "dataset_file_audit.csv",
        },
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    csv_path = output / "dataset_file_audit.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report["artifact_sha256"] = {
        name: sha256_file(output / filename) for name, filename in report["artifacts"].items()
    }
    report_path = output / "dataset_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "scanned": len(scan_ids), "report": str(report_path)}))
    if failures:
        raise RuntimeError(f"Dataset audit failed with {len(failures)} issue(s)")


if __name__ == "__main__":
    main()
