#!/usr/bin/env python3
"""Create audited historical or result-blind confirmatory split roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


DEFAULTS = {
    "vaihingen": {
        "historical": ("13",),
        "confirmatory": ("32", "37"),
    },
    "potsdam": {
        "historical": ("7-10", "7-11"),
        "confirmatory": ("3-11", "6-8"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DEFAULTS), required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=("confirmatory", "historical"), default="confirmatory")
    parser.add_argument("--salt", default="qalf-confirmatory-v1")
    parser.add_argument("--target-fraction", type=float, default=0.10)
    parser.add_argument("--link-mode", choices=("symlink", "none"), default="symlink")
    parser.add_argument("--allow-selection-drift", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def tile_id(sample_id: str, split_prefix: str) -> str:
    prefix = f"{split_prefix}_"
    if not sample_id.startswith(prefix):
        raise ValueError(f"Unexpected {split_prefix} sample id: {sample_id}")
    remainder = sample_id[len(prefix) :]
    if "_" not in remainder:
        raise ValueError(f"Cannot parse tile id: {sample_id}")
    return remainder.split("_", 1)[0]


def sha256_text(lines: list[str]) -> str:
    payload = "".join(f"{line}\n" for line in lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def ranking_key(salt: str, dataset: str, tile: str) -> str:
    value = f"{salt}|{dataset.title()}|{tile}".encode("utf-8")
    return hashlib.sha256(value).hexdigest().upper()


def select_confirmatory_tiles(
    dataset: str,
    train_ids: list[str],
    salt: str,
    target_fraction: float,
) -> tuple[list[str], list[dict[str, object]]]:
    counts = Counter(tile_id(sample_id, "train") for sample_id in train_ids)
    excluded = set(DEFAULTS[dataset]["historical"])
    candidates = sorted(
        (tile for tile in counts if tile not in excluded),
        key=lambda tile: ranking_key(salt, dataset, tile),
    )
    if not candidates:
        raise RuntimeError("No confirmatory tile candidates remain after historical exclusions")

    target = len(train_ids) * target_fraction
    cumulative = 0
    choices: list[tuple[float, int, list[str]]] = []
    ranking: list[dict[str, object]] = []
    for rank, tile in enumerate(candidates, start=1):
        cumulative += counts[tile]
        selected = candidates[:rank]
        choices.append((abs(cumulative - target), rank, selected))
        ranking.append(
            {
                "rank": rank,
                "tile": tile,
                "sha256_key": ranking_key(salt, dataset, tile),
                "patches": counts[tile],
                "cumulative_patches": cumulative,
            }
        )
    return min(choices, key=lambda item: (item[0], item[1]))[2], ranking


def write_split(path: Path, ids: list[str]) -> None:
    path.write_text("".join(f"{sample_id}\n" for sample_id in ids), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.target_fraction < 0.5:
        raise ValueError("target-fraction must be between 0 and 0.5")

    source = Path(args.source_root).resolve()
    output = Path(args.output_root).resolve()
    split_dir = source / "splits"
    train_ids = read_ids(split_dir / "train.txt")
    val_ids = read_ids(split_dir / "val.txt")
    source_test_ids = read_ids(split_dir / "test.txt")
    if source_test_ids:
        raise RuntimeError("The prepared-v4 source test split must be empty")

    ranking: list[dict[str, object]] = []
    if args.mode == "historical":
        selected_tiles = list(DEFAULTS[args.dataset]["historical"])
    else:
        selected_tiles, ranking = select_confirmatory_tiles(
            args.dataset,
            train_ids,
            args.salt,
            args.target_fraction,
        )
        expected = list(DEFAULTS[args.dataset]["confirmatory"])
        if selected_tiles != expected and not args.allow_selection_drift:
            raise RuntimeError(f"Protocol selection drift: expected={expected} actual={selected_tiles}")

    selected = set(selected_tiles)
    test_ids = [sample_id for sample_id in train_ids if tile_id(sample_id, "train") in selected]
    new_train_ids = [sample_id for sample_id in train_ids if tile_id(sample_id, "train") not in selected]
    split_sets = [set(new_train_ids), set(val_ids), set(test_ids)]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Generated train/val/test splits overlap")
    if set(new_train_ids) | set(test_ids) != set(train_ids):
        raise RuntimeError("Generated split union does not recover the source train split")

    manifest = {
        "protocol_version": "qalf-confirmatory-v1" if args.mode == "confirmatory" else "qalf-historical-v1",
        "mode": args.mode,
        "dataset": args.dataset,
        "source_root": str(source),
        "output_root": str(output),
        "selection_uses_image_or_label_content": False,
        "ranking_key_format": "SHA256(<salt>|<DatasetTitle>|<tile_id>)",
        "dataset_token": args.dataset.title(),
        "salt": args.salt if args.mode == "confirmatory" else None,
        "target_fraction": args.target_fraction if args.mode == "confirmatory" else None,
        "historical_tiles_excluded_from_confirmatory_candidates": list(DEFAULTS[args.dataset]["historical"]),
        "selected_tiles": selected_tiles,
        "counts": {"train": len(new_train_ids), "val": len(val_ids), "test": len(test_ids)},
        "split_sha256": {
            "train": sha256_text(new_train_ids),
            "val": sha256_text(val_ids),
            "test": sha256_text(test_ids),
        },
        "source_split_sha256": {
            "train": sha256_text(train_ids),
            "val": sha256_text(val_ids),
            "test": sha256_text(source_test_ids),
        },
        "tile_ranking": ranking,
    }

    if args.dry_run:
        print(json.dumps({"dataset": args.dataset, "mode": args.mode, "selected_tiles": selected_tiles, "counts": manifest["counts"]}))
        return
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing split root: {output}")

    output.mkdir(parents=True)
    (output / "splits").mkdir()
    write_split(output / "splits" / "train.txt", new_train_ids)
    write_split(output / "splits" / "val.txt", val_ids)
    write_split(output / "splits" / "test.txt", test_ids)
    if args.link_mode == "symlink":
        for directory in ("images", "dsm", "masks"):
            target = source / directory
            if not target.is_dir():
                raise FileNotFoundError(target)
            relative_target = os.path.relpath(target, output)
            (output / directory).symlink_to(relative_target, target_is_directory=True)
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "selected_tiles": selected_tiles, "counts": manifest["counts"]}))


if __name__ == "__main__":
    main()
