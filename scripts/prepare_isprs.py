from __future__ import annotations

import argparse
import math
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm


POTSDAM_TRAIN_IDS = [
    "2_10", "2_11", "2_12", "3_10", "3_11", "3_12", "4_10", "4_11",
    "4_12", "5_10", "5_11", "5_12", "6_10", "6_11", "6_12", "6_7",
    "6_8", "6_9", "7_10", "7_11", "7_12", "7_7", "7_8", "7_9",
]
POTSDAM_TEST_IDS = [
    "5_15", "6_15", "6_13", "3_13", "4_14", "6_14", "5_14", "2_13",
    "4_15", "2_14", "5_13", "4_13", "3_14", "7_13",
]

VAIHINGEN_TRAIN_IDS = [
    "1", "11", "13", "15", "17", "21", "23", "26",
    "28", "3", "30", "32", "34", "37", "5", "7",
]
VAIHINGEN_TEST_IDS = [
    "6", "24", "35", "16", "14", "22", "10", "4",
    "2", "20", "8", "31", "33", "27", "38", "12", "29",
]

CLASS_COLORS = {
    (255, 255, 255): 0,  # impervious
    (0, 0, 255): 1,      # building
    (0, 255, 255): 2,    # low vegetation
    (0, 255, 0): 3,      # tree
    (255, 255, 0): 4,    # car
    (255, 0, 0): 5,      # clutter/background
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ISPRS Potsdam/Vaihingen for RS_MM_Seg_Exp")
    parser.add_argument("--dataset", choices=["potsdam", "vaihingen"], required=True)
    parser.add_argument("--raw-zip", type=str, required=True, help="Path to downloaded top-level zip")
    parser.add_argument("--output-root", type=str, required=True, help="Prepared dataset output dir")
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--train-stride", type=int, default=256)
    parser.add_argument("--eval-stride", type=int, default=512)
    parser.add_argument("--val-tiles", type=int, default=2, help="How many official train tiles to reserve as validation")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_zip(zip_path: Path, dst: Path) -> None:
    ensure_dir(dst)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)


def maybe_extract_nested_zip(path: Path) -> None:
    for nested_zip in path.rglob("*.zip"):
        extract_dir = nested_zip.with_suffix("")
        if extract_dir.exists():
            continue
        try:
            extract_zip(nested_zip, extract_dir)
        except zipfile.BadZipFile:
            continue


def load_raster(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        arr = tifffile.imread(str(path))
    else:
        arr = np.array(Image.open(path))
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] in {3, 4} and arr.shape[-1] not in {3, 4}:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def save_image(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(arr).save(path)


def rgb_mask_to_label(mask_rgb: np.ndarray) -> np.ndarray:
    if mask_rgb.ndim == 2:
        return mask_rgb.astype(np.uint8)

    label = np.full(mask_rgb.shape[:2], 255, dtype=np.uint8)
    for color, cls_idx in CLASS_COLORS.items():
        match = np.all(mask_rgb[:, :, :3] == np.array(color, dtype=np.uint8), axis=-1)
        label[match] = cls_idx
    return label


def normalize_aux(aux: np.ndarray) -> np.ndarray:
    aux = aux.astype(np.float32)
    min_val = np.nanmin(aux)
    max_val = np.nanmax(aux)
    aux = (aux - min_val) / max(max_val - min_val, 1e-6)
    aux = np.clip(aux * 255.0, 0, 255).astype(np.uint8)
    return aux


def sliding_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    if positions[-1] != length - patch_size:
        positions.append(length - patch_size)
    return positions


def find_first(base: Path, patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(base.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find any file for patterns: {patterns}")


def build_specs(dataset: str, val_tiles: int) -> dict:
    if dataset == "potsdam":
        def potsdam_id_variants(tid: str) -> list[str]:
            a, b = tid.split("_")
            return [tid, f"{int(a):02d}_{b}"]

        train_ids = POTSDAM_TRAIN_IDS.copy()
        val_ids = train_ids[-val_tiles:]
        train_ids = train_ids[:-val_tiles]
        return {
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": POTSDAM_TEST_IDS,
            "image_glob": lambda tid: [f"**/top_potsdam_{tid}_RGB.tif", f"**/top_potsdam_{tid}_IRRG.tif"],
            "aux_glob": lambda tid: [
                pattern
                for variant in potsdam_id_variants(tid)
                for pattern in [
                    f"**/*potsdam_{variant}_*normalized*.tif",
                    f"**/*potsdam_{variant}_*normalized*.jpg",
                    f"**/*{variant}*dsm*.tif",
                    f"**/*{variant}*dsm*.jpg",
                ]
            ],
            "mask_glob": lambda tid: [
                f"**/top_potsdam_{tid}_label_noBoundary.tif",
                f"**/top_potsdam_{tid}_label.tif",
            ],
        }

    train_ids = VAIHINGEN_TRAIN_IDS.copy()
    val_ids = train_ids[-val_tiles:]
    train_ids = train_ids[:-val_tiles]
    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": VAIHINGEN_TEST_IDS,
        "image_glob": lambda tid: [f"**/top_mosaic_09cm_area{tid}.tif"],
        "aux_glob": lambda tid: [
            f"**/dsm_09cm_matching_area{tid}.tif",
            f"**/*area{tid}*dsm*.tif",
            f"**/*area{tid}*ndsm*.tif",
        ],
        "mask_glob": lambda tid: [
            f"**/top_mosaic_09cm_area{tid}_noBoundary.tif",
            f"**/top_mosaic_09cm_area{tid}.tif",
        ],
    }


def write_split(split_path: Path, items: list[str]) -> None:
    with open(split_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def process_tile(
    image: np.ndarray,
    aux: np.ndarray,
    mask: np.ndarray,
    split_name: str,
    tile_id: str,
    output_root: Path,
    patch_size: int,
    stride: int,
) -> list[str]:
    h, w = mask.shape[:2]
    ys = sliding_positions(h, patch_size, stride)
    xs = sliding_positions(w, patch_size, stride)
    ids: list[str] = []

    for y in ys:
        for x in xs:
            image_patch = image[y : y + patch_size, x : x + patch_size]
            aux_patch = aux[y : y + patch_size, x : x + patch_size]
            mask_patch = mask[y : y + patch_size, x : x + patch_size]
            sample_id = f"{split_name}_{tile_id}_{y}_{x}"
            save_image(output_root / "images" / f"{sample_id}.png", image_patch)
            save_image(output_root / "dsm" / f"{sample_id}.png", aux_patch)
            save_image(output_root / "masks" / f"{sample_id}.png", mask_patch)
            ids.append(sample_id)
    return ids


def main() -> None:
    args = parse_args()
    raw_zip = Path(args.raw_zip)
    output_root = Path(args.output_root)

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)

    ensure_dir(output_root)
    ensure_dir(output_root / "images")
    ensure_dir(output_root / "dsm")
    ensure_dir(output_root / "masks")
    ensure_dir(output_root / "splits")

    extract_root = raw_zip.parent / f"{args.dataset}_extracted"
    if args.overwrite and extract_root.exists():
        shutil.rmtree(extract_root)

    if not extract_root.exists():
        print(f"Extracting top-level archive: {raw_zip}")
        extract_zip(raw_zip, extract_root)

    print("Extracting nested zip archives when present...")
    maybe_extract_nested_zip(extract_root)

    specs = build_specs(args.dataset, args.val_tiles)
    split_map = {
        "train": specs["train_ids"],
        "val": specs["val_ids"],
        "test": specs["test_ids"],
    }

    split_records: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for split_name, tile_ids in split_map.items():
        stride = args.train_stride if split_name == "train" else args.eval_stride
        for tile_id in tqdm(tile_ids, desc=f"prepare-{split_name}"):
            image_path = find_first(extract_root, specs["image_glob"](tile_id))
            aux_path = find_first(extract_root, specs["aux_glob"](tile_id))
            mask_path = find_first(extract_root, specs["mask_glob"](tile_id))

            image = load_raster(image_path)
            aux = normalize_aux(load_raster(aux_path))
            mask = rgb_mask_to_label(load_raster(mask_path))

            if image.ndim == 3 and image.shape[2] > 3:
                image = image[:, :, :3]
            if image.ndim == 2:
                image = np.repeat(image[:, :, None], repeats=3, axis=2)

            if aux.ndim == 3:
                aux = aux[:, :, 0]

            patch_ids = process_tile(
                image=image.astype(np.uint8),
                aux=aux.astype(np.uint8),
                mask=mask.astype(np.uint8),
                split_name=split_name,
                tile_id=tile_id.replace("_", "-"),
                output_root=output_root,
                patch_size=args.patch_size,
                stride=stride,
            )
            split_records[split_name].extend(patch_ids)

    write_split(output_root / "splits" / "train.txt", split_records["train"])
    write_split(output_root / "splits" / "val.txt", split_records["val"])
    write_split(output_root / "splits" / "test.txt", split_records["test"])

    print("Preparation completed.")
    for split_name, items in split_records.items():
        print(f"{split_name}: {len(items)} patches")


if __name__ == "__main__":
    main()
