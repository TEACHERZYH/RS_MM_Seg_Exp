from __future__ import annotations

import argparse
import io
import shutil
import zipfile
from pathlib import Path

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
    (255, 255, 255): 0,
    (0, 0, 255): 1,
    (0, 255, 255): 2,
    (0, 255, 0): 3,
    (255, 255, 0): 4,
    (255, 0, 0): 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ISPRS datasets directly from zip files")
    parser.add_argument("--dataset", choices=["potsdam", "vaihingen"], required=True)
    parser.add_argument("--raw-zip", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--train-stride", type=int, default=256)
    parser.add_argument("--eval-stride", type=int, default=512)
    parser.add_argument("--val-tiles", type=int, default=2)
    parser.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated subset of splits to prepare")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_image(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
    Image.fromarray(arr).save(str(path))


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
    return np.clip(aux * 255.0, 0, 255).astype(np.uint8)


def sliding_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    if positions[-1] != length - patch_size:
        positions.append(length - patch_size)
    return positions


def write_split(path: Path, items: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def read_split(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


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
            sample_id = f"{split_name}_{tile_id}_{y}_{x}"
            save_image(output_root / "images" / f"{sample_id}.png", image[y : y + patch_size, x : x + patch_size])
            save_image(output_root / "dsm" / f"{sample_id}.png", aux[y : y + patch_size, x : x + patch_size])
            save_image(output_root / "masks" / f"{sample_id}.png", mask[y : y + patch_size, x : x + patch_size])
            ids.append(sample_id)
    return ids


def load_array_from_member(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    with zf.open(member) as fp:
        data = fp.read()
    suffix = Path(member).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        arr = tifffile.imread(io.BytesIO(data))
    else:
        arr = np.array(Image.open(io.BytesIO(data)))
    if arr.ndim == 3 and arr.shape[0] in {3, 4} and arr.shape[-1] not in {3, 4}:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def prepare_potsdam(args: argparse.Namespace, output_root: Path) -> None:
    nested_root = Path(args.raw_zip).parent / "potsdam_extracted" / "Potsdam"
    rgb_zip = zipfile.ZipFile(nested_root / "2_Ortho_RGB.zip")
    dsm_zip = zipfile.ZipFile(nested_root / "1_DSM_normalisation.zip")
    label_zip = zipfile.ZipFile(nested_root / "5_Labels_all_noBoundary.zip")

    train_ids = POTSDAM_TRAIN_IDS.copy()
    val_ids = train_ids[-args.val_tiles :]
    train_ids = train_ids[:-args.val_tiles]
    split_map = {"train": train_ids, "val": val_ids, "test": POTSDAM_TEST_IDS}
    split_records: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    selected_splits = {item.strip() for item in args.splits.split(",") if item.strip()}

    for split_name, tile_ids in split_map.items():
        if split_name not in selected_splits:
            continue
        stride = args.train_stride if split_name == "train" else args.eval_stride
        for tile_id in tqdm(tile_ids, desc=f"prepare-{split_name}"):
            print(f"[potsdam] {split_name} tile {tile_id}", flush=True)
            a, b = tile_id.split("_")
            dsm_id = f"{int(a):02d}_{int(b):02d}"
            image_member = f"2_Ortho_RGB/top_potsdam_{tile_id}_RGB.tif"
            aux_member = f"1_DSM_normalisation/dsm_potsdam_{dsm_id}_normalized_lastools.jpg"
            if aux_member not in dsm_zip.namelist():
                aux_member = f"1_DSM_normalisation/dsm_potsdam_{dsm_id}_normalized_ownapproach.jpg"
            mask_member = f"top_potsdam_{tile_id}_label_noBoundary.tif"

            image = load_array_from_member(rgb_zip, image_member)
            aux = normalize_aux(load_array_from_member(dsm_zip, aux_member))
            mask = rgb_mask_to_label(load_array_from_member(label_zip, mask_member))
            image = image[:, :, :3].astype(np.uint8)
            if aux.ndim == 3:
                aux = aux[:, :, 0]

            split_records[split_name].extend(
                process_tile(image, aux.astype(np.uint8), mask.astype(np.uint8), split_name, tile_id.replace("_", "-"), output_root, args.patch_size, stride)
            )

    rgb_zip.close()
    dsm_zip.close()
    label_zip.close()
    for split_name in ["train", "val", "test"]:
        split_path = output_root / "splits" / f"{split_name}.txt"
        items = split_records[split_name] if split_name in selected_splits else read_split(split_path)
        write_split(split_path, items)


def ensure_vaihingen_semantic_zip(raw_zip: Path) -> Path:
    cache_zip = raw_zip.parent / "Vaihingen_semantic_labeling.zip"
    if cache_zip.exists() and cache_zip.stat().st_size > 0:
        return cache_zip

    with zipfile.ZipFile(raw_zip, "r") as outer, outer.open("Vaihingen/ISPRS_semantic_labeling_Vaihingen.zip") as src, open(cache_zip, "wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    return cache_zip


def prepare_vaihingen(args: argparse.Namespace, output_root: Path) -> None:
    semantic_zip_path = ensure_vaihingen_semantic_zip(Path(args.raw_zip))
    semantic_zip = zipfile.ZipFile(semantic_zip_path)

    train_ids = VAIHINGEN_TRAIN_IDS.copy()
    val_ids = train_ids[-args.val_tiles :]
    train_ids = train_ids[:-args.val_tiles]
    split_map = {"train": train_ids, "val": val_ids, "test": VAIHINGEN_TEST_IDS}
    split_records: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    selected_splits = {item.strip() for item in args.splits.split(",") if item.strip()}

    for split_name, tile_ids in split_map.items():
        if split_name not in selected_splits:
            continue
        stride = args.train_stride if split_name == "train" else args.eval_stride
        for tile_id in tqdm(tile_ids, desc=f"prepare-{split_name}"):
            print(f"[vaihingen] {split_name} tile {tile_id}", flush=True)
            image_member = f"top/top_mosaic_09cm_area{tile_id}.tif"
            aux_member = f"dsm/dsm_09cm_matching_area{tile_id}.tif"
            mask_member = f"gts_for_participants/top_mosaic_09cm_area{tile_id}.tif"

            image = load_array_from_member(semantic_zip, image_member)
            aux = normalize_aux(load_array_from_member(semantic_zip, aux_member))
            mask = rgb_mask_to_label(load_array_from_member(semantic_zip, mask_member))
            image = image[:, :, :3].astype(np.uint8)
            if aux.ndim == 3:
                aux = aux[:, :, 0]

            split_records[split_name].extend(
                process_tile(image, aux.astype(np.uint8), mask.astype(np.uint8), split_name, tile_id, output_root, args.patch_size, stride)
            )

    semantic_zip.close()
    for split_name in ["train", "val", "test"]:
        split_path = output_root / "splits" / f"{split_name}.txt"
        items = split_records[split_name] if split_name in selected_splits else read_split(split_path)
        write_split(split_path, items)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)

    ensure_dir(output_root / "images")
    ensure_dir(output_root / "dsm")
    ensure_dir(output_root / "masks")
    ensure_dir(output_root / "splits")

    if args.dataset == "potsdam":
        prepare_potsdam(args, output_root)
    else:
        prepare_vaihingen(args, output_root)

    print("Preparation completed.")
    for split_name in ["train", "val", "test"]:
        split_path = output_root / "splits" / f"{split_name}.txt"
        if not split_path.exists():
            print(f"{split_name}: 0 patches")
            continue
        with open(split_path, "r", encoding="utf-8") as f:
            count = sum(1 for _ in f)
        print(f"{split_name}: {count} patches")


if __name__ == "__main__":
    main()
