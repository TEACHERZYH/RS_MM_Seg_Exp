from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def make_dirs(root: Path) -> None:
    for sub in ["images", "dsm", "masks", "splits"]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def generate_sample(size: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    dsm = np.zeros((size, size), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    for cls in range(6):
        x0 = rng.integers(0, size // 2)
        y0 = rng.integers(0, size // 2)
        w = rng.integers(size // 8, size // 3)
        h = rng.integers(size // 8, size // 3)
        color = rng.integers(40, 220, size=3, dtype=np.uint8)
        height = int(rng.integers(10, 245))
        image[y0 : y0 + h, x0 : x0 + w] = color
        dsm[y0 : y0 + h, x0 : x0 + w] = height
        mask[y0 : y0 + h, x0 : x0 + w] = cls

    noise = rng.integers(0, 20, size=image.shape, dtype=np.uint8)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return image, dsm, mask


def write_split(path: Path, items: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/Vaihingen")
    parser.add_argument("--num-train", type=int, default=12)
    parser.add_argument("--num-val", type=int, default=4)
    parser.add_argument("--num-test", type=int, default=4)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    root = Path(args.root)
    make_dirs(root)

    all_ids: list[str] = []
    total = args.num_train + args.num_val + args.num_test
    for idx in range(total):
        sample_id = f"tile_{idx:03d}"
        image, dsm, mask = generate_sample(args.size, idx + 7)
        Image.fromarray(image).save(root / "images" / f"{sample_id}.png")
        Image.fromarray(dsm).save(root / "dsm" / f"{sample_id}.png")
        Image.fromarray(mask).save(root / "masks" / f"{sample_id}.png")
        all_ids.append(sample_id)

    train_ids = all_ids[: args.num_train]
    val_ids = all_ids[args.num_train : args.num_train + args.num_val]
    test_ids = all_ids[args.num_train + args.num_val :]

    write_split(root / "splits" / "train.txt", train_ids)
    write_split(root / "splits" / "val.txt", val_ids)
    write_split(root / "splits" / "test.txt", test_ids)

    print(f"Dummy dataset created at: {root}")


if __name__ == "__main__":
    main()
