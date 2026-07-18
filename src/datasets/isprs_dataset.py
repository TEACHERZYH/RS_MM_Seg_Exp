from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ISPRSMultimodalDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split_file: str,
        image_dir: str,
        aux_dir: str,
        mask_dir: str,
        image_suffix: str,
        aux_suffix: str,
        mask_suffix: str,
        input_size: int,
        missing_prob: float = 0.0,
        degradation_prob: float = 0.0,
        degradation_noise_std: float = 12.0,
        degradation_blur_kernel: int = 5,
        degradation_lowres_scale: int = 4,
        degradation_mask_min_fraction: float = 0.125,
        degradation_mask_max_fraction: float = 1.0 / 3.0,
        degradation_mask_position: str = "legacy_half",
        normalize_aux: bool = True,
        training: bool = True,
        enable_missing: bool | None = None,
        enable_degradation: bool | None = None,
        augment: bool = False,
        hflip_prob: float = 0.0,
        vflip_prob: float = 0.0,
        rotate90_prob: float = 0.0,
        color_jitter: float = 0.0,
        missing_target: str = "aux",
        augmentation_seed: int = 0,
        corruption_seed: int = 0,
        corruption_manifest: str | None = None,
        corruption_scenario: str | None = None,
        corruption_trial: int | None = None,
        return_pre_availability_inputs: bool = False,
        sample_ids_override: Sequence[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / image_dir
        self.aux_dir = self.root_dir / aux_dir
        self.mask_dir = self.root_dir / mask_dir
        self.split_file = self.root_dir / split_file
        self.image_suffix = image_suffix
        self.aux_suffix = aux_suffix
        self.mask_suffix = mask_suffix
        self.input_size = input_size
        self.missing_prob = missing_prob
        self.degradation_prob = degradation_prob
        self.degradation_noise_std = float(degradation_noise_std)
        self.degradation_blur_kernel = int(degradation_blur_kernel)
        self.degradation_lowres_scale = int(degradation_lowres_scale)
        self.degradation_mask_min_fraction = float(degradation_mask_min_fraction)
        self.degradation_mask_max_fraction = float(degradation_mask_max_fraction)
        self.degradation_mask_position = degradation_mask_position
        if self.degradation_noise_std < 0:
            raise ValueError("degradation_noise_std must be non-negative")
        if self.degradation_blur_kernel <= 0 or self.degradation_blur_kernel % 2 == 0:
            raise ValueError("degradation_blur_kernel must be a positive odd integer")
        if self.degradation_lowres_scale < 1:
            raise ValueError("degradation_lowres_scale must be at least 1")
        if not 0 < self.degradation_mask_min_fraction < self.degradation_mask_max_fraction <= 1:
            raise ValueError("Invalid degradation mask fraction interval")
        if self.degradation_mask_position not in {"legacy_half", "uniform_valid"}:
            raise ValueError(f"Unsupported degradation_mask_position: {self.degradation_mask_position}")
        self.normalize_aux = normalize_aux
        self.training = training
        self.enable_missing = training if enable_missing is None else enable_missing
        self.enable_degradation = training if enable_degradation is None else enable_degradation
        self.augment = augment and training
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate90_prob = rotate90_prob
        if missing_target not in {"aux", "main"}:
            raise ValueError(f"Unsupported missing_target: {missing_target}")
        self.missing_target = missing_target
        self.color_jitter = color_jitter
        self.augmentation_seed = int(augmentation_seed)
        self.corruption_seed = int(corruption_seed)
        self.return_pre_availability_inputs = bool(return_pre_availability_inputs)
        self._reset_rngs(worker_seed=0)
        split_ids = self._read_split()
        if sample_ids_override is None:
            self.sample_ids = split_ids
        else:
            self.sample_ids = [str(sample_id) for sample_id in sample_ids_override]
            missing_ids = sorted(set(self.sample_ids) - set(split_ids))
            if missing_ids or len(self.sample_ids) != len(set(self.sample_ids)):
                raise ValueError(
                    f"Invalid sample_ids_override: missing={len(missing_ids)} duplicates="
                    f"{len(self.sample_ids) - len(set(self.sample_ids))}"
                )
        self.corruption_records = self._read_corruption_manifest(
            corruption_manifest,
            corruption_scenario,
            corruption_trial,
        )

    def _reset_rngs(self, worker_seed: int) -> None:
        self._augmentation_rng = np.random.default_rng(
            np.random.SeedSequence([self.augmentation_seed, int(worker_seed)])
        )
        self._corruption_rng = np.random.default_rng(
            np.random.SeedSequence([self.corruption_seed, int(worker_seed)])
        )

    def set_worker_seed(self, worker_seed: int) -> None:
        self._reset_rngs(worker_seed)

    def _read_corruption_manifest(
        self,
        manifest_path: str | None,
        scenario: str | None,
        trial: int | None,
    ) -> dict[str, dict[str, str]] | None:
        if manifest_path is None:
            return None
        if scenario is None or trial is None:
            raise ValueError("corruption_scenario and corruption_trial are required with a manifest")

        records: dict[str, dict[str, str]] = {}
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("scenario") != scenario or int(row.get("trial", -1)) != int(trial):
                    continue
                sample_id = row["sample_id"]
                if sample_id in records:
                    raise ValueError(f"Duplicate corruption record for {scenario}/{trial}/{sample_id}")
                records[sample_id] = row
        missing = sorted(set(self.sample_ids) - set(records))
        extra = sorted(set(records) - set(self.sample_ids))
        if missing or extra:
            raise ValueError(
                f"Corruption manifest mismatch for {scenario}/{trial}: missing={len(missing)} extra={len(extra)}"
            )
        return records

    def _read_split(self) -> list[str]:
        with open(self.split_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_rgb(self, path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        return np.array(image)

    def _load_aux(self, path: Path) -> np.ndarray:
        aux = Image.open(path).convert("L")
        return np.array(aux)

    def _load_mask(self, path: Path) -> np.ndarray:
        mask = Image.open(path)
        return np.array(mask)

    def _resize(self, arr: np.ndarray, is_mask: bool = False) -> np.ndarray:
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.resize(arr, (self.input_size, self.input_size), interpolation=interpolation)

    def _apply_missing(
        self,
        image: np.ndarray,
        aux: np.ndarray,
        sample_id: str,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        main_available = 1.0
        aux_available = 1.0
        if self.corruption_records is not None:
            record = self.corruption_records[sample_id]
            main_available = float(record["main_available"])
            aux_available = float(record["aux_available"])
            if main_available == 0.0:
                image = np.zeros_like(image)
            if aux_available == 0.0:
                aux = np.zeros_like(aux)
            return image, aux, main_available, aux_available

        if self.enable_missing and self._corruption_rng.random() < self.missing_prob:
            if self.missing_target == "main":
                image = np.zeros_like(image)
                main_available = 0.0
            else:
                aux = np.zeros_like(aux)
                aux_available = 0.0
        return image, aux, main_available, aux_available

    def _apply_degradation(
        self,
        image: np.ndarray,
        aux: np.ndarray,
        sample_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        record = self.corruption_records[sample_id] if self.corruption_records is not None else None
        if record is not None:
            mode = record["degradation_type"]
            if mode == "none":
                return image, aux
        else:
            if not self.enable_degradation or self._corruption_rng.random() >= self.degradation_prob:
                return image, aux
            mode = str(self._corruption_rng.choice(["noise", "blur", "mask", "lowres"]))

        if mode == "noise":
            noise_std = float(record["noise_std"]) if record is not None else self.degradation_noise_std
            noise_rng = (
                np.random.default_rng(int(record["noise_seed"]))
                if record is not None
                else self._corruption_rng
            )
            noise = noise_rng.normal(0, noise_std, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif mode == "blur":
            kernel = int(record["blur_kernel"]) if record is not None else self.degradation_blur_kernel
            image = cv2.GaussianBlur(image, (kernel, kernel), 0)
            aux = cv2.GaussianBlur(aux, (kernel, kernel), 0)
        elif mode == "mask":
            h, w = aux.shape[:2]
            if record is not None:
                x0 = int(record["mask_x0"])
                y0 = int(record["mask_y0"])
                bw = int(record["mask_width"])
                bh = int(record["mask_height"])
            else:
                min_w = max(int(w * self.degradation_mask_min_fraction), 1)
                max_w = max(int(w * self.degradation_mask_max_fraction), min_w + 1)
                min_h = max(int(h * self.degradation_mask_min_fraction), 1)
                max_h = max(int(h * self.degradation_mask_max_fraction), min_h + 1)
                bw = int(self._corruption_rng.integers(min_w, min(max_w, w + 1)))
                bh = int(self._corruption_rng.integers(min_h, min(max_h, h + 1)))
                if self.degradation_mask_position == "uniform_valid":
                    x0 = int(self._corruption_rng.integers(0, max(w - bw + 1, 1)))
                    y0 = int(self._corruption_rng.integers(0, max(h - bh + 1, 1)))
                else:
                    x0 = int(self._corruption_rng.integers(0, max(w // 2, 1)))
                    y0 = int(self._corruption_rng.integers(0, max(h // 2, 1)))
            image[y0 : y0 + bh, x0 : x0 + bw] = 0
            aux[y0 : y0 + bh, x0 : x0 + bw] = 0
        elif mode == "lowres":
            scale = int(record["lowres_scale"]) if record is not None else self.degradation_lowres_scale
            small = cv2.resize(image, (self.input_size // scale, self.input_size // scale))
            image = cv2.resize(small, (self.input_size, self.input_size))
        else:
            raise ValueError(f"Unsupported degradation type in manifest: {mode}")
        return image, aux

    def _apply_spatial_augment(
        self,
        image: np.ndarray,
        aux: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.augment:
            return image, aux, mask

        if self._augmentation_rng.random() < self.hflip_prob:
            image = np.flip(image, axis=1)
            aux = np.flip(aux, axis=1)
            mask = np.flip(mask, axis=1)
        if self._augmentation_rng.random() < self.vflip_prob:
            image = np.flip(image, axis=0)
            aux = np.flip(aux, axis=0)
            mask = np.flip(mask, axis=0)
        if self._augmentation_rng.random() < self.rotate90_prob:
            k = int(self._augmentation_rng.integers(1, 4))
            image = np.rot90(image, k, axes=(0, 1))
            aux = np.rot90(aux, k, axes=(0, 1))
            mask = np.rot90(mask, k, axes=(0, 1))

        return image.copy(), aux.copy(), mask.copy()

    def _apply_color_jitter(self, image: np.ndarray) -> np.ndarray:
        if not self.augment or self.color_jitter <= 0.0:
            return image

        image_f = image.astype(np.float32)
        contrast = 1.0 + self._augmentation_rng.uniform(-self.color_jitter, self.color_jitter)
        brightness = self._augmentation_rng.uniform(-255.0 * self.color_jitter, 255.0 * self.color_jitter)
        mean = image_f.mean(axis=(0, 1), keepdims=True)
        image_f = (image_f - mean) * contrast + mean + brightness
        return np.clip(image_f, 0, 255).astype(np.uint8)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self.sample_ids[index]
        image = self._load_rgb(self.image_dir / f"{sample_id}{self.image_suffix}")
        aux = self._load_aux(self.aux_dir / f"{sample_id}{self.aux_suffix}")
        mask = self._load_mask(self.mask_dir / f"{sample_id}{self.mask_suffix}")

        image = self._resize(image)
        aux = self._resize(aux)
        mask = self._resize(mask, is_mask=True)

        image, aux, mask = self._apply_spatial_augment(image, aux, mask)
        image = self._apply_color_jitter(image)

        teacher_image = image.copy()
        teacher_aux = aux.copy()

        image, aux = self._apply_degradation(image, aux, sample_id)
        pre_availability_image = image.copy()
        pre_availability_aux = aux.copy()
        image, aux, main_available, aux_available = self._apply_missing(image, aux, sample_id)

        image = image.astype(np.float32) / 255.0
        teacher_image = teacher_image.astype(np.float32) / 255.0
        image_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        image_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - image_mean) / image_std
        teacher_image = (teacher_image - image_mean) / image_std
        pre_availability_image = pre_availability_image.astype(np.float32) / 255.0
        pre_availability_image = (pre_availability_image - image_mean) / image_std

        aux = aux.astype(np.float32)
        teacher_aux = teacher_aux.astype(np.float32)
        if self.normalize_aux:
            aux = aux / max(aux.max(), 1.0)
            teacher_aux = teacher_aux / max(teacher_aux.max(), 1.0)
            pre_availability_aux = pre_availability_aux.astype(np.float32)
            pre_availability_aux = pre_availability_aux / max(pre_availability_aux.max(), 1.0)
        else:
            pre_availability_aux = pre_availability_aux.astype(np.float32)
        aux = aux[None, ...]
        teacher_aux = teacher_aux[None, ...]

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        aux_tensor = torch.from_numpy(aux).float()
        teacher_image_tensor = torch.from_numpy(teacher_image.transpose(2, 0, 1)).float()
        teacher_aux_tensor = torch.from_numpy(teacher_aux).float()
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        result = {
            "image": image_tensor,
            "aux": aux_tensor,
            "teacher_image": teacher_image_tensor,
            "teacher_aux": teacher_aux_tensor,
            "mask": mask_tensor,
            "sample_id": sample_id,
            "aux_available": torch.tensor(aux_available, dtype=torch.float32),
            "main_available": torch.tensor(main_available, dtype=torch.float32),
        }
        if self.return_pre_availability_inputs:
            result["pre_availability_image"] = torch.from_numpy(
                pre_availability_image.transpose(2, 0, 1)
            ).float()
            result["pre_availability_aux"] = torch.from_numpy(pre_availability_aux[None, ...]).float()
        if self.corruption_records is not None:
            record = self.corruption_records[sample_id]
            result["corruption_pair_id"] = record.get("corruption_pair_id", "")
            result["corruption_payload_sha256"] = record.get("payload_sha256", "")
        return result
