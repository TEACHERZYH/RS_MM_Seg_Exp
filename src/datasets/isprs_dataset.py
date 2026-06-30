from __future__ import annotations

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
        self.sample_ids = self._read_split()

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

    def _apply_missing(self, image: np.ndarray, aux: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        main_available = 1.0
        aux_available = 1.0
        if self.enable_missing and np.random.rand() < self.missing_prob:
            if self.missing_target == "main":
                image = np.zeros_like(image)
                main_available = 0.0
            else:
                aux = np.zeros_like(aux)
                aux_available = 0.0
        return image, aux, main_available, aux_available

    def _apply_degradation(self, image: np.ndarray, aux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.enable_degradation or np.random.rand() >= self.degradation_prob:
            return image, aux

        mode = np.random.choice(["noise", "blur", "mask", "lowres"])
        if mode == "noise":
            noise = np.random.normal(0, 12, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        elif mode == "blur":
            image = cv2.GaussianBlur(image, (5, 5), 0)
            aux = cv2.GaussianBlur(aux, (5, 5), 0)
        elif mode == "mask":
            h, w = aux.shape[:2]
            x0 = np.random.randint(0, w // 2)
            y0 = np.random.randint(0, h // 2)
            bw = np.random.randint(w // 8, w // 3)
            bh = np.random.randint(h // 8, h // 3)
            image[y0 : y0 + bh, x0 : x0 + bw] = 0
            aux[y0 : y0 + bh, x0 : x0 + bw] = 0
        elif mode == "lowres":
            scale = 4
            small = cv2.resize(image, (self.input_size // scale, self.input_size // scale))
            image = cv2.resize(small, (self.input_size, self.input_size))
        return image, aux

    def _apply_spatial_augment(
        self,
        image: np.ndarray,
        aux: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.augment:
            return image, aux, mask

        if np.random.rand() < self.hflip_prob:
            image = np.flip(image, axis=1)
            aux = np.flip(aux, axis=1)
            mask = np.flip(mask, axis=1)
        if np.random.rand() < self.vflip_prob:
            image = np.flip(image, axis=0)
            aux = np.flip(aux, axis=0)
            mask = np.flip(mask, axis=0)
        if np.random.rand() < self.rotate90_prob:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k, axes=(0, 1))
            aux = np.rot90(aux, k, axes=(0, 1))
            mask = np.rot90(mask, k, axes=(0, 1))

        return image.copy(), aux.copy(), mask.copy()

    def _apply_color_jitter(self, image: np.ndarray) -> np.ndarray:
        if not self.augment or self.color_jitter <= 0.0:
            return image

        image_f = image.astype(np.float32)
        contrast = 1.0 + np.random.uniform(-self.color_jitter, self.color_jitter)
        brightness = np.random.uniform(-255.0 * self.color_jitter, 255.0 * self.color_jitter)
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

        image, aux, main_available, aux_available = self._apply_missing(image, aux)
        image, aux = self._apply_degradation(image, aux)

        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        teacher_image = teacher_image.astype(np.float32) / 255.0
        teacher_image = (teacher_image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )

        aux = aux.astype(np.float32)
        teacher_aux = teacher_aux.astype(np.float32)
        if self.normalize_aux:
            aux = aux / max(aux.max(), 1.0)
            teacher_aux = teacher_aux / max(teacher_aux.max(), 1.0)
        aux = aux[None, ...]
        teacher_aux = teacher_aux[None, ...]

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        aux_tensor = torch.from_numpy(aux).float()
        teacher_image_tensor = torch.from_numpy(teacher_image.transpose(2, 0, 1)).float()
        teacher_aux_tensor = torch.from_numpy(teacher_aux).float()
        mask_tensor = torch.from_numpy(mask.astype(np.int64))

        return {
            "image": image_tensor,
            "aux": aux_tensor,
            "teacher_image": teacher_image_tensor,
            "teacher_aux": teacher_aux_tensor,
            "mask": mask_tensor,
            "sample_id": sample_id,
            "aux_available": torch.tensor(aux_available, dtype=torch.float32),
            "main_available": torch.tensor(main_available, dtype=torch.float32),
        }
