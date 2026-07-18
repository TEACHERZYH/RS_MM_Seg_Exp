#!/usr/bin/env python3
"""Frozen severity-case and corruption primitives for the QALF M2 campaign."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LOWRES_FACTORS = (2, 3, 4, 6, 8)
MASK_FRACTIONS = (0.135, 0.190, 0.245, 0.300, 0.355)
CORRUPTIONS = ("noise", "blur", "mask", "lowres")


@dataclass(frozen=True)
class EvalCase:
    scenario: str
    severity: int
    corruption: str
    trial: int
    missing_aux: bool


def build_m2_cases(include_combined: bool = True) -> list[EvalCase]:
    cases: list[EvalCase] = []
    states = (("degraded", False), ("missing_aux_and_degraded", True)) if include_combined else (("degraded", False),)
    for scenario, missing_aux in states:
        cases.append(EvalCase(scenario, 0, "none", 0, missing_aux))
        for severity in range(1, 6):
            for corruption in CORRUPTIONS:
                for trial in range(3):
                    cases.append(EvalCase(scenario, severity, corruption, trial, missing_aux))
    return cases


def case_seed(dataset_key: str, sample_id: str, case: EvalCase) -> int:
    payload = "|".join(
        (
            "qalf-severity-m2",
            dataset_key,
            sample_id,
            case.scenario,
            case.corruption,
            str(case.severity),
            str(case.trial),
        )
    ).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**63 - 1)


def mask_box(height: int, width: int, severity: int, seed: int) -> tuple[int, int, int, int]:
    if severity < 1 or severity > 5:
        raise ValueError(f"Mask severity must be 1..5, found {severity}")
    fraction = MASK_FRACTIONS[severity - 1]
    box_h = min(max(int(round(height * fraction)), 1), height)
    box_w = min(max(int(round(width * fraction)), 1), width)
    rng = np.random.default_rng(seed)
    y0 = int(rng.integers(0, height - box_h + 1))
    x0 = int(rng.integers(0, width - box_w + 1))
    return x0, y0, box_w, box_h


def _denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(device=image.device, dtype=image.dtype)
    std = STD.to(device=image.device, dtype=image.dtype)
    return torch.clamp(image * std + mean, 0.0, 1.0)


def _normalize(image: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(device=image.device, dtype=image.dtype)
    std = STD.to(device=image.device, dtype=image.dtype)
    return (torch.clamp(image, 0.0, 1.0) - mean) / std


def apply_m2_case(
    image: torch.Tensor,
    aux: torch.Tensor,
    sample_ids: Sequence[str],
    dataset_key: str,
    case: EvalCase,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(sample_ids) != image.shape[0]:
        raise ValueError("sample_ids length does not match batch size")
    if case.severity == 0:
        if case.corruption != "none" or case.trial != 0:
            raise ValueError("Severity level 0 must be the single identity case")
        return image, aux
    if case.severity < 1 or case.severity > 5 or case.corruption not in CORRUPTIONS:
        raise ValueError(f"Invalid M2 severity case: {case}")

    output_image = image.clone()
    output_aux = aux.clone()
    for index, sample_id in enumerate(sample_ids):
        seed = case_seed(dataset_key, str(sample_id), case)
        item_image = output_image[index : index + 1]
        item_aux = output_aux[index : index + 1]
        if case.corruption == "noise":
            generator = torch.Generator(device=item_image.device)
            generator.manual_seed(seed)
            image01 = _denormalize(item_image)
            noise = torch.randn(
                image01.shape,
                generator=generator,
                device=image01.device,
                dtype=image01.dtype,
            )
            output_image[index : index + 1] = _normalize(image01 + noise * (0.025 * case.severity))
        elif case.corruption == "blur":
            kernel = 2 * case.severity + 1
            output_image[index : index + 1] = F.avg_pool2d(
                item_image, kernel_size=kernel, stride=1, padding=kernel // 2
            )
            output_aux[index : index + 1] = F.avg_pool2d(
                item_aux, kernel_size=kernel, stride=1, padding=kernel // 2
            )
        elif case.corruption == "lowres":
            _, _, height, width = item_image.shape
            factor = LOWRES_FACTORS[case.severity - 1]
            small = F.interpolate(
                item_image,
                size=(max(height // factor, 4), max(width // factor, 4)),
                mode="bilinear",
                align_corners=False,
            )
            output_image[index : index + 1] = F.interpolate(
                small, size=(height, width), mode="bilinear", align_corners=False
            )
        else:
            _, _, height, width = item_image.shape
            x0, y0, box_w, box_h = mask_box(height, width, case.severity, seed)
            output_image[index, :, y0 : y0 + box_h, x0 : x0 + box_w] = 0.0
            output_aux[index, :, y0 : y0 + box_h, x0 : x0 + box_w] = 0.0
    return output_image, output_aux


def assert_m2_case_contract(cases: Sequence[EvalCase]) -> None:
    expected = 2 * (1 + 5 * 4 * 3)
    if len(cases) != expected:
        raise RuntimeError(f"M2 severity case count drift: expected={expected} actual={len(cases)}")
    for scenario in ("degraded", "missing_aux_and_degraded"):
        identity = [case for case in cases if case.scenario == scenario and case.severity == 0]
        if identity != [EvalCase(scenario, 0, "none", 0, scenario.startswith("missing_aux"))]:
            raise RuntimeError(f"Invalid singleton identity case: {scenario}")
