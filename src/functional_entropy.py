from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import deterministic_cross_entropy


@dataclass(frozen=True)
class ProbeDecision:
    global_step: int
    block_index: int
    active_offset: int
    active: bool
    selected_batch_index: int | None
    reason: str


class BlockRandomProbeScheduler:
    """Independent checkpointable RNG for one random DN-MFE probe per step block."""

    def __init__(self, seed: int, interval: int = 4) -> None:
        if interval <= 0:
            raise ValueError("DN-MFE probe interval must be positive")
        self.seed = int(seed)
        self.interval = int(interval)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)
        self.current_block = -1
        self.active_offset = -1
        self.trace: list[dict[str, int | bool | str | None]] = []

    def decide(
        self,
        global_step: int,
        main_available: torch.Tensor,
        aux_available: torch.Tensor,
    ) -> ProbeDecision:
        global_step = int(global_step)
        if global_step < 0:
            raise ValueError("DN-MFE global step must be non-negative")
        block_index = global_step // self.interval
        if block_index < self.current_block:
            raise RuntimeError("DN-MFE probe scheduler received a non-monotonic global step")
        if block_index != self.current_block:
            if block_index != self.current_block + 1:
                raise RuntimeError("DN-MFE probe scheduler cannot skip unsampled step blocks")
            self.current_block = block_index
            self.active_offset = int(torch.randint(self.interval, (1,), generator=self.generator).item())
        active = global_step % self.interval == self.active_offset
        selected: int | None = None
        reason = "inactive_offset"
        if active:
            candidates = torch.nonzero(
                main_available.detach().gt(0.5).cpu() & aux_available.detach().gt(0.5).cpu(),
                as_tuple=False,
            ).flatten()
            if candidates.numel() == 0:
                reason = "no_coavailable_sample"
            else:
                position = int(torch.randint(int(candidates.numel()), (1,), generator=self.generator).item())
                selected = int(candidates[position].item())
                reason = "selected_uniform_coavailable"
        decision = ProbeDecision(
            global_step=global_step,
            block_index=block_index,
            active_offset=self.active_offset,
            active=active,
            selected_batch_index=selected,
            reason=reason,
        )
        self.trace.append(
            {
                "global_step": decision.global_step,
                "block_index": decision.block_index,
                "active_offset": decision.active_offset,
                "active": decision.active,
                "selected_batch_index": decision.selected_batch_index,
                "reason": decision.reason,
            }
        )
        return decision

    def state_dict(self) -> dict:
        return {
            "schema": "qalf-dn-mfe-probe-rng-v1",
            "seed": self.seed,
            "interval": self.interval,
            "generator_state": self.generator.get_state(),
            "current_block": self.current_block,
            "active_offset": self.active_offset,
            "trace": list(self.trace),
        }

    def load_state_dict(self, state: dict) -> None:
        if (
            state.get("schema") != "qalf-dn-mfe-probe-rng-v1"
            or int(state.get("seed", -1)) != self.seed
            or int(state.get("interval", -1)) != self.interval
        ):
            raise ValueError("DN-MFE probe scheduler checkpoint contract mismatch")
        self.generator.set_state(state["generator_state"].cpu())
        self.current_block = int(state["current_block"])
        self.active_offset = int(state["active_offset"])
        self.trace = list(state.get("trace", []))


class DimensionNormalizedFunctionalEntropy(nn.Module):
    """Training-only regularizer based on a finite-sample Fisher-information proxy."""

    def __init__(
        self,
        prediction_weight: float = 0.30,
        feature_weight: float = 0.02,
        feature_scales: Sequence[int] = (0, 1, 2, 3),
        eps: float = 1e-6,
        stability_offset: float = 0.05,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        if prediction_weight < 0.0 or feature_weight < 0.0:
            raise ValueError("Functional-entropy weights must be non-negative")
        if eps <= 0.0:
            raise ValueError("Functional-entropy eps must be positive")
        if stability_offset <= 0.0:
            raise ValueError("Functional-entropy stability_offset must be positive")
        self.prediction_weight = float(prediction_weight)
        self.feature_weight = float(feature_weight)
        self.feature_scales = tuple(int(index) for index in feature_scales)
        self.eps = float(eps)
        self.stability_offset = float(stability_offset)
        self.ignore_index = int(ignore_index)

    def _per_sample_cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pixel_loss = deterministic_cross_entropy(
            logits,
            targets,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid = targets.ne(self.ignore_index)
        valid_count = valid.flatten(1).sum(dim=1)
        per_sample = (pixel_loss * valid).flatten(1).sum(dim=1) / valid_count.clamp_min(1)
        return per_sample, valid_count.gt(0)

    def _information(self, gradient: torch.Tensor, ce_per_sample: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
        gradient_energy = gradient.float().reshape(gradient.shape[0], -1).square().mean(dim=1)
        ratio = gradient_energy[sample_mask] / (ce_per_sample[sample_mask].float() + self.eps)
        return ratio.mean()

    def _pair_regularizer(
        self,
        main_gradient: torch.Tensor,
        aux_gradient: torch.Tensor,
        ce_per_sample: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        main_information = self._information(main_gradient, ce_per_sample, sample_mask)
        aux_information = self._information(aux_gradient, ce_per_sample, sample_mask)
        information = torch.stack([main_information, aux_information])
        detached_scale = information.detach().mean().clamp_min(torch.finfo(information.dtype).tiny)
        normalized_information = information / detached_scale
        regularizer = (normalized_information.clamp_min(0.0) + self.stability_offset).reciprocal().mean()
        return regularizer, main_information, aux_information

    def forward(
        self,
        outputs: dict,
        targets: torch.Tensor,
        image: torch.Tensor,
        aux: torch.Tensor,
        main_available: torch.Tensor,
        aux_available: torch.Tensor,
        strength: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        logits = outputs["logits"]
        zero = logits.sum() * 0.0
        ce_per_sample, has_valid_target = self._per_sample_cross_entropy(logits, targets)
        co_available = main_available.gt(0.5) & aux_available.gt(0.5) & has_valid_target
        if not bool(co_available.any().item()) or (self.prediction_weight == 0.0 and self.feature_weight == 0.0):
            return {
                "fe_total": zero,
                "fe_pred": zero,
                "fe_feat": zero,
                "fisher_pred_main": zero,
                "fisher_pred_aux": zero,
                "fisher_feat_main": zero,
                "fisher_feat_aux": zero,
            }

        grad_targets: list[torch.Tensor] = []
        if self.prediction_weight > 0.0:
            if not image.requires_grad or not aux.requires_grad:
                raise RuntimeError("DN-MFE prediction term requires gradient-enabled input tensors")
            grad_targets.extend([image, aux])

        main_features = outputs.get("main_features", [])
        aux_features = outputs.get("aux_features", [])
        selected_features: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self.feature_weight > 0.0:
            if not main_features or not aux_features:
                raise RuntimeError("DN-MFE feature term requires per-modality feature outputs")
            for scale in self.feature_scales:
                if scale < 0 or scale >= len(main_features) or scale >= len(aux_features):
                    raise IndexError(f"DN-MFE feature scale {scale} is unavailable")
                selected_features.append((main_features[scale], aux_features[scale]))
                grad_targets.extend(selected_features[-1])

        functional = ce_per_sample[co_available].sum()
        gradients = torch.autograd.grad(
            functional,
            grad_targets,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )

        cursor = 0
        pred_regularizer = zero
        pred_main_information = zero
        pred_aux_information = zero
        if self.prediction_weight > 0.0:
            pred_regularizer, pred_main_information, pred_aux_information = self._pair_regularizer(
                gradients[cursor], gradients[cursor + 1], ce_per_sample, co_available
            )
            cursor += 2

        feature_regularizers: list[torch.Tensor] = []
        feature_main_information: list[torch.Tensor] = []
        feature_aux_information: list[torch.Tensor] = []
        for _ in selected_features:
            regularizer, main_information, aux_information = self._pair_regularizer(
                gradients[cursor], gradients[cursor + 1], ce_per_sample, co_available
            )
            feature_regularizers.append(regularizer)
            feature_main_information.append(main_information)
            feature_aux_information.append(aux_information)
            cursor += 2

        feature_regularizer = torch.stack(feature_regularizers).mean() if feature_regularizers else zero
        feat_main_information = torch.stack(feature_main_information).mean() if feature_main_information else zero
        feat_aux_information = torch.stack(feature_aux_information).mean() if feature_aux_information else zero

        ramp = float(max(0.0, min(1.0, strength)))
        prediction_loss = ramp * self.prediction_weight * pred_regularizer
        feature_loss = ramp * self.feature_weight * feature_regularizer
        total = prediction_loss + feature_loss
        if not bool(torch.isfinite(total).all().item()):
            raise FloatingPointError("DN-MFE produced a non-finite loss")

        return {
            "fe_total": total,
            "fe_pred": prediction_loss,
            "fe_feat": feature_loss,
            "fisher_pred_main": pred_main_information,
            "fisher_pred_aux": pred_aux_information,
            "fisher_feat_main": feat_main_information,
            "fisher_feat_aux": feat_aux_information,
        }


def functional_entropy_probe(
    model: nn.Module,
    regularizer: DimensionNormalizedFunctionalEntropy,
    targets: torch.Tensor,
    image: torch.Tensor,
    aux: torch.Tensor,
    main_available: torch.Tensor,
    aux_available: torch.Tensor,
    *,
    probe_size: int = 128,
    probe_batch_size: int = 1,
    batch_index: int = 0,
    selected_batch_index: int | None = None,
    probe_generator: torch.Generator | None = None,
    strength: float = 1.0,
) -> dict[str, torch.Tensor] | None:
    """Estimate DN-MFE on an explicitly sampled co-available low-resolution probe."""
    if probe_size <= 0 or probe_batch_size <= 0:
        raise ValueError("Functional-entropy probe dimensions must be positive")
    candidates = torch.nonzero(
        main_available.gt(0.5) & aux_available.gt(0.5),
        as_tuple=False,
    ).flatten()
    if candidates.numel() == 0:
        return None
    if selected_batch_index is not None:
        if probe_batch_size != 1:
            raise ValueError("An explicit DN-MFE sample requires probe_batch_size=1")
        if selected_batch_index < 0 or selected_batch_index >= int(main_available.shape[0]):
            raise IndexError("DN-MFE selected batch index is outside the primary batch")
        selected = torch.tensor([selected_batch_index], device=candidates.device, dtype=torch.long)
        if not bool((candidates == selected_batch_index).any().item()):
            raise ValueError("DN-MFE selected batch index is not co-available")
        count = 1
    else:
        count = min(int(probe_batch_size), int(candidates.numel()))
        if probe_generator is not None:
            permutation = torch.randperm(int(candidates.numel()), generator=probe_generator)[:count]
            positions = permutation.to(device=candidates.device)
        else:
            positions = (torch.arange(count, device=candidates.device) + int(batch_index)) % candidates.numel()
        selected = candidates.index_select(0, positions)
    size = (
        min(int(probe_size), int(image.shape[-2])),
        min(int(probe_size), int(image.shape[-1])),
    )
    probe_image = F.interpolate(
        image.index_select(0, selected).detach(),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).requires_grad_(True)
    probe_aux = F.interpolate(
        aux.index_select(0, selected).detach(),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).requires_grad_(True)
    probe_targets = F.interpolate(
        targets.index_select(0, selected).detach().unsqueeze(1).float(),
        size=size,
        mode="nearest",
    ).squeeze(1).long()
    probe_main_available = main_available.index_select(0, selected).detach()
    probe_aux_available = aux_available.index_select(0, selected).detach()

    batch_norm_states = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            batch_norm_states.append((module, module.training))
            module.training = False
    fork_devices = [probe_image.device.index or 0] if probe_image.is_cuda else []
    try:
        with torch.random.fork_rng(devices=fork_devices, enabled=True):
            outputs = model(
                probe_image,
                probe_aux,
                probe_aux_available,
                probe_main_available,
                return_modality_features=True,
            )
            losses = regularizer(
                outputs,
                probe_targets,
                probe_image,
                probe_aux,
                probe_main_available,
                probe_aux_available,
                strength=strength,
            )
    finally:
        for module, training in batch_norm_states:
            module.training = training
    losses["fe_probe_count"] = outputs["logits"].new_tensor(float(count))
    losses["fe_probe_selected_batch_index"] = outputs["logits"].new_tensor(float(selected[0].item()))
    return losses
