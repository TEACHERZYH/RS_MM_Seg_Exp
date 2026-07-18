from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SecondOrderHardsigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0


class SecondOrderHardswish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0)


def replace_hard_activations_for_second_order(module: nn.Module) -> int:
    replacements = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Hardsigmoid):
            setattr(module, name, SecondOrderHardsigmoid())
            replacements += 1
        elif isinstance(child, nn.Hardswish):
            setattr(module, name, SecondOrderHardswish())
            replacements += 1
        else:
            replacements += replace_hard_activations_for_second_order(child)
    return replacements


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaseEncoder(nn.Module):
    def output_channels(self) -> list[int]:
        raise NotImplementedError


class LightweightEncoder(BaseEncoder):
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        self.stem = ConvBNAct(in_channels, base_channels, stride=2)
        self.stage1 = ConvBNAct(base_channels, base_channels * 2, stride=2)
        self.stage2 = ConvBNAct(base_channels * 2, base_channels * 4, stride=2)
        self.stage3 = ConvBNAct(base_channels * 4, base_channels * 8, stride=2)
        self._output_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stem(x)
        f2 = self.stage1(f1)
        f3 = self.stage2(f2)
        f4 = self.stage3(f3)
        return [f1, f2, f3, f4]

    def output_channels(self) -> list[int]:
        return self._output_channels


class TimmEncoder(BaseEncoder):
    def __init__(
        self,
        model_name: str,
        in_channels: int,
        pretrained: bool,
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required when encoder_type='timm'. Please install it first.") from exc

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained and checkpoint_path is None,
            in_chans=in_channels,
            features_only=True,
            out_indices=out_indices,
        )
        if checkpoint_path is not None:
            checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if not isinstance(state_dict, dict):
                raise TypeError(f"Unsupported timm checkpoint payload: {checkpoint_path}")
            self.backbone.load_state_dict(state_dict, strict=True)
        self._output_channels = list(self.backbone.feature_info.channels())

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(self.backbone(x))

    def output_channels(self) -> list[int]:
        return self._output_channels


def adapt_input_conv_weight(weight: torch.Tensor, in_channels: int) -> torch.Tensor:
    if in_channels == weight.shape[1]:
        return weight
    if in_channels == 1:
        return weight.mean(dim=1, keepdim=True)
    if in_channels < weight.shape[1]:
        return weight[:, :in_channels]

    repeats = (in_channels + weight.shape[1] - 1) // weight.shape[1]
    expanded = weight.repeat(1, repeats, 1, 1)[:, :in_channels]
    return expanded * (weight.shape[1] / float(in_channels))


class TorchvisionConvNeXtEncoder(BaseEncoder):
    def __init__(self, model_name: str, in_channels: int, pretrained: bool) -> None:
        super().__init__()
        from torchvision.models import (
            ConvNeXt_Base_Weights,
            ConvNeXt_Small_Weights,
            ConvNeXt_Tiny_Weights,
            convnext_base,
            convnext_small,
            convnext_tiny,
        )

        factory = {
            "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
            "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT),
            "convnext_base": (convnext_base, ConvNeXt_Base_Weights.DEFAULT),
        }
        if model_name not in factory:
            raise ValueError(f"Unsupported torchvision ConvNeXt model: {model_name}")

        build_fn, default_weights = factory[model_name]
        weights = default_weights if pretrained else None
        backbone = build_fn(weights=weights)

        stem_conv = backbone.features[0][0]
        if in_channels != stem_conv.in_channels:
            new_conv = nn.Conv2d(
                in_channels,
                stem_conv.out_channels,
                kernel_size=stem_conv.kernel_size,
                stride=stem_conv.stride,
                padding=stem_conv.padding,
                bias=stem_conv.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight.copy_(adapt_input_conv_weight(stem_conv.weight, in_channels))
                if stem_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(stem_conv.bias)
            backbone.features[0][0] = new_conv

        self.backbone = backbone.features
        self._output_indices = (1, 3, 5, 7)
        self._output_channels = [96, 192, 384, 768] if "tiny" in model_name else (
            [96, 192, 384, 768] if "small" in model_name else [128, 256, 512, 1024]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            if idx in self._output_indices:
                features.append(x)
        return features

    def output_channels(self) -> list[int]:
        return self._output_channels


class TorchvisionMobileNetV3Encoder(BaseEncoder):
    def __init__(
        self,
        model_name: str,
        in_channels: int,
        pretrained: bool,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        from torchvision.models import (
            MobileNet_V3_Large_Weights,
            MobileNet_V3_Small_Weights,
            mobilenet_v3_large,
            mobilenet_v3_small,
        )

        factory = {
            "mobilenet_v3_large": (mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT, (1, 3, 6, 12), [16, 24, 40, 112]),
            "mobilenet_v3_small": (mobilenet_v3_small, MobileNet_V3_Small_Weights.DEFAULT, (0, 2, 7, 10), [16, 24, 48, 96]),
        }
        if model_name not in factory:
            raise ValueError(f"Unsupported torchvision MobileNetV3 model: {model_name}")

        build_fn, default_weights, output_indices, output_channels = factory[model_name]
        if pretrained and checkpoint_path:
            raise ValueError("Use either torchvision download-based pretrained weights or an explicit checkpoint, not both")
        weights = default_weights if pretrained else None
        backbone = build_fn(weights=weights)
        if checkpoint_path:
            checkpoint = Path(checkpoint_path)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            try:
                state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(checkpoint, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            if not isinstance(state_dict, dict):
                raise TypeError(f"Unsupported torchvision checkpoint payload: {checkpoint}")
            state_dict = {
                key.removeprefix("module."): value
                for key, value in state_dict.items()
            }
            backbone.load_state_dict(state_dict, strict=True)

        stem_conv = backbone.features[0][0]
        if in_channels != stem_conv.in_channels:
            new_conv = nn.Conv2d(
                in_channels,
                stem_conv.out_channels,
                kernel_size=stem_conv.kernel_size,
                stride=stem_conv.stride,
                padding=stem_conv.padding,
                bias=stem_conv.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight.copy_(adapt_input_conv_weight(stem_conv.weight, in_channels))
                if stem_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(stem_conv.bias)
            backbone.features[0][0] = new_conv

        self.backbone = backbone.features[: max(output_indices) + 1]
        self._output_indices = output_indices
        self._output_channels = output_channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            if idx in self._output_indices:
                features.append(x)
        return features

    def output_channels(self) -> list[int]:
        return self._output_channels


class ModalityQualityEstimator(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 4, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, feature: torch.Tensor, available: torch.Tensor | None = None) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(feature, output_size=1).flatten(1)
        quality = torch.sigmoid(self.mlp(pooled))
        if available is not None:
            quality = quality * available.unsqueeze(1)
        return quality


class DynamicGatedFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.main_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.aux_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1),
        )

    def forward(
        self,
        main_feat: torch.Tensor,
        aux_feat: torch.Tensor,
        main_quality: torch.Tensor,
        aux_quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        main_feat = self.main_proj(main_feat)
        aux_feat = self.aux_proj(aux_feat)

        gate_logits = self.gate_conv(torch.cat([main_feat, aux_feat], dim=1))
        local_gates = torch.softmax(gate_logits, dim=1)

        global_main = main_quality.view(-1, 1, 1, 1)
        global_aux = aux_quality.view(-1, 1, 1, 1)

        gate_main = local_gates[:, 0:1] * global_main
        gate_aux = local_gates[:, 1:2] * global_aux
        normalizer = gate_main + gate_aux + 1e-6
        gate_main = gate_main / normalizer
        gate_aux = gate_aux / normalizer

        fused = gate_main * main_feat + gate_aux * aux_feat
        return fused, torch.cat([gate_main, gate_aux], dim=1)


class LightweightDecoder(nn.Module):
    def __init__(self, in_channels: list[int], decoder_channels: int, num_classes: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(ch, decoder_channels, kernel_size=1, bias=False) for ch in in_channels
        ])
        self.smooth = nn.ModuleList([
            ConvBNAct(decoder_channels, decoder_channels) for _ in in_channels[:-1]
        ])
        self.head = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        feats = [proj(feat) for proj, feat in zip(self.lateral, features)]
        x = feats[-1]
        for idx in range(len(feats) - 2, -1, -1):
            x = F.interpolate(x, size=feats[idx].shape[-2:], mode="bilinear", align_corners=False)
            x = self.smooth[idx](x + feats[idx])
        return self.head(x)


class QALFNet(nn.Module):
    def __init__(
        self,
        in_channels_main: int,
        in_channels_aux: int,
        base_channels: int,
        decoder_channels: int,
        num_classes: int,
        dropout: float = 0.1,
        encoder_type: str = "lightweight",
        encoder_name: str = "convnext_tiny.in12k",
        encoder_pretrained: bool = False,
        aux_encoder_name: str | None = None,
        aux_encoder_pretrained: bool | None = None,
        freeze_main_encoder: bool = False,
        freeze_aux_encoder: bool = False,
        fusion_mode: str = "dynamic_gated",
        encoder_out_indices: tuple[int, ...] = (0, 1, 2, 3),
        aux_encoder_out_indices: tuple[int, ...] | None = None,
        encoder_checkpoint: str | None = None,
        aux_encoder_checkpoint: str | None = None,
        second_order_compatible_activations: bool = False,
    ) -> None:
        super().__init__()
        supported_fusion_modes = {
            "dynamic_gated",
            "quality_weighted",
            "fixed_average",
            "availability_masked_average",
            "main_only",
            "aux_only",
            "early_fusion",
        }
        if fusion_mode not in supported_fusion_modes:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.fusion_mode = fusion_mode

        self.main_encoder = self._build_encoder(
            encoder_type=encoder_type,
            in_channels=in_channels_main,
            base_channels=base_channels,
            model_name=encoder_name,
            pretrained=encoder_pretrained,
            out_indices=tuple(encoder_out_indices),
            checkpoint_path=encoder_checkpoint,
        )
        self.aux_encoder = self._build_encoder(
            encoder_type=encoder_type,
            in_channels=in_channels_aux,
            base_channels=base_channels,
            model_name=aux_encoder_name or encoder_name,
            pretrained=encoder_pretrained if aux_encoder_pretrained is None else aux_encoder_pretrained,
            out_indices=tuple(aux_encoder_out_indices or encoder_out_indices),
            checkpoint_path=aux_encoder_checkpoint,
        )
        encoder_dims = self.main_encoder.output_channels()
        aux_dims = self.aux_encoder.output_channels()
        if encoder_dims != aux_dims:
            raise ValueError(f"Main/aux encoder feature dims must match, got {encoder_dims} vs {aux_dims}")

        self.second_order_activation_replacements = 0
        if second_order_compatible_activations:
            self.second_order_activation_replacements += replace_hard_activations_for_second_order(self.main_encoder)
            self.second_order_activation_replacements += replace_hard_activations_for_second_order(self.aux_encoder)

        self.quality_estimators_main = nn.ModuleList([ModalityQualityEstimator(ch) for ch in encoder_dims])
        self.quality_estimators_aux = nn.ModuleList([ModalityQualityEstimator(ch) for ch in encoder_dims])
        self.fusions = nn.ModuleList([DynamicGatedFusion(ch) for ch in encoder_dims])
        self.decoder = LightweightDecoder(encoder_dims, decoder_channels, num_classes)
        self.dropout = nn.Dropout2d(p=dropout)

        if freeze_main_encoder:
            for param in self.main_encoder.parameters():
                param.requires_grad = False
        if freeze_aux_encoder:
            for param in self.aux_encoder.parameters():
                param.requires_grad = False
        if fusion_mode in {"main_only", "early_fusion"}:
            for module in (self.aux_encoder, self.quality_estimators_main, self.quality_estimators_aux, self.fusions):
                for param in module.parameters():
                    param.requires_grad = False
        if fusion_mode == "aux_only":
            for module in (self.main_encoder, self.quality_estimators_main, self.quality_estimators_aux, self.fusions):
                for param in module.parameters():
                    param.requires_grad = False
        if fusion_mode in {"fixed_average", "availability_masked_average"}:
            for module in (self.quality_estimators_main, self.quality_estimators_aux):
                for param in module.parameters():
                    param.requires_grad = False
            for fusion in self.fusions:
                for param in fusion.gate_conv.parameters():
                    param.requires_grad = False
        if fusion_mode == "quality_weighted":
            for fusion in self.fusions:
                for param in fusion.gate_conv.parameters():
                    param.requires_grad = False

    @staticmethod
    def _build_encoder(
        encoder_type: str,
        in_channels: int,
        base_channels: int,
        model_name: str,
        pretrained: bool,
        out_indices: tuple[int, ...],
        checkpoint_path: str | None,
    ) -> BaseEncoder:
        if encoder_type == "lightweight":
            return LightweightEncoder(in_channels, base_channels)
        if encoder_type == "timm":
            return TimmEncoder(
                model_name=model_name,
                in_channels=in_channels,
                pretrained=pretrained,
                out_indices=out_indices,
                checkpoint_path=checkpoint_path,
            )
        if encoder_type == "torchvision_convnext":
            if checkpoint_path is not None:
                raise ValueError("Explicit checkpoints are not implemented for torchvision ConvNeXt")
            return TorchvisionConvNeXtEncoder(model_name=model_name, in_channels=in_channels, pretrained=pretrained)
        if encoder_type == "torchvision_mobilenetv3":
            return TorchvisionMobileNetV3Encoder(
                model_name=model_name,
                in_channels=in_channels,
                pretrained=pretrained,
                checkpoint_path=checkpoint_path,
            )
        raise ValueError(f"Unsupported encoder_type: {encoder_type}")

    def forward(
        self,
        image: torch.Tensor,
        aux: torch.Tensor,
        aux_available: torch.Tensor | None = None,
        main_available: torch.Tensor | None = None,
        return_modality_features: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if self.fusion_mode == "early_fusion":
            main_input = torch.cat([image, aux], dim=1)
            main_feats = self.main_encoder(main_input)
            logits = self.decoder([self.dropout(feat) for feat in main_feats])
            logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
            outputs = {
                "logits": logits,
                "fused_features": main_feats,
                "quality_main": [],
                "quality_aux": [],
                "gate_maps": [],
            }
            if return_modality_features:
                outputs["main_features"] = main_feats
                outputs["aux_features"] = []
            return outputs

        if self.fusion_mode == "main_only":
            main_feats = self.main_encoder(image)
            logits = self.decoder([self.dropout(feat) for feat in main_feats])
            logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
            outputs = {
                "logits": logits,
                "fused_features": main_feats,
                "quality_main": [],
                "quality_aux": [],
                "gate_maps": [],
            }
            if return_modality_features:
                outputs["main_features"] = main_feats
                outputs["aux_features"] = []
            return outputs

        if self.fusion_mode == "aux_only":
            aux_feats = self.aux_encoder(aux)
            logits = self.decoder([self.dropout(feat) for feat in aux_feats])
            logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
            outputs = {
                "logits": logits,
                "fused_features": aux_feats,
                "quality_main": [],
                "quality_aux": [],
                "gate_maps": [],
            }
            if return_modality_features:
                outputs["main_features"] = []
                outputs["aux_features"] = aux_feats
            return outputs

        main_feats = self.main_encoder(image)
        aux_feats = self.aux_encoder(aux)

        fused_feats = []
        quality_main_list = []
        quality_aux_list = []
        gate_maps = []

        if aux_available is None:
            aux_available = torch.ones(image.size(0), device=image.device)
        if main_available is None:
            main_available = torch.ones_like(aux_available)
        availability_sum = main_available.to(dtype=torch.float32) + aux_available.to(dtype=torch.float32)
        if bool(availability_sum.le(0).any().item()):
            raise ValueError(f"{self.fusion_mode} requires at least one available modality per sample")

        for idx, (main_feat, aux_feat) in enumerate(zip(main_feats, aux_feats)):
            if self.fusion_mode in {"fixed_average", "availability_masked_average"}:
                main_proj = self.fusions[idx].main_proj(main_feat)
                aux_proj = self.fusions[idx].aux_proj(aux_feat)
                if self.fusion_mode == "fixed_average":
                    fused_template = main_proj[:, :1]
                    gate_main = torch.full_like(fused_template, 0.5)
                    gate_aux = torch.full_like(fused_template, 0.5)
                else:
                    main_weight = main_available.to(dtype=main_proj.dtype).view(-1, 1, 1, 1)
                    aux_weight = aux_available.to(dtype=aux_proj.dtype).view(-1, 1, 1, 1)
                    denominator = main_weight + aux_weight
                    if bool(denominator.le(0).any().item()):
                        raise ValueError("availability_masked_average requires at least one available modality per sample")
                    gate_main = main_weight / denominator
                    gate_aux = aux_weight / denominator
                fused = gate_main * main_proj + gate_aux * aux_proj
                gates = torch.cat(
                    [
                        gate_main.expand(-1, 1, fused.shape[-2], fused.shape[-1]),
                        gate_aux.expand(-1, 1, fused.shape[-2], fused.shape[-1]),
                    ],
                    dim=1,
                )
                fused_feats.append(self.dropout(fused))
                gate_maps.append(gates)
                continue

            q_main = self.quality_estimators_main[idx](main_feat, main_available)
            q_aux = self.quality_estimators_aux[idx](aux_feat, aux_available)
            if self.fusion_mode == "quality_weighted":
                main_proj = self.fusions[idx].main_proj(main_feat)
                aux_proj = self.fusions[idx].aux_proj(aux_feat)
                gate_main = q_main.view(-1, 1, 1, 1)
                gate_aux = q_aux.view(-1, 1, 1, 1)
                normalizer = gate_main + gate_aux + 1e-6
                gate_main = gate_main / normalizer
                gate_aux = gate_aux / normalizer
                fused = gate_main * main_proj + gate_aux * aux_proj
                gates = torch.cat(
                    [
                        gate_main.expand(-1, 1, fused.shape[-2], fused.shape[-1]),
                        gate_aux.expand(-1, 1, fused.shape[-2], fused.shape[-1]),
                    ],
                    dim=1,
                )
            else:
                fused, gates = self.fusions[idx](main_feat, aux_feat, q_main, q_aux)

            fused_feats.append(self.dropout(fused))
            quality_main_list.append(q_main)
            quality_aux_list.append(q_aux)
            gate_maps.append(gates)

        logits = self.decoder(fused_feats)
        logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)

        outputs = {
            "logits": logits,
            "fused_features": fused_feats,
            "quality_main": quality_main_list,
            "quality_aux": quality_aux_list,
            "gate_maps": gate_maps,
        }
        if return_modality_features:
            outputs["main_features"] = main_feats
            outputs["aux_features"] = aux_feats
        return outputs
