from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.qalf_net import QALFNet
from src.utils import load_checkpoint, load_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def nvidia_smi_inventory() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,persistence_mode,pstate,power.limit,power.draw,clocks.sm,clocks.mem,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "rows": result.stdout.strip().splitlines() if result.returncode == 0 else [],
        "stderr": result.stderr.strip() if result.returncode != 0 else "",
    }


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def module_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def deployed_params(model: QALFNet) -> int:
    """Count the parameters needed by the pruned inference graph."""
    fusion_mode = getattr(model, "fusion_mode", "dynamic_gated")
    decoder = module_params(model.decoder)

    if fusion_mode in {"main_only", "early_fusion"}:
        return module_params(model.main_encoder) + decoder
    if fusion_mode == "aux_only":
        return module_params(model.aux_encoder) + decoder

    encoder_params = module_params(model.main_encoder) + module_params(model.aux_encoder)
    projection_params = sum(
        module_params(fusion.main_proj) + module_params(fusion.aux_proj)
        for fusion in model.fusions
    )

    if fusion_mode in {"fixed_average", "availability_masked_average"}:
        return encoder_params + projection_params + decoder

    quality_params = module_params(model.quality_estimators_main) + module_params(model.quality_estimators_aux)
    if fusion_mode == "quality_weighted":
        return encoder_params + projection_params + quality_params + decoder

    gate_params = sum(module_params(fusion.gate_conv) for fusion in model.fusions)
    return encoder_params + projection_params + gate_params + quality_params + decoder


def route_call_counts(
    model: QALFNet,
    image: torch.Tensor,
    aux: torch.Tensor,
    aux_available: torch.Tensor,
) -> dict[str, int | bool]:
    counts = {"main_encoder": 0, "aux_encoder": 0, "decoder": 0, "projection": 0, "quality": 0, "gate": 0}
    handles = []

    def hook(name: str):
        def increment(_module, _inputs, _output) -> None:
            counts[name] += 1

        return increment

    handles.append(model.main_encoder.register_forward_hook(hook("main_encoder")))
    handles.append(model.aux_encoder.register_forward_hook(hook("aux_encoder")))
    handles.append(model.decoder.register_forward_hook(hook("decoder")))
    for fusion in model.fusions:
        handles.append(fusion.main_proj.register_forward_hook(hook("projection")))
        handles.append(fusion.aux_proj.register_forward_hook(hook("projection")))
        handles.append(fusion.gate_conv.register_forward_hook(hook("gate")))
    for estimator in list(model.quality_estimators_main) + list(model.quality_estimators_aux):
        handles.append(estimator.register_forward_hook(hook("quality")))
    try:
        with torch.no_grad():
            model(image, aux, aux_available)
    finally:
        for handle in handles:
            handle.remove()

    mode = model.fusion_mode
    expected = {
        "main_only": {"main_encoder": 1, "aux_encoder": 0, "decoder": 1, "projection": 0, "quality": 0, "gate": 0},
        "aux_only": {"main_encoder": 0, "aux_encoder": 1, "decoder": 1, "projection": 0, "quality": 0, "gate": 0},
        "early_fusion": {"main_encoder": 1, "aux_encoder": 0, "decoder": 1, "projection": 0, "quality": 0, "gate": 0},
        "fixed_average": {"main_encoder": 1, "aux_encoder": 1, "decoder": 1, "projection": 8, "quality": 0, "gate": 0},
        "availability_masked_average": {"main_encoder": 1, "aux_encoder": 1, "decoder": 1, "projection": 8, "quality": 0, "gate": 0},
        "quality_weighted": {"main_encoder": 1, "aux_encoder": 1, "decoder": 1, "projection": 8, "quality": 8, "gate": 0},
        "dynamic_gated": {"main_encoder": 1, "aux_encoder": 1, "decoder": 1, "projection": 8, "quality": 8, "gate": 4},
    }[mode]
    return {**counts, "verified": all(counts[key] == value for key, value in expected.items())}


def resolve_checkpoint(config: dict, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    output_dir = Path(config["experiment"]["output_dir"])
    candidate = output_dir / "last_model.pt"
    return candidate if candidate.exists() else None


def load_model(config_path: str, checkpoint: str | None, device: torch.device) -> tuple[QALFNet, dict, Path | None]:
    config = load_config(config_path)
    model_config = dict(config["model"])
    model_config["encoder_pretrained"] = False
    model_config["aux_encoder_pretrained"] = False
    model = QALFNet(**model_config).to(device)
    ckpt_path = resolve_checkpoint(config, checkpoint)
    if ckpt_path is not None:
        ckpt = load_checkpoint(str(ckpt_path), map_location=device.type)
        model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config, ckpt_path


def estimate_flops(model: QALFNet, image: torch.Tensor, aux: torch.Tensor, aux_available: torch.Tensor) -> int | None:
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception:
        return None
    try:
        activities = [ProfilerActivity.CPU]
        if image.is_cuda:
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, with_flops=True, record_shapes=False) as prof:
            with torch.no_grad():
                model(image, aux, aux_available)
        flops = sum(getattr(evt, "flops", 0) or 0 for evt in prof.key_averages())
        return int(flops) if flops > 0 else None
    except Exception:
        return None


def measure_latency(
    model: QALFNet,
    image: torch.Tensor,
    aux: torch.Tensor,
    aux_available: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[float, float, float, float, float | None]:
    times_ms: list[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            model(image, aux, aux_available)
        if image.is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(image.device)
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(image, aux, aux_available)
                end.record()
                torch.cuda.synchronize()
                times_ms.append(float(start.elapsed_time(end)))
            peak_memory_mb = torch.cuda.max_memory_allocated(image.device) / (1024.0**2)
        else:
            for _ in range(iterations):
                start = time.perf_counter()
                model(image, aux, aux_available)
                times_ms.append((time.perf_counter() - start) * 1000.0)
            peak_memory_mb = None
    mean_ms = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)
    p95_ms = sorted(times_ms)[max(0, int(len(times_ms) * 0.95) - 1)]
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return mean_ms, median_ms, p95_ms, fps, peak_memory_mb


def parse_item(item: str) -> tuple[str, str, str | None]:
    parts = item.split("|")
    if len(parts) == 2:
        return parts[0], parts[1], None
    if len(parts) == 3:
        return parts[0], parts[1], parts[2] or None
    raise ValueError(f"Invalid --item format: {item}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", action="append", required=True, help="label|config|checkpoint(optional)")
    parser.add_argument("--output-dir", default="outputs/efficiency")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--require-route-verification", action="store_true")
    parser.add_argument("--require-flops", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA efficiency measurement requested but CUDA is unavailable")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    metadata = {
        "device": str(device),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and device.type == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "items": args.item,
        "pretrained_initialization_disabled": True,
        "nvidia_smi_inventory": nvidia_smi_inventory() if torch.cuda.is_available() else None,
    }

    for item in args.item:
        label, config_path, checkpoint = parse_item(item)
        model, config, ckpt_path = load_model(config_path, checkpoint, device)
        input_size = int(config["dataset"]["input_size"])
        image = torch.randn(args.batch_size, 3, input_size, input_size, device=device)
        aux = torch.randn(args.batch_size, 1, input_size, input_size, device=device)
        aux_available = torch.ones(args.batch_size, device=device)
        total_params, trainable_params = count_params(model)
        active_params = deployed_params(model)
        route_counts = route_call_counts(model, image, aux, aux_available)
        if args.require_route_verification and not route_counts["verified"]:
            raise RuntimeError(f"Route graph verification failed for {label}: {route_counts}")
        flops = estimate_flops(model, image, aux, aux_available)
        if args.require_flops and flops is None:
            raise RuntimeError(f"FLOP measurement is unavailable for {label}")
        mean_ms, median_ms, p95_ms, fps, peak_memory_mb = measure_latency(
            model, image, aux, aux_available, args.warmup, args.iterations
        )
        rows.append(
            {
                "label": label,
                "config": config_path,
                "config_sha256": sha256_file(Path(config_path)),
                "checkpoint": str(ckpt_path) if ckpt_path is not None else "",
                "checkpoint_sha256": sha256_file(ckpt_path) if ckpt_path is not None else "",
                "fusion_mode": config["model"].get("fusion_mode", "dynamic_gated"),
                "input_size": input_size,
                "batch_size": args.batch_size,
                "total_params_m": f"{total_params / 1e6:.4f}",
                "trainable_params_m": f"{trainable_params / 1e6:.4f}",
                "route_params_m": f"{active_params / 1e6:.4f}",
                "active_deployed_params_m": f"{active_params / 1e6:.4f}",
                "route_graph_verified": str(bool(route_counts["verified"])).lower(),
                "quality_estimator_calls": route_counts["quality"],
                "gate_predictor_calls": route_counts["gate"],
                "projection_calls": route_counts["projection"],
                "flops_g": "" if flops is None else f"{flops / 1e9:.4f}",
                "latency_mean_ms": f"{mean_ms:.4f}",
                "latency_median_ms": f"{median_ms:.4f}",
                "latency_p95_ms": f"{p95_ms:.4f}",
                "fps": f"{fps:.4f}",
                "peak_allocated_memory_mb": "" if peak_memory_mb is None else f"{peak_memory_mb:.2f}",
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = output_dir / "efficiency_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metadata.update(
        {
            "schema": "qalf-efficiency-metadata-v2",
            "summary_csv_sha256": sha256_file(csv_path),
            "row_count": len(rows),
        }
    )
    json_path = output_dir / "efficiency_metadata.json"
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
