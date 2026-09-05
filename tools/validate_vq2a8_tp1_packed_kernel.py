#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the TP1 M=1 packed VQ2A8 Triton kernel on CUDA or Ascend."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm_ascend.quantization.vq2a8_reference import (
    decode_repacked_vq2a8_codebook_weight,
    prepare_repacked_vq2a8_activation_reference,
    vq2a8_predecoded_matmul_reference,
)
from vllm_ascend.quantization.vq2a8_runtime import (
    VQ2TP1Artifact,
    open_vq2a8_tp1_artifact,
)
from vllm_ascend.quantization.vq2a8_triton import vq2a8_tp1_m1_packed_gemm


@dataclass(frozen=True)
class Probe:
    layer_index: int
    expert_id: int


def parse_probes(value: str) -> tuple[Probe, ...]:
    probes: list[Probe] = []
    seen: set[tuple[int, int]] = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
        parts = item.split(":")
        if len(parts) != 2 or any(not part.isdigit() for part in parts):
            raise argparse.ArgumentTypeError(
                f"Invalid probe {item!r}; use comma-separated layer:expert pairs such as 0:0,3:255."
            )
        pair = int(parts[0]), int(parts[1])
        if pair in seen:
            raise argparse.ArgumentTypeError(f"Duplicate probe {item!r}.")
        seen.add(pair)
        probes.append(Probe(*pair))
    if not probes:
        raise argparse.ArgumentTypeError("At least one layer:expert probe is required.")
    return tuple(probes)


def deterministic_activation(width: int, probe_index: int, kind_index: int) -> torch.Tensor:
    columns = torch.arange(width, dtype=torch.int64)
    multiplier = probe_index * 4 + kind_index * 2 + 3
    numerator = ((columns * multiplier + probe_index * 7 + kind_index * 11).remainder(61) - 30).float()
    block_gain = torch.div(columns, 128, rounding_mode="floor").remainder(5).float() + 1
    return (numerator * block_gain / 64).to(torch.bfloat16).unsqueeze(0).contiguous()


def _synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
    else:
        raise ValueError(f"Unsupported accelerator: {device}.")


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().float().cpu()
    finite = torch.isfinite(values)
    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(finite.all()),
        "nan_count": int(torch.isnan(values).sum()),
        "inf_count": int(torch.isinf(values).sum()),
    }
    if bool(finite.any()):
        finite_values = values[finite]
        result.update(
            {
                "min": float(finite_values.min()),
                "max": float(finite_values.max()),
                "absmax": float(finite_values.abs().max()),
                "l2": float(torch.linalg.vector_norm(finite_values)),
            }
        )
    return result


def _comparison_summary(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    expected_float = expected.detach().float().cpu()
    actual_float = actual.detach().float().cpu()
    if expected_float.shape != actual_float.shape:
        raise AssertionError(f"Shape mismatch: expected={expected_float.shape}, actual={actual_float.shape}.")
    if not bool(torch.isfinite(expected_float).all()):
        raise AssertionError("CPU oracle produced a non-finite tensor.")
    if not bool(torch.isfinite(actual_float).all()):
        raise AssertionError("Packed device kernel produced a non-finite tensor.")
    difference = (actual_float - expected_float).abs()
    close = torch.isclose(actual_float, expected_float, rtol=rtol, atol=atol)
    denominator = expected_float.abs().clamp_min(1e-12)
    result = {
        "allclose": bool(close.all()),
        "mismatch_count": int((~close).sum()),
        "numel": expected_float.numel(),
        "max_abs_error": float(difference.max()) if difference.numel() else 0.0,
        "max_rel_error": float((difference / denominator).max()) if difference.numel() else 0.0,
        "expected": _tensor_summary(expected_float),
        "actual": _tensor_summary(actual_float),
    }
    if not result["allclose"]:
        mismatch = (~close).flatten()
        mismatch_indices = torch.where(mismatch)[0][:8]
        samples = [
            {
                "index": int(index),
                "expected": float(expected_float.flatten()[index]),
                "actual": float(actual_float.flatten()[index]),
                "abs_error": float(difference.flatten()[index]),
            }
            for index in mismatch_indices
        ]
        raise AssertionError(
            f"Packed kernel mismatch: count={result['mismatch_count']}/{result['numel']}, "
            f"max_abs={result['max_abs_error']}, max_rel={result['max_rel_error']}, "
            f"rtol={rtol}, atol={atol}, samples={samples}."
        )
    return result


def _run_projection(
    artifact: VQ2TP1Artifact,
    probe: Probe,
    probe_index: int,
    kind: str,
    device: torch.device,
    *,
    warmups: int,
    repeats: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    payload_cpu, spec = artifact.load_expert(probe.layer_index, probe.expert_id, kind)
    activation_cpu = deterministic_activation(spec.rht_true_columns, probe_index, int(kind == "down"))
    activation_padded_cpu = activation_cpu
    if spec.rht_true_columns != spec.columns:
        activation_padded_cpu = torch.nn.functional.pad(
            activation_cpu,
            (0, spec.columns - spec.rht_true_columns),
        )

    # This dense decode exists only in the independent CPU oracle.  The device
    # path below receives packed indices and codebooks directly.
    codebook_weight_cpu = decode_repacked_vq2a8_codebook_weight(
        payload_cpu,
        spec,
        compute_dtype=torch.float32,
    )
    expected = vq2a8_predecoded_matmul_reference(
        activation_cpu,
        codebook_weight_cpu,
        payload_cpu["weight_scale"],
        payload_cpu["weight_bias"],
        payload_cpu["rht_sign"],
        spec,
        compute_dtype=torch.float32,
        dynamic_a8=True,
    ).to(torch.bfloat16)

    activation_device = activation_padded_cpu.to(device=device)
    weight_scale = payload_cpu["weight_scale"].to(device=device)
    weight_bias = payload_cpu["weight_bias"].to(device=device)
    rht_sign = payload_cpu["rht_sign"].to(device=device)
    quantized, activation_scale, bias_correction = prepare_repacked_vq2a8_activation_reference(
        activation_device,
        weight_scale,
        weight_bias,
        rht_sign,
        spec.rht_block_size,
    )
    # FP8 values are mirrored into BF16 for this first cross-backend tl.dot
    # gate.  This is lossless because BF16 has a wider significand/exponent
    # range than E4M3FN.
    prepared_activation = quantized.to(torch.bfloat16).contiguous()
    packed_indices = payload_cpu["packed_indices"].to(device=device).contiguous()
    codebooks = payload_cpu["codebooks"].to(device=device, dtype=torch.bfloat16).contiguous()
    codebook_tile_ids = payload_cpu["codebook_tile_ids"].to(device=device).contiguous()

    actual = vq2a8_tp1_m1_packed_gemm(
        prepared_activation,
        activation_scale.contiguous(),
        bias_correction.contiguous(),
        packed_indices,
        codebooks,
        codebook_tile_ids,
    )
    _synchronize(device)
    comparison = _comparison_summary(expected, actual, rtol=rtol, atol=atol)

    for _ in range(warmups):
        vq2a8_tp1_m1_packed_gemm(
            prepared_activation,
            activation_scale,
            bias_correction,
            packed_indices,
            codebooks,
            codebook_tile_ids,
        )
    _synchronize(device)

    elapsed_ms: list[float] = []
    repeated_output = actual
    for _ in range(repeats):
        started = time.perf_counter()
        repeated_output = vq2a8_tp1_m1_packed_gemm(
            prepared_activation,
            activation_scale,
            bias_correction,
            packed_indices,
            codebooks,
            codebook_tile_ids,
        )
        _synchronize(device)
        elapsed_ms.append((time.perf_counter() - started) * 1000)
    determinism = _comparison_summary(actual, repeated_output, rtol=0.0, atol=0.0)

    result = {
        "projection": f"{probe.layer_index}:{probe.expert_id}:{kind}",
        "shape": {"m": 1, "n": spec.rows, "k": spec.columns},
        "activation_prepare_backend": "validated_eager_dynamic_a8",
        "packed_projection_backend": "triton_bf16_cube_aligned_table_v2",
        "device_dense_weight_materialized": False,
        "comparison": comparison,
        "determinism": determinism,
        "kernel_ms": {
            "min": min(elapsed_ms),
            "median": statistics.median(elapsed_ms),
            "max": max(elapsed_ms),
            "repeats": repeats,
        },
    }
    print("KERNEL_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    del (
        payload_cpu,
        activation_padded_cpu,
        codebook_weight_cpu,
        expected,
        activation_device,
        weight_scale,
        weight_bias,
        rht_sign,
        quantized,
        activation_scale,
        bias_correction,
        prepared_activation,
        packed_indices,
        codebooks,
        codebook_tile_ids,
        actual,
        repeated_output,
    )
    gc.collect()
    if device.type == "npu":
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()
    return result


def _initialize_device(device: torch.device) -> dict[str, Any]:
    if device.type == "npu":
        import torch_npu

        if not torch.npu.is_available() or torch.npu.device_count() != 1:
            raise RuntimeError("Exactly one logical NPU must be visible for the Ascend gate.")
        if device.index not in (None, 0):
            raise ValueError(f"The isolated Ascend gate requires logical npu:0, got {device}.")
        torch.npu.set_device(0)
        soc_version = torch_npu.npu.get_soc_version()
        device_name = torch.npu.get_device_name(0)
        if soc_version != 260 or "Ascend950" not in device_name:
            raise RuntimeError(f"Expected Ascend 950 (SoC 260), got soc={soc_version}, name={device_name!r}.")
        return {"type": "npu", "logical": "npu:0", "name": device_name, "soc": soc_version}
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(index)
        return {
            "type": "cuda",
            "logical": f"cuda:{index}",
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
        }
    raise ValueError(f"The packed kernel gate requires npu or cuda, got {device}.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument(
        "--probes",
        type=parse_probes,
        default=parse_probes("0:0,3:0,3:127,3:255,42:255"),
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--allow-partial-artifact",
        action="store_true",
        help=(
            "Allow a layer-subset artifact for developer validation. "
            "The Ascend 950 wrapper intentionally never enables this."
        ),
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--atol", type=float, default=0.05)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats must be positive.")
    model = args.model.expanduser().resolve(strict=True)
    artifact_path = args.artifact.expanduser().resolve(strict=True)
    device = torch.device(args.device)
    print("DEVICE " + json.dumps(_initialize_device(device), sort_keys=True), flush=True)
    artifact = open_vq2a8_tp1_artifact(
        artifact_path,
        model / "config.json",
        require_complete=not args.allow_partial_artifact,
        require_reference_identity=True,
        verify_tensor_hashes=False,
    )
    print(
        "ARTIFACT_RESULT "
        + json.dumps(
            {
                "complete": artifact.manifest["complete"],
                "format": artifact.manifest["format"],
                "layers": len(artifact.layers),
                "root": str(artifact.root),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    results: list[dict[str, Any]] = []
    for probe_index, probe in enumerate(args.probes):
        layer = artifact.layer(probe.layer_index)
        if probe.expert_id not in layer.expert_ids:
            raise ValueError(f"Artifact has no probe {probe.layer_index}:{probe.expert_id}.")
        for kind in ("gate_up", "down"):
            print(f"KERNEL {probe.layer_index}:{probe.expert_id}:{kind} stage=packed_gemm", flush=True)
            results.append(
                _run_projection(
                    artifact,
                    probe,
                    probe_index,
                    kind,
                    device,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    rtol=args.rtol,
                    atol=args.atol,
                )
            )

    print(
        "VQ2A8_TP1_M1_PACKED_KERNEL_GATE=PASS "
        + json.dumps(
            {
                "artifact": str(artifact.root),
                "device": str(device),
                "projections": [result["projection"] for result in results],
                "dense_weight_on_device": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
