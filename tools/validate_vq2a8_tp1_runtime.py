#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate a complete TP1 VQ2A8 artifact and run an Ascend 950 eager oracle."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm_ascend.quantization.vq2a8_reference import (
    deepseek_v4_swiglu_reference,
    vq2a8_repacked_matmul_reference,
)
from vllm_ascend.quantization.vq2a8_runtime import (
    VQ2TP1Artifact,
    open_vq2a8_tp1_artifact,
)


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


def deterministic_activation(width: int, probe_index: int) -> torch.Tensor:
    """Create a finite BF16 row without backend-dependent random generation."""
    columns = torch.arange(width, dtype=torch.int64)
    numerator = ((columns * (probe_index * 2 + 3) + probe_index * 7).remainder(61) - 30).float()
    block_gain = torch.div(columns, 128, rounding_mode="floor").remainder(5).float() + 1
    return (numerator * block_gain / 64).to(torch.bfloat16).unsqueeze(0).contiguous()


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
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


def comparison_summary(expected: torch.Tensor, actual: torch.Tensor, *, rtol: float, atol: float) -> dict[str, Any]:
    expected_float = expected.detach().float().cpu()
    actual_float = actual.detach().float().cpu()
    if expected_float.shape != actual_float.shape:
        raise AssertionError(f"Shape mismatch: CPU={tuple(expected_float.shape)}, NPU={tuple(actual_float.shape)}.")
    if not bool(torch.isfinite(expected_float).all()):
        raise AssertionError("CPU oracle produced a non-finite tensor.")
    if not bool(torch.isfinite(actual_float).all()):
        raise AssertionError("Ascend eager oracle produced a non-finite tensor.")
    difference = (actual_float - expected_float).abs()
    denominator = expected_float.abs().clamp_min(1e-12)
    close = torch.isclose(actual_float, expected_float, rtol=rtol, atol=atol)
    result = {
        "allclose": bool(close.all()),
        "mismatch_count": int((~close).sum()),
        "numel": expected_float.numel(),
        "max_abs_error": float(difference.max()) if difference.numel() else 0.0,
        "max_rel_error": float((difference / denominator).max()) if difference.numel() else 0.0,
        "cpu": tensor_summary(expected_float),
        "npu": tensor_summary(actual_float),
    }
    if not result["allclose"]:
        raise AssertionError(
            f"CPU/NPU mismatch: count={result['mismatch_count']}/{result['numel']}, "
            f"max_abs={result['max_abs_error']}, max_rel={result['max_rel_error']}, "
            f"rtol={rtol}, atol={atol}."
        )
    return result


def _projection_pair(
    activation_cpu: torch.Tensor,
    payload_cpu: dict[str, torch.Tensor],
    spec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cpu = vq2a8_repacked_matmul_reference(
        activation_cpu,
        payload_cpu,
        spec,
        compute_dtype=torch.float32,
        dynamic_a8=True,
    ).to(torch.bfloat16)
    payload_npu = {name: tensor.to(device=device) for name, tensor in payload_cpu.items()}
    activation_npu = activation_cpu.to(device=device)
    npu = vq2a8_repacked_matmul_reference(
        activation_npu,
        payload_npu,
        spec,
        compute_dtype=torch.float32,
        dynamic_a8=True,
    ).to(torch.bfloat16)
    torch.npu.synchronize()
    return cpu, npu


def run_probe(
    artifact: VQ2TP1Artifact,
    probe: Probe,
    probe_index: int,
    device: torch.device,
    swiglu_limit: float | None,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    layer = artifact.layer(probe.layer_index)
    if probe.expert_id not in layer.expert_ids:
        raise ValueError(
            f"Probe {probe.layer_index}:{probe.expert_id} is invalid; layer has expert IDs "
            f"[{layer.expert_ids[0]}, {layer.expert_ids[-1]}]."
        )
    gate_payload, gate_spec = artifact.load_expert(probe.layer_index, probe.expert_id, "gate_up")
    down_payload, down_spec = artifact.load_expert(probe.layer_index, probe.expert_id, "down")
    activation_cpu = deterministic_activation(gate_spec.rht_true_columns, probe_index)

    print(f"PROBE {probe.layer_index}:{probe.expert_id} stage=gate_up", flush=True)
    gate_cpu, gate_npu = _projection_pair(activation_cpu, gate_payload, gate_spec, device)
    gate_comparison = comparison_summary(gate_cpu, gate_npu, rtol=rtol, atol=atol)

    print(f"PROBE {probe.layer_index}:{probe.expert_id} stage=swiglu", flush=True)
    activated_cpu = deepseek_v4_swiglu_reference(gate_cpu, swiglu_limit)
    activated_npu = deepseek_v4_swiglu_reference(gate_npu, swiglu_limit)
    torch.npu.synchronize()
    activation_comparison = comparison_summary(activated_cpu, activated_npu, rtol=rtol, atol=atol)

    print(f"PROBE {probe.layer_index}:{probe.expert_id} stage=down", flush=True)
    down_cpu = vq2a8_repacked_matmul_reference(
        activated_cpu,
        down_payload,
        down_spec,
        compute_dtype=torch.float32,
        dynamic_a8=True,
    ).to(torch.bfloat16)
    down_payload_npu = {name: tensor.to(device=device) for name, tensor in down_payload.items()}
    down_npu = vq2a8_repacked_matmul_reference(
        activated_npu,
        down_payload_npu,
        down_spec,
        compute_dtype=torch.float32,
        dynamic_a8=True,
    ).to(torch.bfloat16)
    torch.npu.synchronize()
    down_comparison = comparison_summary(down_cpu, down_npu, rtol=rtol, atol=atol)

    result = {
        "probe": f"{probe.layer_index}:{probe.expert_id}",
        "gate_up": gate_comparison,
        "swiglu": activation_comparison,
        "down": down_comparison,
        "elapsed_seconds": time.perf_counter() - started,
    }
    del gate_payload, down_payload, gate_cpu, gate_npu, activated_cpu, activated_npu, down_cpu, down_npu
    del down_payload_npu
    gc.collect()
    torch.npu.empty_cache()
    print("PROBE_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def _load_model_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Model config must contain an object: {path}.")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Model directory containing config.json.")
    parser.add_argument(
        "--artifact",
        type=Path,
        help="TP1 artifact root; default: MODEL/experts_vq_ascend_v2.",
    )
    parser.add_argument(
        "--probes",
        type=parse_probes,
        default=parse_probes("0:0,3:0,3:127,3:255,42:255"),
        help="Comma-separated layer:expert pairs.",
    )
    parser.add_argument("--device", default="npu:0", help="Logical device; the gate requires npu:0.")
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument(
        "--verify-tensor-hashes",
        action="store_true",
        help="Re-hash every layer tensor file (about 62 GiB for the reference model).",
    )
    parser.add_argument(
        "--require-reference-identity",
        action="store_true",
        help="Require the externally verified reference producer identity.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    model = args.model.expanduser().resolve(strict=True)
    config_path = model / "config.json"
    artifact_path = (
        args.artifact.expanduser().resolve(strict=True)
        if args.artifact is not None
        else (model / "experts_vq_ascend_v2").resolve(strict=True)
    )
    device = torch.device(args.device)
    if device.type != "npu" or device.index not in (None, 0):
        raise ValueError(f"The isolated Ascend 950 gate requires logical npu:0, got {device}.")

    import torch_npu

    if not torch.npu.is_available() or torch.npu.device_count() != 1:
        raise RuntimeError(
            "Exactly one logical NPU must be visible. Set ASCEND_RT_VISIBLE_DEVICES to one physical device."
        )
    torch.npu.set_device(0)
    soc_version = torch_npu.npu.get_soc_version()
    device_name = torch.npu.get_device_name(0)
    if soc_version != 260 or "Ascend950" not in device_name:
        raise RuntimeError(f"Expected Ascend 950 (SoC 260), got soc={soc_version!r}, name={device_name!r}.")

    print(
        "DEVICE "
        + json.dumps(
            {
                "logical": "npu:0",
                "name": device_name,
                "soc": soc_version,
                "properties": str(torch.npu.get_device_properties(0)),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("ARTIFACT stage=validate_all_layer_headers", flush=True)
    artifact = open_vq2a8_tp1_artifact(
        artifact_path,
        config_path,
        require_complete=True,
        require_reference_identity=args.require_reference_identity,
        verify_tensor_hashes=args.verify_tensor_hashes,
    )
    print(
        "ARTIFACT_RESULT "
        + json.dumps(
            {
                "complete": artifact.manifest["complete"],
                "format": artifact.manifest["format"],
                "layers": len(artifact.layers),
                "model_config": str(config_path),
                "root": str(artifact.root),
                "tensor_hashes_verified": args.verify_tensor_hashes,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    model_config = _load_model_config(config_path)
    swiglu_limit = model_config.get("swiglu_limit")
    if swiglu_limit is not None:
        swiglu_limit = float(swiglu_limit)

    results = [
        run_probe(
            artifact,
            probe,
            index,
            torch.device("npu:0"),
            swiglu_limit,
            rtol=args.rtol,
            atol=args.atol,
        )
        for index, probe in enumerate(args.probes)
    ]
    print(
        "VQ2A8_TP1_ASCEND950_RUNTIME_GATE=PASS "
        + json.dumps(
            {
                "artifact": str(artifact.root),
                "probes": [result["probe"] for result in results],
                "swiglu_limit": swiglu_limit,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
