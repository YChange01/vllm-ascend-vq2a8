#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the TP1 M=1 packed VQ2A8 Triton kernel on CUDA or Ascend."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from vllm_ascend.quantization.vq2a8_reference import (
    decode_repacked_vq2a8_codebook_weight,
    deepseek_v4_swiglu_reference,
    prepare_repacked_vq2a8_activation_reference,
    vq2a8_predecoded_matmul_reference,
)
from vllm_ascend.quantization.vq2a8_runtime import (
    VQ2TP1Artifact,
    open_vq2a8_tp1_artifact,
)
from vllm_ascend.quantization.vq2a8_triton import vq2a8_tp1_m1_packed_gemm
from vllm_ascend.quantization.vq2a8_validation import (
    audit_model_storage,
    error_metrics,
    tensor_layout,
    validate_tolerances,
)

RELATIVE_L2_LIMIT = 0.03
PREPARED_INPUT_RTOL = 0.01
PREPARED_INPUT_ATOL = 0.001


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


def activation_case(width: int, probe_index: int, kind_index: int, case: str) -> torch.Tensor:
    values = deterministic_activation(width, probe_index, kind_index)
    if case == "zero":
        return torch.zeros_like(values)
    if case == "impulse":
        values.zero_()
        values[0, (width - 1)] = 1
    elif case == "small":
        values *= 1e-6
    elif case == "large":
        values *= 32
    elif case != "deterministic":
        raise ValueError(f"Unknown activation case: {case}.")
    return values


def environment_report() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "triton", "triton-ascend", "safetensors"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    repo = Path(__file__).resolve().parents[1]
    git_identity: dict[str, Any] = {}
    try:
        for key, arguments in (("head", ["rev-parse", "HEAD"]), ("status", ["status", "--porcelain"])):
            result = subprocess.run(
                ["git", *arguments], cwd=repo, capture_output=True, text=True, timeout=10, check=False
            )
            git_identity[key] = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired) as error:
        git_identity["error"] = str(error)
    source_hashes = {}
    for name in (
        "vq2a8_triton.py",
        "vq2a8_kernel_contract.py",
        "vq2a8_reference.py",
        "vq2a8_runtime.py",
        "vq2a8_repack.py",
        "vq2a8_root_fp8.py",
        "vq2a8_root_fp8_triton.py",
        "vq2a8_vector_gather.py",
        "vq2a8_phase4_micro.py",
        "vq2a8_moe.py",
        "vq2a8_offline.py",
        "vq2a8_execution.py",
    ):
        source = repo / "vllm_ascend/quantization" / name
        source_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    for name in (
        "vllm_ascend/models/deepseek_v4.py",
        "vllm_ascend/attention/dsa_v1.py",
        "vllm_ascend/ops/linear.py",
        "vllm_ascend/patch/worker/vq2a8_offline_model.py",
        "tools/validate_vq2a8_tp1_offline.py",
        "tools/validate_vq2a8_qli_metadata.py",
        "tools/validate_vq2a8_sas_attention.py",
        "csrc/attention/kv_quant_sparse_attn_sharedkv_metadata/op_kernel_aicpu/kv_quant_sparse_attn_sharedkv_metadata_aicpu.cpp",
        "csrc/attention/kv_quant_sparse_attn_sharedkv/op_kernel/kv_quant_sparse_attn_sharedkv_metadata.h",
        "csrc/attention/vllm_quant_lightning_indexer_metadata/op_kernel_aicpu/vllm_quant_lightning_indexer_metadata_aicpu.cpp",
        "tools/validate_vq2a8_tp1_moe.py",
        "tools/validate_vq2a8_tp1_acceptance.py",
        "tools/vq2a8_live_log.py",
        "tools/vq2a8_baseline.py",
        "tools/benchmark_vq2a8_host_load.py",
        "tools/validate_vq2a8_tp1_phase1.py",
        "tools/validate_vq2a8_tp1_phase3.py",
        "tools/validate_vq2a8_root_fp8.py",
        "tools/validate_vq2a8_phase4_kernel.py",
        "tools/validate_vq2a8_tp1_phase4.py",
    ):
        source_hashes[name] = hashlib.sha256((repo / name).read_bytes()).hexdigest()
    return {
        "python": sys.executable,
        "repo": str(repo),
        "git": git_identity,
        "source_sha256": source_hashes,
        "torch_runtime_version": torch.__version__,
        "cpu_threads": {"intraop": torch.get_num_threads(), "interop": torch.get_num_interop_threads()},
        "arguments": sys.argv[1:],
        "packages": packages,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_LAUNCH_BLOCKING",
                "ASCEND_HOME_PATH",
                "ASCEND_TOOLKIT_HOME",
            )
        },
        "toolkit_latest_resolved": str(Path("/usr/local/Ascend/ascend-toolkit/latest").resolve()),
    }


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
    validate_tolerances(rtol, atol)
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
        **error_metrics(expected_float, actual_float),
    }
    # A fixed atol can otherwise accept a completely wrong low-amplitude
    # vector. Also require a scale-normalized whole-vector error bound.
    result["relative_l2_limit"] = RELATIVE_L2_LIMIT
    if result["relative_l2_error"] > result["relative_l2_limit"]:
        result["allclose"] = False
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
        print("NUMERIC_FAILURE " + json.dumps(dict(result, samples=samples, rtol=rtol, atol=atol)), flush=True)
        raise AssertionError(
            f"Packed kernel mismatch: count={result['mismatch_count']}/{result['numel']}, "
            f"max_abs={result['max_abs_error']}, max_rel={result['max_rel_error']}, "
            f"relative_l2={result['relative_l2_error']}, "
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
    case: str = "deterministic",
    activation_cpu: torch.Tensor | None = None,
    activation_device: torch.Tensor | None = None,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    payload_cpu, spec = artifact.load_expert(probe.layer_index, probe.expert_id, kind)
    if activation_cpu is None:
        activation_cpu = activation_case(spec.rht_true_columns, probe_index, int(kind == "down"), case)
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

    if activation_device is None:
        activation_device = activation_padded_cpu.to(device=device)
    elif spec.rht_true_columns != spec.columns:
        activation_device = torch.nn.functional.pad(activation_device, (0, spec.columns - spec.rht_true_columns))
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
    prepared_activation = quantized.contiguous()
    packed_indices = payload_cpu["packed_indices"].to(device=device).contiguous()
    codebooks = payload_cpu["codebooks"].to(device=device).contiguous()
    codebook_tile_ids = payload_cpu["codebook_tile_ids"].to(device=device).contiguous()
    activation_scale = activation_scale.contiguous()
    bias_correction = bias_correction.contiguous()

    # Compare the packed kernel with an oracle using exactly the same prepared
    # FP8 bytes and scales. This separates decode/MAC errors from CPU/NPU RHT
    # differences crossing an FP8 rounding threshold.
    prepared_expected = prepared_activation.cpu().double() @ codebook_weight_cpu.double().T
    prepared_expected *= activation_scale.cpu().double().unsqueeze(-1)
    prepared_expected += bias_correction.cpu().double().unsqueeze(-1)
    prepared_expected = prepared_expected.to(torch.bfloat16)
    layouts = {
        name: tensor_layout(tensor)
        for name, tensor in (
            ("activation", prepared_activation),
            ("activation_scale", activation_scale),
            ("bias_correction", bias_correction),
            ("packed_indices", packed_indices),
            ("codebooks", codebooks),
            ("codebook_tile_ids", codebook_tile_ids),
        )
    }
    print(
        "KERNEL_INPUT "
        + json.dumps({"probe": f"{probe.layer_index}:{probe.expert_id}:{kind}", "case": case, "tensors": layouts}),
        flush=True,
    )

    actual = vq2a8_tp1_m1_packed_gemm(
        prepared_activation,
        activation_scale.contiguous(),
        bias_correction.contiguous(),
        packed_indices,
        codebooks,
        codebook_tile_ids,
    )
    _synchronize(device)
    prepared_comparison = _comparison_summary(
        prepared_expected, actual, rtol=min(rtol, PREPARED_INPUT_RTOL), atol=min(atol, PREPARED_INPUT_ATOL)
    )
    # Retain this evidence even if the independent end-to-end oracle fails.
    print(
        "PREPARED_INPUT_RESULT "
        + json.dumps(
            {
                "projection": f"{probe.layer_index}:{probe.expert_id}:{kind}",
                "case": case,
                "comparison": prepared_comparison,
            }
        ),
        flush=True,
    )
    comparison = _comparison_summary(
        expected, actual, rtol=0.0 if case == "zero" else rtol, atol=0.0 if case == "zero" else atol
    )

    for _ in range(warmups):
        warm_output = vq2a8_tp1_m1_packed_gemm(
            prepared_activation,
            activation_scale,
            bias_correction,
            packed_indices,
            codebooks,
            codebook_tile_ids,
        )
        _synchronize(device)
        _comparison_summary(actual, warm_output, rtol=0.0, atol=0.0)
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
        # Every repeat must match, including an intermittent bad middle run.
        determinism = _comparison_summary(actual, repeated_output, rtol=0.0, atol=0.0)

    result = {
        "projection": f"{probe.layer_index}:{probe.expert_id}:{kind}",
        "case": case,
        "shape": {"m": 1, "n": spec.rows, "k": spec.columns},
        "activation_prepare_backend": "validated_eager_dynamic_a8",
        "activation_storage_dtype": str(prepared_activation.dtype),
        "codebook_storage_dtype": str(codebooks.dtype),
        "packed_projection_backend": (
            "triton_packed_e4m3_vector_reduce_v7" if device.type == "npu" else "triton_native_e4m3_dot_v7"
        ),
        "ascend_compile_profile": ("pure_vector_no_cube_bridge" if device.type == "npu" else None),
        "native_fp8_storage": True,
        "native_fp8_dot": device.type == "cuda",
        "npu_correctness_fallback": device.type == "npu",
        "device_dense_weight_materialized": False,
        "comparison": comparison,
        "same_prepared_input_comparison": prepared_comparison,
        "determinism": determinism,
        "repeats_checked": repeats,
        "synchronized_call_ms": {
            "min": min(elapsed_ms),
            "median": statistics.median(elapsed_ms),
            "max": max(elapsed_ms),
            "repeats": repeats,
            "includes": "Python launch, allocation and synchronization; not device-event kernel timing",
        },
    }
    print("KERNEL_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    del (
        payload_cpu,
        activation_padded_cpu,
        codebook_weight_cpu,
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
        repeated_output,
    )
    gc.collect()
    if device.type == "npu":
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()
    return result, expected, actual


def run_probe_cases(artifact: VQ2TP1Artifact, probe: Probe, device: torch.device, args) -> list[dict[str, Any]]:
    results = []
    config = json.loads(artifact.model_config_path.read_text(encoding="utf-8"))
    probe_seed = probe.layer_index * artifact.model_layout.num_routed_experts + probe.expert_id
    for case in args.cases:
        kwargs = dict(warmups=args.warmups, repeats=args.repeats, rtol=args.rtol, atol=args.atol, case=case)
        print(f"KERNEL {probe.layer_index}:{probe.expert_id}:gate_up case={case}", flush=True)
        gate_result, gate_cpu, gate_device = _run_projection(artifact, probe, probe_seed, "gate_up", device, **kwargs)
        results.append(gate_result)
        print(f"KERNEL {probe.layer_index}:{probe.expert_id}:down case={case}", flush=True)
        down_result, _, _ = _run_projection(artifact, probe, probe_seed, "down", device, **kwargs)
        results.append(down_result)
        if args.chain:
            print(f"CHAIN {probe.layer_index}:{probe.expert_id} case={case} stage=swiglu", flush=True)
            activated_cpu = deepseek_v4_swiglu_reference(gate_cpu, config.get("swiglu_limit"))
            activated_device = deepseek_v4_swiglu_reference(gate_device, config.get("swiglu_limit"))
            swiglu = _comparison_summary(activated_cpu, activated_device, rtol=args.rtol, atol=args.atol)
            print(f"CHAIN {probe.layer_index}:{probe.expert_id} case={case} stage=down", flush=True)
            chain_result, _, _ = _run_projection(
                artifact,
                probe,
                probe_seed,
                "down",
                device,
                activation_cpu=activated_cpu,
                activation_device=activated_device,
                **kwargs,
            )
            chain_result = dict(chain_result, path="gate_up_swiglu_down", swiglu=swiglu)
            results.append(chain_result)
            print("CHAIN_RESULT " + json.dumps(chain_result, sort_keys=True), flush=True)
    return results


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
        properties = torch.npu.get_device_properties(0)
        free_bytes, total_bytes = torch.npu.mem_get_info()
        return {
            "type": "npu",
            "logical": "npu:0",
            "name": device_name,
            "soc": soc_version,
            "properties": str(properties),
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
        }
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
    parser.add_argument(
        "--cases", nargs="+", default=["deterministic"], choices=["deterministic", "zero", "impulse", "small", "large"]
    )
    parser.add_argument("--chain", action="store_true", help="Also validate packed gate_up -> SwiGLU -> down.")
    parser.add_argument(
        "--audit-model", action="store_true", help="Inventory storage and verify hash routing coverage."
    )
    parser.add_argument("--verify-tensor-hashes", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    validate_tolerances(args.rtol, args.atol)
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats must be positive.")
    model = args.model.expanduser().resolve(strict=True)
    artifact_path = args.artifact.expanduser().resolve(strict=True)
    device = torch.device(args.device)
    print("ENVIRONMENT " + json.dumps(environment_report(), sort_keys=True), flush=True)
    print("DEVICE " + json.dumps(_initialize_device(device), sort_keys=True), flush=True)
    artifact = open_vq2a8_tp1_artifact(
        artifact_path,
        model / "config.json",
        require_complete=not args.allow_partial_artifact,
        require_reference_identity=True,
        verify_tensor_hashes=args.verify_tensor_hashes,
    )
    print(
        "ARTIFACT_RESULT "
        + json.dumps(
            {
                "complete": artifact.manifest["complete"],
                "format": artifact.manifest["format"],
                "layers": len(artifact.layers),
                "root": str(artifact.root),
                "tensor_hashes_verified": args.verify_tensor_hashes,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if args.audit_model:
        print("MODEL_AUDIT " + json.dumps(audit_model_storage(artifact), sort_keys=True), flush=True)
    results: list[dict[str, Any]] = []
    for probe in args.probes:
        layer = artifact.layer(probe.layer_index)
        if probe.expert_id not in layer.expert_ids:
            raise ValueError(f"Artifact has no probe {probe.layer_index}:{probe.expert_id}.")
        results.extend(run_probe_cases(artifact, probe, device, args))

    print(
        "VQ2A8_TP1_M1_PACKED_KERNEL_GATE=PASS "
        + json.dumps(
            {
                "artifact": str(artifact.root),
                "activation_dtype": "torch.float8_e4m3fn",
                "codebook_dtype": "torch.float8_e4m3fn",
                "device": str(device),
                "projections": [result["projection"] for result in results],
                "dense_weight_on_device": False,
                "native_fp8_storage": True,
                "native_fp8_dot": device.type == "cuda",
                "npu_correctness_fallback": device.type == "npu",
                "cases": args.cases,
                "expert_chain_verified": args.chain,
                "model_audit_performed": args.audit_model,
                "tensor_hashes_verified": args.verify_tensor_hashes,
                "serving_integration_verified": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
