#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VQ2A8 v2 candidate preflight. Run through accept_vq2a8_ascendc_v2.py on a real NPU.

Synthetic tests require exact BF16 results. Real prepared projections retain
the existing 0.01/0.001 oracle tolerance; this is NOT model quality acceptance.
No old native library or dense real weight is used on the device.
"""

from __future__ import annotations

# ruff: noqa: E402
import os as _bootstrap_os
import sys as _bootstrap_sys

if not __package__:
    _bootstrap_sys.path[0] = _bootstrap_os.path.dirname(
        _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))
    )

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
RELATIVE_L2_LIMIT = 0.03  # Same existing packed-kernel gate, not a relaxed v2 threshold.
SYNTHETIC_ROWS = ((1,), (32,), (1, 2, 15, 16, 17, 31))
REAL_ROWS = (1, 3, 32)
REAL_CASES = ("deterministic", "zero", "impulse")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def expected_cases():
    synthetic = {f"synthetic:k{k}:g{len(rows)}:m{rows[0]}" for k in (2048, 4096) for rows in SYNTHETIC_ROWS}
    real = {f"real:{kind}:m{m}:{case}" for kind in ("gate_up", "down") for m in REAL_ROWS for case in REAL_CASES}
    return synthetic | real


def python_source_hashes():
    paths = (
        "tools/validate_vq2a8_ascendc_v2.py",
        "tools/validate_vq2a8_tp1_offline.py",
        "tools/validate_vq2a8_phase4_kernel.py",
        "tools/validate_vq2a8_tp1_packed_kernel.py",
        "vllm_ascend/quantization/vq2a8_ascendc_v2.py",
        "vllm_ascend/quantization/vq2a8_activation.py",
        "vllm_ascend/quantization/vq2a8_execution.py",
        "vllm_ascend/quantization/vq2a8_offline.py",
        "vllm_ascend/quantization/vq2a8_optimization.py",
        "vllm_ascend/quantization/vq2a8_reference.py",
        "vllm_ascend/quantization/vq2a8_runtime.py",
        "vllm_ascend/quantization/vq2a8_repack.py",
        "vllm_ascend/quantization/vq2a8_moe.py",
        "vllm_ascend/patch/worker/vq2a8_offline_model.py",
    )
    return {name: sha256(REPO / name) for name in paths}


def model_identity(model):
    model = Path(model).resolve(strict=True)
    return {
        "path": str(model),
        "config_sha256": sha256(model / "config.json"),
        "artifact_manifest_sha256": sha256(model / "experts_vq_ascend_v2/manifest.json"),
    }


def validate_receipt(report, identity, model, physical_npu):
    """Host-only receipt checks, shared by supervisor and model child."""
    records = report.get("cases", {})
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("status") != "passed"
        or report.get("implementation") != "ascendc_v2"
        or report.get("device_execution_verified") is not True
        or report.get("physical_runtime") is not True
        or report.get("library") != identity
        or report.get("python_source_sha256") != python_source_hashes()
        or report.get("model") != model_identity(model)
        or not isinstance(physical_npu, str)
        or not physical_npu.isdecimal()
        or report.get("physical_npu") != physical_npu
        or not str(report.get("soc", "")).startswith("Ascend950")
        or not isinstance(records, dict)
        or set(records) != expected_cases()
        or any(
            not isinstance(r, dict) or r.get("passed") is not True or r.get("repeat_exact") is not True
            for r in records.values()
        )
    ):
        raise ValueError("VQ2A8 v2 preflight is incomplete, stale, or belongs to another model/device/library.")
    for key, record in records.items():
        exact = key.startswith("synthetic:") or key.endswith(":zero")
        oracle = record.get("oracle")
        group_rows = (
            next(rows for rows in SYNTHETIC_ROWS if f":g{len(rows)}:m{rows[0]}" in key)
            if key.startswith("synthetic:")
            else (int(key.split(":")[2][1:]),)
        )
        if (
            record.get("grouped_exact") is not True
            or record.get("current_stream_exact") is not True
            or record.get("oracle_exact_required") is not exact
            or not isinstance(oracle, list)
            or len(oracle) != len(group_rows)
        ):
            raise ValueError(f"Incomplete projection checks: {key}.")
        for metrics, rows in zip(oracle, group_rows):
            if (
                not isinstance(metrics, dict)
                or metrics.get("allclose") is not True
                or metrics.get("mismatch_count") != 0
                or metrics.get("numel") != rows * 4096
                or any(
                    type(metrics.get(name)) not in (float, int) or not math.isfinite(metrics[name]) or metrics[name] < 0
                    for name in ("max_abs_error", "relative_l2_error")
                )
                or metrics["relative_l2_error"] > RELATIVE_L2_LIMIT
                or (exact and metrics["max_abs_error"] != 0)
            ):
                raise ValueError(f"Invalid oracle evidence: {key}.")


def checked_model_preflight(library, receipt, model):
    """No stale/simulator/partial receipt may authorize full model allocation."""
    from vllm_ascend.quantization.vq2a8_ascendc_v2 import validate_build_manifest

    identity = validate_build_manifest(library, sha256(library))
    report = json.loads(Path(receipt).read_text(encoding="utf-8"))
    validate_receipt(report, identity, model, os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
    return identity


def _synthetic(rows, k, offset):
    """Integers and binary scales keep the complete FP32 reduction exact."""
    import torch

    from tools.validate_vq2a8_phase4_kernel import synthetic_inputs

    x, scale, bias, words, books, _ = synthetic_inputs(rows, 4096, k, k // 256)
    ids = (torch.arange(k) % (k // 256)).byte()
    x = ((torch.arange(rows * k).reshape(rows, k) * 7 + offset) % 5 - 2).to(torch.float8_e4m3fn)
    books = ((torch.arange(books.numel()).reshape(books.shape) * 3 + offset) % 7 - 3).to(torch.float8_e4m3fn)
    scale = ((torch.arange(rows) % 5 - 2).float() / 8).contiguous()
    bias = ((torch.arange(rows) + offset) % 7 - 3).float() / 2
    return x, scale, bias, words, books, ids


def _convert_inputs(inputs, device):
    from types import SimpleNamespace

    import torch

    from vllm_ascend.quantization.vq2a8_ascendc_v2 import convert_expert_payload, gather_prepared_activation

    x, scale, bias, words, books, ids = inputs
    k = x.shape[1]
    payload = {
        "packed_indices": words,
        "codebooks": books,
        "codebook_tile_ids": ids,
        "weight_scale": torch.ones(k),
        "weight_bias": torch.zeros(k),
        "rht_sign": torch.ones(k, dtype=torch.int8),
    }
    converted = convert_expert_payload(payload, SimpleNamespace(columns=k, rht_true_columns=k))
    q = gather_prepared_activation(x.to(device), converted["activation_order"].to(device))
    return q, scale.to(device), bias.to(device), converted["packed_zn"].to(device), converted["pair_lut"].to(device)


def _check(inputs, expected, *, exact):
    import torch

    from tools.validate_vq2a8_phase4_kernel import bitwise_equal, compare
    from vllm_ascend.quantization.vq2a8_ascendc_v2 import grouped_projection

    actual = grouped_projection(inputs)
    torch.npu.synchronize()
    if len(actual) != len(expected):
        raise AssertionError("Native output list length differs from submitted jobs.")
    metrics = []
    for output, golden in zip(actual, expected):
        if output.dtype != torch.bfloat16 or output.shape != golden.shape:
            raise AssertionError("Native output shape/dtype mismatch.")
        metrics.append(compare(golden, output))
        if exact and not bitwise_equal(output, golden):
            raise AssertionError("Exact integer/binary-scale synthetic oracle mismatch.")
    repeated = grouped_projection(inputs)
    if not all(bitwise_equal(a, b) for a, b in zip(actual, repeated)):
        raise AssertionError("VQ2A8 v2 candidate is not bitwise repeatable.")
    # Exercise descriptor lifetime, row tails and the same tensors on a
    # non-default stream, ordered by explicit events rather than global device.
    ready = torch.npu.Event()
    ready.record()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        stream.wait_event(ready)
        alternate = grouped_projection(inputs)
    stream.synchronize()
    if not all(bitwise_equal(a, b) for a, b in zip(actual, alternate)):
        raise AssertionError("Native current-stream ordering mismatch.")
    for values, output in zip(inputs, actual):
        separate = grouped_projection([values])[0]
        if not bitwise_equal(output, separate):
            raise AssertionError("Grouped dispatch differs from isolated job output.")
    return {
        "passed": True,
        "repeat_exact": True,
        "grouped_exact": True,
        "current_stream_exact": True,
        "oracle_exact_required": exact,
        "oracle": metrics,
    }


def run(args):
    import torch
    import torch_npu  # noqa: F401

    from tools.validate_vq2a8_ascendc import require_hardware_runtime
    from tools.validate_vq2a8_phase4_kernel import prepare_rows, same_fp8_oracle, synthetic_dense_oracle
    from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device, activation_case
    from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
    from vllm_ascend.quantization.vq2a8_ascendc_v2 import (
        convert_expert_payload,
        gather_prepared_activation,
        load_pinned_library,
    )
    from vllm_ascend.quantization.vq2a8_reference import decode_repacked_vq2a8_codebook_weight
    from vllm_ascend.quantization.vq2a8_runtime import open_vq2a8_tp1_artifact

    report = {
        "schema_version": SCHEMA_VERSION,
        "implementation": "ascendc_v2",
        "status": "running",
        "device_execution_verified": False,
        "physical_runtime": False,
        "cases": {},
        "model_integration_verified": False,
        "quality_verified": False,
        "serving_verified": False,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def emit(key, value):
        report["cases"][key] = value
        save()
        print(f"ASCENDC_V2_PREFLIGHT_CASE={key} PASS", flush=True)

    save()
    try:
        require_hardware_runtime()
        device = torch.device("npu:0")
        report["device"] = _initialize_device(device)
        report["soc"] = torch.npu.get_device_name(0)
        require_hardware_runtime()
        report.update(
            physical_runtime=True,
            physical_npu=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            model=model_identity(args.model),
            python_source_sha256=python_source_hashes(),
        )
        report["library"] = load_pinned_library(args.library, sha256(args.library))
        build = json.loads((args.library.parent / "build-manifest.json").read_text(encoding="utf-8"))
        if build.get("soc") != report["soc"]:
            raise ValueError("V2 library was compiled for a different exact SoC; rebuild for this device.")
        for k in (2048, 4096):
            for rows in SYNTHETIC_ROWS:
                key = f"synthetic:k{k}:g{len(rows)}:m{rows[0]}"
                print(f"ASCENDC_V2_PREFLIGHT_START={key}", flush=True)
                native, golden = [], []
                for index, m in enumerate(rows):
                    inputs = _synthetic(m, k, index)
                    dense = synthetic_dense_oracle(*inputs[3:])
                    golden.append(same_fp8_oracle(inputs[:3], dense))
                    native.append(_convert_inputs(inputs, device))
                emit(key, _check(native, golden, exact=True))
                del native, golden, dense, inputs
        artifact = open_vq2a8_tp1_artifact(
            args.model / "experts_vq_ascend_v2",
            args.model / "config.json",
            require_complete=True,
            require_reference_identity=True,
        )
        for kind in ("gate_up", "down"):
            host, spec = artifact.load_expert(3, 0, kind)
            dense = decode_repacked_vq2a8_codebook_weight(host, spec, compute_dtype=torch.float64)
            converted = {name: value.to(device) for name, value in convert_expert_payload(host, spec).items()}
            original = {name: value.to(device) for name, value in host.items()}
            for m in REAL_ROWS:
                for case in REAL_CASES:
                    key = f"real:{kind}:m{m}:{case}"
                    print(f"ASCENDC_V2_PREFLIGHT_START={key}", flush=True)
                    hidden = torch.cat([activation_case(spec.rht_true_columns, i, 0, case) for i in range(m)]).to(
                        device
                    )
                    prepared = RowwiseVQ2A8Preparation().rows(hidden, converted, spec)
                    reference = prepare_rows(hidden, original, spec)
                    from tools.validate_vq2a8_phase4_kernel import bitwise_equal

                    if not all(bitwise_equal(a, b) for a, b in zip(prepared, reference)):
                        raise AssertionError("Expert preparation differs before K gathering.")
                    q, scale, bias = prepared
                    inputs = (
                        gather_prepared_activation(q, converted["activation_order"]),
                        scale,
                        bias,
                        converted["packed_zn"],
                        converted["pair_lut"],
                    )
                    emit(key, _check([inputs], [same_fp8_oracle(reference, dense)], exact=case == "zero"))
            del dense, converted, original, host
        report.update(status="passed", device_execution_verified=True)
        save()
        checked_model_preflight(args.library, args.output, args.model)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    print(f"ASCENDC_V2_PREFLIGHT=PASS REPORT={args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("--output must be a new file in an existing report directory.")
    args.library, args.model = args.library.resolve(strict=True), args.model.resolve(strict=True)
    run(args)


if __name__ == "__main__":
    main()
