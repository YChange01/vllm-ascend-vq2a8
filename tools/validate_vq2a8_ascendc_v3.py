#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Short hash-bound v3 operator preflight, before full-resident model allocation."""

from __future__ import annotations

# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import math
from pathlib import Path

from tools.build_vq2a8_ascendc_v3 import ABI_VERSION, LIBRARY_NAME, REPO, sha256, source_hashes
from tools.validate_vq2a8_ascendc_v2 import model_identity as base_model_identity
from tools.vq2a8_baseline import capture_input_identity

SCHEMA_VERSION = 1
RELATIVE_L2_LIMIT = 0.03
SYNTHETIC = ((1,), (1, 3, 17, 32))
REAL_ROWS = (1, 3)
REAL_CASES = ("deterministic", "zero", "impulse")


def model_identity(model):
    model = Path(model).resolve(strict=True)
    artifact = model / "experts_vq_ascend_v2"
    payloads = list(model.glob("*.safetensors")) + list(artifact.rglob("*.safetensors"))
    return {
        **base_model_identity(model),
        "input_metadata": capture_input_identity(model, artifact),
        "payload_stat_only": {
            path.relative_to(model).as_posix(): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in sorted(payloads)
        },
    }


def python_source_hashes():
    paths = [
        "tools/build_vq2a8_ascendc_v3.py",
        "tools/build_vq2a8_ascendc_v2.py",
        "tools/validate_vq2a8_ascendc_v3.py",
        "tools/validate_vq2a8_ascendc_v2.py",
        "tools/benchmark_vq2a8_ascendc_v3.py",
        "tools/validate_vq2a8_phase4_kernel.py",
        "tools/validate_vq2a8_tp1_packed_kernel.py",
        "tools/benchmark_vq2a8_offline.py",
        "tools/validate_vq2a8_tp1_offline.py",
        "tools/build_vq2a8_ascendc.py",
        "tools/validate_vq2a8_ascendc.py",
        "tools/vq2a8_baseline.py",
        "tools/vq2a8_perf_report.py",
        "tools/vq2a8_v3_progress.py",
        "vllm_ascend/quantization/vq2a8_ascendc_v3.py",
        "vllm_ascend/quantization/vq2a8_ascendc.py",
        "vllm_ascend/quantization/vq2a8_activation.py",
        "vllm_ascend/quantization/vq2a8_execution.py",
        "vllm_ascend/quantization/vq2a8_offline.py",
        "vllm_ascend/quantization/vq2a8_execution_v3.py",
        "vllm_ascend/quantization/vq2a8_reference.py",
        "vllm_ascend/quantization/vq2a8_runtime.py",
        "vllm_ascend/quantization/vq2a8_optimization.py",
        "vllm_ascend/quantization/vq2a8_moe.py",
        "vllm_ascend/patch/worker/vq2a8_offline_model.py",
    ]
    return {name: sha256(REPO / name) for name in paths}


def library_identity(path):
    path = Path(path).resolve(strict=True)
    manifest = json.loads((path.parent / "build-manifest.json").read_text(encoding="utf-8"))
    digest = sha256(path)
    if (
        path.name != LIBRARY_NAME
        or manifest.get("status") != "built"
        or manifest.get("implementation") != "ascendc_v3"
        or manifest.get("abi_version") != ABI_VERSION
        or manifest.get("library_sha256") != digest
        or manifest.get("source_sha256") != source_hashes()
        or manifest.get("build_tool_sha256") != sha256(REPO / "tools/build_vq2a8_ascendc_v3.py")
    ):
        raise ValueError("V3 library/build manifest is missing, stale or mismatched; rebuild v3.")
    return {"path": str(path), "sha256": digest, "namespace": "vq2a8_ascendc_v3", "abi_version": ABI_VERSION}


def expected_cases():
    return (
        {f"synthetic:k{k}:g{len(rows)}" for k in (512, 2048) for rows in SYNTHETIC}
        | {f"real:{kind}:m{m}:{case}" for kind in ("gate_up", "down") for m in REAL_ROWS for case in REAL_CASES}
        | {f"prepared:k{k}:g{jobs}" for k in (512, 2048, 4096) for jobs in (1, 6)}
    )


def validate_receipt(report, identity, model, physical_npu):
    records = report.get("cases", {})
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("implementation") != "ascendc_v3"
        or report.get("status") != "PASS"
        or report.get("device_execution_verified") is not True
        or report.get("physical_runtime") is not True
        or report.get("library") != identity
        or report.get("model") != model_identity(model)
        or report.get("python_source_sha256") != python_source_hashes()
        or not isinstance(physical_npu, str)
        or not physical_npu.isdecimal()
        or report.get("physical_npu") != physical_npu
        or not str(report.get("soc", "")).startswith("Ascend950")
        or not isinstance(records, dict)
        or set(records) != expected_cases()
    ):
        raise ValueError("V3 preflight incomplete, stale or for another device/model/library.")
    for key, record in records.items():
        exact = key.startswith(("synthetic:", "prepared:")) or key.endswith(":zero")
        count = int(key.rsplit("g", 1)[1]) if key.startswith(("synthetic:", "prepared:")) else 1
        if (
            any(record.get(k) is not True for k in ("passed", "repeat_exact", "grouped_exact", "current_stream_exact"))
            or record.get("oracle_exact_required") is not exact
            or len(record.get("oracle", [])) != count
        ):
            raise ValueError(f"Incomplete projection evidence: {key}")
        if (key.startswith("prepared:") or ":m1:" in key) and record.get("prepared_exact") is not True:
            raise ValueError(f"Missing prepared out-ABI evidence: {key}")
        for metric in record["oracle"]:
            if (
                metric.get("allclose") is not True
                or metric.get("mismatch_count") != 0
                or any(
                    type(metric.get(k)) not in (int, float) or not math.isfinite(metric[k]) or metric[k] < 0
                    for k in ("max_abs_error", "relative_l2_error")
                )
                or metric["relative_l2_error"] > RELATIVE_L2_LIMIT
                or (exact and metric["max_abs_error"] != 0)
            ):
                raise ValueError(f"Failed oracle evidence: {key}")


def checked_model_preflight(library, receipt, model):
    identity = library_identity(library)
    report = json.loads(Path(receipt).read_text(encoding="utf-8"))
    validate_receipt(report, identity, model, os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
    return identity


def check_projection(inputs, golden, exact):
    import torch

    from tools.validate_vq2a8_phase4_kernel import bitwise_equal, compare
    from vllm_ascend.quantization.vq2a8_ascendc_v3 import grouped_projection_v3

    actual = grouped_projection_v3(inputs)
    torch.npu.synchronize()
    if len(actual) != len(golden):
        raise ValueError("Grouped output count mismatch")
    metrics = [compare(g, a) for g, a in zip(golden, actual)]
    if exact and not all(bitwise_equal(a, g) for a, g in zip(actual, golden)):
        raise ValueError("Exact synthetic/zero oracle mismatch")
    repeated = grouped_projection_v3(inputs)
    isolated = [grouped_projection_v3([row])[0] for row in inputs]
    ready, stream = torch.npu.Event(), torch.npu.Stream()
    ready.record()
    with torch.npu.stream(stream):
        stream.wait_event(ready)
        alternate = grouped_projection_v3(inputs)
    stream.synchronize()
    if any(
        not all(bitwise_equal(a, b) for a, b in zip(actual, outputs)) for outputs in (repeated, isolated, alternate)
    ):
        raise ValueError("Repeat, grouped-isolated or current-stream mismatch")
    return dict(
        passed=True,
        repeat_exact=True,
        grouped_exact=True,
        current_stream_exact=True,
        oracle_exact_required=exact,
        oracle=metrics,
    )


def check_prepared(inputs):
    """Exercise the new fixed-M1 constants/descriptors/output ABI, not only legacy glue."""
    import torch

    from tools.validate_vq2a8_phase4_kernel import bitwise_equal
    from vllm_ascend.quantization.vq2a8_ascendc_v3 import grouped_projection_out, grouped_projection_v3, make_constants

    jobs, k, n, tiles = len(inputs), inputs[0][0].shape[1], inputs[0][3].shape[0] * 2, inputs[0][4].shape[0]
    if any(row[0].shape[0] != 1 for row in inputs):
        raise ValueError("Prepared preflight is fixed M1")
    outputs = [torch.empty((1, n), dtype=torch.bfloat16, device=inputs[0][0].device) for _ in inputs]
    records = [
        [*(value.data_ptr() for value in row), out.data_ptr(), 1, n, k, tiles, 0] for row, out in zip(inputs, outputs)
    ]
    descriptors = torch.tensor(records, dtype=torch.int64, device="cpu").to(inputs[0][0].device)
    constants = make_constants(inputs[0][0])
    owners = [value for row in inputs for value in row] + outputs
    expected = grouped_projection_v3(inputs)

    def launch(pipeline=False):
        grouped_projection_out(descriptors, constants, owners, jobs=jobs, m=1, n=n, k=k, tiles=tiles, pipeline=pipeline)

    for pipeline in (False, True, False):
        launch(pipeline)
        torch.npu.synchronize()
        if not all(bitwise_equal(a, b) for a, b in zip(outputs, expected)):
            raise ValueError("Prepared out-ABI differs from legacy projection / repeat / pipeline")
    ready, stream = torch.npu.Event(), torch.npu.Stream()
    ready.record()
    with torch.npu.stream(stream):
        stream.wait_event(ready)
        launch()
    stream.synchronize()
    if not all(bitwise_equal(a, b) for a, b in zip(outputs, expected)):
        raise ValueError("Prepared out-ABI current-stream mismatch")
    return True


def run(args):
    from tools.validate_vq2a8_ascendc import require_hardware_runtime

    require_hardware_runtime()
    import torch
    import torch_npu  # noqa: F401

    from tools.validate_vq2a8_phase4_kernel import (
        prepare_rows,
        same_fp8_oracle,
        synthetic_dense_oracle,
        synthetic_inputs,
    )
    from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device, activation_case
    from vllm_ascend.quantization.vq2a8_ascendc_v3 import load_pinned_library
    from vllm_ascend.quantization.vq2a8_reference import decode_repacked_vq2a8_codebook_weight
    from vllm_ascend.quantization.vq2a8_runtime import open_vq2a8_tp1_artifact

    report = dict(
        schema_version=SCHEMA_VERSION,
        implementation="ascendc_v3",
        status="RUNNING",
        cases={},
        device_execution_verified=False,
        physical_runtime=False,
        full_model_graph_verified=False,
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    try:
        identity = library_identity(args.library)
        report.update(library=identity, model=model_identity(args.model), python_source_sha256=python_source_hashes())
        device = torch.device("npu:0")
        report["device"] = _initialize_device(device)
        report["soc"] = torch.npu.get_device_name(0)
        manifest = json.loads((args.library.parent / "build-manifest.json").read_text(encoding="utf-8"))
        if report["soc"] != manifest.get("soc"):
            raise ValueError("V3 library built for a different exact SoC")
        require_hardware_runtime()
        report.update(physical_runtime=True, physical_npu=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
        load_pinned_library(identity["path"], identity["sha256"])
        for k in (512, 2048):
            for rows in SYNTHETIC:
                key = f"synthetic:k{k}:g{len(rows)}"
                print(f"V3_PREFLIGHT_START={key}", flush=True)
                native, golden = [], []
                for offset, m in enumerate(rows):
                    x, scale, bias, words, books, ids = synthetic_inputs(m, 64, k, k // 256)
                    x = ((torch.arange(m * k).reshape(m, k) * 7 + offset) % 5 - 2).to(torch.float8_e4m3fn)
                    books = ((torch.arange(books.numel()).reshape(books.shape) * 3 + offset) % 7 - 3).to(
                        torch.float8_e4m3fn
                    )
                    scale = ((torch.arange(m) % 5 - 2).float() / 8).contiguous()
                    bias = ((torch.arange(m) + offset) % 7 - 3).float() / 2
                    values = (x, scale, bias, words, books, ids)
                    golden.append(same_fp8_oracle(values[:3], synthetic_dense_oracle(*values[3:])))
                    native.append(tuple(t.to(device) for t in values))
                report["cases"][key] = check_projection(native, golden, True)
                save()
                print(f"V3_PREFLIGHT_PASS={key}", flush=True)
        for k in (512, 2048, 4096):
            for jobs in (1, 6):
                key = f"prepared:k{k}:g{jobs}"
                print(f"V3_PREFLIGHT_START={key}", flush=True)
                native, golden = [], []
                for offset in range(jobs):
                    x, scale, bias, words, books, ids = synthetic_inputs(1, 64, k, k // 256)
                    x = ((torch.arange(k).reshape(1, k) + offset) % 5 - 2).to(torch.float8_e4m3fn)
                    books = ((torch.arange(books.numel()).reshape(books.shape) + offset) % 7 - 3).to(
                        torch.float8_e4m3fn
                    )
                    scale, bias = torch.tensor([0.125]), torch.tensor([0.5])
                    values = (x, scale, bias, words, books, ids)
                    native.append(tuple(value.to(device) for value in values))
                    golden.append(same_fp8_oracle(values[:3], synthetic_dense_oracle(*values[3:])))
                record = check_projection(native, golden, True)
                record["prepared_exact"] = check_prepared(native)
                report["cases"][key] = record
                save()
                print(f"V3_PREFLIGHT_PASS={key}", flush=True)
        artifact = open_vq2a8_tp1_artifact(
            args.model / "experts_vq_ascend_v2",
            args.model / "config.json",
            require_complete=True,
            require_reference_identity=True,
        )
        for kind in ("gate_up", "down"):
            host, spec = artifact.load_expert(3, 0, kind)
            dense = decode_repacked_vq2a8_codebook_weight(host, spec, compute_dtype=torch.float64)
            payload = {name: value.to(device) for name, value in host.items()}
            packed = tuple(payload[name] for name in ("packed_indices", "codebooks", "codebook_tile_ids"))
            for m in REAL_ROWS:
                for case in REAL_CASES:
                    key = f"real:{kind}:m{m}:{case}"
                    print(f"V3_PREFLIGHT_START={key}", flush=True)
                    hidden = torch.cat([activation_case(spec.rht_true_columns, i, 0, case) for i in range(m)]).to(
                        device
                    )
                    prepared = prepare_rows(hidden, payload, spec)
                    report["cases"][key] = check_projection(
                        [(*prepared, *packed)], [same_fp8_oracle(prepared, dense)], case == "zero"
                    )
                    if m == 1:
                        report["cases"][key]["prepared_exact"] = check_prepared([(*prepared, *packed)])
                    save()
                    print(f"V3_PREFLIGHT_PASS={key}", flush=True)
        report.update(status="PASS", device_execution_verified=True)
        save()
        checked_model_preflight(args.library, args.output, args.model)
    except Exception as exc:
        report.update(status="FAIL", device_execution_verified=False, error=f"{type(exc).__name__}: {exc}")
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "library", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(dict(scope="plan_only_no_device_execution", cases=sorted(expected_cases())), indent=2))
        return 0
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("--output must be a new file in an existing directory")
    args.library, args.model = args.library.resolve(strict=True), args.model.resolve(strict=True)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
