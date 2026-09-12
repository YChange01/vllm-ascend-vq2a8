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

from tools.build_vq2a8_ascendc_v3 import (
    ABI_VERSION,
    LIBRARY_NAME,
    REPO,
    RESIDENT_ABI_VERSION,
    RESIDENT_CAPABILITIES_REQUIRED,
    RESIDENT_KERNEL,
    RESIDENT_LAYOUT,
    sha256,
    source_hashes,
)
from tools.validate_vq2a8_ascendc_v2 import (
    REAL_CASES,
    REAL_ROWS,
    SYNTHETIC_ROWS,
    _convert_inputs,
    _synthetic,
)
from tools.validate_vq2a8_ascendc_v2 import (
    expected_cases as v2_expected_cases,
)
from tools.validate_vq2a8_ascendc_v2 import (
    model_identity as base_model_identity,
)
from tools.vq2a8_baseline import capture_input_identity

SCHEMA_VERSION = 2
RELATIVE_L2_LIMIT = 0.03


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
        "tools/vq2a8_v3_prepare_check.py",
        "vllm_ascend/quantization/vq2a8_ascendc_v3.py",
        "vllm_ascend/quantization/vq2a8_ascendc_v2.py",
        "vllm_ascend/quantization/vq2a8_ascendc.py",
        "vllm_ascend/quantization/vq2a8_activation.py",
        "vllm_ascend/quantization/vq2a8_execution.py",
        "vllm_ascend/quantization/vq2a8_offline.py",
        "vllm_ascend/quantization/vq2a8_execution_v3.py",
        "vllm_ascend/quantization/vq2a8_v3_graph.py",
        "vllm_ascend/quantization/vq2a8_v3_workspace.py",
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
        or manifest.get("resident_abi_version") != RESIDENT_ABI_VERSION
        or manifest.get("resident_capabilities_required") != RESIDENT_CAPABILITIES_REQUIRED
        or manifest.get("layout") != RESIDENT_LAYOUT
        or manifest.get("resident_kernel") != RESIDENT_KERNEL
        or manifest.get("library_sha256") != digest
        or manifest.get("source_sha256") != source_hashes()
        or manifest.get("build_tool_sha256") != sha256(REPO / "tools/build_vq2a8_ascendc_v3.py")
    ):
        raise ValueError("V3 library/build manifest is missing, stale or mismatched; rebuild v3.")
    return dict(
        path=str(path),
        sha256=digest,
        namespace="vq2a8_ascendc_v3",
        abi_version=ABI_VERSION,
        resident_abi_version=RESIDENT_ABI_VERSION,
    )


def expected_cases():
    return v2_expected_cases() | {f"resident:k{k}:g{jobs}" for k in (2048, 4096) for jobs in (1, 6)}


def validate_receipt(report, identity, model, physical_npu, preparation=None):
    records = report.get("cases", {})
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("implementation") != "ascendc_v3"
        or report.get("status") != "PASS"
        or report.get("device_execution_verified") is not True
        or report.get("physical_runtime") is not True
        or report.get("resident_abi_version") != RESIDENT_ABI_VERSION
        or type(report.get("native_resident_capabilities")) is not int
        or report["native_resident_capabilities"] < 0
        or report["native_resident_capabilities"] & RESIDENT_CAPABILITIES_REQUIRED != RESIDENT_CAPABILITIES_REQUIRED
        or report.get("layout") != RESIDENT_LAYOUT
        or report.get("preparation_mode") not in ("eager", "fused")
        or (preparation is not None and report["preparation_mode"] != preparation)
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
    if report["preparation_mode"] == "fused":
        validate_prepare_receipt(report.get("preparation_checks"), report["native_resident_capabilities"])
    elif report.get("preparation_checks") is not None:
        raise ValueError("Eager preflight cannot substitute fused preparation evidence")
    for key, record in records.items():
        exact = key.startswith(("synthetic:", "resident:")) or key.endswith(":zero")
        count = int(key.split(":g", 1)[1].split(":", 1)[0]) if key.startswith(("synthetic:", "resident:")) else 1
        if (
            any(record.get(k) is not True for k in ("passed", "repeat_exact", "grouped_exact", "current_stream_exact"))
            or record.get("oracle_exact_required") is not exact
            or len(record.get("oracle", [])) != count
        ):
            raise ValueError(f"Incomplete projection evidence: {key}")
        if record.get("native_path") != "grouped_projection_resident":
            raise ValueError(f"Legacy projection cannot certify the resident implementation: {key}")
        if key.startswith("resident:") or ":m1:" in key or (count == 1 and key.endswith(":m1")):
            if record.get("resident_out_exact") is not True or record.get("descriptor_reuse_exact") is not True:
                raise ValueError(f"Missing resident out-ABI evidence: {key}")
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


def validate_prepare_receipt(receipt, capabilities):
    from tools.vq2a8_v3_prepare_check import PREPARE_PREFLIGHT_CASES

    if type(capabilities) is not int or capabilities < 0 or capabilities & 2 != 2 or not isinstance(receipt, dict):
        raise ValueError("Fused preparation requires native capability and numerical evidence")
    cases = receipt.get("cases", [])
    if (
        any(receipt.get(key) is not True for key in ("passed", "exactbitwise", "device_execution_verified"))
        or receipt.get("scope") != "fused_preparation_after_rht_and_bias_gemv"
        or not isinstance(cases, list)
        or len(cases) != len(PREPARE_PREFLIGHT_CASES)
        or any(not isinstance(case, dict) for case in cases)
        or {case.get("name") for case in cases} != set(PREPARE_PREFLIGHT_CASES)
        or any(case.get("passed") is not True or case.get("exactbitwise") is not True for case in cases)
    ):
        raise ValueError("Fused preparation preflight is incomplete or did not preserve exact bytes")


def checked_model_preflight(library, receipt, model, *, preparation="eager"):
    identity = library_identity(library)
    report = json.loads(Path(receipt).read_text(encoding="utf-8"))
    validate_receipt(report, identity, model, os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), preparation)
    return identity


def check_projection(inputs, golden, exact):
    import torch

    from tools.validate_vq2a8_phase4_kernel import bitwise_equal, compare
    from vllm_ascend.quantization.vq2a8_ascendc_v3 import grouped_projection_resident

    actual = grouped_projection_resident(inputs)
    torch.npu.synchronize()
    if len(actual) != len(golden):
        raise ValueError("Grouped output count mismatch")
    if any(a.dtype != torch.bfloat16 or a.shape != g.shape for a, g in zip(actual, golden)):
        raise ValueError("Resident native output shape/dtype differs from the projection contract")
    metrics = [compare(g, a) for g, a in zip(golden, actual)]
    if exact and not all(bitwise_equal(a, g) for a, g in zip(actual, golden)):
        raise ValueError("Exact synthetic/zero oracle mismatch")
    repeated = grouped_projection_resident(inputs)
    isolated = [grouped_projection_resident([row])[0] for row in inputs]
    ready, stream = torch.npu.Event(), torch.npu.Stream()
    ready.record()
    with torch.npu.stream(stream):
        stream.wait_event(ready)
        alternate = grouped_projection_resident(inputs)
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
        native_path="grouped_projection_resident",
    )


def check_resident(inputs):
    """Compare resident writes, descriptor updates and streams against the eager ABI."""
    import torch

    from tools.validate_vq2a8_phase4_kernel import bitwise_equal
    from vllm_ascend.quantization.vq2a8_ascendc_v3 import (
        grouped_projection_resident,
        grouped_projection_resident_out,
    )

    jobs, k, n = len(inputs), inputs[0][0].shape[1], inputs[0][3].shape[0] * 32
    if any(row[0].shape[0] != 1 for row in inputs):
        raise ValueError("Resident preflight is fixed M1")
    outputs = [torch.empty((1, n), dtype=torch.bfloat16, device=inputs[0][0].device) for _ in inputs]
    records = [[*(value.data_ptr() for value in row), out.data_ptr(), 1, n, k] for row, out in zip(inputs, outputs)]
    descriptors = torch.tensor(records, dtype=torch.int64, device="cpu").to(inputs[0][0].device)
    owners = [value for row in inputs for value in row] + outputs
    expected = grouped_projection_resident(inputs)

    def launch():
        grouped_projection_resident_out(descriptors, owners, jobs=jobs, m=1, n=n, k=k)

    for _ in range(2):
        for output in outputs:
            output.fill_(float("nan"))
        launch()
        torch.npu.synchronize()
        if not all(bitwise_equal(a, b) for a, b in zip(outputs, expected)):
            raise ValueError("Resident out-ABI differs from eager projection / repeat")
    # Change the activation address in the SAME descriptor allocation; retain
    # the replacement owners and compare against an independent eager request.
    changed_inputs = [(torch.zeros_like(row[0]), *row[1:]) for row in inputs]
    changed_expected = grouped_projection_resident(changed_inputs)
    owners.extend(row[0] for row in changed_inputs)
    replacement = torch.tensor([row[0].data_ptr() for row in changed_inputs], dtype=torch.int64, device="cpu")
    descriptors[:, 0].copy_(replacement.to(descriptors.device))
    launch()
    torch.npu.synchronize()
    if not all(bitwise_equal(a, b) for a, b in zip(outputs, changed_expected)):
        raise ValueError("Resident descriptor reuse ignored updated activation pointers")
    descriptors[:, 0].copy_(
        torch.tensor([row[0].data_ptr() for row in inputs], dtype=torch.int64).to(descriptors.device)
    )
    ready, stream = torch.npu.Event(), torch.npu.Stream()
    ready.record()
    with torch.npu.stream(stream):
        stream.wait_event(ready)
        launch()
    stream.synchronize()
    if not all(bitwise_equal(a, b) for a, b in zip(outputs, expected)):
        raise ValueError("Resident out-ABI current-stream mismatch")
    return dict(resident_out_exact=True, descriptor_reuse_exact=True)


def run(args):
    from tools.validate_vq2a8_ascendc import require_hardware_runtime

    require_hardware_runtime()
    import torch
    import torch_npu  # noqa: F401

    from tools.validate_vq2a8_phase4_kernel import (
        bitwise_equal,
        prepare_rows,
        same_fp8_oracle,
        synthetic_dense_oracle,
    )
    from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device, activation_case
    from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
    from vllm_ascend.quantization.vq2a8_ascendc_v2 import convert_expert_payload, gather_prepared_activation
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
        resident_abi_version=RESIDENT_ABI_VERSION,
        layout=RESIDENT_LAYOUT,
        preparation_mode=args.preparation,
        preparation_checks=None,
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
        if torch.ops.vq2a8_ascendc_v3.resident_abi_version() != RESIDENT_ABI_VERSION:
            raise ValueError("Loaded resident ABI differs from the preflight contract")
        capabilities = torch.ops.vq2a8_ascendc_v3.resident_capabilities()
        if (
            type(capabilities) is not int
            or capabilities & RESIDENT_CAPABILITIES_REQUIRED != RESIDENT_CAPABILITIES_REQUIRED
        ):
            raise ValueError("Loaded library has no resident projection capability")
        report["native_resident_capabilities"] = capabilities
        if args.preparation == "fused":
            if capabilities & 2 != 2:
                raise ValueError("Fused preparation requested but the loaded candidate lacks its capability")
            from tools.vq2a8_v3_prepare_check import run_prepare_preflight

            print("V3_PREFLIGHT_START=fused_preparation", flush=True)
            report["preparation_checks"] = run_prepare_preflight()
            validate_prepare_receipt(report["preparation_checks"], capabilities)
            save()
            print("V3_PREFLIGHT_PASS=fused_preparation", flush=True)
        for k in (2048, 4096):
            for rows in SYNTHETIC_ROWS:
                key = f"synthetic:k{k}:g{len(rows)}:m{rows[0]}"
                print(f"V3_PREFLIGHT_START={key}", flush=True)
                native, golden = [], []
                for offset, m in enumerate(rows):
                    values = _synthetic(m, k, offset)
                    golden.append(same_fp8_oracle(values[:3], synthetic_dense_oracle(*values[3:])))
                    native.append(_convert_inputs(values, device))
                report["cases"][key] = check_projection(native, golden, True)
                if rows == (1,):
                    report["cases"][key].update(check_resident(native))
                save()
                print(f"V3_PREFLIGHT_PASS={key}", flush=True)
        for k in (2048, 4096):
            for jobs in (1, 6):
                key = f"resident:k{k}:g{jobs}"
                print(f"V3_PREFLIGHT_START={key}", flush=True)
                native, golden = [], []
                for offset in range(jobs):
                    values = _synthetic(1, k, offset)
                    native.append(_convert_inputs(values, device))
                    golden.append(same_fp8_oracle(values[:3], synthetic_dense_oracle(*values[3:])))
                record = check_projection(native, golden, True)
                record.update(check_resident(native))
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
            payload = {name: value.to(device) for name, value in convert_expert_payload(host, spec).items()}
            original = {name: value.to(device) for name, value in host.items()}
            for m in REAL_ROWS:
                for case in REAL_CASES:
                    key = f"real:{kind}:m{m}:{case}"
                    print(f"V3_PREFLIGHT_START={key}", flush=True)
                    hidden = torch.cat([activation_case(spec.rht_true_columns, i, 0, case) for i in range(m)]).to(
                        device
                    )
                    reference = prepare_rows(hidden, original, spec)
                    prepared = RowwiseVQ2A8Preparation().rows(hidden, payload, spec)
                    if not all(bitwise_equal(a, b) for a, b in zip(prepared, reference)):
                        raise ValueError("Resident preparation differs before the exact-byte K gather")
                    q, scale, bias = prepared
                    native = [
                        (
                            gather_prepared_activation(q, payload["activation_order"]),
                            scale,
                            bias,
                            payload["packed_zn"],
                            payload["pair_lut"],
                        )
                    ]
                    report["cases"][key] = check_projection(native, [same_fp8_oracle(reference, dense)], case == "zero")
                    if m == 1:
                        report["cases"][key].update(check_resident(native))
                    save()
                    print(f"V3_PREFLIGHT_PASS={key}", flush=True)
        report.update(status="PASS", device_execution_verified=True)
        save()
        checked_model_preflight(args.library, args.output, args.model, preparation=args.preparation)
    except Exception as exc:
        report.update(status="FAIL", device_execution_verified=False, error=f"{type(exc).__name__}: {exc}")
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "library", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--preparation", choices=("eager", "fused"), default="eager")
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
