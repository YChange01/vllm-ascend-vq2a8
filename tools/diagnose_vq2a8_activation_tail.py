#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded stage-by-stage diagnosis of the old native activation quantizer.

This tool does not enable fused preparation, alter tolerance, or load weights.
The same-input Torch NPU tail is the oracle; a second comparison to the unchanged
native quantizer checks that diagnostic snapshot fences did not alter outputs.
MISMATCH is a completed diagnosis, not a numerical-validation PASS. It exits 2.
"""

from __future__ import annotations

# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import faulthandler
import hashlib
import json
import math
import subprocess
import tempfile
import traceback
from pathlib import Path

from tools.diagnose_vq2a8_tp1_startup import child_environment, emit, parse_snapshot, run_child, stage_recorder

CASE = "v4_v2_activation_tail_diagnostic"
REPO = Path(__file__).resolve().parents[1]
LIBRARY_NAME = "libvq2a8_ascendc_v4_v2.so"
STAGES = ("transformed", "amax", "divided_scale", "scale", "normalized", "clamped", "q", "valid")
CASE_KINDS = ("legacy_m32_g6", "decode_m1_g6", "fp8_boundaries", "scale_boundaries")
DIAGNOSTIC_ABI = 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc-v4-v2" / LIBRARY_NAME)
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--max-mismatches", type=int, default=8)
    parser.add_argument("--max-saved-rows", type=int, default=2)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.physical_npu < 0 or args.timeout_s <= 0:
        parser.error("Require physical NPU >= 0 and timeout > 0")
    if not 1 <= args.max_mismatches <= 64 or not 0 <= args.max_saved_rows <= 4:
        parser.error("Require max-mismatches 1..64 and max-saved-rows 0..4")
    if args.child and (args.plan_only or args.report_dir is None):
        parser.error("Child requires report-dir and cannot use plan-only")
    args.launch_blocking = "0"
    return args


def probe_environment(args, environ=None):
    result = child_environment(args, environ)
    result.setdefault("TASK_QUEUE_ENABLE", "1")
    return result


def child_command(args, directory):
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--child",
        "--library",
        str(args.library.resolve()),
        "--physical-npu",
        str(args.physical_npu),
        "--timeout-s",
        str(args.timeout_s),
        "--report-dir",
        str(directory),
        "--max-mismatches",
        str(args.max_mismatches),
        "--max-saved-rows",
        str(args.max_saved_rows),
    ]


def require_diagnostic_abi(native):
    try:
        version = native.activation_tail_diagnostic_version()
        operator = native.activation_tail_diagnostic
        legacy = native.activation_quantize
    except (AttributeError, RuntimeError) as error:
        raise ValueError("Rebuild with tail diagnostic ABI 1; no fallback or serving mode is selected") from error
    if type(version) is not int or version != DIAGNOSTIC_ABI:
        raise ValueError(f"Tail diagnostic ABI mismatch: {version!r}")
    return operator, legacy


def _coordinate(flat_index, shape):
    coordinates = []
    for size in reversed(shape):
        coordinates.append(flat_index % size)
        flat_index //= size
    return list(reversed(coordinates))


def _ordered_float_bits(bits):
    # A monotonic bit ordering for finite IEEE FP32, with -0/+0 adjacent.
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits | 0x80000000


def compare_bits(actual, expected, *, limit):
    """CPU-computable report; no relaxed tolerance, even for signed zero/NaN."""
    import torch

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Diagnostic stage shape/dtype mismatch")
    actual = actual.detach().contiguous().cpu()
    expected = expected.detach().contiguous().cpu()
    byte_width = actual.element_size()
    actual_bytes = actual.view(torch.uint8).reshape(-1, byte_width)
    expected_bytes = expected.view(torch.uint8).reshape(-1, byte_width)
    different = actual_bytes != expected_bytes
    indices = different.any(-1).nonzero().flatten()
    samples = []
    for index in indices[:limit].tolist():
        left = int.from_bytes(bytes(actual_bytes[index].tolist()), byteorder=sys.byteorder)
        right = int.from_bytes(bytes(expected_bytes[index].tolist()), byteorder=sys.byteorder)
        actual_value = actual.flatten()[index].item()
        expected_value = expected.flatten()[index].item()
        sample = {
            "index": _coordinate(index, actual.shape),
            "actual_hex": f"0x{left:0{byte_width * 2}x}",
            "expected_hex": f"0x{right:0{byte_width * 2}x}",
            "actual_value": repr(actual_value),
            "expected_value": repr(expected_value),
        }
        if actual.dtype == torch.float32:
            sample["ulp_distance"] = (
                abs(_ordered_float_bits(left) - _ordered_float_bits(right))
                if math.isfinite(actual_value) and math.isfinite(expected_value)
                else None
            )
        samples.append(sample)
    return {
        "equal": indices.numel() == 0,
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "unequal_bytes": int(different.sum()),
        "unequal_elements": indices.numel(),
        "samples": samples,
    }


def torch_tail_stages(rotated, weight_scale, row_bias):
    import torch

    from vllm_ascend.quantization.vq2a8_reference import VQ2_FP8_MIN_SCALE

    transformed = rotated * weight_scale
    maximum = transformed.abs().amax(dim=-1)
    divided = maximum / torch.finfo(torch.float8_e4m3fn).max
    scale = torch.clamp(divided, min=VQ2_FP8_MIN_SCALE)
    normalized = transformed / scale.unsqueeze(-1)
    clamped = torch.clamp(normalized, -448, 448)
    quantized = clamped.to(torch.float8_e4m3fn)
    valid = (torch.isfinite(transformed).all(-1) & torch.isfinite(row_bias)).int()
    return dict(zip(STAGES, (transformed, maximum, divided, scale, normalized, clamped, quantized, valid)))


def legacy_tail_inputs(device, native, width, rows):
    """Reproduce original fixture seed, conversion and one-row GEMM geometry.

    Six fixture invocations intentionally have the same seed, exactly as the
    failing old M32/G6 probe. This is a synthetic tail diagnosis, not evidence
    that a different RHT algorithm is equivalent or that model outputs match.
    """
    import torch

    from tools.validate_vq2a8_activation_fused import fixture
    from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation

    requests = [fixture(device, width, rows) for _ in range(6)]
    values = [
        (row, payload["weight_scale"], payload["weight_bias"], payload["rht_sign"])
        for hidden, payload, _ in requests
        for row in hidden.split(1)
    ]
    x = torch.cat([value[0] for value in values]).float()
    weights, biases, signs = (torch.stack([value[index] for value in values]) for index in (1, 2, 3))
    preparation = RowwiseVQ2A8Preparation(compact=True)
    preparation.prepare_for_graph(device, 128)
    signed, input_valid = native.activation_sign(x, weights, biases, signs)
    sign_report = compare_bits(signed, x * signs.float(), limit=8)
    if not sign_report["equal"] or not bool(input_valid.all()):
        raise AssertionError("Legacy signing differs before tail diagnosis; cannot attribute error to tail")
    blocks = signed.reshape(-1, width // 128, 128)
    rotated = [(row @ preparation._hadamard).reshape(1, width) for row in blocks.split(1)]
    row_bias = torch.cat([row @ values[index][2] for index, row in enumerate(rotated)])
    return torch.cat(rotated), weights, row_bias


def boundary_tail_inputs(device, width, kind):
    import torch

    from vllm_ascend.quantization.vq2a8_reference import VQ2_FP8_MIN_SCALE

    if kind == "fp8_boundaries":
        values = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        mid = (values[:-1] + values[1:]) / 2
        positive = torch.cat(
            (mid, torch.nextafter(mid, torch.full_like(mid, float("inf"))), torch.nextafter(mid, mid * 0))
        )
        pattern = torch.cat((positive, -positive, torch.tensor([0.0, -0.0, 448.0, -448.0])))
        x = pattern.repeat((width + pattern.numel() - 1) // pattern.numel())[:width].reshape(1, width)
        x[0, -1] = 448
    elif kind == "scale_boundaries":
        threshold = torch.tensor(448 * VQ2_FP8_MIN_SCALE, dtype=torch.float32)
        maxima = torch.stack(
            (
                torch.nextafter(threshold, torch.tensor(0.0)),
                threshold,
                torch.nextafter(threshold, torch.tensor(float("inf"))),
            )
        )
        x = torch.linspace(-1, 1, width).repeat(3, 1) * maxima[:, None]
    else:
        raise ValueError(f"Unknown diagnostic boundary {kind}")
    x = x.to(device)
    return x, torch.ones_like(x), torch.zeros(x.shape[0], device=device)


def diagnose_case(native, inputs, *, limit, save_rows, directory):
    import torch

    diagnostic, legacy = require_diagnostic_abi(native)
    rotated, weights, bias = inputs
    expected = torch_tail_stages(*inputs)
    if not bool(expected["valid"].all()):
        raise AssertionError("Finite diagnostic case unexpectedly has invalid rows")
    raw = diagnostic(*inputs)
    if len(raw) != len(STAGES):
        raise ValueError("Tail diagnostic ABI returned incomplete stage evidence")
    actual = dict(zip(STAGES, raw))
    old_q, old_scale, old_valid = legacy(*inputs)
    comparisons = {name: compare_bits(actual[name], expected[name], limit=limit) for name in STAGES}
    instrument = {
        name: compare_bits(actual[name], value, limit=limit)
        for name, value in (("q", old_q), ("scale", old_scale), ("valid", old_valid))
    }
    first = next((name for name in STAGES if not comparisons[name]["equal"]), None)
    preserved = all(value["equal"] for value in instrument.values())
    saved = []
    if first is not None and save_rows:
        # Save only first-mismatch rows, not complete tensors or model weights.
        indices = []
        for sample in comparisons[first]["samples"]:
            row = sample["index"][0]
            if row not in indices:
                indices.append(row)
        for row in indices[:save_rows]:
            evidence = {
                "row": row,
                "first_mismatch_stage": first,
                # clone the row-sized CPU storage: serializing a view can
                # otherwise save the entire underlying allocation.
                "rotated": rotated[row : row + 1].detach().cpu().clone(),
                "weight_scale": weights[row : row + 1].detach().cpu().clone(),
                "row_bias": bias[row : row + 1].detach().cpu().clone(),
                **{f"actual_{name}": value[row : row + 1].detach().cpu().clone() for name, value in actual.items()},
                **{f"expected_{name}": value[row : row + 1].detach().cpu().clone() for name, value in expected.items()},
            }
            path = directory / f"mismatch_row_{row}.pt"
            if path.exists():
                raise FileExistsError(path)
            torch.save(evidence, path)
            saved.append({"path": str(path), "bytes": path.stat().st_size, "row": row})
    return {
        "first_mismatch_stage": first,
        "diagnostic_matches_legacy_native": preserved,
        "first_mismatch_attribution_supported": first is not None and preserved,
        "stages": comparisons,
        "legacy_native_comparisons": instrument,
        "saved_rows": saved,
    }


def check_fresh_validity(native, device):
    import torch

    diagnostic, legacy = require_diagnostic_abi(native)
    x = torch.ones((1, 2048), device=device)
    weights = torch.ones_like(x)
    bias = torch.zeros(1, device=device)
    reports = []
    for state, invalid in (("valid", False), ("invalid", True), ("recovered", False)):
        bias.fill_(float("nan") if invalid else 0)
        values = dict(zip(STAGES, diagnostic(x, weights, bias)))
        q, scale, valid = legacy(x, weights, bias)
        comparisons = {
            name: compare_bits(values[name], expected, limit=1)
            for name, expected in (("q", q), ("scale", scale), ("valid", valid))
        }
        expected_flag = 0 if invalid else 1
        if int(values["valid"].cpu().item()) != expected_flag or not all(
            item["equal"] for item in comparisons.values()
        ):
            raise AssertionError(f"Diagnostic stale validity or poison differs from old native quantizer: {state}")
        reports.append(state)
    return reports


def run_diagnostic_cases(device, native, stage, args):
    """Run the bounded matrix; exposed separately for CPU orchestration tests."""
    cases = []
    # First case exactly matches the old K2048/M32/G6 fixture seed.
    for kind in CASE_KINDS:
        for width in (2048, 4096):
            name = f"{kind}_k{width}"
            directory = args.report_dir / name
            directory.mkdir(exist_ok=False)
            with stage(name):
                if kind in ("legacy_m32_g6", "decode_m1_g6"):
                    rows = 32 if kind == "legacy_m32_g6" else 1
                    inputs = legacy_tail_inputs(device, native, width, rows)
                else:
                    inputs = boundary_tail_inputs(device, width, kind)
                result = diagnose_case(
                    native, inputs, limit=args.max_mismatches, save_rows=args.max_saved_rows, directory=directory
                )
                result.update(case=name, synthetic_fixture=True)
                (directory / "stages.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                emit(CASE, "TAIL_STAGES", diagnostic_case=name, result=result)
                cases.append(result)
    with stage("fresh_validity"):
        fresh = check_fresh_validity(native, device)
    return cases, fresh


def run_case_child(args):
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical_npu):
        raise ValueError("Child physical NPU mapping differs from requested device")
    faulthandler.enable()
    faulthandler.dump_traceback_later(min(30, args.timeout_s), repeat=True)
    sync = lambda: None
    stage = stage_recorder(CASE, lambda: sync())
    try:
        with stage("imports"):
            import torch
            import torch_npu  # noqa: F401

            from tools.validate_vq2a8_ascendc import require_hardware_runtime
            from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device
        with stage("device"):
            require_hardware_runtime()
            torch.set_num_threads(4)
            device = torch.device("npu:0")
            info = _initialize_device(device)
            torch.npu.config.allow_internal_format = False
            sync = torch.npu.synchronize
        with stage("library"):
            path = args.library.resolve(strict=True)
            if path.name != LIBRARY_NAME:
                raise ValueError(f"Require {LIBRARY_NAME}; no fallback")
            identity = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            torch.ops.load_library(str(path))
            native = torch.ops.vq2a8_ascendc_v4_v2
            require_diagnostic_abi(native)
            emit(CASE, "INFO", library=identity, device=info, diagnostic_only=True)
        with torch.inference_mode():
            cases, fresh = run_diagnostic_cases(device, native, stage, args)
        with stage("final_sync"):
            pass
        if not all(case["diagnostic_matches_legacy_native"] for case in cases):
            raise AssertionError(
                "Diagnostic snapshots changed legacy native outputs; stage attribution is not reliable"
            )
        mismatch = any(case["first_mismatch_stage"] is not None for case in cases)
        emit(
            CASE,
            "CASE_FAIL" if mismatch else "CASE_PASS",
            scope="synthetic_tail_diagnostic_not_preparation_acceptance",
            diagnostic_complete=True,
            diagnostic_outcome="MISMATCH" if mismatch else "ALL_MATCH",
            cases=cases,
            fresh_validity=fresh,
            library=identity,
            device_execution_verified=True,
            preparation_verified=False,
            graph_verified=False,
            model_integration_verified=False,
            performance_verified=False,
            error="Native tail differs from same-input Torch oracle" if mismatch else None,
        )
        return 2 if mismatch else 0
    except Exception as error:
        traceback.print_exc()
        emit(CASE, "CASE_FAIL", error=str(error), diagnostic_complete=False, device_execution_verified=False)
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()


def accept_completed_child(result):
    """Require complete scoped evidence, not just a marker or zero exit code."""
    final = next(
        (event for event in reversed(result.get("events", [])) if event.get("diagnostic_complete") is True), {}
    )
    cases = final.get("cases", [])
    expected_names = {f"{kind}_k{width}" for kind in CASE_KINDS for width in (2048, 4096)}
    if (
        result.get("reaped") is not True
        or final.get("case") != CASE
        or final.get("device_execution_verified") is not True
        or final.get("fresh_validity") != ["valid", "invalid", "recovered"]
        or len(cases) != len(expected_names)
        or {case.get("case") for case in cases} != expected_names
        or not all(
            case.get("diagnostic_matches_legacy_native") is True
            and set(case.get("stages", {})) == set(STAGES)
            and all(type(item.get("equal")) is bool for item in case["stages"].values())
            for case in cases
        )
    ):
        return None
    mismatch = any(not all(stage["equal"] for stage in case["stages"].values()) for case in cases)
    outcome = "MISMATCH" if mismatch else "ALL_MATCH"
    if (
        final.get("diagnostic_outcome") != outcome
        or final.get("event") != ("CASE_FAIL" if mismatch else "CASE_PASS")
        or result.get("exit_code") != (2 if mismatch else 0)
    ):
        return None
    if result.get("status") not in ("PASS", "FAIL"):
        return None
    return final


def main(argv=None):
    args = parse_args(argv)
    if args.child:
        return run_case_child(args)
    report = {
        "scope": "synthetic_tail_diagnostic_not_preparation_acceptance",
        "status": "PLANNED",
        "diagnostic_abi": DIAGNOSTIC_ABI,
        "case_order": [f"{kind}_k{width}" for kind in CASE_KINDS for width in (2048, 4096)],
        "max_mismatch_samples_per_stage": args.max_mismatches,
        "max_saved_rows_per_case": args.max_saved_rows,
        "device_execution_verified": False,
        "preparation_verified": False,
        "graph_verified": False,
        "model_integration_verified": False,
        "performance_verified": False,
    }
    if args.plan_only:
        print(json.dumps(report, indent=2))
        return 0
    if os.name != "posix":
        raise RuntimeError("Tail diagnostic requires Linux + NPU; use --plan-only elsewhere")
    directory = args.report_dir or Path(tempfile.mkdtemp(prefix="vq2-tail-diagnostic-"))
    if args.report_dir is not None:
        directory.mkdir(parents=True, exist_ok=False)
    report.update(status="FAIL", command=child_command(args, directory))
    try:
        snapshot = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True, timeout=20)
        (directory / "npu.log").write_text(snapshot.stdout + snapshot.stderr, encoding="utf-8")
        state = parse_snapshot(snapshot.stdout, args.physical_npu)
        if state == "unknown" or (state == "busy" and not args.allow_busy):
            report.update(status="BLOCKED", device_state=state)
        else:
            result = run_child(report["command"], probe_environment(args), directory / "diagnostic.log", args.timeout_s)
            report.update(status=result["status"], result=result)
            completed = accept_completed_child(result)
            if completed:
                report.update(status=completed["diagnostic_outcome"], device_execution_verified=True)
            elif result["status"] == "PASS":
                report.update(status="FAIL", error="Incomplete diagnostic evidence")
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        traceback.print_exc()
    finally:
        (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"V4_TAIL_DIAGNOSTIC={report['status']} SUMMARY={directory / 'summary.json'}", flush=True)
    return 0 if report["status"] == "ALL_MATCH" else 2 if report["status"] == "MISMATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
