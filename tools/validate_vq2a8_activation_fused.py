#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded native V4/v2 activation fusion acceptance; never starts a model.

Requires byte-identical FP8, row scales and bias relative to the unchanged
one-row Torch preparation on this NPU. Includes dynamic graph values and
invalid-value rejection. A PASS is not a serving or performance claim.
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
import subprocess
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

from tools.diagnose_vq2a8_tp1_startup import child_environment, emit, parse_snapshot, run_child, stage_recorder

CASE = "v4_v2_activation_fused"
REPO = Path(__file__).resolve().parents[1]
LIBRARY_NAME = "libvq2a8_ascendc_v4_v2.so"
QUEUE_ITERATIONS = 2049


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc-v4-v2" / LIBRARY_NAME)
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--queue-lifetime", action="store_true")
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.physical_npu < 0 or args.timeout_s <= 0 or (args.child and args.plan_only):
        parser.error("Require physical NPU >= 0, timeout > 0; --child cannot use --plan-only")
    args.launch_blocking = "0"
    return args


def child_command(args):
    command = [
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
    ]
    return command + (["--queue-lifetime"] if args.queue_lifetime else [])


def assert_bits(actual, expected, name):
    import torch

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: shape/dtype mismatch")
    actual_bytes = actual.detach().view(torch.uint8).cpu()
    expected_bytes = expected.detach().view(torch.uint8).cpu()
    if not torch.equal(actual_bytes, expected_bytes):
        differences = int((actual_bytes != expected_bytes).sum())
        raise AssertionError(f"{name}: {differences} unequal bytes; no relaxed tolerance or silent fallback")


def fixture(device, width, rows, *, true_width=None):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(width + rows)
    true_width = true_width or width
    hidden = torch.randn(rows, true_width, generator=generator).bfloat16().to(device)
    payload = {
        "weight_scale": torch.randn(width, generator=generator).to(device),
        "weight_bias": torch.randn(width, generator=generator).to(device),
        "rht_sign": torch.where(torch.arange(width) % 3 == 0, -1, 1).to(device, dtype=torch.int8),
    }
    return hidden, payload, SimpleNamespace(columns=width, rht_true_columns=true_width, rht_block_size=128)


def run_numerical_checks(device, native, stage):
    import torch

    from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
    from vllm_ascend.quantization.vq2a8_activation_fused import FusedV4V2Preparation
    from vllm_ascend.quantization.vq2a8_reference import VQ2_FP8_MIN_SCALE

    completed = []
    for width in (2048, 4096):
        for rows, jobs, case in (
            (1, 1, "random"),
            (1, 6, "zero"),
            (2, 6, "small"),
            (32, 6, "random"),
            (1, 1, "impulse"),
            (2, 1, "padded"),
        ):
            name = f"k{width}_m{rows}_g{jobs}_{case}"
            with stage(name):
                requests = [
                    fixture(device, width, rows, true_width=width - 17 if case == "padded" else None)
                    for _ in range(jobs)
                ]
                for hidden, _, _ in requests:
                    if case == "zero":
                        hidden.zero_()
                    elif case == "small":
                        hidden.mul_(1e-15)
                    elif case == "impulse":
                        hidden.zero_()
                        hidden[:, -1] = -1
                expected = RowwiseVQ2A8Preparation(compact=True).many(requests)
                actual = FusedV4V2Preparation().many(requests)
                for job, (got, want) in enumerate(zip(actual, expected)):
                    for field, x, y in zip(("q", "scale", "bias"), got, want):
                        assert_bits(x, y, f"{name}_{job}_{field}")
            completed.append(name)
        # Exhaust positive finite FP8 midpoints and both adjacent FP32 values.
        # The maximum 448 fixes row scale to 1, including sign/zero boundaries.
        with stage(f"k{width}_fp8_rounding_boundaries"):
            values = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
            mid = (values[:-1] + values[1:]) / 2
            positive = torch.cat(
                (
                    mid,
                    torch.nextafter(mid, torch.full_like(mid, float("inf"))),
                    torch.nextafter(mid, torch.zeros_like(mid)),
                )
            )
            pattern = torch.cat((positive, -positive, torch.tensor([0.0, -0.0, 448.0, -448.0])))
            x = pattern.repeat((width + pattern.numel() - 1) // pattern.numel())[:width].reshape(1, width).to(device)
            x[0, -1] = 448
            weights = torch.ones_like(x)
            q, scale, valid = native.activation_quantize(x, weights, torch.zeros(1, device=device))
            expected_scale = (x.abs().amax(-1) / 448).clamp(min=VQ2_FP8_MIN_SCALE)
            expected_q = (x / expected_scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
            assert_bits(q, expected_q, "FP8 nearest-even boundaries")
            assert_bits(scale, expected_scale, "FP8 boundary row scale")
            if not bool(valid.all()):
                raise AssertionError("Finite FP8 boundary data was rejected")
        completed.append(f"k{width}_fp8_rounding_boundaries")
        with stage(f"k{width}_minimum_scale_boundaries"):
            threshold = torch.tensor(448 * VQ2_FP8_MIN_SCALE, dtype=torch.float32)
            maxima = torch.stack(
                (
                    torch.nextafter(threshold, torch.tensor(0.0)),
                    threshold,
                    torch.nextafter(threshold, torch.tensor(float("inf"))),
                )
            )
            x = torch.linspace(-1, 1, width).repeat(3, 1) * maxima[:, None]
            x = x.to(device)
            q, scale, valid = native.activation_quantize(x, torch.ones_like(x), torch.zeros(3, device=device))
            expected_scale = (x.abs().amax(-1) / 448).clamp(min=VQ2_FP8_MIN_SCALE)
            expected_q = (x / expected_scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
            assert_bits(q, expected_q, "minimum scale boundary FP8")
            assert_bits(scale, expected_scale, "minimum scale boundary scale")
            if not bool(valid.all()):
                raise AssertionError("Finite minimum-scale boundary data rejected")
        completed.append(f"k{width}_minimum_scale_boundaries")
    return completed


def run_invalid_checks(device, native, stage):
    import torch

    from vllm_ascend.quantization.vq2a8_activation_fused import FusedV4V2Preparation

    with stage("invalid_values"):
        for field, bad in (
            ("x", float("nan")),
            ("weight_scale", float("inf")),
            ("weight_bias", -float("inf")),
            ("rht_sign", 0),
            ("rht_sign", -128),
        ):
            hidden, payload, spec = fixture(device, 2048, 1)
            target = hidden if field == "x" else payload[field]
            target.view(-1)[-1] = bad
            flags = []
            FusedV4V2Preparation(validity=flags.append).many([(hidden, payload, spec)])
            if len(flags) != 1 or bool(flags[0]):
                raise AssertionError(f"Invalid {field} accepted")
        x = torch.ones(1, 2048, device=device)
        for value, weights, bias in (
            (x * 1e30, x * 1e30, torch.zeros(1, device=device)),
            (x, x, torch.full((1,), float("inf"), device=device)),
        ):
            _, _, valid = native.activation_quantize(value, weights, bias)
            if bool(valid.all()):
                raise AssertionError("Output overflow/row-bias Inf accepted")
    with stage("native_metadata_rejections"):
        x = torch.ones(1, 2048, device=device)
        signs = torch.ones_like(x, dtype=torch.int8)
        bad_calls = [
            lambda: native.activation_sign(x.bfloat16(), x, x, signs),
            lambda: native.activation_sign(x, x, x, signs.int()),
            lambda: native.activation_sign(x, x.cpu(), x, signs),
            lambda: native.activation_sign(x[:, :1024], x[:, :1024], x[:, :1024], signs[:, :1024]),
            lambda: native.activation_quantize(x, x, torch.ones(2, device=device)),
        ]
        for call in bad_calls:
            try:
                call()
            except RuntimeError:
                continue
            raise AssertionError("Invalid native metadata was not rejected")
    return True


def run_graph_checks(device, stage):
    import torch

    from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
    from vllm_ascend.quantization.vq2a8_activation_fused import FusedV4V2Preparation

    with stage("graph_prepare"):
        hidden, payload, spec = fixture(device, 2048, 1)
        flags = []
        prepare = FusedV4V2Preparation(validity=flags.append)
        prepare.prepare_for_graph(device, spec.rht_block_size)
        for _ in range(2):
            prepare.many([(hidden, payload, spec)])
        torch.npu.synchronize()
        flags.clear()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs = prepare.many([(hidden, payload, spec)])[0]
        captured_flag = flags[-1]
    try:
        for case in ("normal", "different", "invalid", "recovered"):
            with stage(f"graph_replay_{case}"):
                hidden.fill_(float("nan") if case == "invalid" else 0.5 if case == "different" else -0.25)
                graph.replay()
                torch.npu.synchronize()
                if bool(captured_flag) != (case != "invalid"):
                    raise AssertionError("Graph captured stale validity instead of live input values")
                if case != "invalid":
                    expected = RowwiseVQ2A8Preparation(compact=True).many([(hidden, payload, spec)])[0]
                    for field, got, want in zip(("q", "scale", "bias"), outputs, expected):
                        assert_bits(got, want, f"{case}_{field}")
        return True
    finally:
        torch.npu.synchronize()
        graph.reset()


def run_queue_checks(device, native, stage):
    import torch

    with stage("queue_lifetime_setup"):
        x = torch.ones(1, 2048, device=device)
        weights = torch.ones_like(x)
        bias = torch.zeros_like(x)
        signs = torch.ones_like(x, dtype=torch.int8)
        row_bias = torch.zeros(1, device=device)
    with stage("queue_lifetime_wrap"):
        for _ in range(QUEUE_ITERATIONS):
            # Each callback receives short-lived Python tensors. No explicit
            # per-iteration synchronize; scalar stream lookup stays caller-side.
            signed, valid = native.activation_sign(x + 0.5, weights.clone(), bias.clone(), signs.clone())
            q, scale, quant_valid = native.activation_quantize(signed, weights.clone(), row_bias.clone())
            ordinary = q.float().sum()
        torch.npu.synchronize()
        if not bool(valid.all() & quant_valid.all()) or not bool(torch.isfinite(ordinary)):
            raise AssertionError("Queue-lifetime output invalid")
        expected_scale = (x + 0.5).abs().amax(-1) / 448
        assert_bits(scale, expected_scale, "queue lifetime scale")
    return {"iterations": QUEUE_ITERATIONS, "explicit_per_iteration_sync": False}


def run_case_child(args):
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical_npu):
        raise ValueError("Child physical NPU mapping differs from the requested device")
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
                raise ValueError(f"Require {LIBRARY_NAME}; no baseline library fallback")
            identity = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            torch.ops.load_library(str(path))
            native = torch.ops.vq2a8_ascendc_v4_v2
            if native.activation_preparation_version() != 1:
                raise ValueError("Fused activation preparation ABI mismatch")
            emit(CASE, "INFO", library=identity, device=info)
        with torch.inference_mode():
            results = {
                "numeric": run_numerical_checks(device, native, stage),
                "invalid": run_invalid_checks(device, native, stage),
                "graph": run_graph_checks(device, stage),
            }
            if args.queue_lifetime:
                results["queue_lifetime"] = run_queue_checks(device, native, stage)
        with stage("final_sync"):
            pass
        emit(
            CASE,
            "CASE_PASS",
            results=results,
            library=identity,
            device_execution_verified=True,
            graph_verified=True,
            model_integration_verified=False,
            performance_verified=False,
        )
        return 0
    except Exception as error:
        traceback.print_exc()
        emit(CASE, "CASE_FAIL", error=str(error), device_execution_verified=False)
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()


def main(argv=None):
    args = parse_args(argv)
    if args.child:
        return run_case_child(args)
    report = {
        "scope": "fused_activation_only",
        "command": child_command(args),
        "status": "PLANNED",
        "device_execution_verified": False,
        "graph_verified": False,
        "model_integration_verified": False,
        "performance_verified": False,
    }
    if args.plan_only:
        print(json.dumps(report, indent=2))
        return 0
    if os.name != "posix":
        raise RuntimeError("Native fused activation validation requires Linux + NPU; use --plan-only elsewhere")
    directory = args.report_dir or Path(tempfile.mkdtemp(prefix="vq2-activation-fused-"))
    if args.report_dir is not None:
        directory.mkdir(parents=True, exist_ok=False)
    report["status"] = "FAIL"
    try:
        snapshot = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True, timeout=20)
        (directory / "npu.log").write_text(snapshot.stdout + snapshot.stderr, encoding="utf-8")
        state = parse_snapshot(snapshot.stdout, args.physical_npu)
        if state == "unknown" or (state == "busy" and not args.allow_busy):
            report.update(status="BLOCKED", device_state=state)
        else:
            result = run_child(
                child_command(args), child_environment(args), directory / "validation.log", args.timeout_s
            )
            report.update(status=result["status"], result=result)
            final = next((event for event in reversed(result["events"]) if event.get("event") == "CASE_PASS"), {})
            required = {"numeric", "invalid", "graph"} | ({"queue_lifetime"} if args.queue_lifetime else set())
            if result["status"] == "PASS":
                if (
                    final.get("case") != CASE
                    or final.get("device_execution_verified") is not True
                    or final.get("graph_verified") is not True
                    or set(final.get("results", {})) != required
                    or not all(final["results"].values())
                ):
                    report.update(status="FAIL", error="Incomplete child evidence")
                else:
                    report.update(device_execution_verified=True, graph_verified=True)
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        traceback.print_exc()
    finally:
        (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"V4_FUSED_ACTIVATION={report['status']} SUMMARY={directory / 'summary.json'}", flush=True)
    return 0 if report["status"] == "PASS" else 130 if report["status"] == "INTERRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
