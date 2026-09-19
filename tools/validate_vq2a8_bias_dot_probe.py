#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H: experimental row-dot bitwise gate; never enables a model optimization.

The candidate uses vector Mul/ReduceSum, whose rounding may differ from CANN
MatMul. Any bit difference is a FAIL, not grounds to loosen the comparison.
Optional --inputs accepts CPU FP32 tensors {'rotated', 'weight_bias'} captured
from real activation preparation, via torch.load(weights_only=True).
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

from tools.diagnose_vq2a8_tp1_startup import child_environment, emit, parse_snapshot, run_child, stage_recorder

CASE = "v4_v2_bias_dot_probe"
LIBRARY_NAME = "libvq2a8_ascendc_v4_v2.so"
WIDTHS = (2048, 4096)
PATTERNS = ("random", "cancellation", "wide_exponents", "zeros", "subnormal", "nonfinite")
SEEDS = (17, 83)
QUEUE_ITERATIONS = 513
PRESSURE_BYTES = 2 * 1024 * 1024
CONTRACT_CASES = 9


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--queue-lifetime", action="store_true")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.physical_npu < 0 or not 1 <= args.timeout_s <= 7200 or (args.child and args.plan_only):
        parser.error("Require nonnegative NPU, timeout 1..7200; no child with plan-only")
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
    if args.queue_lifetime:
        command.append("--queue-lifetime")
    if args.inputs is not None:
        command += ["--inputs", str(args.inputs.resolve())]
    return command


def cases():
    return [
        (width, rows, pattern, seed)
        for width in WIDTHS
        for rows in range(1, 7)
        for pattern in PATTERNS
        for seed in SEEDS
    ]


def numeric_names():
    return [f"k{k}_r{r}_{pattern}_s{seed}" for k, r, pattern, seed in cases()]


def graph_names():
    return [f"graph_k{k}_{phase}" for k in WIDTHS for phase in ("normal", "changed", "recovered")]


def fixture(width, rows, pattern, seed):
    import torch

    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, width, generator=generator, dtype=torch.float32)
    weight = torch.randn(rows, width, generator=generator, dtype=torch.float32)
    if pattern == "cancellation":
        x[:, 1::2] = x[:, ::2]
        weight[:, 1::2] = -weight[:, ::2]
        x[:, -1] += 2**-17
    elif pattern == "wide_exponents":
        exponent = (torch.arange(width) % 41 - 20).float()
        x *= torch.exp2(exponent)
        weight *= torch.exp2(-exponent.roll(7))
    elif pattern == "zeros":
        x.zero_()
        x[:, 1::2] = -0.0
    elif pattern == "subnormal":
        x *= 2**-140
        weight.fill_(1.0)
    elif pattern == "nonfinite":
        x[0, 0] = float("nan")
        weight[-1, -1] = float("inf")
    elif pattern != "random":
        raise ValueError("Unknown bias dot fixture")
    return x, weight


def validate_inputs(payload):
    import torch

    if type(payload) is not dict or set(payload) != {"rotated", "weight_bias"}:
        raise ValueError("Real inputs must have exactly rotated and weight_bias")
    x, weight = payload["rotated"], payload["weight_bias"]
    for value in (x, weight):
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or value.ndim != 2
            or not value.is_contiguous()
            or not 1 <= value.shape[0] <= 6
            or value.shape[1] not in WIDTHS
        ):
            raise ValueError("Real inputs require contiguous CPU FP32[R1..6,K2048/4096]")
    if x.shape != weight.shape:
        raise ValueError("Real input shapes differ")
    return x, weight


def reference(x, weight):
    import torch

    # Reproduce the production rank, geometry and out= dispatch exactly.
    result = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    for row in range(x.shape[0]):
        torch.matmul(x[row : row + 1], weight[row], out=result[row : row + 1])
    return result


def assert_bits(actual, expected, name):
    import torch

    if actual.dtype != torch.float32 or actual.shape != expected.shape:
        raise AssertionError(f"{name}: dot output contract changed")
    a = actual.detach().cpu().contiguous().view(torch.int32)
    b = expected.detach().cpu().contiguous().view(torch.int32)
    mismatch = (a != b).nonzero().flatten()
    if mismatch.numel():
        row = int(mismatch[0])
        raise AssertionError(
            f"{name}: {mismatch.numel()} differing FP32 rows; first={row}, "
            f"actual_bits=0x{int(a[row]) & 0xFFFFFFFF:08x}, "
            f"expected_bits=0x{int(b[row]) & 0xFFFFFFFF:08x}; NOT model-compatible"
        )


def run_numeric(device, operation, stage, external=None):
    names = []
    for (width, rows, pattern, seed), name in zip(cases(), numeric_names()):
        x, weight = (value.to(device) for value in fixture(width, rows, pattern, seed))
        with stage(name):
            expected, actual = reference(x, weight), operation(x, weight)
        assert_bits(actual, expected, name)
        names.append(name)
    if external is not None:
        x, weight = (value.to(device) for value in external)
        with stage("real_inputs"):
            actual, expected = operation(x, weight), reference(x, weight)
        assert_bits(actual, expected, "real_inputs")
        names.append("real_inputs")
    return names


def run_contract(device, operation, stage):
    import torch

    x = torch.zeros(2, 2048, device=device)
    bad = [
        (x.half(), x),
        (x, x.half()),
        (x.flatten(), x),
        (x[:0], x[:0]),
        (torch.zeros(7, 2048, device=device), torch.zeros(7, 2048, device=device)),
        (x[:, :1024].contiguous(), x[:, :1024].contiguous()),
        (x, x[:1]),
        (torch.zeros(2, 4096, device=device)[:, ::2], x),
        (x.cpu(), x),
    ]
    for index, pair in enumerate(bad):
        with stage(f"contract_{index}"):
            try:
                operation(*pair)
            except (ValueError, RuntimeError):
                pass
            else:
                raise AssertionError(f"Native dot probe accepted malformed case {index}")
    return len(bad)


def run_graph(device, operation, stage):
    import torch

    names = []
    for width in WIDTHS:
        stream = torch.npu.Stream(device=device)
        with torch.npu.stream(stream):
            x, weight = (value.to(device) for value in fixture(width, 6, "random", 17))
            for _ in range(3):
                operation(x, weight)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with stage(f"graph_k{width}_capture"), torch.npu.graph(graph, stream=stream):
            actual = operation(x, weight)
        for phase, seed in (("normal", 17), ("changed", 83), ("recovered", 17)):
            with torch.npu.stream(stream):
                source = fixture(width, 6, "random", seed)
                x.copy_(source[0])
                weight.copy_(source[1])
                expected = reference(x, weight)
            torch.npu.synchronize()
            name = f"graph_k{width}_{phase}"
            with stage(name), torch.npu.stream(stream):
                graph.replay()
            assert_bits(actual, expected, name)
            names.append(name)
        with stage(f"graph_k{width}_reset"):
            graph.reset()
    return names


def queue_evidence():
    return {
        "iterations": QUEUE_ITERATIONS,
        "allocation_pressure_bytes": PRESSURE_BYTES,
        "explicit_per_iteration_synchronize": False,
        "cpu_reads_inside_loop": False,
        "python_input_owners_dropped_before_fence": True,
        "ordinary_reference": "same_device_rowwise_matmul_then_add",
        "task_queue_enable": "1",
        "runtime_queue_slots_measured": False,
        "native_stream_check_may_drain_host_queue": True,
    }


def run_queue(device, operation, stage):
    import torch

    # CPU snapshots do not keep queued NPU input owners alive. Device fixtures
    # are uploaded before the hot loop to avoid an accidental H2D fence there.
    templates = []
    for width in WIDTHS:
        for seed in SEEDS:
            x, weight = (value.to(device) for value in fixture(width, 6, "random", seed))
            expected = reference(x, weight)
            ordinary = (expected + 1.0).cpu().clone()
            templates.append((x, weight, expected.cpu().clone(), ordinary))
            del expected, ordinary, x, weight
    outputs = []
    with stage("queue_lifetime"):
        for index in range(QUEUE_ITERATIONS):
            template_x, template_weight, expected, ordinary_expected = templates[index % len(templates)]
            x, weight = template_x.clone(), template_weight.clone()
            actual = operation(x, weight)
            ordinary = actual + 1.0
            outputs.append((actual, ordinary, expected, ordinary_expected))
            del x, weight, actual, ordinary
            pressure = torch.empty(PRESSURE_BYTES, device=device, dtype=torch.uint8)
            pressure.fill_(index % 251)
            del pressure
        templates.clear()
        del template_x, template_weight
    with stage("queue_lifetime_verify"):
        for index, (actual, ordinary, expected, ordinary_expected) in enumerate(outputs):
            assert_bits(actual, expected, f"queue_{index}")
            assert_bits(ordinary, ordinary_expected, f"queue_{index}_ordinary")
    return queue_evidence()


def validate_child_evidence(result, queue_lifetime, real_inputs=False):
    events = result.get("events", [])
    final = next((event for event in reversed(events) if event.get("event") == "CASE_PASS"), {})
    expected = {
        "numeric": numeric_names() + (["real_inputs"] if real_inputs else []),
        "native_contract": CONTRACT_CASES,
        "graph": graph_names(),
    }
    if queue_lifetime:
        expected["queue_lifetime"] = queue_evidence()
    digest = final.get("library", {}).get("sha256", "")
    if (
        result.get("status") != "PASS"
        or result.get("exit_code") != 0
        or result.get("reaped") is not True
        or any(event.get("event") in ("FAIL", "CASE_FAIL") for event in events)
        or final.get("case") != CASE
        or type(final.get("native_abi")) is not int
        or final.get("native_abi") != 1
        or final.get("results") != expected
        or final.get("device_execution_verified") is not True
        or final.get("graph_verified") is not True
        or final.get("model_integration_verified") is not False
        or final.get("performance_verified") is not False
        or final.get("model_dispatch_enabled") is not False
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        or Path(final.get("library", {}).get("path", "")).name != LIBRARY_NAME
        or not any(e.get("event") == "PASS" and e.get("stage") == "final_sync" for e in events)
        or (
            queue_lifetime
            and not any(e.get("event") == "PASS" and e.get("stage") == "queue_lifetime_verify" for e in events)
        )
    ):
        raise ValueError("Incomplete bias dot probe evidence; cannot certify bitwise compatibility")
    if real_inputs:
        identity = final.get("real_inputs", {})
        value = identity.get("sha256", "")
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Missing real-input identity")


def run_case_child(args):
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical_npu):
        raise ValueError("Child physical NPU mapping differs")
    faulthandler.enable()
    faulthandler.dump_traceback_later(min(30, args.timeout_s), repeat=True)
    synchronize = lambda: None
    stage = stage_recorder(CASE, lambda: synchronize())
    try:
        queue = os.environ.get("TASK_QUEUE_ENABLE")
        if queue not in ("0", "1") or (args.queue_lifetime and queue != "1"):
            raise ValueError("Require TASK_QUEUE_ENABLE=0/1; queue lifetime requires 1")
        with stage("imports"):
            import torch
            import torch_npu  # noqa: F401

            from tools.validate_vq2a8_ascendc import require_hardware_runtime
            from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device
        with stage("device"):
            require_hardware_runtime()
            if torch.npu.device_count() != 1:
                raise RuntimeError("Require exactly one visible NPU")
            torch.set_num_threads(4)
            device = torch.device("npu:0")
            _initialize_device(device)
            torch.npu.config.allow_internal_format = False
            synchronize = torch.npu.synchronize
        with stage("library"):
            path = args.library.resolve(strict=True)
            if path.name != LIBRARY_NAME:
                raise ValueError(f"Require {LIBRARY_NAME}")
            identity = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            torch.ops.load_library(str(path))
            native = torch.ops.vq2a8_ascendc_v4_v2
            version = native.bias_dot_rows_probe_version()
            if type(version) is not int or version != 1:
                raise RuntimeError("Bias dot probe requires ABI 1; no implicit fallback")
            operation = native.bias_dot_rows_probe
        external, real_identity = None, None
        if args.inputs is not None:
            with stage("load_real_inputs"):
                path = args.inputs.resolve(strict=True)
                external = validate_inputs(torch.load(path, map_location="cpu", weights_only=True))
                real_identity = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        with torch.inference_mode():
            results = {
                "native_contract": run_contract(device, operation, stage),
                "numeric": run_numeric(device, operation, stage, external),
                "graph": run_graph(device, operation, stage),
            }
            if args.queue_lifetime:
                results["queue_lifetime"] = run_queue(device, operation, stage)
        with stage("final_sync"):
            pass
        emit(
            CASE,
            "CASE_PASS",
            results=results,
            native_abi=1,
            library=identity,
            real_inputs=real_identity,
            device_execution_verified=True,
            graph_verified=True,
            model_integration_verified=False,
            performance_verified=False,
            model_dispatch_enabled=False,
        )
        return 0
    except Exception as error:
        traceback.print_exc()
        emit(CASE, "CASE_FAIL", error=str(error), device_execution_verified=False, model_dispatch_enabled=False)
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()


def main(argv=None):
    args = parse_args(argv)
    if args.child:
        return run_case_child(args)
    report = {
        "scope": "experimental_row_dot_only",
        "command": child_command(args),
        "status": "PLANNED",
        "device_execution_verified": False,
        "graph_verified": False,
        "model_integration_verified": False,
        "performance_verified": False,
        "model_dispatch_enabled": False,
    }
    if args.plan_only:
        print(json.dumps(report, indent=2))
        return 0
    if os.name != "posix":
        raise RuntimeError("Bias dot probe requires Linux + NPU; use --plan-only elsewhere")
    directory = args.report_dir or Path(tempfile.mkdtemp(prefix="vq2-bias-dot-"))
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
            environment = child_environment(args)
            environment.setdefault("TASK_QUEUE_ENABLE", "1")
            result = run_child(child_command(args), environment, directory / "validation.log", args.timeout_s)
            report.update(status=result["status"], result=result)
            if result["status"] == "PASS":
                validate_child_evidence(result, args.queue_lifetime, args.inputs is not None)
                report.update(device_execution_verified=True, graph_verified=True)
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        traceback.print_exc()
    finally:
        summary = directory / "summary.json"
        summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"V4_BIAS_DOT_PROBE={report['status']} SUMMARY={summary}", flush=True)
    return 0 if report["status"] == "PASS" else 130 if report["status"] == "INTERRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
