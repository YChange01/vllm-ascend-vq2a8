#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""I: strict same-NPU VQ2 BF16 SwiGLU/select/sign acceptance, not model proof.

The candidate uses vector Exp/Div. BF16 boundaries alone do not prove equality
with CANN SiLU: any output-bit difference fails, with no tolerance or fallback.
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
from tools.validate_vq2a8_select_sign import (
    LIBRARY_NAME,
    METADATA_CASES,
    PRESSURE_BYTES,
    QUEUE_ITERATIONS,
    assert_bits,
    assert_ordinary_bits,
    check_owners,
    ids_values,
    make_bank,
    snapshot_owners,
)

CASE = "v4_v2_swiglu_select_sign"
WIDTHS = (2048, 4096)
LAYOUTS = ("contiguous", "expanded", "padded")
LIMITS = (0.0, 7.0, 7.1)
PATTERNS = ("random", "clamp_edges", "wide", "zeros", "nonfinite")
GRAPH_PHASES = (
    "normal",
    "invalid_slot",
    "recovered_slot",
    "nan_input",
    "recovered_input",
    "invalid_sign",
    "recovered_sign",
)
NATIVE_CONTRACT_CASES = 18


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--queue-lifetime", action="store_true")
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
    return command + (["--queue-lifetime"] if args.queue_lifetime else [])


def numeric_cases():
    return [
        (k, groups, layout, limit, pattern)
        for k in WIDTHS
        for groups in range(1, 7)
        for layout in LAYOUTS
        for limit in LIMITS
        for pattern in PATTERNS
    ]


def case_name(width, groups, layout, limit, pattern):
    return f"k{width}_g{groups}_{layout}_limit{limit:g}_{pattern}"


def numeric_names():
    names = [case_name(*case) for case in numeric_cases()]
    for width in WIDTHS:
        names += [f"metadata_k{width}_{pattern}" for pattern in METADATA_CASES]
        names += [f"slot_k{width}_{pattern}" for pattern in ("duplicate", "low", "high", "min", "max")]
        names += [
            f"bf16_sweep_k{width}_limit{limit:g}_chunk{chunk}" for limit in LIMITS for chunk in range(65536 // width)
        ]
    return names


def graph_names():
    return [
        f"graph_k{k}_{layout}_limit{limit:g}_{phase}"
        for k in WIDTHS
        for layout in LAYOUTS
        for limit in (0.0, 7.0)
        for phase in GRAPH_PHASES
    ]


def gate_up_view(owner, groups, width, layout):
    if layout == "expanded":
        return owner.expand(groups, 2 * width)
    if layout == "padded":
        return owner[:, 16 : 16 + 2 * width]
    return owner


def make_gate_up(device, width, groups, layout, pattern="random"):
    import torch

    rows = 1 if layout == "expanded" else groups
    columns = 2 * width + (32 if layout == "padded" else 0)
    values = torch.randn(rows, columns, generator=torch.Generator().manual_seed(197)) * 5
    if pattern == "clamp_edges":
        boundary = torch.tensor([-8.0, -7.125, -7.1, -7.0, -6.96875, 6.96875, 7.0, 7.1, 7.125, 8.0])
        values = boundary[torch.arange(rows * columns).remainder(boundary.numel())].reshape(rows, columns)
    elif pattern == "wide":
        exponent = (torch.arange(columns).remainder(121) - 60).float()
        values *= torch.exp2(exponent)
    elif pattern == "zeros":
        values.zero_()
        values[:, 1::2] = -0.0
    elif pattern == "nonfinite":
        special = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 2**-133, -(2**-133)])
        values = special[torch.arange(rows * columns).remainder(special.numel())].reshape(rows, columns)
    elif pattern != "random":
        raise ValueError("Unknown SwiGLU fixture")
    owner = values.to(dtype=torch.bfloat16, device=device)
    return gate_up_view(owner, groups, width, layout), owner


def reference(bank, gate_up, ids, limit, native):
    from vllm_ascend.quantization.vq2a8_reference import deepseek_v4_swiglu_reference

    activated = deepseek_v4_swiglu_reference(gate_up, limit)
    scale, bias, signs, selected = bank.select(ids)
    signed, input_status = native.activation_sign_strided(activated, scale, bias, signs)
    return signed, scale, bias, selected, input_status


def fused(bank, gate_up, ids, limit):
    operation = getattr(bank, "swiglu_select_sign", None)
    if not callable(operation):
        raise RuntimeError("Native bank has no swiglu_select_sign; no fallback")
    return operation(gate_up, ids, limit)


def require_abi(native):
    version = getattr(native, "swiglu_select_sign_version", None)
    abi = version() if callable(version) else None
    if type(abi) is not int or abi != 1:
        raise RuntimeError("Require SwiGLU/select/sign ABI 1; no fallback")


def run_numeric(device, factory, native, stage):
    import torch

    banks = {width: make_bank(device, factory, width) for width in WIDTHS}
    names = []
    for width, groups, layout, limit, pattern in numeric_cases():
        name = case_name(width, groups, layout, limit, pattern)
        bank, owners = banks[width]
        gate_up, owner = make_gate_up(device, width, groups, layout, pattern)
        id_owner = torch.tensor([73, *ids_values(groups, 3, "valid"), 91], dtype=torch.int64, device=device)
        ids = id_owner[1:-1]
        inspected = [owner, id_owner, *owners[3], *owners[4], *owners[5]]
        before = snapshot_owners(inspected)
        expected = reference(bank, gate_up, ids, limit, native)
        # Verification is inside its own stage, never confuse execution PASS
        # with numerical acceptance as happened in the earlier H probe log.
        with stage(name + "_execute"):
            actual = fused(bank, gate_up, ids, limit)
        with stage(name):
            assert_bits(actual, expected, name)
            check_owners(before, inspected, name)
        names.append(name)
    for width in WIDTHS:
        gate_up, owner = make_gate_up(device, width, 6, "contiguous")
        ids = torch.tensor([0, 1, 2, 2, 1, 0], dtype=torch.int64, device=device)
        for pattern in METADATA_CASES:
            name = f"metadata_k{width}_{pattern}"
            bank, owners = make_bank(device, factory, width, pattern=pattern)
            expected = reference(bank, gate_up, ids, 7.0, native)
            with stage(name):
                actual = fused(bank, gate_up, ids, 7.0)
                assert_bits(actual, expected, name)
                if bool(actual[4].cpu().any()):
                    raise AssertionError(f"{name}: invalid metadata accepted")
            names.append(name)
        bank, owners = banks[width]
        for pattern in ("duplicate", "low", "high", "min", "max"):
            name = f"slot_k{width}_{pattern}"
            slots = torch.tensor(ids_values(6, 3, pattern), dtype=torch.int64, device=device)
            expected = reference(bank, gate_up, slots, 7.0, native)
            with stage(name):
                assert_bits(fused(bank, gate_up, slots, 7.0), expected, name)
            names.append(name)
        # All 65536 gate bit patterns, up=1, signs=+1 explicitly expose
        # the BF16 activation rounding through signed FP32 output bits.
        bank, owners = make_bank(device, factory, width)
        for signs in owners[5]:
            signs.fill_(1)
        all_gate_bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        slots = torch.zeros(1, dtype=torch.int64, device=device)
        for limit in LIMITS:
            for chunk in range(65536 // width):
                name = f"bf16_sweep_k{width}_limit{limit:g}_chunk{chunk}"
                gate = all_gate_bits[chunk * width : (chunk + 1) * width].reshape(1, width)
                gate_up = torch.cat((gate, torch.ones_like(gate)), dim=1).to(device)
                expected = reference(bank, gate_up, slots, limit, native)
                with stage(name):
                    assert_bits(fused(bank, gate_up, slots, limit), expected, name)
                names.append(name)
    return names


def run_native_contract(device, factory, stage):
    import torch

    width = 2048
    bank, owners = make_bank(device, factory, width)
    x = torch.zeros((2, width * 2), dtype=torch.bfloat16, device=device)
    ids = torch.zeros(2, dtype=torch.int64, device=device)
    malformed = [
        (x.float(), ids, 7.0),
        (x.half(), ids, 7.0),
        (x.flatten(), ids, 7.0),
        (x[:, :width].contiguous(), ids, 7.0),
        (x[:0], ids[:0], 7.0),
        (
            torch.zeros((7, width * 2), dtype=x.dtype, device=device),
            torch.zeros(7, dtype=ids.dtype, device=device),
            7.0,
        ),
        (torch.zeros((2, width * 4), dtype=x.dtype, device=device)[:, ::2], ids, 7.0),
        (x.as_strided((2, width * 2), (1, 1)), ids, 7.0),
        (torch.zeros((2, width * 2 + 1), dtype=x.dtype, device=device)[:, : width * 2], ids, 7.0),
        (torch.zeros(width * 4 + 1, dtype=x.dtype, device=device)[1:].reshape(2, width * 2), ids, 7.0),
        (x, ids.int(), 7.0),
        (x, ids[:1], 7.0),
        (x, torch.zeros(4, dtype=ids.dtype, device=device)[::2], 7.0),
        (x.cpu(), ids, 7.0),
        (x, ids.cpu(), 7.0),
        (x, ids, -1.0),
        (x, ids, float("nan")),
        (x, ids, float("inf")),
    ]
    for index, (value, slots, limit) in enumerate(malformed):
        with stage(f"contract_{index}"):
            try:
                fused(bank, value, slots, limit)
            except (RuntimeError, ValueError):
                pass
            else:
                raise AssertionError(f"Native SwiGLU/select/sign accepted malformed case {index}")
    return len(malformed)


def run_graph(device, factory, native, stage):
    import torch

    names = []
    for width in WIDTHS:
        for layout in LAYOUTS:
            for limit in (0.0, 7.0):
                prefix = f"graph_k{width}_{layout}_limit{limit:g}"
                stream = torch.npu.Stream(device=device)
                with torch.npu.stream(stream):
                    bank, owners = make_bank(device, factory, width)
                    x, owner = make_gate_up(device, width, 6, layout)
                    clean_owner = owner.clone()
                    ids = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64, device=device)
                    clean_ids, clean_sign = ids.clone(), owners[5][0].clone()
                    for _ in range(3):
                        fused(bank, x, ids, limit)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with stage(prefix + "_capture"), torch.npu.graph(graph, stream=stream):
                    actual = fused(bank, x, ids, limit)
                for phase in GRAPH_PHASES:
                    name = prefix + "_" + phase
                    with torch.npu.stream(stream):
                        owner.copy_(clean_owner)
                        ids.copy_(clean_ids)
                        owners[5][0].copy_(clean_sign)
                        if phase == "invalid_slot":
                            ids[-1:].fill_(-(1 << 63))
                        elif phase == "nan_input":
                            owner[:, 16:17].fill_(float("nan"))
                        elif phase == "invalid_sign":
                            owners[5][0][-1:].zero_()
                        expected = tuple(t.clone() for t in reference(bank, x, ids, limit, native))
                    torch.npu.synchronize()
                    inspected = [owner, ids, *owners[3], *owners[4], *owners[5]]
                    before = snapshot_owners(inspected)
                    with stage(name + "_execute"), torch.npu.stream(stream):
                        graph.replay()
                    with stage(name):
                        assert_bits(actual, expected, name)
                        check_owners(before, inspected, name)
                    names.append(name)
                with stage(prefix + "_reset"):
                    graph.reset()
    return names


def queue_evidence():
    return {
        "iterations": QUEUE_ITERATIONS,
        "input_upload_before_loop": True,
        "explicit_per_iteration_synchronize": False,
        "cpu_reads_inside_loop": False,
        "fresh_device_clones": True,
        "python_input_owners_dropped_before_fence": True,
        "bank_owners_dropped_before_fence": True,
        "allocation_pressure_bytes": PRESSURE_BYTES,
        "ordinary_reference": "same_device_torch_swiglu_select_then_add",
        "task_queue_enable": "1",
        "runtime_queue_slots_measured": False,
        "native_stream_check_may_drain_host_queue": True,
    }


def run_queue(device, factory, native, stage):
    import torch

    templates, banks = [], {}
    for width in WIDTHS:
        banks[width] = make_bank(device, factory, width)
        for layout in LAYOUTS:
            for limit in (0.0, 7.0):
                for pattern in ("valid", "min", "duplicate"):
                    x, owner = make_gate_up(device, width, 6, layout)
                    ids = torch.tensor(ids_values(6, 3, pattern), dtype=torch.int64, device=device)
                    expected_device = reference(banks[width][0], x, ids, limit, native)
                    ordinary_expected = (expected_device[1] + 1.0).cpu().clone()
                    expected = tuple(t.cpu().clone() for t in expected_device)
                    templates.append((width, layout, limit, owner, ids, expected, ordinary_expected))
                    del expected_device
    torch.npu.synchronize()
    outputs = []
    with stage("queue_lifetime"):
        for iteration in range(QUEUE_ITERATIONS):
            width, layout, limit, template_owner, template_ids, expected, ordinary_expected = templates[
                iteration % len(templates)
            ]
            owner, ids = template_owner.clone(), template_ids.clone()
            x = gate_up_view(owner, 6, width, layout)
            values = fused(banks[width][0], x, ids, limit)
            outputs.append((values, values[1] + 1.0, expected, ordinary_expected))
            del x, owner, ids, values
            pressure = torch.empty(PRESSURE_BYTES, dtype=torch.uint8, device=device)
            pressure.fill_(iteration % 251)
            del pressure
        banks.clear()
        pressure = torch.empty(PRESSURE_BYTES, dtype=torch.uint8, device=device)
        pressure.fill_(91)
        del pressure
    with stage("queue_lifetime_verify"):
        for index, (values, ordinary, expected, ordinary_expected) in enumerate(outputs):
            assert_bits(tuple(value.cpu() for value in values), expected, f"queue_{index}")
            assert_ordinary_bits(ordinary, ordinary_expected, f"queue_{index}")
    return queue_evidence()


def validate_child_evidence(result, queue_lifetime):
    events = result.get("events", [])
    final = next((event for event in reversed(events) if event.get("event") == "CASE_PASS"), {})
    expected = {"numeric": numeric_names(), "native_contract": NATIVE_CONTRACT_CASES, "graph": graph_names()}
    if queue_lifetime:
        expected["queue_lifetime"] = queue_evidence()
    identity = final.get("library", {})
    digest = identity.get("sha256", "")
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
        or not isinstance(identity.get("path"), str)
        or Path(identity["path"]).name != LIBRARY_NAME
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        or (
            queue_lifetime
            and not any(
                event.get("event") == "PASS" and event.get("stage") == "queue_lifetime_verify" for event in events
            )
        )
        or not any(event.get("event") == "PASS" and event.get("stage") == "final_sync" for event in events)
    ):
        raise ValueError("Incomplete SwiGLU/select/sign child evidence")


def run_case_child(args):
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical_npu):
        raise ValueError("Child physical NPU mapping differs from requested device")
    faulthandler.enable()
    faulthandler.dump_traceback_later(min(30, args.timeout_s), repeat=True)
    synchronize = lambda: None
    stage = stage_recorder(CASE, lambda: synchronize())
    try:
        queue = os.environ.get("TASK_QUEUE_ENABLE")
        if queue not in ("0", "1") or (args.queue_lifetime and queue != "1"):
            raise ValueError("Graph requires TASK_QUEUE_ENABLE=0/1; queue lifetime requires 1")
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
            info = _initialize_device(device)
            torch.npu.config.allow_internal_format = False
            synchronize = torch.npu.synchronize
        with stage("library"):
            path = args.library.resolve(strict=True)
            if path.name != LIBRARY_NAME:
                raise ValueError(f"Require {LIBRARY_NAME}; no fallback")
            identity = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            torch.ops.load_library(str(path))
            native = torch.ops.vq2a8_ascendc_v4_v2
            require_abi(native)
            factory = torch.classes.vq2a8_ascendc_v4_v2.ResidentBank
            emit(CASE, "INFO", library=identity, device=info, native_abi=1, task_queue_enable=queue)
        with torch.inference_mode():
            results = {
                "native_contract": run_native_contract(device, factory, stage),
                "numeric": run_numeric(device, factory, native, stage),
                "graph": run_graph(device, factory, native, stage),
            }
            if args.queue_lifetime:
                results["queue_lifetime"] = run_queue(device, factory, native, stage)
        with stage("final_sync"):
            pass
        emit(
            CASE,
            "CASE_PASS",
            results=results,
            native_abi=1,
            library=identity,
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
        "scope": "swiglu_resident_select_sign_only",
        "command": child_command(args),
        "status": "PLANNED",
        "device_execution_verified": False,
        "graph_verified": False,
        "model_integration_verified": False,
        "performance_verified": False,
        "model_dispatch_enabled": False,
        "queue_lifetime_requested": args.queue_lifetime,
    }
    if args.plan_only:
        print(json.dumps(report, indent=2))
        return 0
    if os.name != "posix":
        raise RuntimeError("SwiGLU/select/sign validation requires Linux + NPU; use --plan-only elsewhere")
    directory = args.report_dir or Path(tempfile.mkdtemp(prefix="vq2-swiglu-select-sign-"))
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
                validate_child_evidence(result, args.queue_lifetime)
                report.update(device_execution_verified=True, graph_verified=True)
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        traceback.print_exc()
    finally:
        summary = directory / "summary.json"
        summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"V4_SWIGLU_SELECT_SIGN={report['status']} SUMMARY={summary}", flush=True)
    return 0 if report["status"] == "PASS" else 130 if report["status"] == "INTERRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
