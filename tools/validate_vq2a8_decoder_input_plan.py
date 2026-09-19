#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated G integer input-plan gate; not real-model/performance evidence.

Input preparation runs outside decoder capture, so this probe does not claim
graph capture validation. The real-model general/G/general gate is separate.
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

CASE = "v4_v2_decoder_input_plan"
LIBRARY_NAME = "libvq2a8_ascendc_v4_v2.so"
QUEUE_ITERATIONS = 513
CONTRACT_CASES = 12
DEFENSIVE_CASES = ("negative", "oversized", "out_of_range", "bad_query", "recovery")
SENTINEL = -777


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--queue-lifetime", action="store_true")
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.physical_npu < 0 or not 0 < args.timeout_s <= 7200 or (args.child and args.plan_only):
        parser.error("Require nonnegative card, timeout 1..7200 and no child plan-only")
    args.launch_blocking = "0"
    return args


def case_names():
    return [
        f"g{groups}_offset{offset}_p{position}_round{round_id}"
        for groups in (1, 6)
        for offset in (0, 1)
        for position in (0, 1, 2, 7)
        for round_id in range(3)
    ]


def row_values(groups, shift):
    return [[gid + shift + 2, gid + shift + 11, -2, (1 << 31) - 1] for gid in range(groups)]


def expected_owners(groups, offset, position, shift):
    """Literal integer oracle, including untouched row/tails and int32 overflow."""
    result = []
    for gid, row in enumerate(row_values(groups, shift)):
        size, limit = 2 ** (gid % 3 + 1), 5 + gid % 2
        slot = row[position // size] * size + position % size
        slot = (slot + (1 << 31)) % (1 << 32) - (1 << 31)
        result.append(
            (
                [SENTINEL] * offset + row + [SENTINEL] * (4 + offset),
                [SENTINEL] * offset + [slot] + [-1] * (limit - 1) + [SENTINEL] * (8 - limit + offset),
            )
        )
    return result


def fixture(torch, device, groups, offset):
    owners, tables, slots = [], [], []
    for _ in range(groups):
        a = torch.full((8 + offset * 2,), SENTINEL, dtype=torch.int32, device=device)
        b = torch.full((8 + offset * 2,), SENTINEL, dtype=torch.int32, device=device)
        owners.append((a, b))
        tables.append(a[offset : offset + 8].view(2, 4))
        slots.append(b[offset : offset + 8])
    sizes, limits = [2 ** (i % 3 + 1) for i in range(groups)], [5 + i % 2 for i in range(groups)]
    plan = torch.classes.vq2a8_ascendc_v4_v2.DecoderInputPlan(tables, slots, sizes, limits)
    return plan, owners, tables, slots, sizes, limits


def inputs(torch, device, groups, offset, position, shift):
    values = [x for row in row_values(groups, shift) for x in row]
    packed = torch.tensor([SENTINEL] * offset + values + [SENTINEL] * offset, dtype=torch.int32, device=device)
    query = torch.tensor([SENTINEL] * offset + [0, 1] + [SENTINEL] * offset, dtype=torch.int32, device=device)
    positions = torch.tensor([SENTINEL] * offset + [position] + [SENTINEL] * offset, dtype=torch.int64, device=device)
    return packed, query, positions


def assert_owners(torch, owners, expected):
    for actual_pair, expected_pair in zip(owners, expected):
        for actual, values in zip(actual_pair, expected_pair):
            torch.testing.assert_close(actual.cpu(), torch.tensor(values, dtype=torch.int32), rtol=0, atol=0)


def numeric_checks(torch, device, stage):
    completed = []
    for groups in (1, 6):
        for offset in (0, 1):
            plan, owners, *_ = fixture(torch, device, groups, offset)
            for position in (0, 1, 2, 7):
                for round_id, shift in enumerate((0, 19, 0)):
                    name = f"g{groups}_offset{offset}_p{position}_round{round_id}"
                    full = inputs(torch, device, groups, offset, position, shift)
                    before = tuple(value.cpu() for value in full)
                    packed, query, positions = full
                    with stage(name):
                        plan.copy_rows(packed[offset : offset + groups * 4])
                        plan.slot_mapping(query[offset : offset + 2], positions[offset : offset + 1])
                    assert_owners(torch, owners, expected_owners(groups, offset, position, shift))
                    for value, saved in zip(full, before):
                        torch.testing.assert_close(value.cpu(), saved, rtol=0, atol=0)
                    completed.append(name)
    return completed


def contract_checks(torch, device, stage):
    plan, owners, tables, slots, sizes, limits = fixture(torch, device, 1, 0)
    ctor = torch.classes.vq2a8_ascendc_v4_v2.DecoderInputPlan
    packed, query, positions = inputs(torch, device, 1, 0, 1, 0)
    calls = [
        lambda: ctor([], [], [], []),
        lambda: ctor(tables, slots, [0], limits),
        lambda: ctor(tables, slots, sizes, [0]),
        lambda: ctor(tables, slots, sizes, [9]),
        lambda: ctor(tables, [tables[0].view(-1)], sizes, limits),
        lambda: plan.copy_rows(packed[:3]),
        lambda: plan.copy_rows(packed.to(torch.int64)),
        lambda: plan.slot_mapping(query[:1], positions),
        lambda: plan.slot_mapping(query, positions.to(torch.int32)),
        lambda: plan.copy_rows(tables[0][0]),
        lambda: plan.copy_rows(packed.cpu()),
        lambda: ctor([tables[0].to(torch.int64)], slots, sizes, limits),
    ]
    with stage("native_contract_rejections"):
        for operation in calls:
            try:
                operation()
            except (RuntimeError, ValueError):
                continue
            raise AssertionError("Native input contract unexpectedly accepted invalid arguments")
    assert_owners(torch, owners, [([SENTINEL] * 8, [SENTINEL] * 8)])
    return len(calls)


def defensive_checks(torch, device, stage):
    plan, owners, *_ = fixture(torch, device, 1, 1)
    for name in DEFENSIVE_CASES:
        position = {"negative": -1, "oversized": (1 << 63) - 1, "out_of_range": 8, "bad_query": 1, "recovery": 1}[name]
        packed, query, positions = inputs(torch, device, 1, 1, position, 0)
        if name == "bad_query":
            query[1] = 1
        with stage("defensive_" + name):
            plan.copy_rows(packed[1:5])
            plan.slot_mapping(query[1:3], positions[1:2])
        expected = expected_owners(1, 1, 1, 0)
        if name != "recovery":
            expected[0][1][1] = -1
        assert_owners(torch, owners, expected)
    return list(DEFENSIVE_CASES)


def queue_checks(torch, device, stage):
    plan, owners, tables, slots, _, _ = fixture(torch, device, 6, 1)
    templates = [inputs(torch, device, 6, 1, position, shift) for position, shift in ((1, 0), (2, 19), (7, 0))]
    expected = [expected_owners(6, 1, p, s) for p, s in ((1, 0), (2, 19), (7, 0))]
    torch.npu.synchronize()
    snapshots = []
    with stage("queue_owner_release_513"):
        for iteration in range(QUEUE_ITERATIONS):
            packed, query, positions = (value.clone() for value in templates[iteration % 3])
            plan.copy_rows(packed[1:25])
            plan.slot_mapping(query[1:3], positions[1:2])
            snapshots.append([(a.clone(), b.clone()) for a, b in owners])
            del packed, query, positions
            pressure = torch.empty(256 * 1024, dtype=torch.uint8, device=device)
            del pressure
        # Release the custom class and all original output owners before the
        # only post-loop synchronization. Callbacks/allocator must retain them.
        del plan, owners, tables, slots
    for iteration, values in enumerate(snapshots):
        assert_owners(torch, values, expected[iteration % 3])
    return {
        "iterations": QUEUE_ITERATIONS,
        "input_and_plan_owners_released": True,
        "per_iteration_synchronize": False,
        "runtime_queue_slots_measured": False,
    }


def validate_child_evidence(result, queue_lifetime):
    events = result.get("events", [])
    final = next((e for e in reversed(events) if e.get("event") == "CASE_PASS"), {})
    values = final.get("results", {})
    library = final.get("library", {})
    digest = library.get("sha256", "")
    if (
        result.get("status") != "PASS"
        or result.get("exit_code") != 0
        or result.get("reaped") is not True
        or any(e.get("event") in ("FAIL", "CASE_FAIL") for e in events)
        or final.get("case") != CASE
        or type(final.get("native_abi")) is not int
        or final.get("native_abi") != 1
        or final.get("device_execution_verified") is not True
        or final.get("model_integration_verified") is not False
        or final.get("performance_verified") is not False
        or values.get("numeric") != case_names()
        or values.get("native_contract") != CONTRACT_CASES
        or values.get("defensive_device_inputs") != list(DEFENSIVE_CASES)
        or set(values)
        != (
            {"numeric", "native_contract", "defensive_device_inputs", "queue"}
            if queue_lifetime
            else {"numeric", "native_contract", "defensive_device_inputs"}
        )
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        or Path(library.get("path", "")).name != LIBRARY_NAME
        or not any(e.get("event") == "PASS" and e.get("stage") == "final_sync" for e in events)
    ):
        raise ValueError("Incomplete B1 input-plan evidence")
    if queue_lifetime and values.get("queue") != {
        "iterations": QUEUE_ITERATIONS,
        "input_and_plan_owners_released": True,
        "per_iteration_synchronize": False,
        "runtime_queue_slots_measured": False,
    }:
        raise ValueError("Incomplete B1 input-plan queue evidence")


def run_case_child(args):
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical_npu):
        raise ValueError("Physical NPU mapping differs")
    faulthandler.enable()
    faulthandler.dump_traceback_later(30, repeat=True)
    synchronize = lambda: None
    stage = stage_recorder(CASE, lambda: synchronize())
    try:
        if args.queue_lifetime and os.environ.get("TASK_QUEUE_ENABLE") != "1":
            raise ValueError("Queue gate requires TASK_QUEUE_ENABLE=1")
        with stage("imports"):
            import torch
            import torch_npu  # noqa: F401

            from tools.validate_vq2a8_ascendc import require_hardware_runtime
            from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device
        with stage("device"):
            require_hardware_runtime()
            if torch.npu.device_count() != 1:
                raise ValueError("Require exactly one visible NPU")
            device = torch.device("npu:0")
            info = _initialize_device(device)
            torch.npu.config.allow_internal_format = False
            synchronize = torch.npu.synchronize
        with stage("library"):
            path = args.library.resolve(strict=True)
            if path.name != LIBRARY_NAME:
                raise ValueError("Unexpected library name")
            identity = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            torch.ops.load_library(str(path))
            version = torch.ops.vq2a8_ascendc_v4_v2.decoder_input_plan_version()
            if type(version) is not int or version != 1:
                raise ValueError("Unsupported input-plan ABI")
            emit(CASE, "INFO", library=identity, device=info)
        with torch.inference_mode():
            results = {
                "numeric": numeric_checks(torch, device, stage),
                "native_contract": contract_checks(torch, device, stage),
                "defensive_device_inputs": defensive_checks(torch, device, stage),
            }
            if args.queue_lifetime:
                results["queue"] = queue_checks(torch, device, stage)
        with stage("final_sync"):
            pass
        emit(
            CASE,
            "CASE_PASS",
            results=results,
            native_abi=1,
            library=identity,
            device_execution_verified=True,
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
    ] + (["--queue-lifetime"] if args.queue_lifetime else [])
    report = {
        "scope": "block_table_upload_and_grouped_slot_mapping",
        "command": command,
        "status": "PLANNED",
        "device_execution_verified": False,
        "model_integration_verified": False,
        "performance_verified": False,
    }
    if args.plan_only:
        print(json.dumps(report, indent=2))
        return 0
    if os.name != "posix":
        raise RuntimeError("Requires Linux+NPU; use --plan-only elsewhere")
    directory = args.report_dir or Path(tempfile.mkdtemp(prefix="vq2-decoder-input-"))
    if args.report_dir is not None:
        directory.mkdir(parents=True, exist_ok=False)
    try:
        snapshot = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True, timeout=20)
        (directory / "npu.log").write_text(snapshot.stdout + snapshot.stderr, encoding="utf-8")
        state = parse_snapshot(snapshot.stdout, args.physical_npu)
        if state == "unknown" or (state == "busy" and not args.allow_busy):
            report.update(status="BLOCKED", device_state=state)
        else:
            environment = child_environment(args)
            environment.setdefault("TASK_QUEUE_ENABLE", "1")
            result = run_child(command, environment, directory / "validation.log", args.timeout_s)
            report.update(status=result["status"], result=result)
            if result["status"] == "PASS":
                validate_child_evidence(result, args.queue_lifetime)
                report["device_execution_verified"] = True
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        traceback.print_exc()
    finally:
        summary = directory / "summary.json"
        summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"V4_DECODER_INPUT_PLAN={report['status']} SUMMARY={summary}", flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
