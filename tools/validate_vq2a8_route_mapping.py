#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded integer route-mapping acceptance, not model/performance acceptance.

The oracle preserves safe-clamp/gather/where and all(slots >= 0) semantics.
Positive out-of-bank lookup values are intentionally not rejected here: the
downstream resident bank owns that check. No weights or model are loaded.
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
import random
import subprocess
import tempfile
import traceback
from pathlib import Path

from tools.diagnose_vq2a8_tp1_startup import child_environment, emit, parse_snapshot, run_child, stage_recorder

CASE = "v4_v2_route_mapping"
REPO = Path(__file__).resolve().parents[1]
LIBRARY_NAME = "libvq2a8_ascendc_v4_v2.so"
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
LOOKUP_SIZES = (1, 2, 256)
PATTERNS = (
    "sequential",
    "duplicate",
    "low",
    "high",
    "min",
    "max",
    "negative_lookup",
    "large_lookup",
    "sparse",
    "random",
)
GRAPH_CASES = (
    "valid",
    "low",
    "recovered_low",
    "max",
    "recovered_max",
    "negative_lookup",
    "large_lookup",
    "recovered_lookup",
)
QUEUE_ITERATIONS = 513
QUEUE_TEMPLATE_COUNT = math.lcm(6, len(LOOKUP_SIZES), len(PATTERNS), 2)
PRESSURE_BYTES = 2 * 1024 * 1024
NATIVE_CONTRACT_CASES = 12


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc-v4-v2-route-mapping" / LIBRARY_NAME)
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


def probe_environment(args, environ=None):
    environment = child_environment(args, environ)
    environment.setdefault("TASK_QUEUE_ENABLE", "1")
    return environment


def case_values(groups, size, pattern):
    """Deterministic Python inputs, including values unsafe for truncated IDs."""
    if groups not in range(1, 7) or size not in LOOKUP_SIZES or pattern not in PATTERNS:
        raise ValueError("Unsupported probe fixture")
    lookup = list(range(size))
    ids = [index % size for index in range(groups)]
    if pattern == "duplicate":
        ids = [size - 1] * groups
    elif pattern in ("low", "high", "min", "max"):
        ids[-1] = {"low": -1, "high": size, "min": INT64_MIN, "max": INT64_MAX}[pattern]
    elif pattern in ("negative_lookup", "large_lookup"):
        ids[-1] = size - 1
        lookup[-1] = INT64_MIN if pattern == "negative_lookup" else INT64_MAX
    elif pattern == "sparse":
        lookup = [index // 2 if index % 2 else -1 for index in range(size)]
        ids = [(index * 37) % size for index in range(groups)]
    elif pattern == "random":
        generator = random.Random(1701 + groups * 257 + size)
        lookup = [generator.choice((-1, 0, 1, 255, INT64_MAX)) for _ in range(size)]
        ids = [generator.choice((INT64_MIN, -1, 0, size - 1, size, INT64_MAX)) for _ in range(groups)]
    return ids, lookup


def literal_reference(ids, lookup):
    slots = [lookup[index] if 0 <= index < len(lookup) else -1 for index in ids]
    return slots, all(slot >= 0 for slot in slots)


def reference(ids, lookup):
    import torch

    in_range = (ids >= 0) & (ids < lookup.numel())
    mapped = lookup.index_select(0, ids.clamp(0, lookup.numel() - 1))
    slots = torch.where(in_range, mapped, -1).contiguous()
    return slots, (slots >= 0).all()


def fixture(device, groups, size, pattern, *, offset=0):
    import torch

    ids, lookup = case_values(groups, size, pattern)
    # Slice only after upload so nonzero storage offsets remain visible to the
    # actual binding. The leading/trailing sentinels also detect stray writes.
    owners = [torch.tensor([73] * offset + values + [91], dtype=torch.int64, device=device) for values in (ids, lookup)]
    views = tuple(owner[offset:-1] for owner in owners)
    return views, owners


def assert_equal(actual, expected, name):
    import torch

    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        raise AssertionError(f"{name}: require slots and validity outputs")
    for index, (got, want) in enumerate(zip(actual, expected)):
        if got.dtype != want.dtype or got.shape != want.shape or got.device != want.device:
            raise AssertionError(f"{name}_{index}: output dtype/shape/device mismatch")
        if not torch.equal(got.cpu(), want.cpu()):
            raise AssertionError(f"{name}_{index}: exact integer/BOOL mismatch; no relaxed tolerance")


def numerical_case_names():
    return [
        f"g{groups}_n{size}_{pattern}_offset{offset}"
        for groups in range(1, 7)
        for size in LOOKUP_SIZES
        for pattern in PATTERNS
        for offset in (0, 1)
    ]


def run_numeric_checks(device, checker, stage):
    import torch

    names = []
    for groups in range(1, 7):
        for size in LOOKUP_SIZES:
            for pattern in PATTERNS:
                for offset in (0, 1):
                    name = f"g{groups}_n{size}_{pattern}_offset{offset}"
                    with stage(name):
                        values, owners = fixture(device, groups, size, pattern, offset=offset)
                        before = [owner.clone() for owner in owners]
                        slots, valid = literal_reference(*case_values(groups, size, pattern))
                        expected = (
                            torch.tensor(slots, dtype=torch.int64, device=device),
                            torch.tensor(valid, device=device),
                        )
                        assert_equal(reference(*values), expected, f"{name}_torch_oracle")
                        assert_equal(checker(*values), expected, name)
                        for owner, original in zip(owners, before):
                            if not torch.equal(owner.cpu(), original.cpu()):
                                raise AssertionError(f"{name}: input storage/sentinels modified")
                    names.append(name)
    return names


def run_native_contract_checks(device, native, stage):
    import torch

    with stage("native_metadata_rejections"):
        values, _ = fixture(device, 6, 256, "sequential")
        ids, lookup = values
        bad_inputs = [
            (ids[:0], lookup),
            (torch.zeros(7, dtype=torch.int64, device=device), lookup),
            (ids.reshape(2, 3), lookup),
            (ids.int(), lookup),
            (torch.zeros(12, dtype=torch.int64, device=device)[::2], lookup),
            (ids, lookup[:0]),
            (ids, torch.zeros(257, dtype=torch.int64, device=device)),
            (ids, lookup.reshape(2, 128)),
            (ids, lookup.int()),
            (ids, torch.zeros(512, dtype=torch.int64, device=device)[::2]),
            (ids.cpu(), lookup),
            (ids, lookup.cpu()),
        ]
        for index, inputs in enumerate(bad_inputs):
            try:
                native.route_mapping(*inputs)
            except (RuntimeError, NotImplementedError):
                continue
            raise AssertionError(f"Invalid native route metadata accepted: case {index}")
        if len(bad_inputs) != NATIVE_CONTRACT_CASES:
            raise AssertionError("Incomplete native contract coverage")
    return len(bad_inputs)


def graph_case_names():
    return [f"graph_g{groups}_n{size}_{case}" for groups in (1, 6) for size in (1, 256) for case in GRAPH_CASES]


def run_graph_checks(device, checker, stage):
    import torch

    names = []
    for groups in (1, 6):
        for size in (1, 256):
            with stage(f"graph_prepare_g{groups}_n{size}"):
                values, owners = fixture(device, groups, size, "sequential", offset=1)
                for _ in range(2):
                    checker(*values)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    actual = checker(*values)
            # On a driver failure propagate immediately, never graph.reset()
            # from finally: the bounded parent will terminate a stuck child.
            for case in GRAPH_CASES:
                name = f"graph_g{groups}_n{size}_{case}"
                with stage(name):
                    pattern = case if case in PATTERNS else "sequential"
                    fresh, _ = fixture(device, groups, size, pattern)
                    for target, source in zip(values, fresh):
                        target.copy_(source)
                    before = [owner.clone() for owner in owners]
                    wanted_slots, wanted_valid = literal_reference(*case_values(groups, size, pattern))
                    expected = (
                        torch.tensor(wanted_slots, dtype=torch.int64, device=device),
                        torch.tensor(wanted_valid, dtype=torch.bool, device=device),
                    )
                    graph.replay()
                    assert_equal(actual, expected, name)
                    assert_equal(reference(*values), expected, f"{name}_torch_oracle")
                    for owner, original in zip(owners, before):
                        if not torch.equal(owner.cpu(), original.cpu()):
                            raise AssertionError(f"{name}: input storage/sentinels modified")
                names.append(name)
            with stage(f"graph_reset_g{groups}_n{size}"):
                torch.npu.synchronize()
                graph.reset()
    return names


def queue_evidence():
    return {
        "iterations": QUEUE_ITERATIONS,
        "all_outputs_checked": True,
        "ordinary_outputs_checked": True,
        "owners_dropped_before_fence": True,
        "input_upload_before_loop": True,
        "input_templates": QUEUE_TEMPLATE_COUNT,
        "fresh_device_clone_owners_each_iteration": True,
        "explicit_per_iteration_synchronize": False,
        "allocation_pressure_bytes_per_iteration": PRESSURE_BYTES,
        "runtime_queue_slots_measured": False,
    }


def prepare_queue_templates(device):
    """Upload one LCM period before the asynchronous stress loop begins."""
    templates = []
    for iteration in range(QUEUE_TEMPLATE_COUNT):
        groups, size = iteration % 6 + 1, LOOKUP_SIZES[iteration % len(LOOKUP_SIZES)]
        pattern, offset = PATTERNS[iteration % len(PATTERNS)], iteration % 2
        _, owners = fixture(device, groups, size, pattern, offset=offset)
        ids, lookup = case_values(groups, size, pattern)
        wanted_slots, wanted_valid = literal_reference(ids, lookup)
        templates.append(
            {
                "owners": owners,
                "offset": offset,
                "expected_slots": wanted_slots,
                "expected_valid": wanted_valid,
                "expected_ordinary": [index >= 0 for index in ids],
            }
        )
    return templates


def run_queue_checks(device, checker, stage):
    import torch

    actual_slots, actual_valid, ordinary = [], [], []
    expected_slots, expected_valid, expected_ordinary = [], [], []
    # torch.tensor(Python values, device=NPU) can entail a synchronous H2D
    # transfer. Finish all uploads before stressing asynchronous owner release.
    templates = prepare_queue_templates(device)
    torch.npu.synchronize()
    with stage("queue_owner_release_and_allocation_pressure"):
        for iteration in range(QUEUE_ITERATIONS):
            template = templates[iteration % len(templates)]
            # Fresh owning device allocations, not aliases of the permanent
            # templates. clone is a device copy; the loop has no H2D upload.
            owners = [owner.clone() for owner in template["owners"]]
            values = tuple(owner[template["offset"] : -1] for owner in owners)
            slots, valid = checker(*values)
            actual_slots.append(slots)
            actual_valid.append(valid)
            ordinary.append(values[0] >= 0)
            expected_slots.extend(template["expected_slots"])
            expected_valid.append(template["expected_valid"])
            expected_ordinary.extend(template["expected_ordinary"])
            del values, owners
            pressure = torch.empty(PRESSURE_BYTES, dtype=torch.uint8, device=device)
            pressure.fill_(0 if iteration % 2 else 0xFF)
            del pressure
        # No device host read or explicit synchronization inside the loop.
        # Assert every retained output, not merely the final queue element.
        torch.npu.synchronize()
        for actual, expected, dtype, name in (
            (torch.cat(actual_slots), expected_slots, torch.int64, "slots"),
            (torch.stack(actual_valid), expected_valid, torch.bool, "validity"),
            (torch.cat(ordinary), expected_ordinary, torch.bool, "ordinary"),
        ):
            if not torch.equal(actual.cpu(), torch.tensor(expected, dtype=dtype)):
                raise AssertionError(f"Queue owner-release {name} mismatch")
    return queue_evidence()


def validate_child_evidence(result, *, queue_lifetime):
    events = result.get("events", [])
    final = next((event for event in reversed(events) if event.get("event") == "CASE_PASS"), {})
    values = final.get("results", {})
    identity = final.get("library", {})
    digest = identity.get("sha256")
    expected_keys = {"numeric", "native_contract", "graph"} | ({"queue_lifetime"} if queue_lifetime else set())
    if (
        result.get("status") != "PASS"
        or result.get("exit_code") != 0
        or result.get("reaped") is not True
        or any(event.get("event") in ("FAIL", "CASE_FAIL") for event in events)
        or final.get("case") != CASE
        or type(final.get("native_abi")) is not int
        or final.get("native_abi") != 1
        or final.get("device_execution_verified") is not True
        or final.get("graph_verified") is not True
        or final.get("model_integration_verified") is not False
        or final.get("performance_verified") is not False
        or not isinstance(identity.get("path"), str)
        or Path(identity["path"]).name != LIBRARY_NAME
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or set(values) != expected_keys
        or values.get("numeric") != numerical_case_names()
        or values.get("native_contract") != NATIVE_CONTRACT_CASES
        or values.get("graph") != graph_case_names()
        or not any(event.get("event") == "PASS" and event.get("stage") == "final_sync" for event in events)
    ):
        raise ValueError("Incomplete route-mapping child evidence")
    if queue_lifetime and values["queue_lifetime"] != {**queue_evidence(), "task_queue_enable": "1"}:
        raise ValueError("Incomplete route-mapping queue evidence")


def run_case_child(args):
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical_npu):
        raise ValueError("Child physical NPU mapping differs from the requested device")
    faulthandler.enable()
    faulthandler.dump_traceback_later(min(30, args.timeout_s), repeat=True)
    synchronize = lambda: None
    stage = stage_recorder(CASE, lambda: synchronize())
    try:
        queue_mode = os.environ.get("TASK_QUEUE_ENABLE")
        if queue_mode not in ("0", "1") or (args.queue_lifetime and queue_mode != "1"):
            raise ValueError("NPU graph requires TASK_QUEUE_ENABLE=0/1; --queue-lifetime requires 1")
        with stage("imports"):
            import torch
            import torch_npu  # noqa: F401

            from tools.validate_vq2a8_ascendc import require_hardware_runtime
            from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device
            from vllm_ascend.quantization.vq2a8_route_mapping import FusedRouteMapping
        with stage("device"):
            require_hardware_runtime()
            if torch.npu.device_count() != 1:
                raise RuntimeError("Require exactly one visible NPU; refusing ambiguous mapping")
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
            checker = FusedRouteMapping(native_ops=native)
            emit(CASE, "INFO", library=identity, device=info, native_abi=1, task_queue_enable=queue_mode)
        with torch.inference_mode():
            results = {
                "numeric": run_numeric_checks(device, checker, stage),
                "native_contract": run_native_contract_checks(device, native, stage),
                "graph": run_graph_checks(device, checker, stage),
            }
            if args.queue_lifetime:
                results["queue_lifetime"] = {
                    **run_queue_checks(device, checker, stage),
                    "task_queue_enable": queue_mode,
                }
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
        "scope": "integer_route_mapping_only",
        "command": child_command(args),
        "status": "PLANNED",
        "native_abi": 1,
        "device_execution_verified": False,
        "graph_verified": False,
        "model_integration_verified": False,
        "performance_verified": False,
        "queue_lifetime_requested": args.queue_lifetime,
    }
    if args.plan_only:
        print(json.dumps(report, indent=2))
        return 0
    if os.name != "posix":
        raise RuntimeError("Route mapping validation requires Linux + NPU; use --plan-only elsewhere")
    directory = args.report_dir or Path(tempfile.mkdtemp(prefix="vq2-route-mapping-"))
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
                child_command(args), probe_environment(args), directory / "validation.log", args.timeout_s
            )
            report.update(status=result["status"], result=result)
            if result["status"] == "PASS":
                validate_child_evidence(result, queue_lifetime=args.queue_lifetime)
                report.update(device_execution_verified=True, graph_verified=True)
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        traceback.print_exc()
    finally:
        summary = directory / "summary.json"
        summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"V4_ROUTE_MAPPING={report['status']} SUMMARY={summary}", flush=True)
    return 0 if report["status"] == "PASS" else 130 if report["status"] == "INTERRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
