#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded exact layer-validity acceptance, not model/performance acceptance.

Compares six INT32 nonzero predicates, three BF16 finite scans and every route
BOOL scalar against the unchanged Torch expression. Includes live graph
mutation/recovery and optional asynchronous owner-release pressure.
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

CASE = "v4_v2_validity_fused"
REPO = Path(__file__).resolve().parents[1]
LIBRARY_NAME = "libvq2a8_ascendc_v4_v2.so"
QUEUE_ITERATIONS = 513
PRESSURE_BYTES = 2 * 1024 * 1024
WIDTH_PAIRS = ((2048, 2048), (2048, 4096), (4096, 2048), (4096, 4096))
GRAPH_CASES = ("valid", "status", "recovered_status", "output", "recovered_output", "route", "recovered_route")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc-v4-v2-validity" / LIBRARY_NAME)
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


def fixture(device, groups=6, gate_width=4096, down_width=4096, flag_count=8, *, projected_3d=False, value=0.5):
    import torch

    statuses = [torch.full((groups,), 1 if i % 2 == 0 else -7, dtype=torch.int32, device=device) for i in range(6)]
    outputs = [
        torch.full((groups, gate_width), value, dtype=torch.bfloat16, device=device),
        torch.full((groups, down_width), value, dtype=torch.bfloat16, device=device),
        torch.full((1, down_width), value, dtype=torch.bfloat16, device=device),
    ]
    if projected_3d:
        outputs[:2] = [output.unsqueeze(1) for output in outputs[:2]]
    flags = [torch.tensor(True, dtype=torch.bool, device=device) for _ in range(flag_count)]
    return statuses, outputs, flags


def reference(statuses, outputs, route_flags):
    import torch

    predicates = [(status != 0).all() for status in statuses]
    predicates.extend(torch.isfinite(output).all() for output in outputs)
    predicates.extend(route_flags)
    return torch.stack(predicates).all()


def assert_equal(actual, expected, name):
    import torch

    if actual.dtype != torch.bool or actual.shape != () or actual.device != expected.device:
        raise AssertionError(f"{name}: require matching device BOOL scalar")
    if not torch.equal(actual.cpu(), expected.cpu()):
        raise AssertionError(f"{name}: predicate mismatch; no relaxed tolerance")


def numerical_case_names():
    return [
        f"g{groups}_gate{gate}_down{down}_flags{flags}"
        for groups in range(1, 7)
        for gate, down in WIDTH_PAIRS
        for flags in (0, 8)
    ]


def run_numeric_checks(device, checker, stage):
    import torch

    names = []
    pattern_bits = torch.arange(0x7F80, dtype=torch.int32)
    pattern_bits = torch.cat((pattern_bits, pattern_bits + 0x8000)).to(torch.uint16)
    finite_pattern = pattern_bits.view(torch.bfloat16)
    pattern_cursor = 0
    for groups in range(1, 7):
        for gate, down in WIDTH_PAIRS:
            for flag_count in (0, 8):
                name = f"g{groups}_gate{gate}_down{down}_flags{flag_count}"
                with stage(name):
                    values = fixture(device, groups, gate, down, flag_count, projected_3d=groups % 2 == 0)
                    # Both signs of every finite BF16 bit pattern, distributed
                    # across the output buffers. Classification must not reject
                    # subnormals, signed zero, or the largest finite values.
                    for output in values[1]:
                        offset = pattern_cursor % finite_pattern.numel()
                        pattern = finite_pattern.roll(-offset).repeat(
                            (output.numel() + finite_pattern.numel() - 1) // finite_pattern.numel()
                        )
                        output.copy_(pattern[: output.numel()].reshape(output.shape).to(device))
                        pattern_cursor += output.numel()
                    assert_equal(checker(*values), reference(*values), name)
                names.append(name)
    return names


def run_invalid_checks(device, checker, stage):
    with stage("every_status_route_and_output_position"):
        values = fixture(device)
        statuses, outputs, flags = values
        count = 0
        for status in statuses:
            for row in range(status.numel()):
                status[row] = 0
                assert_equal(checker(*values), reference(*values), f"status_{count}")
                status[row] = -3
                assert_equal(checker(*values), reference(*values), f"recovered_status_{count}")
                count += 1
        for flag in flags:
            flag.fill_(False)
            assert_equal(checker(*values), reference(*values), f"route_{count}")
            flag.fill_(True)
            assert_equal(checker(*values), reference(*values), f"recovered_route_{count}")
            count += 1
        for output in outputs:
            for position in (0, output.numel() // 2, output.numel() - 1):
                for invalid in (float("nan"), float("inf"), -float("inf")):
                    output.view(-1)[position] = invalid
                    assert_equal(checker(*values), reference(*values), f"output_{count}")
                    output.view(-1)[position] = 0.5
                    assert_equal(checker(*values), reference(*values), f"recovered_output_{count}")
                    count += 1
        # Protect the oracle itself: malformed input must not compare equal only
        # because both paths accidentally returned True.
        if count != 71 or not bool(reference(*values)):
            raise AssertionError("Invalid predicate coverage incomplete")
    return {"invalid_cases": count, "recovery_after_each": True}


def run_nonfinite_pattern_checks(device, checker, stage):
    import torch

    # Do not construct Python float NaNs: that could canonicalize their payload
    # before the NPU ever sees signaling/quiet NaN and negative NaN patterns.
    bits = torch.arange(0x7F80, 0x8000, dtype=torch.int32)
    bits = torch.cat((bits, bits + 0x8000)).to(torch.int16)
    device_bits = bits.to(device)
    values = fixture(device)
    count = 0
    for chunk in range(0, bits.numel(), 16):
        with stage(f"nonfinite_bits_{chunk:03d}_{chunk + 15:03d}"):
            for index in range(chunk, chunk + 16):
                output = values[1][index % len(values[1])]
                position = (0, output.numel() // 2, output.numel() - 1)[(index // 3) % 3]
                raw = output.view(torch.int16).reshape(-1)[position : position + 1]
                raw.copy_(device_bits[index : index + 1])
                if not torch.equal(raw.cpu(), bits[index : index + 1]):
                    raise AssertionError(f"nonfinite_bits_{index}: input bit pattern was changed")
                expected = reference(*values)
                if bool(expected):
                    raise AssertionError(f"nonfinite_bits_{index}: reference accepted a nonfinite pattern")
                assert_equal(checker(*values), expected, f"nonfinite_bits_{index}")
                # 0x3f00 is exactly BF16 0.5, written through the integer view.
                raw.fill_(0x3F00)
                assert_equal(checker(*values), reference(*values), f"recovered_nonfinite_bits_{index}")
                count += 1
    if count != 256 or not bool(reference(*values)):
        raise AssertionError("Nonfinite BF16 bit-pattern coverage incomplete")
    return {"patterns": count, "raw_bits_verified": True, "recovery_after_each": True}


def run_native_contract_checks(device, native, stage):
    with stage("native_metadata_rejections"):
        statuses, outputs, flags = fixture(device)
        bad_calls = [
            lambda: native.layer_validity(statuses[:5], outputs, flags),
            lambda: native.layer_validity(statuses, outputs[:2], flags),
            lambda: native.layer_validity(statuses, outputs, flags + flags[:1]),
            lambda: native.layer_validity([statuses[0].long(), *statuses[1:]], outputs, flags),
            lambda: native.layer_validity([statuses[0][:5], *statuses[1:]], outputs, flags),
            lambda: native.layer_validity(statuses, [outputs[0].float(), *outputs[1:]], flags),
            lambda: native.layer_validity(statuses, [outputs[0][:, ::2], *outputs[1:]], flags),
            lambda: native.layer_validity(statuses, [outputs[0].cpu(), *outputs[1:]], flags),
            lambda: native.layer_validity(statuses, outputs, [flags[0].reshape(1), *flags[1:]]),
            lambda: native.layer_validity(statuses, outputs, [flags[0].int(), *flags[1:]]),
            lambda: native.layer_validity(statuses, [outputs[0], outputs[1], outputs[2][:, :2048]], flags),
        ]
        for call in bad_calls:
            try:
                call()
            except RuntimeError:
                continue
            raise AssertionError("Invalid native geometry was accepted")
    return len(bad_calls)


def run_graph_checks(device, checker, stage):
    import torch

    names = []
    for groups in (1, 6):
        with stage(f"graph_prepare_g{groups}"):
            values = fixture(device, groups, projected_3d=True)
            for _ in range(2):
                checker(*values)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                actual = checker(*values)
        # Reset only on a successful run. A failed driver replay must not enter
        # a potentially blocking graph.reset() before the child reports FAIL.
        for case in GRAPH_CASES:
            name = f"graph_g{groups}_{case}"
            with stage(name):
                values[0][-1].fill_(0 if case == "status" else -2)
                values[1][-1].fill_(float("nan") if case == "output" else -0.25)
                values[2][-1].fill_(case != "route")
                graph.replay()
                assert_equal(actual, reference(*values), name)
            names.append(name)
        with stage(f"graph_reset_g{groups}"):
            torch.npu.synchronize()
            graph.reset()
    return names


def run_queue_checks(device, checker, stage):
    import torch

    actual, ordinary, expected, expected_ordinary = [], [], [], []
    with stage("queue_owner_release_and_allocation_pressure"):
        for iteration in range(QUEUE_ITERATIONS):
            value = (iteration % 7 - 3) / 8
            values = fixture(device, value=value, projected_3d=True)
            case = iteration % 4
            if case == 1:
                values[0][-1][-1] = 0
            elif case == 2:
                values[1][-1][0, -1] = float("nan")
            elif case == 3:
                values[2][-1].fill_(False)
            actual.append(checker(*values))
            ordinary.append(values[1][0].float().sum())
            expected.append(case == 0)
            expected_ordinary.append(value * values[1][0].numel())
            del values
            pressure = torch.empty(PRESSURE_BYTES, dtype=torch.uint8, device=device)
            pressure.fill_(0 if iteration % 2 else 0xFF)
            del pressure
        # There is deliberately no host read or explicit synchronize inside
        # the iteration loop. Validate every queued result, not only the last.
        torch.npu.synchronize()
        if not torch.equal(torch.stack(actual).cpu(), torch.tensor(expected, dtype=torch.bool)):
            raise AssertionError("Queue owner-release validity mismatch")
        if not torch.equal(torch.stack(ordinary).cpu(), torch.tensor(expected_ordinary, dtype=torch.float32)):
            raise AssertionError("Queue ordinary-op result mismatch")
    return {
        "iterations": QUEUE_ITERATIONS,
        "all_outputs_checked": True,
        "owners_dropped_before_fence": True,
        "explicit_per_iteration_synchronize": False,
        "allocation_pressure_bytes_per_iteration": PRESSURE_BYTES,
        "runtime_queue_slots_measured": False,
    }


def validate_child_evidence(result, *, queue_lifetime):
    final = next((event for event in reversed(result.get("events", [])) if event.get("event") == "CASE_PASS"), {})
    expected_keys = {"numeric", "invalid", "nonfinite_patterns", "native_contract", "graph"} | (
        {"queue_lifetime"} if queue_lifetime else set()
    )
    values = final.get("results", {})
    if (
        final.get("case") != CASE
        or type(final.get("native_abi")) is not int
        or final.get("native_abi") != 1
        or final.get("device_execution_verified") is not True
        or final.get("graph_verified") is not True
        or set(values) != expected_keys
        or values.get("numeric") != numerical_case_names()
        or values.get("invalid") != {"invalid_cases": 71, "recovery_after_each": True}
        or values.get("nonfinite_patterns") != {"patterns": 256, "raw_bits_verified": True, "recovery_after_each": True}
        or values.get("native_contract") != 11
        or values.get("graph") != [f"graph_g{groups}_{case}" for groups in (1, 6) for case in GRAPH_CASES]
    ):
        raise ValueError("Incomplete layer-validity child evidence")
    if queue_lifetime:
        queue = values["queue_lifetime"]
        if (
            queue.get("iterations") != QUEUE_ITERATIONS
            or queue.get("all_outputs_checked") is not True
            or queue.get("owners_dropped_before_fence") is not True
            or queue.get("explicit_per_iteration_synchronize") is not False
            or queue.get("allocation_pressure_bytes_per_iteration") != PRESSURE_BYTES
            or queue.get("task_queue_enable") != "1"
        ):
            raise ValueError("Incomplete layer-validity queue evidence")


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
            from vllm_ascend.quantization.vq2a8_validity_fused import FusedLayerValidity
        with stage("device"):
            require_hardware_runtime()
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
            checker = FusedLayerValidity(native)
            emit(CASE, "INFO", library=identity, device=info, native_abi=1)
        with torch.inference_mode():
            results = {
                "numeric": run_numeric_checks(device, checker, stage),
                "invalid": run_invalid_checks(device, checker, stage),
                "nonfinite_patterns": run_nonfinite_pattern_checks(device, checker, stage),
                "native_contract": run_native_contract_checks(device, native, stage),
                "graph": run_graph_checks(device, checker, stage),
            }
            if args.queue_lifetime:
                results["queue_lifetime"] = run_queue_checks(device, checker, stage)
                results["queue_lifetime"]["task_queue_enable"] = queue_mode
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
        "scope": "layer_validity_only",
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
        raise RuntimeError("Layer validity validation requires Linux + NPU; use --plan-only elsewhere")
    directory = args.report_dir or Path(tempfile.mkdtemp(prefix="vq2-validity-fused-"))
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
        print(f"V4_FUSED_VALIDITY={report['status']} SUMMARY={summary}", flush=True)
    return 0 if report["status"] == "PASS" else 130 if report["status"] == "INTERRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
