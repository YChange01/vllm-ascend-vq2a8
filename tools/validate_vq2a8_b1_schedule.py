#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate K: exact B1 tile-major scheduling gate, not speed acceptance.

The candidate and reference receive the same prepared FP8 bytes and FP32
scales/biases. By default reorder remains the original vectorized implementation;
--reorder-chunks 2/4 separately validates J+K. No change to tile shape, K order, Mmad,
scale/bias or BF16 rounding is permitted by this gate.
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

from tools import validate_vq2a8_reorder_row_reuse as fixtures_common
from tools import validate_vq2a8_tail_reorder as common
from tools.diagnose_vq2a8_tp1_startup import emit, parse_snapshot, run_child, stage_recorder

CASE = "v4_v2_b1_schedule"
REPO = Path(__file__).resolve().parents[1]
LIBRARY_NAME = common.LIBRARY_NAME
WIDTHS = common.WIDTHS
EXPERTS = common.EXPERTS
INVALID_IDS = common.INVALID_IDS
QUEUE_ITERATIONS = 513
QUEUE_TEMPLATES = 24
CONTRACT_CASES = 11
GRAPH_CASES = ("finite", "changed", "invalid", "all_invalid", "recovered", "duplicate")
DISPATCH_SCOPE = "m1_tile_major_explicit_reorder_chunks"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc-v4-v2-b1-schedule" / LIBRARY_NAME)
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--queue-lifetime", action="store_true")
    parser.add_argument(
        "--reorder-chunks",
        type=int,
        choices=(0, 2, 4),
        default=0,
        help="0 isolates K; 2/4 validates the explicit J+K combination",
    )
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.physical_npu < 0 or args.timeout_s <= 0 or (args.child and args.plan_only):
        parser.error("Require physical NPU >=0, positive timeout, and no child plan-only")
    args.launch_blocking = "0"
    return args


def child_command(args):
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
        "--reorder-chunks",
        str(args.reorder_chunks),
    ] + (["--queue-lifetime"] if args.queue_lifetime else [])


def require_abi(native, reorder_chunks=0):
    try:
        version = native.b1_schedule_version()
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError("B1 schedule ABI missing; no fallback") from error
    if type(version) is not int or version != 1:
        raise RuntimeError(f"B1 schedule requires independent ABI 1, got {version!r}")
    if reorder_chunks:
        try:
            version = native.activation_reorder_chunk_reuse_version()
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError("J+K requires chunk-reuse ABI; no fallback") from error
        if type(version) is not int or version != 1:
            raise RuntimeError(f"J+K requires chunk-reuse ABI 1, got {version!r}")


def project_candidate(bank, inputs, reorder_chunks=0):
    # Zero is the original vectorized reorder; one is tile-major B1 scheduling.
    return bank.project_candidate(*inputs, reorder_chunks, 1)


def inputs_fixture(fixture, groups, *, offset=0, ids=None, rank3=False):
    return fixtures_common.inputs_fixture(fixture, groups, "finite", offset=offset, ids=ids, rank3=rank3)


def baseline(fixture, inputs, literal_ids):
    import torch

    output, valid = fixture.bank.project_vectorized(*inputs)
    expected_valid = torch.tensor([int(0 <= slot < EXPERTS) for slot in literal_ids], dtype=torch.int32)
    common.assert_bits(valid.cpu(), expected_valid, "baseline_route_validity")
    for row, slot in enumerate(literal_ids):
        if 0 <= slot < EXPERTS:
            if not bool(torch.isfinite(output[row]).all()):
                raise AssertionError("Finite synthetic reference must produce finite valid output")
        elif not bool(torch.isnan(output[row]).all()):
            raise AssertionError("Invalid reference route must be completely NaN-poisoned")
    return output.cpu().clone(), valid.cpu().clone()


def checked_case(fixture, inputs, literal_ids, owner, name, reorder_chunks=0):
    snapshots = [common.bytes_cpu(tensor) for tensor in (*inputs, owner)]
    reference = baseline(fixture, inputs, literal_ids)
    actual = project_candidate(fixture.bank, inputs, reorder_chunks)
    common.check_projected(fixture, actual, reference, name)
    for tensor, snapshot in zip((*inputs, owner), snapshots):
        common.assert_bits(common.bytes_cpu(tensor), snapshot, name + "_input_unchanged")


def numeric_names():
    return [
        f"k{k}_g{g}_{routes}_rank{rank}"
        for k in WIDTHS
        for g in range(1, 7)
        for routes in ("sequential", "duplicate")
        for rank in (2, 3)
    ]


def run_numeric_checks(fixtures, stage, reorder_chunks=0):
    names = []
    for k, fixture in fixtures.items():
        for groups in range(1, 7):
            for routes in ("sequential", "duplicate"):
                for rank in (2, 3):
                    name = f"k{k}_g{groups}_{routes}_rank{rank}"
                    with stage(name):
                        inputs, ids, owner = inputs_fixture(
                            fixture,
                            groups,
                            offset=groups,
                            ids=[2] * groups if routes == "duplicate" else None,
                            rank3=rank == 3,
                        )
                        checked_case(fixture, inputs, ids, owner, name, reorder_chunks)
                    names.append(name)
    return names


def invalid_names():
    return [
        f"k{k}_invalid{i}_{state}"
        for k in WIDTHS
        for i in range(len(INVALID_IDS))
        for state in ("invalid", "all_invalid", "recovered")
    ]


def run_invalid_checks(fixtures, stage, reorder_chunks=0):
    names = []
    for k, fixture in fixtures.items():
        for index, invalid in enumerate(INVALID_IDS):
            inputs, ids, owner = inputs_fixture(fixture, 6, ids=[0, 1, 2, 0, 1, invalid])
            for state in ("invalid", "all_invalid", "recovered"):
                name = f"k{k}_invalid{index}_{state}"
                with stage(name):
                    if state == "all_invalid":
                        ids[:] = [invalid] * 6
                        inputs[-1].fill_(invalid)
                    elif state == "recovered":
                        ids[:] = [2] * 6
                        inputs[-1].fill_(2)
                    checked_case(fixture, inputs, ids, owner, name, reorder_chunks)
                names.append(name)
    return names


def run_contract_checks(fixture, stage, reorder_chunks=0):
    import torch

    (x, scale, bias, ids), _, _ = inputs_fixture(fixture, 6)
    bad_inputs = [
        (x.float(), scale, bias, ids),
        (
            x.view(torch.uint8)[:, None].repeat(1, 2, 1).view(x.dtype),
            scale[:, None].repeat(1, 2),
            bias[:, None].repeat(1, 2),
            ids,
        ),
        (x[:, ::2], scale, bias, ids),
        (x, scale.long(), bias, ids),
        (x, scale, bias[:, None], ids),
        (x, scale, bias, ids.int()),
        (x[:0], scale[:0], bias[:0], ids[:0]),
        (x, scale[:-1], bias, ids),
        (x, scale, bias, torch.zeros(7, dtype=torch.int64, device=x.device)),
        (
            torch.empty(x.numel() + 1, dtype=torch.uint8, device=x.device)[1:].view(x.dtype).reshape(x.shape),
            scale,
            bias,
            ids,
        ),
    ]
    count = 0
    with stage("native_metadata_rejections"):
        for values in bad_inputs:
            try:
                project_candidate(fixture.bank, values, reorder_chunks)
            except RuntimeError:
                count += 1
            else:
                raise AssertionError("B1 schedule accepted invalid native metadata")
        with torch.npu.stream(torch.npu.Stream()):
            try:
                project_candidate(fixture.bank, (x, scale, bias, ids), reorder_chunks)
            except RuntimeError:
                count += 1
            else:
                raise AssertionError("B1 schedule accepted a different bank stream")
    if count != CONTRACT_CASES:
        raise AssertionError("B1 schedule native contract coverage incomplete")
    return count


def graph_names():
    return [f"graph_k{k}_g{g}_{case}" for k in WIDTHS for g in (1, 6) for case in GRAPH_CASES]


def run_graph_checks(device, factory, stage, reorder_chunks=0):
    import torch

    names = []
    for k in WIDTHS:
        for groups in (1, 6):
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                with stage(f"graph_k{k}_g{groups}_prepare"):
                    fixture = common.make_fixture(k, device, factory)
                    static, _, owner = inputs_fixture(fixture, groups)
                    for _ in range(2):
                        project_candidate(fixture.bank, static, reorder_chunks)
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        projected = project_candidate(fixture.bank, static, reorder_chunks)
                for index, case in enumerate(GRAPH_CASES):
                    name = f"graph_k{k}_g{groups}_{case}"
                    with stage(name):
                        ids = [(row + index) % EXPERTS for row in range(groups)]
                        if case == "invalid":
                            ids[-1] = (1 << 63) - 1
                        elif case == "all_invalid":
                            ids[:] = [-1] * groups
                        elif case == "duplicate":
                            ids[:] = [1] * groups
                        current, _, _ = inputs_fixture(fixture, groups, offset=17 * index, ids=ids)
                        for dst, src in zip(static, current):
                            dst.view(torch.uint8).copy_(src.view(torch.uint8))
                        snapshots = [common.bytes_cpu(tensor) for tensor in (*static, owner)]
                        reference = baseline(fixture, static, ids)
                        graph.replay()
                        common.check_projected(fixture, projected, reference, name)
                        for tensor, snapshot in zip((*static, owner), snapshots):
                            common.assert_bits(common.bytes_cpu(tensor), snapshot, name + "_input_unchanged")
                    names.append(name)
                with stage(f"graph_k{k}_g{groups}_reset"):
                    stream.synchronize()
                    graph.reset()
    return names


def queue_evidence(reorder_chunks=0):
    return {
        **common.queue_evidence(),
        "iterations": QUEUE_ITERATIONS,
        "input_templates": QUEUE_TEMPLATES,
        "oracle": "same_device_original_vectorized_projection_exact_bytes",
        "reorder_chunks": reorder_chunks,
        "schedule": 1,
    }


def run_queue_checks(device, factory, stage, reorder_chunks=0):
    import torch

    templates = []
    with stage("queue_preupload_and_oracles"):
        queue_fixtures = {k: common.make_fixture(k, device, factory) for k in WIDTHS}
        for index in range(QUEUE_TEMPLATES):
            fixture = queue_fixtures[WIDTHS[index % len(WIDTHS)]]
            groups = index // 2 % 6 + 1
            ids = [(row + index) % EXPERTS for row in range(groups)]
            if index >= QUEUE_TEMPLATES // 2:
                ids[-1] = INVALID_IDS[index % len(INVALID_IDS)]
            inputs, ids, owner = inputs_fixture(fixture, groups, offset=index, ids=ids)
            reference = baseline(fixture, inputs, ids)
            templates.append((fixture, inputs, reference, [common.bytes_cpu(tensor) for tensor in inputs]))
        del inputs, owner
        torch.npu.synchronize()
    pending = []
    with stage("queue_owner_release_and_allocation_pressure"):
        for iteration in range(QUEUE_ITERATIONS):
            fixture, template, reference, snapshots = templates[iteration % QUEUE_TEMPLATES]
            inputs = tuple(tensor.view(torch.uint8).clone().view(tensor.dtype) for tensor in template)
            projected = project_candidate(fixture.bank, inputs, reorder_chunks)
            ordinary = [tensor.reshape(-1).view(torch.uint8).clone() for tensor in inputs]
            geometry = SimpleNamespace(k=fixture.k, device=fixture.device)
            pending.append((geometry, projected, reference, ordinary, snapshots))
            del inputs
            pressure = torch.empty(common.PRESSURE_BYTES, dtype=torch.uint8, device=device)
            pressure.fill_(iteration % 256)
            del pressure
        templates.clear()
        queue_fixtures.clear()
        del fixture, template
        pressure = [
            torch.empty(common.PRESSURE_BYTES, dtype=torch.uint8, device=device).fill_(chunk + 1)
            for chunk in range(common.POST_RELEASE_PRESSURE_CHUNKS)
        ]
        del pressure
        torch.npu.synchronize()
    with stage("queue_lifetime_verify"):
        for index, (geometry, projected, reference, ordinary, snapshots) in enumerate(pending):
            common.check_projected(geometry, projected, reference, f"queue{index}")
            for got, want in zip(ordinary, snapshots):
                common.assert_bits(got.cpu(), want, f"queue{index}_input")
    return queue_evidence(reorder_chunks)


def validate_child_evidence(result, *, queue_lifetime, reorder_chunks=0):
    events = result.get("events", [])
    final = next((event for event in reversed(events) if event.get("event") == "CASE_PASS"), {})
    expected = {
        "numeric": numeric_names(),
        "invalid": invalid_names(),
        "native_contract": CONTRACT_CASES,
        "graph": graph_names(),
    }
    if queue_lifetime:
        expected["queue_lifetime"] = {**queue_evidence(reorder_chunks), "task_queue_enable": "1"}
    identity = final.get("library", {})
    digest = identity.get("sha256")
    required_passes = {"final_sync"} | ({"queue_lifetime_verify"} if queue_lifetime else set())
    passes = {event.get("stage") for event in events if event.get("event") == "PASS"}
    if (
        result.get("status") != "PASS"
        or result.get("exit_code") != 0
        or result.get("reaped") is not True
        or any(event.get("event") in ("FAIL", "CASE_FAIL") for event in events)
        or final.get("case") != CASE
        or type(final.get("native_abi")) is not int
        or final.get("native_abi") != 1
        or final.get("dispatch_scope") != DISPATCH_SCOPE
        or type(final.get("reorder_chunks")) is not int
        or final.get("reorder_chunks") != reorder_chunks
        or final.get("results") != expected
        or final.get("device_execution_verified") is not True
        or final.get("graph_verified") is not True
        or final.get("model_integration_verified") is not False
        or final.get("performance_verified") is not False
        or not isinstance(identity.get("path"), str)
        or Path(identity["path"]).name != LIBRARY_NAME
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        or not required_passes.issubset(passes)
    ):
        raise ValueError("Incomplete B1 schedule child evidence")


def run_case_child(args):
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical_npu):
        raise ValueError("Physical NPU mapping differs from request")
    faulthandler.enable()
    faulthandler.dump_traceback_later(min(30, args.timeout_s), repeat=True)
    synchronize = lambda: None
    stage = stage_recorder(CASE, lambda: synchronize())
    try:
        queue_mode = os.environ.get("TASK_QUEUE_ENABLE")
        if queue_mode not in ("0", "1") or (args.queue_lifetime and queue_mode != "1"):
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
                raise ValueError("Wrong native library name; no fallback")
            identity = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            torch.ops.load_library(str(path))
            require_abi(torch.ops.vq2a8_ascendc_v4_v2, args.reorder_chunks)
            factory = torch.classes.vq2a8_ascendc_v4_v2.ResidentBank
            emit(
                CASE,
                "INFO",
                library=identity,
                device=info,
                native_abi=1,
                dispatch_scope=DISPATCH_SCOPE,
                reorder_chunks=args.reorder_chunks,
            )
        with torch.inference_mode():
            with stage("synthetic_nonzero_banks"):
                fixtures = {k: common.make_fixture(k, device, factory) for k in WIDTHS}
            results = {
                "numeric": run_numeric_checks(fixtures, stage, args.reorder_chunks),
                "invalid": run_invalid_checks(fixtures, stage, args.reorder_chunks),
                "native_contract": run_contract_checks(fixtures[2048], stage, args.reorder_chunks),
                "graph": run_graph_checks(device, factory, stage, args.reorder_chunks),
            }
            if args.queue_lifetime:
                results["queue_lifetime"] = {
                    **run_queue_checks(device, factory, stage, args.reorder_chunks),
                    "task_queue_enable": queue_mode,
                }
        with stage("final_sync"):
            pass
        emit(
            CASE,
            "CASE_PASS",
            results=results,
            native_abi=1,
            dispatch_scope=DISPATCH_SCOPE,
            reorder_chunks=args.reorder_chunks,
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
        "scope": "b1_work_enumeration_only",
        "dispatch_scope": DISPATCH_SCOPE,
        "reorder_chunks": args.reorder_chunks,
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
        raise RuntimeError("B1 schedule validation requires Linux + NPU; use --plan-only elsewhere")
    directory = args.report_dir or Path(tempfile.mkdtemp(prefix="vq2-b1-schedule-"))
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
                child_command(args), common.probe_environment(args), directory / "validation.log", args.timeout_s
            )
            report.update(status=result["status"], result=result)
            if result["status"] == "PASS":
                validate_child_evidence(result, queue_lifetime=args.queue_lifetime, reorder_chunks=args.reorder_chunks)
                report.update(device_execution_verified=True, graph_verified=True)
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        traceback.print_exc()
    finally:
        summary = directory / "summary.json"
        summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"V4_B1_SCHEDULE={report['status']} SUMMARY={summary}", flush=True)
    return 0 if report["status"] == "PASS" else 130 if report["status"] == "INTERRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
