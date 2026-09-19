#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate J bounded 2/4-chunk M=1 FP8-byte reorder gate, not speed acceptance.

Derived from the row-reuse gate; uses explicitly selected chunk-reuse ABI and
methods only. Reuses the tail-reorder fixture, byte/descriptor checks and bounded
supervisor. No FP32 clamp/cast is under test: inputs include all 256 raw FP8
encodings, including both NaNs and signed zeros. M>1 production dispatch remains
the existing vectorized path and is covered separately by CPU dispatch tests.
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

from tools import validate_vq2a8_tail_reorder as common
from tools.diagnose_vq2a8_tp1_startup import emit, parse_snapshot, run_child, stage_recorder

CASE = "v4_v2_reorder_chunk_reuse"
REPO = Path(__file__).resolve().parents[1]
LIBRARY_NAME = common.LIBRARY_NAME
WIDTHS = common.WIDTHS
EXPERTS = common.EXPERTS
INVALID_IDS = common.INVALID_IDS
QUEUE_ITERATIONS = 513
QUEUE_TEMPLATES = 24
CONTRACT_CASES = 32
GRAPH_CASES = ("finite", "changed", "invalid", "recovered", "raw_bytes", "recovered_raw")
DISPATCH_SCOPE = "m1_only_m_gt1_vectorized"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc-v4-v2-chunk-reuse" / LIBRARY_NAME)
    parser.add_argument("--chunks", type=int, choices=(2, 4), default=2)
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--queue-lifetime", action="store_true")
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
        "--chunks",
        str(args.chunks),
        "--timeout-s",
        str(args.timeout_s),
    ] + (["--queue-lifetime"] if args.queue_lifetime else [])


def require_abi(native):
    try:
        version = native.activation_reorder_chunk_reuse_version()
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError("Chunk-reuse reorder ABI missing; no fallback") from error
    if type(version) is not int or version != 1:
        raise RuntimeError(f"Chunk-reuse reorder requires independent ABI 1, got {version!r}")


def inputs_fixture(fixture, groups, pattern, *, offset=0, ids=None, rank3=False):
    import torch

    codes = torch.arange(256, dtype=torch.uint8)
    if pattern == "finite":
        codes = codes[(codes != 0x7F) & (codes != 0xFF)]
    elif pattern != "raw_bytes":
        raise ValueError("Unknown raw FP8 pattern")
    codes = codes.roll(offset % codes.numel())
    count = groups * fixture.k
    # Aligned nonzero storage offset with sentinels on both sides. Transfer
    # uint8, then reinterpret; no floating-point conversion may change NaNs.
    owner = torch.full((count + 64,), 0x5A, dtype=torch.uint8)
    owner[32:-32].copy_(codes.repeat((count + codes.numel() - 1) // codes.numel())[:count])
    owner = owner.to(fixture.device)
    q = owner[32:-32].reshape(groups, fixture.k).view(torch.float8_e4m3fn)
    scale = (((torch.arange(groups).float() + offset) % 5 + 1) / 64).to(fixture.device)
    bias = (((torch.arange(groups).float() + offset) % 7 - 3) / 8).to(fixture.device)
    literal_ids = list(ids) if ids is not None else [(row + offset) % EXPERTS for row in range(groups)]
    ids_tensor = torch.tensor(literal_ids, dtype=torch.int64, device=fixture.device)
    if rank3:
        q, scale, bias = q[:, None, :], scale[:, None], bias[:, None]
    return (q, scale, bias, ids_tensor), literal_ids, owner


def baseline(fixture, inputs, literal_ids, *, projection):
    import torch

    q, scale, bias, ids = inputs
    prepared = fixture.bank.prepare_vectorized(q, scale, bias, ids)
    projected = fixture.bank.project_vectorized(q, scale, bias, ids) if projection else None
    q_bytes = q.view(torch.uint8).reshape(len(literal_ids), fixture.k).cpu()
    prepared_bytes = prepared[0].view(torch.uint8).reshape(len(literal_ids), fixture.k)
    expected = []
    for row, slot in enumerate(literal_ids):
        if 0 <= slot < EXPERTS:
            ordered = q_bytes[row].index_select(0, fixture.orders[slot])
            common.assert_bits(prepared_bytes[row].cpu(), ordered, "baseline_permutation")
            expected.append(ordered.clone())
        else:
            expected.append(None)
    wanted_valid = torch.tensor([int(0 <= slot < EXPERTS) for slot in literal_ids], dtype=torch.int32)
    common.assert_bits(prepared[1].cpu(), wanted_valid, "baseline_slots")
    if projected is not None:
        common.assert_bits(projected[1].cpu(), wanted_valid, "baseline_projection_slots")
        projected = tuple(tensor.cpu().clone() for tensor in projected)
    return expected, wanted_valid, projected


def check_prepared(fixture, prepared, oracle, descriptors, name, *, rank3=False):
    if not isinstance(prepared, (tuple, list)) or len(prepared) != 4:
        raise AssertionError(f"{name}: prepare_chunk_reuse must expose four outputs")
    if rank3:
        groups = len(oracle[0])
        if prepared[0].shape != (groups, 1, fixture.k) or prepared[3].shape != (groups, 1, common.OUTPUT_WIDTH):
            raise AssertionError(f"{name}: native rank3 shape mismatch")
        prepared = (prepared[0][:, 0], prepared[1], prepared[2], prepared[3][:, 0])
    common.check_prepared(fixture, prepared, oracle, descriptors, name)


def checked_case(fixture, inputs, literal_ids, owner, name, *, chunks, projection=True):
    snapshots = [common.bytes_cpu(tensor) for tensor in (*inputs, owner)]
    oracle = baseline(fixture, inputs, literal_ids, projection=projection)
    prepared = fixture.bank.prepare_chunk_reuse(*inputs, chunks)
    descriptors = common.descriptor_reference(fixture, inputs, literal_ids, prepared)
    projected = fixture.bank.project_candidate(*inputs, chunks, 0) if projection else None
    check_prepared(fixture, prepared, oracle, descriptors, name, rank3=inputs[0].ndim == 3)
    if projected is not None:
        common.check_projected(fixture, projected, oracle[2], name + "_projection")
    if projection:
        common.check_projected(
            fixture, fixture.bank.project_chunk_reuse(*inputs, chunks), oracle[2], name + "_direct_projection"
        )
    for tensor, snapshot in zip((*inputs, owner), snapshots):
        common.assert_bits(common.bytes_cpu(tensor), snapshot, name + "_input_unchanged")


def numeric_names():
    return [
        f"k{k}_g{g}_{pattern}_{routes}_rank{rank}"
        for k in WIDTHS
        for g in range(1, 7)
        for pattern in ("finite", "raw_bytes")
        for routes in ("sequential", "duplicate")
        for rank in (2, 3)
    ]


def run_numeric_checks(fixtures, stage, chunks):
    names = []
    for k, fixture in fixtures.items():
        for groups in range(1, 7):
            for pattern in ("finite", "raw_bytes"):
                for routes in ("sequential", "duplicate"):
                    for rank in (2, 3):
                        name = f"k{k}_g{groups}_{pattern}_{routes}_rank{rank}"
                        with stage(name):
                            inputs, ids, owner = inputs_fixture(
                                fixture,
                                groups,
                                pattern,
                                offset=groups,
                                ids=[2] * groups if routes == "duplicate" else None,
                                rank3=rank == 3,
                            )
                            checked_case(
                                fixture, inputs, ids, owner, name, chunks=chunks, projection=pattern == "finite"
                            )
                        names.append(name)
    return names


def invalid_names():
    return [
        f"k{k}_invalid{i}_{state}"
        for k in WIDTHS
        for i in range(len(INVALID_IDS))
        for state in ("invalid", "recovered")
    ]


def run_invalid_checks(fixtures, stage, chunks):
    names = []
    for k, fixture in fixtures.items():
        for index, invalid in enumerate(INVALID_IDS):
            inputs, ids, owner = inputs_fixture(fixture, 6, "finite", ids=[0, 1, 2, 0, 1, invalid])
            for state in ("invalid", "recovered"):
                name = f"k{k}_invalid{index}_{state}"
                with stage(name):
                    if state == "recovered":
                        ids[-1] = 2
                        inputs[-1][-1] = 2
                    checked_case(fixture, inputs, ids, owner, name, chunks=chunks)
                names.append(name)
    return names


def run_contract_checks(fixture, stage, chunks):
    import torch

    (x, scale, bias, ids), _, _ = inputs_fixture(fixture, 6, "finite")
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
        for method in (fixture.bank.prepare_chunk_reuse, fixture.bank.project_chunk_reuse):
            for values in bad_inputs:
                try:
                    method(*values, chunks)
                except RuntimeError:
                    count += 1
                else:
                    raise AssertionError("Chunk-reuse native binding accepted invalid metadata")
            for rejected_chunks in (-1, 0, 1, 3, 8):
                try:
                    method(x, scale, bias, ids, rejected_chunks)
                except RuntimeError:
                    count += 1
                else:
                    raise AssertionError("Chunk-reuse accepted unsupported chunk count")
            with torch.npu.stream(torch.npu.Stream()):
                try:
                    method(x, scale, bias, ids, chunks)
                except RuntimeError:
                    count += 1
                else:
                    raise AssertionError("Chunk-reuse native binding accepted a different bank stream")
    if count != CONTRACT_CASES:
        raise AssertionError("Chunk-reuse native contract coverage incomplete")
    return count


def graph_names():
    return [f"graph_k{k}_g{g}_{case}" for k in WIDTHS for g in (1, 6) for case in GRAPH_CASES]


def run_graph_checks(device, factory, stage, chunks):
    import torch

    names = []
    for k in WIDTHS:
        for groups in (1, 6):
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                with stage(f"graph_k{k}_g{groups}_prepare"):
                    fixture = common.make_fixture(k, device, factory)
                    static, _, owner = inputs_fixture(fixture, groups, "finite")
                    for _ in range(2):
                        fixture.bank.prepare_chunk_reuse(*static, chunks)
                        fixture.bank.project_candidate(*static, chunks, 0)
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, stream=stream):
                        prepared = fixture.bank.prepare_chunk_reuse(*static, chunks)
                        projected = fixture.bank.project_candidate(*static, chunks, 0)
                for index, case in enumerate(GRAPH_CASES):
                    name = f"graph_k{k}_g{groups}_{case}"
                    with stage(name):
                        ids = [(row + index) % EXPERTS for row in range(groups)]
                        if case == "invalid":
                            ids[-1] = (1 << 63) - 1
                        pattern = "raw_bytes" if case == "raw_bytes" else "finite"
                        current, _, _ = inputs_fixture(fixture, groups, pattern, offset=index * 17, ids=ids)
                        # Preserve FP8 NaN bytes across static-input updates.
                        for dst, src in zip(static, current):
                            dst.view(torch.uint8).copy_(src.view(torch.uint8))
                        snapshots = [common.bytes_cpu(tensor) for tensor in (*static, owner)]
                        oracle = baseline(fixture, static, ids, projection=case != "raw_bytes")
                        descriptors = common.descriptor_reference(fixture, static, ids, prepared)
                        graph.replay()
                        check_prepared(fixture, prepared, oracle, descriptors, name)
                        if oracle[2] is not None:
                            common.check_projected(fixture, projected, oracle[2], name + "_projection")
                        for tensor, snapshot in zip((*static, owner), snapshots):
                            common.assert_bits(common.bytes_cpu(tensor), snapshot, name + "_input_unchanged")
                    names.append(name)
                with stage(f"graph_k{k}_g{groups}_reset"):
                    stream.synchronize()
                    graph.reset()
    return names


def queue_evidence():
    return {
        **common.queue_evidence(),
        "iterations": QUEUE_ITERATIONS,
        "input_templates": QUEUE_TEMPLATES,
        "raw_fp8_nan_and_signed_zero_bytes": True,
        "projection_checked_on_finite_inputs": True,
        "oracle": "same_device_vectorized_plus_cpu_byte_permutation",
    }


def run_queue_checks(device, factory, stage, chunks):
    import torch

    templates = []
    with stage("queue_preupload_and_oracles"):
        queue_fixtures = {k: common.make_fixture(k, device, factory) for k in WIDTHS}
        for index in range(QUEUE_TEMPLATES):
            fixture = queue_fixtures[WIDTHS[index % len(WIDTHS)]]
            groups = index // 4 % 6 + 1
            ids = [(row + index) % EXPERTS for row in range(groups)]
            if index >= QUEUE_TEMPLATES // 2:
                ids[-1] = INVALID_IDS[index % len(INVALID_IDS)]
            pattern = "finite" if index // 2 % 2 == 0 else "raw_bytes"
            inputs, ids, owner = inputs_fixture(fixture, groups, pattern, offset=index, ids=ids)
            oracle = baseline(fixture, inputs, ids, projection=pattern == "finite")
            templates.append((fixture, inputs, ids, oracle, [common.bytes_cpu(tensor) for tensor in inputs]))
        del inputs, owner
        torch.npu.synchronize()
    pending = []
    with stage("queue_owner_release_and_allocation_pressure"):
        for iteration in range(QUEUE_ITERATIONS):
            fixture, template, ids, oracle, snapshots = templates[iteration % QUEUE_TEMPLATES]
            inputs = tuple(tensor.view(torch.uint8).clone().view(tensor.dtype) for tensor in template)
            prepared = fixture.bank.prepare_chunk_reuse(*inputs, chunks)
            descriptors = common.descriptor_reference(fixture, inputs, ids, prepared)
            projected = fixture.bank.project_candidate(*inputs, chunks, 0) if oracle[2] is not None else None
            ordinary = [tensor.reshape(-1).view(torch.uint8).clone() for tensor in inputs]
            geometry = SimpleNamespace(k=fixture.k, device=fixture.device)
            pending.append((geometry, prepared, projected, oracle, descriptors, ordinary, snapshots))
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
        for index, (geometry, prepared, projected, oracle, descriptors, ordinary, snapshots) in enumerate(pending):
            check_prepared(geometry, prepared, oracle, descriptors, f"queue{index}")
            if projected is not None:
                common.check_projected(geometry, projected, oracle[2], f"queue{index}_projection")
            for got, want in zip(ordinary, snapshots):
                common.assert_bits(got.cpu(), want, f"queue{index}_input")
    return queue_evidence()


def validate_child_evidence(result, *, queue_lifetime, chunks):
    events = result.get("events", [])
    final = next((event for event in reversed(events) if event.get("event") == "CASE_PASS"), {})
    expected = {
        "numeric": numeric_names(),
        "invalid": invalid_names(),
        "native_contract": CONTRACT_CASES,
        "graph": graph_names(),
    }
    if queue_lifetime:
        expected["queue_lifetime"] = {**queue_evidence(), "task_queue_enable": "1"}
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
        or type(final.get("chunks")) is not int
        or final.get("chunks") != chunks
        or chunks not in (2, 4)
        or final.get("dispatch_scope") != DISPATCH_SCOPE
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
        raise ValueError("Incomplete chunk-reuse child evidence")


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
            require_abi(torch.ops.vq2a8_ascendc_v4_v2)
            factory = torch.classes.vq2a8_ascendc_v4_v2.ResidentBank
            emit(
                CASE,
                "INFO",
                library=identity,
                device=info,
                native_abi=1,
                chunks=args.chunks,
                dispatch_scope=DISPATCH_SCOPE,
            )
        with torch.inference_mode():
            with stage("synthetic_nonzero_banks"):
                fixtures = {k: common.make_fixture(k, device, factory) for k in WIDTHS}
            results = {
                "numeric": run_numeric_checks(fixtures, stage, args.chunks),
                "invalid": run_invalid_checks(fixtures, stage, args.chunks),
                "native_contract": run_contract_checks(fixtures[2048], stage, args.chunks),
                "graph": run_graph_checks(device, factory, stage, args.chunks),
            }
            if args.queue_lifetime:
                results["queue_lifetime"] = {
                    **run_queue_checks(device, factory, stage, args.chunks),
                    "task_queue_enable": queue_mode,
                }
        with stage("final_sync"):
            pass
        emit(
            CASE,
            "CASE_PASS",
            results=results,
            native_abi=1,
            chunks=args.chunks,
            dispatch_scope=DISPATCH_SCOPE,
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
        "scope": "raw_fp8_byte_reorder_only",
        "dispatch_scope": DISPATCH_SCOPE,
        "command": child_command(args),
        "status": "PLANNED",
        "native_abi": 1,
        "chunks": args.chunks,
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
        raise RuntimeError("Chunk-reuse validation requires Linux + NPU; use --plan-only elsewhere")
    directory = args.report_dir or Path(tempfile.mkdtemp(prefix="vq2-chunk-reuse-"))
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
                validate_child_evidence(result, queue_lifetime=args.queue_lifetime, chunks=args.chunks)
                report.update(device_execution_verified=True, graph_verified=True)
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        traceback.print_exc()
    finally:
        summary = directory / "summary.json"
        summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"V4_REORDER_CHUNK_REUSE={report['status']} SUMMARY={summary}", flush=True)
    return 0 if report["status"] == "PASS" else 130 if report["status"] == "INTERRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
