#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, staged V4 + v2 projection validation on one explicitly selected NPU.

Default: standalone compute before resident integration. Later phases include
their kernel/resident prerequisites. No model loading, serving or performance
claim; --model additionally reads just one real expert's gate/up and down.
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from tools.diagnose_vq2a8_tp1_startup import child_environment, emit, parse_snapshot, run_child, stage_recorder

REPO = Path(__file__).resolve().parents[1]
CASE = "v4_v2"
LIBRARY_NAME = "libvq2a8_ascendc_v4_v2.so"
FIELDS = ("packed_zn", "pair_lut", "activation_order", "weight_scale", "weight_bias", "rht_sign")
REDUCTIONS = (2048, 4096)
ROWS = (1, 2, 15, 16, 17, 31, 32)
EXPERTS = 3
N = 4096
QUEUE_CAPACITY_REFERENCE = 4096
QUEUE_MIN_ITERATIONS = 1366  # At least select + project + ordinary op, not measured queue slots.
QUEUE_DEFAULT_ITERATIONS = 2049
QUEUE_MAX_ITERATIONS = 8192
BANK_RELEASE_ITERATIONS = 3
PRESSURE_CHUNK_BYTES = 2 * 1024 * 1024
PRESSURE_CHUNKS = 16


def phases_for(phase):
    if phase not in ("kernel", "resident", "lifetime", "graph", "all"):
        raise ValueError("Unknown validation phase")
    return (
        ["kernel"]
        + (["resident"] if phase != "kernel" else [])
        + (["lifetime", "graph"] if phase == "all" else [phase] if phase in ("lifetime", "graph") else [])
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc-v4-v2" / LIBRARY_NAME)
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument("--phase", choices=("kernel", "resident", "lifetime", "graph", "all"), default="kernel")
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--queue-iterations", type=int, default=QUEUE_DEFAULT_ITERATIONS)
    parser.add_argument("--model", type=Path, help="optional real artifact root; never starts a model")
    parser.add_argument("--expert", default="0:0", help="one real artifact layer:expert")
    parser.add_argument("--allow-busy", action="store_true", help="explicitly allow known NPU occupancy")
    parser.add_argument("--report-dir", type=Path, help="new directory; existing results are never overwritten")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.physical_npu < 0 or args.timeout_s <= 0:
        parser.error("Require --physical-npu >= 0 and --timeout-s > 0")
    if not QUEUE_MIN_ITERATIONS <= args.queue_iterations <= QUEUE_MAX_ITERATIONS:
        parser.error(f"--queue-iterations must be {QUEUE_MIN_ITERATIONS}..{QUEUE_MAX_ITERATIONS}")
    if args.phase not in ("lifetime", "all") and args.queue_iterations != QUEUE_DEFAULT_ITERATIONS:
        parser.error("--queue-iterations requires --phase lifetime or all")
    pieces = args.expert.split(":")
    if len(pieces) != 2 or not all(value.isdecimal() for value in pieces):
        parser.error("--expert requires nonnegative layer:expert")
    if args.child and args.plan_only:
        parser.error("--child and --plan-only cannot be combined")
    args.launch_blocking = "0"  # Exercise normal asynchronous execution, not a synchronous workaround.
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
        "--phase",
        args.phase,
        "--timeout-s",
        str(args.timeout_s),
        "--queue-iterations",
        str(args.queue_iterations),
        "--expert",
        args.expert,
    ]
    if args.model is not None:
        command.extend(("--model", str(args.model.resolve())))
    return command


def validation_plan(args):
    return {
        "scope": "synthetic_and_optional_real_projection_only",
        "phase": args.phase,
        "phases": phases_for(args.phase),
        "command": child_command(args),
        "physical_npu": args.physical_npu,
        "geometry": {"n": N, "k": list(REDUCTIONS), "rows": list(ROWS), "jobs": [1, 6], "experts": EXPERTS},
        "queue_lifetime": {
            "enabled": args.phase in ("lifetime", "all"),
            "iterations": args.queue_iterations,
            "queue_capacity_reference": QUEUE_CAPACITY_REFERENCE,
            "runtime_queue_slots_measured": False,
            "requires_runtime_task_queue_enabled": True,
            "explicit_per_iteration_synchronize": False,
            "native_stream_check_may_drain_host_queue": True,
            "bank_release_iterations": BANK_RELEASE_ITERATIONS,
        },
        "real_projection_requested": args.model is not None,
        "full_model_loaded": False,
        "model_integration_verified": False,
        "serving_verified": False,
        "performance_verified": False,
        "device_execution_verified": False,
        "graph_verified": False,
    }


def synthetic_fixture(k, expert):
    """CPU integer/binary-scale oracle, arbitrary paired LUTs and shuffled K IDs."""
    import torch

    from vllm_ascend.quantization.vq2a8_v4_v2 import convert_expert_payload

    columns = torch.arange(k)
    pair_rows = torch.arange(N // 2)[:, None]
    codes = ((columns[None, :] * 7 + pair_rows * 3 + expert * 5) % 16).int()
    packed = (codes.reshape(N // 2, k // 8, 8).long() << (torch.arange(8) * 4)).sum(-1).int()
    books = (
        (torch.arange((k // 256) * (N // 32) * 32).reshape(k // 256, N // 32, 16, 2) * (expert * 2 + 3) + expert) % 31
        - 15
    ).to(torch.float8_e4m3fn)
    ids = ((columns * 5 + expert) % (k // 256)).byte()
    payload = {
        "packed_indices": packed,
        "codebooks": books,
        "codebook_tile_ids": ids,
        "weight_scale": (columns.float() % 7 + expert + 1) / 16,
        "weight_bias": (columns.float() % 13 - 6 + expert) / 128,
        "rht_sign": torch.where((columns + expert) % 3 == 0, -1, 1).to(torch.int8),
    }
    # Independent direct codebook indexing in ORIGINAL K order, not a decode of
    # the candidate's converted layout. Integer FP8 values make FP32 dots exact.
    dense = torch.empty(N, k, dtype=torch.float32)
    book_float = books.float()
    tile = ids.long()[None, :]
    block = (torch.arange(N // 2) // 16)[:, None]
    dense[0::2] = book_float[tile, block, codes.long(), 0]
    dense[1::2] = book_float[tile, block, codes.long(), 1]
    converted = convert_expert_payload(payload, SimpleNamespace(rows=N, columns=k, rht_true_columns=k))
    return {"converted": converted, "dense": dense, "expert": expert, "k": k}


def prepared_inputs(k, rows, offset):
    import torch

    q = ((torch.arange(rows * k).reshape(rows, k) * 7 + offset) % 5 - 2).to(torch.float8_e4m3fn)
    scale = ((torch.arange(rows) + offset) % 5 - 2).float() / 8
    bias = ((torch.arange(rows) + offset) % 7 - 3).float() / 2
    return q, scale, bias


def projection_oracle(fixture, prepared):
    q, scale, bias = prepared
    return (q.float() @ fixture["dense"].T * scale[:, None] + bias[:, None]).bfloat16()


def check_output(actual, expected, label, *, exact=True):
    import torch

    got = actual.detach().cpu()
    if got.shape != expected.shape or got.dtype != expected.dtype:
        raise AssertionError(f"{label}: shape/dtype mismatch")
    if not torch.isfinite(got.float()).all():
        raise AssertionError(f"{label}: non-finite valid output")
    if exact:
        passed = torch.equal(got.view(torch.uint8), expected.contiguous().view(torch.uint8))
    else:
        passed = torch.allclose(got.float(), expected.float(), rtol=0.01, atol=0.001)
    if not passed:
        error = (got.float() - expected.float()).abs().max().item()
        raise AssertionError(f"{label}: oracle mismatch, max_abs_error={error}")


def run_kernel_checks(device, grouped, stage, fixtures):
    import torch

    completed = []
    for k, experts in fixtures.items():
        for jobs in (1, 6):
            for rows in ROWS:
                name = f"kernel_k{k}_g{jobs}_m{rows}"
                inputs, expected = [], []
                with stage(name + "_setup"):
                    for job in range(jobs):
                        fixture = experts[job % EXPERTS]
                        q, scale, bias = prepared_inputs(k, rows, job + rows)
                        expected.append(projection_oracle(fixture, (q, scale, bias)))
                        payload = fixture["converted"]
                        ordered = q.view(torch.uint8).index_select(1, payload["activation_order"]).contiguous()
                        inputs.append(
                            (
                                ordered.view(torch.float8_e4m3fn).to(device),
                                scale.to(device),
                                bias.to(device),
                                payload["packed_zn"].to(device),
                                payload["pair_lut"].to(device),
                            )
                        )
                with stage(name):
                    outputs = grouped(*(list(values) for values in zip(*inputs)))
                if len(outputs) != jobs:
                    raise AssertionError("Standalone kernel returned a different job count")
                for output, oracle in zip(outputs, expected):
                    check_output(output, oracle, name)
                completed.append(name)
    return completed


def make_bank(experts, device, factory):
    payloads = [{name: value.to(device) for name, value in fixture["converted"].items()} for fixture in experts]
    return factory(*([payload[name] for payload in payloads] for name in FIELDS))


def routed_inputs(experts, routes, rows, iteration, device):
    import torch

    prepared = [prepared_inputs(experts[0]["k"], rows, iteration + index) for index in range(len(routes))]
    q, scale, bias = (torch.stack([values[index] for values in prepared]).to(device) for index in range(3))
    expected = [
        projection_oracle(experts[expert], values) if 0 <= expert < len(experts) else None
        for expert, values in zip(routes, prepared)
    ]
    if rows == 1:
        q, scale, bias = q[:, 0].contiguous(), scale[:, 0].contiguous(), bias[:, 0].contiguous()
        expected = [value[0] if value is not None else None for value in expected]
    return (q, scale, bias, torch.tensor(routes, dtype=torch.int64, device=device)), expected


def check_routed(outputs, expected, label):
    import torch

    output, valid = outputs
    flags = valid.detach().cpu().tolist()
    if len(flags) != len(expected) or output.shape[0] != len(expected):
        raise AssertionError(f"{label}: route count mismatch")
    for index, oracle in enumerate(expected):
        if bool(flags[index]) != (oracle is not None):
            raise AssertionError(f"{label}: invalid route mask")
        if oracle is None:
            if not torch.isnan(output[index].detach().cpu().float()).all():
                raise AssertionError(f"{label}: invalid route must produce NaNs, not stale values")
        else:
            check_output(output[index], oracle, label)


def run_resident_checks(device, factory, stage, fixtures):
    import torch

    banks, completed = {}, []
    routes = ((0,), (2, 1, 0, 2, 1, 0), (2, 2, 2, 2, 2, 2), (0, -1, 3, 2**32, -(2**63), 2**63 - 1))
    for k, experts in fixtures.items():
        with stage(f"resident_k{k}_bank_upload"):
            bank = make_bank(experts, device, factory)
        if list(bank.metadata())[:3] != [EXPERTS, N, k]:
            raise AssertionError("Resident metadata geometry mismatch")
        banks[k] = bank
        for iteration, selected in enumerate(routes):
            for rows in ROWS:
                name = f"resident_k{k}_ids{iteration}_m{rows}"
                inputs, expected = routed_inputs(experts, selected, rows, iteration, device)
                with stage(name):
                    metadata = bank.select(inputs[-1])
                    outputs = bank.project(*inputs)
                check_routed(outputs, expected, name)
                metadata_valid = metadata[3].detach().cpu().tolist()
                for index, expert in enumerate(selected):
                    if bool(metadata_valid[index]) != (0 <= expert < EXPERTS):
                        raise AssertionError("Resident select validity mismatch")
                    if 0 <= expert < EXPERTS:
                        for field, result in zip(FIELDS[3:], metadata[:3]):
                            wanted = experts[expert]["converted"][field]
                            if not torch.equal(result[index].detach().cpu(), wanted):
                                raise AssertionError(f"Resident select {field} mismatch")
                completed.append(name)
    return banks, completed


def run_lifetime_checks(device, bank, experts, stage, iterations):
    """Exercise queue wrap and off-thread final decrefs, without per-loop fences."""
    import torch

    with stage("lifetime_setup"):
        templates = [routed_inputs(experts, (expert,), 1, expert, device) for expert in range(EXPERTS)]
        expected_device = [expected[0].to(device) for _, expected in templates]
        all_valid = torch.ones((), dtype=torch.bool, device=device)
        left = torch.arange(16, device=device, dtype=torch.float32).reshape(1, 16)
        right = torch.eye(16, device=device, dtype=torch.float32)
    for phase in ("lifetime_wrap", "lifetime_concurrent_release"):
        # Only final reference destruction occurs in the extra Python thread;
        # all native calls remain on the bank's construction stream.
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = None
            with stage(phase):
                for iteration in range(iterations):
                    expert = iteration % EXPERTS
                    inputs = tuple(value.clone() for value in templates[expert][0])
                    metadata = bank.select(inputs[-1])
                    output, valid = bank.project(*inputs)
                    correct = (output[0].view(torch.uint8) == expected_device[expert].view(torch.uint8)).all()
                    all_valid = all_valid & correct & valid.bool().all() & metadata[3].bool().all()
                    ordinary = left @ right
                    all_valid = all_valid & (ordinary == left).all()
                    if phase == "lifetime_concurrent_release":
                        if pending is not None:
                            pending.result()  # CPU ownership fence, not an NPU synchronization.
                        owners = [inputs, metadata, output, valid, correct, ordinary]
                        del inputs, metadata, output, valid, correct, ordinary
                        pending = executor.submit(list.clear, owners)
                        del owners
                    else:
                        del inputs, metadata, output, valid, correct, ordinary
                    if iteration < 8 or (iteration + 1) % 256 == 0:
                        emit(CASE, "PROGRESS", stage=phase, iterations=iteration + 1)
                if pending is not None:
                    pending.result()
        if not bool(all_valid.cpu()):
            raise AssertionError(f"{phase}: asynchronous output/validity regression")
    return {
        "iterations_per_phase": iterations,
        "minimum_native_or_ordinary_calls_per_phase": iterations * 3,
        "runtime_queue_slots_measured": False,
        "requires_runtime_task_queue_enabled": True,
        "explicit_per_iteration_synchronize": False,
        "full_model_verified": False,
    }


def run_bank_release_checks(device, factory, fixture, stage):
    """Drop the bank and every uploaded/input Python owner before a device fence.

    This is deliberately separate from the long queue loops: three temporary
    banks only. Native queued state capture/stream lifetime tracking must keep
    resources alive, rather than the probe retaining a global bank/payload list.
    """
    import torch

    for iteration in range(BANK_RELEASE_ITERATIONS):
        name = f"lifetime_bank_release_{iteration}"
        with stage(name + "_setup"):
            uploaded = {name: value.to(device, copy=True) for name, value in fixture["converted"].items()}
            bank = factory(*([uploaded[name]] for name in FIELDS))
            inputs, expected = routed_inputs([fixture], (0,), 1, iteration + 11, device)
            left = torch.arange(16, device=device, dtype=torch.float32).reshape(1, 16)
            right = torch.eye(16, device=device, dtype=torch.float32)
        with stage(name):
            output, valid = bank.project(*inputs)
            del bank, inputs, uploaded
            # Repeated allocation AND writes make premature bank/input reuse
            # observable; the probe holds no dense expert device weights.
            for index in range(PRESSURE_CHUNKS):
                pressure = torch.empty(PRESSURE_CHUNK_BYTES, device=device, dtype=torch.uint8)
                pressure.fill_((iteration * PRESSURE_CHUNKS + index) % 251 + 1)
                del pressure
            ordinary = left @ right
        check_routed((output, valid), expected, name)
        if not torch.equal(ordinary.cpu(), left.cpu()):
            raise AssertionError(f"{name}: ordinary operation was corrupted")
        del output, valid, ordinary, left, right
    return {
        "iterations": BANK_RELEASE_ITERATIONS,
        "python_bank_and_input_owners_dropped_before_fence": True,
        "uploaded_python_payload_owners_dropped_before_fence": True,
        "allocation_pressure_bytes_per_iteration": PRESSURE_CHUNKS * PRESSURE_CHUNK_BYTES,
        "per_pressure_allocation_bytes": PRESSURE_CHUNK_BYTES,
        "allocator_peak_memory_measured": False,
        "projection_and_ordinary_outputs_checked": True,
    }


def run_graph_checks(device, factory, fixtures, stage):
    import torch

    completed = []
    for k in REDUCTIONS:
        experts = fixtures[k]
        for jobs in (1, 6):
            name = f"graph_k{k}_g{jobs}_m1"
            caller = torch.npu.current_stream()
            owner = torch.npu.Stream()
            # Native calls remain on the bank's construction stream. Capture
            # there, then exercise the V4 current-caller-stream replay contract.
            with stage(name + "_warmup"), torch.npu.stream(owner):
                bank = make_bank(experts, device, factory)
                static, _ = routed_inputs(experts, (0,) * jobs, 1, 0, device)
                for _ in range(3):
                    bank.select(static[-1])
                    bank.project(*static)
            graph = torch.npu.NPUGraph()
            with stage(name + "_capture"), torch.npu.graph(graph, stream=owner):
                captured_metadata = bank.select(static[-1])
                captured = bank.project(*static)
            try:
                for iteration in range(7):
                    routes = tuple((iteration + index) % EXPERTS for index in range(jobs))
                    if iteration == 4:
                        routes = (-1,) * jobs
                    elif iteration == 5:
                        routes = (2**32,) * jobs
                    current, expected = routed_inputs(experts, routes, 1, iteration + 11, device)
                    with stage(name + f"_replay{iteration}"):
                        if torch.npu.current_stream().npu_stream != caller.npu_stream:
                            raise RuntimeError("Graph replay left its bound caller stream")
                        for dst, src in zip(static, current):
                            dst.copy_(src)
                        graph.replay()
                    check_routed(captured, expected, name)
                    with stage(name + f"_eager{iteration}"), torch.npu.stream(owner):
                        owner.wait_stream(caller)
                        eager_metadata = bank.select(current[-1])
                        eager = bank.project(*current)
                    check_routed(eager, expected, name)
                    if not torch.equal(captured_metadata[3].cpu(), eager_metadata[3].cpu()):
                        raise AssertionError("Captured route metadata used stale IDs")
                    for index, oracle in enumerate(expected):
                        if oracle is not None:
                            for captured_value, eager_value in zip(captured_metadata[:3], eager_metadata[:3]):
                                if not torch.equal(captured_value[index].cpu(), eager_value[index].cpu()):
                                    raise AssertionError("Captured preparation metadata used stale IDs")
                            check_output(captured[0][index], eager[0][index].cpu(), name + "_eager")
                completed.append(name)
            finally:
                # A failure in synchronization is fatal to this isolated child;
                # the parent deadline still bounds a hung reset/destructor.
                with stage(name + "_reset"):
                    caller.synchronize()
                    owner.synchronize()
                    graph.reset()
    return completed


def prepare_real_rows(hidden, payload, spec):
    """CPU reference explicitly prepares independent rows, matching V4 arithmetic."""
    import torch

    from vllm_ascend.quantization.vq2a8_reference import prepare_repacked_vq2a8_activation_reference

    prepared = [
        prepare_repacked_vq2a8_activation_reference(
            row,
            payload["weight_scale"],
            payload["weight_bias"],
            payload["rht_sign"],
            spec.rht_block_size,
        )
        for row in hidden.split(1)
    ]
    return tuple(torch.cat([values[index] for values in prepared]).contiguous() for index in range(3))


def run_real_checks(args, device, grouped, factory, stage):
    import torch

    from tools.validate_vq2a8_tp1_packed_kernel import activation_case
    from vllm_ascend.quantization.vq2a8_reference import decode_repacked_vq2a8_codebook_weight
    from vllm_ascend.quantization.vq2a8_runtime import open_vq2a8_tp1_artifact
    from vllm_ascend.quantization.vq2a8_v4_v2 import convert_expert_payload

    layer, expert = map(int, args.expert.split(":"))
    artifact = open_vq2a8_tp1_artifact(args.model / "experts_vq_ascend_v2", args.model / "config.json")
    completed = []
    for kind in ("gate_up", "down"):
        with stage(f"real_{kind}_load"):
            payload, spec = artifact.load_expert(layer, expert, kind)
            converted = convert_expert_payload(payload, spec)
            dense = decode_repacked_vq2a8_codebook_weight(payload, spec, compute_dtype=torch.float64)
            uploaded = {name: value.to(device) for name, value in converted.items()}
            bank = factory(*([uploaded[name]] for name in FIELDS)) if args.phase != "kernel" else None
        for case in ("deterministic", "zero", "impulse"):
            for rows in (1, 3, 32):
                name = f"real_{kind}_{case}_m{rows}"
                hidden = torch.cat([activation_case(spec.rht_true_columns, row, 0, case) for row in range(rows)])
                if spec.columns != spec.rht_true_columns:
                    hidden = torch.nn.functional.pad(hidden, (0, spec.columns - spec.rht_true_columns))
                prepared = prepare_real_rows(hidden, payload, spec)
                q, scale, bias = prepared
                expected = (q.double() @ dense.T * scale.double()[:, None] + bias.double()[:, None]).bfloat16()
                ordered = q.view(torch.uint8).index_select(1, converted["activation_order"]).contiguous()
                with stage(name):
                    output = grouped(
                        [ordered.view(torch.float8_e4m3fn).to(device)],
                        [scale.to(device)],
                        [bias.to(device)],
                        [uploaded["packed_zn"]],
                        [uploaded["pair_lut"]],
                    )[0]
                check_output(output, expected, name, exact=case == "zero")
                if bank is not None:
                    with stage(name + "_resident"):
                        actual, valid = bank.project(
                            q.to(device).unsqueeze(0),
                            scale.to(device).unsqueeze(0),
                            bias.to(device).unsqueeze(0),
                            torch.zeros(1, device=device, dtype=torch.int64),
                        )
                    check_output(actual[0], expected, name + "_resident", exact=case == "zero")
                    if not bool(valid.cpu().all()):
                        raise AssertionError("Valid real expert rejected")
                completed.append(name)
    return completed


def run_phases(args, device, grouped, factory, stage, fixtures):
    """Do not construct a bank after a failed standalone kernel prerequisite."""
    results = {"kernel": run_kernel_checks(device, grouped, stage, fixtures)}
    requested = phases_for(args.phase)
    if "resident" in requested:
        banks, results["resident"] = run_resident_checks(device, factory, stage, fixtures)
    if "lifetime" in requested:
        results["lifetime"] = run_lifetime_checks(device, banks[2048], fixtures[2048], stage, args.queue_iterations)
        results["lifetime"]["bank_release"] = run_bank_release_checks(device, factory, fixtures[2048][0], stage)
    if "graph" in requested:
        results["graph"] = run_graph_checks(device, factory, fixtures, stage)
    if args.model is not None:
        results["real_projection"] = run_real_checks(args, device, grouped, factory, stage)
    return results


def run_case_child(args):
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical_npu):
        raise ValueError("Child device mapping differs from requested physical NPU")
    faulthandler.enable()
    faulthandler.dump_traceback_later(min(30, max(1, args.timeout_s / 2)), repeat=True)
    synchronize = lambda: None
    stage = stage_recorder(CASE, lambda: synchronize())
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
            device_info = _initialize_device(device)
            torch.npu.config.allow_internal_format = False
            synchronize = torch.npu.synchronize
        with stage("library"):
            path = args.library.resolve(strict=True)
            if path.name != LIBRARY_NAME:
                raise ValueError(f"Choose {LIBRARY_NAME}; no baseline fallback")
            identity = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            torch.ops.load_library(str(path))
            native = torch.ops.vq2a8_ascendc_v4_v2
            if native.abi_version() != 1:
                raise ValueError("V4 + v2 ABI mismatch")
            factory = torch.classes.vq2a8_ascendc_v4_v2.ResidentBank
            emit(CASE, "INFO", library=identity, device=device_info)
        with torch.inference_mode():
            with stage("synthetic_setup"):
                fixtures = {k: [synthetic_fixture(k, expert) for expert in range(EXPERTS)] for k in REDUCTIONS}
            results = run_phases(args, device, native.grouped_projection, factory, stage, fixtures)
        with stage("final_sync"):
            pass
        emit(
            CASE,
            "CASE_PASS",
            results=results,
            library=identity,
            device_execution_verified=True,
            graph_verified="graph" in results,
            model_integration_verified=False,
            performance_verified=False,
            full_model_loaded=False,
            real_projection_verified="real_projection" in results,
        )
        return 0
    except Exception as exc:
        traceback.print_exc()
        emit(CASE, "CASE_FAIL", error=str(exc), device_execution_verified=False)
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()


def main(argv=None):
    args = parse_args(argv)
    if args.child:
        return run_case_child(args)
    plan = validation_plan(args)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    if os.name != "posix":
        raise RuntimeError("Physical NPU validation requires Linux; use --plan-only elsewhere")
    directory = args.report_dir or Path(tempfile.mkdtemp(prefix="vq2-v4-v2-"))
    if args.report_dir is not None:
        directory.mkdir(parents=True, exist_ok=False)
    report = {**plan, "status": "FAIL"}
    try:
        snapshot = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True, timeout=20)
        (directory / "npu.log").write_text(snapshot.stdout + snapshot.stderr, encoding="utf-8")
        state = parse_snapshot(snapshot.stdout, args.physical_npu)
        report["device_state"] = state
        if state == "unknown" or (state == "busy" and not args.allow_busy):
            report["status"] = "BLOCKED"
            print(f"Device state={state}; --allow-busy overrides known occupancy only", flush=True)
        else:
            result = run_child(
                child_command(args), child_environment(args), directory / "validation.log", args.timeout_s
            )
            report["result"], report["status"] = result, result["status"]
            if result["status"] == "PASS":
                final = next((event for event in reversed(result["events"]) if event.get("event") == "CASE_PASS"), {})
                expected_phases = set(plan["phases"]) | ({"real_projection"} if args.model is not None else set())
                completed = final.get("results", {})
                if (
                    final.get("case") != CASE
                    or final.get("device_execution_verified") is not True
                    or set(completed) != expected_phases
                    or not all(completed.values())
                ):
                    report.update(status="FAIL", error="Child returned incomplete projection evidence")
                else:
                    report["device_execution_verified"] = True
                    report["graph_verified"] = final.get("graph_verified") is True and "graph" in completed
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
    except Exception as exc:
        report["error"] = str(exc)
        traceback.print_exc()
    finally:
        (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"V4_V2_VALIDATION={report['status']} SUMMARY={directory / 'summary.json'}", flush=True)
    return 0 if report["status"] == "PASS" else 130 if report["status"] == "INTERRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
