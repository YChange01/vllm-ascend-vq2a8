#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact clamp/cast/resident-reorder acceptance; not model or performance evidence.

Never reads unspecified invalid reordered rows or uninitialized valid prepare
outputs. The scalar/vectorized baseline and CPU byte permutation are independent
oracles. All Torch/NPU imports are child-only or inside explicitly called helpers.
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

CASE = "v4_v2_tail_reorder"
REPO = Path(__file__).resolve().parents[1]
LIBRARY_NAME = "libvq2a8_ascendc_v4_v2.so"
WIDTHS = (2048, 4096)
EXPERTS = 3
OUTPUT_WIDTH = 4096
PATTERNS = ("finite", "rounding", "special", "random")
INVALID_IDS = (-1, EXPERTS, -(1 << 63), (1 << 63) - 1, 1 << 32)
GRAPH_CASES = ("finite", "changed", "invalid", "recovered", "special", "recovered_special")
QUEUE_ITERATIONS = 513
QUEUE_TEMPLATES = 24
PRESSURE_BYTES = 2 * 1024 * 1024
POST_RELEASE_PRESSURE_CHUNKS = 4
CONTRACT_CASES = 20


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc-v4-v2-tail" / LIBRARY_NAME)
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
    result = [
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
    return result + (["--queue-lifetime"] if args.queue_lifetime else [])


def probe_environment(args, environ=None):
    result = child_environment(args, environ)
    result.setdefault("TASK_QUEUE_ENABLE", "1")
    return result


def require_abi(native):
    try:
        version = native.activation_tail_reorder_version()
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError("Tail reorder ABI missing; no fallback") from error
    if type(version) is not int or version != 1:
        raise RuntimeError(f"Tail reorder requires independent ABI 1, got {version!r}")


def make_fixture(k, device, factory):
    from tools.validate_vq2a8_v4_v2 import FIELDS, make_bank, synthetic_fixture

    experts = [synthetic_fixture(k, expert) for expert in range(EXPERTS)]
    payloads = {}

    def owning_factory(*columns):
        payloads.update(zip(FIELDS, columns))
        return factory(*columns)

    bank = make_bank(experts, device, owning_factory)
    orders = [item["converted"]["activation_order"].clone() for item in experts]
    if any(order.equal(order.sort().values) for order in orders):
        raise AssertionError("Synthetic tail fixtures must exercise nonidentity permutations")
    return SimpleNamespace(bank=bank, payloads=payloads, orders=orders, k=k, device=device)


def normalized_pattern(k, groups, pattern, offset=0):
    import torch

    if pattern == "finite":
        values = torch.tensor(
            [-4096, -449, -448, -447, -1.5, -0.0, 0, 0.5, 1, 447, 448, 449, 4096], dtype=torch.float32
        )
    elif pattern == "rounding":
        # Every adjacent nonnegative finite E4M3 pair: midpoint and its FP32
        # neighbours, both signs, plus subnormal underflow/signed-zero edges.
        fp8 = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        midpoint = (fp8[:-1] + fp8[1:]) / 2
        values = torch.cat(
            (
                torch.nextafter(midpoint, torch.full_like(midpoint, -torch.inf)),
                midpoint,
                torch.nextafter(midpoint, torch.full_like(midpoint, torch.inf)),
            )
        )
        values = torch.cat((values, -values))
    elif pattern == "special":
        bits = [
            0,
            0x80000000,
            1,
            0x80000001,
            0x007FFFFF,
            0x807FFFFF,
            0x00800000,
            0x80800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
            0xFFC00000,
            0x7FC12345,
            0xFFC12345,
            0x7F800001,
            0xFF800001,
        ]
        values = torch.tensor(
            [value if value < (1 << 31) else value - (1 << 32) for value in bits], dtype=torch.int32
        ).view(torch.float32)
    elif pattern == "random":
        values = torch.randn(k, generator=torch.Generator().manual_seed(1701 + offset)) * 600
    else:
        raise ValueError("Unknown normalized input pattern")
    values = values.roll(offset % values.numel())
    return values.repeat((groups * k + values.numel() - 1) // values.numel())[: groups * k].reshape(groups, k).clone()


def inputs_fixture(fixture, groups, pattern, *, offset=0, ids=None):
    import torch

    k, device = fixture.k, fixture.device
    cpu = normalized_pattern(k, groups, pattern, offset)
    # Preserve raw NaN payloads and exercise an aligned nonzero storage offset.
    owner = torch.full((groups * k + 16,), 123.0, dtype=torch.float32)
    owner[8:-8].view(torch.int32).copy_(cpu.reshape(-1).view(torch.int32))
    owner = owner.to(device)
    normalized = owner[8:-8].reshape(groups, k)
    scale = ((torch.arange(groups).float() + offset) % 5 + 1).to(device) / 64
    bias = ((torch.arange(groups).float() + offset) % 7 - 3).to(device) / 8
    literal_ids = list(ids) if ids is not None else [(row + offset) % EXPERTS for row in range(groups)]
    ids_tensor = torch.tensor(literal_ids, dtype=torch.int64, device=device)
    return (normalized, scale, bias, ids_tensor), literal_ids, owner


def bytes_cpu(tensor):
    import torch

    return tensor.detach().reshape(-1).view(torch.uint8).cpu().clone()


def assert_bits(actual, expected, name):
    import torch

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: dtype/shape mismatch")
    if not torch.equal(bytes_cpu(actual), bytes_cpu(expected)):
        raise AssertionError(f"{name}: exact bytes differ; no relaxed tolerance")


def baseline(fixture, inputs, literal_ids, *, projection):
    import torch

    normalized, scale, bias, ids = inputs
    quantized = torch.clamp(normalized, -448.0, 448.0).to(torch.float8_e4m3fn)
    prepared = fixture.bank.prepare_vectorized(quantized, scale, bias, ids)
    projected = fixture.bank.project_vectorized(quantized, scale, bias, ids) if projection else None
    q_bytes = quantized.view(torch.uint8).cpu()
    expected = []
    for row, slot in enumerate(literal_ids):
        if 0 <= slot < EXPERTS:
            ordered = q_bytes[row].index_select(0, fixture.orders[slot])
            assert_bits(prepared[0][row].view(torch.uint8).cpu(), ordered, "baseline_permutation")
            expected.append(ordered.clone())
        else:
            expected.append(None)
    wanted_valid = torch.tensor([int(0 <= slot < EXPERTS) for slot in literal_ids], dtype=torch.int32)
    assert_bits(prepared[1].cpu(), wanted_valid, "baseline_slots")
    if projected is not None:
        assert_bits(projected[1].cpu(), wanted_valid, "baseline_projection_slots")
        projected = (projected[0].cpu().clone(), projected[1].cpu().clone())
    return expected, wanted_valid, projected


def descriptor_reference(fixture, inputs, literal_ids, prepared):
    reordered, _valid, _descriptors, output = prepared
    _normalized, scale, bias, _ids = inputs
    return [
        [
            reordered.data_ptr() + row * fixture.k,
            scale.data_ptr() + row * 4,
            bias.data_ptr() + row * 4,
            fixture.payloads["packed_zn"][slot].data_ptr(),
            fixture.payloads["pair_lut"][slot].data_ptr(),
            output.data_ptr() + row * OUTPUT_WIDTH * 2,
            1,
            OUTPUT_WIDTH,
            fixture.k,
        ]
        if 0 <= slot < EXPERTS
        else [0] * 9
        for row, slot in enumerate(literal_ids)
    ]


def check_prepared(fixture, prepared, oracle, descriptors, name):
    import torch

    if not isinstance(prepared, (tuple, list)) or len(prepared) != 4:
        raise AssertionError(f"{name}: prepare_tail must expose four outputs")
    reordered, valid, records, output = prepared
    expected, wanted_valid, _projection = oracle
    groups = len(expected)
    for tensor, dtype, shape in (
        (reordered, torch.float8_e4m3fn, (groups, fixture.k)),
        (valid, torch.int32, (groups,)),
        (records, torch.int64, (groups, 9)),
        (output, torch.bfloat16, (groups, OUTPUT_WIDTH)),
    ):
        if tensor.dtype != dtype or tensor.shape != shape or tensor.device != torch.device(fixture.device):
            raise AssertionError(f"{name}: native output contract mismatch")
    assert_bits(valid.cpu(), wanted_valid, name + "_slots")
    assert_bits(records.cpu(), torch.tensor(descriptors, dtype=torch.int64), name + "_descriptor")
    for row, expected_bytes in enumerate(expected):
        if expected_bytes is not None:
            assert_bits(reordered[row].view(torch.uint8).cpu(), expected_bytes, name + "_bytes")
            # output[row] is uninitialized on a valid prepare: never read it.
        else:
            # reordered[row] is unspecified for invalid routes: never read it.
            poison = torch.full((OUTPUT_WIDTH,), 0x7FC0, dtype=torch.int16)
            assert_bits(output[row].view(torch.int16).cpu(), poison, name + "_poison")


def checked_case(fixture, inputs, literal_ids, owner, name, *, projection=True):
    snapshots = [bytes_cpu(tensor) for tensor in (*inputs, owner)]
    oracle = baseline(fixture, inputs, literal_ids, projection=projection)
    prepared = fixture.bank.prepare_tail(*inputs)
    descriptors = descriptor_reference(fixture, inputs, literal_ids, prepared)
    actual_projection = fixture.bank.project_tail(*inputs) if projection else None
    check_prepared(fixture, prepared, oracle, descriptors, name)
    if actual_projection is not None:
        check_projected(fixture, actual_projection, oracle[2], name + "_projection")
    for tensor, snapshot in zip((*inputs, owner), snapshots):
        assert_bits(bytes_cpu(tensor), snapshot, name + "_input_unchanged")


def check_projected(fixture, projected, expected, name):
    import torch

    if not isinstance(projected, (tuple, list)) or len(projected) != 2:
        raise AssertionError(f"{name}: project_tail must return two outputs")
    for got, want in zip(projected, expected):
        if got.device != torch.device(fixture.device):
            raise AssertionError(f"{name}: projection output device mismatch")
        assert_bits(got.cpu(), want, name)


def numeric_names():
    return [
        f"k{k}_g{g}_{pattern}_{routes}"
        for k in WIDTHS
        for g in range(1, 7)
        for pattern in PATTERNS
        for routes in ("sequential", "duplicate")
    ]


def run_numeric_checks(fixtures, stage):
    names = []
    for k, fixture in fixtures.items():
        for groups in range(1, 7):
            for pattern in PATTERNS:
                for routes in ("sequential", "duplicate"):
                    name = f"k{k}_g{groups}_{pattern}_{routes}"
                    with stage(name):
                        inputs, ids, owner = inputs_fixture(
                            fixture, groups, pattern, offset=groups, ids=[2] * groups if routes == "duplicate" else None
                        )
                        checked_case(fixture, inputs, ids, owner, name, projection=pattern != "special")
                    names.append(name)
    return names


def invalid_names():
    return [
        f"k{k}_invalid{index}_{state}"
        for k in WIDTHS
        for index in range(len(INVALID_IDS))
        for state in ("invalid", "recovered")
    ]


def run_invalid_checks(fixtures, stage):
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
                    checked_case(fixture, inputs, ids, owner, name)
                names.append(name)
    return names


def run_contract_checks(fixture, stage):
    import torch

    inputs, _ids, _owner = inputs_fixture(fixture, 6, "finite")
    x, scale, bias, ids = inputs
    bad_inputs = [
        (x.bfloat16(), scale, bias, ids),
        (x.unsqueeze(1), scale[:, None], bias[:, None], ids),
        (x[:, ::2], scale, bias, ids),
        (x, scale.long(), bias, ids),
        (x, scale, bias[:, None], ids),
        (x, scale, bias, ids.int()),
        (x[:0], scale[:0], bias[:0], ids[:0]),
        (x, scale[:-1], bias, ids),
        (x, scale, bias, torch.zeros(7, dtype=torch.int64, device=x.device)),
        (torch.empty(x.numel() + 1, device=x.device)[1:].reshape(x.shape), scale, bias, ids),
    ]
    count = 0
    with stage("native_metadata_rejections"):
        for method in (fixture.bank.prepare_tail, fixture.bank.project_tail):
            for values in bad_inputs:
                try:
                    method(*values)
                except RuntimeError:
                    count += 1
                else:
                    raise AssertionError("Tail native binding accepted invalid metadata")
    if count != CONTRACT_CASES:
        raise AssertionError("Tail native contract coverage incomplete")
    return count


def graph_names():
    return [f"graph_k{k}_g{g}_{case}" for k in WIDTHS for g in (1, 6) for case in GRAPH_CASES]


def run_graph_checks(device, factory, stage):
    import torch

    names = []
    for k in WIDTHS:
        for groups in (1, 6):
            owner_stream = torch.npu.Stream()
            with torch.npu.stream(owner_stream):
                with stage(f"graph_k{k}_g{groups}_prepare"):
                    fixture = make_fixture(k, device, factory)
                    static, _, owner = inputs_fixture(fixture, groups, "finite")
                    for _ in range(2):
                        fixture.bank.prepare_tail(*static)
                        fixture.bank.project_tail(*static)
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    # Both paths are captured. Nonfinite rows are byte-checked
                    # but their projection output is not a finite-output claim.
                    with torch.npu.graph(graph, stream=owner_stream):
                        prepared = fixture.bank.prepare_tail(*static)
                        projected = fixture.bank.project_tail(*static)
                for index, case in enumerate(GRAPH_CASES):
                    name = f"graph_k{k}_g{groups}_{case}"
                    with stage(name):
                        pattern = "special" if case == "special" else "rounding" if case == "changed" else "finite"
                        ids = [(row + index) % EXPERTS for row in range(groups)]
                        if case == "invalid":
                            ids[-1] = (1 << 63) - 1
                        current, _, _ = inputs_fixture(fixture, groups, pattern, offset=index, ids=ids)
                        for dst, src in zip(static, current):
                            dst.copy_(src)
                        snapshots = [bytes_cpu(tensor) for tensor in (*static, owner)]
                        oracle = baseline(fixture, static, ids, projection=case != "special")
                        descriptors = descriptor_reference(fixture, static, ids, prepared)
                        graph.replay()
                        check_prepared(fixture, prepared, oracle, descriptors, name)
                        if oracle[2] is not None:
                            check_projected(fixture, projected, oracle[2], name + "_projection")
                        for tensor, snapshot in zip((*static, owner), snapshots):
                            assert_bits(bytes_cpu(tensor), snapshot, name + "_input_unchanged")
                    names.append(name)
                # Only success resets; a failed replay must reach the bounded
                # supervisor instead of hanging in graph.reset() cleanup.
                with stage(f"graph_k{k}_g{groups}_reset"):
                    owner_stream.synchronize()
                    graph.reset()
    return names


def queue_evidence():
    return {
        "iterations": QUEUE_ITERATIONS,
        "input_templates": QUEUE_TEMPLATES,
        "input_upload_before_loop": True,
        "independent_queue_banks": True,
        "bank_payload_upload_before_loop": True,
        "fresh_device_clones": True,
        "all_outputs_checked": True,
        "all_input_bytes_checked": True,
        "owners_dropped_before_fence": True,
        "input_owners_dropped_before_fence": True,
        "bank_owners_dropped_before_fence": True,
        "payload_python_owners_dropped_before_fence": True,
        "explicit_per_iteration_synchronize": False,
        "runtime_queue_slots_measured": False,
        "native_stream_check_may_drain_host_queue": True,
        "allocation_pressure_bytes_per_iteration": PRESSURE_BYTES,
        "allocation_pressure_bytes_after_bank_release": POST_RELEASE_PRESSURE_CHUNKS * PRESSURE_BYTES,
    }


def run_queue_checks(device, factory, stage):
    import torch

    templates = []
    with stage("queue_preupload_and_oracles"):
        # These banks belong only to this test. Numeric/graph fixture owners
        # must not accidentally keep the queued indirect payloads alive.
        queue_fixtures = {k: make_fixture(k, device, factory) for k in WIDTHS}
        for index in range(QUEUE_TEMPLATES):
            fixture = queue_fixtures[WIDTHS[index % 2]]
            groups = index // 2 % 6 + 1
            ids = [(row + index) % EXPERTS for row in range(groups)]
            if index >= QUEUE_TEMPLATES // 2:
                ids[-1] = INVALID_IDS[index % len(INVALID_IDS)]
            inputs, ids, owner = inputs_fixture(fixture, groups, "finite", offset=index, ids=ids)
            oracle = baseline(fixture, inputs, ids, projection=True)
            templates.append((fixture, inputs, ids, oracle, [bytes_cpu(tensor) for tensor in inputs]))
        del inputs, owner
        torch.npu.synchronize()
    pending = []
    with stage("queue_owner_release_and_allocation_pressure"):
        for iteration in range(QUEUE_ITERATIONS):
            fixture, template, ids, oracle, snapshots = templates[iteration % QUEUE_TEMPLATES]
            inputs = tuple(tensor.clone() for tensor in template)
            prepared = fixture.bank.prepare_tail(*inputs)
            descriptors = descriptor_reference(fixture, inputs, ids, prepared)
            projected = fixture.bank.project_tail(*inputs)
            # Ordinary device clones both stress lifetime and expose a native
            # write to any input; none retain the original Python input owners.
            ordinary = [tensor.reshape(-1).view(torch.uint8).clone() for tensor in inputs]
            # Verification needs only geometry/device, never bank or payload
            # owners. Descriptor references above contain addresses as ints.
            geometry = SimpleNamespace(k=fixture.k, device=fixture.device)
            pending.append((geometry, prepared, projected, oracle, descriptors, ordinary, snapshots))
            del inputs
            pressure = torch.empty(PRESSURE_BYTES, dtype=torch.uint8, device=fixture.device)
            pressure.fill_(iteration % 256)
            del pressure
        templates.clear()
        queue_fixtures.clear()
        del fixture, template
        # Stress allocations after the last Python bank/payload owner has
        # gone, while stream recording/native queue ownership must protect
        # indirect reads. No new bank or host upload occurs in this interval.
        post_release_pressure = []
        for chunk in range(POST_RELEASE_PRESSURE_CHUNKS):
            pressure = torch.empty(PRESSURE_BYTES, dtype=torch.uint8, device=device)
            pressure.fill_(chunk + 1)
            post_release_pressure.append(pressure)
        del pressure, post_release_pressure
        torch.npu.synchronize()
        for index, (fixture, prepared, projected, oracle, descriptors, ordinary, snapshots) in enumerate(pending):
            check_prepared(fixture, prepared, oracle, descriptors, f"queue{index}")
            check_projected(fixture, projected, oracle[2], f"queue{index}_projection")
            for got, want in zip(ordinary, snapshots):
                assert_bits(got.cpu(), want, f"queue{index}_input")
    return queue_evidence()


def validate_child_evidence(result, *, queue_lifetime):
    events = result.get("events", [])
    final = next((event for event in reversed(events) if event.get("event") == "CASE_PASS"), {})
    values, identity = final.get("results", {}), final.get("library", {})
    digest = identity.get("sha256")
    keys = {"numeric", "invalid", "native_contract", "graph"} | ({"queue_lifetime"} if queue_lifetime else set())
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
        or any(c not in "0123456789abcdef" for c in digest)
        or set(values) != keys
        or values.get("numeric") != numeric_names()
        or values.get("invalid") != invalid_names()
        or values.get("native_contract") != CONTRACT_CASES
        or values.get("graph") != graph_names()
        or not any(event.get("event") == "PASS" and event.get("stage") == "final_sync" for event in events)
    ):
        raise ValueError("Incomplete tail-reorder child evidence")
    if queue_lifetime and values["queue_lifetime"] != {**queue_evidence(), "task_queue_enable": "1"}:
        raise ValueError("Incomplete tail-reorder queue evidence")


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
            emit(CASE, "INFO", library=identity, device=info, native_abi=1)
        with torch.inference_mode():
            with stage("synthetic_nonzero_banks"):
                fixtures = {k: make_fixture(k, device, factory) for k in WIDTHS}
            results = {
                "numeric": run_numeric_checks(fixtures, stage),
                "invalid": run_invalid_checks(fixtures, stage),
                "native_contract": run_contract_checks(fixtures[2048], stage),
                "graph": run_graph_checks(device, factory, stage),
            }
            if args.queue_lifetime:
                results["queue_lifetime"] = {
                    **run_queue_checks(device, factory, stage),
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
        "scope": "normalized_tail_clamp_cast_reorder_only",
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
        raise RuntimeError("Tail reorder validation requires Linux + NPU; use --plan-only elsewhere")
    directory = args.report_dir or Path(tempfile.mkdtemp(prefix="vq2-tail-reorder-"))
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
        print(f"V4_TAIL_REORDER={report['status']} SUMMARY={summary}", flush=True)
    return 0 if report["status"] == "PASS" else 130 if report["status"] == "INTERRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
