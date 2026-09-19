#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded resident select/sign acceptance; not model or performance proof."""

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

CASE = "v4_v2_select_sign"
REPO = Path(__file__).resolve().parents[1]
LIBRARY_NAME = "libvq2a8_ascendc_v4_v2.so"
WIDTHS = (2048, 4096)
DTYPES = ("bfloat16", "float32")
LAYOUTS = ("contiguous", "expanded", "padded")
ID_CASES = ("valid", "duplicate", "low", "high", "min", "max")
METADATA_CASES = ("scale_nan", "bias_inf", "sign_zero", "sign_two", "sign_min")
GRAPH_PHASES = (
    "normal",
    "invalid_slot",
    "recovered_slot",
    "nan_input",
    "recovered_input",
    "invalid_sign",
    "recovered_sign",
)
QUEUE_ITERATIONS = 513
PRESSURE_BYTES = 2 * 1024 * 1024
NATIVE_CONTRACT_CASES = 14


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
    if args.physical_npu < 0 or args.timeout_s <= 0 or (args.child and args.plan_only):
        parser.error("Require physical NPU >= 0, timeout > 0; no --child with --plan-only")
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
        (k, dtype, layout, groups, pattern)
        for k in WIDTHS
        for dtype in DTYPES
        for layout in LAYOUTS
        for groups in range(1, 7)
        for pattern in ID_CASES
    ]


def numeric_names():
    return (
        [f"k{k}_{dtype}_{layout}_g{g}_{pattern}" for k, dtype, layout, g, pattern in numeric_cases()]
        + [f"metadata_k{k}_{dtype}_{pattern}" for k in WIDTHS for dtype in DTYPES for pattern in METADATA_CASES]
        + [f"last_expert_k{k}" for k in WIDTHS]
    )


def graph_names():
    return [
        f"graph_k{k}_{dtype}_{layout}_{phase}"
        for k in WIDTHS
        for dtype in DTYPES
        for layout in LAYOUTS
        for phase in GRAPH_PHASES
    ]


def ids_values(groups, experts, pattern):
    ids = [i % experts for i in range(groups)]
    if pattern == "duplicate":
        ids = [experts - 1] * groups
    elif pattern in ("low", "high", "min", "max"):
        ids[-1] = {"low": -1, "high": experts, "min": -(1 << 63), "max": (1 << 63) - 1}[pattern]
    return ids


def make_bank(device, factory, width, experts=3, pattern="normal"):
    import torch

    # Projection payload is never used here; share its immutable owners to
    # exercise N=256 without allocating 256 full compressed matrices.
    packed = torch.zeros((128, width // 16, 16, 8), dtype=torch.uint8, device=device)
    books = torch.zeros((width // 256, 128, 32), dtype=torch.uint8, device=device)
    order = torch.arange(width, dtype=torch.int64, device=device)
    scale, bias, sign = [], [], []
    for expert in range(experts):
        positions = torch.arange(width)
        s = ((positions % 17).float() - 8) / 16 + expert / 128
        b = ((positions % 13).float() - 6) / 8 - expert / 64
        signs = torch.where(positions % 2 == expert % 2, -1, 1).to(torch.int8)
        if pattern == "scale_nan":
            s[-1] = float("nan")
        elif pattern == "bias_inf":
            b[width // 2] = float("inf")
        elif pattern in ("sign_zero", "sign_two", "sign_min"):
            signs[0] = {"sign_zero": 0, "sign_two": 2, "sign_min": -128}[pattern]
        scale.append(s.to(device))
        bias.append(b.to(device))
        sign.append(signs.to(device))
    owners = ([packed] * experts, [books] * experts, [order] * experts, scale, bias, sign)
    return factory(*owners), owners


def make_hidden(device, width, groups, dtype, layout):
    import torch

    rows = 1 if layout == "expanded" else groups
    columns = width + 32 if layout == "padded" else width
    values = ((torch.arange(rows * columns).reshape(rows, columns) % 127).float() - 63) / 32
    # Include both signs of zero and exactly representable negative values.
    values[:, 0] = 0.0
    values[:, 1] = -0.0
    owner = values.to(dtype=getattr(torch, dtype), device=device)
    return hidden_view(owner, groups, width, layout), owner


def hidden_view(owner, groups, width, layout):
    if layout == "expanded":
        return owner.expand(groups, width)
    if layout == "padded":
        return owner[:, 16 : 16 + width]
    return owner


def reference(bank, hidden, ids, native):
    scale, bias, signs, selected = bank.select(ids)
    signed, input_status = native.activation_sign_strided(hidden, scale, bias, signs)
    return signed, scale, bias, selected, input_status


def assert_bits(actual, expected, name):
    import torch

    if len(actual) != 5 or len(expected) != 5:
        raise AssertionError(f"{name}: five outputs required")
    for field, got, want in zip(("signed", "scale", "bias", "select_status", "input_status"), actual, expected):
        if got.dtype != want.dtype or got.shape != want.shape or got.device != want.device or not got.is_contiguous():
            raise AssertionError(f"{name}/{field}: dtype/shape/device/layout differs")
        if not torch.equal(
            got.detach().cpu().contiguous().view(torch.uint8), want.detach().cpu().contiguous().view(torch.uint8)
        ):
            raise AssertionError(f"{name}/{field}: unequal bytes; no tolerance or fallback")


def check_owners(before, after, name):
    import torch

    for previous, current in zip(before, after):
        if not torch.equal(previous, current.cpu().contiguous().view(torch.uint8)):
            raise AssertionError(f"{name}: input or metadata owner mutated")


def snapshot_owners(owners):
    import torch

    return [owner.cpu().contiguous().view(torch.uint8).clone() for owner in owners]


def run_numeric(device, factory, native, fused, stage):
    import torch

    banks = {k: make_bank(device, factory, k) for k in WIDTHS}
    names = []
    for width, dtype, layout, groups, pattern in numeric_cases():
        name = f"k{width}_{dtype}_{layout}_g{groups}_{pattern}"
        bank, owners = banks[width]
        hidden, owner = make_hidden(device, width, groups, dtype, layout)
        # Nonzero INT64 storage offset is naturally aligned, not 32-byte aligned.
        id_owner = torch.tensor([73, *ids_values(groups, 3, pattern), 91], dtype=torch.int64, device=device)
        ids = id_owner[1:-1]
        inspected = [owner, id_owner, *owners[3], *owners[4], *owners[5]]
        before = snapshot_owners(inspected)
        expected = reference(bank, hidden, ids, native)
        with stage(name):
            actual = fused(bank, hidden, ids)
        assert_bits(actual, expected, name)
        check_owners(before, inspected, name)
        names.append(name)
    for width in WIDTHS:
        for dtype in DTYPES:
            for pattern in METADATA_CASES:
                name = f"metadata_k{width}_{dtype}_{pattern}"
                bank, owners = make_bank(device, factory, width, pattern=pattern)
                hidden, owner = make_hidden(device, width, 6, dtype, "expanded")
                ids = torch.tensor([0, 1, 2, 2, 1, 0], dtype=torch.int64, device=device)
                expected = reference(bank, hidden, ids, native)
                with stage(name):
                    actual = fused(bank, hidden, ids)
                assert_bits(actual, expected, name)
                if bool(actual[4].cpu().any()):
                    raise AssertionError(f"{name}: invalid preparation metadata accepted")
                names.append(name)
    for width in WIDTHS:
        name = f"last_expert_k{width}"
        bank, owners = make_bank(device, factory, width, experts=256)
        hidden, owner = make_hidden(device, width, 6, "float32", "padded")
        ids = torch.tensor([255, 0, 255, -1, 256, 1], dtype=torch.int64, device=device)
        expected = reference(bank, hidden, ids, native)
        with stage(name):
            actual = fused(bank, hidden, ids)
        assert_bits(actual, expected, name)
        names.append(name)
    return names


def run_graph(device, factory, native, fused, stage):
    import torch

    names = []
    for width in WIDTHS:
        for dtype in DTYPES:
            for layout in LAYOUTS:
                prefix = f"graph_k{width}_{dtype}_{layout}"
                stream = torch.npu.Stream(device=device)
                with torch.npu.stream(stream):
                    bank, owners = make_bank(device, factory, width)
                    hidden, owner = make_hidden(device, width, 6, dtype, layout)
                    clean_owner = owner.clone()
                    ids = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64, device=device)
                    clean_ids = ids.clone()
                    clean_sign = owners[5][0].clone()
                    for _ in range(3):
                        fused(bank, hidden, ids)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with stage(prefix + "_capture"), torch.npu.graph(graph, stream=stream):
                    actual = fused(bank, hidden, ids)
                for phase in GRAPH_PHASES:
                    name = prefix + "_" + phase
                    with torch.npu.stream(stream):
                        owner.copy_(clean_owner)
                        ids.copy_(clean_ids)
                        owners[5][0].copy_(clean_sign)
                        if phase == "invalid_slot":
                            ids[-1:].fill_(-(1 << 63))
                        elif phase == "nan_input":
                            # Mutate backing storage, never fill an expanded view.
                            owner[:, 16:17].fill_(float("nan"))
                        elif phase == "invalid_sign":
                            # Deliberate validity injection only, not supported
                            # serving-time mutation of immutable expert payloads.
                            owners[5][0][-1:].zero_()
                        expected = tuple(value.clone() for value in reference(bank, hidden, ids, native))
                    torch.npu.synchronize()
                    inspected = [owner, ids, *owners[3], *owners[4], *owners[5]]
                    before = snapshot_owners(inspected)
                    with stage(name), torch.npu.stream(stream):
                        graph.replay()
                    assert_bits(actual, expected, name)
                    check_owners(before, inspected, name)
                    names.append(name)
                # Do not attempt reset after a failed replay/driver error.
                with stage(prefix + "_reset"):
                    graph.reset()
    return names


def run_native_contract(device, factory, stage):
    import torch

    width = 2048
    bank, owners = make_bank(device, factory, width)
    hidden = torch.zeros((2, width), dtype=torch.float32, device=device)
    ids = torch.zeros(2, dtype=torch.int64, device=device)
    malformed = [
        (hidden.half(), ids),
        (hidden.reshape(-1), ids),
        (hidden[:, :1024].contiguous(), ids),
        (hidden[:0], ids[:0]),
        (torch.zeros((7, width), device=device), torch.zeros(7, dtype=torch.int64, device=device)),
        (torch.zeros((2, width * 2), device=device)[:, ::2], ids),
        (hidden.as_strided((2, width), (1, 1)), ids),
        (torch.zeros((2, width + 1), device=device)[:, :width], ids),
        (torch.zeros(2 * width + 1, device=device)[1:].reshape(2, width), ids),
        (hidden, ids.int()),
        (hidden, ids[:1]),
        (hidden, torch.zeros(4, dtype=torch.int64, device=device)[::2]),
        (hidden.cpu(), ids),
        (hidden, ids.cpu()),
    ]
    for index, (value, slots) in enumerate(malformed):
        with stage(f"native_contract_{index}"):
            try:
                bank.select_sign(value, slots)
            except (RuntimeError, ValueError):
                pass
            else:
                raise AssertionError(f"Native select/sign accepted malformed case {index}")
    return len(malformed)


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
        "task_queue_enable": "1",
        "runtime_queue_slots_measured": False,
    }


def run_queue(device, factory, native, fused, stage):
    import torch

    templates = []
    banks = {}
    for width in WIDTHS:
        banks[width] = make_bank(device, factory, width)
        for dtype in DTYPES:
            for layout in LAYOUTS:
                for pattern in ("valid", "min", "duplicate"):
                    hidden, owner = make_hidden(device, width, 6, dtype, layout)
                    ids = torch.tensor(ids_values(6, 3, pattern), dtype=torch.int64, device=device)
                    expected = tuple(t.cpu().clone() for t in reference(banks[width][0], hidden, ids, native))
                    templates.append((width, layout, owner, ids, expected))
    torch.npu.synchronize()
    outputs = []
    with stage("queue_lifetime"):
        for iteration in range(QUEUE_ITERATIONS):
            width, layout, template_owner, template_ids, expected = templates[iteration % len(templates)]
            owner = template_owner.clone()
            hidden = hidden_view(owner, 6, width, layout)
            ids = template_ids.clone()
            values = fused(banks[width][0], hidden, ids)
            ordinary = values[1] + 1.0
            outputs.append((values, ordinary, expected))
            del hidden, owner, ids, values
            pressure = torch.empty(PRESSURE_BYTES, dtype=torch.uint8, device=device)
            pressure.fill_(iteration % 251)
            del pressure
        # Drop native bank plus Python metadata owners while queued indirect
        # reads may still be outstanding. Templates contain no bank metadata.
        banks.clear()
        pressure = torch.empty(PRESSURE_BYTES, dtype=torch.uint8, device=device)
        pressure.fill_(91)
        del pressure
    for index, (values, ordinary, expected) in enumerate(outputs):
        assert_bits(tuple(value.cpu() for value in values), expected, f"queue_{index}")
        want = expected[1] + 1.0
        if not torch.equal(ordinary.cpu().view(torch.uint8), want.view(torch.uint8)):
            raise AssertionError(f"queue_{index}: ordinary operation changed")
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
        or not isinstance(identity.get("path"), str)
        or Path(identity["path"]).name != LIBRARY_NAME
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        or not any(event.get("event") == "PASS" and event.get("stage") == "final_sync" for event in events)
    ):
        raise ValueError("Incomplete resident select/sign child evidence")


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
            from vllm_ascend.quantization.vq2a8_select_sign import FusedSelectSign
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
            fused = FusedSelectSign(native)
            factory = torch.classes.vq2a8_ascendc_v4_v2.ResidentBank
            emit(CASE, "INFO", library=identity, device=info, native_abi=1, task_queue_enable=queue)
        with torch.inference_mode():
            results = {
                "numeric": run_numeric(device, factory, native, fused, stage),
                "native_contract": run_native_contract(device, factory, stage),
                "graph": run_graph(device, factory, native, fused, stage),
            }
            if args.queue_lifetime:
                results["queue_lifetime"] = run_queue(device, factory, native, fused, stage)
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
        "scope": "resident_select_sign_only",
        "command": child_command(args),
        "status": "PLANNED",
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
        raise RuntimeError("Select/sign validation requires Linux + NPU; use --plan-only elsewhere")
    directory = args.report_dir or Path(tempfile.mkdtemp(prefix="vq2-select-sign-"))
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
        print(f"V4_SELECT_SIGN={report['status']} SUMMARY={summary}", flush=True)
    return 0 if report["status"] == "PASS" else 130 if report["status"] == "INTERRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
