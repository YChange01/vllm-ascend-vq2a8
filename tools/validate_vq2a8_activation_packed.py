#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded Ascend 950 acceptance for packed M=1 activation preparation.

Both the pure-Torch and activation-sign-fused modes must be byte-identical to
the unchanged row-wise oracle.  The probe never loads model weights and makes
no serving or performance claim.
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

CASE = "v4_v2_activation_packed"
REPO = Path(__file__).resolve().parents[1]
LIBRARY_NAME = "libvq2a8_ascendc_v4_v2.so"
QUEUE_ITERATIONS = 513
DEFAULT_PREPARATION_MODES = ("rowwise_packed", "sign_fused")
STRIDED_PREPARATION_MODES = ("sign_fused_strided", "sign_fused_direct")
PREPARATION_MODES = DEFAULT_PREPARATION_MODES + STRIDED_PREPARATION_MODES
INPUT_DTYPES = ("bf16", "fp32")
INPUT_LAYOUTS = ("contiguous", "expanded", "padded")
PAD_COLUMNS = 16  # Keep the BF16 and FP32 row starts 32-byte aligned.
BF16_BOUNDARIES = ("tiny", "subnormal_min", "signed_zero", "boundary_mixed")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc-v4-v2" / LIBRARY_NAME)
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--queue-lifetime", action="store_true")
    parser.add_argument("--preparation-modes", nargs="+", choices=PREPARATION_MODES, default=DEFAULT_PREPARATION_MODES)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if len(set(args.preparation_modes)) != len(args.preparation_modes):
        parser.error("--preparation-modes must not contain duplicates")
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
        "--preparation-modes",
        *args.preparation_modes,
    ]
    return command + (["--queue-lifetime"] if args.queue_lifetime else [])


def probe_environment(args, environ=None):
    """Isolate the device and default, but never override, task-queue policy."""

    environment = child_environment(args, environ)
    if args.queue_lifetime:
        # This probe also captures NPU graphs; the target runtime rejects
        # queue mode 2 during capture. Keep explicit caller policy untouched.
        environment.setdefault("TASK_QUEUE_ENABLE", "1")
    return environment


def assert_bits(actual, expected, name):
    import torch

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: shape/dtype mismatch")
    actual_bytes = actual.detach().view(torch.uint8).cpu()
    expected_bytes = expected.detach().view(torch.uint8).cpu()
    if not torch.equal(actual_bytes, expected_bytes):
        differences = int((actual_bytes != expected_bytes).sum())
        raise AssertionError(f"{name}: {differences} unequal bytes; no tolerance or fallback")


def fixture(device, width, groups, *, dtype=None):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(width * 10 + groups)
    hidden = torch.randn(groups, width, generator=generator).to(dtype=dtype or torch.bfloat16).to(device)
    scale = torch.randn(groups, width, generator=generator).to(device)
    bias = torch.randn(groups, width, generator=generator).to(device)
    signs = torch.where(torch.arange(groups * width).reshape(groups, width) % 3 == 0, -1, 1)
    signs = signs.to(device=device, dtype=torch.int8)
    spec = SimpleNamespace(columns=width, rht_true_columns=width, rht_block_size=128)
    return hidden, scale, bias, signs, spec


def preparation_for_mode(mode, native):
    from vllm_ascend.quantization.vq2a8_activation_packed import PackedRowwiseVQ2A8Preparation

    if mode not in PREPARATION_MODES:
        raise ValueError(f"Unsupported activation preparation mode {mode}")
    options = {"fuse_sign": mode != "rowwise_packed", "native_ops": native}
    if mode in STRIDED_PREPARATION_MODES:
        options.update(strided_sign=True, direct_output=mode == "sign_fused_direct")
    return PackedRowwiseVQ2A8Preparation(**options)


def require_mode_abis(native, modes):
    """Fail closed without probing new symbols for the old perf3 modes."""

    version = native.activation_preparation_version()
    if type(version) is not int or version != 1:
        raise ValueError(f"Activation-sign ABI mismatch: {version!r}")
    versions = {"activation_preparation": version}
    if any(mode in STRIDED_PREPARATION_MODES for mode in modes):
        try:
            version = native.activation_sign_strided_version()
        except (AttributeError, RuntimeError) as error:
            raise ValueError(
                "Selected strided/direct preparation requires a rebuilt strided-sign ABI; no fallback"
            ) from error
        if type(version) is not int or version != 1:
            raise ValueError(f"Strided activation-sign ABI mismatch: {version!r}")
        versions["activation_sign_strided"] = version
    return versions


def input_layout_values(device, width, groups, dtype, layout):
    import torch

    values = list(fixture(device, width, groups, dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]))
    if layout == "expanded":
        owner = values[0][:1].clone()
        values[0] = owner.expand(groups, -1)
    elif layout == "padded":
        owner = torch.full((groups, width + 2 * PAD_COLUMNS), 19, dtype=values[0].dtype, device=device)
        values[0] = owner[:, PAD_COLUMNS : PAD_COLUMNS + width].copy_(values[0])
    elif layout == "contiguous":
        owner = values[0]
    else:
        raise ValueError(f"Unknown input layout {layout}")
    return values, owner


def extended_numeric_cases(modes):
    return [
        (f"{mode}_k{width}_g{groups}_{dtype}_{layout}", mode, width, groups, dtype, layout)
        for width in (2048, 4096)
        for groups in range(1, 7)
        for dtype in INPUT_DTYPES
        for layout in INPUT_LAYOUTS
        for mode in modes
        if mode in STRIDED_PREPARATION_MODES
    ]


def bf16_boundary_cases(modes):
    return [
        (f"{mode}_k{width}_g6_bf16_{layout}_{boundary}", mode, width, layout, boundary)
        for width in (2048, 4096)
        for layout in INPUT_LAYOUTS
        for boundary in BF16_BOUNDARIES
        for mode in modes
        if mode in STRIDED_PREPARATION_MODES
    ]


def bf16_boundary_pattern(boundary):
    import torch

    # Build exact BF16 bits on the CPU, so CPU arithmetic/flush-to-zero policy
    # cannot erase the subnormal input before the device conversion is tested.
    # 0x0080 is finfo(BF16).tiny; 0x0001 is tiny / 128.
    patterns = {
        "tiny": (0x0080, 0x8080),
        "subnormal_min": (0x0001, 0x8001),
        "signed_zero": (0x0000, 0x8000),
        "boundary_mixed": (0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x8080, 0x0081, 0x8081),
    }
    return torch.tensor(patterns[boundary], dtype=torch.uint16).view(torch.bfloat16)


def check_native_sign_boundary(native, values, name):
    """Compare the signed FP32 intermediate before FP8 can hide a mismatch."""

    hidden, scale, bias, signs, _ = values
    actual, actual_valid = native.activation_sign_strided(hidden, scale, bias, signs)
    expected, expected_valid = native.activation_sign(hidden.float().contiguous(), scale, bias, signs)
    assert_bits(actual, expected, f"{name}_signed_fp32")
    assert_bits(actual_valid, expected_valid, f"{name}_native_validity")


def run_bf16_boundary_checks(device, native, stage, modes):
    import torch

    completed = []
    for name, mode, width, layout, boundary in bf16_boundary_cases(modes):
        with stage(name):
            values, owner = input_layout_values(device, width, 6, "bf16", layout)
            pattern = bf16_boundary_pattern(boundary)
            row = pattern.repeat((width + pattern.numel() - 1) // pattern.numel())[:width]
            target = owner if layout == "expanded" else values[0]
            target.copy_(row.expand(target.shape[0], -1).contiguous().to(device))
            # Check raw copied BF16 bits as well, including the sign of zero.
            assert_bits(values[0].contiguous(), row.expand(6, -1).contiguous(), f"{name}_input_bits")
            check_native_sign_boundary(native, values, name)
            flags = []
            outputs = preparation_for_mode(mode, native).packed(*values, validity=flags.append)
            torch.npu.synchronize()
            if len(flags) != 1 or not bool(flags[0]):
                raise AssertionError(f"{name} rejected finite boundary input")
            check_outputs(outputs, reference(values), name)
        completed.append(name)
    return completed


def reference(values):
    import torch

    from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation

    hidden, scale, bias, signs, spec = values
    requests = [
        (
            hidden[index : index + 1],
            {"weight_scale": scale[index], "weight_bias": bias[index], "rht_sign": signs[index]},
            spec,
        )
        for index in range(hidden.shape[0])
    ]
    prepared = RowwiseVQ2A8Preparation(compact=True).many(requests)
    return tuple(torch.cat(fields).contiguous() for fields in zip(*prepared))


def check_outputs(actual, expected, name):
    for field, got, want in zip(("q", "scale", "bias"), actual, expected):
        assert_bits(got, want, f"{name}_{field}")


def expected_numeric_results(modes=DEFAULT_PREPARATION_MODES):
    results = []
    for width in (2048, 4096):
        for groups in range(1, 7):
            for mode in modes:
                results.append(f"{mode}_k{width}_g{groups}_random")
        for case in ("zero", "tiny", "impulse", "expanded"):
            for mode in modes:
                results.append(f"{mode}_k{width}_g6_{case}")
    return (
        results + [case[0] for case in extended_numeric_cases(modes)] + [case[0] for case in bf16_boundary_cases(modes)]
    )


def graph_cases(modes=DEFAULT_PREPARATION_MODES):
    cases = []
    for width in (2048, 4096):
        cases.extend(
            (f"{mode}_k{width}_{layout}", mode, width, "bf16", layout)
            for layout in ("contiguous", "expanded")
            for mode in modes
            if mode in DEFAULT_PREPARATION_MODES
        )
        cases.extend(
            (f"{mode}_k{width}_{dtype}_{layout}", mode, width, dtype, layout)
            for dtype in INPUT_DTYPES
            for layout in INPUT_LAYOUTS
            for mode in modes
            if mode in STRIDED_PREPARATION_MODES
        )
    return cases


def expected_graph_results(modes=DEFAULT_PREPARATION_MODES):
    return [case[0] for case in graph_cases(modes)]


def run_numeric_checks(device, native, stage, modes=DEFAULT_PREPARATION_MODES):
    import torch

    completed = []
    for width in (2048, 4096):
        for groups in range(1, 7):
            values = fixture(device, width, groups)
            for mode in modes:
                name = f"{mode}_k{width}_g{groups}_random"
                with stage(name):
                    preparation = preparation_for_mode(mode, native)
                    flags = []
                    actual = preparation.packed(*values, validity=flags.append)
                    torch.npu.synchronize()
                    if len(flags) != 1 or not bool(flags[0]):
                        raise AssertionError(f"{name} rejected finite input")
                    check_outputs(actual, reference(values), name)
                completed.append(name)
        for case in ("zero", "tiny", "impulse", "expanded"):
            values = list(fixture(device, width, 6))
            if case == "zero":
                values[0].zero_()
            elif case == "tiny":
                values[0].mul_(1e-15)
            elif case == "impulse":
                values[0].zero_()
                values[0][:, -1] = -1
            else:
                values[0] = values[0][:1].expand(6, -1)
            for mode in modes:
                name = f"{mode}_k{width}_g6_{case}"
                with stage(name):
                    preparation = preparation_for_mode(mode, native)
                    flags = []
                    actual = preparation.packed(*values, validity=flags.append)
                    torch.npu.synchronize()
                    if len(flags) != 1 or not bool(flags[0]):
                        raise AssertionError(f"{name} rejected finite input")
                    check_outputs(actual, reference(values), name)
                completed.append(name)
    for name, mode, width, groups, dtype, layout in extended_numeric_cases(modes):
        with stage(name):
            values, owner = input_layout_values(device, width, groups, dtype, layout)
            before = owner.clone()
            flags = []
            actual = preparation_for_mode(mode, native).packed(*values, validity=flags.append)
            torch.npu.synchronize()
            if len(flags) != 1 or not bool(flags[0]):
                raise AssertionError(f"{name} rejected finite input")
            check_outputs(actual, reference(values), name)
            assert_bits(owner, before, f"{name}_input_storage_unchanged")
        completed.append(name)
    completed.extend(run_bf16_boundary_checks(device, native, stage, modes))
    return completed


def run_invalid_checks(device, native, stage, modes=DEFAULT_PREPARATION_MODES):
    import torch

    with stage("invalid_values"):
        for mode in modes:
            for field, bad in ((0, float("nan")), (1, float("inf")), (2, -float("inf")), (3, 0)):
                values = list(fixture(device, 2048, 6))
                values[field].view(-1)[-1] = bad
                flags = []
                preparation = preparation_for_mode(mode, native)
                preparation.packed(*values, validity=flags.append)
                torch.npu.synchronize()
                if len(flags) != 1 or bool(flags[0]):
                    raise AssertionError(f"mode={mode} accepted invalid field {field}")
    return True


def run_graph_checks(device, native, stage, modes=DEFAULT_PREPARATION_MODES):
    import torch

    completed = []
    cases = (
        ("normal", None, None, True),
        ("different", "hidden", 0.5, True),
        ("invalid_hidden", "hidden", float("nan"), False),
        ("recovered_hidden", None, None, True),
        ("invalid_scale", "scale", float("inf"), False),
        ("recovered_scale", None, None, True),
        ("invalid_bias", "bias", -float("inf"), False),
        ("recovered_bias", None, None, True),
        ("invalid_sign", "signs", 0, False),
        ("recovered_sign", None, None, True),
    )
    for width in (2048, 4096):
        for graph_name, mode, case_width, dtype, layout in graph_cases(modes):
            if case_width != width:
                continue
            with stage(f"graph_prepare_{graph_name}"):
                values, hidden_owner = input_layout_values(device, width, 6, dtype, layout)
                flags = []
                preparation = preparation_for_mode(mode, native)
                preparation.prepare_for_graph(device, values[-1].rht_block_size)
                for _ in range(2):
                    preparation.packed(*values, validity=flags.append)
                torch.npu.synchronize()
                flags.clear()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    outputs = preparation.packed(*values, validity=flags.append)
                captured_flag = flags[-1]
            try:
                for case, target, bad, expected_valid in cases:
                    with stage(f"graph_replay_{graph_name}_{case}"):
                        hidden_owner.fill_(-0.25)
                        values[1].fill_(1.0)
                        values[2].fill_(0.125)
                        values[3].fill_(1)
                        targets = {
                            # Expanded rows share the owner; padded rows need
                            # a logical element, not the padding sentinel.
                            "hidden": hidden_owner if layout == "expanded" else values[0],
                            "scale": values[1],
                            "bias": values[2],
                            "signs": values[3],
                        }
                        if target is not None:
                            targets[target][-1, -1] = bad
                        graph.replay()
                        torch.npu.synchronize()
                        if bool(captured_flag) != expected_valid:
                            raise AssertionError(f"{graph_name} replayed stale validity for {case}")
                        if expected_valid:
                            check_outputs(outputs, reference(values), f"{graph_name}_{case}")
                completed.append(graph_name)
            finally:
                torch.npu.synchronize()
                graph.reset()
    return completed


def queue_input_cases(modes):
    return [
        (f"{mode}_k{width}_{dtype}_{layout}", mode, width, dtype, layout)
        for mode in modes
        if mode in STRIDED_PREPARATION_MODES
        for width in (2048, 4096)
        for dtype in INPUT_DTYPES
        for layout in INPUT_LAYOUTS
    ]


def run_strided_queue_case(device, native, stage, case):
    import torch

    name, mode, width, dtype, layout = case
    with stage(f"queue_lifetime_{name}"):
        template, template_owner = input_layout_values(device, width, 6, dtype, layout)
        preparation = preparation_for_mode(mode, native)
        expected_first = reference(template)
        flags = []
        first = preparation.packed(*template, validity=flags.append)
        first_valid = flags[-1]
        for iteration in range(QUEUE_ITERATIONS):
            owner = template_owner.clone()
            if layout == "expanded":
                hidden = owner.expand(6, -1)
            elif layout == "padded":
                hidden = owner[:, PAD_COLUMNS : PAD_COLUMNS + width]
            else:
                hidden = owner
            # NPU fill/ViewCopy cannot write a stride-zero expanded target.
            # Fill its single-row owner but still submit the expanded view;
            # padded inputs keep their padding untouched. No extra owner alias
            # may survive the explicit release below.
            (owner if layout == "expanded" else hidden).fill_((iteration % 7 - 3) / 8)
            values = [hidden, *(value.clone() for value in template[1:4]), template[4]]
            flags.clear()
            output = preparation.packed(*values, validity=flags.append)
            # Device work must not rely on these Python owners remaining alive.
            del values, hidden, owner
        final_valid = flags[-1]
        output_finite = torch.stack([torch.isfinite(value.float()).all() for value in output]).all()
        torch.npu.synchronize()
        if not bool(first_valid & final_valid & output_finite):
            raise AssertionError(f"{name} queue-lifetime validity failed")
        check_outputs(first, expected_first, f"{name}_retained")
        template_owner.fill_(((QUEUE_ITERATIONS - 1) % 7 - 3) / 8)
        check_outputs(output, reference(template), f"{name}_final")
    return name


def run_queue_checks(device, native, stage, modes=DEFAULT_PREPARATION_MODES):
    import torch

    completed = []
    for mode in modes:
        if mode in STRIDED_PREPARATION_MODES:
            continue
        with stage(f"queue_lifetime_{mode}"):
            values = list(fixture(device, 2048, 6))
            preparation = preparation_for_mode(mode, native)
            flags = []
            first = tuple(value.clone() for value in preparation.packed(*values, validity=flags.append))
            first_valid = flags[-1]
            for iteration in range(QUEUE_ITERATIONS):
                values[0].fill_((iteration % 7 - 3) / 8)
                flags.clear()
                output = preparation.packed(*values, validity=flags.append)
            final_valid = flags[-1]
            output_finite = torch.stack([torch.isfinite(value.float()).all() for value in output]).all()
            torch.npu.synchronize()
            if not bool(first_valid & final_valid & output_finite):
                raise AssertionError(f"{mode} queue-lifetime validity failed")
            check_outputs(first, reference(fixture(device, 2048, 6)), f"{mode}_retained")
            check_outputs(output, reference(values), f"{mode}_final")
        completed.append(mode)
    input_cases = [run_strided_queue_case(device, native, stage, case) for case in queue_input_cases(modes)]
    evidence = {"modes": list(modes), "iterations": QUEUE_ITERATIONS, "explicit_per_iteration_sync": False}
    if input_cases:
        evidence.update(input_cases=input_cases, input_owners_dropped_before_fence=True, retained_outputs_checked=True)
    return evidence


def validate_child_evidence(result, *, queue_lifetime, modes=DEFAULT_PREPARATION_MODES):
    """Reject a successful exit unless its final event proves every gate."""

    if result.get("status") != "PASS":
        raise ValueError("Child process did not report PASS")
    events = result.get("events")
    final = events[-1] if isinstance(events, list) and events else {}
    required = {"numeric", "invalid", "graph"} | ({"queue_lifetime"} if queue_lifetime else set())
    results = final.get("results")
    library = final.get("library")
    if (
        final.get("case") != CASE
        or final.get("event") != "CASE_PASS"
        or final.get("device_execution_verified") is not True
        or final.get("graph_verified") is not True
        or final.get("model_integration_verified") is not False
        or final.get("performance_verified") is not False
        or not isinstance(results, dict)
        or set(results) != required
        or results.get("invalid") is not True
        or results.get("numeric") != expected_numeric_results(modes)
        or results.get("graph") != expected_graph_results(modes)
        or not isinstance(library, dict)
        or not library.get("path")
        or len(library.get("sha256", "")) != 64
    ):
        raise ValueError("Incomplete or spoofed child evidence")
    if any(mode in STRIDED_PREPARATION_MODES for mode in modes):
        native_abis = final.get("native_abis")
        if (
            final.get("preparation_modes") != list(modes)
            or not isinstance(native_abis, dict)
            or native_abis != {"activation_preparation": 1, "activation_sign_strided": 1}
            or any(type(version) is not int for version in native_abis.values())
        ):
            raise ValueError("Incomplete selected-mode or strided ABI evidence")
    if queue_lifetime:
        lifetime = results["queue_lifetime"]
        if (
            not isinstance(lifetime, dict)
            or lifetime.get("modes") != list(modes)
            or lifetime.get("iterations") != QUEUE_ITERATIONS
            or lifetime.get("explicit_per_iteration_sync") is not False
            or lifetime.get("requires_runtime_task_queue_enabled") is not True
            or lifetime.get("task_queue_enable") not in ("1", "2")
        ):
            raise ValueError("Incomplete queue-lifetime child evidence")
        expected_cases = [case[0] for case in queue_input_cases(modes)]
        if expected_cases and (
            lifetime.get("input_cases") != expected_cases
            or lifetime.get("input_owners_dropped_before_fence") is not True
            or lifetime.get("retained_outputs_checked") is not True
        ):
            raise ValueError("Incomplete strided queue-lifetime input coverage")
    return final


def run_case_child(args):
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical_npu):
        raise ValueError("Child physical NPU mapping differs from the requested device")
    faulthandler.enable()
    faulthandler.dump_traceback_later(min(30, args.timeout_s), repeat=True)

    def synchronize():
        return None

    stage = stage_recorder(CASE, lambda: synchronize())
    try:
        task_queue_enable = os.environ.get("TASK_QUEUE_ENABLE")
        if args.queue_lifetime and task_queue_enable not in ("1", "2"):
            raise RuntimeError("--queue-lifetime requires TASK_QUEUE_ENABLE=1 or 2 in the isolated child")
        with stage("imports"):
            import torch
            import torch_npu  # noqa: F401

            from tools.validate_vq2a8_ascendc import require_hardware_runtime
            from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device
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
                raise ValueError(f"Require {LIBRARY_NAME}; no library fallback")
            identity = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            torch.ops.load_library(str(path))
            native = torch.ops.vq2a8_ascendc_v4_v2
            native_abis = require_mode_abis(native, args.preparation_modes)
            emit(
                CASE,
                "INFO",
                library=identity,
                device=info,
                preparation_modes=args.preparation_modes,
                native_abis=native_abis,
            )
        with torch.inference_mode():
            results = {
                "numeric": run_numeric_checks(device, native, stage, args.preparation_modes),
                "invalid": run_invalid_checks(device, native, stage, args.preparation_modes),
                "graph": run_graph_checks(device, native, stage, args.preparation_modes),
            }
            if args.queue_lifetime:
                results["queue_lifetime"] = run_queue_checks(device, native, stage, args.preparation_modes)
                results["queue_lifetime"].update(
                    task_queue_enable=task_queue_enable,
                    requires_runtime_task_queue_enabled=True,
                )
        with stage("final_sync"):
            pass
        emit(
            CASE,
            "CASE_PASS",
            results=results,
            library=identity,
            preparation_modes=list(args.preparation_modes),
            native_abis=native_abis,
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
        "scope": "packed_m1_activation_only",
        "command": child_command(args),
        "preparation_modes": list(args.preparation_modes),
        "requires_strided_sign_abi": any(mode in STRIDED_PREPARATION_MODES for mode in args.preparation_modes),
        "status": "PLANNED",
        "device_execution_verified": False,
        "graph_verified": False,
        "model_integration_verified": False,
        "performance_verified": False,
        "queue_lifetime": {
            "enabled": args.queue_lifetime,
            "iterations": QUEUE_ITERATIONS if args.queue_lifetime else 0,
            "runtime_queue_slots_measured": False,
            "requires_runtime_task_queue_enabled": True,
            "explicit_per_iteration_synchronize": False,
        },
    }
    if args.plan_only:
        print(json.dumps(report, indent=2))
        return 0
    if os.name != "posix":
        raise RuntimeError("Packed activation validation requires Linux + NPU; use --plan-only elsewhere")
    directory = args.report_dir or Path(tempfile.mkdtemp(prefix="vq2-activation-packed-"))
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
                try:
                    validate_child_evidence(result, queue_lifetime=args.queue_lifetime, modes=args.preparation_modes)
                except ValueError as error:
                    report.update(status="FAIL", error=str(error))
                else:
                    report.update(device_execution_verified=True, graph_verified=True)
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        traceback.print_exc()
    finally:
        summary = directory / "summary.json"
        summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"V4_PACKED_ACTIVATION={report['status']} SUMMARY={summary}", flush=True)
    return 0 if report["status"] == "PASS" else 130 if report["status"] == "INTERRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
