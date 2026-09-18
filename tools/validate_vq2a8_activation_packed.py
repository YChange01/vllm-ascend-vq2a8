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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc-v4-v2" / LIBRARY_NAME)
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
    """Isolate the device and default, but never override, task-queue policy."""

    environment = child_environment(args, environ)
    if args.queue_lifetime:
        environment.setdefault("TASK_QUEUE_ENABLE", "2")
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


def fixture(device, width, groups):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(width * 10 + groups)
    hidden = torch.randn(groups, width, generator=generator).bfloat16().to(device)
    scale = torch.randn(groups, width, generator=generator).to(device)
    bias = torch.randn(groups, width, generator=generator).to(device)
    signs = torch.where(torch.arange(groups * width).reshape(groups, width) % 3 == 0, -1, 1)
    signs = signs.to(device=device, dtype=torch.int8)
    spec = SimpleNamespace(columns=width, rht_true_columns=width, rht_block_size=128)
    return hidden, scale, bias, signs, spec


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


def expected_numeric_results():
    results = []
    for width in (2048, 4096):
        for groups in range(1, 7):
            for mode in ("rowwise_packed", "sign_fused"):
                results.append(f"{mode}_k{width}_g{groups}_random")
        for case in ("zero", "tiny", "impulse", "expanded"):
            for mode in ("rowwise_packed", "sign_fused"):
                results.append(f"{mode}_k{width}_g6_{case}")
    return results


def expected_graph_results():
    return [
        f"{mode}_k{width}_{layout}"
        for width in (2048, 4096)
        for layout in ("contiguous", "expanded")
        for mode in ("rowwise_packed", "sign_fused")
    ]


def run_numeric_checks(device, native, stage):
    import torch

    from vllm_ascend.quantization.vq2a8_activation_packed import PackedRowwiseVQ2A8Preparation

    completed = []
    for width in (2048, 4096):
        for groups in range(1, 7):
            values = fixture(device, width, groups)
            for mode, fuse_sign in (("rowwise_packed", False), ("sign_fused", True)):
                name = f"{mode}_k{width}_g{groups}_random"
                with stage(name):
                    preparation = PackedRowwiseVQ2A8Preparation(fuse_sign=fuse_sign, native_ops=native)
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
            for mode, fuse_sign in (("rowwise_packed", False), ("sign_fused", True)):
                name = f"{mode}_k{width}_g6_{case}"
                with stage(name):
                    preparation = PackedRowwiseVQ2A8Preparation(fuse_sign=fuse_sign, native_ops=native)
                    flags = []
                    actual = preparation.packed(*values, validity=flags.append)
                    torch.npu.synchronize()
                    if len(flags) != 1 or not bool(flags[0]):
                        raise AssertionError(f"{name} rejected finite input")
                    check_outputs(actual, reference(values), name)
                completed.append(name)
    return completed


def run_invalid_checks(device, native, stage):
    import torch

    from vllm_ascend.quantization.vq2a8_activation_packed import PackedRowwiseVQ2A8Preparation

    with stage("invalid_values"):
        for fuse_sign in (False, True):
            for field, bad in ((0, float("nan")), (1, float("inf")), (2, -float("inf")), (3, 0)):
                values = list(fixture(device, 2048, 6))
                values[field].view(-1)[-1] = bad
                flags = []
                preparation = PackedRowwiseVQ2A8Preparation(fuse_sign=fuse_sign, native_ops=native)
                preparation.packed(*values, validity=flags.append)
                torch.npu.synchronize()
                if len(flags) != 1 or bool(flags[0]):
                    raise AssertionError(f"mode={fuse_sign} accepted invalid field {field}")
    return True


def run_graph_checks(device, native, stage):
    import torch

    from vllm_ascend.quantization.vq2a8_activation_packed import PackedRowwiseVQ2A8Preparation

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
        for expanded in (False, True):
            for mode, fuse_sign in (("rowwise_packed", False), ("sign_fused", True)):
                layout = "expanded" if expanded else "contiguous"
                graph_name = f"{mode}_k{width}_{layout}"
                with stage(f"graph_prepare_{graph_name}"):
                    values = list(fixture(device, width, 6))
                    hidden_owner = values[0]
                    if expanded:
                        hidden_owner = values[0][:1].clone()
                        values[0] = hidden_owner.expand(6, -1)
                    flags = []
                    preparation = PackedRowwiseVQ2A8Preparation(fuse_sign=fuse_sign, native_ops=native)
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
                                "hidden": hidden_owner,
                                "scale": values[1],
                                "bias": values[2],
                                "signs": values[3],
                            }
                            if target is not None:
                                targets[target].view(-1)[-1] = bad
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


def run_queue_checks(device, native, stage):
    import torch

    from vllm_ascend.quantization.vq2a8_activation_packed import PackedRowwiseVQ2A8Preparation

    completed = []
    for mode, fuse_sign in (("rowwise_packed", False), ("sign_fused", True)):
        with stage(f"queue_lifetime_{mode}"):
            values = list(fixture(device, 2048, 6))
            preparation = PackedRowwiseVQ2A8Preparation(fuse_sign=fuse_sign, native_ops=native)
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
    return {"modes": completed, "iterations": QUEUE_ITERATIONS, "explicit_per_iteration_sync": False}


def validate_child_evidence(result, *, queue_lifetime):
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
        or results.get("numeric") != expected_numeric_results()
        or results.get("graph") != expected_graph_results()
        or not isinstance(library, dict)
        or not library.get("path")
        or len(library.get("sha256", "")) != 64
    ):
        raise ValueError("Incomplete or spoofed child evidence")
    if queue_lifetime:
        lifetime = results["queue_lifetime"]
        if (
            not isinstance(lifetime, dict)
            or lifetime.get("modes") != ["rowwise_packed", "sign_fused"]
            or lifetime.get("iterations") != QUEUE_ITERATIONS
            or lifetime.get("explicit_per_iteration_sync") is not False
            or lifetime.get("requires_runtime_task_queue_enabled") is not True
            or lifetime.get("task_queue_enable") not in ("1", "2")
        ):
            raise ValueError("Incomplete queue-lifetime child evidence")
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
            if native.activation_preparation_version() != 1:
                raise ValueError("Activation-sign ABI mismatch")
            emit(CASE, "INFO", library=identity, device=info)
        with torch.inference_mode():
            results = {
                "numeric": run_numeric_checks(device, native, stage),
                "invalid": run_invalid_checks(device, native, stage),
                "graph": run_graph_checks(device, native, stage),
            }
            if args.queue_lifetime:
                results["queue_lifetime"] = run_queue_checks(device, native, stage)
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
                    validate_child_evidence(result, queue_lifetime=args.queue_lifetime)
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
