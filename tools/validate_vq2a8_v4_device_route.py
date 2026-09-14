#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate V4 device-selected V1 projections using small synthetic experts.

No model/artifact loading, recompilation, global package audit or performance
claim. One isolated child has a deadline and never resets a shared NPU.
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
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

from tools.diagnose_vq2a8_tp1_startup import child_environment, emit, parse_snapshot, run_child, stage_recorder

REPO = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO / "build/vq2a8-ascendc-v4-device-route"
CASE = "v4_device_route"
SCOPE = "synthetic_v1_layout_device_route_only"
EXPERTS = 4
OUTPUT_COLUMNS = 64
REDUCTIONS = (512, 1024)
PAYLOAD_FIELDS = ("packed_indices", "codebooks", "codebook_tile_ids", "weight_scale", "weight_bias", "rht_sign")
VALID_ROUTES = ((0,), (3, 1), (0, 1, 2, 3, 1, 0), (3, 3, 3, 3, 3, 3), (2, 0, 3, 1, 2, 0))
QUEUE_CAPACITY_REFERENCE = 4096
QUEUE_CALLS_PER_ITERATION = 3  # select, project, ordinary matmul; excludes clones/checks.
QUEUE_MIXED_CALLS_PER_ITERATION = 6  # Original three plus single/grouped/pipeline projection.
QUEUE_MIN_ITERATIONS = QUEUE_CAPACITY_REFERENCE // QUEUE_CALLS_PER_ITERATION + 1
QUEUE_DEFAULT_ITERATIONS = 2049
QUEUE_MAX_ITERATIONS = 8192
QUEUE_PROGRESS_INTERVAL = 256
QUEUE_EARLY_PROGRESS_ITERATIONS = 8


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-npu", type=int, default=2)
    parser.add_argument("--library", type=Path, default=BUILD_DIR / "libvq2a8_ascendc.so")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--allow-busy", action="store_true", help="allow a known busy NPU; may affect other jobs")
    parser.add_argument("--launch-blocking", choices=("0", "1"), default="0")
    parser.add_argument(
        "--queue-lifetime",
        action="store_true",
        help="also run resident-only then mixed V1/resident asynchronous lifetime pressure; no model loading",
    )
    parser.add_argument(
        "--queue-iterations",
        type=int,
        default=QUEUE_DEFAULT_ITERATIONS,
        help=f"queue-lifetime iterations ({QUEUE_MIN_ITERATIONS}..{QUEUE_MAX_ITERATIONS}; default %(default)s)",
    )
    parser.add_argument("--report-dir", type=Path, help="new report directory; never overwrite an old result")
    parser.add_argument("--plan-only", action="store_true", help="print commands; do not import torch or use an NPU")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.physical_npu < 0 or args.timeout_s <= 0:
        parser.error("Require --physical-npu >= 0 and --timeout-s > 0")
    if args.child and args.plan_only:
        parser.error("--child and --plan-only cannot be combined")
    if not QUEUE_MIN_ITERATIONS <= args.queue_iterations <= QUEUE_MAX_ITERATIONS:
        parser.error(f"--queue-iterations must be {QUEUE_MIN_ITERATIONS}..{QUEUE_MAX_ITERATIONS}")
    if args.queue_lifetime and args.launch_blocking != "0":
        parser.error("--queue-lifetime requires --launch-blocking 0")
    if not args.queue_lifetime and args.queue_iterations != QUEUE_DEFAULT_ITERATIONS:
        parser.error("--queue-iterations requires --queue-lifetime")
    return args


def child_command(args):
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--child",
        "--physical-npu",
        str(args.physical_npu),
        "--library",
        str(args.library.resolve()),
        "--timeout-s",
        str(args.timeout_s),
        "--launch-blocking",
        args.launch_blocking,
    ]
    if args.queue_lifetime:
        command.extend(["--queue-lifetime", "--queue-iterations", str(args.queue_iterations)])
    return command


def queue_lifetime_phases(iterations):
    resident_paths = ["resident_select", "resident_project", "matmul"]
    return [
        {
            "stage": "queue_lifetime_wrap",
            "iterations": iterations,
            "minimum_operator_calls": iterations * QUEUE_CALLS_PER_ITERATION,
            "operator_paths": resident_paths,
            "grouped_descriptor_copy_is_blocking": False,
        },
        {
            "stage": "queue_lifetime_mixed",
            "iterations": iterations,
            "minimum_operator_calls": iterations * QUEUE_MIXED_CALLS_PER_ITERATION,
            "operator_paths": [*resident_paths, "projection", "grouped_projection", "grouped_projection_pipeline"],
            "grouped_descriptor_copy_is_blocking": True,
        },
    ]


def queue_lifetime_plan(args):
    phases = queue_lifetime_phases(args.queue_iterations if args.queue_lifetime else 0)
    return {
        "enabled": args.queue_lifetime,
        "iterations": args.queue_iterations if args.queue_lifetime else 0,
        "minimum_operator_calls": sum(phase["minimum_operator_calls"] for phase in phases),
        "phases": phases,
        "short_checks_and_pressure_share_process": True,
        "queue_capacity_reference": QUEUE_CAPACITY_REFERENCE,
        "runtime_queue_enabled_or_slots_measured": False,
        "requires_runtime_task_queue_enabled": True,
        "explicit_per_iteration_synchronize": False,
        "native_stream_check_may_drain_host_queue": True,
        "retained_output_samples": 2,
        "model_weights_loaded": False,
        "performance_verified": False,
    }


def library_identity(path):
    """Record actual bytes; do not turn old manifests/version pins into gates."""
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.suffix != ".so":
        raise ValueError("Select a built libvq2a8_ascendc.so library")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def resident_bank_class(torch):
    try:
        return torch.classes.vq2a8_ascendc.ResidentBank
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "The selected library has no ResidentBank ABI. Rebuild separately with "
            "python tools/build_vq2a8_ascendc.py --soc <actual-Ascend950-SoC> "
            "--build-dir build/vq2a8-ascendc-v4-device-route ; "
            "then retry in a fresh process with that library."
        ) from exc


def synthetic_experts(reduction, *, experts=EXPERTS, columns=OUTPUT_COLUMNS):
    """CPU-only bounded original V1 layout; every expert has distinct metadata."""
    import torch

    from tools.validate_vq2a8_phase4_kernel import synthetic_inputs

    if reduction not in REDUCTIONS or not 1 <= experts <= EXPERTS or columns != OUTPUT_COLUMNS:
        raise ValueError("Only the bounded synthetic validation geometry is supported")
    values = []
    with torch.device("cpu"):
        column = torch.arange(reduction)
        for expert in range(experts):
            # Different tile counts also catch accidental first-expert strides.
            tiles = (1, 3, 7, 4)[expert]
            _, _, _, packed, books, tile_ids = synthetic_inputs(1, columns, reduction, tiles)
            values.append(
                {
                    "packed_indices": packed.roll(expert, dims=0).contiguous(),
                    "codebooks": (books.float() + expert / 8).to(torch.float8_e4m3fn),
                    "codebook_tile_ids": ((tile_ids.int() + expert) % tiles).byte(),
                    "weight_scale": (column.float() % 7 + expert + 1) / 16,
                    "weight_bias": (column.float() % 13 - 6 + expert) / 128,
                    "rht_sign": torch.where((column + expert) % (expert + 3) == 0, -1, 1).to(torch.int8),
                }
            )
    return values


def exact_tensor(actual, expected, label):
    """Explicit validation boundary, including signed-zero/FP8 byte equality."""
    import torch

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{label}: shape/dtype mismatch")
    got = actual.detach().cpu().contiguous().view(torch.uint8)
    want = expected.detach().cpu().contiguous().view(torch.uint8)
    if not torch.equal(got, want):
        raise AssertionError(f"{label}: differs bitwise from the V1 baseline")


def _selected_requests(hidden, selected, spec):
    return [
        (hidden[index : index + 1], {name: value[index] for name, value in zip(PAYLOAD_FIELDS[3:], selected)}, spec)
        for index in range(hidden.shape[0])
    ]


def run_synthetic_checks(device, *, bank_factory, projection, synchronize, stage=None):
    """Run real tensor checks. Injected CPU functions are for UT only, not fallback.

    The production caller supplies ONLY the loaded native bank and V1 op on
    physical NPU. This helper itself does not certify any hardware execution.
    """
    import torch

    from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation

    stage = stage or (lambda _: nullcontext())
    checks = []
    for reduction in REDUCTIONS:
        with stage(f"k{reduction}_bank_upload"):
            hosts = synthetic_experts(reduction)
            payloads = [{name: value.to(device) for name, value in payload.items()} for payload in hosts]
            bank = bank_factory(*([payload[name] for payload in payloads] for name in PAYLOAD_FIELDS))
            spec = SimpleNamespace(columns=reduction, rht_true_columns=reduction, rht_block_size=128)
            preparation = RowwiseVQ2A8Preparation(compact=True, validity=lambda _: None)
        preserved = []
        for iteration, routes in enumerate(VALID_ROUTES):
            with stage(f"k{reduction}_valid_{iteration}"):
                ids = torch.tensor(routes, dtype=torch.int64, device=device)
                with torch.device("cpu"):
                    raw = ((torch.arange(reduction) * (iteration + 3) % 31 - 15) / (16 + iteration)).bfloat16()
                # One token replicated across expert slots, as in B1 decode.
                hidden = raw.reshape(1, reduction).to(device).expand(len(routes), -1).contiguous()
                selected = bank.select(ids)
                synchronize()
                if len(selected) != 4:
                    raise AssertionError("ResidentBank.select must return scale, bias, sign, valid")
                for field, actual in zip(PAYLOAD_FIELDS[3:], selected[:3]):
                    expected = torch.stack([payloads[expert][field] for expert in routes])
                    exact_tensor(actual, expected, f"selected_{field}")
                exact_tensor(selected[3], torch.ones(len(routes), dtype=torch.int32), "select_valid")
                candidate = preparation.many(_selected_requests(hidden, selected[:3], spec))
                baseline = preparation.many(
                    [(hidden[row : row + 1], payloads[expert], spec) for row, expert in enumerate(routes)]
                )
                for row, (actual, expected) in enumerate(zip(candidate, baseline)):
                    for field, left, right in zip(("fp8", "scale", "bias"), actual, expected):
                        exact_tensor(left, right, f"prepared_{row}_{field}")
                prepared = tuple(torch.cat([entry[index] for entry in candidate]).contiguous() for index in range(3))
                output, valid = bank.project(*prepared, ids)
                expected = torch.cat(
                    [
                        projection(*baseline[row], *(payloads[expert][name] for name in PAYLOAD_FIELDS[:3]))
                        for row, expert in enumerate(routes)
                    ]
                )
                synchronize()
                exact_tensor(output, expected, "prepared_projection")
                exact_tensor(valid, torch.ones(len(routes), dtype=torch.int32), "projection_valid")
                if not bool(torch.isfinite(output.float()).all()):
                    raise AssertionError("V1 comparison produced nonfinite output")
                # Reuse inputs/ids with new values to detect stale descriptors.
                rotated_routes = routes[1:] + routes[:1]
                ids.copy_(torch.tensor(rotated_routes, dtype=torch.int64, device=device))
                repeated, repeat_valid = bank.project(*prepared, ids)
                rotated_expected = torch.cat(
                    [
                        projection(*candidate[row], *(payloads[expert][name] for name in PAYLOAD_FIELDS[:3]))
                        for row, expert in enumerate(rotated_routes)
                    ]
                )
                synchronize()
                exact_tensor(repeated, rotated_expected, "changed_routes_same_prepared")
                exact_tensor(repeat_valid, valid, "changed_routes_valid")
                for previous, snapshot in preserved:
                    exact_tensor(previous, snapshot, "previous_output_ownership")
                preserved.append((output, output.detach().cpu().clone()))
                checks.append({"k": reduction, "routes": list(routes), "iteration": iteration, "exact": True})
        with stage(f"k{reduction}_invalid_ids"):
            routes = (-1, EXPERTS, 2**40, 1)
            ids = torch.tensor(routes, dtype=torch.int64, device=device)
            selected = bank.select(ids)
            # Validate projection independently: valid FP8 input, not NaNs from select.
            x = torch.ones((len(routes), reduction), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
            scale = torch.ones(len(routes), dtype=torch.float32, device=device)
            bias = torch.zeros(len(routes), dtype=torch.float32, device=device)
            output, valid = bank.project(x, scale, bias, ids)
            synchronize()
            expected_valid = torch.tensor([0, 0, 0, 1], dtype=torch.int32)
            exact_tensor(selected[3], expected_valid, "invalid_select_flags")
            exact_tensor(valid, expected_valid, "invalid_projection_flags")
            for value in selected[:2]:
                if not bool(torch.isnan(value[:3].cpu()).all()):
                    raise AssertionError("Invalid metadata rows must contain NaN")
            exact_tensor(selected[2][:3], torch.zeros((3, reduction), dtype=torch.int8), "invalid_sign_zero")
            if not bool(torch.isnan(output[:3].float().cpu()).all()):
                raise AssertionError("Invalid projection rows must contain NaN")
            for field, value in zip(PAYLOAD_FIELDS[3:], selected[:3]):
                exact_tensor(value[3], payloads[1][field], f"mixed_valid_{field}")
            expected = projection(x[3:], scale[3:], bias[3:], *(payloads[1][name] for name in PAYLOAD_FIELDS[:3]))
            synchronize()
            exact_tensor(output[3:], expected, "mixed_valid_projection")
            checks.append({"k": reduction, "routes": list(routes), "invalid_ids_safe": True, "exact": True})
        with stage(f"k{reduction}_resident_immutable"):
            for payload, host in zip(payloads, hosts):
                for name in PAYLOAD_FIELDS:
                    exact_tensor(payload[name], host[name], f"resident_{name}")
            for previous, snapshot in preserved:
                exact_tensor(previous, snapshot, "previous_output_ownership")
    return checks


def run_native_contract_checks(device, *, bank_factory, projection, synchronize, stage):
    """Reject malformed metadata and another stream before device submission."""
    import torch

    def rejected(call, expected):
        try:
            call()
        except RuntimeError as exc:
            if expected not in str(exc):
                raise AssertionError(f"Unexpected rejection instead of {expected}: {exc}") from exc
        else:
            raise AssertionError(f"Native bank did not reject {expected}")

    with stage("native_contract_setup"):
        reduction = REDUCTIONS[0]
        payloads = [
            {name: value.to(device) for name, value in payload.items()} for payload in synthetic_experts(reduction)
        ]
        fields = tuple([payload[name] for payload in payloads] for name in PAYLOAD_FIELDS)
        bank = bank_factory(*fields)
        if list(bank.metadata()) != [EXPERTS, OUTPUT_COLUMNS, reduction, EXPERTS * 64]:
            raise AssertionError("ResidentBank metadata differs from its selected synthetic geometry")
        ids = torch.tensor([0, 3], dtype=torch.int64, device=device)
        x = torch.ones((2, reduction), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
        scale = torch.ones(2, dtype=torch.float32, device=device)
        bias = torch.zeros(2, dtype=torch.float32, device=device)
    with stage("native_contract_rejections"):
        rejected(lambda: bank.select(ids.int()), "wrong dtype/rank")
        rejected(lambda: bank.select(ids.reshape(1, 2)), "wrong dtype/rank")
        rejected(lambda: bank.select(torch.empty(0, dtype=torch.int64, device=device)), "1..6 route slots")
        rejected(lambda: bank.select(torch.zeros(7, dtype=torch.int64, device=device)), "1..6 route slots")
        rejected(lambda: bank.select(torch.zeros(4, dtype=torch.int64, device=device)[::2]), "contiguous")
        rejected(lambda: bank.project(x.bfloat16(), scale, bias, ids), "wrong dtype/rank")
        rejected(lambda: bank.project(x, scale[:1], bias, ids), "Resident projection requires")
        rejected(lambda: bank_factory(fields[0][:-1], *fields[1:]), "identical lengths")
    with stage("native_construction_stream"):
        alternate = torch.npu.Stream(device=device)
        with torch.npu.stream(alternate):
            rejected(lambda: bank.select(ids), "construction NPU stream")
            rejected(lambda: bank.project(x, scale, bias, ids), "construction NPU stream")
        # Rejected side-stream use must not poison normal owner-stream calls.
        output, valid = bank.project(x, scale, bias, ids)
        expected = torch.cat(
            [
                projection(
                    x[row : row + 1],
                    scale[row : row + 1],
                    bias[row : row + 1],
                    *(payloads[expert][name] for name in PAYLOAD_FIELDS[:3]),
                )
                for row, expert in enumerate((0, 3))
            ]
        )
        synchronize()
        exact_tensor(output, expected, "owner_stream_after_rejected_other_stream")
        exact_tensor(valid, torch.ones(2, dtype=torch.int32), "owner_stream_valid")
    return {"malformed_metadata_rejected": True, "other_stream_rejected": True, "owner_stream_reuse_exact": True}


def _queue_log_iteration(completed):
    return completed <= QUEUE_EARLY_PROGRESS_ITERATIONS or completed % QUEUE_PROGRESS_INTERVAL == 0


def _queue_step_recorder(phase, completed):
    @contextmanager
    def step(name):
        # Unlike stage_recorder, these markers NEVER query a Tensor or fence.
        fields = {"stage": phase, "iteration": completed, "step": name}
        if _queue_log_iteration(completed):
            emit(CASE, "QUEUE_STEP", phase="BEGIN", **fields)
        try:
            yield
        except BaseException:
            emit(CASE, "QUEUE_STEP", phase="FAIL", **fields)
            raise
        if _queue_log_iteration(completed):
            emit(CASE, "QUEUE_STEP", phase="RETURN", **fields)

    return step


def _queue_lifetime_iteration(bank, template, *, step=nullcontext):
    """Drop Python input references before the next ordinary operator is queued."""
    import torch

    with step("ids_clone"):
        ids = template["ids"].clone()
    with step("resident_select"):
        selected = bank.select(ids)
    with step("prepared_clone"):
        prepared = tuple(value.clone() for value in template["prepared"])
    with step("resident_project"):
        output, project_valid = bank.project(*prepared, ids)
    # The native handler/allocator, not this Python loop, must keep these alive.
    with step("resident_release_inputs"):
        del prepared, ids
    with step("left_clone"):
        left = template["left"].clone()
    with step("right_clone"):
        right = template["right"].clone()
    with step("matmul"):
        product = torch.matmul(left, right)
    with step("matmul_release_inputs"):
        del left, right
    with step("resident_verify"):
        valid = (project_valid == 1).all() & (selected[3] == 1).all()
        valid = valid & torch.isfinite(output.float()).all()
        valid = valid & (output.view(torch.uint8) == template["expected"].view(torch.uint8)).all()
        valid = valid & (product == template["matmul_expected"]).all()
        for actual, expected in zip(selected[:3], template["metadata"]):
            valid = valid & (actual == expected).all()
        del actual, expected
    with step("resident_release_outputs"):
        del selected, project_valid, product
    return output, valid


def _queue_mixed_projection_checks(templates, projection, grouped_projection, grouped_projection_pipeline, step):
    import torch

    valid = None
    for name, call in (
        ("projection", projection),
        ("grouped_projection", grouped_projection),
        ("grouped_projection_pipeline", grouped_projection_pipeline),
    ):
        selected_templates = templates[:1] if name == "projection" else templates
        with step(f"{name}_clone"):
            jobs = [
                (*[value.clone() for value in template["prepared"]], *template["projection_payload"])
                for template in selected_templates
            ]
        with step(name):
            outputs = [call(*jobs[0])] if name == "projection" else call(jobs)
        with step(f"{name}_release_inputs"):
            del jobs
        with step(f"{name}_verify"):
            if len(outputs) != len(selected_templates):
                raise AssertionError(f"Queue lifetime {name} returned the wrong number of outputs")
            for output, template in zip(outputs, selected_templates):
                expected = template["expected"]
                if output.shape != expected.shape or output.dtype != expected.dtype:
                    raise AssertionError(f"Queue lifetime {name} output shape/dtype differs from V1 oracle")
                current = torch.isfinite(output.float()).all()
                current = current & (output.view(torch.uint8) == expected.view(torch.uint8)).all()
                valid = current if valid is None else valid & current
            del output, current
        with step(f"{name}_release_outputs"):
            del outputs
    return valid


def run_queue_lifetime_checks(
    device,
    *,
    bank_factory,
    projection,
    synchronize,
    grouped_projection=None,
    grouped_projection_pipeline=None,
    iterations=QUEUE_DEFAULT_ITERATIONS,
    stage=None,
    progress=None,
    mixed_progress=None,
):
    """Bounded lifetime regression; CPU injection tests orchestration, not an NPU.

    Setup and the end of EACH phase explicitly synchronize. Keep the original
    resident-only phase free of grouped descriptor copies, which are blocking
    in the existing ABI. The additional mixed phase does not replace it.
    ResidentBank's stream check can drain the host task queue. Counts are call
    lower bounds, NOT measured queue-slot indices or a claim of zero waiting.
    """
    import torch

    from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation

    if type(iterations) is not int or not QUEUE_MIN_ITERATIONS <= iterations <= QUEUE_MAX_ITERATIONS:
        raise ValueError(f"Queue lifetime iterations must be {QUEUE_MIN_ITERATIONS}..{QUEUE_MAX_ITERATIONS}")
    if not callable(grouped_projection) or not callable(grouped_projection_pipeline):
        raise ValueError("Queue lifetime requires real grouped and grouped-pipeline entry points; no fallback")
    stage = stage or stage_recorder(CASE, synchronize)
    progress = progress or (
        lambda completed: emit(CASE, "QUEUE_PROGRESS", stage="queue_lifetime_wrap", iterations=completed)
    )
    mixed_progress = mixed_progress or (
        lambda completed: emit(CASE, "QUEUE_PROGRESS", stage="queue_lifetime_mixed", iterations=completed)
    )
    with stage("queue_lifetime_setup"):
        reduction = REDUCTIONS[0]
        payloads = [
            {name: value.to(device) for name, value in payload.items()} for payload in synthetic_experts(reduction)
        ]
        bank = bank_factory(*([payload[name] for payload in payloads] for name in PAYLOAD_FIELDS))
        preparation = RowwiseVQ2A8Preparation(compact=True, validity=lambda _: None)
        spec = SimpleNamespace(columns=reduction, rht_true_columns=reduction, rht_block_size=128)
        templates = []
        for expert, payload in enumerate(payloads):
            with torch.device("cpu"):
                hidden = ((torch.arange(reduction) * (expert + 3) % 31 - 15) / 16).bfloat16().reshape(1, -1)
                left = torch.arange(16, dtype=torch.float32).reshape(1, 16)
                right = torch.eye(16, dtype=torch.float32)
            prepared = tuple(value.contiguous() for value in preparation.many([(hidden.to(device), payload, spec)])[0])
            templates.append(
                {
                    "ids": torch.tensor([expert], dtype=torch.int64, device=device),
                    "prepared": prepared,
                    "projection_payload": tuple(payload[name] for name in PAYLOAD_FIELDS[:3]),
                    "expected": projection(*prepared, *(payload[name] for name in PAYLOAD_FIELDS[:3])),
                    "metadata": tuple(payload[name].unsqueeze(0) for name in PAYLOAD_FIELDS[3:]),
                    "left": left.to(device),
                    "right": right.to(device),
                    "matmul_expected": left.to(device),
                }
            )
        valid = torch.ones((), dtype=torch.bool, device=device)
        preserved = []
    with stage("queue_lifetime_wrap"):
        for iteration in range(iterations):
            template = templates[iteration % len(templates)]
            step = _queue_step_recorder("queue_lifetime_wrap", iteration + 1)
            output, current_valid = _queue_lifetime_iteration(bank, template, step=step)
            valid = valid & current_valid
            if iteration in (0, iterations - 1):
                preserved.append((output, template["expected"]))
            del output, current_valid
            if _queue_log_iteration(iteration + 1) or iteration + 1 == iterations:
                progress(iteration + 1)  # Host counters only; no Tensor formatting/reads.
    with stage("queue_lifetime_mixed"):
        for iteration in range(iterations):
            template = templates[iteration % len(templates)]
            peer = templates[(iteration + 1) % len(templates)]
            step = _queue_step_recorder("queue_lifetime_mixed", iteration + 1)
            output, current_valid = _queue_lifetime_iteration(bank, template, step=step)
            valid = valid & current_valid
            del output, current_valid
            current_valid = _queue_mixed_projection_checks(
                [template, peer], projection, grouped_projection, grouped_projection_pipeline, step
            )
            valid = valid & current_valid
            del current_valid
            if _queue_log_iteration(iteration + 1) or iteration + 1 == iterations:
                mixed_progress(iteration + 1)
    # All device checks were submitted in the loop; read one scalar after its
    # end fence. Preserve only first/last outputs, never every input/handler.
    if not bool(valid.cpu()):
        raise AssertionError("Queue lifetime metadata/projection/matmul validity or bitwise projection check failed")
    for output, expected in preserved:
        exact_tensor(output, expected, "queue_lifetime_preserved_output")
    phases = queue_lifetime_phases(iterations)
    return {
        "scope": "synthetic_mixed_async_handler_lifetime_only",
        "iterations": iterations,
        "minimum_operator_calls": sum(phase["minimum_operator_calls"] for phase in phases),
        "phases": phases,
        "mixed_grouped_jobs": 2,
        "queue_capacity_reference": QUEUE_CAPACITY_REFERENCE,
        "runtime_queue_enabled_or_slots_measured": False,
        "requires_runtime_task_queue_enabled": True,
        "k": reduction,
        "n": OUTPUT_COLUMNS,
        "experts": EXPERTS,
        "routes": 1,
        "explicit_synchronization_stages": ["queue_lifetime_setup", "queue_lifetime_wrap", "queue_lifetime_mixed"],
        "explicit_per_iteration_synchronize": False,
        "native_stream_check_may_drain_host_queue": True,
        "temporary_inputs_dropped_before_matmul": True,
        "retained_output_samples": len(preserved),
        "all_iteration_checks_passed": True,
        "preserved_outputs_bitwise_exact": True,
        "model_weights_loaded": False,
        "full_model_verified": False,
        "performance_verified": False,
    }


def run_case_child(args):
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical_npu):
        raise ValueError("Child device mapping differs from the selected physical NPU")
    faulthandler.enable()
    faulthandler.dump_traceback_later(min(30, max(1, args.timeout_s / 2)), repeat=True)
    sync = lambda: None
    stage = stage_recorder(CASE, lambda: sync())
    try:
        with stage("imports"):
            import torch
            import torch_npu  # noqa: F401

            from tools.validate_vq2a8_ascendc import require_hardware_runtime
            from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device
            from vllm_ascend.quantization.vq2a8_ascendc import (
                grouped_projection,
                grouped_projection_pipeline,
                load_library,
                vq2a8_ascendc,
            )

        with stage("device"):
            require_hardware_runtime()
            torch.set_num_threads(4)
            device = torch.device("npu:0")
            info = _initialize_device(device)
            torch.npu.config.allow_internal_format = False
            sync = torch.npu.synchronize
        with stage("library"):
            identity = library_identity(args.library)
            load_library(args.library)
            bank_factory = resident_bank_class(torch)
            emit(CASE, "INFO", library=identity, device=info, scope=SCOPE)
        with torch.inference_mode():
            checks = run_synthetic_checks(
                device, bank_factory=bank_factory, projection=vq2a8_ascendc, synchronize=sync, stage=stage
            )
            native_contracts = run_native_contract_checks(
                device, bank_factory=bank_factory, projection=vq2a8_ascendc, synchronize=sync, stage=stage
            )
            queue_lifetime = {"enabled": False}
            if args.queue_lifetime:
                queue_lifetime = {
                    "enabled": True,
                    **run_queue_lifetime_checks(
                        device,
                        bank_factory=bank_factory,
                        projection=vq2a8_ascendc,
                        grouped_projection=grouped_projection,
                        grouped_projection_pipeline=grouped_projection_pipeline,
                        synchronize=sync,
                        iterations=args.queue_iterations,
                        stage=stage,
                    ),
                }
        with stage("final_sync"):
            pass
        emit(
            CASE,
            "CASE_PASS",
            scope=SCOPE,
            checks=checks,
            native_contracts=native_contracts,
            queue_lifetime=queue_lifetime,
            library=identity,
            device_execution_verified=True,
            model_weights_loaded=False,
            full_model_verified=False,
            graph_verified=False,
            performance_verified=False,
            timing_valid=False,
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
    plan = {
        "scope": SCOPE,
        "physical_npu": args.physical_npu,
        "command": child_command(args),
        "synthetic_geometry": {"experts": EXPERTS, "n": OUTPUT_COLUMNS, "k": list(REDUCTIONS), "max_routes": 6},
        "queue_lifetime": queue_lifetime_plan(args),
        "model_weights_loaded": False,
        "full_model_verified": False,
        "graph_verified": False,
        "performance_verified": False,
        "timing_valid": False,
        "device_execution_verified": False,
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    if os.name != "posix":
        raise RuntimeError("Physical NPU validation requires Linux; use --plan-only on other platforms")
    directory = Path(tempfile.mkdtemp(prefix="vq2-v4-device-route-")) if args.report_dir is None else args.report_dir
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
            print(f"Device state={state}; --allow-busy only overrides known occupancy", flush=True)
        else:
            if state == "busy":
                print("WARNING: synthetic validation still uses NPU resources and may affect other jobs", flush=True)
            result = run_child(
                child_command(args), child_environment(args), directory / "validation.log", args.timeout_s
            )
            report["result"] = result
            report["status"] = result["status"]
            report["device_execution_verified"] = result["status"] == "PASS"
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
    except Exception as exc:
        report["error"] = str(exc)
        traceback.print_exc()
    finally:
        (directory / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"V4_DEVICE_ROUTE={report['status']} SUMMARY={directory / 'summary.json'}", flush=True)
    return 0 if report["status"] == "PASS" else 130 if report["status"] == "INTERRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
