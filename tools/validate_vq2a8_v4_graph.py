#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, isolated native V4 MoE graph checks, without loading model weights.

Only the Linux child can certify hardware execution. CPU-injected helpers test
orchestration, never NPUGraph support, model accuracy, serving, or performance.
Capture uses the resident bank's current owner stream; unsupported capture is
a failure, not a reason to change streams, disable validation, or use eager.
"""

from __future__ import annotations

# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import faulthandler
import json
import subprocess
import tempfile
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace

from tools.diagnose_vq2a8_tp1_startup import child_environment, emit, parse_snapshot, run_child, stage_recorder
from tools.validate_vq2a8_v4_device_route import (
    BUILD_DIR,
    EXPERTS,
    OUTPUT_COLUMNS,
    PAYLOAD_FIELDS,
    REDUCTIONS,
    _queue_mixed_projection_checks,
    _selected_requests,
    exact_tensor,
    library_identity,
    resident_bank_class,
    synthetic_experts,
)

CASE = "v4_graph"
SCOPE = "synthetic_v4_native_select_pipeline_and_moe_graph_only"
GRAPH_WARMUPS = 2
ROUTE_COUNTS = (1, 6)
QUEUE_MIN_ITERATIONS = 2049
QUEUE_MAX_ITERATIONS = 8192
QUEUE_PROGRESS_INTERVAL = 256
MOE_WIDTH = 512
SHARED_WIDTH = 32
MEMORY_GROWTH_TOLERANCE_BYTES = 64 * 1024**2
DYNAMIC_CASE_NAMES = (
    "a",
    "same_hidden_ids_b",
    "ids_a_again",
    "same_ids_hidden_b",
    "a_again",
    "invalid_id",
    "valid_after_invalid",
    "invalid_int64_id",
    "valid_after_large_id",
    "nan_hidden",
    "valid_after_nan",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-npu", type=int, default=1)
    parser.add_argument("--library", type=Path, default=BUILD_DIR / "libvq2a8_ascendc.so")
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--launch-blocking", choices=("0",), default="0")
    parser.add_argument("--queue-lifetime", action="store_true", help="also run bounded asynchronous replay pressure")
    parser.add_argument("--queue-iterations", type=int, default=QUEUE_MIN_ITERATIONS)
    parser.add_argument("--report-dir", type=Path, help="new output directory; never overwrite an old receipt")
    parser.add_argument("--plan-only", action="store_true", help="no torch/vLLM import, file write, or device work")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.physical_npu < 0 or not 1 <= args.timeout_s <= 3600:
        parser.error("Require physical-npu >=0 and timeout-s in [1,3600].")
    if not QUEUE_MIN_ITERATIONS <= args.queue_iterations <= QUEUE_MAX_ITERATIONS:
        parser.error(f"queue-iterations must be in [{QUEUE_MIN_ITERATIONS},{QUEUE_MAX_ITERATIONS}].")
    if not args.queue_lifetime and args.queue_iterations != QUEUE_MIN_ITERATIONS:
        parser.error("--queue-iterations requires --queue-lifetime.")
    if args.child and args.plan_only:
        parser.error("--child and --plan-only cannot be combined.")
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
        "0",
    ]
    if args.queue_lifetime:
        command.extend(["--queue-lifetime", "--queue-iterations", str(args.queue_iterations)])
    return command


def memory_observation(device):
    """Read only at explicit test boundaries, never in the replay loop."""
    import torch

    result = {}
    if device.type == "npu":
        result.update(
            allocated_bytes=torch.npu.memory_allocated(device), reserved_bytes=torch.npu.memory_reserved(device)
        )
    proc = Path("/proc/self/status")
    if proc.exists():
        for line in proc.read_text().splitlines():
            if line.startswith("VmRSS:"):
                result["rss_bytes"] = int(line.split()[1]) * 1024
    return result


class CapturedTuple:
    """Small real-NPUGraph probe; graph/input/output owners live until close."""

    def __init__(self, compute, inputs, *, synchronize):
        import torch

        if any(value.device.type != "npu" for value in inputs):
            raise ValueError("CapturedTuple requires native NPU inputs; no CPU fallback.")
        self.backend, self.synchronize, self.compute = torch.npu, synchronize, compute
        self.inputs = tuple(value.clone() for value in inputs)
        self.stream = self.backend.current_stream(inputs[0].device)
        self.graph, self.outputs = None, None
        self.captures = self.replays = 0
        self.failed = False
        started = time.perf_counter()
        for _ in range(GRAPH_WARMUPS):
            compute(*self.inputs)
        synchronize()
        self.warmup_s = time.perf_counter() - started
        started = time.perf_counter()
        graph = self.backend.NPUGraph()
        with self.backend.graph(graph, stream=self.stream):
            self.outputs = tuple(compute(*self.inputs))
        self.graph = graph
        self.captures = 1
        synchronize()
        self.capture_s = time.perf_counter() - started

    def replay(self, *inputs):
        if self.failed or self.graph is None:
            raise RuntimeError("Graph probe is failed or closed; no eager fallback.")
        if len(inputs) != len(self.inputs):
            raise ValueError("Graph input arity changed.")
        if self.backend.current_stream(inputs[0].device).npu_stream != self.stream.npu_stream:
            raise RuntimeError("Graph probe changed its resident owner stream.")
        for actual, static in zip(inputs, self.inputs):
            if (actual.shape, actual.dtype, actual.device) != (static.shape, static.dtype, static.device):
                raise ValueError("Graph input signature changed; no implicit recapture.")
        try:
            for actual, static in zip(inputs, self.inputs):
                static.copy_(actual)
            self.graph.replay()
            self.replays += 1
            # The returned tensors belong to this invocation, not the next replay.
            return tuple(value.clone() for value in self.outputs)
        except BaseException:
            self.failed = True
            raise

    def snapshot(self):
        return {
            "captures": self.captures,
            "replays": self.replays,
            "entries": int(self.graph is not None),
            "pool_count": int(self.graph is not None),
            "pool_policy": "one_backend_default_private_pool_per_graph",
            "static_input_pointers": [value.data_ptr() for value in self.inputs],
            "static_buffer_bytes": sum(value.numel() * value.element_size() for value in self.inputs + self.outputs),
            "warmup_s": self.warmup_s,
            "capture_s": self.capture_s,
        }

    def close(self):
        # If the fence fails keep all owners attached; the supervised child exits.
        self.synchronize()
        self.graph = self.outputs = self.compute = None
        self.inputs = ()


def pipeline_compute(bank, spec):
    """Original rowwise preparation; all validity is an output of this call."""
    import torch

    from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation

    flags = []
    preparation = RowwiseVQ2A8Preparation(compact=True, validity=flags.append)

    def compute(hidden, ids):
        flags.clear()
        selected = bank.select(ids)
        prepared = preparation.many(_selected_requests(hidden.expand(ids.numel(), -1), selected[:3], spec))
        packed = tuple(torch.cat([row[index] for row in prepared]).contiguous() for index in range(3))
        output, valid = bank.project(*packed, ids)
        status = (selected[3] == 1).all() & (valid == 1).all() & torch.isfinite(output).all()
        for flag in flags:
            status = status & flag
        return (*selected, *packed, output, valid, status)

    return compute


def dynamic_inputs(device, reduction, routes):
    """Bounded fixed-shape same-value/different-value cases; no expert matrix."""
    import torch

    if reduction not in REDUCTIONS or routes not in ROUTE_COUNTS:
        raise ValueError("Only bounded graph synthetic shapes are supported.")
    first = (0,) if routes == 1 else (0, 1, 2, 3, 1, 0)
    second = (3,) if routes == 1 else (3, 2, 1, 0, 2, 3)
    hidden = ((torch.arange(reduction, device=device).float() * 3 % 31 - 15) / 16).bfloat16().reshape(1, -1)
    changed = (hidden.float() * -0.75 + 0.125).bfloat16()
    ids_a, ids_b = (torch.tensor(value, dtype=torch.int64, device=device) for value in (first, second))
    invalid = ids_a.clone()
    invalid[0] = -1
    large = ids_a.clone()
    large[0] = 2**40
    return [
        ("a", hidden, ids_a, True),
        ("same_hidden_ids_b", hidden, ids_b, True),
        ("ids_a_again", hidden, ids_a, True),
        ("same_ids_hidden_b", changed, ids_a, True),
        ("a_again", hidden, ids_a, True),
        ("invalid_id", hidden, invalid, False),
        ("valid_after_invalid", hidden, ids_a, True),
        ("invalid_int64_id", hidden, large, False),
        ("valid_after_large_id", hidden, ids_a, True),
        ("nan_hidden", torch.full_like(hidden, float("nan")), ids_a, False),
        ("valid_after_nan", hidden, ids_a, True),
    ]


def check_tuple(actual, expected, label):
    if len(actual) != len(expected):
        raise AssertionError(f"{label}: graph output arity changed")
    for index, (left, right) in enumerate(zip(actual, expected)):
        exact_tensor(left, right, f"{label}_{index}")


def exercise_graph(compute, cases, *, synchronize, capture_factory=CapturedTuple, select_only=False):
    """Injected CPU capture is orchestration coverage, never a hardware receipt."""
    import torch

    first = cases[0]
    graph = capture_factory(compute, (first[1], first[2]), synchronize=synchronize)
    before = graph.snapshot()
    preserved = None
    checks, changed_outputs = [], []
    for label, hidden, ids, valid in cases:
        actual = graph.replay(hidden, ids)
        expected = tuple(compute(hidden, ids))
        synchronize()
        check_tuple(actual, expected, label)
        wanted = bool((ids >= 0).all() & (ids < EXPERTS).all()) if select_only else valid
        status = (actual[3] == 1).all() if select_only else actual[-1]
        if bool(status) != wanted:
            raise AssertionError(f"{label}: replay validity did not refresh")
        if preserved is not None:
            check_tuple(preserved[0], preserved[1], "previous_output_ownership")
        if label == "a":
            preserved = (actual, tuple(value.detach().cpu().clone() for value in actual))
        if label in ("a", "same_hidden_ids_b", "same_ids_hidden_b"):
            index = 0 if select_only else 7
            changed_outputs.append(actual[index].detach().cpu().contiguous().view(torch.uint8).clone())
        checks.append({"case": label, "bit_exact": True, "valid": wanted})
    if torch.equal(changed_outputs[0], changed_outputs[1]):
        raise AssertionError("Changing route IDs did not change graph output; stale native capture suspected")
    if not select_only and torch.equal(changed_outputs[0], changed_outputs[2]):
        raise AssertionError("Changing hidden did not change graph output; stale native capture suspected")
    after = graph.snapshot()
    if after["captures"] != 1 or after["entries"] != 1 or after["replays"] != len(cases):
        raise AssertionError("Expected one captured graph and one replay per dynamic case")
    if before["static_input_pointers"] != after["static_input_pointers"]:
        raise AssertionError("Static graph input addresses changed")
    graph.close()
    return {"checks": checks, "graph": after, "same_static_addresses": True, "previous_outputs_preserved": True}


def run_synthetic_checks(device, *, bank_factory, synchronize, stage=None, capture_factory=CapturedTuple):
    stage = stage or (lambda _: nullcontext())
    checks = []
    for reduction in REDUCTIONS:
        with stage(f"k{reduction}_bank_upload"):
            payloads = [
                {key: value.to(device) for key, value in payload.items()} for payload in synthetic_experts(reduction)
            ]
            bank = bank_factory(*([payload[key] for payload in payloads] for key in PAYLOAD_FIELDS))
            spec = SimpleNamespace(columns=reduction, rht_true_columns=reduction, rht_block_size=128)
        for routes in ROUTE_COUNTS:
            cases = dynamic_inputs(device, reduction, routes)
            for kind, compute in (
                ("select", lambda hidden, ids, owner=bank: owner.select(ids)),
                ("pipeline", pipeline_compute(bank, spec)),
            ):
                with stage(f"k{reduction}_routes{routes}_{kind}_graph"):
                    result = exercise_graph(
                        compute,
                        cases,
                        synchronize=synchronize,
                        capture_factory=capture_factory,
                        select_only=kind == "select",
                    )
                checks.append({"k": reduction, "n": OUTPUT_COLUMNS, "routes": routes, "kind": kind, **result})
        with stage(f"k{reduction}_teardown"):
            del bank, payloads, compute, cases
    return checks


def make_moe_runtime(device, *, bank_factory, hash_route, top_k):
    """Four packed slots, sparse eight-ID router, H=512; never real weights."""
    import torch

    from vllm_ascend.quantization.vq2a8_moe import VQ2TP1MoE
    from vllm_ascend.quantization.vq2a8_optimization import configure_runtime

    if (hash_route, top_k) not in ((True, 1), (True, 6), (False, 1)):
        raise ValueError("Only bounded hash top-k 1/6 and non-hash top-k 1 are supported.")
    banks, owners = {}, []
    expert_ids = (0, 2, 5, 7)
    cache = {expert: {} for expert in expert_ids}
    for kind, columns in (("gate_up", 2 * MOE_WIDTH), ("down", MOE_WIDTH)):
        payloads = synthetic_experts(MOE_WIDTH)
        for payload in payloads:
            payload["packed_indices"] = payload["packed_indices"].repeat(columns // OUTPUT_COLUMNS, 1)
            payload["codebooks"] = payload["codebooks"].repeat(1, columns // OUTPUT_COLUMNS, 1, 1)
        payloads = [{key: value.to(device) for key, value in payload.items()} for payload in payloads]
        spec = SimpleNamespace(rows=columns, columns=MOE_WIDTH, rht_true_columns=MOE_WIDTH, rht_block_size=128)
        banks[kind] = (bank_factory(*([payload[key] for payload in payloads] for key in PAYLOAD_FIELDS)), spec)
        for expert, payload in zip(expert_ids, payloads):
            cache[expert][kind] = (payload, spec)
        owners.extend(payloads)
    root = {"gate.weight": torch.zeros((8, MOE_WIDTH), dtype=torch.float32, device=device)}
    for name, shape in (
        ("w1", (SHARED_WIDTH, MOE_WIDTH)),
        ("w3", (SHARED_WIDTH, MOE_WIDTH)),
        ("w2", (MOE_WIDTH, SHARED_WIDTH)),
    ):
        root[f"shared_experts.{name}.weight"] = (
            ((torch.arange(shape[0] * shape[1], device=device).float() % 17 - 8) / 4096).reshape(shape).bfloat16()
        )
    if hash_route:
        root["gate.tid2eid"] = torch.tensor(
            [[7, 2, 2, 7, 0, 5][:top_k], [5, 7, 0, 2, 7, 5][:top_k], [1] * top_k],
            dtype=torch.int64,
            device=device,
        )
    else:
        root["gate.weight"][0].fill_(0.125)
        root["gate.weight"][7].fill_(-0.125)
        root["gate.bias"] = torch.tensor(
            [0, -1000, -1000, -1000, -1000, -1000, -1000, 0], dtype=torch.float32, device=device
        )
    runtime = SimpleNamespace(
        execution_policy="ascendc_v4",
        device=device,
        root=root,
        config=SimpleNamespace(
            hidden_size=MOE_WIDTH,
            num_experts=8,
            top_k=top_k,
            renormalize=True,
            num_shared=1,
            swiglu_limit=7.0,
            routed_scale=1.5,
        ),
        layer=SimpleNamespace(expert_ids=expert_ids),
        cache_experts=4,
        token_chunk=2,
        native_calls=0,
        native_rows=0,
        native_launches=0,
        native_experts=0,
        projection_rows=0,
        prepare_batches=0,
        _require_ready=lambda: None,
        _device_route_banks={"lookup": torch.tensor([0, -1, 1, -1, -1, 2, -1, 3], device=device), **banks},
        _synthetic_payload_owners=owners,
        _cache=cache,
    )
    # Exercise the original shared gate/up/SwiGLU/down implementation, not a
    # placeholder identity branch. Only its synthetic width is reduced.
    runtime.shared = MethodType(VQ2TP1MoE.shared, runtime)
    configure_runtime(runtime, "device_route_decode")
    return runtime


def run_moe_checks(device, *, bank_factory, synchronize, stage=None, lifetime_iterations=0):
    import torch

    stage = stage or (lambda _: nullcontext())
    if lifetime_iterations and not QUEUE_MIN_ITERATIONS <= lifetime_iterations <= QUEUE_MAX_ITERATIONS:
        raise ValueError("MoE lifetime iterations must be bounded in [2049,8192].")
    checks = []
    for hash_route, top_k in ((True, 1), (True, 6), (False, 1)):
        with stage(f"moe_hash{int(hash_route)}_topk{top_k}"):
            runtime = make_moe_runtime(device, bank_factory=bank_factory, hash_route=hash_route, top_k=top_k)
            state = runtime._optimization
            eager_banks = runtime._device_route_banks
            # Exercise the actual production metadata-only bank construction,
            # private capture stream and caller-stream input/output bridges.
            state.prepare_graph(runtime)
            hidden = torch.full((1, MOE_WIDTH), 0.125, dtype=torch.bfloat16, device=device)
            token = torch.zeros(1, dtype=torch.int64, device=device)
            cases = [
                ("a", hidden, token, True),
                ("b", -hidden, token + int(hash_route), True),
                ("a_again", hidden, token, True),
                ("nan", torch.full_like(hidden, float("nan")), token, False),
                ("valid_after_nan", hidden, token, True),
            ]
            if hash_route:
                cases[2:2] = [
                    ("same_hidden_token_b", hidden, token + 1, True),
                    ("same_token_hidden_b", -hidden, token, True),
                ]
                cases.extend(
                    [
                        ("invalid_token", hidden, token - 1, False),
                        ("recover_token", hidden, token, True),
                        ("missing_slot", hidden, token + 2, False),
                        ("recover_slot", hidden, token, True),
                    ]
                )
            preserved = None
            rows = []
            templates = []
            proof_outputs = {}
            for label, value, ids, wanted in cases:
                state.valid = None
                actual_output = state.forward_graph(runtime, value, ids)
                actual = (actual_output, state.valid)
                state.valid = None
                expected_output = state.forward(runtime, value, ids)
                expected = (expected_output, state.valid)
                synchronize()
                check_tuple(actual, expected, f"moe_{label}")
                if bool(actual[1]) != wanted:
                    raise AssertionError(f"MoE {label}: replay validity did not refresh")
                if preserved is not None:
                    check_tuple(preserved[0], preserved[1], "moe_previous_output")
                else:
                    preserved = (actual, tuple(value.detach().cpu().clone() for value in actual))
                if label in ("a", "b"):
                    templates.append((value, ids, expected_output))
                if label in ("a", "same_hidden_token_b", "same_token_hidden_b"):
                    proof_outputs[label] = expected_output.detach().cpu().contiguous().view(torch.uint8).clone()
                rows.append({"case": label, "bit_exact": True, "valid": wanted})
            if torch.equal(templates[0][2].view(torch.uint8), templates[1][2].view(torch.uint8)):
                raise AssertionError("MoE A/B input change did not change output; stale graph suspected")
            if hash_route and any(
                torch.equal(proof_outputs["a"], proof_outputs[name])
                for name in ("same_hidden_token_b", "same_token_hidden_b")
            ):
                raise AssertionError("MoE token-only or hidden-only change did not change the eager oracle output")
            report = state.graph_snapshot()
            if report["captures"] != 1 or report["replays"] != len(cases) or report["stream_bridges"] != len(cases):
                raise AssertionError("MoE graph recaptured or lacked real replay coverage")
            if runtime._device_route_banks is not eager_banks or report["graph_payload_copy_bytes"] != 0:
                raise AssertionError("MoE graph replaced the eager banks or copied resident payloads")
            lifetime = {"enabled": False}
            if lifetime_iterations and hash_route and top_k == 6:
                lifetime = {
                    "enabled": True,
                    **exercise_moe_lifetime(
                        state, runtime, templates, synchronize=synchronize, iterations=lifetime_iterations, stage=stage
                    ),
                }
            state.close_graph()
            checks.append(
                {
                    "hash_route": hash_route,
                    "top_k": top_k,
                    "hidden": MOE_WIDTH,
                    "checks": rows,
                    "graph": report,
                    "production_stream_bridge_verified": True,
                    "shared_expert_arithmetic": "original_gate_up_swiglu_down",
                    "queue_lifetime": lifetime,
                }
            )
            del state, runtime, actual, expected, preserved, eager_banks
    return checks


def exercise_moe_lifetime(state, runtime, templates, *, synchronize, iterations, stage, observe=memory_observation):
    """Stress actual production stream bridges, early input release and reuse."""
    import torch

    if type(iterations) is not int or not QUEUE_MIN_ITERATIONS <= iterations <= QUEUE_MAX_ITERATIONS:
        raise ValueError("MoE lifetime iterations must be bounded in [2049,8192].")
    before = state.graph_snapshot()
    valid = torch.ones((), dtype=torch.bool, device=runtime.device)
    left = torch.arange(16, dtype=torch.float32, device=runtime.device).reshape(1, 16)
    right = torch.eye(16, dtype=torch.float32, device=runtime.device)
    synchronize()
    memory_before = observe(runtime.device)
    preserved = []
    with stage("moe_production_bridge_lifetime"):
        for iteration in range(iterations):
            hidden, ids, expected = templates[iteration % 2]
            hidden, ids = hidden.clone(), ids.clone()
            state.valid = None
            output = state.forward_graph(runtime, hidden, ids)
            del hidden, ids
            lhs, rhs = left.clone(), right.clone()
            product = torch.matmul(lhs, rhs)
            del lhs, rhs
            valid.logical_and_(state.valid & (output.view(torch.uint8) == expected.view(torch.uint8)).all())
            valid.logical_and_((product == left).all())
            if iteration in (0, iterations - 1):
                preserved.append((output, expected))
            del output, product
            if (iteration + 1) % QUEUE_PROGRESS_INTERVAL == 0 or iteration + 1 == iterations:
                emit(CASE, "QUEUE_PROGRESS", stage="moe_production_bridge_lifetime", iterations=iteration + 1)
    synchronize()
    memory_after = observe(runtime.device)
    if not bool(valid.cpu()):
        raise AssertionError("Production MoE bridge lifetime changed outputs or validity")
    for output, expected in preserved:
        exact_tensor(output, expected, "moe_bridge_preserved_output")
    after = state.graph_snapshot()
    if (
        after["captures"] != before["captures"]
        or after["entries"] != 1
        or after["replays"] - before["replays"] != iterations
        or after["stream_bridges"] - before["stream_bridges"] != iterations
    ):
        raise AssertionError("Production MoE bridge lifetime recaptured or missed replay/stream work")
    for key in ("allocated_bytes", "reserved_bytes", "rss_bytes"):
        if (
            key in memory_before
            and key in memory_after
            and memory_after[key] - memory_before[key] > MEMORY_GROWTH_TOLERANCE_BYTES
        ):
            raise AssertionError(f"MoE bridge lifetime {key} grew beyond the fixed 64 MiB probe tolerance")
    return {
        "iterations": iterations,
        "graph_before": before,
        "graph_after": after,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "memory_growth_tolerance_bytes": MEMORY_GROWTH_TOLERANCE_BYTES,
        "explicit_per_iteration_synchronize": False,
        "retained_output_samples": len(preserved),
        "temporary_inputs_dropped_before_matmul": True,
        "all_iteration_checks_passed": True,
    }


def run_queue_lifetime_checks(
    device,
    *,
    bank_factory,
    projection,
    grouped_projection,
    grouped_projection_pipeline,
    synchronize,
    iterations=QUEUE_MIN_ITERATIONS,
    stage=None,
    capture_factory=CapturedTuple,
    observe=memory_observation,
):
    """Two bounded phases: replay+eager, then replay+mixed grouped eager ops.

    There is no per-iteration fence or tensor host read. Original grouped ABI may
    block for descriptor H2D; the first phase deliberately excludes that path.
    Replay counts are NOT actual operator submission/queue-slot measurements.
    """
    import torch

    if type(iterations) is not int or not QUEUE_MIN_ITERATIONS <= iterations <= QUEUE_MAX_ITERATIONS:
        raise ValueError("Graph lifetime iterations must be bounded in [2049,8192].")
    if not all(callable(value) for value in (projection, grouped_projection, grouped_projection_pipeline)):
        raise ValueError("Graph lifetime needs all real eager projection entries; no fallback.")
    stage = stage or (lambda _: nullcontext())
    with stage("graph_lifetime_setup"):
        payloads = [{key: value.to(device) for key, value in payload.items()} for payload in synthetic_experts(512)]
        bank = bank_factory(*([payload[key] for payload in payloads] for key in PAYLOAD_FIELDS))
        compute = pipeline_compute(bank, SimpleNamespace(columns=512, rht_true_columns=512, rht_block_size=128))
        cases = dynamic_inputs(device, 512, 1)
        templates = []
        for index in (0, 1):
            _, hidden, ids, _ = cases[index]
            expected = tuple(compute(hidden, ids))
            payload = payloads[0 if index == 0 else 3]
            templates.append(
                {
                    "hidden": hidden,
                    "ids": ids,
                    "expected": expected[7],
                    "prepared": expected[4:7],
                    "projection_payload": tuple(payload[key] for key in PAYLOAD_FIELDS[:3]),
                }
            )
        graph = capture_factory(compute, (templates[0]["hidden"], templates[0]["ids"]), synchronize=synchronize)
        left = torch.arange(16, dtype=torch.float32, device=device).reshape(1, 16)
        right = torch.eye(16, dtype=torch.float32, device=device)
        valid = torch.ones((), dtype=torch.bool, device=device)
        preserved = []
    synchronize()
    memory = [observe(device)]
    baseline = graph.snapshot()
    for phase in ("graph_lifetime_replay_eager", "graph_lifetime_mixed"):
        with stage(phase):
            for iteration in range(iterations):
                template = templates[iteration % 2]
                hidden, ids = template["hidden"].clone(), template["ids"].clone()
                result = graph.replay(hidden, ids)
                del hidden, ids
                lhs, rhs = left.clone(), right.clone()
                product = torch.matmul(lhs, rhs)
                del lhs, rhs
                valid.logical_and_(
                    result[-1] & (result[7].view(torch.uint8) == template["expected"].view(torch.uint8)).all()
                )
                valid.logical_and_((product == left).all())
                if phase == "graph_lifetime_mixed":
                    valid.logical_and_(
                        _queue_mixed_projection_checks(
                            templates, projection, grouped_projection, grouped_projection_pipeline, nullcontext
                        )
                    )
                if iteration in (0, iterations - 1):
                    preserved.append((result[7], template["expected"]))
                del result, product
                if (iteration + 1) % QUEUE_PROGRESS_INTERVAL == 0 or iteration + 1 == iterations:
                    emit(CASE, "QUEUE_PROGRESS", stage=phase, iterations=iteration + 1)
        synchronize()
        memory.append(observe(device))
    if not bool(valid.cpu()):
        raise AssertionError("Graph lifetime replay/eager/grouped result or validity failed")
    for actual, expected in preserved:
        exact_tensor(actual, expected, "graph_lifetime_preserved_output")
    after = graph.snapshot()
    if after["captures"] != baseline["captures"] or after["entries"] != 1 or after["replays"] != 2 * iterations:
        raise AssertionError("Graph lifetime recaptured, leaked entries or missed replay work")
    for key in ("allocated_bytes", "reserved_bytes", "rss_bytes"):
        # Allocator warmup can grow the first phase. Bound growth of the second
        # phase explicitly; this is not a proof of arbitrary service longevity.
        if key in memory[1] and key in memory[2] and memory[2][key] - memory[1][key] > MEMORY_GROWTH_TOLERANCE_BYTES:
            raise AssertionError(f"Graph lifetime {key} grew beyond the fixed 64 MiB probe tolerance")
    graph.close()
    return {
        "iterations_per_phase": iterations,
        "phases": 2,
        "graph": after,
        "memory": memory,
        "memory_growth_tolerance_bytes": MEMORY_GROWTH_TOLERANCE_BYTES,
        "explicit_per_iteration_synchronize": False,
        "grouped_descriptor_h2d_may_block": True,
        "replays_are_not_queue_slot_measurements": True,
        "retained_output_samples": len(preserved),
        "all_iteration_checks_passed": True,
        "model_weights_loaded": False,
        "performance_verified": False,
    }


def run_case_child(args):
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != str(args.physical_npu):
        raise ValueError("Child device mapping differs from selected physical NPU.")
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
            # Select/project banks are constructed on the SAME managed stream
            # used to capture them. Never capture the default stream or disable
            # the native bank's construction-stream guard to make a test pass.
            owner_stream = torch.npu.Stream(device=device)
            owner_stream.wait_stream(torch.npu.current_stream(device))
            with torch.npu.stream(owner_stream):
                checks = run_synthetic_checks(device, bank_factory=bank_factory, synchronize=sync, stage=stage)
            moe = run_moe_checks(
                device,
                bank_factory=bank_factory,
                synchronize=sync,
                stage=stage,
                lifetime_iterations=args.queue_iterations if args.queue_lifetime else 0,
            )
            lifetime = {"enabled": False}
            if args.queue_lifetime:
                with torch.npu.stream(owner_stream):
                    lifetime = {
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
            if library_identity(args.library) != identity:
                raise ValueError("Native library changed during graph probe.")
        emit(
            CASE,
            "CASE_PASS",
            scope=SCOPE,
            checks=checks,
            moe_checks=moe,
            queue_lifetime=lifetime,
            library=identity,
            device_execution_verified=True,
            graph_functional_verified=True,
            full_model_graph_verified=False,
            model_weights_loaded=False,
            full_model_verified=False,
            serving_verified=False,
            performance_verified=False,
            timing_valid=False,
        )
        return 0
    except Exception as exc:
        traceback.print_exc()
        emit(CASE, "CASE_FAIL", error=str(exc), device_execution_verified=False, graph_functional_verified=False)
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()


def plan(args):
    return {
        "scope": SCOPE,
        "physical_npu": args.physical_npu,
        "command": child_command(args),
        "synthetic_geometry": {
            "experts": EXPERTS,
            "n": OUTPUT_COLUMNS,
            "k": list(REDUCTIONS),
            "route_counts": list(ROUTE_COUNTS),
            "moe_hidden": MOE_WIDTH,
        },
        "queue_lifetime": {
            "enabled": args.queue_lifetime,
            "iterations_per_phase": args.queue_iterations if args.queue_lifetime else 0,
        },
        "device_execution_verified": False,
        "graph_functional_verified": False,
        "model_weights_loaded": False,
        "full_model_verified": False,
        "full_model_graph_verified": False,
        "serving_verified": False,
        "performance_verified": False,
        "timing_valid": False,
    }


def validate_receipt(result, *, require_lifetime=False, iterations=QUEUE_MIN_ITERATIONS):
    receipts = [
        event for event in result.get("events", []) if event.get("case") == CASE and event.get("event") == "CASE_PASS"
    ]
    if result.get("status") != "PASS" or result.get("exit_code") != 0 or len(receipts) != 1:
        raise ValueError("Graph child did not finish with exactly one matching receipt.")
    receipt = receipts[0]
    if (
        receipt.get("scope") != SCOPE
        or receipt.get("device_execution_verified") is not True
        or receipt.get("graph_functional_verified") is not True
    ):
        raise ValueError("Graph child lacks actual native graph verification.")
    excluded = (
        "model_weights_loaded",
        "full_model_verified",
        "full_model_graph_verified",
        "serving_verified",
        "performance_verified",
        "timing_valid",
    )
    if any(receipt.get(key) is not False for key in excluded):
        raise ValueError("Synthetic graph receipt overstates its model/serving/performance scope.")
    low = receipt.get("checks", [])
    expected = {(k, n, kind) for k in REDUCTIONS for n in ROUTE_COUNTS for kind in ("select", "pipeline")}
    if len(low) != len(expected) or {(row.get("k"), row.get("routes"), row.get("kind")) for row in low} != expected:
        raise ValueError("Graph child lacks complete select/pipeline geometry coverage.")
    for row in low:
        graph, checks = row.get("graph", {}), row.get("checks", [])
        invalid = {"invalid_id", "invalid_int64_id"}
        if row["kind"] == "pipeline":
            invalid.add("nan_hidden")
        if (
            graph.get("captures") != 1
            or graph.get("entries") != 1
            or graph.get("replays") != len(DYNAMIC_CASE_NAMES)
            or row.get("same_static_addresses") is not True
            or row.get("previous_outputs_preserved") is not True
            or [check.get("case") for check in checks] != list(DYNAMIC_CASE_NAMES)
            or any(
                check.get("bit_exact") is not True or check.get("valid") is not (check.get("case") not in invalid)
                for check in checks
            )
        ):
            raise ValueError("Graph child lacks dynamic bit-exact/static-address/ownership evidence.")
    moe = receipt.get("moe_checks", [])
    if len(moe) != 3 or {(row.get("hash_route"), row.get("top_k")) for row in moe} != {
        (True, 1),
        (True, 6),
        (False, 1),
    }:
        raise ValueError("Graph child lacks full-MoE route coverage.")
    for row in moe:
        graph, checks = row.get("graph", {}), row.get("checks", [])
        invalid = {"nan", "invalid_token", "missing_slot"}
        names = ["a", "b", "a_again", "nan", "valid_after_nan"]
        if row["hash_route"]:
            names[2:2] = ["same_hidden_token_b", "same_token_hidden_b"]
            names += ["invalid_token", "recover_token", "missing_slot", "recover_slot"]
        if (
            graph.get("captures") != 1
            or graph.get("entries") != 1
            or graph.get("replays") != len(names)
            or graph.get("stream_bridges") != len(names)
            or graph.get("graph_payload_copy_bytes") != 0
            or row.get("production_stream_bridge_verified") is not True
            or [check.get("case") for check in checks] != names
            or any(
                check.get("bit_exact") is not True or check.get("valid") is not (check.get("case") not in invalid)
                for check in checks
            )
        ):
            raise ValueError("Graph child lacks production MoE replay/stream/validity evidence.")
        if require_lifetime and row["hash_route"] and row["top_k"] == 6:
            lifetime = row.get("queue_lifetime", {})
            before, after = lifetime.get("graph_before", {}), lifetime.get("graph_after", {})
            if (
                lifetime.get("enabled") is not True
                or lifetime.get("iterations") != iterations
                or lifetime.get("all_iteration_checks_passed") is not True
                or lifetime.get("explicit_per_iteration_synchronize") is not False
                or after.get("captures") != before.get("captures")
                or after.get("replays", -1) - before.get("replays", -1) != iterations
                or after.get("stream_bridges", -1) - before.get("stream_bridges", -1) != iterations
            ):
                raise ValueError("Requested production MoE bridge lifetime evidence is missing.")
    if require_lifetime:
        lifetime = receipt.get("queue_lifetime", {})
        graph = lifetime.get("graph", {})
        if (
            lifetime.get("enabled") is not True
            or lifetime.get("iterations_per_phase") != iterations
            or lifetime.get("phases") != 2
            or lifetime.get("all_iteration_checks_passed") is not True
            or lifetime.get("explicit_per_iteration_synchronize") is not False
            or graph.get("captures") != 1
            or graph.get("entries") != 1
            or graph.get("replays") != 2 * iterations
        ):
            raise ValueError("Requested native pipeline graph lifetime evidence is missing.")
    return receipt


def main(argv=None):
    args = parse_args(argv)
    if args.child:
        return run_case_child(args)
    report = plan(args)
    if args.plan_only:
        print(json.dumps(report, indent=2))
        return 0
    if os.name != "posix":
        raise RuntimeError("Physical NPU validation requires Linux; use --plan-only on other platforms.")
    directory = Path(tempfile.mkdtemp(prefix="vq2-v4-graph-")) if args.report_dir is None else args.report_dir
    if args.report_dir is not None:
        directory.mkdir(parents=True, exist_ok=False)
    report["status"] = "FAIL"
    try:
        snapshot = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=True, timeout=20)
        (directory / "npu.log").write_text(snapshot.stdout + snapshot.stderr, encoding="utf-8")
        state = parse_snapshot(snapshot.stdout, args.physical_npu)
        report["device_state"] = state
        if state != "idle":
            report["status"] = "BLOCKED"
            print(f"Graph probe device state={state}; no jobs stopped and no device work started.", flush=True)
        else:
            result = run_child(
                child_command(args), child_environment(args), directory / "validation.log", args.timeout_s
            )
            report["result"] = result
            report["status"] = result["status"]
            validate_receipt(result, require_lifetime=args.queue_lifetime, iterations=args.queue_iterations)
            report.update(status="PASS", device_execution_verified=True, graph_functional_verified=True)
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
    except Exception as exc:
        if report["status"] == "PASS":
            report["status"] = "FAIL"
        report["error"] = str(exc)
        traceback.print_exc()
    finally:
        (directory / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"V4_GRAPH={report['status']} SUMMARY={directory / 'summary.json'}", flush=True)
    return 0 if report["status"] == "PASS" else 130 if report["status"] == "INTERRUPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
