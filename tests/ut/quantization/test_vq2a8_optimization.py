# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts only: no CANN compile/device execution/performance claim."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tools.vq2a8_optimization_report import graph_captures, numerical_gate, validate_sample
from vllm_ascend.quantization import vq2a8_optimization as opt
from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_activation_fast import (
    BatchedFWHTPreparation,
    PreparationGraph,
    butterfly_reference,
)
from vllm_ascend.quantization.vq2a8_moe import mix_vq2a8_routes, route_vq2a8
from vllm_ascend.quantization.vq2a8_reference import _sylvester_hadamard, deepseek_v4_swiglu_reference


@pytest.mark.parametrize("tokens", [1, 2, 31, 32, 33, 96, 128])
@pytest.mark.parametrize("topk", [1, 6])
def test_route_plan_all_slots_exactly_once_and_unique_expert_tokens(tokens, topk):
    ids = [[(row + slot // 2) % 5 for slot in range(topk)] for row in range(tokens)]
    jobs = opt.route_plan(ids, set(range(5)))
    visited, assignments = [], []
    for expert, rows, sources, destinations in jobs:
        assert 1 <= len(rows) <= 32
        assert len(rows) == len(set(rows))
        assignments.extend((token, expert) for token in rows)
        for source, destination in zip(sources, destinations):
            token, slot = divmod(destination, topk)
            assert rows[source] == token and ids[token][slot] == expert
        visited.extend(destinations)
    assert sorted(visited) == list(range(tokens * topk))
    assert len(assignments) == len(set(assignments))


@pytest.mark.parametrize("ids", [[], [[]], [[1], [1, 2]], [[1.0]], [[-1]], [[7]], [[0] * 7], [[0]] * 129])
def test_route_plan_rejects_unsafe_indices_before_gather(ids):
    with pytest.raises(ValueError):
        opt.route_plan(ids, {0, 1, 2})


@pytest.mark.parametrize("width", [128, 512, 2048, 4096, 8192])
@pytest.mark.parametrize("case", ["random", "zero", "impulse"])
def test_butterfly_equation_matches_sylvester_oracle(width, case):
    generator = torch.Generator().manual_seed(width)
    x = torch.randn(3, width, generator=generator).bfloat16()
    signs = torch.where(torch.arange(width) % 3 == 0, -1, 1).to(torch.int8)
    if case != "random":
        x.zero_()
        if case == "impulse":
            x[:, -1] = 1
    actual = butterfly_reference(x, signs)
    expected = ((x.double() * signs).reshape(-1, 128) @ _sylvester_hadamard(128)).reshape(x.shape).float()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_deferred_preparation_still_scans_mutated_metadata_and_does_not_read_host(monkeypatch):
    flags = []
    prepare = RowwiseVQ2A8Preparation(compact=True, validity=flags.append)
    x = torch.ones(1, 512).bfloat16()
    scale, bias, sign = torch.ones(512), torch.zeros(512), torch.ones(512, dtype=torch.int8)
    original = torch.Tensor.__bool__
    calls = []
    monkeypatch.setattr(torch.Tensor, "__bool__", lambda tensor: calls.append(1) or original(tensor))
    prepare(x, scale, bias, sign, 128)
    sign[3] = 0
    prepare(x, scale, bias, sign, 128)
    assert not calls
    assert original(flags[0]) and not original(flags[1])


def test_deferred_router_never_drops_bias_validation_or_hash_bounds():
    flags = []
    route_vq2a8(
        torch.ones(2, 6), 6, renormalize=True, correction_bias=torch.full((6,), float("nan")), validity=flags.append
    )
    assert any(not bool(flag) for flag in flags)
    with pytest.raises(ValueError, match="out of range"):
        route_vq2a8(
            torch.ones(1, 6),
            6,
            renormalize=True,
            hash_table=torch.zeros((1, 6), dtype=torch.int64),
            input_ids=torch.tensor([-1]),
            validity=flags.append,
        )


@pytest.mark.parametrize("tokens", [1, 2, 33, 96, 128])
@pytest.mark.parametrize("cache_limit", [1, 2, 6])
@pytest.mark.parametrize("preset", ["fast", "batched"])
def test_fast_moe_matches_reference_duplicate_hash_slots_and_split_jobs(monkeypatch, tokens, cache_limit, preset):
    ids = torch.tensor([[(r + slot // 2) % 8 for slot in range(6)] for r in range(tokens)])
    weights = torch.tensor([[0.05, 0.1, 0.15, 0.2, 0.23, 0.27]]).repeat(tokens, 1)
    x = (torch.arange(tokens * 8).reshape(tokens, 8) % 13 / 16).bfloat16()
    monkeypatch.setattr(opt, "route_vq2a8", lambda *a, **kw: (weights, ids))
    loads, jobs_per_call = [], []

    def payload(index):
        loads.append(index)
        return {kind: (dict(index=index, kind=kind), None) for kind in ("gate_up", "down")}

    def project(hidden, p, _):
        if p["kind"] == "gate_up":
            return torch.cat((hidden + p["index"] / 16, hidden - p["index"] / 16), dim=1)
        return hidden * (p["index"] + 1) / 16

    def project_many(requests):
        jobs_per_call.append(len(requests))
        assert all(1 <= r[0].shape[0] <= 32 for r in requests)
        return [project(*r) for r in requests]

    runtime = NS(
        device=torch.device("cpu"),
        root={"gate.weight": torch.ones(8, 8)},
        config=NS(hidden_size=8, top_k=6, renormalize=True, num_shared=1, swiglu_limit=7.0, routed_scale=1.5),
        layer=NS(expert_ids=list(range(8))),
        cache_experts=cache_limit,
        native_experts=0,
        token_chunk=2,
        _get_expert=payload,
        _projections_many=project_many,
        shared=lambda v: v / 16,
    )

    def expert(index, hidden):
        p = {kind: (dict(index=index, kind=kind), None) for kind in ("gate_up", "down")}
        return project(deepseek_v4_swiglu_reference(project(hidden, *p["gate_up"]), 7.0), *p["down"])

    expected = mix_vq2a8_routes(x, weights, ids, expert, routed_scale=1.5, shared=runtime.shared)
    state = opt.FastMoEState(runtime, opt.OptimizationOptions.preset(preset))
    actual = state.forward(runtime, x, None)
    assert torch.equal(actual, expected)
    assert bool(state.valid) and state.stats["route_host_reads"] == 1
    assert max(jobs_per_call) <= cache_limit
    assert len(loads) == state.stats["jobs"] == runtime.native_experts


def test_runtime_presets_retain_graph_owner_across_baseline_switch():
    runtime = NS(root={})
    opt.configure_runtime(runtime, "prepare_graph")
    state, graph = runtime._optimization, runtime._row_preparation
    opt.configure_runtime(runtime, None)
    assert runtime._optimization is None and isinstance(runtime._row_preparation, RowwiseVQ2A8Preparation)
    opt.configure_runtime(runtime, "prepare_graph")
    assert runtime._optimization is state and runtime._row_preparation is graph
    with pytest.raises(ValueError):
        opt.configure_runtime(runtime, "unsafe")


def test_pipeline_can_be_tested_without_accepting_fwht_rounding():
    candidate = opt.OptimizationOptions.preset("pipeline")
    assert candidate.pipeline and candidate.preparation == "rowwise"
    assert not candidate.shared_batch


@pytest.mark.parametrize("reuse", [False, True])
def test_supervisor_commands_preserve_old_build_and_cover_order(tmp_path, reuse):
    from tools.accept_vq2a8_optimizations import PRESETS, commands

    assert PRESETS == opt.PRESETS
    args = NS(
        library=tmp_path / "new.so" if reuse else None,
        build_dir=tmp_path / "candidate",
        soc="Ascend950DT_9574",
        jobs=4,
        model=tmp_path / "weights",
        physical_npu=4,
        cases="10:4,32:32,96:32",
        warmups=2,
        repeats=5,
        presets=",".join(PRESETS),
        profile=True,
    )
    plan = commands(args, tmp_path / "report")
    assert [name for name, _ in plan] == (
        ["environment", "preflight", "performance"] if reuse else ["environment", "build", "preflight", "performance"]
    )
    assert "--profile-optimization" in plan[-1][1]
    assert "--optimization-presets" in plan[-1][1]
    assert not any(word in str(plan) for word in ("pip install", "git reset", "csrc/build", "vq2a8-ascendc-v026"))
    assert not (tmp_path / "report").exists()


def test_fwht_runtime_has_no_cpu_fallback():
    with pytest.raises(ValueError, match="NPU-only"):
        BatchedFWHTPreparation(validity=lambda _: None).many([(torch.ones(1, 512), {}, None)])


def test_graph_refreshes_static_metadata_bounds_cache_and_rejects_other_stream(monkeypatch):
    # Protocol-only fake capture: explicitly not a device-graph execution test.
    context = NS(active=None, current=NS(npu_stream=1, wait_stream=lambda s: None))

    class Graph:
        def replay(self):
            dst, src = self.operation
            dst.copy_(src[0] * src[1])

    class Capture:
        def __init__(self, graph):
            self.graph = graph

        def __enter__(self):
            context.active = self.graph

        def __exit__(self, *args):
            context.active = None

    class Prepare:
        def pack(self, requests):
            return requests

        def check(self, packed):
            pass

        def compute(self, packed):
            out = packed[0] * packed[1]
            if context.active is not None:
                context.active.operation = out, packed
            return (out,)

    monkeypatch.setattr(
        torch,
        "npu",
        NS(
            current_stream=lambda: context.current,
            Stream=lambda: NS(wait_stream=lambda s: None),
            stream=lambda s: nullcontext(),
            synchronize=lambda: None,
            NPUGraph=Graph,
            graph=Capture,
        ),
        raising=False,
    )
    graph = PreparationGraph(Prepare())
    for i in (1, 3, 7):
        actual = graph.many(([1, 1], (torch.full((2, 4), float(i)), torch.full((2, 4), float(i + 2)))))
        assert torch.equal(torch.cat([x[0] for x in actual]), torch.full((2, 4), float(i * (i + 2))))
    assert graph.captures == 1 and graph.replays == 3
    graph.many(([1], (torch.ones(1, 8), torch.ones(1, 8))))
    graph.many(([1], (torch.ones(1, 16), torch.ones(1, 16))))
    graph.many(([2], (torch.ones(2, 4), torch.ones(2, 4))))
    assert len(graph.entries) == 2 and graph.bypasses == 2
    context.current.npu_stream = 2
    with pytest.raises(RuntimeError, match="single owner stream"):
        graph.many(([1], (torch.ones(1, 8), torch.ones(1, 8))))


def test_strict_gate_does_not_hide_rounding_token_or_signed_zero_changes():
    reference = torch.tensor([[1.0, 0.0]])
    assert numerical_gate([1], reference, [1], reference)["accepted"]
    for tokens, value in (
        ([2], reference),
        ([1], reference + 1e-6),
        ([1], torch.tensor([[1.0, -0.0]])),
        ([1], torch.full_like(reference, float("nan"))),
    ):
        assert not numerical_gate([1], reference, tokens, value)["accepted"]


def test_measured_sample_must_not_capture_graph():
    before = {"0": {"graph": {"captures": 1}}}
    after = {"0": {"graph": {"captures": 2}}}
    sample = dict(finite=True, tokens=[1, 2], forwards=2, optimization_before=before, optimization_after=after)
    assert graph_captures(before) == 1
    validate_sample(sample, [1, 2], 2, measured=False)
    with pytest.raises(ValueError, match="capture contaminated"):
        validate_sample(sample, [1, 2], 2, measured=True)


def test_profile_hooks_are_opt_in_and_removed():
    model = torch.nn.Module()
    model.self_attn = torch.nn.Linear(4, 4)
    opt.install_profile_hooks(model, True)
    assert len(model._optimization_profile_handles) == 2
    model.self_attn(torch.ones(1, 4))
    opt.install_profile_hooks(model, False)
    assert not model.self_attn._forward_hooks and not model.self_attn._forward_pre_hooks


def test_pipeline_source_preserves_original_entry_and_balances_slot_fences():
    source = (Path(__file__).resolve().parents[3] / "csrc/vq2a8_ascendc/kernel.cpp").read_text()
    assert "ProjectionKernel<> op" in source and "ProjectionKernel<true> op" in source
    assert "start >= kBuffers * kK" in source
    assert "slot < kBuffers && slot * kK < k_" in source
    assert "Slot(start) * kM * kK" in source and "Slot(start) * kN * kK" in source


@pytest.mark.parametrize("buffers", [1, 2])
@pytest.mark.parametrize("tiles", [1, 2, 3, 4, 16, 32, 512])
def test_pipeline_event_protocol_model_drains_every_slot(buffers, tiles):
    # Simple event ownership model (not the CANN runtime), including odd tile
    # counts and repeated experts. Source binding is checked separately above.
    for _expert in range(3):
        outstanding = [False] * buffers
        for tile in range(tiles):
            slot = tile % buffers
            if tile >= buffers:
                assert outstanding[slot]
                outstanding[slot] = False
            assert not outstanding[slot]
            outstanding[slot] = True  # Cube read acknowledged; producer may reuse
        for slot in range(min(buffers, tiles)):
            assert outstanding[slot]
            outstanding[slot] = False
        assert not any(outstanding)
