# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tools.validate_vq2a8_tp1_acceptance import format_compact_summary, summarize_log
from vllm_ascend.quantization import vq2a8_execution as execution
from vllm_ascend.quantization.vq2a8_offline import OfflineMoEOwner, offline_engine_options
from vllm_ascend.quantization.vq2a8_reference import prepare_repacked_vq2a8_activation_reference
from vllm_ascend.quantization.vq2a8_runtime import VQ2TP1Layer


def layer_header(index, count):
    return VQ2TP1Layer(
        layer_index=index,
        expert_ids=tuple(range(count)),
        tensor_path=Path("/unused"),
        metadata_path=Path("/unused"),
        tensor_sha256="",
        metadata_sha256="",
        specs={},
        tensor_shapes={
            f"{kind}_{field}": (count, 4) for kind in ("gate_up", "down") for field in execution.VQ2_TP1_TORCH_DTYPES
        },
    )


def test_plan_rounds_each_tensor_and_accounts_single_expert_hash_layers():
    unit = 12 * 512
    layers = [layer_header(0, 1), layer_header(3, 256)]
    plan = execution.packed_cache_plan(layers, unit * 18)
    assert plan["layer_limits"] == {0: 1, 3: 17}
    assert plan["planned_bytes"] == unit * 18
    assert plan["full_packed_bytes"] == unit * 257
    assert not plan["all_experts_fit"]
    full = execution.packed_cache_plan(layers, unit * 300)
    assert full["all_experts_fit"] and full["planned_bytes"] == unit * 257
    limited = execution.packed_cache_plan(layers, unit * 300, expert_limit=8)
    assert limited["layer_limits"] == {0: 1, 3: 8}


@pytest.mark.parametrize("budget", [0, -1, 1.5, True, 12 * 512 - 1])
def test_invalid_or_too_small_budget_fails_before_expert_allocation(budget):
    with pytest.raises(ValueError):
        execution.packed_cache_plan([layer_header(0, 1)], budget)


def test_budget_reserves_headroom_and_reuses_allocator_cache_only_once(monkeypatch):
    gib = execution.GIB
    monkeypatch.setattr(execution, "synchronize_execution", lambda device: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (60 * gib, 100 * gib))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 20 * gib)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 30 * gib)
    plan = execution.device_cache_budget("cuda:0")
    assert plan["budget_bytes"] == 54 * gib  # min(60+10,90-20) -16
    assert execution.device_cache_budget("cuda:0", budget_gib=30)["budget_bytes"] == 30 * gib
    with pytest.raises(ValueError, match="exceeds"):
        execution.device_cache_budget("cuda:0", budget_gib=55)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reserve_gib": 0},
        {"budget_gib": -1},
        {"budget_gib": float("nan")},
        {"memory_fraction": 2},
        {"reserve_gib": True},
    ],
)
def test_invalid_device_budget_options_rejected_without_device_access(kwargs):
    with pytest.raises(ValueError):
        execution.device_cache_budget("cuda:0", **kwargs)


def cache_only_runtime(limit=3):
    runtime = execution.CachedVQ2TP1MoE.__new__(execution.CachedVQ2TP1MoE)
    runtime.device = torch.device("cpu")
    runtime.layer_index = 0
    runtime.layer = NS(expert_ids=(0, 1, 2))
    runtime._cache = OrderedDict()
    runtime.cache_experts = limit
    runtime.cache_hits = runtime.cache_loads = runtime.cache_peak_bytes = 0
    runtime._resident_bytes = runtime.evictions = 0
    runtime.progress = False
    runtime.verbose_experts = False
    runtime.timing = dict.fromkeys(
        ("host_load_validate_s", "host_read_s", "host_validate_s", "h2d_s", "prepare_s", "packed_projection_s"), 0.0
    )
    runtime.prepare_batches = runtime.projection_rows = 0
    return runtime


def test_expert_verbosity_defaults_to_off_even_with_layer_progress(monkeypatch):
    monkeypatch.setattr(execution.VQ2TP1MoE, "__init__", lambda *args, **kwargs: None)
    runtime = execution.CachedVQ2TP1MoE(progress=True)
    assert runtime.progress is True and runtime.verbose_experts is False


@pytest.mark.parametrize("verbose_experts", [False, True])
def test_expert_verbosity_keeps_layer_timing_cache_stats_and_errors(monkeypatch, capsys, verbose_experts):
    runtime = cache_only_runtime()
    runtime.progress = True
    runtime.verbose_experts = verbose_experts
    runtime.artifact = NS(load_expert=lambda *args, **kwargs: ({"packed": torch.ones(4)}, None))

    def expert(self, expert_id, hidden):
        self._get_expert(expert_id)
        return hidden

    def forward(self, hidden, input_ids):
        self.expert(0, hidden)
        return self.expert(0, hidden)

    monkeypatch.setattr(execution.VQ2TP1MoE, "expert", expert)
    monkeypatch.setattr(execution.VQ2TP1MoE, "forward", forward)
    hidden = torch.ones(3, 4)
    assert runtime.forward(hidden) is hidden
    lines = capsys.readouterr().out.splitlines()
    assert any("stage=expert_start" in line for line in lines) is verbose_experts
    assert any("stage=expert_done" in line for line in lines) is verbose_experts
    assert any("stage=expert_load_start" in line for line in lines) is verbose_experts
    assert any("stage=expert_load_done" in line for line in lines) is verbose_experts
    reports = [
        json.loads(line.removeprefix("MODEL_MOE_TIMING ")) for line in lines if line.startswith("MODEL_MOE_TIMING ")
    ]
    assert len(reports) == 1
    assert reports[0]["cache_loads"] == reports[0]["cache_hits"] == 1
    assert reports[0]["tokens"] == 3 and reports[0]["resident_bytes"] == 32
    if not verbose_experts:
        assert len(lines) == 1
    with pytest.raises(ValueError, match="no stored expert"):
        runtime.expert(999, hidden)


@pytest.mark.parametrize("verbose_experts", [False, True])
def test_owner_keeps_layer_progress_and_propagates_expert_verbosity(monkeypatch, verbose_experts):
    monkeypatch.setattr(execution.VQ2TP1MoE, "__init__", lambda *args, **kwargs: None)
    owner = OfflineMoEOwner.__new__(OfflineMoEOwner)
    owner.options = {"execution_policy": "cached", "verbose_experts": verbose_experts}
    owner.layers, owner.calls = {}, {}
    owner.artifact = None
    owner.device = torch.device("cpu")
    layer = owner.create_layer(23)
    assert layer.progress is True and layer.verbose_experts is verbose_experts


def test_cached_weights_validate_only_on_misses_and_survive_repeated_passes():
    runtime = cache_only_runtime()
    calls = []

    def load(layer, expert, kind, **kwargs):
        assert set(kwargs) == {"device", "timings"}  # Never disable payload validation.
        assert kwargs["device"] == "cpu"
        for key, seconds in (("host_read_s", 0.25), ("host_validate_s", 0.75)):
            kwargs["timings"][key] = kwargs["timings"].get(key, 0.0) + seconds
        calls.append((expert, kind))
        return {"packed": torch.ones(4, dtype=torch.int32)}, None

    runtime.artifact = NS(load_expert=load)
    for _ in range(2):
        for expert in (0, 1, 2):
            runtime._get_expert(expert)
    assert len(calls) == 6  # two projections once per expert, not per pass
    assert runtime.timing["host_read_s"] == 1.5 and runtime.timing["host_validate_s"] == 4.5
    assert runtime.cache_stats() == dict(
        resident_experts=3, resident_bytes=96, peak_packed_bytes=96, loads=3, hits=3, evictions=0
    )
    runtime.clear_cache()
    assert runtime.cache_stats()["resident_bytes"] == 0
    assert runtime.cache_stats()["resident_experts"] == 0


def test_bounded_lru_and_failed_load_never_publish_partial_expert():
    runtime = cache_only_runtime(limit=2)
    runtime.artifact = NS(load_expert=lambda *args, **kwargs: ({"packed": torch.ones(4)}, None))
    for index in (0, 1, 0, 2):
        runtime._get_expert(index)
    assert list(runtime._cache) == [0, 2]
    assert runtime.cache_stats()["resident_bytes"] == 64
    assert runtime.evictions == 1

    def bad(*args, **kwargs):
        raise ValueError("Invalid payload")

    runtime.artifact.load_expert = bad
    with pytest.raises(ValueError, match="Invalid payload"):
        runtime._get_expert(1)
    assert runtime.cache_loads == 3 and 1 not in runtime._cache
    assert runtime._resident_bytes == 32
    with pytest.raises(ValueError, match="no stored"):
        runtime._get_expert(999)


@pytest.mark.parametrize("count", [1, 3, 10, 32])
@pytest.mark.parametrize("case", ["normal", "zero", "impulse", "small"])
def test_cached_policy_preserves_per_row_preparation_and_aligned_scalars(monkeypatch, count, case):
    runtime = cache_only_runtime()
    # Execute host math and a capturing kernel stand-in; no device claims.
    runtime.device = torch.device("cuda:0")
    monkeypatch.setattr(execution, "synchronize_execution", lambda device: None)
    calls = []

    def capture(q, scale, bias, *weights):
        assert q.shape == (1, 512)
        assert scale.shape == bias.shape == (1,)
        assert scale.storage_offset() == bias.storage_offset() == 0
        calls.append((q.clone(), scale.clone(), bias.clone()))
        return (q.float() * scale + bias).to(torch.bfloat16)

    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization.vq2a8_triton", NS(vq2a8_tp1_m1_packed_gemm=capture))
    x = ((torch.arange(count * 512).reshape(count, 512) % 17) - 8).to(torch.bfloat16) / 16
    if case == "zero":
        x.zero_()
    elif case == "impulse":
        x.zero_()
        x[:, 0] = 1
    elif case == "small":
        x *= 0.0001
    columns = torch.arange(512)
    payload = {
        "weight_scale": torch.ones(512),
        "weight_bias": torch.zeros(512),
        "rht_sign": torch.where(columns % 2 == 0, 1, -1).to(torch.int8),
        "packed_indices": None,
        "codebooks": None,
        "codebook_tile_ids": None,
    }
    actual = runtime._projection(x, payload, NS(columns=512, rht_true_columns=512, rht_block_size=128))
    assert actual.shape == (count, 512)
    assert len(calls) == count and runtime.prepare_batches == count and runtime.projection_rows == count
    for row, (q, scale, bias) in zip(x.split(1), calls):
        ref_q, ref_scale, ref_bias = prepare_repacked_vq2a8_activation_reference(
            row, payload["weight_scale"], payload["weight_bias"], payload["rht_sign"], 128
        )
        assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
        torch.testing.assert_close(scale, ref_scale, rtol=0, atol=0)
        torch.testing.assert_close(bias, ref_bias, rtol=0, atol=0)


@pytest.mark.parametrize("count", [1, 17, 32, 33, 65])
@pytest.mark.parametrize("columns", [512, 480])
def test_native_policy_batches_only_after_accepted_row_preparation(monkeypatch, count, columns):
    runtime = execution.AscendCVQ2TP1MoE.__new__(execution.AscendCVQ2TP1MoE)
    runtime.__dict__.update(cache_only_runtime().__dict__)
    runtime.device = NS(type="npu")  # Host capture only; no actual NPU assertion.
    runtime.reset_native_trace()
    monkeypatch.setattr(execution, "synchronize_execution", lambda device: None)
    calls = []
    payload = {
        "weight_scale": torch.ones(512),
        "weight_bias": torch.zeros(512),
        "rht_sign": torch.ones(512, dtype=torch.int8),
        "packed_indices": torch.zeros(1, dtype=torch.int32),
        "codebooks": torch.zeros(1),
        "codebook_tile_ids": torch.zeros(1),
    }

    def native(q, scale, bias, packed, book, ids):
        assert 1 <= q.shape[0] <= 32 and q.shape[1] == 512
        assert scale.shape == bias.shape == (q.shape[0],)
        assert (
            packed is payload["packed_indices"] and book is payload["codebooks"] and ids is payload["codebook_tile_ids"]
        )
        calls.append((q.clone(), scale.clone(), bias.clone()))
        return (q.float() * scale[:, None] + bias[:, None]).to(torch.bfloat16)

    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization.vq2a8_ascendc", NS(vq2a8_ascendc=native))
    hidden = ((torch.arange(count * columns).reshape(count, columns) % 17 - 8) / 16).bfloat16()
    output = runtime._projection(hidden, payload, NS(columns=512, rht_true_columns=columns, rht_block_size=128))
    expected = []
    for row in hidden.split(1):
        prepared = prepare_repacked_vq2a8_activation_reference(
            torch.nn.functional.pad(row, (0, 512 - columns)),
            payload["weight_scale"],
            payload["weight_bias"],
            payload["rht_sign"],
            128,
        )
        expected.append(prepared)
    for actual, ref in zip((torch.cat(v) for v in zip(*calls)), (torch.cat(v) for v in zip(*expected))):
        assert torch.equal(actual.view(torch.uint8), ref.view(torch.uint8))
    assert output.shape == (count, 512)
    assert runtime.native_calls == (count + 31) // 32
    assert runtime.native_rows == runtime.projection_rows == runtime.prepare_batches == count


def test_native_policy_no_host_fallback_and_no_count_on_failure(monkeypatch):
    with pytest.raises(ValueError, match="no CPU/CUDA fallback"):
        execution.AscendCVQ2TP1MoE(None, 0, "cpu")
    runtime = execution.AscendCVQ2TP1MoE.__new__(execution.AscendCVQ2TP1MoE)
    runtime.__dict__.update(cache_only_runtime().__dict__)
    runtime.reset_native_trace()
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.quantization.vq2a8_ascendc", NS(vq2a8_ascendc=lambda *a: pytest.fail("fallback"))
    )
    with pytest.raises(ValueError, match="never fall back"):
        runtime._projection(torch.zeros(1, 512), {}, None)
    assert runtime.native_calls == 0


def test_native_trace_reset_excludes_profile_and_preserves_cache(monkeypatch):
    runtime = execution.AscendCVQ2TP1MoE.__new__(execution.AscendCVQ2TP1MoE)
    runtime.native_calls, runtime.native_rows, runtime.native_experts = 20, 40, 10
    runtime.native_steps = [{"tokens": 999}]
    runtime._cache = {0: "packed"}
    runtime.reset_native_trace()

    def forward(self, hidden, input_ids):
        self.native_calls += 2
        self.native_rows += 2 * len(hidden)
        self.native_experts += 1
        self.native_launches += 2
        return hidden

    monkeypatch.setattr(execution.CachedVQ2TP1MoE, "forward", forward)
    hidden = torch.zeros(3, 4)
    assert runtime.forward(hidden) is hidden
    assert runtime.native_steps == [
        {"tokens": 3, "projection_calls": 2, "projection_rows": 6, "expert_calls": 1, "kernel_launches": 2}
    ]
    runtime.reset_native_trace()
    assert runtime.native_steps == [] and runtime.native_calls == 0
    assert runtime._cache == {0: "packed"}


@pytest.mark.parametrize("hashed", [True, False])
@pytest.mark.parametrize("verbose_experts", [False, True])
def test_native_policy_preserves_full_routed_chain_and_shared_scale_on_host(
    monkeypatch, capsys, hashed, verbose_experts
):
    """Real scheduler/preparation/SwiGLU with a CPU projection stand-in, not NPU evidence."""
    monkeypatch.setattr(execution, "synchronize_execution", lambda device: None)

    def projection(q, scale, bias, packed, book, ids):
        value = (q.float() * scale[:, None] + bias[:, None]).bfloat16()
        return torch.cat((value, value / 2), dim=1) if packed == "gate_up" else value

    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization.vq2a8_triton", NS(vq2a8_tp1_m1_packed_gemm=projection))
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.quantization.vq2a8_ascendc",
        NS(vq2a8_ascendc=projection, grouped_projection=lambda inputs: [projection(*args) for args in inputs]),
    )
    baseline = cache_only_runtime()
    candidate = execution.AscendCVQ2TP1MoE.__new__(execution.AscendCVQ2TP1MoE)
    candidate.__dict__.update(cache_only_runtime().__dict__)
    candidate.reset_native_trace()
    for runtime, device in ((baseline, "cuda"), (candidate, "npu")):
        runtime.device = NS(type=device)
        runtime.progress = True
        runtime.verbose_experts = verbose_experts
        runtime.token_chunk = 2
        runtime.config = NS(
            hidden_size=512, top_k=2, renormalize=True, num_shared=1, routed_scale=1.5, swiglu_limit=None
        )
        runtime.root = {"gate.weight": torch.zeros(2, 512)}
        runtime.root["gate.tid2eid" if hashed else "gate.bias"] = (
            torch.tensor([[0, 0], [1, 0]]) if hashed else torch.zeros(2)
        )
        runtime.shared = lambda hidden: torch.full_like(hidden, 2)
        for expert in (0, 1):
            runtime._cache[expert] = {
                kind: (
                    {
                        "weight_scale": torch.full((512,), 1 + expert / 2),
                        "weight_bias": torch.zeros(512),
                        "rht_sign": torch.ones(512, dtype=torch.int8),
                        "packed_indices": kind,
                        "codebooks": None,
                        "codebook_tile_ids": None,
                    },
                    NS(columns=512, rht_true_columns=512, rht_block_size=128),
                )
                for kind in ("gate_up", "down")
            }
    hidden = ((torch.arange(3 * 512).reshape(3, 512) % 11 - 5) / 16).bfloat16()
    ids = torch.tensor([0, 1, 0])
    expected = baseline.forward(hidden, ids)
    capsys.readouterr()
    actual = candidate.forward(hidden, ids)
    output = capsys.readouterr().out
    assert ("stage=expert_group_start" in output) is verbose_experts
    assert ("stage=expert_group_done" in output) is verbose_experts
    assert "MODEL_MOE_TIMING" in output
    assert torch.equal(actual, expected)
    assert candidate.cache_loads == 0  # existing packed residency reused
    step = candidate.native_steps[0]
    assert step["projection_calls"] == 2 * step["expert_calls"]
    assert step["projection_rows"] == (8 if hashed else 12)
    assert step["tokens"] == 3


@pytest.mark.parametrize("capacity", [1, 2, 6, 12])
@pytest.mark.parametrize("tokens", [1, 5])
def test_grouped_scheduler_preserves_duplicate_slots_and_bounds_pinned_payloads(monkeypatch, capacity, tokens):
    """Exercise real preparation/activation/mixer; native calls are CPU stand-ins."""
    monkeypatch.setattr(execution, "synchronize_execution", lambda device: None)
    launches = []

    def projection(q, scale, bias, packed, book, ids):
        kind, index = packed
        value = (q.float() * scale[:, None] + bias[:, None] + index / 16).bfloat16()
        return torch.cat((value, value / 2), dim=1) if kind == "gate_up" else value

    def single(*args):
        launches.append(1)
        return projection(*args)

    def grouped(inputs):
        launches.append(len(inputs))
        assert 2 <= len(inputs) <= min(6, capacity)
        return [projection(*values) for values in inputs]

    monkeypatch.setitem(
        sys.modules, "vllm_ascend.quantization.vq2a8_ascendc", NS(vq2a8_ascendc=single, grouped_projection=grouped)
    )
    runtime = execution.AscendCVQ2TP1MoE.__new__(execution.AscendCVQ2TP1MoE)
    runtime.__dict__.update(cache_only_runtime(limit=capacity).__dict__)
    runtime.layer = NS(expert_ids=tuple(range(12)))
    runtime.reset_native_trace()
    runtime.device = NS(type="npu")
    runtime.token_chunk = 2
    runtime.config = NS(routed_scale=1.5, swiglu_limit=None, num_shared=1)
    runtime.shared = lambda hidden: hidden / 8
    hidden = ((torch.arange(tokens * 512).reshape(tokens, 512) % 17 - 8) / 16).bfloat16()
    # Two duplicate slots in each token must not disappear from the reduction.
    ids = torch.tensor([[((row * 5 + i) % 12) for i in (0, 1, 2, 2, 3, 4, 5, 5)] for row in range(tokens)])
    weights = (torch.arange(tokens * 8).reshape(tokens, 8).float() % 7 + 1) / 32
    runtime.route = lambda h, inputs: (weights, ids)
    loaded = []

    def get_expert(index):
        loaded.append(index)
        if index not in runtime._cache:
            if len(runtime._cache) == capacity:
                runtime._cache.popitem(last=False)
            runtime._cache[index] = {
                kind: (
                    {
                        "weight_scale": torch.full((512,), 1 + index / 32),
                        "weight_bias": torch.zeros(512),
                        "rht_sign": torch.ones(512, dtype=torch.int8),
                        "packed_indices": (kind, index),
                        "codebooks": None,
                        "codebook_tile_ids": None,
                    },
                    NS(columns=512, rht_true_columns=512, rht_block_size=128),
                )
                for kind in ("gate_up", "down")
            }
        runtime._cache.move_to_end(index)
        return runtime._cache[index]

    runtime._get_expert = get_expert
    # Accepted per-expert schedule with the same native projection stand-in.
    expected = execution.VQ2TP1MoE.forward(runtime, hidden)
    expected_order = list(loaded)
    runtime.reset_native_trace()
    launches.clear()
    loaded.clear()
    actual = runtime.forward(hidden)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert loaded == expected_order
    assert len(runtime._cache) <= capacity
    record = runtime.native_steps[0]
    assert record["projection_calls"] == 2 * len(loaded) == sum(launches)
    assert record["kernel_launches"] == len(launches)
    if tokens == 1 and capacity >= 6:
        assert record["projection_calls"] == 12 and record["kernel_launches"] == 2
    if capacity == 1:
        assert record["kernel_launches"] == record["projection_calls"]


def test_grouped_projection_failure_cannot_count_as_completed_work(monkeypatch):
    runtime = execution.AscendCVQ2TP1MoE.__new__(execution.AscendCVQ2TP1MoE)
    runtime.__dict__.update(cache_only_runtime().__dict__)
    runtime.reset_native_trace()
    runtime.device = NS(type="npu")
    monkeypatch.setattr(execution, "synchronize_execution", lambda device: None)
    runtime._row_preparation = NS(many=lambda requests: [(None, None, None)] * len(requests))
    payload = dict.fromkeys(("packed_indices", "codebooks", "codebook_tile_ids"))

    def failure(inputs):
        raise RuntimeError("device failure")

    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization.vq2a8_ascendc", NS(grouped_projection=failure))
    with pytest.raises(RuntimeError, match="device failure"):
        runtime._projections_many([(torch.zeros(1, 512), payload, None)] * 2)
    assert runtime.native_calls == runtime.native_rows == runtime.native_experts == runtime.native_launches == 0


def test_grouped_schedule_rejects_missing_expert_before_any_projection():
    runtime = execution.AscendCVQ2TP1MoE.__new__(execution.AscendCVQ2TP1MoE)
    runtime.layer = NS(expert_ids=(0, 1))
    runtime.token_chunk = 2
    runtime.route = lambda hidden, inputs: (torch.ones(1, 2), torch.tensor([[0, 2]]))
    with pytest.raises(ValueError, match="Router selected experts missing"):
        runtime._forward(torch.zeros(1, 512), None)


def test_owner_configures_actual_layer_limits_before_any_expert_is_loaded(monkeypatch):
    owner = OfflineMoEOwner.__new__(OfflineMoEOwner)
    owner.options = {"execution_policy": "cached", "cache_experts": 256}
    owner.device = torch.device("cuda:0")
    owner.layers = {
        index: NS(layer=layer_header(index, count), cache_experts=1, cache_stats=lambda: {"resident_experts": 0})
        for index, count in ((0, 1), (3, 256))
    }
    monkeypatch.setattr(
        "vllm_ascend.quantization.vq2a8_offline.device_cache_budget",
        lambda *args, **kwargs: {"budget_bytes": 18 * 12 * 512},
    )
    owner.configure_cache(0.9)
    assert owner.layers[0].cache_experts == 1 and owner.layers[3].cache_experts == 17
    owner.layers[3].cache_stats = lambda: {"resident_experts": 1}
    with pytest.raises(ValueError, match="before"):
        owner.configure_cache(0.9)


def test_offline_policy_preserves_the_old_plan_as_explicit_baseline():
    baseline = offline_engine_options(Path("/m"), Path("/a"), execution_policy="baseline")
    cached = offline_engine_options(Path("/m"), Path("/a"))
    assert baseline["additional_config"]["vq2a8_offline"]["cache_experts"] == 2
    assert baseline["additional_config"]["vq2a8_offline"]["token_chunk"] == 2
    assert cached["additional_config"]["vq2a8_offline"]["cache_experts"] == 256
    assert cached["additional_config"]["vq2a8_offline"]["token_chunk"] == 2
    assert cached["kv_cache_memory_bytes"] == baseline["kv_cache_memory_bytes"]


def test_timeout_summary_reports_unknown_and_counts_only_finished_forwards(tmp_path):
    log = tmp_path / "child.log"
    log.write_text(
        'MODEL_FORWARD_TIMING {"phase":"profile","elapsed_s":10}\n'
        "MODEL stage=forward_start phase=prefill step=0 tokens=10\n"
        "MODEL layer=38 stage=expert_start expert=4 tokens=1\n"
    )
    result = summarize_log(log, None, timed_out=True)
    result["probe"] = "full_model"
    text = format_compact_summary({"stage": "model", "status": "failed", "probes": ["full_model"], "results": [result]})
    assert "finite=unknown repeat_exact=unknown" in text
    assert "FORWARD=profile completed=1 total_s=10.000" in text
    assert "FORWARD=prefill completed=0" in text
    assert "FORWARD=decode completed=0" in text
    assert "timeout=True" in text and "layer=38" in text


@pytest.mark.parametrize(
    "comparison,expected",
    [
        ({"allclose": True, "max_abs_error": 0}, "PASS"),
        ({"allclose": True, "max_abs_error": 0.01}, "UNKNOWN_OR_FAIL"),
        (None, "UNKNOWN_OR_FAIL"),
    ],
)
def test_cached_short_report_distinguishes_exact_agreement_from_tolerance(comparison, expected):
    text = format_compact_summary(
        {
            "stage": "moe",
            "probes": ["layer0"],
            "status": "running",
            "results": [
                {
                    "probe": "layer0",
                    "records": [
                        {
                            "type": "MOE_RESULT",
                            "data": {
                                "execution_policy": "cached",
                                "accepted_policy_comparison": comparison,
                            },
                        }
                    ],
                }
            ],
        }
    )
    assert f"baseline_exact={expected}" in text
