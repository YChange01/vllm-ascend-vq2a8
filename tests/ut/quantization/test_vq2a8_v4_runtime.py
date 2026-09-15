# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_ascend.quantization import vq2a8_execution as v1
from vllm_ascend.quantization import vq2a8_execution_v4 as v4
from vllm_ascend.quantization.vq2a8_repack import VQ2_DIRECT_TP1_FORMAT
from vllm_ascend.quantization.vq2a8_runtime import VQ2_TP1_TORCH_DTYPES, VQ2TP1Layer


def layer_header(index=0, experts=(0, 1, 2)):
    return VQ2TP1Layer(
        layer_index=index,
        expert_ids=experts,
        tensor_path=Path("/unused"),
        metadata_path=Path("/unused"),
        tensor_sha256="",
        metadata_sha256="",
        specs={"gate_up": NS(kind="gate_up"), "down": NS(kind="down")},
        tensor_shapes={
            f"{kind}_{field}": (len(experts), 4) for kind in ("gate_up", "down") for field in VQ2_TP1_TORCH_DTYPES
        },
    )


@pytest.fixture
def runtime(monkeypatch):
    """Replace only the hardware/root constructor, not V1 loading or V4 logic."""
    reads = []
    layer = layer_header()

    def load_expert(layer_index, expert_id, kind, **kwargs):
        assert kwargs.keys() == {"device", "timings"}
        assert kwargs["device"] == "cpu"
        assert not torch.is_inference_mode_enabled()
        reads.append((layer_index, expert_id, kind))
        kwargs["timings"]["host_read_s"] = kwargs["timings"].get("host_read_s", 0) + 0.01
        kwargs["timings"]["host_validate_s"] = kwargs["timings"].get("host_validate_s", 0) + 0.02
        return (
            {field: torch.full((4,), expert_id + 1, dtype=dtype) for field, dtype in VQ2_TP1_TORCH_DTYPES.items()},
            layer.specs[kind],
        )

    artifact = NS(manifest={"format": VQ2_DIRECT_TP1_FORMAT}, load_expert=load_expert)

    def init(self, provided_artifact, layer_index, *, device, progress=False, **kwargs):
        assert provided_artifact is artifact and layer_index == 0 and device == "cpu"
        self.artifact = provided_artifact
        self.device = torch.device(device)
        self.layer = layer
        self.layer_index = layer_index
        self._cache = OrderedDict()
        self.cache_experts = kwargs.get("cache_experts", 1)
        self.cache_loads = self.cache_hits = self.cache_peak_bytes = 0
        self._resident_bytes = self.evictions = self.h2d_bytes = 0
        self.progress = progress
        self.verbose_experts = False
        self.measurement_mode = False
        self.timing = dict.fromkeys(
            ("host_load_validate_s", "host_read_s", "host_validate_s", "h2d_s", "prepare_s", "packed_projection_s"),
            0.0,
        )

    monkeypatch.setattr(v1.AscendCVQ2TP1MoE, "__init__", init)
    result = v4.AscendCV4VQ2TP1MoE(artifact, 0, device="cpu", cache_experts=1)
    result.test_reads = reads
    return result


def initialize(runtime):
    plan = v4.packed_resident_plan([runtime.layer], 12 * 512 * len(runtime.layer.expert_ids))
    return runtime.initialize_resident(budget_bytes=plan["planned_bytes"])


class FakeGraphState:
    def __init__(self):
        self.prepared = self.closed = False
        self.replays = 0

    def prepare_graph(self, runtime):
        self.prepared = True

    def forward_graph(self, runtime, hidden, input_ids):
        self.replays += 1
        return hidden + 1

    def graph_snapshot(self):
        return {"prepared": self.prepared, "captures": int(self.prepared), "replays": self.replays, "entries": 1}

    def close_graph(self):
        self.closed = True


def test_replay_stream_selection_delegates_to_existing_prepared_state(runtime):
    initialize(runtime)
    with pytest.raises(RuntimeError, match="Prepare V4"):
        runtime.set_v4_graph_replay_stream("caller")
    runtime._optimization = state = FakeGraphState()
    runtime.prepare_v4_graph()
    selected = []
    state.set_graph_replay_stream = selected.append
    runtime.set_v4_graph_replay_stream("caller")
    runtime.set_v4_graph_replay_stream("owner")
    assert selected == ["caller", "owner"]
    assert runtime._v4_graph_state is state and state.prepared
    assert runtime.cache_loads == len(runtime.layer.expert_ids)


def test_graph_requires_explicit_prepare_and_semantic_decode(runtime, monkeypatch):
    initialize(runtime)
    with pytest.raises(RuntimeError, match="not prepared"):
        runtime.set_v4_graph_enabled(True)
    runtime._optimization = state = FakeGraphState()
    runtime.trace_native, runtime.native_steps = True, []
    runtime.prepare_v4_graph()
    runtime.set_v4_graph_enabled(True)
    hidden, ids = torch.zeros((1, 4), dtype=torch.bfloat16), torch.zeros((1,), dtype=torch.int32)
    eager = []
    monkeypatch.setattr(v1.AscendCVQ2TP1MoE, "forward", lambda *a: eager.append(True) or hidden)
    assert runtime.forward(hidden, ids) is hidden  # M1 prefill is not decode.
    assert state.replays == 0
    runtime.set_v4_graph_phase(True)
    assert torch.equal(runtime.forward(hidden, ids), hidden + 1)
    assert state.replays == 1 and len(eager) == 1
    assert runtime.native_steps == [
        {"tokens": 1, "graph_replays": 1, "counter_scope": "graph_replay_not_native_launch"}
    ]
    assert runtime.v4_graph_report()["ready"] is True
    runtime.set_v4_graph_enabled(False)
    runtime.forward(hidden, ids)
    assert state.replays == 1 and len(eager) == 2
    with pytest.raises(RuntimeError, match="single-shot"):
        runtime.prepare_v4_graph()


def test_graph_abort_releases_graph_before_resident_payload(runtime, monkeypatch):
    initialize(runtime)
    runtime._optimization = state = FakeGraphState()
    runtime.prepare_v4_graph()
    original = state.close_graph

    def close():
        assert runtime._cache
        original()

    monkeypatch.setattr(state, "close_graph", close)
    runtime.abort_residency()
    assert state.closed and not runtime._cache and runtime._v4_graph_state is None


def test_graph_failed_fence_retains_every_owner(runtime, monkeypatch):
    initialize(runtime)
    runtime._optimization = state = FakeGraphState()
    runtime.prepare_v4_graph()
    monkeypatch.setattr(v4, "synchronize_execution", lambda device: (_ for _ in ()).throw(RuntimeError("fence failed")))
    with pytest.raises(RuntimeError, match="fence failed"):
        runtime.abort_residency()
    assert runtime._v4_graph_state is state and runtime._cache and not state.closed
    assert runtime._resident_failed


def test_plan_is_v1_rounded_full_residency_not_a_reduced_lru_cap():
    unit = 12 * 512
    layers = [layer_header(0, (0,)), layer_header(3, tuple(range(256)))]
    plan = v4.packed_resident_plan(layers, unit * 257)
    original = v1.packed_cache_plan(layers, unit * 257)
    for field in ("planned_bytes", "full_packed_bytes", "budget_bytes", "layer_limits", "all_experts_fit"):
        assert plan[field] == original[field]
    assert plan["allocation"] == "eager_packed_only"
    assert plan["layout"] == "v1_packed"
    assert plan["layer_plans"][0] == {"experts": 1, "payload_bytes": 120, "planned_bytes": unit}
    assert plan["layer_plans"][3] == {"experts": 256, "payload_bytes": 120 * 256, "planned_bytes": unit * 256}
    with pytest.raises(ValueError, match="full residency.*no cache fallback"):
        v4.packed_resident_plan(layers, unit * 257 - 1)


@pytest.mark.parametrize("budget", [True, -1, 0, 1.5])
def test_plan_rejects_invalid_or_insufficient_budget(budget):
    with pytest.raises(ValueError):
        v4.packed_resident_plan([layer_header()], budget)


@pytest.mark.parametrize("experts", [(), (0, 0), (True,), (-1,), (1.5,)])
def test_plan_rejects_invalid_inventory(experts):
    with pytest.raises(ValueError, match="inventory"):
        v4.packed_resident_plan([layer_header(experts=experts)], 1 << 30)


def test_plan_rejects_duplicate_layers_and_z_n_fields():
    with pytest.raises(ValueError, match="Duplicate layer"):
        v4.packed_resident_plan([layer_header(), layer_header()], 1 << 30)
    layer = layer_header()
    layer.tensor_shapes.pop("gate_up_codebook_tile_ids")
    layer.tensor_shapes["gate_up_activation_order"] = (3, 4)
    with pytest.raises(KeyError, match="codebook_tile_ids"):
        v4.packed_resident_plan([layer], 1 << 30)


def test_full_model_v1_payload_uses_less_hbm_than_v3_expansion():
    def model_layer(index, count):
        layer = layer_header(index, tuple(range(count)))
        for kind, rows, columns in (("gate_up", 4096, 4096), ("down", 4096, 2048)):
            layer.tensor_shapes[f"{kind}_packed_indices"] = (count, rows // 2, columns // 8)
            layer.tensor_shapes[f"{kind}_codebooks"] = (count, columns // 256, rows // 32, 16, 2)
            for field in ("codebook_tile_ids", "weight_scale", "weight_bias", "rht_sign"):
                layer.tensor_shapes[f"{kind}_{field}"] = (count, columns)
        return layer

    layers = [model_layer(index, 1 if index < 3 else 256) for index in range(43)]
    plan = v4.packed_resident_plan(layers, 66_079_641_600)
    assert plan["full_packed_bytes"] == 66_079_641_600
    assert sum(item["payload_bytes"] for item in plan["layer_plans"].values()) == 66_079_641_600
    assert plan["full_packed_bytes"] < 66_520_172_544  # V3 expanded pair-LUT/gather layout.


@pytest.mark.parametrize("expert_id", [True, 1.0, "1", -1, 3])
def test_ready_lookup_rejects_invalid_ids_without_loading(runtime, expert_id):
    initialize(runtime)
    before = list(runtime.test_reads)
    with pytest.raises(ValueError, match="no stored expert"):
        runtime._get_expert(expert_id)
    assert runtime.test_reads == before
    assert runtime.check_resident_integrity()["post_init_loads"] == 0


def test_initialization_preloads_all_experts_once_with_original_reader(runtime):
    assert runtime.v4_report()["ready"] is False
    assert not hasattr(runtime, "_optimization")
    report = initialize(runtime)
    assert runtime.cache_experts == 3  # Never inherit the caller's initial LRU cap.
    assert runtime.test_reads == [(0, expert, kind) for expert in range(3) for kind in ("gate_up", "down")]
    assert report["ready"] is True and report["failed"] is False
    assert report["layout"] == "v1_packed" and report["fallback_enabled"] is False
    assert report["preload_loads"] == report["resident_experts"] == report["expected_experts"] == 3
    assert report["payload_bytes"] == 360
    assert report["planned_bytes"] == 12 * 512 * 3
    assert report["preload_h2d_bytes"] == 0  # CPU contracts must not claim actual device transfers.
    assert report["preload_elapsed_s"] >= 0
    assert report["post_init_loads"] == report["post_init_h2d_bytes"] == report["post_init_evictions"] == 0
    assert report["preload_evictions"] == 0
    assert not hasattr(runtime, "_optimization")
    assert runtime.timing["host_read_s"] == pytest.approx(0.06)
    assert runtime.timing["host_validate_s"] == pytest.approx(0.12)
    assert runtime.check_resident_integrity() == report


def test_v4_inherits_v1_arithmetic_and_preparation_without_any_override():
    for name in ("_projection", "_projections_many", "_forward", "_prepare_host_expert", "_timing_sync"):
        assert name not in v4.AscendCV4VQ2TP1MoE.__dict__
        assert getattr(v4.AscendCV4VQ2TP1MoE, name) is getattr(v1.AscendCVQ2TP1MoE, name)


def test_production_constructor_retains_v1_npu_only_gate():
    with pytest.raises(ValueError, match="requires NPU; no CPU/CUDA fallback"):
        v4.AscendCV4VQ2TP1MoE(None, 0, device="cpu")


def test_forward_is_ready_guard_then_exact_v1_delegation(runtime, monkeypatch):
    hidden, tokens = object(), object()
    calls = []

    def forward(self, hidden_arg, input_ids=None):
        calls.append((self, hidden_arg, input_ids))
        return hidden_arg

    monkeypatch.setattr(v1.AscendCVQ2TP1MoE, "forward", forward)
    with pytest.raises(RuntimeError, match="not ready"):
        runtime.forward(hidden, tokens)
    assert calls == []
    initialize(runtime)
    runtime.measurement_mode = True
    assert runtime.forward(hidden, tokens) is hidden
    assert calls == [(runtime, hidden, tokens)]
    assert runtime.measurement_mode is True


def test_lookup_after_preload_is_hit_only_without_lru_or_io(runtime):
    initialize(runtime)
    before = tuple(runtime._cache)
    runtime.artifact.load_expert = lambda *a, **kw: pytest.fail("Runtime disk read is forbidden")
    for expert_id in (1, 0, 2, 0):
        assert runtime._get_expert(expert_id) is runtime._cache[expert_id]
    assert runtime.cache_hits == 4
    assert tuple(runtime._cache) == before
    assert runtime.v4_report()["post_init_loads"] == 0
    with pytest.raises(ValueError, match="no stored expert"):
        runtime._get_expert(99)
    del runtime._cache[0]
    with pytest.raises(RuntimeError, match="missing or replaced.*no cache fallback"):
        runtime._get_expert(0)
    assert runtime.cache_loads == 3 and runtime.evictions == 0


def test_lookup_rejects_replacement_instead_of_returning_foreign_payload(runtime):
    initialize(runtime)
    runtime._cache[0] = dict(runtime._cache[0])
    with pytest.raises(RuntimeError, match="missing or replaced"):
        runtime._get_expert(0)


@pytest.mark.parametrize("when", ["before", "after"])
def test_clear_cache_never_silently_evicts_residents(runtime, when):
    if when == "after":
        initialize(runtime)
    count = len(runtime._cache)
    with pytest.raises(RuntimeError, match="cannot be evicted"):
        runtime.clear_cache()
    assert len(runtime._cache) == count


def test_insufficient_budget_fails_before_any_read_or_device_fence(runtime, monkeypatch):
    monkeypatch.setattr(v4, "synchronize_execution", lambda *a: pytest.fail("unexpected device access"))
    with pytest.raises(ValueError, match="full residency"):
        runtime.initialize_resident(budget_bytes=12 * 512 * 3 - 1)
    assert runtime.test_reads == [] and not runtime._cache
    assert runtime.v4_report()["ready"] is False


@pytest.mark.parametrize("format_name", ["vq2a8_zn_tp1_v1", "vq2a8_zn_tp2_v1", "unknown", None])
def test_initializer_explicitly_rejects_non_v1_artifacts(runtime, format_name):
    runtime.artifact.manifest["format"] = format_name
    with pytest.raises(ValueError, match="V1 direct-TP1"):
        initialize(runtime)
    assert runtime.test_reads == [] and not runtime._cache


def test_reinitialization_is_rejected_without_io(runtime):
    initialize(runtime)
    reads = len(runtime.test_reads)
    with pytest.raises(RuntimeError, match="once from an empty cache"):
        initialize(runtime)
    assert len(runtime.test_reads) == reads


def test_existing_cache_is_never_reinterpreted_as_full_residency(runtime):
    runtime._cache[0] = {}
    with pytest.raises(RuntimeError, match="empty cache"):
        initialize(runtime)
    assert runtime.test_reads == []


def test_failed_load_rolls_back_only_after_synchronization(runtime, monkeypatch):
    original = runtime.artifact.load_expert
    fences = []

    def load(layer, expert, kind, **kwargs):
        if expert == 1 and kind == "down":
            raise ValueError("strict artifact validation failed")
        return original(layer, expert, kind, **kwargs)

    runtime.artifact.load_expert = load
    monkeypatch.setattr(v4, "synchronize_execution", lambda device: fences.append(tuple(runtime._cache)))
    with pytest.raises(ValueError, match="strict artifact validation failed"):
        initialize(runtime)
    assert fences == [(0,)]  # Partial layer remains strongly owned until rollback fence.
    assert not runtime._cache and not runtime._resident_fingerprints
    report = runtime.v4_report()
    assert report["failed"] is True and report["ready"] is False
    assert report["preload_loads"] == 1 and report["payload_bytes"] == 0
    with pytest.raises(RuntimeError, match="once"):
        initialize(runtime)
    with pytest.raises(RuntimeError, match="not ready"):
        runtime._get_expert(0)


def test_failed_cleanup_preserves_original_failure_and_held_weights(runtime, monkeypatch):
    original = runtime.artifact.load_expert

    def load(layer, expert, kind, **kwargs):
        if expert == 1:
            raise ValueError("original loading failure")
        return original(layer, expert, kind, **kwargs)

    def fail_fence(device):
        raise RuntimeError("completion not established")

    runtime.artifact.load_expert = load
    monkeypatch.setattr(v4, "synchronize_execution", fail_fence)
    with pytest.raises(ValueError, match="original loading failure"):
        initialize(runtime)
    assert tuple(runtime._cache) == (0,) and runtime._resident_fingerprints
    assert runtime.v4_report()["failed"] is True
    assert "completion not established" in runtime.v4_report()["cleanup_error"]


def test_owner_can_abort_previously_ready_layer_after_model_wide_failure(runtime, monkeypatch):
    initialize(runtime)
    resident = runtime._cache[0]

    def fence(device):
        assert runtime._cache[0] is resident
        assert runtime.v4_report()["ready"] is False

    monkeypatch.setattr(v4, "synchronize_execution", fence)
    runtime.abort_residency()
    assert not runtime._cache and not runtime._resident_fingerprints
    assert runtime.v4_report()["failed"] is True
    with pytest.raises(RuntimeError, match="not ready"):
        runtime.forward(torch.zeros(1))


def test_ready_layer_abort_fence_failure_keeps_all_strong_references(runtime, monkeypatch):
    initialize(runtime)
    resident = tuple(runtime._cache.values())

    def fence(device):
        raise RuntimeError("device still owns payloads")

    monkeypatch.setattr(v4, "synchronize_execution", fence)
    with pytest.raises(RuntimeError, match="still owns payloads"):
        runtime.abort_residency()
    assert all(left is right for left, right in zip(resident, runtime._cache.values()))
    assert len(runtime._cache) == 3 and len(runtime._resident_fingerprints) == 3
    assert runtime.v4_report()["failed"] is True and runtime.v4_report()["ready"] is False
    with pytest.raises(RuntimeError, match="not ready"):
        runtime._get_expert(0)


@pytest.mark.parametrize("counter", ["cache_loads", "h2d_bytes", "evictions"])
def test_post_initialization_counter_changes_are_reported_and_rejected(runtime, counter):
    initialize(runtime)
    setattr(runtime, counter, getattr(runtime, counter) + 1)
    report_name = {
        "cache_loads": "post_init_loads",
        "h2d_bytes": "post_init_h2d_bytes",
        "evictions": "post_init_evictions",
    }
    assert runtime.v4_report()[report_name[counter]] == 1
    with pytest.raises(RuntimeError):
        runtime.check_resident_integrity()


@pytest.mark.parametrize("mutation", ["missing", "capacity", "bytes", "tensor", "dtype", "shape", "field", "spec"])
def test_request_boundary_integrity_rejects_inventory_and_storage_changes(runtime, mutation):
    initialize(runtime)
    payload, spec = runtime._cache[0]["gate_up"]
    if mutation == "missing":
        del runtime._cache[0]
    elif mutation == "capacity":
        runtime.cache_experts = 1
    elif mutation == "bytes":
        runtime._resident_bytes += 1
    elif mutation == "tensor":
        payload["packed_indices"] = payload["packed_indices"].clone()
    elif mutation == "dtype":
        payload["packed_indices"] = payload["packed_indices"].to(torch.int64)
    elif mutation == "shape":
        payload["packed_indices"] = payload["packed_indices"].view(2, 2)
    elif mutation == "field":
        payload["extra"] = torch.zeros(1)
    elif mutation == "spec":
        runtime._cache[0]["gate_up"] = (payload, NS(kind="gate_up"))
    with pytest.raises(RuntimeError):
        runtime.check_resident_integrity()


def test_request_boundary_integrity_is_metadata_only(runtime, monkeypatch):
    initialize(runtime)
    monkeypatch.setattr(v4, "synchronize_execution", lambda device: pytest.fail("unexpected fence"))
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *a, **kw: pytest.fail("unexpected D2H"))
    monkeypatch.setattr(torch.Tensor, "item", lambda *a, **kw: pytest.fail("unexpected scalar sync"))
    assert runtime.check_resident_integrity()["ready"] is True


def test_preload_progress_is_visible_without_expert_verbose_mode(runtime, capsys):
    runtime.progress = True
    initialize(runtime)
    output = capsys.readouterr().out
    assert "stage=v4_resident_payload loaded=3 total=3" in output
    assert "loaded_bytes=360" in output
