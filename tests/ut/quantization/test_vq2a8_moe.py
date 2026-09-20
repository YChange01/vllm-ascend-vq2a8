# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.quantization.vq2a8_moe import (
    VQ2MoEConfig,
    VQ2TP1MoE,
    load_vq2a8_moe_root_weights,
    mix_vq2a8_routes,
    route_vq2a8,
)


def test_router_bias_selects_but_does_not_weight_and_scale_is_external():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    weights, ids = route_vq2a8(logits, 2, correction_bias=torch.tensor([20.0, 10.0, 0.0, 0.0]))
    assert ids.tolist() == [[0, 1]]
    expected = torch.log1p(logits.double().exp()).sqrt()[:, :2]
    torch.testing.assert_close(weights, (expected / expected.sum()).float())
    torch.testing.assert_close(weights.sum(1), torch.ones(1))


def test_router_ties_choose_smallest_id_and_hash_keeps_duplicate_slots():
    _, ids = route_vq2a8(torch.zeros(3, 4), 3)
    assert ids.tolist() == [[0, 1, 2]] * 3
    weights, ids = route_vq2a8(
        torch.zeros(3, 4), 3, hash_table=torch.tensor([[2, 2, 2], [1, 0, 1]]), input_ids=torch.tensor([0, 1, 0])
    )
    assert ids.tolist() == [[2, 2, 2], [1, 0, 1], [2, 2, 2]]
    torch.testing.assert_close(weights, torch.full((3, 3), 1 / 3))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_ids": None},
        {"input_ids": torch.tensor([-1])},
        {"input_ids": torch.tensor([2])},
        {"input_ids": torch.tensor([0.0])},
        {"input_ids": torch.tensor([0]), "correction_bias": torch.zeros(4)},
    ],
)
def test_hash_router_rejects_invalid_token_contract(kwargs):
    with pytest.raises(ValueError):
        route_vq2a8(torch.zeros(1, 4), 2, hash_table=torch.zeros((2, 2), dtype=torch.int64), **kwargs)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_router_rejects_nonfinite_logits(value):
    with pytest.raises(ValueError, match="non-finite"):
        route_vq2a8(torch.full((1, 4), value), 2)


def test_duplicate_routes_evaluate_once_but_all_contribute_and_shared_runs_once():
    hidden = torch.arange(12).reshape(3, 4).to(torch.bfloat16)
    ids = torch.tensor([[0, 0, 1], [1, 1, 1], [0, 1, 0]])
    weights = torch.tensor([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5], [0.25, 0.5, 0.25]])
    calls, shared_calls = [], []

    def expert(index, values):
        calls.append((index, values.shape[0]))
        return values + index + 1

    def shared(values):
        shared_calls.append(values.shape[0])
        return torch.full_like(values, 8)

    actual = mix_vq2a8_routes(hidden, weights, ids, expert, routed_scale=1.5, shared=shared)
    expected = torch.zeros_like(hidden, dtype=torch.float32)
    for row in range(3):
        for slot in range(3):
            expected[row] += (hidden[row] + int(ids[row, slot]) + 1).float() * weights[row, slot]
    torch.testing.assert_close(actual, (expected * 1.5 + 8).to(torch.bfloat16))
    assert calls == [(0, 2), (1, 3)]
    assert shared_calls == [3]


def test_empty_batch_executes_neither_shared_nor_routed():
    def unexpected(*args):
        raise AssertionError("must not execute")

    result = mix_vq2a8_routes(
        torch.empty(0, 4, dtype=torch.bfloat16),
        torch.empty(0, 2),
        torch.empty(0, 2, dtype=torch.int64),
        unexpected,
        routed_scale=1.5,
        shared=unexpected,
    )
    assert result.shape == (0, 4)


def _layer_fixture(tmp_path, hash_layer=False):
    config = {
        "hidden_size": 4,
        "moe_intermediate_size": 2,
        "n_routed_experts": 3,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "num_hash_layers": int(hash_layer),
        "vocab_size": 5,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
        "scoring_func": "sqrtsoftplus",
        "hidden_act": "silu",
        "n_group": None,
        "topk_group": None,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    root = {
        "gate.weight": torch.zeros(3, 4, dtype=torch.bfloat16),
        "shared_experts.w1.weight": torch.ones(2, 4, dtype=torch.bfloat16),
        "shared_experts.w3.weight": torch.ones(2, 4, dtype=torch.bfloat16),
        "shared_experts.w2.weight": torch.ones(4, 2, dtype=torch.bfloat16),
    }
    if hash_layer:
        root["gate.tid2eid"] = torch.zeros(5, 2, dtype=torch.int64)
    else:
        root["gate.bias"] = torch.tensor([0.0, 1.0, 2.0])
    save_file({f"layers.0.ffn.{k}": v for k, v in root.items()}, str(tmp_path / "model.safetensors"))

    def load(*args, **kwargs):
        return {"packed": torch.ones(4, dtype=torch.int32)}, None

    return SimpleNamespace(
        model_config_path=path, layer=lambda i: SimpleNamespace(expert_ids=(0, 1, 2)), load_expert=load
    )


def test_real_root_names_and_cache_eviction_do_not_accumulate_experts(tmp_path):
    artifact = _layer_fixture(tmp_path)
    layer = VQ2TP1MoE(artifact, 0, "cpu", cache_experts=2)
    for expert_id in (0, 1, 0, 2):
        layer._get_expert(expert_id)
    assert list(layer._cache) == [0, 2]
    assert layer.cache_stats() == {
        "resident_experts": 2,
        "resident_bytes": 64,
        "peak_packed_bytes": 64,
        "loads": 3,
        "hits": 1,
    }
    layer.clear_cache()
    assert layer.cache_stats()["resident_bytes"] == 0


def test_chunked_multi_token_layer_equals_repeated_single_token(tmp_path, monkeypatch):
    layer = VQ2TP1MoE(_layer_fixture(tmp_path), 0, "cpu", token_chunk=2)
    monkeypatch.setattr(layer, "expert", lambda index, hidden: hidden + index)
    hidden = torch.arange(12).reshape(3, 4).to(torch.bfloat16) / 16
    batched = layer.forward(hidden)
    separate = torch.cat([layer.forward(row) for row in hidden.split(1)])
    torch.testing.assert_close(batched, separate, rtol=0, atol=0)


def test_hash_runtime_requires_token_ids_and_checks_artifact_coverage(tmp_path):
    artifact = _layer_fixture(tmp_path, hash_layer=True)
    layer = VQ2TP1MoE(artifact, 0, "cpu")
    with pytest.raises(ValueError, match="input_ids"):
        layer.forward(torch.zeros(1, 4, dtype=torch.bfloat16))
    artifact.layer = lambda i: SimpleNamespace(expert_ids=(1, 2))
    with pytest.raises(ValueError, match="missing"):
        VQ2TP1MoE(artifact, 0, "cpu")


def test_missing_shared_weight_is_not_silently_skipped(tmp_path):
    artifact = _layer_fixture(tmp_path)
    save_file({"layers.0.ffn.gate.weight": torch.zeros(3, 4)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="Missing MoE root tensors"):
        load_vq2a8_moe_root_weights(tmp_path, 0, VQ2MoEConfig.from_json(artifact.model_config_path))


def test_tp4_cannot_enter_tp1_runtime(tmp_path):
    with pytest.raises(ValueError, match="TP1"):
        VQ2TP1MoE(_layer_fixture(tmp_path), 0, "cpu", tp_size=4)


@pytest.mark.parametrize(
    "options",
    [{"cache_experts": 0}, {"token_chunk": 0}, {"cache_experts": 1.5}, {"token_chunk": True}, {"tp_size": True}],
)
def test_runtime_rejects_invalid_cache_and_chunk_limits(tmp_path, options):
    with pytest.raises(ValueError, match="TP1"):
        VQ2TP1MoE(_layer_fixture(tmp_path), 0, "cpu", **options)


@pytest.mark.parametrize("top_k,renormalize", [(True, True), (1.0, True), (2, 1)])
def test_router_rejects_ambiguous_option_types(top_k, renormalize):
    with pytest.raises(ValueError, match="router"):
        route_vq2a8(torch.zeros(1, 4), top_k, renormalize=renormalize)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("failure", ["nan", "shape", "dtype"])
def test_bad_expert_result_cannot_enter_reduction(shared, failure):
    hidden = torch.ones(1, 4, dtype=torch.bfloat16)

    def bad(values):
        if failure == "shape":
            return values[:, :1]
        if failure == "dtype":
            return values.float()
        return torch.full_like(values, float("nan"))

    with pytest.raises(ValueError):
        mix_vq2a8_routes(
            hidden,
            torch.ones(1, 1),
            torch.zeros(1, 1, dtype=torch.int64),
            lambda index, values: values if shared else bad(values),
            routed_scale=1.5,
            shared=bad if shared else None,
        )


def test_grouped_routing_is_rejected_not_silently_ignored(tmp_path):
    artifact = _layer_fixture(tmp_path)
    path = artifact.model_config_path
    config = json.loads(path.read_text())
    config["n_group"] = 2
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Grouped"):
        VQ2TP1MoE(artifact, 0, "cpu")


def test_empty_layer_batch_does_not_load_experts(tmp_path):
    layer = VQ2TP1MoE(_layer_fixture(tmp_path), 0, "cpu")
    assert layer.forward(torch.empty(0, 4, dtype=torch.bfloat16)).shape == (0, 4)
    assert layer.cache_stats()["loads"] == 0
