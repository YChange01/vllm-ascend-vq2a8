# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU arithmetic/control-flow contracts, not native execution or speed proof."""

from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_ascend.quantization import vq2a8_v4_device_route as device_route
from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_moe import route_vq2a8
from vllm_ascend.quantization.vq2a8_optimization import FastMoEState, OptimizationOptions, configure_runtime


@contextmanager
def no_host_tensor_reads(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("Singleton device routing must not materialize tensor values on the host")

    with monkeypatch.context() as scoped:
        for name in ("cpu", "numpy", "tolist", "item", "__bool__", "__int__", "__index__"):
            scoped.setattr(torch.Tensor, name, reject)
        yield


@pytest.mark.parametrize("hash_route", [False, True])
@pytest.mark.parametrize("renormalize", [False, True])
def test_device_router_matches_original_without_host_reads(monkeypatch, hash_route, renormalize):
    logits = torch.tensor([[0.0, 0.0, -2.0, 4.0, 4.0, 1.0]])
    options = {"renormalize": renormalize}
    if hash_route:
        options.update(hash_table=torch.tensor([[4, 4, 1, 0, 1, 5]]), input_ids=torch.tensor([0]))
    else:
        options["correction_bias"] = torch.tensor([0.5, 0.5, 10, 0, 0, -1])
    expected = route_vq2a8(logits, 6, **options)
    flags = []
    with no_host_tensor_reads(monkeypatch):
        actual = route_vq2a8(logits, 6, **options, device_only=True, validity=flags.append)
    for left, right in zip(actual, expected):
        assert torch.equal(left, right)
    assert all(bool(v) for v in flags)


@pytest.mark.parametrize("token,expert", [(-1, 0), (1, 0), (0, -1), (0, 6)])
def test_invalid_hash_indices_are_safe_and_deferred_not_silently_accepted(monkeypatch, token, expert):
    flags = []
    with no_host_tensor_reads(monkeypatch):
        route_vq2a8(
            torch.ones(1, 6),
            1,
            hash_table=torch.tensor([[expert]]),
            input_ids=torch.tensor([token]),
            device_only=True,
            validity=flags.append,
        )
    assert not all(bool(v) for v in flags)


@pytest.mark.parametrize("mode", [True, 1, "true"])
def test_device_router_requires_explicit_deferred_validation(mode):
    with pytest.raises(ValueError, match="deferred validity"):
        route_vq2a8(torch.ones(1, 6), 1, device_only=mode)


class CpuBankOracle:
    """Device-indexed tensor oracle; does not imitate or claim a native kernel."""

    def __init__(self, experts, n, k, generator):
        self.weight_scale = torch.rand(experts, k, generator=generator) + 0.1
        self.weight_bias = torch.randn(experts, k, generator=generator) / 128
        self.signs = (torch.randint(0, 2, (experts, k), generator=generator) * 2 - 1).to(torch.int8)
        self.weights = torch.randn(experts, n, k, generator=generator).to(torch.float8_e4m3fn)

    def select(self, ids):
        valid = (ids >= 0) & (ids < self.weights.shape[0])
        safe = ids.clamp(0, self.weights.shape[0] - 1)
        return [
            self.weight_scale.index_select(0, safe),
            self.weight_bias.index_select(0, safe),
            self.signs.index_select(0, safe),
            valid.int(),
        ]

    def project(self, x, scale, bias, ids):
        valid = (ids >= 0) & (ids < self.weights.shape[0])
        safe = ids.clamp(0, self.weights.shape[0] - 1)
        weights = self.weights.float().index_select(0, safe)
        # One-row GEMM on both oracle paths; output ordering follows route slots.
        output = torch.cat([x[i : i + 1].float() @ weights[i].T for i in range(x.shape[0])])
        output = (output * scale[:, None] + bias[:, None]).bfloat16()
        return [torch.where(valid[:, None], output, float("nan")), valid.int()]


def make_runtime(*, hash_route=False, sparse=False):
    generator = torch.Generator().manual_seed(17)
    count, width, top_k = 8, 16, 6
    banks = {
        kind: CpuBankOracle(count, width * 2 if kind == "gate_up" else width, width, generator)
        for kind in ("gate_up", "down")
    }
    spec = NS(rows=width, columns=width, rht_true_columns=width, rht_block_size=8)
    root = {"gate.weight": torch.randn(count, width, generator=generator)}
    if hash_route:
        root["gate.tid2eid"] = torch.tensor([[6, 1, 1, 6, 0, 7], [7, 6, 0, 1, 6, 7]])
    else:
        root["gate.bias"] = torch.tensor([0.0, 0.0, -2, -1, 1, 2, 0.5, 0.5])
    runtime = NS(
        execution_policy="ascendc_v4",
        device=torch.device("cpu"),
        root=root,
        config=NS(
            hidden_size=width,
            num_experts=count,
            top_k=top_k,
            renormalize=True,
            num_shared=1,
            swiglu_limit=7.0,
            routed_scale=1.5,
        ),
        layer=NS(expert_ids=tuple(range(count))),
        cache_experts=count,
        token_chunk=2,
        native_calls=0,
        native_rows=0,
        native_launches=0,
        native_experts=0,
        projection_rows=0,
        prepare_batches=0,
        _require_ready=lambda: None,
        shared=lambda x: x / 16,
    )
    lookup = torch.arange(count)
    if sparse:
        lookup[7] = -1
    runtime._device_route_banks = {"lookup": lookup, **{kind: (bank, spec) for kind, bank in banks.items()}}
    baseline_prepare = RowwiseVQ2A8Preparation(compact=True, validity=lambda flag: None)

    def get_expert(index):
        return {
            kind: (
                {
                    "weight_scale": bank.weight_scale[index],
                    "weight_bias": bank.weight_bias[index],
                    "rht_sign": bank.signs[index],
                    "test_index": index,
                    "test_kind": kind,
                },
                spec,
            )
            for kind, bank in banks.items()
        }

    def projections(requests):
        prepared = baseline_prepare.many(requests)
        return [
            banks[p["test_kind"]].project(*values, torch.tensor([p["test_index"]]))[0]
            for values, (_, p, _) in zip(prepared, requests)
        ]

    runtime._get_expert, runtime._projections_many = get_expert, projections
    hidden = torch.randn(1, width, generator=generator).bfloat16()
    return runtime, hidden


@pytest.mark.parametrize("hash_route", [False, True])
def test_singleton_path_matches_batched_v4_on_changing_tokens_and_routes(monkeypatch, hash_route):
    runtime, hidden = make_runtime(hash_route=hash_route)
    baseline = FastMoEState(runtime, OptimizationOptions.preset("batched"))
    configure_runtime(runtime, "device_route_decode")
    candidate = runtime._optimization
    for token, factor in ((0, 1), (1, -1), (0, 0.125), (1, 2)):
        x, tokens = hidden * factor, torch.tensor([token])
        expected = baseline.forward(runtime, x, tokens)
        with no_host_tensor_reads(monkeypatch):
            actual = candidate.forward(runtime, x, tokens)
        assert torch.equal(actual, expected)
        assert bool(candidate.valid)
    assert candidate.stats["route_host_reads"] == 0
    assert candidate.stats["singleton_forwards"] == 4
    assert candidate.stats["device_select_calls"] == 8
    assert candidate.report()["singleton_descriptor_h2d_bytes"] == 0


def test_absent_resident_expert_is_flagged_without_host_lookup(monkeypatch):
    runtime, hidden = make_runtime(hash_route=True, sparse=True)
    configure_runtime(runtime, "device_route_decode")
    with no_host_tensor_reads(monkeypatch):
        result = runtime._optimization.forward(runtime, hidden, torch.tensor([0]))
    assert not bool(runtime._optimization.valid)
    assert not bool(torch.isfinite(result).all())


def test_multitoken_prefill_uses_original_batched_state(monkeypatch):
    runtime, hidden = make_runtime()
    configure_runtime(runtime, "device_route_decode")
    calls = []
    monkeypatch.setattr(FastMoEState, "forward", lambda *a: calls.append(a) or hidden)
    actual = runtime._optimization.forward(runtime, hidden.repeat(10, 1), None)
    assert actual is hidden and len(calls) == 1
    assert runtime._optimization.stats["batched_prefill_forwards"] == 1
    assert runtime._optimization.stats["singleton_forwards"] == 0


def test_baseline_switch_preserves_banks_and_does_not_allocate_or_reload(monkeypatch):
    runtime, _ = make_runtime()
    banks = runtime._device_route_banks
    configure_runtime(runtime, "device_route_decode")
    state = runtime._optimization
    configure_runtime(runtime, "batched")
    assert type(runtime._optimization) is FastMoEState
    configure_runtime(runtime, "device_route_decode")
    assert runtime._optimization is state
    assert runtime._device_route_banks is banks
    assert state.valid is None


def test_new_runtime_rejects_nonresident_policy_or_missing_initialization():
    runtime, _ = make_runtime()
    runtime.execution_policy = "ascendc"
    with pytest.raises(ValueError, match="only supported"):
        configure_runtime(runtime, "device_route_decode")
    runtime.execution_policy = "ascendc_v4"
    runtime._device_route_banks = None
    with pytest.raises(ValueError, match="outside the forward"):
        configure_runtime(runtime, "device_route_decode")


def test_bank_initialization_only_uploads_sparse_lookup_and_keeps_tensor_owners(monkeypatch):
    from vllm_ascend.quantization.vq2a8_runtime import VQ2_TP1_FIELDS

    spec = NS(rows=32, columns=512, rht_true_columns=512, rht_block_size=128)
    cached = {
        i: {kind: ({field: torch.ones(1) for field in VQ2_TP1_FIELDS}, spec) for kind in ("gate_up", "down")}
        for i in (1, 4, 7)
    }
    fake_device = NS(type="npu")
    calls, uploads = [], []

    def bank_type(*fields):
        calls.append(fields)
        return NS(metadata=lambda: [3, 32, 512, 3 * 64])

    original_to = torch.Tensor.to

    def copy_mapping(tensor, *args, **kwargs):
        if args == (fake_device,):
            uploads.append(tensor.clone())
            return tensor.clone()
        return original_to(tensor, *args, **kwargs)

    runtime = NS(
        _require_ready=lambda: None,
        check_resident_integrity=lambda: None,
        device=fake_device,
        config=NS(num_experts=8, top_k=6),
        layer=NS(expert_ids=(1, 4, 7)),
        _cache=cached,
    )
    monkeypatch.setattr(device_route, "require_device_route_library", lambda: bank_type)
    monkeypatch.setattr(torch.Tensor, "to", copy_mapping)
    device_route.initialize_device_route_banks(runtime)
    assert len(uploads) == 1 and uploads[0].tolist() == [-1, 0, -1, -1, 1, -1, -1, 2]
    assert runtime._device_route_banks["metadata_bytes"] == 8 * 8 + 2 * 3 * 64
    for bank_args, kind in zip(calls, ("gate_up", "down")):
        for field_tensors, field in zip(bank_args, VQ2_TP1_FIELDS):
            assert all(tensor is cached[i][kind][0][field] for i, tensor in zip((1, 4, 7), field_tensors))
    banks = runtime._device_route_banks
    device_route.initialize_device_route_banks(runtime)
    assert runtime._device_route_banks is banks and len(calls) == 2 and len(uploads) == 1
