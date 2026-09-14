# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU arithmetic / lifecycle contracts; not a CANN or HCCL execution gate."""

from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_ascend.quantization import vq2a8_execution_tp2 as tp2
from vllm_ascend.quantization import vq2a8_execution_v3 as v3
from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_optimization import FastMoEState, OptimizationOptions
from vllm_ascend.quantization.vq2a8_v3_workspace import RESIDENT_FIELDS, ResidentV2ProjectionWorkspace, resident_shapes


def disk_layer(*, experts=(0,), down_k=1280):
    specs = {
        "gate_up": tp2.TP2ComputeSpec(2048, 4096, 4096),
        "down": tp2.TP2ComputeSpec(4096, down_k, 1024),
    }
    return NS(layer_index=3, expert_ids=experts, spec_for=lambda expert, kind: specs[kind], specs=specs)


def payload(spec):
    n, k = spec.rows, spec.columns
    generator = torch.Generator().manual_seed(k)
    return {
        "packed_zn": torch.randint(0, 256, (n // 32, k // 16, 16, 8), dtype=torch.uint8, generator=generator),
        "pair_lut": torch.randint(0, 120, (k // 256, n // 32, 32), dtype=torch.uint8, generator=generator),
        "activation_order": torch.randperm(k, generator=generator),
        "weight_scale": torch.cat((torch.ones(spec.rht_true_columns), torch.zeros(k - spec.rht_true_columns))),
        "weight_bias": torch.zeros(k),
        "rht_sign": torch.ones(k, dtype=torch.int8),
    }


@pytest.mark.parametrize("disk_k", [1024, 1280, 1536, 1792, 2048])
def test_tp2_down_always_has_two_v2_k1024_iterations(disk_k):
    disk = disk_layer(down_k=disk_k)
    layer = tp2.TP2ComputeLayer.from_disk(disk)
    shapes = resident_shapes(layer, "down")
    assert shapes["packed_zn"] == (1, 128, 128, 16, 8)
    assert layer.specs["down"].columns == 2048
    source = payload(disk.specs["down"])
    bank = {field: torch.empty(shapes[field], dtype=dtype)[0] for field, dtype, _ in RESIDENT_FIELDS}
    tp2.copy_tp2_payload(bank, source, disk.specs["down"], layer.specs["down"])
    assert torch.equal(bank["packed_zn"][:, : disk_k // 16], source["packed_zn"])
    assert not bank["packed_zn"][:, disk_k // 16 :].count_nonzero()
    assert torch.equal(bank["pair_lut"][: disk_k // 256], source["pair_lut"])
    assert not bank["pair_lut"][disk_k // 256 :].count_nonzero()
    assert torch.equal(bank["activation_order"][:disk_k], source["activation_order"])
    assert torch.equal(bank["activation_order"].sort().values, torch.arange(2048))
    assert not bank["weight_scale"][1024:].count_nonzero()
    assert not bank["weight_bias"][1024:].count_nonzero()
    assert torch.equal(bank["rht_sign"][disk_k:], torch.ones(2048 - disk_k, dtype=torch.int8))
    # The byte gather preserves the original packed positions; appended K is
    # zero after the same rank-local RHT/A8 preparation.
    x = torch.arange(1024).remainder(11).reshape(1, 1024).bfloat16()
    preparation = RowwiseVQ2A8Preparation(compact=True)
    before = preparation.rows(x, source, disk.specs["down"])
    after = preparation.rows(x, bank, layer.specs["down"])
    # A larger host GEMM can change signed zero; TP2 does not claim bytewise
    # identity with a different K geometry. Numeric FP8 values must agree.
    assert torch.equal(before[0].float(), after[0][:, :disk_k].float())
    # GEMM dispatch for extra zero blocks may also move the FP32 maximum by
    # an ulp. This is an arithmetic-padding check, not an exact-width oracle.
    torch.testing.assert_close(before[1], after[1], rtol=1e-6, atol=0)
    torch.testing.assert_close(before[2], after[2], rtol=1e-6, atol=0)
    gathered = after[0].view(torch.uint8).index_select(1, bank["activation_order"])
    assert not gathered[:, disk_k:].count_nonzero()


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("down", "columns", 2304),
        ("down", "columns", 1152),
        ("gate_up", "rows", 4096),
        ("down", "rht_true_columns", 2048),
    ],
)
def test_tp2_compute_layer_rejects_non_v2_shapes(kind, field, value):
    layer = disk_layer()
    spec = layer.specs[kind]
    layer.specs[kind] = NS(**{**vars(spec), field: value})
    with pytest.raises(ValueError, match="V2-compatible"):
        tp2.TP2ComputeLayer.from_disk(layer)


def test_tp2_resident_budget_uses_uniform_not_smaller_disk_k():
    small = tp2.TP2ComputeLayer.from_disk(disk_layer(experts=tuple(range(256)), down_k=1024))
    large = tp2.TP2ComputeLayer.from_disk(disk_layer(experts=tuple(range(256)), down_k=2048))
    p0 = v3.resident_plan([small], 1 << 40)
    p1 = v3.resident_plan([large], 1 << 40)
    assert p0 == p1
    assert p0["layer_plans"][3]["jobs"] == 6
    with pytest.raises(ValueError, match="no cache fallback"):
        v3.resident_plan([small], p0["planned_bytes"] - 1)


def runtime_shell():
    value = tp2.AscendCV3VQ2TP2MoE.__new__(tp2.AscendCV3VQ2TP2MoE)
    value.tp_rank, value.tp_size, value.tp_collectives = 0, 2, 0
    value.device = torch.device("cpu")
    value._v3_prefill_state = NS(scope=lambda _: nullcontext(), valid=None)
    return value


def test_tp2_initialize_streams_once_into_final_banks(monkeypatch):
    value = runtime_shell()
    disk = disk_layer()
    value.layer = tp2.TP2ComputeLayer.from_disk(disk)
    value.layer_index = 3
    value.config = NS(top_k=1)
    value.root = {"gate.weight": torch.zeros(1, 4096)}
    value._resident_ready = value._resident_failed = False
    value._cache = OrderedDict()
    value.projection_kernel, value.v3_preparation, value.v3_decode_graph = "v2", "eager", "none"
    value.progress = False
    value.cache_loads = value.cache_hits = value.h2d_bytes = 0
    value.decode_device_calls = value.prefill_legacy_calls = 0
    calls = []

    def shards(index, *, device):
        calls.append((index, device))
        yield {0: {kind: (payload(disk.specs[kind]), disk.specs[kind]) for kind in v3.KINDS}}

    value.artifact = NS(iter_rank_shards=shards)
    monkeypatch.setattr(tp2, "resident_library_capabilities", lambda **kwargs: 7 if kwargs["require_tp2"] else 0)
    report = value.initialize_resident(budget_bytes=1 << 30)
    assert report["ready"] and calls == [(3, "cpu")]
    assert report["tensor_parallel_size"] == 2
    for kind in v3.KINDS:
        ws = value._resident_workspaces[kind]
        assert isinstance(ws, ResidentV2ProjectionWorkspace)
        assert ws.launcher.keywords == {"tp_size": 2}
        for field in ws.banks:
            assert value._cache[0][kind][0][field].data_ptr() == ws.banks[field][0].data_ptr()
        assert value._cache[0][kind][1] == value.layer.specs[kind]
    value.check_resident_immutable()


def test_tp2_grouped_prefill_launch_is_explicit(monkeypatch):
    value = runtime_shell()
    calls = []
    monkeypatch.setattr(tp2, "grouped_projection_resident", lambda inputs, **kw: calls.append((inputs, kw)))
    value._launch_resident(["job"])
    assert calls == [(["job"], {"tp_size": 2})]


@pytest.mark.parametrize("tokens", [1, 2, 42])
def test_tp2_sum_routes_once_before_replicated_shared_in_decode_and_prefill(tokens):
    value = runtime_shell()
    calls = []
    value.tp_group = NS(all_reduce=lambda result: calls.append(result.clone()) or result * 2)
    value.config = NS(hidden_size=4, top_k=1, num_shared=1, renormalize=True, swiglu_limit=7.0, routed_scale=1.5)
    value.root = {"gate.weight": torch.ones(1, 4)}
    value.layer = NS(expert_ids=(0,))
    value.cache_experts, value.token_chunk = 1, 2
    value.native_experts = 0
    value.shared = lambda hidden: torch.full_like(hidden, 5)
    value._resident_valid = None
    hidden = torch.ones(tokens, 4).bfloat16()
    if tokens == 1:
        value._resident_lookup = torch.zeros(1, dtype=torch.int64)
        value._route_device = lambda *args: (torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int64))
        value._resident_workspaces = {
            "gate_up": NS(jobs=1, project=lambda *args: torch.ones(1, 8).bfloat16()),
            "down": NS(project=lambda *args: torch.full((1, 4), 2, dtype=torch.bfloat16)),
        }
        result = value._decode(hidden, torch.tensor([0]))
    else:
        value._get_expert = lambda expert: {kind: ({"kind": kind}, NS()) for kind in v3.KINDS}
        value._projections_many = lambda reqs: [
            torch.full((x.shape[0], 8 if p["kind"] == "gate_up" else 4), 2, dtype=torch.bfloat16) for x, p, _ in reqs
        ]
        result = FastMoEState(value, OptimizationOptions.preset("batched")).forward(value, hidden, None)
    assert len(calls) == value.tp_collectives == 1
    assert torch.equal(calls[0], torch.full((tokens, 4), 3.0))  # weighted routed partial only
    assert torch.equal(result, torch.full_like(hidden, 11))  # 3 + 3 + shared(5), NOT 16


@pytest.mark.parametrize("rank,world,group_rank", [(2, 2, 0), (0, 1, 0), (0, 2, 1)])
def test_tp2_constructor_rejects_wrong_distributed_context(rank, world, group_rank):
    with pytest.raises(ValueError, match="TP2"):
        tp2.AscendCV3VQ2TP2MoE(
            NS(tp_rank=0),
            3,
            "npu:0",
            tp_rank=rank,
            tp_group=NS(world_size=world, rank_in_group=group_rank, all_reduce=lambda x: x),
        )
