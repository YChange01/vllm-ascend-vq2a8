# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU fake-op contracts only: no CANN compilation, latency or graph claim."""

from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_ascend.quantization import vq2a8_ascendc_v3 as binding
from vllm_ascend.quantization import vq2a8_execution_v3 as v3
from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_moe import route_vq2a8
from vllm_ascend.quantization.vq2a8_reference import deepseek_v4_swiglu_reference


def inventory(experts=(0, 1, 2, 3, 4, 5), index=3):
    specs = {
        kind: NS(rows=1024 if kind == "gate_up" else 512, columns=512, rht_true_columns=512, rht_block_size=128)
        for kind in v3.KINDS
    }
    shapes = {}
    for kind, spec in specs.items():
        count, n, k = len(experts), spec.rows, spec.columns
        shapes.update(
            {
                f"{kind}_packed_indices": (count, n // 2, k // 8),
                f"{kind}_codebooks": (count, 1, n // 32, 16, 2),
                f"{kind}_codebook_tile_ids": (count, k),
                **{f"{kind}_{field}": (count, k) for field in v3.TRANSFORM_FIELDS},
            }
        )
    return NS(layer_index=index, expert_ids=experts, specs=specs, tensor_shapes=shapes)


def payload_banks(layer, kind):
    banks = {
        field: torch.zeros(layer.tensor_shapes[f"{kind}_{field}"], dtype=v3.VQ2_TP1_TORCH_DTYPES[field])
        for field in v3.ELEMENT_BYTES
    }
    for row in range(len(layer.expert_ids)):
        banks["weight_scale"][row].fill_(1 + row / 16)
        banks["weight_bias"][row].fill_(row / 32)
        banks["rht_sign"][row].fill_(1)
        banks["rht_sign"][row, row::3] = -1
    return banks


def fake_projection(descriptors, constants, owners, **geometry):
    # Explicit CPU stand-in, NOT execution of pointers or the native kernel.
    x, scale, bias, output = owners[1:5]
    assert descriptors.shape == (geometry["jobs"], 12)
    assert constants.numel() == binding.CONSTANT_WORDS
    value = x.float().sum(1) * scale + bias
    output.copy_(value[:, None].expand_as(output).bfloat16())


def workspace(layer, kind, jobs, *, launcher=fake_projection, flags=None):
    banks = payload_banks(layer, kind)
    payloads = [{field: banks[field][row] for field in banks} for row in range(len(layer.expert_ids))]
    prepare = RowwiseVQ2A8Preparation(compact=True, validity=(flags if flags is not None else []).append)
    prepare._ensure_hadamard(torch.device("cpu"), 128)
    return v3.ResidentProjectionWorkspace(
        payloads,
        layer.specs[kind],
        jobs,
        torch.zeros(binding.CONSTANT_WORDS, dtype=torch.int32),
        prepare,
        banks=banks,
        launcher=launcher,
    )


def runtime(monkeypatch, *, singleton=False, hash_routes=False):
    experts = (0,) if singleton else tuple(range(6))
    layer = inventory(experts)
    value = v3.AscendCV3VQ2TP1MoE.__new__(v3.AscendCV3VQ2TP1MoE)
    value.layer, value.layer_index, value.device = layer, layer.layer_index, torch.device("cpu")
    # Retain coverage for the original diagnostic runtime. The default V3
    # converted-layout workspace is exercised in the resident runtime tests.
    value.projection_kernel = "legacy"
    value.v3_preparation, value.v3_decode_graph = "eager", "none"
    value._decode_graph = None
    value.config = NS(hidden_size=512, top_k=6, renormalize=True, num_shared=0, swiglu_limit=7.0, routed_scale=1.5)
    value.root = {"gate.weight": torch.arange(6 * 512).reshape(6, 512).float() / (6 * 512 * 512)}
    if hash_routes or singleton:
        value.root["gate.tid2eid"] = (
            torch.zeros((4, 6), dtype=torch.int64)
            if singleton
            else torch.tensor([[0, 0, 2, 2, 4, 4], [5, 3, 1, 5, 3, 1], [0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]])
        )
    value._resident_ready = True
    value._resident_failed = False
    value._resident_valid = None
    value._resident_busy = False
    value._resident_stream = None
    value._resident_lookup = (
        torch.arange(6, dtype=torch.int64) if not singleton else torch.tensor([0, -1, -1, -1, -1, -1])
    )
    jobs = 1 if singleton else 6
    value._resident_workspaces = {kind: workspace(layer, kind, jobs) for kind in v3.KINDS}
    for ws in value._resident_workspaces.values():
        ws.preparation.validity = value._retain_valid
    value.native_calls = value.native_rows = value.native_experts = value.native_launches = 0
    value.projection_rows = value.prepare_batches = value.decode_device_calls = value.prefill_legacy_calls = 0
    value._v3_prefill_state = NS(valid=None, scope=lambda name: nullcontext())
    monkeypatch.setattr(torch, "npu", NS(current_stream=lambda *args: NS(npu_stream=7)), raising=False)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    return value


def test_v3_full_resident_plan_never_shrinks_expert_inventory():
    layers = [inventory((0,), 0), inventory(tuple(range(6)), 3)]
    plan = v3.resident_plan(layers, 1 << 30, kernel="legacy")
    assert plan["layer_limits"] == {0: 1, 3: 6}
    assert plan["layer_plans"][0]["jobs"] == 1
    assert plan["layer_plans"][3]["jobs"] == 6
    assert plan["payload_bytes"] + plan["workspace_bytes"] == plan["planned_bytes"]
    assert v3.resident_plan(layers, plan["planned_bytes"], kernel="legacy")["all_experts_fit"]
    with pytest.raises(ValueError, match="no cache fallback"):
        v3.resident_plan(layers, plan["planned_bytes"] - 1, kernel="legacy")


@pytest.mark.parametrize("budget", [0, -1, True, 1.2])
def test_v3_resident_plan_requires_explicit_byte_budget(budget):
    with pytest.raises(ValueError):
        v3.resident_plan([inventory()], budget, kernel="legacy")


@pytest.mark.parametrize("slots", [[0, 1, 2, 3, 4, 5], [5, 3, 1, 5, 3, 1]])
def test_v3_workspace_changes_device_pointers_and_preserves_exact_v1_preparation(slots):
    layer = inventory()
    ws = workspace(layer, "gate_up", 6)
    x = torch.randn(1, 512, generator=torch.Generator().manual_seed(12)).bfloat16()
    addresses = (
        ws.x.data_ptr(),
        ws.scale.data_ptr(),
        ws.bias.data_ptr(),
        ws.output.data_ptr(),
        ws.descriptors.data_ptr(),
    )
    ws.project(x, torch.tensor(slots))
    requests = [(x, ws.payloads[index], ws.spec) for index in slots]
    expected = RowwiseVQ2A8Preparation(compact=True).many(requests)
    for index, (quantized, scale, bias) in enumerate(expected):
        assert torch.equal(ws.x[index : index + 1].view(torch.uint8), quantized.view(torch.uint8))
        assert torch.equal(ws.scale[index : index + 1], scale)
        assert torch.equal(ws.bias[index : index + 1], bias)
        assert torch.equal(ws.descriptors[index, 3:6], ws.pointer_bank[slots[index]])
    ws.project(x * 2, torch.tensor(list(reversed(slots))))
    assert addresses == (
        ws.x.data_ptr(),
        ws.scale.data_ptr(),
        ws.bias.data_ptr(),
        ws.output.data_ptr(),
        ws.descriptors.data_ptr(),
    )


@pytest.mark.parametrize("width,true_width", [(512, 508), (2048, 2048), (4096, 4096)])
@pytest.mark.parametrize("case", ["zero", "large", "rounding"])
def test_v3_bulk_preparation_padding_and_rounding_cases_match_v1_many(width, true_width, case):
    layer = inventory()
    spec = layer.specs["gate_up"]
    spec.columns, spec.rht_true_columns = width, true_width
    for field in v3.ELEMENT_BYTES:
        shape = layer.tensor_shapes[f"gate_up_{field}"]
        if field == "packed_indices":
            shape = (*shape[:-1], width // 8)
        elif field != "codebooks":
            shape = (*shape[:-1], width)
        layer.tensor_shapes[f"gate_up_{field}"] = shape
    ws = workspace(layer, "gate_up", 6)
    x = torch.randn(6, true_width, generator=torch.Generator().manual_seed(width)).bfloat16()
    if case == "zero":
        x.zero_()
    elif case == "large":
        x *= 256
    else:
        x = (x.float() / 32 + 1.0625).bfloat16()
    slots = torch.tensor([5, 0, 3, 3, 1, 2])
    ws.project(x, slots)
    expected = RowwiseVQ2A8Preparation(compact=True).many(
        [(x[index : index + 1], ws.payloads[expert], spec) for index, expert in enumerate(slots.tolist())]
    )
    assert torch.equal(ws.x.view(torch.uint8), torch.cat([q for q, _, _ in expected]).view(torch.uint8))
    assert torch.equal(ws.scale, torch.cat([s for _, s, _ in expected]))
    assert torch.equal(ws.bias, torch.cat([b for _, _, b in expected]))


@pytest.mark.parametrize("singleton,hash_routes", [(False, False), (False, True), (True, True)])
def test_v3_decode_has_no_host_tensor_reads_and_keeps_slot_reduction_exact(monkeypatch, singleton, hash_routes):
    value = runtime(monkeypatch, singleton=singleton, hash_routes=hash_routes)
    hidden = torch.ones(1, 512).bfloat16()
    token = torch.tensor([0])
    expected_weights, expected_ids = route_vq2a8(
        F_linear(hidden.float(), value.root["gate.weight"]),
        6,
        hash_table=value.root.get("gate.tid2eid"),
        input_ids=token,
    )
    slots = value._resident_lookup[expected_ids.flatten()]
    ws_gate, ws_down = (value._resident_workspaces[kind] for kind in v3.KINDS)
    selected = slots[:1] if singleton else slots
    gate = ws_gate.project(hidden, selected)
    down = ws_down.project(deepseek_v4_swiglu_reference(gate, 7.0), selected)
    expanded = down.expand(6, -1) if singleton else down
    expected = ((expanded.float().view(1, 6, 512) * expected_weights.unsqueeze(-1)).sum(1) * 1.5).bfloat16()
    with monkeypatch.context() as patch:

        def forbidden(*args, **kwargs):
            raise AssertionError("decode attempted a host tensor read")

        for method in ("cpu", "tolist", "item", "__bool__"):
            patch.setattr(torch.Tensor, method, forbidden)
        actual = value._forward(hidden, token)
    assert torch.equal(actual, expected)
    assert actual.data_ptr() != ws_down.output.data_ptr()
    saved = actual.clone()
    value._forward(hidden * 2, torch.tensor([1]))
    assert torch.equal(actual, saved)
    assert value.native_launches == 4
    assert value.native_calls == (4 if singleton else 24)


def F_linear(hidden, weight):
    return torch.nn.functional.linear(hidden, weight)


@pytest.mark.parametrize("token", [-1, 4, 2**60])
def test_v3_hash_invalid_input_is_clamped_before_gather_and_retains_failure(monkeypatch, token):
    value = runtime(monkeypatch, hash_routes=True)
    output = value._forward(torch.ones(1, 512).bfloat16(), torch.tensor([token]))
    assert output.shape == (1, 512)
    assert not bool(value.v3_validity())


def test_v3_deterministic_topk_ties_match_original_router(monkeypatch):
    value = runtime(monkeypatch)
    value.root["gate.weight"].zero_()
    weights, ids = value._route_device(torch.ones(1, 512).bfloat16(), None)
    assert torch.equal(ids, torch.arange(6).view(1, 6))
    expected, _ = route_vq2a8(torch.zeros(1, 6), 6)
    assert torch.equal(weights, expected)


def test_v3_workspace_rejects_other_stream_reentrancy_and_uninitialized_state(monkeypatch):
    value = runtime(monkeypatch)
    hidden = torch.ones(1, 512).bfloat16()
    value._forward(hidden, None)
    monkeypatch.setattr(torch.npu, "current_stream", lambda *args: NS(npu_stream=8))
    with pytest.raises(RuntimeError, match="one stream"):
        value._forward(hidden, None)
    value._resident_busy = True
    with pytest.raises(RuntimeError, match="concurrently"):
        value._forward(hidden, None)
    value._resident_ready = False
    with pytest.raises(RuntimeError, match="initialization"):
        value._forward(hidden, None)


def test_v3_no_eviction_or_missing_expert_fallback(monkeypatch):
    value = runtime(monkeypatch)
    value._cache = {0: {}}
    with pytest.raises(RuntimeError, match="cannot be evicted"):
        value.clear_cache()
    with pytest.raises(RuntimeError, match="no lazy cache fallback"):
        value._get_expert(1)


def test_v3_integrity_gate_detects_inplace_mutation(monkeypatch):
    value = runtime(monkeypatch)
    tensor = value.root["gate.weight"]
    value._resident_seals = tuple((tensor, value._seal(tensor)) for tensor in value._immutable_tensors())
    value.check_resident_integrity()
    tensor.add_(1)
    with pytest.raises(RuntimeError, match="changed"):
        value.check_resident_integrity()
    assert value._resident_failed


def test_v3_integrity_gate_detects_live_bank_replacement(monkeypatch):
    value = runtime(monkeypatch)
    value._resident_seals = tuple((tensor, value._seal(tensor)) for tensor in value._immutable_tensors())
    value._resident_workspaces["down"].pointer_bank = value._resident_workspaces["down"].pointer_bank.clone()
    with pytest.raises(RuntimeError, match="changed"):
        value.check_resident_integrity()


def test_v3_stream_records_all_storage_once(monkeypatch):
    value = runtime(monkeypatch)
    recorded = []
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: recorded.append(tensor.data_ptr()))
    hidden = torch.ones(1, 512).bfloat16()
    value._forward(hidden, None)
    first = list(recorded)
    assert first and len(first) == len(set(first))
    expected = {tensor.data_ptr() for tensor in value._immutable_tensors()}
    assert expected.issubset(recorded)
    value._forward(hidden, None)
    assert recorded == first


@pytest.mark.parametrize("inference", [False, True])
def test_v3_initialization_loads_directly_into_final_banks_and_never_duplicates_device_weights(monkeypatch, inference):
    value = runtime(monkeypatch)
    value._resident_ready = False
    value._resident_workspaces = {}
    value._cache = OrderedDict()
    value.progress = False
    value.cache_loads = value.cache_hits = value.cache_peak_bytes = value.h2d_bytes = 0
    source = {kind: payload_banks(value.layer, kind) for kind in v3.KINDS}
    loads = []

    def load_expert(layer_index, expert, kind, *, device):
        assert device == "cpu"
        loads.append((expert, kind))
        return ({field: bank[expert].clone() for field, bank in source[kind].items()}, value.layer.specs[kind])

    value.artifact = NS(load_expert=load_expert)
    monkeypatch.setattr(v3, "make_constants", lambda anchor: torch.zeros(binding.CONSTANT_WORDS, dtype=torch.int32))
    with torch.inference_mode(inference):
        report = value.initialize_resident(budget_bytes=1 << 30)
    assert report["ready"] and len(loads) == 12
    for expert in range(6):
        for kind in v3.KINDS:
            for field in v3.ELEMENT_BYTES:
                payload = value._cache[expert][kind][0][field]
                bank = value._resident_workspaces[kind].banks[field]
                assert payload.data_ptr() == bank[expert].data_ptr()
                assert bank._version >= 6
    with pytest.raises(RuntimeError, match="fresh empty"):
        value.initialize_resident(budget_bytes=1 << 30)
    with pytest.raises(TypeError):
        value.root["gate.weight"] = torch.zeros(6, 512)


def test_v3_hash_int32_table_remains_integer_gather_safe(monkeypatch):
    value = runtime(monkeypatch, hash_routes=True)
    value.root["gate.tid2eid"] = value.root["gate.tid2eid"].int()
    _, ids = value._route_device(torch.ones(1, 512).bfloat16(), torch.tensor([0], dtype=torch.int32))
    assert ids.dtype == torch.int64
    value._forward(torch.ones(1, 512).bfloat16(), torch.tensor([0], dtype=torch.int32))


def test_v3_initialization_budget_failure_precedes_payload_reads(monkeypatch):
    value = runtime(monkeypatch)
    value._resident_ready = False
    value._cache = OrderedDict()
    value.artifact = NS(load_expert=lambda *args, **kwargs: pytest.fail("budget failure must not read payloads"))
    with pytest.raises(ValueError, match="exceeds budget"):
        value.initialize_resident(budget_bytes=1)


def test_v3_partial_residency_failure_is_terminal_and_never_falls_back(monkeypatch):
    value = runtime(monkeypatch)
    value._resident_ready = False
    value._cache = OrderedDict()

    def failed_load(*args, **kwargs):
        raise RuntimeError("simulated host payload failure")

    value.artifact = NS(load_expert=failed_load)
    with pytest.raises(RuntimeError, match="host payload failure"):
        value.initialize_resident(budget_bytes=1 << 30)
    assert value._resident_failed and not value._resident_ready
    with pytest.raises(RuntimeError, match="fresh empty"):
        value.initialize_resident(budget_bytes=1 << 30)
    with pytest.raises(RuntimeError, match="initialization"):
        value._forward(torch.ones(1, 512).bfloat16(), None)


def test_v3_prefill_uses_explicit_v3_legacy_grouped_path_only(monkeypatch):
    value = runtime(monkeypatch)
    value.cache_hits, value.cache_experts, value.token_chunk = 0, 6, 2
    value._cache = {
        expert: {
            kind: (value._resident_workspaces[kind].payloads[expert], value.layer.specs[kind]) for kind in v3.KINDS
        }
        for expert in range(6)
    }
    value._row_preparation = value._resident_workspaces["gate_up"].preparation
    value._v3_prefill_state = v3.FastMoEState(value, v3.OptimizationOptions.preset("batched"))
    calls = []

    def legacy_projection(inputs):
        calls.append(len(inputs))
        outputs = []
        for quantized, scale, bias, packed, _, _ in inputs:
            reduced = quantized.float().sum(1) * scale + bias
            outputs.append(reduced[:, None].expand(-1, packed.shape[0] * 2).bfloat16())
        return outputs

    monkeypatch.setattr(v3, "grouped_projection_v3", legacy_projection)
    result = value._forward(torch.ones(2, 512).bfloat16(), None)
    assert result.shape == (2, 512) and result.dtype == torch.bfloat16
    assert calls == [6, 6]
    assert value.prefill_legacy_calls == 1 and value.decode_device_calls == 0
    assert value._v3_prefill_state.stats["route_host_reads"] == 1


def test_v3_probe_rejects_fwht_without_replacing_owned_state(monkeypatch):
    value = runtime(monkeypatch)
    value._row_preparation = value._resident_workspaces["gate_up"].preparation
    value._resident_seals = tuple((tensor, value._seal(tensor)) for tensor in value._immutable_tensors())
    original = value._resident_workspaces
    value.configure_v3_probe(measurement=True, optimization="batched", compact=True)
    assert value.measurement_mode and value._resident_workspaces is original
    with pytest.raises(ValueError, match="FWHT"):
        value.configure_v3_probe(measurement=True, optimization="fwht")
    assert value._resident_workspaces is original


def test_v3_wrapper_refuses_missing_library_and_wrong_host_geometry(monkeypatch):
    monkeypatch.setattr(binding, "_require_loaded", lambda: None)
    with pytest.raises(ValueError, match="geometry"):
        binding.grouped_projection_out(
            torch.zeros(6, 12, dtype=torch.int64),
            torch.zeros(6656, dtype=torch.int32),
            [torch.ones(1)],
            jobs=6,
            m=1,
            n=32,
            k=513,
            tiles=1,
        )


def test_v3_source_keeps_namespace_and_does_not_enable_fwht():
    from pathlib import Path

    source = Path(v3.__file__).read_text(encoding="utf-8")
    assert "BatchedFWHTPreparation" not in source
    assert "grouped_projection_v3(inputs)" in source
    assert v3.AscendCV3VQ2TP1MoE.execution_policy == "ascendc_v3"
