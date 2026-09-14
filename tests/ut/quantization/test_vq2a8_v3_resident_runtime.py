# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU conversion/preparation with explicit stand-ins for NPU launchers."""

import math
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_ascend.quantization import vq2a8_execution_v3 as runtime
from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_ascendc_v2 import convert_expert_payload, gather_prepared_activation
from vllm_ascend.quantization.vq2a8_reference import VQ2_FP8_MIN_SCALE
from vllm_ascend.quantization.vq2a8_v3_workspace import ResidentV2ProjectionWorkspace, resident_shapes


def _converted_case(k, experts=3):
    n = 4096
    spec = NS(rows=n, columns=k, rht_true_columns=k, rht_block_size=128)
    sources, converted = [], []
    for expert in range(experts):
        rng = np.random.default_rng(k + expert)
        words = rng.integers(0, 256, (n // 2, k // 8, 4), dtype=np.uint8).view(np.int32).reshape(n // 2, k // 8)
        books = rng.integers(0, 256, (k // 256, n // 32, 16, 2), dtype=np.uint8)
        books[(books & 127) == 127] = 128
        ids = np.repeat(np.arange(k // 256, dtype=np.uint8), 256)
        rng.shuffle(ids)
        source = {
            "packed_indices": torch.from_numpy(words),
            "codebooks": torch.from_numpy(books).view(torch.float8_e4m3fn),
            "codebook_tile_ids": torch.from_numpy(ids),
            "weight_scale": torch.linspace(0.125 + expert / 8, 1 + expert / 8, k),
            "weight_bias": torch.linspace(-0.125, 0.25 + expert / 32, k),
            "rht_sign": torch.where((torch.arange(k) + expert) % 3 == 0, -1, 1).to(torch.int8),
        }
        sources.append(source)
        converted.append(convert_expert_payload(source, spec))
    banks = {field: torch.stack([payload[field] for payload in converted]) for field in converted[0]}
    payloads = [{field: bank[expert] for field, bank in banks.items()} for expert in range(experts)]
    return NS(spec=spec, banks=banks, payloads=payloads, sources=sources)


@pytest.fixture(scope="module")
def cases():
    return {k: _converted_case(k) for k in (2048, 4096)}


def _fake_projection(descriptors, owners, **geometry):
    # This deliberately tests workspace plumbing, never native pointer access.
    x, scale, bias, output = owners[1:5]
    assert descriptors.shape == (geometry["jobs"], 9)
    assert (geometry["m"], geometry["n"], geometry["k"]) == (1, 4096, x.shape[1])
    values = x.float().sum(1) * scale + bias
    output.copy_(values[:, None].expand_as(output).bfloat16())


def _fake_prepare(rotated, weight_scale, order, input_bias, quantized, scale, bias, valid):
    # Real Torch arithmetic oracle replacing only the unexecutable AscendC op.
    transformed = rotated * weight_scale
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    expected_scale = torch.clamp(transformed.abs().amax(dim=-1) / fp8_max, min=VQ2_FP8_MIN_SCALE)
    raw = torch.clamp(transformed / expected_scale.unsqueeze(-1), -fp8_max, fp8_max).to(torch.float8_e4m3fn)
    torch.gather(raw.view(torch.uint8), 1, order, out=quantized.view(torch.uint8))
    scale.copy_(expected_scale)
    bias.copy_(input_bias)
    valid.fill_(1)


def _workspace(
    case, *, jobs=6, true_width=None, mode="eager", flags=None, prepare_launcher=_fake_prepare, preparation=None
):
    spec = NS(**vars(case.spec))
    if true_width is not None:
        spec.rht_true_columns = true_width
    if preparation is None:
        preparation = RowwiseVQ2A8Preparation(compact=True, validity=(flags if flags is not None else []).append)
    preparation._ensure_hadamard(case.banks["weight_scale"].device, spec.rht_block_size)
    return ResidentV2ProjectionWorkspace(
        case.payloads,
        spec,
        jobs,
        preparation,
        banks=case.banks,
        preparation_mode=mode,
        launcher=_fake_projection,
        prepare_launcher=prepare_launcher,
    )


def _assert_reference_preparation(workspace, hidden, slots):
    requests = [
        (
            hidden if hidden.shape[0] == 1 else hidden[row : row + 1],
            workspace.payloads[expert],
            workspace.spec,
        )
        for row, expert in enumerate(slots.tolist())
    ]
    expected = RowwiseVQ2A8Preparation(compact=True).many(requests)
    for row, ((quantized, scale, bias), expert) in enumerate(zip(expected, slots.tolist())):
        gathered = gather_prepared_activation(quantized, workspace.payloads[expert]["activation_order"])
        assert torch.equal(workspace.x[row : row + 1].view(torch.uint8), gathered.view(torch.uint8))
        assert torch.equal(workspace.scale[row : row + 1], scale)
        assert torch.equal(workspace.bias[row : row + 1], bias)
        assert torch.equal(workspace.descriptors[row, 3:5], workspace.pointer_bank[expert])
        assert workspace.descriptors[row, 0] == workspace.x[row].data_ptr()
        assert workspace.descriptors[row, 1] == workspace.scale[row:].data_ptr()
        assert workspace.descriptors[row, 2] == workspace.bias[row:].data_ptr()
        assert workspace.descriptors[row, 5] == workspace.output[row].data_ptr()
        assert torch.equal(workspace.descriptors[row, 6:], torch.tensor([1, 4096, workspace.k]))


@pytest.mark.parametrize("k", [2048, 4096])
@pytest.mark.parametrize("rows,padding", [(1, 0), (6, 4)])
@pytest.mark.parametrize("mode", ["eager", "fused"])
def test_converted_workspace_matches_v2_preparation_for_duplicate_dynamic_slots(cases, k, rows, padding, mode):
    workspace = _workspace(cases[k], true_width=k - padding, mode=mode)
    hidden = torch.randn(rows, k - padding, generator=torch.Generator().manual_seed(124)).bfloat16()
    addresses = tuple(tensor.data_ptr() for tensor in (*workspace.owners, workspace.descriptors))
    for slots, multiplier in (([2, 0, 2, 1, 0, 2], 1), ([0, 2, 1, 0, 2, 0], 2)):
        slots = torch.tensor(slots)
        current = hidden * multiplier
        output = workspace.project(current, slots)
        assert output is workspace.output
        _assert_reference_preparation(workspace, current, slots)
        assert addresses == tuple(tensor.data_ptr() for tensor in (*workspace.owners, workspace.descriptors))


@pytest.mark.parametrize("case_name", ["zero", "large", "rounding"])
def test_workspace_fp8_boundaries_keep_quantization_before_permutation(cases, case_name):
    workspace = _workspace(cases[2048])
    hidden = torch.randn(6, 2048, generator=torch.Generator().manual_seed(501)).bfloat16()
    if case_name == "zero":
        hidden.zero_()
    elif case_name == "large":
        hidden *= 256
    else:
        hidden = (hidden.float() / 32 + 1.0625).bfloat16()
    slots = torch.tensor([2, 1, 2, 0, 1, 0])
    workspace.project(hidden, slots)
    _assert_reference_preparation(workspace, hidden, slots)


def test_fused_launcher_receives_owned_outputs_and_retains_native_failure(cases):
    flags, calls = [], []

    def preparation(*args):
        calls.append(args)
        _fake_prepare(*args)
        args[-1][2] = 0

    workspace = _workspace(cases[2048], mode="fused", flags=flags, prepare_launcher=preparation)
    workspace.project(torch.ones(1, 2048).bfloat16(), torch.tensor([2, 0, 2, 1, 0, 1]))
    assert len(calls) == 1
    expected = (
        workspace.rotated,
        workspace.selected["weight_scale"],
        workspace.selected["activation_order"],
        workspace.input_bias,
        workspace.x,
        workspace.scale,
        workspace.bias,
        workspace.valid,
    )
    assert all(actual is wanted for actual, wanted in zip(calls[0], expected))
    assert bool(flags[0])
    assert not bool(flags[-1])
    assert workspace.valid.tolist() == [1, 1, 0, 1, 1, 1]
    owner_storage = {tensor.untyped_storage().data_ptr() for tensor in workspace.owners}
    assert all(tensor.untyped_storage().data_ptr() in owner_storage for tensor in expected)


@pytest.mark.parametrize("error", ["width", "rows", "dtype", "slot_dtype", "slot_shape", "slot_range"])
def test_workspace_rejects_invalid_inputs_before_projection(cases, error):
    workspace = _workspace(cases[2048])
    calls = []
    workspace.launcher = lambda *args, **kwargs: calls.append((args, kwargs))
    hidden, slots = torch.ones(1, 2048).bfloat16(), torch.tensor([0, 1, 2, 0, 1, 2])
    if error == "width":
        hidden = hidden[:, :-1]
    elif error == "rows":
        hidden = hidden.expand(2, -1)
    elif error == "dtype":
        hidden = hidden.float()
    elif error == "slot_dtype":
        slots = slots.int()
    elif error == "slot_shape":
        slots = slots[:-1]
    else:
        slots[0] = 3
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        workspace.project(hidden, slots)
    assert not calls


def _inventory(cases):
    specs = {"gate_up": cases[4096].spec, "down": cases[2048].spec}
    shapes = {
        f"{kind}_{field}": (3, *tensor.shape)
        for kind, k in (("gate_up", 4096), ("down", 2048))
        for field, tensor in cases[k].sources[0].items()
    }
    return NS(layer_index=3, expert_ids=(0, 1, 2), specs=specs, tensor_shapes=shapes)


def _rounded_storage(tensors):
    storages = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes() for tensor in tensors}
    return sum(
        math.ceil(size / runtime.ALLOCATION_GRANULARITY) * runtime.ALLOCATION_GRANULARITY for size in storages.values()
    )


def test_resident_plan_matches_converted_banks_and_every_persistent_workspace_allocation(cases):
    layer = _inventory(cases)
    plan = runtime.resident_plan([layer], 1 << 40, kernel="v2")
    preparation = RowwiseVQ2A8Preparation(compact=True, validity=lambda valid: None)
    workspaces = [_workspace(cases[k], preparation=preparation) for k in (4096, 2048)]
    banks = [bank for workspace in workspaces for bank in workspace.banks.values()]
    payload_bytes = _rounded_storage(banks)
    lookup = torch.empty(256, dtype=torch.int64)
    validity = torch.ones((), dtype=torch.bool)
    all_owners = [owner for workspace in workspaces for owner in (*workspace.owners, workspace.descriptors)]
    all_owners.extend((preparation._hadamard, lookup, validity))
    workspace_bytes = _rounded_storage(all_owners) - payload_bytes
    assert plan["payload_bytes"] == payload_bytes
    assert plan["workspace_bytes"] == workspace_bytes
    assert plan["planned_bytes"] == payload_bytes + workspace_bytes
    assert plan["layout"] == "zn_pair_lut_k256"
    assert runtime.resident_plan([layer], plan["planned_bytes"], kernel="v2")["all_experts_fit"]
    with pytest.raises(ValueError, match="no cache fallback"):
        runtime.resident_plan([layer], plan["planned_bytes"] - 1, kernel="v2")
    for kind, workspace in zip(("gate_up", "down"), workspaces):
        assert resident_shapes(layer, kind) == {field: tuple(bank.shape) for field, bank in workspace.banks.items()}


def _uninitialized_runtime(cases, monkeypatch):
    value = runtime.AscendCV3VQ2TP1MoE.__new__(runtime.AscendCV3VQ2TP1MoE)
    value.layer, value.layer_index, value.device = _inventory(cases), 3, torch.device("cpu")
    value.config = NS(hidden_size=4096, top_k=6, renormalize=True, num_shared=0, swiglu_limit=7.0, routed_scale=1.5)
    value.root = {
        "gate.weight": torch.zeros(3, 4096),
        "gate.tid2eid": torch.tensor([[2, 0, 2, 1, 0, 1], [0, 1, 0, 2, 1, 2]]),
    }
    value.projection_kernel, value.v3_preparation, value.v3_decode_graph = "v2", "eager", "none"
    value._resident_ready = value._resident_failed = value._resident_busy = False
    value._resident_stream = value._resident_valid = value._decode_graph = None
    value._resident_workspaces, value._cache = {}, {}
    value.progress = False
    value.cache_loads = value.cache_hits = value.cache_peak_bytes = value.h2d_bytes = 0
    value.native_calls = value.native_rows = value.native_experts = value.native_launches = 0
    value.projection_rows = value.prepare_batches = value.decode_device_calls = value.prefill_legacy_calls = 0
    value._v3_prefill_state = runtime.FastMoEState(value, runtime.OptimizationOptions.preset("batched"))
    loads = []

    def load_expert(layer_index, expert, kind, *, device):
        assert layer_index == 3 and device == "cpu"
        loads.append((expert, kind))
        case = cases[4096 if kind == "gate_up" else 2048]
        return {field: tensor.clone() for field, tensor in case.sources[expert].items()}, case.spec

    value.artifact = NS(load_expert=load_expert)
    monkeypatch.setattr(runtime, "resident_library_capabilities", lambda: 3)
    monkeypatch.setattr(torch, "npu", NS(current_stream=lambda *args: NS(npu_stream=7)), raising=False)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    return value, loads


@pytest.mark.parametrize("inference", [False, True])
def test_actual_resident_initialization_converts_once_and_updates_device_selection(cases, monkeypatch, inference):
    value, loads = _uninitialized_runtime(cases, monkeypatch)
    conversions = []

    def convert(source, spec):
        conversions.append(spec.columns)
        return convert_expert_payload(source, spec)

    monkeypatch.setattr(runtime, "convert_expert_payload", convert)
    with torch.inference_mode(inference):
        report = value.initialize_resident(budget_bytes=1 << 30)
    assert report["ready"] and report["layout"] == "zn_pair_lut_k256"
    assert conversions == [4096, 2048] * 3
    assert loads == [(expert, kind) for expert in range(3) for kind in ("gate_up", "down")]
    for kind, k in (("gate_up", 4096), ("down", 2048)):
        workspace = value._resident_workspaces[kind]
        workspace.launcher = _fake_projection
        for field, expected in cases[k].banks.items():
            bank = workspace.banks[field]
            assert torch.equal(bank.view(torch.uint8), expected.view(torch.uint8))
            assert bank._version >= 3
            for expert in range(3):
                assert value._cache[expert][kind][0][field].data_ptr() == bank[expert].data_ptr()
    flag = value._resident_valid
    flag_pointer = flag.data_ptr()
    recorded = []
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: recorded.append(tensor.data_ptr()))
    for token in (0, 1):
        output = value._forward(torch.ones(1, 4096).bfloat16(), torch.tensor([token]))
        assert output.shape == (1, 4096) and torch.isfinite(output).all()
        slots = value.root["gate.tid2eid"][token]
        for workspace in value._resident_workspaces.values():
            assert torch.equal(workspace.descriptors[:, 3:5], workspace.pointer_bank.index_select(0, slots))
    assert flag_pointer in recorded
    assert len(loads) == len(conversions) == 6
    assert value.decode_device_calls == 2 and value.native_launches == 4
    value._retain_valid(torch.tensor(False))
    value._retain_valid(torch.tensor(True))
    assert value._resident_valid is flag and not bool(flag)
    value.configure_v3_probe(measurement=True)
    assert value._resident_valid.data_ptr() == flag_pointer and bool(value._resident_valid)
    with pytest.raises(RuntimeError, match="fresh empty"):
        value.initialize_resident(budget_bytes=1 << 30)


def test_prefill_uses_converted_native_layout_and_original_preparation(cases, monkeypatch):
    value, _ = _uninitialized_runtime(cases, monkeypatch)
    value.initialize_resident(budget_bytes=1 << 30)
    workspace = value._resident_workspaces["down"]
    generator = torch.Generator().manual_seed(81)
    requests = [
        (torch.randn(rows, 2048, generator=generator).bfloat16(), workspace.payloads[expert], workspace.spec)
        for rows, expert in ((2, 2), (3, 0))
    ]
    expected = RowwiseVQ2A8Preparation(compact=True).many(requests)
    calls = []

    def projection(inputs):
        calls.append(inputs)
        for (quantized, scale, bias, packed, table), reference, (_, payload, _) in zip(inputs, expected, requests):
            raw, reference_scale, reference_bias = reference
            gathered = gather_prepared_activation(raw, payload["activation_order"])
            assert torch.equal(quantized.view(torch.uint8), gathered.view(torch.uint8))
            assert torch.equal(scale, reference_scale) and torch.equal(bias, reference_bias)
            assert packed is payload["packed_zn"] and table is payload["pair_lut"]
        return [torch.ones(row[0].shape[0], 4096).bfloat16() for row in inputs]

    monkeypatch.setattr(runtime, "grouped_projection_resident", projection)
    monkeypatch.setattr(runtime, "grouped_projection_v3", lambda inputs: pytest.fail("legacy layout was selected"))
    output = value._projections_many(requests)
    assert [tuple(tensor.shape) for tensor in output] == [(2, 4096), (3, 4096)]
    assert len(calls) == value.resident_prefill_launches == value.native_launches == 1


def test_invalid_conversion_is_terminal_before_any_native_projection(cases, monkeypatch):
    value, _ = _uninitialized_runtime(cases, monkeypatch)
    original = value.artifact.load_expert

    def bad_population(*args, **kwargs):
        source, spec = original(*args, **kwargs)
        source["codebook_tile_ids"][0] = (source["codebook_tile_ids"][0] + 1) % (spec.columns // 256)
        return source, spec

    value.artifact.load_expert = bad_population
    with pytest.raises(ValueError, match="homogeneous K256"):
        value.initialize_resident(budget_bytes=1 << 30)
    assert value._resident_failed and not value._resident_ready
    assert not value._cache and not value._resident_workspaces and value.native_launches == 0
    with pytest.raises(RuntimeError, match="fresh empty"):
        value.initialize_resident(budget_bytes=1 << 30)
