# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Combined candidate wiring with real CPU Tensor math, not native proof."""

from types import SimpleNamespace as NS

import pytest
import torch

from tests.ut.quantization.test_vq2a8_activation_fused import TorchPreparationOps
from tests.ut.quantization.test_vq2a8_v4_graph import no_host_tensor_reads
from tests.ut.quantization.test_vq2a8_v4_v2_runtime import fake_runtime
from vllm_ascend.quantization.vq2a8_activation_fused import FusedV4V2Preparation
from vllm_ascend.quantization.vq2a8_optimization import configure_runtime
from vllm_ascend.quantization.vq2a8_v4_device_route import DeviceRouteGraphCompute


class ProjectionSelectionOracle:
    """Cheap deterministic projection, checking selection and Tensor contracts."""

    def __init__(self, width, output_width):
        self.scale = torch.ones(6, width)
        self.bias = torch.zeros(6, width)
        self.signs = torch.ones(6, width, dtype=torch.int8)
        self.output_width = output_width
        self.calls = []

    def select(self, slots):
        valid = (slots >= 0) & (slots < 6)
        safe = slots.clamp(0, 5)
        return self.scale[safe], self.bias[safe], self.signs[safe], valid.int()

    def _project(self, method, q, scale, bias, slots):
        self.calls.append((method, tuple(q.shape)))
        # No dense fake weights, and no claim of native projection accuracy.
        # Both methods consume q/scale/bias so skipping preparation is visible.
        value = (q.float().sum(-1) * scale + bias) / q.shape[-1]
        output = value[..., None].expand(*value.shape, self.output_width).bfloat16().clone()
        valid = (slots >= 0) & (slots < 6)
        mask = valid.reshape((slots.numel(),) + (1,) * (output.ndim - 1))
        return torch.where(mask, output, float("nan")), valid.int()

    def project(self, *args):
        return self._project("scalar", *args)

    def project_vectorized(self, *args):
        return self._project("vectorized", *args)


def runtime_and_ops(monkeypatch):
    runtime = fake_runtime()
    runtime.v4_activation_preparation = "fused"
    runtime.v4_activation_reorder = "vectorized"
    runtime.config.hidden_size = 2048
    runtime.root["gate.weight"] = torch.zeros(6, 2048)
    runtime._cache = {}
    runtime._v2_payload_locations = {}
    del runtime._row_preparation
    for kind in ("gate_up", "down"):
        n, k = (4096, 2048) if kind == "gate_up" else (2048, 2048)
        bank = ProjectionSelectionOracle(k, n)
        spec = NS(rows=n, columns=k, rht_true_columns=k, rht_block_size=128)
        runtime._device_route_banks[kind] = bank, spec
        for slot in range(6):
            payload = {"weight_scale": bank.scale[slot], "weight_bias": bank.bias[slot], "rht_sign": bank.signs[slot]}
            runtime._cache.setdefault(slot, {})[kind] = payload, spec
            runtime._v2_payload_locations[id(payload)] = kind, slot
    ops = TorchPreparationOps()
    monkeypatch.setattr(torch.ops, "vq2a8_ascendc_v4_v2", ops)
    return runtime, ops


def test_eager_prefill_and_pure_graph_compute_use_both_candidates(monkeypatch):
    runtime, ops = runtime_and_ops(monkeypatch)
    configure_runtime(runtime, "device_route_decode")
    assert isinstance(runtime._row_preparation, FusedV4V2Preparation)
    compute = DeviceRouteGraphCompute(runtime)
    assert all(isinstance(value, FusedV4V2Preparation) for value in compute.preparations.values())
    hidden = torch.randn(1, 2048).bfloat16()
    for token in (0, 1, 0):
        ids = torch.tensor([token])
        with no_host_tensor_reads(monkeypatch):
            eager = runtime._optimization.forward(runtime, hidden, ids)
            captured, valid = compute(hidden, ids)
        assert torch.equal(eager, captured) and bool(valid)
    requests = [(hidden.expand(count, -1), *runtime._cache[slot]["gate_up"]) for slot, count in ((0, 1), (4, 3))]
    with no_host_tensor_reads(monkeypatch):
        outputs = runtime._projections_many(requests)
    assert [tuple(output.shape) for output in outputs] == [(1, 4096), (3, 4096)]
    assert ops.calls and {name for name, _ in ops.calls} == {"sign", "quantize"}
    for kind in ("gate_up", "down"):
        assert all(method == "vectorized" for method, _ in runtime._device_route_banks[kind][0].calls)
    assert compute.signature["activation_reorder"] == "vectorized"
    assert compute.signature["activation_preparation"] == "fused"


@pytest.mark.parametrize(
    "option,replacement", [("v4_activation_reorder", "scalar"), ("v4_activation_preparation", "rowwise")]
)
def test_graph_contract_rejects_candidate_switch_after_capture(monkeypatch, option, replacement):
    runtime, _ = runtime_and_ops(monkeypatch)
    configure_runtime(runtime, "device_route_decode")
    compute = DeviceRouteGraphCompute(runtime)
    setattr(runtime, option, replacement)
    with pytest.raises(RuntimeError, match="signature changed"):
        compute.check_runtime_contract(runtime)


def test_combined_pure_compute_invalid_then_valid_not_stale(monkeypatch):
    runtime, _ = runtime_and_ops(monkeypatch)
    configure_runtime(runtime, "device_route_decode")
    compute = DeviceRouteGraphCompute(runtime)
    hidden = torch.ones(1, 2048).bfloat16()
    assert not bool(compute(hidden, torch.tensor([-1]))[1])
    assert bool(compute(hidden, torch.tensor([0]))[1])
    hidden[0, -1] = float("nan")
    assert not bool(compute(hidden, torch.tensor([0]))[1])
    hidden[0, -1] = 1
    assert bool(compute(hidden, torch.tensor([0]))[1])
