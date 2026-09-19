# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU preparation with an explicit projection oracle, not native evidence."""

from types import SimpleNamespace

import pytest
import torch

from tests.ut.quantization.test_vq2a8_activation_direct import StridedSignOracle
from tests.ut.quantization.test_vq2a8_activation_packed import assert_bytes, fixture, reference
from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_activation_packed import PackedRowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_v4_device_route import DeviceRouteDecodeState, DeviceRouteGraphCompute
from vllm_ascend.quantization.vq2a8_v4_v2 import AscendCV4V2VQ2TP1MoE


class ProjectionInputOracle:
    """Record actual preparation inputs while returning a tiny CPU projection."""

    def __init__(self, scale, bias, signs):
        self.metadata = (scale, bias, signs)
        self.inputs = None

    def select(self, slots):
        return (*(value.index_select(0, slots) for value in self.metadata), torch.ones_like(slots, dtype=torch.int32))

    def project(self, quantized, scale, bias, slots):
        self.inputs = (quantized, scale, bias)
        output = (quantized.float().sum(-1) * scale + bias).reshape(-1, 1).bfloat16()
        return output, torch.ones_like(slots, dtype=torch.int32)


@pytest.mark.parametrize("mode", ["rowwise", "rowwise_packed", "sign_fused", "sign_fused_strided", "sign_fused_direct"])
@pytest.mark.parametrize("path", ["eager", "graph_compute"])
@pytest.mark.parametrize("invalid_sign", [False, True])
def test_device_projection_uses_selected_preparation_and_retains_each_input_flag(monkeypatch, mode, path, invalid_sign):
    native = StridedSignOracle()
    monkeypatch.setattr(torch.ops, "vq2a8_ascendc_v4_v2", native)
    factory_owner = AscendCV4V2VQ2TP1MoE.__new__(AscendCV4V2VQ2TP1MoE)
    factory_owner.v4_activation_preparation = mode
    factory_owner.v4_activation_reorder = "scalar"
    flags = []
    preparation = factory_owner.make_v4_preparation(compact=True, validity=flags.append)
    if mode == "rowwise":
        assert type(preparation) is RowwiseVQ2A8Preparation
    else:
        assert isinstance(preparation, PackedRowwiseVQ2A8Preparation)
        assert preparation.fuse_sign is mode.startswith("sign_fused")
        assert preparation.strided_sign is (mode in ("sign_fused_strided", "sign_fused_direct"))
        assert preparation.direct_output is (mode == "sign_fused_direct")

    hidden, scale, bias, signs, spec = fixture(2048, 3)
    hidden = hidden[:1].expand(3, -1)
    slots = torch.tensor([2, 0, 2], dtype=torch.int64)
    if invalid_sign:
        signs[2, -1] = 0
    bank = ProjectionInputOracle(scale, bias, signs)
    runtime = SimpleNamespace(
        v4_activation_preparation=mode,
        _row_preparation=preparation,
        _device_route_banks={"gate_up": (bank, spec)},
        project_v4_prepared=factory_owner.project_v4_prepared,
        native_calls=0,
        native_rows=0,
        native_launches=0,
        projection_rows=0,
        prepare_batches=0,
    )
    if path == "eager":
        state = DeviceRouteDecodeState.__new__(DeviceRouteDecodeState)
        state.valid, state.profile = None, False
        state.stats = {"device_select_calls": 0, "preparation_calls": 0}
        preparation.validity = state.retain
        output = state._project(runtime, hidden, slots, "gate_up")
        flags.append(state.valid)
        assert state.stats == {"device_select_calls": 1, "preparation_calls": 1}
        assert runtime.native_launches == 1 and runtime.prepare_batches == 3
    else:
        compute = DeviceRouteGraphCompute.__new__(DeviceRouteGraphCompute)
        compute.runtime = runtime
        compute.banks = runtime._device_route_banks
        compute.preparations = {"gate_up": preparation}
        output = compute._project(hidden, slots, "gate_up", flags.append)
    assert output.shape == (3, 1) and output.dtype == torch.bfloat16
    assert all(bool(flag) for flag in flags) is (not invalid_sign)
    assert len(native.calls) == (1 if mode.startswith("sign_fused") else 0)
    if not invalid_sign:
        selected = tuple(value.index_select(0, slots) for value in (scale, bias, signs))
        expected = reference(hidden, *selected, spec)
        for actual_value, expected_value in zip(bank.inputs, expected):
            assert_bytes(actual_value, expected_value)
