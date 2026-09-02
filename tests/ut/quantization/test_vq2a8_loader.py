# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import inspect
import json
import sys
import types

import pytest
import torch
from safetensors.torch import save_file
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

from vllm_ascend.quantization.vq2a8_config import AscendVQ2A8Config
from vllm_ascend.quantization.vq2a8_method import (
    ASCEND_VQ2_FIELDS,
    ASCEND_VQ2_KINDS,
    ASCEND_VQ2_TP_FORMAT,
    AscendVQ2A8MoEMethod,
    _validate_runtime_options,
    _wrap_fused_experts_result,
    create_compact_expert_parameters,
    load_repacked_layer,
)


def _repacked_tensors(num_experts: int) -> dict[str, torch.Tensor]:
    tensors = {}
    for kind in ASCEND_VQ2_KINDS:
        for field in ASCEND_VQ2_FIELDS:
            tensors[f"{kind}_{field}"] = torch.zeros(num_experts, 1)
    return tensors


def _write_repacked_layer(tmp_path, num_experts: int = 2) -> None:
    rank_path = tmp_path / "tp4" / "rank1"
    rank_path.mkdir(parents=True)
    metadata = {
        "format": ASCEND_VQ2_TP_FORMAT,
        "layer": 7,
        "tp_size": 4,
        "tp_rank": 1,
        "expert_ids": list(range(num_experts)),
        "complete": True,
        "rht_block_size": 4,
        "row_group_size": 2,
    }
    (rank_path / "experts_vq_layer_7.json").write_text(json.dumps(metadata), encoding="utf-8")
    save_file(
        _repacked_tensors(num_experts),
        rank_path / "experts_vq_layer_7.safetensors",
    )


def test_compact_parameters_do_not_allocate_dense_experts() -> None:
    layer = torch.nn.Module()
    create_compact_expert_parameters(layer, 256, torch.bfloat16, {})
    assert tuple(layer.w13_weight.shape) == (256, 0)
    assert tuple(layer.w2_weight.shape) == (256, 0)
    assert layer.w13_weight.numel() == 0
    assert layer.w2_weight.numel() == 0
    assert layer.vq_gate_up_packed_indices.numel() == 0


def test_config_registers_and_resolves_model_relative_paths(tmp_path) -> None:
    (tmp_path / "experts_vq").mkdir()
    config = AscendVQ2A8Config.from_config({"experts_path": "experts_vq"})
    config.maybe_update_config(str(tmp_path))
    assert "vq2a8" in QUANTIZATION_METHODS
    assert config.experts_path == str(tmp_path / "experts_vq")
    assert config.kernel_path == str(tmp_path / "experts_vq_ascend")
    assert config.allow_reference_fallback is True
    assert config.quant_description == {}
    assert not any("norm.bias" in name for name in config.quant_description)


def test_config_keeps_dense_linears_unquantized() -> None:
    config = AscendVQ2A8Config("unused")
    layer = object.__new__(LinearBase)
    method = config.get_quant_method(layer, "model.layers.0.self_attn.wq_a")

    assert isinstance(method, UnquantizedLinearMethod)


def test_load_repacked_layer_validates_and_loads_headers(tmp_path) -> None:
    _write_repacked_layer(tmp_path)
    tensors, metadata = load_repacked_layer(tmp_path, 7, 4, 1, 2)
    assert metadata["expert_ids"] == [0, 1]
    assert set(tensors) == {f"{kind}_{field}" for kind in ASCEND_VQ2_KINDS for field in ASCEND_VQ2_FIELDS}


def test_load_repacked_layer_rejects_wrong_expert_count(tmp_path) -> None:
    _write_repacked_layer(tmp_path)
    with pytest.raises(ValueError, match="has 2 experts, expected 256"):
        load_repacked_layer(tmp_path, 7, 4, 1, 256)


def test_load_repacked_layer_accepts_compact_hash_experts(tmp_path) -> None:
    _write_repacked_layer(tmp_path, num_experts=1)
    tensors, metadata = load_repacked_layer(
        tmp_path,
        7,
        4,
        1,
        expected_experts=256,
        allow_compact_experts=True,
    )
    assert metadata["expert_ids"] == [0]
    assert tensors["gate_up_packed_indices"].shape[0] == 1


def test_config_can_require_compiled_operators() -> None:
    config = AscendVQ2A8Config.from_config(
        {
            "experts_path": "experts_vq",
            "allow_reference_fallback": False,
        }
    )
    assert config.allow_reference_fallback is False


def test_profile_force_load_balance_is_not_treated_as_eplb() -> None:
    _validate_runtime_options(
        enable_force_load_balance=True,
        log2phy=None,
        global_redundant_expert_num=0,
        mc2_mask=None,
    )


@pytest.mark.parametrize(
    ("log2phy", "global_redundant_expert_num", "mc2_mask"),
    [
        (torch.empty(0), 0, None),
        (None, 1, None),
        (None, 0, torch.empty(0)),
    ],
)
def test_runtime_guard_still_rejects_eplb_and_mc2(
    log2phy: torch.Tensor | None,
    global_redundant_expert_num: int,
    mc2_mask: torch.Tensor | None,
) -> None:
    with pytest.raises(NotImplementedError, match="without EPLB or MC2"):
        _validate_runtime_options(
            enable_force_load_balance=True,
            log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            mc2_mask=mc2_mask,
        )


def test_vq2_output_uses_fused_experts_result_protocol(monkeypatch) -> None:
    module_name = "vllm_ascend.ops.fused_moe.moe_comm_method"
    fake_module = types.ModuleType(module_name)

    class FakeFusedExpertsResult:
        def __init__(self, *, routed_out: torch.Tensor) -> None:
            self.routed_out = routed_out

    fake_module.FusedExpertsResult = FakeFusedExpertsResult
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    output = torch.arange(4)

    result = _wrap_fused_experts_result(output)

    assert result.routed_out is output


def test_vq2_quant_method_defers_tp_reduce_to_moe_runner() -> None:
    source = inspect.getsource(AscendVQ2A8MoEMethod.apply)

    assert "tensor_model_parallel_all_reduce" not in source
    assert "tp_reduction_deferred" in source
