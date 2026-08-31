# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json

import pytest
import torch
from safetensors.torch import save_file
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

from vllm_ascend.quantization.vq2a8_config import AscendVQ2A8Config
from vllm_ascend.quantization.vq2a8_method import (
    ASCEND_VQ2_FIELDS,
    ASCEND_VQ2_KINDS,
    ASCEND_VQ2_TP_FORMAT,
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
    }
    (rank_path / "experts_vq_layer_7.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
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


def test_load_repacked_layer_validates_and_loads_headers(tmp_path) -> None:
    _write_repacked_layer(tmp_path)
    tensors, metadata = load_repacked_layer(tmp_path, 7, 4, 1, 2)
    assert metadata["expert_ids"] == [0, 1]
    assert set(tensors) == {
        f"{kind}_{field}"
        for kind in ASCEND_VQ2_KINDS
        for field in ASCEND_VQ2_FIELDS
    }


def test_load_repacked_layer_rejects_wrong_expert_count(tmp_path) -> None:
    _write_repacked_layer(tmp_path)
    with pytest.raises(ValueError, match="has 2 experts, expected 256"):
        load_repacked_layer(tmp_path, 7, 4, 1, 256)
