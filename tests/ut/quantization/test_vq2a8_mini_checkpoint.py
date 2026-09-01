# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch

from tools.create_vq2a8_mini_checkpoint import (
    mini_config,
    selected_model_weights,
    slice_router_tensor,
)


def test_mini_config_updates_depth_experts_and_quantization_paths() -> None:
    source = {
        "num_hidden_layers": 43,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "num_hash_layers": 3,
        "num_nextn_predict_layers": 1,
        "compress_ratios": [0, 0, 4, 128, 4],
        "quantization_config": {
            "quant_method": "vq2a8",
            "experts_path": "experts_vq",
        },
    }
    actual = mini_config(source, num_layers=4, num_experts=8)
    assert actual["num_hidden_layers"] == 4
    assert actual["n_routed_experts"] == 8
    assert actual["num_experts_per_tok"] == 6
    assert actual["num_hash_layers"] == 3
    assert actual["num_nextn_predict_layers"] == 0
    assert actual["compress_ratios"] == [0, 0, 4, 128]
    assert actual["quantization_config"]["kernel_path"] == "experts_vq_ascend"


def test_selected_model_weights_keeps_globals_and_prefix_layers() -> None:
    weight_map = {
        "embed.weight": "a",
        "layers.0.attn.weight": "a",
        "layers.3.ffn.gate.weight": "b",
        "layers.4.attn.weight": "b",
        "norm.weight": "c",
    }
    assert selected_model_weights(weight_map, 4) == {
        "embed.weight": "a",
        "layers.0.attn.weight": "a",
        "layers.3.ffn.gate.weight": "b",
        "norm.weight": "c",
    }


def test_slice_router_tensor_reduces_router_outputs() -> None:
    weight = torch.arange(32).reshape(8, 4)
    actual = slice_router_tensor("layers.3.ffn.gate.weight", weight, 8, 3)
    torch.testing.assert_close(actual, weight[:3])


def test_slice_router_tensor_rejects_out_of_range_hash_route() -> None:
    routes = torch.tensor([[0, 1], [2, 7]])
    with pytest.raises(ValueError, match="outside the requested"):
        slice_router_tensor("layers.0.ffn.gate.tid2eid", routes, 8, 3)
