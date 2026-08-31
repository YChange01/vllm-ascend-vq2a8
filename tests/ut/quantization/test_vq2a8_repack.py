# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json

import torch
from safetensors.torch import save_file

from vllm_ascend.quantization.vq2a8_format import (
    decode_expert_weight,
    decode_repacked_vq2_weight,
    shard_decoded_expert_weight,
)
from vllm_ascend.quantization.vq2a8_method import load_repacked_layer
from vllm_ascend.quantization.vq2a8_repack import repack_layer


def _pack_indices(indices: torch.Tensor) -> torch.Tensor:
    words = torch.zeros((indices.numel() + 7) // 8, dtype=torch.int64)
    for index, value in enumerate(indices.tolist()):
        words[index // 8] |= value << (index % 8 * 4)
    return words.to(torch.int32)


def _matrix_metadata() -> dict:
    return {
        "rows": 8,
        "cols": 8,
        "n_row_tiles": 4,
        "n_col_tiles": 2,
        "row_group_size": 2,
        "group_size": 4,
        "K": 16,
        "index_bits": 4,
        "vector_len": 2,
        "n_vectors": 32,
        "n_elements": 64,
        "orig_shape": [8, 8],
        "norm_dim": 0,
        "enable_perm": True,
        "enable_norm": True,
        "enable_rht": True,
        "rht_block_size": 4,
        "rht_true_columns": 8,
    }


def _canonical_tensors() -> dict[str, torch.Tensor]:
    codebooks = torch.empty(2, 4, 16, 2)
    for column_tile in range(2):
        for row_tile in range(4):
            for code in range(16):
                codebooks[column_tile, row_tile, code] = torch.tensor(
                    [code + row_tile, code - column_tile]
                )
    return {
        "packed_indices": _pack_indices(torch.arange(32).remainder(16)),
        "codebooks": codebooks,
        "perm": torch.tensor([3, 0, 7, 2, 5, 1, 6, 4], dtype=torch.int32),
        "weight_scale": torch.linspace(0.5, 1.25, 8),
        "weight_bias": torch.linspace(-0.25, 0.5, 8),
        "rht_sign": torch.tensor([1, -1, 1, 1, -1, 1, -1, 1], dtype=torch.int8),
    }


def _write_canonical_layer(path) -> None:
    metadata = _matrix_metadata()
    entries = {
        "0.mlp.experts.0.gate_up": metadata,
        "0.mlp.experts.0.down": metadata,
    }
    (path / "experts_vq_layer_0.json").write_text(
        json.dumps(entries), encoding="utf-8"
    )
    tensors = {}
    for name in entries:
        tensors.update(
            {f"{name}.{field}": value for field, value in _canonical_tensors().items()}
        )
    save_file(tensors, path / "experts_vq_layer_0.safetensors")


def test_repack_tp_shards_decode_to_canonical_weights(tmp_path) -> None:
    source_path = tmp_path / "canonical"
    output_path = tmp_path / "ascend"
    source_path.mkdir()
    _write_canonical_layer(source_path)
    repack_layer(source_path, output_path, layer_index=0, tp_size=2)

    canonical = _canonical_tensors()
    metadata = _matrix_metadata()
    for tp_rank in range(2):
        payload, repack_metadata = load_repacked_layer(
            output_path, 0, 2, tp_rank, expected_experts=1
        )
        for kind in ("gate_up", "down"):
            decoded = decode_repacked_vq2_weight(
                payload[f"{kind}_packed_indices"][0],
                payload[f"{kind}_codebooks"][0],
                payload[f"{kind}_codebook_tile_ids"][0],
                payload[f"{kind}_weight_scale"][0],
                payload[f"{kind}_weight_bias"][0],
                payload[f"{kind}_rht_sign"][0],
                repack_metadata["rht_block_size"],
                repack_metadata["row_group_size"],
            )
            full = decode_expert_weight(canonical, metadata, "cpu")
            expected = shard_decoded_expert_weight(
                full, f"0.mlp.experts.0.{kind}", 2, tp_rank
            )
            torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
