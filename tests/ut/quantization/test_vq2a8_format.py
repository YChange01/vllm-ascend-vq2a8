# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.quantization.vq2a8_format import (
    apply_vq2_weight_transforms,
    decode_codebook_weight,
    decode_expert_weight,
    inspect_vq2_directory,
    parse_expert_name,
    transform_vq2_activation,
    unpack_vq2_indices,
    vq2_matmul_reference,
)


def _pack_vq2_indices(values: torch.Tensor) -> torch.Tensor:
    words = torch.zeros((values.numel() + 7) // 8, dtype=torch.int64)
    for index, value in enumerate(values.tolist()):
        words[index // 8] |= value << (index % 8 * 4)
    return words.to(torch.int32)


def _metadata(rows: int = 4, columns: int = 4) -> dict:
    return {
        "rows": rows,
        "cols": columns,
        "n_row_tiles": 2,
        "n_col_tiles": 1,
        "row_group_size": 2,
        "group_size": columns,
        "K": 16,
        "index_bits": 4,
        "vector_len": 2,
        "n_vectors": rows * columns // 2,
        "n_elements": rows * columns,
        "orig_shape": [rows, columns],
        "norm_dim": 0,
        "enable_perm": True,
        "enable_norm": True,
        "enable_rht": True,
        "rht_block_size": 4,
        "rht_true_columns": columns,
    }


def _tensors() -> dict[str, torch.Tensor]:
    indices = torch.arange(8, dtype=torch.int64)
    codebooks = torch.empty((1, 2, 16, 2), dtype=torch.float32)
    for index in range(16):
        codebooks[0, 0, index] = torch.tensor([index + 0.25, index - 0.5])
        codebooks[0, 1, index] = torch.tensor([index + 1.0, index + 2.0])
    return {
        "packed_indices": _pack_vq2_indices(indices),
        "codebooks": codebooks,
        "perm": torch.tensor([2, 0, 3, 1], dtype=torch.int32),
        "weight_scale": torch.tensor([1.25, 0.75, 1.5, 0.5]),
        "weight_bias": torch.tensor([0.5, -0.25, 0.125, 1.0]),
        "rht_sign": torch.tensor([1, -1, 1, 1], dtype=torch.int8),
    }


def test_unpack_vq2_indices_round_trips_nibbles() -> None:
    expected = torch.arange(97, dtype=torch.int64).remainder(16)
    actual = unpack_vq2_indices(
        _pack_vq2_indices(expected), expected.numel(), "cpu"
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_direct_reference_matches_fully_decoded_weight() -> None:
    metadata = _metadata()
    tensors = _tensors()
    activation = torch.tensor(
        [[0.5, -1.0, 2.0, 0.25], [1.5, 0.75, -0.5, 2.0]]
    )
    decoded = decode_expert_weight(tensors, metadata, "cpu").float()
    expected = activation @ decoded.transpose(0, 1)
    actual = vq2_matmul_reference(activation, tensors, metadata)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)


def test_activation_transform_matches_weight_transform_in_fp32() -> None:
    metadata = _metadata()
    tensors = _tensors()
    activation = torch.randn(3, 4)
    codebook_weight = decode_codebook_weight(tensors, metadata, "cpu")
    decoded = apply_vq2_weight_transforms(codebook_weight, tensors, metadata)
    transformed, correction = transform_vq2_activation(
        activation, tensors, metadata
    )
    actual = transformed @ codebook_weight.transpose(0, 1)
    assert correction is not None
    actual += correction.unsqueeze(-1)
    torch.testing.assert_close(actual, activation @ decoded.transpose(0, 1))


def test_inspect_directory_validates_expert_pairs_and_tensor_headers(
    tmp_path,
) -> None:
    metadata = _metadata()
    matrices = {
        "0.mlp.experts.0.gate_up": metadata,
        "0.mlp.experts.0.down": metadata,
    }
    (tmp_path / "experts_vq_layer_0.json").write_text(
        json.dumps(matrices), encoding="utf-8"
    )
    tensors = {}
    for name in matrices:
        tensors.update({f"{name}.{field}": value for field, value in _tensors().items()})
    save_file(tensors, tmp_path / "experts_vq_layer_0.safetensors")

    summaries = inspect_vq2_directory(tmp_path)
    assert len(summaries) == 1
    assert summaries[0].expert_ids == (0,)
    assert summaries[0].matrix_count == 2
    assert summaries[0].tensor_count == 12


def test_inspect_directory_rejects_missing_matrix_kind(tmp_path) -> None:
    name = "0.mlp.experts.0.gate_up"
    (tmp_path / "experts_vq_layer_0.json").write_text(
        json.dumps({name: _metadata()}), encoding="utf-8"
    )
    save_file(
        {f"{name}.{field}": value for field, value in _tensors().items()},
        tmp_path / "experts_vq_layer_0.safetensors",
    )
    with pytest.raises(ValueError, match="expected.*gate_up.*down"):
        inspect_vq2_directory(tmp_path)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("3.mlp.experts.17.gate_up", (3, 17, "gate_up")),
        ("42.mlp.experts.255.down", (42, 255, "down")),
    ],
)
def test_parse_expert_name(name: str, expected: tuple[int, int, str]) -> None:
    assert parse_expert_name(name) == expected
