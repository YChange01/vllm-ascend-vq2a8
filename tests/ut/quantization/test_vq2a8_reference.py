# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import pytest
import torch

from vllm_ascend.quantization.vq2a8_artifact import VQ2MatrixSpec
from vllm_ascend.quantization.vq2a8_reference import (
    decode_codebook_weight,
    decode_expert_weight,
    decode_expert_weight_literal,
    prepare_repacked_vq2a8_activation_reference,
    transform_vq2_activation,
    unpack_vq2_indices,
    vq2_matmul_literal,
    vq2_matmul_reference,
)


def _pack_indices(values: torch.Tensor) -> torch.Tensor:
    words = torch.zeros(math.ceil(values.numel() / 8), dtype=torch.int64)
    for position, value in enumerate(values.tolist()):
        words[position // 8] |= int(value) << (position % 8 * 4)
    return words.to(torch.int32)


def _spec(
    *,
    rows: int = 4,
    columns: int = 8,
    true_columns: int = 5,
    row_group_size: int = 2,
    group_size: int = 3,
    rht_block_size: int = 4,
    transforms: bool = True,
) -> VQ2MatrixSpec:
    metadata = {
        "rows": rows,
        "cols": columns,
        "n_row_tiles": rows // row_group_size,
        "n_col_tiles": math.ceil(columns / group_size),
        "row_group_size": row_group_size,
        "group_size": group_size,
        "K": 16,
        "index_bits": 4,
        "vector_len": 2,
        "n_vectors": rows * columns // 2,
        "n_elements": rows * columns,
        "orig_shape": [rows, true_columns if transforms else columns],
        "norm_dim": 0,
        "enable_perm": transforms,
        "enable_norm": transforms,
        "enable_rht": transforms,
    }
    if transforms:
        metadata.update(
            {
                "rht_block_size": rht_block_size,
                "rht_true_columns": true_columns,
            }
        )
    return VQ2MatrixSpec.from_dict("0.mlp.experts.0.gate_up", metadata)


def _tensors(spec: VQ2MatrixSpec) -> dict[str, torch.Tensor]:
    indices = torch.arange(spec.num_vectors, dtype=torch.int64).remainder(16)
    codebooks = torch.empty((spec.column_tiles, spec.row_tiles, 16, 2), dtype=torch.float64)
    for column_tile in range(spec.column_tiles):
        for row_tile in range(spec.row_tiles):
            for code in range(16):
                for component in range(2):
                    codebooks[column_tile, row_tile, code, component] = (
                        1000 * column_tile + 100 * row_tile + 2 * code + component
                    ) / 16
    tensors = {
        "packed_indices": _pack_indices(indices),
        "codebooks": codebooks,
    }
    if spec.enable_permutation:
        permutation = torch.roll(torch.arange(spec.columns, dtype=torch.int32), shifts=-1)
        signs = torch.where(
            torch.arange(spec.columns).remainder(3) == 0,
            torch.tensor(-1, dtype=torch.int8),
            torch.tensor(1, dtype=torch.int8),
        )
        tensors.update(
            {
                "perm": permutation,
                "weight_scale": torch.linspace(-1.0, 1.5, spec.columns),
                "weight_bias": torch.linspace(0.5, -0.5, spec.columns),
                "rht_sign": signs,
            }
        )
    return tensors


@pytest.mark.parametrize("count", [0, 1, 7, 8, 9, 97])
def test_unpack_vq2_indices_handles_word_boundaries(count: int) -> None:
    expected = torch.arange(count, dtype=torch.int64).remainder(16)
    actual = unpack_vq2_indices(_pack_indices(expected), count)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_unpack_vq2_indices_treats_signed_word_as_uint32() -> None:
    packed = torch.tensor([-0x01234568], dtype=torch.int32)
    expected = torch.tensor([8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.int64)
    torch.testing.assert_close(unpack_vq2_indices(packed, 8), expected, rtol=0, atol=0)


def test_unpack_vq2_indices_requires_exact_word_count() -> None:
    with pytest.raises(ValueError, match="has 2 words, expected 1"):
        unpack_vq2_indices(torch.zeros(2, dtype=torch.int32), 1)


def test_decode_traversal_handles_unequal_tiles_and_partial_column_tile() -> None:
    spec = _spec(rows=4, columns=7, true_columns=7, group_size=3, transforms=False)
    tensors = _tensors(spec)
    vectorized = decode_codebook_weight(tensors, spec, compute_dtype=torch.float64)
    literal = decode_expert_weight_literal(tensors, spec)
    torch.testing.assert_close(vectorized, literal, rtol=0, atol=0)

    assert spec.column_tiles == 3
    assert spec.row_tiles == 2
    assert vectorized.shape == (4, 7)
    assert vectorized[0, 0] == tensors["codebooks"][0, 0, 0, 0]
    last_tile_first_index = 2 * spec.row_tiles * spec.group_size * spec.vectors_per_row_group
    last_code = last_tile_first_index % 16
    assert vectorized[0, 6] == tensors["codebooks"][2, 0, last_code, 0]


def test_decode_traversal_matches_hard_coded_matrix() -> None:
    spec = _spec(rows=4, columns=3, true_columns=3, group_size=3, transforms=False)
    indices = torch.tensor([0, 1, 2, 3, 4, 5])
    codebooks = torch.empty((1, 2, 16, 2), dtype=torch.float64)
    for code in range(16):
        codebooks[0, 0, code] = torch.tensor([100 + code, 200 + code])
        codebooks[0, 1, code] = torch.tensor([300 + code, 400 + code])
    tensors = {"packed_indices": _pack_indices(indices), "codebooks": codebooks}
    expected = torch.tensor(
        [[100, 101, 102], [200, 201, 202], [303, 304, 305], [403, 404, 405]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        decode_codebook_weight(tensors, spec, compute_dtype=torch.float64), expected, rtol=0, atol=0
    )


def test_decode_traversal_matches_real_row_group_size_32_sentinel() -> None:
    spec = _spec(
        rows=32,
        columns=3,
        true_columns=3,
        row_group_size=32,
        group_size=2,
        transforms=False,
    )
    indices = torch.tensor([*range(16), *reversed(range(16)), *((value + 5) % 16 for value in range(16))])
    codebooks = torch.empty((2, 1, 16, 2), dtype=torch.float64)
    for column_tile in range(2):
        for code in range(16):
            codebooks[column_tile, 0, code] = torch.tensor(
                [100 * column_tile + 10 * code, 100 * column_tile + 10 * code + 1],
                dtype=torch.float64,
            )
    tensors = {"packed_indices": _pack_indices(indices), "codebooks": codebooks}
    expected = torch.tensor(
        [
            [0, 150, 150],
            [1, 151, 151],
            [10, 140, 160],
            [11, 141, 161],
            [20, 130, 170],
            [21, 131, 171],
            [30, 120, 180],
            [31, 121, 181],
            [40, 110, 190],
            [41, 111, 191],
            [50, 100, 200],
            [51, 101, 201],
            [60, 90, 210],
            [61, 91, 211],
            [70, 80, 220],
            [71, 81, 221],
            [80, 70, 230],
            [81, 71, 231],
            [90, 60, 240],
            [91, 61, 241],
            [100, 50, 250],
            [101, 51, 251],
            [110, 40, 100],
            [111, 41, 101],
            [120, 30, 110],
            [121, 31, 111],
            [130, 20, 120],
            [131, 21, 121],
            [140, 10, 130],
            [141, 11, 131],
            [150, 0, 140],
            [151, 1, 141],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        decode_codebook_weight(tensors, spec, compute_dtype=torch.float64), expected, rtol=0, atol=0
    )


def test_full_transform_matches_pinned_nvidia_golden() -> None:
    """Golden generated by the reference implementation at 2d75468d."""
    metadata = {
        "rows": 4,
        "cols": 4,
        "n_row_tiles": 2,
        "n_col_tiles": 2,
        "row_group_size": 2,
        "group_size": 3,
        "K": 16,
        "index_bits": 4,
        "vector_len": 2,
        "n_vectors": 8,
        "n_elements": 16,
        "orig_shape": [4, 3],
        "norm_dim": 0,
        "enable_perm": True,
        "enable_norm": True,
        "enable_rht": True,
        "rht_block_size": 4,
        "rht_true_columns": 3,
    }
    spec = VQ2MatrixSpec.from_dict("0.mlp.experts.0.gate_up", metadata)
    codebooks = torch.empty((2, 2, 16, 2), dtype=torch.float32)
    for column_tile in range(2):
        for row_tile in range(2):
            for code in range(16):
                codebooks[column_tile, row_tile, code] = torch.tensor(
                    [
                        (100 * column_tile + 20 * row_tile + 2 * code + 1) / 16,
                        (100 * column_tile + 20 * row_tile + 2 * code + 2) / 16,
                    ]
                )
    tensors = {
        "packed_indices": _pack_indices(torch.arange(8)),
        "codebooks": codebooks.to(torch.float8_e4m3fn),
        "perm": torch.tensor([2, 0, 3, 1], dtype=torch.int32),
        "weight_scale": torch.tensor([1.25, 0.75, 1.5, 0.5]),
        "weight_bias": torch.tensor([0.5, -0.25, 0.125, 1.0]),
        "rht_sign": torch.tensor([1, -1, 1, -1], dtype=torch.int8),
    }
    activation = torch.tensor([[0.5, -1.0, 2.0], [1.5, 0.75, -0.5]])
    expected_weight = torch.tensor(
        [
            [3.5546875, 2.6015625, 2.1796875],
            [3.65625, 2.53125, 2.15625],
            [6.59375, 1.15625, 1.84375],
            [6.671875, 1.078125, 1.921875],
        ]
    )
    expected_output = torch.tensor(
        [
            [3.53515625, 3.609375, 5.828125, 6.1015625],
            [6.193359375, 6.3046875, 9.8359375, 9.85546875],
        ]
    )
    actual_weight = decode_expert_weight(tensors, spec)
    actual_output = vq2_matmul_reference(activation, tensors, spec)
    assert actual_weight.dtype == torch.float32
    torch.testing.assert_close(actual_weight, expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("enable_permutation", "enable_normalization", "enable_rht"),
    [(False, False, False), (True, False, False), (False, True, False), (False, False, True), (True, True, True)],
)
def test_each_transform_matches_literal_oracle(
    enable_permutation: bool, enable_normalization: bool, enable_rht: bool
) -> None:
    metadata = {
        "rows": 4,
        "cols": 4,
        "n_row_tiles": 2,
        "n_col_tiles": 2,
        "row_group_size": 2,
        "group_size": 3,
        "K": 16,
        "index_bits": 4,
        "vector_len": 2,
        "n_vectors": 8,
        "n_elements": 16,
        "orig_shape": [4, 4],
        "norm_dim": 0,
        "enable_perm": enable_permutation,
        "enable_norm": enable_normalization,
        "enable_rht": enable_rht,
    }
    if enable_rht:
        metadata.update({"rht_block_size": 4, "rht_true_columns": 4})
    spec = VQ2MatrixSpec.from_dict("0.mlp.experts.0.down", metadata)
    tensors = {
        "packed_indices": _pack_indices(torch.arange(8)),
        "codebooks": torch.arange(2 * 2 * 16 * 2, dtype=torch.float32).reshape(2, 2, 16, 2) / 16,
    }
    if enable_permutation:
        tensors["perm"] = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
    if enable_normalization:
        tensors["weight_scale"] = torch.tensor([1.25, 0.75, 1.5, 0.5])
        tensors["weight_bias"] = torch.tensor([0.5, -0.25, 0.125, 1.0])
    if enable_rht:
        tensors["rht_sign"] = torch.tensor([1, -1, 1, -1], dtype=torch.int8)
    vectorized = decode_expert_weight(tensors, spec, compute_dtype=torch.float64)
    literal = decode_expert_weight_literal(tensors, spec)
    torch.testing.assert_close(vectorized, literal, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("rht_block_size", [2, 4, 8])
def test_vectorized_decode_matches_independent_literal_oracle(rht_block_size: int) -> None:
    spec = _spec(rht_block_size=rht_block_size)
    tensors = _tensors(spec)
    vectorized = decode_expert_weight(tensors, spec, compute_dtype=torch.float64)
    literal = decode_expert_weight_literal(tensors, spec)
    torch.testing.assert_close(vectorized, literal, rtol=1e-12, atol=1e-12)


def test_target_rht_block_128_matches_literal_oracle() -> None:
    spec = _spec(
        rows=2,
        columns=128,
        true_columns=127,
        row_group_size=2,
        group_size=31,
        rht_block_size=128,
    )
    tensors = _tensors(spec)
    vectorized = decode_expert_weight(tensors, spec, compute_dtype=torch.float64)
    literal = decode_expert_weight_literal(tensors, spec)
    torch.testing.assert_close(vectorized, literal, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("rht_block_size", [2, 4, 8])
def test_activation_side_transform_matches_full_weight(rht_block_size: int) -> None:
    spec = _spec(rht_block_size=rht_block_size)
    tensors = _tensors(spec)
    activation = torch.tensor(
        [[0.5, -1.0, 2.0, 0.25, -0.75], [1.5, 0.75, -0.5, 2.0, 0.125]],
        dtype=torch.float64,
    )

    decoded_weight = decode_expert_weight(tensors, spec, compute_dtype=torch.float64)
    expected = activation @ decoded_weight.transpose(0, 1)
    transformed, bias_correction = transform_vq2_activation(activation, tensors, spec, compute_dtype=torch.float64)
    from_parts = transformed @ decode_codebook_weight(tensors, spec, compute_dtype=torch.float64).transpose(0, 1)
    assert bias_correction is not None
    from_parts += bias_correction.unsqueeze(-1)

    torch.testing.assert_close(from_parts, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        vq2_matmul_reference(activation, tensors, spec, compute_dtype=torch.float64),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(vq2_matmul_literal(activation, tensors, spec), expected, rtol=1e-12, atol=1e-12)


def test_dynamic_fp8_reference_keeps_zero_activation_finite() -> None:
    activation = torch.zeros((3, 8), dtype=torch.bfloat16)
    scale = torch.linspace(-1.0, 1.0, 8)
    bias = torch.linspace(-0.5, 0.5, 8)
    sign = torch.tensor([1, -1, 1, 1, -1, 1, -1, -1], dtype=torch.int8)
    quantized, activation_scale, bias_correction = prepare_repacked_vq2a8_activation_reference(
        activation,
        scale,
        bias,
        sign,
        rht_block_size=4,
    )
    assert quantized.dtype == torch.float8_e4m3fn
    assert bool(torch.isfinite(quantized.float()).all())
    assert bool((quantized.float() == 0).all())
    torch.testing.assert_close(activation_scale, torch.full((3,), 1e-12), rtol=0, atol=0)
    torch.testing.assert_close(bias_correction, torch.zeros(3), rtol=0, atol=0)


def test_dynamic_fp8_reference_matches_hand_computed_nonzero_case() -> None:
    activation = torch.tensor(
        [[7.5, 2.5, -4.5, -1.5], [-3.75, -1.25, 2.25, 0.75]],
        dtype=torch.float32,
    )
    scale = torch.ones(4)
    bias = torch.tensor([0.5, -0.25, 0.125, 1.0])
    sign = torch.tensor([1, -1, 1, -1], dtype=torch.int8)
    quantized, activation_scale, bias_correction = prepare_repacked_vq2a8_activation_reference(
        activation, scale, bias, sign, rht_block_size=4
    )
    expected_quantized = torch.tensor([[56.0, 112.0, 224.0, 448.0], [-56.0, -112.0, -224.0, -448.0]]).to(
        torch.float8_e4m3fn
    )
    torch.testing.assert_close(quantized.float(), expected_quantized.float(), rtol=0, atol=0)
    torch.testing.assert_close(activation_scale, torch.tensor([8 / 448, 4 / 448]), rtol=0, atol=0)
    torch.testing.assert_close(bias_correction, torch.tensor([8.5, -4.25]), rtol=0, atol=0)


def _fwht_activation_oracle(activation: torch.Tensor, signs: torch.Tensor, block_size: int) -> torch.Tensor:
    """Independent butterfly oracle for the matrix-based RHT."""
    result = (activation.float() * signs.float()).reshape(
        activation.shape[0],
        activation.shape[1] // block_size,
        block_size,
    )
    result = result.clone()
    stride = 1
    while stride < block_size:
        groups = result.reshape(*result.shape[:-1], -1, 2 * stride)
        left = groups[..., :stride].clone()
        right = groups[..., stride:].clone()
        groups[..., :stride] = left + right
        groups[..., stride:] = left - right
        stride *= 2
    return (result / math.sqrt(block_size)).reshape_as(activation.float())


@pytest.mark.parametrize("input_size", [512, 2048, 4096])
@pytest.mark.parametrize("tokens", [1, 72])
def test_target_sized_dynamic_fp8_reference_matches_independent_oracle(input_size: int, tokens: int) -> None:
    block_size = 128
    columns = torch.arange(input_size, dtype=torch.int64)
    token_ids = torch.arange(tokens, dtype=torch.int64).unsqueeze(1)
    block_ids = torch.div(columns, block_size, rounding_mode="floor")
    activation = (
        ((columns.remainder(29) - 14).unsqueeze(0) * (token_ids.remainder(5) + 1))
        * (block_ids.remainder(4) + 1).unsqueeze(0)
        / 64
    ).to(torch.bfloat16)
    signs = torch.where(
        (columns + block_ids).remainder(3) == 0,
        torch.tensor(-1, dtype=torch.int8),
        torch.tensor(1, dtype=torch.int8),
    )
    weight_scale = (0.5 + columns.remainder(17).float() / 32).to(torch.float32)
    weight_bias = ((columns.remainder(11).float() - 5) / 64).to(torch.float32)

    quantized, activation_scale, bias_correction = prepare_repacked_vq2a8_activation_reference(
        activation,
        weight_scale,
        weight_bias,
        signs,
        rht_block_size=block_size,
    )
    rotated = _fwht_activation_oracle(activation, signs, block_size)
    transformed = rotated * weight_scale.unsqueeze(0)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    expected_scale = torch.clamp(transformed.abs().amax(dim=-1) / fp8_max, min=1e-12)
    expected_quantized = torch.clamp(
        transformed / expected_scale.unsqueeze(-1),
        -fp8_max,
        fp8_max,
    ).to(torch.float8_e4m3fn)
    expected_bias = rotated @ weight_bias

    assert quantized.shape == (tokens, input_size)
    assert quantized.numel() == activation.numel()
    assert bool(torch.isfinite(quantized.float()).all())
    quantized_float = quantized.float()
    expected_quantized_float = expected_quantized.float()
    mismatched = quantized_float != expected_quantized_float
    assert int(mismatched.sum()) <= max(1, quantized.numel() // 5_000)
    if bool(mismatched.any()):
        assert float((quantized_float - expected_quantized_float).abs().max()) <= 0.5
    torch.testing.assert_close(activation_scale, expected_scale, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(bias_correction, expected_bias, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        quantized_float * activation_scale.unsqueeze(-1),
        expected_quantized_float * expected_scale.unsqueeze(-1),
        rtol=0.1,
        atol=1e-6,
    )


@pytest.mark.parametrize("block_size", [True, 0, 3, 8])
def test_dynamic_fp8_reference_rejects_invalid_block_size(block_size: int) -> None:
    with pytest.raises(ValueError, match="rht_block_size"):
        prepare_repacked_vq2a8_activation_reference(
            torch.zeros(1, 4),
            torch.ones(4),
            torch.zeros(4),
            torch.ones(4, dtype=torch.int8),
            block_size,
        )


def test_dynamic_fp8_reference_rejects_bad_shape_sign_and_nonfinite() -> None:
    activation = torch.zeros(1, 4)
    with pytest.raises(ValueError, match="weight_scale has shape"):
        prepare_repacked_vq2a8_activation_reference(
            activation,
            torch.ones(1, 4),
            torch.zeros(4),
            torch.ones(4, dtype=torch.int8),
            4,
        )
    with pytest.raises(ValueError, match="other than -1 and 1"):
        prepare_repacked_vq2a8_activation_reference(
            activation,
            torch.ones(4),
            torch.zeros(4),
            torch.tensor([1, 1, 0, -1], dtype=torch.int8),
            4,
        )
    activation[0, 0] = float("nan")
    with pytest.raises(ValueError, match="activation contains non-finite"):
        prepare_repacked_vq2a8_activation_reference(
            activation,
            torch.ones(4),
            torch.zeros(4),
            torch.ones(4, dtype=torch.int8),
            4,
        )


def test_reference_rejects_wrong_activation_width() -> None:
    spec = _spec()
    with pytest.raises(ValueError, match="Activation width is 4, expected 5"):
        transform_vq2_activation(torch.zeros(1, 4), _tensors(spec), spec)
