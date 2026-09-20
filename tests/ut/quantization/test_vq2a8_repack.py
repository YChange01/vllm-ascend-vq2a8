# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import vllm_ascend.quantization.vq2a8_repack as vq2a8_repack_module
from vllm_ascend.quantization.vq2a8_artifact import VQ2MatrixSpec
from vllm_ascend.quantization.vq2a8_reference import (
    decode_expert_weight,
    decode_repacked_vq2a8_codebook_weight,
    decode_repacked_vq2a8_weight,
    prepare_repacked_vq2a8_activation_reference,
    vq2_matmul_reference,
    vq2a8_predecoded_matmul_reference,
    vq2a8_repacked_matmul_reference,
)
from vllm_ascend.quantization.vq2a8_repack import (
    canonical_index_grid,
    pack_repacked_indices,
    repack_matrix_tp1,
    unpack_repacked_indices,
    validate_repacked_matrix,
)

REPACKED_FIELDS = {
    "packed_indices",
    "codebooks",
    "codebook_tile_ids",
    "weight_scale",
    "weight_bias",
    "rht_sign",
}


def _metadata(
    *,
    rows: int = 8,
    columns: int = 8,
    true_columns: int | None = None,
    row_group_size: int = 4,
    group_size: int = 3,
    rht_block_size: int = 4,
    enable_permutation: bool = True,
    enable_normalization: bool = True,
    enable_rht: bool = True,
) -> dict[str, object]:
    if true_columns is None:
        true_columns = columns
    if not enable_rht:
        true_columns = columns
    metadata: dict[str, object] = {
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
        "orig_shape": [rows, true_columns],
        "norm_dim": 0,
        "enable_perm": enable_permutation,
        "enable_norm": enable_normalization,
        "enable_rht": enable_rht,
    }
    if enable_rht:
        metadata.update(
            {
                "rht_block_size": rht_block_size,
                "rht_true_columns": true_columns,
            }
        )
    return metadata


def _spec(name: str = "0.mlp.experts.0.down", **overrides: object) -> VQ2MatrixSpec:
    return VQ2MatrixSpec.from_dict(name, _metadata(**overrides))


def _pack_flat_codes(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.to(torch.int64).reshape(-1)
    words = torch.zeros(math.ceil(flat.numel() / 8), dtype=torch.int64)
    for position, code in enumerate(flat.tolist()):
        words[position // 8] |= int(code) << (4 * (position % 8))
    return words.to(torch.int32)


def _canonical_flat_from_grid(grid: torch.Tensor, spec: VQ2MatrixSpec) -> torch.Tensor:
    tiled = grid.reshape(spec.row_tiles, spec.vectors_per_row_group, spec.columns)
    pieces = []
    for column_tile in range(spec.column_tiles):
        start = column_tile * spec.group_size
        end = min(start + spec.group_size, spec.columns)
        pieces.append(tiled[:, :, start:end].permute(0, 2, 1).reshape(-1))
    return torch.cat(pieces)


def _codebooks(spec: VQ2MatrixSpec) -> torch.Tensor:
    count = spec.column_tiles * spec.row_tiles * 16 * 2
    values = (torch.arange(count, dtype=torch.float32).remainder(31) - 15) / 8
    return values.reshape(spec.column_tiles, spec.row_tiles, 16, 2).to(torch.float8_e4m3fn)


def _permutation(columns: int) -> torch.Tensor:
    return torch.cat(
        (
            torch.arange(1, columns, 2, dtype=torch.int32),
            torch.arange(0, columns, 2, dtype=torch.int32),
        )
    )


def _canonical_payload(
    spec: VQ2MatrixSpec,
    grid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    output_pairs = spec.rows // 2
    if grid is None:
        pair_ids = torch.arange(output_pairs, dtype=torch.int64).unsqueeze(1)
        columns = torch.arange(spec.columns, dtype=torch.int64).unsqueeze(0)
        grid = (3 * pair_ids + 5 * columns + pair_ids * columns).remainder(16).to(torch.uint8)
    payload = {
        "packed_indices": _pack_flat_codes(_canonical_flat_from_grid(grid, spec)),
        "codebooks": _codebooks(spec),
    }
    if spec.enable_permutation:
        payload["perm"] = _permutation(spec.columns)
    if spec.enable_normalization:
        columns = torch.arange(spec.columns, dtype=torch.float32)
        payload["weight_scale"] = 0.5 + columns.remainder(5) / 8
        payload["weight_bias"] = (columns.remainder(7) - 3) / 16
    if spec.enable_rht:
        payload["rht_sign"] = torch.where(
            torch.arange(spec.columns).remainder(3) == 0,
            torch.tensor(-1, dtype=torch.int8),
            torch.tensor(1, dtype=torch.int8),
        )
    return payload


def _clone_payload(payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in payload.items()}


def test_cpu_payload_validation_checks_every_e4m3_byte_pattern():
    spec = _spec()
    payload = repack_matrix_tp1(_canonical_payload(spec), spec)
    for byte in range(256):
        payload["codebooks"].view(torch.uint8).fill_(byte)
        finite = bool(torch.isfinite(payload["codebooks"].float()).all())
        if finite:
            validate_repacked_matrix(payload, spec)
        else:
            with pytest.raises(ValueError, match="non-finite"):
                validate_repacked_matrix(payload, spec)


def test_cpu_payload_validation_avoids_torch_reductions_and_thread_mutation(monkeypatch):
    spec = _spec(columns=512, group_size=2, rht_block_size=128)
    payload = repack_matrix_tp1(_canonical_payload(spec), spec)

    def forbidden(*args, **kwargs):
        raise AssertionError("Payload validation must not create a torch reduction thread team")

    monkeypatch.setattr(torch, "isfinite", forbidden)
    monkeypatch.setattr(torch, "bincount", forbidden)
    monkeypatch.setattr(torch, "set_num_threads", forbidden)
    validate_repacked_matrix(payload, spec)  # includes all 256 tile IDs
    payload["weight_scale"][-1] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        validate_repacked_matrix(payload, spec)


def _manual_repacked_codebook_weight(
    tensors: dict[str, torch.Tensor],
    spec: VQ2MatrixSpec,
) -> torch.Tensor:
    indices = unpack_repacked_indices(tensors["packed_indices"], spec.columns)
    tile_ids = tensors["codebook_tile_ids"].to(torch.int64)
    codebooks = tensors["codebooks"].to(torch.float64)
    weight = torch.empty((spec.rows, spec.columns), dtype=torch.float64)
    for output_pair in range(spec.rows // 2):
        row_tile = output_pair // spec.vectors_per_row_group
        for column in range(spec.columns):
            code = int(indices[output_pair, column])
            tile = int(tile_ids[column])
            weight[2 * output_pair : 2 * output_pair + 2, column] = codebooks[tile, row_tile, code]
    return weight


def test_canonical_index_grid_uses_frozen_traversal_with_partial_tile() -> None:
    spec = _spec(
        rows=8,
        columns=7,
        row_group_size=4,
        group_size=3,
        enable_permutation=False,
        enable_normalization=False,
        enable_rht=False,
    )
    flat_codes = torch.arange(spec.num_vectors).remainder(16)
    expected = torch.tensor(
        [
            [0, 2, 4, 12, 14, 0, 8],
            [1, 3, 5, 13, 15, 1, 9],
            [6, 8, 10, 2, 4, 6, 10],
            [7, 9, 11, 3, 5, 7, 11],
        ],
        dtype=torch.uint8,
    )

    actual = canonical_index_grid(_pack_flat_codes(flat_codes), spec)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()


def test_signed_int32_words_are_little_endian_in_both_layouts() -> None:
    signed_word = torch.tensor([-0x01234568], dtype=torch.int32)
    expected = torch.arange(8, 16, dtype=torch.uint8).reshape(1, 8)
    spec = _spec(
        rows=2,
        columns=8,
        row_group_size=2,
        group_size=3,
        enable_permutation=False,
        enable_normalization=False,
        enable_rht=False,
    )

    torch.testing.assert_close(canonical_index_grid(signed_word, spec), expected, rtol=0, atol=0)
    torch.testing.assert_close(unpack_repacked_indices(signed_word.reshape(1, 1), 8), expected, rtol=0, atol=0)
    assert int(pack_repacked_indices(expected)[0, 0]) == -0x01234568


def test_repacked_pack_round_trip_zeroes_and_rejects_padding_nibbles() -> None:
    indices = torch.arange(30, dtype=torch.int64).reshape(3, 10).remainder(16).to(torch.uint8)
    packed = pack_repacked_indices(indices)

    assert packed.dtype == torch.int32
    assert packed.shape == (3, 2)
    torch.testing.assert_close(unpack_repacked_indices(packed, 10), indices, rtol=0, atol=0)
    unsigned_last_words = packed[:, -1].to(torch.int64) & 0xFFFFFFFF
    assert bool(((unsigned_last_words >> 8) == 0).all())

    corrupted = packed.clone()
    corrupted[0, -1] = int(corrupted[0, -1]) | (7 << 8)
    with pytest.raises(ValueError, match="unused high padding nibbles must be zero"):
        unpack_repacked_indices(corrupted, 10)


def test_tp1_repack_absorbs_argsort_permutation_only_into_codes_and_tile_ids() -> None:
    spec = _spec(true_columns=7)
    payload = _canonical_payload(spec)
    canonical_grid = canonical_index_grid(payload["packed_indices"], spec)
    inverse_permutation = torch.argsort(payload["perm"].to(torch.int64))

    repacked = repack_matrix_tp1(payload, spec)

    expected_grid = canonical_grid[:, inverse_permutation]
    expected_tile_ids = torch.div(inverse_permutation, spec.group_size, rounding_mode="floor").to(torch.uint8)
    torch.testing.assert_close(
        unpack_repacked_indices(repacked["packed_indices"], spec.columns),
        expected_grid,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(repacked["codebook_tile_ids"], expected_tile_ids, rtol=0, atol=0)
    for name in ("codebooks", "weight_scale", "weight_bias", "rht_sign"):
        assert torch.equal(repacked[name], payload[name])
        assert repacked[name].data_ptr() != payload[name].data_ptr()
    assert "perm" not in repacked


def test_repacked_schema_has_exactly_six_fields_with_frozen_dtypes_and_shapes() -> None:
    spec = _spec(columns=12, true_columns=11, group_size=5)
    repacked = repack_matrix_tp1(_canonical_payload(spec), spec)

    validate_repacked_matrix(repacked, spec)
    assert set(repacked) == REPACKED_FIELDS
    assert (repacked["packed_indices"].dtype, tuple(repacked["packed_indices"].shape)) == (
        torch.int32,
        (spec.rows // 2, 2),
    )
    assert (repacked["codebooks"].dtype, tuple(repacked["codebooks"].shape)) == (
        torch.float8_e4m3fn,
        (3, 2, 16, 2),
    )
    assert (repacked["codebook_tile_ids"].dtype, tuple(repacked["codebook_tile_ids"].shape)) == (
        torch.uint8,
        (12,),
    )
    assert (repacked["weight_scale"].dtype, tuple(repacked["weight_scale"].shape)) == (torch.float32, (12,))
    assert (repacked["weight_bias"].dtype, tuple(repacked["weight_bias"].shape)) == (torch.float32, (12,))
    assert (repacked["rht_sign"].dtype, tuple(repacked["rht_sign"].shape)) == (torch.int8, (12,))
    assert all(tensor.device.type == "cpu" and tensor.is_contiguous() for tensor in repacked.values())


def test_repacked_schema_rejects_missing_extra_wrong_dtype_shape_and_stride() -> None:
    spec = _spec()
    valid = repack_matrix_tp1(_canonical_payload(spec), spec)

    invalid_payloads = []
    missing = dict(valid)
    missing.pop("weight_bias")
    invalid_payloads.append(missing)
    extra = dict(valid)
    extra["unexpected"] = torch.zeros(1)
    invalid_payloads.append(extra)
    for name, replacement in (
        ("packed_indices", valid["packed_indices"].reshape(-1)),
        ("codebooks", valid["codebooks"].float()),
        ("codebook_tile_ids", valid["codebook_tile_ids"].to(torch.int32)),
        ("weight_scale", valid["weight_scale"].to(torch.float64)),
        ("weight_bias", valid["weight_bias"][:-1].clone()),
        ("rht_sign", valid["rht_sign"].to(torch.int16)),
    ):
        invalid = dict(valid)
        invalid[name] = replacement
        invalid_payloads.append(invalid)
    noncontiguous = dict(valid)
    noncontiguous["weight_scale"] = torch.stack((valid["weight_scale"], valid["weight_scale"]), dim=1)[:, 0]
    assert not noncontiguous["weight_scale"].is_contiguous()
    invalid_payloads.append(noncontiguous)

    for invalid in invalid_payloads:
        with pytest.raises(ValueError):
            validate_repacked_matrix(invalid, spec)


@pytest.mark.parametrize(
    ("enable_permutation", "enable_normalization", "enable_rht"),
    [(False, True, True), (True, False, True), (True, True, False)],
)
def test_repack_rejects_disabled_required_transforms(
    enable_permutation: bool,
    enable_normalization: bool,
    enable_rht: bool,
) -> None:
    spec = _spec(
        enable_permutation=enable_permutation,
        enable_normalization=enable_normalization,
        enable_rht=enable_rht,
    )
    with pytest.raises(ValueError, match="requires permutation, normalization, and RHT"):
        repack_matrix_tp1({}, spec)


def test_repack_rejects_invalid_canonical_values_and_non_cpu_tensors() -> None:
    spec = _spec()
    valid = _canonical_payload(spec)

    bad_permutation = _clone_payload(valid)
    bad_permutation["perm"][0] = bad_permutation["perm"][1]
    with pytest.raises(ValueError, match="perm is not a bijection"):
        repack_matrix_tp1(bad_permutation, spec)

    for field, value in (("weight_scale", float("nan")), ("weight_bias", float("inf"))):
        nonfinite = _clone_payload(valid)
        nonfinite[field][0] = value
        with pytest.raises(ValueError, match="normalization contains non-finite"):
            repack_matrix_tp1(nonfinite, spec)

    nonfinite_codebook = _clone_payload(valid)
    codebooks = nonfinite_codebook["codebooks"].float()
    codebooks[0, 0, 0, 0] = float("nan")
    nonfinite_codebook["codebooks"] = codebooks.to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="codebooks contain non-finite"):
        repack_matrix_tp1(nonfinite_codebook, spec)

    bad_sign = _clone_payload(valid)
    bad_sign["rht_sign"][0] = 0
    with pytest.raises(ValueError, match="rht_sign"):
        repack_matrix_tp1(bad_sign, spec)

    non_cpu = _clone_payload(valid)
    non_cpu["packed_indices"] = torch.empty_like(valid["packed_indices"], device="meta")
    with pytest.raises(ValueError, match="must be a CPU tensor"):
        repack_matrix_tp1(non_cpu, spec)


def test_repacked_validation_rejects_bad_tile_ids_values_signs_and_device() -> None:
    spec = _spec()
    valid = repack_matrix_tp1(_canonical_payload(spec), spec)

    out_of_range_tile = _clone_payload(valid)
    out_of_range_tile["codebook_tile_ids"][0] = spec.column_tiles
    with pytest.raises(ValueError, match="outside"):
        validate_repacked_matrix(out_of_range_tile, spec)

    wrong_tile_population = _clone_payload(valid)
    source = int(torch.nonzero(wrong_tile_population["codebook_tile_ids"] == 1)[0])
    wrong_tile_population["codebook_tile_ids"][source] = 0
    with pytest.raises(ValueError, match="counts"):
        validate_repacked_matrix(wrong_tile_population, spec)

    nonfinite_codebook = _clone_payload(valid)
    codebooks = nonfinite_codebook["codebooks"].float()
    codebooks[0, 0, 0, 0] = float("nan")
    nonfinite_codebook["codebooks"] = codebooks.to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="non-finite"):
        validate_repacked_matrix(nonfinite_codebook, spec)

    nonfinite_scale = _clone_payload(valid)
    nonfinite_scale["weight_scale"][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_repacked_matrix(nonfinite_scale, spec)

    bad_sign = _clone_payload(valid)
    bad_sign["rht_sign"][0] = 0
    with pytest.raises(ValueError, match="other than -1 and 1"):
        validate_repacked_matrix(bad_sign, spec)

    non_cpu = _clone_payload(valid)
    non_cpu["packed_indices"] = torch.empty_like(valid["packed_indices"], device="meta")
    with pytest.raises(ValueError, match="must be a CPU tensor"):
        validate_repacked_matrix(non_cpu, spec)


@pytest.mark.parametrize("columns", [4, 8, 12, 16])
def test_runtime_validator_never_expands_index_grid_and_preserves_all_codes(monkeypatch, columns):
    spec = _spec(columns=columns)
    payload = repack_matrix_tp1(_canonical_payload(spec), spec)
    # Exhaust all possible nibble values, including signed int32 high bits.
    for code in range(16):
        codes = torch.full((spec.rows // 2, columns), code, dtype=torch.uint8)
        payload["packed_indices"] = pack_repacked_indices(codes)
        before = {name: tensor.view(torch.uint8).clone() for name, tensor in payload.items()}

        def forbidden(*args, **kwargs):
            raise AssertionError("Runtime validation must not allocate an unpacked grid")

        with monkeypatch.context() as patch:
            patch.setattr(vq2a8_repack_module, "unpack_repacked_indices", forbidden)
            patch.setattr(vq2a8_repack_module, "_unpack_packed_rows", forbidden)
            validate_repacked_matrix(payload, spec)
        for name, tensor in payload.items():
            assert torch.equal(tensor.view(torch.uint8), before[name])
        assert torch.equal(unpack_repacked_indices(payload["packed_indices"], columns), codes)
    if columns % 8:
        payload["packed_indices"][:, -1] |= 1 << ((columns % 8) * 4)
        with pytest.raises(ValueError, match="padding nibbles"):
            validate_repacked_matrix(payload, spec)


def test_pack_rejects_out_of_range_wrong_dtype_and_non_cpu_indices() -> None:
    out_of_range = torch.zeros((1, 8), dtype=torch.uint8)
    out_of_range[0, -1] = 16
    with pytest.raises(ValueError, match=r"uint4 range \[0, 15\]"):
        pack_repacked_indices(out_of_range)
    with pytest.raises(ValueError, match="two-dimensional uint8"):
        pack_repacked_indices(torch.zeros((1, 8), dtype=torch.int8))
    with pytest.raises(ValueError, match="must be a CPU tensor"):
        pack_repacked_indices(torch.empty((1, 8), dtype=torch.uint8, device="meta"))


def test_canonical_and_repacked_dense_weight_and_projection_are_equivalent() -> None:
    spec = _spec(true_columns=7)
    canonical = _canonical_payload(spec)
    repacked = repack_matrix_tp1(canonical, spec)
    activation = ((torch.arange(21, dtype=torch.float64).reshape(3, 7).remainder(13) - 6) / 8).contiguous()

    canonical_weight = decode_expert_weight(canonical, spec, compute_dtype=torch.float64)
    repacked_weight = decode_repacked_vq2a8_weight(repacked, spec, compute_dtype=torch.float64)
    canonical_output = vq2_matmul_reference(activation, canonical, spec, compute_dtype=torch.float64)
    repacked_output = vq2a8_repacked_matmul_reference(
        activation,
        repacked,
        spec,
        compute_dtype=torch.float64,
        dynamic_a8=False,
    )
    codebook_weight = decode_repacked_vq2a8_codebook_weight(
        repacked,
        spec,
        compute_dtype=torch.float64,
    )
    predecoded_output = vq2a8_predecoded_matmul_reference(
        activation,
        codebook_weight,
        repacked["weight_scale"],
        repacked["weight_bias"],
        repacked["rht_sign"],
        spec,
        compute_dtype=torch.float64,
        dynamic_a8=False,
    )

    torch.testing.assert_close(repacked_weight, canonical_weight, rtol=0, atol=0)
    torch.testing.assert_close(repacked_output, canonical_output, rtol=0, atol=0)
    torch.testing.assert_close(predecoded_output, repacked_output, rtol=0, atol=0)
    torch.testing.assert_close(repacked_output, activation @ canonical_weight.T, rtol=0, atol=0)


@pytest.mark.parametrize("columns", [4, 8, 12])
@pytest.mark.parametrize("tokens", [1, 72])
def test_dynamic_a8_repacked_projection_covers_small_k_and_batch_boundaries(columns: int, tokens: int) -> None:
    spec = _spec(columns=columns, true_columns=columns - 1, group_size=3)
    repacked = repack_matrix_tp1(_canonical_payload(spec), spec)
    token_ids = torch.arange(1, tokens + 1, dtype=torch.int64).unsqueeze(1)
    input_columns = torch.arange(spec.rht_true_columns, dtype=torch.int64).unsqueeze(0)
    activation = (((token_ids * (input_columns.remainder(5) - 2)).remainder(31) - 15) / 8).to(torch.bfloat16)

    actual = vq2a8_repacked_matmul_reference(
        activation,
        repacked,
        spec,
        compute_dtype=torch.float64,
        dynamic_a8=True,
    )
    codebook_weight = decode_repacked_vq2a8_codebook_weight(
        repacked,
        spec,
        compute_dtype=torch.float64,
    )
    predecoded = vq2a8_predecoded_matmul_reference(
        activation,
        codebook_weight,
        repacked["weight_scale"],
        repacked["weight_bias"],
        repacked["rht_sign"],
        spec,
        compute_dtype=torch.float64,
        dynamic_a8=True,
    )
    padded = F.pad(activation, (0, spec.columns - spec.rht_true_columns))
    quantized, activation_scale, bias_correction = prepare_repacked_vq2a8_activation_reference(
        padded,
        repacked["weight_scale"],
        repacked["weight_bias"],
        repacked["rht_sign"],
        spec.rht_block_size,
    )
    codebook_weight = _manual_repacked_codebook_weight(repacked, spec)
    expected = quantized.to(torch.float64) @ codebook_weight.T
    expected *= activation_scale.to(torch.float64).unsqueeze(-1)
    expected += bias_correction.to(torch.float64).unsqueeze(-1)

    assert actual.shape == (tokens, spec.rows)
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(predecoded, actual, rtol=0, atol=0)
