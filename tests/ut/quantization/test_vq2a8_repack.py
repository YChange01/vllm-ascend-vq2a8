# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import math
import os
import runpy
import sys
from pathlib import Path

import pytest
import safetensors.torch as safetensors_torch
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

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
    VQ2_DIRECT_TP1_FORMAT,
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


def _repack_tool_path() -> Path:
    return next(
        candidate
        for parent in Path(__file__).resolve().parents
        if (candidate := parent / "tools" / "repack_vq2a8_tp1.py").is_file()
    )


def _write_cli_model(
    tmp_path: Path,
    *,
    num_hidden_layers: int = 1,
    present_layers: tuple[int, ...] = (0,),
    num_hash_layers: int = 0,
    num_routed_experts: int = 2,
) -> tuple[Path, Path, dict[int, dict[int, dict[str, int]]]]:
    model_path = tmp_path / "model"
    source = model_path / "experts_vq"
    source.mkdir(parents=True)
    sentinels: dict[int, dict[int, dict[str, int]]] = {}
    for layer_index in present_layers:
        expert_count = 1 if layer_index < num_hash_layers else num_routed_experts
        metadata = {}
        tensors = {}
        sentinels[layer_index] = {}
        # Deliberately write metadata in reverse expert order so the CLI must
        # stack by the explicit expert_ids contract rather than mapping order.
        for expert_id in reversed(range(expert_count)):
            sentinels[layer_index][expert_id] = {}
            for kind, rows, true_columns in (("gate_up", 4, 4), ("down", 4, 2)):
                name = f"{layer_index}.mlp.experts.{expert_id}.{kind}"
                matrix_metadata = _metadata(rows=rows, columns=4, true_columns=true_columns, row_group_size=2)
                spec = VQ2MatrixSpec.from_dict(name, matrix_metadata)
                sentinel = 1 + 4 * expert_id + (2 if kind == "down" else 0)
                grid = torch.full((spec.rows // 2, spec.columns), sentinel, dtype=torch.uint8)
                payload = _canonical_payload(spec, grid)
                payload["weight_bias"].fill_(sentinel / 16)
                metadata[name] = matrix_metadata
                sentinels[layer_index][expert_id][kind] = sentinel
                for field, tensor in payload.items():
                    tensors[f"{name}.{field}"] = tensor
        metadata_path = source / f"experts_vq_layer_{layer_index}.json"
        tensor_path = source / f"experts_vq_layer_{layer_index}.safetensors"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        save_file(tensors, tensor_path)
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": num_hidden_layers,
                "num_hash_layers": num_hash_layers,
                "n_routed_experts": num_routed_experts,
                "hidden_size": 4,
                "moe_intermediate_size": 2,
                "quantization_config": {"quant_method": "vq2a8"},
            }
        ),
        encoding="utf-8",
    )
    return model_path, source, sentinels


def _run_repack_cli(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    output: Path,
    *extra_arguments: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_repack_tool_path()),
            "--input",
            str(source),
            "--output",
            str(output),
            *extra_arguments,
        ],
    )
    runpy.run_path(str(_repack_tool_path()), run_name="__main__")


def _source_files(source: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(source.iterdir()) if path.is_file()}


def test_repack_cli_writes_auditable_multi_expert_tp1_artifact_without_touching_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, source, sentinels = _write_cli_model(tmp_path)
    output = tmp_path / "direct"
    source_before = _source_files(source)

    _run_repack_cli(monkeypatch, source, output)

    lines = capsys.readouterr().out.strip().splitlines()
    report = json.loads(lines[-1])
    assert report == {
        "complete": True,
        "format": VQ2_DIRECT_TP1_FORMAT,
        "layers": [0],
        "output": str(output.resolve()),
    }
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["format"] == VQ2_DIRECT_TP1_FORMAT
    assert manifest["tp_size"] == 1
    assert manifest["complete"] is True
    assert manifest["repacker"] == {
        "contract_revision": 1,
        "tool": "tools/repack_vq2a8_tp1.py",
        "tool_sha256": hashlib.sha256(_repack_tool_path().read_bytes()).hexdigest(),
    }
    assert manifest["output"] == {
        "artifact_root": ".",
        "path_semantics": "relative_to_manifest",
    }
    assert manifest["packing"] == {
        "axis": "input_column",
        "endianness": "little",
        "index_bits": 4,
        "indices_per_word": 8,
        "nibble_order": "least_significant_first",
        "padding_nibbles": "zero",
        "vector_length": 2,
        "word_dtype": "I32",
    }
    assert manifest["communication"] == {
        "multi_rank_owner": "moe_runner",
        "reduction_required": False,
        "tp_reduction_count": 0,
        "tp_reduction_owner": "none_for_tp1",
    }
    assert manifest["layers"][0]["expert_count"] == 2
    assert not Path(manifest["layers"][0]["tensor_file"]).is_absolute()
    assert not Path(manifest["layers"][0]["metadata_file"]).is_absolute()
    direct_tensors = output / "tp1" / "rank0" / "experts_vq_layer_0.safetensors"
    with safe_open(direct_tensors, framework="pt", device="cpu") as handle:
        tensor_names = handle.keys()
        assert set(tensor_names) == {f"{kind}_{field}" for kind in ("gate_up", "down") for field in REPACKED_FIELDS}
        actual_shapes = {name: list(handle.get_tensor(name).shape) for name in tensor_names}
        for kind in ("gate_up", "down"):
            packed = handle.get_tensor(f"{kind}_packed_indices")
            bias = handle.get_tensor(f"{kind}_weight_bias")
            assert packed.shape[0] == 2
            assert bias.shape[0] == 2
            for expert_id in range(2):
                sentinel = sentinels[0][expert_id][kind]
                expected_codes = torch.full((2, 4), sentinel, dtype=torch.uint8)
                torch.testing.assert_close(unpack_repacked_indices(packed[expert_id], 4), expected_codes)
                torch.testing.assert_close(bias[expert_id], torch.full((4,), sentinel / 16))

    layer_manifest_path = output / "tp1" / "rank0" / "experts_vq_layer_0.json"
    layer_manifest = json.loads(layer_manifest_path.read_text(encoding="utf-8"))
    assert layer_manifest["schema_version"] == 1
    assert layer_manifest["expert_ids"] == [0, 1]
    assert layer_manifest["communication"] == {
        "multi_rank_owner": "moe_runner",
        "reduction_required": False,
        "tp_reduction_count": 0,
        "tp_reduction_owner": "none_for_tp1",
    }
    assert layer_manifest["output"]["tensor_file"] == "experts_vq_layer_0.safetensors"
    assert not Path(layer_manifest["output"]["tensor_file"]).is_absolute()
    expected_tensor_contract = {
        "codebook_tile_ids": ("U8", ["expert", "input_column"]),
        "codebooks": (
            "F8_E4M3",
            ["expert", "codebook_column_tile", "row_tile", "code", "vector_component"],
        ),
        "packed_indices": ("I32", ["expert", "output_pair", "packed_input_word"]),
        "rht_sign": ("I8", ["expert", "input_column"]),
        "weight_bias": ("F32", ["expert", "input_column"]),
        "weight_scale": ("F32", ["expert", "input_column"]),
    }
    for kind in ("gate_up", "down"):
        tensors = layer_manifest["layout"][kind]["tensors"]
        assert set(tensors) == REPACKED_FIELDS
        for field, (dtype, axes) in expected_tensor_contract.items():
            assert tensors[field]["dtype"] == dtype
            assert tensors[field]["axes"] == axes
            assert tensors[field]["shape"] == actual_shapes[f"{kind}_{field}"]
    assert _source_files(source) == source_before


def test_repack_cli_all_is_fail_closed_when_configured_layer_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source, _ = _write_cli_model(tmp_path, num_hidden_layers=2, present_layers=(0,))
    output = tmp_path / "direct"

    with pytest.raises(ValueError, match="--layers=all requires exactly the model layers"):
        _run_repack_cli(monkeypatch, source, output)

    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}.tmp-*")) == []

    _run_repack_cli(monkeypatch, source, output, "--layers", "0")

    subset_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert subset_manifest["layers_selected"] == [0]
    assert subset_manifest["layers_expected"] == 2
    assert subset_manifest["complete"] is False


def test_repack_cli_refuses_an_existing_output_without_modifying_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source, _ = _write_cli_model(tmp_path)
    output = tmp_path / "direct"
    output.mkdir()
    marker = output / "owned-by-user"
    marker.write_text("keep", encoding="utf-8")
    source_before = _source_files(source)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run_repack_cli(monkeypatch, source, output)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert _source_files(source) == source_before


@pytest.mark.parametrize("output_location", ["below_input", "above_input"])
def test_repack_cli_refuses_input_output_ancestor_relationships(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_location: str,
) -> None:
    model_path, source, _ = _write_cli_model(tmp_path)
    output = source / "direct" if output_location == "below_input" else model_path
    source_before = _source_files(source)

    with pytest.raises(ValueError, match="neither may contain the other"):
        _run_repack_cli(monkeypatch, source, output)

    if output_location == "below_input":
        assert not output.exists()
    assert _source_files(source) == source_before


@pytest.mark.parametrize("failure_stage", ["repack", "write", "publish"])
def test_repack_cli_failure_never_publishes_output_and_removes_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    _, source, _ = _write_cli_model(tmp_path)
    output = tmp_path / "direct"

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError(f"injected {failure_stage} failure")

    if failure_stage == "repack":
        monkeypatch.setattr(vq2a8_repack_module, "repack_matrix_tp1", fail)
    elif failure_stage == "write":
        monkeypatch.setattr(safetensors_torch, "save_file", fail)
    else:
        original_replace = os.replace

        def fail_final_publish(source_path: str | os.PathLike[str], destination_path: str | os.PathLike[str]) -> None:
            if Path(destination_path) == output:
                raise RuntimeError("injected publish failure")
            original_replace(source_path, destination_path)

        monkeypatch.setattr(os, "replace", fail_final_publish)

    with pytest.raises(RuntimeError, match=f"injected {failure_stage} failure"):
        _run_repack_cli(monkeypatch, source, output)

    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}.tmp-*")) == []


def test_repack_cli_detects_source_snapshot_change_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source, _ = _write_cli_model(tmp_path)
    output = tmp_path / "direct"
    metadata_path = source / "experts_vq_layer_0.json"
    original_repack = vq2a8_repack_module.repack_matrix_tp1
    mutated = False

    def mutate_source_after_first_repack(
        tensors: dict[str, torch.Tensor],
        spec: VQ2MatrixSpec,
    ) -> dict[str, torch.Tensor]:
        nonlocal mutated
        result = original_repack(tensors, spec)
        if not mutated:
            metadata_path.write_text(metadata_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            mutated = True
        return result

    monkeypatch.setattr(vq2a8_repack_module, "repack_matrix_tp1", mutate_source_after_first_repack)

    with pytest.raises(RuntimeError, match="Source snapshot changed during repack"):
        _run_repack_cli(monkeypatch, source, output)

    assert mutated
    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}.tmp-*")) == []


def test_repack_cli_can_require_the_pinned_reference_identity_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, source, _ = _write_cli_model(tmp_path)
    output = tmp_path / "direct"
    source_before = _source_files(source)

    with pytest.raises(ValueError, match="Reference identity is required"):
        _run_repack_cli(
            monkeypatch,
            source,
            output,
            "--require-reference-identity",
        )

    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}.tmp-*")) == []
    assert _source_files(source) == source_before


@pytest.mark.parametrize(
    "corruption",
    ["tensor_bytes", "metadata_locator", "root_communication", "root_complete"],
)
def test_repack_cli_validates_staging_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    _, source, _ = _write_cli_model(tmp_path)
    output = tmp_path / "direct"
    original_replace = os.replace
    corrupted = False

    def corrupt_after_root_manifest_write(
        source_path: str | os.PathLike[str],
        destination_path: str | os.PathLike[str],
    ) -> None:
        nonlocal corrupted
        original_replace(source_path, destination_path)
        destination = Path(destination_path)
        if destination.name != "manifest.json":
            return
        corrupted = True
        manifest = json.loads(destination.read_text(encoding="utf-8"))
        if corruption == "tensor_bytes":
            tensor_path = destination.parent / manifest["layers"][0]["tensor_file"]
            with tensor_path.open("ab") as tensor_file:
                tensor_file.write(b"corrupt")
            return
        if corruption == "metadata_locator":
            manifest["layers"][0]["metadata_file"] = "../outside.json"
        elif corruption == "root_communication":
            manifest["communication"]["tp_reduction_count"] = 1
        else:
            manifest["complete"] = False
        destination.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(os, "replace", corrupt_after_root_manifest_write)

    with pytest.raises(ValueError, match="Staging artifact validation failed"):
        _run_repack_cli(monkeypatch, source, output)

    assert corrupted
    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}.tmp-*")) == []
