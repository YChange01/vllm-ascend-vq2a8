# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent CPU layout/math oracles; no TP2 runtime or NPU claim."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from vllm_ascend.quantization.vq2a8_artifact import VQ2MatrixSpec
from vllm_ascend.quantization.vq2a8_tp2_layout import repack_matrix_tp2

FIELD_DTYPES = {
    "packed_zn": torch.uint8,
    "pair_lut": torch.uint8,
    "activation_order": torch.int64,
    "weight_scale": torch.float32,
    "weight_bias": torch.float32,
    "rht_sign": torch.int8,
}


def _canonical(kind="down", k=512, *, identity_perm=False, n=512):
    rng = np.random.default_rng(1901 + k)
    spec = VQ2MatrixSpec(
        name=f"3.mlp.experts.7.{kind}",
        layer_index=3,
        expert_id=7,
        kind=kind,
        rows=n,
        columns=k,
        row_tiles=n // 32,
        column_tiles=k // 256,
        row_group_size=32,
        group_size=256,
        num_vectors=n * k // 2,
        num_elements=n * k,
        original_shape=(n, k),
        norm_dimension=0,
        enable_permutation=True,
        enable_normalization=True,
        enable_rht=True,
        rht_block_size=128,
        rht_true_columns=k,
    )
    # Produce the canonical column-tile/row-tile/column/pair serialization
    # ourselves, without any production repack/unpack routine.
    grid = rng.integers(0, 16, (n // 2, k), dtype=np.uint8)
    flat = grid.reshape(n // 32, 16, k // 256, 256).transpose(2, 0, 3, 1).reshape(-1)
    shifts = np.arange(8, dtype=np.uint32) * 4
    words = np.sum(flat.reshape(-1, 8).astype(np.uint32) << shifts, axis=1, dtype=np.uint32)
    books = torch.from_numpy(rng.integers(-32, 33, (k // 256, n // 32, 16, 2)).astype(np.float32) / 8)
    books = books.to(torch.float8_e4m3fn)
    books.view(torch.uint8)[0, 0, 0, 0] = 128  # Preserve negative-zero bytes too.
    perm = np.arange(k, dtype=np.int32) if identity_perm else rng.permutation(k).astype(np.int32)
    source = {
        "packed_indices": torch.from_numpy(words.view(np.int32)),
        "codebooks": books,
        "perm": torch.from_numpy(perm),
        "weight_scale": torch.from_numpy(rng.uniform(0.125, 1.5, k).astype(np.float32)),
        "weight_bias": torch.from_numpy(rng.uniform(-0.25, 0.25, k).astype(np.float32)),
        "rht_sign": torch.from_numpy(rng.choice(np.array([-1, 1], dtype=np.int8), k)),
    }
    columns = np.argsort(perm)
    rows = np.arange(n)
    codes = grid[(rows // 2)[:, None], columns[None, :]]
    physical_bytes = books.view(torch.uint8).numpy()[
        (columns // 256)[None, :], (rows // 32)[:, None], codes, (rows % 2)[:, None]
    ]
    return source, spec, torch.from_numpy(physical_bytes.copy())


def _selected_rows(spec, rank):
    if spec.kind == "down":
        return torch.arange(spec.rows)
    width = spec.rows // 4
    return torch.cat(
        (
            torch.arange(rank * width, (rank + 1) * width),
            torch.arange(spec.rows // 2 + rank * width, spec.rows // 2 + (rank + 1) * width),
        )
    )


def _columns(spec, rank):
    return (
        (0, spec.columns) if spec.kind == "gate_up" else (rank * (spec.columns // 2), (rank + 1) * (spec.columns // 2))
    )


def _decode_zn_bytes(payload):
    """Decode zN + pair LUT directly, independently of the exporter."""
    packed = payload["packed_zn"].numpy()
    nt, kt, _, _ = packed.shape
    codes = np.stack((packed & 15, packed >> 4), axis=-1).reshape(nt, kt, 16, 16)
    codes = codes.transpose(0, 3, 1, 2).reshape(nt * 16, kt * 16)
    rows, columns = np.arange(nt * 32), np.arange(kt * 16)
    paired_codes = codes[(rows // 2)[:, None], columns[None, :]]
    result = payload["pair_lut"].numpy()[
        (columns // 256)[None, :], (rows // 32)[:, None], paired_codes * 2 + (rows % 2)[:, None]
    ]
    return torch.from_numpy(result.copy())


def _hadamard(dtype):
    h = torch.ones((1, 1), dtype=dtype)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / np.sqrt(128)


def _rotate(hidden, sign):
    return ((hidden * sign.to(hidden.dtype)).reshape(hidden.shape[0], -1, 128) @ _hadamard(hidden.dtype)).reshape_as(
        hidden
    )


def _project(hidden, weight, scale, bias, sign, *, a8=False):
    rotated = _rotate(hidden, sign)
    transformed = rotated * scale.to(hidden.dtype)
    correction = rotated @ bias.to(hidden.dtype)
    if a8:
        activation_scale = (transformed.abs().amax(-1) / 448).clamp_min(1e-12)
        quantized = (transformed / activation_scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        transformed = quantized.to(hidden.dtype) * activation_scale[:, None]
    return transformed @ weight.to(hidden.dtype).T + correction[:, None]


def _project_exported(hidden, payload, *, a8=False):
    padded_k = payload["activation_order"].numel()
    padded = torch.nn.functional.pad(hidden, (0, padded_k - hidden.shape[1]))
    # Restore physical extended K only in this independent test oracle.
    weight = _decode_zn_bytes(payload)[:, torch.argsort(payload["activation_order"])].view(torch.float8_e4m3fn)
    return _project(padded, weight, payload["weight_scale"], payload["weight_bias"], payload["rht_sign"], a8=a8)


@pytest.mark.parametrize("kind,k", [("gate_up", 512), ("down", 256), ("down", 512)])
@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_bytes_match_independent_physical_weight_slice(kind, k, rank):
    source, spec, physical = _canonical(kind, k)
    before = {field: tensor.view(torch.uint8).clone() for field, tensor in source.items()}
    payload, metadata = repack_matrix_tp2(source, spec, rank)
    rows = _selected_rows(spec, rank)
    start, end = _columns(spec, rank)
    local_k, n = end - start, rows.numel()
    padded_k = payload["activation_order"].numel()
    assert set(payload) == set(FIELD_DTYPES)
    for field, dtype in FIELD_DTYPES.items():
        assert payload[field].dtype == dtype and payload[field].device.type == "cpu"
        assert payload[field].is_contiguous()
        assert metadata["tensor_shapes"][field] == list(payload[field].shape)
    assert payload["packed_zn"].shape == (n // 32, padded_k // 16, 16, 8)
    assert payload["pair_lut"].shape == (padded_k // 256, n // 32, 32)
    assert metadata["format"] == "vq2a8_zn_tp2_v1"
    assert metadata["tp_size"] == 2 and metadata["tp_rank"] == rank
    assert metadata["runtime_supported"] is False
    assert metadata["canonical_shape"] == [spec.rows, k]
    assert metadata["logical_shape"] == [n, local_k]
    assert metadata["packed_shape"] == [n, padded_k]
    assert metadata["input_column_range"] == [start, end]
    assert metadata["padding_columns"] == padded_k - local_k
    if kind == "gate_up":
        width = spec.rows // 4
        assert metadata["output_row_ranges"] == [
            [rank * width, (rank + 1) * width],
            [spec.rows // 2 + rank * width, spec.rows // 2 + (rank + 1) * width],
        ]
    else:
        assert metadata["output_row_ranges"] == [[0, spec.rows]]
    order = payload["activation_order"]
    assert torch.equal(order.sort().values, torch.arange(padded_k))
    unpacked = _decode_zn_bytes(payload)[:, torch.argsort(order)]
    assert torch.equal(unpacked[:, :local_k], physical[rows, start:end])
    for field in ("weight_scale", "weight_bias", "rht_sign"):
        assert torch.equal(payload[field][:local_k], source[field][start:end])
        expected_dummy = 1 if field == "rht_sign" else 0
        assert torch.all(payload[field][local_k:] == expected_dummy)
    assert all(torch.equal(before[field], source[field].view(torch.uint8)) for field in source)


@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_cross_rank_permutation_pads_each_lut_without_repeating_real_columns(rank):
    source, spec, _ = _canonical()
    payload, metadata = repack_matrix_tp2(source, spec, rank)
    start, end = _columns(spec, rank)
    ids = torch.argsort(source["perm"].long())[start:end] // 256
    counts = torch.bincount(ids, minlength=spec.column_tiles)
    assert torch.all((counts > 0) & (counts < 256))  # Ordinary K slicing does not fit the unpadded LUT ABI.
    assert metadata["lut_source_tile_ids"] == list(range(spec.column_tiles))
    assert metadata["tile_valid_counts"] == counts.tolist()
    assert metadata["padding_columns"] == 256
    local_k = end - start
    for block, tile in enumerate(metadata["lut_source_tile_ids"]):
        order = payload["activation_order"][block * 256 : (block + 1) * 256]
        real = torch.nonzero(ids == tile).flatten()
        assert torch.equal(order[: real.numel()], real)  # Stable physical-column order within each LUT.
        assert torch.all(order[real.numel() :] >= local_k)
    hidden = torch.randn(2, local_k, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    padded = torch.nn.functional.pad(hidden, (0, payload["activation_order"].numel() - local_k))
    transformed = _rotate(padded, payload["rht_sign"]) * payload["weight_scale"]
    assert torch.count_nonzero(transformed[:, local_k:]) == 0
    assert torch.count_nonzero(payload["weight_bias"][local_k:]) == 0


@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_omits_empty_codebooks_instead_of_padding_them(rank):
    source, spec, _ = _canonical(identity_perm=True)
    payload, metadata = repack_matrix_tp2(source, spec, rank)
    assert metadata["lut_source_tile_ids"] == [rank]
    assert metadata["tile_valid_counts"] == [256]
    assert metadata["padding_columns"] == 0
    assert payload["activation_order"].numel() == 256


def test_tp2_local_rht_and_local_bias_sum_recover_unquantized_full_down_projection():
    source, spec, physical = _canonical()
    hidden = torch.randn(3, spec.columns, generator=torch.Generator().manual_seed(71), dtype=torch.float64)
    expected = _project(
        hidden, physical.view(torch.float8_e4m3fn), source["weight_scale"], source["weight_bias"], source["rht_sign"]
    )
    parts = []
    for rank in (0, 1):
        payload, _ = repack_matrix_tp2(source, spec, rank)
        start, end = _columns(spec, rank)
        parts.append(_project_exported(hidden[:, start:end], payload))
    torch.testing.assert_close(parts[0] + parts[1], expected, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("kind", ["gate_up", "down"])
def test_tp2_padded_projection_matches_independent_local_a8_not_a_tp1_bitwise_claim(kind):
    source, spec, physical = _canonical(kind)
    hidden = torch.randn(2, spec.columns, generator=torch.Generator().manual_seed(47)).bfloat16().float()
    for rank in (0, 1):
        payload, _ = repack_matrix_tp2(source, spec, rank)
        start, end = _columns(spec, rank)
        rows = _selected_rows(spec, rank)
        expected = _project(
            hidden[:, start:end],
            physical[rows, start:end].view(torch.float8_e4m3fn),
            source["weight_scale"][start:end],
            source["weight_bias"][start:end],
            source["rht_sign"][start:end],
            a8=True,
        )
        actual = _project_exported(hidden[:, start:end], payload, a8=True)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)


def test_tp2_local_dynamic_a8_is_not_silently_identified_with_full_k_a8():
    source, spec, physical = _canonical()
    hidden = torch.randn(1, spec.columns, generator=torch.Generator().manual_seed(611))
    hidden[:, spec.columns // 2 :] *= 128
    full = _project(
        hidden,
        physical.view(torch.float8_e4m3fn),
        source["weight_scale"],
        source["weight_bias"],
        source["rht_sign"],
        a8=True,
    )
    parts = []
    for rank in (0, 1):
        payload, _ = repack_matrix_tp2(source, spec, rank)
        start, end = _columns(spec, rank)
        parts.append(_project_exported(hidden[:, start:end], payload, a8=True))
    assert not torch.allclose(parts[0] + parts[1], full, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("rank", [-1, 2, True, False, 0.0, "0", None])
def test_tp2_rejects_invalid_rank(rank):
    source, spec, _ = _canonical()
    with pytest.raises((ValueError, TypeError)):
        repack_matrix_tp2(source, spec, rank)


@pytest.mark.parametrize(
    "error",
    [
        "book_nan",
        "scale_nan",
        "bias_inf",
        "sign",
        "perm_duplicate",
        "perm_range",
        "perm_dtype",
        "packed_shape",
        "scale_shape",
        "missing",
        "extra",
    ],
)
def test_tp2_rejects_invalid_canonical_payload(error):
    source, spec, _ = _canonical()
    if error == "book_nan":
        source["codebooks"].view(torch.uint8).reshape(-1)[0] = 127
    elif error == "scale_nan":
        source["weight_scale"][0] = float("nan")
    elif error == "bias_inf":
        source["weight_bias"][0] = float("inf")
    elif error == "sign":
        source["rht_sign"][0] = 0
    elif error == "perm_duplicate":
        source["perm"][0] = source["perm"][1]
    elif error == "perm_range":
        source["perm"][0] = spec.columns
    elif error == "perm_dtype":
        source["perm"] = source["perm"].long()
    elif error == "packed_shape":
        source["packed_indices"] = source["packed_indices"].reshape(1, -1)
    elif error == "scale_shape":
        source["weight_scale"] = source["weight_scale"][:-1]
    elif error == "missing":
        del source["weight_bias"]
    else:
        source["unexpected"] = torch.zeros(1)
    with pytest.raises((ValueError, TypeError)):
        repack_matrix_tp2(source, spec, 0)


@pytest.mark.parametrize(
    "change",
    [
        {"rht_block_size": 64},
        {"group_size": 128},
        {"row_group_size": 16},
        {"rht_true_columns": 511},
        {"norm_dimension": 1},
        {"enable_permutation": False},
        {"enable_normalization": False},
        {"enable_rht": False},
        {"rows": 510},
        {"columns": 384},
    ],
)
def test_tp2_rejects_unsupported_geometry_and_transform_contract(change):
    source, spec, _ = _canonical()
    with pytest.raises((ValueError, TypeError)):
        repack_matrix_tp2(source, replace(spec, **change), 0)
