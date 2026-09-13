# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte oracles against the unchanged V2/V3 startup converter, not NPU timing."""

import numpy as np
import pytest
import torch

from tests.ut.quantization.test_vq2a8_tp2_layout import _canonical, _decode_zn_bytes
from vllm_ascend.quantization.vq2a8_ascendc_v2 import convert_expert_payload
from vllm_ascend.quantization.vq2a8_repack import repack_matrix_tp1
from vllm_ascend.quantization.vq2a8_tp2_layout import repack_matrix_zn


@pytest.mark.parametrize("kind,k", [("gate_up", 4096), ("down", 2048)])
@pytest.mark.parametrize("permutation", ["identity", "random", "reverse", "cross_tiles"])
def test_tp1_zn_matches_existing_startup_converter_byte_for_byte(kind, k, permutation):
    source, spec, _ = _canonical(kind, k, n=4096, identity_perm=permutation == "identity")
    if permutation == "reverse":
        source["perm"] = torch.arange(k - 1, -1, -1, dtype=torch.int32)
    elif permutation == "cross_tiles":
        source["perm"] = torch.arange(k, dtype=torch.int32).reshape(-1, 256).T.contiguous().reshape(-1)
    # Include signed zero, subnormals and extreme finite FP8 values. Values
    # 127/255 are NaNs and deliberately excluded from this valid byte oracle.
    finite_bytes = np.array([0, 128, 1, 129, 126, 254, 56, 184], dtype=np.uint8)
    source["codebooks"].view(torch.uint8).reshape(-1)[:8].copy_(torch.from_numpy(finite_bytes))
    source["weight_scale"][0] = -0.0
    source["weight_bias"][0] = -0.0
    before = {field: value.view(torch.uint8).clone() for field, value in source.items()}
    expected = convert_expert_payload(repack_matrix_tp1(source, spec), spec)
    actual, metadata = repack_matrix_zn(source, spec, rank=0, tp_size=1)
    assert set(actual) == set(expected)
    for field, value in actual.items():
        reference = expected[field]
        assert value.device.type == "cpu" and value.is_contiguous()
        assert value.dtype == reference.dtype and value.shape == reference.shape
        assert torch.equal(value.view(torch.uint8), reference.view(torch.uint8)), field
    for field, value in source.items():
        assert torch.equal(value.view(torch.uint8), before[field]), field
    assert metadata["packed_shape"] == metadata["logical_shape"] == [4096, k]
    assert metadata["padding_columns"] == 0
    assert metadata["lut_source_tile_ids"] == list(range(k // 256))
    assert metadata["tile_valid_counts"] == [256] * (k // 256)


@pytest.mark.parametrize("kind,k", [("gate_up", 4096), ("down", 2048)])
def test_tp1_zn_preserves_full_canonical_physical_weight_bytes(kind, k):
    source, spec, physical_bytes = _canonical(kind, k, n=4096)
    payload, metadata = repack_matrix_zn(source, spec, rank=0, tp_size=1)
    restored = _decode_zn_bytes(payload)[:, torch.argsort(payload["activation_order"])]
    assert torch.equal(restored, physical_bytes)
    assert metadata["input_column_range"] == [0, k]
    assert metadata["output_row_ranges"] == ([[0, 2048], [2048, 4096]] if kind == "gate_up" else [[0, 4096]])
