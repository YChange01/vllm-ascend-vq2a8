# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure CPU V4+v2 compressed layout shared by runtime and offline export.

This is the original runtime conversion, not a second quantizer or the old V3
resident implementation. Layout versioning is independent of native-library
build identity so unchanged payloads survive arithmetic/kernel rebuilds.
"""

from __future__ import annotations

import numpy as np
import torch

from .vq2a8_runtime import VQ2_TP1_TORCH_DTYPES

V4_V2_ABI_VERSION = 1
V4_V2_MAX_EXPERTS = 256
V4_V2_SUPPORTED_N = 4096
V4_V2_SUPPORTED_K = (2048, 4096)
V4_V2_FIELDS = ("packed_zn", "pair_lut", "activation_order", "weight_scale", "weight_bias", "rht_sign")
V4_V2_DTYPES = (torch.uint8, torch.uint8, torch.int64, torch.float32, torch.float32, torch.int8)
V4_V2_BANK_WORDS = 8
V4_V2_BANK_COPIES = 2  # eager plus optional graph; payload storage is shared


def _geometry(n, k):
    if n != V4_V2_SUPPORTED_N or k not in V4_V2_SUPPORTED_K:
        raise ValueError("V4 v2 supports N=4096 and K=2048/4096 only; no alternate-kernel fallback.")


def _cpu_tensor(tensor, name, dtype, shape=None):
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype or tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU {dtype} tensor before conversion.")
    if not tensor.is_contiguous() or (shape is not None and tuple(tensor.shape) != tuple(shape)):
        raise ValueError(f"Invalid contiguous geometry for {name}.")
    return tensor


def convert_expert_payload(payload, spec):
    """Convert one validated CPU expert, before its only payload H2D upload.

    Nibble packing and arbitrary 16-entry FP8 pair LUTs are lossless. This
    creates compressed zN data, not a dense FP8 expert or a second device bank.
    Source preparation metadata deliberately remains in original K order.
    """
    if set(payload) != set(VQ2_TP1_TORCH_DTYPES):
        raise ValueError("V4 v2 conversion requires exactly the six direct-TP1 fields.")
    words = _cpu_tensor(payload["packed_indices"], "packed_indices", torch.int32)
    if words.ndim != 2 or min(words.shape) <= 0:
        raise ValueError("packed_indices requires positive [N/2,K/8] geometry.")
    n, k = words.shape[0] * 2, words.shape[1] * 8
    _geometry(n, k)
    if spec.rows != n or spec.columns != k or not 0 < spec.rht_true_columns <= k:
        raise ValueError("V4 v2 conversion geometry disagrees with matrix spec.")
    books = _cpu_tensor(payload["codebooks"], "codebooks", torch.float8_e4m3fn)
    if books.ndim != 4 or not 1 <= books.shape[0] <= 256 or tuple(books.shape[1:]) != (n // 32, 16, 2):
        raise ValueError("codebooks requires [1..256,N/32,16,2] arbitrary FP8 pairs.")
    ids = _cpu_tensor(payload["codebook_tile_ids"], "codebook_tile_ids", torch.uint8, (k,)).numpy()
    if np.any(ids.astype(np.int64) >= books.shape[0]):
        raise ValueError("V4 v2 codebook tile ID is out of range.")
    byte_books = books.view(torch.uint8).numpy()
    if np.any((byte_books & np.uint8(127)) == np.uint8(127)):
        raise ValueError("V4 v2 codebooks contain E4M3FN NaN.")
    retained = {}
    for name in ("weight_scale", "weight_bias", "rht_sign"):
        tensor = _cpu_tensor(payload[name], name, VQ2_TP1_TORCH_DTYPES[name], (k,))
        valid = ((tensor == -1) | (tensor == 1)).all() if name == "rht_sign" else torch.isfinite(tensor).all()
        if not bool(valid):
            raise ValueError(f"Invalid expert preparation metadata: {name}.")
        retained[name] = tensor
    order = np.argsort(ids, kind="stable").astype(np.int64)
    blocks = ids[order].reshape(-1, 256)
    if not np.all(blocks == blocks[:, :1]):
        raise ValueError("V4 v2 tile populations cannot form homogeneous K256 blocks; no lossy fallback.")
    unsigned = words.numpy().view(np.uint32)
    shifts = np.arange(8, dtype=np.uint32) * np.uint32(4)
    indices = ((unsigned[..., None] >> shifts) & np.uint32(15)).astype(np.uint8).reshape(n // 2, k)
    indices = indices[:, order]
    zn = indices.reshape(n // 32, 16, k // 16, 16).transpose(0, 2, 3, 1)
    packed_zn = np.ascontiguousarray(zn[..., 0::2] | (zn[..., 1::2] << np.uint8(4)))
    pair_lut = np.ascontiguousarray(byte_books[blocks[:, 0]].reshape(k // 256, n // 32, 32))
    return {
        "packed_zn": torch.from_numpy(packed_zn),
        "pair_lut": torch.from_numpy(pair_lut),
        "activation_order": torch.from_numpy(order),
        **retained,
    }
