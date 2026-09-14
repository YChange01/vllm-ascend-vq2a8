"""CPU-only, byte-preserving VQ2 layout bridge; NOT an executable NPU backend.

The input is the *prepared* direct-TP1 contract, after RHT/sign/normalization.
Preserve K order by default. Sorting K is a separate experiment, since it changes
floating-point reduction order. Neither mode makes the original MX/BF16 kernel
compatible: a full-FP32 row-scale/bias epilogue is still required before BF16.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ZN_K0 = 16
ZN_N0 = 32
LUT_K = 256
PAIR_SIZE = 2
LUT_SIZE = 16
INDICES_PER_WORD = 8
MX_GROUP = 32
E8M0_ONE = 127


def _array(value: np.ndarray, name: str, dtype, ndim: int) -> np.ndarray:
    # Validate before any cast: uint8(256) or uint8(-1) must not hide bad input.
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must be an ndarray with dtype {np.dtype(dtype)}")
    if value.ndim != ndim or any(size <= 0 for size in value.shape):
        raise ValueError(f"{name} must have {ndim} positive dimensions")
    return value


def _finite_fp8(data: np.ndarray, name: str) -> None:
    # E4M3FN: +/-0x7f are NaN; preserve every other byte including signed zero.
    if np.any((data & np.uint8(0x7F)) == np.uint8(0x7F)):
        raise ValueError(f"{name} contains E4M3FN NaN")


def unpack_direct_words(words: np.ndarray) -> np.ndarray:
    """int32 [N/2,K/8] -> uint8 [N/2,K], low nibble is the first K."""
    words = _array(words, "packed_indices", np.int32, 2)
    unsigned = np.ascontiguousarray(words).view(np.uint32)
    shifts = np.arange(INDICES_PER_WORD, dtype=np.uint32) * np.uint32(4)
    return ((unsigned[..., None] >> shifts) & np.uint32(15)).astype(np.uint8).reshape(words.shape[0], -1)


def pack_pairs_zn(indices: np.ndarray) -> np.ndarray:
    """Arbitrary VQ uint4 pair indices -> zN bytes, WITHOUT scalar fitting.

    Input [N/2,K]; output [N/32,K/16,16,8]. The two nibbles in each byte
    select two adjacent N pairs, each pair containing two arbitrary FP8 bytes.
    """
    indices = _array(indices, "pair_indices", np.uint8, 2)
    pairs, k = indices.shape
    if pairs % (ZN_N0 // PAIR_SIZE) or k % ZN_K0:
        raise ValueError("pair_indices requires N%32=0 and K%16=0")
    if np.any(indices >= LUT_SIZE):
        raise ValueError("pair_indices must be in [0,15]")
    zn = indices.reshape(-1, ZN_N0 // PAIR_SIZE, k // ZN_K0, ZN_K0).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(zn[..., 0::2] | (zn[..., 1::2] << np.uint8(4)))


def unpack_pairs_zn(packed: np.ndarray) -> np.ndarray:
    packed = _array(packed, "packed_zn", np.uint8, 4)
    if packed.shape[-2:] != (ZN_K0, ZN_N0 // 4):
        raise ValueError("packed_zn must end in [16,8]")
    pairs = np.empty(packed.shape[:-1] + (ZN_N0 // PAIR_SIZE,), dtype=np.uint8)
    pairs[..., 0::2] = packed & np.uint8(15)
    pairs[..., 1::2] = packed >> np.uint8(4)
    return np.ascontiguousarray(pairs.transpose(0, 3, 1, 2).reshape(-1, packed.shape[1] * ZN_K0))


@dataclass(frozen=True)
class BridgeWeights:
    packed_zn: np.ndarray
    codebook_bytes: np.ndarray
    codebook_tile_ids: np.ndarray
    # target[k] = source[activation_gather[k]], for BOTH A and W.
    activation_gather: np.ndarray
    # Only present when every target K256 has one LUT. This is NOT a full ABI.
    fixed_k256_lut: np.ndarray | None
    k_reordered: bool


def validate_bridge_weights(weights: BridgeWeights) -> None:
    """Reject hand-made or mutated plans that silently duplicate/drop K columns."""
    if not isinstance(weights, BridgeWeights):
        raise TypeError("weights must be BridgeWeights")
    packed = _array(weights.packed_zn, "packed_zn", np.uint8, 4)
    if packed.shape[-2:] != (ZN_K0, ZN_N0 // 4):
        raise ValueError("packed_zn must end in [16,8]")
    k, n = packed.shape[1] * ZN_K0, packed.shape[0] * ZN_N0
    books = _array(weights.codebook_bytes, "codebook_bytes", np.uint8, 4)
    ids = _array(weights.codebook_tile_ids, "codebook_tile_ids", np.uint8, 1)
    order = _array(weights.activation_gather, "activation_gather", np.int64, 1)
    if k % LUT_K or books.shape[0] > 256 or books.shape[1:] != (n // ZN_N0, LUT_SIZE, PAIR_SIZE):
        raise ValueError("bridge weight geometry is inconsistent")
    if ids.shape != (k,) or np.any(ids.astype(np.int64) >= books.shape[0]):
        raise ValueError("bridge tile IDs do not match weights")
    if order.shape != (k,) or not np.array_equal(np.sort(order), np.arange(k)):
        raise ValueError("activation_gather must be a bijection over [0,K)")
    if not isinstance(weights.k_reordered, bool) or weights.k_reordered != (not np.array_equal(order, np.arange(k))):
        raise ValueError("k_reordered does not match activation_gather")
    _finite_fp8(books, "codebook_bytes")
    if weights.fixed_k256_lut is not None:
        fixed = _array(weights.fixed_k256_lut, "fixed_k256_lut", np.uint8, 3)
        blocks = ids.reshape(-1, LUT_K)
        if not np.all(blocks == blocks[:, :1]):
            raise ValueError("fixed K256 LUT cannot represent mixed tile IDs")
        expected = books[blocks[:, 0]].reshape(k // LUT_K, n // ZN_N0, 32)
        if not np.array_equal(fixed, expected):
            raise ValueError("fixed K256 LUT bytes do not match selected codebooks")


def bridge_direct_weights(
    packed_indices: np.ndarray,
    codebook_bytes: np.ndarray,
    codebook_tile_ids: np.ndarray,
    *,
    k_order: str = "preserve",
) -> BridgeWeights:
    """Copy/rearrange only bytes; do not write artifacts or modify source arrays.

    preserve: keep original physical K and require per-K codebook IDs in AIV.
    codebook: stable-sort by tile ID and require matching prepared-A gather;
              reject if the resulting fixed-K256 LUT cannot represent the data.
    """
    indices = unpack_direct_words(packed_indices)
    n, k = indices.shape[0] * PAIR_SIZE, indices.shape[1]
    books = _array(codebook_bytes, "codebook_bytes", np.uint8, 4)
    ids = _array(codebook_tile_ids, "codebook_tile_ids", np.uint8, 1)
    if n % ZN_N0 or k % LUT_K:
        raise ValueError("bridge requires N%32=0 and K%256=0")
    if books.shape[1:] != (n // ZN_N0, LUT_SIZE, PAIR_SIZE) or books.shape[0] > 256:
        raise ValueError("codebooks must be uint8 [1..256,N/32,16,2] FP8 bytes")
    if ids.shape != (k,) or np.any(ids.astype(np.int64) >= books.shape[0]):
        raise ValueError("codebook_tile_ids shape/range does not match weights")
    _finite_fp8(books, "codebook_bytes")
    if k_order not in ("preserve", "codebook"):
        raise ValueError("k_order must be preserve or codebook")
    order = np.arange(k, dtype=np.int64)
    if k_order == "codebook":
        order = np.argsort(ids, kind="stable").astype(np.int64)
    target_ids = np.ascontiguousarray(ids[order])
    blocks = target_ids.reshape(-1, LUT_K)
    fixed_lut = None
    if np.all(blocks == blocks[:, :1]):
        fixed_lut = np.ascontiguousarray(books[blocks[:, 0]].reshape(k // LUT_K, n // ZN_N0, 32))
    elif k_order == "codebook":
        raise ValueError("tile ID populations cannot form homogeneous K256 blocks; retain per-K IDs")
    return BridgeWeights(
        packed_zn=pack_pairs_zn(indices[:, order]),
        codebook_bytes=books.copy(order="C"),
        codebook_tile_ids=target_ids,
        activation_gather=order,
        fixed_k256_lut=fixed_lut,
        k_reordered=not np.array_equal(order, np.arange(k)),
    )


def decode_bridge_bytes(weights: BridgeWeights) -> np.ndarray:
    """Independent CPU byte lookup for audit; not a production dense-W path."""
    validate_bridge_weights(weights)
    indices = unpack_pairs_zn(weights.packed_zn)
    row_groups = np.arange(indices.shape[0], dtype=np.int64) // (ZN_N0 // PAIR_SIZE)
    pairs = weights.codebook_bytes[weights.codebook_tile_ids[None, :], row_groups[:, None], indices]
    return np.ascontiguousarray(pairs.transpose(0, 2, 1).reshape(indices.shape[0] * PAIR_SIZE, -1))


def bridge_prepared_activation(
    activation_bytes: np.ndarray, row_scale: np.ndarray, row_bias: np.ndarray, weights: BridgeWeights
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preserve existing FP8 quantization. MX scales are ONE, not row scales.

    Returns (gathered_fp8_bytes, mx_ones, fp32_row_scale, fp32_row_bias).
    Existing RHT/sign/weight normalization MUST run before this function.
    The last two outputs must be consumed by a new FP32 epilogue; the received
    expert kernel cannot consume them and must NOT be called as a replacement.
    """
    validate_bridge_weights(weights)
    activation = _array(activation_bytes, "activation_bytes", np.uint8, 2)
    scale = _array(row_scale, "row_scale", np.float32, 1)
    bias = _array(row_bias, "row_bias", np.float32, 1)
    m, k = activation.shape
    if k != weights.activation_gather.size or scale.shape != (m,) or bias.shape != (m,):
        raise ValueError("activation/scales do not match [M,K]/[M]/[M]")
    _finite_fp8(activation, "activation_bytes")
    if not np.isfinite(scale).all() or np.any(scale <= 0) or not np.isfinite(bias).all():
        raise ValueError("row_scale must be positive finite and row_bias finite")
    return (
        np.ascontiguousarray(activation[:, weights.activation_gather]),
        np.full((m, k // MX_GROUP), E8M0_ONE, dtype=np.uint8),
        scale.copy(),
        bias.copy(),
    )


def fp32_epilogue(accumulator: np.ndarray, row_scale: np.ndarray, row_bias: np.ndarray) -> np.ndarray:
    """CPU semantic check; BF16 conversion happens only AFTER this result."""
    accumulator = _array(accumulator, "accumulator", np.float32, 2)
    scale = _array(row_scale, "row_scale", np.float32, 1)
    bias = _array(row_bias, "row_bias", np.float32, 1)
    if scale.shape != accumulator.shape[:1] or bias.shape != scale.shape:
        raise ValueError("epilogue row metadata shape mismatch")
    if not all(np.isfinite(value).all() for value in (accumulator, scale, bias)) or np.any(scale <= 0):
        raise ValueError("epilogue requires finite values and positive row scales")
    return accumulator * scale[:, None] + bias[:, None]


def audit_direct_contract(packed_indices, codebook_bytes, codebook_tile_ids) -> dict:
    weights = bridge_direct_weights(packed_indices, codebook_bytes, codebook_tile_ids)
    n, k = packed_indices.shape[0] * PAIR_SIZE, codebook_tile_ids.size
    unique_per_block = [int(np.unique(block).size) for block in codebook_tile_ids.reshape(-1, LUT_K)]
    populations = np.bincount(codebook_tile_ids, minlength=codebook_bytes.shape[0])
    return {
        "scope": "cpu_layout_contract_only",
        "n": n,
        "k": k,
        "mixed_k256_blocks": sum(count > 1 for count in unique_per_block),
        "total_k256_blocks": len(unique_per_block),
        "max_codebooks_per_k256": max(unique_per_block),
        "tile_populations": populations.tolist(),
        "stable_grouping_can_form_k256": bool(np.all(populations % LUT_K == 0)),
        "packed_bytes_preserved": weights.packed_zn.nbytes == packed_indices.nbytes,
        "original_host_shape_supported": n % 1024 == 0 and k % 1024 == 0,
        "fixed_k256_lut_compatible_without_k_reorder": weights.fixed_k256_lut is not None,
        "fp32_row_scale_bias_epilogue_required": True,
        "drop_in_replacement_ready": False,
        "device_execution_verified": False,
    }
