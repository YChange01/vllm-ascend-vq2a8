"""User-supplied W2 packed zN and pair-LUT layouts; transcription, see ../README.md."""

from __future__ import annotations

import numpy as np

ZN_K0 = 16
ZN_N0 = 32
CODEBOOK_K = 256
CODEBOOK_N = 32


def validate_w2_shape(codes: np.ndarray) -> None:
    if codes.dtype != np.uint8:
        raise TypeError(f"codes 必须是 uint8，实际为 {codes.dtype}")
    if codes.shape[-1] % 4 != 0:
        raise ValueError("最后一维必须是 4 的倍数")
    if codes.size and int(codes.max()) > 3:
        raise ValueError("2-bit code 必须位于 [0,3]")


def pack_2bit_codes(codes: np.ndarray) -> np.ndarray:
    """将最后一维每 4 个 2-bit code 打包为 1 个 uint8。"""
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    validate_w2_shape(codes)
    packed = codes[..., 0::4] | (codes[..., 1::4] << np.uint8(2))
    packed |= (codes[..., 2::4] << np.uint8(4)) | (codes[..., 3::4] << np.uint8(6))
    return np.ascontiguousarray(packed, dtype=np.uint8)


def unpack_2bit_codes(packed: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    out = np.empty(packed.shape[:-1] + (packed.shape[-1] * 4,), dtype=np.uint8)
    out[..., 0::4] = packed & np.uint8(0x03)
    out[..., 1::4] = (packed >> np.uint8(2)) & np.uint8(0x03)
    out[..., 2::4] = (packed >> np.uint8(4)) & np.uint8(0x03)
    out[..., 3::4] = (packed >> np.uint8(6)) & np.uint8(0x03)
    return out


def pack_zn_2bit_codes(codes: np.ndarray) -> np.ndarray:
    """逻辑 [...,N,K] -> packed zN [...,N/32,K/16,16,8]。"""
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    validate_w2_shape(codes)
    if codes.ndim < 2:
        raise ValueError("codes 至少需要 N、K 两维")
    n, k = codes.shape[-2:]
    if n % ZN_N0 != 0:
        raise ValueError(f"N 必须是 {ZN_N0} 的倍数")
    if k % ZN_K0 != 0:
        raise ValueError(f"K 必须是 {ZN_K0} 的倍数")
    prefix = codes.shape[:-2]
    dims = tuple(range(len(prefix)))
    logical_kn = np.swapaxes(codes, -2, -1)
    zn_codes = logical_kn.reshape(prefix + (k // ZN_K0, ZN_K0, n // ZN_N0, ZN_N0))
    zn_codes = zn_codes.transpose(dims + (len(prefix) + 2, len(prefix), len(prefix) + 1, len(prefix) + 3))
    return pack_2bit_codes(zn_codes)


def unpack_zn_2bit_codes(packed: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    if packed.ndim < 4:
        raise ValueError("packed zN 末四维必须是 [N/32,K/16,16,8]")
    if packed.shape[-2] != ZN_K0 or packed.shape[-1] != ZN_N0 // 4:
        raise ValueError(f"packed zN 最后两维必须是 [{ZN_K0},{ZN_N0 // 4}]")
    prefix = packed.shape[:-4]
    dims = tuple(range(len(prefix)))
    n_blocks, k_blocks = packed.shape[-4:-2]
    zn_codes = unpack_2bit_codes(packed)
    logical_kn = zn_codes.transpose(dims + (len(prefix) + 1, len(prefix) + 2, len(prefix), len(prefix) + 3))
    logical_kn = logical_kn.reshape(prefix + (k_blocks * ZN_K0, n_blocks * ZN_N0))
    return np.ascontiguousarray(np.swapaxes(logical_kn, -2, -1))


def reshape_codes_by_codebook(codes: np.ndarray) -> np.ndarray:
    """逻辑 code -> [...,K/256,N/32,32,256]。"""
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    validate_w2_shape(codes)
    if codes.ndim < 2:
        raise ValueError("codes 至少需要 N、K 两维")
    n, k = codes.shape[-2:]
    if n % CODEBOOK_N != 0:
        raise ValueError(f"N 必须是 {CODEBOOK_N} 的倍数")
    if k % CODEBOOK_K != 0:
        raise ValueError(f"K 必须是 {CODEBOOK_K} 的倍数")
    prefix = codes.shape[:-2]
    dims = tuple(range(len(prefix)))
    grouped = codes.reshape(prefix + (n // CODEBOOK_N, CODEBOOK_N, k // CODEBOOK_K, CODEBOOK_K))
    return np.ascontiguousarray(
        grouped.transpose(dims + (len(prefix) + 2, len(prefix), len(prefix) + 1, len(prefix) + 3))
    )


def restore_codes_from_codebook_groups(grouped: np.ndarray) -> np.ndarray:
    grouped = np.ascontiguousarray(grouped)
    if grouped.ndim < 4:
        raise ValueError("grouped code 末四维必须是 [K/256,N/32,32,256]")
    if grouped.shape[-2:] != (CODEBOOK_N, CODEBOOK_K):
        raise ValueError(f"grouped code 最后两维必须是 [{CODEBOOK_N},{CODEBOOK_K}]")
    prefix = grouped.shape[:-4]
    dims = tuple(range(len(prefix)))
    k_blocks, n_groups = grouped.shape[-4:-2]
    logical = grouped.transpose(dims + (len(prefix) + 1, len(prefix) + 2, len(prefix), len(prefix) + 3))
    return np.ascontiguousarray(logical.reshape(prefix + (n_groups * CODEBOOK_N, k_blocks * CODEBOOK_K)))


def build_pair_lut(levels: np.ndarray) -> np.ndarray:
    """4 scalar levels -> 16 pair entries; kernel LUT need not be separable."""
    if levels.shape[-1] != 4:
        raise ValueError("levels 最后一维必须包含 4 个值")
    pair_index = np.arange(16, dtype=np.uint8)
    first = np.take(levels, pair_index & np.uint8(0x03), axis=-1)
    second = np.take(levels, pair_index >> np.uint8(2), axis=-1)
    return np.ascontiguousarray(np.stack((first, second), axis=-1))
