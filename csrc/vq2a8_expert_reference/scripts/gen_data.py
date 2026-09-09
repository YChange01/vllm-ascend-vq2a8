#!/usr/bin/env python3
"""用户提供的 grouped MXFP8/W2 数据生成器整理稿；并非真实 VQ checkpoint 转换器。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ml_dtypes
import numpy as np
from w2_layout import (
    CODEBOOK_K,
    CODEBOOK_N,
    ZN_K0,
    ZN_N0,
    build_pair_lut,
    pack_zn_2bit_codes,
    reshape_codes_by_codebook,
    restore_codes_from_codebook_groups,
)

MAX_GROUP_COUNT = 128
REDUCTION_TILE = 1024
OUTPUT_TILE = 1024
MX_SCALE_GROUP = 32
E4M3FN_MAX = 448.0
E8M0_EXP_BIAS = 127
E8M0_MIN_EXP = -127
E8M0_MAX_EXP = 127


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 MXFP8 A/E8M0 scale、packed zN B、pair LUT、group_list 和 FP32 Golden"
    )
    parser.add_argument("G", type=int, help=f"组/专家数，范围 1..{MAX_GROUP_COUNT}")
    parser.add_argument("K", type=int, help=f"归约维，必须是 {REDUCTION_TILE} 的倍数")
    parser.add_argument("N", type=int, help=f"输出维，必须是 {OUTPUT_TILE} 的倍数")
    parser.add_argument("--group-list", help="逗号分隔的非累计 M_i；默认每组 1 行，如 2,1,0,3,0,0,1,1")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save-dequant-weight", action="store_true", help="额外保存较大的 output/golden_b_fp8.bin")
    return parser.parse_args()


def parse_group_list(text: str | None, group_count: int) -> np.ndarray:
    if text is None:
        return np.ones(group_count, dtype=np.int64)
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != group_count or any(not part for part in parts):
        raise ValueError(f"--group-list 必须恰好包含 {group_count} 个逗号分隔整数，实际为 {text!r}")
    try:
        return np.asarray([int(part) for part in parts], dtype=np.int64)
    except ValueError as error:
        raise ValueError(f"无效的 --group-list：{text!r}") from error


def validate_shape(group_count: int, k: int, n: int, group_list: np.ndarray) -> None:
    if not 1 <= group_count <= MAX_GROUP_COUNT:
        raise ValueError(f"G 必须位于 [1,{MAX_GROUP_COUNT}]，实际为 {group_count}")
    if k <= 0 or k % REDUCTION_TILE != 0:
        raise ValueError(f"K 必须是 {REDUCTION_TILE} 的正整数倍，实际为 {k}")
    if n <= 0 or n % OUTPUT_TILE != 0:
        raise ValueError(f"N 必须是 {OUTPUT_TILE} 的正整数倍，实际为 {n}")
    if group_list.shape != (group_count,):
        raise ValueError(f"group_list 形状应为 ({group_count},)，实际为 {group_list.shape}")
    if np.any(group_list < 0):
        raise ValueError(f"group_list[i] 必须为非负整数，实际为 {group_list.tolist()}")
    if int(group_list.sum()) == 0:
        raise ValueError("group_list 至少需要包含一个 token 行")


def quantize_mxfp8_e4m3(activation_f32: np.ndarray, fp8_dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """沿 K 轴每 32 元素生成一个 E8M0 scale，并将 A 量化为 E4M3FN。"""
    if activation_f32.ndim != 2 or activation_f32.shape[1] % MX_SCALE_GROUP != 0:
        raise ValueError(f"A 必须是二维矩阵且 K 可被 {MX_SCALE_GROUP} 整除，实际 shape={activation_f32.shape}")
    grouped = activation_f32.reshape(activation_f32.shape[0], -1, MX_SCALE_GROUP)
    max_abs = np.max(np.abs(grouped), axis=-1)
    safe_ratio = np.maximum(max_abs / E4M3FN_MAX, np.finfo(np.float32).tiny)
    exponents = np.ceil(np.log2(safe_ratio)).astype(np.int32)
    exponents = np.where(max_abs == 0.0, 0, exponents)
    exponents = np.clip(exponents, E8M0_MIN_EXP, E8M0_MAX_EXP)
    scale_values = np.exp2(exponents.astype(np.float32)).astype(np.float32)
    normalized = grouped / scale_values[..., None]
    normalized = np.clip(normalized, -E4M3FN_MAX, E4M3FN_MAX)
    activation_fp8 = np.ascontiguousarray(normalized.reshape(activation_f32.shape).astype(fp8_dtype))
    scale_codes = np.ascontiguousarray((exponents + E8M0_EXP_BIAS).astype(np.uint8))
    return activation_fp8, scale_codes, scale_values


def generate(group_count: int, k: int, n: int, group_list: np.ndarray, seed: int, save_b: bool) -> None:
    validate_shape(group_count, k, n, group_list)
    total_m = int(group_list.sum())
    input_dir, output_dir = Path("input"), Path("output")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fp8 = ml_dtypes.float8_e4m3fn
    if np.dtype(fp8).itemsize != 1:
        raise RuntimeError("float8_e4m3fn 必须占 1 Byte")
    k_codebook_count, n_group_count = k // CODEBOOK_K, n // CODEBOOK_N
    activation_source = rng.standard_normal((total_m, k), dtype=np.float32)
    activation, activation_scale, activation_scale_f32 = quantize_mxfp8_e4m3(activation_source, fp8)
    scales = rng.uniform(0.25, 1.5, size=(group_count, k_codebook_count, n_group_count, 1)).astype(np.float32)
    base_levels = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
    levels = (scales * base_levels).astype(fp8)
    scalar_codes = rng.integers(0, 4, size=(group_count, n, k), dtype=np.uint8)
    packed_codes_zn = pack_zn_2bit_codes(scalar_codes)
    pair_lut = build_pair_lut(levels)
    table = pair_lut.reshape(group_count, k_codebook_count, n_group_count, 32)
    codes_grouped = reshape_codes_by_codebook(scalar_codes)
    dequant_grouped = np.take_along_axis(levels[..., None, None, :], codes_grouped[..., None], axis=-1)[..., 0]
    dequant_weight = np.ascontiguousarray(restore_codes_from_codebook_groups(dequant_grouped), dtype=fp8)
    # Golden is relative to QUANTIZED A and B, not to unquantized activations.
    activation_f32 = (
        activation.astype(np.float32).reshape(total_m, k // MX_SCALE_GROUP, MX_SCALE_GROUP)
        * activation_scale_f32[..., None]
    ).reshape(total_m, k)
    golden_c = np.empty((total_m, n), dtype=np.float32)
    row_begin = 0
    for group_id, group_m_value in enumerate(group_list):
        group_m = int(group_m_value)
        row_end = row_begin + group_m
        if group_m:
            golden_c[row_begin:row_end] = (
                activation_f32[row_begin:row_end] @ dequant_weight[group_id].astype(np.float32).T
            )
        row_begin = row_end
    activation.tofile(input_dir / "input_a.bin")
    activation_scale.tofile(input_dir / "input_a_scale.bin")
    packed_codes_zn.tofile(input_dir / "input_b.bin")
    table.tofile(input_dir / "input_table.bin")
    group_list.tofile(input_dir / "input_group_list.bin")
    golden_c.tofile(output_dir / "golden_c.bin")
    if save_b:
        dequant_weight.tofile(output_dir / "golden_b_fp8.bin")
    metadata = {
        "groups": group_count,
        "experts": group_count,
        "group_list": group_list.tolist(),
        "group_list_semantics": "non-cumulative M_i",
        "total_m": total_m,
        "k": k,
        "n": n,
        "fractal_k0": ZN_K0,
        "fractal_n0": ZN_N0,
        "fractal_contiguous_axis": "N0",
        "decoded_fractal_order": ["N1", "K1", "K0", "N0"],
        "packed_fractal_order": ["N1", "K1", "K0", "N0/4"],
        "codebook_k": CODEBOOK_K,
        "codebook_n": CODEBOOK_N,
        "k0_fractals_per_codebook": CODEBOOK_K // ZN_K0,
        "n0_fractals_per_codebook": CODEBOOK_N // ZN_N0,
        "weights_per_codebook": CODEBOOK_K * CODEBOOK_N,
        "activation_dtype": "mxfp8_e4m3fn",
        "activation_scale_dtype": "float8_e8m0",
        "activation_scale_group_size": MX_SCALE_GROUP,
        "activation_scale_layout": "GM ND [M,K/64,2], two adjacent E8M0 reinterpreted as BF16 for DN2NZ",
        "weight_code_bits": 2,
        "weight_format": "zN K0=16 N0=32, N0-contiguous, packed W2",
        "weight_lut_dtype": "float8_e4m3fn",
        "output_dtype": "bfloat16",
        "golden_output_dtype": "float32",
        "input_group_list_shape": [group_count],
        "input_group_list_dtype": "int64",
        "input_a_shape": [total_m, k],
        "input_a_scale_shape": [total_m, k // 64, 2],
        "input_b_packed_zn_shape": [group_count, n // ZN_N0, k // ZN_K0, ZN_K0, ZN_N0 // 4],
        "input_table_shape": [group_count, k // CODEBOOK_K, n // CODEBOOK_N, 32],
        "output_shape": [total_m, n],
        "seed": seed,
    }
    (input_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Group list int64:   {group_list.tolist()}, {group_list.nbytes} bytes")
    print(f"A MXFP8 E4M3:       {activation.shape}, {activation.nbytes} bytes")
    print(f"A E8M0 scale:       {(total_m, k // 64, 2)}, {activation_scale.nbytes} bytes")
    print(f"B W2 packed zN:     {packed_codes_zn.shape}, {packed_codes_zn.nbytes} bytes")
    print("  B axes:            [G, N1, K1, K0=16, N0/4=8]")
    print(f"Pair LUT K256xN32:  {table.shape}, {table.nbytes} bytes")
    print(f"Golden C FP32:      {golden_c.shape}, {golden_c.nbytes} bytes")


if __name__ == "__main__":
    args = parse_args()
    generate(args.G, args.K, args.N, parse_group_list(args.group_list, args.G), args.seed, args.save_dequant_weight)
