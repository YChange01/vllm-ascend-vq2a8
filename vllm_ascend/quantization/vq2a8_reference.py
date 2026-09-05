# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU references for canonical VQ2A8 expert matrices."""

from __future__ import annotations

import functools
import math

import torch

from vllm_ascend.quantization.vq2a8_artifact import (
    VQ2_CODEBOOK_SIZE,
    VQ2_INDEX_BITS,
    VQ2_INDICES_PER_WORD,
    VQ2_VECTOR_LENGTH,
    VQ2MatrixSpec,
)

LITERAL_ORACLE_MAX_ELEMENTS = 1_000_000
VQ2_FP8_MIN_SCALE = 1e-12


def unpack_vq2_indices(
    packed: torch.Tensor,
    num_vectors: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Unpack little-endian 4-bit indices from signed int32 words."""
    if not isinstance(num_vectors, int) or isinstance(num_vectors, bool) or num_vectors < 0:
        raise ValueError(f"num_vectors must be non-negative, got {num_vectors}.")
    if packed.dtype != torch.int32 or packed.ndim != 1:
        raise ValueError(
            f"Packed VQ2A8 indices must be one-dimensional int32, got "
            f"dtype={packed.dtype}, shape={tuple(packed.shape)}."
        )
    required_words = math.ceil(num_vectors / VQ2_INDICES_PER_WORD)
    words = packed.reshape(-1)
    if words.numel() != required_words:
        raise ValueError(
            f"Packed VQ2A8 tensor has {words.numel()} words, expected {required_words} for {num_vectors} vectors."
        )
    words = words.to(device=device, dtype=torch.int64) & 0xFFFFFFFF
    positions = torch.arange(num_vectors, device=device, dtype=torch.int64)
    word_indices = torch.div(positions, VQ2_INDICES_PER_WORD, rounding_mode="floor")
    shifts = positions.remainder(VQ2_INDICES_PER_WORD) * VQ2_INDEX_BITS
    return (words[word_indices] >> shifts) & (VQ2_CODEBOOK_SIZE - 1)


@functools.cache
def _sylvester_hadamard(size: int) -> torch.Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {size}.")
    matrix = torch.tensor([[1.0]], dtype=torch.float64)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(size)


def _inverse_rht(
    weight: torch.Tensor,
    block_size: int,
    signs: torch.Tensor,
) -> torch.Tensor:
    width = signs.numel()
    if weight.shape[-1] != width:
        raise ValueError(f"RHT width mismatch: weight={weight.shape[-1]}, signs={width}.")
    hadamard = _sylvester_hadamard(block_size).to(device=weight.device, dtype=weight.dtype)
    shape = weight.shape
    blocks = weight.reshape(*shape[:-1], width // block_size, block_size)
    blocks = blocks @ hadamard
    blocks *= signs.to(device=weight.device, dtype=weight.dtype).reshape(width // block_size, block_size)
    return blocks.reshape(shape)


def _forward_rht_activation(
    activation: torch.Tensor,
    block_size: int,
    signs: torch.Tensor,
) -> torch.Tensor:
    width = signs.numel()
    if activation.shape[-1] != width:
        raise ValueError(f"RHT width mismatch: activation={activation.shape[-1]}, signs={width}.")
    hadamard = _sylvester_hadamard(block_size).to(device=activation.device, dtype=activation.dtype)
    shape = activation.shape
    blocks = activation.reshape(*shape[:-1], width // block_size, block_size)
    blocks = blocks * signs.to(device=activation.device, dtype=activation.dtype).reshape(
        width // block_size, block_size
    )
    return (blocks @ hadamard).reshape(shape)


def decode_codebook_weight(
    tensors: dict[str, torch.Tensor],
    spec: VQ2MatrixSpec,
    device: torch.device | str = "cpu",
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Vectorized decode before permutation, normalization, and RHT."""
    indices = unpack_vq2_indices(tensors["packed_indices"], spec.num_vectors, device)
    codebooks = tensors["codebooks"].to(device=device, dtype=compute_dtype)
    expected_shape = (
        spec.column_tiles,
        spec.row_tiles,
        VQ2_CODEBOOK_SIZE,
        VQ2_VECTOR_LENGTH,
    )
    if tuple(codebooks.shape) != expected_shape:
        raise ValueError(f"Codebook shape is {tuple(codebooks.shape)}, expected {expected_shape}.")

    weight = torch.empty((spec.rows, spec.columns), device=device, dtype=compute_dtype)
    position = 0
    for column_tile in range(spec.column_tiles):
        start = column_tile * spec.group_size
        end = min(start + spec.group_size, spec.columns)
        width = end - start
        tile_vector_count = spec.row_tiles * width * spec.vectors_per_row_group
        tile_indices = indices[position : position + tile_vector_count].reshape(
            spec.row_tiles,
            width * spec.vectors_per_row_group,
        )
        position += tile_vector_count
        vectors = torch.gather(
            codebooks[column_tile],
            1,
            tile_indices.unsqueeze(-1).expand(-1, -1, VQ2_VECTOR_LENGTH),
        )
        block = vectors.reshape(spec.row_tiles, width, spec.row_group_size).transpose(1, 2).reshape(spec.rows, width)
        weight[:, start:end] = block
    if position != spec.num_vectors:
        raise ValueError(f"Decoded {position} VQ2A8 vectors, expected {spec.num_vectors}.")
    return weight


def apply_vq2_weight_transforms(
    weight: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    spec: VQ2MatrixSpec,
) -> torch.Tensor:
    """Apply the producer's inverse transforms to a codebook matrix."""
    if spec.enable_permutation:
        permutation = tensors["perm"].to(device=weight.device, dtype=torch.int64)
        weight = weight[:, torch.argsort(permutation)]
    if spec.enable_normalization:
        scale = tensors["weight_scale"].to(device=weight.device, dtype=weight.dtype)
        bias = tensors["weight_bias"].to(device=weight.device, dtype=weight.dtype)
        weight = weight * scale.unsqueeze(spec.norm_dimension)
        weight += bias.unsqueeze(spec.norm_dimension)
    if spec.enable_rht:
        weight = _inverse_rht(
            weight,
            spec.rht_block_size,
            tensors["rht_sign"],
        )
        weight = weight[..., : spec.rht_true_columns]
    return weight


def decode_expert_weight(
    tensors: dict[str, torch.Tensor],
    spec: VQ2MatrixSpec,
    device: torch.device | str = "cpu",
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode one canonical expert matrix as ``W[out, in]``."""
    weight = decode_codebook_weight(tensors, spec, device, compute_dtype)
    return apply_vq2_weight_transforms(weight, tensors, spec).reshape(spec.original_shape)


def transform_vq2_activation(
    activation: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    spec: VQ2MatrixSpec,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Move permutation, normalization, and RHT to the activation side."""
    if activation.ndim == 0:
        raise ValueError("Activation must have at least one dimension.")
    if activation.shape[-1] != spec.rht_true_columns:
        raise ValueError(f"Activation width is {activation.shape[-1]}, expected {spec.rht_true_columns}.")
    transformed = activation.to(compute_dtype)
    if spec.rht_true_columns < spec.columns:
        transformed = torch.nn.functional.pad(
            transformed,
            (0, spec.columns - spec.rht_true_columns),
        )
    if spec.enable_rht:
        transformed = _forward_rht_activation(
            transformed,
            spec.rht_block_size,
            tensors["rht_sign"],
        )

    bias_correction = None
    if spec.enable_normalization:
        scale = tensors["weight_scale"].to(device=transformed.device, dtype=transformed.dtype)
        bias = tensors["weight_bias"].to(device=transformed.device, dtype=transformed.dtype)
        bias_correction = transformed @ bias
        transformed = transformed * scale
    if spec.enable_permutation:
        permutation = tensors["perm"].to(device=transformed.device, dtype=torch.int64)
        transformed = transformed[..., permutation]
    return transformed, bias_correction


def vq2_matmul_reference(
    activation: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    spec: VQ2MatrixSpec,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Evaluate ``activation @ W.T`` without materializing transformed W."""
    codebook_weight = decode_codebook_weight(tensors, spec, activation.device, compute_dtype)
    transformed, bias_correction = transform_vq2_activation(activation, tensors, spec, compute_dtype)
    output = transformed @ codebook_weight.transpose(0, 1)
    if bias_correction is not None:
        output += bias_correction.unsqueeze(-1)
    return output


def prepare_repacked_vq2a8_activation_reference(
    activation: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    rht_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare activation for a TP-local repacked direct-kernel artifact.

    The canonical permutation must already have been absorbed by the offline
    repack. This helper therefore applies RHT, normalization and dynamic A8,
    but intentionally does not apply ``perm``.
    """
    if activation.ndim != 2:
        raise ValueError(f"Expected a two-dimensional activation, got {activation.shape}.")
    input_size = activation.shape[1]
    if input_size <= 0:
        raise ValueError("Activation width must be positive.")
    for name, tensor in (
        ("weight_scale", weight_scale),
        ("weight_bias", weight_bias),
        ("rht_sign", rht_sign),
    ):
        if tuple(tensor.shape) != (input_size,):
            raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected ({input_size},).")
    if weight_scale.dtype != torch.float32 or weight_bias.dtype != torch.float32:
        raise ValueError("weight_scale and weight_bias must be float32.")
    if rht_sign.dtype != torch.int8:
        raise ValueError("rht_sign must be int8.")
    if (
        not isinstance(rht_block_size, int)
        or isinstance(rht_block_size, bool)
        or rht_block_size <= 0
        or rht_block_size & (rht_block_size - 1)
        or input_size % rht_block_size
    ):
        raise ValueError(
            f"rht_block_size must be a positive power of two that divides "
            f"activation width {input_size}, got {rht_block_size!r}."
        )
    if not bool(torch.isfinite(activation.float()).all()):
        raise ValueError("activation contains non-finite values.")
    if not bool(torch.isfinite(weight_scale).all()) or not bool(torch.isfinite(weight_bias).all()):
        raise ValueError("weight_scale or weight_bias contains non-finite values.")
    signs = rht_sign.to(torch.int16)
    if not bool(((signs == -1) | (signs == 1)).all()):
        raise ValueError("rht_sign contains values other than -1 and 1.")
    rotated = _forward_rht_activation(
        activation.float(),
        rht_block_size,
        rht_sign,
    )
    bias_correction = rotated @ weight_bias.float()
    transformed = rotated * weight_scale.float().unsqueeze(0)
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    absolute_maximum = transformed.abs().amax(dim=-1)
    activation_scale = torch.clamp(
        absolute_maximum / fp8_max,
        min=VQ2_FP8_MIN_SCALE,
    )
    quantized = torch.clamp(
        transformed / activation_scale.unsqueeze(-1),
        -fp8_max,
        fp8_max,
    ).to(fp8_dtype)
    return quantized, activation_scale, bias_correction


def decode_expert_weight_literal(
    tensors: dict[str, torch.Tensor],
    spec: VQ2MatrixSpec,
) -> torch.Tensor:
    """Slow scalar oracle independent of the vectorized decode helpers."""
    if spec.num_elements > LITERAL_ORACLE_MAX_ELEMENTS:
        raise ValueError(
            f"Literal oracle refuses {spec.num_elements} elements; limit is {LITERAL_ORACLE_MAX_ELEMENTS}."
        )
    if any(tensor.device.type != "cpu" for tensor in tensors.values()):
        raise ValueError("Literal oracle requires CPU tensors.")
    _validate_literal_inputs(tensors, spec)

    packed_words = [int(value) & 0xFFFFFFFF for value in tensors["packed_indices"].reshape(-1).tolist()]
    codebooks = tensors["codebooks"].to(torch.float64)
    untransformed = torch.empty((spec.rows, spec.columns), dtype=torch.float64)
    flat_position = 0
    for column_tile in range(spec.column_tiles):
        column_start = column_tile * spec.group_size
        tile_width = min(spec.group_size, spec.columns - column_start)
        for row_tile in range(spec.row_tiles):
            for column_offset in range(tile_width):
                for vector_offset in range(spec.vectors_per_row_group):
                    word = packed_words[flat_position // VQ2_INDICES_PER_WORD]
                    shift = (flat_position % VQ2_INDICES_PER_WORD) * VQ2_INDEX_BITS
                    code = (word >> shift) & (VQ2_CODEBOOK_SIZE - 1)
                    for component in range(VQ2_VECTOR_LENGTH):
                        row = row_tile * spec.row_group_size + vector_offset * VQ2_VECTOR_LENGTH + component
                        column = column_start + column_offset
                        untransformed[row, column] = codebooks[
                            column_tile,
                            row_tile,
                            code,
                            component,
                        ]
                    flat_position += 1
    if flat_position != spec.num_vectors:
        raise ValueError(f"Literal oracle consumed {flat_position} vectors, expected {spec.num_vectors}.")

    if spec.enable_permutation:
        permutation = [int(value) for value in tensors["perm"].tolist()]
        inverse_permutation = [0] * spec.columns
        for source_column, physical_column in enumerate(permutation):
            inverse_permutation[physical_column] = source_column
    else:
        inverse_permutation = list(range(spec.columns))

    scale = tensors["weight_scale"].to(torch.float64).tolist() if spec.enable_normalization else [1.0] * spec.columns
    bias = tensors["weight_bias"].to(torch.float64).tolist() if spec.enable_normalization else [0.0] * spec.columns
    normalized = torch.empty_like(untransformed)
    for row in range(spec.rows):
        for column in range(spec.columns):
            value = float(untransformed[row, inverse_permutation[column]])
            normalized[row, column] = value * scale[column] + bias[column]

    if not spec.enable_rht:
        return normalized[:, : spec.rht_true_columns]

    signs = [int(value) for value in tensors["rht_sign"].tolist()]
    weight = torch.empty_like(normalized)
    inverse_norm = 1.0 / math.sqrt(spec.rht_block_size)
    for row in range(spec.rows):
        for block_start in range(0, spec.columns, spec.rht_block_size):
            for output_offset in range(spec.rht_block_size):
                value = 0.0
                for input_offset in range(spec.rht_block_size):
                    parity = (input_offset & output_offset).bit_count() & 1
                    coefficient = -inverse_norm if parity else inverse_norm
                    value += float(normalized[row, block_start + input_offset]) * coefficient
                output_column = block_start + output_offset
                weight[row, output_column] = value * signs[output_column]
    return weight[:, : spec.rht_true_columns]


def vq2_matmul_literal(
    activation: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    spec: VQ2MatrixSpec,
) -> torch.Tensor:
    """Evaluate a small matrix through the independent scalar oracle."""
    if activation.device.type != "cpu":
        raise ValueError("Literal oracle requires a CPU activation.")
    weight = decode_expert_weight_literal(tensors, spec)
    return activation.to(torch.float64) @ weight.transpose(0, 1)


def _validate_literal_inputs(tensors: dict[str, torch.Tensor], spec: VQ2MatrixSpec) -> None:
    expected_fields = set(spec.expected_tensor_headers())
    if set(tensors) != expected_fields:
        raise ValueError(
            f"Literal oracle fields differ; missing={sorted(expected_fields - set(tensors))}, "
            f"extra={sorted(set(tensors) - expected_fields)}."
        )
    packed = tensors["packed_indices"]
    if packed.dtype != torch.int32 or tuple(packed.shape) != (spec.packed_word_count,):
        raise ValueError(
            f"Literal packed_indices must be int32[{spec.packed_word_count}], got "
            f"dtype={packed.dtype}, shape={tuple(packed.shape)}."
        )
    codebooks = tensors["codebooks"]
    expected_codebook_shape = (
        spec.column_tiles,
        spec.row_tiles,
        VQ2_CODEBOOK_SIZE,
        VQ2_VECTOR_LENGTH,
    )
    if tuple(codebooks.shape) != expected_codebook_shape or not codebooks.dtype.is_floating_point:
        raise ValueError(
            f"Literal codebooks must be floating point with shape "
            f"{expected_codebook_shape}, got dtype={codebooks.dtype}, "
            f"shape={tuple(codebooks.shape)}."
        )
    if not bool(torch.isfinite(codebooks.float()).all()):
        raise ValueError("Literal codebooks contain non-finite values.")
    if spec.enable_permutation:
        permutation = tensors["perm"]
        if permutation.dtype != torch.int32 or tuple(permutation.shape) != (spec.columns,):
            raise ValueError("Literal perm must be one-dimensional int32 with cols entries.")
        expected = torch.arange(spec.columns, dtype=torch.int64)
        if not torch.equal(torch.sort(permutation.to(torch.int64)).values, expected):
            raise ValueError("Literal perm is not a bijection over [0, cols).")
    if spec.enable_normalization:
        for name in ("weight_scale", "weight_bias"):
            tensor = tensors[name]
            if tensor.dtype != torch.float32 or tuple(tensor.shape) != (spec.columns,):
                raise ValueError(f"Literal {name} must be float32[cols].")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Literal {name} contains non-finite values.")
    if spec.enable_rht:
        signs = tensors["rht_sign"]
        if signs.dtype != torch.int8 or tuple(signs.shape) != (spec.columns,):
            raise ValueError("Literal rht_sign must be int8[cols].")
        signs_int = signs.to(torch.int16)
        if not bool(((signs_int == -1) | (signs_int == 1)).all()):
            raise ValueError("Literal rht_sign contains values other than -1 and 1.")
