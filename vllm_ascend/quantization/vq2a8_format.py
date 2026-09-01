# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical VPTQ VQ2 expert artifact parsing and CPU references."""

from __future__ import annotations

import functools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

VQ2_CODEBOOK_SIZE = 16
VQ2_INDEX_BITS = 4
VQ2_VECTOR_LENGTH = 2
VQ2_MATRIX_KINDS = ("gate_up", "down")
ASCEND_VQ2_TP_FORMAT = "vq2a8_ascend_tp_v1"

_EXPERT_NAME_PATTERN = re.compile(
    r"^(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<kind>gate_up|down)$"
)


@dataclass(frozen=True)
class VQ2LayerSummary:
    """Header-only summary of one canonical VQ2 layer artifact."""

    layer_index: int
    expert_ids: tuple[int, ...]
    matrix_count: int
    tensor_count: int


def extract_decoder_layer_index(prefix: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", prefix)
    if match is None:
        raise ValueError(f"Cannot extract decoder layer index from {prefix!r}.")
    return int(match.group(1))


def parse_expert_name(name: str) -> tuple[int, int, str]:
    """Return ``(layer, expert, kind)`` from a canonical matrix name."""
    match = _EXPERT_NAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Invalid canonical VQ2 matrix name: {name!r}.")
    return (
        int(match.group("layer")),
        int(match.group("expert")),
        match.group("kind"),
    )


def layer_artifact_paths(
    experts_path: str | Path, layer_index: int
) -> tuple[Path, Path]:
    root = Path(experts_path)
    stem = f"experts_vq_layer_{layer_index}"
    return root / f"{stem}.json", root / f"{stem}.safetensors"


def load_layer_metadata(
    experts_path: str | Path, layer_index: int
) -> dict[str, dict[str, Any]]:
    metadata_path, tensor_path = layer_artifact_paths(experts_path, layer_index)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing VQ2 metadata: {metadata_path}.")
    if not tensor_path.is_file():
        raise FileNotFoundError(f"Missing VQ2 tensors: {tensor_path}.")
    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError(f"VQ2 metadata must be a non-empty object: {metadata_path}.")
    for name, matrix_metadata in metadata.items():
        name_layer, _, _ = parse_expert_name(name)
        if name_layer != layer_index:
            raise ValueError(
                f"Matrix {name!r} belongs to layer {name_layer}, expected "
                f"layer {layer_index}."
            )
        if not isinstance(matrix_metadata, dict):
            raise ValueError(f"VQ2 matrix metadata must be an object: {name!r}.")
    return metadata


def required_tensor_fields(metadata: dict[str, Any]) -> tuple[str, ...]:
    fields = ["packed_indices", "codebooks"]
    if metadata.get("enable_perm", False):
        fields.append("perm")
    if metadata.get("enable_norm", False):
        fields.extend(("weight_scale", "weight_bias"))
    if metadata.get("enable_rht", False):
        fields.append("rht_sign")
    return tuple(fields)


def inspect_layer_artifact(
    experts_path: str | Path, layer_index: int
) -> VQ2LayerSummary:
    """Validate metadata/tensor headers without reading tensor payloads."""
    metadata = load_layer_metadata(experts_path, layer_index)
    _, tensor_path = layer_artifact_paths(experts_path, layer_index)
    expert_kinds: dict[int, set[str]] = {}
    expected_keys: set[str] = set()
    for name, matrix_metadata in metadata.items():
        _, expert_id, kind = parse_expert_name(name)
        expert_kinds.setdefault(expert_id, set()).add(kind)
        expected_keys.update(
            f"{name}.{field}" for field in required_tensor_fields(matrix_metadata)
        )
    for expert_id, kinds in expert_kinds.items():
        if kinds != set(VQ2_MATRIX_KINDS):
            raise ValueError(
                f"Layer {layer_index} expert {expert_id} has matrices "
                f"{sorted(kinds)}, expected {list(VQ2_MATRIX_KINDS)}."
            )
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
    missing = sorted(expected_keys - actual_keys)
    if missing:
        raise ValueError(
            f"Layer {layer_index} is missing {len(missing)} VQ2 tensors; "
            f"first missing key: {missing[0]}."
        )
    return VQ2LayerSummary(
        layer_index=layer_index,
        expert_ids=tuple(sorted(expert_kinds)),
        matrix_count=len(metadata),
        tensor_count=len(actual_keys),
    )


def inspect_vq2_directory(experts_path: str | Path) -> list[VQ2LayerSummary]:
    root = Path(experts_path)
    if not root.is_dir():
        raise FileNotFoundError(f"VQ2 expert directory not found: {root}.")
    layer_indices: list[int] = []
    for metadata_path in root.glob("experts_vq_layer_*.json"):
        suffix = metadata_path.stem.removeprefix("experts_vq_layer_")
        if suffix.isdigit():
            layer_indices.append(int(suffix))
    if not layer_indices:
        raise ValueError(f"No canonical VQ2 layer artifacts found in {root}.")
    layer_indices.sort()
    expected = list(range(layer_indices[0], layer_indices[-1] + 1))
    if layer_indices != expected:
        missing = sorted(set(expected) - set(layer_indices))
        raise ValueError(f"VQ2 layer sequence has gaps; missing layers: {missing}.")
    return [inspect_layer_artifact(root, index) for index in layer_indices]


def load_expert_tensors(
    experts_path: str | Path,
    layer_index: int,
    expert_id: int,
    kind: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if kind not in VQ2_MATRIX_KINDS:
        raise ValueError(f"Unknown VQ2 matrix kind: {kind!r}.")
    metadata = load_layer_metadata(experts_path, layer_index)
    name = f"{layer_index}.mlp.experts.{expert_id}.{kind}"
    if name not in metadata:
        raise KeyError(f"VQ2 matrix not found: {name}.")
    matrix_metadata = metadata[name]
    _, tensor_path = layer_artifact_paths(experts_path, layer_index)
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        tensors = {
            field: handle.get_tensor(f"{name}.{field}")
            for field in required_tensor_fields(matrix_metadata)
        }
    return tensors, matrix_metadata


def unpack_vq2_indices(
    packed: torch.Tensor,
    num_vectors: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Unpack little-endian 4-bit VQ indices stored in int32 words."""
    if num_vectors < 0:
        raise ValueError(f"num_vectors must be non-negative, got {num_vectors}.")
    words = packed.reshape(-1).to(device=device, dtype=torch.int64) & 0xFFFFFFFF
    required_words = (num_vectors + 7) // 8
    if words.numel() < required_words:
        raise ValueError(
            f"Packed VQ2 tensor has {words.numel()} words, but {required_words} "
            f"are required for {num_vectors} vectors."
        )
    positions = torch.arange(num_vectors, device=device, dtype=torch.int64)
    word_indices = torch.div(positions, 8, rounding_mode="floor")
    shifts = positions.remainder(8) * VQ2_INDEX_BITS
    return (words[word_indices] >> shifts) & (VQ2_CODEBOOK_SIZE - 1)


def unpack_repacked_indices(
    packed: torch.Tensor, input_size: int
) -> torch.Tensor:
    """Unpack a TP-local index grid to ``[output_pair, input]``."""
    if packed.ndim != 2:
        raise ValueError(
            f"Repacked VQ2 indices must be 2D, got {tuple(packed.shape)}."
        )
    positions = torch.arange(input_size, device=packed.device, dtype=torch.int64)
    words = packed.to(torch.int64) & 0xFFFFFFFF
    return (words[:, positions // 8] >> ((positions % 8) * 4)) & 15


@functools.cache
def _sylvester_hadamard(size: int) -> torch.Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {size}.")
    matrix = torch.tensor([[1.0]])
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
    weight: torch.Tensor, block_size: int, sign: torch.Tensor
) -> torch.Tensor:
    width = sign.numel()
    if weight.shape[-1] != width:
        raise ValueError(
            f"RHT width mismatch: weight={weight.shape[-1]}, sign={width}."
        )
    hadamard = _sylvester_hadamard(block_size).to(
        device=weight.device, dtype=weight.dtype
    )
    shape = weight.shape
    blocks = weight.reshape(*shape[:-1], width // block_size, block_size)
    blocks = blocks @ hadamard
    blocks *= sign.to(device=weight.device, dtype=weight.dtype).reshape(
        width // block_size, block_size
    )
    return blocks.reshape(shape)


def _forward_rht_activation(
    activation: torch.Tensor, block_size: int, sign: torch.Tensor
) -> torch.Tensor:
    width = sign.numel()
    if activation.shape[-1] != width:
        raise ValueError(
            f"RHT width mismatch: activation={activation.shape[-1]}, sign={width}."
        )
    hadamard = _sylvester_hadamard(block_size).to(
        device=activation.device, dtype=activation.dtype
    )
    shape = activation.shape
    blocks = activation.reshape(*shape[:-1], width // block_size, block_size)
    blocks = blocks * sign.to(device=activation.device, dtype=activation.dtype).reshape(
        width // block_size, block_size
    )
    return (blocks @ hadamard).reshape(shape)


def decode_codebook_weight(
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    device: torch.device | str,
) -> torch.Tensor:
    """Decode indices and codebooks without inverse weight transforms."""
    rows = int(metadata["rows"])
    columns = int(metadata["cols"])
    row_tiles = int(metadata["n_row_tiles"])
    column_tiles = int(metadata["n_col_tiles"])
    row_group_size = int(metadata["row_group_size"])
    group_size = int(metadata["group_size"])
    codebook_size = int(metadata["K"])
    index_bits = int(metadata["index_bits"])
    vector_length = int(metadata["vector_len"])
    if (
        index_bits != VQ2_INDEX_BITS
        or codebook_size != VQ2_CODEBOOK_SIZE
        or vector_length != VQ2_VECTOR_LENGTH
    ):
        raise ValueError(
            "VQ2A8 requires index_bits=4, K=16, and vector_len=2, got "
            f"index_bits={index_bits}, K={codebook_size}, "
            f"vector_len={vector_length}."
        )
    if row_group_size % vector_length:
        raise ValueError(
            f"row_group_size={row_group_size} is not divisible by "
            f"vector_len={vector_length}."
        )
    if rows != row_tiles * row_group_size:
        raise ValueError(
            f"rows={rows} does not match n_row_tiles * row_group_size "
            f"({row_tiles * row_group_size})."
        )

    num_vectors = int(metadata["n_vectors"])
    indices = unpack_vq2_indices(tensors["packed_indices"], num_vectors, device)
    codebooks = tensors["codebooks"].to(device=device).float()
    expected_codebook_shape = (
        column_tiles,
        row_tiles,
        codebook_size,
        vector_length,
    )
    if tuple(codebooks.shape) != expected_codebook_shape:
        raise ValueError(
            f"Codebook shape is {tuple(codebooks.shape)}, expected "
            f"{expected_codebook_shape}."
        )

    weight = torch.empty((rows, columns), device=device, dtype=torch.float32)
    vectors_per_row_group = row_group_size // vector_length
    position = 0
    for column_tile in range(column_tiles):
        start = column_tile * group_size
        end = min(start + group_size, columns)
        width = end - start
        tile_vector_count = row_tiles * width * vectors_per_row_group
        tile_indices = indices[position : position + tile_vector_count].reshape(
            row_tiles, width * vectors_per_row_group
        )
        position += tile_vector_count
        vectors = torch.gather(
            codebooks[column_tile],
            1,
            tile_indices.unsqueeze(-1).expand(-1, -1, vector_length),
        )
        block = (
            vectors.reshape(row_tiles, width, row_group_size)
            .transpose(1, 2)
            .reshape(rows, width)
        )
        weight[:, start:end] = block
    if position != num_vectors:
        raise ValueError(
            f"Decoded {position} VQ vectors, but metadata declares {num_vectors}."
        )
    return weight


def apply_vq2_weight_transforms(
    weight: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> torch.Tensor:
    """Apply inverse permutation, normalization, and RHT transforms."""
    if metadata.get("enable_perm", False):
        permutation = tensors["perm"].to(device=weight.device, dtype=torch.int64)
        weight = weight[:, torch.argsort(permutation)]
    norm_dimension = int(metadata["norm_dim"])
    if metadata.get("enable_norm", False):
        scale = tensors["weight_scale"].to(device=weight.device).float()
        bias = tensors["weight_bias"].to(device=weight.device).float()
        weight = weight * scale.unsqueeze(norm_dimension)
        weight += bias.unsqueeze(norm_dimension)
    if metadata.get("enable_rht", False):
        weight = _inverse_rht(
            weight, int(metadata["rht_block_size"]), tensors["rht_sign"]
        )
        weight = weight[..., : int(metadata["rht_true_columns"])]
    return weight


def transform_vq2_activation(
    activation: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Move inverse weight transforms to the activation side."""
    if int(metadata["norm_dim"]) != 0:
        raise ValueError("VQ2A8 direct matmul requires norm_dim=0.")
    width = int(metadata["cols"])
    true_width = int(metadata.get("rht_true_columns", width))
    if activation.shape[-1] != true_width:
        raise ValueError(
            f"Activation width is {activation.shape[-1]}, expected {true_width}."
        )
    transformed = activation.float()
    if true_width < width:
        transformed = torch.nn.functional.pad(transformed, (0, width - true_width))
    if metadata.get("enable_rht", False):
        transformed = _forward_rht_activation(
            transformed, int(metadata["rht_block_size"]), tensors["rht_sign"]
        )

    bias_correction = None
    if metadata.get("enable_norm", False):
        scale = tensors["weight_scale"].to(
            device=transformed.device, dtype=transformed.dtype
        )
        bias = tensors["weight_bias"].to(
            device=transformed.device, dtype=transformed.dtype
        )
        bias_correction = transformed @ bias
        transformed = transformed * scale
    if metadata.get("enable_perm", False):
        permutation = tensors["perm"].to(
            device=transformed.device, dtype=torch.int64
        )
        transformed = transformed[..., permutation]
    return transformed, bias_correction


def vq2_matmul_reference(
    activation: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> torch.Tensor:
    """CPU correctness reference for direct VQ2 matrix multiplication."""
    codebook_weight = decode_codebook_weight(tensors, metadata, activation.device)
    transformed, bias_correction = transform_vq2_activation(
        activation, tensors, metadata
    )
    output = transformed @ codebook_weight.transpose(0, 1)
    if bias_correction is not None:
        output += bias_correction.unsqueeze(-1)
    return output


def decode_expert_weight(
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    device: torch.device | str,
) -> torch.Tensor:
    """Decode one canonical VQ2 expert matrix to BF16."""
    weight = decode_codebook_weight(tensors, metadata, device)
    weight = apply_vq2_weight_transforms(weight, tensors, metadata)
    return weight.reshape(metadata["orig_shape"]).to(torch.bfloat16)


def decode_repacked_vq2_weight(
    packed_indices: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_tile_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    rht_block_size: int,
    row_group_size: int = 32,
) -> torch.Tensor:
    """Decode one TP-local matrix for repack correctness tests."""
    input_size = codebook_tile_ids.numel()
    indices = unpack_repacked_indices(packed_indices, input_size)
    output_size = indices.shape[0] * VQ2_VECTOR_LENGTH
    if output_size % row_group_size:
        raise ValueError(
            f"Output size {output_size} is not divisible by "
            f"row group size {row_group_size}."
        )
    expected_codebook_shape = (
        codebooks.shape[0],
        output_size // row_group_size,
        VQ2_CODEBOOK_SIZE,
        VQ2_VECTOR_LENGTH,
    )
    if tuple(codebooks.shape) != expected_codebook_shape:
        raise ValueError(
            f"Codebook shape is {tuple(codebooks.shape)}, expected "
            f"{expected_codebook_shape}."
        )
    output_rows = torch.arange(output_size, device=indices.device)
    vector_rows = torch.div(output_rows, VQ2_VECTOR_LENGTH, rounding_mode="floor")
    row_tiles = torch.div(output_rows, row_group_size, rounding_mode="floor")
    components = output_rows.remainder(VQ2_VECTOR_LENGTH)
    source_tiles = codebook_tile_ids.to(torch.int64)
    # Ascend aclnnIndex cannot use an FP8 tensor as the indexed source. The
    # reference path reconstructs BF16 weights, and BF16 represents every
    # finite E4M3FN value exactly, so cast before the advanced lookup.
    lookup_codebooks = codebooks
    if lookup_codebooks.dtype == torch.float8_e4m3fn:
        lookup_codebooks = lookup_codebooks.to(torch.bfloat16)
    weight = lookup_codebooks[
        source_tiles.unsqueeze(0),
        row_tiles.unsqueeze(1),
        indices[vector_rows],
        components.unsqueeze(1),
    ].float()
    weight = weight * weight_scale.float().unsqueeze(0)
    weight += weight_bias.float().unsqueeze(0)
    return _inverse_rht(weight, rht_block_size, rht_sign).to(torch.bfloat16)


def shard_decoded_expert_weight(
    weight: torch.Tensor, name: str, tp_size: int, tp_rank: int
) -> torch.Tensor:
    """Shard a decoded matrix using vLLM routed-expert TP semantics."""
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"Invalid TP geometry: size={tp_size}, rank={tp_rank}.")
    if tp_size == 1:
        return weight
    if name.endswith(".gate_up"):
        if weight.shape[0] % (2 * tp_size):
            raise ValueError(
                f"gate_up rows {weight.shape[0]} are not divisible by "
                f"2 * tp_size ({2 * tp_size})."
            )
        gate, up = weight.chunk(2, dim=0)
        shard_size = gate.shape[0] // tp_size
        start = tp_rank * shard_size
        return torch.cat(
            (gate.narrow(0, start, shard_size), up.narrow(0, start, shard_size)),
            dim=0,
        )
    if name.endswith(".down"):
        if weight.shape[1] % tp_size:
            raise ValueError(
                f"down columns {weight.shape[1]} are not divisible by "
                f"tp_size ({tp_size})."
            )
        shard_size = weight.shape[1] // tp_size
        return weight.narrow(1, tp_rank * shard_size, shard_size)
    raise ValueError(f"Unknown VQ2 expert tensor {name!r}.")
