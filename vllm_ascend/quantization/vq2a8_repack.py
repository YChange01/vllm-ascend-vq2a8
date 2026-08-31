# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Repack canonical VQ2 artifacts into a TP-local Ascend layout."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm_ascend.quantization.vq2a8_format import (
    ASCEND_VQ2_TP_FORMAT,
    VQ2_MATRIX_KINDS,
    load_layer_metadata,
    parse_expert_name,
    required_tensor_fields,
    unpack_vq2_indices,
)


def pack_repacked_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pack ``[output_pair, input]`` uint4 codes along the input dimension."""
    if indices.ndim != 2:
        raise ValueError(f"Expected a 2D index matrix, got {indices.shape}.")
    rows, columns = indices.shape
    padded_columns = (columns + 7) // 8 * 8
    if padded_columns != columns:
        indices = torch.nn.functional.pad(indices, (0, padded_columns - columns))
    values = indices.reshape(rows, -1, 8).to(torch.int64)
    shifts = torch.arange(8, dtype=torch.int64).reshape(1, 1, 8) * 4
    return torch.sum(values << shifts, dim=-1).to(torch.int32)


def serialized_index_grid(
    packed: torch.Tensor, metadata: dict[str, Any]
) -> torch.Tensor:
    """Return canonical codes as ``[output_pair, input]``."""
    row_tiles = int(metadata["n_row_tiles"])
    columns = int(metadata["cols"])
    group_size = int(metadata["group_size"])
    vectors_per_tile = int(metadata["row_group_size"]) // int(
        metadata["vector_len"]
    )
    flat = unpack_vq2_indices(packed, int(metadata["n_vectors"]), "cpu")
    pieces = []
    position = 0
    for column_tile in range(int(metadata["n_col_tiles"])):
        start = column_tile * group_size
        width = min(start + group_size, columns) - start
        count = row_tiles * width * vectors_per_tile
        piece = flat[position : position + count].reshape(
            row_tiles, width, vectors_per_tile
        )
        pieces.append(piece)
        position += count
    if position != flat.numel():
        raise ValueError(
            f"Consumed {position} indices but artifact contains {flat.numel()}."
        )
    grid = torch.cat(pieces, dim=1)
    return grid.permute(0, 2, 1).reshape(-1, columns).contiguous()


def _gate_up_row_tiles(
    metadata: dict[str, Any], tp_size: int, tp_rank: int
) -> torch.Tensor:
    rows = int(metadata["rows"])
    row_group_size = int(metadata["row_group_size"])
    if rows % (2 * tp_size * row_group_size):
        raise ValueError("gate_up output rows do not align to TP row tiles.")
    half_tiles = rows // 2 // row_group_size
    local_tiles = half_tiles // tp_size
    gate_start = tp_rank * local_tiles
    up_start = half_tiles + gate_start
    return torch.cat(
        (
            torch.arange(gate_start, gate_start + local_tiles),
            torch.arange(up_start, up_start + local_tiles),
        )
    )


def _output_pairs(
    row_tiles: torch.Tensor, metadata: dict[str, Any]
) -> torch.Tensor:
    pairs_per_tile = int(metadata["row_group_size"]) // int(
        metadata["vector_len"]
    )
    offsets = torch.arange(pairs_per_tile)
    return (row_tiles[:, None] * pairs_per_tile + offsets).reshape(-1)


def repack_matrix(
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    kind: str,
    tp_size: int,
    tp_rank: int,
) -> dict[str, torch.Tensor]:
    """Resolve producer permutation and apply vLLM expert TP sharding."""
    if (
        int(metadata["K"]) != 16
        or int(metadata["index_bits"]) != 4
        or int(metadata["vector_len"]) != 2
        or int(metadata["norm_dim"]) != 0
    ):
        raise ValueError(f"Unsupported VQ2 geometry for {kind}: {metadata}.")
    if not all(
        metadata.get(flag, False)
        for flag in ("enable_perm", "enable_norm", "enable_rht")
    ):
        raise ValueError(
            f"{ASCEND_VQ2_TP_FORMAT} requires perm, norm, and RHT for {kind}."
        )

    columns = int(metadata["cols"])
    row_tiles = int(metadata["n_row_tiles"])
    group_size = int(metadata["group_size"])
    source_columns = torch.argsort(tensors["perm"].to(torch.int64))
    physical_columns = torch.arange(columns)
    if kind == "gate_up":
        selected_row_tiles = _gate_up_row_tiles(metadata, tp_size, tp_rank)
    elif kind == "down":
        if columns % tp_size:
            raise ValueError("down input columns are not divisible by TP size.")
        width = columns // tp_size
        start = tp_rank * width
        physical_columns = physical_columns[start : start + width]
        source_columns = source_columns[start : start + width]
        selected_row_tiles = torch.arange(row_tiles)
    else:
        raise ValueError(f"Unknown expert matrix kind {kind!r}.")

    grid = serialized_index_grid(tensors["packed_indices"], metadata)
    selected_indices = grid[_output_pairs(selected_row_tiles, metadata)][
        :, source_columns
    ]
    return {
        "packed_indices": pack_repacked_indices(selected_indices),
        "codebooks": tensors["codebooks"][:, selected_row_tiles].contiguous(),
        "codebook_tile_ids": torch.div(
            source_columns, group_size, rounding_mode="floor"
        )
        .to(torch.uint8)
        .contiguous(),
        "weight_scale": tensors["weight_scale"][physical_columns].contiguous(),
        "weight_bias": tensors["weight_bias"][physical_columns].contiguous(),
        "rht_sign": tensors["rht_sign"][physical_columns].contiguous(),
    }


def _load_matrix_tensors(handle, name: str, metadata: dict[str, Any]):
    return {
        field: handle.get_tensor(f"{name}.{field}")
        for field in required_tensor_fields(metadata)
    }


def _expert_bases(entries: dict[str, Any]) -> tuple[list[int], dict[int, str]]:
    bases: dict[int, str] = {}
    for name in entries:
        _, expert_id, kind = parse_expert_name(name)
        if kind == "gate_up":
            bases[expert_id] = name.rsplit(".", 1)[0]
    return sorted(bases), bases


def _stack_payloads(
    payloads: list[dict[str, torch.Tensor]], kind: str
) -> dict[str, torch.Tensor]:
    if not payloads:
        raise ValueError(f"Cannot stack an empty {kind} payload.")
    return {
        f"{kind}_{field}": torch.stack([payload[field] for payload in payloads])
        for field in payloads[0]
    }


def repack_layer(
    source_path: str | Path,
    output_path: str | Path,
    layer_index: int,
    tp_size: int,
) -> list[Path]:
    """Repack one canonical layer for every TP rank."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    entries = load_layer_metadata(source_path, layer_index)
    expert_ids, bases = _expert_bases(entries)
    if not expert_ids:
        raise ValueError(f"Layer {layer_index} contains no VQ2 experts.")
    tensor_path = source_path / f"experts_vq_layer_{layer_index}.safetensors"
    representative = entries[f"{bases[expert_ids[0]]}.gate_up"]
    written: list[Path] = []
    for tp_rank in range(tp_size):
        by_kind: dict[str, list[dict[str, torch.Tensor]]] = {
            kind: [] for kind in VQ2_MATRIX_KINDS
        }
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            for expert_id in expert_ids:
                base = bases[expert_id]
                for kind in VQ2_MATRIX_KINDS:
                    name = f"{base}.{kind}"
                    by_kind[kind].append(
                        repack_matrix(
                            _load_matrix_tensors(handle, name, entries[name]),
                            entries[name],
                            kind,
                            tp_size,
                            tp_rank,
                        )
                    )
        output_tensors: dict[str, torch.Tensor] = {}
        for kind, payloads in by_kind.items():
            output_tensors.update(_stack_payloads(payloads, kind))
        rank_path = output_path / f"tp{tp_size}" / f"rank{tp_rank}"
        rank_path.mkdir(parents=True, exist_ok=True)
        stem = f"experts_vq_layer_{layer_index}"
        output_tensor_path = rank_path / f"{stem}.safetensors"
        output_metadata_path = rank_path / f"{stem}.json"
        temporary_tensor_path = output_tensor_path.with_suffix(".safetensors.tmp")
        temporary_metadata_path = output_metadata_path.with_suffix(".json.tmp")
        save_file(output_tensors, temporary_tensor_path)
        layer_metadata = {
            "format": ASCEND_VQ2_TP_FORMAT,
            "layer": layer_index,
            "tp_size": tp_size,
            "tp_rank": tp_rank,
            "expert_ids": expert_ids,
            "complete": True,
            "index_bits": 4,
            "vector_len": 2,
            "effective_bits_per_weight": 2,
            "row_group_size": int(representative["row_group_size"]),
            "group_size": int(representative["group_size"]),
            "rht_block_size": int(representative["rht_block_size"]),
        }
        with temporary_metadata_path.open("w", encoding="utf-8") as output_file:
            json.dump(layer_metadata, output_file, indent=2)
            output_file.write("\n")
        os.replace(temporary_tensor_path, output_tensor_path)
        os.replace(temporary_metadata_path, output_metadata_path)
        written.extend((output_metadata_path, output_tensor_path))
    return written
