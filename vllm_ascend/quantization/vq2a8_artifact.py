# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation helpers for canonical VQ2A8 routed-expert artifacts."""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

VQ2_CODEBOOK_SIZE = 16
VQ2_INDEX_BITS = 4
VQ2_INDICES_PER_WORD = 8
VQ2_VECTOR_LENGTH = 2
VQ2_MATRIX_KINDS = ("gate_up", "down")
VQ2_CONSUMER_REFERENCE_COMMIT = "2d75468d44857582f9d21c983d451d69bea50ad7"
_MAX_SAFETENSORS_HEADER_SIZE = 100_000_000

_MATRIX_NAME_PATTERN = re.compile(
    r"^(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<kind>gate_up|down)$"
)
_SAFETENSORS_DTYPES = {
    "F8_E4M3": 1,
    "F32": 4,
    "I8": 1,
    "I32": 4,
}
_TORCH_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F32": torch.float32,
    "I8": torch.int8,
    "I32": torch.int32,
}
_MATRIX_METADATA_FIELDS = {
    "rows",
    "cols",
    "n_row_tiles",
    "n_col_tiles",
    "row_group_size",
    "group_size",
    "K",
    "index_bits",
    "vector_len",
    "n_vectors",
    "n_elements",
    "orig_shape",
    "norm_dim",
    "enable_perm",
    "enable_norm",
    "enable_rht",
}
_RHT_METADATA_FIELDS = {"rht_block_size", "rht_true_columns"}


@dataclass(frozen=True)
class VQ2TensorHeader:
    """One tensor entry from a safetensors header."""

    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


@dataclass(frozen=True)
class VQ2MatrixSpec:
    """Validated metadata for one canonical expert matrix."""

    name: str
    layer_index: int
    expert_id: int
    kind: str
    rows: int
    columns: int
    row_tiles: int
    column_tiles: int
    row_group_size: int
    group_size: int
    num_vectors: int
    num_elements: int
    original_shape: tuple[int, int]
    norm_dimension: int
    enable_permutation: bool
    enable_normalization: bool
    enable_rht: bool
    rht_block_size: int
    rht_true_columns: int

    @classmethod
    def from_dict(cls, name: str, metadata: dict[str, Any]) -> VQ2MatrixSpec:
        """Parse and validate producer metadata for one matrix."""
        layer_index, expert_id, kind = parse_matrix_name(name)
        expected_fields = set(_MATRIX_METADATA_FIELDS)
        if metadata.get("enable_rht") is True:
            expected_fields.update(_RHT_METADATA_FIELDS)
        actual_fields = set(metadata)
        if actual_fields != expected_fields:
            missing = sorted(expected_fields - actual_fields)
            extra = sorted(actual_fields - expected_fields)
            raise ValueError(f"{name}: metadata fields differ; missing={missing}, extra={extra}.")
        rows = _positive_int(metadata, "rows")
        columns = _positive_int(metadata, "cols")
        row_tiles = _positive_int(metadata, "n_row_tiles")
        column_tiles = _positive_int(metadata, "n_col_tiles")
        row_group_size = _positive_int(metadata, "row_group_size")
        group_size = _positive_int(metadata, "group_size")
        num_vectors = _positive_int(metadata, "n_vectors")
        num_elements = _positive_int(metadata, "n_elements")
        norm_dimension = _nonnegative_int(metadata, "norm_dim")
        enable_permutation = _bool_value(metadata, "enable_perm")
        enable_normalization = _bool_value(metadata, "enable_norm")
        enable_rht = _bool_value(metadata, "enable_rht")

        codebook_size = _positive_int(metadata, "K")
        index_bits = _positive_int(metadata, "index_bits")
        vector_length = _positive_int(metadata, "vector_len")
        if (codebook_size, index_bits, vector_length) != (
            VQ2_CODEBOOK_SIZE,
            VQ2_INDEX_BITS,
            VQ2_VECTOR_LENGTH,
        ):
            raise ValueError(
                f"{name}: expected K=16, index_bits=4, vector_len=2; got "
                f"K={codebook_size}, index_bits={index_bits}, "
                f"vector_len={vector_length}."
            )
        if norm_dimension != 0:
            raise ValueError(f"{name}: norm_dim must be 0, got {norm_dimension}.")
        if rows != row_tiles * row_group_size:
            raise ValueError(
                f"{name}: rows={rows} does not equal n_row_tiles * row_group_size ({row_tiles * row_group_size})."
            )
        if column_tiles != math.ceil(columns / group_size):
            raise ValueError(
                f"{name}: n_col_tiles={column_tiles} does not equal "
                f"ceil(cols / group_size) ({math.ceil(columns / group_size)})."
            )
        if row_group_size % vector_length:
            raise ValueError(f"{name}: row_group_size={row_group_size} is not divisible by vector_len={vector_length}.")
        if num_elements != rows * columns:
            raise ValueError(f"{name}: n_elements={num_elements} does not equal rows * cols ({rows * columns}).")
        if num_elements % vector_length or num_vectors != num_elements // vector_length:
            raise ValueError(
                f"{name}: n_vectors={num_vectors} does not equal "
                f"n_elements / vector_len ({num_elements // vector_length})."
            )

        original_shape = _shape_pair(metadata, "orig_shape")
        if enable_rht:
            rht_block_size = _positive_int(metadata, "rht_block_size")
            rht_true_columns = _positive_int(metadata, "rht_true_columns")
            if rht_block_size & (rht_block_size - 1):
                raise ValueError(f"{name}: rht_block_size must be a power of two, got {rht_block_size}.")
            if columns % rht_block_size:
                raise ValueError(f"{name}: cols={columns} is not divisible by rht_block_size={rht_block_size}.")
            if rht_true_columns > columns:
                raise ValueError(f"{name}: rht_true_columns={rht_true_columns} exceeds cols={columns}.")
        else:
            rht_block_size = 1
            rht_true_columns = columns
        expected_original_shape = (rows, rht_true_columns)
        if original_shape != expected_original_shape:
            raise ValueError(
                f"{name}: orig_shape={list(original_shape)} does not equal {list(expected_original_shape)}."
            )

        return cls(
            name=name,
            layer_index=layer_index,
            expert_id=expert_id,
            kind=kind,
            rows=rows,
            columns=columns,
            row_tiles=row_tiles,
            column_tiles=column_tiles,
            row_group_size=row_group_size,
            group_size=group_size,
            num_vectors=num_vectors,
            num_elements=num_elements,
            original_shape=original_shape,
            norm_dimension=norm_dimension,
            enable_permutation=enable_permutation,
            enable_normalization=enable_normalization,
            enable_rht=enable_rht,
            rht_block_size=rht_block_size,
            rht_true_columns=rht_true_columns,
        )

    @property
    def vectors_per_row_group(self) -> int:
        return self.row_group_size // VQ2_VECTOR_LENGTH

    @property
    def packed_word_count(self) -> int:
        return math.ceil(self.num_vectors / VQ2_INDICES_PER_WORD)

    def expected_tensor_headers(self) -> dict[str, tuple[str, tuple[int, ...]]]:
        """Return the exact tensor schema for this matrix."""
        expected = {
            "packed_indices": ("I32", (self.packed_word_count,)),
            "codebooks": (
                "F8_E4M3",
                (
                    self.column_tiles,
                    self.row_tiles,
                    VQ2_CODEBOOK_SIZE,
                    VQ2_VECTOR_LENGTH,
                ),
            ),
        }
        if self.enable_permutation:
            expected["perm"] = ("I32", (self.columns,))
        if self.enable_normalization:
            expected["weight_scale"] = ("F32", (self.columns,))
            expected["weight_bias"] = ("F32", (self.columns,))
        if self.enable_rht:
            expected["rht_sign"] = ("I8", (self.columns,))
        return expected


@dataclass(frozen=True)
class VQ2LayerSummary:
    """Header-only validation result for one canonical layer."""

    layer_index: int
    expert_ids: tuple[int, ...]
    matrix_count: int
    tensor_count: int
    tensor_file_bytes: int
    gate_up_shape: tuple[int, int] | None = None
    down_shape: tuple[int, int] | None = None


@dataclass(frozen=True)
class VQ2PayloadSummary:
    """Value-level validation result for one matrix payload."""

    name: str
    codebook_min: float
    codebook_max: float
    scale_min: float | None
    scale_max: float | None


@dataclass(frozen=True)
class VQ2ModelLayout:
    """Expected expert layout read from a model ``config.json``."""

    num_hidden_layers: int
    num_hash_layers: int
    num_routed_experts: int
    hidden_size: int
    moe_intermediate_size: int

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> VQ2ModelLayout:
        quantization_config = config.get("quantization_config")
        if not isinstance(quantization_config, dict):
            raise ValueError("config.json has no quantization_config object.")
        if quantization_config.get("quant_method") != "vq2a8":
            raise ValueError("config.json quantization_config.quant_method must be 'vq2a8'.")
        num_hidden_layers = _positive_int(config, "num_hidden_layers")
        num_hash_layers = _nonnegative_int(config, "num_hash_layers")
        num_routed_experts = _positive_int(config, "n_routed_experts")
        hidden_size = _positive_int(config, "hidden_size")
        moe_intermediate_size = _positive_int(config, "moe_intermediate_size")
        if num_hash_layers > num_hidden_layers:
            raise ValueError(f"num_hash_layers={num_hash_layers} exceeds num_hidden_layers={num_hidden_layers}.")
        return cls(
            num_hidden_layers,
            num_hash_layers,
            num_routed_experts,
            hidden_size,
            moe_intermediate_size,
        )

    def expected_expert_ids(self, layer_index: int) -> tuple[int, ...]:
        if not 0 <= layer_index < self.num_hidden_layers:
            raise ValueError(f"Layer {layer_index} is outside [0, {self.num_hidden_layers}).")
        count = 1 if layer_index < self.num_hash_layers else self.num_routed_experts
        return tuple(range(count))


def parse_matrix_name(name: str) -> tuple[int, int, str]:
    """Return ``(layer, expert, kind)`` from a canonical matrix name."""
    match = _MATRIX_NAME_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Invalid canonical VQ2A8 matrix name: {name!r}.")
    layer_index = int(match.group("layer"))
    expert_id = int(match.group("expert"))
    kind = match.group("kind")
    canonical_name = f"{layer_index}.mlp.experts.{expert_id}.{kind}"
    if name != canonical_name:
        raise ValueError(f"Non-canonical VQ2A8 matrix name {name!r}; expected {canonical_name!r}.")
    return layer_index, expert_id, kind


def layer_artifact_paths(experts_path: str | Path, layer_index: int) -> tuple[Path, Path]:
    root = Path(experts_path)
    stem = f"experts_vq_layer_{layer_index}"
    return root / f"{stem}.json", root / f"{stem}.safetensors"


def load_layer_specs(experts_path: str | Path, layer_index: int) -> dict[str, VQ2MatrixSpec]:
    """Load and validate all matrix metadata for one layer."""
    metadata_path, tensor_path = layer_artifact_paths(experts_path, layer_index)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing VQ2A8 metadata: {metadata_path}.")
    if not tensor_path.is_file():
        raise FileNotFoundError(f"Missing VQ2A8 tensors: {tensor_path}.")
    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file, object_pairs_hook=_object_without_duplicates)
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError(f"VQ2A8 metadata must be a non-empty object: {metadata_path}.")

    specs: dict[str, VQ2MatrixSpec] = {}
    for name, matrix_metadata in metadata.items():
        if not isinstance(matrix_metadata, dict):
            raise ValueError(f"VQ2A8 matrix metadata must be an object: {name!r}.")
        spec = VQ2MatrixSpec.from_dict(name, matrix_metadata)
        if spec.layer_index != layer_index:
            raise ValueError(f"Matrix {name!r} belongs to layer {spec.layer_index}, expected layer {layer_index}.")
        specs[name] = spec
    return specs


def read_safetensors_header(tensor_path: str | Path) -> dict[str, VQ2TensorHeader]:
    """Read and structurally validate a safetensors header without payload I/O."""
    path = Path(tensor_path)
    file_size = path.stat().st_size
    with path.open("rb") as tensor_file:
        length_bytes = tensor_file.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Truncated safetensors length prefix: {path}.")
        header_size = struct.unpack("<Q", length_bytes)[0]
        if header_size > _MAX_SAFETENSORS_HEADER_SIZE:
            raise ValueError(
                f"Safetensors header size {header_size} exceeds the "
                f"{_MAX_SAFETENSORS_HEADER_SIZE}-byte safety limit: {path}."
            )
        if header_size > file_size - 8:
            raise ValueError(f"Safetensors header size {header_size} exceeds file size {file_size}: {path}.")
        header_bytes = tensor_file.read(header_size)
    try:
        raw_header = json.loads(header_bytes, object_pairs_hook=_object_without_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid safetensors JSON header: {path}.") from error
    if not isinstance(raw_header, dict):
        raise ValueError(f"Safetensors header must be an object: {path}.")

    data_size = file_size - 8 - header_size
    headers: dict[str, VQ2TensorHeader] = {}
    occupied_ranges: list[tuple[int, int, str]] = []
    for name, entry in raw_header.items():
        if name == "__metadata__":
            if not isinstance(entry, dict) or any(
                not isinstance(key, str) or not isinstance(value, str) for key, value in entry.items()
            ):
                raise ValueError(f"Invalid safetensors __metadata__: {path}.")
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid safetensors entry for {name!r}: {path}.")
        if set(entry) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"Safetensors entry {name!r} must contain exactly dtype, shape and data_offsets: {path}.")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if dtype not in _SAFETENSORS_DTYPES:
            raise ValueError(f"Unsupported safetensors dtype {dtype!r} for {name!r}.")
        if not isinstance(shape, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in shape
        ):
            raise ValueError(f"Invalid safetensors shape for {name!r}: {shape!r}.")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
        ):
            raise ValueError(f"Invalid safetensors offsets for {name!r}: {offsets!r}.")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise ValueError(f"Out-of-range safetensors offsets for {name!r}: {offsets!r}, data size={data_size}.")
        expected_bytes = math.prod(shape) * _SAFETENSORS_DTYPES[dtype]
        if end - start != expected_bytes:
            raise ValueError(
                f"Tensor {name!r} stores {end - start} bytes, expected "
                f"{expected_bytes} for dtype={dtype}, shape={shape}."
            )
        headers[name] = VQ2TensorHeader(dtype, tuple(shape), (start, end))
        occupied_ranges.append((start, end, name))

    previous_end = 0
    previous_name = "<data-start>"
    for start, end, name in sorted(occupied_ranges):
        if start < previous_end:
            raise ValueError(f"Safetensors data for {name!r} overlaps {previous_name!r}: {path}.")
        if start > previous_end:
            raise ValueError(
                f"Safetensors data has a gap before {name!r}: expected offset {previous_end}, got {start}: {path}."
            )
        previous_end = end
        previous_name = name
    if previous_end != data_size:
        raise ValueError(f"Safetensors data has {data_size - previous_end} trailing bytes: {path}.")
    return headers


def inspect_layer_artifact(experts_path: str | Path, layer_index: int) -> VQ2LayerSummary:
    """Validate metadata and tensor headers without reading tensor payloads."""
    specs = load_layer_specs(experts_path, layer_index)
    _, tensor_path = layer_artifact_paths(experts_path, layer_index)
    headers = read_safetensors_header(tensor_path)

    expert_kinds: dict[int, set[str]] = {}
    geometry_by_kind: dict[str, tuple[object, ...]] = {}
    shape_by_kind: dict[str, tuple[int, int]] = {}
    expected_headers: dict[str, tuple[str, tuple[int, ...]]] = {}
    for spec in specs.values():
        expert_kinds.setdefault(spec.expert_id, set()).add(spec.kind)
        geometry = (
            spec.rows,
            spec.columns,
            spec.row_tiles,
            spec.column_tiles,
            spec.row_group_size,
            spec.group_size,
            spec.num_vectors,
            spec.num_elements,
            spec.original_shape,
            spec.norm_dimension,
            spec.enable_permutation,
            spec.enable_normalization,
            spec.enable_rht,
            spec.rht_block_size,
            spec.rht_true_columns,
        )
        previous_geometry = geometry_by_kind.setdefault(spec.kind, geometry)
        if geometry != previous_geometry:
            raise ValueError(f"Layer {layer_index} {spec.kind} metadata is inconsistent across experts.")
        shape_by_kind.setdefault(spec.kind, spec.original_shape)
        expected_headers.update(
            {f"{spec.name}.{field}": tensor_header for field, tensor_header in spec.expected_tensor_headers().items()}
        )
    expected_kinds = set(VQ2_MATRIX_KINDS)
    for expert_id, kinds in expert_kinds.items():
        if kinds != expected_kinds:
            raise ValueError(
                f"Layer {layer_index} expert {expert_id} has matrices "
                f"{sorted(kinds)}, expected {sorted(expected_kinds)}."
            )
    gate_up_shape = shape_by_kind["gate_up"]
    down_shape = shape_by_kind["down"]
    if gate_up_shape[1] != down_shape[0] or gate_up_shape[0] != 2 * down_shape[1]:
        raise ValueError(
            f"Layer {layer_index} has incompatible MoE matrix shapes: "
            f"gate_up={list(gate_up_shape)}, down={list(down_shape)}; expected "
            "gate_up=[2 * intermediate_size, hidden_size] and "
            "down=[hidden_size, intermediate_size]."
        )
    expert_ids = tuple(sorted(expert_kinds))
    if expert_ids != tuple(range(len(expert_ids))):
        raise ValueError(f"Layer {layer_index} expert IDs must be contiguous from zero, got {list(expert_ids)}.")

    missing = sorted(set(expected_headers) - set(headers))
    extra = sorted(set(headers) - set(expected_headers))
    if missing:
        raise ValueError(
            f"Layer {layer_index} is missing {len(missing)} VQ2A8 tensors; first missing key: {missing[0]}."
        )
    if extra:
        raise ValueError(f"Layer {layer_index} has {len(extra)} unexpected VQ2A8 tensors; first extra key: {extra[0]}.")
    for name, (expected_dtype, expected_shape) in expected_headers.items():
        actual = headers[name]
        if (actual.dtype, actual.shape) != (expected_dtype, expected_shape):
            raise ValueError(
                f"Tensor {name!r} has dtype={actual.dtype}, "
                f"shape={list(actual.shape)}; expected dtype={expected_dtype}, "
                f"shape={list(expected_shape)}."
            )

    return VQ2LayerSummary(
        layer_index=layer_index,
        expert_ids=expert_ids,
        matrix_count=len(specs),
        tensor_count=len(headers),
        tensor_file_bytes=tensor_path.stat().st_size,
        gate_up_shape=shape_by_kind.get("gate_up"),
        down_shape=shape_by_kind.get("down"),
    )


def inspect_vq2_directory(experts_path: str | Path) -> list[VQ2LayerSummary]:
    """Validate a contiguous directory of canonical layer artifacts."""
    root = Path(experts_path)
    if not root.is_dir():
        raise FileNotFoundError(f"VQ2A8 expert directory not found: {root}.")
    metadata_layers = {
        int(match.group(1))
        for path in root.glob("experts_vq_layer_*.json")
        if (match := re.fullmatch(r"experts_vq_layer_(\d+)\.json", path.name))
    }
    tensor_layers = {
        int(match.group(1))
        for path in root.glob("experts_vq_layer_*.safetensors")
        if (match := re.fullmatch(r"experts_vq_layer_(\d+)\.safetensors", path.name))
    }
    if not metadata_layers and not tensor_layers:
        raise ValueError(f"No canonical VQ2A8 layer artifacts found in {root}.")
    if metadata_layers != tensor_layers:
        missing_metadata = sorted(tensor_layers - metadata_layers)
        missing_tensors = sorted(metadata_layers - tensor_layers)
        raise ValueError(
            f"VQ2A8 layer files are not paired; missing_metadata={missing_metadata}, missing_tensors={missing_tensors}."
        )
    layer_indices = sorted(metadata_layers)
    expected = list(range(layer_indices[-1] + 1))
    if layer_indices != expected:
        missing = sorted(set(expected) - set(layer_indices))
        raise ValueError(f"VQ2A8 layer sequence has gaps; missing layers: {missing}.")
    return [inspect_layer_artifact(root, layer_index) for layer_index in layer_indices]


def load_model_layout(config_path: str | Path) -> VQ2ModelLayout:
    """Load the expected VQ2A8 layer/expert layout from ``config.json``."""
    path = Path(config_path)
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file, object_pairs_hook=_object_without_duplicates)
    if not isinstance(config, dict):
        raise ValueError(f"Model config must be an object: {path}.")
    return VQ2ModelLayout.from_dict(config)


def validate_model_layout(
    summaries: list[VQ2LayerSummary],
    layout: VQ2ModelLayout,
    *,
    require_all_layers: bool,
) -> None:
    """Match validated layer summaries against model-level expectations."""
    actual_layers = tuple(summary.layer_index for summary in summaries)
    if len(set(actual_layers)) != len(actual_layers):
        raise ValueError(f"Duplicate layer summaries: {list(actual_layers)}.")
    if require_all_layers:
        expected_layers = tuple(range(layout.num_hidden_layers))
        if tuple(sorted(actual_layers)) != expected_layers:
            raise ValueError(
                f"Artifact layers {list(sorted(actual_layers))} do not match config layers {list(expected_layers)}."
            )
    for summary in summaries:
        expected_experts = layout.expected_expert_ids(summary.layer_index)
        if summary.expert_ids != expected_experts:
            raise ValueError(
                f"Layer {summary.layer_index} has expert IDs "
                f"{list(summary.expert_ids)}, expected {list(expected_experts)}."
            )
        expected_gate_up = (2 * layout.moe_intermediate_size, layout.hidden_size)
        expected_down = (layout.hidden_size, layout.moe_intermediate_size)
        if summary.gate_up_shape != expected_gate_up:
            raise ValueError(
                f"Layer {summary.layer_index} gate_up shape is {summary.gate_up_shape}, expected {expected_gate_up}."
            )
        if summary.down_shape != expected_down:
            raise ValueError(
                f"Layer {summary.layer_index} down shape is {summary.down_shape}, expected {expected_down}."
            )


def load_matrix_tensors(
    experts_path: str | Path,
    layer_index: int,
    expert_id: int,
    kind: str,
) -> tuple[dict[str, torch.Tensor], VQ2MatrixSpec]:
    """Load one matrix payload from a canonical layer artifact."""
    if kind not in VQ2_MATRIX_KINDS:
        raise ValueError(f"Unknown VQ2A8 matrix kind: {kind!r}.")
    specs = load_layer_specs(experts_path, layer_index)
    name = f"{layer_index}.mlp.experts.{expert_id}.{kind}"
    if name not in specs:
        raise KeyError(f"VQ2A8 matrix not found: {name}.")
    spec = specs[name]
    _, tensor_path = layer_artifact_paths(experts_path, layer_index)
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        tensors = {field: handle.get_tensor(f"{name}.{field}") for field in spec.expected_tensor_headers()}
    return tensors, spec


def validate_matrix_payload(tensors: dict[str, torch.Tensor], spec: VQ2MatrixSpec) -> VQ2PayloadSummary:
    """Validate tensor values that cannot be checked from headers alone."""
    expected_fields = set(spec.expected_tensor_headers())
    actual_fields = set(tensors)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        extra = sorted(actual_fields - expected_fields)
        raise ValueError(f"{spec.name}: payload fields differ; missing={missing}, extra={extra}.")
    for field, (safetensors_dtype, expected_shape) in spec.expected_tensor_headers().items():
        tensor = tensors[field]
        expected_dtype = _TORCH_DTYPES[safetensors_dtype]
        if tensor.dtype != expected_dtype or tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{spec.name}.{field}: dtype={tensor.dtype}, shape={list(tensor.shape)}; "
                f"expected dtype={expected_dtype}, shape={list(expected_shape)}."
            )

    codebooks = tensors["codebooks"].float()
    if not bool(torch.isfinite(codebooks).all()):
        raise ValueError(f"{spec.name}: codebooks contain non-finite values.")

    if spec.enable_permutation:
        permutation = tensors["perm"].to(torch.int64)
        expected = torch.arange(spec.columns, dtype=torch.int64)
        if not torch.equal(torch.sort(permutation).values, expected):
            raise ValueError(f"{spec.name}: perm is not a bijection over [0, cols).")

    scale_min: float | None = None
    scale_max: float | None = None
    if spec.enable_normalization:
        scale = tensors["weight_scale"].float()
        bias = tensors["weight_bias"].float()
        if not bool(torch.isfinite(scale).all()) or not bool(torch.isfinite(bias).all()):
            raise ValueError(f"{spec.name}: normalization contains non-finite values.")
        scale_min = float(scale.min())
        scale_max = float(scale.max())

    if spec.enable_rht:
        signs = tensors["rht_sign"].to(torch.int16)
        if not bool(((signs == -1) | (signs == 1)).all()):
            raise ValueError(f"{spec.name}: rht_sign contains values other than -1 and 1.")

    return VQ2PayloadSummary(
        name=spec.name,
        codebook_min=float(codebooks.min()),
        codebook_max=float(codebooks.max()),
        scale_min=scale_min,
        scale_max=scale_max,
    )


def _positive_int(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}.")
    return value


def _nonnegative_int(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer, got {value!r}.")
    return value


def _bool_value(metadata: dict[str, Any], key: str) -> bool:
    value = metadata.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean, got {value!r}.")
    return value


def _shape_pair(metadata: dict[str, Any], key: str) -> tuple[int, int]:
    value = metadata.get(key)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0 for dimension in value)
    ):
        raise ValueError(f"{key} must contain two positive integers, got {value!r}.")
    return value[0], value[1]


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result
