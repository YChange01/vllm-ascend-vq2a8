# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only, rank-bound reader for the frozen TP2 packed-zN artifact.

Loading this format does not upgrade the exporter's historical runtime/device
validation claims. Metadata/headers cover both ranks; optional tensor SHA-256
checks cover this reader's rank. Value checks precede every device transfer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch
from safetensors import safe_open

from .vq2a8_artifact import VQ2ModelLayout, load_model_layout

VQ2_TP2_ZN_FORMAT = "vq2a8_zn_tp2_v1"
VQ2_TP2_FIELDS = ("packed_zn", "pair_lut", "activation_order", "weight_scale", "weight_bias", "rht_sign")
_DTYPES = {
    "packed_zn": "U8",
    "pair_lut": "U8",
    "activation_order": "I64",
    "weight_scale": "F32",
    "weight_bias": "F32",
    "rht_sign": "I8",
}
_TORCH_DTYPES = {"U8": torch.uint8, "I64": torch.int64, "F32": torch.float32, "I8": torch.int8}
_DTYPE_BYTES = {"U8": 1, "I64": 8, "F32": 4, "I8": 1}
_KINDS = ("gate_up", "down")
_MAX_HEADER_BYTES = 64 * 2**20
_PREPARATION_ORDER = [
    "physical_rht128",
    "physical_bias_gemv_and_weight_scale",
    "rank_local_dynamic_fp8",
    "byte_gather_activation_order",
]
_COMMUNICATION = {
    "gate_up": "column_parallel_separate_gate_and_up_slices_concatenated_per_rank",
    "down": "row_parallel_contiguous_physical_input_slice_sum_partials_across_tp_ranks",
    "activation_quantization": "per_rank_per_row_amax_after_local_RHT128_and_weight_scale",
    "bias_correction": "local_input_contribution_only_before_down_partial_sum",
    "routing": "same_token_expert_assignments_on_both_ranks_not_expert_parallel",
}


def _same(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_same(actual[key], value) for key, value in expected.items())
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(_same(left, right) for left, right in zip(actual, expected))
    return actual == expected


def _equal(actual: Any, expected: Any, label: str) -> None:
    # JSON bool and int must not compare interchangeably in an artifact contract.
    if not _same(actual, expected):
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}.")


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return value


def _integers(value: Any, label: str, *, minimum: int = 0) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    return tuple(_integer(item, label, minimum=minimum) for item in value)


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}.")
        result[key] = value
    return result


def _json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _no_links(path: Path) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink() or getattr(component, "is_junction", lambda: False)():
            raise ValueError(f"Artifact/config paths must not traverse symlinks or junctions: {component}.")
    return absolute.resolve(strict=True)


def _file(root: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise ValueError(f"Invalid artifact-relative path: {name!r}.")
    relative = PurePosixPath(name)
    if relative.is_absolute() or PureWindowsPath(name).drive or relative.as_posix() != name or ".." in relative.parts:
        raise ValueError(f"Artifact path must be canonical and cannot escape its root: {name!r}.")
    path = _no_links(root.joinpath(*relative.parts))
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Artifact path must identify a regular file inside its root: {name!r}.")
    return path


def _identity(path: Path) -> tuple[int, int, int, int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _header(path: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Read I64-capable headers without safe_open or touching tensor payloads."""
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors header: {path}.")
        length = struct.unpack("<Q", prefix)[0]
        if length < 2 or length > _MAX_HEADER_BYTES or length + 8 > path.stat().st_size:
            raise ValueError(f"Invalid safetensors header length: {path}.")
        value = json.loads(stream.read(length), object_pairs_hook=_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid safetensors header object: {path}.")
    result, spans = {}, []
    for key, entry in value.items():
        if key == "__metadata__":
            if not isinstance(entry, dict) or any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in entry.items()
            ):
                raise ValueError(f"Invalid safetensors string metadata: {path}.")
            continue
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"Invalid safetensors tensor header: {key}.")
        dtype = entry["dtype"]
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
            raise ValueError(f"Unsupported TP2 tensor dtype: {dtype!r}.")
        shape = _integers(entry["shape"], f"{key}.shape", minimum=1)
        offsets = _integers(entry["data_offsets"], f"{key}.data_offsets")
        if not shape or len(offsets) != 2 or offsets[1] - offsets[0] != math.prod(shape) * _DTYPE_BYTES[dtype]:
            raise ValueError(f"Invalid safetensors byte span: {key}.")
        result[key] = (dtype, shape)
        spans.append(offsets)
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError(f"Overlapping/gapped safetensors payload: {path}.")
        cursor = end
    if 8 + length + cursor != path.stat().st_size:
        raise ValueError(f"Safetensors payload/file size mismatch: {path}.")
    return result


@dataclass(frozen=True)
class VQ2TP2MatrixSpec:
    name: str
    layer_index: int
    expert_id: int
    kind: str
    tp_rank: int
    canonical_shape: tuple[int, int]
    logical_shape: tuple[int, int]
    packed_shape: tuple[int, int]
    lut_source_tile_ids: tuple[int, ...]
    tile_valid_counts: tuple[int, ...]
    metadata: dict[str, Any]

    @property
    def rows(self) -> int:
        return self.logical_shape[0]

    @property
    def columns(self) -> int:
        return self.packed_shape[1]

    @property
    def rht_true_columns(self) -> int:
        return self.logical_shape[1]

    @property
    def padding_columns(self) -> int:
        return self.columns - self.rht_true_columns

    @property
    def output_row_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple(tuple(pair) for pair in self.metadata["output_row_ranges"])

    @property
    def input_column_range(self) -> tuple[int, int]:
        return tuple(self.metadata["input_column_range"])

    @property
    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        n, k = self.packed_shape
        return {
            "packed_zn": (n // 32, k // 16, 16, 8),
            "pair_lut": (k // 256, n // 32, 32),
            **{field: (k,) for field in VQ2_TP2_FIELDS[2:]},
        }

    rht_block_size = 128
    row_group_size = 32
    group_size = 256


@dataclass(frozen=True)
class VQ2TP2Shard:
    layer_index: int
    tp_rank: int
    expert_ids: tuple[int, ...]
    tensor_path: Path
    metadata_path: Path
    tensor_sha256: str
    metadata_sha256: str
    payload_bytes: int
    specs: dict[tuple[int, str], VQ2TP2MatrixSpec]
    tensor_identity: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class VQ2TP2Layer:
    layer_index: int
    tp_rank: int
    expert_ids: tuple[int, ...]
    shards: tuple[VQ2TP2Shard, ...]
    matrix_specs: dict[str, dict[int, VQ2TP2MatrixSpec]]

    def spec_for(self, expert_id: int, kind: str) -> VQ2TP2MatrixSpec:
        _integer(expert_id, "expert_id")
        if kind not in _KINDS:
            raise ValueError(f"Unknown TP2 matrix kind: {kind!r}.")
        try:
            return self.matrix_specs[kind][expert_id]
        except KeyError as error:
            raise KeyError(f"TP2 layer {self.layer_index} has no expert {expert_id}.") from error


def validate_tp2_runtime_payload(tensors: dict[str, torch.Tensor], spec: VQ2TP2MatrixSpec) -> None:
    """Check local mappings/dummy columns on CPU; cannot prove source equivalence."""
    _equal(set(tensors), set(VQ2_TP2_FIELDS), f"{spec.name} fields")
    for field, shape in spec.tensor_shapes.items():
        value = tensors[field]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or not value.is_contiguous():
            raise ValueError(f"{spec.name}.{field} must be a contiguous CPU tensor.")
        if value.dtype != _TORCH_DTYPES[_DTYPES[field]] or tuple(value.shape) != shape:
            raise ValueError(f"{spec.name}.{field} dtype/shape mismatch.")
    order = tensors["activation_order"]
    k, logical_k = spec.columns, spec.rht_true_columns
    if not torch.equal(torch.sort(order).values, torch.arange(k, dtype=torch.int64, device="cpu")):
        raise ValueError(f"{spec.name}: activation_order is not a bijection over packed K.")
    packed = tensors["packed_zn"].reshape(spec.rows // 32, k, 8)
    next_dummy = logical_k
    for block, count in enumerate(spec.tile_valid_counts):
        mapping = order[block * 256 : (block + 1) * 256]
        real = mapping[:count]
        if bool((real >= logical_k).any()) or (count > 1 and not bool((real[1:] > real[:-1]).all())):
            raise ValueError(f"{spec.name}: activation_order disagrees with tile_valid_counts/stable real mapping.")
        padding = 256 - count
        if not torch.equal(
            mapping[count:], torch.arange(next_dummy, next_dummy + padding, dtype=torch.int64, device="cpu")
        ):
            raise ValueError(f"{spec.name}: dummy activation mapping disagrees with tile_valid_counts.")
        if padding and bool(packed[:, block * 256 + count : (block + 1) * 256, :].any()):
            raise ValueError(f"{spec.name}: dummy packed codes must be zero.")
        next_dummy += padding
    for field in ("weight_scale", "weight_bias"):
        if not bool(torch.isfinite(tensors[field]).all()) or bool(tensors[field][logical_k:].any()):
            raise ValueError(f"{spec.name}.{field} must be finite with zero dummy metadata.")
    signs = tensors["rht_sign"]
    if not bool(((signs == -1) | (signs == 1)).all()) or not bool((signs[logical_k:] == 1).all()):
        raise ValueError(f"{spec.name}: invalid RHT signs or dummy signs.")
    if bool(((tensors["pair_lut"] & 127) == 127).any()):
        raise ValueError(f"{spec.name}: pair LUT contains non-finite E4M3FN bytes.")


@dataclass(frozen=True)
class VQ2TP2Artifact:
    root: Path
    model_config_path: Path
    model_layout: VQ2ModelLayout
    tp_rank: int
    layers: dict[int, VQ2TP2Layer]
    manifest: dict[str, Any]
    tensor_hashes_verified: bool

    def layer(self, layer_index: int) -> VQ2TP2Layer:
        _integer(layer_index, "layer_index")
        try:
            return self.layers[layer_index]
        except KeyError as error:
            raise KeyError(f"TP2 artifact has no layer {layer_index}.") from error

    def _read_shard(
        self, shard: VQ2TP2Shard, *, device: torch.device | str, non_blocking: bool, only: tuple[int, str] | None = None
    ) -> dict[int, dict[str, tuple[dict[str, torch.Tensor], VQ2TP2MatrixSpec]]]:
        path = _file(self.root, shard.tensor_path.relative_to(self.root).as_posix())
        if _identity(path) != shard.tensor_identity:
            raise ValueError(f"TP2 tensor file changed after artifact validation: {path}.")
        result: dict[int, dict[str, tuple[dict[str, torch.Tensor], VQ2TP2MatrixSpec]]] = {}
        with safe_open(path, framework="pt", device="cpu") as handle:
            for (expert, kind), spec in shard.specs.items():
                if only is not None and (expert, kind) != only:
                    continue
                tensors = {
                    field: handle.get_tensor(f"{expert}.{kind}.{field}").contiguous() for field in VQ2_TP2_FIELDS
                }
                validate_tp2_runtime_payload(tensors, spec)
                result.setdefault(expert, {})[kind] = (tensors, spec)
        if _identity(path) != shard.tensor_identity:
            raise ValueError(f"TP2 tensor file changed while reading: {path}.")
        target = torch.device(device)
        if target.type != "cpu":
            for matrices in result.values():
                for kind, (tensors, spec) in matrices.items():
                    matrices[kind] = (
                        {
                            field: tensor.to(device=target, non_blocking=non_blocking)
                            for field, tensor in tensors.items()
                        },
                        spec,
                    )
        return result

    def iter_rank_shards(
        self,
        layer_index: int,
        rank: int | None = None,
        *,
        device: torch.device | str = "cpu",
        non_blocking: bool = False,
    ) -> Iterator[dict[int, dict[str, tuple[dict[str, torch.Tensor], VQ2TP2MatrixSpec]]]]:
        """Yield this rank's expert dictionaries using one safe_open per shard."""
        if rank is not None:
            _equal(rank, self.tp_rank, "Requested TP2 rank")
        for shard in self.layer(layer_index).shards:
            yield self._read_shard(shard, device=device, non_blocking=non_blocking)

    def load_expert(
        self,
        layer_index: int,
        expert_id: int,
        kind: str,
        *,
        device: torch.device | str = "cpu",
        non_blocking: bool = False,
    ) -> tuple[dict[str, torch.Tensor], VQ2TP2MatrixSpec]:
        """Read one matrix; resident initialization should prefer iter_rank_shards."""
        layer = self.layer(layer_index)
        layer.spec_for(expert_id, kind)
        shard = next(item for item in layer.shards if expert_id in item.expert_ids)
        return self._read_shard(shard, device=device, non_blocking=non_blocking, only=(expert_id, kind))[expert_id][
            kind
        ]


def _matrix(entry: Any, layout: VQ2ModelLayout, layer: int, rank: int, expert_ids: tuple[int, ...]) -> VQ2TP2MatrixSpec:
    if not isinstance(entry, dict):
        raise ValueError("TP2 matrix metadata must be an object.")
    expert = _integer(entry.get("expert_id"), "matrix.expert_id")
    kind = entry.get("kind")
    if expert not in expert_ids or kind not in _KINDS:
        raise ValueError("TP2 matrix has invalid expert/kind identity.")
    name = f"{layer}.mlp.experts.{expert}.{kind}"
    canonical = (
        (2 * layout.moe_intermediate_size, layout.hidden_size)
        if kind == "gate_up"
        else (layout.hidden_size, layout.moe_intermediate_size)
    )
    n, source_k = canonical
    logical = (n // 2, source_k) if kind == "gate_up" else (n, source_k // 2)
    width = n // 4
    output_ranges = (
        [[rank * width, (rank + 1) * width], [n // 2 + rank * width, n // 2 + (rank + 1) * width]]
        if kind == "gate_up"
        else [[0, n]]
    )
    input_range = [0, source_k] if kind == "gate_up" else [rank * logical[1], (rank + 1) * logical[1]]
    for key, expected in {
        "name": name,
        "format": VQ2_TP2_ZN_FORMAT,
        "tp_size": 2,
        "tp_rank": rank,
        "canonical_shape": list(canonical),
        "logical_shape": list(logical),
        "output_row_ranges": output_ranges,
        "input_column_range": input_range,
        "range_convention": "zero_based_half_open",
        "rht_block_size": 128,
        "rht_true_columns": logical[1],
        "runtime_supported": False,
    }.items():
        _equal(entry.get(key), expected, f"{name}.{key}")
    packed_shape = _integers(entry.get("packed_shape"), f"{name}.packed_shape", minimum=1)
    tiles = _integers(entry.get("lut_source_tile_ids"), f"{name}.lut_source_tile_ids")
    counts = _integers(entry.get("tile_valid_counts"), f"{name}.tile_valid_counts", minimum=1)
    if not tiles or tiles != tuple(sorted(set(tiles))) or tiles[-1] >= source_k // 256:
        raise ValueError(f"{name}: invalid source LUT tile mapping.")
    if len(counts) != len(tiles) or max(counts) > 256 or sum(counts) != logical[1]:
        raise ValueError(f"{name}: invalid tile_valid_counts.")
    if packed_shape != (logical[0], len(tiles) * 256) or packed_shape[1] < logical[1] or packed_shape[1] > source_k:
        raise ValueError(f"{name}: invalid packed_shape.")
    if kind == "gate_up" and (tiles != tuple(range(source_k // 256)) or any(count != 256 for count in counts)):
        raise ValueError(f"{name}: gate/up must retain every source K256 block.")
    _equal(entry.get("padding_columns"), packed_shape[1] - logical[1], f"{name}.padding_columns")
    semantics = entry.get("activation_semantics")
    if not isinstance(semantics, dict):
        raise ValueError(f"{name}: missing activation semantics.")
    for key, expected in {
        "physical_input": "rank-local logical input followed by padding_columns zeros",
        "preparation_order": _PREPARATION_ORDER,
        "dummy_metadata": {"weight_scale": 0.0, "weight_bias": 0.0, "rht_sign": 1},
        "quantization": "per-token per-expert TP-rank-local E4M3FN amax/448 with min_scale=1e-12",
        "bias_correction": "rank-local rotated-input dot weight_bias, added once to that rank projection",
        "down_aggregation": "sum rank-local down projection partials; never duplicate a full-K bias",
        "tp1_bitwise_equivalent": False,
    }.items():
        _equal(semantics.get(key), expected, f"{name}.activation_semantics.{key}")
    spec = VQ2TP2MatrixSpec(name, layer, expert, kind, rank, canonical, logical, packed_shape, tiles, counts, entry)
    shapes = {field: list(shape) for field, shape in spec.tensor_shapes.items()}
    _equal(entry.get("tensor_shapes"), shapes, f"{name}.tensor_shapes")
    records = {
        field: {"key": f"{expert}.{kind}.{field}", "dtype": _DTYPES[field], "shape": shape}
        for field, shape in shapes.items()
    }
    _equal(entry.get("tensors"), records, f"{name}.tensors")
    return spec


def open_vq2a8_tp2_artifact(
    artifact_path: str | Path, model_config_path: str | Path, *, tp_rank: int = 0, verify_tensor_hashes: bool = True
) -> VQ2TP2Artifact:
    """Strictly open a complete artifact; no vLLM/NPU imports or device claims.

    SHA verification is performed once per local-rank shard when enabled.
    File identities are rechecked on reads; the artifact tree must remain
    immutable for the reader's lifetime. Matrix values are validated on load.
    """
    if type(tp_rank) is not int or tp_rank not in (0, 1) or type(verify_tensor_hashes) is not bool:
        raise ValueError("TP2 rank must be integer 0/1 and verify_tensor_hashes must be bool.")
    root, config = _no_links(Path(artifact_path)), _no_links(Path(model_config_path))
    if not root.is_dir() or not config.is_file():
        raise ValueError("TP2 artifact/config must be a directory/regular file.")
    manifest = _json(_file(root, "manifest.json"))
    for key, expected in {
        "schema_version": 1,
        "format": VQ2_TP2_ZN_FORMAT,
        "tp_size": 2,
        "tp_ranks": [0, 1],
        "complete": True,
        "dry_run": False,
        "runtime_compatible": False,
        "tensor_values_verified": True,
        "tp1_a8_bitwise_equivalent": False,
    }.items():
        _equal(manifest.get(key), expected, f"manifest.{key}")
    _equal(manifest.get("communication"), _COMMUNICATION, "manifest.communication")
    layout = load_model_layout(config)
    if layout.hidden_size % 256 or layout.moe_intermediate_size % 256:
        raise ValueError("TP2 model geometry must support K256 and complete local RHT128 blocks.")
    _equal(manifest.get("model_layout"), asdict(layout), "manifest.model_layout")
    _equal(manifest.get("layers_expected"), layout.num_hidden_layers, "manifest.layers_expected")
    selected = _integers(manifest.get("layers_selected"), "manifest.layers_selected")
    _equal(selected, tuple(range(layout.num_hidden_layers)), "manifest.layers_selected")
    source = manifest.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("config"), dict):
        raise ValueError("TP2 manifest is missing its source config binding.")
    _equal(_digest(source["config"].get("sha256"), "source config SHA"), _sha256(config), "source config SHA")
    entries = manifest.get("shards")
    if not isinstance(entries, list) or not entries:
        raise ValueError("TP2 manifest must contain shards.")
    grouped: dict[tuple[int, int], list[VQ2TP2Shard]] = {(layer, rank): [] for layer in selected for rank in (0, 1)}
    used_paths: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("TP2 shard entries must be objects.")
        layer = _integer(entry.get("layer"), "shard.layer")
        rank = _integer(entry.get("rank"), "shard.rank")
        if (layer, rank) not in grouped:
            raise ValueError("TP2 shard references an invalid layer/rank.")
        experts = _integers(entry.get("expert_ids"), "shard.expert_ids")
        if not experts or experts != tuple(range(experts[0], experts[-1] + 1)):
            raise ValueError("TP2 shard expert_ids must be sorted, contiguous and unique.")
        stem = f"tp2/rank{rank}/layer_{layer:03d}/experts_{experts[0]:04d}_{experts[-1] + 1:04d}"
        tensor_path, metadata_path = _file(root, entry.get("file")), _file(root, entry.get("metadata_file"))
        _equal(entry["file"], stem + ".safetensors", "shard.file")
        _equal(entry["metadata_file"], stem + ".json", "shard.metadata_file")
        if tensor_path in used_paths or metadata_path in used_paths:
            raise ValueError("Duplicate TP2 shard path.")
        used_paths.update((tensor_path, metadata_path))
        tensor_sha = _digest(entry.get("sha256"), "shard.sha256")
        metadata_sha = _digest(entry.get("metadata_sha256"), "shard.metadata_sha256")
        _equal(_sha256(metadata_path), metadata_sha, "TP2 metadata SHA-256")
        metadata = _json(metadata_path)
        for key, expected in {"layer": layer, "rank": rank, "sha256": tensor_sha}.items():
            _equal(metadata.get(key), expected, f"shard metadata.{key}")
        matrices = metadata.get("matrices")
        if not isinstance(matrices, list) or len(matrices) != 2 * len(experts):
            raise ValueError("TP2 shard must contain gate_up and down metadata for every expert.")
        specs = {}
        for matrix in matrices:
            spec = _matrix(matrix, layout, layer, rank, experts)
            key = (spec.expert_id, spec.kind)
            if key in specs:
                raise ValueError("Duplicate TP2 matrix metadata.")
            specs[key] = spec
        expected_keys = {(expert, kind) for expert in experts for kind in _KINDS}
        _equal(set(specs), expected_keys, "TP2 matrix coverage")
        expected_header = {
            f"{expert}.{kind}.{field}": (_DTYPES[field], shape)
            for (expert, kind), spec in specs.items()
            for field, shape in spec.tensor_shapes.items()
        }
        before = _identity(tensor_path)
        _equal(_header(tensor_path), expected_header, "TP2 tensor header")
        payload_bytes = sum(math.prod(shape) * _DTYPE_BYTES[dtype] for dtype, shape in expected_header.values())
        _equal(entry.get("payload_bytes"), payload_bytes, "shard.payload_bytes")
        if verify_tensor_hashes and rank == tp_rank:
            _equal(_sha256(tensor_path), tensor_sha, "TP2 tensor SHA-256")
        if _identity(tensor_path) != before:
            raise ValueError("TP2 tensor file changed during validation.")
        grouped[layer, rank].append(
            VQ2TP2Shard(
                layer, rank, experts, tensor_path, metadata_path, tensor_sha, metadata_sha, payload_bytes, specs, before
            )
        )
    rank_bytes = [0, 0]
    layers = {}
    for (layer, rank), shards in grouped.items():
        shards.sort(key=lambda shard: shard.expert_ids[0])
        experts = tuple(expert for shard in shards for expert in shard.expert_ids)
        _equal(experts, layout.expected_expert_ids(layer), f"TP2 layer {layer} rank {rank} expert coverage")
        rank_bytes[rank] += sum(shard.payload_bytes for shard in shards)
        if rank == tp_rank:
            matrix_specs = {
                kind: {expert: shard.specs[expert, kind] for shard in shards for expert in shard.expert_ids}
                for kind in _KINDS
            }
            layers[layer] = VQ2TP2Layer(layer, rank, experts, tuple(shards), matrix_specs)
    _equal(manifest.get("per_rank_payload_bytes"), rank_bytes, "manifest.per_rank_payload_bytes")
    for layer in selected:
        by_rank = [
            {expert: shard.specs[expert, "down"] for shard in grouped[layer, rank] for expert in shard.expert_ids}
            for rank in (0, 1)
        ]
        for expert in layout.expected_expert_ids(layer):
            populations = [0] * (layout.moe_intermediate_size // 256)
            for rank in (0, 1):
                spec = by_rank[rank][expert]
                for tile, count in zip(spec.lut_source_tile_ids, spec.tile_valid_counts):
                    populations[tile] += count
            if any(count != 256 for count in populations):
                raise ValueError(
                    f"Layer {layer} expert {expert}: down rank tile populations do not partition canonical K."
                )
    return VQ2TP2Artifact(root, config, layout, tp_rank, layers, manifest, verify_tensor_hashes)
