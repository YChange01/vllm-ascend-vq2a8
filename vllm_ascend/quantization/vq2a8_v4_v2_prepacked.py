# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable, CPU-only reader for the current V4/v2 compressed payload.

These are final compressed bytes, not old V3 banks or device pointer tables.
All files have an explicit expert axis. Startup always checks model binding,
coverage and tensor headers; large payload hashes are optionally verified.
"""

from __future__ import annotations

import json
import math
import struct
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
import torch
from safetensors import safe_open

from .vq2a8_artifact import VQ2_MATRIX_KINDS, VQ2MatrixSpec, VQ2ModelLayout, load_model_layout
from .vq2a8_repack import VQ2_DIRECT_TP1_FORMAT
from .vq2a8_runtime import (
    _expected_stacked_shapes,
    _integer_sequence,
    _object_without_duplicates,
    _read_json_object,
    _require_file_hash,
    _resolve_artifact_file,
    _sha256_value,
    _spec_from_runtime_layout,
    _validate_spec_against_model,
)
from .vq2a8_v4_v2_layout import V4_V2_DTYPES, V4_V2_FIELDS, _geometry

V4_V2_PREPACKED_FORMAT = "vq2a8_v4_v2_prepacked_v1"
V4_V2_PREPACKED_SCHEMA_VERSION = 1
V4_V2_PREPACKED_LAYOUT_VERSION = 1
V4_V2_PREPACKED_DTYPES = ("U8", "U8", "I64", "F32", "F32", "I8")
MAX_PREPACKED_HEADER_BYTES = 1024 * 1024
V4_V2_PREPACKED_VALIDATION_SCOPE = "direct_tp1_validation_current_conversion_serialization_not_device_or_model_accuracy"


def serialize_spec(spec: VQ2MatrixSpec) -> dict[str, Any]:
    """Serialize the same preparation contract as the direct TP1 artifact."""
    return {
        "canonical_shape": [spec.rows, spec.columns],
        "original_shape": list(spec.original_shape),
        "row_tiles": spec.row_tiles,
        "column_tiles": spec.column_tiles,
        "row_group_size": spec.row_group_size,
        "group_size": spec.group_size,
        "rht_block_size": spec.rht_block_size,
        "rht_true_columns": spec.rht_true_columns,
    }


def prepacked_projection_shapes(spec: VQ2MatrixSpec) -> tuple[tuple[int, ...], ...]:
    """Real final layout shapes, distinct from source direct-TP1 headers."""
    n, k = spec.rows, spec.columns
    _geometry(n, k)
    return ((n // 32, k // 16, 16, 8), (k // 256, n // 32, 32), (k,), (k,), (k,), (k,))


def validate_prepacked_payload(payload: Mapping[str, torch.Tensor], spec: VQ2MatrixSpec) -> None:
    """Validate final bytes without decoding packed indices or re-converting.

    CPU checks are load-time only. Arbitrary finite FP8 LUT encodings, including
    negative zero, are preserved; integer nibbles themselves need no scan.
    """
    if set(payload) != set(V4_V2_FIELDS):
        raise ValueError("Prepacked expert must contain exactly the six V4/v2 fields.")
    for field, dtype, shape in zip(V4_V2_FIELDS, V4_V2_DTYPES, prepacked_projection_shapes(spec), strict=True):
        tensor = payload[field]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"Invalid prepacked {field}: require contiguous CPU {dtype} with shape {shape}.")
    lut = payload["pair_lut"].numpy()
    if np.any((lut & np.uint8(127)) == np.uint8(127)):
        raise ValueError("Prepacked pair_lut contains E4M3FN NaN bytes.")
    order = payload["activation_order"].numpy()
    if not np.array_equal(np.sort(order), np.arange(spec.columns, dtype=np.int64)):
        raise ValueError("Prepacked activation_order must be an exact permutation of [0,K).")
    for field in ("weight_scale", "weight_bias"):
        if not np.isfinite(payload[field].numpy()).all():
            raise ValueError(f"Prepacked {field} must be finite.")
    sign = payload["rht_sign"].numpy()
    if not np.all((sign == -1) | (sign == 1)):
        raise ValueError("Prepacked rht_sign must contain only -1 and +1.")


@dataclass(frozen=True)
class V4V2PrepackedShard:
    tensor_path: Path
    sha256: str
    expert_ids: tuple[int, ...]


@dataclass(frozen=True)
class V4V2PrepackedLayer:
    layer_index: int
    expert_ids: tuple[int, ...]
    specs: Mapping[str, VQ2MatrixSpec]
    source_tensor_shapes: Mapping[str, tuple[int, ...]]
    shards: tuple[V4V2PrepackedShard, ...]
    source_tensor_sha256: str
    source_metadata_sha256: str
    v4_v2_prepacked: ClassVar[bool] = True

    def projection_shapes(self, kind: str) -> tuple[tuple[int, ...], ...]:
        if kind not in VQ2_MATRIX_KINDS:
            raise ValueError(f"Unknown V4/v2 matrix kind {kind!r}.")
        return prepacked_projection_shapes(self.specs[kind])

    def spec_for(self, expert_id: int, kind: str) -> VQ2MatrixSpec:
        if type(expert_id) is not int or expert_id not in self.expert_ids:
            raise KeyError(f"Layer {self.layer_index} has no stored expert {expert_id!r}.")
        if kind not in VQ2_MATRIX_KINDS:
            raise ValueError(f"Unknown V4/v2 matrix kind {kind!r}.")
        return replace(self.specs[kind], name=f"{self.layer_index}.mlp.experts.{expert_id}.{kind}", expert_id=expert_id)


@dataclass(frozen=True)
class V4V2PrepackedArtifact:
    root: Path
    model_config_path: Path
    model_layout: VQ2ModelLayout
    layers: Mapping[int, V4V2PrepackedLayer]
    manifest: dict[str, Any]

    def layer(self, layer_index: int) -> V4V2PrepackedLayer:
        if type(layer_index) is not int or layer_index not in self.layers:
            raise KeyError(f"Prepacked V4/v2 artifact has no layer {layer_index!r}.")
        return self.layers[layer_index]

    def load_expert(
        self,
        layer_index: int,
        expert_id: int,
        kind: str,
        *,
        device: torch.device | str = "cpu",
        validate_payload: bool = True,
        timings: dict[str, float] | None = None,
    ) -> tuple[dict[str, torch.Tensor], VQ2MatrixSpec]:
        """Read bounded expert slices; no original payload or conversion needed.

        Shape/dtype checks are unconditional. Optional value validation scans
        only LUT and preparation metadata. Mapping/page faults count toward
        host read/validation time, not a separate physical disk-I/O metric.
        """
        read_start = time.perf_counter()
        layer = self.layer(layer_index)
        spec = layer.spec_for(expert_id, kind)
        shard = next(shard for shard in layer.shards if expert_id in shard.expert_ids)
        # Recheck locators in case files were replaced after opening the artifact.
        locator = shard.tensor_path.relative_to(self.root).as_posix()
        path = _resolve_artifact_file(self.root, locator, "prepacked tensor shard")
        slot = shard.expert_ids.index(expert_id)
        tensors = {}
        with safe_open(path, framework="pt", device="cpu") as handle:
            for field, dtype, shape in zip(V4_V2_FIELDS, V4_V2_DTYPES, layer.projection_shapes(kind), strict=True):
                tensor = handle.get_slice(f"{kind}_{field}")[slot]
                if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                    raise ValueError(f"Prepacked {kind}_{field} changed shape or dtype after artifact validation.")
                tensors[field] = tensor.contiguous()
        read_s = time.perf_counter() - read_start
        validation_start = time.perf_counter()
        if validate_payload:
            validate_prepacked_payload(tensors, spec)
        validation_s = time.perf_counter() - validation_start if validate_payload else 0.0
        target = torch.device(device)
        if target.type != "cpu":
            tensors = {field: tensor.to(target, non_blocking=False) for field, tensor in tensors.items()}
        if timings is not None:
            for key, duration in (("host_read_s", read_s), ("host_validate_s", validation_s)):
                timings[key] = timings.get(key, 0.0) + duration
        return tensors, spec


def _exact_fields(value: object, fields, description: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{description} must contain exactly {sorted(fields)}.")
    return value


def _strict_equal(actual: object, expected: object, description: str) -> None:
    # bool is an int subclass; manifest version/dimension values must not exploit it.
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{description} must be {expected!r}, got {actual!r}.")


def _validate_optional_provenance(manifest: dict[str, Any]) -> None:
    """Exporter evidence is recorded, never compared with installed code hashes."""
    if "producer" in manifest:
        producer = _exact_fields(manifest["producer"], ("tool", "files"), "producer")
        if producer["tool"] not in (
            "tools/prepack_vq2a8_v4_v2.py",  # Existing artifacts retain their original provenance.
            "vllm_ascend.quantization.vq2a8_prepack",
        ):
            raise ValueError("producer.tool must identify a supported VQ2 prepack exporter.")
        files = producer["files"]
        if not isinstance(files, list) or not files:
            raise ValueError("producer.files must contain conversion source identities.")
        seen = set()
        for entry in files:
            _exact_fields(entry, ("file", "sha256"), "producer file")
            name = entry["file"]
            if (
                not isinstance(name, str)
                or not name
                or name in {".", ".."}
                or any(character in name for character in ("/", "\\", ":"))
                or name in seen
            ):
                raise ValueError("producer files require unique portable basenames.")
            seen.add(name)
            _sha256_value(entry["sha256"], "producer source hash")
    if "tensor_bytes_verified" in manifest:
        _strict_equal(manifest["tensor_bytes_verified"], True, "tensor_bytes_verified")
    if "validation_scope" in manifest:
        _strict_equal(manifest["validation_scope"], V4_V2_PREPACKED_VALIDATION_SCOPE, "validation_scope")
    if "payload_bytes" in manifest:
        size = manifest["payload_bytes"]
        if type(size) is not int or size <= 0:
            raise ValueError("payload_bytes must be a positive integer.")


def _validate_header(path: Path, layer: V4V2PrepackedLayer, count: int) -> None:
    """Validate exact safetensors header and payload spans without reading weights."""
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Truncated prepacked safetensors header: {path}.")
        length = struct.unpack("<Q", length_bytes)[0]
        if not 0 < length <= MAX_PREPACKED_HEADER_BYTES or length > file_size - 8:
            raise ValueError(f"Invalid prepacked safetensors header size: {path}.")
        header = json.loads(stream.read(length), object_pairs_hook=_object_without_duplicates)
    if not isinstance(header, dict):
        raise ValueError("Prepacked safetensors header must be an object.")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()
    ):
        raise ValueError("Prepacked safetensors metadata must contain string values.")
    expected_keys = {f"{kind}_{field}" for kind in VQ2_MATRIX_KINDS for field in V4_V2_FIELDS}
    _exact_fields(header, expected_keys, "prepacked safetensors tensors")
    spans = []
    for kind in VQ2_MATRIX_KINDS:
        for field, dtype, torch_dtype, shape in zip(
            V4_V2_FIELDS, V4_V2_PREPACKED_DTYPES, V4_V2_DTYPES, layer.projection_shapes(kind), strict=True
        ):
            name = f"{kind}_{field}"
            entry = _exact_fields(header[name], ("dtype", "shape", "data_offsets"), name)
            _strict_equal(entry["dtype"], dtype, f"{name}.dtype")
            dimensions = entry["shape"]
            if (
                not isinstance(dimensions, list)
                or any(type(dimension) is not int for dimension in dimensions)
                or dimensions != [count, *shape]
            ):
                raise ValueError(f"Prepacked {name} shape differs from model and shard expert inventory.")
            offsets = entry["data_offsets"]
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(offset) is not int or offset < 0 for offset in offsets)
                or offsets[1] - offsets[0] != count * math.prod(shape) * torch_dtype.itemsize
            ):
                raise ValueError(f"Prepacked {name} data offsets do not match its tensor size.")
            spans.append(tuple(offsets))
    end = 0
    for begin, stop in sorted(spans):
        if begin != end:
            raise ValueError("Prepacked safetensors tensor spans overlap or leave holes.")
        end = stop
    if end != file_size - length - 8:
        raise ValueError("Prepacked safetensors payload is truncated or has trailing data.")


def open_vq2a8_v4_v2_prepacked_artifact(
    artifact_path: str | Path,
    model_config_path: str | Path,
    *,
    verify_tensor_hashes: bool = False,
) -> V4V2PrepackedArtifact:
    """Open a complete immutable inventory, independent of original TP1 files."""
    unresolved_root = Path(artifact_path).expanduser()
    if unresolved_root.is_symlink():
        raise ValueError("Prepacked artifact root must not be a symbolic link.")
    root = unresolved_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    config = Path(model_config_path).expanduser()
    if config.is_symlink():
        raise ValueError("Model config must not be a symbolic link.")
    config = config.resolve(strict=True)
    if not config.is_file():
        raise FileNotFoundError(config)
    manifest = _read_json_object(
        _resolve_artifact_file(root, "manifest.json", "prepacked manifest"), "prepacked manifest"
    )
    required_fields = (
        "format",
        "schema_version",
        "layout_version",
        "complete",
        "model_config_sha256",
        "model_layout",
        "source",
        "layers",
    )
    optional_fields = ("producer", "payload_bytes", "tensor_bytes_verified", "validation_scope")
    if set(manifest) - set(required_fields) - set(optional_fields) or set(required_fields) - set(manifest):
        raise ValueError("Prepacked manifest must contain exactly required schema fields and optional provenance.")
    _validate_optional_provenance(manifest)
    for key, expected in (
        ("format", V4_V2_PREPACKED_FORMAT),
        ("schema_version", V4_V2_PREPACKED_SCHEMA_VERSION),
        ("layout_version", V4_V2_PREPACKED_LAYOUT_VERSION),
        ("complete", True),
    ):
        _strict_equal(manifest[key], expected, f"prepacked manifest.{key}")
    _require_file_hash(config, _sha256_value(manifest["model_config_sha256"], "model config hash"), "model config")
    layout = load_model_layout(config)
    layout_fields = _exact_fields(manifest["model_layout"], asdict(layout), "model_layout")
    for key, expected in asdict(layout).items():
        _strict_equal(layout_fields[key], expected, f"model_layout.{key}")
    if layout.num_routed_experts > 256:
        raise ValueError("V4/v2 prepacked layout supports at most 256 experts.")
    source = _exact_fields(manifest["source"], ("format", "manifest_sha256"), "source")
    _strict_equal(source["format"], VQ2_DIRECT_TP1_FORMAT, "source.format")
    _sha256_value(source["manifest_sha256"], "source.manifest_sha256")
    entries = manifest["layers"]
    if not isinstance(entries, list) or len(entries) != layout.num_hidden_layers:
        raise ValueError("Prepacked artifact must cover every model layer exactly once.")
    layers = {}
    seen_files = set()
    for index, entry in enumerate(entries):
        _exact_fields(
            entry,
            (
                "layer_index",
                "expert_ids",
                "specs",
                "source_tensor_shapes",
                "source_tensor_sha256",
                "source_metadata_sha256",
                "shards",
            ),
            f"layer {index}",
        )
        _strict_equal(entry["layer_index"], index, "layer_index")
        ids = _integer_sequence(entry["expert_ids"], f"layer {index}.expert_ids")
        if ids != layout.expected_expert_ids(index):
            raise ValueError(f"Prepacked layer {index} expert inventory does not match model config.")
        spec_entries = _exact_fields(entry["specs"], VQ2_MATRIX_KINDS, f"layer {index}.specs")
        specs, source_shapes = {}, {}
        for kind in VQ2_MATRIX_KINDS:
            spec = _spec_from_runtime_layout(index, kind, spec_entries[kind])
            _exact_fields(spec_entries[kind], serialize_spec(spec), f"layer {index}.{kind} spec")
            _validate_spec_against_model(spec, layout)
            prepacked_projection_shapes(spec)
            if spec.row_group_size != 32 or not 1 <= spec.column_tiles <= 256:
                raise ValueError("V4/v2 prepacked source spec requires row_group_size=32 and 1..256 codebook tiles.")
            specs[kind] = spec
            source_shapes.update(
                {f"{kind}_{field}": shape for field, shape in _expected_stacked_shapes(spec, len(ids)).items()}
            )
        shape_entries = _exact_fields(entry["source_tensor_shapes"], source_shapes, "source_tensor_shapes")
        for name, shape in source_shapes.items():
            value = shape_entries[name]
            if not isinstance(value, list) or any(type(dim) is not int for dim in value) or tuple(value) != shape:
                raise ValueError(f"Prepacked source_tensor_shapes.{name} disagrees with source spec.")
        source_tensor_hash = _sha256_value(entry["source_tensor_sha256"], "source tensor hash")
        source_metadata_hash = _sha256_value(entry["source_metadata_sha256"], "source metadata hash")
        shards = []
        shard_entries = entry["shards"]
        if not isinstance(shard_entries, list) or not shard_entries:
            raise ValueError("Prepacked layer requires nonempty shards.")
        covered = []
        for shard in shard_entries:
            _exact_fields(shard, ("file", "sha256", "expert_ids"), "prepacked shard")
            path = _resolve_artifact_file(root, shard["file"], "prepacked shard")
            if path in seen_files:
                raise ValueError("Duplicate prepacked shard path.")
            seen_files.add(path)
            shard_ids = _integer_sequence(shard["expert_ids"], "shard.expert_ids")
            covered.extend(shard_ids)
            digest = _sha256_value(shard["sha256"], "shard.sha256")
            shards.append(V4V2PrepackedShard(path, digest, shard_ids))
        if tuple(covered) != ids:
            raise ValueError("Prepacked shards must cover layer expert IDs once in exact order.")
        layer = V4V2PrepackedLayer(
            index,
            ids,
            MappingProxyType(specs),
            MappingProxyType(source_shapes),
            tuple(shards),
            source_tensor_hash,
            source_metadata_hash,
        )
        for shard in shards:
            _validate_header(shard.tensor_path, layer, len(shard.expert_ids))
            if verify_tensor_hashes:
                _require_file_hash(shard.tensor_path, shard.sha256, "prepacked shard")
        layers[index] = layer
    if "payload_bytes" in manifest:
        payload_bytes = sum(
            len(layer.expert_ids) * math.prod(shape) * dtype.itemsize
            for layer in layers.values()
            for kind in VQ2_MATRIX_KINDS
            for shape, dtype in zip(layer.projection_shapes(kind), V4_V2_DTYPES, strict=True)
        )
        _strict_equal(manifest["payload_bytes"], payload_bytes, "payload_bytes")
    return V4V2PrepackedArtifact(root, config, layout, MappingProxyType(layers), manifest)
