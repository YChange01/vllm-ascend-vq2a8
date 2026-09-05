# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict reader for the frozen TP1 VQ2A8 runtime artifact.

The offline repacker writes one safetensors file per layer with an explicit
expert axis.  This module validates the complete model/artifact binding and
then exposes bounded, one-expert-at-a-time reads for correctness bring-up.
It deliberately does not register a vLLM quantization method or a device
kernel; those integrations are enabled only after the standalone runtime gate
passes on Ascend 950.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch
from safetensors import safe_open

from vllm_ascend.quantization.vq2a8_artifact import (
    VQ2_CODEBOOK_SIZE,
    VQ2_CONSUMER_REFERENCE_COMMIT,
    VQ2_INDEX_BITS,
    VQ2_INDICES_PER_WORD,
    VQ2_MATRIX_KINDS,
    VQ2_VECTOR_LENGTH,
    VQ2MatrixSpec,
    VQ2ModelLayout,
    load_model_layout,
    read_safetensors_header,
)
from vllm_ascend.quantization.vq2a8_repack import (
    VQ2_DIRECT_TP1_FORMAT,
    validate_repacked_matrix,
)

VQ2_TP1_FIELDS = (
    "packed_indices",
    "codebooks",
    "codebook_tile_ids",
    "weight_scale",
    "weight_bias",
    "rht_sign",
)
VQ2_TP1_SCHEMA_VERSION = 1
VQ2_TP1_REPACK_CONTRACT_REVISION = 1
VQ2_TP1_AXES = {
    "packed_indices": ("expert", "output_pair", "packed_input_word"),
    "codebooks": ("expert", "codebook_column_tile", "row_tile", "code", "vector_component"),
    "codebook_tile_ids": ("expert", "input_column"),
    "weight_scale": ("expert", "input_column"),
    "weight_bias": ("expert", "input_column"),
    "rht_sign": ("expert", "input_column"),
}
VQ2_TP1_DTYPES = {
    "packed_indices": "I32",
    "codebooks": "F8_E4M3",
    "codebook_tile_ids": "U8",
    "weight_scale": "F32",
    "weight_bias": "F32",
    "rht_sign": "I8",
}
VQ2_TP1_TORCH_DTYPES = {
    "packed_indices": torch.int32,
    "codebooks": torch.float8_e4m3fn,
    "codebook_tile_ids": torch.uint8,
    "weight_scale": torch.float32,
    "weight_bias": torch.float32,
    "rht_sign": torch.int8,
}
VQ2_TP1_COMMUNICATION = {
    "reduction_required": False,
    "tp_reduction_owner": "none_for_tp1",
    "tp_reduction_count": 0,
    "multi_rank_owner": "moe_runner",
}
VQ2_TP1_PACKING = {
    "axis": "input_column",
    "word_dtype": "I32",
    "endianness": "little",
    "index_bits": VQ2_INDEX_BITS,
    "indices_per_word": VQ2_INDICES_PER_WORD,
    "nibble_order": "least_significant_first",
    "padding_nibbles": "zero",
    "vector_length": VQ2_VECTOR_LENGTH,
}
VQ2_TP1_TRANSFORMS = {
    "permutation": "absorbed_offline_with_argsort_perm",
    "normalization": "activation_side_scale_and_bias_correction",
    "rht": "activation_side_normalized_sylvester_once",
}

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class VQ2TP1Layer:
    """Validated locators and homogeneous matrix geometry for one layer."""

    layer_index: int
    expert_ids: tuple[int, ...]
    tensor_path: Path
    metadata_path: Path
    tensor_sha256: str
    metadata_sha256: str
    specs: dict[str, VQ2MatrixSpec]
    tensor_shapes: dict[str, tuple[int, ...]]

    def spec_for(self, expert_id: int, kind: str) -> VQ2MatrixSpec:
        if kind not in VQ2_MATRIX_KINDS:
            raise ValueError(f"Unknown VQ2A8 matrix kind: {kind!r}.")
        if expert_id not in self.expert_ids:
            raise KeyError(
                f"Layer {self.layer_index} has no expert {expert_id}; "
                f"valid range is [{self.expert_ids[0]}, {self.expert_ids[-1]}]."
            )
        representative = self.specs[kind]
        return replace(
            representative,
            name=f"{self.layer_index}.mlp.experts.{expert_id}.{kind}",
            expert_id=expert_id,
        )


@dataclass(frozen=True)
class VQ2TP1Artifact:
    """A validated complete TP1 artifact bound to one model config."""

    root: Path
    model_config_path: Path
    model_layout: VQ2ModelLayout
    layers: dict[int, VQ2TP1Layer]
    manifest: dict[str, Any]

    def layer(self, layer_index: int) -> VQ2TP1Layer:
        try:
            return self.layers[layer_index]
        except KeyError as error:
            raise KeyError(f"TP1 artifact has no layer {layer_index}.") from error

    def load_expert(
        self,
        layer_index: int,
        expert_id: int,
        kind: str,
        *,
        device: torch.device | str = "cpu",
        validate_payload: bool = True,
    ) -> tuple[dict[str, torch.Tensor], VQ2MatrixSpec]:
        """Load one expert slice without reading the layer's full expert axis."""
        layer = self.layer(layer_index)
        spec = layer.spec_for(expert_id, kind)
        expert_position = layer.expert_ids.index(expert_id)
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(layer.tensor_path, framework="pt", device="cpu") as handle:
            for field in VQ2_TP1_FIELDS:
                name = f"{kind}_{field}"
                tensor = handle.get_slice(name)[expert_position]
                expected_shape = layer.tensor_shapes[name][1:]
                expected_dtype = VQ2_TP1_TORCH_DTYPES[field]
                if tensor.dtype != expected_dtype or tuple(tensor.shape) != expected_shape:
                    raise ValueError(
                        f"Layer {layer_index} expert {expert_id} {name} loaded as "
                        f"dtype={tensor.dtype}, shape={tuple(tensor.shape)}; expected "
                        f"dtype={expected_dtype}, shape={expected_shape}."
                    )
                tensors[field] = tensor.contiguous()
        if validate_payload:
            validate_repacked_matrix(tensors, spec)
        target = torch.device(device)
        if target.type != "cpu":
            tensors = {name: tensor.to(device=target, non_blocking=False) for name, tensor in tensors.items()}
        return tensors, spec


def open_vq2a8_tp1_artifact(
    artifact_path: str | Path,
    model_config_path: str | Path,
    *,
    require_complete: bool = True,
    require_reference_identity: bool = False,
    verify_tensor_hashes: bool = False,
) -> VQ2TP1Artifact:
    """Validate and open a repacked TP1 artifact.

    Header validation is always performed for every layer.  Tensor payload
    SHA-256 verification is opt-in because the production artifact is about
    62 GiB; metadata hashes and the exact model-config hash are always checked.
    """
    root = Path(artifact_path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"VQ2A8 TP1 artifact is not a directory: {root}.")
    unresolved_config_path = Path(model_config_path).expanduser()
    if unresolved_config_path.is_symlink():
        raise FileNotFoundError(f"Model config must be a regular, non-symlink file: {unresolved_config_path}.")
    config_path = unresolved_config_path.resolve(strict=True)
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config must be a regular file: {config_path}.")

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(f"TP1 manifest must be a regular, non-symlink file: {manifest_path}.")
    manifest = _read_json_object(manifest_path, "root manifest")
    _require_equal(manifest, "schema_version", VQ2_TP1_SCHEMA_VERSION, "root manifest")
    _require_equal(manifest, "format", VQ2_DIRECT_TP1_FORMAT, "root manifest")
    _require_equal(manifest, "tp_size", 1, "root manifest")
    _require_equal(manifest, "tp_ranks", [0], "root manifest")
    _require_equal(manifest, "consumer_reference_commit", VQ2_CONSUMER_REFERENCE_COMMIT, "root manifest")
    _require_equal(
        manifest,
        "output",
        {"artifact_root": ".", "path_semantics": "relative_to_manifest"},
        "root manifest",
    )
    _require_equal(manifest, "packing", VQ2_TP1_PACKING, "root manifest")
    _require_equal(manifest, "communication", VQ2_TP1_COMMUNICATION, "root manifest")
    _validate_repacker(manifest.get("repacker"))

    model_layout = load_model_layout(config_path)
    _validate_model_binding(manifest, model_layout, config_path)
    if require_reference_identity:
        evidence = manifest.get("producer_evidence")
        if not isinstance(evidence, dict) or evidence.get("reference_identity_match") is not True:
            raise ValueError("TP1 artifact does not carry the required externally verified producer identity.")

    expected_layers = tuple(range(model_layout.num_hidden_layers))
    selected_layers = _integer_sequence(manifest.get("layers_selected"), "root manifest.layers_selected")
    complete = manifest.get("complete")
    if not isinstance(complete, bool):
        raise ValueError(f"root manifest.complete must be bool, got {complete!r}.")
    if manifest.get("layers_expected") != model_layout.num_hidden_layers:
        raise ValueError(
            "root manifest.layers_expected does not match model config: "
            f"{manifest.get('layers_expected')!r} != {model_layout.num_hidden_layers}."
        )
    if complete != (selected_layers == expected_layers):
        raise ValueError("root manifest.complete is inconsistent with layers_selected and model config.")
    if require_complete and not complete:
        raise ValueError("A complete TP1 artifact is required, but root manifest.complete is false.")

    layer_entries = manifest.get("layers")
    if not isinstance(layer_entries, list) or len(layer_entries) != len(selected_layers):
        raise ValueError("root manifest.layers must contain one entry per selected layer.")
    layers: dict[int, VQ2TP1Layer] = {}
    for expected_layer, entry in zip(selected_layers, layer_entries, strict=True):
        layer = _validate_layer(
            root,
            entry,
            expected_layer,
            model_layout,
            verify_tensor_hash=verify_tensor_hashes,
        )
        layers[expected_layer] = layer
    return VQ2TP1Artifact(
        root=root,
        model_config_path=config_path,
        model_layout=model_layout,
        layers=layers,
        manifest=manifest,
    )


def _validate_layer(
    root: Path,
    entry: object,
    layer_index: int,
    model_layout: VQ2ModelLayout,
    *,
    verify_tensor_hash: bool,
) -> VQ2TP1Layer:
    if not isinstance(entry, dict):
        raise ValueError(f"Root layer {layer_index} entry must be an object.")
    _require_equal(entry, "layer", layer_index, f"root layer {layer_index}")
    expected_expert_ids = model_layout.expected_expert_ids(layer_index)
    _require_equal(entry, "expert_count", len(expected_expert_ids), f"root layer {layer_index}")

    expected_stem = f"experts_vq_layer_{layer_index}"
    metadata_locator = f"tp1/rank0/{expected_stem}.json"
    tensor_locator = f"tp1/rank0/{expected_stem}.safetensors"
    _require_equal(entry, "metadata_file", metadata_locator, f"root layer {layer_index}")
    _require_equal(entry, "tensor_file", tensor_locator, f"root layer {layer_index}")
    metadata_path = _resolve_artifact_file(root, metadata_locator, f"layer {layer_index} metadata")
    tensor_path = _resolve_artifact_file(root, tensor_locator, f"layer {layer_index} tensors")
    metadata_sha256 = _sha256_value(entry.get("metadata_sha256"), f"root layer {layer_index}.metadata_sha256")
    tensor_sha256 = _sha256_value(entry.get("tensor_sha256"), f"root layer {layer_index}.tensor_sha256")
    _require_file_hash(metadata_path, metadata_sha256, f"layer {layer_index} metadata")
    if verify_tensor_hash:
        _require_file_hash(tensor_path, tensor_sha256, f"layer {layer_index} tensors")

    layer_manifest = _read_json_object(metadata_path, f"layer {layer_index} manifest")
    description = f"layer {layer_index} manifest"
    for key, expected in (
        ("schema_version", VQ2_TP1_SCHEMA_VERSION),
        ("format", VQ2_DIRECT_TP1_FORMAT),
        ("layer", layer_index),
        ("tp_size", 1),
        ("tp_rank", 0),
        ("complete", True),
        ("consumer_reference_commit", VQ2_CONSUMER_REFERENCE_COMMIT),
        ("transforms", VQ2_TP1_TRANSFORMS),
        ("communication", VQ2_TP1_COMMUNICATION),
    ):
        _require_equal(layer_manifest, key, expected, description)
    expert_ids = _integer_sequence(layer_manifest.get("expert_ids"), f"{description}.expert_ids")
    if expert_ids != expected_expert_ids:
        raise ValueError(f"{description}.expert_ids is {list(expert_ids)}, expected {list(expected_expert_ids)}.")
    _require_equal(layer_manifest, "num_expected_experts", len(expected_expert_ids), description)
    output = layer_manifest.get("output")
    if not isinstance(output, dict):
        raise ValueError(f"{description}.output must be an object.")
    _require_equal(output, "tensor_file", tensor_path.name, f"{description}.output")
    _require_equal(output, "tensor_sha256", tensor_sha256, f"{description}.output")

    layouts = layer_manifest.get("layout")
    if not isinstance(layouts, dict) or set(layouts) != set(VQ2_MATRIX_KINDS):
        raise ValueError(f"{description}.layout must contain exactly {list(VQ2_MATRIX_KINDS)}.")
    specs: dict[str, VQ2MatrixSpec] = {}
    expected_headers: dict[str, tuple[str, tuple[int, ...]]] = {}
    tensor_shapes: dict[str, tuple[int, ...]] = {}
    for kind in VQ2_MATRIX_KINDS:
        matrix_layout = layouts[kind]
        if not isinstance(matrix_layout, dict):
            raise ValueError(f"{description}.layout.{kind} must be an object.")
        spec = _spec_from_runtime_layout(layer_index, kind, matrix_layout)
        _validate_spec_against_model(spec, model_layout)
        specs[kind] = spec
        tensor_layouts = matrix_layout.get("tensors")
        if not isinstance(tensor_layouts, dict) or set(tensor_layouts) != set(VQ2_TP1_FIELDS):
            raise ValueError(f"{description}.layout.{kind}.tensors has the wrong fields.")
        expected_shapes = _expected_stacked_shapes(spec, len(expected_expert_ids))
        for field in VQ2_TP1_FIELDS:
            tensor_layout = tensor_layouts[field]
            if not isinstance(tensor_layout, dict):
                raise ValueError(f"{description}.layout.{kind}.tensors.{field} must be an object.")
            expected_shape = expected_shapes[field]
            expected_axes = list(VQ2_TP1_AXES[field])
            expected_dtype = VQ2_TP1_DTYPES[field]
            if tensor_layout.get("shape") != list(expected_shape):
                raise ValueError(
                    f"{description} {kind}.{field} shape is {tensor_layout.get('shape')!r}, "
                    f"expected {list(expected_shape)}."
                )
            if tensor_layout.get("axes") != expected_axes or tensor_layout.get("dtype") != expected_dtype:
                raise ValueError(f"{description} {kind}.{field} dtype or axes contract is invalid.")
            tensor_name = f"{kind}_{field}"
            expected_headers[tensor_name] = (expected_dtype, expected_shape)
            tensor_shapes[tensor_name] = expected_shape

    headers = read_safetensors_header(tensor_path)
    if set(headers) != set(expected_headers):
        missing = sorted(set(expected_headers) - set(headers))
        extra = sorted(set(headers) - set(expected_headers))
        raise ValueError(f"Layer {layer_index} tensor names differ; missing={missing}, extra={extra}.")
    for name, expected in expected_headers.items():
        header = headers[name]
        if (header.dtype, header.shape) != expected:
            raise ValueError(
                f"Layer {layer_index} tensor {name} has dtype={header.dtype}, shape={header.shape}; "
                f"expected dtype={expected[0]}, shape={expected[1]}."
            )
    return VQ2TP1Layer(
        layer_index=layer_index,
        expert_ids=expert_ids,
        tensor_path=tensor_path,
        metadata_path=metadata_path,
        tensor_sha256=tensor_sha256,
        metadata_sha256=metadata_sha256,
        specs=specs,
        tensor_shapes=tensor_shapes,
    )


def _spec_from_runtime_layout(layer_index: int, kind: str, layout: dict[str, Any]) -> VQ2MatrixSpec:
    canonical_shape = _shape_pair(layout.get("canonical_shape"), f"layer {layer_index} {kind}.canonical_shape")
    original_shape = _shape_pair(layout.get("original_shape"), f"layer {layer_index} {kind}.original_shape")
    rows, columns = canonical_shape
    metadata = {
        "rows": rows,
        "cols": columns,
        "n_row_tiles": _positive_int(layout.get("row_tiles"), f"layer {layer_index} {kind}.row_tiles"),
        "n_col_tiles": _positive_int(layout.get("column_tiles"), f"layer {layer_index} {kind}.column_tiles"),
        "row_group_size": _positive_int(layout.get("row_group_size"), f"layer {layer_index} {kind}.row_group_size"),
        "group_size": _positive_int(layout.get("group_size"), f"layer {layer_index} {kind}.group_size"),
        "K": VQ2_CODEBOOK_SIZE,
        "index_bits": VQ2_INDEX_BITS,
        "vector_len": VQ2_VECTOR_LENGTH,
        "n_vectors": rows * columns // VQ2_VECTOR_LENGTH,
        "n_elements": rows * columns,
        "orig_shape": list(original_shape),
        "norm_dim": 0,
        "enable_perm": True,
        "enable_norm": True,
        "enable_rht": True,
        "rht_block_size": _positive_int(layout.get("rht_block_size"), f"layer {layer_index} {kind}.rht_block_size"),
        "rht_true_columns": _positive_int(
            layout.get("rht_true_columns"), f"layer {layer_index} {kind}.rht_true_columns"
        ),
    }
    return VQ2MatrixSpec.from_dict(f"{layer_index}.mlp.experts.0.{kind}", metadata)


def _validate_spec_against_model(spec: VQ2MatrixSpec, layout: VQ2ModelLayout) -> None:
    expected = (
        (2 * layout.moe_intermediate_size, layout.hidden_size)
        if spec.kind == "gate_up"
        else (layout.hidden_size, layout.moe_intermediate_size)
    )
    if spec.original_shape != expected:
        raise ValueError(f"{spec.name} original shape is {spec.original_shape}, expected {expected} from config.json.")


def _expected_stacked_shapes(spec: VQ2MatrixSpec, num_experts: int) -> dict[str, tuple[int, ...]]:
    return {
        "packed_indices": (
            num_experts,
            spec.rows // VQ2_VECTOR_LENGTH,
            math.ceil(spec.columns / VQ2_INDICES_PER_WORD),
        ),
        "codebooks": (
            num_experts,
            spec.column_tiles,
            spec.row_tiles,
            VQ2_CODEBOOK_SIZE,
            VQ2_VECTOR_LENGTH,
        ),
        "codebook_tile_ids": (num_experts, spec.columns),
        "weight_scale": (num_experts, spec.columns),
        "weight_bias": (num_experts, spec.columns),
        "rht_sign": (num_experts, spec.columns),
    }


def _validate_model_binding(manifest: dict[str, Any], layout: VQ2ModelLayout, config_path: Path) -> None:
    expected_layout = {
        "num_hidden_layers": layout.num_hidden_layers,
        "num_hash_layers": layout.num_hash_layers,
        "num_routed_experts": layout.num_routed_experts,
        "hidden_size": layout.hidden_size,
        "moe_intermediate_size": layout.moe_intermediate_size,
    }
    _require_equal(manifest, "model_layout", expected_layout, "root manifest")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("root manifest.source must be an object.")
    _require_equal(source, "model_config_file", config_path.name, "root manifest.source")
    expected_hash = _sha256_value(source.get("model_config_sha256"), "root manifest.source.model_config_sha256")
    _require_file_hash(config_path, expected_hash, "model config")


def _validate_repacker(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("root manifest.repacker must be an object.")
    _require_equal(value, "tool", "tools/repack_vq2a8_tp1.py", "root manifest.repacker")
    _require_equal(
        value,
        "contract_revision",
        VQ2_TP1_REPACK_CONTRACT_REVISION,
        "root manifest.repacker",
    )
    _sha256_value(value.get("tool_sha256"), "root manifest.repacker.tool_sha256")


def _resolve_artifact_file(root: Path, locator: str, description: str) -> Path:
    if not isinstance(locator, str) or not locator:
        raise ValueError(f"{description} locator must be a non-empty string.")
    posix = PurePosixPath(locator)
    windows = PureWindowsPath(locator)
    parts = locator.split("/")
    if (
        "\\" in locator
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"{description} has an unsafe locator: {locator!r}.")
    candidate = root.joinpath(*parts)
    component = root
    for part in parts:
        component /= part
        if component.is_symlink():
            raise ValueError(f"{description} must not traverse a symbolic link: {locator!r}.")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{description} escapes artifact root: {locator!r}.") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} is not a regular file: {resolved}.")
    return resolved


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as json_file:
            value = json.load(json_file, object_pairs_hook=_object_without_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {description} at {path}: {error}.") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}.")
    return value


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _require_equal(payload: dict[str, Any], key: str, expected: object, description: str) -> None:
    if payload.get(key) != expected:
        raise ValueError(f"{description}.{key} must be {expected!r}, got {payload.get(key)!r}.")


def _sha256_value(value: object, description: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256 digest, got {value!r}.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file_hash(path: Path, expected: str, description: str) -> None:
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"{description} SHA-256 mismatch: expected {expected}, got {actual} ({path}).")


def _integer_sequence(value: object, description: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not int or item < 0 for item in value)
        or value != sorted(set(value))
    ):
        raise ValueError(f"{description} must be non-empty sorted unique non-negative integers, got {value!r}.")
    return tuple(value)


def _positive_int(value: object, description: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{description} must be a positive integer, got {value!r}.")
    return value


def _shape_pair(value: object, description: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in value)
    ):
        raise ValueError(f"{description} must contain two positive integers, got {value!r}.")
    return value[0], value[1]
