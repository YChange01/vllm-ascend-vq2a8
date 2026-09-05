# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repack canonical VQ2A8 experts into the frozen TP1 direct layout.

This is an offline, CPU-only conversion.  It never materializes dense BF16
weights and never mutates the canonical ``experts_vq`` source directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm_ascend.quantization.vq2a8_artifact import (
    VQ2_CODEBOOK_SIZE,
    VQ2_CONSUMER_REFERENCE_COMMIT,
    VQ2_INDEX_BITS,
    VQ2_INDICES_PER_WORD,
    VQ2_MATRIX_KINDS,
    VQ2_VECTOR_LENGTH,
    VQ2MatrixSpec,
    inspect_layer_artifact,
    load_layer_specs,
    load_model_layout,
    validate_model_layout,
)
from vllm_ascend.quantization.vq2a8_repack import (
    VQ2_DIRECT_TP1_FORMAT,
    repack_matrix_tp1,
)

_OUTPUT_FIELDS = (
    "packed_indices",
    "codebooks",
    "codebook_tile_ids",
    "weight_scale",
    "weight_bias",
    "rht_sign",
)
_LAYER_FILE_PATTERN = re.compile(r"experts_vq_layer_(0|[1-9]\d*)\.(json|safetensors)")
_LAYER_FILE_CANDIDATE_PATTERN = re.compile(r"experts_vq_layer_(\d+)\.(json|safetensors)")
_LAYER_SELECTION_PATTERN = re.compile(r"(0|[1-9]\d*)(?:-(0|[1-9]\d*))?")
_SHA256SUM_LINE_PATTERN = re.compile(r"([0-9a-f]{64}) ([ *])(.+)")
_MANIFEST_SCHEMA_VERSION = 1
_REPACK_CONTRACT_REVISION = 1
_REFERENCE_MODEL_CONFIG_SHA256 = "df08fb78e87c77407d60670e242e86bc78e9044549b94243dc5f40b9047b9552"
_REFERENCE_SHA256SUMS_SHA256 = "8913f673ec6952ab8c62de6ab48bffbaa3983fdd71649b61174058a86a976cac"
_PER_LAYER_HEADER_RESERVE_BYTES = 1024 * 1024
_ROOT_MANIFEST_RESERVE_BYTES = 4 * 1024 * 1024
_MINIMUM_DISK_SAFETY_MARGIN_BYTES = 64 * 1024 * 1024
_DISK_SAFETY_MARGIN_RATIO = 0.02
_STABLE_DTYPES = {
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float32: "F32",
    torch.int8: "I8",
    torch.int32: "I32",
    torch.uint8: "U8",
}
_OUTPUT_AXES = {
    "packed_indices": ("expert", "output_pair", "packed_input_word"),
    "codebooks": ("expert", "codebook_column_tile", "row_tile", "code", "vector_component"),
    "codebook_tile_ids": ("expert", "input_column"),
    "weight_scale": ("expert", "input_column"),
    "weight_bias": ("expert", "input_column"),
    "rht_sign": ("expert", "input_column"),
}
_OUTPUT_DTYPE_NAMES = {
    "packed_indices": "I32",
    "codebooks": "F8_E4M3",
    "codebook_tile_ids": "U8",
    "weight_scale": "F32",
    "weight_bias": "F32",
    "rht_sign": "I8",
}
_TP1_COMMUNICATION_CONTRACT = {
    "reduction_required": False,
    "tp_reduction_owner": "none_for_tp1",
    "tp_reduction_count": 0,
    "multi_rank_owner": "moe_runner",
}
_PACKING_CONTRACT = {
    "axis": "input_column",
    "word_dtype": "I32",
    "endianness": "little",
    "index_bits": VQ2_INDEX_BITS,
    "indices_per_word": VQ2_INDICES_PER_WORD,
    "nibble_order": "least_significant_first",
    "padding_nibbles": "zero",
    "vector_length": VQ2_VECTOR_LENGTH,
}
_REFERENCE_PRODUCER_EVIDENCE = {
    "repository_url": "https://github.com/mehrantgn/VPTQ-plusplus",
    "repository_revision": "8b86f2ea6f4da1ce8008f4b1bece0443fa44859b",
    "repository_branch": "tile-codebook",
    "codec_path": "vptq/utils/expert_vq.py",
    "codec_sha256": "d3886161949394bca124363c5fa60c2f9a05a5c066927ccfad0b5cf127a5cd46",
    "codec_tracked_at_revision": False,
    "generator_script_sha256": "cf70968e5f75b1b75a0734ba25e45f3b152b7d7a89133e39d6afc74d6f515796",
    "generator_log_sha256": "179d3cb063617282974d19a0eb532bbc7729c7179d4e8d0be57ccf9083c8179c",
    "generator_exit_code": 0,
    "generator_weight_checks": 30729,
    "generator_weight_mismatches": 0,
}


def _parse_layers(specification: str, available: tuple[int, ...]) -> tuple[int, ...]:
    if specification == "all":
        return available
    selected: set[int] = set()
    for raw_part in specification.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("Layer selection contains an empty item.")
        match = _LAYER_SELECTION_PATTERN.fullmatch(part)
        if match is None:
            raise ValueError(f"Invalid layer selection item: {part!r}.")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        if end < start:
            raise ValueError(f"Layer range ends before it starts: {part!r}.")
        selected.update(range(start, end + 1))
    missing = sorted(selected - set(available))
    if missing:
        raise ValueError(f"Selected layers are absent from the source artifact: {missing}.")
    if not selected:
        raise ValueError("At least one layer must be selected.")
    return tuple(sorted(selected))


def _discover_layers(source: Path) -> tuple[int, ...]:
    by_suffix: dict[str, set[int]] = {"json": set(), "safetensors": set()}
    for path in source.iterdir():
        candidate = _LAYER_FILE_CANDIDATE_PATTERN.fullmatch(path.name)
        if candidate is not None and _LAYER_FILE_PATTERN.fullmatch(path.name) is None:
            raise ValueError(f"Non-canonical layer artifact filename with a leading zero: {path.name!r}.")
        match = _LAYER_FILE_PATTERN.fullmatch(path.name)
        if match is not None:
            if not path.is_file():
                raise ValueError(f"Canonical layer artifact is not a file: {path}.")
            by_suffix[match.group(2)].add(int(match.group(1)))
    if not by_suffix["json"] and not by_suffix["safetensors"]:
        raise ValueError(f"No canonical VQ2A8 layer artifacts found in {source}.")
    if by_suffix["json"] != by_suffix["safetensors"]:
        raise ValueError(
            "Canonical layer files are not paired; "
            f"missing_json={sorted(by_suffix['safetensors'] - by_suffix['json'])}, "
            f"missing_safetensors={sorted(by_suffix['json'] - by_suffix['safetensors'])}."
        )
    return tuple(sorted(by_suffix["json"]))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_file_snapshot(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "sha256": _sha256_file(path),
    }


def _capture_layer_snapshots(source: Path, layers: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    snapshots: dict[int, dict[str, Any]] = {}
    for layer_index in layers:
        stem = f"experts_vq_layer_{layer_index}"
        snapshots[layer_index] = {
            "metadata": _source_file_snapshot(source / f"{stem}.json"),
            "tensors": _source_file_snapshot(source / f"{stem}.safetensors"),
        }
    return snapshots


def _canonical_checksum_path(raw_path: str, *, line_number: int) -> str:
    if raw_path.startswith("./"):
        raw_path = raw_path[2:]
    if (
        not raw_path
        or raw_path.startswith("/")
        or "\\" in raw_path
        or "\x00" in raw_path
        or any(part in {"", ".", ".."} for part in raw_path.split("/"))
    ):
        raise ValueError(f"SHA256SUMS line {line_number} has an unsafe or non-canonical relative path: {raw_path!r}.")
    return raw_path


def _parse_sha256sums(contents: bytes) -> dict[str, str]:
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("SHA256SUMS must be valid UTF-8.") from error

    entries: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        match = _SHA256SUM_LINE_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"Malformed SHA256SUMS line {line_number}: {line!r}.")
        relative_path = _canonical_checksum_path(match.group(3), line_number=line_number)
        if relative_path in entries:
            raise ValueError(f"SHA256SUMS contains duplicate path {relative_path!r}.")
        entries[relative_path] = match.group(1)
    return entries


def _capture_checksum_snapshot(config: Path) -> tuple[dict[str, Any], dict[str, str]]:
    checksum_path = config.parent / "SHA256SUMS"
    if checksum_path.exists() and not checksum_path.is_file():
        raise ValueError(f"SHA256SUMS exists but is not a file: {checksum_path}.")
    if not checksum_path.is_file():
        return {"file": checksum_path.name, "present": False, "sha256": None}, {}
    contents = checksum_path.read_bytes()
    return (
        {
            "file": checksum_path.name,
            "present": True,
            "sha256": hashlib.sha256(contents).hexdigest(),
        },
        _parse_sha256sums(contents),
    )


def _verify_source_snapshot(
    source: Path,
    config: Path,
    config_sha256: str,
    checksum_snapshot: dict[str, Any],
    layer_snapshots: dict[int, dict[str, Any]],
) -> None:
    def verify(path: Path, expected_sha256: str) -> None:
        if not path.is_file():
            raise RuntimeError(f"Source snapshot changed during repack: missing file {path}.")
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Source snapshot changed during repack: {path} expected sha256={expected_sha256}, got {actual_sha256}."
            )

    verify(config, config_sha256)
    checksum_path = config.parent / str(checksum_snapshot["file"])
    checksum_present = checksum_path.is_file()
    if checksum_present != checksum_snapshot["present"]:
        raise RuntimeError(
            "Source snapshot changed during repack: "
            f"{checksum_path} presence changed from {checksum_snapshot['present']} to {checksum_present}."
        )
    if checksum_present:
        verify(checksum_path, str(checksum_snapshot["sha256"]))
    for layer_index, snapshot in layer_snapshots.items():
        verify(source / snapshot["metadata"]["file"], snapshot["metadata"]["sha256"])
        verify(source / snapshot["tensors"]["file"], snapshot["tensors"]["sha256"])


def _checksum_path_for(path: Path, model_root: Path) -> str | None:
    try:
        relative = path.relative_to(model_root)
    except ValueError:
        return None
    if not relative.parts:
        return None
    return "/".join(relative.parts)


def _selected_snapshot_matches_checksum_manifest(
    source: Path,
    config: Path,
    config_sha256: str,
    layer_snapshots: dict[int, dict[str, Any]],
    checksum_entries: dict[str, str],
) -> tuple[bool, int]:
    model_root = config.parent
    selected_files: list[tuple[Path, str]] = [(config, config_sha256)]
    for snapshot in layer_snapshots.values():
        for artifact_kind in ("metadata", "tensors"):
            artifact = snapshot[artifact_kind]
            selected_files.append((source / artifact["file"], artifact["sha256"]))

    for path, observed_sha256 in selected_files:
        relative_path = _checksum_path_for(path, model_root)
        if relative_path is None or checksum_entries.get(relative_path) != observed_sha256:
            return False, len(selected_files)
    return True, len(selected_files)


def _producer_evidence(
    source: Path,
    config: Path,
    config_sha256: str,
    checksum_snapshot: dict[str, Any],
    checksum_entries: dict[str, str],
    layer_snapshots: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    checksum_sha256 = checksum_snapshot["sha256"]
    selected_snapshot_match, selected_snapshot_file_count = _selected_snapshot_matches_checksum_manifest(
        source,
        config,
        config_sha256,
        layer_snapshots,
        checksum_entries,
    )
    reference_identity_match = (
        config_sha256 == _REFERENCE_MODEL_CONFIG_SHA256
        and checksum_snapshot["present"]
        and checksum_sha256 == _REFERENCE_SHA256SUMS_SHA256
        and selected_snapshot_match
    )
    return {
        "evidence_status": (
            "externally_verified_for_this_input" if reference_identity_match else "unverified_for_this_input"
        ),
        "reference_identity_match": reference_identity_match,
        "reference_identity": {
            "expected_model_config_sha256": _REFERENCE_MODEL_CONFIG_SHA256,
            "observed_model_config_sha256": config_sha256,
            "expected_sha256sums_sha256": _REFERENCE_SHA256SUMS_SHA256,
            "observed_sha256sums_sha256": checksum_sha256,
            "selected_snapshot_file_count": selected_snapshot_file_count,
            "selected_snapshot_matches_sha256sums": selected_snapshot_match,
            "verification_scope": "model_config_and_selected_layer_files",
        },
        "reference_evidence": dict(_REFERENCE_PRODUCER_EVIDENCE),
    }


def _estimate_output_bytes(
    source: Path,
    layers: tuple[int, ...],
    expected_experts: dict[int, tuple[int, ...]],
) -> tuple[int, int]:
    tensor_payload_bytes = 0
    for layer_index in layers:
        specs = load_layer_specs(source, layer_index)
        expert_ids = expected_experts[layer_index]
        representative_expert = expert_ids[0]
        num_experts = len(expert_ids)
        for kind in VQ2_MATRIX_KINDS:
            spec = specs[f"{layer_index}.mlp.experts.{representative_expert}.{kind}"]
            output_pairs = spec.rows // VQ2_VECTOR_LENGTH
            packed_words = math.ceil(spec.columns / VQ2_INDICES_PER_WORD)
            tensor_payload_bytes += num_experts * output_pairs * packed_words * torch.int32.itemsize
            tensor_payload_bytes += (
                num_experts * spec.column_tiles * spec.row_tiles * VQ2_CODEBOOK_SIZE * VQ2_VECTOR_LENGTH
            )
            tensor_payload_bytes += num_experts * spec.columns * torch.uint8.itemsize
            tensor_payload_bytes += 2 * num_experts * spec.columns * torch.float32.itemsize
            tensor_payload_bytes += num_experts * spec.columns * torch.int8.itemsize
    metadata_reserve_bytes = len(layers) * _PER_LAYER_HEADER_RESERVE_BYTES + _ROOT_MANIFEST_RESERVE_BYTES
    return tensor_payload_bytes, metadata_reserve_bytes


def _fsync_file(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_paths(source: Path, output: Path, config: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Canonical experts_vq directory not found: {source}.")
    if not config.is_file():
        raise FileNotFoundError(f"Model config not found: {config}.")
    if source == output or _is_relative_to(output, source) or _is_relative_to(source, output):
        raise ValueError(
            "Input and output must be separate sibling trees; neither may contain the other "
            f"(input={source}, output={output})."
        )
    if output.exists():
        raise FileExistsError(f"Output path already exists; refusing to overwrite it: {output}.")


def _source_payload(handle: Any, spec: VQ2MatrixSpec) -> dict[str, torch.Tensor]:
    return {field: handle.get_tensor(f"{spec.name}.{field}") for field in spec.expected_tensor_headers()}


def _matrix_manifest(spec: VQ2MatrixSpec, tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    tensor_manifests: dict[str, Any] = {}
    for field in _OUTPUT_FIELDS:
        tensor = tensors[field]
        dtype = _STABLE_DTYPES.get(tensor.dtype)
        if dtype is None:
            raise ValueError(f"No stable manifest dtype for {spec.name}.{field}: {tensor.dtype}.")
        axes = _OUTPUT_AXES[field]
        if tensor.ndim != len(axes):
            raise ValueError(
                f"Manifest axes for {spec.name}.{field} have length {len(axes)}, but tensor rank is {tensor.ndim}."
            )
        tensor_manifests[field] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "axes": list(axes),
        }
    return {
        "canonical_shape": [spec.rows, spec.columns],
        "original_shape": list(spec.original_shape),
        "row_group_size": spec.row_group_size,
        "group_size": spec.group_size,
        "row_tiles": spec.row_tiles,
        "column_tiles": spec.column_tiles,
        "rht_block_size": spec.rht_block_size,
        "rht_true_columns": spec.rht_true_columns,
        "tensors": tensor_manifests,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8", newline="\n") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _write_safetensors_atomic(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    save_file(tensors, temporary)
    _fsync_file(temporary)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _raise_staging_validation_error(message: str) -> None:
    raise ValueError(f"Staging artifact validation failed: {message}")


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _raise_staging_validation_error(f"cannot read {description} at {path}: {error}.")
    if not isinstance(payload, dict):
        _raise_staging_validation_error(f"{description} must contain a JSON object.")
    return payload


def _resolve_staging_file(staging: Path, locator: object, *, description: str) -> Path:
    if not isinstance(locator, str) or not locator:
        _raise_staging_validation_error(f"{description} must be a non-empty relative path string.")
    posix_path = PurePosixPath(locator)
    windows_path = PureWindowsPath(locator)
    parts = locator.split("/")
    if (
        "\\" in locator
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in parts)
    ):
        _raise_staging_validation_error(f"{description} has an unsafe artifact locator: {locator!r}.")

    staging_root = staging.resolve(strict=True)
    candidate = staging.joinpath(*parts)
    component = staging
    for part in parts:
        component /= part
        if component.is_symlink():
            _raise_staging_validation_error(f"{description} must not traverse a symbolic link: {locator!r}.")
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        _raise_staging_validation_error(f"{description} does not resolve to a staged file: {locator!r} ({error}).")
    if not _is_relative_to(resolved, staging_root):
        _raise_staging_validation_error(f"{description} escapes the staging directory: {locator!r}.")
    if not resolved.is_file():
        _raise_staging_validation_error(f"{description} is not a regular file: {locator!r}.")
    return resolved


def _manifest_sha256(value: object, *, description: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _raise_staging_validation_error(f"{description} is not a lowercase SHA-256 digest.")
    return value


def _require_manifest_value(
    payload: dict[str, Any],
    key: str,
    expected: object,
    *,
    description: str,
) -> None:
    if payload.get(key) != expected:
        _raise_staging_validation_error(f"{description}.{key} must be {expected!r}, got {payload.get(key)!r}.")


def _validate_staging_artifact(staging: Path, expected_manifest: dict[str, Any]) -> None:
    manifest_path = staging / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        _raise_staging_validation_error("manifest.json must be a regular, non-symlink file.")
    manifest = _read_json_object(manifest_path, description="root manifest")

    for key, expected in (
        ("schema_version", _MANIFEST_SCHEMA_VERSION),
        ("format", VQ2_DIRECT_TP1_FORMAT),
        ("tp_size", 1),
        ("tp_ranks", [0]),
        ("output", {"artifact_root": ".", "path_semantics": "relative_to_manifest"}),
        ("packing", _PACKING_CONTRACT),
        ("communication", _TP1_COMMUNICATION_CONTRACT),
    ):
        _require_manifest_value(manifest, key, expected, description="root manifest")

    repacker = manifest.get("repacker")
    if not isinstance(repacker, dict):
        _raise_staging_validation_error("root manifest.repacker must be an object.")
    _require_manifest_value(
        repacker,
        "tool",
        "tools/repack_vq2a8_tp1.py",
        description="root manifest.repacker",
    )
    _require_manifest_value(
        repacker,
        "contract_revision",
        _REPACK_CONTRACT_REVISION,
        description="root manifest.repacker",
    )
    _manifest_sha256(repacker.get("tool_sha256"), description="root manifest.repacker.tool_sha256")

    selected_layers = manifest.get("layers_selected")
    expected_layer_count = manifest.get("layers_expected")
    complete = manifest.get("complete")
    layers = manifest.get("layers")
    if (
        not isinstance(selected_layers, list)
        or any(type(layer) is not int or layer < 0 for layer in selected_layers)
        or selected_layers != sorted(set(selected_layers))
    ):
        _raise_staging_validation_error("root manifest.layers_selected must be sorted unique non-negative integers.")
    if type(expected_layer_count) is not int or expected_layer_count <= 0:
        _raise_staging_validation_error("root manifest.layers_expected must be a positive integer.")
    expected_complete = selected_layers == list(range(expected_layer_count))
    if type(complete) is not bool or complete is not expected_complete:
        _raise_staging_validation_error(
            "root manifest.complete is inconsistent with layers_selected and layers_expected."
        )
    if not isinstance(layers, list) or len(layers) != len(selected_layers):
        _raise_staging_validation_error("root manifest.layers must contain one entry per selected layer.")

    referenced_files: set[Path] = set()
    for expected_layer, layer_entry in zip(selected_layers, layers, strict=True):
        if not isinstance(layer_entry, dict):
            _raise_staging_validation_error(f"root layer {expected_layer} entry must be an object.")
        _require_manifest_value(layer_entry, "layer", expected_layer, description=f"root layer {expected_layer}")
        expert_count = layer_entry.get("expert_count")
        if type(expert_count) is not int or expert_count <= 0:
            _raise_staging_validation_error(f"root layer {expected_layer}.expert_count must be positive.")

        expected_metadata_locator = f"tp1/rank0/experts_vq_layer_{expected_layer}.json"
        expected_tensor_locator = f"tp1/rank0/experts_vq_layer_{expected_layer}.safetensors"
        metadata_path = _resolve_staging_file(
            staging,
            layer_entry.get("metadata_file"),
            description=f"root layer {expected_layer}.metadata_file",
        )
        tensor_path = _resolve_staging_file(
            staging,
            layer_entry.get("tensor_file"),
            description=f"root layer {expected_layer}.tensor_file",
        )
        _require_manifest_value(
            layer_entry,
            "metadata_file",
            expected_metadata_locator,
            description=f"root layer {expected_layer}",
        )
        _require_manifest_value(
            layer_entry,
            "tensor_file",
            expected_tensor_locator,
            description=f"root layer {expected_layer}",
        )
        if metadata_path in referenced_files or tensor_path in referenced_files:
            _raise_staging_validation_error(f"root layer {expected_layer} reuses a staged file locator.")
        referenced_files.update((metadata_path, tensor_path))

        metadata_sha256 = _manifest_sha256(
            layer_entry.get("metadata_sha256"),
            description=f"root layer {expected_layer}.metadata_sha256",
        )
        tensor_sha256 = _manifest_sha256(
            layer_entry.get("tensor_sha256"),
            description=f"root layer {expected_layer}.tensor_sha256",
        )
        actual_metadata_sha256 = _sha256_file(metadata_path)
        if actual_metadata_sha256 != metadata_sha256:
            _raise_staging_validation_error(
                f"root layer {expected_layer} metadata SHA-256 mismatch: "
                f"expected {metadata_sha256}, got {actual_metadata_sha256}."
            )

        layer_manifest = _read_json_object(
            metadata_path,
            description=f"layer {expected_layer} manifest",
        )
        for key, expected in (
            ("schema_version", _MANIFEST_SCHEMA_VERSION),
            ("format", VQ2_DIRECT_TP1_FORMAT),
            ("layer", expected_layer),
            ("tp_size", 1),
            ("tp_rank", 0),
            ("complete", True),
            ("communication", _TP1_COMMUNICATION_CONTRACT),
        ):
            _require_manifest_value(
                layer_manifest,
                key,
                expected,
                description=f"layer {expected_layer} manifest",
            )
        expert_ids = layer_manifest.get("expert_ids")
        if (
            not isinstance(expert_ids, list)
            or any(type(expert_id) is not int or expert_id < 0 for expert_id in expert_ids)
            or expert_ids != sorted(set(expert_ids))
            or len(expert_ids) != expert_count
        ):
            _raise_staging_validation_error(
                f"layer {expected_layer} manifest.expert_ids is inconsistent with expert_count."
            )
        _require_manifest_value(
            layer_manifest,
            "num_expected_experts",
            expert_count,
            description=f"layer {expected_layer} manifest",
        )
        layout = layer_manifest.get("layout")
        if not isinstance(layout, dict) or set(layout) != set(VQ2_MATRIX_KINDS):
            _raise_staging_validation_error(
                f"layer {expected_layer} manifest.layout must contain exactly {list(VQ2_MATRIX_KINDS)}."
            )
        for kind in VQ2_MATRIX_KINDS:
            matrix_layout = layout[kind]
            if not isinstance(matrix_layout, dict) or not isinstance(matrix_layout.get("tensors"), dict):
                _raise_staging_validation_error(
                    f"layer {expected_layer} manifest.layout.{kind}.tensors must be an object."
                )
            tensor_layouts = matrix_layout["tensors"]
            if set(tensor_layouts) != set(_OUTPUT_FIELDS):
                _raise_staging_validation_error(
                    f"layer {expected_layer} manifest.layout.{kind}.tensors has the wrong fields."
                )
            for field in _OUTPUT_FIELDS:
                tensor_layout = tensor_layouts[field]
                expected_axes = list(_OUTPUT_AXES[field])
                if not isinstance(tensor_layout, dict) or tensor_layout.get("axes") != expected_axes:
                    _raise_staging_validation_error(
                        f"layer {expected_layer} {kind}.{field} has the wrong axes contract."
                    )
                shape = tensor_layout.get("shape")
                if (
                    not isinstance(shape, list)
                    or len(shape) != len(expected_axes)
                    or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
                    or shape[0] != expert_count
                ):
                    _raise_staging_validation_error(
                        f"layer {expected_layer} {kind}.{field} has an invalid shape contract."
                    )
                if tensor_layout.get("dtype") != _OUTPUT_DTYPE_NAMES[field]:
                    _raise_staging_validation_error(
                        f"layer {expected_layer} {kind}.{field} has an invalid stable dtype."
                    )

        layer_output = layer_manifest.get("output")
        if not isinstance(layer_output, dict):
            _raise_staging_validation_error(f"layer {expected_layer} manifest.output must be an object.")
        _require_manifest_value(
            layer_output,
            "tensor_file",
            tensor_path.name,
            description=f"layer {expected_layer} manifest.output",
        )
        _require_manifest_value(
            layer_output,
            "tensor_sha256",
            tensor_sha256,
            description=f"layer {expected_layer} manifest.output",
        )
        actual_tensor_sha256 = _sha256_file(tensor_path)
        if actual_tensor_sha256 != tensor_sha256:
            _raise_staging_validation_error(
                f"root layer {expected_layer} tensor SHA-256 mismatch: "
                f"expected {tensor_sha256}, got {actual_tensor_sha256}."
            )
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            expected_tensor_names = {f"{kind}_{field}" for kind in VQ2_MATRIX_KINDS for field in _OUTPUT_FIELDS}
            if set(handle.keys()) != expected_tensor_names:
                _raise_staging_validation_error(f"layer {expected_layer} safetensors file has the wrong tensor names.")
            for kind in VQ2_MATRIX_KINDS:
                for field in _OUTPUT_FIELDS:
                    tensor_slice = handle.get_slice(f"{kind}_{field}")
                    tensor_layout = layer_manifest["layout"][kind]["tensors"][field]
                    if list(tensor_slice.get_shape()) != tensor_layout["shape"]:
                        _raise_staging_validation_error(
                            f"layer {expected_layer} {kind}.{field} header shape disagrees with its manifest."
                        )
                    if tensor_slice.get_dtype() != tensor_layout["dtype"]:
                        _raise_staging_validation_error(
                            f"layer {expected_layer} {kind}.{field} header dtype disagrees with its manifest."
                        )

    if manifest != expected_manifest:
        _raise_staging_validation_error("root manifest differs from the manifest generated in memory.")


def _repack_layer(
    source: Path,
    rank_directory: Path,
    layer_index: int,
    expected_expert_ids: tuple[int, ...],
    source_snapshot: dict[str, Any],
) -> dict[str, Any]:
    summary = inspect_layer_artifact(source, layer_index)
    if summary.expert_ids != expected_expert_ids:
        raise ValueError(
            f"Layer {layer_index} has expert IDs {list(summary.expert_ids)}, expected {list(expected_expert_ids)}."
        )
    specs = load_layer_specs(source, layer_index)
    source_tensors = source / f"experts_vq_layer_{layer_index}.safetensors"

    stacked: dict[str, torch.Tensor] = {}
    representative_specs: dict[str, VQ2MatrixSpec] = {}
    with safe_open(source_tensors, framework="pt", device="cpu") as handle:
        for expert_index, expert_id in enumerate(expected_expert_ids):
            for kind in VQ2_MATRIX_KINDS:
                name = f"{layer_index}.mlp.experts.{expert_id}.{kind}"
                spec = specs[name]
                payload = _source_payload(handle, spec)
                repacked = repack_matrix_tp1(payload, spec)
                if kind not in representative_specs:
                    representative_specs[kind] = spec
                    for field in _OUTPUT_FIELDS:
                        value = repacked[field]
                        stacked[f"{kind}_{field}"] = torch.empty(
                            (len(expected_expert_ids), *value.shape),
                            dtype=value.dtype,
                            device="cpu",
                        )
                for field in _OUTPUT_FIELDS:
                    destination = stacked[f"{kind}_{field}"][expert_index]
                    value = repacked[field]
                    if destination.dtype != value.dtype or destination.shape != value.shape:
                        raise ValueError(
                            f"Layer {layer_index} {kind}.{field} changed geometry at expert {expert_id}: "
                            f"expected dtype={destination.dtype}, shape={tuple(destination.shape)}; "
                            f"got dtype={value.dtype}, shape={tuple(value.shape)}."
                        )
                    destination.copy_(value)
    tensor_name = f"experts_vq_layer_{layer_index}.safetensors"
    tensor_path = rank_directory / tensor_name
    _write_safetensors_atomic(tensor_path, stacked)

    kind_manifests = {
        kind: _matrix_manifest(
            representative_specs[kind],
            {field: stacked[f"{kind}_{field}"] for field in _OUTPUT_FIELDS},
        )
        for kind in VQ2_MATRIX_KINDS
    }
    layer_manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "format": VQ2_DIRECT_TP1_FORMAT,
        "layer": layer_index,
        "tp_size": 1,
        "tp_rank": 0,
        "expert_ids": list(expected_expert_ids),
        "num_expected_experts": len(expected_expert_ids),
        "complete": True,
        "consumer_reference_commit": VQ2_CONSUMER_REFERENCE_COMMIT,
        "source": source_snapshot,
        "output": {
            "tensor_file": tensor_name,
            "tensor_sha256": _sha256_file(tensor_path),
        },
        "layout": kind_manifests,
        "transforms": {
            "permutation": "absorbed_offline_with_argsort_perm",
            "normalization": "activation_side_scale_and_bias_correction",
            "rht": "activation_side_normalized_sylvester_once",
        },
        "communication": dict(_TP1_COMMUNICATION_CONTRACT),
    }
    metadata_name = f"experts_vq_layer_{layer_index}.json"
    metadata_path = rank_directory / metadata_name
    _write_json_atomic(metadata_path, layer_manifest)
    return {
        "layer": layer_index,
        "expert_count": len(expected_expert_ids),
        "tensor_file": f"tp1/rank0/{tensor_name}",
        "tensor_sha256": layer_manifest["output"]["tensor_sha256"],
        "metadata_file": f"tp1/rank0/{metadata_name}",
        "metadata_sha256": _sha256_file(metadata_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Canonical experts_vq directory.")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New artifact root; it must not already exist.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        help="Model config.json; default is the input directory's sibling config.json.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Layer selection such as 0,3-5, or all (default: all).",
    )
    parser.add_argument(
        "--require-reference-identity",
        action="store_true",
        help="Fail unless config and selected source files match the pinned reference SHA256SUMS identity.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    repacker_path = Path(__file__).resolve(strict=True)
    repacker_sha256 = _sha256_file(repacker_path)
    config = (
        args.model_config.expanduser().resolve() if args.model_config is not None else source.parent / "config.json"
    )
    _validate_paths(source, output, config)
    config_sha256 = _sha256_file(config)
    checksum_snapshot, checksum_entries = _capture_checksum_snapshot(config)
    layout = load_model_layout(config)
    available_layers = _discover_layers(source)
    expected_model_layers = tuple(range(layout.num_hidden_layers))
    if args.layers == "all":
        if available_layers != expected_model_layers:
            raise ValueError(
                "--layers=all requires exactly the model layers; "
                f"source has {list(available_layers)}, config expects {list(expected_model_layers)}."
            )
        selected_layers = expected_model_layers
    else:
        selected_layers = _parse_layers(args.layers, available_layers)
    complete = selected_layers == expected_model_layers
    expected_experts = {layer_index: layout.expected_expert_ids(layer_index) for layer_index in selected_layers}
    layer_snapshots = _capture_layer_snapshots(source, selected_layers)
    producer_evidence = _producer_evidence(
        source,
        config,
        config_sha256,
        checksum_snapshot,
        checksum_entries,
        layer_snapshots,
    )
    if args.require_reference_identity and producer_evidence["reference_identity_match"] is not True:
        raise ValueError(
            "Reference identity is required, but config and selected source files do not exactly match "
            "the pinned config/SHA256SUMS identity."
        )

    summaries = [inspect_layer_artifact(source, layer) for layer in selected_layers]
    validate_model_layout(
        summaries,
        layout,
        require_all_layers=complete,
    )
    tensor_payload_bytes, metadata_reserve_bytes = _estimate_output_bytes(
        source,
        selected_layers,
        expected_experts,
    )
    estimated_output_bytes = tensor_payload_bytes + metadata_reserve_bytes
    disk_safety_margin_bytes = max(
        _MINIMUM_DISK_SAFETY_MARGIN_BYTES,
        math.ceil(estimated_output_bytes * _DISK_SAFETY_MARGIN_RATIO),
    )
    required_free_bytes = estimated_output_bytes + disk_safety_margin_bytes
    output.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(output.parent)
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < required_free_bytes:
        raise OSError(
            f"Insufficient free space under {output.parent}: need at least "
            f"{required_free_bytes} bytes (estimated output={estimated_output_bytes}, "
            f"safety margin={disk_safety_margin_bytes}), have {free_bytes}."
        )

    staging = output.parent / f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    rank_directory = staging / "tp1" / "rank0"
    staging_created = False
    try:
        if staging.exists():
            raise FileExistsError(f"Unexpected staging collision: {staging}.")
        staging.mkdir()
        staging_created = True
        _fsync_directory(output.parent)
        (staging / "tp1").mkdir()
        _fsync_directory(staging)
        rank_directory.mkdir()
        _fsync_directory(staging / "tp1")

        layer_manifests = []
        for layer_index in selected_layers:
            layer_manifest = _repack_layer(
                source,
                rank_directory,
                layer_index,
                expected_experts[layer_index],
                layer_snapshots[layer_index],
            )
            layer_manifests.append(layer_manifest)
            print(
                f"layer={layer_index} experts={layer_manifest['expert_count']} format={VQ2_DIRECT_TP1_FORMAT}",
                flush=True,
            )
        _verify_source_snapshot(
            source,
            config,
            config_sha256,
            checksum_snapshot,
            layer_snapshots,
        )
        if _sha256_file(repacker_path) != repacker_sha256:
            raise RuntimeError(f"Repacker changed during conversion: {repacker_path}.")
        root_manifest = {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "format": VQ2_DIRECT_TP1_FORMAT,
            "tp_size": 1,
            "tp_ranks": [0],
            "consumer_reference_commit": VQ2_CONSUMER_REFERENCE_COMMIT,
            "repacker": {
                "tool": "tools/repack_vq2a8_tp1.py",
                "contract_revision": _REPACK_CONTRACT_REVISION,
                "tool_sha256": repacker_sha256,
            },
            "producer_evidence": producer_evidence,
            "source": {
                "model_config_file": config.name,
                "model_config_sha256": config_sha256,
                "checksum_manifest": checksum_snapshot,
            },
            "output": {
                "artifact_root": ".",
                "path_semantics": "relative_to_manifest",
            },
            "build_provenance": {
                "informational": True,
                "absolute_paths": {
                    "canonical_experts": str(source),
                    "model_config": str(config),
                    "requested_output": str(output),
                },
            },
            "model_layout": {
                "num_hidden_layers": layout.num_hidden_layers,
                "num_hash_layers": layout.num_hash_layers,
                "num_routed_experts": layout.num_routed_experts,
                "hidden_size": layout.hidden_size,
                "moe_intermediate_size": layout.moe_intermediate_size,
            },
            "packing": dict(_PACKING_CONTRACT),
            "communication": dict(_TP1_COMMUNICATION_CONTRACT),
            "storage": {
                "estimated_tensor_payload_bytes": tensor_payload_bytes,
                "metadata_and_header_reserve_bytes": metadata_reserve_bytes,
                "disk_safety_margin_bytes": disk_safety_margin_bytes,
            },
            "layers_selected": list(selected_layers),
            "layers_expected": layout.num_hidden_layers,
            "complete": complete,
            "layers": layer_manifests,
        }
        _write_json_atomic(staging / "manifest.json", root_manifest)
        _fsync_directory(staging)
        _validate_staging_artifact(staging, root_manifest)
        if _sha256_file(repacker_path) != repacker_sha256:
            raise RuntimeError(f"Repacker changed during conversion: {repacker_path}.")
        if output.exists():
            raise FileExistsError(f"Output path appeared during repack; refusing to overwrite it: {output}.")
        os.replace(staging, output)
        staging_created = False
        _fsync_directory(output.parent)
    except BaseException:
        if staging_created:
            shutil.rmtree(staging, ignore_errors=True)
            _fsync_directory(output.parent)
        raise

    print(
        json.dumps(
            {
                "format": VQ2_DIRECT_TP1_FORMAT,
                "output": str(output),
                "layers": list(selected_layers),
                "complete": complete,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
