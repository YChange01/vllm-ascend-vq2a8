#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only direct-TP1 -> reusable V4+v2 compressed expert layout.

The input is an existing experts_vq_ascend_v2 artifact, NOT canonical experts_vq
or the old V3 format. Only static layout conversion is persisted; there are no
device pointers, dense expert expansion, NPU imports or native builds.
"""

from __future__ import annotations

# Avoid tools/bisect.py shadowing the standard library on direct execution.
# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import importlib
import importlib.machinery
import json
import math
import shutil
import stat
import tempfile
import time
import types
import uuid
from dataclasses import asdict
from pathlib import Path

from tools.repack_vq2a8_tp2 import _publish_directory, _sha256, _sync_directory, _sync_file, _write_json


def _cpu_modules():
    """Import CPU siblings without initializing public vLLM/plugin packages."""
    directory = Path(__file__).resolve().parents[1] / "vllm_ascend" / "quantization"
    package_name = "_vq2a8_v4_v2_prepack_cpu"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(directory)]
        package.__spec__ = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[package_name] = package
    elif list(sys.modules[package_name].__path__) != [str(directory)]:
        raise RuntimeError("CPU prepack helper package is already bound to another repository.")
    return tuple(
        importlib.import_module(f"{package_name}.{module}")
        for module in ("vq2a8_runtime", "vq2a8_v4_v2_layout", "vq2a8_v4_v2_prepacked")
    )


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="existing direct-TP1 artifact directory")
    parser.add_argument("--output", required=True, type=Path, help="new separate directory; must not exist")
    parser.add_argument("--model-config", type=Path, help="default: INPUT/../config.json")
    parser.add_argument("--experts-per-shard", type=_positive, default=16, help="bounded CPU memory, default 16")
    parser.add_argument("--threads", type=_positive, default=4, help="CPU PyTorch threads; default 4")
    parser.add_argument("--plan-only", action="store_true", help="validate headers and estimate bytes; write nothing")
    return parser.parse_args(argv)


def _progress(stage, **fields):
    print(
        "V4_V2_PREPACK_STAGE=" + stage + " " + " ".join(f"{key}={value}" for key, value in fields.items()), flush=True
    )


def _no_symlinks(path):
    """Reject symlinks/junctions before resolution, including ancestor aliases."""
    absolute = Path(os.path.abspath(path.expanduser()))
    for part in (absolute, *absolute.parents):
        if part.is_symlink():
            raise ValueError(f"Prepack paths must not contain symlinks: {part}")
        try:
            attributes = getattr(part.lstat(), "st_file_attributes", 0)
        except FileNotFoundError:
            continue
        if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ValueError(f"Prepack paths must not contain filesystem junctions/reparse points: {part}")
    return absolute


def _paths(args):
    source_path = _no_symlinks(args.input)
    output_path = _no_symlinks(args.output)
    if output_path.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_path}")
    source = source_path.resolve(strict=True)
    output = output_path.resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"Direct-TP1 artifact must be a directory: {source}")
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Input and output must be separate trees; neither may contain the other.")
    config = _no_symlinks(args.model_config or source.parent / "config.json").resolve(strict=True)
    if not config.is_file():
        raise FileNotFoundError(f"Model config must be a regular file: {config}")
    if config == output or config.is_relative_to(output):
        raise ValueError("Output must not contain the source model config.")
    # The strict TP1 reader resolves each manifest locator. Reject any aliases
    # first so resolved files cannot hide a symlinked source subtree.
    for candidate in source.rglob("*"):
        _no_symlinks(candidate)
    return source, output, config


def _snapshot(paths):
    return {str(path): _sha256(path) for path in paths}


def _assert_unchanged(snapshot):
    for filename, expected in snapshot.items():
        path = _no_symlinks(Path(filename))
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Source or conversion code changed; refusing to publish: {path}")


def _invalidate_staging_manifest(staging, expected_identity):
    """Recoverably remove the ready marker only from our unpublished tree."""
    _no_symlinks(staging)
    actual = staging.stat()
    if (actual.st_dev, actual.st_ino) != expected_identity or not staging.is_dir():
        raise RuntimeError("Staging directory identity changed; refusing to modify another tree.")
    manifest = staging / "manifest.json"
    if not manifest.exists():
        return
    _no_symlinks(manifest)
    failed = staging / "failed_manifest.json"
    while True:
        try:
            # The same atomic no-replace rename primitive handles files too.
            # Do not overwrite an independently-created diagnostic file.
            _publish_directory(manifest, failed)
            return
        except FileExistsError:
            failed = staging / f"failed_manifest-{uuid.uuid4().hex}.json"


def _write_shard(staging, layer_index, expert_ids, tensors):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    directory = staging / f"layer_{layer_index:03d}"
    directory.mkdir(exist_ok=True)
    path = directory / f"experts_{expert_ids[0]:03d}_{expert_ids[-1]:03d}.safetensors"
    partial = path.with_suffix(".safetensors.partial")
    save_file(tensors, partial)
    _sync_file(partial)
    with safe_open(partial, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(tensors):
            raise RuntimeError(f"Serialized shard keys differ: {partial}")
        for key, expected in tensors.items():
            actual = handle.get_tensor(key)
            # Byte comparison preserves signed zero and every FP8/nibble bit;
            # numerical float equality alone is insufficient for this claim.
            if (
                actual.dtype != expected.dtype
                or actual.shape != expected.shape
                or not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            ):
                raise RuntimeError(f"Serialized tensor raw bytes differ: {partial}:{key}")
    os.rename(partial, path)
    _sync_directory(directory)
    return {"file": path.relative_to(staging).as_posix(), "sha256": _sha256(path), "expert_ids": list(expert_ids)}


def _convert_layer(layer, staging, shard_size, runtime, converter, reader):
    import torch
    from safetensors import safe_open

    started = time.monotonic()
    shards = []
    with safe_open(layer.tensor_path, framework="pt", device="cpu") as handle:
        for offset in range(0, len(layer.expert_ids), shard_size):
            expert_ids = layer.expert_ids[offset : offset + shard_size]
            tensors = {
                f"{kind}_{field}": torch.empty((len(expert_ids), *shape), dtype=dtype)
                for kind in ("gate_up", "down")
                for field, dtype, shape in zip(
                    converter.V4_V2_FIELDS,
                    converter.V4_V2_DTYPES,
                    reader.prepacked_projection_shapes(layer.specs[kind]),
                )
            }
            for local, expert in enumerate(expert_ids):
                for kind in ("gate_up", "down"):
                    spec = layer.spec_for(expert, kind)
                    payload = {
                        field: handle.get_slice(f"{kind}_{field}")[offset + local].contiguous()
                        for field in runtime.VQ2_TP1_FIELDS
                    }
                    runtime.validate_repacked_matrix(payload, spec)
                    converted = converter.convert_expert_payload(payload, spec)
                    for field in converter.V4_V2_FIELDS:
                        tensors[f"{kind}_{field}"][local].copy_(converted[field])
                    del payload, converted
            shards.append(_write_shard(staging, layer.layer_index, expert_ids, tensors))
            del tensors
            _progress(
                "shard_verified",
                layer=layer.layer_index,
                experts=f"{offset + len(expert_ids)}/{len(layer.expert_ids)}",
                elapsed_s=f"{time.monotonic() - started:.3f}",
            )
    return {
        "layer_index": layer.layer_index,
        "expert_ids": list(layer.expert_ids),
        "specs": {kind: reader.serialize_spec(spec) for kind, spec in layer.specs.items()},
        "source_tensor_shapes": {key: list(shape) for key, shape in layer.tensor_shapes.items()},
        "source_tensor_sha256": layer.tensor_sha256,
        "source_metadata_sha256": layer.metadata_sha256,
        "shards": shards,
    }


def run(args):
    source, output, config = _paths(args)
    runtime, converter, reader = _cpu_modules()
    import torch

    if type(args.threads) is not int or args.threads < 1 or type(args.experts_per_shard) is not int:
        raise ValueError("CPU threads and experts-per-shard must be positive integers.")
    if not 1 <= args.experts_per_shard <= converter.V4_V2_MAX_EXPERTS:
        raise ValueError("experts-per-shard must be between 1 and 256.")
    torch.set_num_threads(args.threads)
    initial = _snapshot((source / "manifest.json", config))
    _progress("source_validation", input=source, payload_hashes=not args.plan_only)
    artifact = runtime.open_vq2a8_tp1_artifact(
        source, config, require_reference_identity=True, verify_tensor_hashes=not args.plan_only
    )
    payload_bytes = 0
    for layer in artifact.layers.values():
        if len(layer.expert_ids) > converter.V4_V2_MAX_EXPERTS:
            raise ValueError("V4 v2 supports at most 256 experts per layer.")
        for spec in layer.specs.values():
            converter._geometry(spec.rows, spec.columns)
            payload_bytes += len(layer.expert_ids) * sum(
                math.prod(shape) * dtype.itemsize
                for shape, dtype in zip(reader.prepacked_projection_shapes(spec), converter.V4_V2_DTYPES)
            )
    plan = {
        "input": str(source),
        "output": str(output),
        "layers": len(artifact.layers),
        "experts_per_shard": args.experts_per_shard,
        "payload_bytes": payload_bytes,
        "payload_gib": payload_bytes / 2**30,
        "plan_only": args.plan_only,
        "payload_values_verified": False,
        "scope": "expert_compressed_layout_only_not_device_or_model_acceptance",
    }
    _assert_unchanged(initial)
    print("V4_V2_PREPACK_PLAN=" + json.dumps(plan, sort_keys=True), flush=True)
    if args.plan_only:
        return plan
    # Runtime validation just hashed payloads and metadata. Bind the exact
    # bytes it checked; re-hash before publication to reject mid-run edits.
    source_snapshot = {
        **initial,
        **{str(layer.tensor_path): layer.tensor_sha256 for layer in artifact.layers.values()},
        **{str(layer.metadata_path): layer.metadata_sha256 for layer in artifact.layers.values()},
    }
    producer_paths = (
        Path(__file__).resolve(),
        Path(converter.__file__).resolve(),
        Path(runtime.__file__).resolve(),
        Path(reader.__file__).resolve(),
        Path(runtime.__file__).with_name("vq2a8_repack.py"),
        Path(runtime.__file__).with_name("vq2a8_artifact.py"),
        Path(_write_json.__code__.co_filename).resolve(),
    )
    producer_snapshot = _snapshot(producer_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    _no_symlinks(output)
    free_bytes = shutil.disk_usage(output.parent).free
    required_bytes = payload_bytes + max(64 * 2**20, payload_bytes // 20)
    if free_bytes < required_bytes:
        raise ValueError(f"Insufficient output disk space: free={free_bytes}, required={required_bytes} bytes.")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)).resolve()
    staging_stat = staging.stat()
    staging_identity = (staging_stat.st_dev, staging_stat.st_ino)
    published = False
    _progress("convert", staging=staging)
    try:
        layers = [
            _convert_layer(layer, staging, args.experts_per_shard, runtime, converter, reader)
            for layer in artifact.layers.values()
        ]
        manifest = {
            "format": reader.V4_V2_PREPACKED_FORMAT,
            "schema_version": reader.V4_V2_PREPACKED_SCHEMA_VERSION,
            "layout_version": reader.V4_V2_PREPACKED_LAYOUT_VERSION,
            "complete": True,
            "model_config_sha256": initial[str(config)],
            "model_layout": asdict(artifact.model_layout),
            "source": {
                "format": runtime.VQ2_DIRECT_TP1_FORMAT,
                "manifest_sha256": initial[str(source / "manifest.json")],
            },
            "layers": layers,
            "producer": {
                "tool": "tools/prepack_vq2a8_v4_v2.py",
                "files": [{"file": path.name, "sha256": producer_snapshot[str(path)]} for path in producer_paths],
            },
            "payload_bytes": payload_bytes,
            "tensor_bytes_verified": True,
            "validation_scope": "direct_tp1_validation_current_conversion_serialization_not_device_or_model_accuracy",
        }
        # This is the completion marker: it is written after every shard was
        # fsynced and byte-verified. The final output remains absent until the
        # strict reader accepts the complete staging tree and atomic publish.
        _write_json(staging / "manifest.json", manifest)
        reader.open_vq2a8_v4_v2_prepacked_artifact(staging, config, verify_tensor_hashes=True)
        _progress("source_recheck")
        _assert_unchanged(source_snapshot)
        _assert_unchanged(producer_snapshot)
        _no_symlinks(output)
        _publish_directory(staging, output)
        published = True
        _sync_directory(output.parent)
    except BaseException as error:
        # Preserve our partial tree for diagnosis; never remove source data
        # or an independently-created output directory on any failure path.
        if not published:
            try:
                _invalidate_staging_manifest(staging, staging_identity)
            except Exception as cleanup_error:
                error.add_note(f"Could not quarantine staging completion marker: {cleanup_error}")
            _progress("failed_partial_preserved", staging=staging)
        else:
            # Publication succeeded but parent-directory fsync failed. Never
            # resolve or mutate a subsequently-created path at old staging.
            _progress("published_output_preserved", output=output)
        raise
    _progress("complete", output=output, payload_bytes=payload_bytes)
    return manifest


def main(argv=None):
    try:
        run(parse_args(argv))
    except Exception as error:
        print(f"V4_V2_PREPACK=FAIL ERROR={type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    print("V4_V2_PREPACK=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
