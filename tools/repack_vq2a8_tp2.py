#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only canonical experts_vq -> TP2 packed-zN artifact generation.

This generates expert weights, NOT a runnable TP2 model. The current V3
TP1 loader must not consume this format. No NPU, vLLM, or compiled library
is imported. See docs/vq2a8_tp2_offline.md for the communication contract.
"""

from __future__ import annotations

# Direct execution must not let tools/bisect shadow the standard library.
# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import hashlib
import importlib
import importlib.machinery
import json
import re
import shutil
import tempfile
import time
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _cpu_modules() -> tuple[Any, Any]:
    # Load real CPU helpers under an isolated package. Importing the public
    # vllm_ascend package would initialize its vLLM logger/plugin dependencies.
    # Do not stub vllm, torch_npu, or any device/kernel implementation.
    directory = Path(__file__).resolve().parents[1] / "vllm_ascend" / "quantization"
    package_name = "_vq2a8_offline_cpu"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(directory)]
        package.__spec__ = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[package_name] = package
    elif list(sys.modules[package_name].__path__) != [str(directory)]:
        raise RuntimeError("Offline CPU helper package is already bound to another repository.")
    return (
        importlib.import_module(f"{package_name}.vq2a8_artifact"),
        importlib.import_module(f"{package_name}.vq2a8_tp2_layout"),
    )


def _progress(stage: str, **fields: Any) -> None:
    print(f"VQ2_TP2_STAGE={stage} " + " ".join(f"{key}={value}" for key, value in fields.items()), flush=True)


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="original canonical experts_vq directory")
    parser.add_argument("--output", required=True, type=Path, help="new artifact directory; must not exist")
    parser.add_argument("--model-config", type=Path, help="default: INPUT/../config.json")
    parser.add_argument("--layers", default="all", help="all, or a subset such as 3 or 0,3-5 (incomplete artifact)")
    parser.add_argument("--experts-per-shard", type=_positive, default=32, help="bounded RAM/output shard size")
    parser.add_argument("--threads", type=_positive, default=4, help="CPU PyTorch threads (no NPU used)")
    parser.add_argument(
        "--dry-run", action="store_true", help="check headers/geometry and estimate bytes; write nothing"
    )
    return parser.parse_args(argv)


def _discover_layers(source: Path) -> tuple[int, ...]:
    suffixes: dict[str, set[int]] = {"json": set(), "safetensors": set()}
    for path in source.iterdir():
        match = re.fullmatch(r"experts_vq_layer_(\d+)\.(json|safetensors)", path.name)
        if match is None:
            continue
        layer = int(match[1])
        if str(layer) != match[1] or not path.is_file() or path.is_symlink():
            raise ValueError(f"Invalid canonical layer file (or symlink): {path}")
        suffixes[match[2]].add(layer)
    if not suffixes["json"] or suffixes["json"] != suffixes["safetensors"]:
        raise ValueError("Input must contain paired experts_vq_layer_N.json/.safetensors files.")
    return tuple(sorted(suffixes["json"]))


def _select_layers(selection: str, available: tuple[int, ...]) -> tuple[int, ...]:
    if selection == "all":
        return available
    selected: set[int] = set()
    for part in selection.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part.strip())
        if match is None:
            raise ValueError(f"Invalid --layers item: {part!r}")
        start, end = int(match[1]), int(match[2] or match[1])
        if end < start or end > max(available):
            raise ValueError(f"Invalid or missing layer range: {part!r}")
        selected.update(range(start, end + 1))
    if not selected or selected - set(available):
        raise ValueError(f"Selected layers are absent from input: {sorted(selected - set(available))}")
    return tuple(sorted(selected))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_file(path: Path) -> None:
    # Windows _commit/fsync requires a writable file descriptor. This helper
    # only handles newly-written output files, never the canonical source.
    with path.open("rb+") as stream:
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _publish_directory(staging: Path, output: Path) -> None:
    if os.name == "nt":
        # Windows rename already refuses an existing destination directory.
        os.rename(staging, output)
        return
    if sys.platform != "linux":
        raise OSError("Atomic no-replace publication is supported on Linux and Windows only.")
    import ctypes
    import errno

    # POSIX rename may overwrite a concurrently-created EMPTY directory.
    # Use Linux RENAME_NOREPLACE instead of an exists()/rename() race; fail
    # closed if the filesystem/libc does not provide this primitive.
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        rename = libc.renameat2
    except AttributeError as error:
        raise OSError(errno.ENOTSUP, "libc.renameat2 is required for no-replace publication") from error
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    at_fdcwd, rename_noreplace = -100, 1
    if rename(at_fdcwd, os.fsencode(staging), at_fdcwd, os.fsencode(output), rename_noreplace) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(output))


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _matrix_tensor_metadata(expert: int, kind: str, tensors: dict[str, Any]) -> dict[str, Any]:
    dtype_names = {"torch.uint8": "U8", "torch.int64": "I64", "torch.float32": "F32", "torch.int8": "I8"}
    return {
        field: {"key": f"{expert}.{kind}.{field}", "dtype": dtype_names[str(tensor.dtype)], "shape": list(tensor.shape)}
        for field, tensor in tensors.items()
    }


def _write_shard(
    staging: Path,
    layer: int,
    rank: int,
    experts: tuple[int, ...],
    tensors: dict[str, Any],
    matrices: list[dict[str, Any]],
) -> dict[str, Any]:
    # Lazy dependencies keep --help usable even before installing CPU torch.
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    directory = staging / "tp2" / f"rank{rank}" / f"layer_{layer:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"experts_{experts[0]:04d}_{experts[-1] + 1:04d}.safetensors"
    temporary = path.with_suffix(".safetensors.partial")
    save_file(tensors, temporary)
    _sync_file(temporary)
    # Validate actual serialized bytes, including I64 activation_order. The
    # canonical input header parser intentionally supports different dtypes.
    with safe_open(temporary, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(tensors):
            raise RuntimeError(f"Saved shard has wrong tensor keys: {temporary}")
        for key, expected in tensors.items():
            actual = handle.get_tensor(key)
            if actual.dtype != expected.dtype or actual.shape != expected.shape or not torch.equal(actual, expected):
                raise RuntimeError(f"Saved tensor verification failed: {temporary}:{key}")
    os.replace(temporary, path)
    metadata_path = path.with_suffix(".json")
    sha256 = _sha256(path)
    _write_json(metadata_path, {"layer": layer, "rank": rank, "sha256": sha256, "matrices": matrices})
    return {
        "layer": layer,
        "rank": rank,
        "expert_ids": list(experts),
        "file": path.relative_to(staging).as_posix(),
        "metadata_file": metadata_path.relative_to(staging).as_posix(),
        "sha256": sha256,
        "metadata_sha256": _sha256(metadata_path),
        "payload_bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
    }


def _payload_upper_bound(spec: Any) -> int:
    # With canonical K256 groups, each rank holds at most the original number
    # of K256 blocks. Down padding may therefore retain the FULL original K.
    rows = spec.rows // 2 if spec.kind == "gate_up" else spec.rows
    columns = spec.columns
    return rows * columns // 4 + columns // 256 * rows + columns * (8 + 4 + 4 + 1)


def _source_snapshot(paths: tuple[Path, ...]) -> list[dict[str, str]]:
    return [{"file": path.name, "sha256": _sha256(path)} for path in paths]


def _convert_layer(
    source: Path,
    staging: Path,
    layer: int,
    expert_ids: tuple[int, ...],
    shard_size: int,
    artifact: Any,
    converter: Any,
    planned_metadata_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from safetensors import safe_open

    started = time.monotonic()
    paths = artifact.layer_artifact_paths(source, layer)
    _progress("source_hash", layer=layer)
    snapshot = _source_snapshot(paths)
    if snapshot[0]["sha256"] != planned_metadata_sha256:
        raise RuntimeError(f"Source layer {layer} metadata changed after planning; refusing to publish.")
    # Re-read metadata after hashing; changes during conversion fail below.
    specs = artifact.load_layer_specs(source, layer)
    entries: list[dict[str, Any]] = []
    with safe_open(paths[1], framework="pt", device="cpu") as handle:
        for offset in range(0, len(expert_ids), shard_size):
            experts = expert_ids[offset : offset + shard_size]
            shard_tensors: list[dict[str, Any]] = [{}, {}]
            shard_matrices: list[list[dict[str, Any]]] = [[], []]
            for position, expert in enumerate(experts, offset + 1):
                for kind in artifact.VQ2_MATRIX_KINDS:
                    spec = specs[f"{layer}.mlp.experts.{expert}.{kind}"]
                    original = {
                        field: handle.get_tensor(f"{spec.name}.{field}") for field in spec.expected_tensor_headers()
                    }
                    for rank in (0, 1):
                        tensors, metadata = converter.repack_matrix_tp2(original, spec, rank)
                        tensor_metadata = _matrix_tensor_metadata(expert, kind, tensors)
                        shard_matrices[rank].append(
                            {
                                **metadata,
                                "name": spec.name,
                                "expert_id": expert,
                                "kind": kind,
                                "tensors": tensor_metadata,
                            }
                        )
                        shard_tensors[rank].update({tensor_metadata[field]["key"]: t for field, t in tensors.items()})
                    del original
                if position % 8 == 0 or position == len(expert_ids):
                    _progress(
                        "convert",
                        layer=layer,
                        experts=f"{position}/{len(expert_ids)}",
                        elapsed_s=f"{time.monotonic() - started:.1f}",
                    )
            for rank in (0, 1):
                _progress("write_verify", layer=layer, rank=rank, experts=f"{experts[0]}..{experts[-1]}")
                entries.append(_write_shard(staging, layer, rank, experts, shard_tensors[rank], shard_matrices[rank]))
                shard_tensors[rank].clear()
    _progress("source_recheck", layer=layer)
    if snapshot != _source_snapshot(paths):
        raise RuntimeError(f"Source layer {layer} changed during conversion; refusing to publish.")
    _progress("layer_done", layer=layer, elapsed_s=f"{time.monotonic() - started:.1f}")
    return entries, {"layer": layer, "files": snapshot}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Plan, convert and publish a source-bound artifact without touching runtime.

    Keep validation, bounded conversion, and the final source/producer checks
    inside one publication transaction; a partial tree is never a model.
    """
    source = args.input.resolve()
    # Refuse both live and broken output symlinks before resolve follows them.
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {args.output}")
    output = args.output.resolve()
    config = (args.model_config or source.parent / "config.json").resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Input and output must be separate trees; neither may contain the other.")
    if not source.is_dir() or not config.is_file():
        raise FileNotFoundError(f"Canonical input or model config missing: input={source}, config={config}")
    artifact, converter = _cpu_modules()
    import torch

    torch.set_num_threads(args.threads)
    started = time.monotonic()
    selected = _select_layers(args.layers, _discover_layers(source))
    config_sha256 = _sha256(config)
    layout = artifact.load_model_layout(config)
    summaries = []
    planned_metadata = {}
    bytes_per_rank_upper = 0
    matrices = 0
    for layer in selected:
        _progress("inspect", layer=layer, selected=len(selected))
        metadata_path = artifact.layer_artifact_paths(source, layer)[0]
        planned_metadata[layer] = _sha256(metadata_path)
        summaries.append(artifact.inspect_layer_artifact(source, layer))
        for spec in artifact.load_layer_specs(source, layer).values():
            converter.validate_tp2_spec(spec)
            bytes_per_rank_upper += _payload_upper_bound(spec)
            matrices += 1
        if _sha256(metadata_path) != planned_metadata[layer]:
            raise RuntimeError(f"Source layer {layer} metadata changed during planning.")
    complete = selected == tuple(range(layout.num_hidden_layers))
    artifact.validate_model_layout(summaries, layout, require_all_layers=args.layers == "all")
    plan = {
        "format": converter.VQ2_TP2_ZN_FORMAT,
        "tp_size": 2,
        "tp_ranks": [0, 1],
        "runtime_compatible": False,
        "complete": complete,
        "layers_selected": list(selected),
        "layers_expected": layout.num_hidden_layers,
        "per_rank_payload_upper_bytes": bytes_per_rank_upper,
        "per_rank_payload_upper_gib": bytes_per_rank_upper / 2**30,
        "total_payload_upper_bytes": bytes_per_rank_upper * 2,
        "scope": "expert_payload_only_excludes_root_weights_kv_workspace_graph_and_runtime",
        "dry_run": args.dry_run,
    }
    print("VQ2_TP2_PLAN=" + json.dumps(plan, sort_keys=True), flush=True)
    if args.dry_run:
        _progress("dry_run_done", PAYLOAD_VALUES_VERIFIED=False, RUNTIME_COMPATIBLE=False)
        return plan

    producer_paths = (
        Path(__file__).resolve(),
        Path(artifact.__file__).resolve(),
        Path(converter.__file__).resolve(),
        Path(converter.__file__).with_name("vq2a8_repack.py").resolve(),
    )
    producer = _source_snapshot(producer_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Allow space for headers/JSON and filesystem overhead, not just tensors.
    disk_required = 2 * bytes_per_rank_upper + max(64 * 2**20, bytes_per_rank_upper // 10, matrices * 16384)
    free_disk = shutil.disk_usage(output.parent).free
    if free_disk < disk_required:
        raise ValueError(f"Insufficient output disk space: free={free_disk}, required_upper={disk_required} bytes")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)).resolve()
    published = False
    _progress("convert_start", output=output, staging=staging, RUNTIME_COMPATIBLE=False)
    try:
        # Discover unsupported publication primitives/filesystems before doing
        # a potentially long conversion, using only our private empty tree.
        probe_source, probe_output = staging / ".publish-source", staging / ".publish-target"
        probe_source.mkdir()
        _publish_directory(probe_source, probe_output)
        probe_output.rmdir()
        shards: list[dict[str, Any]] = []
        source_layers: list[dict[str, Any]] = []
        for summary in summaries:
            layer_shards, snapshot = _convert_layer(
                source,
                staging,
                summary.layer_index,
                summary.expert_ids,
                args.experts_per_shard,
                artifact,
                converter,
                planned_metadata[summary.layer_index],
            )
            shards.extend(layer_shards)
            source_layers.append(snapshot)
        # A completed early layer may change while later layers are converted.
        # Recheck the full selected source snapshot before final publication.
        for snapshot in source_layers:
            layer = snapshot["layer"]
            _progress("final_source_recheck", layer=layer)
            if _source_snapshot(artifact.layer_artifact_paths(source, layer)) != snapshot["files"]:
                raise RuntimeError(f"Source layer {layer} changed before final publication.")
        if _sha256(config) != config_sha256 or _source_snapshot(producer_paths) != producer:
            raise RuntimeError("Model config or conversion code changed during conversion; refusing to publish.")
        actual_bytes = [sum(shard["payload_bytes"] for shard in shards if shard["rank"] == rank) for rank in (0, 1)]
        manifest = {
            **plan,
            "schema_version": 1,
            "dry_run": False,
            "model_layout": asdict(layout),
            "source": {"config": {"file": config.name, "sha256": config_sha256}, "layers": source_layers},
            "producer": {"tool": "tools/repack_vq2a8_tp2.py", "files": producer},
            "per_rank_payload_bytes": actual_bytes,
            "shards": shards,
            "tensor_values_verified": True,
            "validation_scope": "canonical_checks_layout_roundtrip_serialization_not_device_or_model_accuracy",
            "tp1_a8_bitwise_equivalent": False,
            "communication": {
                "gate_up": "column_parallel_separate_gate_and_up_slices_concatenated_per_rank",
                "down": "row_parallel_contiguous_physical_input_slice_sum_partials_across_tp_ranks",
                "activation_quantization": "per_rank_per_row_amax_after_local_RHT128_and_weight_scale",
                "bias_correction": "local_input_contribution_only_before_down_partial_sum",
                "routing": "same_token_expert_assignments_on_both_ranks_not_expert_parallel",
            },
            "runtime_requirements": [
                "A new TP2 loader: current TP1 serving rejects this format.",
                "Native local-N support (gate_up N2048 for this model).",
                "Prepare local physical activation; append zero dummy columns, then gather by activation_order.",
                "Per-expert variable packed K handling where padding differs; persistent workspace budgeting.",
                "TP routing and down partial SUM, root model TP and shared-expert ownership integration.",
                "Device numerical, model quality, peak memory and performance validation.",
            ],
        }
        _write_json(staging / "manifest.json", manifest)
        _sync_directory(staging)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output appeared during conversion; refusing to overwrite: {output}")
        _publish_directory(staging, output)
        published = True
        _sync_directory(output.parent)
    finally:
        if not published and staging.exists():
            # Only delete the exact newly-created private staging directory.
            if staging.parent != output.parent or not staging.name.startswith(f".{output.name}.partial-"):
                raise RuntimeError(f"Refusing to clean an unexpected staging path: {staging}")
            shutil.rmtree(staging)
            _progress("partial_output_removed", path=staging, SOURCE_WRITES_BY_CONVERTER=False)
    _progress(
        "done",
        OUTPUT=output,
        RUNTIME_COMPATIBLE=False,
        rank0_gib=f"{actual_bytes[0] / 2**30:.3f}",
        rank1_gib=f"{actual_bytes[1] / 2**30:.3f}",
        elapsed_s=f"{time.monotonic() - started:.1f}",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"VQ2_TP2_ERROR={error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
