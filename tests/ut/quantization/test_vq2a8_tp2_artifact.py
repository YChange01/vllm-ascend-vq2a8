# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU artifact-reader contracts, including real subprocess import isolation."""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tools.repack_vq2a8_tp2 import _write_shard
from vllm_ascend.quantization import vq2a8_tp2_runtime as reader
from vllm_ascend.quantization.vq2a8_artifact import VQ2MatrixSpec, VQ2ModelLayout
from vllm_ascend.quantization.vq2a8_tp2_layout import repack_matrix_tp2

REPO = Path(__file__).resolve().parents[3]
DTYPES = {
    "packed_zn": "U8",
    "pair_lut": "U8",
    "activation_order": "I64",
    "weight_scale": "F32",
    "weight_bias": "F32",
    "rht_sign": "I8",
}
COMMUNICATION = {
    "gate_up": "column_parallel_separate_gate_and_up_slices_concatenated_per_rank",
    "down": "row_parallel_contiguous_physical_input_slice_sum_partials_across_tp_ranks",
    "activation_quantization": "per_rank_per_row_amax_after_local_RHT128_and_weight_scale",
    "bias_correction": "local_input_contribution_only_before_down_partial_sum",
    "routing": "same_token_expert_assignments_on_both_ranks_not_expert_parallel",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def canonical(layer, expert, kind):
    n, k = (1024, 512) if kind == "gate_up" else (512, 512)
    spec = VQ2MatrixSpec(
        name=f"{layer}.mlp.experts.{expert}.{kind}",
        layer_index=layer,
        expert_id=expert,
        kind=kind,
        rows=n,
        columns=k,
        row_tiles=n // 32,
        column_tiles=k // 256,
        row_group_size=32,
        group_size=256,
        num_vectors=n * k // 2,
        num_elements=n * k,
        original_shape=(n, k),
        norm_dimension=0,
        enable_permutation=True,
        enable_normalization=True,
        enable_rht=True,
        rht_block_size=128,
        rht_true_columns=k,
    )
    books = (torch.arange(k // 256 * n // 32 * 32) % 120).to(torch.uint8).reshape(k // 256, n // 32, 16, 2)
    books[..., 0, 0] = 128  # Preserve signed zero without rejecting finite LUT bytes.
    payload = {
        "packed_indices": torch.full((n * k // 16,), 0x11111111, dtype=torch.int32),
        "codebooks": books.view(torch.float8_e4m3fn),
        "perm": torch.roll(torch.arange(k, dtype=torch.int32), 17 if expert else 0),
        "weight_scale": torch.linspace(-0.5, 1.5, k),
        "weight_bias": torch.linspace(-0.25, 0.25, k),
        "rht_sign": torch.where(torch.arange(k) % 2 == 0, 1, -1).to(torch.int8),
    }
    return payload, spec


@pytest.fixture(scope="module")
def artifact_template(tmp_path_factory):
    base = tmp_path_factory.mktemp("tp2-artifact-template")
    root, config = base / "tp2", base / "config.json"
    root.mkdir()
    config_value = {
        "num_hidden_layers": 2,
        "num_hash_layers": 1,
        "n_routed_experts": 2,
        "hidden_size": 512,
        "moe_intermediate_size": 512,
        "quantization_config": {"quant_method": "vq2a8"},
    }
    write_json(config, config_value)
    layout = VQ2ModelLayout.from_dict(config_value)
    shards = []
    for layer in range(2):
        experts = layout.expected_expert_ids(layer)
        for rank in range(2):
            tensors, matrices = {}, []
            for expert in experts:
                for kind in ("gate_up", "down"):
                    original, spec = canonical(layer, expert, kind)
                    payload, metadata = repack_matrix_tp2(original, spec, rank)
                    records = {
                        field: {"key": f"{expert}.{kind}.{field}", "dtype": DTYPES[field], "shape": list(value.shape)}
                        for field, value in payload.items()
                    }
                    matrices.append({**metadata, "name": spec.name, "expert_id": expert, "tensors": records})
                    tensors.update({records[field]["key"]: value for field, value in payload.items()})
            shards.append(_write_shard(root, layer, rank, experts, tensors, matrices))
    manifest = {
        "schema_version": 1,
        "format": "vq2a8_zn_tp2_v1",
        "tp_size": 2,
        "tp_ranks": [0, 1],
        "runtime_compatible": False,
        "complete": True,
        "dry_run": False,
        "layers_selected": [0, 1],
        "layers_expected": 2,
        "model_layout": asdict(layout),
        "source": {"config": {"file": "config.json", "sha256": digest(config)}},
        "shards": shards,
        "per_rank_payload_bytes": [
            sum(shard["payload_bytes"] for shard in shards if shard["rank"] == rank) for rank in (0, 1)
        ],
        "tensor_values_verified": True,
        "tp1_a8_bitwise_equivalent": False,
        "communication": COMMUNICATION,
    }
    write_json(root / "manifest.json", manifest)
    return base


@pytest.fixture
def artifact_paths(artifact_template, tmp_path):
    shutil.copytree(artifact_template, tmp_path / "model")
    return tmp_path / "model/tp2", tmp_path / "model/config.json"


def test_tp2_reader_cpu_factories_ignore_model_default_device(artifact_paths):
    # vLLM constructs inside a default NPU device context. Meta emulates that
    # factory override on a CPU host and catches unqualified torch.arange.
    with torch.device("meta"):
        artifact = reader.open_vq2a8_tp2_artifact(*artifact_paths, tp_rank=1)
        shards = list(artifact.iter_rank_shards(1, device="cpu"))
    assert shards
    assert all(
        tensor.device.type == "cpu"
        for shard in shards
        for values in shard.values()
        for tensors, _ in values.values()
        for tensor in tensors.values()
    )


def shard_entry(root, *, layer=1, rank=0):
    manifest = read_json(root / "manifest.json")
    entry = next(entry for entry in manifest["shards"] if entry["layer"] == layer and entry["rank"] == rank)
    return manifest, entry


def change_metadata(root, change, *, layer=1, rank=0):
    manifest, entry = shard_entry(root, layer=layer, rank=rank)
    path = root / entry["metadata_file"]
    metadata = read_json(path)
    change(metadata)
    write_json(path, metadata)
    entry["metadata_sha256"] = digest(path)
    write_json(root / "manifest.json", manifest)


def change_tensor(root, change, *, layer=1, rank=0):
    manifest, entry = shard_entry(root, layer=layer, rank=rank)
    path = root / entry["file"]
    tensors = {key: tensor.clone() for key, tensor in load_file(path).items()}
    change(tensors)
    save_file(tensors, path)
    entry["sha256"] = digest(path)
    metadata_path = root / entry["metadata_file"]
    metadata = read_json(metadata_path)
    metadata["sha256"] = entry["sha256"]
    write_json(metadata_path, metadata)
    entry["metadata_sha256"] = digest(metadata_path)
    write_json(root / "manifest.json", manifest)


@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_artifact_complete_rank_bound_reader_preserves_variable_k_and_claims(artifact_paths, rank):
    root, config = artifact_paths
    before = digest(root / "manifest.json")
    artifact = reader.open_vq2a8_tp2_artifact(root, config, tp_rank=rank)
    assert artifact.root == root.resolve() and artifact.model_config_path == config.resolve()
    assert artifact.tp_rank == rank and artifact.tensor_hashes_verified is True
    assert artifact.layer(0).expert_ids == (0,)
    assert artifact.layer(1).expert_ids == (0, 1)  # TP partitions matrices, not expert IDs.
    assert set(artifact.layers) == {0, 1}
    assert artifact.model_layout.hidden_size == 512
    widths = []
    for expert in (0, 1):
        payload, spec = artifact.load_expert(1, expert, "down", device="cpu", non_blocking=True)
        widths.append(spec.columns)
        assert payload["activation_order"].dtype == torch.int64
        assert spec.rht_true_columns == 256 and spec.rows == 512 and spec.rht_block_size == 128
        assert spec.canonical_shape == (512, 512)
        assert spec.input_column_range == (rank * 256, (rank + 1) * 256)
        assert spec.output_row_ranges == ((0, 512),)
        assert spec.metadata["runtime_supported"] is False
        assert bool((payload["pair_lut"] == 128).any())
    assert widths == [256, 512]
    assert artifact.manifest["runtime_compatible"] is False
    assert artifact.manifest["tp1_a8_bitwise_equivalent"] is False
    assert digest(root / "manifest.json") == before


def test_tp2_artifact_rank_shard_iterator_opens_once_not_per_expert(artifact_paths, monkeypatch):
    root, config = artifact_paths
    calls = []
    real_open = reader.safe_open

    def counted_open(*args, **kwargs):
        calls.append((args, kwargs))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(reader, "safe_open", counted_open)
    artifact = reader.open_vq2a8_tp2_artifact(root, config)
    assert calls == []  # Header inspection does not open payloads through safetensors.
    chunks = list(artifact.iter_rank_shards(1, 0))
    assert len(calls) == len(chunks) == 1
    assert set(chunks[0]) == {0, 1}
    for matrices in chunks[0].values():
        assert set(matrices) == {"gate_up", "down"}
        for tensors, _ in matrices.values():
            assert set(tensors) == set(DTYPES)
    with pytest.raises(ValueError, match="rank"):
        list(artifact.iter_rank_shards(1, 1))


@pytest.mark.parametrize(
    "field,value",
    [
        ("complete", False),
        ("tp_size", 1),
        ("tp_ranks", [0]),
        ("tp_ranks", [False, 1]),
        ("layers_selected", [0]),
        ("layers_expected", 1),
        ("schema_version", True),
        ("dry_run", True),
        ("runtime_compatible", True),
        ("tp1_a8_bitwise_equivalent", True),
        ("format", "vq2a8_direct_tp1_v1"),
    ],
)
def test_tp2_artifact_rejects_manifest_contract_tampering(artifact_paths, field, value):
    root, config = artifact_paths
    manifest = read_json(root / "manifest.json")
    manifest[field] = value
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError):
        reader.open_vq2a8_tp2_artifact(root, config)


@pytest.mark.parametrize("rank", [-1, 2, True, 0.0])
def test_tp2_artifact_rejects_invalid_bound_rank(artifact_paths, rank):
    with pytest.raises(ValueError, match="rank"):
        reader.open_vq2a8_tp2_artifact(*artifact_paths, tp_rank=rank)


def test_tp2_artifact_config_hash_and_geometry_are_bound(artifact_paths):
    root, config = artifact_paths
    config.write_bytes(config.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="config SHA"):
        reader.open_vq2a8_tp2_artifact(root, config)


def test_tp2_artifact_rejects_wrong_model_geometry_even_with_matching_config_hash(artifact_paths):
    root, config = artifact_paths
    value = read_json(config)
    value["hidden_size"] = 1024
    write_json(config, value)
    manifest = read_json(root / "manifest.json")
    manifest["source"]["config"]["sha256"] = digest(config)
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="model_layout"):
        reader.open_vq2a8_tp2_artifact(root, config)


@pytest.mark.parametrize("mode", ["missing_rank", "duplicate_shard", "ep_partition"])
def test_tp2_artifact_rejects_missing_duplicate_or_ep_expert_coverage(artifact_paths, mode):
    root, config = artifact_paths
    manifest = read_json(root / "manifest.json")
    if mode == "missing_rank":
        manifest["shards"] = [entry for entry in manifest["shards"] if entry["rank"] == 0]
    elif mode == "duplicate_shard":
        manifest["shards"].append(manifest["shards"][0])
    else:
        manifest["communication"]["routing"] = "experts_partitioned_across_ranks"
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError):
        reader.open_vq2a8_tp2_artifact(root, config)


@pytest.mark.parametrize(
    "name", ["../outside.safetensors", "/absolute", "C:/outside", "tp2\\rank0\\shard", "tp2//bad", "tp2/./bad"]
)
def test_tp2_artifact_rejects_escape_and_noncanonical_paths(artifact_paths, name):
    root, config = artifact_paths
    manifest = read_json(root / "manifest.json")
    manifest["shards"][0]["file"] = name
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="path"):
        reader.open_vq2a8_tp2_artifact(root, config)


@pytest.mark.parametrize("target", ["file", "directory", "root", "config"])
def test_tp2_artifact_rejects_symlinks_in_any_path_component(artifact_paths, target):
    root, config = artifact_paths
    _, entry = shard_entry(root)
    original = {"file": root / entry["file"], "directory": root / "tp2/rank0", "root": root, "config": config}[target]
    moved = original.with_name(original.name + "-actual")
    original.rename(moved)
    try:
        original.symlink_to(moved, target_is_directory=moved.is_dir())
    except OSError as error:
        pytest.skip(f"Creating a symlink is unavailable on this host: {error}")
    with pytest.raises(ValueError, match="symlink"):
        reader.open_vq2a8_tp2_artifact(root, config)


@pytest.mark.parametrize("target", ["file", "directory", "root", "config"])
def test_tp2_artifact_symlink_guard_branch_without_host_link_privileges(artifact_paths, target, monkeypatch):
    root, config = artifact_paths
    _, entry = shard_entry(root)
    link = {"file": root / entry["file"], "directory": root / "tp2/rank0", "root": root, "config": config}[target]
    real_is_symlink = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == link or real_is_symlink(path))
    with pytest.raises(ValueError, match="symlink"):
        reader.open_vq2a8_tp2_artifact(root, config)


def test_tp2_artifact_rejects_metadata_hash_mismatch(artifact_paths):
    root, config = artifact_paths
    _, entry = shard_entry(root)
    path = root / entry["metadata_file"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="metadata SHA"):
        reader.open_vq2a8_tp2_artifact(root, config)


def test_tp2_artifact_tensor_hash_checks_only_the_selected_rank(artifact_paths):
    root, config = artifact_paths
    _, entry = shard_entry(root, rank=1)
    path = root / entry["file"]
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    assert reader.open_vq2a8_tp2_artifact(root, config, tp_rank=0).tensor_hashes_verified is True
    with pytest.raises(ValueError, match="tensor SHA"):
        reader.open_vq2a8_tp2_artifact(root, config, tp_rank=1)
    assert (
        reader.open_vq2a8_tp2_artifact(root, config, tp_rank=1, verify_tensor_hashes=False).tensor_hashes_verified
        is False
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("logical_shape", [512, 512]),
        ("input_column_range", [256, 512]),
        ("canonical_shape", [1024, 512]),
        ("tile_valid_counts", [257, 1]),
        ("lut_source_tile_ids", [0, 2]),
        ("padding_columns", 0),
        ("tp_rank", 1),
        ("runtime_supported", True),
        ("rht_block_size", 64),
    ],
)
def test_tp2_artifact_rejects_matrix_metadata_tampering_with_valid_sha(artifact_paths, field, value):
    root, config = artifact_paths

    def change(metadata):
        matrix = next(item for item in metadata["matrices"] if item["expert_id"] == 1 and item["kind"] == "down")
        matrix[field] = value

    change_metadata(root, change)
    with pytest.raises(ValueError):
        reader.open_vq2a8_tp2_artifact(root, config)


def test_tp2_artifact_rejects_cross_rank_tile_population_mismatch(artifact_paths):
    root, config = artifact_paths

    def change(metadata):
        matrix = next(item for item in metadata["matrices"] if item["expert_id"] == 1 and item["kind"] == "down")
        matrix["tile_valid_counts"][0] -= 1
        matrix["tile_valid_counts"][1] += 1

    change_metadata(root, change)
    with pytest.raises(ValueError, match="populations"):
        reader.open_vq2a8_tp2_artifact(root, config)


@pytest.mark.parametrize(
    "mode",
    [
        "duplicate_order",
        "dummy_order",
        "dummy_code",
        "dummy_scale",
        "dummy_bias",
        "dummy_sign",
        "nan_lut",
        "nan_scale",
        "bad_sign",
        "order_dtype",
        "packed_shape",
    ],
)
def test_tp2_artifact_rejects_invalid_payload_even_when_hashes_are_updated(artifact_paths, mode):
    root, config = artifact_paths
    prefix = "1.down."

    def change(tensors):
        if mode == "duplicate_order":
            tensors[prefix + "activation_order"][0] = tensors[prefix + "activation_order"][1]
        elif mode == "dummy_order":
            order = tensors[prefix + "activation_order"]
            dummy = torch.where(order >= 256)[0][:2]
            order[dummy] = order[dummy.flip(0)]
        elif mode == "dummy_code":
            order = tensors[prefix + "activation_order"]
            dummy_column = int(torch.where(order >= 256)[0][0])
            tensors[prefix + "packed_zn"].reshape(16, 512, 8)[0, dummy_column, 0] = 1
        elif mode.startswith("dummy_"):
            field = {"dummy_scale": "weight_scale", "dummy_bias": "weight_bias", "dummy_sign": "rht_sign"}[mode]
            tensors[prefix + field][256] = -1
        elif mode == "nan_lut":
            tensors[prefix + "pair_lut"].flatten()[0] = 127
        elif mode == "nan_scale":
            tensors[prefix + "weight_scale"][0] = float("nan")
        elif mode == "bad_sign":
            tensors[prefix + "rht_sign"][0] = 0
        elif mode == "order_dtype":
            tensors[prefix + "activation_order"] = tensors[prefix + "activation_order"].to(torch.int32)
        else:
            tensors[prefix + "packed_zn"] = tensors[prefix + "packed_zn"].flatten()

    change_tensor(root, change)
    with pytest.raises(ValueError):
        artifact = reader.open_vq2a8_tp2_artifact(root, config)
        artifact.load_expert(1, 1, "down")


def test_tp2_artifact_rejects_tensor_changed_after_open(artifact_paths):
    root, config = artifact_paths
    artifact = reader.open_vq2a8_tp2_artifact(root, config)
    _, entry = shard_entry(root)
    path = root / entry["file"]
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="changed after"):
        list(artifact.iter_rank_shards(1))


def test_tp2_artifact_rejects_noncontiguous_safetensor_offsets(artifact_paths):
    root, config = artifact_paths
    _, entry = shard_entry(root)
    path = root / entry["file"]
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + length])
    first = min(header, key=lambda key: header[key]["data_offsets"][0])
    header[first]["data_offsets"] = [offset + 1 for offset in header[first]["data_offsets"]]
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + length :])
    with pytest.raises(ValueError, match="Overlapping/gapped"):
        reader.open_vq2a8_tp2_artifact(root, config)


def test_tp2_artifact_subprocess_cpu_import_and_load_does_not_import_runtime(artifact_paths):
    root, config = artifact_paths
    script = """
import importlib, importlib.abc, importlib.machinery, sys, types
from pathlib import Path
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.')
               for name in ('vllm', 'torch_npu', 'vllm_ascend')):
            raise AssertionError('Unexpected runtime import: ' + fullname)
        return None
sys.meta_path.insert(0, BlockRuntime())
package = types.ModuleType('_tp2_cpu_test')
package.__path__ = [str(Path(sys.argv[1]) / 'vllm_ascend/quantization')]
package.__spec__ = importlib.machinery.ModuleSpec('_tp2_cpu_test', loader=None, is_package=True)
sys.modules['_tp2_cpu_test'] = package
reader = importlib.import_module('_tp2_cpu_test.vq2a8_tp2_runtime')
artifact = reader.open_vq2a8_tp2_artifact(sys.argv[2], sys.argv[3], tp_rank=1)
assert set(next(artifact.iter_rank_shards(1))) == {0, 1}
assert artifact.manifest['runtime_compatible'] is False
print('TP2_CPU_READER=PASS')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", "-c", script, str(REPO), str(root), str(config)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TP2_CPU_READER=PASS" in result.stdout
