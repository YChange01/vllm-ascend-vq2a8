# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU bytes/schema tests; no native library or NPU is imported."""

import hashlib
import json
import math
import struct
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import load_file, save_file

from vllm_ascend.quantization.vq2a8_artifact import VQ2ModelLayout
from vllm_ascend.quantization.vq2a8_repack import VQ2_DIRECT_TP1_FORMAT
from vllm_ascend.quantization.vq2a8_runtime import _expected_stacked_shapes, _spec_from_runtime_layout
from vllm_ascend.quantization.vq2a8_v4_v2_layout import V4_V2_FIELDS, convert_expert_payload
from vllm_ascend.quantization.vq2a8_v4_v2_prepacked import (
    V4_V2_PREPACKED_FORMAT,
    V4_V2_PREPACKED_VALIDATION_SCOPE,
    open_vq2a8_v4_v2_prepacked_artifact,
    prepacked_projection_shapes,
    serialize_spec,
    validate_prepacked_payload,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix_spec(kind):
    k = 4096 if kind == "gate_up" else 2048
    return _spec_from_runtime_layout(
        0,
        kind,
        {
            "canonical_shape": [4096, k],
            "original_shape": [4096, k],
            "row_tiles": 128,
            "column_tiles": k // 256,
            "row_group_size": 32,
            "group_size": 256,
            "rht_block_size": 128,
            "rht_true_columns": k,
        },
    )


def source_payload(spec, expert=0):
    k = spec.columns
    books = (torch.arange(spec.column_tiles * 128 * 32) % 126).to(torch.uint8).reshape(spec.column_tiles, 128, 16, 2)
    books.flatten()[0] = 128  # finite negative zero must survive byte-for-byte
    return {
        "packed_indices": torch.full((2048, k // 8), 0x76543210 + expert, dtype=torch.int32),
        "codebooks": books.view(torch.float8_e4m3fn),
        "codebook_tile_ids": (torch.arange(k) % spec.column_tiles).to(torch.uint8),
        "weight_scale": torch.linspace(0.5, 1.5, k),
        "weight_bias": torch.linspace(-0.1, 0.1, k),
        "rht_sign": ((torch.arange(k) % 2) * 2 - 1).to(torch.int8),
    }


def make_artifact(tmp_path, *, experts=1):
    root = tmp_path / "prepacked"
    root.mkdir()
    config = tmp_path / "config.json"
    config_dict = {
        "num_hidden_layers": 1,
        "num_hash_layers": 0,
        "n_routed_experts": experts,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "quantization_config": {"quant_method": "vq2a8"},
    }
    config.write_text(json.dumps(config_dict), encoding="utf-8")
    specs = {kind: matrix_spec(kind) for kind in ("gate_up", "down")}
    shards = []
    for expert in range(experts):
        tensors = {}
        for kind, spec in specs.items():
            payload = convert_expert_payload(source_payload(spec, expert), spec)
            tensors.update({f"{kind}_{field}": tensor.unsqueeze(0).clone() for field, tensor in payload.items()})
        path = root / f"expert_{expert}.safetensors"
        save_file(tensors, path)
        shards.append({"file": path.name, "sha256": digest(path), "expert_ids": [expert]})
    manifest = {
        "format": V4_V2_PREPACKED_FORMAT,
        "schema_version": 1,
        "layout_version": 1,
        "complete": True,
        "model_config_sha256": digest(config),
        "model_layout": asdict(VQ2ModelLayout.from_dict(config_dict)),
        "source": {"format": VQ2_DIRECT_TP1_FORMAT, "manifest_sha256": "1" * 64},
        "layers": [
            {
                "layer_index": 0,
                "expert_ids": list(range(experts)),
                "specs": {kind: serialize_spec(spec) for kind, spec in specs.items()},
                "source_tensor_shapes": {
                    f"{kind}_{field}": list(shape)
                    for kind, spec in specs.items()
                    for field, shape in _expected_stacked_shapes(spec, experts).items()
                },
                "source_tensor_sha256": "2" * 64,
                "source_metadata_sha256": "3" * 64,
                "shards": shards,
            }
        ],
    }
    write_manifest(root, manifest)
    return root, config, manifest


def write_manifest(root, manifest):
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def artifact_files(tmp_path):
    return make_artifact(tmp_path)


def test_prepacked_is_exact_current_conversion_standalone_and_bounded(tmp_path, monkeypatch):
    root, config, _ = make_artifact(tmp_path, experts=2)
    artifact = open_vq2a8_v4_v2_prepacked_artifact(root, config, verify_tensor_hashes=True)
    layer = artifact.layer(0)
    assert layer.v4_v2_prepacked and layer.expert_ids == (0, 1)
    assert not hasattr(layer, "tensor_shapes")
    assert len(layer.shards) == 2
    with pytest.raises(TypeError):
        artifact.layers[1] = layer
    with pytest.raises(TypeError):
        layer.specs["gate_up"] = layer.specs["down"]
    expected = {
        (expert, kind): convert_expert_payload(source_payload(matrix_spec(kind), expert), matrix_spec(kind))
        for expert in (0, 1)
        for kind in ("gate_up", "down")
    }
    # Runtime reading must never call the converter or depend on source files.
    monkeypatch.setattr(
        "vllm_ascend.quantization.vq2a8_v4_v2_layout.convert_expert_payload",
        lambda *args, **kwargs: pytest.fail("runtime conversion was invoked"),
    )
    for (expert, kind), converted in expected.items():
        timings = {}
        payload, spec = artifact.load_expert(0, expert, kind, timings=timings)
        assert spec.expert_id == expert and spec.name == f"0.mlp.experts.{expert}.{kind}"
        assert layer.projection_shapes(kind) == prepacked_projection_shapes(spec)
        assert set(payload) == set(V4_V2_FIELDS)
        for field, tensor in payload.items():
            assert tensor.device.type == "cpu" and tensor.is_contiguous()
            assert torch.equal(tensor.view(torch.uint8), converted[field].view(torch.uint8))
        assert set(timings) == {"host_read_s", "host_validate_s"}
        assert all(value >= 0 for value in timings.values())


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda m: m.update(format="old_v3"), "format"),
        (lambda m: m.update(schema_version=True), "schema_version"),
        (lambda m: m.update(layout_version=2), "layout_version"),
        (lambda m: m.update(complete=False), "complete"),
        (lambda m: m.update(unknown=True), "exactly"),
        (lambda m: m.update(layers=[]), "every model layer"),
        (lambda m: m["model_layout"].update(num_hidden_layers=True), "num_hidden_layers"),
        (lambda m: m["source"].update(manifest_sha256="bad"), "SHA-256"),
        (lambda m: m["source"].update(format="v3"), "source.format"),
        (lambda m: m["layers"][0].update(layer_index=True), "layer_index"),
        (lambda m: m["layers"][0].update(expert_ids=[1]), "inventory"),
        (lambda m: m["layers"][0].update(expert_ids=[0, 0]), "sorted unique"),
        (lambda m: m["layers"][0].update(shards=[]), "nonempty"),
        (lambda m: m["layers"][0]["shards"][0].update(expert_ids=[1]), "cover layer"),
        (lambda m: m["layers"][0]["shards"].append(m["layers"][0]["shards"][0].copy()), "Duplicate.*path"),
        (lambda m: m["layers"][0]["specs"]["down"].update(rht_block_size=3), "power of two"),
        (lambda m: m["layers"][0]["specs"]["down"].update(extra=1), "exactly"),
        (
            lambda m: m["layers"][0]["source_tensor_shapes"]["down_weight_scale"].__setitem__(1, 1024),
            "source_tensor_shapes",
        ),
    ],
)
def test_bad_manifests_rejected(artifact_files, mutation, match):
    root, config, manifest = artifact_files
    mutation(manifest)
    write_manifest(root, manifest)
    with pytest.raises(ValueError, match=match):
        open_vq2a8_v4_v2_prepacked_artifact(root, config)


@pytest.mark.parametrize(
    "locator", ["../outside.safetensors", "/absolute", "C:/absolute", "a\\b", "./expert_0.safetensors", "a//b"]
)
def test_unsafe_path_rejected(artifact_files, locator):
    root, config, manifest = artifact_files
    manifest["layers"][0]["shards"][0]["file"] = locator
    write_manifest(root, manifest)
    with pytest.raises(ValueError, match="unsafe"):
        open_vq2a8_v4_v2_prepacked_artifact(root, config)


def test_model_hash_and_config_geometry_are_binding(artifact_files):
    root, config, manifest = artifact_files
    value = json.loads(config.read_text())
    value["hidden_size"] = 2048
    config.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        open_vq2a8_v4_v2_prepacked_artifact(root, config)
    manifest["model_config_sha256"] = digest(config)
    manifest["model_layout"]["hidden_size"] = 2048
    write_manifest(root, manifest)
    with pytest.raises(ValueError, match="original shape"):
        open_vq2a8_v4_v2_prepacked_artifact(root, config)


def test_optional_large_hash_catches_finite_weight_tamper(artifact_files):
    root, config, manifest = artifact_files
    shard = root / manifest["layers"][0]["shards"][0]["file"]
    tensors = load_file(shard)
    tensors["gate_up_packed_zn"].flatten()[0] ^= 1
    save_file(tensors, shard)
    open_vq2a8_v4_v2_prepacked_artifact(root, config)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        open_vq2a8_v4_v2_prepacked_artifact(root, config, verify_tensor_hashes=True)


def add_provenance(manifest):
    manifest.update(
        producer={
            "tool": "tools/prepack_vq2a8_v4_v2.py",
            "files": [{"file": "vq2a8_v4_v2_layout.py", "sha256": "4" * 64}],
        },
        payload_bytes=sum(
            math.prod(shape) * dtype.itemsize
            for kind in ("gate_up", "down")
            for shape, dtype in zip(
                prepacked_projection_shapes(matrix_spec(kind)),
                (torch.uint8, torch.uint8, torch.int64, torch.float32, torch.float32, torch.int8),
                strict=True,
            )
        ),
        tensor_bytes_verified=True,
        validation_scope=V4_V2_PREPACKED_VALIDATION_SCOPE,
    )


def test_exporter_provenance_does_not_pin_current_code(artifact_files):
    root, config, manifest = artifact_files
    add_provenance(manifest)
    write_manifest(root, manifest)
    artifact = open_vq2a8_v4_v2_prepacked_artifact(root, config)
    assert artifact.manifest["producer"]["files"][0]["sha256"] == "4" * 64


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda m: m.update(payload_bytes=True), "positive integer"),
        (lambda m: m.update(payload_bytes=1), "payload_bytes"),
        (lambda m: m.update(tensor_bytes_verified=False), "tensor_bytes_verified"),
        (lambda m: m.update(validation_scope="device_correctness_proven"), "validation_scope"),
        (lambda m: m["producer"].update(files=[]), "source identities"),
        (lambda m: m["producer"]["files"][0].update(file="../bad.py"), "portable basenames"),
        (lambda m: m["producer"]["files"][0].update(sha256="broken"), "SHA-256"),
    ],
)
def test_provenance_remains_well_formed(artifact_files, mutation, match):
    root, config, manifest = artifact_files
    add_provenance(manifest)
    mutation(manifest)
    write_manifest(root, manifest)
    with pytest.raises(ValueError, match=match):
        open_vq2a8_v4_v2_prepacked_artifact(root, config)


@pytest.mark.parametrize("mutation", ["missing", "extra", "dtype", "shape", "truncated"])
def test_header_checks_are_not_optional(artifact_files, mutation):
    root, config, manifest = artifact_files
    shard = root / manifest["layers"][0]["shards"][0]["file"]
    tensors = load_file(shard)
    if mutation == "missing":
        tensors.pop("down_rht_sign")
    elif mutation == "extra":
        tensors["device_pointer_table"] = torch.zeros(1, dtype=torch.int64)
    elif mutation == "dtype":
        tensors["down_activation_order"] = tensors["down_activation_order"].int()
    elif mutation == "shape":
        tensors["down_activation_order"] = tensors["down_activation_order"][:, :-1].contiguous()
    save_file(tensors, shard)
    if mutation == "truncated":
        with shard.open("r+b") as stream:
            stream.truncate(shard.stat().st_size - 1)
    with pytest.raises(ValueError):
        open_vq2a8_v4_v2_prepacked_artifact(root, config)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("pair_lut", 127, "NaN"),
        ("activation_order", 0, "permutation"),
        ("weight_scale", float("inf"), "finite"),
        ("weight_bias", float("nan"), "finite"),
        ("rht_sign", 0, "-1 and \\+1"),
    ],
)
def test_payload_values_checked_without_expansion(artifact_files, field, value, match):
    root, config, _ = artifact_files
    artifact = open_vq2a8_v4_v2_prepacked_artifact(root, config)
    payload, spec = artifact.load_expert(0, 0, "down")
    payload[field].flatten()[-1] = value
    with pytest.raises(ValueError, match=match):
        validate_prepacked_payload(payload, spec)


def test_load_error_lookup_and_unvalidated_timing(artifact_files):
    root, config, _ = artifact_files
    artifact = open_vq2a8_v4_v2_prepacked_artifact(root, config)
    for layer, expert, kind, error in (
        (True, 0, "down", KeyError),
        (0, True, "down", KeyError),
        (0, 1, "down", KeyError),
        (0, 0, "other", ValueError),
    ):
        with pytest.raises(error):
            artifact.load_expert(layer, expert, kind)
    times = {"host_read_s": 1.0, "host_validate_s": 2.0}
    artifact.load_expert(0, 0, "down", validate_payload=False, timings=times)
    assert times["host_read_s"] >= 1 and times["host_validate_s"] == 2


def test_duplicate_json_keys_rejected(artifact_files):
    root, config, _ = artifact_files
    path = root / "manifest.json"
    path.write_text(
        path.read_text().replace('"complete": true', '"complete": true, "complete": true'), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate JSON"):
        open_vq2a8_v4_v2_prepacked_artifact(root, config)


def test_duplicate_safetensors_keys_rejected(artifact_files):
    root, config, manifest = artifact_files
    path = root / manifest["layers"][0]["shards"][0]["file"]
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = raw[8 : 8 + length].decode()
    # Preserve all offsets; duplicate metadata key has otherwise no weight impact.
    header = '{"__metadata__":{},"__metadata__":{},' + header[1:]
    data = header.encode()
    data += b" " * (-len(data) % 8)
    path.write_bytes(struct.pack("<Q", len(data)) + data + raw[8 + length :])
    with pytest.raises(ValueError, match="Duplicate JSON"):
        open_vq2a8_v4_v2_prepacked_artifact(root, config)


def test_symlink_shard_rejected(artifact_files, tmp_path):
    root, config, manifest = artifact_files
    source = root / manifest["layers"][0]["shards"][0]["file"]
    link = root / "linked.safetensors"
    try:
        link.symlink_to(source)
    except OSError as error:
        pytest.skip(f"Platform does not allow test symlink: {error}")
    manifest["layers"][0]["shards"][0]["file"] = link.name
    write_manifest(root, manifest)
    with pytest.raises(ValueError, match="symbolic link"):
        open_vq2a8_v4_v2_prepacked_artifact(root, config)
