# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU storage/direct-residency contracts, not NPU numerical validation."""

import hashlib
import json
import shutil
from collections import OrderedDict
from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest
import torch
from safetensors.torch import load_file, save_file

from tools.repack_vq2a8_tp2 import _write_shard
from vllm_ascend.quantization import vq2a8_execution_tp1_zn as runtime
from vllm_ascend.quantization import vq2a8_execution_v3 as v3
from vllm_ascend.quantization import vq2a8_offline as offline
from vllm_ascend.quantization import vq2a8_tp2_runtime as storage
from vllm_ascend.quantization.vq2a8_artifact import VQ2MatrixSpec, VQ2ModelLayout
from vllm_ascend.quantization.vq2a8_repack import VQ2_DIRECT_TP1_FORMAT
from vllm_ascend.quantization.vq2a8_tp1_zn_runtime import artifact_format, open_vq2a8_tp1_zn_artifact
from vllm_ascend.quantization.vq2a8_tp2_layout import repack_matrix_zn
from vllm_ascend.quantization.vq2a8_v3_workspace import RESIDENT_FIELDS, resident_shapes
from vllm_ascend.quantization.vq2a8_zn_contract import VQ2_TP1_ZN_FORMAT, communication_contract


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def canonical(expert, kind):
    n, k = (1024, 512) if kind == "gate_up" else (512, 512)
    spec = VQ2MatrixSpec(
        name=f"0.mlp.experts.{expert}.{kind}",
        layer_index=0,
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
    tensors = {
        "packed_indices": torch.full((n * k // 16,), 0x76543210, dtype=torch.int32),
        "codebooks": (torch.arange(k // 256 * n // 32 * 32) % 120)
        .to(torch.uint8)
        .reshape(k // 256, n // 32, 16, 2)
        .view(torch.float8_e4m3fn),
        "perm": torch.roll(torch.arange(k, dtype=torch.int32), 17),
        "weight_scale": torch.linspace(-0.5, 1.5, k),
        "weight_bias": torch.linspace(-0.25, 0.25, k),
        "rht_sign": torch.where(torch.arange(k) % 2 == 0, 1, -1).to(torch.int8),
    }
    return tensors, spec


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    base = tmp_path_factory.mktemp("tp1-zn-template")
    root, config = base / "artifact", base / "config.json"
    root.mkdir()
    cfg = {
        "num_hidden_layers": 1,
        "num_hash_layers": 0,
        "n_routed_experts": 2,
        "hidden_size": 512,
        "moe_intermediate_size": 512,
        "quantization_config": {"quant_method": "vq2a8"},
    }
    write_json(config, cfg)
    layout = VQ2ModelLayout.from_dict(cfg)
    tensors, matrices = {}, []
    for expert in (0, 1):
        for kind in ("gate_up", "down"):
            source, spec = canonical(expert, kind)
            payload, metadata = repack_matrix_zn(source, spec, tp_size=1)
            records = {
                field: {"key": f"{expert}.{kind}.{field}", "dtype": storage._DTYPES[field], "shape": list(value.shape)}
                for field, value in payload.items()
            }
            tensors.update({records[field]["key"]: value for field, value in payload.items()})
            matrices.append({**metadata, "name": spec.name, "expert_id": expert, "tensors": records})
    shard = _write_shard(root, 0, 0, (0, 1), tensors, matrices, tp_size=1)
    write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "format": VQ2_TP1_ZN_FORMAT,
            "tp_size": 1,
            "tp_ranks": [0],
            "runtime_compatible": False,
            "complete": True,
            "dry_run": False,
            "layers_selected": [0],
            "layers_expected": 1,
            "model_layout": asdict(layout),
            "source": {"config": {"file": "config.json", "sha256": digest(config)}},
            "shards": [shard],
            "per_rank_payload_bytes": [shard["payload_bytes"]],
            "tensor_values_verified": True,
            "tp1_a8_bitwise_equivalent": False,
            "communication": communication_contract(1),
        },
    )
    return base


@pytest.fixture
def paths(template, tmp_path):
    shutil.copytree(template, tmp_path / "model")
    return tmp_path / "model/artifact", tmp_path / "model/config.json"


def rewrite(paths, *, matrix_change=None, tensor_change=None):
    root, _ = paths
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["shards"][0]
    metadata_path = root / entry["metadata_file"]
    metadata = json.loads(metadata_path.read_text())
    if matrix_change:
        matrix_change(metadata["matrices"][1])  # expert zero down
    if tensor_change:
        tensor_path = root / entry["file"]
        tensors = load_file(tensor_path)
        tensor_change(tensors)
        save_file(tensors, tensor_path)
        entry["sha256"] = metadata["sha256"] = digest(tensor_path)
    write_json(metadata_path, metadata)
    entry["metadata_sha256"] = digest(metadata_path)
    write_json(root / "manifest.json", manifest)


def test_tp1_reader_preserves_full_k_and_opens_each_shard_once(paths, monkeypatch):
    before = digest(paths[0] / "manifest.json")
    artifact = open_vq2a8_tp1_zn_artifact(*paths)
    opened = []
    original = storage.safe_open
    monkeypatch.setattr(storage, "safe_open", lambda *a, **kw: opened.append(a[0]) or original(*a, **kw))
    with torch.device("meta"):
        shards = list(artifact.iter_rank_shards(0, device="cpu"))
    assert len(opened) == 1 and len(shards) == 1
    assert artifact_format(paths[0]) == VQ2_TP1_ZN_FORMAT
    assert artifact.tp_rank == 0 and artifact.tensor_hashes_verified
    assert artifact.manifest["runtime_compatible"] is False
    for expert, matrices in shards[0].items():
        for kind, (tensors, spec) in matrices.items():
            assert spec.canonical_shape == spec.logical_shape == spec.packed_shape
            assert spec.columns == spec.rht_true_columns == 512 and spec.padding_columns == 0
            assert spec.tile_valid_counts == (256, 256)
            assert spec.lut_source_tile_ids == (0, 1)
            assert all(t.device.type == "cpu" for t in tensors.values())
            source, original_spec = canonical(expert, kind)
            expected, _ = repack_matrix_zn(source, original_spec, tp_size=1)
            assert all(torch.equal(tensors[field], expected[field]) for field in expected)
    assert digest(paths[0] / "manifest.json") == before
    with pytest.raises(ValueError, match="rank"):
        list(artifact.iter_rank_shards(0, rank=1))


def test_readers_do_not_cross_load_tp_formats(paths):
    with pytest.raises(ValueError, match="manifest.format"):
        storage.open_vq2a8_tp2_artifact(*paths)
    root, config = paths
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["format"] = "vq2a8_zn_tp2_v1"
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest.format"):
        open_vq2a8_tp1_zn_artifact(root, config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("tp_size", 2),
        ("tp_size", True),
        ("tp_ranks", [0, 1]),
        ("tp_ranks", [False]),
        ("complete", False),
        ("dry_run", True),
        ("tensor_values_verified", False),
        ("communication", communication_contract(2)),
    ],
)
def test_tp1_reader_rejects_false_or_cross_rank_manifest_claims(paths, key, value):
    root, _ = paths
    manifest = json.loads((root / "manifest.json").read_text())
    manifest[key] = value
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest"):
        open_vq2a8_tp1_zn_artifact(*paths)


@pytest.mark.parametrize(
    "key,value",
    [
        ("tp_rank", 1),
        ("tp_size", 2),
        ("padding_columns", 256),
        ("logical_shape", [512, 256]),
        ("packed_shape", [512, 768]),
        ("input_column_range", [256, 512]),
        ("output_row_ranges", [[0, 256]]),
        ("tile_valid_counts", [128, 384]),
        ("lut_source_tile_ids", [1, 0]),
        ("runtime_supported", True),
    ],
)
def test_tp1_reader_rejects_sharded_or_padded_metadata(paths, key, value):
    rewrite(paths, matrix_change=lambda entry: entry.update({key: value}))
    with pytest.raises(ValueError):
        open_vq2a8_tp1_zn_artifact(*paths)


@pytest.mark.parametrize(
    "field,value",
    [
        ("activation_order", 0),
        ("weight_scale", float("nan")),
        ("weight_bias", float("inf")),
        ("rht_sign", 0),
        ("pair_lut", 127),
    ],
)
def test_tp1_reader_validates_cpu_values_before_transfer(paths, field, value):
    rewrite(paths, tensor_change=lambda tensors: tensors[f"0.down.{field}"].fill_(value))
    artifact = open_vq2a8_tp1_zn_artifact(*paths)
    with pytest.raises(ValueError):
        list(artifact.iter_rank_shards(0, device="meta"))


def test_tp1_reader_rejects_hash_changes_and_path_escape(paths):
    root, _ = paths
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["shards"][0]
    entry["metadata_sha256"] = "0" * 64
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="SHA-256"):
        open_vq2a8_tp1_zn_artifact(*paths)
    entry["file"] = "../outside.safetensors"
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="escape"):
        open_vq2a8_tp1_zn_artifact(*paths)


def compute_disk():
    specs = {}
    for kind, k in (("gate_up", 4096), ("down", 2048)):
        n = 4096
        shapes = {
            "packed_zn": (n // 32, k // 16, 16, 8),
            "pair_lut": (k // 256, n // 32, 32),
            **{field: (k,) for field, _, _ in RESIDENT_FIELDS[2:]},
        }
        specs[kind] = NS(
            rows=n,
            columns=k,
            rht_true_columns=k,
            rht_block_size=128,
            tp_rank=0,
            canonical_shape=(n, k),
            logical_shape=(n, k),
            packed_shape=(n, k),
            metadata={"format": VQ2_TP1_ZN_FORMAT, "tp_size": 1},
            tensor_shapes=shapes,
        )
    return NS(layer_index=0, expert_ids=(0,), spec_for=lambda expert, kind: specs[kind])


def test_tp1_zn_resident_shapes_are_full_width_without_direct_headers():
    layer = runtime.TP1ZNComputeLayer.from_disk(compute_disk())
    assert resident_shapes(layer, "gate_up")["packed_zn"] == (1, 128, 256, 16, 8)
    assert resident_shapes(layer, "down")["packed_zn"] == (1, 128, 128, 16, 8)
    assert "down_packed_indices" not in layer.tensor_shapes
    assert layer.specs["down"].rht_true_columns == 2048
    plan = v3.resident_plan([layer], 1 << 30)
    with pytest.raises(ValueError, match="no cache fallback"):
        v3.resident_plan([layer], plan["planned_bytes"] - 1)
    direct_shapes = {}
    for kind, spec in layer.specs.items():
        n, k = spec.rows, spec.columns
        direct_shapes.update(
            {
                f"{kind}_packed_indices": (1, n // 2, k // 8),
                f"{kind}_codebooks": (1, k // 256, n // 32, 16, 2),
                **{
                    f"{kind}_{field}": (1, k)
                    for field in ("codebook_tile_ids", "weight_scale", "weight_bias", "rht_sign")
                },
            }
        )
    direct = NS(layer_index=0, expert_ids=(0,), specs=layer.specs, tensor_shapes=direct_shapes)
    assert v3.resident_plan([direct], 1 << 30) == plan


@pytest.mark.parametrize(
    "kind,key,value",
    [
        ("gate_up", "rows", 2048),
        ("down", "rht_true_columns", 1024),
        ("down", "columns", 1024),
        ("down", "tp_rank", 1),
    ],
)
def test_tp1_zn_compute_rejects_tp2_geometry(kind, key, value):
    disk = compute_disk()
    setattr(disk.spec_for(0, kind), key, value)
    with pytest.raises(ValueError):
        runtime.TP1ZNComputeLayer.from_disk(disk)


def test_tp1_zn_residency_direct_copies_once_and_never_converts_or_communicates(monkeypatch):
    disk = compute_disk()
    value = runtime.AscendCV3VQ2TP1ZNMoE.__new__(runtime.AscendCV3VQ2TP1ZNMoE)
    value.layer = runtime.TP1ZNComputeLayer.from_disk(disk)
    value.layer_index, value.device, value.config = 0, torch.device("cpu"), NS(top_k=1)
    value.root = {"gate.weight": torch.zeros(1, 4096)}
    value._resident_ready = value._resident_failed = False
    value._cache = OrderedDict()
    value.projection_kernel, value.v3_preparation, value.v3_decode_graph = "v2", "eager", "none"
    value.progress = False
    value.cache_loads = value.cache_hits = value.h2d_bytes = 0
    value.decode_device_calls = value.prefill_legacy_calls = 0
    sources = {}
    for kind in ("gate_up", "down"):
        spec = disk.spec_for(0, kind)
        payload = {field: torch.ones(spec.tensor_shapes[field], dtype=dtype) for field, dtype, _ in RESIDENT_FIELDS}
        payload["activation_order"] = torch.arange(spec.columns, dtype=torch.int64).flip(0)
        sources[kind] = payload, spec
    reads = []

    def shards(index, *, device):
        reads.append((index, device))
        yield {0: sources}

    value.artifact = NS(iter_rank_shards=shards)
    monkeypatch.setattr(v3, "convert_expert_payload", lambda *a: pytest.fail("unexpected online conversion"))
    monkeypatch.setattr(v3, "resident_library_capabilities", lambda: 7)
    report = value.initialize_resident(budget_bytes=1 << 30)
    assert report["ready"] and report["payload_load"] == "prepacked_direct"
    assert reads == [(0, "cpu")]
    for kind, (source, _) in sources.items():
        workspace = value._resident_workspaces[kind]
        for field, tensor in source.items():
            assert torch.equal(workspace.banks[field][0], tensor)
            assert value._cache[0][kind][0][field].data_ptr() == workspace.banks[field][0].data_ptr()
    routed = torch.ones(1, 4096)
    assert value._reduce_routed(routed) is routed
    value.check_resident_immutable()


def test_tp1_zn_constructor_rejects_legacy_projection_before_load():
    with pytest.raises(ValueError, match="no legacy fallback"):
        runtime.AscendCV3VQ2TP1ZNMoE(NS(), 0, "npu:0", projection_kernel="legacy")


def test_tp1_zn_constructor_runs_real_parent_without_direct_layer_headers(tmp_path, monkeypatch):
    from vllm_ascend.quantization import vq2a8_moe as moe

    disk = compute_disk()
    artifact = NS(
        manifest={"format": VQ2_TP1_ZN_FORMAT},
        tp_rank=0,
        layer=lambda index: disk,
        model_config_path=tmp_path / "config.json",
    )
    root = NS(to=lambda *a, **kw: root)
    fake_device = NS(type="npu", index=0)
    monkeypatch.setattr(torch, "device", lambda *a, **kw: fake_device)
    monkeypatch.setattr(moe.VQ2MoEConfig, "from_json", lambda *a: NS())
    monkeypatch.setattr(moe, "load_vq2a8_moe_root_weights", lambda *a: {"gate.weight": root})
    value = runtime.AscendCV3VQ2TP1ZNMoE(artifact, 0, "npu:0", v3_preparation="fused", v3_decode_graph="moe")
    assert isinstance(value.layer, runtime.TP1ZNComputeLayer)
    assert value.artifact is artifact and value.layer.specs["down"].columns == 2048
    assert value.root["gate.weight"] is root and value._cache == {}
    assert value.projection_kernel == "v2" and value.v3_decode_graph == "moe"


def test_tp1_zn_uses_tp1_launch_for_prefill_and_workspace(monkeypatch):
    calls = []
    value = runtime.AscendCV3VQ2TP1ZNMoE.__new__(runtime.AscendCV3VQ2TP1ZNMoE)
    value.v3_preparation = "eager"
    monkeypatch.setattr(v3, "grouped_projection_resident", lambda inputs, **kw: calls.append((inputs, kw)))
    value._launch_resident(["job"])
    assert calls == [(["job"], {})]
    monkeypatch.setattr(v3, "ResidentV2ProjectionWorkspace", lambda *a, **kw: kw)
    workspace = value._make_resident_workspace([], NS(), 1, NS(), {})
    assert "launcher" not in workspace  # TP1 default; no TP2 partial launcher.


@pytest.mark.parametrize("format_name", [VQ2_TP1_ZN_FORMAT, VQ2_DIRECT_TP1_FORMAT, "unexpected"])
def test_owner_selects_reader_by_format_without_fallback(tmp_path, monkeypatch, format_name):
    from vllm_ascend.quantization import vq2a8_ascendc_v3 as native

    calls = []
    monkeypatch.setattr(native, "load_pinned_library", lambda *a: None)
    monkeypatch.setattr(offline, "artifact_format", lambda *a: format_name)
    monkeypatch.setattr(offline, "audit_offline_root", lambda *a: {})
    artifact = NS(manifest={"format": format_name})
    monkeypatch.setattr(offline, "open_vq2a8_tp1_zn_artifact", lambda *a, **kw: calls.append(("zn", kw)) or artifact)
    monkeypatch.setattr(offline, "open_vq2a8_tp1_artifact", lambda *a, **kw: calls.append(("direct", kw)) or artifact)
    options = {
        "execution_policy": "ascendc_v3",
        "artifact": "unused",
        "ascendc_v3_library": "unused",
        "ascendc_v3_sha256": "a" * 64,
    }
    if format_name == "unexpected":
        with pytest.raises(ValueError, match="no format fallback"):
            offline.OfflineMoEOwner(tmp_path, options, NS(type="npu"))
        assert not calls
    else:
        owner = offline.OfflineMoEOwner(tmp_path, options, NS(type="npu"))
        assert owner.artifact is artifact
        assert (
            calls == [("zn", {"verify_tensor_hashes": True})]
            if format_name == VQ2_TP1_ZN_FORMAT
            else calls == [("direct", {"require_complete": True, "require_reference_identity": True})]
        )


def test_owner_does_not_accept_zn_for_legacy_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(offline, "artifact_format", lambda *a: VQ2_TP1_ZN_FORMAT)
    with pytest.raises(ValueError, match="execution_policy=ascendc_v3"):
        offline.OfflineMoEOwner(tmp_path, {"execution_policy": "cached", "artifact": "unused"}, torch.device("cpu"))


def test_standard_serve_accepts_explicit_tp1_zn_artifact_without_legacy_directory(tmp_path):
    from tools import serve_vq2a8_v3 as server

    model = tmp_path / "model"
    artifact = model / "experts_vq_tp1_zn"
    artifact.mkdir(parents=True)
    library = tmp_path / "candidate.so"
    library.write_bytes(b"CPU command fixture, not a native library")
    args = server.parse_args(
        [
            "--model",
            str(model),
            "--artifact",
            str(artifact),
            "--library",
            str(library),
            "--tensor-parallel-size",
            "1",
            "--physical-npu",
            "2",
            "--preparation",
            "fused",
            "--decode-graph",
            "moe",
        ]
    )
    command = server.build_command(args)
    additional = json.loads(command[command.index("--additional-config") + 1])["vq2a8_offline"]
    assert additional["artifact"] == str(artifact.resolve())
    assert additional["execution_policy"] == "ascendc_v3"
    assert additional["v3_preparation"] == "fused" and additional["v3_decode_graph"] == "moe"
    assert command[command.index("--tensor-parallel-size") + 1] == "1"
    assert command[command.index("--distributed-executor-backend") + 1] == "uni"
    assert server.server_environment(args)["ASCEND_RT_VISIBLE_DEVICES"] == "2"
    assert not (model / "experts_vq_ascend_v2").exists()
