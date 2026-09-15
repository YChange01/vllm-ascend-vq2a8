# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU serialization contracts; not native-kernel or NPU acceptance."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.ut.quantization import test_vq2a8_runtime as source_fixture
from tools import prepack_vq2a8_v4_v2 as tool
from vllm_ascend.quantization.vq2a8_artifact import VQ2MatrixSpec
from vllm_ascend.quantization.vq2a8_runtime import open_vq2a8_tp1_artifact
from vllm_ascend.quantization.vq2a8_v4_v2 import convert_expert_payload as runtime_convert
from vllm_ascend.quantization.vq2a8_v4_v2_layout import convert_expert_payload

REPO = Path(__file__).resolve().parents[3]


def _spec(kind):
    n, k = 4096, 4096 if kind == "gate_up" else 2048
    return VQ2MatrixSpec.from_dict(
        f"0.mlp.experts.0.{kind}",
        {
            "rows": n,
            "cols": k,
            "n_row_tiles": n // 32,
            "n_col_tiles": k // 256,
            "row_group_size": 32,
            "group_size": 256,
            "K": 16,
            "index_bits": 4,
            "vector_len": 2,
            "n_vectors": n * k // 2,
            "n_elements": n * k,
            "orig_shape": [n, k],
            "norm_dim": 0,
            "enable_perm": True,
            "enable_norm": True,
            "enable_rht": True,
            "rht_block_size": 128,
            "rht_true_columns": k,
        },
    )


def _payload(spec, expert):
    k = spec.columns
    # Nontrivial arbitrary finite FP8 pairs, including the negative-zero byte.
    books = ((torch.arange(spec.column_tiles * spec.row_tiles * 32) + expert) % 127).to(torch.uint8)
    books[::7] |= 128
    books = books.reshape(spec.column_tiles, spec.row_tiles, 16, 2)
    columns = torch.arange(k)
    bias = (columns % 11).float() / 4
    bias[::13] = -0.0
    return {
        "packed_indices": torch.full((spec.rows // 2, k // 8), 0x01234567 + expert, dtype=torch.int32),
        "codebooks": books.view(torch.float8_e4m3fn),
        "codebook_tile_ids": (columns % spec.column_tiles).to(torch.uint8),
        "weight_scale": torch.ones(k, dtype=torch.float32) * (expert + 1),
        "weight_bias": bias,
        "rht_sign": torch.where(columns % 2 == 0, 1, -1).to(torch.int8),
    }


def write_source(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(source_fixture, "_spec", _spec)
    monkeypatch.setattr(source_fixture, "_expert_payload", _payload)
    artifact, config = source_fixture._write_artifact(tmp_path)
    config_data = json.loads(config.read_text())
    config_data.update(hidden_size=4096, moe_intermediate_size=2048)
    config.write_text(json.dumps(config_data, sort_keys=True))
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_layout"].update(hidden_size=4096, moe_intermediate_size=2048)
    manifest["source"]["model_config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    return artifact, config


def _args(source, output, *flags):
    return tool.parse_args(["--input", str(source), "--output", str(output), "--threads", "2", *flags])


def _source_bytes(root):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}


def test_runtime_reexports_the_identical_cpu_converter():
    assert runtime_convert is convert_expert_payload


def test_cpu_imports_do_not_initialize_vllm_or_npu():
    script = (
        "from tools.prepack_vq2a8_v4_v2 import _cpu_modules; import sys; "
        "_cpu_modules(); "
        "assert not any(n == p or n.startswith(p + '.') "
        "for p in ('vllm', 'vllm_ascend', 'torch_npu') for n in sys.modules)"
    )
    process = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr


def test_plan_only_checks_real_geometry_but_creates_nothing(tmp_path, monkeypatch):
    source, _ = write_source(tmp_path, monkeypatch)
    output = tmp_path / "missing_parent" / "prepacked"
    plan = tool.run(_args(source, output, "--plan-only"))
    assert plan["plan_only"]
    assert plan["layers"] == 1
    assert plan["payload_bytes"] > 0
    assert plan["payload_values_verified"] is False
    assert not output.parent.exists()


@pytest.mark.parametrize("shard_size", [1, 2])
def test_complete_roundtrip_is_exact_and_does_not_need_source_after_export(tmp_path, monkeypatch, shard_size):
    source, config = write_source(tmp_path, monkeypatch)
    output = tmp_path / "prepacked"
    before = _source_bytes(source.parent)
    manifest = tool.run(_args(source, output, "--experts-per-shard", str(shard_size)))
    assert _source_bytes(source.parent) == before
    assert manifest["complete"] is True
    assert manifest["tensor_bytes_verified"] is True
    assert len(manifest["layers"][0]["shards"]) == (2 + shard_size - 1) // shard_size
    assert not list(output.rglob("*.partial"))
    assert not any("pointer" in key for key in manifest)
    original = open_vq2a8_tp1_artifact(source, config, verify_tensor_hashes=True)
    _, _, reader = tool._cpu_modules()
    packed = reader.open_vq2a8_v4_v2_prepacked_artifact(output, config, verify_tensor_hashes=True)
    expected = {}
    for expert in range(2):
        for kind in ("gate_up", "down"):
            payload, spec = original.load_expert(0, expert, kind)
            expected[expert, kind] = convert_expert_payload(payload, spec)
    # Rename the test-owned fixture: loader cannot accidentally re-open input.
    source.rename(source.with_name("original_not_available"))
    for (expert, kind), reference in expected.items():
        payload, spec = packed.load_expert(0, expert, kind)
        assert spec.expert_id == expert
        for name in reference:
            assert payload[name].device.type == "cpu"
            assert torch.equal(payload[name].view(torch.uint8), reference[name].view(torch.uint8)), name


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    source, _ = write_source(tmp_path, monkeypatch)
    output = tmp_path / "prepacked"
    output.mkdir()
    sentinel = output / "user_file"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        tool.run(_args(source, output))
    assert sentinel.read_text() == "keep"


def test_output_inside_source_is_rejected(tmp_path, monkeypatch):
    source, _ = write_source(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="separate trees"):
        tool.run(_args(source, source / "prepacked", "--plan-only"))


def test_symlink_paths_are_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    link = tmp_path / "alias"
    try:
        link.symlink_to(source, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation not permitted on this host.")
    with pytest.raises(ValueError, match="symlink"):
        tool._paths(_args(link, tmp_path / "output"))


def test_failed_conversion_does_not_publish_or_modify_source(tmp_path, monkeypatch):
    source, _ = write_source(tmp_path, monkeypatch)
    output = tmp_path / "prepacked"
    before = _source_bytes(source.parent)
    _, converter, _ = tool._cpu_modules()

    def fail(*args):
        raise RuntimeError("injected conversion failure")

    monkeypatch.setattr(converter, "convert_expert_payload", fail)
    with pytest.raises(RuntimeError, match="injected conversion failure"):
        tool.run(_args(source, output))
    assert not output.exists()
    assert _source_bytes(source.parent) == before
    assert len(list(tmp_path.glob(".prepacked.partial-*"))) == 1


@pytest.mark.parametrize("changed_file", ["config", "manifest", "tensor"])
def test_source_change_during_export_prevents_publication(tmp_path, monkeypatch, changed_file):
    source, config = write_source(tmp_path, monkeypatch)
    output = tmp_path / "prepacked"
    original = tool._write_shard

    def mutate_source(*args):
        result = original(*args)
        path = {"config": config, "manifest": source / "manifest.json", "tensor": next(source.rglob("*.safetensors"))}[
            changed_file
        ]
        if changed_file == "tensor":
            # Windows forbids truncating a mapped file, but permits changing
            # an existing byte, which is precisely the race being tested.
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                value = stream.read(1)[0]
                stream.seek(-1, 2)
                stream.write(bytes([value ^ 1]))
        else:
            path.write_bytes(path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(tool, "_write_shard", mutate_source)
    with pytest.raises((RuntimeError, ValueError), match="changed; refusing to publish|SHA-256 mismatch"):
        tool.run(_args(source, output))
    assert not output.exists()
    partial = next(tmp_path.glob(".prepacked.partial-*"))
    assert not (partial / "manifest.json").exists()
    assert (partial / "failed_manifest.json").is_file()
    _, _, reader = tool._cpu_modules()
    with pytest.raises(FileNotFoundError):
        reader.open_vq2a8_v4_v2_prepacked_artifact(partial, config)


def test_unaccepted_source_reference_identity_cannot_be_laundered_by_export(tmp_path, monkeypatch):
    source, _ = write_source(tmp_path, monkeypatch)
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["producer_evidence"]["reference_identity_match"] = False
    path.write_text(json.dumps(manifest))
    output = tmp_path / "prepacked"
    with pytest.raises(ValueError, match="producer identity"):
        tool.run(_args(source, output))
    assert not output.exists()
    assert not list(tmp_path.glob(".prepacked.partial-*"))


def test_bad_source_hash_fails_before_creating_output(tmp_path, monkeypatch):
    source, _ = write_source(tmp_path, monkeypatch)
    tensor = next(source.rglob("*.safetensors"))
    data = bytearray(tensor.read_bytes())
    data[-1] ^= 1
    tensor.write_bytes(data)
    output = tmp_path / "prepacked"
    with pytest.raises(ValueError, match="SHA-256"):
        tool.run(_args(source, output))
    assert not output.exists()
    assert not list(tmp_path.glob(".prepacked.partial-*"))


def test_serialization_rechecks_raw_bytes_including_negative_zero(tmp_path, monkeypatch):
    import safetensors.torch

    original = safetensors.torch.save_file

    def corrupt_zero(tensors, filename):
        changed = dict(tensors)
        changed["weight_bias"] = torch.zeros_like(tensors["weight_bias"])
        original(changed, filename)

    monkeypatch.setattr(safetensors.torch, "save_file", corrupt_zero)
    with pytest.raises(RuntimeError, match="raw bytes differ"):
        tool._write_shard(tmp_path, 0, (0,), {"weight_bias": torch.tensor([[-0.0]])})


def test_concurrent_destination_is_not_replaced(tmp_path, monkeypatch):
    source, _ = write_source(tmp_path, monkeypatch)
    output = tmp_path / "prepacked"
    original = tool._publish_directory

    def collide(staging, destination):
        if staging.is_dir():
            destination.mkdir()
        original(staging, destination)

    monkeypatch.setattr(tool, "_publish_directory", collide)
    with pytest.raises(OSError):
        tool.run(_args(source, output))
    assert output.is_dir()
    assert not list(output.iterdir())
    assert len(list(tmp_path.glob(".prepacked.partial-*"))) == 1


def test_successfully_published_artifact_is_not_modified_after_parent_sync_failure(tmp_path, monkeypatch):
    source, config = write_source(tmp_path, monkeypatch)
    output = tmp_path / "prepacked"
    original = tool._sync_directory

    def fail_parent_sync(path):
        if path == output.parent:
            raise OSError("injected parent sync failure")
        original(path)

    monkeypatch.setattr(tool, "_sync_directory", fail_parent_sync)
    with pytest.raises(OSError, match="injected parent sync failure"):
        tool.run(_args(source, output))
    assert (output / "manifest.json").is_file()
    assert not (output / "failed_manifest.json").exists()
    _, _, reader = tool._cpu_modules()
    reader.open_vq2a8_v4_v2_prepacked_artifact(output, config, verify_tensor_hashes=True)


def test_quarantine_does_not_overwrite_existing_diagnostic_file(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "manifest.json").write_text("complete")
    (staging / "failed_manifest.json").write_text("keep old evidence")
    status = staging.stat()
    tool._invalidate_staging_manifest(staging, (status.st_dev, status.st_ino))
    assert not (staging / "manifest.json").exists()
    assert (staging / "failed_manifest.json").read_text() == "keep old evidence"
    assert next(staging.glob("failed_manifest-*.json")).read_text() == "complete"


@pytest.mark.parametrize("option", ["--threads", "--experts-per-shard"])
def test_cli_rejects_nonpositive_bounds(option):
    with pytest.raises(SystemExit):
        tool.parse_args(["--input", "source", "--output", "dest", option, "0"])
