# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production compressed conversion and fail-closed publication, on CPU."""

import json

import pytest
import torch

from tests.ut.quantization import test_vq2a8_runtime as direct_fixture
from tests.ut.quantization.test_vq2a8_v4_v2_prepacked import digest, matrix_spec, source_payload
from vllm_ascend.quantization import vq2a8_prepack as prepack
from vllm_ascend.quantization.vq2a8_artifact_io import _publish_directory
from vllm_ascend.quantization.vq2a8_v4_v2_layout import convert_expert_payload
from vllm_ascend.quantization.vq2a8_v4_v2_prepacked import open_vq2a8_v4_v2_prepacked_artifact


@pytest.fixture
def direct_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(direct_fixture, "_spec", matrix_spec)
    monkeypatch.setattr(direct_fixture, "_expert_payload", source_payload)
    root, config = direct_fixture._write_artifact(tmp_path)
    data = json.loads(config.read_text())
    data.update(hidden_size=4096, moe_intermediate_size=2048)
    config.write_text(json.dumps(data), encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"]["model_config_sha256"] = digest(config)
    manifest["model_layout"].update(hidden_size=4096, moe_intermediate_size=2048)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, config


def args_for(root, config, output, *extra):
    return prepack.parse_args(
        [
            "--input",
            str(root),
            "--model-config",
            str(config),
            "--output",
            str(output),
            "--experts-per-shard",
            "1",
            *extra,
        ]
    )


def test_plan_then_convert_and_read_exact_bytes(direct_artifact, tmp_path):
    root, config = direct_artifact
    before = {p: digest(p) for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "result"
    plan = prepack.run(args_for(root, config, output, "--plan-only"))
    assert plan["payload_bytes"] > 0
    assert not output.exists()
    manifest = prepack.run(args_for(root, config, output))
    assert manifest["producer"]["tool"] == "vllm_ascend.quantization.vq2a8_prepack"
    artifact = open_vq2a8_v4_v2_prepacked_artifact(output, config, verify_tensor_hashes=True)
    for expert in (0, 1):
        for kind in ("gate_up", "down"):
            actual, spec = artifact.load_expert(0, expert, kind)
            expected = convert_expert_payload(source_payload(spec, expert), spec)
            for name in expected:
                assert torch.equal(actual[name].view(torch.uint8), expected[name].view(torch.uint8))
    assert {p: digest(p) for p in before} == before


def test_existing_or_nested_output_is_rejected(direct_artifact, tmp_path):
    root, config = direct_artifact
    with pytest.raises(FileExistsError):
        prepack.run(args_for(root, config, tmp_path))
    with pytest.raises(ValueError, match="separate trees"):
        prepack.run(args_for(root, config, root / "nested"))


def test_source_change_does_not_publish_ready_artifact(direct_artifact, tmp_path, monkeypatch):
    root, config = direct_artifact
    output = tmp_path / "result"
    original = prepack._convert_layer

    def mutate_source(*args, **kwargs):
        value = original(*args, **kwargs)
        with (root / "manifest.json").open("a") as handle:
            handle.write(" ")
        return value

    monkeypatch.setattr(prepack, "_convert_layer", mutate_source)
    with pytest.raises(RuntimeError, match="changed"):
        prepack.run(args_for(root, config, output))
    assert not output.exists()
    staging = list(tmp_path.glob(".result.partial-*"))
    assert len(staging) == 1
    assert not (staging[0] / "manifest.json").exists()
    assert (staging[0] / "failed_manifest.json").is_file()


def test_atomic_publish_never_replaces_existing_directory(tmp_path):
    source, output = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    output.mkdir()
    marker = output / "user-data"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(OSError):
        _publish_directory(source, output)
    assert source.is_dir()
    assert marker.read_text() == "keep"
