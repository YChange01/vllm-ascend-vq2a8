# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP1 packed-zN contracts and real CPU-only CLI publication regression tests."""

import json
import subprocess
import sys

import pytest
import torch
from safetensors import safe_open

from tests.ut.quantization.test_vq2a8_tp2_layout import _canonical
from tests.ut.quantization.test_vq2a8_tp2_repack_cli import (
    CPU_ENTRYPOINT,
    FIELD_DTYPES,
    REPO,
    assert_no_staging,
    digest,
    require_success,
    snapshot,
    write_model,
)
from tools.repack_vq2a8_tp2 import parse_args as parse_tp2_args
from vllm_ascend.quantization.vq2a8_tp1_zn_runtime import open_vq2a8_tp1_zn_artifact
from vllm_ascend.quantization.vq2a8_tp2_layout import repack_matrix_tp2, repack_matrix_zn
from vllm_ascend.quantization.vq2a8_zn_contract import activation_semantics, communication_contract, zn_format

SCRIPT = REPO / "tools/repack_vq2a8_tp1_zn.py"


def run_cli(source, output, *flags, inject=None):
    entrypoint = CPU_ENTRYPOINT
    if inject is not None:
        entrypoint = entrypoint.replace(
            "runpy.run_path(sys.argv[0], run_name='__main__')",
            "namespace = runpy.run_path(sys.argv[0], run_name='_tp1_cli_test')\n"
            "cli_globals = namespace['common'].main.__globals__\n" + inject + "\nraise SystemExit(namespace['main']())",
        )
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-X",
            "utf8",
            "-c",
            entrypoint,
            str(SCRIPT),
            "--input",
            str(source),
            "--output",
            str(output),
            *flags,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
    )


@pytest.mark.parametrize("tp_size", [None, 0, 3, True, False, 1.0, "1"])
def test_zn_contract_rejects_ambiguous_tp_size(tp_size):
    for function in (zn_format, communication_contract, activation_semantics):
        with pytest.raises(ValueError):
            function(tp_size)


def test_zn_contract_is_fresh_and_distinguishes_full_k_from_local_a8():
    first = activation_semantics(1)
    first["preparation_order"].clear()
    first["dummy_metadata"]["weight_scale"] = 7
    assert activation_semantics(1)["dummy_metadata"]["weight_scale"] == 0
    assert "full_K_dynamic_fp8" in activation_semantics(1)["preparation_order"]
    assert "rank_local_dynamic_fp8" in activation_semantics(2)["preparation_order"]
    assert "no TP collective" in activation_semantics(1)["down_aggregation"]
    assert activation_semantics(1)["tp1_bitwise_equivalent"] is False
    assert zn_format(1) == "vq2a8_zn_tp1_v1" and zn_format(2) == "vq2a8_zn_tp2_v1"


@pytest.mark.parametrize("rank", [1, 2, -1, True, False, 0.0, "0", None])
def test_tp1_zn_rejects_invalid_rank(rank):
    source, spec, _ = _canonical()
    with pytest.raises(ValueError):
        repack_matrix_zn(source, spec, rank=rank, tp_size=1)


@pytest.mark.parametrize("kind", ["gate_up", "down"])
@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_wrapper_still_has_identical_tensor_and_metadata_contract(kind, rank):
    source, spec, _ = _canonical(kind)
    direct, direct_metadata = repack_matrix_tp2(source, spec, rank)
    shared, shared_metadata = repack_matrix_zn(source, spec, rank=rank, tp_size=2)
    assert direct_metadata == shared_metadata
    for key in direct:
        assert torch.equal(direct[key].view(torch.uint8), shared[key].view(torch.uint8))


@pytest.mark.parametrize("experts_per_shard", [1, 2, 32])
def test_tp1_zn_cli_publishes_full_rank0_bytes_with_bounded_shards_and_hashes(tmp_path, experts_per_shard):
    model, source = write_model(tmp_path)
    output = tmp_path / "tp1 zn"
    before = snapshot(model)
    result = run_cli(source, output, "--experts-per-shard", str(experts_per_shard))
    require_success(result)
    assert "VQ2_TP1_STAGE=done" in result.stdout and "rank1_gib" not in result.stdout
    assert "VQ2_TP2_STAGE=" not in result.stdout
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["format"] == "vq2a8_zn_tp1_v1"
    assert manifest["tp_size"] == 1 and manifest["tp_ranks"] == [0]
    assert manifest["complete"] is True and manifest["tensor_values_verified"] is True
    assert manifest["runtime_compatible"] is False and manifest["tp1_a8_bitwise_equivalent"] is False
    assert manifest["communication"] == communication_contract(1)
    assert manifest["producer"]["tool"] == "tools/repack_vq2a8_tp1_zn.py"
    producer_files = {entry["file"] for entry in manifest["producer"]["files"]}
    assert {"repack_vq2a8_tp1_zn.py", "repack_vq2a8_tp2.py", "vq2a8_zn_contract.py"} <= producer_files
    shard_bytes = 0
    assert len(manifest["shards"]) == (2 + experts_per_shard - 1) // experts_per_shard
    for shard in manifest["shards"]:
        assert shard["rank"] == 0 and shard["file"].startswith("tp1/rank0/")
        assert digest(output / shard["file"]) == shard["sha256"]
        assert digest(output / shard["metadata_file"]) == shard["metadata_sha256"]
        metadata = json.loads((output / shard["metadata_file"]).read_text())
        shard_bytes += shard["payload_bytes"]
        with safe_open(output / shard["file"], framework="pt", device="cpu") as handle:
            for matrix in metadata["matrices"]:
                k = 512 if matrix["kind"] == "gate_up" else 256
                assert matrix["canonical_shape"] == matrix["logical_shape"] == matrix["packed_shape"] == [512, k]
                assert matrix["padding_columns"] == 0
                assert matrix["tile_valid_counts"] == [256] * (k // 256)
                assert matrix["activation_semantics"] == activation_semantics(1)
                for field, expected_dtype in FIELD_DTYPES.items():
                    entry = matrix["tensors"][field]
                    assert entry["dtype"] == handle.get_slice(entry["key"]).get_dtype() == expected_dtype
                    assert entry["shape"] == handle.get_slice(entry["key"]).get_shape()
                order = handle.get_tensor(matrix["tensors"]["activation_order"]["key"])
                assert torch.equal(order.sort().values, torch.arange(k))
    assert manifest["per_rank_payload_bytes"] == [shard_bytes]
    assert manifest["total_payload_upper_bytes"] == manifest["per_rank_payload_upper_bytes"] == shard_bytes
    # Exercise the real writer -> strict reader seam, not a hand-made manifest.
    artifact = open_vq2a8_tp1_zn_artifact(output, model / "config.json")
    loaded = {
        expert: matrices for shard in artifact.iter_rank_shards(0, device="cpu") for expert, matrices in shard.items()
    }
    assert set(loaded) == {0, 1}
    for matrices in loaded.values():
        for payload, spec in matrices.values():
            assert spec.canonical_shape == spec.logical_shape == spec.packed_shape
            assert spec.padding_columns == 0 and spec.tp_rank == 0
            assert payload["activation_order"].numel() == spec.columns
    assert snapshot(model) == before
    assert not (output / "tp2").exists()


def test_tp1_zn_cli_dry_run_has_no_output_and_does_not_change_legacy_defaults(tmp_path):
    model, source = write_model(tmp_path, experts=1)
    output = tmp_path / "missing parent" / "tp1"
    before = snapshot(model)
    result = run_cli(source, output, "--dry-run")
    require_success(result)
    assert not output.parent.exists() and snapshot(model) == before
    plan = json.loads(
        next(line.split("=", 1)[1] for line in result.stdout.splitlines() if line.startswith("VQ2_TP1_PLAN="))
    )
    assert plan["tp_size"] == 1 and plan["dry_run"] is True
    legacy = parse_tp2_args(["--input", str(source), "--output", str(output)])
    assert not hasattr(legacy, "tp_size")  # Existing TP2 CLI does not silently switch format.


def test_tp1_zn_cli_refuses_existing_output_and_preserves_tp2_sibling(tmp_path):
    model, source = write_model(tmp_path, experts=1)
    old_tp2 = model / "experts_vq_tp2_zn"
    old_tp2.mkdir()
    (old_tp2 / "keep.bin").write_bytes(b"existing TP2 payload")
    before = snapshot(model)
    result = run_cli(source, old_tp2)
    assert result.returncode != 0 and "refusing to overwrite" in result.stderr
    assert snapshot(model) == before


def test_tp1_zn_cli_bad_payload_cleans_private_staging_without_source_writes(tmp_path):
    model, source = write_model(tmp_path, invalid_last_permutation=True)
    before = snapshot(model)
    output = tmp_path / "bad"
    result = run_cli(source, output, "--experts-per-shard", "1")
    assert result.returncode != 0 and "perm" in result.stderr.lower()
    assert_no_staging(output)
    assert snapshot(model) == before


def test_tp1_zn_cli_source_change_refuses_publication(tmp_path):
    _, source = write_model(tmp_path, experts=1)
    output = tmp_path / "changed"
    result = run_cli(
        source,
        output,
        inject="""
real_write = cli_globals['_write_shard']
source_path = cli_globals['Path'](sys.argv[sys.argv.index('--input') + 1])
def mutate(*args, **kwargs):
    result = real_write(*args, **kwargs)
    path = source_path / 'experts_vq_layer_0.json'
    path.write_bytes(path.read_bytes() + b'\\n')
    return result
cli_globals['_write_shard'] = mutate
""",
    )
    assert result.returncode != 0 and "Source layer 0 changed during conversion" in result.stderr
    assert_no_staging(output)


def test_tp1_zn_cli_help_without_runtime_dependencies(tmp_path):
    result = run_cli(tmp_path / "unused", tmp_path / "unused-output", "--help")
    require_success(result)
    assert "TP1 packed-zN" in result.stdout and "--dry-run" in result.stdout
