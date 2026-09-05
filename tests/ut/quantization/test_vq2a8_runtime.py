# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.quantization.vq2a8_artifact import (
    VQ2_CONSUMER_REFERENCE_COMMIT,
    VQ2MatrixSpec,
)
from vllm_ascend.quantization.vq2a8_reference import (
    deepseek_v4_swiglu_reference,
    vq2a8_repacked_matmul_reference,
)
from vllm_ascend.quantization.vq2a8_repack import (
    VQ2_DIRECT_TP1_FORMAT,
    pack_repacked_indices,
)
from vllm_ascend.quantization.vq2a8_runtime import (
    VQ2_TP1_AXES,
    VQ2_TP1_COMMUNICATION,
    VQ2_TP1_DTYPES,
    VQ2_TP1_FIELDS,
    VQ2_TP1_PACKING,
    VQ2_TP1_TRANSFORMS,
    open_vq2a8_tp1_artifact,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec(kind: str) -> VQ2MatrixSpec:
    rows, columns, block_size = (4, 4, 4) if kind == "gate_up" else (4, 2, 2)
    metadata = {
        "rows": rows,
        "cols": columns,
        "n_row_tiles": 2,
        "n_col_tiles": math.ceil(columns / 2),
        "row_group_size": 2,
        "group_size": 2,
        "K": 16,
        "index_bits": 4,
        "vector_len": 2,
        "n_vectors": rows * columns // 2,
        "n_elements": rows * columns,
        "orig_shape": [rows, columns],
        "norm_dim": 0,
        "enable_perm": True,
        "enable_norm": True,
        "enable_rht": True,
        "rht_block_size": block_size,
        "rht_true_columns": columns,
    }
    return VQ2MatrixSpec.from_dict(f"0.mlp.experts.0.{kind}", metadata)


def _expert_payload(spec: VQ2MatrixSpec, expert_id: int) -> dict[str, torch.Tensor]:
    output_pairs = spec.rows // 2
    codes = (
        (torch.arange(output_pairs * spec.columns, dtype=torch.int64).reshape(output_pairs, spec.columns) + expert_id)
        .remainder(16)
        .to(torch.uint8)
    )
    codebooks = torch.empty(
        (spec.column_tiles, spec.row_tiles, 16, 2),
        dtype=torch.float32,
    )
    for column_tile in range(spec.column_tiles):
        for row_tile in range(spec.row_tiles):
            for code in range(16):
                codebooks[column_tile, row_tile, code] = torch.tensor(
                    [
                        (column_tile * 32 + row_tile * 8 + code + expert_id) / 16,
                        (column_tile * 32 + row_tile * 8 + code + expert_id + 1) / 16,
                    ]
                )
    columns = torch.arange(spec.columns)
    return {
        "packed_indices": pack_repacked_indices(codes),
        "codebooks": codebooks.to(torch.float8_e4m3fn),
        "codebook_tile_ids": torch.div(columns, spec.group_size, rounding_mode="floor").to(torch.uint8),
        "weight_scale": (0.5 + columns.float() / 8).to(torch.float32),
        "weight_bias": torch.full((spec.columns,), expert_id / 16, dtype=torch.float32),
        "rht_sign": torch.where(columns.remainder(2) == 0, 1, -1).to(torch.int8),
    }


def _matrix_layout(spec: VQ2MatrixSpec, stacked: dict[str, torch.Tensor], kind: str) -> dict:
    return {
        "canonical_shape": [spec.rows, spec.columns],
        "original_shape": list(spec.original_shape),
        "row_group_size": spec.row_group_size,
        "group_size": spec.group_size,
        "row_tiles": spec.row_tiles,
        "column_tiles": spec.column_tiles,
        "rht_block_size": spec.rht_block_size,
        "rht_true_columns": spec.rht_true_columns,
        "tensors": {
            field: {
                "dtype": VQ2_TP1_DTYPES[field],
                "shape": list(stacked[f"{kind}_{field}"].shape),
                "axes": list(VQ2_TP1_AXES[field]),
            }
            for field in VQ2_TP1_FIELDS
        },
    }


def _write_artifact(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "model"
    artifact = model / "experts_vq_ascend_v2"
    rank = artifact / "tp1" / "rank0"
    rank.mkdir(parents=True)
    config_path = model / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "num_hidden_layers": 1,
                "num_hash_layers": 0,
                "n_routed_experts": 2,
                "hidden_size": 4,
                "moe_intermediate_size": 2,
                "quantization_config": {"quant_method": "vq2a8", "experts_path": "experts_vq"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    specs = {kind: _spec(kind) for kind in ("gate_up", "down")}
    stacked: dict[str, torch.Tensor] = {}
    for kind, spec in specs.items():
        payloads = [_expert_payload(spec, expert_id) for expert_id in range(2)]
        for field in VQ2_TP1_FIELDS:
            stacked[f"{kind}_{field}"] = torch.stack([payload[field] for payload in payloads])
    tensor_path = rank / "experts_vq_layer_0.safetensors"
    save_file(stacked, tensor_path)
    tensor_sha256 = _sha256(tensor_path)
    layer_manifest = {
        "schema_version": 1,
        "format": VQ2_DIRECT_TP1_FORMAT,
        "layer": 0,
        "tp_size": 1,
        "tp_rank": 0,
        "expert_ids": [0, 1],
        "num_expected_experts": 2,
        "complete": True,
        "consumer_reference_commit": VQ2_CONSUMER_REFERENCE_COMMIT,
        "source": {},
        "output": {
            "tensor_file": tensor_path.name,
            "tensor_sha256": tensor_sha256,
        },
        "layout": {kind: _matrix_layout(spec, stacked, kind) for kind, spec in specs.items()},
        "transforms": VQ2_TP1_TRANSFORMS,
        "communication": VQ2_TP1_COMMUNICATION,
    }
    metadata_path = rank / "experts_vq_layer_0.json"
    metadata_path.write_text(json.dumps(layer_manifest, sort_keys=True), encoding="utf-8")
    root_manifest = {
        "schema_version": 1,
        "format": VQ2_DIRECT_TP1_FORMAT,
        "tp_size": 1,
        "tp_ranks": [0],
        "consumer_reference_commit": VQ2_CONSUMER_REFERENCE_COMMIT,
        "repacker": {
            "tool": "tools/repack_vq2a8_tp1.py",
            "contract_revision": 1,
            "tool_sha256": "0" * 64,
        },
        "producer_evidence": {"reference_identity_match": True},
        "source": {
            "model_config_file": "config.json",
            "model_config_sha256": _sha256(config_path),
            "checksum_manifest": None,
        },
        "output": {"artifact_root": ".", "path_semantics": "relative_to_manifest"},
        "model_layout": {
            "num_hidden_layers": 1,
            "num_hash_layers": 0,
            "num_routed_experts": 2,
            "hidden_size": 4,
            "moe_intermediate_size": 2,
        },
        "packing": VQ2_TP1_PACKING,
        "communication": VQ2_TP1_COMMUNICATION,
        "storage": {},
        "layers_selected": [0],
        "layers_expected": 1,
        "complete": True,
        "layers": [
            {
                "layer": 0,
                "expert_count": 2,
                "tensor_file": "tp1/rank0/experts_vq_layer_0.safetensors",
                "tensor_sha256": tensor_sha256,
                "metadata_file": "tp1/rank0/experts_vq_layer_0.json",
                "metadata_sha256": _sha256(metadata_path),
            }
        ],
    }
    (artifact / "manifest.json").write_text(json.dumps(root_manifest, sort_keys=True), encoding="utf-8")
    return artifact, config_path


def test_runtime_reader_binds_complete_artifact_and_loads_one_expert(tmp_path: Path) -> None:
    artifact_path, config_path = _write_artifact(tmp_path)

    artifact = open_vq2a8_tp1_artifact(
        artifact_path,
        config_path,
        require_reference_identity=True,
        verify_tensor_hashes=True,
    )
    tensors, spec = artifact.load_expert(0, 1, "gate_up")

    assert artifact.model_layout.hidden_size == 4
    assert tuple(artifact.layers) == (0,)
    assert spec.name == "0.mlp.experts.1.gate_up"
    assert spec.original_shape == (4, 4)
    assert set(tensors) == set(VQ2_TP1_FIELDS)
    torch.testing.assert_close(tensors["weight_bias"], torch.full((4,), 1 / 16))


def test_runtime_reader_selected_payload_executes_repacked_reference(tmp_path: Path) -> None:
    artifact_path, config_path = _write_artifact(tmp_path)
    artifact = open_vq2a8_tp1_artifact(artifact_path, config_path)
    gate_payload, gate_spec = artifact.load_expert(0, 1, "gate_up")
    down_payload, down_spec = artifact.load_expert(0, 1, "down")
    activation = torch.tensor([[0.5, -1.0, 0.25, 2.0]], dtype=torch.bfloat16)

    gate_up = vq2a8_repacked_matmul_reference(
        activation,
        gate_payload,
        gate_spec,
        dynamic_a8=True,
    ).to(torch.bfloat16)
    activated = deepseek_v4_swiglu_reference(gate_up, 10.0)
    output = vq2a8_repacked_matmul_reference(
        activated,
        down_payload,
        down_spec,
        dynamic_a8=True,
    )

    assert gate_up.shape == (1, 4)
    assert activated.shape == (1, 2)
    assert output.shape == (1, 4)
    assert bool(torch.isfinite(output).all())


def test_runtime_reader_rejects_model_config_hash_mismatch(tmp_path: Path) -> None:
    artifact_path, config_path = _write_artifact(tmp_path)
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="model config SHA-256 mismatch"):
        open_vq2a8_tp1_artifact(artifact_path, config_path)


def test_runtime_reader_payload_hash_is_optional_but_enforceable(tmp_path: Path) -> None:
    artifact_path, config_path = _write_artifact(tmp_path)
    tensor_path = artifact_path / "tp1" / "rank0" / "experts_vq_layer_0.safetensors"
    data = bytearray(tensor_path.read_bytes())
    data[-1] ^= 1
    tensor_path.write_bytes(data)

    open_vq2a8_tp1_artifact(artifact_path, config_path, verify_tensor_hashes=False)
    with pytest.raises(ValueError, match="layer 0 tensors SHA-256 mismatch"):
        open_vq2a8_tp1_artifact(artifact_path, config_path, verify_tensor_hashes=True)


def test_runtime_reader_rejects_partial_or_inconsistent_root(tmp_path: Path) -> None:
    artifact_path, config_path = _write_artifact(tmp_path)
    manifest_path = artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="complete is inconsistent"):
        open_vq2a8_tp1_artifact(artifact_path, config_path)


def test_runtime_reader_rejects_empty_layer_selection(tmp_path: Path) -> None:
    artifact_path, config_path = _write_artifact(tmp_path)
    manifest_path = artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["layers_selected"] = []
    manifest["layers"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="must be non-empty"):
        open_vq2a8_tp1_artifact(artifact_path, config_path)


def test_runtime_reader_rejects_wrong_tensor_axes_before_payload_read(tmp_path: Path) -> None:
    artifact_path, config_path = _write_artifact(tmp_path)
    metadata_path = artifact_path / "tp1" / "rank0" / "experts_vq_layer_0.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["layout"]["gate_up"]["tensors"]["packed_indices"]["axes"] = ["expert", "wrong", "axis"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    manifest_path = artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["layers"][0]["metadata_sha256"] = _sha256(metadata_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="dtype or axes contract is invalid"):
        open_vq2a8_tp1_artifact(artifact_path, config_path)


@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        (None, torch.nn.functional.silu(torch.tensor([[2.0, -3.0]])) * torch.tensor([[20.0, -30.0]])),
        (10.0, torch.nn.functional.silu(torch.tensor([[2.0, -3.0]])) * torch.tensor([[10.0, -10.0]])),
    ],
)
def test_deepseek_v4_swiglu_reference_clamp_contract(limit: float | None, expected: torch.Tensor) -> None:
    gate_up = torch.tensor([[2.0, -3.0, 20.0, -30.0]])
    torch.testing.assert_close(deepseek_v4_swiglu_reference(gate_up, limit), expected)


@pytest.mark.parametrize("bad_limit", [True, -1.0, "10"])
def test_deepseek_v4_swiglu_reference_rejects_invalid_limit(bad_limit: object) -> None:
    with pytest.raises(ValueError, match="swiglu_limit"):
        deepseek_v4_swiglu_reference(torch.ones(1, 4), bad_limit)  # type: ignore[arg-type]
