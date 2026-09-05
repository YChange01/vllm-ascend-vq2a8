# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import math
import runpy
import struct
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.quantization.vq2a8_artifact import (
    VQ2LayerSummary,
    VQ2MatrixSpec,
    inspect_layer_artifact,
    inspect_vq2_directory,
    load_matrix_tensors,
    load_model_layout,
    parse_matrix_name,
    read_safetensors_header,
    validate_matrix_payload,
    validate_model_layout,
)


def _metadata(layer: int, expert: int, kind: str) -> dict[str, object]:
    if kind == "gate_up":
        rows, columns, true_columns = 4, 8, 6
    else:
        rows, columns, true_columns = 6, 4, 2
    row_group_size = 2
    group_size = 3
    return {
        "rows": rows,
        "cols": columns,
        "n_row_tiles": rows // row_group_size,
        "n_col_tiles": math.ceil(columns / group_size),
        "row_group_size": row_group_size,
        "group_size": group_size,
        "K": 16,
        "index_bits": 4,
        "vector_len": 2,
        "n_vectors": rows * columns // 2,
        "n_elements": rows * columns,
        "orig_shape": [rows, true_columns],
        "norm_dim": 0,
        "enable_perm": True,
        "enable_norm": True,
        "enable_rht": True,
        "rht_block_size": 4,
        "rht_true_columns": true_columns,
    }


def _pack_indices(values: torch.Tensor) -> torch.Tensor:
    words = torch.zeros((values.numel() + 7) // 8, dtype=torch.int64)
    for position, value in enumerate(values.tolist()):
        words[position // 8] |= int(value) << (position % 8 * 4)
    return words.to(torch.int32)


def _matrix_tensors(metadata: dict[str, object]) -> dict[str, torch.Tensor]:
    columns = int(metadata["cols"])
    row_tiles = int(metadata["n_row_tiles"])
    column_tiles = int(metadata["n_col_tiles"])
    num_vectors = int(metadata["n_vectors"])
    codebooks = (
        torch.arange(column_tiles * row_tiles * 16 * 2, dtype=torch.float32)
        .remainder(31)
        .reshape(column_tiles, row_tiles, 16, 2)
    )
    return {
        "packed_indices": _pack_indices(torch.arange(num_vectors).remainder(16)),
        "codebooks": codebooks.to(torch.float8_e4m3fn),
        "perm": torch.roll(torch.arange(columns, dtype=torch.int32), shifts=-1),
        "weight_scale": torch.linspace(-1.0, 1.5, columns),
        "weight_bias": torch.linspace(-0.5, 0.5, columns),
        "rht_sign": torch.where(
            torch.arange(columns).remainder(3) == 0,
            torch.tensor(-1, dtype=torch.int8),
            torch.tensor(1, dtype=torch.int8),
        ),
    }


def _write_layer(
    root: Path,
    layer: int = 0,
    expert_ids: tuple[int, ...] = (0,),
    *,
    drop_tensor: str | None = None,
    extra_tensor: bool = False,
    codebook_dtype: torch.dtype = torch.float8_e4m3fn,
) -> None:
    metadata: dict[str, dict[str, object]] = {}
    tensors: dict[str, torch.Tensor] = {}
    for expert in expert_ids:
        for kind in ("gate_up", "down"):
            name = f"{layer}.mlp.experts.{expert}.{kind}"
            metadata[name] = _metadata(layer, expert, kind)
            for field, tensor in _matrix_tensors(metadata[name]).items():
                if field == "codebooks":
                    tensor = tensor.to(codebook_dtype)
                key = f"{name}.{field}"
                if key != drop_tensor:
                    tensors[key] = tensor
    if extra_tensor:
        tensors["unexpected"] = torch.zeros(1)
    (root / f"experts_vq_layer_{layer}.json").write_text(json.dumps(metadata), encoding="utf-8")
    save_file(tensors, root / f"experts_vq_layer_{layer}.safetensors")


def test_inspect_layer_and_payload_accept_canonical_schema(tmp_path: Path) -> None:
    _write_layer(tmp_path)
    summary = inspect_layer_artifact(tmp_path, 0)
    assert summary.layer_index == 0
    assert summary.expert_ids == (0,)
    assert summary.matrix_count == 2
    assert summary.tensor_count == 12

    tensors, spec = load_matrix_tensors(tmp_path, 0, 0, "gate_up")
    payload = validate_matrix_payload(tensors, spec)
    assert payload.name == "0.mlp.experts.0.gate_up"
    assert payload.scale_min == -1.0
    assert payload.scale_max == 1.5


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("K", 8, "expected K=16"),
        ("norm_dim", 1, "norm_dim must be 0"),
        ("n_row_tiles", 3, "rows=.*n_row_tiles"),
        ("n_col_tiles", 2, r"ceil\(cols / group_size\)"),
        ("n_vectors", 15, "n_vectors=.*n_elements"),
        ("rht_block_size", 3, "power of two"),
        ("rht_true_columns", 9, "exceeds cols"),
        ("orig_shape", [4, 8], "orig_shape"),
        ("orig_shape", [True, 5], "orig_shape"),
    ],
)
def test_matrix_spec_rejects_inconsistent_metadata(field: str, value: object, match: str) -> None:
    metadata = _metadata(0, 0, "gate_up")
    metadata[field] = value
    with pytest.raises(ValueError, match=match):
        VQ2MatrixSpec.from_dict("0.mlp.experts.0.gate_up", metadata)


def test_matrix_spec_rejects_unknown_and_disabled_rht_fields() -> None:
    metadata = _metadata(0, 0, "gate_up")
    metadata["unknown"] = 1
    with pytest.raises(ValueError, match="metadata fields differ.*unknown"):
        VQ2MatrixSpec.from_dict("0.mlp.experts.0.gate_up", metadata)

    metadata = _metadata(0, 0, "gate_up")
    metadata["enable_rht"] = False
    with pytest.raises(ValueError, match="metadata fields differ.*rht_block_size"):
        VQ2MatrixSpec.from_dict("0.mlp.experts.0.gate_up", metadata)


def test_inspect_layer_rejects_missing_extra_and_wrong_dtype(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()
    _write_layer(missing, drop_tensor="0.mlp.experts.0.down.rht_sign")
    with pytest.raises(ValueError, match="missing.*rht_sign"):
        inspect_layer_artifact(missing, 0)

    extra = tmp_path / "extra"
    extra.mkdir()
    _write_layer(extra, extra_tensor=True)
    with pytest.raises(ValueError, match="unexpected.*unexpected"):
        inspect_layer_artifact(extra, 0)

    wrong_dtype = tmp_path / "wrong_dtype"
    wrong_dtype.mkdir()
    _write_layer(wrong_dtype, codebook_dtype=torch.float32)
    with pytest.raises(ValueError, match="dtype=F32.*expected dtype=F8_E4M3"):
        inspect_layer_artifact(wrong_dtype, 0)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("perm", torch.tensor([0, 0, 2, 3, 4, 5, 6, 7]), "perm is not a bijection"),
        ("rht_sign", torch.tensor([1, -1, 0, 1, 1, 1, 1, 1]), "rht_sign"),
        ("weight_scale", torch.tensor([1.0, 1.0, float("nan"), 1.0, 1.0, 1.0, 1.0, 1.0]), "non-finite"),
    ],
)
def test_payload_rejects_invalid_values(field: str, replacement: torch.Tensor, match: str) -> None:
    spec = VQ2MatrixSpec.from_dict("0.mlp.experts.0.gate_up", _metadata(0, 0, "gate_up"))
    tensors = _matrix_tensors(_metadata(0, 0, "gate_up"))
    tensors[field] = replacement.to(tensors[field].dtype)
    with pytest.raises(ValueError, match=match):
        validate_matrix_payload(tensors, spec)


def test_payload_rejects_nonfinite_codebook() -> None:
    spec = VQ2MatrixSpec.from_dict("0.mlp.experts.0.down", _metadata(0, 0, "down"))
    tensors = _matrix_tensors(_metadata(0, 0, "down"))
    codebooks = tensors["codebooks"].float()
    codebooks[0, 0, 0, 0] = float("nan")
    tensors["codebooks"] = codebooks.to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="codebooks contain non-finite"):
        validate_matrix_payload(tensors, spec)


def test_payload_rejects_wrong_dtype_and_shape() -> None:
    spec = VQ2MatrixSpec.from_dict("0.mlp.experts.0.down", _metadata(0, 0, "down"))
    tensors = _matrix_tensors(_metadata(0, 0, "down"))
    tensors["perm"] = tensors["perm"].to(torch.int64)
    with pytest.raises(ValueError, match="expected dtype=torch.int32"):
        validate_matrix_payload(tensors, spec)

    tensors = _matrix_tensors(_metadata(0, 0, "down"))
    tensors["weight_bias"] = tensors["weight_bias"][:-1]
    with pytest.raises(ValueError, match=r"expected dtype=torch.float32, shape=\[4\]"):
        validate_matrix_payload(tensors, spec)


def test_directory_requires_paired_contiguous_layers(tmp_path: Path) -> None:
    _write_layer(tmp_path, layer=0)
    _write_layer(tmp_path, layer=2)
    with pytest.raises(ValueError, match="sequence has gaps.*1"):
        inspect_vq2_directory(tmp_path)

    (tmp_path / "experts_vq_layer_2.json").unlink()
    with pytest.raises(ValueError, match=r"not paired.*missing_metadata=\[2\]"):
        inspect_vq2_directory(tmp_path)


def test_model_layout_checks_hash_and_routed_expert_counts(tmp_path: Path) -> None:
    config = {
        "num_hidden_layers": 3,
        "num_hash_layers": 1,
        "n_routed_experts": 2,
        "hidden_size": 6,
        "moe_intermediate_size": 2,
        "quantization_config": {"quant_method": "vq2a8", "experts_path": "experts_vq"},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    layout = load_model_layout(config_path)
    summaries = [
        VQ2LayerSummary(0, (0,), 2, 12, 1, (4, 6), (6, 2)),
        VQ2LayerSummary(1, (0, 1), 4, 24, 1, (4, 6), (6, 2)),
        VQ2LayerSummary(2, (0, 1), 4, 24, 1, (4, 6), (6, 2)),
    ]
    validate_model_layout(summaries, layout, require_all_layers=True)

    with pytest.raises(ValueError, match="Layer 1 has expert IDs"):
        validate_model_layout(
            [VQ2LayerSummary(1, (0,), 2, 12, 1, (4, 6), (6, 2))],
            layout,
            require_all_layers=False,
        )
    with pytest.raises(ValueError, match="do not match config layers"):
        validate_model_layout(summaries[:-1], layout, require_all_layers=True)

    with pytest.raises(ValueError, match="gate_up shape"):
        validate_model_layout(
            [VQ2LayerSummary(0, (0,), 2, 12, 1, (8, 4), (6, 2))],
            layout,
            require_all_layers=False,
        )


def test_layer_rejects_wrong_layer_and_noncontiguous_expert_id(tmp_path: Path) -> None:
    wrong_layer = tmp_path / "wrong_layer"
    wrong_layer.mkdir()
    _write_layer(wrong_layer, layer=1)
    (wrong_layer / "experts_vq_layer_1.json").rename(wrong_layer / "experts_vq_layer_0.json")
    (wrong_layer / "experts_vq_layer_1.safetensors").rename(wrong_layer / "experts_vq_layer_0.safetensors")
    with pytest.raises(ValueError, match="belongs to layer 1, expected layer 0"):
        inspect_layer_artifact(wrong_layer, 0)

    noncontiguous = tmp_path / "noncontiguous"
    noncontiguous.mkdir()
    _write_layer(noncontiguous, expert_ids=(1,))
    with pytest.raises(ValueError, match="contiguous from zero"):
        inspect_layer_artifact(noncontiguous, 0)


def test_layer_rejects_geometry_that_differs_across_experts(tmp_path: Path) -> None:
    _write_layer(tmp_path, expert_ids=(0, 1))
    metadata_path = tmp_path / "experts_vq_layer_0.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["0.mlp.experts.1.gate_up"]["group_size"] = 4
    metadata["0.mlp.experts.1.gate_up"]["n_col_tiles"] = 2
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="gate_up metadata is inconsistent"):
        inspect_layer_artifact(tmp_path, 0)


def test_layer_rejects_incompatible_gate_up_and_down_shapes(tmp_path: Path) -> None:
    _write_layer(tmp_path)
    metadata_path = tmp_path / "experts_vq_layer_0.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    down = metadata["0.mlp.experts.0.down"]
    down.update(
        {
            "rows": 4,
            "n_row_tiles": 2,
            "n_vectors": 8,
            "n_elements": 16,
            "orig_shape": [4, 2],
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible MoE matrix shapes"):
        inspect_layer_artifact(tmp_path, 0)


def _run_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], arguments: list[str]) -> dict:
    tool_path = next(
        candidate
        for parent in Path(__file__).resolve().parents
        if (candidate := parent / "tools" / "validate_vq2a8_artifact.py").is_file()
    )
    monkeypatch.setattr(sys, "argv", [str(tool_path), *arguments])
    runpy.run_path(str(tool_path), run_name="__main__")
    return json.loads(capsys.readouterr().out)


def test_cli_reports_header_only_as_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    model_path = tmp_path / "model"
    experts_path = model_path / "experts_vq"
    experts_path.mkdir(parents=True)
    _write_layer(experts_path)
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 1,
                "num_hash_layers": 1,
                "n_routed_experts": 2,
                "hidden_size": 6,
                "moe_intermediate_size": 2,
                "quantization_config": {"quant_method": "vq2a8"},
            }
        ),
        encoding="utf-8",
    )

    report = _run_cli(monkeypatch, capsys, [str(experts_path)])
    assert report["validation"] == {
        "complete": False,
        "layers_checked": 1,
        "layers_expected": 1,
        "matrix_headers_checked": 2,
        "matrix_payloads_checked": 0,
        "matrix_payloads_in_checked_layers": 2,
        "scope": "header_only",
    }

    report = _run_cli(monkeypatch, capsys, [str(experts_path), "--all-payloads"])
    assert report["validation"]["scope"] == "all_payloads"
    assert report["validation"]["complete"] is True
    assert report["validation"]["matrix_payloads_checked"] == 2


def test_cli_distinguishes_verified_codec_from_external_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    experts_path = tmp_path / "experts_vq"
    experts_path.mkdir()
    _write_layer(experts_path)
    codec_path = tmp_path / "producer.py"
    codec_path.write_bytes(b"canonical producer\n")

    verified = _run_cli(monkeypatch, capsys, [str(experts_path), "--producer-codec", str(codec_path)])
    assert verified["producer_codec"] == {
        "path": str(codec_path.resolve()),
        "sha256": "fc6dea449f4908bc59f7de78fdcefd02bb1ac50b3bc1e5f9c6548c1cddf161db",
        "verified_from_file": True,
    }

    claimed_hash = "A" * 64
    claimed = _run_cli(
        monkeypatch,
        capsys,
        [str(experts_path), "--claimed-producer-codec-sha256", claimed_hash],
    )
    assert claimed["producer_codec"] == {
        "path": None,
        "sha256": claimed_hash.lower(),
        "verified_from_file": False,
    }


def _write_raw_safetensors(path: Path, header: str | dict[str, object], payload: bytes) -> None:
    header_bytes = header.encode() if isinstance(header, str) else json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)


@pytest.mark.parametrize(
    ("header", "payload", "match"),
    [
        (
            {"x": {"dtype": "I8", "shape": [True], "data_offsets": [0, 1]}},
            b"\0",
            "Invalid safetensors shape",
        ),
        (
            {
                "x": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]},
                "y": {"dtype": "I8", "shape": [1], "data_offsets": [2, 3]},
            },
            b"\0\0\0",
            "has a gap",
        ),
        (
            {
                "x": {"dtype": "I8", "shape": [2], "data_offsets": [0, 2]},
                "y": {"dtype": "I8", "shape": [1], "data_offsets": [1, 2]},
            },
            b"\0\0",
            "overlaps",
        ),
        (
            {"x": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]}},
            b"\0\0",
            "trailing bytes",
        ),
        (
            {
                "__metadata__": {"producer": 1},
                "x": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]},
            },
            b"\0",
            "Invalid safetensors __metadata__",
        ),
        (
            {"x": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1], "extra": 1}},
            b"\0",
            "must contain exactly",
        ),
    ],
)
def test_safetensors_header_rejects_structural_corruption(
    tmp_path: Path, header: dict[str, object], payload: bytes, match: str
) -> None:
    path = tmp_path / "bad.safetensors"
    _write_raw_safetensors(path, header, payload)
    with pytest.raises(ValueError, match=match):
        read_safetensors_header(path)


def test_safetensors_header_rejects_duplicate_tensor_key(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.safetensors"
    entry = '{"dtype":"I8","shape":[1],"data_offsets":[0,1]}'
    _write_raw_safetensors(path, f'{{"x":{entry},"x":{entry}}}', b"\0")
    with pytest.raises(ValueError, match="Duplicate JSON key: 'x'"):
        read_safetensors_header(path)


def test_matrix_name_rejects_leading_zero_aliases() -> None:
    with pytest.raises(ValueError, match="Non-canonical"):
        parse_matrix_name("00.mlp.experts.0.gate_up")
    with pytest.raises(ValueError, match="Non-canonical"):
        parse_matrix_name("0.mlp.experts.01.down")
