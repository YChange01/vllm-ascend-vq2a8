# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""I configuration integration; CPU checks do not certify NPU/model execution."""

import argparse
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.ut.quantization.test_vq2a8_abcd_options import common_options
from tests.ut.quantization.test_vq2a8_v4_v2_integration import config, options
from tools import serve_vq2a8_v4 as serve
from tools import vq2a8_candidate_options as parent_options
from vllm_ascend.quantization import vq2a8_offline as offline
from vllm_ascend.quantization import vq2a8_v4_v2 as v2
from vllm_ascend.quantization.vq2a8_abcd import validate_candidates


def i_options(**extra):
    return {**common_options(), "v4_select_sign": "fused", "v4_swiglu_mode": "fused_select_sign", **extra}


def serving_flags(tmp_path):
    artifact = tmp_path / "model" / "experts_vq_v4_v2_prepacked"
    artifact.mkdir(parents=True)
    library = tmp_path / "libvq2a8_ascendc_v4_v2.so"
    library.write_bytes(b"CPU fixture, not a hardware library")
    return [
        "--model",
        str(artifact.parent),
        "--artifact",
        str(artifact),
        "--library",
        str(library),
        "--compute-backend",
        "v2",
        "--device-route-decode",
        "--decode-graph",
        "decoder",
        "--graph-replay-stream",
        "caller",
        "--max-model-len",
        "16",
        "--activation-preparation",
        "sign_fused_direct",
        "--activation-reorder",
        "vectorized",
        "--select-sign",
        "fused",
    ]


@pytest.mark.parametrize("mode", ("torch", "fused_select_sign"))
def test_i_worker_and_serving_round_trip_preserve_default_keys(tmp_path, mode):
    selected = options(tmp_path, **i_options(v4_swiglu_mode=mode))
    assert offline.validate_offline_config(config(selected)) is selected
    args = serve.parse_args(serving_flags(tmp_path) + ["--swiglu-mode", mode])
    command = serve.build_command(args)
    serving = json.loads(command[command.index("--additional-config") + 1])["vq2a8_offline"]
    assert offline.validate_offline_config(config(serving)) is serving
    for value in (selected, serving):
        assert value.get("v4_swiglu_mode", "torch") == mode
        assert ("v4_swiglu_mode" in value) is (mode != "torch")
    assert "--enforce-eager" in command


@pytest.mark.parametrize("mode", ("torch", "fused_select_sign"))
def test_i_owner_checks_abi_before_weights_and_forwards_only_nondefault(monkeypatch, tmp_path, mode):
    selected = options(tmp_path, **i_options(v4_swiglu_mode=mode))
    calls = []
    artifact = NS(root=tmp_path, manifest={"format": offline.VQ2_DIRECT_TP1_FORMAT})
    monkeypatch.setattr(v2, "load_v4_v2_library", lambda *a: calls.append("library"))
    monkeypatch.setattr(v2, "require_v4_v2_features", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(offline, "artifact_format", lambda *a: offline.VQ2_DIRECT_TP1_FORMAT)
    monkeypatch.setattr(offline, "open_vq2a8_tp1_artifact", lambda *a, **kw: calls.append("weights") or artifact)
    monkeypatch.setattr(offline, "audit_offline_root", lambda *a: {})
    owner = offline.OfflineMoEOwner(tmp_path, selected, NS(type="npu"))
    assert calls[0] == "library" and calls[2] == "weights"
    assert calls[1].get("swiglu_mode", "torch") == mode
    assert ("swiglu_mode" in calls[1]) is (mode != "torch")

    class Layer:
        def __init__(self, *args, **kwargs):
            self.options = kwargs

    monkeypatch.setattr(v2, "AscendCV4V2VQ2TP1MoE", Layer)
    kwargs = owner.create_layer(0).options
    assert kwargs.get("v4_swiglu_mode", "torch") == mode
    assert ("v4_swiglu_mode" in kwargs) is (mode != "torch")


@pytest.mark.parametrize("bad", (None, True, False, 1, "fused", "", "auto"))
def test_i_invalid_modes_fail_in_generator_and_worker(tmp_path, bad):
    with pytest.raises(ValueError, match="swiglu_mode"):
        options(tmp_path, **i_options(v4_swiglu_mode=bad))
    selected = options(tmp_path, **i_options())
    selected["v4_swiglu_mode"] = bad
    with pytest.raises(ValueError, match="swiglu_mode"):
        offline.validate_offline_config(config(selected))


@pytest.mark.parametrize(
    "extra",
    (
        {"backend": "v1"},
        {"policy": "cached"},
        {"device_route": False},
        {"graph_mode": "none"},
        {"graph_mode": "moe"},
        {"select_sign": "separate"},
        {"preparation": "rowwise"},
        {"preparation": "sign_fused"},
        {"activation_tail": "fused_reorder"},
        {"reorder": "scalar"},
        {"reorder": "row_reuse"},
        {"reorder": "chunk_reuse2"},
        {"reorder": "chunk_reuse4"},
        {"b1_schedule": "tile_major"},
    ),
)
def test_i_parent_and_worker_contracts_reject_out_of_scope(extra):
    selected = dict(swiglu_mode="fused_select_sign", select_sign="fused", graph_mode="decoder")
    selected.update(extra)
    for check in (validate_candidates, parent_options.validate_candidates):
        with pytest.raises(ValueError):
            check(**selected)


@pytest.mark.parametrize("graph_mode", (None, "decoder"))
def test_i_internal_and_decoder_graph_contracts_allowed(graph_mode):
    validate_candidates(swiglu_mode="fused_select_sign", select_sign="fused", graph_mode=graph_mode)


@pytest.mark.parametrize(
    "extra",
    (
        ["--compute-backend", "v1"],
        ["--decode-graph", "none"],
        ["--decode-graph", "moe"],
        ["--select-sign", "separate"],
        ["--activation-preparation", "rowwise"],
        ["--activation-tail", "fused_reorder"],
        ["--activation-reorder", "scalar"],
        ["--activation-reorder", "row_reuse"],
        ["--activation-reorder", "chunk_reuse2"],
        ["--activation-reorder", "chunk_reuse4"],
        ["--b1-schedule", "tile_major"],
    ),
)
def test_i_serving_scope_matches_worker(tmp_path, extra):
    with pytest.raises(SystemExit):
        serve.parse_args(serving_flags(tmp_path) + ["--swiglu-mode", "fused_select_sign"] + extra)


def test_i_serving_requires_device_route(tmp_path):
    argv = serving_flags(tmp_path)
    argv.remove("--device-route-decode")
    with pytest.raises(SystemExit):
        serve.parse_args(argv + ["--swiglu-mode", "fused_select_sign"])


def test_i_shared_parser_and_validation_inherit_switch():
    parser = argparse.ArgumentParser()
    parent_options.add_candidate_arguments(parser)
    assert parser.parse_args([]).swiglu_mode == "torch"
    args = parser.parse_args(["--swiglu-mode", "fused_select_sign", "--select-sign", "fused"])
    args.compute_backend = "v2"
    args.activation_preparation = "sign_fused_direct"
    args.activation_reorder = "vectorized"
    parent_options.validate_candidate_args(args, graph_mode="decoder", device_route=True)
    args.activation_reorder = "chunk_reuse2"
    with pytest.raises(ValueError):
        parent_options.validate_candidate_args(args, graph_mode="decoder", device_route=True)


def test_i_default_config_does_not_leak_to_other_policies(tmp_path):
    selected = offline.offline_engine_options(tmp_path, tmp_path)["additional_config"]["vq2a8_offline"]
    assert "v4_swiglu_mode" not in selected
    selected["v4_swiglu_mode"] = "torch"
    with pytest.raises(ValueError, match="execution_policy=ascendc_v4"):
        offline.validate_offline_config(config(selected))


def test_i_serving_stays_standard_library_only():
    tree = ast.parse(Path(serve.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module]
        else:
            continue
        assert all(name.split(".")[0] in sys.stdlib_module_names for name in names)


@pytest.mark.parametrize("version", (None, True, False, 0, 2, "1", 1))
def test_i_independent_native_abi_is_strict(version):
    native = NS(
        activation_reorder_version=lambda: 1,
        activation_preparation_version=lambda: 1,
        activation_sign_strided_version=lambda: 1,
        select_sign_version=lambda: 1,
    )
    if version is not None:
        native.swiglu_select_sign_version = lambda: version
    if type(version) is int and version == 1:
        v2.require_v4_v2_features(
            "vectorized", "sign_fused_direct", select_sign="fused", swiglu_mode="fused_select_sign", native_ops=native
        )
    else:
        with pytest.raises(RuntimeError, match="swiglu_select_sign_version"):
            v2.require_v4_v2_features(
                "vectorized",
                "sign_fused_direct",
                select_sign="fused",
                swiglu_mode="fused_select_sign",
                native_ops=native,
            )


def test_i_default_does_not_require_swiglu_abi():
    v2.require_v4_v2_features(native_ops=NS(activation_reorder_version=lambda: 1))
