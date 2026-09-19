# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent ABCD option plumbing, fail-closed ABIs and CPU-only receipts."""

import json
from itertools import product
from types import SimpleNamespace as NS

import pytest

from tests.ut.quantization.test_vq2a8_v4_v2_integration import config, options
from tools import serve_vq2a8_v4 as server
from tools import validate_vq2a8_v4_decoder_graph as probe
from tools import vq2a8_candidate_options as parent_options
from vllm_ascend.quantization import vq2a8_offline as offline
from vllm_ascend.quantization.vq2a8_abcd import ABCD_DEFAULTS, validate_candidates
from vllm_ascend.quantization.vq2a8_v4_v2 import require_v4_v2_features

COMBINATIONS = tuple(
    product(("signature", "planned"), ("fused", "fused_vectorized"), ("separate", "fused"), ("torch", "fused_reorder"))
)


def common_options():
    return dict(
        v4_compute_backend="v2",
        v4_device_route_decode=True,
        v4_decode_graph="decoder",
        v4_graph_replay_stream="caller",
        v4_activation_preparation="sign_fused_direct",
        v4_activation_reorder="vectorized",
        v4_decoder_metadata_mode="position_template",
        v4_route_mapping="fused",
    )


@pytest.mark.parametrize("guard,validity,select,tail", COMBINATIONS)
def test_all_abcd_combinations_round_trip(tmp_path, guard, validity, select, tail):
    modes = dict(runtime_guard=guard, validity_mode=validity, select_sign=select, activation_tail=tail)
    opts = options(tmp_path, **common_options(), **{"v4_" + key: value for key, value in modes.items()})
    assert offline.validate_offline_config(config(opts)) is opts
    for key, value in modes.items():
        assert opts.get("v4_" + key, ABCD_DEFAULTS.get(key)) == value


@pytest.mark.parametrize("guard,validity,select,tail", COMBINATIONS)
def test_server_and_probe_match_offline_modes(tmp_path, capsys, guard, validity, select, tail):
    model = tmp_path / "model"
    artifact = model / "experts_vq_v4_v2_prepacked"
    artifact.mkdir(parents=True)
    library = tmp_path / "libvq2a8_ascendc_v4_v2.so"
    library.write_bytes(b"CPU CLI fixture, not a native library")
    shared = [
        "--model",
        str(model),
        "--artifact",
        str(artifact),
        "--library",
        str(library),
        "--compute-backend",
        "v2",
        "--activation-preparation",
        "sign_fused_direct",
        "--activation-reorder",
        "vectorized",
        "--runtime-guard",
        guard,
        "--validity-mode",
        validity,
        "--select-sign",
        select,
        "--activation-tail",
        tail,
        "--route-mapping",
        "fused",
        "--decoder-metadata-mode",
        "position_template",
    ]
    args = server.parse_args(
        shared
        + [
            "--device-route-decode",
            "--decode-graph",
            "decoder",
            "--graph-replay-stream",
            "caller",
            "--max-model-len",
            "16",
        ]
    )
    command = server.build_command(args)
    opts = json.loads(command[command.index("--additional-config") + 1])["vq2a8_offline"]
    offline.validate_offline_config(config(opts))
    assert "--enforce-eager" in command
    assert probe.main(shared + ["--plan-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["device_execution"] is False
    for key in ("runtime_guard", "validity_mode", "select_sign", "activation_tail"):
        assert report[key] == getattr(args, key) == opts.get("v4_" + key, ABCD_DEFAULTS.get(key))


@pytest.mark.parametrize("name", ("runtime_guard", "select_sign", "activation_tail"))
@pytest.mark.parametrize("bad", (None, True, False, 1, "auto", ""))
def test_unknown_modes_rejected_by_both_config_layers(tmp_path, name, bad):
    with pytest.raises(ValueError, match=name):
        options(tmp_path, **common_options(), **{"v4_" + name: bad})
    opts = options(tmp_path, **common_options())
    opts["v4_" + name] = bad
    with pytest.raises(ValueError, match=name):
        offline.validate_offline_config(config(opts))


@pytest.mark.parametrize(
    "extra",
    (
        dict(runtime_guard="planned", backend="v1"),
        dict(runtime_guard="planned", graph_mode="none"),
        dict(select_sign="fused", policy="cached"),
        dict(select_sign="fused", device_route=False),
        dict(select_sign="fused", preparation="sign_fused"),
        dict(activation_tail="fused_reorder", preparation="rowwise"),
        dict(activation_tail="fused_reorder", reorder="scalar"),
    ),
)
def test_parent_and_worker_reject_same_out_of_scope_combinations(extra):
    for check in (validate_candidates, parent_options.validate_candidates):
        with pytest.raises(ValueError):
            check(**extra)


@pytest.mark.parametrize(
    "extra,feature",
    (
        (dict(validity_mode="fused_vectorized"), "layer_validity_vectorized_version"),
        (dict(select_sign="fused"), "select_sign_version"),
        (dict(activation_tail="fused_reorder"), "activation_tail_reorder_version"),
    ),
)
@pytest.mark.parametrize("version", (None, True, 0, 2, "1", 1))
def test_native_candidate_feature_gates(extra, feature, version):
    native = NS(
        activation_reorder_version=lambda: 1,
        activation_preparation_version=lambda: 1,
        activation_sign_strided_version=lambda: 1,
    )
    # Missing symbols must fail before loading tens of gigabytes of weights.
    with pytest.raises(RuntimeError, match="Rebuild"):
        require_v4_v2_features("vectorized", "sign_fused_direct", native_ops=native, **extra)
    setattr(native, feature, lambda: version)
    if type(version) is int and version == 1:
        require_v4_v2_features("vectorized", "sign_fused_direct", native_ops=native, **extra)
    else:
        with pytest.raises(RuntimeError, match="require 1"):
            require_v4_v2_features("vectorized", "sign_fused_direct", native_ops=native, **extra)


def test_baseline_never_requires_new_abis_or_config_keys(tmp_path):
    opts = options(tmp_path, **common_options())
    assert all("v4_" + key not in opts for key in ABCD_DEFAULTS)
    native = NS(
        activation_reorder_version=lambda: 1,
        activation_preparation_version=lambda: 1,
        activation_sign_strided_version=lambda: 1,
        layer_validity_version=lambda: 1,
    )
    require_v4_v2_features("vectorized", "sign_fused_direct", validity_mode="fused", native_ops=native)


@pytest.mark.parametrize("name,value", tuple(ABCD_DEFAULTS.items()))
def test_v4_candidate_keys_do_not_silently_leak_into_other_policies(tmp_path, name, value):
    opts = offline.offline_engine_options(tmp_path, tmp_path)["additional_config"]["vq2a8_offline"]
    opts["v4_" + name] = value
    with pytest.raises(ValueError, match="execution_policy=ascendc_v4"):
        offline.validate_offline_config(config(opts))


@pytest.mark.parametrize("location", ("top", "graph"))
@pytest.mark.parametrize("name", ("runtime_guard", "select_sign", "activation_tail", "validity_mode"))
@pytest.mark.parametrize("bad", (None, "wrong", False))
def test_model_receipt_cannot_hide_a_missing_or_different_candidate(location, name, bad):
    modes = dict(
        runtime_guard="planned",
        select_sign="fused",
        activation_tail="fused_reorder",
        validity_mode="fused_vectorized",
        route_mapping="fused",
    )
    args = NS(**modes, decoder_metadata_mode="recursive")
    receipt = dict(
        status="PASS",
        hardware_execution_verified=True,
        **modes,
        cases=[
            {"round": round_id, "prompt": prompt, "output": output}
            for round_id in range(probe.REUSE_ROUNDS)
            for prompt, output in probe.CASES
        ],
        graph={**modes, "decoder": {"replays": probe.REUSE_ROUNDS * sum(n - 1 for _, n in probe.CASES)}},
    )
    probe.validate_receipt(args, receipt)
    (receipt if location == "top" else receipt["graph"])[name] = bad
    with pytest.raises(ValueError, match="requested modes"):
        probe.validate_receipt(args, receipt)
