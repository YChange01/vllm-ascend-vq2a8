# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EFG option/ABI contracts atop ABC; CPU wiring tests, never NPU acceptance.

H is intentionally a standalone compatibility probe, not a model switch. D is
off here: row reuse consumes prepared FP8 and cannot select fused tail reorder.
"""

import json
from itertools import product
from types import SimpleNamespace as NS

import pytest

from tests.ut.quantization.test_vq2a8_v4_v2_integration import config, options
from tools import serve_vq2a8_v4 as server
from tools import validate_vq2a8_v4_decoder_graph as probe
from tools import vq2a8_candidate_options as parent_options
from vllm_ascend.quantization import vq2a8_offline as offline
from vllm_ascend.quantization import vq2a8_v4_v2 as runtime
from vllm_ascend.quantization.vq2a8_abcd import validate_candidates

COMBINATIONS = tuple(product(("planned", "native"), ("vectorized", "row_reuse"), ("general", "b1_packed")))
EFG_DEFAULTS = (("runtime_guard", "signature"), ("activation_reorder", "scalar"), ("decoder_input_mode", "general"))
EFG_FEATURES = (
    ("runtime_guard", "native", "runtime_guard_version"),
    ("reorder", "row_reuse", "activation_reorder_row_reuse_version"),
    ("decoder_input_mode", "b1_packed", "decoder_input_plan_version"),
)
ABC_ABIS = frozenset(
    (
        "activation_reorder_version",
        "activation_preparation_version",
        "activation_sign_strided_version",
        "layer_validity_vectorized_version",
        "route_mapping_version",
        "select_sign_version",
    )
)


def abc_modes(guard="planned", reorder="vectorized", inputs="general"):
    return dict(
        compute_backend="v2",
        activation_preparation="sign_fused_direct",
        validity_mode="fused_vectorized",
        select_sign="fused",
        activation_tail="torch",
        route_mapping="fused",
        decoder_metadata_mode="position_template",
        runtime_guard=guard,
        activation_reorder=reorder,
        decoder_input_mode=inputs,
    )


def offline_modes(guard="planned", reorder="vectorized", inputs="general"):
    return {
        **{"v4_" + key: value for key, value in abc_modes(guard, reorder, inputs).items()},
        "v4_device_route_decode": True,
        "v4_decode_graph": "decoder",
        "v4_graph_replay_stream": "caller",
    }


def argv_options(values):
    return [part for key, value in values.items() for part in ("--" + key.replace("_", "-"), str(value))]


@pytest.fixture
def cli_paths(tmp_path):
    model = tmp_path / "model"
    artifact = model / "experts_vq_v4_v2_prepacked"
    artifact.mkdir(parents=True)
    library = tmp_path / "libvq2a8_ascendc_v4_v2.so"
    library.write_bytes(b"CPU CLI path fixture, not a compiled native library")
    return dict(model=model, artifact=artifact, library=library)


@pytest.mark.parametrize("guard,reorder,inputs", COMBINATIONS)
def test_efg_eight_combinations_offline_round_trip(tmp_path, guard, reorder, inputs):
    modes = abc_modes(guard, reorder, inputs)
    opts = options(tmp_path, **offline_modes(guard, reorder, inputs))
    assert offline.validate_offline_config(config(opts)) is opts
    for key, value in modes.items():
        assert opts.get("v4_" + key, dict(EFG_DEFAULTS).get(key, "torch")) == value
    assert "v4_activation_tail" not in opts  # D stays independently disabled.


@pytest.mark.parametrize("guard,reorder,inputs", COMBINATIONS)
def test_efg_server_and_model_probe_select_same_eight_combinations(cli_paths, capsys, guard, reorder, inputs):
    modes = abc_modes(guard, reorder, inputs)
    shared = argv_options({**cli_paths, **modes})
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
    assert json.loads(command[command.index("--compilation-config") + 1])["cudagraph_mode"] == "NONE"
    assert probe.main(shared + ["--plan-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["device_execution"] is False
    for key, value in modes.items():
        assert report[key] == getattr(args, key) == value
        assert opts.get("v4_" + key, dict(EFG_DEFAULTS).get(key, "torch")) == value


@pytest.mark.parametrize("guard,reorder,inputs", COMBINATIONS)
def test_efg_owner_forwards_selected_abis_before_artifact_load_and_layer_modes(
    monkeypatch, tmp_path, guard, reorder, inputs
):
    opts = options(tmp_path, **offline_modes(guard, reorder, inputs))
    calls = []
    artifact = NS(root=tmp_path, manifest={"format": offline.VQ2_DIRECT_TP1_FORMAT})
    monkeypatch.setattr(runtime, "load_v4_v2_library", lambda *args: calls.append("load_library"))
    monkeypatch.setattr(runtime, "require_v4_v2_features", lambda *args, **kw: calls.append((args, kw)))
    monkeypatch.setattr(offline, "artifact_format", lambda path: offline.VQ2_DIRECT_TP1_FORMAT)
    monkeypatch.setattr(offline, "open_vq2a8_tp1_artifact", lambda *args, **kw: calls.append("artifact") or artifact)
    monkeypatch.setattr(offline, "audit_offline_root", lambda path: {})
    owner = offline.OfflineMoEOwner(tmp_path, opts, NS(type="npu"))
    assert calls == [
        "load_library",
        (
            (reorder, "sign_fused_direct"),
            dict(
                validity_mode="fused_vectorized",
                route_mapping="fused",
                select_sign="fused",
                activation_tail="torch",
                runtime_guard=guard,
                decoder_input_mode=inputs,
            ),
        ),
        "artifact",
    ]

    class Layer:
        def __init__(self, actual_artifact, index, device, **kwargs):
            assert actual_artifact is artifact and index == 0 and device is owner.device
            self.kwargs = kwargs

    monkeypatch.setattr(runtime, "AscendCV4V2VQ2TP1MoE", Layer)
    layer = owner.create_layer(0)
    for name in ("runtime_guard", "activation_reorder", "validity_mode", "select_sign", "activation_tail"):
        assert layer.kwargs["v4_" + name] == abc_modes(guard, reorder, inputs)[name]
    # G is a runner-wide input adapter, not a per-MoE kernel mode.
    assert "v4_decoder_input_mode" not in layer.kwargs
    assert owner.options.get("v4_decoder_input_mode", "general") == inputs


def test_efg_defaults_unchanged_at_cli_probe_offline_and_native_gate(tmp_path, cli_paths):
    serving = server.parse_args([])
    validator = probe.parse_args(argv_options(cli_paths))
    for name, default in EFG_DEFAULTS:
        assert getattr(serving, name) == getattr(validator, name) == default
    assert serving.compute_backend == "v1" and validator.compute_backend == "v2"
    for opts in (options(tmp_path), options(tmp_path, v4_compute_backend="v2")):
        assert all("v4_" + name not in opts for name, _ in EFG_DEFAULTS)
    # No new symbols are required for the original scalar/rowwise baseline.
    runtime.require_v4_v2_features(native_ops=NS())
    args = server.parse_args(argv_options({**cli_paths, "compute_backend": "v2"}))
    command = server.build_command(args)
    opts = json.loads(command[command.index("--additional-config") + 1])["vq2a8_offline"]
    assert all("v4_" + name not in opts for name, _ in EFG_DEFAULTS)


@pytest.mark.parametrize("name", ("runtime_guard", "activation_reorder", "decoder_input_mode"))
@pytest.mark.parametrize("bad", (None, True, False, 1, "auto", "", "NATIVE"))
def test_efg_unknown_modes_rejected_by_builder_and_worker(tmp_path, name, bad):
    modes = offline_modes()
    modes["v4_" + name] = bad
    with pytest.raises(ValueError, match=name):
        options(tmp_path, **modes)
    opts = options(tmp_path, **offline_modes())
    opts["v4_" + name] = bad
    with pytest.raises(ValueError, match=name):
        offline.validate_offline_config(config(opts))


@pytest.mark.parametrize(
    "extra",
    (
        dict(runtime_guard="native", backend="v1"),
        dict(runtime_guard="native", policy="cached"),
        dict(runtime_guard="native", device_route=False),
        dict(runtime_guard="native", graph_mode="none"),
        dict(decoder_input_mode="b1_packed", graph_mode="none"),
        dict(decoder_input_mode="b1_packed", graph_mode="moe"),
        dict(decoder_input_mode="b1_packed", graph_mode="decoder", backend="v1"),
        dict(decoder_input_mode="b1_packed", graph_mode="decoder", device_route=False),
        dict(decoder_input_mode="b1_packed", graph_mode="decoder", policy="cached"),
        dict(activation_tail="fused_reorder", reorder="row_reuse"),
    ),
)
def test_efg_parent_worker_scope_checks_agree(extra):
    for check in (validate_candidates, parent_options.validate_candidates):
        with pytest.raises(ValueError):
            check(**extra)


@pytest.mark.parametrize(
    "selected,extra",
    (
        (dict(v4_runtime_guard="native"), dict(v4_compute_backend="v1")),
        (dict(v4_runtime_guard="native"), dict(v4_device_route_decode=False)),
        (dict(v4_runtime_guard="native"), dict(v4_decode_graph="none")),
        (dict(v4_activation_reorder="row_reuse"), dict(v4_compute_backend="v1")),
        (dict(v4_decoder_input_mode="b1_packed"), dict(v4_compute_backend="v1")),
        (dict(v4_decoder_input_mode="b1_packed"), dict(v4_device_route_decode=False)),
        (dict(v4_decoder_input_mode="b1_packed"), dict(v4_decode_graph="none")),
        (dict(v4_decoder_input_mode="b1_packed"), dict(v4_decode_graph="moe")),
    ),
)
def test_efg_scope_rejected_by_both_offline_layers(tmp_path, selected, extra):
    base = dict(
        v4_compute_backend="v2", v4_device_route_decode=True, v4_decode_graph="decoder", v4_graph_replay_stream="caller"
    )
    with pytest.raises(ValueError):
        options(tmp_path, **{**base, **selected, **extra})
    opts = options(tmp_path, **{**base, **selected})
    opts.update(extra)
    with pytest.raises(ValueError):
        offline.validate_offline_config(config(opts))


@pytest.mark.parametrize(
    "extra",
    (
        ["--runtime-guard", "native"],
        ["--compute-backend", "v2", "--device-route-decode", "--runtime-guard", "native"],
        ["--activation-reorder", "row_reuse"],
        ["--compute-backend", "v2", "--device-route-decode", "--decoder-input-mode", "b1_packed"],
        [
            "--compute-backend",
            "v2",
            "--device-route-decode",
            "--decode-graph",
            "moe",
            "--decoder-input-mode",
            "b1_packed",
        ],
    ),
)
def test_efg_invalid_server_scope_fails_before_loading(extra):
    with pytest.raises(SystemExit):
        server.parse_args(extra)


@pytest.mark.parametrize("guard,inputs", tuple(product(("planned", "native"), ("general", "b1_packed"))))
def test_f_plus_d_rejected_by_every_public_options_entrypoint(tmp_path, cli_paths, guard, inputs):
    modes = {**abc_modes(guard, "row_reuse", inputs), "activation_tail": "fused_reorder"}
    argv = argv_options({**cli_paths, **modes})
    with pytest.raises(SystemExit):
        server.parse_args(
            argv
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
    with pytest.raises(SystemExit):
        probe.parse_args(argv)
    kwargs = {**offline_modes(guard, "row_reuse", inputs), "v4_activation_tail": "fused_reorder"}
    with pytest.raises(ValueError, match="vectorized activation reorder"):
        options(tmp_path, **kwargs)
    opts = options(tmp_path, **offline_modes(guard, "row_reuse", inputs))
    opts["v4_activation_tail"] = "fused_reorder"
    with pytest.raises(ValueError, match="vectorized activation reorder"):
        offline.validate_offline_config(config(opts))
    with pytest.raises(ValueError, match="vectorized activation reorder"):
        runtime.require_v4_v2_features(
            "row_reuse", "sign_fused_direct", activation_tail="fused_reorder", native_ops=NS()
        )


class NativeFeatureSpy:
    """Only ABI lookup is mocked; this does not claim native operator execution."""

    def __init__(self, features, versions=None, missing=(), error=AttributeError):
        self.features = features
        self.versions = {} if versions is None else versions
        self.missing = missing
        self.error = error
        self.lookups = []

    def __getattr__(self, name):
        self.lookups.append(name)
        if name in self.missing:
            raise self.error(name)
        if name == "route_mapping":
            return lambda *args: pytest.fail("feature inspection must not execute device operators")
        if name not in self.features:
            raise AssertionError(f"Unselected ABI accessed: {name}")
        return lambda: self.versions.get(name, 1)


def abc_native_options(guard="planned", reorder="vectorized", inputs="general"):
    return dict(
        reorder=reorder,
        preparation="sign_fused_direct",
        validity_mode="fused_vectorized",
        route_mapping="fused",
        select_sign="fused",
        runtime_guard=guard,
        decoder_input_mode=inputs,
    )


@pytest.mark.parametrize("guard,reorder,inputs", COMBINATIONS)
def test_efg_eight_combinations_require_only_selected_abis(guard, reorder, inputs):
    selected = abc_native_options(guard, reorder, inputs)
    expected = ABC_ABIS | {abi for key, value, abi in EFG_FEATURES if selected[key] == value}
    native = NativeFeatureSpy(expected)
    runtime.require_v4_v2_features(**selected, native_ops=native)
    assert set(native.lookups) == expected | {"route_mapping"}
    assert "activation_tail_reorder_version" not in native.lookups
    assert not any("bias_dot" in name for name in native.lookups)  # H never enters model startup.


@pytest.mark.parametrize("key,selected,feature", EFG_FEATURES)
@pytest.mark.parametrize("bad", (None, True, False, 0, 2, "1", 1.0))
def test_efg_selected_abi_must_be_integer_one(key, selected, feature, bad):
    kwargs = {**abc_native_options(), key: selected}
    native = NativeFeatureSpy(ABC_ABIS | {feature}, versions={feature: bad})
    with pytest.raises(RuntimeError, match=feature + ".*require 1"):
        runtime.require_v4_v2_features(**kwargs, native_ops=native)


@pytest.mark.parametrize("key,selected,feature", EFG_FEATURES)
@pytest.mark.parametrize("error", (AttributeError, RuntimeError))
def test_efg_selected_missing_or_unloadable_abi_has_no_fallback(key, selected, feature, error):
    native = NativeFeatureSpy(ABC_ABIS | {feature}, missing=(feature,), error=error)
    with pytest.raises(RuntimeError, match="Rebuild.*" + feature + ".*no fallback"):
        runtime.require_v4_v2_features(**{**abc_native_options(), key: selected}, native_ops=native)


@pytest.mark.parametrize("guard", ("signature", "planned"))
def test_efg_original_and_planned_guards_do_not_require_native_host_abi(guard):
    native = NativeFeatureSpy(ABC_ABIS)
    runtime.require_v4_v2_features(**abc_native_options(guard), native_ops=native)
    assert "runtime_guard_version" not in native.lookups


@pytest.mark.parametrize("flag", ("--bias-dot-probe", "--bias-dot-mode"))
def test_h_has_no_serving_or_model_validation_switch(cli_paths, flag):
    for parser in (server.parse_args, probe.parse_args):
        with pytest.raises(SystemExit):
            parser(argv_options(cli_paths) + [flag, "vectorized"])
