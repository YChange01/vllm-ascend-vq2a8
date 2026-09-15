# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent optimization switches: CPU wiring contracts, not NPU proof."""

import json
from types import SimpleNamespace as NS

import pytest

from tests.ut.quantization.test_vq2a8_v4_v2_integration import config, options
from tools import serve_vq2a8_v4 as server
from vllm_ascend.quantization import vq2a8_offline as offline
from vllm_ascend.quantization import vq2a8_optimization as optimization
from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_v4_v2 import AscendCV4V2VQ2TP1MoE, require_v4_v2_features


def test_old_native_library_still_supports_unmodified_baseline():
    require_v4_v2_features(native_ops=NS())
    with pytest.raises(RuntimeError, match="Rebuild"):
        require_v4_v2_features("vectorized", native_ops=NS())
    with pytest.raises(RuntimeError, match="Rebuild"):
        require_v4_v2_features(preparation="fused", native_ops=NS())


@pytest.mark.parametrize("version", [None, True, 0, 2, "1"])
def test_invalid_native_feature_version_never_falls_back(version):
    native = NS(activation_reorder_version=lambda: version, activation_preparation_version=lambda: version)
    with pytest.raises(RuntimeError, match="require 1"):
        require_v4_v2_features("vectorized", "fused", native_ops=native)


def test_independent_native_feature_checks():
    require_v4_v2_features("vectorized", native_ops=NS(activation_reorder_version=lambda: 1))
    require_v4_v2_features(preparation="fused", native_ops=NS(activation_preparation_version=lambda: 1))


@pytest.mark.parametrize("reorder", ["scalar", "vectorized"])
@pytest.mark.parametrize("preparation", ["rowwise", "fused"])
@pytest.mark.parametrize("graph", ["none", "moe", "decoder"])
def test_independent_v4_perf_options(tmp_path, reorder, preparation, graph):
    opts = options(
        tmp_path,
        v4_compute_backend="v2",
        v4_activation_reorder=reorder,
        v4_activation_preparation=preparation,
        v4_decode_graph=graph,
        v4_device_route_decode=True,
        v4_graph_replay_stream="owner" if graph == "none" else "caller",
    )
    assert offline.validate_offline_config(config(opts)) == opts
    assert opts.get("v4_activation_reorder", "scalar") == reorder
    assert opts.get("v4_activation_preparation", "rowwise") == preparation


@pytest.mark.parametrize("key", ["v4_activation_reorder", "v4_activation_preparation"])
@pytest.mark.parametrize("value", [None, True, 1, "auto", "v3"])
def test_unknown_activation_option_rejected(tmp_path, key, value):
    with pytest.raises(ValueError, match=key):
        options(tmp_path, v4_compute_backend="v2", **{key: value})
    opts = options(tmp_path, v4_compute_backend="v2")
    opts[key] = value
    with pytest.raises(ValueError, match=key):
        offline.validate_offline_config(config(opts))


@pytest.mark.parametrize("extra", [{"v4_activation_reorder": "vectorized"}, {"v4_activation_preparation": "fused"}])
def test_new_native_optimizations_never_leak_to_v1(tmp_path, extra):
    with pytest.raises(ValueError, match="v4_compute_backend=v2"):
        options(tmp_path, **extra)


def test_decoder_graph_scope_is_checked_before_allocation(tmp_path):
    with pytest.raises(ValueError, match="caller"):
        options(tmp_path, v4_device_route_decode=True, v4_decode_graph="decoder")
    opts = options(tmp_path, v4_device_route_decode=True, v4_decode_graph="decoder", v4_graph_replay_stream="caller")
    conf = config(opts)
    conf.model_config.max_model_len = 17
    with pytest.raises(ValueError, match="max_model_len"):
        offline.validate_offline_config(conf)


def test_cli_wires_all_three_candidates_and_keeps_engine_eager(tmp_path):
    model = tmp_path / "model"
    (model / "experts_vq_ascend_v2").mkdir(parents=True)
    library = tmp_path / "libvq2a8_ascendc_v4_v2.so"
    library.write_bytes(b"fixture only")
    args = server.parse_args(
        [
            "--model",
            str(model),
            "--library",
            str(library),
            "--compute-backend",
            "v2",
            "--activation-reorder",
            "vectorized",
            "--activation-preparation",
            "fused",
            "--device-route-decode",
            "--decode-graph",
            "decoder",
            "--graph-replay-stream",
            "caller",
            "--max-model-len",
            "16",
            "--kv-cache-mib",
            "256",
            "--reserve-gib",
            "3",
        ]
    )
    command = server.build_command(args)
    opts = json.loads(command[command.index("--additional-config") + 1])["vq2a8_offline"]
    assert opts["v4_activation_reorder"] == "vectorized"
    assert opts["v4_activation_preparation"] == "fused"
    assert opts["v4_decode_graph"] == "decoder"
    # Custom scoped decoder graph, not unverified whole-engine graph support.
    assert "--enforce-eager" in command
    assert json.loads(command[command.index("--compilation-config") + 1])["cudagraph_mode"] == "NONE"
    offline.validate_offline_config(config(opts))


@pytest.mark.parametrize(
    "argv",
    [
        ["--activation-reorder", "vectorized"],
        ["--activation-preparation", "fused"],
        ["--decode-graph", "decoder", "--device-route-decode"],
        ["--decode-graph", "decoder", "--device-route-decode", "--graph-replay-stream", "caller"],
    ],
)
def test_invalid_cli_combinations_fail(argv):
    with pytest.raises(SystemExit):
        server.parse_args(argv)


def test_runtime_project_selects_only_explicit_method():
    runtime = AscendCV4V2VQ2TP1MoE.__new__(AscendCV4V2VQ2TP1MoE)
    calls = []
    bank = NS(
        project=lambda *values: calls.append(("scalar", values)),
        project_vectorized=lambda *values: calls.append(("vectorized", values)),
    )
    values = tuple(object() for _ in range(4))
    runtime.project_v4_prepared(bank, *values)
    runtime.v4_activation_reorder = "vectorized"
    runtime.project_v4_prepared(bank, *values)
    assert calls == [("scalar", values), ("vectorized", values)]


def test_runtime_reference_factory_unchanged():
    runtime = AscendCV4V2VQ2TP1MoE.__new__(AscendCV4V2VQ2TP1MoE)
    callback = object()
    preparation = runtime.make_v4_preparation(compact=True, validity=callback)
    assert type(preparation) is RowwiseVQ2A8Preparation
    assert preparation.compact and preparation.validity is callback


def test_optimization_switch_uses_candidate_factory_and_keeps_preset_cache(monkeypatch):
    created = []

    def factory(**kwargs):
        value = NS(kwargs=kwargs)
        created.append(value)
        return value

    runtime = NS(make_v4_preparation=factory)
    monkeypatch.setattr(optimization, "FastMoEState", lambda *a, **kw: NS(retain=lambda flag: None))
    optimization.configure_runtime(runtime, "batched")
    first = runtime._row_preparation
    assert first.kwargs["compact"] and callable(first.kwargs["validity"])
    optimization.configure_runtime(runtime, None)
    assert runtime._row_preparation is created[0]
    optimization.configure_runtime(runtime, "batched")
    assert runtime._row_preparation is first and len(created) == 2
