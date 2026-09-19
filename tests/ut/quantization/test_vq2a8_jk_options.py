# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""J/K host integration gates; never evidence of NPU execution or speed."""

import json
from itertools import product
from types import SimpleNamespace as NS

import pytest
import torch

from tests.ut.quantization.test_vq2a8_abcd_options import common_options
from tests.ut.quantization.test_vq2a8_v4_v2_integration import config, options
from tools import serve_vq2a8_v4 as serve
from tools import validate_vq2a8_v4_decoder_graph as decoder
from vllm_ascend.quantization import vq2a8_offline as offline
from vllm_ascend.quantization import vq2a8_v4_v2 as v2
from vllm_ascend.quantization.vq2a8_abcd import validate_candidates
from vllm_ascend.quantization.vq2a8_offline import validate_offline_config
from vllm_ascend.quantization.vq2a8_runtime_guard import RUNTIME_FIELDS
from vllm_ascend.quantization.vq2a8_v4_device_route import _candidate_projection, _graph_projector
from vllm_ascend.quantization.vq2a8_v4_v2 import AscendCV4V2VQ2TP1MoE as Runtime
from vllm_ascend.quantization.vq2a8_v4_v2 import require_v4_v2_features

MODES = tuple(product(("vectorized", "chunk_reuse2", "chunk_reuse4"), ("baseline", "tile_major")))


@pytest.mark.parametrize("reorder,schedule", MODES)
def test_jk_worker_config_round_trip(tmp_path, reorder, schedule):
    values = {**common_options(), "v4_activation_reorder": reorder, "v4_b1_schedule": schedule}
    selected = options(tmp_path, **values)
    assert validate_offline_config(config(selected)) is selected
    assert selected.get("v4_b1_schedule", "baseline") == schedule
    assert selected["v4_activation_reorder"] == reorder


@pytest.mark.parametrize("reorder,schedule", MODES)
def test_jk_owner_checks_selected_abi_before_weights_and_passes_modes_to_layer(
    monkeypatch, tmp_path, reorder, schedule
):
    opts = options(tmp_path, **{**common_options(), "v4_activation_reorder": reorder, "v4_b1_schedule": schedule})
    calls = []
    artifact = NS(root=tmp_path, manifest={"format": offline.VQ2_DIRECT_TP1_FORMAT})
    monkeypatch.setattr(v2, "load_v4_v2_library", lambda *a: calls.append("library"))
    monkeypatch.setattr(v2, "require_v4_v2_features", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(offline, "artifact_format", lambda *a: offline.VQ2_DIRECT_TP1_FORMAT)
    monkeypatch.setattr(offline, "open_vq2a8_tp1_artifact", lambda *a, **kw: calls.append("weights") or artifact)
    monkeypatch.setattr(offline, "audit_offline_root", lambda *a: {})
    owner = offline.OfflineMoEOwner(tmp_path, opts, NS(type="npu"))
    assert calls[0] == "library" and calls[2] == "weights"
    assert calls[1][0][0] == reorder
    assert calls[1][1].get("b1_schedule", "baseline") == schedule

    class Layer:
        def __init__(self, *args, **kwargs):
            self.options = kwargs

    monkeypatch.setattr(v2, "AscendCV4V2VQ2TP1MoE", Layer)
    layer = owner.create_layer(0)
    assert layer.options["v4_activation_reorder"] == reorder
    assert layer.options.get("v4_b1_schedule", "baseline") == schedule


@pytest.mark.parametrize("reorder,schedule", MODES)
def test_jk_cli_config_and_plan_match(tmp_path, capsys, reorder, schedule):
    artifact = tmp_path / "model" / "experts_vq_v4_v2_prepacked"
    artifact.mkdir(parents=True)
    library = tmp_path / "libvq2a8_ascendc_v4_v2.so"
    library.write_bytes(b"not a hardware library")
    shared = [
        "--model",
        str(artifact.parent),
        "--artifact",
        str(artifact),
        "--library",
        str(library),
        "--compute-backend",
        "v2",
        "--activation-reorder",
        reorder,
        "--b1-schedule",
        schedule,
    ]
    args = serve.parse_args(
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
    command = serve.build_command(args)
    selected = json.loads(command[command.index("--additional-config") + 1])["vq2a8_offline"]
    assert validate_offline_config(config(selected)) is selected
    assert decoder.main(shared + ["--plan-only"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["device_execution"] is False
    assert plan["activation_reorder"] == selected["v4_activation_reorder"] == reorder
    assert plan["b1_schedule"] == selected.get("v4_b1_schedule", "baseline") == schedule


@pytest.mark.parametrize(
    "overrides",
    [
        {"graph_mode": "none"},
        {"graph_mode": "moe"},
        {"backend": "v1"},
        {"device_route": False},
        {"activation_tail": "fused_reorder"},
        {"reorder": "row_reuse"},
        {"reorder": "scalar"},
        {"policy": "cached"},
        {"b1_schedule": True},
    ],
)
def test_jk_out_of_scope_fails_closed(overrides):
    with pytest.raises(ValueError):
        validate_candidates(
            **{"reorder": "vectorized", "b1_schedule": "tile_major", "graph_mode": "decoder", **overrides}
        )


@pytest.mark.parametrize("reorder,schedule", MODES)
@pytest.mark.parametrize("rank3", [False, True])
def test_jk_graph_dispatch_has_independent_eager_and_prefill_reference(reorder, schedule, rank3):
    calls = []
    bank = NS(
        project_vectorized=lambda *a: calls.append(("reference", a)),
        project_candidate=lambda *a: calls.append(("candidate", a)),
    )
    runtime = object.__new__(Runtime)
    runtime.v4_activation_reorder, runtime.v4_b1_schedule = reorder, schedule
    q = torch.empty((2, 1, 2048) if rank3 else (2, 2048))
    inputs = (q, object(), object(), object())
    runtime.project_v4_prepared(bank, *inputs)
    assert calls[-1] == ("reference", inputs)
    runtime.project_v4_graph_prepared(bank, *inputs)
    chunks = {"chunk_reuse2": 2, "chunk_reuse4": 4}.get(reorder, 0)
    candidate = bool(chunks or schedule != "baseline")
    assert calls[-1] == (
        ("candidate", (*inputs, chunks, int(schedule == "tile_major"))) if candidate else ("reference", inputs)
    )
    # M>1 prefill cannot accidentally invoke the new native entrypoint.
    prefill = (torch.empty(2, 4, 2048), *inputs[1:])
    runtime.project_v4_prepared(bank, *prefill)
    assert calls[-1] == ("reference", prefill)
    if candidate:
        with pytest.raises(ValueError, match="M=1"):
            runtime.project_v4_graph_prepared(bank, *prefill)
        assert runtime.v4_candidate_graph_build_calls == 1
        assert runtime.v4_candidate_reference_calls == 2


@pytest.mark.parametrize("graph", [False, True])
def test_jk_fused_select_sign_helper_keeps_reference_separate(graph):
    calls = []
    output, status = torch.ones(1), torch.ones(1, dtype=torch.int32)

    def reference(*_):
        calls.append("reference")
        return output, status

    def candidate(*_):
        calls.append("candidate")
        return output, status

    runtime = NS(
        v4_select_sign="fused",
        v4_activation_tail="torch",
        project_v4_prepared=reference,
        project_v4_graph_prepared=candidate,
    )
    preparation = NS(packed_resident=lambda *a, **kw: (None, None, None))
    statuses = []
    assert _candidate_projection(runtime, None, None, preparation, None, None, None, statuses, graph=graph) is output
    assert calls == (["candidate"] if graph else ["reference"])
    assert statuses == [status]


@pytest.mark.parametrize(
    "reorder,schedule,feature",
    [
        ("chunk_reuse2", "baseline", "activation_reorder_chunk_reuse_version"),
        ("chunk_reuse4", "baseline", "activation_reorder_chunk_reuse_version"),
        ("vectorized", "tile_major", "b1_schedule_version"),
    ],
)
@pytest.mark.parametrize("version", [None, True, 0, 2, "1", 1])
def test_jk_independent_abi_gate(reorder, schedule, feature, version):
    ops = NS(activation_reorder_version=lambda: 1)
    if version is not None:
        setattr(ops, feature, lambda: version)
    if type(version) is int and version == 1:
        require_v4_v2_features(reorder=reorder, b1_schedule=schedule, native_ops=ops)
    else:
        with pytest.raises(RuntimeError, match=feature):
            require_v4_v2_features(reorder=reorder, b1_schedule=schedule, native_ops=ops)


def test_jk_baseline_does_not_require_new_abis_and_guard_tracks_schedule():
    require_v4_v2_features(reorder="vectorized", native_ops=NS(activation_reorder_version=lambda: 1))
    assert ("v4_b1_schedule", "baseline") in RUNTIME_FIELDS


@pytest.mark.parametrize(
    "modes",
    [
        dict(v4_activation_reorder="chunk_reuse2"),
        dict(v4_activation_reorder="chunk_reuse4"),
        dict(v4_b1_schedule="tile_major"),
    ],
)
def test_jk_missing_graph_dispatch_never_silently_uses_reference(modes):
    with pytest.raises(RuntimeError, match="no fallback"):
        _graph_projector(NS(**modes), lambda *args: None)
