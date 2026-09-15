# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU configuration/loader contracts; not NPU execution certification."""

import json
from types import SimpleNamespace as NS

import pytest
import torch

from tools import serve_vq2a8_v4 as server
from vllm_ascend.quantization import vq2a8_offline as offline


def options(tmp_path, **extra):
    return offline.offline_engine_options(
        tmp_path / "model",
        tmp_path / "artifact",
        execution_policy="ascendc_v4",
        ascendc_library=tmp_path / "libvq2a8_ascendc_v4_v2.so",
        ascendc_sha256="a" * 64,
        **extra,
    )["additional_config"]["vq2a8_offline"]


def config(opts):
    return NS(
        additional_config={"vq2a8_offline": opts},
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        model_config=NS(enforce_eager=True, quantization=None, dtype=torch.bfloat16, max_model_len=16),
        quant_config=None,
        scheduler_config=NS(max_num_seqs=1, max_num_batched_tokens=16),
        compilation_config=NS(mode=0, cudagraph_mode=0),
        cache_config=NS(gpu_memory_utilization=0.9, kv_cache_memory_bytes=256 * 1024**2),
        load_config=NS(load_format="safetensors"),
    )


def test_backend_selection_retains_v4_topology_and_default(tmp_path):
    baseline = options(tmp_path)
    assert "v4_compute_backend" not in baseline
    candidate = options(tmp_path, v4_compute_backend="v2")
    assert offline.validate_offline_config(config(candidate))["v4_compute_backend"] == "v2"
    assert candidate.pop("v4_compute_backend") == "v2"
    assert candidate == baseline


@pytest.mark.parametrize("backend", [None, True, False, 1, "auto", "v3", "V2"])
def test_unknown_backend_rejected_at_both_entrypoints(tmp_path, backend):
    with pytest.raises(ValueError, match="v4_compute_backend"):
        options(tmp_path, v4_compute_backend=backend)
    opts = options(tmp_path)
    opts["v4_compute_backend"] = backend
    with pytest.raises(ValueError, match="v4_compute_backend"):
        offline.validate_offline_config(config(opts))


def test_candidate_is_not_standalone_v2_or_v3_policy(tmp_path):
    with pytest.raises(ValueError, match="v4_compute_backend"):
        offline.offline_engine_options(tmp_path, tmp_path, execution_policy="cached", v4_compute_backend="v2")
    opts = options(tmp_path, v4_compute_backend="v2", v4_device_route_decode=True, v4_decode_graph="moe")
    assert offline.validate_offline_config(config(opts))["execution_policy"] == "ascendc_v4"


def test_server_selects_candidate_without_changing_other_options(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "experts_vq_ascend_v2").mkdir()
    library = tmp_path / "libvq2a8_ascendc_v4_v2.so"
    library.write_bytes(b"CPU command fixture only")
    args = server.parse_args(
        [
            "--model",
            str(model),
            "--library",
            str(library),
            "--compute-backend",
            "v2",
            "--physical-npu",
            "1",
            "--device-route-decode",
            "--decode-graph",
            "moe",
            "--graph-replay-stream",
            "caller",
            "--max-model-len",
            "16",
            "--kv-cache-mib",
            "256",
            "--memory-fraction",
            "1.0",
            "--reserve-gib",
            "3",
        ]
    )
    command = server.build_command(args)
    opts = json.loads(command[command.index("--additional-config") + 1])["vq2a8_offline"]
    assert opts["v4_compute_backend"] == "v2" and opts["execution_policy"] == "ascendc_v4"
    assert opts["ascendc_library"] == str(library.resolve())
    assert opts["v4_decode_graph"] == "moe" and opts["v4_graph_replay_stream"] == "caller"
    assert "--enforce-eager" in command
    assert server.server_environment(args, {})["ASCEND_RT_VISIBLE_DEVICES"] == "1"
    assert offline.validate_offline_config(config(opts))["v4_compute_backend"] == "v2"


@pytest.mark.parametrize(
    "backend,filename",
    [
        ("v2", "libvq2a8_ascendc.so"),
        ("v2", "libvq2a8_ascendc_v2.so"),
        ("v2", "libvq2a8_ascendc_v3.so"),
        ("v1", "libvq2a8_ascendc_v4_v2.so"),
    ],
)
def test_server_rejects_wrong_library_before_spawning(tmp_path, backend, filename):
    model = tmp_path / "model"
    model.mkdir()
    (model / "experts_vq_ascend_v2").mkdir()
    library = tmp_path / filename
    library.write_bytes(b"fixture")
    args = server.parse_args(["--model", str(model), "--library", str(library), "--compute-backend", backend])
    with pytest.raises(ValueError, match="compute-backend v2"):
        server.build_command(args)


def test_owner_loads_only_candidate_and_selects_candidate_class(monkeypatch, tmp_path):
    from vllm_ascend.quantization import vq2a8_ascendc, vq2a8_v4_v2

    opts = options(tmp_path, v4_compute_backend="v2", v4_device_route_decode=True)
    loaded = []
    monkeypatch.setattr(vq2a8_v4_v2, "load_v4_v2_library", lambda path, sha: loaded.append((path, sha)) or {"v2": True})

    def forbidden(*args, **kwargs):
        raise AssertionError("candidate must not load or execute old native backend")

    monkeypatch.setattr(vq2a8_ascendc, "load_pinned_library", forbidden)
    monkeypatch.setattr(offline, "artifact_format", lambda path: offline.VQ2_DIRECT_TP1_FORMAT)
    artifact = NS(root=tmp_path / "artifact", manifest={"format": offline.VQ2_DIRECT_TP1_FORMAT})
    monkeypatch.setattr(offline, "open_vq2a8_tp1_artifact", lambda *args, **kwargs: artifact)
    monkeypatch.setattr(offline, "audit_offline_root", lambda path: {})
    owner = offline.OfflineMoEOwner(tmp_path / "model", opts, NS(type="npu"))
    assert loaded == [(opts["ascendc_library"], opts["ascendc_sha256"])]
    assert owner.native_library == {"v2": True}

    class Candidate:
        def __init__(self, actual_artifact, index, device, **kwargs):
            assert actual_artifact is artifact and index == 0

    monkeypatch.setattr(vq2a8_v4_v2, "AscendCV4V2VQ2TP1MoE", Candidate)
    assert isinstance(owner.create_layer(0), Candidate)


def test_model_planner_uses_candidate_layout(monkeypatch, tmp_path):
    from vllm_ascend.quantization import vq2a8_execution_v4, vq2a8_v4_v2

    owner = offline.OfflineMoEOwner.__new__(offline.OfflineMoEOwner)
    owner.options = {"v4_compute_backend": "v2"}
    header = NS(expert_ids=(0,))
    calls = []
    owner.artifact = NS(layers={0: header})
    owner.layers = {
        0: NS(
            layer=header,
            initialize_resident=lambda **kw: calls.append(kw),
            check_resident_integrity=lambda: {},
            v4_report=lambda: {
                "layer_index": 0,
                "source_format": offline.VQ2_DIRECT_TP1_FORMAT,
                "startup_conversion": True,
                "preload_host_convert_s": 0.0,
                "preload_host_read_s": 0.0,
                "preload_host_validate_s": 0.0,
                "preload_h2d_s": 0.0,
            },
        )
    }

    def candidate_plan(layers, budget):
        assert layers == [header] and budget == 1234
        return {"layer_plans": {0: {"planned_bytes": 1000}}, "layout": "v2_zn", "planned_bytes": 1000}

    monkeypatch.setattr(vq2a8_v4_v2, "v4_v2_resident_plan", candidate_plan)
    monkeypatch.setattr(vq2a8_execution_v4, "packed_resident_plan", lambda *args: pytest.fail("old layout planner"))
    owner._configure_v4_residency({"budget_bytes": 1234})
    assert calls == [{"budget_bytes": 1000}]
    assert owner.cache_plan["layout"] == "v2_zn" and owner.cache_plan["preload_complete"] is True
