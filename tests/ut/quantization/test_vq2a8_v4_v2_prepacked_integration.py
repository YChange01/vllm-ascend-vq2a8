# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU prepacked wiring/ownership tests, not NPU or startup-speed acceptance."""

import json
from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

from tests.ut.quantization.test_vq2a8_v4_v2_integration import config, options
from tests.ut.quantization.test_vq2a8_v4_v2_runtime import layer_header, source_payload
from tools import serve_vq2a8_v4 as server
from tools import validate_vq2a8_v4_decoder_graph as decoder_probe
from tools import validate_vq2a8_v4_v2 as projection_probe
from vllm_ascend.quantization import vq2a8_execution as execution
from vllm_ascend.quantization import vq2a8_execution_v4 as v4
from vllm_ascend.quantization import vq2a8_offline as offline
from vllm_ascend.quantization import vq2a8_v4_v2 as v2
from vllm_ascend.quantization import vq2a8_v4_v2_prepacked as prepacked
from vllm_ascend.quantization.vq2a8_repack import VQ2_DIRECT_TP1_FORMAT
from vllm_ascend.quantization.vq2a8_zn_contract import VQ2_TP1_ZN_FORMAT


def forbidden(*args, **kwargs):
    raise AssertionError("prepacked loading must not convert or open a source/other-format artifact")


def prepacked_header(direct, payload):
    """Actual reader header; no original-format tensor_shapes field."""
    header = prepacked.V4V2PrepackedLayer(
        layer_index=direct.layer_index,
        expert_ids=direct.expert_ids,
        specs=direct.specs,
        source_tensor_shapes=direct.tensor_shapes,
        shards=(),
        source_tensor_sha256="a" * 64,
        source_metadata_sha256="b" * 64,
    )
    assert header.projection_shapes("gate_up") == tuple(tuple(payload[field].shape) for field in v2.V4_V2_FIELDS)
    return header


@pytest.mark.parametrize("k", [2048, 4096])
def test_prepacked_planner_uses_converted_shapes_and_keeps_device_budget(k):
    payload, spec = source_payload(k)
    direct = layer_header(payload, spec, experts=(0, 2))
    converted = v2.convert_expert_payload(payload, spec)
    header = prepacked_header(direct, converted)
    assert not hasattr(header, "tensor_shapes")
    for kind in ("gate_up", "down"):
        assert v2._projection_shapes(header, kind) == v2._projection_shapes(direct, kind)
    assert v2.v4_v2_resident_plan([header], 10**10) == v2.v4_v2_resident_plan([direct], 10**10)


@pytest.mark.parametrize("source_format", [VQ2_DIRECT_TP1_FORMAT, prepacked.V4_V2_PREPACKED_FORMAT])
def test_v4_v2_resident_artifact_hook_accepts_only_supported_sources(source_format):
    runtime = v2.AscendCV4V2VQ2TP1MoE.__new__(v2.AscendCV4V2VQ2TP1MoE)
    runtime.artifact = NS(manifest={"format": source_format})
    runtime.layer = NS(v4_v2_prepacked=source_format == prepacked.V4_V2_PREPACKED_FORMAT)
    runtime._validate_resident_artifact()


@pytest.mark.parametrize("source_format", [VQ2_TP1_ZN_FORMAT, "vq2a8_unknown", None])
def test_v4_v2_rejects_old_v3_or_unknown_layout(source_format):
    runtime = v2.AscendCV4V2VQ2TP1MoE.__new__(v2.AscendCV4V2VQ2TP1MoE)
    runtime.artifact = NS(manifest={"format": source_format})
    with pytest.raises(ValueError):
        runtime._validate_resident_artifact()


def test_v1_resident_hook_does_not_accept_prepacked_candidate():
    runtime = v4.AscendCV4VQ2TP1MoE.__new__(v4.AscendCV4VQ2TP1MoE)
    runtime.artifact = NS(manifest={"format": prepacked.V4_V2_PREPACKED_FORMAT})
    with pytest.raises(ValueError):
        runtime._validate_resident_artifact()


def test_prepacked_manifest_cannot_relabel_a_direct_layer():
    runtime = v2.AscendCV4V2VQ2TP1MoE.__new__(v2.AscendCV4V2VQ2TP1MoE)
    runtime.artifact = NS(manifest={"format": prepacked.V4_V2_PREPACKED_FORMAT})
    runtime.layer = NS()
    with pytest.raises(ValueError, match="dedicated validated reader"):
        runtime._validate_resident_artifact()


def initialize_cpu_runtime(monkeypatch, source_format, layer, load):
    """Execute real residency logic with CPU transfers and an explicit bank stub."""
    artifact = NS(manifest={"format": source_format}, load_expert=load)

    def init(self, artifact, index, device, **kwargs):
        self.artifact, self.layer, self.layer_index, self.device = artifact, layer, index, torch.device(device)
        self._cache = OrderedDict()
        self.cache_experts = 1
        self.cache_loads = self.cache_hits = self.cache_peak_bytes = 0
        self._resident_bytes = self.evictions = self.h2d_bytes = 0
        self.progress = self.verbose_experts = self.measurement_mode = False
        self.timing = dict.fromkeys(
            ("host_load_validate_s", "host_read_s", "host_validate_s", "host_convert_s", "h2d_s"), 0.0
        )

    monkeypatch.setattr(execution.AscendCVQ2TP1MoE, "__init__", init)
    runtime = v2.AscendCV4V2VQ2TP1MoE(artifact, 0, "cpu")
    built = []

    def bank_factory():
        built.append(tuple(runtime._cache))
        return {"metadata_bytes": 32}

    monkeypatch.setattr(runtime, "_create_device_route_banks", bank_factory)
    return runtime, built


@pytest.mark.parametrize("use_prepacked", [False, True])
def test_preload_reuses_transfer_and_fenced_lifetime_without_reconversion(monkeypatch, use_prepacked):
    source, spec = source_payload()
    direct_layer = layer_header(source, spec, experts=(0,))
    converted = v2.convert_expert_payload(source, spec)
    source_format = prepacked.V4_V2_PREPACKED_FORMAT if use_prepacked else VQ2_DIRECT_TP1_FORMAT
    payload = converted if use_prepacked else source
    layer = prepacked_header(direct_layer, converted) if use_prepacked else direct_layer
    reads, conversions, transfers, fences = [], [], [], []

    def load(index, expert, kind, **kwargs):
        assert kwargs["device"] == "cpu"
        reads.append((index, expert, kind))
        return payload, spec

    runtime, built = initialize_cpu_runtime(monkeypatch, source_format, layer, load)

    def convert(actual, actual_spec):
        assert actual is source and actual_spec is spec
        conversions.append(actual)
        return converted

    monkeypatch.setattr(v2, "convert_expert_payload", forbidden if use_prepacked else convert)
    original_to = torch.Tensor.to

    def observed_to(tensor, *args, **kwargs):
        transfers.append((tensor, args, kwargs))
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", observed_to)
    monkeypatch.setattr(execution, "synchronize_execution", lambda device: fences.append(("load", device)))
    monkeypatch.setattr(v4, "synchronize_execution", lambda device: fences.append(("resident", device)))
    plan = v2.v4_v2_resident_plan([layer], 10**10)
    report = runtime.initialize_resident(budget_bytes=plan["planned_bytes"])
    assert reads == [(0, 0, "gate_up"), (0, 0, "down")]
    assert len(conversions) == (0 if use_prepacked else 2)
    assert built == [(0,)]
    assert [name for name, _ in fences] == ["load", "resident"]
    # All six already-converted tensors use the same blocking transfer path.
    assert len(transfers) == 2 * len(v2.V4_V2_FIELDS)
    for tensor, args, kwargs in transfers:
        assert any(tensor is original for original in converted.values())
        assert args == (torch.device("cpu"),) and kwargs == {"non_blocking": False}
    assert report["preload_loads"] == 1
    assert report["post_init_loads"] == report["post_init_h2d_bytes"] == 0
    assert report["dual_payload_residency"] is False
    assert report["startup_conversion"] is (not use_prepacked)
    if use_prepacked:
        assert runtime.timing["host_convert_s"] == 0
    else:
        assert runtime.timing["host_convert_s"] > 0
    assert runtime._get_expert(0) is runtime._cache[0]
    assert len(reads) == 2  # Cache hits never reopen the prepacked or direct source.
    for kind in ("gate_up", "down"):
        assert set(runtime._cache[0][kind][0]) == set(v2.V4_V2_FIELDS)
        assert "packed_indices" not in runtime._cache[0][kind][0]
    runtime.check_resident_integrity()
    runtime._cache[0]["gate_up"][0]["activation_order"] = converted["activation_order"].clone()
    with pytest.raises(RuntimeError, match="storage changed"):
        runtime.check_resident_integrity()


def test_prepacked_prepare_returns_same_payload_objects(monkeypatch):
    source, spec = source_payload()
    converted = v2.convert_expert_payload(source, spec)
    runtime = v2.AscendCV4V2VQ2TP1MoE.__new__(v2.AscendCV4V2VQ2TP1MoE)
    runtime.artifact = NS(manifest={"format": prepacked.V4_V2_PREPACKED_FORMAT})
    runtime.layer = prepacked_header(layer_header(source, spec), converted)
    host = {kind: (converted, spec) for kind in ("gate_up", "down")}
    monkeypatch.setattr(v2, "convert_expert_payload", forbidden)
    actual = runtime._prepare_host_expert(host)
    assert actual is host
    assert actual["gate_up"][0] is converted
    assert all(actual["gate_up"][0][name] is value for name, value in converted.items())


@pytest.mark.parametrize("use_prepacked", [False, True])
def test_owner_dispatches_source_only_to_selected_v4_v2_reader(monkeypatch, tmp_path, use_prepacked):
    opts = options(tmp_path, v4_compute_backend="v2")
    source_format = prepacked.V4_V2_PREPACKED_FORMAT if use_prepacked else VQ2_DIRECT_TP1_FORMAT
    monkeypatch.setattr(v2, "load_v4_v2_library", lambda *args: {"cpu_loader_stub": True})
    monkeypatch.setattr(offline, "artifact_format", lambda path: source_format)
    monkeypatch.setattr(offline, "audit_offline_root", lambda path: {})
    artifact, calls = NS(root=tmp_path / "artifact", manifest={"format": source_format}), []

    def open_selected(path, model_config, **kwargs):
        calls.append((path, model_config, kwargs))
        return artifact

    monkeypatch.setattr(offline, "open_vq2a8_tp1_artifact", forbidden if use_prepacked else open_selected)
    monkeypatch.setattr(offline, "open_vq2a8_tp1_zn_artifact", forbidden)
    monkeypatch.setattr(prepacked, "open_vq2a8_v4_v2_prepacked_artifact", open_selected if use_prepacked else forbidden)
    if hasattr(offline, "open_vq2a8_v4_v2_prepacked_artifact"):
        monkeypatch.setattr(
            offline, "open_vq2a8_v4_v2_prepacked_artifact", open_selected if use_prepacked else forbidden
        )
    owner = offline.OfflineMoEOwner(tmp_path / "model", opts, NS(type="npu"))
    assert owner.artifact is artifact and len(calls) == 1
    assert str(calls[0][0]) == opts["artifact"]
    assert calls[0][1] == tmp_path / "model" / "config.json"


@pytest.mark.parametrize("policy", ["baseline", "cached", "ascendc_v4", "ascendc_v3"])
def test_prepacked_owner_rejects_non_v4_v2_backend(monkeypatch, tmp_path, policy):
    from vllm_ascend.quantization import vq2a8_ascendc, vq2a8_ascendc_v3

    opts = {"artifact": str(tmp_path / "packed"), "execution_policy": policy}
    opts.update(dict.fromkeys(("ascendc_library", "ascendc_sha256", "ascendc_v3_library", "ascendc_v3_sha256"), ""))
    monkeypatch.setattr(vq2a8_ascendc, "load_pinned_library", lambda *args: None)
    monkeypatch.setattr(vq2a8_ascendc_v3, "load_pinned_library", lambda *args: None)
    monkeypatch.setattr(offline, "artifact_format", lambda path: prepacked.V4_V2_PREPACKED_FORMAT)
    monkeypatch.setattr(offline, "open_vq2a8_tp1_artifact", forbidden)
    monkeypatch.setattr(offline, "open_vq2a8_tp1_zn_artifact", forbidden)
    monkeypatch.setattr(prepacked, "open_vq2a8_v4_v2_prepacked_artifact", forbidden)
    if hasattr(offline, "open_vq2a8_v4_v2_prepacked_artifact"):
        monkeypatch.setattr(offline, "open_vq2a8_v4_v2_prepacked_artifact", forbidden)
    with pytest.raises(ValueError, match="v4.*v2|V4.*v2"):
        offline.OfflineMoEOwner(tmp_path / "model", opts, NS(type="npu"))


def test_server_explicit_artifact_preserves_three_optimization_options(tmp_path):
    model, artifact = tmp_path / "model", tmp_path / "prepacked"
    model.mkdir()
    artifact.mkdir()
    assert not (model / "experts_vq_ascend_v2").exists()
    library = tmp_path / "libvq2a8_ascendc_v4_v2.so"
    library.write_bytes(b"command fixture, not a native library")
    args = server.parse_args(
        [
            "--model",
            str(model),
            "--artifact",
            str(artifact),
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
    assert opts["artifact"] == str(artifact.resolve())
    assert opts["v4_compute_backend"] == "v2"
    assert opts["v4_activation_reorder"] == "vectorized"
    assert opts["v4_activation_preparation"] == "fused"
    assert opts["v4_decode_graph"] == "decoder"
    assert opts["v4_graph_replay_stream"] == "caller"
    assert offline.validate_offline_config(config(opts)) == opts


def test_server_default_artifact_remains_direct(tmp_path):
    model = tmp_path / "model"
    (model / "experts_vq_ascend_v2").mkdir(parents=True)
    library = tmp_path / "libvq2a8_ascendc.so"
    library.write_bytes(b"command fixture")
    args = server.parse_args(["--model", str(model), "--library", str(library)])
    command = server.build_command(args)
    opts = json.loads(command[command.index("--additional-config") + 1])["vq2a8_offline"]
    assert opts["artifact"] == str((model / "experts_vq_ascend_v2").resolve())
    assert "v4_compute_backend" not in opts


@pytest.mark.parametrize("explicit", [False, True])
def test_projection_probe_plan_forwards_selected_artifact_without_hardware(tmp_path, explicit):
    argv = ["--model", str(tmp_path / "model"), "--phase", "resident", "--plan-only"]
    if explicit:
        argv += ["--artifact", str(tmp_path / "prepacked")]
    args = projection_probe.parse_args(argv)
    plan = projection_probe.validation_plan(args)
    assert plan["prepacked_projection_requested"] is explicit
    assert plan["real_projection_requested"] is True
    assert plan["device_execution_verified"] is False
    assert plan["model_integration_verified"] is False
    command = plan["command"]
    assert command[command.index("--model") + 1] == str((tmp_path / "model").resolve())
    if explicit:
        assert command[command.index("--artifact") + 1] == str((tmp_path / "prepacked").resolve())
    else:
        assert "--artifact" not in command


def test_projection_probe_requires_model_for_independent_prepacked_oracle(tmp_path):
    with pytest.raises(SystemExit):
        projection_probe.parse_args(["--artifact", str(tmp_path / "prepacked")])


@pytest.mark.parametrize("explicit", [False, True])
def test_decoder_probe_plan_reports_artifact_without_opening_model(tmp_path, capsys, monkeypatch, explicit):
    argv = ["--model", str(tmp_path / "model"), "--library", str(tmp_path / "candidate.so"), "--plan-only"]
    if explicit:
        argv += ["--artifact", str(tmp_path / "prepacked")]
    monkeypatch.setattr(decoder_probe, "run_model", forbidden)
    monkeypatch.setattr(decoder_probe, "run_child", forbidden)
    assert decoder_probe.main(argv) == 0
    plan = json.loads(capsys.readouterr().out)
    expected = tmp_path / "prepacked" if explicit else tmp_path / "model" / "experts_vq_ascend_v2"
    assert plan["artifact"] == str(expected)
    assert plan["device_execution"] is False


def test_decoder_probe_child_command_preserves_explicit_artifact(tmp_path, monkeypatch):
    argv = [
        "--model",
        str(tmp_path / "model"),
        "--library",
        str(tmp_path / "candidate.so"),
        "--artifact",
        str(tmp_path / "prepacked"),
        "--output-dir",
        str(tmp_path / "report"),
    ]
    commands = []

    def supervised(command, environment, log, timeout):
        commands.append(command)
        # No child or NPU is run. Exercise only command forwarding and failure
        # receipt handling, never fabricate a hardware-success result.
        return {"status": "TIMEOUT"}

    monkeypatch.setattr(decoder_probe, "child_environment", lambda args: {})
    monkeypatch.setattr(decoder_probe, "run_child", supervised)
    assert decoder_probe.main(argv) == 1
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--artifact") + 1] == str(tmp_path / "prepacked")
    assert "--child" in command


@pytest.mark.parametrize("mutation", [None, "packed_byte", "metadata_signed_zero", "spec"])
def test_real_projection_probe_uploads_stored_bytes_and_rejects_differences(monkeypatch, tmp_path, mutation):
    """Tiny CPU wiring oracle, no native kernel or device correctness claim."""
    from tools import validate_vq2a8_tp1_packed_kernel as original_probe
    from vllm_ascend.quantization import vq2a8_reference, vq2a8_runtime

    spec = NS(rows=2, columns=2, rht_true_columns=2, rht_block_size=2)
    source = {"direct": object()}
    converted = {
        "packed_zn": torch.tensor([3, 5], dtype=torch.uint8),
        "pair_lut": torch.tensor([0, 1], dtype=torch.uint8),
        "activation_order": torch.tensor([1, 0], dtype=torch.int64),
        "weight_scale": torch.ones(2, dtype=torch.float32),
        "weight_bias": torch.zeros(2, dtype=torch.float32),
        "rht_sign": torch.tensor([-1, 1], dtype=torch.int8),
    }
    stored = {name: value.clone() for name, value in converted.items()}
    stored_spec = spec
    if mutation == "packed_byte":
        stored["packed_zn"][0] = 4
    elif mutation == "metadata_signed_zero":
        stored["weight_bias"][0] = -0.0
        assert torch.equal(stored["weight_bias"], converted["weight_bias"])
    elif mutation == "spec":
        stored_spec = NS(rows=2, columns=2, rht_true_columns=2, rht_block_size=1)
    reads, conversions, bank_calls, grouped_calls = [], [], [], []

    def direct_load(layer, expert, kind):
        reads.append(("direct", layer, expert, kind))
        return source, spec

    def stored_load(layer, expert, kind):
        reads.append(("prepacked", layer, expert, kind))
        return stored, stored_spec

    def convert(payload, actual_spec):
        assert payload is source and actual_spec is spec
        conversions.append(payload)
        return converted

    monkeypatch.setattr(vq2a8_runtime, "open_vq2a8_tp1_artifact", lambda *args: NS(load_expert=direct_load))
    monkeypatch.setattr(prepacked, "open_vq2a8_v4_v2_prepacked_artifact", lambda *args: NS(load_expert=stored_load))
    monkeypatch.setattr(v2, "convert_expert_payload", convert)
    monkeypatch.setattr(
        vq2a8_reference, "decode_repacked_vq2a8_codebook_weight", lambda *args, **kwargs: torch.zeros(2, 2).double()
    )
    monkeypatch.setattr(original_probe, "activation_case", lambda *args: torch.ones(1, 2))
    monkeypatch.setattr(
        projection_probe,
        "prepare_real_rows",
        lambda hidden, *args: (hidden.to(torch.float8_e4m3fn), torch.ones(len(hidden)), torch.zeros(len(hidden))),
    )

    def bank_factory(*fields):
        assert len(fields) == len(projection_probe.FIELDS)
        for name, group in zip(projection_probe.FIELDS, fields):
            assert len(group) == 1 and group[0] is stored[name]
            assert group[0] is not converted[name]
        bank_calls.append(fields)

        def project(q, scale, bias, slots):
            return torch.zeros(*q.shape[:-1], 2).bfloat16(), torch.ones(1, dtype=torch.int32)

        return NS(project=project)

    def grouped(q, scale, bias, packed, lut):
        assert packed[0] is stored["packed_zn"] and lut[0] is stored["pair_lut"]
        grouped_calls.append(packed)
        return [torch.zeros(q[0].shape[0], 2).bfloat16()]

    args = NS(model=tmp_path / "model", artifact=tmp_path / "prepacked", expert="2:3", phase="resident")
    if mutation is not None:
        with pytest.raises(AssertionError, match="changed"):
            projection_probe.run_real_checks(args, "cpu", grouped, bank_factory, lambda name: nullcontext())
        assert not bank_calls and not grouped_calls
        return
    completed = projection_probe.run_real_checks(args, "cpu", grouped, bank_factory, lambda name: nullcontext())
    assert len(completed) == len(grouped_calls) == 18
    assert len(bank_calls) == len(conversions) == 2
    assert reads == [
        ("direct", 2, 3, "gate_up"),
        ("prepacked", 2, 3, "gate_up"),
        ("direct", 2, 3, "down"),
        ("prepacked", 2, 3, "down"),
    ]
