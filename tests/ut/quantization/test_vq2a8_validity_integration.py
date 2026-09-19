# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate wiring contracts, not native execution or latency acceptance."""

from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tests.ut.quantization.test_vq2a8_activation_packed import TorchSignOps, assert_bytes, fixture
from tests.ut.quantization.test_vq2a8_v4_device_route import make_runtime, no_host_tensor_reads
from tools import serve_vq2a8_v4 as serve
from tools import validate_vq2a8_v4_decoder_graph as decoder_probe
from vllm_ascend.quantization import vq2a8_v4_device_route as route
from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_activation_packed import PackedRowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_offline import offline_engine_options
from vllm_ascend.quantization.vq2a8_optimization import configure_runtime
from vllm_ascend.quantization.vq2a8_v4_v2 import require_v4_v2_features


def test_raw_sign_flags_preserve_bytes_and_avoid_scalar_reduction():
    values = fixture(2048, 6)
    prep = PackedRowwiseVQ2A8Preparation(fuse_sign=True, native_ops=TorchSignOps())
    reference = prep.packed(*values)
    raw, reduced = [], []
    actual = prep.packed(*values, validity=reduced.append, raw_input_validity=raw.append)
    assert len(raw) == 1 and raw[0].dtype == torch.int32 and raw[0].shape == (6,)
    assert reduced == []
    for x, y in zip(actual, reference):
        assert_bytes(x, y)
    values[3][0, 0] = 0
    prep.packed(*values, raw_input_validity=raw.append)
    assert not bool(raw[-1][0])
    values[3][0, 0] = 1
    prep.packed(*values, raw_input_validity=raw.append)
    assert bool(raw[-1].all()) and not bool(raw[-2][0])


def test_raw_flags_reject_non_native_sign():
    prep = PackedRowwiseVQ2A8Preparation()
    with pytest.raises(ValueError, match="Raw input validity"):
        prep.packed(*fixture(2048, 1), raw_input_validity=lambda x: None)


class PackedProtocolOracle:
    """Tiny geometry oracle for route/control flow; production ABI uses 2048/4096."""

    def prepare_for_graph(self, *args):
        pass

    def packed(self, hidden, scale, bias, signs, spec, *, validity=None, raw_input_validity=None):
        flags = []
        requests = [
            (hidden[i : i + 1], {"weight_scale": scale[i], "weight_bias": bias[i], "rht_sign": signs[i]}, spec)
            for i in range(hidden.shape[0])
        ]
        result = RowwiseVQ2A8Preparation(compact=True).many(requests, validity=flags.append)
        valid = torch.stack(flags).all().expand(hidden.shape[0]).int().contiguous()
        if raw_input_validity is not None:
            raw_input_validity(valid)
        else:
            validity(valid.all())
        return tuple(torch.cat(x).contiguous() for x in zip(*result))


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("hash_route", [False, True])
def test_route_fused_equivalence_and_fresh_flags(monkeypatch, graph, hash_route):
    runtime, hidden = make_runtime(hash_route=hash_route)
    runtime.v4_compute_backend = "v2"
    runtime.v4_activation_preparation = "sign_fused"
    runtime._row_preparation = PackedProtocolOracle()
    runtime.make_v4_preparation = lambda **kw: PackedProtocolOracle()
    configure_runtime(runtime, "device_route_decode")
    baseline = route.DeviceRouteGraphCompute(runtime) if graph else runtime._optimization
    expected = baseline(hidden, torch.tensor([0]))[0] if graph else baseline.forward(runtime, hidden, torch.tensor([0]))
    calls = []

    def check(statuses, outputs, flags):
        assert len(statuses) == 6 and len(outputs) == 3
        assert all(x.dtype == torch.int32 and x.shape == (6,) for x in statuses)
        calls.append((statuses, outputs, flags))
        return torch.stack(
            [*(x.ne(0).all() for x in statuses), *(torch.isfinite(x).all() for x in outputs), *flags]
        ).all()

    monkeypatch.setattr(route, "_make_layer_validity", lambda r: check)
    runtime.v4_validity_mode = "fused"
    runtime._optimization = route.DeviceRouteDecodeState(runtime)
    candidate = route.DeviceRouteGraphCompute(runtime) if graph else runtime._optimization
    outputs = []
    for token in (0, 0, 0):
        # Select flags exercise valid -> invalid -> valid without changing arithmetic.
        if len(calls) == 1:
            runtime._device_route_banks["lookup"].fill_(-1)
        elif len(calls) == 2:
            runtime._device_route_banks["lookup"].copy_(torch.arange(8))
        if not graph:
            candidate.valid = None  # model boundary starts a fresh forward
        with no_host_tensor_reads(monkeypatch):
            if graph:
                output, valid = candidate(hidden, torch.tensor([token]))
            else:
                output = candidate.forward(runtime, hidden, torch.tensor([token]))
                valid = candidate.valid
        outputs.append((output, valid))
    assert torch.equal(outputs[0][0], expected) and torch.equal(outputs[2][0], expected)
    assert [bool(x[1]) for x in outputs] == [True, False, True]
    assert len({id(x[0]) for x in calls}) == 3
    if graph:
        runtime.v4_validity_mode = "torch"
        with pytest.raises(RuntimeError, match="signature changed"):
            candidate.check_runtime_contract(runtime)


def test_feature_gate_and_engine_options():
    native = NS(activation_preparation_version=lambda: 1)
    with pytest.raises(RuntimeError, match="layer_validity_version"):
        require_v4_v2_features(preparation="sign_fused", validity_mode="fused", native_ops=native)
    native.layer_validity_version = lambda: 1
    require_v4_v2_features(preparation="sign_fused", validity_mode="fused", native_ops=native)
    options = dict(
        execution_policy="ascendc_v4",
        v4_compute_backend="v2",
        v4_device_route_decode=True,
        v4_activation_preparation="sign_fused",
        v4_validity_mode="fused",
    )
    result = offline_engine_options(Path("model"), Path("artifact"), **options)
    assert result["additional_config"]["vq2a8_offline"]["v4_validity_mode"] == "fused"
    for change in (
        {"v4_device_route_decode": False},
        {"v4_compute_backend": "v1"},
        {"v4_activation_preparation": "rowwise"},
        {"v4_validity_mode": "silent"},
    ):
        with pytest.raises(ValueError):
            offline_engine_options(Path("model"), Path("artifact"), **(options | change))


def test_cli_flags_and_template_evidence(tmp_path):
    model, artifact = tmp_path / "model", tmp_path / "artifact"
    model.mkdir()
    artifact.mkdir()
    library = tmp_path / "libvq2a8_ascendc_v4_v2.so"
    library.write_bytes(b"cpu contract only")
    args = [
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
        "--validity-mode",
        "fused",
        "--decoder-metadata-mode",
        "position_template",
    ]
    parsed = serve.parse_args(
        args
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
    command = serve.build_command(parsed)
    assert "v4_validity_mode" in str(command) and "position_template" in str(command)
    assert decoder_probe.parse_args(args).validity_mode == "fused"
    evidence = {
        "reference_verification_enabled": True,
        "reference_checks": 86,
        "reference_positions": list(range(1, 15)),
        "original_builder_skips": 58,
    }
    decoder_probe.require_template_evidence({"decoder": {"position_template": evidence}})
    for change in (
        {"reference_checks": 0},
        {"reference_positions": [1]},
        {"reference_verification_enabled": False},
        {"original_builder_skips": 0},
    ):
        with pytest.raises(AssertionError, match="equivalence evidence"):
            decoder_probe.require_template_evidence({"decoder": {"position_template": evidence | change}})


@pytest.mark.parametrize("failure", [None, "replays", "skips", "tokens"])
def test_template_probe_exercises_actual_serving_without_shadow(failure):
    reference = NS(token_ids=[1, 2, 3, 4], logprobs=[{1: NS(logprob=-0.25)}] * 4)

    class Engine:
        calls = []
        replays = skips = 0
        shadow = True

        def collective_rpc(self, method, args=()):
            if method is decoder_probe.set_template_verification:
                self.shadow = args[0]
                self.calls.append(self.shadow)
                return {}
            assert method is decoder_probe.graph_report
            return {"decoder": {"replays": self.replays, "position_template": {"original_builder_skips": self.skips}}}

        def generate(self, prompts, params, use_tqdm):
            assert not self.shadow and prompts == [{"prompt_token_ids": [3]}] and not use_tqdm
            self.replays += 2 if failure == "replays" else 3
            self.skips += 0 if failure == "skips" else 3
            result = NS(token_ids=[9] * 4, logprobs=reference.logprobs) if failure == "tokens" else reference
            return [NS(outputs=[result])]

    engine = Engine()
    if failure:
        with pytest.raises(AssertionError):
            decoder_probe.verify_template_serving_path(engine, [3], None, reference, lambda x: x)
    else:
        assert decoder_probe.verify_template_serving_path(engine, [3], None, reference, lambda x: x) == 0
        assert engine.calls == [False, True]
