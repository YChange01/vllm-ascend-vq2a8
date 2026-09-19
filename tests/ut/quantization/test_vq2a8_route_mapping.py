# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract/control-flow tests; not Ascend execution or speed acceptance."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tests.ut.quantization.test_vq2a8_offline import config_from_decoder_plan, decoder_candidate_plan
from tests.ut.quantization.test_vq2a8_v4_device_route import make_runtime, no_host_tensor_reads
from tools import serve_vq2a8_v4 as serve
from tools import validate_vq2a8_v4_decoder_graph as decoder_probe
from vllm_ascend.quantization import vq2a8_route_mapping as mapping
from vllm_ascend.quantization import vq2a8_v4_device_route as route
from vllm_ascend.quantization.vq2a8_offline import offline_engine_options, validate_offline_config
from vllm_ascend.quantization.vq2a8_optimization import FastMoEState, configure_runtime
from vllm_ascend.quantization.vq2a8_v4_v2 import require_v4_v2_features

NATIVE = Path(__file__).resolve().parents[3] / "csrc/vq2a8_ascendc_v4_v2"


def native_reference(**changes):
    return NS(**{"route_mapping_version": lambda: 1, "route_mapping": mapping.torch_route_mapping, **changes})


@pytest.mark.parametrize("version", [None, True, 0, 2, "1"])
def test_abi_is_exact_integer(version):
    with pytest.raises(RuntimeError, match="ABI"):
        mapping.FusedRouteMapping(native_reference(route_mapping_version=lambda: version))


@pytest.mark.parametrize("name", ["route_mapping_version", "route_mapping"])
def test_no_silent_fallback(name):
    native = native_reference()
    delattr(native, name)
    with pytest.raises(RuntimeError, match="no implicit fallback"):
        mapping.FusedRouteMapping(native)


@pytest.mark.parametrize("groups", range(1, 7))
@pytest.mark.parametrize("experts", [1, 2, 256])
def test_exact_integer_boundaries_and_offset_views(monkeypatch, groups, experts):
    bounds = torch.iinfo(torch.int64)
    choices = [-1, 0, experts - 1, experts, bounds.min, bounds.max]
    lookup = torch.tensor([0, *[index if index % 2 else -7 for index in range(experts)]])[1:]
    ids = torch.tensor([0, *choices[:groups]])[1:]
    expected = torch.tensor([lookup[index].item() if 0 <= index < experts else -1 for index in choices[:groups]])
    checker = mapping.FusedRouteMapping(native_reference())
    with no_host_tensor_reads(monkeypatch):
        slots, valid = checker(ids, lookup)
    assert torch.equal(slots, expected)
    assert torch.equal(valid, (expected >= 0).all())
    assert slots.shape == ids.shape and slots.dtype == torch.int64
    assert valid.shape == () and valid.dtype == torch.bool


def test_positive_invalid_slot_is_not_silently_clamped():
    lookup = torch.tensor([torch.iinfo(torch.int64).max])
    slots, valid = mapping.FusedRouteMapping(native_reference())(torch.tensor([0, 0]), lookup)
    assert slots.tolist() == [torch.iinfo(torch.int64).max] * 2
    assert bool(valid)  # The resident bank, not the lookup, enforces slot upper bounds.


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("failure", ["none", "dtype", "rank", "empty", "too_large", "strided", "device"])
def test_input_contract_before_native_call(which, failure):
    args = [torch.arange(6), torch.arange(256)]
    value = args[which]
    if failure == "none":
        args[which] = None
    elif failure == "dtype":
        args[which] = value.int()
    elif failure == "rank":
        args[which] = value.reshape(1, -1)
    elif failure == "empty":
        args[which] = value[:0]
    elif failure == "too_large":
        args[which] = torch.arange(value.numel() + 1)
    elif failure == "strided":
        args[which] = value[::2]
    else:
        args[which] = value.to("meta")
    calls = []
    checker = mapping.FusedRouteMapping(native_reference(route_mapping=lambda *args: calls.append(args)))
    with pytest.raises(ValueError, match="Route mapping"):
        checker(*args)
    assert not calls


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("hash_route", [False, True])
def test_eager_and_graph_integration_preserves_outputs_and_invalid_recovery(monkeypatch, graph, hash_route):
    runtime, hidden = make_runtime(hash_route=hash_route)
    runtime.v4_compute_backend = "v2"
    configure_runtime(runtime, "device_route_decode")
    baseline = route.DeviceRouteGraphCompute(runtime) if graph else runtime._optimization
    tokens = torch.tensor([0])
    expected = baseline(hidden, tokens)[0] if graph else baseline.forward(runtime, hidden, tokens)
    calls = []

    def operation(ids, lookup):
        calls.append((ids, lookup))
        return mapping.torch_route_mapping(ids, lookup)

    monkeypatch.setattr(mapping, "FusedRouteMapping", lambda: operation)
    runtime.v4_route_mapping = "fused"
    runtime._optimization = route.DeviceRouteDecodeState(runtime)
    candidate = route.DeviceRouteGraphCompute(runtime) if graph else runtime._optimization
    for bad_slot in (None, -1, 1000, None):
        lookup = runtime._device_route_banks["lookup"]
        if bad_slot is None:
            lookup.copy_(torch.arange(lookup.numel()))
        else:
            lookup.fill_(bad_slot)
        if not graph:
            candidate.valid = None
        with no_host_tensor_reads(monkeypatch):
            if graph:
                actual, valid = candidate(hidden, tokens)
            else:
                actual = candidate.forward(runtime, hidden, tokens)
                valid = candidate.valid
        assert bool(valid) == (bad_slot is None)
        if bad_slot is None:
            assert torch.equal(actual, expected)
    assert len(calls) == 4
    if graph:
        assert candidate.signature["route_mapping"] == "fused"
        runtime.v4_route_mapping = "torch"
        with pytest.raises(RuntimeError, match="signature changed"):
            candidate.check_runtime_contract(runtime)


def test_prefill_does_not_enter_mapping(monkeypatch):
    runtime, hidden = make_runtime()
    runtime.v4_compute_backend = "v2"
    runtime.v4_route_mapping = "fused"
    monkeypatch.setattr(mapping, "FusedRouteMapping", lambda: lambda *args: pytest.fail("decode mapper in prefill"))
    configure_runtime(runtime, "device_route_decode")
    calls = []
    monkeypatch.setattr(FastMoEState, "forward", lambda *args: calls.append(args) or hidden)
    assert runtime._optimization.forward(runtime, hidden.repeat(2, 1), None) is hidden
    assert len(calls) == 1


def test_feature_gate_and_engine_wiring():
    native = NS()
    with pytest.raises(RuntimeError, match="route_mapping_version"):
        require_v4_v2_features(route_mapping="fused", native_ops=native)
    native.route_mapping_version = lambda: 1
    with pytest.raises(RuntimeError, match="no implicit fallback"):
        require_v4_v2_features(route_mapping="fused", native_ops=native)
    native.route_mapping = mapping.torch_route_mapping
    require_v4_v2_features(route_mapping="fused", native_ops=native)
    require_v4_v2_features(native_ops=NS())  # Old library remains sufficient for defaults.
    options = dict(execution_policy="ascendc_v4", v4_compute_backend="v2", v4_device_route_decode=True)
    value = offline_engine_options(Path("model"), Path("artifact"), **options, v4_route_mapping="fused")
    assert value["additional_config"]["vq2a8_offline"]["v4_route_mapping"] == "fused"
    default = offline_engine_options(Path("model"), Path("artifact"), **options)
    assert "v4_route_mapping" not in default["additional_config"]["vq2a8_offline"]
    for changes in ({"v4_compute_backend": "v1"}, {"v4_device_route_decode": False}, {"execution_policy": "cached"}):
        with pytest.raises(ValueError, match="route mapping|v4_compute_backend"):
            offline_engine_options(Path("model"), Path("artifact"), **(options | changes), v4_route_mapping="fused")
    with pytest.raises(ValueError, match="route_mapping"):
        offline_engine_options(Path("model"), Path("artifact"), **options, v4_route_mapping="silent")


def test_cli_wiring_and_no_default_change(tmp_path, capsys):
    model, artifact = tmp_path / "model", tmp_path / "artifact"
    model.mkdir()
    artifact.mkdir()
    library = tmp_path / "libvq2a8_ascendc_v4_v2.so"
    library.write_bytes(b"contract-only")
    args = ["--model", str(model), "--artifact", str(artifact), "--library", str(library), "--compute-backend", "v2"]
    assert serve.parse_args(args).route_mapping == "torch"
    assert decoder_probe.parse_args(args).route_mapping == "torch"
    fused = args + ["--route-mapping", "fused"]
    with pytest.raises(SystemExit):
        serve.parse_args(fused)  # Must select device-route decode too.
    parsed = serve.parse_args(fused + ["--device-route-decode"])
    command = serve.build_command(parsed)
    extra = json.loads(command[command.index("--additional-config") + 1])
    assert extra["vq2a8_offline"]["v4_route_mapping"] == "fused"
    assert decoder_probe.main(fused + ["--plan-only"]) == 0
    assert json.loads(capsys.readouterr().out)["route_mapping"] == "fused"


@pytest.mark.parametrize("mode", ["torch", "fused"])
def test_offline_config_accepts_selected_mapping(tmp_path, mode):
    cfg = config_from_decoder_plan(decoder_candidate_plan(tmp_path, v4_route_mapping=mode))
    assert validate_offline_config(cfg).get("v4_route_mapping", "torch") == mode


@pytest.mark.parametrize("invalid", [None, True, 1, "silent"])
def test_offline_config_rejects_invalid_mapping(tmp_path, invalid):
    cfg = config_from_decoder_plan(decoder_candidate_plan(tmp_path))
    cfg.additional_config["vq2a8_offline"]["v4_route_mapping"] = invalid
    with pytest.raises(ValueError, match="v4_route_mapping"):
        validate_offline_config(cfg)


@pytest.mark.parametrize("backend,device_route", [("v1", False), ("v1", True), ("v2", False)])
def test_offline_config_mapping_requires_v2_device_route(tmp_path, backend, device_route):
    cfg = config_from_decoder_plan(decoder_candidate_plan(tmp_path))
    cfg.additional_config["vq2a8_offline"].update(
        v4_compute_backend=backend, v4_device_route_decode=device_route, v4_route_mapping="fused"
    )
    with pytest.raises(ValueError, match="Fused route mapping"):
        validate_offline_config(cfg)


def test_hot_path_metadata_only_and_native_ownership_contract():
    source = inspect.getsource(mapping.FusedRouteMapping.__call__)
    for forbidden in (".cpu(", ".item(", ".tolist(", ".all(", ".contiguous("):
        assert forbidden not in source
    binding = (NATIVE / "route_mapping_binding.cpp").read_text()
    enqueue = binding.index('OpCommand::RunOpApi("Vq2a8V4V2RouteMapping"')
    assert binding.index("const auto launchStream = stream.stream();") < enqueue
    assert "stream.stream()" not in binding[enqueue:]
    assert "[launchStream, ids, lookup, slots, valid, groups, experts]" in binding
    for name in ("ids", "lookup"):
        assert f"recordStream({name}.storage().data_ptr(), stream)" in binding
    for predicate in ("tensor.storage().nbytes()", "tensor.storage_offset() >= 0", "tensor.is_contiguous()"):
        assert predicate in binding
    kernel = (NATIVE / "route_mapping_kernel.cpp").read_text()
    assert kernel.index("id >= int64_t(0)") < kernel.index("lookup_.GetValue")
    assert "<<<1, nullptr, stream>>>" in kernel
    assert "DataCopyPad(slots_, slots, slotCopy)" in kernel
