# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU guard contracts only, not NPU capture or performance acceptance."""

import gc
import inspect
import weakref
from copy import copy
from types import SimpleNamespace

import pytest
import torch

from tests.ut.quantization.test_vq2a8_v4_device_route import make_runtime, no_host_tensor_reads
from vllm_ascend.quantization.vq2a8_optimization import configure_runtime
from vllm_ascend.quantization.vq2a8_runtime_guard import (
    CONFIG_FIELDS,
    GEOMETRY_FIELDS,
    RUNTIME_FIELDS,
    PlannedRuntimeGuard,
)
from vllm_ascend.quantization.vq2a8_v4_device_route import DeviceRouteGraphCompute


def captured_compute(mode="signature"):
    runtime, _ = make_runtime()
    runtime.v4_runtime_guard = mode
    configure_runtime(runtime, "device_route_decode")
    return runtime, DeviceRouteGraphCompute(runtime)


@pytest.mark.parametrize("mode", ["signature", "planned"])
@pytest.mark.parametrize(
    "change",
    [
        "runtime_owner",
        "config_owner",
        "banks_owner",
        "gate_bank",
        "down_bank",
        "root_replaced_tensor",
        "root_set",
        "root_shape",
        "root_stride",
        "root_offset",
        "root_dtype",
        "root_added_key",
        "root_removed_key",
        "lookup_replaced_tensor",
        "lookup_set",
        "lookup_stride",
        "lookup_offset",
        "lookup_shape",
        "lookup_dtype",
        "mode",
    ],
)
def test_signature_and_planned_guards_reject_mutation_before_compute(mode, change):
    runtime, compute = captured_compute(mode)
    compute.check_runtime_contract(runtime)
    if change == "runtime_owner":
        runtime = copy(runtime)
    elif change == "config_owner":
        runtime.config = copy(runtime.config)
    elif change == "banks_owner":
        runtime._device_route_banks = dict(runtime._device_route_banks)
    elif change in ("gate_bank", "down_bank"):
        kind = "gate_up" if change == "gate_bank" else "down"
        _, spec = runtime._device_route_banks[kind]
        runtime._device_route_banks[kind] = (object(), spec)
    elif change == "root_replaced_tensor":
        runtime.root["gate.weight"] = runtime.root["gate.weight"].clone()
    elif change == "root_set":
        runtime.root["gate.weight"].set_(runtime.root["gate.weight"].clone())
    elif change == "root_shape":
        runtime.root["gate.weight"].resize_(4, 32)
    elif change == "root_stride":
        runtime.root["gate.weight"].as_strided_((8, 16), (1, 8))
    elif change == "root_offset":
        runtime.root["gate.weight"].as_strided_((8, 16), (0, 1), 1)
    elif change == "root_dtype":
        runtime.root["gate.weight"] = runtime.root["gate.weight"].double()
    elif change == "root_added_key":
        runtime.root["new"] = torch.zeros(1)
    elif change == "root_removed_key":
        del runtime.root["gate.bias"]
    elif change == "lookup_replaced_tensor":
        runtime._device_route_banks["lookup"] = runtime._device_route_banks["lookup"].clone()
    elif change == "lookup_set":
        runtime._device_route_banks["lookup"].set_(runtime._device_route_banks["lookup"].clone())
    elif change == "lookup_stride":
        runtime._device_route_banks["lookup"].as_strided_((8,), (0,))
    elif change == "lookup_offset":
        runtime._device_route_banks["lookup"].as_strided_((8,), (0,), 1)
    elif change == "lookup_shape":
        runtime._device_route_banks["lookup"].resize_(2, 4)
    elif change == "lookup_dtype":
        runtime._device_route_banks["lookup"] = runtime._device_route_banks["lookup"].int()
    else:
        runtime.v4_runtime_guard = "planned" if mode == "signature" else "signature"
    with pytest.raises(RuntimeError, match="signature changed"):
        compute.check_runtime_contract(runtime)


@pytest.mark.parametrize("mode", ["signature", "planned"])
@pytest.mark.parametrize("name,default", RUNTIME_FIELDS[:-1])
def test_every_runtime_mode_is_rechecked(mode, name, default):
    runtime, compute = captured_compute(mode)
    setattr(runtime, name, default + "_changed")
    with pytest.raises(RuntimeError, match="signature changed"):
        compute.check_runtime_contract(runtime)


@pytest.mark.parametrize("mode", ["signature", "planned"])
@pytest.mark.parametrize("name", CONFIG_FIELDS)
def test_every_config_scalar_is_rechecked(mode, name):
    runtime, compute = captured_compute(mode)
    value = getattr(runtime.config, name)
    setattr(runtime.config, name, not value if type(value) is bool else value + 1)
    with pytest.raises(RuntimeError, match="signature changed"):
        compute.check_runtime_contract(runtime)


@pytest.mark.parametrize("mode", ["signature", "planned"])
@pytest.mark.parametrize("kind", ["gate_up", "down"])
@pytest.mark.parametrize("name", GEOMETRY_FIELDS)
def test_every_projection_geometry_is_rechecked(mode, kind, name):
    runtime, compute = captured_compute(mode)
    spec = runtime._device_route_banks[kind][1]
    setattr(spec, name, getattr(spec, name) + 1)
    with pytest.raises(RuntimeError, match="signature changed"):
        compute.check_runtime_contract(runtime)


@pytest.mark.parametrize("mode", ["signature", "planned"])
def test_reordering_root_keys_or_equal_spec_rewrap_keeps_contract(mode):
    runtime, compute = captured_compute(mode)
    runtime.root = dict(reversed(tuple(runtime.root.items())))
    bank, spec = runtime._device_route_banks["down"]
    runtime._device_route_banks["down"] = (bank, copy(spec))
    compute.check_runtime_contract(runtime)


@pytest.mark.parametrize("mode", ["signature", "planned"])
@pytest.mark.parametrize("invalid", [None, 7, "not_a_lookup"])
def test_malformed_lookup_is_a_guard_failure(mode, invalid):
    runtime, compute = captured_compute(mode)
    runtime._device_route_banks["lookup"] = invalid
    with pytest.raises(RuntimeError, match="signature changed"):
        compute.check_runtime_contract(runtime)


@pytest.mark.parametrize("mode", ["signature", "planned"])
def test_changed_runtime_is_rejected_before_input_copy_or_graph_replay(mode):
    runtime, compute = captured_compute(mode)
    state = runtime._optimization
    reached = []

    def unexpected(*args, **kwargs):
        pytest.fail("Guard must fail before graph inputs/copies/replay")

    state._graph_compute = compute
    state._decode_graph = SimpleNamespace(
        _require_open=lambda: reached.append("open"), _check_inputs=unexpected, replay=unexpected
    )
    runtime._device_route_banks["lookup"] = runtime._device_route_banks["lookup"].clone()
    with pytest.raises(RuntimeError, match="signature changed"):
        state.forward_graph(runtime, torch.zeros(1, 16), torch.zeros(1, dtype=torch.int64))
    assert reached == ["open"]


def test_planned_does_not_rebuild_the_signature_at_replay(monkeypatch):
    runtime, compute = captured_compute("planned")

    def unexpected(*args, **kwargs):
        raise AssertionError("Do not reconstruct the runtime signature")

    monkeypatch.setattr(compute, "_contract", unexpected)
    with no_host_tensor_reads(monkeypatch):
        for _ in range(3):
            compute.check_runtime_contract(runtime)
    runtime.config.top_k += 1
    with pytest.raises(RuntimeError, match="signature changed"):
        compute.check_runtime_contract(runtime)
    assert "sorted(" not in inspect.getsource(PlannedRuntimeGuard.check).split("# No sorted()", 1)[-1]


def test_signature_default_and_report_mode_are_explicit():
    runtime, _ = make_runtime()
    configure_runtime(runtime, "device_route_decode")
    compute = DeviceRouteGraphCompute(runtime)
    assert compute._runtime_guard_mode == "signature"
    assert compute._runtime_guard_plan is None
    assert compute.signature["runtime_guard"] == "signature"
    assert compute.signature["select_sign"] == "separate"
    assert compute.signature["activation_tail"] == "torch"
    runtime, compute = captured_compute("planned")
    assert compute.signature["runtime_guard"] == "planned"


@pytest.mark.parametrize("mode", ["off", "none", "", None, True])
def test_no_guard_disabling_or_unknown_mode(mode):
    runtime, _ = make_runtime()
    runtime.v4_runtime_guard = mode
    configure_runtime(runtime, "device_route_decode")
    with pytest.raises(ValueError, match="signature or planned"):
        DeviceRouteGraphCompute(runtime)


def test_guard_retains_replaced_tensor_and_bank_owners():
    runtime, _ = make_runtime()
    plan = PlannedRuntimeGuard(runtime)
    root = weakref.ref(runtime.root["gate.weight"])
    lookup = weakref.ref(runtime._device_route_banks["lookup"])
    bank = weakref.ref(runtime._device_route_banks["gate_up"][0])
    runtime.root["gate.weight"] = runtime.root["gate.weight"].clone()
    runtime._device_route_banks["lookup"] = runtime._device_route_banks["lookup"].clone()
    _, spec = runtime._device_route_banks["gate_up"]
    runtime._device_route_banks["gate_up"] = (object(), spec)
    gc.collect()
    assert root() is not None and lookup() is not None and bank() is not None
    with pytest.raises(RuntimeError, match="signature changed"):
        plan.check(runtime)


@pytest.mark.parametrize("before_compile", [False, True])
def test_device_tensor_cannot_be_used_as_a_host_config_scalar(monkeypatch, before_compile):
    runtime, _ = make_runtime()
    plan = None if before_compile else PlannedRuntimeGuard(runtime)
    runtime.config.top_k = torch.tensor(runtime.config.top_k)
    with no_host_tensor_reads(monkeypatch), pytest.raises(RuntimeError, match="signature changed"):
        if before_compile:
            PlannedRuntimeGuard(runtime)
        else:
            plan.check(runtime)


def test_plan_strongly_owns_runtime_and_config_after_external_references_drop():
    class Runtime(SimpleNamespace):
        pass

    runtime, _ = make_runtime()
    runtime = Runtime(**vars(runtime))
    plan = PlannedRuntimeGuard(runtime)
    reference = weakref.ref(runtime)
    del runtime
    gc.collect()
    assert reference() is plan.runtime
    plan.check(reference())
