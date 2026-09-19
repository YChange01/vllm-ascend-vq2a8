# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU Python/oracle tests, not native compilation or NPU acceptance."""

import ast
import gc
import inspect
import json
import weakref
from copy import copy
from types import SimpleNamespace

import pytest
import torch

from tests.ut.quantization.test_vq2a8_v4_decoder_graph import make_bank
from tests.ut.quantization.test_vq2a8_v4_device_route import make_runtime, no_host_tensor_reads
from tools import validate_vq2a8_runtime_guard_native as probe
from vllm_ascend.quantization import vq2a8_runtime_guard as module


class HostMetadataOracle:
    """Tests Python batching with independent CPU metadata snapshots."""

    def __init__(self, tensors, labels):
        if len(tensors) != len(labels):
            raise RuntimeError("length")
        self.snapshots = []
        self.storage_owners = []
        self.calls = 0
        for tensor, label in zip(tensors, labels):
            if tensor.layout != torch.strided or tensor.device.type == "meta":
                raise RuntimeError("storage-backed tensors")
            self.snapshots.append((tensor, self.metadata(tensor), label))
            self.storage_owners.append(tensor.untyped_storage())

    @staticmethod
    def metadata(tensor):
        return (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.dtype,
            tensor.device,
        )

    def append(self, other):
        if self.calls or other is self:
            raise RuntimeError("startup only")
        self.snapshots.extend(other.snapshots)
        self.storage_owners.extend(other.storage_owners)

    def check(self, tensors):
        self.calls += 1
        if len(tensors) != len(self.snapshots):
            raise RuntimeError("signature changed")
        for current, (owner, metadata, _label) in zip(tensors, self.snapshots):
            if current is not owner or self.metadata(current) != metadata:
                raise RuntimeError("signature changed")


def make_compute():
    runtime, _ = make_runtime()
    runtime.v4_runtime_guard = "native"
    plan = module.NativeRuntimeGuard(
        runtime,
        native_ops=SimpleNamespace(runtime_guard_version=lambda: 1),
        native_factory=HostMetadataOracle,
    )
    return SimpleNamespace(runtime=runtime, _runtime_guard_mode="native", _runtime_guard_plan=plan)


@pytest.mark.parametrize("version", [None, "1", False, True, 0, 2])
def test_native_abi_requires_exact_int_one(version):
    with pytest.raises(RuntimeError, match="independent ABI"):
        module.native_runtime_guard_factory(
            native_ops=SimpleNamespace(runtime_guard_version=lambda: version), native_factory=HostMetadataOracle
        )


def test_missing_abi_has_no_python_fallback():
    with pytest.raises(RuntimeError, match="missing; no fallback"):
        module.native_runtime_guard_factory(native_ops=SimpleNamespace(), native_factory=HostMetadataOracle)


def test_device_route_native_plan_integrates_without_signature_rebuild(monkeypatch):
    from tests.ut.quantization.test_vq2a8_runtime_guard import captured_compute

    monkeypatch.setattr(module, "native_runtime_guard_factory", lambda **_kwargs: HostMetadataOracle)
    runtime, compute = captured_compute("native")
    assert isinstance(compute._runtime_guard_plan, module.NativeRuntimeGuard)
    assert compute.signature["runtime_guard"] == "native"

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Native runtime replay must not rebuild the signature")

    monkeypatch.setattr(compute, "_contract", unexpected)
    with no_host_tensor_reads(monkeypatch):
        compute.check_runtime_contract(runtime)
    runtime.root["gate.weight"].set_(runtime.root["gate.weight"].clone())
    with pytest.raises(RuntimeError, match="signature changed"):
        compute.check_runtime_contract(runtime)


@pytest.mark.parametrize("name,default", module.RUNTIME_FIELDS)
def test_native_runtime_scalars_are_live(name, default):
    compute = make_compute()
    setattr(compute.runtime, name, default + "_changed")
    with pytest.raises(RuntimeError, match="signature changed"):
        compute._runtime_guard_plan.check(compute.runtime)
    assert compute._runtime_guard_plan.native_plan.calls == 0


@pytest.mark.parametrize("name", module.CONFIG_FIELDS)
def test_native_config_scalars_are_live(name):
    compute = make_compute()
    old = getattr(compute.runtime.config, name)
    setattr(compute.runtime.config, name, not old if type(old) is bool else old + 1)
    with pytest.raises(RuntimeError, match="signature changed"):
        compute._runtime_guard_plan.check(compute.runtime)


@pytest.mark.parametrize("kind", ["gate_up", "down"])
@pytest.mark.parametrize("name", module.GEOMETRY_FIELDS)
def test_native_geometry_is_live(kind, name):
    compute = make_compute()
    spec = compute.runtime._device_route_banks[kind][1]
    setattr(spec, name, getattr(spec, name) + 1)
    with pytest.raises(RuntimeError, match="signature changed"):
        compute._runtime_guard_plan.check(compute.runtime)


@pytest.mark.parametrize(
    "mutation",
    ["runtime", "config", "banks", "bank", "root_tensor", "lookup", "root_add", "root_remove", "scalar_tensor"],
)
def test_native_host_owners_keys_and_types_are_checked_before_cpp(monkeypatch, mutation):
    compute = make_compute()
    runtime, plan = compute.runtime, compute._runtime_guard_plan
    if mutation == "runtime":
        runtime = copy(runtime)
    elif mutation == "config":
        runtime.config = copy(runtime.config)
    elif mutation == "banks":
        runtime._device_route_banks = dict(runtime._device_route_banks)
    elif mutation == "bank":
        _, spec = runtime._device_route_banks["down"]
        runtime._device_route_banks["down"] = (object(), spec)
    elif mutation == "root_tensor":
        runtime.root["gate.weight"] = runtime.root["gate.weight"].view_as(runtime.root["gate.weight"])
    elif mutation == "lookup":
        runtime._device_route_banks["lookup"] = runtime._device_route_banks["lookup"].clone()
    elif mutation == "root_add":
        runtime.root["new"] = torch.ones(1)
    elif mutation == "root_remove":
        del runtime.root["gate.bias"]
    else:
        runtime.config.top_k = torch.tensor(runtime.config.top_k)
    with no_host_tensor_reads(monkeypatch), pytest.raises(RuntimeError, match="signature changed"):
        plan.check(runtime)
    assert plan.native_plan.calls == 0


@pytest.mark.parametrize("target", ["root", "lookup"])
@pytest.mark.parametrize("mutation", ["set", "shape", "stride", "offset", "dtype"])
def test_native_cpp_receives_live_mutated_tensor_after_success(target, mutation):
    compute = make_compute()
    runtime, plan = compute.runtime, compute._runtime_guard_plan
    tensor = runtime.root["gate.weight"] if target == "root" else runtime._device_route_banks["lookup"]
    plan.check(runtime)
    if mutation == "set":
        tensor.set_(tensor.clone())
    elif mutation == "shape":
        tensor.resize_(tensor.numel() // 2, 2)
    elif mutation == "stride":
        tensor.as_strided_(tensor.shape, (0,) * tensor.ndim)
    elif mutation == "offset":
        tensor.as_strided_(tensor.shape, (0,) * tensor.ndim, 1)
    else:
        tensor.data = tensor.double()
    with pytest.raises(RuntimeError, match="signature changed"):
        plan.check(runtime)
    assert plan.native_plan.calls == 2


def test_native_collection_uses_only_identity_not_python_tensor_metadata(monkeypatch):
    compute = make_compute()
    runtime, plan = compute.runtime, compute._runtime_guard_plan
    observed = []
    plan.native_plan.check = lambda tensors: observed.append(tensors)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Python replay must not inspect tensor metadata")

    monkeypatch.setattr(torch.Tensor, "data_ptr", forbidden)
    monkeypatch.setattr(torch.Tensor, "stride", forbidden)
    monkeypatch.setattr(torch.Tensor, "storage_offset", forbidden)
    with no_host_tensor_reads(monkeypatch):
        plan.check(runtime)
    assert observed[0][0] is runtime.root["gate.weight"]
    assert observed[0][-1] is runtime._device_route_banks["lookup"]


def test_native_root_rewrap_and_equal_geometry_rewrap_are_supported():
    compute = make_compute()
    runtime = compute.runtime
    runtime.root = dict(reversed(tuple(runtime.root.items())))
    bank, spec = runtime._device_route_banks["gate_up"]
    runtime._device_route_banks["gate_up"] = (bank, copy(spec))
    compute._runtime_guard_plan.check(runtime)


def test_native_batch_one_call_all_layers_and_no_cross_step_pass(monkeypatch):
    computes = tuple(make_compute() for _ in range(43))
    batch = module.NativeRuntimeGuardBatch(computes)
    with no_host_tensor_reads(monkeypatch):
        batch.check(computes)
        batch.check(computes)
    assert batch.native_plan.calls == 2
    assert all(compute._runtime_guard_plan.native_plan.calls == 0 for compute in computes)
    assert len(batch.native_plan.snapshots) == 43 * 3
    computes[-1].runtime.root["gate.weight"].set_(torch.zeros_like(computes[-1].runtime.root["gate.weight"]))
    with pytest.raises(RuntimeError, match="signature changed"):
        batch.check(computes)
    assert batch.native_plan.calls == 3


def test_native_batch_does_not_recapture_metadata_on_aggregation():
    compute = make_compute()
    compute.runtime.root["gate.weight"].transpose_(0, 1)
    batch = module.NativeRuntimeGuardBatch([compute])
    with pytest.raises(RuntimeError, match="signature changed"):
        batch.check([compute])


@pytest.mark.parametrize("mutation", ["compute", "count", "order", "plan", "mode", "runtime"])
def test_batch_structure_cannot_bypass_capture_contract(mutation):
    computes = [make_compute(), make_compute()]
    batch = module.NativeRuntimeGuardBatch(computes)
    if mutation == "compute":
        computes[0] = copy(computes[0])
    elif mutation == "count":
        computes.pop()
    elif mutation == "order":
        computes.reverse()
    elif mutation == "plan":
        computes[0]._runtime_guard_plan = object()
    elif mutation == "mode":
        computes[0]._runtime_guard_mode = "planned"
    else:
        computes[0].runtime = copy(computes[0].runtime)
    with pytest.raises(RuntimeError, match="signature changed"):
        batch.check(computes)
    assert batch.native_plan.calls == 0


def test_native_plan_retains_old_owners_and_checks_live_replacement():
    compute = make_compute()
    runtime, plan = compute.runtime, compute._runtime_guard_plan
    old_root = weakref.ref(runtime.root["gate.weight"])
    old_lookup = weakref.ref(runtime._device_route_banks["lookup"])
    old_bank = weakref.ref(runtime._device_route_banks["gate_up"][0])
    runtime.root["gate.weight"] = runtime.root["gate.weight"].clone()
    runtime._device_route_banks["lookup"] = runtime._device_route_banks["lookup"].clone()
    _, spec = runtime._device_route_banks["gate_up"]
    runtime._device_route_banks["gate_up"] = (object(), spec)
    gc.collect()
    assert old_root() is not None and old_lookup() is not None and old_bank() is not None
    with pytest.raises(RuntimeError, match="signature changed"):
        plan.check(runtime)


def test_decoder_uses_one_native_batch_before_metadata_and_input_copy(monkeypatch):
    bank, backend, _, context = make_bank()
    bank.ready = False
    bank.computes = (make_compute(), make_compute())
    bank.prepare_runtime_guard()
    bank.ready = True
    backend.fail_sync = True
    bank.replay(torch.tensor([10]), torch.tensor([3]), context)
    assert bank._runtime_guard_batch.native_plan.calls == 1
    copies = bank.entries[3]["metadata"].copies
    tokens = bank.entries[3]["tokens"].clone()
    bank.computes[-1].runtime.root["gate.weight"] = bank.computes[-1].runtime.root["gate.weight"].clone()
    with pytest.raises(RuntimeError, match="signature changed"):
        bank.replay(torch.tensor([20]), torch.tensor([3]), context)
    assert bank.replays == 1 and bank.failed
    assert bank.entries[3]["metadata"].copies == copies
    assert torch.equal(bank.entries[3]["tokens"], tokens)
    backend.fail_sync = False
    bank.close()
    assert bank._runtime_guard_batch is None


def test_decoder_does_not_lazily_capture_a_missing_native_guard():
    bank, _, _, context = make_bank()
    bank.computes = (make_compute(),)
    with pytest.raises(RuntimeError, match="not aggregated at startup"):
        bank.replay(torch.tensor([10]), torch.tensor([3]), context)
    assert bank.replays == 0


def test_native_guard_startup_and_mixed_modes_fail_closed():
    with pytest.raises(ValueError, match="requires captured computes"):
        module.NativeRuntimeGuardBatch([])
    compute = make_compute()
    compute._runtime_guard_mode = "planned"
    with pytest.raises(ValueError, match="native plans for every"):
        module.NativeRuntimeGuardBatch([make_compute(), compute])
    bank, _, _, _ = make_bank()
    with pytest.raises(RuntimeError, match="startup-only"):
        bank.prepare_runtime_guard()


def test_host_probe_plan_is_torch_free_and_does_not_claim_native_success(capsys):
    assert probe.main(["--build-cpu", "--plan-only"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PLANNED" and result["native_host_verified"] is False
    assert not any(result[name] for name in result if name.endswith("_verified"))
    for node in ast.parse(inspect.getsource(probe)).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert "torch" not in ast.unparse(node)


def test_host_probe_matrix_with_python_oracle_only():
    result = probe.run_checks(HostMetadataOracle)
    assert set(probe.MUTATIONS) <= set(result)
    assert "append_preserves_snapshot" in result and "strong_owner_and_release" in result


def test_native_source_is_host_only_and_checks_complete_metadata():
    source = probe.SOURCE.read_text(encoding="utf-8")
    for requirement in (
        "at::Tensor owner",
        "c10::Storage storage_owner",
        "current.unsafeGetTensorImpl() == implementation",
        "current.layout() == layout",
        "current.device() == device",
        "current.scalar_type() == dtype",
        "current.sizes().equals(sizes)",
        "current.strides().equals(strides)",
        "current.storage_offset() == offset",
        "current.const_data_ptr() == pointer",
        "other->snapshots_.begin()",
    ):
        assert requirement in source
    for forbidden in ('#include "acl/', '#include "torch_npu/', "getCurrentNPUStream(", ".cpu(", ".item("):
        assert forbidden not in source
