# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transactional fast-plan checks and a CPU-only metadata DAG microbenchmark."""

import weakref
from copy import copy
from dataclasses import dataclass
from statistics import median
from time import perf_counter

import pytest
import torch

from vllm_ascend.quantization import vq2a8_v4_decoder_graph as graph_module
from vllm_ascend.quantization.vq2a8_v4_decoder_graph import (
    FastPlannedDecoderMetadataBuffers,
    PlannedDecoderMetadataBuffers,
    V4DecoderGraphBank,
)


class DeviceTensor(torch.Tensor):
    """CPU storage/non-CPU tag for copy protocol checks, never device evidence."""

    @property
    def device(self):
        return torch.device("cuda:0")


def device_tensor(value):
    return torch.Tensor._make_subclass(DeviceTensor, torch.tensor(value), False)


@dataclass
class SharedMetadata:
    tensors: dict
    positions: list
    valid: bool = True


@dataclass
class LayerMetadata:
    shared: SharedMetadata
    live_views: tuple
    num_decodes: int = 1


def make_dag():
    """43 wrappers, shared containers, plus distinct views of shared storage."""
    tensors = tuple(torch.ones(4, dtype=torch.int64) for _ in range(8))
    shared = SharedMetadata({str(i): value for i, value in enumerate(tensors)}, [3])
    return {str(layer): LayerMetadata(shared, tuple(value.view_as(value) for value in tensors)) for layer in range(43)}


@pytest.mark.parametrize("buffers", (PlannedDecoderMetadataBuffers, FastPlannedDecoderMetadataBuffers))
def test_fast_dag_shared_paths_and_replacements(buffers):
    source = make_dag()
    owner = buffers(source)
    for _ in range(3):
        current = make_dag()
        owner.update(current)
        # Replacing a previously shared container must not bypass its checks.
        current["42"].shared = copy(current["42"].shared)
        owner.update(current)
        current["42"].shared.positions = [4]
        with pytest.raises(ValueError, match="constant"):
            owner.update(current)


def test_fast_live_sources_and_failed_updates_are_transactional():
    owner = FastPlannedDecoderMetadataBuffers({"dynamic": device_tensor([1]), "constant": 3})
    target = owner.tree["dynamic"]
    for value in (7, 9, 11):
        source = {"dynamic": device_tensor([value]), "constant": 3}
        owner.update(source)
        assert target.tolist() == [value]
        source["dynamic"].fill_(value + 1)
        owner.update(source)
        assert target.tolist() == [value + 1]
    assert owner.copies == 6
    with pytest.raises(ValueError, match="constant"):
        owner.update({"dynamic": device_tensor([99]), "constant": 4})
    assert target.tolist() == [12]
    assert owner.copies == 6


@pytest.mark.parametrize("change", ("dtype", "shape", "stride", "values", "type", "length", "keys"))
def test_fast_bad_late_source_leaves_never_modify_early_device_target(change):
    source = {"device": device_tensor([1]), "cpu": [torch.ones(2, 2)]}
    owner = FastPlannedDecoderMetadataBuffers(source)
    source["device"].fill_(8)
    if change == "dtype":
        source["cpu"][0] = source["cpu"][0].double()
    elif change == "shape":
        source["cpu"][0] = source["cpu"][0].reshape(4)
    elif change == "stride":
        source["cpu"][0] = source["cpu"][0].t()
    elif change == "values":
        source["cpu"][0].fill_(2)
    elif change == "type":
        source["cpu"] = tuple(source["cpu"])
    elif change == "length":
        source["cpu"].append(None)
    else:
        source["new"] = None
    with pytest.raises(ValueError):
        owner.update(source)
    assert owner.tree["device"].tolist() == [1]
    assert owner.copies == 0


@pytest.mark.parametrize("change", ("shape", "storage", "dtype", "stride"))
def test_fast_checks_captured_target_contract_and_pointer(change):
    value = device_tensor([[1, 2], [3, 4]])
    owner = FastPlannedDecoderMetadataBuffers({"x": value})
    target = owner.tree["x"]
    if change == "shape":
        target.resize_(8)
    elif change == "storage":
        target.set_(target.clone())
    elif change == "dtype":
        target.data = target.float()
    else:
        target.transpose_(0, 1)
    with pytest.raises(ValueError, match="tensor contract"):
        owner.update({"x": value})
    assert owner.copies == 0


def test_fast_alias_divergence_checks_current_source_each_update():
    original = device_tensor([1])
    owner = FastPlannedDecoderMetadataBuffers({"a": original, "b": original.view_as(original)})
    for value in (5, 7):
        live = device_tensor([value])
        owner.update({"a": live, "b": live.view_as(live)})
        assert owner.tree["a"].tolist() == [value]
        with pytest.raises(ValueError, match="alias topology"):
            owner.update({"a": live, "b": live.clone()})
    assert owner.copies == 2


def test_fast_memoizes_tensor_contracts_only_within_update(monkeypatch):
    source = make_dag()
    owner = FastPlannedDecoderMetadataBuffers(source)
    original = graph_module._tensor_contract
    seen = []

    def measured(value):
        seen.append(id(value))
        return original(value)

    monkeypatch.setattr(graph_module, "_tensor_contract", measured)
    owner.update(source)
    # Eight distinct views per layer plus eight original leaves and eight
    # captured targets, regardless of the number of shared-container paths.
    assert len(seen) == len(set(seen)) == 43 * 8 + 8 + 8
    once = list(seen)
    owner.update(source)
    assert seen == once + once


def test_fast_does_not_retain_replacement_sources():
    owner = FastPlannedDecoderMetadataBuffers({"x": device_tensor([1])})
    replacement = device_tensor([2])
    reference = weakref.ref(replacement)
    owner.update({"x": replacement})
    del replacement
    assert reference() is None


def test_fast_preserves_immutable_context_and_rejects_storage_changes():
    @dataclass
    class Fields:
        mutable: list
        full_compress_cos: list

    value = device_tensor([1])
    source = [value]
    owner = FastPlannedDecoderMetadataBuffers(Fields(source, source))
    assert owner.tree.mutable[0] is not value
    assert owner.tree.full_compress_cos[0] is value
    with pytest.raises(ValueError, match="immutable"):
        owner.update(Fields(source, [value.clone()]))
    assert owner.copies == 0


def test_fast_rejects_schema_changes_without_runtime_field_discovery(monkeypatch):
    source = make_dag()
    owner = FastPlannedDecoderMetadataBuffers(source)
    monkeypatch.setattr(graph_module, "fields", lambda _: pytest.fail("no runtime dataclass reflection"))
    owner.update(source)
    monkeypatch.setitem(SharedMetadata.__dataclass_fields__, "future", SharedMetadata.__dataclass_fields__["valid"])
    with pytest.raises(ValueError, match="dataclass fields"):
        owner.update(source)


@pytest.mark.parametrize("replacement", (True, 1.0, "1", None))
def test_fast_constant_types_are_exact(replacement):
    owner = FastPlannedDecoderMetadataBuffers({"value": 1})
    with pytest.raises(ValueError, match="type"):
        owner.update({"value": replacement})


def test_fast_rejects_cycles_and_unknown_objects():
    value = {}
    value["cycle"] = value
    with pytest.raises(TypeError, match="Cyclic"):
        FastPlannedDecoderMetadataBuffers(value)
    with pytest.raises(TypeError, match="Unsupported"):
        FastPlannedDecoderMetadataBuffers({"unknown": object()})
    owner = FastPlannedDecoderMetadataBuffers({"value": 1})
    with pytest.raises(ValueError, match="type"):
        owner.update(owner)


def test_bank_accepts_fast_mode_without_changing_default():
    assert V4DecoderGraphBank(None, 4, backend=object()).metadata_mode == "recursive"
    assert V4DecoderGraphBank(None, 4, backend=object(), metadata_mode="planned_fast").metadata_mode == "planned_fast"


def test_cpu_metadata_dag_microbenchmark():
    """Report host validation cost only; no flaky wall-time speed assertion."""
    source = make_dag()
    owners = {
        "planned": PlannedDecoderMetadataBuffers(source),
        "planned_fast": FastPlannedDecoderMetadataBuffers(source),
    }
    iterations = 100
    samples = {name: [] for name in owners}
    for owner in owners.values():
        for _ in range(10):
            owner.update(source)
    # Alternate ordering each round to avoid consistently favoring the second
    # implementation; fixture, values and CPU equality work remain identical.
    for round_index in range(6):
        names = tuple(owners) if round_index % 2 == 0 else tuple(reversed(owners))
        for name in names:
            start = perf_counter()
            for _ in range(iterations):
                owners[name].update(source)
            samples[name].append((perf_counter() - start) * 1e6 / iterations)
    medians = {name: round(median(values), 3) for name, values in samples.items()}
    assert owners["planned"].copies == owners["planned_fast"].copies == 0
    print(f"CPU_METADATA_DAG_MICROBENCH_US={medians}; not NPU or end-to-end TPOT evidence")
