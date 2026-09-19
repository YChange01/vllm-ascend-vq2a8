# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""I model plumbing with independent CPU oracles, not NPU integration proof."""

from types import SimpleNamespace as NS

import pytest
import torch

from tests.ut.quantization.test_vq2a8_activation_direct import StridedSignOracle
from tests.ut.quantization.test_vq2a8_activation_packed import assert_bytes, fixture
from tests.ut.quantization.test_vq2a8_v4_combined_activation import ProjectionSelectionOracle
from tests.ut.quantization.test_vq2a8_v4_device_route import no_host_tensor_reads
from tests.ut.quantization.test_vq2a8_v4_v2_runtime import fake_runtime
from vllm_ascend.quantization import vq2a8_v4_device_route as route
from vllm_ascend.quantization.vq2a8_activation_packed import PackedRowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_optimization import configure_runtime
from vllm_ascend.quantization.vq2a8_reference import deepseek_v4_swiglu_reference
from vllm_ascend.quantization.vq2a8_select_sign import FusedSwigluSelectSign


class NativeOracle(StridedSignOracle):
    def select_sign_version(self):
        return 1

    def swiglu_select_sign_version(self):
        return 1


class BankOracle(ProjectionSelectionOracle):
    """Torch reference in place of tested native code, not an NPU emulator."""

    def __init__(self, width, output_width):
        super().__init__(width, output_width)
        self.native_swiglu_calls = []
        self.prepared = []

    def select(self, slots):
        scale, bias, signs, valid = super().select(slots)
        return (
            torch.where(valid[:, None] != 0, scale, float("nan")),
            torch.where(valid[:, None] != 0, bias, float("nan")),
            torch.where(valid[:, None] != 0, signs, 0),
            valid,
        )

    def select_sign(self, hidden, slots):
        scale, bias, signs, valid = self.select(slots)
        signed, input_valid = NativeOracle().activation_sign_strided(hidden, scale, bias, signs)
        return signed, scale, bias, valid, input_valid

    def swiglu_select_sign(self, gate_up, slots, limit):
        self.native_swiglu_calls.append((gate_up.data_ptr(), slots.data_ptr(), limit))
        return self.select_sign(deepseek_v4_swiglu_reference(gate_up, limit), slots)

    def _project(self, method, quantized, scale, bias, slots):
        self.prepared.append((quantized, scale, bias))
        return super()._project(method, quantized, scale, bias, slots)


def preparation(*, fused=True, native=None):
    return PackedRowwiseVQ2A8Preparation(
        fuse_sign=True,
        strided_sign=True,
        direct_output=True,
        fuse_select=True,
        fuse_swiglu=fused,
        native_ops=NativeOracle() if native is None else native,
    )


@pytest.mark.parametrize("abi", (None, True, False, 0, 2, "1", 1.0))
def test_i_projection_wrapper_requires_strict_abi(abi):
    with pytest.raises(RuntimeError, match="ABI"):
        FusedSwigluSelectSign(NS(swiglu_select_sign_version=lambda: abi))


def test_i_projection_default_never_requires_new_abi():
    native = StridedSignOracle()
    native.select_sign_version = lambda: 1
    assert preparation(fused=False, native=native)._swiglu_select_sign is None
    with pytest.raises(RuntimeError, match="no implicit fallback"):
        preparation(native=native)
    with pytest.raises(RuntimeError, match="no implicit fallback"):
        FusedSwigluSelectSign(NativeOracle())(
            NS(), torch.zeros(1, 4096, dtype=torch.bfloat16), torch.zeros(1, dtype=torch.int64), 7.0
        )


@pytest.mark.parametrize("limit,forwarded", ((None, 0.0), (0, 0.0), (7, 7.0), (7.1, 7.1)))
def test_i_projection_wrapper_forwards_original_storage_without_host_reads(monkeypatch, limit, forwarded):
    x = torch.zeros(2, 4096, dtype=torch.bfloat16)
    slots = torch.zeros(2, dtype=torch.int64)
    calls, result = [], object()

    def native(value, ids, bound):
        calls.append((value is x, ids is slots, bound))
        return result

    candidate = FusedSwigluSelectSign(NativeOracle())
    with no_host_tensor_reads(monkeypatch):
        assert candidate(NS(swiglu_select_sign=native), x, slots, limit) is result
    assert calls == [(True, True, forwarded)]


@pytest.mark.parametrize("limit", (-1, True, False, "7", float("inf"), float("nan"), 1e300))
def test_i_projection_invalid_limit_rejected_before_native(limit):
    with pytest.raises(ValueError, match="limit"):
        FusedSwigluSelectSign(NativeOracle())(
            NS(swiglu_select_sign=lambda *_: pytest.fail("native called")),
            torch.zeros(1, 4096, dtype=torch.bfloat16),
            torch.zeros(1, dtype=torch.int64),
            limit,
        )


@pytest.mark.parametrize("bad", range(11))
def test_i_projection_invalid_geometry_rejected_before_native(bad):
    x = torch.zeros(2, 4096, dtype=torch.bfloat16)
    ids = torch.zeros(2, dtype=torch.int64)
    cases = [
        (None, ids),
        (x.float(), ids),
        (x.flatten(), ids),
        (x[:, :2048], ids),
        (x[:0], ids[:0]),
        (x.as_strided((2, 4096), (32, 1)), ids),
        (torch.zeros(2, 8192, dtype=x.dtype)[:, ::2], ids),
        (torch.zeros(8193, dtype=x.dtype)[1:].reshape(2, 4096), ids),
        (x, ids.int()),
        (x, ids[:1]),
        (x, torch.zeros(4, dtype=torch.int64)[::2]),
    ]
    with pytest.raises(ValueError, match="SwiGLU/select/sign"):
        FusedSwigluSelectSign(NativeOracle())(
            NS(swiglu_select_sign=lambda *_: pytest.fail("native called")), *cases[bad], 7.0
        )


@pytest.mark.parametrize("width", (2048, 4096))
@pytest.mark.parametrize("groups", (1, 3, 6))
@pytest.mark.parametrize("limit", (None, 0.0, 7.0, 7.1))
@pytest.mark.parametrize("raw", (False, True))
def test_i_projection_reuses_unchanged_rht_bias_quantization_byte_for_byte(monkeypatch, width, groups, limit, raw):
    x, scale, bias, signs, spec = fixture(width, groups)
    gate_up = torch.cat((x, x.flip(-1)), dim=-1)
    slots = torch.arange(groups, dtype=torch.int64)
    bank = BankOracle(width, 1)
    bank.scale[:groups], bank.bias[:groups], bank.signs[:groups] = scale, bias, signs
    want = preparation(fused=False).packed_resident(
        bank,
        deepseek_v4_swiglu_reference(gate_up, limit),
        slots,
        spec,
        validity=lambda _: None,
    )
    flags, statuses = [], [] if raw else None
    candidate = preparation()
    with no_host_tensor_reads(monkeypatch):
        got = candidate.packed_resident_swiglu(
            bank,
            gate_up,
            slots,
            spec,
            swiglu_limit=limit,
            validity=flags.append,
            raw_statuses=statuses,
        )
    for actual, expected in zip(got, want):
        assert_bytes(actual, expected)
    assert len(bank.native_swiglu_calls) == 1
    if raw:
        assert flags == []
        assert len(statuses) == 2
        assert all(value.dtype == torch.int32 and tuple(value.shape) == (groups,) for value in statuses)
    else:
        assert len(flags) == 2 and all(bool(flag) for flag in flags)


def test_i_projection_packed_explicit_only_and_geometry_fail_closed():
    gate = torch.zeros(1, 4096, dtype=torch.bfloat16)
    ids = torch.zeros(1, dtype=torch.int64)
    spec = NS(columns=2048, rht_true_columns=2048, rht_block_size=128)
    with pytest.raises(ValueError, match="explicit SwiGLU"):
        preparation(fused=False).packed_resident_swiglu(None, gate, ids, spec, swiglu_limit=7, validity=lambda _: None)
    spec.rht_block_size = 64
    with pytest.raises(ValueError, match="RHT128"):
        preparation().packed_resident_swiglu(None, gate, ids, spec, swiglu_limit=7, validity=lambda _: None)
    with pytest.raises(ValueError, match="explicit resident"):
        PackedRowwiseVQ2A8Preparation(fuse_swiglu=True)


def runtime_fixture(monkeypatch):
    runtime = fake_runtime()
    runtime.v4_activation_preparation = "sign_fused_direct"
    runtime.v4_activation_reorder = "vectorized"
    runtime.v4_select_sign = "fused"
    runtime.v4_swiglu_mode = "fused_select_sign"
    runtime.config.hidden_size = 2048
    runtime.root["gate.weight"] = torch.zeros(6, 2048)
    runtime._cache, runtime._v2_payload_locations = {}, {}
    del runtime._row_preparation
    for kind in ("gate_up", "down"):
        n, k = (4096, 2048) if kind == "gate_up" else (2048, 2048)
        bank = BankOracle(k, n)
        spec = NS(rows=n, columns=k, rht_true_columns=k, rht_block_size=128)
        runtime._device_route_banks[kind] = bank, spec
        for slot in range(6):
            payload = {"weight_scale": bank.scale[slot], "weight_bias": bank.bias[slot], "rht_sign": bank.signs[slot]}
            runtime._cache.setdefault(slot, {})[kind] = payload, spec
            runtime._v2_payload_locations[id(payload)] = kind, slot
    monkeypatch.setattr(torch.ops, "vq2a8_ascendc_v4_v2", NativeOracle())
    configure_runtime(runtime, "device_route_decode")
    return runtime


@pytest.mark.parametrize("raw", (False, True))
def test_i_projection_graph_only_down_eager_reference_and_shared_unchanged(monkeypatch, raw):
    runtime = runtime_fixture(monkeypatch)
    retained = []
    if raw:

        def layer_check(statuses, outputs, flags):
            assert len(statuses) == 6
            retained.append(statuses)
            return torch.stack(
                [*(x.ne(0).all() for x in statuses), *(torch.isfinite(x).all() for x in outputs), *flags]
            ).all()

        monkeypatch.setattr(route, "_make_layer_validity", lambda _: layer_check)
    shared_calls = []
    runtime.config.num_shared = 1
    runtime.shared = lambda value: shared_calls.append(value.data_ptr()) or value / 16
    compute = route.DeviceRouteGraphCompute(runtime)
    assert compute.signature["swiglu_mode"] == "fused_select_sign"
    hidden = torch.randn(1, 2048, generator=torch.Generator().manual_seed(8)).bfloat16()
    token = torch.zeros(1, dtype=torch.int64)
    baseline_swiglu = route.deepseek_v4_swiglu_reference
    references = []

    def original_reference(value, limit):
        references.append(value.data_ptr())
        return baseline_swiglu(value, limit)

    monkeypatch.setattr(route, "deepseek_v4_swiglu_reference", original_reference)
    with no_host_tensor_reads(monkeypatch):
        eager = runtime._optimization.forward(runtime, hidden, token)
        eager_valid = runtime._optimization.valid
        eager_stats = dict(runtime._optimization.stats)
        eager_preparation = runtime._row_preparation
        actual, valid = compute(hidden, token)
    assert torch.equal(eager, actual) and bool(valid)
    assert runtime._optimization.valid is eager_valid
    assert runtime._optimization.stats == eager_stats
    assert runtime._row_preparation is eager_preparation
    assert all(item is not eager_preparation for item in compute.preparations.values())
    assert len(references) == 1  # graph has no original Torch SwiGLU call
    assert len(shared_calls) == 2
    assert runtime.v4_swiglu_reference_calls == runtime.v4_swiglu_graph_build_calls == 1
    assert not runtime._device_route_banks["gate_up"][0].native_swiglu_calls
    assert len(runtime._device_route_banks["down"][0].native_swiglu_calls) == 1
    if raw:
        assert len(retained) == 1
    # General projection/prefill remains ordinary preparation, even with I set.
    requests = [(hidden.expand(count, -1), *runtime._cache[slot]["gate_up"]) for slot, count in ((0, 1), (4, 3))]
    runtime._projections_many(requests)
    assert len(runtime._device_route_banks["down"][0].native_swiglu_calls) == 1


@pytest.mark.parametrize("raw", (False, True))
@pytest.mark.parametrize("corrupt", ("slot", "input", "scale", "sign"))
def test_i_projection_invalid_then_recovered_flags_are_fresh(monkeypatch, corrupt, raw):
    runtime = runtime_fixture(monkeypatch)
    if raw:

        def layer_check(statuses, outputs, flags):
            assert len(statuses) == 6
            return torch.stack(
                [*(x.ne(0).all() for x in statuses), *(torch.isfinite(x).all() for x in outputs), *flags]
            ).all()

        monkeypatch.setattr(route, "_make_layer_validity", lambda _: layer_check)
    # A previous eager error must neither poison graph output nor be erased.
    eager_invalid = torch.tensor(False)
    runtime._optimization.valid = eager_invalid
    compute = route.DeviceRouteGraphCompute(runtime)
    hidden = torch.ones(1, 2048, dtype=torch.bfloat16)
    token = torch.zeros(1, dtype=torch.int64)
    original, valid = compute(hidden, token)
    assert bool(valid)
    bank = runtime._device_route_banks["down"][0]
    target = {"slot": runtime._device_route_banks["lookup"], "input": hidden, "scale": bank.scale, "sign": bank.signs}[
        corrupt
    ]
    backup = target.clone()
    target.fill_({"slot": -1, "input": float("nan"), "scale": float("nan"), "sign": 0}[corrupt])
    _, invalid = compute(hidden, token)
    assert not bool(invalid)
    target.copy_(backup)
    recovered, valid_again = compute(hidden, token)
    assert bool(valid_again) and not bool(invalid)
    assert torch.equal(recovered, original)
    assert runtime._optimization.valid is eager_invalid
    assert not bool(eager_invalid)


def test_i_projection_contract_rejects_mode_change_after_construction(monkeypatch):
    runtime = runtime_fixture(monkeypatch)
    compute = route.DeviceRouteGraphCompute(runtime)
    runtime.v4_swiglu_mode = "torch"
    with pytest.raises(RuntimeError, match="signature changed"):
        compute.check_runtime_contract(runtime)
