# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C/D projection wiring and A validity CPU contracts, not NPU acceptance."""

from types import SimpleNamespace as NS

import pytest
import torch

from tests.ut.quantization.test_vq2a8_activation_packed import assert_bytes, fixture, reference
from tests.ut.quantization.test_vq2a8_v4_device_route import no_host_tensor_reads
from vllm_ascend.quantization import vq2a8_v4_device_route as route
from vllm_ascend.quantization.vq2a8_activation_packed import PackedRowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_validity_fused import FusedLayerValidity


def sign_oracle(hidden, scale, bias, signs):
    widened = hidden.float()
    valid = torch.isfinite(widened).all(-1) & torch.isfinite(scale).all(-1) & torch.isfinite(bias).all(-1)
    valid &= ((signs == -1) | (signs == 1)).all(-1)
    return widened * signs.float(), valid.int()


class SignOpsOracle:
    """Only the sign primitive is substituted; production RHT/quantization runs."""

    def __init__(self):
        self.calls = []
        self.statuses = []

    def activation_preparation_version(self):
        return 1

    def activation_sign_strided_version(self):
        return 1

    def select_sign_version(self):
        return 1

    def activation_sign(self, *args):
        raise AssertionError("Direct strided preparation must not materialize the old sign input")

    def activation_sign_strided(self, *args):
        self.calls.append(args)
        result = sign_oracle(*args)
        self.statuses.append(result[1])
        return result


class BankOracle:
    """Device-indexed selection with the native invalid-row poison contract."""

    def __init__(self, width):
        _, self.scale, self.bias, self.signs, self.spec = fixture(width, 6)
        self.select_calls = []
        self.fused_calls = []
        self.selection_statuses = []
        self.input_statuses = []

    def selected(self, slots):
        valid = (slots >= 0) & (slots < self.scale.shape[0])
        safe = slots.clamp(0, self.scale.shape[0] - 1)
        scale = torch.where(valid[:, None], self.scale.index_select(0, safe), float("nan"))
        bias = torch.where(valid[:, None], self.bias.index_select(0, safe), float("nan"))
        signs = torch.where(valid[:, None], self.signs.index_select(0, safe), 0)
        status = valid.int()
        self.selection_statuses.append(status)
        return scale, bias, signs, status

    def select(self, slots):
        self.select_calls.append(slots)
        return self.selected(slots)

    def select_sign(self, hidden, slots):
        self.fused_calls.append((hidden, slots))
        scale, bias, signs, selected = self.selected(slots)
        signed, valid = sign_oracle(hidden, scale, bias, signs)
        self.input_statuses.append(valid)
        return signed, scale, bias, selected, valid


def preparation(native, *, fused):
    return PackedRowwiseVQ2A8Preparation(
        fuse_sign=True, strided_sign=True, direct_output=True, fuse_select=fused, native_ops=native
    )


def hidden_view(width, dtype, layout, groups=6):
    values = fixture(width, groups, dtype=dtype)[0]
    if layout == "expanded":
        return values[:1].expand(groups, -1)
    if layout == "padded":
        owner = torch.zeros(groups, width + 32, dtype=dtype)
        view = owner[:, 16 : 16 + width]
        view.copy_(values)
        return view
    return values


def fp8_tail(normalized):
    limit = torch.finfo(torch.float8_e4m3fn).max
    return normalized.clamp(-limit, limit).to(torch.float8_e4m3fn).contiguous()


@pytest.mark.parametrize("width", [2048, 4096])
@pytest.mark.parametrize("groups", [1, 2, 6])
@pytest.mark.parametrize("direct_output", [False, True])
@pytest.mark.parametrize("case", ["random", "zero", "tiny", "impulse"])
def test_normalized_boundary_preserves_original_torch_tail_bytes(width, groups, direct_output, case):
    hidden, scale, bias, signs, spec = fixture(width, groups)
    if case == "zero":
        hidden.zero_()
    elif case == "tiny":
        hidden.mul_(1e-15)
    elif case == "impulse":
        hidden.zero_()
        hidden[:, -1] = -1
    prep = PackedRowwiseVQ2A8Preparation(
        fuse_sign=direct_output,
        strided_sign=direct_output,
        direct_output=direct_output,
        native_ops=SignOpsOracle(),
    )
    prep.prepare_for_graph(hidden.device, 128)
    signed = hidden.float() * signs.float()
    expected = prep._from_signed(signed, scale, bias, spec)
    normalized, actual_scale, actual_bias = prep._from_signed(signed, scale, bias, spec, return_normalized=True)
    assert normalized.dtype == torch.float32 and normalized.is_contiguous()
    actual = fp8_tail(normalized), actual_scale, actual_bias
    original = reference(hidden, scale, bias, signs, spec)
    for got, want, old in zip(actual, expected, original):
        assert_bytes(got, want)
        assert_bytes(got, old)
        assert got.is_contiguous()


@pytest.mark.parametrize("width", [2048, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["contiguous", "expanded", "padded"])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("raw", [False, True])
def test_resident_preparation_matches_select_then_sign_and_status_order(
    monkeypatch, width, dtype, layout, normalized, raw
):
    bank, native = BankOracle(width), SignOpsOracle()
    hidden = hidden_view(width, dtype, layout)
    slots = torch.tensor([5, 0, 3, 3, 0, 1], dtype=torch.int64)
    scale, bias, signs, selected = bank.select(slots)
    expected_statuses, expected_flags = [selected], []
    expected = preparation(native, fused=False).packed(
        hidden,
        scale,
        bias,
        signs,
        bank.spec,
        validity=expected_flags.append,
        raw_input_validity=expected_statuses.append,
        return_normalized=normalized,
    )
    raw_statuses, flags = ([] if raw else None), []
    prep = preparation(native, fused=True)
    with no_host_tensor_reads(monkeypatch):
        actual = prep.packed_resident(
            bank,
            hidden,
            slots,
            bank.spec,
            validity=flags.append,
            raw_statuses=raw_statuses,
            return_normalized=normalized,
        )
    assert len(bank.select_calls) == len(bank.fused_calls) == len(native.calls) == 1
    assert bank.fused_calls[0][0] is hidden and bank.fused_calls[0][1] is slots
    for got, want in zip(actual, expected):
        assert_bytes(got, want)
    if raw:
        assert flags == [] and len(raw_statuses) == 2
        assert raw_statuses[0] is bank.selection_statuses[-1]
        assert raw_statuses[1] is bank.input_statuses[-1]
        for got, want in zip(raw_statuses, expected_statuses):
            assert_bytes(got, want)
    else:
        assert len(flags) == 2 and all(bool(flag) for flag in flags)


class ProjectionOracle:
    """Projection spy: tail dispatch, statuses and output ownership are observable."""

    def __init__(self, select_sign, tail):
        self.v4_select_sign = select_sign
        self.v4_activation_tail = tail
        self.calls = []
        self.statuses = []
        self.outputs = []
        self.failure = None

    def _project(self, mode, bank, activation, scale, bias, slots):
        self.calls.append((mode, bank, activation, scale, bias, slots))
        output = torch.zeros(slots.numel(), 2048, dtype=torch.bfloat16)
        valid = torch.ones(slots.numel(), dtype=torch.int32)
        if self.failure == "projection":
            valid[0] = 0
        if self.failure == "output":
            output[0, -1] = float("nan")
        self.statuses.append(valid)
        self.outputs.append(output)
        return output, valid

    def project_v4_prepared(self, *args):
        return self._project("prepared", *args)

    def project_v4_normalized(self, *args):
        return self._project("normalized", *args)


@pytest.mark.parametrize(
    "select_sign,tail", [("fused", "torch"), ("separate", "fused_reorder"), ("fused", "fused_reorder")]
)
@pytest.mark.parametrize("raw", [False, True])
def test_c_d_dispatch_once_without_duplicate_select_sign_or_preparation(monkeypatch, select_sign, tail, raw):
    bank, native = BankOracle(2048), SignOpsOracle()
    runtime = ProjectionOracle(select_sign, tail)
    hidden = hidden_view(2048, torch.bfloat16, "expanded")
    slots = torch.tensor([5, 1, 2, 1, 0, 5], dtype=torch.int64)
    prep = preparation(native, fused=select_sign == "fused")
    transform_calls = []
    original_transform = prep._from_signed

    def transform(*args, **kwargs):
        transform_calls.append((args, kwargs))
        return original_transform(*args, **kwargs)

    monkeypatch.setattr(prep, "_from_signed", transform)
    flags, raw_statuses = [], [] if raw else None
    with no_host_tensor_reads(monkeypatch):
        output = route._candidate_projection(runtime, bank, bank.spec, prep, hidden, slots, flags.append, raw_statuses)
    assert route._uses_projection_candidate(runtime)
    fused = select_sign == "fused"
    assert len(bank.fused_calls) == int(fused)
    assert len(bank.select_calls) == len(native.calls) == int(not fused)
    assert len(transform_calls) == len(runtime.calls) == 1
    assert transform_calls[0][1] == {"return_normalized": tail == "fused_reorder"}
    mode, called_bank, activation, scale, bias, called_slots = runtime.calls[0]
    assert mode == ("normalized" if tail == "fused_reorder" else "prepared")
    assert activation.dtype == (torch.float32 if tail == "fused_reorder" else torch.float8_e4m3fn)
    assert called_bank is bank and called_slots is slots and output is runtime.outputs[0]
    if raw:
        assert flags == [] and len(raw_statuses) == 3
        assert raw_statuses[0] is bank.selection_statuses[0]
        assert raw_statuses[1] is (bank.input_statuses[0] if fused else native.statuses[0])
        assert raw_statuses[2] is runtime.statuses[0]
    else:
        assert len(flags) == 3 and all(bool(flag) for flag in flags)
    expected = reference(
        hidden,
        bank.scale.index_select(0, slots),
        bank.bias.index_select(0, slots),
        bank.signs.index_select(0, slots),
        bank.spec,
    )
    prepared = fp8_tail(activation) if tail == "fused_reorder" else activation
    for got, want in zip((prepared, scale, bias), expected):
        assert_bytes(got, want)


def validity_oracle(statuses, outputs, route_flags):
    return torch.stack(
        [*(status.ne(0).all() for status in statuses), *(torch.isfinite(out).all() for out in outputs), *route_flags]
    ).all()


@pytest.mark.parametrize(
    "select_sign,tail", [("fused", "torch"), ("separate", "fused_reorder"), ("fused", "fused_reorder")]
)
@pytest.mark.parametrize("validity_mode", ["torch", "fused", "fused_vectorized"])
@pytest.mark.parametrize("failure", ["selection", "input", "projection", "output", "combined", "route"])
def test_candidate_validity_is_not_lost_and_recovers_each_forward(
    monkeypatch, select_sign, tail, validity_mode, failure
):
    bank, native = BankOracle(2048), SignOpsOracle()
    runtime = ProjectionOracle(select_sign, tail)
    prep = preparation(native, fused=select_sign == "fused")
    base_hidden = hidden_view(2048, torch.bfloat16, "contiguous")
    native_check = NS(
        layer_validity_version=lambda: 1,
        layer_validity=validity_oracle,
        layer_validity_vectorized_version=lambda: 1,
        layer_validity_vectorized=validity_oracle,
    )
    checker = FusedLayerValidity(
        native_check, reduction="vectorized" if validity_mode == "fused_vectorized" else "scalar"
    )
    observed, all_status_lists = [], []
    for invalid in (False, True, False):
        hidden = base_hidden.clone()
        slots = torch.tensor([5, 1, 2, 1, 0, 5], dtype=torch.int64)
        runtime.failure = failure if invalid else None
        if invalid and failure == "selection":
            slots[0] = -(1 << 63)
        if invalid and failure == "input":
            hidden[0, 0] = float("nan")
        route_flags = [torch.tensor(not (invalid and failure == "route"))]
        raw_statuses = None if validity_mode == "torch" else []
        flags = []
        before = len(runtime.outputs)
        with no_host_tensor_reads(monkeypatch):
            outputs = [
                route._candidate_projection(runtime, bank, bank.spec, prep, hidden, slots, flags.append, raw_statuses)
                for _ in range(2)
            ]
            combined = torch.zeros(1, 2048, dtype=torch.bfloat16)
            if invalid and failure == "combined":
                combined[0, -1] = float("inf")
            if raw_statuses is None:
                valid = torch.stack([*flags, *route_flags, torch.isfinite(combined).all()]).all()
            else:
                valid = checker(raw_statuses, [*outputs, combined], route_flags)
        assert outputs[0] is runtime.outputs[before] and outputs[1] is runtime.outputs[before + 1]
        if raw_statuses is not None:
            assert flags == [] and len(raw_statuses) == 6
            for index in range(2):
                status_offset = before + index
                assert raw_statuses[3 * index] is bank.selection_statuses[status_offset]
                input_statuses = bank.input_statuses if select_sign == "fused" else native.statuses
                assert raw_statuses[3 * index + 1] is input_statuses[status_offset]
                assert raw_statuses[3 * index + 2] is runtime.statuses[status_offset]
            all_status_lists.append(raw_statuses)
        else:
            assert len(flags) == 6
        observed.append(bool(valid))
    assert observed == [True, False, True]
    assert len({id(statuses) for statuses in all_status_lists}) == len(all_status_lists)


def test_default_runtime_does_not_enter_candidate_projection():
    assert not route._uses_projection_candidate(NS())
    assert not route._uses_projection_candidate(NS(v4_select_sign="separate", v4_activation_tail="torch"))
