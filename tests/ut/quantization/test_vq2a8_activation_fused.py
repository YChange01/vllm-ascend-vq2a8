# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU orchestration contracts; native arithmetic needs the bounded NPU probe."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import validate_vq2a8_activation_fused as validate
from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_activation_fused import FusedV4V2Preparation
from vllm_ascend.quantization.vq2a8_reference import VQ2_FP8_MIN_SCALE


class TorchPreparationOps:
    """Independent Torch oracle, not a mock claim about native execution."""

    def __init__(self):
        self.calls = []

    def activation_preparation_version(self):
        return 1

    def activation_sign(self, x, scale, bias, signs):
        self.calls.append(("sign", tuple(x.shape)))
        valid = torch.isfinite(x).all(-1) & torch.isfinite(scale).all(-1) & torch.isfinite(bias).all(-1)
        valid &= ((signs == -1) | (signs == 1)).all(-1)
        return x * signs.float(), valid.int()

    def activation_quantize(self, rotated, weight_scale, bias):
        self.calls.append(("quantize", tuple(rotated.shape)))
        transformed = rotated * weight_scale
        scale = (transformed.abs().amax(-1) / 448).clamp(min=VQ2_FP8_MIN_SCALE)
        quantized = (transformed / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        valid = torch.isfinite(transformed).all(-1) & torch.isfinite(bias)
        return quantized, scale, valid.int()


def request(width=2048, rows=1, *, true_width=None, dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(width + rows)
    true_width = true_width or width
    hidden = torch.randn(rows, true_width, generator=generator).to(dtype)
    payload = {
        "weight_scale": torch.randn(width, generator=generator),
        "weight_bias": torch.randn(width, generator=generator),
        "rht_sign": torch.where(torch.arange(width) % 3 == 0, -1, 1).to(torch.int8),
    }
    spec = SimpleNamespace(columns=width, rht_true_columns=true_width, rht_block_size=128)
    return hidden, payload, spec


@pytest.mark.parametrize("width", [2048, 4096])
@pytest.mark.parametrize("rows,jobs", [(1, 1), (2, 6), (17, 1), (32, 1)])
@pytest.mark.parametrize("case", ["random", "zero", "small", "impulse"])
def test_fused_wrapper_retains_reference_rounding_with_oracle_ops(width, rows, jobs, case):
    requests = [request(width, rows) for _ in range(jobs)]
    for hidden, _, _ in requests:
        if case == "zero":
            hidden.zero_()
        elif case == "small":
            hidden.mul_(1e-15)
        elif case == "impulse":
            hidden.zero_()
            hidden[:, -1] = -1
    native = TorchPreparationOps()
    fused = FusedV4V2Preparation(native_ops=native)
    expected = RowwiseVQ2A8Preparation(compact=True).many(requests)
    actual = fused.many(requests)
    for got, want in zip(actual, expected):
        for x, y in zip(got, want):
            assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))
    assert native.calls == [("sign", (rows * jobs, width)), ("quantize", (rows * jobs, width))]


def test_fused_padding_and_mixed_dtype_keep_conversion_order():
    requests = [request(true_width=2000), request(true_width=2000, dtype=torch.float64)]
    actual = FusedV4V2Preparation(native_ops=TorchPreparationOps()).many(requests)
    expected = RowwiseVQ2A8Preparation(compact=True).many(requests)
    for got, want in zip(actual, expected):
        for x, y in zip(got, want):
            assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))


@pytest.mark.parametrize(
    "target,bad",
    [
        ("x", float("nan")),
        ("weight_scale", float("inf")),
        ("weight_bias", -float("inf")),
        ("rht_sign", 0),
        ("rht_sign", -128),
    ],
)
def test_fused_rechecks_mutated_metadata_and_dynamic_values(target, bad):
    hidden, payload, spec = request()
    flags = []
    fused = FusedV4V2Preparation(native_ops=TorchPreparationOps(), validity=flags.append)
    fused.many([(hidden, payload, spec)])
    assert bool(flags[-1])
    tensor = hidden if target == "x" else payload[target]
    tensor.view(-1)[-1] = bad
    fused.many([(hidden, payload, spec)])
    assert not bool(flags[-1])


def test_fused_checks_output_overflow_before_sampling():
    hidden, payload, spec = request(dtype=torch.float32)
    hidden.fill_(1e30)
    payload["weight_bias"].fill_(1e30)
    with pytest.raises(ValueError, match="Invalid activation"):
        FusedV4V2Preparation(native_ops=TorchPreparationOps()).many([(hidden, payload, spec)])


def test_fused_deferred_validity_never_reads_a_device_value(monkeypatch):
    flags = []
    fused = FusedV4V2Preparation(native_ops=TorchPreparationOps())
    fused.prepare_for_graph(torch.device("cpu"), 128)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected scalar read in fused preparation")

    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    fused.many([request()], validity=flags.append)
    assert len(flags) == 1 and flags[0].shape == ()


def test_fused_graph_constants_are_frozen():
    fused = FusedV4V2Preparation(native_ops=TorchPreparationOps())
    fused.prepare_for_graph(torch.device("cpu"), 128)
    hidden, payload, spec = request()
    spec.rht_block_size = 64
    with pytest.raises(RuntimeError, match="geometry changed"):
        fused.many([(hidden, payload, spec)])


@pytest.mark.parametrize("requests", [[], [request()] * 7, [request(width=512)], [request(rows=33)]])
def test_fused_rejects_unsupported_geometry_without_launch(requests):
    native = TorchPreparationOps()
    with pytest.raises(ValueError):
        FusedV4V2Preparation(native_ops=native).many(requests)
    assert not native.calls


def test_fused_requires_explicit_native_feature():
    with pytest.raises(RuntimeError, match="rebuilt"):
        FusedV4V2Preparation(native_ops=SimpleNamespace())
    ops = TorchPreparationOps()
    ops.activation_preparation_version = lambda: 2
    with pytest.raises(RuntimeError, match="ABI 2"):
        FusedV4V2Preparation(native_ops=ops)
    ops.activation_preparation_version = lambda: True
    with pytest.raises(RuntimeError, match="ABI True"):
        FusedV4V2Preparation(native_ops=ops)


@pytest.mark.parametrize("compact", [True, False])
@pytest.mark.parametrize("entry", ["rows", "call"])
def test_fused_all_entry_points_launch_fusion(compact, entry):
    hidden, payload, spec = request()
    ops = TorchPreparationOps()
    fused = FusedV4V2Preparation(native_ops=ops, compact=compact)
    if entry == "rows":
        actual = fused.rows(hidden, payload, spec)
    else:
        actual = fused(hidden, payload["weight_scale"], payload["weight_bias"], payload["rht_sign"], 128)
    expected = RowwiseVQ2A8Preparation(compact=True).many([(hidden, payload, spec)])[0]
    assert len(ops.calls) == 2
    for got, want in zip(actual, expected):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


def test_native_fusion_retains_queue_owners_and_device_checks():
    root = Path(__file__).resolve().parents[3] / "csrc/vq2a8_ascendc_v4_v2"
    binding = (root / "activation_binding.cpp").read_text()
    kernel = (root / "activation_kernel.cpp").read_text()
    assert "RunOpApiV2" not in binding
    assert binding.count("const auto launchStream = stream.stream();") == 2
    assert "[launchStream, blocks, x, weightScale, weightBias, signs, output, valid, rows, width]" in binding
    assert "[launchStream, blocks, rotated, weightScale, rowBias, quantized, scale, valid, rows, width]" in binding
    assert "recordStream(tensor.storage().data_ptr(), stream)" in binding
    assert "CMPMODE::LE" in kernel and "CMPMODE::EQ" in kernel
    assert "RoundMode::CAST_RINT" in kernel
    assert "Div(x, x, divisor, width_)" in kernel and "Divs(reduced, reduced, kFp8Maximum, 1)" in kernel
    assert "1.0e-12f" in kernel


def test_fused_probe_plan_is_bounded_and_cannot_claim_execution(capsys):
    assert validate.main(["--physical-npu", "1", "--timeout-s", "75", "--queue-lifetime", "--plan-only"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert not plan["device_execution_verified"] and not plan["graph_verified"]
    assert not plan["model_integration_verified"] and not plan["performance_verified"]
    assert "--queue-lifetime" in plan["command"]
    assert plan["command"][plan["command"].index("--timeout-s") + 1] == "75"


@pytest.mark.parametrize("arguments", [["--physical-npu", "-1"], ["--timeout-s", "0"], ["--child", "--plan-only"]])
def test_fused_probe_rejects_unsafe_or_conflicting_arguments(arguments):
    with pytest.raises(SystemExit):
        validate.parse_args(arguments)


def test_fused_probe_bit_oracle_preserves_signed_zero_and_dtype():
    validate.assert_bits(torch.tensor([1.0]), torch.tensor([1.0]), "same")
    with pytest.raises(AssertionError, match="unequal bytes"):
        validate.assert_bits(torch.tensor([0.0]), torch.tensor([-0.0]), "signed zero")
    with pytest.raises(AssertionError, match="shape/dtype"):
        validate.assert_bits(torch.tensor([1.0]), torch.tensor([1.0]).half(), "dtype")
