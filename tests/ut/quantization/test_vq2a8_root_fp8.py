# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_ascend.quantization.vq2a8_root_fp8 import (
    ROOT_FP8_POLICY,
    TOKEN_SCALE_MIN,
    RootFP8State,
    inverse_rope_fp32,
    quantize_root_activation,
    quantize_root_weight,
    root_fp8_matmul_npu,
    root_fp8_matmul_reference,
    root_linear_kind,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.layers.0.self_attn.wq_a", "tensor"),
        ("layers.42.attn.wo_a.weight", "block128"),
        ("model.layers.2.self_attn.indexer.wq_b", "tensor"),
        ("layers.2.attn.compressor.wkv.weight", None),
        ("layers.2.attn.indexer.compressor.wgate.weight", None),
        ("layers.2.attn.indexer.weights_proj.weight", None),
        ("layers.2.ffn.gate.weight", None),
        ("lm_head", None),
        ("mtp.layers.0.self_attn.wo_a", None),
        ("model.layers.0.self_attn.dsa_attn.wq_a", None),
    ],
)
def test_explicit_root_allowlist(name, expected):
    assert root_linear_kind(name) == expected


@pytest.mark.parametrize("kind", ["tensor", "block128"])
@pytest.mark.parametrize("case", ["deterministic", "zero", "small"])
def test_quantization_preserves_source_and_expected_scale_recipe(kind, case):
    weight = ((torch.arange(256 * 512).reshape(256, 512) % 97 - 48).float() / 32).to(torch.bfloat16)
    if case == "zero":
        weight.zero_()
    elif case == "small":
        weight *= 1e-6
    original = weight.clone()
    qw, sw = quantize_root_weight(weight, kind)
    assert torch.equal(original, weight) and qw.dtype == torch.float8_e4m3fn and sw.dtype == torch.float32
    assert bool(torch.isfinite(qw.float()).all()) and bool((sw > 0).all())
    if kind == "block128":
        independent = torch.empty_like(sw)
        for row in range(2):
            for col in range(4):
                independent[row, col] = weight[
                    row * 128 : (row + 1) * 128, col * 128 : (col + 1) * 128
                ].float().abs().max().clamp_min(1e-4) * (1.0 / 448)
        assert torch.equal(sw, independent)
    elif case != "zero":
        assert torch.equal(sw, weight.float().abs().max().reshape(1) / torch.tensor(448.0))
    qx, sx = quantize_root_activation(weight[:3], kind)
    assert qx.dtype == torch.float8_e4m3fn
    if kind == "tensor":
        expected = (weight[:3].float().abs().amax(-1, keepdim=True) / torch.tensor(448.0)).clamp_min(TOKEN_SCALE_MIN)
        assert torch.equal(sx, expected)
    else:
        assert torch.equal(torch.log2(sx), torch.log2(sx).round())
    if case == "zero":
        assert torch.count_nonzero(qx.float()) == 0 and torch.count_nonzero(qw.float()) == 0


@pytest.mark.parametrize("bad", ["dtype", "shape", "nan", "inf", "empty", "unaligned", "noncontiguous"])
def test_weight_rejects_unsupported_inputs(bad):
    weight = torch.ones(128, 128, dtype=torch.bfloat16)
    if bad == "dtype":
        weight = weight.float()
    elif bad == "shape":
        weight = weight.unsqueeze(0)
    elif bad == "nan":
        weight[0, 0] = float("nan")
    elif bad == "inf":
        weight[0, 0] = float("inf")
    elif bad == "empty":
        weight = weight[:0]
    elif bad == "unaligned":
        weight = weight[:127]
    elif bad == "noncontiguous":
        weight = weight.T
    with pytest.raises(ValueError):
        quantize_root_weight(weight, "block128")


def test_inverse_rope_fp32_retains_values_lost_by_bf16_intermediate():
    x = torch.tensor([[[1.0, 2.0, 0.125, 0.25]]], dtype=torch.bfloat16)
    original = x.clone()
    cos = torch.tensor([[[[0.97, 0.97]]]])
    sin = torch.tensor([[[[0.1234, 0.1234]]]])
    result = inverse_rope_fp32(x, cos, sin, 2)
    expected = torch.tensor([[[1.0, 2.0, 0.125 * 0.97 + 0.25 * 0.1234, 0.25 * 0.97 - 0.125 * 0.1234]]])
    torch.testing.assert_close(result, expected, rtol=1e-7, atol=1e-8)
    assert result.dtype == torch.float32 and torch.equal(x, original)
    assert not torch.equal(result, result.bfloat16().float())
    with pytest.raises(ValueError, match="metadata"):
        inverse_rope_fp32(x, cos.flatten(), sin.flatten(), 2)


def test_block_scale_pins_cuda_fp32_reciprocal_not_cpu_scalar_division():
    weight = torch.full((128, 128), 0.072265625, dtype=torch.bfloat16)
    _, scale = quantize_root_weight(weight, "block128")
    assert scale.view(torch.int32).item() == 958997651


def test_grouped_projection_keeps_independent_groups_and_one_output_cast():
    groups, rank, k = 2, 128, 256
    weight = torch.ones(groups * rank, k, dtype=torch.bfloat16)
    weight[rank:] *= 3
    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))
    state = RootFP8State("block128")
    with pytest.raises(ValueError, match="loaded"):
        state.apply_grouped(layer, torch.ones(3, groups, k), groups, rank)
    state.process(layer)
    with pytest.raises(ValueError, match="twice"):
        state.process(layer)
    x = torch.ones(3, groups, k, dtype=torch.bfloat16)
    x[:, 1] *= 2
    result = state.apply_grouped(layer, x, groups, rank, matmul=root_fp8_matmul_reference)
    assert state.calls == 1 and result.shape == (3, groups * rank)
    assert torch.equal(result[:, :rank], torch.full((3, rank), 256.0, dtype=torch.bfloat16))
    assert torch.equal(result[:, rank:], torch.full((3, rank), 1536.0, dtype=torch.bfloat16))
    assert list(layer.named_parameters())[0][0] == "weight"
    assert dict(layer.named_buffers())["vq2a8_root_scale"].dtype == torch.float32


def test_tensor_projection_restores_leading_dimensions_and_has_no_implicit_fallback():
    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(torch.ones(256, 128, dtype=torch.bfloat16)))
    state = RootFP8State("tensor")
    with pytest.raises(ValueError, match="not been processed"):
        state.apply(layer, torch.ones(1, 128))
    state.process(layer)
    x = torch.ones(2, 3, 128, dtype=torch.bfloat16)
    result = state.apply(layer, x, matmul=root_fp8_matmul_reference)
    assert result.shape == (2, 3, 256) and bool((result == 128).all())
    with pytest.raises(ValueError, match="same NPU"):
        state.apply(layer, x)


@pytest.mark.parametrize("kind", ["tensor", "block128"])
def test_native_call_scale_layout_is_not_mx_or_requantized(monkeypatch, kind):
    device = NS(type="npu")

    def fake(t):
        return NS(
            device=device,
            dtype=t.dtype,
            shape=t.shape,
            ndim=t.ndim,
            tensor=t,
            T=t.T if t.ndim == 2 else t,
            is_contiguous=t.is_contiguous,
            numel=t.numel,
            reshape=t.reshape,
            flatten=t.flatten,
        )

    qw, sw = quantize_root_weight(torch.ones(256, 512, dtype=torch.bfloat16), kind)
    qx, sx = quantize_root_activation(torch.ones(3, 512), kind)
    calls = []
    monkeypatch.setitem(
        sys.modules, "torch_npu", NS(npu_quant_matmul=lambda *a, **kw: calls.append((a, kw)) or "result")
    )
    assert root_fp8_matmul_npu(fake(qx), fake(sx), fake(qw), fake(sw), kind) == "result"
    args, kwargs = calls[0]
    assert args[0].dtype == args[1].dtype == torch.float8_e4m3fn
    assert args[2].dtype == kwargs["pertoken_scale"].dtype == torch.float32
    assert kwargs["output_dtype"] == torch.bfloat16 and "bias" not in kwargs
    if kind == "block128":
        assert kwargs["group_sizes"] == [1, 128, 128]
        assert torch.equal(args[2], sw.T) and torch.equal(kwargs["pertoken_scale"].tensor, sx)
        assert args[2].stride() == sw.T.stride() and not args[2].is_contiguous()
        assert args[1].stride() == qw.T.stride()
    else:
        assert "group_sizes" not in kwargs and args[2].numel() == 1


def test_actual_offline_method_honors_post_load_lifecycle_and_bias_guard():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/worker/vq2a8_offline_model.py"
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "OfflineRootFP8Method"
    )
    scope = {"LinearMethodBase": object, "ROOT_FP8_POLICY": ROOT_FP8_POLICY, "RootFP8State": RootFP8State}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    method = scope["OfflineRootFP8Method"]("tensor")
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.self_attn.wq_a"
    layer.register_parameter("weight", torch.nn.Parameter(torch.ones(128, 128, dtype=torch.bfloat16)))
    method.process_weights_after_loading(layer)
    assert method.state.ready and layer.weight.dtype == torch.float8_e4m3fn
    with pytest.raises(RuntimeError, match="canonical"):
        method.create_weights(layer)
    with pytest.raises(ValueError, match="bias-free"):
        method.apply(layer, torch.ones(1, 128), bias=torch.ones(128))


@pytest.mark.parametrize("fail_step,expected", [("smoke", 1), ("roots", 2), (None, 3)])
def test_phase3_driver_fail_fast_and_no_other_stages(tmp_path, monkeypatch, fail_step, expected):
    from tools import validate_vq2a8_tp1_phase3 as driver

    model = tmp_path / "model"
    (model / "experts_vq_ascend_v2").mkdir(parents=True)
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["phase3", "--model", str(model), "--output-dir", str(output)])
    monkeypatch.setattr(driver, "LiveChildLog", lambda *args: nullcontext())
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        stage = command[command.index("--stage") + 1]
        assert kwargs["timeout"] == (None if stage == "model" else 1800)
        if stage == "model":
            assert "online_fp8_sm90" in command and "--baseline-report" not in command
            (output / "model").mkdir()
            (output / "model/summary.txt").write_text("ROOT_FP8_EXECUTION_VERIFIED=True\n")
            data = {
                "status": "passed",
                "results": [
                    {
                        "passed": True,
                        "records": [
                            {
                                "type": "VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS",
                                "data": {
                                    "root_linear_mode": "online_fp8_sm90",
                                    "root_fp8_execution_verified": True,
                                    "native_fp8_root_matmul": True,
                                    "offline_execution_verified": True,
                                    "repeat_exact": True,
                                },
                            }
                        ],
                    }
                ],
            }
            (output / "model/summary.json").write_text(json.dumps(data))
        else:
            (output / f"{stage}.json").write_text(
                json.dumps({"status": "passed", "stage": stage, "device": "npu:0", "results": [{"passed": True}]})
            )
        return NS(returncode=1 if stage == fail_step else 0)

    monkeypatch.setattr(driver.subprocess, "run", run)
    assert driver.main() == (1 if fail_step else 0)
    assert len(commands) == expected
    report = json.loads((output / "phase3.json").read_text())
    assert report["phase2"] == "skipped_by_request" and report["deferred_phases"] == [4, 5]
    assert not report["quality_verified"] and not report["native_fp8_expert_dot"]


@pytest.mark.parametrize("bad", ["missing", "empty", "cpu", "failed", "bad_json"])
def test_phase3_refuses_missing_or_wrong_operator_evidence(tmp_path, bad):
    from tools.validate_vq2a8_tp1_phase3 import step_evidence_passed

    value = {"status": "passed", "stage": "roots", "device": "npu:0", "results": [{"passed": True}]}
    if bad == "empty":
        value["results"] = []
    elif bad == "cpu":
        value["device"] = "cpu"
    elif bad == "failed":
        value["results"][0]["passed"] = False
    if bad != "missing":
        (tmp_path / "roots.json").write_text("{" if bad == "bad_json" else json.dumps(value))
    assert not step_evidence_passed(tmp_path, "roots")


def test_root_operator_progress_updates_heartbeat_stage(tmp_path):
    from tools.vq2a8_live_log import LiveChildLog

    relay = LiveChildLog(tmp_path / "root.log", "phase3-roots")
    relay._record_stage("ROOT_FP8 stage=weight_pre")
    relay._record_stage("pare name=wo_a\n")
    assert relay._last_stage == "ROOT_FP8 stage=weight_prepare name=wo_a"


def test_inverse_rope_fp8_midpoint_regression_and_bounded_diagnostics():
    from tools.validate_vq2a8_root_fp8 import activation_mismatch_details, rounding_inputs

    heads, cos, sin = rounding_inputs()
    expected = inverse_rope_fp32(heads, cos, sin, 448).reshape(10, 8, 4096)
    r = heads.float()[..., 448:].reshape(10, 64, 32, 2)
    c, s = (t.reshape(10, 1, 32, 2) for t in (cos, sin))
    even = r[..., 0] * c[..., 0] + r[..., 1] * s[..., 0]
    odd = r[..., 1] * c[..., 1] - r[..., 0] * s[..., 1]
    unfused = torch.cat((heads.float()[..., :448], torch.stack((even, odd), -1).flatten(-2)), -1)
    unfused = unfused.reshape_as(expected)
    q, scale = quantize_root_activation(expected[:, 6], "block128")
    dq, dscale = quantize_root_activation(unfused[:, 6], "block128")
    diagnostic = activation_mismatch_details(expected[:, 6], unfused[:, 6], q, dq, scale, dscale)
    assert diagnostic["byte_mismatch_count"] == 1
    sample = diagnostic["mismatch_samples"][0]
    assert sample["index"] == [6, 970]
    assert sample["expected_byte"] == 116 and sample["actual_byte"] == 117
    assert sample["expected_input_bits"] == 885522432 and sample["actual_input_bits"] == 885522433
    assert sample["expected_scaled"] == 200.0 and sample["actual_scaled"] > 200.0
    assert torch.equal(scale, dscale)
    assert sample["expected_fp8_value"] == 192 and sample["actual_fp8_value"] == 208
    many = activation_mismatch_details(
        torch.ones(3, 128),
        torch.zeros(3, 128),
        torch.ones(3, 128).to(torch.float8_e4m3fn),
        torch.zeros(3, 128).to(torch.float8_e4m3fn),
        torch.ones(3, 1),
        torch.ones(3, 1),
    )
    assert many["byte_mismatch_count"] == 384 and len(many["mismatch_samples"]) == 8
    json.dumps(diagnostic, allow_nan=False)


def test_rounding_probe_requires_exact_inputs_scales_and_fp32_before_weight_gates():
    from tools.validate_vq2a8_root_fp8 import rounding_probe

    labels = []

    def compare(label, expected, actual, *, exact, details=None):
        assert exact and torch.equal(expected.float(), actual.float())
        labels.append(label)

    rounding_probe(compare, torch.device("cpu"))
    assert len(labels) == 38
    assert "rounding:small:g6:bytes" in labels and "rounding:zero:inverse_rope_fp32" in labels


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Developer CUDA check, not NPU acceptance")
@pytest.mark.parametrize("size", [0, 1, 65, 1024, 1025])
def test_explicit_device_fma_matches_fused_reference_with_broadcast_and_tail(size):
    from vllm_ascend.quantization.vq2a8_root_fp8_triton import root_fma_fp32

    a = ((torch.arange(size * 6).reshape(size, 6) % 37 - 18).float() / 64)[:, ::2]
    b = torch.tensor([0.9876543, -0.2345678, 0.9234567])
    c = a * 0.7654321
    expected = torch.addcmul(c, a, b)
    actual = root_fma_fp32(a.cuda(), b.cuda(), c.cuda())
    assert torch.equal(expected, actual.cpu()) and actual.dtype == torch.float32
    with pytest.raises(ValueError, match="FP32"):
        root_fma_fp32(a.cuda().bfloat16(), b.cuda(), c.cuda())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Developer CUDA check, not NPU acceptance")
def test_explicit_device_inverse_rope_rounding_regression():
    from tools.validate_vq2a8_root_fp8 import rounding_probe

    def compare(label, expected, actual, *, exact, details=None):
        assert exact
        assert torch.equal(expected.float().cpu(), actual.float().cpu()), label

    rounding_probe(compare, torch.device("cuda"))
