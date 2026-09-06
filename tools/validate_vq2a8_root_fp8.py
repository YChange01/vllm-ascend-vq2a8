#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-3 operator gate, NOT an independent full-model reference or benchmark."""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path


def activation_mismatch_details(x, dx, qx, dqx, sx, dsx):
    """Bounded failure evidence; byte deltas are not FP8 numerical errors."""
    import torch

    qb, dqb = qx.view(torch.uint8).cpu(), dqx.view(torch.uint8).cpu()
    indices = (qb != dqb).nonzero()
    x, dx, sx, dsx = (t.float().cpu() for t in (x, dx, sx, dsx))
    qvalues, dqvalues = qx.float().cpu(), dqx.float().cpu()

    def number(t):
        value = float(t)
        return value if math.isfinite(value) else None

    samples = []
    for row, channel in indices[:8].tolist():
        block = channel // 128 if sx.shape[1] != 1 else 0
        samples.append(
            {
                "index": [row, channel],
                "expected_byte": int(qb[row, channel]),
                "actual_byte": int(dqb[row, channel]),
                "expected_fp8_value": number(qvalues[row, channel]),
                "actual_fp8_value": number(dqvalues[row, channel]),
                "expected_input": number(x[row, channel]),
                "actual_input": number(dx[row, channel]),
                "expected_input_bits": int(x[row, channel].view(torch.int32)),
                "actual_input_bits": int(dx[row, channel].view(torch.int32)),
                "expected_scale": number(sx[row, block]),
                "actual_scale": number(dsx[row, block]),
                "expected_scaled": number(x[row, channel] / sx[row, block]),
                "actual_scaled": number(dx[row, channel] / dsx[row, block]),
            }
        )
    return {"byte_mismatch_count": len(indices), "mismatch_samples": samples}


def rounding_inputs(case="small"):
    """Weight-free reproduction of M=10, G=8, K=4096, group-6 boundary."""
    import torch

    x = (((torch.arange(10 * 8 * 4096).reshape(10, 64, 512) * 7) % 61 - 30).float() / 64).bfloat16()
    if case == "small":
        x *= 1e-6
    elif case == "zero":
        x.zero_()
    else:
        raise ValueError("Unknown rounding regression case.")
    angle = torch.arange(10 * 32).reshape(10, 1, 1, 32).float() / 64
    return x, angle.cos().repeat_interleave(2, -1), angle.sin().repeat_interleave(2, -1)


def fma_conformance_words():
    """FP32 operand/result encodings, independent of host floating arithmetic."""
    return [
        (0x3F800000, 0x3F800000, 0x33800000, 0x3F800000),  # 1 + half ULP -> even
        (0x3F800001, 0x3F800000, 0x33800000, 0x3F800002),  # odd -> even
        (0x00000001, 0x3F000000, 0, 0),  # half minimum subnormal
        (0x00000003, 0x3F000000, 0, 2),
        (0x80000001, 0x3F000000, 0, 0x80000000),
        (0x00800000, 0x3F000000, 0, 0x00400000),
        (0x7F7FFFFF, 0x40000000, 0, 0x7F800000),
        (0x7F7FFFFF, 0x40000000, 0xFF7FFFFF, 0x7F7FFFFF),  # no intermediate overflow
        (0x3F800000, 0x3F800000, 0xBF800000, 0),
        (0x80000000, 0x3F800000, 0x80000000, 0x80000000),
        (0x80000000, 0x3F800000, 0, 0),
    ]


def fma_conformance_probe(compare, device):
    """Tiny first gate for int64 lowering, ties, cancellation and signed zeros."""
    import torch

    words = torch.tensor(fma_conformance_words(), dtype=torch.int64).to(torch.int32)
    a, b, c, expected = (words[:, i].contiguous().view(torch.float32) for i in range(4))
    if device.type == "cpu":
        actual = torch.addcmul(c, a, b)
    else:
        from vllm_ascend.quantization.vq2a8_root_fp8_triton import root_fma_fp32

        actual = root_fma_fp32(a.to(device), b.to(device), c.to(device)).cpu()
    print(
        "ROOT_FP8_FMA "
        + json.dumps(
            {
                "backend": "cpu_reference" if device.type == "cpu" else "integer_single_rounding_rne",
                "actual_words": [f"{bits & 0xFFFFFFFF:08x}" for bits in actual.view(torch.int32).tolist()],
                "expected_words": [f"{bits & 0xFFFFFFFF:08x}" for bits in words[:, 3].tolist()],
            }
        ),
        flush=True,
    )
    # Byte comparisons preserve +/-0 and all 32 bits; converting int32
    # encodings to float32 in compare() would discard low significand bits.
    compare("rounding:fma_conformance:bytes", expected.view(torch.uint8), actual.view(torch.uint8), exact=True)


def rounding_probe(compare, device):
    """Run before weights; isolate FMA from FP8 cast/scale on identical inputs."""
    import torch

    from vllm_ascend.quantization.vq2a8_root_fp8 import inverse_rope_fp32, quantize_root_activation

    print("ROOT_FP8 stage=fma_conformance", flush=True)
    fma_conformance_probe(compare, device)
    for case in ("small", "zero"):
        print(f"ROOT_FP8 stage=rounding_regression case={case} tokens=10 groups=8", flush=True)
        heads, cos, sin = rounding_inputs(case)
        expected = inverse_rope_fp32(heads, cos, sin, 448).reshape(10, 8, 4096)
        actual = inverse_rope_fp32(heads.to(device), cos.to(device), sin.to(device), 448).reshape_as(expected)
        # Diagnostic only: does this installation's old eager addcmul have
        # the same rounding fingerprint as two separate multiplies + add?
        if case == "small":
            r = heads.to(device).float()[..., 448:].reshape(10, 64, 32, 2)
            c, s = (t.to(device).reshape(10, 1, 32, 2) for t in (cos, sin))
            legacy = torch.addcmul(r[..., 1] * s[..., 0], r[..., 0], c[..., 0])
            unfused = r[..., 0] * c[..., 0] + r[..., 1] * s[..., 0]
            print(
                "ROOT_FP8_ROUNDING "
                + json.dumps(
                    {
                        "group": 6,
                        "index": [6, 970],
                        "fma_backend": "cpu_reference" if device.type == "cpu" else "integer_single_rounding_rne",
                        "expected_fp32": float(expected[6, 6, 970]),
                        "explicit_fma_fp32": float(actual[6, 6, 970]),
                        "legacy_addcmul_fp32": float(legacy[6, 49, 5]),
                        "unfused_fp32": float(unfused[6, 49, 5]),
                    }
                ),
                flush=True,
            )
        # Test quantization without NPU RoPE first, so failure identifies
        # cast/scale handling separately from the FMA implementation.
        q, scale = quantize_root_activation(expected[:, 6], "block128")
        same = expected[:, 6].to(device)
        dq, dscale = quantize_root_activation(same, "block128")
        compare(
            f"rounding:{case}:same_input:bytes",
            q.view(torch.uint8),
            dq.view(torch.uint8),
            exact=True,
            details=partial(activation_mismatch_details, expected[:, 6], same, q, dq, scale, dscale),
        )
        compare(f"rounding:{case}:same_input:scale", scale, dscale, exact=True)
        compare(f"rounding:{case}:inverse_rope_fp32", expected, actual, exact=True)
        for group in range(8):
            q, scale = quantize_root_activation(expected[:, group], "block128")
            dq, dscale = quantize_root_activation(actual[:, group], "block128")
            compare(
                f"rounding:{case}:g{group}:bytes",
                q.view(torch.uint8),
                dq.view(torch.uint8),
                exact=True,
                details=partial(
                    activation_mismatch_details, expected[:, group], actual[:, group], q, dq, scale, dscale
                ),
            )
            compare(f"rounding:{case}:g{group}:scale", scale, dscale, exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["smoke", "roots"], default="smoke")
    parser.add_argument("--device", choices=["npu:0", "cpu"], default="npu:0")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new file.")
    import torch
    from safetensors import safe_open

    from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device, environment_report
    from vllm_ascend.quantization.vq2a8_root_fp8 import (
        RootFP8State,
        inverse_rope_fp32,
        quantize_root_activation,
        quantize_root_weight,
        root_fp8_matmul_npu,
        root_fp8_matmul_reference,
    )
    from vllm_ascend.quantization.vq2a8_validation import error_metrics

    device = torch.device(args.device)
    report = {
        "status": "running",
        "stage": args.stage,
        "device": str(device),
        "environment": environment_report(),
        "results": [],
        "independent_model_reference": False,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    print("ENVIRONMENT " + json.dumps(report["environment"]), flush=True)
    if device.type == "npu":
        report["device_info"] = _initialize_device(device)
        save()
        print("DEVICE " + json.dumps(report["device_info"]), flush=True)
    matmul = root_fp8_matmul_npu if device.type == "npu" else root_fp8_matmul_reference

    def compare(label, expected, actual, *, exact=False, details=None):
        expected, actual = expected.float().cpu(), actual.float().cpu()
        finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
        compatible = finite and expected.shape == actual.shape
        metrics = error_metrics(expected, actual) if compatible else {}
        passed = compatible
        if exact:
            passed = passed and torch.equal(expected, actual)
        else:
            # Separate from the expert tolerances. Neither BF16-vs-FP8 nor
            # end-to-end model equivalence is certified by these limits.
            passed = (
                passed
                and torch.allclose(expected, actual, rtol=0.01, atol=0.03125)
                and metrics["relative_l2_error"] <= 0.01
            )
        result = {
            "check": label,
            "passed": bool(passed),
            "exact_required": exact,
            "max_abs_error": float((expected - actual).abs().max()) if compatible else None,
            "finite": finite,
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
            **metrics,
        }
        if not passed and compatible:
            indices = (expected != actual).nonzero()
            result["mismatch_count"] = len(indices)
            result["first_mismatch_indices"] = indices[:8].tolist()
            if details is not None:
                result.update(details())
        report["results"].append(result)
        save()
        print("ROOT_FP8_RESULT " + json.dumps(result), flush=True)
        if not passed:
            raise AssertionError(f"Root FP8 gate failed: {label}.")

    def projection(name, weight, kind, groups=1):
        print(f"ROOT_FP8 stage=weight_prepare name={name} shape={list(weight.shape)} kind={kind}", flush=True)
        qw, sw = quantize_root_weight(weight, kind)
        dev_weight, dev_scale = quantize_root_weight(weight.to(device), kind)
        compare(name + ":weight_bytes", qw.view(torch.uint8), dev_weight.view(torch.uint8), exact=True)
        compare(name + ":weight_scale", sw, dev_scale, exact=True)
        layer = torch.nn.Module()
        layer.register_parameter("weight", torch.nn.Parameter(dev_weight, requires_grad=False))
        layer.register_buffer("vq2a8_root_scale", dev_scale)
        state = RootFP8State(kind)
        state.ready = True
        n, k = weight.shape
        rank = n // groups
        for tokens in (1, 3, 10, 32):
            for case in ("deterministic", "zero", "impulse", "small"):
                print(f"ROOT_FP8 stage=projection name={name} tokens={tokens} case={case}", flush=True)
                x = (((torch.arange(tokens * groups * k).reshape(tokens, groups, k) * 7) % 61 - 30).float() / 64).to(
                    torch.bfloat16
                )
                if case == "zero":
                    x.zero_()
                elif case == "small":
                    x *= 1e-6
                elif case == "impulse":
                    x.zero_()
                    x[:, :, -1] = 1
                if kind == "block128":
                    # Non-identity inverse RoPE, including the 448/64 boundary.
                    head_dim = 512 if k % 512 == 0 else k
                    rope_dim = 64
                    angle = torch.arange(tokens * (rope_dim // 2)).reshape(tokens, 1, 1, rope_dim // 2).float() / 64
                    cos, sin = angle.cos().repeat_interleave(2, -1), angle.sin().repeat_interleave(2, -1)
                    heads = x.reshape(tokens, -1, head_dim)
                    rotated = inverse_rope_fp32(heads, cos, sin, head_dim - rope_dim)
                    dx = inverse_rope_fp32(
                        heads.to(device), cos.to(device), sin.to(device), head_dim - rope_dim
                    ).reshape(tokens, groups, k)
                    x = rotated.reshape(tokens, groups, k)
                else:
                    dx = x.to(device)
                expected_parts = []
                for group in range(groups):
                    qx, sx = quantize_root_activation(x[:, group], kind)
                    dqx, dsx = quantize_root_activation(dx[:, group], kind)
                    label = f"{name}:{tokens}:{case}:g{group}"
                    compare(
                        label + ":activation_bytes",
                        qx.view(torch.uint8),
                        dqx.view(torch.uint8),
                        exact=True,
                        details=partial(activation_mismatch_details, x[:, group], dx[:, group], qx, dqx, sx, dsx),
                    )
                    compare(label + ":activation_scale", sx, dsx, exact=True)
                    scale = sw if kind == "tensor" else sw[group * rank // 128 : (group + 1) * rank // 128]
                    expected_parts.append(
                        root_fp8_matmul_reference(qx, sx, qw[group * rank : (group + 1) * rank], scale, kind)
                    )
                expected = torch.cat(expected_parts, -1)

                def execute(dx=dx):
                    if kind == "block128":
                        return state.apply_grouped(layer, dx, groups, rank, matmul=matmul)
                    return state.apply(layer, dx[:, 0], matmul=matmul)

                actual = execute()
                compare(f"{name}:{tokens}:{case}:output", expected, actual)
                for repeat in range(2):
                    compare(f"{name}:{tokens}:{case}:repeat{repeat + 1}", actual, execute(), exact=True)

    try:
        rounding_probe(compare, device)
        if args.stage == "smoke":
            for kind in ("tensor", "block128"):
                w = (((torch.arange(512 * 512).reshape(512, 512) * 3) % 37 - 18).float() / 64).to(torch.bfloat16)
                projection("synthetic:" + kind, w, kind, groups=2 if kind == "block128" else 1)
        else:
            probes = [
                (f"layers.0.attn.{name}.weight", "block128" if name == "wo_a" else "tensor")
                for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
            ]
            probes += [("layers.2.attn.indexer.wq_b.weight", "tensor"), ("layers.42.attn.wo_a.weight", "block128")]
            config = json.loads((args.model / "config.json").read_text())
            files = {}
            for path in sorted(args.model.glob("*.safetensors")):
                with safe_open(path, framework="pt", device="cpu") as handle:
                    for key in handle.keys():  # noqa: SIM118
                        if key in files:
                            raise ValueError(f"Duplicate root tensor {key}.")
                        files[key] = path
            for name, kind in probes:
                with safe_open(files[name], framework="pt", device="cpu") as handle:
                    weight = handle.get_tensor(name).contiguous()
                projection(name, weight, kind, groups=config["o_groups"] if kind == "block128" else 1)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
    print(f"ROOT_FP8_OPERATOR_GATE=PASS stage={args.stage} checks={len(report['results'])} DEVICE={device}", flush=True)


if __name__ == "__main__":
    main()
