#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-3 operator gate, NOT an independent full-model reference or benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
        print("DEVICE " + json.dumps(_initialize_device(device)), flush=True)
    matmul = root_fp8_matmul_npu if device.type == "npu" else root_fp8_matmul_reference

    def compare(label, expected, actual, *, exact=False):
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
                    compare(label + ":activation_bytes", qx.view(torch.uint8), dqx.view(torch.uint8), exact=True)
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
