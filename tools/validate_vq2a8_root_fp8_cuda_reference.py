#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Developer-only SM90 operator comparison, not phase-2 model certification.

Runs on the accessible NVIDIA host. The referenced Python functions must match
the pinned original source. Compiled upstream weight quantization can round
FP8 boundary values differently from eager PyTorch; report bytes separately
from projection accuracy instead of claiming bitwise upstream equivalence.
"""

from __future__ import annotations

# Direct scripts must not put tools/bisect ahead of the stdlib bisect module.
# ruff: noqa: E402
import os as _bootstrap_os
import sys as _bootstrap_sys

if not __package__:
    _bootstrap_sys.path[0] = _bootstrap_os.path.dirname(
        _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))
    )

import argparse
import ast
import inspect
import json
import subprocess
from importlib import import_module
from pathlib import Path

REFERENCE_COMMIT = "2d75468d44857582f9d21c983d451d69bea50ad7"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--reference-repo", required=True, type=Path)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open
    from vllm import _custom_ops as ops
    from vllm.utils.deep_gemm import fp8_einsum, per_block_cast_to_fp8

    from vllm_ascend.quantization.vq2a8_root_fp8 import (
        inverse_rope_fp32,
        quantize_root_activation,
        quantize_root_weight,
        root_fp8_matmul_reference,
    )
    from vllm_ascend.quantization.vq2a8_validation import error_metrics

    rope = import_module("vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant")

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.reference_repo, text=True).strip()
    if head != REFERENCE_COMMIT or torch.cuda.get_device_capability()[0] != 9:
        raise ValueError("Requires the pinned original source and an SM90 NVIDIA device.")

    def verify_function(function, source):
        tree = ast.parse((args.reference_repo / source).read_text())
        original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function.__name__)
        actual = ast.parse(inspect.getsource(function)).body[0]
        if ast.dump(original) != ast.dump(actual):
            raise ValueError(f"Installed reference differs from pinned source: {source}:{function.__name__}.")

    verify_function(per_block_cast_to_fp8, "vllm/utils/deep_gemm.py")
    for name in (
        "fused_inv_rope_fp8_quant",
        "_fused_inv_rope_fp8_quant_kernel_impl",
        "_fused_inv_rope_fp8_quant_per_head",
    ):
        function = getattr(rope, name)
        verify_function(
            getattr(function, "fn", function), "vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py"
        )
    print(f"REFERENCE_COMMIT={head} DEVICE={torch.cuda.get_device_name()} TORCH={torch.__version__}", flush=True)
    # Strictly tensor numerical comparisons, not a full-model reference run.
    torch.backends.cuda.matmul.allow_tf32 = False
    probes = [
        (f"layers.0.attn.{name}.weight", "block128" if name == "wo_a" else "tensor")
        for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
    ]
    probes += [("layers.2.attn.indexer.wq_b.weight", "tensor"), ("layers.42.attn.wo_a.weight", "block128")]
    files = {}
    for path in args.model.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as handle:
            files.update({name: path for name in handle.keys()})  # noqa: SIM118
    groups = json.loads((args.model / "config.json").read_text())["o_groups"]
    all_weight_bytes_equal = True
    for name, kind in probes:
        with safe_open(files[name], framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(name).cuda().contiguous()
        qw, sw = quantize_root_weight(weight, kind)
        cpu_qw, cpu_sw = quantize_root_weight(weight.cpu(), kind)
        if not torch.equal(cpu_qw.view(torch.uint8), qw.view(torch.uint8).cpu()) or not torch.equal(cpu_sw, sw.cpu()):
            raise AssertionError(f"CPU/CUDA eager weight preparation differs: {name}.")
        if kind == "block128":
            rq, rs = per_block_cast_to_fp8(weight, use_ue8m0=False)
        else:
            rs = weight.abs().max().float().reshape(1) / torch.tensor(448.0, device="cuda", dtype=torch.float32)
            rq, rs = ops.scaled_fp8_quant(weight, scale=rs)
        if not torch.equal(sw, rs):
            raise AssertionError(f"Weight scales differ: {name}.")
        mismatch = int((qw.view(torch.uint8) != rq.view(torch.uint8)).sum())
        all_weight_bytes_equal &= mismatch == 0
        print(
            "CUDA_ROOT_WEIGHT "
            + json.dumps(
                {
                    "name": name,
                    "scale_exact": True,
                    "byte_mismatch_count": mismatch,
                    "numel": qw.numel(),
                    "max_quantized_abs_error": float((qw.float() - rq.float()).abs().max()),
                }
            ),
            flush=True,
        )
        count = groups if kind == "block128" else 1
        rank, k = weight.shape[0] // count, weight.shape[1]
        for tokens in (1, 3, 10, 32):
            for case in ("deterministic", "zero", "impulse", "small"):
                x = (
                    ((torch.arange(tokens * count * k, device="cuda").reshape(tokens, count, k) * 7) % 61 - 30).float()
                    / 64
                ).bfloat16()
                if case == "zero":
                    x.zero_()
                elif case == "small":
                    x *= 1e-6
                elif case == "impulse":
                    x.zero_()
                    x[:, :, -1] = 1
                if kind == "block128":
                    angles = torch.arange(tokens * 32).reshape(tokens, 32).float() / 64
                    c, s = angles.cos().cuda(), angles.sin().cuda()
                    cache = torch.cat((c, s), -1).contiguous()
                    heads = x.reshape(tokens, -1, 512)
                    rx, rxs = rope.fused_inv_rope_fp8_quant(
                        heads, torch.arange(tokens, device="cuda"), cache, count, k // 512
                    )
                    native = torch.empty((tokens, count, rank), dtype=torch.bfloat16, device="cuda")
                    fp8_einsum(
                        "bhr,hdr->bhd",
                        (rx, rxs),
                        (rq.view(count, rank, k), rs.view(count, rank // 128, k // 128)),
                        native,
                        recipe=(1, 128, 128),
                    )
                    rotated = inverse_rope_fp32(
                        heads,
                        c.repeat_interleave(2, -1).reshape(tokens, 1, 1, 64),
                        s.repeat_interleave(2, -1).reshape(tokens, 1, 1, 64),
                        448,
                    )
                    cpu_x = inverse_rope_fp32(
                        heads.cpu(),
                        c.cpu().repeat_interleave(2, -1).reshape(tokens, 1, 1, 64),
                        s.cpu().repeat_interleave(2, -1).reshape(tokens, 1, 1, 64),
                        448,
                    ).reshape_as(x)
                    x = rotated.reshape_as(x)
                else:
                    cpu_x = x.cpu()
                for group in range(count):
                    qx, sx = quantize_root_activation(x[:, group], kind)
                    cpu_qx, cpu_sx = quantize_root_activation(cpu_x[:, group], kind)
                    if not torch.equal(cpu_qx.view(torch.uint8), qx.view(torch.uint8).cpu()) or not torch.equal(
                        cpu_sx, sx.cpu()
                    ):
                        raise AssertionError(
                            f"CPU/CUDA eager activation preparation differs: {name}:{tokens}:{case}:{group}."
                        )
                    if kind == "tensor":
                        rqx, rsx = ops.scaled_fp8_quant(x[:, group].contiguous(), use_per_token_if_dynamic=True)
                        native = ops.cutlass_scaled_mm(rqx, rq.T, rsx, rs, out_dtype=torch.bfloat16).unsqueeze(1)
                    else:
                        rqx, rsx = rx[:, group].contiguous(), rxs[:, group].contiguous()
                    if not torch.equal(qx.view(torch.uint8), rqx.view(torch.uint8)) or not torch.equal(sx, rsx):
                        raise AssertionError(f"Activation bytes/scales differ: {name}:{tokens}:{case}:{group}.")
                    sl = slice(group * rank, (group + 1) * rank)
                    scale = sw if kind == "tensor" else sw[group * rank // 128 : (group + 1) * rank // 128]
                    expected = native[:, group]
                    actual = root_fp8_matmul_reference(qx, sx, qw[sl], scale, kind)
                    metrics = error_metrics(expected.float(), actual.float())
                    if (
                        not torch.allclose(expected, actual, rtol=0.01, atol=0.03125)
                        or metrics["relative_l2_error"] > 0.01
                    ):
                        raise AssertionError(f"Projection differs: {name}:{tokens}:{case}:{group}: {metrics}.")
                print(f"CUDA_ROOT_CASE=PASS name={name} tokens={tokens} case={case} activation_exact=True", flush=True)
    print(
        f"CUDA_ROOT_OPERATOR_REFERENCE=PASS weights_bitwise_equal={all_weight_bytes_equal} "
        "NPU_VERIFIED=False INDEPENDENT_MODEL_REFERENCE=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
