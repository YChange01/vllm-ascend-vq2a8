# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP1 root-linear FP8 contract, separate from the accepted VQ2 expert kernel.

Targets the pinned NVIDIA SM90/Cutlass policy, not MXFP8: tensorwise weights
and tokenwise activations; wo_a uses 128x128 weights, 1x128 pow2 activations.
CPU helpers are operator references only. Production NPU calls never fall back.
"""

from __future__ import annotations

import re

import torch

FP8_MAX = 448.0
FP8_BLOCK = 128
TOKEN_SCALE_MIN = 1.0 / (FP8_MAX * 512.0)
BLOCK_WEIGHT_AMAX_MIN = 1e-4
OPROJ_AMAX_MIN = 1e-10
ROOT_FP8_POLICY = "online_fp8_sm90"


def root_linear_kind(name: str) -> str | None:
    """Explicit allowlist: never quantize compressor, router, head or norms."""
    match = re.fullmatch(
        r"(?:model\.)?layers\.\d+\.(?:self_attn|attn)\.(wq_a|wq_b|wkv|wo_a|wo_b|indexer\.wq_b)(?:\.weight)?", name
    )
    if match is None:
        return None
    return "block128" if match[1] == "wo_a" else "tensor"


def quantize_root_weight(weight: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Online load-time conversion. Keep canonical [N,K] and FP32 scales."""
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
        raise ValueError("Root FP8 requires contiguous BF16 checkpoint [N,K] weights.")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("Non-finite root weight.")
    n, k = weight.shape
    if min(n, k) <= 0:
        raise ValueError("Empty root weight.")
    if kind == "tensor":
        scale = weight.float().abs().amax().reshape(1) / torch.tensor(
            FP8_MAX, dtype=torch.float32, device=weight.device
        )
        # An all-zero synthetic weight needs a defined scale. Real nonzero
        # weights retain the original amax/448, without a new scale floor.
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        quantized = (weight.float() * scale.reciprocal()).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    elif kind == "block128":
        if n % FP8_BLOCK or k % FP8_BLOCK:
            raise ValueError("wo_a weight dimensions must be multiples of 128.")
        blocks = weight.view(n // FP8_BLOCK, FP8_BLOCK, k // FP8_BLOCK, FP8_BLOCK)
        # The pinned CUDA scalar division uses a rounded FP32 reciprocal.
        # Explicit multiplication also gives CPU/NPU the same scale bits;
        # CPU scalar division otherwise differs by one FP32 ULP here.
        scale = blocks.float().abs().amax((1, 3), keepdim=True).clamp_min(BLOCK_WEIGHT_AMAX_MIN) * (1.0 / FP8_MAX)
        quantized = (blocks * scale.reciprocal()).to(torch.float8_e4m3fn).reshape(n, k)
        scale = scale.reshape(n // FP8_BLOCK, k // FP8_BLOCK)
    else:
        raise ValueError(f"Unknown root quantization kind: {kind}.")
    return quantized.contiguous(), scale.float().contiguous()


def quantize_root_activation(x: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or x.dtype not in (torch.bfloat16, torch.float32) or x.shape[1] == 0:
        raise ValueError("Root activation must be BF16/FP32 [M,K].")
    values = x.float()
    if kind == "tensor":
        scale = (
            values.abs().amax(-1, keepdim=True) / torch.tensor(FP8_MAX, dtype=torch.float32, device=x.device)
        ).clamp_min(TOKEN_SCALE_MIN)
        quantized = (values / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    elif kind == "block128":
        if x.shape[1] % FP8_BLOCK:
            raise ValueError("wo_a activation K must be a multiple of 128.")
        blocks = values.reshape(x.shape[0], x.shape[1] // FP8_BLOCK, FP8_BLOCK)
        raw = blocks.abs().amax(-1, keepdim=True).clamp_min(OPROJ_AMAX_MIN) * (1.0 / FP8_MAX)
        scale = torch.exp2(torch.ceil(torch.log2(raw)))
        quantized = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).reshape_as(x)
        scale = scale.squeeze(-1)
    else:
        raise ValueError(f"Unknown root quantization kind: {kind}.")
    return quantized.contiguous(), scale.float().contiguous()


def inverse_rope_fp32(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, nope_dim: int) -> torch.Tensor:
    """Interleaved inverse RoPE without intermediate BF16 rounding.

    Ascend metadata stores [T,1,1,rope_dim] with repeated even/odd phases.
    No in-place change to the attention output or KV cache.
    """
    if x.ndim != 3 or not 0 <= nope_dim < x.shape[-1] or (x.shape[-1] - nope_dim) % 2:
        raise ValueError("Invalid inverse RoPE input/partial slice.")
    tokens, _, dim = x.shape
    rope_dim = dim - nope_dim
    if cos.shape != sin.shape or tuple(cos.shape) != (tokens, 1, 1, rope_dim):
        raise ValueError("Unexpected Ascend inverse RoPE metadata shape.")
    values = x.float()
    rotary = values[..., nope_dim:].reshape(tokens, x.shape[1], rope_dim // 2, 2)
    c = cos.float().reshape(tokens, 1, rope_dim // 2, 2)
    s = sin.float().reshape(tokens, 1, rope_dim // 2, 2)
    # Match the fused NVIDIA kernel's FP32 multiply-add order: the partner
    # product is rounded first, then x*cos + partner is a single addcmul.
    even = torch.addcmul(rotary[..., 1] * s[..., 0], rotary[..., 0], c[..., 0])
    odd = torch.addcmul(-(rotary[..., 0] * s[..., 1]), rotary[..., 1], c[..., 1])
    return torch.cat((values[..., :nope_dim], torch.stack((even, odd), -1).flatten(-2)), -1)


def root_fp8_matmul_reference(qx, sx, qw, sw, kind: str) -> torch.Tensor:
    """Explicit FP32 reference with one BF16 output cast; not a runtime fallback."""
    if kind == "tensor":
        return ((qx.float() @ qw.float().T) * sw.reshape(1, 1) * sx.reshape(-1, 1)).to(torch.bfloat16)
    if kind != "block128":
        raise ValueError(f"Unknown root quantization kind: {kind}.")
    result = torch.zeros(qx.shape[0], qw.shape[0], dtype=torch.float32, device=qx.device)
    for start in range(0, qw.shape[1], FP8_BLOCK):
        block = start // FP8_BLOCK
        part = qx[:, start : start + FP8_BLOCK].float() @ qw[:, start : start + FP8_BLOCK].float().T
        result += part * (sx[:, block, None] * sw[:, block].repeat_interleave(FP8_BLOCK)[None, :])
    return result.to(torch.bfloat16)


def root_fp8_matmul_npu(qx, sx, qw, sw, kind: str) -> torch.Tensor:
    """CANN FP8 matmul with FP32 scales. No experimental mixed Triton kernel."""
    if qx.device.type != "npu" or any(t.device != qx.device for t in (sx, qw, sw)):
        raise ValueError("Root native FP8 requires tensors on the same NPU.")
    if qx.dtype != torch.float8_e4m3fn or qw.dtype != torch.float8_e4m3fn:
        raise ValueError("Root native FP8 requires E4M3 operands.")
    if sx.dtype != torch.float32 or sw.dtype != torch.float32 or qx.ndim != 2 or qw.ndim != 2:
        raise ValueError("Root native FP8 requires 2D operands and FP32 scales.")
    if not qx.is_contiguous() or not qw.is_contiguous() or qx.shape[1] != qw.shape[1]:
        raise ValueError("Root native FP8 requires contiguous [M,K]/[N,K] operands.")
    # Lazy import keeps the arithmetic/selection contract CPU-testable.
    import torch_npu

    if kind == "tensor":
        if sw.numel() != 1 or sx.shape != (qx.shape[0], 1):
            raise ValueError("Invalid tensor/token scales.")
        return torch_npu.npu_quant_matmul(
            qx, qw.T, sw.reshape(1), pertoken_scale=sx.flatten(), output_dtype=torch.bfloat16
        )
    if kind != "block128" or qw.shape[0] % FP8_BLOCK or qw.shape[1] % FP8_BLOCK:
        raise ValueError("Invalid block FP8 kind/dimensions.")
    if sw.shape != (qw.shape[0] // FP8_BLOCK, qw.shape[1] // FP8_BLOCK) or sx.shape != (
        qx.shape[0],
        qw.shape[1] // FP8_BLOCK,
    ):
        raise ValueError("Invalid block FP8 scales.")
    # op-plugin checks weight and block-scale transpose STRIDES as well as
    # shapes. Preserve matching transpose views; making only sw.T contiguous
    # violates that contract even though the numerical scale grid is correct.
    return torch_npu.npu_quant_matmul(
        qx, qw.T, sw.T, pertoken_scale=sx, group_sizes=[1, 128, 128], output_dtype=torch.bfloat16
    )


class RootFP8State:
    """Per-linear ownership, used by the offline adapter and focused gates."""

    def __init__(self, kind: str):
        if kind not in ("tensor", "block128"):
            raise ValueError("Invalid root kind.")
        self.kind = kind
        self.calls = 0
        self.ready = False

    def process(self, layer) -> None:
        if self.ready:
            raise ValueError("Root FP8 weight processing was called twice; reload is not supported.")
        qweight, scale = quantize_root_weight(layer.weight.detach(), self.kind)
        layer.weight.requires_grad_(False)
        layer.weight.data = qweight
        layer.register_buffer("vq2a8_root_scale", scale)
        self.ready = True

    def apply(self, layer, x, *, matmul=root_fp8_matmul_npu):
        if not self.ready:
            raise ValueError("Root FP8 weights have not been processed.")
        original_shape = x.shape
        qx, sx = quantize_root_activation(x.reshape(-1, x.shape[-1]), self.kind)
        result = matmul(qx, sx, layer.weight, layer.vq2a8_root_scale, self.kind)
        self.calls += 1
        return result.reshape(*original_shape[:-1], layer.weight.shape[0])

    def apply_grouped(self, layer, x, groups: int, rank: int, *, matmul=root_fp8_matmul_npu):
        if not self.ready or self.kind != "block128" or x.ndim != 3:
            raise ValueError("Grouped wo_a requires loaded block FP8 weights and [T,G,K] input.")
        if x.shape[1] != groups or tuple(layer.weight.shape) != (groups * rank, x.shape[2]) or rank % FP8_BLOCK:
            raise ValueError("Invalid grouped wo_a weight/activation layout.")
        outputs = []
        for group in range(groups):
            qx, sx = quantize_root_activation(x[:, group].contiguous(), self.kind)
            weight = layer.weight[group * rank : (group + 1) * rank]
            scale = layer.vq2a8_root_scale[group * rank // FP8_BLOCK : (group + 1) * rank // FP8_BLOCK]
            outputs.append(matmul(qx, sx, weight, scale, self.kind))
        self.calls += 1
        return torch.stack(outputs, dim=1).flatten(1)
