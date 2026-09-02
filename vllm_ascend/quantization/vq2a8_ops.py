# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Ascend VQ2A8 operator dispatch and correctness fallback."""

from __future__ import annotations

import importlib
import logging
from enum import Enum

import torch
import torch.nn.functional as F

from vllm_ascend.quantization.vq2a8_format import decode_repacked_vq2_weight

logger = logging.getLogger(__name__)
_warned_reference_fallback = False
_warned_extension_load_failure = False

VQ2A8_CUSTOM_OPS = ("vq2a8_gate_up", "vq2a8_down_reduce")


def _normalize_activation_name(activation: str | Enum) -> str:
    value = activation.value if isinstance(activation, Enum) else activation
    if not isinstance(value, str):
        raise TypeError(
            "VQ2A8 activation must be a string or string-valued enum, "
            f"got {activation!r}."
        )
    return value


def _registered_custom_op(name: str):
    namespace = getattr(torch.ops, "_C_ascend", None)
    if namespace is None:
        return None
    return getattr(namespace, name, None)


def _ensure_vq2a8_extension_loaded() -> bool:
    """Lazily load the extension without enabling every A5 custom op."""
    global _warned_extension_load_failure
    if all(_registered_custom_op(name) is not None for name in VQ2A8_CUSTOM_OPS):
        return True
    try:
        importlib.import_module("vllm_ascend.vllm_ascend_C")
    except ImportError as exc:
        if not _warned_extension_load_failure:
            logger.warning("Failed to load the Ascend VQ2A8 extension: %s", exc)
            _warned_extension_load_failure = True
        return False
    return all(_registered_custom_op(name) is not None for name in VQ2A8_CUSTOM_OPS)


def _custom_op(name: str):
    op = _registered_custom_op(name)
    if op is not None:
        return op
    _ensure_vq2a8_extension_loaded()
    return _registered_custom_op(name)


def custom_vq2a8_ops_available() -> bool:
    """Return whether both Ascend 950 VQ2A8 kernels are registered."""
    return all(_custom_op(name) is not None for name in VQ2A8_CUSTOM_OPS)


def custom_vq2a8_gate_up_available() -> bool:
    """Return whether the Ascend 950 packed gate/up kernel is registered."""
    return _custom_op("vq2a8_gate_up") is not None


def custom_vq2a8_down_reduce_available() -> bool:
    """Return whether the Ascend 950 packed down/reduce kernel is registered."""
    return _custom_op("vq2a8_down_reduce") is not None


def _validate_payload(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    packed_indices: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_tile_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
) -> None:
    if x.ndim != 2:
        raise ValueError(f"VQ2A8 input must be 2D, got {tuple(x.shape)}.")
    if expert_ids.ndim != 1 or expert_ids.shape[0] != x.shape[0]:
        raise ValueError(
            "VQ2A8 expert_ids must be 1D with one ID per input row."
        )
    if packed_indices.ndim != 3:
        raise ValueError(
            "VQ2A8 packed indices must have shape [experts, output_pairs, words]."
        )
    num_experts = packed_indices.shape[0]
    payloads = {
        "codebooks": codebooks,
        "codebook_tile_ids": codebook_tile_ids,
        "weight_scale": weight_scale,
        "weight_bias": weight_bias,
        "rht_sign": rht_sign,
    }
    for name, tensor in payloads.items():
        if tensor.shape[0] != num_experts:
            raise ValueError(
                f"VQ2A8 {name} has {tensor.shape[0]} experts, "
                f"expected {num_experts}."
            )
    input_size = codebook_tile_ids.shape[1]
    if x.shape[1] != input_size:
        raise ValueError(
            f"VQ2A8 input width is {x.shape[1]}, expected {input_size}."
        )
    if packed_indices.shape[2] * 8 < input_size:
        raise ValueError("VQ2A8 packed indices do not cover the input width.")


def _decode_expert(
    expert_index: int,
    packed_indices: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_tile_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    rht_block_size: int,
    row_group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return decode_repacked_vq2_weight(
        packed_indices[expert_index],
        codebooks[expert_index],
        codebook_tile_ids[expert_index],
        weight_scale[expert_index],
        weight_bias[expert_index],
        rht_sign[expert_index],
        rht_block_size,
        row_group_size,
    ).to(dtype=dtype)


def reference_vq2a8_gate_up(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    packed_indices: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_tile_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    rht_block_size: int,
    row_group_size: int,
    activation: str | Enum = "silu",
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """Decode selected gate/up experts and apply SwiGLU for bring-up tests."""
    _validate_payload(
        x,
        expert_ids,
        packed_indices,
        codebooks,
        codebook_tile_ids,
        weight_scale,
        weight_bias,
        rht_sign,
    )
    activation_name = _normalize_activation_name(activation)
    if activation_name not in ("silu", "swiglu"):
        raise NotImplementedError(
            f"VQ2A8 reference fallback only supports SwiGLU, got {activation!r}."
        )
    output_size = packed_indices.shape[1] * 2
    if output_size % 2:
        raise ValueError(f"gate_up output width must be even, got {output_size}.")
    projected = torch.empty(
        (x.shape[0], output_size), device=x.device, dtype=x.dtype
    )
    for expert_index in torch.unique(expert_ids).cpu().tolist():
        if not 0 <= expert_index < packed_indices.shape[0]:
            raise ValueError(f"VQ2A8 expert ID {expert_index} is out of range.")
        rows = torch.where(expert_ids == expert_index)[0]
        weight = _decode_expert(
            expert_index,
            packed_indices,
            codebooks,
            codebook_tile_ids,
            weight_scale,
            weight_bias,
            rht_sign,
            rht_block_size,
            row_group_size,
            x.dtype,
        )
        values = x.index_select(0, rows) @ weight.transpose(0, 1)
        projected.index_copy_(0, rows, values)
    gate, up = projected.chunk(2, dim=-1)
    if swiglu_limit is not None and swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return F.silu(gate) * up


def reference_vq2a8_down_reduce(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    token_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    packed_indices: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_tile_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    rht_block_size: int,
    row_group_size: int,
    num_tokens: int,
) -> torch.Tensor:
    """Decode selected down experts and reduce routed rows to input tokens."""
    _validate_payload(
        x,
        expert_ids,
        packed_indices,
        codebooks,
        codebook_tile_ids,
        weight_scale,
        weight_bias,
        rht_sign,
    )
    if token_ids.shape != expert_ids.shape or routing_weights.shape != expert_ids.shape:
        raise ValueError("token_ids and routing_weights must match expert_ids.")
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}.")
    output_size = packed_indices.shape[1] * 2
    output = torch.zeros(
        (num_tokens, output_size), device=x.device, dtype=x.dtype
    )
    for expert_index in torch.unique(expert_ids).cpu().tolist():
        if not 0 <= expert_index < packed_indices.shape[0]:
            raise ValueError(f"VQ2A8 expert ID {expert_index} is out of range.")
        rows = torch.where(expert_ids == expert_index)[0]
        weight = _decode_expert(
            expert_index,
            packed_indices,
            codebooks,
            codebook_tile_ids,
            weight_scale,
            weight_bias,
            rht_sign,
            rht_block_size,
            row_group_size,
            x.dtype,
        )
        values = x.index_select(0, rows) @ weight.transpose(0, 1)
        values *= routing_weights.index_select(0, rows).to(x.dtype).unsqueeze(-1)
        output.index_add_(0, token_ids.index_select(0, rows).long(), values)
    return output


def vq2a8_gate_up(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    packed_indices: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_tile_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    rht_block_size: int,
    row_group_size: int,
    activation: str | Enum,
    allow_reference_fallback: bool,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    global _warned_reference_fallback
    activation_name = _normalize_activation_name(activation)
    op = _custom_op("vq2a8_gate_up")
    if op is not None:
        return op(
            x,
            expert_ids,
            packed_indices,
            codebooks,
            codebook_tile_ids,
            weight_scale,
            weight_bias,
            rht_sign,
            rht_block_size,
            row_group_size,
            activation_name,
            float(swiglu_limit) if swiglu_limit is not None else 0.0,
        )
    if not allow_reference_fallback:
        raise RuntimeError("Ascend custom op _C_ascend.vq2a8_gate_up is unavailable.")
    if not _warned_reference_fallback:
        logger.warning(
            "Using the VQ2A8 BF16 reference fallback. This validates "
            "correctness but is not suitable for performance measurements."
        )
        _warned_reference_fallback = True
    return reference_vq2a8_gate_up(
        x,
        expert_ids,
        packed_indices,
        codebooks,
        codebook_tile_ids,
        weight_scale,
        weight_bias,
        rht_sign,
        rht_block_size,
        row_group_size,
        activation_name,
        swiglu_limit,
    )


def vq2a8_down_reduce(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    token_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    packed_indices: torch.Tensor,
    codebooks: torch.Tensor,
    codebook_tile_ids: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_bias: torch.Tensor,
    rht_sign: torch.Tensor,
    rht_block_size: int,
    row_group_size: int,
    num_tokens: int,
    allow_reference_fallback: bool,
) -> torch.Tensor:
    op = _custom_op("vq2a8_down_reduce")
    if op is not None:
        return op(
            x,
            expert_ids,
            token_ids,
            routing_weights,
            packed_indices,
            codebooks,
            codebook_tile_ids,
            weight_scale,
            weight_bias,
            rht_sign,
            rht_block_size,
            row_group_size,
            num_tokens,
        )
    if not allow_reference_fallback:
        raise RuntimeError(
            "Ascend custom op _C_ascend.vq2a8_down_reduce is unavailable."
        )
    return reference_vq2a8_down_reduce(
        x,
        expert_ids,
        token_ids,
        routing_weights,
        packed_indices,
        codebooks,
        codebook_tile_ids,
        weight_scale,
        weight_bias,
        rht_sign,
        rht_block_size,
        row_group_size,
        num_tokens,
    )
