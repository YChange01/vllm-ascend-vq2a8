# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compact VQ2 routed-expert loader for Ascend."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.quantization.vq2a8_format import extract_decoder_layer_index

ASCEND_VQ2_TP_FORMAT = "vq2a8_ascend_tp_v1"
ASCEND_VQ2_FIELDS = (
    "packed_indices",
    "codebooks",
    "codebook_tile_ids",
    "weight_scale",
    "weight_bias",
    "rht_sign",
)
ASCEND_VQ2_KINDS = ("gate_up", "down")


def _discard_dense_expert_weight(
    parameter: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    *args: Any,
    return_success: bool = False,
    **kwargs: Any,
) -> bool | None:
    del parameter, loaded_weight, args, kwargs
    return True if return_success else None


def create_compact_expert_parameters(
    layer: torch.nn.Module,
    num_experts: int,
    params_dtype: torch.dtype,
    extra_weight_attrs: dict[str, Any],
) -> None:
    """Register zero-sized dense placeholders and packed VQ2 buffers."""
    attrs = dict(extra_weight_attrs)
    attrs["weight_loader"] = _discard_dense_expert_weight
    for name in ("w13_weight", "w2_weight"):
        parameter = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter(name, parameter)
        set_weight_attrs(parameter, attrs)
    for kind in ASCEND_VQ2_KINDS:
        for field in ASCEND_VQ2_FIELDS:
            layer.register_buffer(
                f"vq_{kind}_{field}", torch.empty(0), persistent=False
            )
    layer.register_buffer("vq_expert_ids", torch.empty(0), persistent=False)


def repacked_layer_paths(
    kernel_path: str | Path,
    layer_index: int,
    tp_size: int,
    tp_rank: int,
) -> tuple[Path, Path]:
    rank_path = Path(kernel_path) / f"tp{tp_size}" / f"rank{tp_rank}"
    stem = f"experts_vq_layer_{layer_index}"
    return rank_path / f"{stem}.json", rank_path / f"{stem}.safetensors"


def load_repacked_layer(
    kernel_path: str | Path,
    layer_index: int,
    tp_size: int,
    tp_rank: int,
    expected_experts: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load and validate one TP-local packed VQ2 layer on CPU."""
    metadata_path, tensor_path = repacked_layer_paths(
        kernel_path, layer_index, tp_size, tp_rank
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Ascend VQ2 metadata: {metadata_path}.")
    if not tensor_path.is_file():
        raise FileNotFoundError(f"Missing Ascend VQ2 tensors: {tensor_path}.")
    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    expected_metadata = {
        "format": ASCEND_VQ2_TP_FORMAT,
        "layer": layer_index,
        "tp_size": tp_size,
        "tp_rank": tp_rank,
    }
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"Ascend VQ2 metadata {key}={metadata.get(key)!r}, "
                f"expected {expected_value!r}: {metadata_path}."
            )
    expert_ids = metadata.get("expert_ids", [])
    if len(expert_ids) != expected_experts:
        raise ValueError(
            f"Layer {layer_index} repack has {len(expert_ids)} experts, "
            f"expected {expected_experts}."
        )
    expected_keys = {
        f"{kind}_{field}"
        for kind in ASCEND_VQ2_KINDS
        for field in ASCEND_VQ2_FIELDS
    }
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
        missing = sorted(expected_keys - actual_keys)
        if missing:
            raise ValueError(
                f"Layer {layer_index} repack is missing {missing[0]!r}."
            )
        tensors = {key: handle.get_tensor(key) for key in sorted(expected_keys)}
    return tensors, metadata


class AscendVQ2A8MoEMethod(FusedMoEMethodBase):
    """vLLM MoE method backed by TP-local packed VQ2 artifacts."""

    def __init__(
        self,
        moe_config,
        kernel_path: str,
        prefix: str,
        tid2eid: torch.Tensor | None = None,
    ) -> None:
        super().__init__(moe_config)
        self.kernel_path = kernel_path
        self.layer_index = extract_decoder_layer_index(prefix)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tid2eid = tid2eid

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del hidden_size, intermediate_size_per_partition
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        create_compact_expert_parameters(
            layer, num_experts, params_dtype, extra_weight_attrs
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_vq2a8_ascend_loaded", False):
            return
        tensors, metadata = load_repacked_layer(
            self.kernel_path,
            self.layer_index,
            self.tp_size,
            self.tp_rank,
            layer.num_experts,
        )
        device = layer.w13_weight.device
        for key, tensor in tensors.items():
            setattr(
                layer,
                f"vq_{key}",
                tensor.to(device=device, non_blocking=True).contiguous(),
            )
        layer.vq_expert_ids = torch.tensor(
            metadata["expert_ids"], device=device, dtype=torch.int32
        )
        layer._vq2a8_ascend_loaded = True

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        del layer
        return None

    @property
    def supports_eplb(self) -> bool:
        return False

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: torch.Tensor | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del (
            layer,
            x,
            router_logits,
            top_k,
            renormalize,
            use_grouped_topk,
            num_experts,
            expert_map,
            topk_group,
            num_expert_group,
            custom_routing_function,
            scoring_func,
            routed_scaling_factor,
            e_score_correction_bias,
            is_prefill,
            enable_force_load_balance,
            log2phy,
            global_redundant_expert_num,
            pertoken_scale,
            activation,
            apply_router_weight_on_input,
            mc2_mask,
        )
        raise NotImplementedError(
            "Ascend 950 VQ2A8 custom operators are not built. "
            "Build the vq2a8_gate_up and vq2a8_down_reduce operators."
        )
