# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compact VQ2 routed-expert loader for Ascend."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.quantization.vq2a8_format import (
    ASCEND_VQ2_TP_FORMAT,
    extract_decoder_layer_index,
)

if TYPE_CHECKING:
    from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult

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
            layer.register_buffer(f"vq_{kind}_{field}", torch.empty(0), persistent=False)
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
    allow_compact_experts: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load and validate one TP-local packed VQ2 layer on CPU."""
    metadata_path, tensor_path = repacked_layer_paths(kernel_path, layer_index, tp_size, tp_rank)
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
                f"Ascend VQ2 metadata {key}={metadata.get(key)!r}, expected {expected_value!r}: {metadata_path}."
            )
    expert_ids = metadata.get("expert_ids", [])
    expert_count_matches = len(expert_ids) == expected_experts
    compact_expert_count_is_valid = allow_compact_experts and 0 < len(expert_ids) <= expected_experts
    if not expert_count_matches and not compact_expert_count_is_valid:
        raise ValueError(f"Layer {layer_index} repack has {len(expert_ids)} experts, expected {expected_experts}.")
    if metadata.get("complete") is not True:
        raise ValueError(f"Layer {layer_index} repack is not marked complete.")
    if expert_ids != list(range(len(expert_ids))):
        raise ValueError(f"Layer {layer_index} VQ2 expert IDs must be contiguous from zero.")
    for key in ("rht_block_size", "row_group_size"):
        if not isinstance(metadata.get(key), int) or metadata[key] <= 0:
            raise ValueError(f"Layer {layer_index} has invalid {key}={metadata.get(key)!r}.")
    expected_keys = {f"{kind}_{field}" for kind in ASCEND_VQ2_KINDS for field in ASCEND_VQ2_FIELDS}
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
        missing = sorted(expected_keys - actual_keys)
        if missing:
            raise ValueError(f"Layer {layer_index} repack is missing {missing[0]!r}.")
        tensors = {key: handle.get_tensor(key) for key in sorted(expected_keys)}
    return tensors, metadata


def _validate_runtime_options(
    enable_force_load_balance: bool,
    log2phy: torch.Tensor | None,
    global_redundant_expert_num: int,
    mc2_mask: torch.Tensor | None,
) -> None:
    # Profile/dummy runs use this flag only to spread synthetic tokens across
    # experts. It does not mean that EPLB or MC2 is enabled.
    del enable_force_load_balance
    if log2phy is not None or global_redundant_expert_num or mc2_mask is not None:
        raise NotImplementedError("VQ2A8 bring-up supports tensor parallelism without EPLB or MC2.")


def _wrap_fused_experts_result(output: torch.Tensor) -> FusedExpertsResult:
    # Keep this import lazy so format/loader tests do not require torch-npu.
    from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult

    return FusedExpertsResult(routed_out=output)


def _selected_expert_payload_on_cpu(
    layer: torch.nn.Module,
    kind: str,
    expert_id: int,
) -> tuple[torch.Tensor, ...]:
    """Copy one real expert's compact payload for an opt-in CPU golden run."""
    return tuple(
        getattr(layer, f"vq_{kind}_{field}")[expert_id : expert_id + 1].detach().cpu() for field in ASCEND_VQ2_FIELDS
    )


def _compare_first_assignment_with_reference(
    *,
    layer: torch.nn.Module,
    expanded_x: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up: torch.Tensor,
    activation: str,
    call_index: int,
    layer_index: int,
    tp_rank: int,
) -> None:
    """Compare one routed row with CPU golden math from the real payload."""
    if expanded_x.shape[0] == 0:
        return

    from vllm_ascend.quantization.vq2a8_debug import (
        log_vq2a8_comparison,
        log_vq2a8_event,
    )
    from vllm_ascend.quantization.vq2a8_ops import (
        custom_vq2a8_prepare_debug_available,
        reference_vq2a8_down_reduce,
        reference_vq2a8_gate_up,
        reference_vq2a8_prepare,
        vq2a8_down_reduce,
        vq2a8_prepare_debug,
    )

    expert_id = int(expert_ids[0].detach().cpu())
    cpu_expert_ids = torch.zeros(1, dtype=torch.int32)
    cpu_token_ids = torch.zeros(1, dtype=torch.int64)
    cpu_x = expanded_x[0:1].detach().cpu()
    cpu_routing_weight = routing_weights[0:1].detach().float().cpu()
    gate_payload = _selected_expert_payload_on_cpu(layer, "gate_up", expert_id)
    down_payload = _selected_expert_payload_on_cpu(layer, "down", expert_id)
    comparison_extra = {
        "physical_expert": expert_id,
        "probe_rows": 1,
        "payload": "real_model",
    }

    if custom_vq2a8_prepare_debug_available():
        native_quantized, native_scale, native_bias = vq2a8_prepare_debug(
            expanded_x[0:1],
            expert_ids[0:1],
            layer.vq_gate_up_weight_scale,
            layer.vq_gate_up_weight_bias,
            layer.vq_gate_up_rht_sign,
            layer.vq_rht_block_size,
        )
        reference_quantized, reference_scale, reference_bias = reference_vq2a8_prepare(
            cpu_x,
            cpu_expert_ids,
            gate_payload[3],
            gate_payload[4],
            gate_payload[5],
            layer.vq_rht_block_size,
        )
        for stage, actual, expected in (
            ("prepare_quantized", native_quantized, reference_quantized),
            ("prepare_scale", native_scale, reference_scale),
            ("prepare_bias", native_bias, reference_bias),
        ):
            log_vq2a8_comparison(
                scope="moe",
                stage=stage,
                actual=actual,
                expected=expected,
                call_index=call_index,
                layer_index=layer_index,
                tp_rank=tp_rank,
                extra=comparison_extra,
            )
    else:
        log_vq2a8_event(
            scope="moe",
            stage="prepare_comparison_unavailable",
            call_index=call_index,
            layer_index=layer_index,
            tp_rank=tp_rank,
            values=comparison_extra,
        )

    reference_gate_up = reference_vq2a8_gate_up(
        cpu_x,
        cpu_expert_ids,
        *gate_payload,
        rht_block_size=layer.vq_rht_block_size,
        row_group_size=layer.vq_row_group_size,
        activation=activation,
        swiglu_limit=getattr(layer, "swiglu_limit", None),
    )
    log_vq2a8_comparison(
        scope="moe",
        stage="gate_up",
        actual=gate_up[0:1],
        expected=reference_gate_up,
        call_index=call_index,
        layer_index=layer_index,
        tp_rank=tp_rank,
        extra=comparison_extra,
    )

    probe_token_ids = torch.zeros(1, dtype=torch.int64, device=expanded_x.device)
    native_down = vq2a8_down_reduce(
        gate_up[0:1],
        expert_ids[0:1],
        probe_token_ids,
        routing_weights[0:1],
        layer.vq_down_packed_indices,
        layer.vq_down_codebooks,
        layer.vq_down_codebook_tile_ids,
        layer.vq_down_weight_scale,
        layer.vq_down_weight_bias,
        layer.vq_down_rht_sign,
        layer.vq_rht_block_size,
        layer.vq_row_group_size,
        1,
        allow_reference_fallback=False,
    )
    reference_down = reference_vq2a8_down_reduce(
        reference_gate_up,
        cpu_expert_ids,
        cpu_token_ids,
        cpu_routing_weight,
        *down_payload,
        rht_block_size=layer.vq_rht_block_size,
        row_group_size=layer.vq_row_group_size,
        num_tokens=1,
    )
    log_vq2a8_comparison(
        scope="moe",
        stage="down_isolated",
        actual=native_down,
        expected=reference_down,
        call_index=call_index,
        layer_index=layer_index,
        tp_rank=tp_rank,
        extra=comparison_extra,
    )


class AscendVQ2A8MoEMethod(FusedMoEMethodBase):
    """vLLM MoE method backed by TP-local packed VQ2 artifacts."""

    def __init__(
        self,
        moe_config,
        kernel_path: str,
        prefix: str,
        tid2eid: torch.Tensor | None = None,
        allow_reference_fallback: bool = True,
    ) -> None:
        super().__init__(moe_config)
        self.kernel_path = kernel_path
        self.layer_index = extract_decoder_layer_index(prefix)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tid2eid = tid2eid
        self.allow_reference_fallback = allow_reference_fallback

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
        create_compact_expert_parameters(layer, num_experts, params_dtype, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_vq2a8_ascend_loaded", False):
            return
        tensors, metadata = load_repacked_layer(
            self.kernel_path,
            self.layer_index,
            self.tp_size,
            self.tp_rank,
            layer.num_experts,
            allow_compact_experts=self.tid2eid is not None,
        )
        device = layer.w13_weight.device
        for key, tensor in tensors.items():
            setattr(
                layer,
                f"vq_{key}",
                tensor.to(device=device, non_blocking=True).contiguous(),
            )
        layer.vq_expert_ids = torch.tensor(metadata["expert_ids"], device=device, dtype=torch.int32)
        layer.vq_rht_block_size = int(metadata["rht_block_size"])
        layer.vq_row_group_size = int(metadata["row_group_size"])
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
    ) -> FusedExpertsResult:
        del is_prefill, pertoken_scale
        if expert_map is not None:
            raise NotImplementedError("VQ2A8 does not support expert parallelism yet.")
        _validate_runtime_options(
            enable_force_load_balance,
            log2phy,
            global_redundant_expert_num,
            mc2_mask,
        )

        from vllm_ascend.ops.fused_moe.experts_selector import select_experts
        from vllm_ascend.quantization.vq2a8_debug import (
            log_vq2a8_event,
            log_vq2a8_tensor,
            reserve_vq2a8_debug_call,
            vq2a8_debug_compare_reference_enabled,
        )
        from vllm_ascend.quantization.vq2a8_ops import (
            vq2a8_down_reduce,
            vq2a8_gate_up,
        )

        debug_call = reserve_vq2a8_debug_call("moe", self.layer_index)
        if debug_call is not None:
            log_vq2a8_tensor(
                scope="moe",
                stage="input",
                tensor=x,
                call_index=debug_call,
                layer_index=self.layer_index,
                tp_rank=self.tp_rank,
            )
            log_vq2a8_tensor(
                scope="moe",
                stage="router_logits",
                tensor=router_logits,
                call_index=debug_call,
                layer_index=self.layer_index,
                tp_rank=self.tp_rank,
            )

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=layer.num_experts if num_experts == -1 else num_experts,
            tid2eid=self.tid2eid,
        )
        token_ids = (
            torch.arange(x.shape[0], device=x.device, dtype=torch.int64).unsqueeze(1).expand_as(topk_ids).reshape(-1)
        )
        expert_ids = topk_ids.reshape(-1).to(torch.int32)
        routing_weights = topk_weights.reshape(-1)
        expanded_x = x.index_select(0, token_ids)
        if debug_call is not None:
            expert_ids_cpu = expert_ids.detach().cpu()
            packed_experts = int(layer.vq_gate_up_packed_indices.shape[0])
            minimum_expert = int(expert_ids_cpu.min()) if expert_ids_cpu.numel() else 0
            maximum_expert = int(expert_ids_cpu.max()) if expert_ids_cpu.numel() else -1
            if minimum_expert < 0 or maximum_expert >= packed_experts:
                raise RuntimeError(
                    f"VQ2A8 layer {self.layer_index} selected expert IDs in "
                    f"[{minimum_expert}, {maximum_expert}], but its packed artifact "
                    f"contains {packed_experts} experts."
                )
            log_vq2a8_tensor(
                scope="moe",
                stage="topk_ids",
                tensor=expert_ids,
                call_index=debug_call,
                layer_index=self.layer_index,
                tp_rank=self.tp_rank,
            )
            log_vq2a8_tensor(
                scope="moe",
                stage="topk_weights",
                tensor=topk_weights,
                call_index=debug_call,
                layer_index=self.layer_index,
                tp_rank=self.tp_rank,
                extra={
                    "row_sums": topk_weights.detach().float().sum(dim=-1).cpu().tolist(),
                    "routed_scaling_factor": routed_scaling_factor,
                },
            )
            log_vq2a8_event(
                scope="moe",
                stage="artifact",
                call_index=debug_call,
                layer_index=self.layer_index,
                tp_rank=self.tp_rank,
                values={
                    "packed_experts": packed_experts,
                    "gate_up_input": list(expanded_x.shape),
                    "gate_up_output": int(layer.vq_gate_up_packed_indices.shape[1]),
                    "down_output": int(layer.vq_down_packed_indices.shape[1] * 2),
                    "rht_block_size": int(layer.vq_rht_block_size),
                    "row_group_size": int(layer.vq_row_group_size),
                },
            )
        if apply_router_weight_on_input:
            expanded_x = expanded_x * routing_weights.to(x.dtype).unsqueeze(-1)
            routing_weights = torch.ones_like(routing_weights)

        gate_up = vq2a8_gate_up(
            expanded_x,
            expert_ids,
            layer.vq_gate_up_packed_indices,
            layer.vq_gate_up_codebooks,
            layer.vq_gate_up_codebook_tile_ids,
            layer.vq_gate_up_weight_scale,
            layer.vq_gate_up_weight_bias,
            layer.vq_gate_up_rht_sign,
            layer.vq_rht_block_size,
            layer.vq_row_group_size,
            activation,
            self.allow_reference_fallback,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
        )
        if debug_call is not None:
            log_vq2a8_tensor(
                scope="moe",
                stage="gate_up",
                tensor=gate_up,
                call_index=debug_call,
                layer_index=self.layer_index,
                tp_rank=self.tp_rank,
            )
            if vq2a8_debug_compare_reference_enabled(self.layer_index):
                _compare_first_assignment_with_reference(
                    layer=layer,
                    expanded_x=expanded_x,
                    expert_ids=expert_ids,
                    routing_weights=routing_weights,
                    gate_up=gate_up,
                    activation=activation,
                    call_index=debug_call,
                    layer_index=self.layer_index,
                    tp_rank=self.tp_rank,
                )
        output = vq2a8_down_reduce(
            gate_up,
            expert_ids,
            token_ids,
            routing_weights,
            layer.vq_down_packed_indices,
            layer.vq_down_codebooks,
            layer.vq_down_codebook_tile_ids,
            layer.vq_down_weight_scale,
            layer.vq_down_weight_bias,
            layer.vq_down_rht_sign,
            layer.vq_rht_block_size,
            layer.vq_row_group_size,
            x.shape[0],
            self.allow_reference_fallback,
        )
        if debug_call is not None:
            log_vq2a8_tensor(
                scope="moe",
                stage="down_before_tp",
                tensor=output,
                call_index=debug_call,
                layer_index=self.layer_index,
                tp_rank=self.tp_rank,
            )
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        if debug_call is not None:
            log_vq2a8_tensor(
                scope="moe",
                stage="down_after_tp",
                tensor=output,
                call_index=debug_call,
                layer_index=self.layer_index,
                tp_rank=self.tp_rank,
            )
        return _wrap_fused_experts_result(output)
