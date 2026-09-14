# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dependency-free TP1/TP2 packed-zN artifact contracts, not device evidence."""

from typing import Any

VQ2_TP1_ZN_FORMAT = "vq2a8_zn_tp1_v1"
VQ2_TP2_ZN_FORMAT = "vq2a8_zn_tp2_v1"


def zn_format(tp_size: int) -> str:
    if not isinstance(tp_size, int) or isinstance(tp_size, bool) or tp_size not in (1, 2):
        raise ValueError("Packed-zN TP size must be integer 1 or 2.")
    return VQ2_TP1_ZN_FORMAT if tp_size == 1 else VQ2_TP2_ZN_FORMAT


def communication_contract(tp_size: int) -> dict[str, str]:
    zn_format(tp_size)
    if tp_size == 2:
        return {
            "gate_up": "column_parallel_separate_gate_and_up_slices_concatenated_per_rank",
            "down": "row_parallel_contiguous_physical_input_slice_sum_partials_across_tp_ranks",
            "activation_quantization": "per_rank_per_row_amax_after_local_RHT128_and_weight_scale",
            "bias_correction": "local_input_contribution_only_before_down_partial_sum",
            "routing": "same_token_expert_assignments_on_both_ranks_not_expert_parallel",
        }
    return {
        "gate_up": "full_gate_and_up_outputs_on_rank0",
        "down": "full_physical_input_projection_on_rank0_no_collective",
        "activation_quantization": "per_row_full_K_amax_after_physical_RHT128_and_weight_scale",
        "bias_correction": "full_input_contribution_added_once_per_projection",
        "routing": "all_token_expert_assignments_on_rank0_not_expert_parallel",
    }


def activation_semantics(tp_size: int) -> dict[str, Any]:
    zn_format(tp_size)
    semantics = {
        "physical_input": "rank-local logical input followed by padding_columns zeros",
        "dummy_metadata": {"weight_scale": 0.0, "weight_bias": 0.0, "rht_sign": 1},
        "preparation_order": [
            "physical_rht128",
            "physical_bias_gemv_and_weight_scale",
            "rank_local_dynamic_fp8",
            "byte_gather_activation_order",
        ],
        "quantization": "per-token per-expert TP-rank-local E4M3FN amax/448 with min_scale=1e-12",
        "bias_correction": "rank-local rotated-input dot weight_bias, added once to that rank projection",
        "down_aggregation": "sum rank-local down projection partials; never duplicate a full-K bias",
        "tp1_bitwise_equivalent": False,
        "floating_point_order": (
            "stable codebook regrouping changes K reduction order; TP2 changes A8 scale/reduction scope"
        ),
    }
    if tp_size == 1:
        semantics.update(
            physical_input="full logical physical input on rank0; no padding columns",
            preparation_order=[
                "physical_rht128",
                "physical_bias_gemv_and_weight_scale",
                "full_K_dynamic_fp8",
                "byte_gather_activation_order",
            ],
            quantization="per-token per-expert full-K E4M3FN amax/448 with min_scale=1e-12",
            bias_correction="full-K rotated-input dot weight_bias, added once to the full projection",
            down_aggregation="full down projection on rank0; no TP collective",
            floating_point_order=(
                "stable codebook regrouping changes K reduction order; device bitwise equivalence is not verified"
            ),
        )
    return semantics
