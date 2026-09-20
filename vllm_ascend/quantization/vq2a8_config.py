# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit supported-mode contract for the extracted TP1 VQ2A8 runtime.

Keep the accepted configuration names readable in existing deployment files,
but reject removed experiments instead of silently selecting a different path.
"""

from types import MappingProxyType

SUPPORTED_MODES = MappingProxyType(
    {
        "execution_policy": "ascendc_v4",
        "root_linear_mode": "bf16",
        "v4_serving": True,
        "v4_device_route_decode": True,
        "v4_decode_graph": "decoder",
        "v4_graph_replay_stream": "caller",
        "v4_compute_backend": "v2",
        "v4_activation_reorder": "vectorized",
        "v4_activation_preparation": "sign_fused_direct",
        "v4_validity_mode": "fused_vectorized",
        "v4_route_mapping": "fused",
        "v4_runtime_guard": "planned",
        "v4_select_sign": "fused",
        "v4_activation_tail": "torch",
        "v4_b1_schedule": "baseline",
        "v4_swiglu_mode": "torch",
        "v4_decoder_metadata_mode": "position_template",
        "v4_decoder_input_mode": "general",
        "v4_host_profile": False,
    }
)


def validate_supported_modes(options):
    """Validate provided mode values; missing values use the accepted defaults."""
    for key, expected in SUPPORTED_MODES.items():
        actual = options.get(key, expected)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"{key} supports only {expected!r}; got {actual!r}. No fallback is enabled.")


def resolve_runtime_options(options):
    """Return owned options with explicit defaults and no unknown settings."""
    if not isinstance(options, dict) or options.get("enabled") is not True:
        raise ValueError("VQ2A8 requires additional_config.vq2a8_offline.enabled=true.")
    allowed = set(SUPPORTED_MODES) | {
        "enabled",
        "artifact",
        "cache_experts",
        "token_chunk",
        "cache_budget_gib",
        "cache_reserve_gib",
        "cache_memory_fraction",
        "ascendc_library",
        "ascendc_sha256",
        "verbose_experts",
    }
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unsupported VQ2A8 options: {sorted(unknown)}.")
    if not isinstance(options.get("artifact"), str) or not options["artifact"]:
        raise ValueError("VQ2A8 requires a nonempty artifact path.")
    validate_supported_modes(options)
    return {**SUPPORTED_MODES, **options}
