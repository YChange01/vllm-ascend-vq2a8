# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-testable contracts for the opt-in, eager offline model adapter."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open

from vllm_ascend.quantization.vq2a8_abcd import validate_candidates
from vllm_ascend.quantization.vq2a8_execution import (
    GIB,
    AscendCVQ2TP1MoE,
    CachedVQ2TP1MoE,
    device_cache_budget,
    packed_cache_plan,
)
from vllm_ascend.quantization.vq2a8_moe import VQ2TP1MoE
from vllm_ascend.quantization.vq2a8_repack import VQ2_DIRECT_TP1_FORMAT
from vllm_ascend.quantization.vq2a8_runtime import open_vq2a8_tp1_artifact
from vllm_ascend.quantization.vq2a8_tp1_zn_runtime import artifact_format, open_vq2a8_tp1_zn_artifact
from vllm_ascend.quantization.vq2a8_v4_v2_prepacked import V4_V2_PREPACKED_FORMAT
from vllm_ascend.quantization.vq2a8_zn_contract import VQ2_TP1_ZN_FORMAT

OFFLINE_CONTEXT_LIMIT = 32
OFFLINE_NEW_TOKENS = 4
OFFLINE_RUNS = 2
V4_DECODER_CONTEXT_LIMIT = 16
CACHE_EXECUTION_POLICIES = ("cached", "ascendc", "ascendc_v2", "ascendc_v3", "ascendc_v4")


def _validate_v4_compute_backend(value, policy):
    if value not in ("v1", "v2") or (value != "v1" and policy != "ascendc_v4"):
        raise ValueError("v4_compute_backend requires v1|v2; v2 requires execution_policy=ascendc_v4.")


def _validate_v4_activation_options(reorder, preparation, backend, policy):
    if reorder not in ("scalar", "vectorized", "row_reuse", "chunk_reuse2", "chunk_reuse4"):
        raise ValueError("Invalid v4_activation_reorder.")
    if preparation not in (
        "rowwise",
        "rowwise_packed",
        "sign_fused",
        "sign_fused_strided",
        "sign_fused_direct",
        "fused",
    ):
        raise ValueError("Invalid v4_activation_preparation mode.")
    if (reorder != "scalar" or preparation != "rowwise") and (backend != "v2" or policy != "ascendc_v4"):
        raise ValueError("V4 activation optimizations require execution_policy=ascendc_v4 and v4_compute_backend=v2.")


def _validate_cache_memory_fraction(value, policy):
    if policy not in CACHE_EXECUTION_POLICIES:
        raise ValueError("cache_memory_fraction requires a cached or native AscendC execution policy.")
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("cache_memory_fraction must be a finite number in (0,1], not a boolean.")


def _validate_v4_host_options(metadata_mode, host_profile, graph_mode, policy):
    if metadata_mode not in ("recursive", "planned", "planned_fast", "position_template"):
        raise ValueError("v4_decoder_metadata_mode requires recursive|planned|planned_fast|position_template.")
    if metadata_mode != "recursive" and (policy != "ascendc_v4" or graph_mode != "decoder"):
        raise ValueError("Planned metadata requires V4 decoder graphs.")
    if type(host_profile) is not bool or (host_profile and (policy != "ascendc_v4" or graph_mode != "decoder")):
        raise ValueError("v4_host_profile must be boolean and requires V4 decoder graphs.")


def _validate_v4_validity_options(mode, preparation, backend, policy, device_route):
    if mode not in ("torch", "fused", "fused_vectorized"):
        raise ValueError("v4_validity_mode requires torch|fused|fused_vectorized.")
    if mode in ("fused", "fused_vectorized") and (
        policy != "ascendc_v4"
        or backend != "v2"
        or device_route is not True
        or preparation not in ("sign_fused", "sign_fused_strided", "sign_fused_direct")
    ):
        raise ValueError("Fused validity requires V4 v2 device-route decode with native sign preparation.")


def _validate_v4_route_mapping(mode, backend, policy, device_route):
    if mode not in ("torch", "fused"):
        raise ValueError("v4_route_mapping requires torch|fused.")
    if mode == "fused" and (backend != "v2" or policy != "ascendc_v4" or device_route is not True):
        raise ValueError("Fused route mapping requires V4 v2 device-route decode.")


def offline_engine_options(
    model_root: Path,
    artifact: Path,
    *,
    execution_policy="cached",
    cache_budget_gib=0.0,
    cache_reserve_gib=16.0,
    cache_memory_fraction=None,
    root_linear_mode="bf16",
    ascendc_library=None,
    ascendc_sha256=None,
    ascendc_v2_library=None,
    ascendc_v2_sha256=None,
    ascendc_v3_library=None,
    ascendc_v3_sha256=None,
    v3_preparation="eager",
    v3_decode_graph="none",
    v3_serving=False,
    v4_serving=False,
    v4_device_route_decode=False,
    v4_decode_graph="none",
    v4_graph_replay_stream="owner",
    v4_compute_backend="v1",
    v4_activation_reorder="scalar",
    v4_activation_preparation="rowwise",
    v4_validity_mode="torch",
    v4_route_mapping="torch",
    v4_runtime_guard="signature",
    v4_select_sign="separate",
    v4_activation_tail="torch",
    v4_b1_schedule="baseline",
    v4_decoder_metadata_mode="recursive",
    v4_decoder_input_mode="general",
    v4_host_profile=False,
    verbose_experts=False,
    tensor_parallel_size=1,
) -> dict:
    """A fixed, bounded bring-up plan, not a general serving configuration."""
    _validate_v4_compute_backend(v4_compute_backend, execution_policy)
    _validate_v4_activation_options(
        v4_activation_reorder, v4_activation_preparation, v4_compute_backend, execution_policy
    )
    _validate_v4_host_options(v4_decoder_metadata_mode, v4_host_profile, v4_decode_graph, execution_policy)
    _validate_v4_validity_options(
        v4_validity_mode, v4_activation_preparation, v4_compute_backend, execution_policy, v4_device_route_decode
    )
    _validate_v4_route_mapping(v4_route_mapping, v4_compute_backend, execution_policy, v4_device_route_decode)
    validate_candidates(
        v4_runtime_guard,
        v4_select_sign,
        v4_activation_tail,
        decoder_input_mode=v4_decoder_input_mode,
        b1_schedule=v4_b1_schedule,
        backend=v4_compute_backend,
        policy=execution_policy,
        preparation=v4_activation_preparation,
        reorder=v4_activation_reorder,
        device_route=v4_device_route_decode,
        graph_mode=v4_decode_graph,
    )
    if (
        v4_activation_preparation in ("rowwise_packed", "sign_fused", "sign_fused_strided", "sign_fused_direct")
        and not v4_device_route_decode
    ):
        raise ValueError("Packed activation preparation requires v4_device_route_decode=true.")
    if execution_policy not in ("ascendc", "ascendc_v4") and (
        ascendc_library is not None or ascendc_sha256 is not None
    ):
        raise ValueError("V1 AscendC library options require execution_policy=ascendc or ascendc_v4.")
    if execution_policy == "ascendc_v4" and root_linear_mode != "bf16":
        raise ValueError("VQ2A8 v4 requires unchanged BF16 roots.")
    if execution_policy != "ascendc_v2" and (ascendc_v2_library is not None or ascendc_v2_sha256 is not None):
        raise ValueError("V2 library options require execution_policy=ascendc_v2.")
    if execution_policy != "ascendc_v3" and (ascendc_v3_library is not None or ascendc_v3_sha256 is not None):
        raise ValueError("V3 library options require execution_policy=ascendc_v3.")
    if v3_preparation not in ("eager", "fused") or v3_decode_graph not in ("none", "moe"):
        raise ValueError("V3 requires preparation=eager|fused and decode_graph=none|moe.")
    if execution_policy != "ascendc_v3" and (v3_preparation != "eager" or v3_decode_graph != "none"):
        raise ValueError("V3 preparation/graph options require execution_policy=ascendc_v3.")
    if type(v3_serving) is not bool or (v3_serving and execution_policy != "ascendc_v3"):
        raise ValueError("v3_serving must be boolean and requires execution_policy=ascendc_v3.")
    if type(v4_serving) is not bool or (v4_serving and execution_policy != "ascendc_v4"):
        raise ValueError("v4_serving must be boolean and requires execution_policy=ascendc_v4.")
    if type(v4_device_route_decode) is not bool or (v4_device_route_decode and execution_policy != "ascendc_v4"):
        raise ValueError("v4_device_route_decode must be boolean and requires execution_policy=ascendc_v4.")
    if v4_decode_graph not in ("none", "moe", "decoder") or (
        v4_decode_graph != "none" and (execution_policy != "ascendc_v4" or not v4_device_route_decode)
    ):
        raise ValueError("v4_decode_graph requires none|moe|decoder; graphs require V4 device-route decode.")
    if v4_graph_replay_stream not in ("owner", "caller") or (
        v4_graph_replay_stream == "caller" and v4_decode_graph == "none"
    ):
        raise ValueError(
            "v4_graph_replay_stream requires owner|caller; caller requires v4_decode_graph=moe or decoder."
        )
    if v4_decode_graph == "decoder" and v4_graph_replay_stream != "caller":
        raise ValueError("V4 decoder graph requires v4_graph_replay_stream=caller.")
    if cache_memory_fraction is not None:
        _validate_cache_memory_fraction(cache_memory_fraction, execution_policy)
    if type(tensor_parallel_size) is not int or tensor_parallel_size not in (1, 2):
        raise ValueError("Offline tensor_parallel_size must be 1 or 2.")
    if tensor_parallel_size == 2 and (
        execution_policy != "ascendc_v3" or root_linear_mode != "bf16" or v3_decode_graph != "none"
    ):
        raise ValueError("TP2 requires V3, BF16 roots and decode_graph=none.")
    return {
        "model": str(model_root),
        "skip_tokenizer_init": True,
        "trust_remote_code": False,
        "hf_overrides": {
            "architectures": [f"VQ2A8TP{tensor_parallel_size}OfflineForCausalLM"],
            "quantization_config": None,
        },
        "dtype": "bfloat16",
        "load_format": "safetensors",
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": 1,
        "distributed_executor_backend": "mp" if tensor_parallel_size == 2 else "uni",
        "enforce_eager": True,
        "compilation_config": {"mode": 0, "cudagraph_mode": "NONE"},
        "async_scheduling": False,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "max_num_seqs": 1,
        "max_model_len": V4_DECODER_CONTEXT_LIMIT if v4_decode_graph == "decoder" else OFFLINE_CONTEXT_LIMIT,
        "max_num_batched_tokens": V4_DECODER_CONTEXT_LIMIT if v4_decode_graph == "decoder" else OFFLINE_CONTEXT_LIMIT,
        "block_size": 128,
        "gpu_memory_utilization": 0.9 if execution_policy in CACHE_EXECUTION_POLICIES else 0.35,
        "kv_cache_memory_bytes": 1024**3,
        "seed": 0,
        "disable_log_stats": True,
        "additional_config": {
            "enable_flashcomm1": False,
            "mix_placement": False,
            "multistream_dsv4_dsa_overlap": False,
            "vq2a8_offline": {
                "enabled": True,
                "artifact": str(artifact),
                "execution_policy": execution_policy,
                **(
                    {"v3_preparation": v3_preparation, "v3_decode_graph": v3_decode_graph}
                    if execution_policy == "ascendc_v3"
                    else {}
                ),
                **({"v3_serving": True} if v3_serving else {}),
                **({"v4_serving": True} if v4_serving else {}),
                **({"v4_compute_backend": v4_compute_backend} if v4_compute_backend != "v1" else {}),
                **({"v4_activation_reorder": v4_activation_reorder} if v4_activation_reorder != "scalar" else {}),
                **(
                    {"v4_activation_preparation": v4_activation_preparation}
                    if v4_activation_preparation != "rowwise"
                    else {}
                ),
                **({"v4_device_route_decode": True} if v4_device_route_decode else {}),
                **({"v4_validity_mode": v4_validity_mode} if v4_validity_mode != "torch" else {}),
                **({"v4_route_mapping": v4_route_mapping} if v4_route_mapping != "torch" else {}),
                **({"v4_runtime_guard": v4_runtime_guard} if v4_runtime_guard != "signature" else {}),
                **({"v4_select_sign": v4_select_sign} if v4_select_sign != "separate" else {}),
                **({"v4_activation_tail": v4_activation_tail} if v4_activation_tail != "torch" else {}),
                **({"v4_b1_schedule": v4_b1_schedule} if v4_b1_schedule != "baseline" else {}),
                **({"v4_decoder_input_mode": v4_decoder_input_mode} if v4_decoder_input_mode != "general" else {}),
                **(
                    {"v4_decoder_metadata_mode": v4_decoder_metadata_mode}
                    if v4_decoder_metadata_mode != "recursive"
                    else {}
                ),
                **({"v4_host_profile": True} if v4_host_profile else {}),
                **({"v4_decode_graph": v4_decode_graph} if v4_decode_graph != "none" else {}),
                **({"v4_graph_replay_stream": v4_graph_replay_stream} if v4_decode_graph != "none" else {}),
                "cache_experts": 256 if execution_policy in CACHE_EXECUTION_POLICIES else 2,
                "token_chunk": 2,
                "cache_budget_gib": cache_budget_gib,
                "cache_reserve_gib": cache_reserve_gib,
                **({"cache_memory_fraction": cache_memory_fraction} if cache_memory_fraction is not None else {}),
                "root_linear_mode": root_linear_mode,
                "verbose_experts": verbose_experts,
                **(
                    {"ascendc_library": str(ascendc_library), "ascendc_sha256": ascendc_sha256}
                    if execution_policy in ("ascendc", "ascendc_v4")
                    else {}
                ),
                **(
                    {"ascendc_v2_library": str(ascendc_v2_library), "ascendc_v2_sha256": ascendc_v2_sha256}
                    if execution_policy == "ascendc_v2"
                    else {}
                ),
                **(
                    {"ascendc_v3_library": str(ascendc_v3_library), "ascendc_v3_sha256": ascendc_v3_sha256}
                    if execution_policy == "ascendc_v3"
                    else {}
                ),
            },
        },
    }


def validate_offline_config(config) -> dict:
    """Reject unsupported integrations before any model-sized allocation."""
    options = (config.additional_config or {}).get("vq2a8_offline")
    if not isinstance(options, dict) or options.get("enabled") is not True:
        raise ValueError("This architecture requires explicit additional_config.vq2a8_offline.enabled=true.")
    allowed = {
        "enabled",
        "artifact",
        "cache_experts",
        "token_chunk",
        "execution_policy",
        "cache_budget_gib",
        "cache_reserve_gib",
        "cache_memory_fraction",
        "root_linear_mode",
        "ascendc_library",
        "ascendc_sha256",
        "ascendc_v2_library",
        "ascendc_v2_sha256",
        "ascendc_v3_library",
        "ascendc_v3_sha256",
        "v3_preparation",
        "v3_decode_graph",
        "v3_serving",
        "v4_serving",
        "v4_device_route_decode",
        "v4_decode_graph",
        "v4_graph_replay_stream",
        "v4_compute_backend",
        "v4_activation_reorder",
        "v4_activation_preparation",
        "v4_validity_mode",
        "v4_route_mapping",
        "v4_runtime_guard",
        "v4_select_sign",
        "v4_activation_tail",
        "v4_b1_schedule",
        "v4_decoder_metadata_mode",
        "v4_decoder_input_mode",
        "v4_host_profile",
        "v3_startup_trace",
        "verbose_experts",
    }
    if set(options) - allowed or not isinstance(options.get("artifact"), str):
        raise ValueError("Invalid vq2a8_offline options/artifact path.")
    parallel, model = config.parallel_config, config.model_config
    tp_size = getattr(parallel, "tensor_parallel_size", None)
    if type(tp_size) is not int or tp_size not in (1, 2):
        raise ValueError("Offline VQ2A8 requires tensor_parallel_size=1 or 2.")
    if tp_size == 2:
        if (
            options.get("execution_policy") != "ascendc_v3"
            or options.get("root_linear_mode", "bf16") != "bf16"
            or options.get("v3_decode_graph", "none") != "none"
        ):
            raise ValueError("TP2 requires V3, BF16 roots and decode_graph=none.")
        if getattr(parallel, "distributed_executor_backend", None) != "mp":
            raise ValueError("TP2 requires the standard multiprocessing (mp) executor.")
        extra = config.additional_config or {}
        for name in ("enable_flashcomm1", "enable_dsa_cp", "mix_placement", "multistream_dsv4_dsa_overlap"):
            if extra.get(name, False):
                raise ValueError(f"TP2 requires {name}=False.")
        finegrained = extra.get("finegrained_tp_config", {})
        if not isinstance(finegrained, dict) or any(finegrained.values()):
            raise ValueError("TP2 does not support finegrained TP overrides.")
    for name in ("pipeline_parallel_size", "data_parallel_size"):
        if getattr(parallel, name, None) != 1:
            raise ValueError(f"Offline VQ2A8 requires {name}=1.")
    for name in ("prefill_context_parallel_size", "decode_context_parallel_size"):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"Offline VQ2A8 requires {name}=1.")
    for name in ("enable_expert_parallel", "enable_eplb", "use_sequence_parallel_moe"):
        if getattr(parallel, name, False):
            raise ValueError(f"Offline VQ2A8 does not support {name}.")
    if not model.enforce_eager or config.quant_config is not None or model.quantization is not None:
        raise ValueError(
            "Offline adapter requires eager execution and canonical BF16 root allocation (no global quantizer)."
        )
    if model.dtype != torch.bfloat16 or config.scheduler_config.max_num_seqs != 1:
        raise ValueError("Offline adapter requires BF16 and max_num_seqs=1.")
    if not 1 <= model.max_model_len <= 128 or config.scheduler_config.max_num_batched_tokens > 128:
        raise ValueError("Offline bring-up is limited to at most 128 context/batched tokens.")
    if getattr(config.scheduler_config, "async_scheduling", False):
        raise ValueError("Async scheduling is not supported by this synchronous correctness baseline.")
    compilation = config.compilation_config
    if (
        getattr(compilation.mode, "value", compilation.mode) != 0
        or getattr(compilation.cudagraph_mode, "value", compilation.cudagraph_mode) != 0
    ):
        raise ValueError("Compilation and graph capture must both be disabled.")
    if any(
        getattr(config, name, None) is not None for name in ("speculative_config", "lora_config", "kv_transfer_config")
    ):
        raise ValueError("Speculation, LoRA and KV transfer are outside offline acceptance.")
    if getattr(config.cache_config, "enable_prefix_caching", False):
        raise ValueError("Prefix caching must be disabled for independent repeated prefills.")
    offload = getattr(config, "offload_config", None)
    if (
        getattr(config.cache_config, "cpu_offload_gb", 0)
        or getattr(getattr(offload, "uva", None), "cpu_offload_gb", 0)
        or getattr(getattr(offload, "prefetch", None), "offload_group_size", 0)
        or getattr(model, "enable_sleep_mode", False)
    ):
        raise ValueError("Generic offload/sleep cannot manage standalone packed-cache ownership.")
    if config.load_config.load_format != "safetensors":
        raise ValueError("Offline adapter requires the canonical safetensors loader; dummy loading is forbidden.")
    if options.get("execution_policy", "baseline") not in ("baseline", *CACHE_EXECUTION_POLICIES):
        raise ValueError("execution_policy must be baseline, cached, ascendc, ascendc_v2, ascendc_v3 or ascendc_v4.")
    if "cache_memory_fraction" in options:
        _validate_cache_memory_fraction(options["cache_memory_fraction"], options.get("execution_policy", "baseline"))
        kv_bytes = getattr(config.cache_config, "kv_cache_memory_bytes", None)
        if type(kv_bytes) is not int or kv_bytes <= 0:
            raise ValueError("cache_memory_fraction requires explicit positive integer kv_cache_memory_bytes.")
    if options.get("execution_policy") in ("ascendc", "ascendc_v4"):
        path, sha = options.get("ascendc_library"), options.get("ascendc_sha256")
        if not isinstance(path, str) or not Path(path).is_absolute() or Path(path).suffix != ".so":
            raise ValueError("AscendC requires an absolute native .so library path.")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("AscendC requires the regression-tested library SHA256.")
    elif "ascendc_library" in options or "ascendc_sha256" in options:
        raise ValueError("Native library options require explicit execution_policy=ascendc or ascendc_v4.")
    _validate_v4_compute_backend(options.get("v4_compute_backend", "v1"), options.get("execution_policy"))
    validate_candidates(
        options.get("v4_runtime_guard", "signature"),
        options.get("v4_select_sign", "separate"),
        options.get("v4_activation_tail", "torch"),
        decoder_input_mode=options.get("v4_decoder_input_mode", "general"),
        b1_schedule=options.get("v4_b1_schedule", "baseline"),
        backend=options.get("v4_compute_backend", "v1"),
        policy=options.get("execution_policy"),
        preparation=options.get("v4_activation_preparation", "rowwise"),
        reorder=options.get("v4_activation_reorder", "scalar"),
        device_route=options.get("v4_device_route_decode", False),
        graph_mode=options.get("v4_decode_graph", "none"),
    )
    _validate_v4_route_mapping(
        options.get("v4_route_mapping", "torch"),
        options.get("v4_compute_backend", "v1"),
        options.get("execution_policy"),
        options.get("v4_device_route_decode", False),
    )
    _validate_v4_validity_options(
        options.get("v4_validity_mode", "torch"),
        options.get("v4_activation_preparation", "rowwise"),
        options.get("v4_compute_backend", "v1"),
        options.get("execution_policy"),
        options.get("v4_device_route_decode", False),
    )
    _validate_v4_activation_options(
        options.get("v4_activation_reorder", "scalar"),
        options.get("v4_activation_preparation", "rowwise"),
        options.get("v4_compute_backend", "v1"),
        options.get("execution_policy"),
    )
    if options.get("execution_policy") != "ascendc_v4" and any(
        key in options
        for key in (
            "v4_activation_reorder",
            "v4_activation_preparation",
            "v4_validity_mode",
            "v4_route_mapping",
            "v4_runtime_guard",
            "v4_select_sign",
            "v4_activation_tail",
            "v4_b1_schedule",
        )
    ):
        raise ValueError("V4 activation options require execution_policy=ascendc_v4.")
    if "v4_compute_backend" in options and options.get("execution_policy") != "ascendc_v4":
        raise ValueError("v4_compute_backend requires execution_policy=ascendc_v4.")
    if options.get("execution_policy") == "ascendc_v4":
        if options.get("root_linear_mode", "bf16") != "bf16":
            raise ValueError("VQ2A8 v4 requires unchanged BF16 roots.")
        if options.get("cache_experts", 256) != 256:
            raise ValueError("VQ2A8 v4 requires full residency, not an eviction cache.")
    if options.get("execution_policy") == "ascendc_v2":
        path, sha = options.get("ascendc_v2_library"), options.get("ascendc_v2_sha256")
        if not isinstance(path, str) or not Path(path).is_absolute() or Path(path).suffix != ".so":
            raise ValueError("VQ2A8 v2 backend requires an absolute native .so library path.")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("VQ2A8 v2 backend requires the regression-tested library SHA256.")
        if options.get("root_linear_mode", "bf16") != "bf16":
            raise ValueError("VQ2A8 v2 candidate bring-up requires BF16 roots; root FP8 is outside its gate.")
    elif "ascendc_v2_library" in options or "ascendc_v2_sha256" in options:
        raise ValueError("VQ2A8 v2 library options require explicit execution_policy=ascendc_v2.")
    if options.get("execution_policy") == "ascendc_v3":
        path, sha = options.get("ascendc_v3_library"), options.get("ascendc_v3_sha256")
        if not isinstance(path, str) or not Path(path).is_absolute() or Path(path).suffix != ".so":
            raise ValueError("VQ2A8 v3 requires an absolute native .so library path.")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("VQ2A8 v3 requires the regression-tested library SHA256.")
        if options.get("root_linear_mode", "bf16") != "bf16":
            raise ValueError("VQ2A8 v3 requires unchanged BF16 roots.")
        if options.get("cache_experts", 256) != 256:
            raise ValueError("VQ2A8 v3 requires full residency, not an eviction cache.")
        if options.get("v3_preparation", "eager") not in ("eager", "fused") or options.get(
            "v3_decode_graph", "none"
        ) not in ("none", "moe"):
            raise ValueError("Invalid V3 preparation/decode graph selection.")
    elif "ascendc_v3_library" in options or "ascendc_v3_sha256" in options:
        raise ValueError("VQ2A8 v3 library options require explicit execution_policy=ascendc_v3.")
    if options.get("execution_policy") != "ascendc_v3" and any(
        key in options for key in ("v3_preparation", "v3_decode_graph")
    ):
        raise ValueError("V3 preparation/graph options require execution_policy=ascendc_v3.")
    if type(options.get("v3_serving", False)) is not bool or (
        "v3_serving" in options and options.get("execution_policy") != "ascendc_v3"
    ):
        raise ValueError("v3_serving must be boolean and requires execution_policy=ascendc_v3.")
    if type(options.get("v4_serving", False)) is not bool or (
        "v4_serving" in options and options.get("execution_policy") != "ascendc_v4"
    ):
        raise ValueError("v4_serving must be boolean and requires execution_policy=ascendc_v4.")
    if type(options.get("v4_device_route_decode", False)) is not bool or (
        "v4_device_route_decode" in options and options.get("execution_policy") != "ascendc_v4"
    ):
        raise ValueError("v4_device_route_decode must be boolean and requires execution_policy=ascendc_v4.")
    graph_mode = options.get("v4_decode_graph", "none")
    _validate_v4_host_options(
        options.get("v4_decoder_metadata_mode", "recursive"),
        options.get("v4_host_profile", False),
        graph_mode,
        options.get("execution_policy"),
    )
    if options.get("execution_policy") != "ascendc_v4" and any(
        key in options for key in ("v4_decoder_metadata_mode", "v4_decoder_input_mode", "v4_host_profile")
    ):
        raise ValueError("V4 host options require execution_policy=ascendc_v4.")
    if options.get("v4_activation_preparation") in (
        "rowwise_packed",
        "sign_fused",
        "sign_fused_strided",
        "sign_fused_direct",
    ) and not options.get("v4_device_route_decode"):
        raise ValueError("Packed activation preparation requires v4_device_route_decode=true.")
    if graph_mode not in ("none", "moe", "decoder") or (
        "v4_decode_graph" in options and options.get("execution_policy") != "ascendc_v4"
    ):
        raise ValueError("v4_decode_graph requires none|moe|decoder and execution_policy=ascendc_v4.")
    if graph_mode != "none" and (tp_size != 1 or options.get("v4_device_route_decode") is not True):
        raise ValueError("V4 MoE decode graph requires TP1 and v4_device_route_decode=true.")
    replay_stream = options.get("v4_graph_replay_stream", "owner")
    if replay_stream not in ("owner", "caller") or (replay_stream == "caller" and graph_mode == "none"):
        raise ValueError(
            "v4_graph_replay_stream requires owner|caller; caller requires v4_decode_graph=moe or decoder."
        )
    if graph_mode == "decoder" and (replay_stream != "caller" or model.max_model_len > V4_DECODER_CONTEXT_LIMIT):
        raise ValueError("V4 decoder graph requires caller replay and max_model_len <=16.")
    if graph_mode != "none":
        graph_kv_bytes = getattr(config.cache_config, "kv_cache_memory_bytes", None)
        if type(graph_kv_bytes) is not int or graph_kv_bytes <= 0:
            raise ValueError(
                "V4 MoE decode graph requires explicit positive kv_cache_memory_bytes for headroom accounting."
            )
        if graph_kv_bytes > options.get("cache_reserve_gib", 16.0) * GIB:
            raise ValueError("V4 MoE graph KV cache must fit inside cache_reserve_gib.")
    trace_mode = options.get("v3_startup_trace", "off")
    if trace_mode not in ("off", "async", "sync"):
        raise ValueError("v3_startup_trace must be off, async or sync.")
    if trace_mode != "off" and (
        tp_size != 1
        or options.get("execution_policy") != "ascendc_v3"
        or options.get("v3_serving") is not True
        or options.get("v3_decode_graph", "none") != "none"
    ):
        raise ValueError("v3_startup_trace requires TP1 V3 serving with decode_graph=none.")
    if options.get("root_linear_mode", "bf16") not in ("bf16", "online_fp8_sm90"):
        raise ValueError("root_linear_mode must be bf16 or online_fp8_sm90.")
    if type(options.get("verbose_experts", False)) is not bool:
        raise ValueError("verbose_experts must be a boolean.")
    for key, default, upper in (("cache_experts", 2, 256), ("token_chunk", 2, 8)):
        value = options.get(key, default)
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"{key} must be an integer in [1,{upper}].")
    for key, default, minimum in (("cache_budget_gib", 0.0, 0), ("cache_reserve_gib", 16.0, 1)):
        value = options.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
            raise ValueError(f"{key} must be finite and >= {minimum}.")
    if "cache_memory_fraction" in options and kv_bytes > options.get("cache_reserve_gib", 16.0) * GIB:
        raise ValueError(
            "Explicit kv_cache_memory_bytes must fit inside cache_reserve_gib when cache fraction is independent."
        )
    return options


def canonical_root_parameter(name: str) -> str | None:
    """Map only non-MoE canonical checkpoint names, without substring guessing."""
    if name.startswith("mtp.") or re.match(r"layers\.\d+\.ffn\.", name):
        return None
    if name == "head.weight":
        return "lm_head.weight"
    if name == "embed.weight":
        return "model.embed_tokens.weight"
    name = re.sub(r"^(layers\.\d+)\.attn\.", r"\1.self_attn.", name)
    name = re.sub(r"^(layers\.\d+)\.attn_norm\.", r"\1.input_layernorm.", name)
    name = re.sub(r"^(layers\.\d+)\.ffn_norm\.", r"\1.post_attention_layernorm.", name)
    return "model." + name


def audit_offline_root(model_root: Path) -> dict:
    """No implicit FP8-to-BF16 conversion: this adapter supports BF16/F32 roots."""
    inventory = {}
    for path in sorted(model_root.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118 -- safe_open is not an iterable dict.
                if name.startswith("mtp."):
                    continue
                if name in inventory:
                    raise ValueError(f"Duplicate root checkpoint tensor: {name}.")
                view = handle.get_slice(name)
                dtype = view.get_dtype()
                allowed = ("I32", "I64") if name.endswith(".gate.tid2eid") else ("BF16", "F32")
                if dtype not in allowed:
                    raise ValueError(f"Unsupported root dtype {dtype} for {name}; do not silently dequantize it.")
                inventory[name] = {"dtype": dtype, "shape": view.get_shape()}
    if not inventory:
        raise ValueError("No root checkpoint tensors found.")
    return inventory


def _validate_tp2_root_geometry(config):
    fields = ("hidden_size", "num_attention_heads", "head_dim", "o_groups", "o_lora_rank", "q_lora_rank", "vocab_size")
    if not isinstance(config, dict) or any(type(config.get(key)) is not int or config[key] <= 0 for key in fields):
        raise ValueError("TP2 roots require positive integer hidden/head/group/LoRA/vocabulary geometry.")
    if (
        config["num_attention_heads"] % 2
        or config["o_groups"] % 2
        or config["num_attention_heads"] * config["head_dim"] % config["o_groups"]
    ):
        raise ValueError("TP2 root attention heads/groups must divide evenly into two valid output shards.")


class OfflineMoEOwner:
    """One rank-bound artifact index; cached or resident experts share a byte budget."""

    def __init__(self, model_root: Path, options: dict, device: torch.device, *, tp_size=1, tp_group=None):
        if type(tp_size) is not int or tp_size not in (1, 2):
            raise ValueError("Offline owner requires TP size 1 or 2.")
        self.tp_size, self.tp_group, self.tp_rank = tp_size, tp_group, 0
        if tp_size == 2:
            if (
                options.get("execution_policy") != "ascendc_v3"
                or options.get("root_linear_mode", "bf16") != "bf16"
                or options.get("v3_decode_graph", "none") != "none"
                or type(getattr(tp_group, "world_size", None)) is not int
                or getattr(tp_group, "world_size", None) != 2
                or type(getattr(tp_group, "rank_in_group", None)) is not int
                or tp_group.rank_in_group not in (0, 1)
                or not callable(getattr(tp_group, "all_reduce", None))
            ):
                raise ValueError("TP2 owner requires V3/BF16/no graph and a matching two-rank TP group.")
            self.tp_rank = tp_group.rank_in_group
            self.root_config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
            _validate_tp2_root_geometry(self.root_config)
        self.native_library = None
        if options.get("execution_policy") in ("ascendc", "ascendc_v4"):
            if device.type != "npu":
                raise ValueError("AscendC offline execution requires an NPU, without fallback.")
            backend = options.get("v4_compute_backend", "v1")
            _validate_v4_compute_backend(backend, options.get("execution_policy"))
            if backend == "v2":
                from vllm_ascend.quantization.vq2a8_v4_v2 import load_v4_v2_library, require_v4_v2_features

                self.native_library = load_v4_v2_library(options["ascendc_library"], options["ascendc_sha256"])
                require_v4_v2_features(
                    options.get("v4_activation_reorder", "scalar"),
                    options.get("v4_activation_preparation", "rowwise"),
                    validity_mode=options.get("v4_validity_mode", "torch"),
                    route_mapping=options.get("v4_route_mapping", "torch"),
                    select_sign=options.get("v4_select_sign", "separate"),
                    activation_tail=options.get("v4_activation_tail", "torch"),
                    **(
                        {"b1_schedule": options["v4_b1_schedule"]}
                        if options.get("v4_b1_schedule", "baseline") != "baseline"
                        else {}
                    ),
                    runtime_guard=options.get("v4_runtime_guard", "signature"),
                    decoder_input_mode=options.get("v4_decoder_input_mode", "general"),
                )
            else:
                from vllm_ascend.quantization.vq2a8_ascendc import load_pinned_library

                self.native_library = load_pinned_library(options["ascendc_library"], options["ascendc_sha256"])
                if options.get("v4_device_route_decode", False):
                    from vllm_ascend.quantization.vq2a8_v4_device_route import require_device_route_library

                    require_device_route_library()
        elif options.get("execution_policy") == "ascendc_v2":
            from vllm_ascend.quantization.vq2a8_ascendc_v2 import load_pinned_library

            if device.type != "npu":
                raise ValueError("VQ2A8 v2 offline execution requires an NPU, without fallback.")
            self.native_library = load_pinned_library(options["ascendc_v2_library"], options["ascendc_v2_sha256"])
        elif options.get("execution_policy") == "ascendc_v3":
            from vllm_ascend.quantization.vq2a8_ascendc_v3 import load_pinned_library, resident_library_capabilities

            if device.type != "npu":
                raise ValueError("VQ2A8 v3 offline execution requires an NPU, without fallback.")
            self.native_library = load_pinned_library(options["ascendc_v3_library"], options["ascendc_v3_sha256"])
            if tp_size == 2:
                resident_library_capabilities(require_tp2=True)
        if tp_size == 2:
            from vllm.platforms import current_platform

            from vllm_ascend.quantization.vq2a8_tp2_runtime import open_vq2a8_tp2_artifact

            logical_device = device.index
            physical_device = current_platform.device_id_to_physical_device_id(logical_device)
            print(
                f"MODEL stage=tp2_owner tp_rank={self.tp_rank} tp_size=2 "
                f"logical_device={logical_device} physical_device={physical_device}",
                flush=True,
            )
            self.artifact = open_vq2a8_tp2_artifact(
                Path(options["artifact"]), model_root / "config.json", tp_rank=self.tp_rank, verify_tensor_hashes=True
            )
        else:
            format_name = artifact_format(options["artifact"])
            if format_name == V4_V2_PREPACKED_FORMAT:
                if options.get("execution_policy") != "ascendc_v4" or options.get("v4_compute_backend", "v1") != "v2":
                    raise ValueError(
                        "Prepacked V4 v2 experts require execution_policy=ascendc_v4 and v4_compute_backend=v2."
                    )
                from vllm_ascend.quantization.vq2a8_v4_v2_prepacked import open_vq2a8_v4_v2_prepacked_artifact

                self.artifact = open_vq2a8_v4_v2_prepacked_artifact(options["artifact"], model_root / "config.json")
            elif format_name == VQ2_TP1_ZN_FORMAT:
                if options.get("execution_policy") != "ascendc_v3":
                    raise ValueError("TP1 packed-zN requires execution_policy=ascendc_v3; no format fallback.")
                self.artifact = open_vq2a8_tp1_zn_artifact(
                    options["artifact"], model_root / "config.json", verify_tensor_hashes=True
                )
            elif format_name == VQ2_DIRECT_TP1_FORMAT:
                self.artifact = open_vq2a8_tp1_artifact(
                    Path(options["artifact"]),
                    model_root / "config.json",
                    require_complete=True,
                    require_reference_identity=True,
                )
            else:
                raise ValueError(f"Unsupported artifact format for TP1: {format_name!r}; no format fallback.")
        self.inventory = audit_offline_root(model_root)
        if options.get("execution_policy") == "ascendc_v4" and options.get("v4_compute_backend", "v1") == "v2":
            print(
                "MODEL_V4_V2_EXPERT_SOURCE "
                + json.dumps(
                    {
                        "artifact": str(self.artifact.root),
                        "format": self.artifact.manifest.get("format"),
                        "startup_conversion": self.artifact.manifest.get("format") != V4_V2_PREPACKED_FORMAT,
                    }
                ),
                flush=True,
            )
        self.options = options
        self.device = device
        self.layers: dict[int, VQ2TP1MoE] = {}
        self.calls: dict[int, int] = {}
        self.cache_plan = None

    def create_layer(self, index: int) -> VQ2TP1MoE:
        if index in self.layers:
            raise ValueError(f"Duplicate offline MoE layer {index}.")
        print(f"MODEL layer={index} stage=moe_root_load_start", flush=True)
        # vLLM constructs under a default NPU device. Keep CPU validation
        # factories on CPU; the tested runtime moves its roots explicitly.
        with torch.device("cpu"):
            runtime_classes = {
                "baseline": VQ2TP1MoE,
                "cached": CachedVQ2TP1MoE,
                "ascendc": AscendCVQ2TP1MoE,
            }
            if self.options.get("execution_policy") == "ascendc_v2":
                from vllm_ascend.quantization.vq2a8_ascendc_v2 import AscendCV2VQ2TP1MoE

                runtime_classes["ascendc_v2"] = AscendCV2VQ2TP1MoE
            if self.options.get("execution_policy") == "ascendc_v4":
                from vllm_ascend.quantization.vq2a8_execution_v4 import AscendCV4VQ2TP1MoE

                runtime_classes["ascendc_v4"] = AscendCV4VQ2TP1MoE
                if self.options.get("v4_compute_backend", "v1") == "v2":
                    from vllm_ascend.quantization.vq2a8_v4_v2 import AscendCV4V2VQ2TP1MoE

                    runtime_classes["ascendc_v4"] = AscendCV4V2VQ2TP1MoE
            if self.options.get("execution_policy") == "ascendc_v3":
                if getattr(self, "tp_size", 1) == 2:
                    from vllm_ascend.quantization.vq2a8_execution_tp2 import AscendCV3VQ2TP2MoE

                    runtime_classes["ascendc_v3"] = AscendCV3VQ2TP2MoE
                elif getattr(self.artifact, "manifest", {}).get("format") == VQ2_TP1_ZN_FORMAT:
                    from vllm_ascend.quantization.vq2a8_execution_tp1_zn import AscendCV3VQ2TP1ZNMoE

                    runtime_classes["ascendc_v3"] = AscendCV3VQ2TP1ZNMoE
                else:
                    from vllm_ascend.quantization.vq2a8_execution_v3 import AscendCV3VQ2TP1MoE

                    runtime_classes["ascendc_v3"] = AscendCV3VQ2TP1MoE
            runtime_class = runtime_classes[self.options.get("execution_policy", "baseline")]
            layer = runtime_class(
                self.artifact,
                index,
                self.device,
                cache_experts=self.options.get("cache_experts", 2),
                token_chunk=self.options.get("token_chunk", 2),
                **(
                    {
                        "v4_activation_reorder": self.options.get("v4_activation_reorder", "scalar"),
                        "v4_activation_preparation": self.options.get("v4_activation_preparation", "rowwise"),
                        "v4_validity_mode": self.options.get("v4_validity_mode", "torch"),
                        "v4_route_mapping": self.options.get("v4_route_mapping", "torch"),
                        "v4_runtime_guard": self.options.get("v4_runtime_guard", "signature"),
                        "v4_select_sign": self.options.get("v4_select_sign", "separate"),
                        "v4_activation_tail": self.options.get("v4_activation_tail", "torch"),
                        **(
                            {"v4_b1_schedule": self.options["v4_b1_schedule"]}
                            if self.options.get("v4_b1_schedule", "baseline") != "baseline"
                            else {}
                        ),
                    }
                    if self.options.get("execution_policy") == "ascendc_v4"
                    and self.options.get("v4_compute_backend", "v1") == "v2"
                    else {}
                ),
                **(
                    {"tp_rank": self.tp_rank, "tp_group": self.tp_group, "projection_kernel": "v2"}
                    if getattr(self, "tp_size", 1) == 2
                    else {}
                ),
                **(
                    {
                        "v3_preparation": self.options.get("v3_preparation", "eager"),
                        "v3_decode_graph": self.options.get("v3_decode_graph", "none"),
                    }
                    if self.options.get("execution_policy") == "ascendc_v3"
                    else {}
                ),
                **(
                    {"progress": True, "verbose_experts": self.options.get("verbose_experts", False)}
                    if issubclass(runtime_class, CachedVQ2TP1MoE)
                    else {}
                ),
            )
        self.layers[index] = layer
        self.calls[index] = 0
        print(f"MODEL layer={index} stage=moe_root_load_done", flush=True)
        return layer

    def configure_cache(self, memory_fraction: float) -> None:
        """Call after strict root load, before profiling populates any cache."""
        policy = self.options.get("execution_policy")
        override = self.options.get("cache_memory_fraction")
        if override is not None:
            _validate_cache_memory_fraction(override, policy)
        if policy not in CACHE_EXECUTION_POLICIES:
            return
        if any(layer.cache_stats()["resident_experts"] for layer in self.layers.values()):
            raise ValueError("Configure the packed cache before the first expert call.")
        cache_memory_fraction = memory_fraction if override is None else override
        requested_gib = self.options.get("cache_budget_gib", 0.0)
        if type(requested_gib) not in (int, float) or not math.isfinite(requested_gib) or requested_gib < 0:
            raise ValueError("cache_budget_gib must be finite and non-negative.")
        # Sample the original safe-budget function once, before applying an
        # explicit byte cap, so even a rejected cap has useful memory evidence.
        budget = device_cache_budget(
            self.device,
            reserve_gib=self.options.get("cache_reserve_gib", 16.0),
            budget_gib=0.0,
            memory_fraction=cache_memory_fraction,
        )
        available = budget["budget_bytes"]
        requested = int(requested_gib * GIB)
        free, allocated, reserved = (
            budget.get("free_bytes"),
            budget.get("allocated_bytes_at_plan"),
            budget.get("reserved_bytes_at_plan"),
        )
        reusable = (
            free + max(0, reserved - allocated)
            if all(value is not None for value in (free, allocated, reserved))
            else None
        )
        evidence = {
            "scope": "requested_cap_within_safe_budget_not_model_residency",
            "execution_policy": policy,
            "engine_memory_fraction": memory_fraction,
            "cache_memory_fraction": cache_memory_fraction,
            "independent_cache_fraction": override is not None,
            "free_bytes": free,
            "reusable_bytes": reusable,
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "reserve_bytes": budget.get("reserve_bytes"),
            "available_bytes": available,
            "requested_bytes": requested,
            "budget_bytes": requested or available,
            "fits": requested <= available,
        }
        evidence.update(
            {
                key.replace("_bytes", "_gib"): value / GIB if value is not None else None
                for key, value in list(evidence.items())
                if key.endswith("_bytes")
            }
        )
        print("MODEL_CACHE_BUDGET " + json.dumps(evidence), flush=True)
        if requested > available:
            raise ValueError(f"Requested packed cache {requested} exceeds safe current budget {available} bytes.")
        budget = {
            **budget,
            "budget_bytes": requested or available,
            "engine_memory_fraction": memory_fraction,
            "cache_memory_fraction": cache_memory_fraction,
        }
        if self.options.get("execution_policy") == "ascendc_v3":
            self._configure_v3_residency(budget)
            return
        if policy == "ascendc_v4":
            self._configure_v4_residency(budget)
            return
        planner = packed_cache_plan
        if self.options.get("execution_policy") == "ascendc_v2":
            from vllm_ascend.quantization.vq2a8_ascendc_v2 import ascendc_v2_cache_plan

            planner = ascendc_v2_cache_plan
        plan = planner(
            [layer.layer for layer in self.layers.values()],
            budget["budget_bytes"],
            expert_limit=self.options.get("cache_experts", 256),
        )
        for index, layer in self.layers.items():
            layer.cache_experts = plan["layer_limits"][index]
        self.cache_plan = {**budget, **plan}
        print("MODEL_CACHE_PLAN " + json.dumps(self.cache_plan), flush=True)

    def _configure_v4_residency(self, budget):
        """Preload one selected packed layout before worker profiling."""
        from vllm_ascend.quantization.vq2a8_execution_v4 import packed_resident_plan

        if not self.layers or set(self.layers) != set(self.artifact.layers):
            raise ValueError("V4 requires all artifact layers before full-residency planning.")
        planner = packed_resident_plan
        if self.options.get("v4_compute_backend", "v1") == "v2":
            from vllm_ascend.quantization.vq2a8_v4_v2 import v4_v2_resident_plan

            planner = v4_v2_resident_plan
        plan = planner([layer.layer for layer in self.layers.values()], budget["budget_bytes"])
        self.cache_plan = {**budget, **plan, "preload_complete": False}
        print("MODEL_CACHE_PLAN " + json.dumps(self.cache_plan), flush=True)
        started = time.perf_counter()
        try:
            for index, layer in self.layers.items():
                print(f"MODEL layer={index} stage=v4_resident_load_start", flush=True)
                layer.initialize_resident(budget_bytes=plan["layer_plans"][index]["planned_bytes"])
                print(f"MODEL layer={index} stage=v4_resident_load_done", flush=True)
                if self.options.get("v4_compute_backend", "v1") == "v2":
                    report = layer.v4_report()
                    print(
                        "MODEL_V4_V2_LOAD "
                        + json.dumps(
                            {
                                key: report[key]
                                for key in (
                                    "layer_index",
                                    "source_format",
                                    "startup_conversion",
                                    "preload_host_convert_s",
                                    "preload_host_read_s",
                                    "preload_host_validate_s",
                                    "preload_h2d_s",
                                )
                            }
                        ),
                        flush=True,
                    )
            for layer in self.layers.values():
                layer.check_resident_integrity()
            if self.options.get("v4_device_route_decode", False):
                from vllm_ascend.quantization.vq2a8_v4_device_route import initialize_device_route_banks

                for layer in self.layers.values():
                    initialize_device_route_banks(layer)
                print(
                    "MODEL_V4_DEVICE_ROUTE_READY "
                    + json.dumps(
                        {
                            "layers": len(self.layers),
                            "metadata_bytes": sum(
                                layer._device_route_banks["metadata_bytes"] for layer in self.layers.values()
                            ),
                            "weight_payload_copied": False,
                            "singleton_host_route_reads": 0,
                            "multi_token_prefill": "batched",
                        }
                    ),
                    flush=True,
                )
        except BaseException:
            # Never execute a partially resident model. Each layer owns the
            # completion fence needed before its allocations can be released.
            for layer in self.layers.values():
                try:
                    layer.abort_residency()
                except Exception as error:
                    print(f"MODEL_V4_CLEANUP_ERROR {error}", flush=True)
            raise
        self.cache_plan["preload_complete"] = True
        self.cache_plan["preload_elapsed_s"] = time.perf_counter() - started
        print(
            "MODEL_V4_RESIDENT_READY "
            + json.dumps(
                {
                    "layers": len(self.layers),
                    "experts": sum(len(layer.layer.expert_ids) for layer in self.layers.values()),
                    "planned_bytes": plan["planned_bytes"],
                    "preload_elapsed_s": self.cache_plan["preload_elapsed_s"],
                    "expert_payload_runtime_loading": False,
                    "native_abi": "v4_v2_abi1" if self.options.get("v4_compute_backend", "v1") == "v2" else "v1",
                }
            ),
            flush=True,
        )

    def _configure_v3_residency(self, budget):
        """Plan ALL layers before admitting the first immutable expert bank.

        Unlike the legacy lazy cache, v3 accounts for fixed decode workspaces
        inside its budget. The explicit reserve still covers subsequent KV
        allocation, transient preparation/attention tensors and fragmentation.
        No automatic reserve reduction or lower-precision root conversion.
        """
        from vllm_ascend.quantization.vq2a8_execution_v3 import resident_plan

        plan = resident_plan(
            [layer.layer for layer in self.layers.values()], budget["budget_bytes"], report_budget=True
        )
        self.cache_plan = {**budget, **plan}
        print("MODEL_CACHE_PLAN " + json.dumps(self.cache_plan), flush=True)
        for index, layer in self.layers.items():
            print(f"MODEL layer={index} stage=v3_resident_load_start", flush=True)
            layer.initialize_resident(budget_bytes=plan["layer_plans"][index]["planned_bytes"])
            print(f"MODEL layer={index} stage=v3_resident_load_done", flush=True)

    def delegated_names(self) -> set[str]:
        return {f"layers.{index}.ffn.{name}" for index, layer in self.layers.items() for name in layer.root}

    def _load_tp2_root(self, name, param, value, loader):
        """Keep full checkpoint tensors; trusted vLLM loaders own sharding/padding.

        Only the standard DeepSeek-V4 BF16 root families below may be sharded.
        All other roots (including HC/RMS/indexer) retain exact replicated shapes.
        Shared experts are delegated to the runtime and remain replicated.
        """
        cfg = self.root_config
        h, heads, dim = cfg["hidden_size"], cfg["num_attention_heads"], cfg["head_dim"]
        groups, rank = cfg["o_groups"], cfg["o_lora_rank"]
        if heads % 2 or groups % 2:
            raise ValueError("TP2 requires attention heads and output groups divisible by two.")
        suffix = re.sub(r"^layers\.\d+\.attn\.", "", name)
        shapes = {
            "wq_b.weight": (heads * dim, cfg["q_lora_rank"]),
            "wo_a.weight": (groups * rank, heads * dim // groups),
            "wo_b.weight": (h, groups * rank),
            "attn_sink": (heads,),
        }
        expected = shapes.get(suffix) if suffix != name else None
        if name in ("embed.weight", "head.weight"):
            expected = (cfg["vocab_size"], h)
        if expected is None:
            if tuple(value.shape) != tuple(param.shape):
                raise ValueError(f"Replicated TP2 root shape mismatch: {name}.")
            getattr(param, "weight_loader", loader)(param, value)
            return
        if tuple(value.shape) != expected:
            raise ValueError(f"Canonical TP2 root shape mismatch: {name}: {tuple(value.shape)} != {expected}.")
        if suffix == "attn_sink":
            local = value.narrow(0, self.tp_rank * (heads // 2), heads // 2)
            if tuple(local.shape) != tuple(param.shape):
                raise ValueError("TP2 attention sink allocation does not match local heads.")
            loader(param, local)
            return
        weight_loader = getattr(param, "weight_loader", None)
        if not callable(weight_loader):
            raise ValueError(f"TP2 root requires its parameter-specific weight loader: {name}.")
        weight_loader(param, value)

    def load_root(self, parameters: dict, weights, loader) -> tuple[set[str], dict]:
        expected_delegated = self.delegated_names()
        seen, loaded, delegated, skipped = set(), set(), set(), set()
        widened = []
        print(f"MODEL stage=root_weight_load_start expected_non_mtp={len(self.inventory)}", flush=True)
        for name, value in weights:
            if name in seen:
                raise ValueError(f"Duplicate streamed checkpoint tensor {name}.")
            seen.add(name)
            if len(seen) == 1 or len(seen) % 64 == 0:
                print(f"MODEL stage=root_weight_load tensors_seen={len(seen)} current={name}", flush=True)
            if name.startswith("mtp."):
                skipped.add(name)
                continue
            if name not in self.inventory:
                raise ValueError(f"Checkpoint tensor was not present in the audited inventory: {name}.")
            if name in expected_delegated:
                delegated.add(name)
                continue
            target = canonical_root_parameter(name)
            if target is None or target not in parameters or target in loaded:
                raise ValueError(f"Unmapped or repeated root tensor {name} -> {target}.")
            param = parameters[target]
            widen_norm = (
                re.fullmatch(r"layers\.\d+\.attn\.(?:indexer\.)?compressor\.norm\.weight", name)
                and value.dtype == torch.bfloat16
                and param.dtype == torch.float32
            )
            tp2 = getattr(self, "tp_size", 1) == 2
            if (not tp2 and tuple(value.shape) != tuple(param.shape)) or (
                value.dtype != param.dtype and not widen_norm
            ):
                raise ValueError(
                    f"Root shape/dtype mismatch: {name} -> {target} ({value.shape}, {value.dtype}) "
                    f"!= ({param.shape}, {param.dtype})."
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Non-finite root tensor {name}.")
            if tp2:
                self._load_tp2_root(name, param, value, loader)
            else:
                loader(param, value)
            if widen_norm:
                widened.append(name)
            loaded.add(target)
        if loaded != set(parameters) or delegated != expected_delegated or seen - skipped != set(self.inventory):
            raise ValueError(
                f"Incomplete root load: parameters={sorted(set(parameters) - loaded)}, "
                f"delegated={sorted(expected_delegated - delegated)}, "
                f"source={sorted(set(self.inventory) - (seen - skipped))}."
            )
        print(f"MODEL stage=root_weight_load_done registered={len(loaded)} delegated={len(delegated)}", flush=True)
        return loaded, {
            "registered_parameters_loaded": len(loaded),
            "moe_root_tensors_loaded": len(delegated),
            "mtp_tensors_skipped": len(skipped),
            "moe_layers": len(self.layers),
            "a5_compressor_norm_bf16_to_fp32": widened,
        }

    def cache_report(self) -> dict:
        stats = [layer.cache_stats() for layer in self.layers.values()]
        return {
            "resident_packed_bytes": sum(item["resident_bytes"] for item in stats),
            "resident_experts": sum(item["resident_experts"] for item in stats),
            "per_layer_cache_limit": max((layer.cache_experts for layer in self.layers.values()), default=0),
            "layer_calls": dict(self.calls),
            "loads": sum(item["loads"] for item in stats),
            "hits": sum(item["hits"] for item in stats),
            "evictions": sum(item.get("evictions", 0) for item in stats),
            "plan": self.cache_plan,
            **({"v4": self.v4_report()} if self.options.get("execution_policy") == "ascendc_v4" else {}),
        }

    def v4_report(self) -> dict:
        return {
            str(index): layer.v4_report()
            for index, layer in self.layers.items()
            if getattr(layer, "execution_policy", None) == "ascendc_v4"
        }

    def reset_backend_trace(self):
        for layer in self.layers.values():
            if isinstance(layer, AscendCVQ2TP1MoE):
                layer.reset_native_trace()

    def backend_report(self) -> dict:
        return {
            "policy": self.options.get("execution_policy", "baseline"),
            "library": self.native_library,
            "fallback_enabled": False,
            "layers": [
                {"layer": index, "steps": list(layer.native_steps)}
                for index, layer in self.layers.items()
                if isinstance(layer, AscendCVQ2TP1MoE)
            ],
        }


def validate_v4_residency_evidence(cache: dict, reports: dict, layers: int) -> dict:
    """Check NPU residency evidence outside timed work, not package versions.

    The plan describes rounded allocations; payload counts describe logical
    tensor bytes. Neither includes roots, KV, temporary tensors or headroom.
    This checks reported real execution, not an independent weight oracle.
    """
    if type(layers) is not int or layers < 1 or not isinstance(cache, dict) or not isinstance(reports, dict):
        raise ValueError("V4 residency requires a complete layer report.")
    plan = cache.get("plan")
    if (
        not isinstance(plan, dict)
        or plan.get("preload_complete") is not True
        or plan.get("all_experts_fit") is not True
        or plan.get("allocation") != "eager_packed_only"
        or plan.get("layout") != "v1_packed"
    ):
        raise ValueError("V4 full-residency preload was not completed.")
    layer_plans = plan.get("layer_plans")
    if not isinstance(layer_plans, dict):
        raise ValueError("V4 requires a full per-layer residency plan.")
    expected_keys = {str(index) for index in range(layers)}
    records = {str(index): record for index, record in reports.items()}
    planned = {str(index): record for index, record in layer_plans.items()}
    if (
        len(records) != len(reports)
        or len(planned) != len(layer_plans)
        or set(records) != expected_keys
        or set(planned) != expected_keys
    ):
        raise ValueError("V4 residency layer coverage is incomplete or duplicated.")
    experts = payload_bytes = allocation_bytes = 0
    for key in sorted(expected_keys, key=int):
        record, entry = records[key], planned[key]
        if not isinstance(record, dict) or not isinstance(entry, dict):
            raise ValueError(f"Invalid V4 layer {key} residency record.")
        if (
            record.get("ready") is not True
            or record.get("failed") is not False
            or record.get("fallback_enabled") is not False
            or record.get("execution_policy") != "ascendc_v4"
            or record.get("layout") != "v1_packed"
            or type(record.get("layer_index")) is not int
            or record["layer_index"] != int(key)
        ):
            raise ValueError(f"V4 layer {key} is not a ready, no-fallback V1-packed runtime.")
        for name in ("experts", "payload_bytes", "planned_bytes"):
            if type(entry.get(name)) is not int or entry[name] <= 0:
                raise ValueError(f"Invalid V4 layer {key} planned {name}.")
        if entry["payload_bytes"] > entry["planned_bytes"]:
            raise ValueError(f"V4 layer {key} payload exceeds its allocation plan.")
        expected = {
            "expected_experts": entry["experts"],
            "resident_experts": entry["experts"],
            "preload_loads": entry["experts"],
            "payload_bytes": entry["payload_bytes"],
            "planned_bytes": entry["planned_bytes"],
            "preload_h2d_bytes": entry["payload_bytes"],
            "preload_evictions": 0,
            "post_init_loads": 0,
            "post_init_h2d_bytes": 0,
            "post_init_evictions": 0,
        }
        if any(type(record.get(name)) is not int or record[name] != value for name, value in expected.items()):
            raise ValueError(f"V4 layer {key} residency changed or expert payload was loaded/evicted at runtime.")
        experts += entry["experts"]
        payload_bytes += entry["payload_bytes"]
        allocation_bytes += entry["planned_bytes"]
    expected_cache = {
        "resident_experts": experts,
        "loads": experts,
        "evictions": 0,
        "resident_packed_bytes": payload_bytes,
    }
    if (
        any(type(cache.get(name)) is not int or cache[name] != value for name, value in expected_cache.items())
        or type(plan.get("planned_bytes")) is not int
        or plan["planned_bytes"] != allocation_bytes
        or type(plan.get("budget_bytes")) is not int
        or allocation_bytes > plan["budget_bytes"]
    ):
        raise ValueError("V4 aggregate residency counters or byte budget do not match its complete plan.")
    return {
        "layers": layers,
        "experts": experts,
        "payload_bytes": payload_bytes,
        "planned_bytes": allocation_bytes,
        "preload_complete": True,
        "runtime_expert_payload_h2d_bytes": 0,
        "runtime_expert_evictions": 0,
        "scope": "expert_payload_residency_only_not_all_device_transfers_or_latency",
    }


def validate_offline_evidence(
    evidence: dict,
    prompt: list[int],
    generated: list[int],
    layers: int,
    vocab: int,
    *,
    execution_policy=None,
    ascendc_sha256=None,
    ascendc_v2_sha256=None,
    ascendc_v3_sha256=None,
) -> dict:
    """Require real prefill followed by decode, all layers, and sampler/logit agreement."""
    if len(prompt) < 2 or len(generated) != OFFLINE_NEW_TOKENS:
        raise ValueError("Gate requires a multi-token prefill and four generated tokens.")
    if any(type(token) is not int or not 0 <= token < vocab for token in prompt + generated):
        raise ValueError("Out-of-vocabulary or invalid token IDs.")
    expected_steps = [{"tokens": len(prompt), "positions": list(range(len(prompt)))}]
    expected_steps.extend({"tokens": 1, "positions": [len(prompt) + index]} for index in range(len(generated) - 1))
    if evidence["steps"] != expected_steps:
        raise ValueError(f"Unexpected prefill/decode positions or padded batch: {evidence['steps']}.")
    backend = evidence.get("expert_backend", {})
    if execution_policy is not None and backend.get("policy") != execution_policy:
        raise ValueError("Requested expert backend did not execute on the model.")
    if execution_policy in ("ascendc", "ascendc_v2", "ascendc_v3", "ascendc_v4"):
        expected_sha256 = {
            "ascendc": ascendc_sha256,
            "ascendc_v2": ascendc_v2_sha256,
            "ascendc_v3": ascendc_v3_sha256,
            "ascendc_v4": ascendc_sha256,
        }[execution_policy]
        records = backend.get("layers", [])
        if (
            not isinstance(expected_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            or backend.get("library", {}).get("sha256") != expected_sha256
            or backend.get("fallback_enabled") is not False
            or len(records) != layers
            or {r["layer"] for r in records} != set(range(layers))
        ):
            raise ValueError("AscendC model library/layer coverage mismatch.")
        for record in records:
            steps = record.get("steps", [])
            if len(steps) != len(expected_steps):
                raise ValueError("AscendC profile calls cannot count as generation coverage.")
            for step, expected in zip(steps, expected_steps):
                if (
                    any(
                        type(step.get(k)) is not int
                        for k in ("tokens", "projection_calls", "projection_rows", "expert_calls", "kernel_launches")
                    )
                    or step["tokens"] != expected["tokens"]
                    or step["expert_calls"] < 1
                    or step["projection_calls"] != 2 * step["expert_calls"]
                    or step["projection_rows"] < 2 * step["tokens"]
                    or step["projection_rows"] % 2
                    or not 2 <= step["kernel_launches"] <= step["projection_calls"]
                    or step["projection_calls"] > 6 * step["kernel_launches"]
                    or step["kernel_launches"] % 2
                ):
                    raise ValueError("AscendC gate/up and down coverage is incomplete for a real model step.")
    calls = {int(index): count for index, count in evidence["cache"]["layer_calls"].items()}
    if calls != {index: len(expected_steps) for index in range(layers)}:
        raise ValueError(f"Not all {layers} MoE layers executed once per model step: {calls}.")
    if evidence["load"].get("moe_layers") != layers or evidence["load"].get("registered_parameters_loaded", 0) <= 0:
        raise ValueError("Strict full-model loading was not evidenced.")
    if evidence["cache"]["resident_experts"] > layers * evidence["cache"]["per_layer_cache_limit"]:
        raise ValueError("Packed expert cache exceeded its declared bound.")
    plan = evidence["cache"].get("plan")
    if plan and evidence["cache"]["resident_packed_bytes"] > plan["planned_bytes"]:
        raise ValueError("Packed expert cache exceeded its planned byte budget.")
    if execution_policy == "ascendc_v4":
        validate_v4_residency_evidence(evidence["cache"], evidence.get("v4", {}), layers)
    logits = evidence["logits"]
    if logits.shape != (len(generated), vocab) or not bool(torch.isfinite(logits).all()):
        raise ValueError("Invalid shape or non-finite generation logits.")
    # Tied greedy maxima may select either tied token; require the sampled
    # logit to be exactly maximal instead of assuming argmax tie breaking.
    selected = logits[torch.arange(len(generated)), torch.tensor(generated)]
    if not torch.equal(selected, logits.max(-1).values):
        raise ValueError("Sampled tokens disagree with greedy captured logits.")
    root = evidence.get("root_fp8", {"mode": "bf16"})
    if root.get("mode") not in ("bf16", "online_fp8_sm90"):
        raise ValueError("Unknown root linear evidence mode.")
    if root.get("mode") == "online_fp8_sm90":
        expected = {
            f"model.layers.{i}.self_attn.{name}"
            for i in range(layers)
            for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")
        }
        records = root.get("layers", [])
        by_name = {r["name"]: r for r in records}
        if (
            len(by_name) != len(records)
            or not root.get("all_processed")
            or not root.get("native_fp8_root_matmul")
            or any(
                not r.get("processed")
                or r.get("weight_dtype") != "torch.float8_e4m3fn"
                or r.get("scale_dtype") != "torch.float32"
                for r in records
            )
            or not expected.issubset(by_name)
            or any(by_name[name].get("calls") != len(expected_steps) for name in expected)
        ):
            raise ValueError("Root FP8 projection coverage/calls incomplete; profile calls cannot count.")
    return {
        "prefill_tokens": len(prompt),
        "steps": evidence["steps"],
        "decode_steps": len(generated) - 1,
        "generated_token_ids": generated,
        "layers_executed": layers,
        "finite_logits": True,
        "greedy_logits_agree": True,
        "cache": evidence["cache"],
        "load": evidence["load"],
        "peak_allocated_bytes": evidence["peak_allocated_bytes"],
        "peak_reserved_bytes": evidence["peak_reserved_bytes"],
        "root_fp8": root,
        "expert_backend": backend,
    }
