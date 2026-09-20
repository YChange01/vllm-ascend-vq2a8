# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict model integration and full-residency ownership for TP1 VQ2A8."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import regex as re
import torch
from safetensors import safe_open

from vllm_ascend.quantization.vq2a8_config import resolve_runtime_options
from vllm_ascend.quantization.vq2a8_execution import GIB, device_cache_budget
from vllm_ascend.quantization.vq2a8_moe import VQ2TP1MoE
from vllm_ascend.quantization.vq2a8_repack import VQ2_DIRECT_TP1_FORMAT
from vllm_ascend.quantization.vq2a8_runtime import open_vq2a8_tp1_artifact
from vllm_ascend.quantization.vq2a8_v4_v2_prepacked import (
    V4_V2_PREPACKED_FORMAT,
    open_vq2a8_v4_v2_prepacked_artifact,
)

V4_DECODER_CONTEXT_LIMIT = 16


def _validate_cache_memory_fraction(value, policy="ascendc_v4"):
    if policy != "ascendc_v4":
        raise ValueError("Only the resident V4/v2 runtime is supported.")
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("cache_memory_fraction must be a finite number in (0,1], not a boolean.")


def offline_engine_options(model_root: Path, artifact: Path, **options) -> dict:
    """Build the bounded single-request configuration, retaining accepted maths."""
    runtime = resolve_runtime_options({"enabled": True, "artifact": str(artifact), **options})
    return {
        "model": str(model_root),
        "skip_tokenizer_init": True,
        "trust_remote_code": False,
        "hf_overrides": {
            "architectures": ["VQ2A8TP1OfflineForCausalLM"],
            "quantization_config": None,
        },
        "dtype": "bfloat16",
        "load_format": "safetensors",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "distributed_executor_backend": "uni",
        "enforce_eager": True,
        "compilation_config": {"mode": 0, "cudagraph_mode": "NONE"},
        "async_scheduling": False,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "max_num_seqs": 1,
        "max_model_len": V4_DECODER_CONTEXT_LIMIT,
        "max_num_batched_tokens": V4_DECODER_CONTEXT_LIMIT,
        "block_size": 128,
        "gpu_memory_utilization": 0.9,
        "kv_cache_memory_bytes": 1024**3,
        "seed": 0,
        "disable_log_stats": True,
        "additional_config": {
            "enable_flashcomm1": False,
            "mix_placement": False,
            "multistream_dsv4_dsa_overlap": False,
            "vq2a8_offline": runtime,
        },
    }


def validate_offline_config(config) -> dict:
    """Reject unsupported integrations before any model-sized allocation."""
    options = resolve_runtime_options((config.additional_config or {}).get("vq2a8_offline"))
    parallel, model = config.parallel_config, config.model_config
    for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
        if type(getattr(parallel, name, None)) is not int or getattr(parallel, name) != 1:
            raise ValueError(f"VQ2A8 requires {name}=1.")
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
    for name, value in (
        ("max_model_len", model.max_model_len),
        ("max_num_batched_tokens", config.scheduler_config.max_num_batched_tokens),
    ):
        if type(value) is not int or not 1 <= value <= V4_DECODER_CONTEXT_LIMIT:
            raise ValueError(f"VQ2A8 requires {name} in [1,16].")
    if config.scheduler_config.max_num_batched_tokens < model.max_model_len:
        raise ValueError("Non-chunked prefill requires max_num_batched_tokens >= max_model_len.")
    if getattr(config.scheduler_config, "enable_chunked_prefill", False):
        raise ValueError("VQ2A8 requires enable_chunked_prefill=False.")
    if getattr(config.cache_config, "block_size", None) != 128:
        raise ValueError("VQ2A8 position templates require block_size=128.")
    extra = config.additional_config or {}
    for name in ("enable_flashcomm1", "enable_dsa_cp", "mix_placement", "multistream_dsv4_dsa_overlap"):
        if extra.get(name, False):
            raise ValueError(f"VQ2A8 requires {name}=False.")
    finegrained = extra.get("finegrained_tp_config", {})
    if not isinstance(finegrained, dict) or any(finegrained.values()):
        raise ValueError("VQ2A8 does not support finegrained TP overrides.")
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
    if options.get("cache_experts", 256) != 256:
        raise ValueError("VQ2A8 requires full residency, not an eviction cache.")
    if type(options.get("token_chunk", 2)) is not int or options.get("token_chunk", 2) != 2:
        raise ValueError("VQ2A8 requires token_chunk=2 to retain shared-expert GEMM geometry.")
    if type(options.get("verbose_experts", False)) is not bool:
        raise ValueError("verbose_experts must be boolean.")
    for key, default, minimum in (("cache_budget_gib", 0.0, 0), ("cache_reserve_gib", 16.0, 1)):
        value = options.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
            raise ValueError(f"{key} must be finite and >= {minimum}.")
    if "cache_memory_fraction" in options:
        _validate_cache_memory_fraction(options["cache_memory_fraction"])
    kv_bytes = getattr(config.cache_config, "kv_cache_memory_bytes", None)
    if type(kv_bytes) is not int or kv_bytes <= 0:
        raise ValueError("VQ2A8 decoder graph requires explicit positive kv_cache_memory_bytes.")
    if kv_bytes > options.get("cache_reserve_gib", 16.0) * GIB:
        raise ValueError("VQ2A8 KV cache must fit inside cache_reserve_gib.")
    path, sha = options.get("ascendc_library"), options.get("ascendc_sha256")
    if path is not None:
        if not isinstance(path, str) or not Path(path).is_absolute() or Path(path).suffix != ".so":
            raise ValueError("An explicit AscendC library requires an absolute .so path.")
        if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
            raise ValueError("An explicit AscendC library requires its regression-tested SHA256.")
    elif sha is not None:
        raise ValueError("ascendc_sha256 requires ascendc_library.")
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


class OfflineMoEOwner:
    """Own the artifact, roots and non-evictable packed expert banks."""

    def __init__(self, model_root: Path, options: dict, device: torch.device, *, tp_size=1, tp_group=None):
        from vllm_ascend.quantization.vq2a8_v4_v2 import load_v4_v2_library, require_v4_v2_features

        if type(tp_size) is not int or tp_size != 1 or tp_group is not None:
            raise ValueError("VQ2A8 supports TP1 only.")
        if device.type != "npu":
            raise ValueError("VQ2A8 requires an NPU, without fallback.")
        options = resolve_runtime_options(options)
        self.native_library = load_v4_v2_library(options.get("ascendc_library"), options.get("ascendc_sha256"))
        require_v4_v2_features()
        manifest = json.loads((Path(options["artifact"]) / "manifest.json").read_text(encoding="utf-8"))
        format_name = manifest.get("format")
        if format_name == V4_V2_PREPACKED_FORMAT:
            self.artifact = open_vq2a8_v4_v2_prepacked_artifact(options["artifact"], model_root / "config.json")
        elif format_name == VQ2_DIRECT_TP1_FORMAT:
            self.artifact = open_vq2a8_tp1_artifact(
                Path(options["artifact"]),
                model_root / "config.json",
                require_complete=True,
                require_reference_identity=True,
            )
        else:
            raise ValueError(f"Unsupported artifact format: {format_name!r}; no format fallback.")
        self.inventory = audit_offline_root(model_root)
        self.options = options
        self.device = device
        self.layers: dict[int, VQ2TP1MoE] = {}
        self.calls: dict[int, int] = {}
        self.cache_plan = None

    def create_layer(self, index: int) -> VQ2TP1MoE:
        from vllm_ascend.quantization.vq2a8_v4_v2 import AscendCV4V2VQ2TP1MoE

        if index in self.layers:
            raise ValueError(f"Duplicate offline MoE layer {index}.")
        with torch.device("cpu"):
            layer = AscendCV4V2VQ2TP1MoE(
                self.artifact,
                index,
                self.device,
                cache_experts=256,
                token_chunk=2,
                progress=True,
                verbose_experts=self.options.get("verbose_experts", False),
            )
        self.layers[index] = layer
        self.calls[index] = 0
        return layer

    def configure_cache(self, memory_fraction: float) -> None:
        """Call after strict root load, before profiling populates any cache."""
        policy = self.options.get("execution_policy")
        override = self.options.get("cache_memory_fraction")
        if override is not None:
            _validate_cache_memory_fraction(override, policy)
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
        self._configure_v4_residency(budget)

    def _configure_v4_residency(self, budget):
        """Preload one selected packed layout before worker profiling."""
        from vllm_ascend.quantization.vq2a8_v4_v2 import v4_v2_resident_plan

        if not self.layers or set(self.layers) != set(self.artifact.layers):
            raise ValueError("V4 requires all artifact layers before full-residency planning.")
        plan = v4_v2_resident_plan([layer.layer for layer in self.layers.values()], budget["budget_bytes"])
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

    def delegated_names(self) -> set[str]:
        return {f"layers.{index}.ffn.{name}" for index, layer in self.layers.items() for name in layer.root}

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
            if (tuple(value.shape) != tuple(param.shape)) or (value.dtype != param.dtype and not widen_norm):
                raise ValueError(
                    f"Root shape/dtype mismatch: {name} -> {target} ({value.shape}, {value.dtype}) "
                    f"!= ({param.shape}, {param.dtype})."
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Non-finite root tensor {name}.")
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
