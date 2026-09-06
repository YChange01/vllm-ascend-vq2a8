# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-testable contracts for the opt-in, eager offline model adapter."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open

from vllm_ascend.quantization.vq2a8_execution import CachedVQ2TP1MoE, device_cache_budget, packed_cache_plan
from vllm_ascend.quantization.vq2a8_moe import VQ2TP1MoE
from vllm_ascend.quantization.vq2a8_runtime import open_vq2a8_tp1_artifact

OFFLINE_CONTEXT_LIMIT = 32
OFFLINE_NEW_TOKENS = 4
OFFLINE_RUNS = 2


def offline_engine_options(
    model_root: Path, artifact: Path, *, execution_policy="cached", cache_budget_gib=0.0, cache_reserve_gib=16.0
) -> dict:
    """A fixed, bounded bring-up plan, not a general serving configuration."""
    return {
        "model": str(model_root),
        "skip_tokenizer_init": True,
        "trust_remote_code": False,
        "hf_overrides": {"architectures": ["VQ2A8TP1OfflineForCausalLM"], "quantization_config": None},
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
        "max_model_len": OFFLINE_CONTEXT_LIMIT,
        "max_num_batched_tokens": OFFLINE_CONTEXT_LIMIT,
        "block_size": 128,
        "gpu_memory_utilization": 0.9 if execution_policy == "cached" else 0.35,
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
                "cache_experts": 256 if execution_policy == "cached" else 2,
                "token_chunk": 2,
                "cache_budget_gib": cache_budget_gib,
                "cache_reserve_gib": cache_reserve_gib,
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
    }
    if set(options) - allowed or not isinstance(options.get("artifact"), str):
        raise ValueError("Invalid vq2a8_offline options/artifact path.")
    parallel, model = config.parallel_config, config.model_config
    for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
        if getattr(parallel, name, None) != 1:
            raise ValueError(f"Offline VQ2A8 requires {name}=1.")
    for name in ("prefill_context_parallel_size", "decode_context_parallel_size"):
        if getattr(parallel, name, 1) != 1:
            raise ValueError(f"Offline VQ2A8 requires {name}=1.")
    for name in ("enable_expert_parallel", "enable_eplb", "use_sequence_parallel_moe"):
        if getattr(parallel, name, False):
            raise ValueError(f"Offline VQ2A8 does not support {name}.")
    if not model.enforce_eager or config.quant_config is not None or model.quantization is not None:
        raise ValueError("Offline adapter requires eager execution and unquantized root layers.")
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
    if options.get("execution_policy", "baseline") not in ("baseline", "cached"):
        raise ValueError("execution_policy must be baseline or cached.")
    for key, default, upper in (("cache_experts", 2, 256), ("token_chunk", 2, 8)):
        value = options.get(key, default)
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"{key} must be an integer in [1,{upper}].")
    for key, default, minimum in (("cache_budget_gib", 0.0, 0), ("cache_reserve_gib", 16.0, 1)):
        value = options.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
            raise ValueError(f"{key} must be finite and >= {minimum}.")
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
    """One artifact index per model; lazy per-layer caches share a byte budget."""

    def __init__(self, model_root: Path, options: dict, device: torch.device):
        self.artifact = open_vq2a8_tp1_artifact(
            Path(options["artifact"]),
            model_root / "config.json",
            require_complete=True,
            require_reference_identity=True,
        )
        self.inventory = audit_offline_root(model_root)
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
            runtime_class = CachedVQ2TP1MoE if self.options.get("execution_policy") == "cached" else VQ2TP1MoE
            layer = runtime_class(
                self.artifact,
                index,
                self.device,
                cache_experts=self.options.get("cache_experts", 2),
                token_chunk=self.options.get("token_chunk", 2),
                **({"progress": True} if runtime_class is CachedVQ2TP1MoE else {}),
            )
        self.layers[index] = layer
        self.calls[index] = 0
        print(f"MODEL layer={index} stage=moe_root_load_done", flush=True)
        return layer

    def configure_cache(self, memory_fraction: float) -> None:
        """Call after strict root load, before profiling populates any cache."""
        if self.options.get("execution_policy") != "cached":
            return
        if any(layer.cache_stats()["resident_experts"] for layer in self.layers.values()):
            raise ValueError("Configure the packed cache before the first expert call.")
        budget = device_cache_budget(
            self.device,
            reserve_gib=self.options.get("cache_reserve_gib", 16.0),
            budget_gib=self.options.get("cache_budget_gib", 0.0),
            memory_fraction=memory_fraction,
        )
        plan = packed_cache_plan(
            [layer.layer for layer in self.layers.values()],
            budget["budget_bytes"],
            expert_limit=self.options.get("cache_experts", 256),
        )
        for index, layer in self.layers.items():
            layer.cache_experts = plan["layer_limits"][index]
        self.cache_plan = {**budget, **plan}
        print("MODEL_CACHE_PLAN " + json.dumps(self.cache_plan), flush=True)

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
            if tuple(value.shape) != tuple(param.shape) or (value.dtype != param.dtype and not widen_norm):
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
        }


def validate_offline_evidence(evidence: dict, prompt: list[int], generated: list[int], layers: int, vocab: int) -> dict:
    """Require real prefill followed by decode, all layers, and sampler/logit agreement."""
    if len(prompt) < 2 or len(generated) != OFFLINE_NEW_TOKENS:
        raise ValueError("Gate requires a multi-token prefill and four generated tokens.")
    if any(type(token) is not int or not 0 <= token < vocab for token in prompt + generated):
        raise ValueError("Out-of-vocabulary or invalid token IDs.")
    expected_steps = [{"tokens": len(prompt), "positions": list(range(len(prompt)))}]
    expected_steps.extend({"tokens": 1, "positions": [len(prompt) + index]} for index in range(len(generated) - 1))
    if evidence["steps"] != expected_steps:
        raise ValueError(f"Unexpected prefill/decode positions or padded batch: {evidence['steps']}.")
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
    logits = evidence["logits"]
    if logits.shape != (len(generated), vocab) or not bool(torch.isfinite(logits).all()):
        raise ValueError("Invalid shape or non-finite generation logits.")
    # Tied greedy maxima may select either tied token; require the sampled
    # logit to be exactly maximal instead of assuming argmax tie breaking.
    selected = logits[torch.arange(len(generated)), torch.tensor(generated)]
    if not torch.equal(selected, logits.max(-1).values):
        raise ValueError("Sampled tokens disagree with greedy captured logits.")
    return {
        "prefill_tokens": len(prompt),
        "decode_steps": len(generated) - 1,
        "generated_token_ids": generated,
        "layers_executed": layers,
        "finite_logits": True,
        "greedy_logits_agree": True,
        "cache": evidence["cache"],
        "load": evidence["load"],
        "peak_allocated_bytes": evidence["peak_allocated_bytes"],
        "peak_reserved_bytes": evidence["peak_reserved_bytes"],
    }
