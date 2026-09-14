# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded eager TP1 MoE bring-up, not a registered serving backend.

NPU/CUDA experts keep packed weights and invoke the accepted M=1 kernel for
each row. Host routing schedules, synchronous validation and an LRU cache
are deliberate correctness-baseline choices, not graph/throughput claims.
The CPU path is a dense oracle only; it never runs on an accelerator.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm_ascend.quantization.vq2a8_reference import (
    decode_repacked_vq2a8_codebook_weight,
    deepseek_v4_swiglu_reference,
    prepare_repacked_vq2a8_activation_reference,
    vq2a8_predecoded_matmul_reference,
)
from vllm_ascend.quantization.vq2a8_runtime import VQ2TP1Artifact


def _finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor.float()).all()):
        raise ValueError(f"{name} contains non-finite values.")


@dataclass(frozen=True)
class VQ2MoEConfig:
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    num_shared: int
    num_hash_layers: int
    vocab_size: int
    renormalize: bool
    routed_scale: float
    swiglu_limit: float | None

    @classmethod
    def from_json(cls, path: Path) -> VQ2MoEConfig:
        config = json.loads(path.read_text(encoding="utf-8"))
        if config.get("scoring_func") != "sqrtsoftplus" or config.get("hidden_act") != "silu":
            raise ValueError("TP1 MoE currently requires sqrtsoftplus routing and silu activation.")
        if any(config.get(key) not in (None, 1) for key in ("n_group", "topk_group")):
            raise ValueError("Grouped top-k is not supported by the TP1 MoE baseline.")
        for key in ("hidden_size", "moe_intermediate_size", "n_routed_experts", "num_experts_per_tok", "vocab_size"):
            if type(config.get(key)) is not int or config[key] <= 0:
                raise ValueError(f"{key} must be a positive integer.")
        for key in ("n_shared_experts", "num_hash_layers"):
            if type(config.get(key)) is not int or config[key] < 0:
                raise ValueError(f"{key} must be a non-negative integer.")
        if config["num_experts_per_tok"] > config["n_routed_experts"]:
            raise ValueError("top-k exceeds the number of routed experts.")
        scale = config.get("routed_scaling_factor")
        if isinstance(scale, bool) or not isinstance(scale, int | float) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("routed_scaling_factor must be finite and positive.")
        if not isinstance(config.get("norm_topk_prob"), bool):
            raise ValueError("norm_topk_prob must be bool.")
        limit = config.get("swiglu_limit")
        deepseek_v4_swiglu_reference(torch.zeros(1, 2), limit)
        return cls(
            config["hidden_size"],
            config["moe_intermediate_size"],
            config["n_routed_experts"],
            config["num_experts_per_tok"],
            config["n_shared_experts"],
            config["num_hash_layers"],
            config["vocab_size"],
            config["norm_topk_prob"],
            float(scale),
            limit,
        )


def route_vq2a8(
    logits: torch.Tensor,
    top_k: int,
    *,
    renormalize: bool = True,
    correction_bias: torch.Tensor | None = None,
    hash_table: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    validity=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unscaled FP32 weights and int64 expert IDs.

    Bias changes expert selection, not mixture weights. Equal biased scores
    select the lowest ID first, matching the pinned dsv4_topk kernel. The
    single routed-scale owner is mix_vq2a8_routes, not this function.
    """
    if (
        logits.ndim != 2
        or not logits.is_floating_point()
        or type(top_k) is not int
        or not 1 <= top_k <= logits.shape[1]
        or type(renormalize) is not bool
    ):
        raise ValueError("Invalid router logits or top-k.")
    if validity is None:
        _finite(logits, "router logits")
    else:
        validity(torch.isfinite(logits).all())
    scores = F.softplus(logits.float()).sqrt()
    if hash_table is not None:
        if correction_bias is not None:
            raise ValueError("Hash routing must not also use a correction bias.")
        if hash_table.ndim != 2 or hash_table.shape[1] != top_k or hash_table.dtype not in (torch.int32, torch.int64):
            raise ValueError("Hash table must be integer [vocab, top_k].")
        if (
            input_ids is None
            or input_ids.shape != (logits.shape[0],)
            or input_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Hash routing requires integer input_ids with one ID per token.")
        if bool(((input_ids < 0) | (input_ids >= hash_table.shape[0])).any()):
            raise ValueError("Hash input token ID is out of range.")
        ids = hash_table[input_ids.to(hash_table.device, dtype=torch.int64)].to(logits.device, dtype=torch.int64)
        if bool(((ids < 0) | (ids >= logits.shape[1])).any()):
            raise ValueError("Hash expert ID is out of range.")
    else:
        choice = scores.clone()
        if correction_bias is not None:
            if correction_bias.shape != (logits.shape[1],) or correction_bias.device != logits.device:
                raise ValueError("Correction bias shape/device does not match router logits.")
            if validity is None:
                _finite(correction_bias, "correction bias")
            else:
                validity(torch.isfinite(correction_bias).all())
            choice += correction_bias.float()
        selected = []
        for _ in range(top_k):
            index = choice.argmax(dim=1, keepdim=True)
            selected.append(index)
            choice.scatter_(1, index, -float("inf"))
        ids = torch.cat(selected, dim=1)
    weights = scores.gather(1, ids)
    if renormalize:
        denominator = weights.sum(dim=1, keepdim=True)
        if validity is None:
            if bool((denominator <= 0).any()):
                raise ValueError("Selected router scores sum to zero.")
        else:
            validity(torch.isfinite(denominator).all() & (denominator > 0).all())
        weights = weights / denominator
    return weights, ids


def mix_vq2a8_routes(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    expert: Callable[[int, torch.Tensor], torch.Tensor],
    *,
    routed_scale: float,
    shared: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Evaluate each unique (token, expert) once; retain every top-k slot.

    Input weights MUST NOT contain routed_scaling_factor. Slotwise FP32
    multiplication/reduction retains duplicate routes without scatter-add
    races. Shared experts run once and are not multiplied by routed_scale.
    The caller bounds the token chunk and validates expert membership.
    """
    if hidden.ndim != 2 or hidden.dtype != torch.bfloat16:
        raise ValueError("MoE hidden states must be BF16 [tokens, hidden].")
    if weights.ndim != 2 or weights.shape != ids.shape or weights.shape[0] != hidden.shape[0] or weights.shape[1] < 1:
        raise ValueError("Route weights/IDs must be [tokens, top_k].")
    if weights.dtype != torch.float32 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Route weights must be FP32 and IDs must be integer.")
    if weights.device != hidden.device or ids.device != hidden.device:
        raise ValueError("Routes and hidden states must share a device.")
    if (
        isinstance(routed_scale, bool)
        or not isinstance(routed_scale, int | float)
        or not math.isfinite(routed_scale)
        or routed_scale <= 0
    ):
        raise ValueError("Routed scale must be finite and positive.")
    _finite(hidden, "hidden states")
    _finite(weights, "route weights")
    if bool((weights < 0).any()) or bool((ids < 0).any()):
        raise ValueError("Route weights and expert IDs must be non-negative.")
    if hidden.shape[0] == 0:
        return torch.empty_like(hidden)
    plan: dict[int, dict[int, list[int]]] = {}
    for token, token_ids in enumerate(ids.cpu().tolist()):
        for slot, expert_id in enumerate(token_ids):
            plan.setdefault(expert_id, {}).setdefault(token, []).append(slot)
    slots = torch.empty((*ids.shape, hidden.shape[1]), device=hidden.device, dtype=torch.bfloat16)
    for expert_id, rows in plan.items():
        token_list = list(rows)
        selected = hidden.index_select(0, torch.tensor(token_list, device=hidden.device, dtype=torch.int64))
        values = expert(expert_id, selected)
        if values.shape != selected.shape or values.dtype != hidden.dtype or values.device != hidden.device:
            raise ValueError(f"Expert {expert_id} returned an invalid shape, dtype or device.")
        _finite(values, f"expert {expert_id} output")
        for position, token in enumerate(token_list):
            for slot in rows[token]:
                slots[token, slot].copy_(values[position])
    result = (slots.float() * weights.unsqueeze(-1)).sum(dim=1) * routed_scale
    if shared is not None:
        shared_output = shared(hidden)
        if (
            shared_output.shape != hidden.shape
            or shared_output.dtype != hidden.dtype
            or shared_output.device != hidden.device
        ):
            raise ValueError("Shared experts returned an invalid shape, dtype or device.")
        _finite(shared_output, "shared expert output")
        result += shared_output.float()
    result = result.to(hidden.dtype)
    _finite(result, "combined MoE output")
    return result


def load_vq2a8_moe_root_weights(model_root: Path, layer: int, config: VQ2MoEConfig) -> dict[str, torch.Tensor]:
    """Read just the router/hash/shared tensors from the canonical checkpoint."""
    prefix = f"layers.{layer}.ffn."
    requested = {"gate.weight": (config.num_experts, config.hidden_size)}
    if layer < config.num_hash_layers:
        requested["gate.tid2eid"] = (config.vocab_size, config.top_k)
    else:
        requested["gate.bias"] = (config.num_experts,)
    if config.num_shared:
        width = config.intermediate_size * config.num_shared
        requested.update(
            {
                "shared_experts.w1.weight": (width, config.hidden_size),
                "shared_experts.w3.weight": (width, config.hidden_size),
                "shared_experts.w2.weight": (config.hidden_size, width),
            }
        )
    tensors = {}
    for path in sorted(model_root.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            names = set(handle.keys())
            for name, shape in requested.items():
                if prefix + name not in names:
                    continue
                if name in tensors:
                    raise ValueError(f"Duplicate checkpoint tensor {prefix + name}.")
                value = handle.get_tensor(prefix + name)
                expected_dtypes = (
                    (torch.int32, torch.int64) if name == "gate.tid2eid" else (torch.bfloat16, torch.float32)
                )
                if tuple(value.shape) != shape or value.dtype not in expected_dtypes:
                    raise ValueError(f"Invalid checkpoint tensor {prefix + name}: {value.dtype} {tuple(value.shape)}.")
                _finite(value, prefix + name)
                tensors[name] = value.contiguous()
    if set(tensors) != set(requested):
        raise ValueError(f"Missing MoE root tensors: {sorted(set(requested) - set(tensors))}.")
    return tensors


class VQ2TP1MoE:
    """Standalone eager layer with explicit device/cache ownership.

    Not nn.Module/state_dict integration: do not attach to a serving model
    until loader, forward-context and graph contracts have been validated.
    """

    def __init__(
        self,
        artifact: VQ2TP1Artifact,
        layer_index: int,
        device: torch.device | str,
        *,
        cache_experts: int = 8,
        token_chunk: int = 8,
        tp_size: int = 1,
    ):
        if (
            any(type(value) is not int for value in (tp_size, cache_experts, token_chunk))
            or tp_size != 1
            or cache_experts < 1
            or token_chunk < 1
        ):
            raise ValueError("Requires TP1 and positive expert-cache/token-chunk limits.")
        self.artifact = artifact
        self.layer_index = layer_index
        self.layer = artifact.layer(layer_index)
        self.config = VQ2MoEConfig.from_json(artifact.model_config_path)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "npu", "cuda"):
            raise ValueError("The TP1 MoE baseline supports CPU, NPU or CUDA only.")
        self.cache_experts = cache_experts
        self.token_chunk = token_chunk
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self.cache_loads = 0
        self.cache_hits = 0
        self.cache_peak_bytes = 0
        root = load_vq2a8_moe_root_weights(artifact.model_config_path.parent, layer_index, self.config)
        table = root.get("gate.tid2eid")
        if table is not None and not set(table.unique().tolist()).issubset(self.layer.expert_ids):
            raise ValueError("Hash table routes to experts missing from this artifact.")
        self.root = {
            key: value.to(
                self.device,
                dtype=(
                    torch.int64
                    if key.endswith("tid2eid")
                    else torch.float32
                    if key.startswith("gate.")
                    else torch.bfloat16
                ),
            )
            for key, value in root.items()
        }

    def cache_stats(self) -> dict:
        sizes = [
            tensor.numel() * tensor.element_size()
            for expert in self._cache.values()
            for payload, _ in expert.values()
            for tensor in payload.values()
        ]
        return {
            "resident_experts": len(self._cache),
            "resident_bytes": sum(sizes),
            "peak_packed_bytes": self.cache_peak_bytes,
            "loads": self.cache_loads,
            "hits": self.cache_hits,
        }

    def clear_cache(self) -> None:
        self._cache.clear()

    def _get_expert(self, expert_id: int) -> dict:
        if expert_id not in self.layer.expert_ids:
            raise ValueError(f"Layer {self.layer_index} has no stored expert {expert_id}.")
        if expert_id in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(expert_id)
            return self._cache[expert_id]
        if len(self._cache) >= self.cache_experts:
            self._cache.popitem(last=False)
        expert = {
            kind: self.artifact.load_expert(self.layer_index, expert_id, kind, device=self.device)
            for kind in ("gate_up", "down")
        }
        self._cache[expert_id] = expert
        self.cache_loads += 1
        self.cache_peak_bytes = max(self.cache_peak_bytes, self.cache_stats()["resident_bytes"])
        return expert

    def _projection(self, hidden: torch.Tensor, payload: dict, spec) -> torch.Tensor:
        if self.device.type == "cpu":
            weight = decode_repacked_vq2a8_codebook_weight(payload, spec)
            return vq2a8_predecoded_matmul_reference(
                hidden,
                weight,
                payload["weight_scale"],
                payload["weight_bias"],
                payload["rht_sign"],
                spec,
                dynamic_a8=True,
            ).to(torch.bfloat16)
        # Lazy import keeps host-only routing tests independent of Triton/NPU.
        from vllm_ascend.quantization.vq2a8_triton import vq2a8_tp1_m1_packed_gemm

        outputs = []
        for row in hidden.split(1):
            if spec.columns != spec.rht_true_columns:
                row = F.pad(row, (0, spec.columns - spec.rht_true_columns))
            quantized, scale, bias = prepare_repacked_vq2a8_activation_reference(
                row, payload["weight_scale"], payload["weight_bias"], payload["rht_sign"], spec.rht_block_size
            )
            outputs.append(
                vq2a8_tp1_m1_packed_gemm(
                    quantized.contiguous(),
                    scale.contiguous(),
                    bias.contiguous(),
                    payload["packed_indices"],
                    payload["codebooks"],
                    payload["codebook_tile_ids"],
                )
            )
        return torch.cat(outputs, dim=0)

    def expert(self, expert_id: int, hidden: torch.Tensor) -> torch.Tensor:
        payload = self._get_expert(expert_id)
        gate = self._projection(hidden, *payload["gate_up"])
        activated = deepseek_v4_swiglu_reference(gate, self.config.swiglu_limit)
        return self._projection(activated, *payload["down"])

    def shared(self, hidden: torch.Tensor) -> torch.Tensor:
        gate = F.linear(hidden, self.root["shared_experts.w1.weight"])
        up = F.linear(hidden, self.root["shared_experts.w3.weight"])
        activated = deepseek_v4_swiglu_reference(torch.cat((gate, up), dim=-1), self.config.swiglu_limit)
        return F.linear(activated, self.root["shared_experts.w2.weight"])

    @torch.inference_mode()
    def route(self, hidden: torch.Tensor, input_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            hidden.ndim != 2
            or hidden.shape[1] != self.config.hidden_size
            or hidden.device != self.root["gate.weight"].device
        ):
            raise ValueError("Hidden shape/device does not match this MoE layer.")
        if hidden.dtype != torch.bfloat16:
            raise ValueError("The TP1 MoE baseline requires BF16 hidden states.")
        logits = F.linear(hidden.float(), self.root["gate.weight"])
        return route_vq2a8(
            logits,
            self.config.top_k,
            renormalize=self.config.renormalize,
            correction_bias=self.root.get("gate.bias"),
            hash_table=self.root.get("gate.tid2eid"),
            input_ids=input_ids,
        )

    @torch.inference_mode()
    def forward(self, hidden: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        weights, ids = self.route(hidden, input_ids)
        if not set(ids.cpu().flatten().tolist()).issubset(self.layer.expert_ids):
            raise ValueError("Router selected experts missing from this artifact.")
        if not hidden.shape[0]:
            return torch.empty_like(hidden)
        outputs = []
        for start in range(0, hidden.shape[0], self.token_chunk):
            stop = start + self.token_chunk
            outputs.append(
                mix_vq2a8_routes(
                    hidden[start:stop],
                    weights[start:stop],
                    ids[start:stop],
                    self.expert,
                    routed_scale=self.config.routed_scale,
                    shared=self.shared if self.config.num_shared else None,
                )
            )
        return torch.cat(outputs, dim=0)
