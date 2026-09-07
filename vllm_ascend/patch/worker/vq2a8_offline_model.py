# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in inheritance adapter. Importing this module does not monkey-patch vLLM.

Only the offline gate selects this architecture. Attention/HC/cache execution
is inherited; MoE allocation, token routing and root loading are explicit.
"""

import json
import time
from pathlib import Path

import torch
from torch import nn
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.models.deepseek_v4 import AscendDeepseekV4ForCausalLM, DeepseekV2DecoderLayer, DeepseekV4Model
from vllm_ascend.quantization.vq2a8_offline import OfflineMoEOwner, validate_offline_config
from vllm_ascend.quantization.vq2a8_root_fp8 import ROOT_FP8_POLICY, RootFP8State, root_linear_kind


class OfflineRootFP8Method(LinearMethodBase):
    """Installed only on the opt-in offline model, before strict BF16 loading."""

    vq2a8_root_mode = ROOT_FP8_POLICY

    def __init__(self, kind):
        self.state = RootFP8State(kind)

    def create_weights(self, *args, **kwargs):
        raise RuntimeError("Offline root FP8 must use the existing canonical BF16 allocation.")

    def process_weights_after_loading(self, layer):
        print(f"MODEL stage=root_fp8_quantize_start prefix={layer.prefix} kind={self.state.kind}", flush=True)
        self.state.process(layer)
        print(f"MODEL stage=root_fp8_quantize_done prefix={layer.prefix}", flush=True)

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("The pinned root FP8 projections are bias-free.")
        return self.state.apply(layer, x)

    def apply_grouped(self, layer, x, groups, rank):
        return self.state.apply_grouped(layer, x, groups, rank)


class OfflineMoEAdapter(nn.Module):
    def __init__(self, owner, layer_index):
        super().__init__()
        self.owner = owner
        self.layer_index = layer_index
        self.runtime = owner.create_layer(layer_index)

    def forward(self, hidden_states, input_ids=None):
        if input_ids is None:
            raise ValueError("Offline MoE must receive actual input_ids, including hash layers.")
        print(f"MODEL layer={self.layer_index} stage=moe_start tokens={input_ids.numel()}", flush=True)
        result = self.runtime.forward(hidden_states, input_ids)
        self.owner.calls[self.layer_index] += 1
        print(f"MODEL layer={self.layer_index} stage=moe_done", flush=True)
        return result


class OfflineDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(self, vllm_config, prefix, topk_indices_buffer, owner):
        self._offline_owner = owner
        super().__init__(vllm_config, prefix, topk_indices_buffer=topk_indices_buffer)
        if owner.options.get("root_linear_mode", "bf16") == ROOT_FP8_POLICY:
            selected = []
            for name, module in self.self_attn.named_modules():
                kind = root_linear_kind(f"{prefix}.self_attn.{name}")
                if kind is None:
                    continue
                if not hasattr(module, "weight") or getattr(module, "tp_size", 1) != 1:
                    raise ValueError(f"Unsupported offline FP8 root module {name}.")
                method = OfflineRootFP8Method(kind)
                module.quant_method = method
                # Ascend custom communication wrappers cache the method during
                # construction. Update the cached reference as well.
                if getattr(module, "custom_op", None) is not None:
                    module.custom_op.update_attrs()
                selected.append(name)
            required = {"wq_a", "wq_b", "wkv", "wo_a", "wo_b"}
            if self.self_attn.indexer is not None:
                required.add("indexer.wq_b")
            if set(selected) != required:
                raise ValueError(f"Root FP8 coverage mismatch: {selected} != {sorted(required)}.")

    def forward(self, *args, **kwargs):
        print(f"MODEL layer={self.layer_idx} stage=decoder_start", flush=True)
        result = super().forward(*args, **kwargs)
        print(f"MODEL layer={self.layer_idx} stage=decoder_done", flush=True)
        return result

    def _build_mlp(self, vllm_config, config, prefix, is_draft_layer):
        if is_draft_layer:
            raise ValueError("Offline VQ2A8 does not support MTP/draft layers.")
        return OfflineMoEAdapter(self._offline_owner, self.layer_idx)


class OfflineDecoderModel(DeepseekV4Model):
    requires_moe_input_ids = True

    def __init__(self, *, vllm_config, prefix=""):
        options = validate_offline_config(vllm_config)
        self.offline_owner = OfflineMoEOwner(
            Path(vllm_config.model_config.model), options, torch.device("npu", torch.npu.current_device())
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def _make_decoder_layer(self, vllm_config, prefix, topk_indices_buffer):
        return OfflineDecoderLayer(vllm_config, prefix, topk_indices_buffer, self.offline_owner)


class VQ2A8TP1OfflineForCausalLM(AscendDeepseekV4ForCausalLM):
    model_cls = OfflineDecoderModel

    def __init__(self, *, vllm_config, prefix=""):
        validate_offline_config(vllm_config)
        ascend = get_ascend_config()
        for name in ("enable_flashcomm1", "mix_placement", "multistream_dsv4_dsa_overlap"):
            if getattr(ascend, name, False):
                raise ValueError(f"Offline VQ2A8 requires {name}=False.")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._offline_loaded = False
        self._offline_trace = False
        self._offline_logits = []
        self._offline_steps = []
        self._offline_load_report = {}
        self._offline_memory_fraction = vllm_config.cache_config.gpu_memory_utilization
        self._offline_root_mode = validate_offline_config(vllm_config).get("root_linear_mode", "bf16")
        self._offline_root_verified = False

    def set_moe_parameters(self):
        # No FusedMoE allocation, expert extraction, EPLB or TP reduction.
        self.expert_weights = []
        self.moe_layers = []
        self.moe_mlp_layers = [layer.mlp for layer in self.model.layers]
        self.num_expert_groups = 1
        self.num_logical_experts = self.num_physical_experts = self.config.n_routed_experts
        self.num_local_physical_experts = self.num_routed_experts = self.config.n_routed_experts
        self.num_shared_experts = self.config.n_shared_experts
        self.num_redundant_experts = 0

    def get_expert_mapping(self):
        return []

    def load_weights(self, weights):
        loaded, report = self.model.offline_owner.load_root(
            dict(self.named_parameters()), weights, default_weight_loader
        )
        self._offline_load_report = report
        if self._offline_root_mode == "bf16":
            self.model.offline_owner.configure_cache(self._offline_memory_fraction)
        self._offline_loaded = True
        print("MODEL_LOAD_RESULT " + json.dumps(report), flush=True)
        return loaded

    def reset_offline_trace(self):
        if not self._offline_loaded:
            raise RuntimeError("Offline model weights have not passed strict loading.")
        self._offline_logits = []
        self._offline_steps = []
        for index in self.model.offline_owner.calls:
            self.model.offline_owner.calls[index] = 0
        self.model.offline_owner.reset_backend_trace()
        for module in self.modules():
            method = getattr(module, "quant_method", None)
            if isinstance(method, OfflineRootFP8Method):
                method.state.calls = 0
        self._offline_trace = True

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if not self._offline_loaded or input_ids is None or inputs_embeds is not None:
            raise ValueError(
                "Offline forward requires loaded weights and real token IDs; inputs_embeds are unsupported."
            )
        if self._offline_root_mode == ROOT_FP8_POLICY and not self._offline_root_verified:
            report = self.root_fp8_evidence()
            if not report["all_processed"]:
                raise ValueError("Root FP8 post-load conversion incomplete before forward.")
            # The vLLM loader has now processed the root modules. Plan against
            # their actual FP8 residency, not the temporary BF16 allocation.
            self.model.offline_owner.configure_cache(self._offline_memory_fraction)
            self._offline_root_verified = True
            print("MODEL_ROOT_FP8_RESULT " + json.dumps(report), flush=True)
        if self._offline_trace and not get_forward_context().attn_metadata:
            raise ValueError("A profiling/dummy attention path cannot count as real model execution.")
        phase = ("prefill" if not self._offline_steps else "decode") if self._offline_trace else "profile"
        print(
            f"MODEL stage=forward_start phase={phase} step={len(self._offline_steps)} tokens={input_ids.numel()}",
            flush=True,
        )
        started = time.perf_counter()
        result = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Non-finite final decoder hidden states.")
        if self._offline_trace:
            if len(self._offline_steps) >= 128:
                raise ValueError("Offline step trace exceeded the bounded gate budget.")
            self._offline_steps.append({"tokens": input_ids.numel(), "positions": positions.cpu().tolist()})
        elapsed = time.perf_counter() - started
        print(f"MODEL stage=forward_done phase={phase} elapsed_s={elapsed:.3f}", flush=True)
        print(
            "MODEL_FORWARD_TIMING "
            + json.dumps(
                {
                    "phase": phase,
                    "tokens": input_ids.numel(),
                    "elapsed_s": elapsed,
                    "cache": self.model.offline_owner.cache_report(),
                }
            ),
            flush=True,
        )
        return result

    def compute_logits(self, hidden_states):
        print(f"MODEL stage=logits_start rows={hidden_states.shape[0]}", flush=True)
        started = time.perf_counter()
        logits = super().compute_logits(hidden_states)
        if logits is None or not bool(torch.isfinite(logits).all()):
            raise ValueError("Non-finite or missing model logits.")
        if self._offline_trace:
            if sum(value.shape[0] for value in self._offline_logits) + logits.shape[0] > 128:
                raise ValueError("Offline logits exceeded the bounded gate budget.")
            self._offline_logits.append(logits.float().cpu())
        print(
            f"MODEL stage=logits_done rows={logits.shape[0]} finite=True elapsed_s={time.perf_counter() - started:.3f}",
            flush=True,
        )
        return logits

    def offline_evidence(self):
        if not self._offline_logits:
            raise ValueError("No real generation logits captured.")
        return {
            "load": self._offline_load_report,
            "steps": self._offline_steps,
            "cache": self.model.offline_owner.cache_report(),
            "logits": torch.cat(self._offline_logits),
            "peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "peak_reserved_bytes": torch.npu.max_memory_reserved(),
            "root_fp8": self.root_fp8_evidence(),
            "expert_backend": self.model.offline_owner.backend_report(),
        }

    def root_fp8_evidence(self):
        records = []
        for name, module in self.named_modules():
            method = getattr(module, "quant_method", None)
            if not isinstance(method, OfflineRootFP8Method):
                continue
            state = method.state
            scale = getattr(module, "vq2a8_root_scale", None)
            records.append(
                {
                    "name": name,
                    "kind": state.kind,
                    "calls": state.calls,
                    "weight_dtype": str(module.weight.dtype),
                    "weight_shape": list(module.weight.shape),
                    "scale_dtype": str(scale.dtype) if scale is not None else None,
                    "processed": state.ready
                    and module.weight.dtype == torch.float8_e4m3fn
                    and scale is not None
                    and scale.dtype == torch.float32,
                }
            )
        return {
            "mode": self._offline_root_mode,
            "layers": records,
            "all_processed": bool(records) and all(r["processed"] for r in records),
            "native_fp8_root_matmul": bool(records)
            and all(r["processed"] and r["calls"] > 0 for r in records if not r["name"].endswith("indexer.wq_b")),
            "native_fp8_expert_dot": False,
            "independent_reference": False,
        }
