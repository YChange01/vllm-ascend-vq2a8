# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in inheritance adapter. Importing this module does not monkey-patch vLLM.

The offline tools and explicit V3/V4 servers select this architecture. Attention/HC/cache execution
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
from vllm_ascend.quantization.vq2a8_v4_decoder_graph import (
    DecoderStateSnapshot,
    V4DecoderGraphBank,
    mutable_decoder_tensors,
)
from vllm_ascend.quantization.vq2a8_v4_device_route import DeviceRouteGraphCompute, create_device_route_banks


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
        quiet = getattr(self.owner, "measurement_mode", False)
        if not quiet:
            print(f"MODEL layer={self.layer_index} stage=moe_start tokens={input_ids.numel()}", flush=True)
        result = self.runtime.forward(hidden_states, input_ids)
        self.owner.calls[self.layer_index] += 1
        if not quiet:
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
        if getattr(self._offline_owner, "optimization_profile", False):
            with torch.profiler.record_function(f"vq2a8::decoder::{self.layer_idx}"):
                return super().forward(*args, **kwargs)
        if getattr(self._offline_owner, "measurement_mode", False):
            return super().forward(*args, **kwargs)
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
        owner_kwargs = {}
        if vllm_config.parallel_config.tensor_parallel_size == 2:
            from vllm.distributed import get_tp_group

            owner_kwargs = {"tp_size": 2, "tp_group": get_tp_group()}
        self.offline_owner = OfflineMoEOwner(
            Path(vllm_config.model_config.model),
            options,
            torch.device("npu", torch.npu.current_device()),
            **owner_kwargs,
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def _make_decoder_layer(self, vllm_config, prefix, topk_indices_buffer):
        return OfflineDecoderLayer(vllm_config, prefix, topk_indices_buffer, self.offline_owner)


class VQ2A8TP1OfflineForCausalLM(AscendDeepseekV4ForCausalLM):
    model_cls = OfflineDecoderModel
    _offline_tp_size = 1

    def __init__(self, *, vllm_config, prefix=""):
        options = validate_offline_config(vllm_config)
        expected_tp = getattr(self, "_offline_tp_size", 1)
        if vllm_config.parallel_config.tensor_parallel_size != expected_tp:
            raise ValueError(f"This VQ2A8 architecture requires tensor_parallel_size={expected_tp}.")
        ascend = get_ascend_config()
        for name in ("enable_flashcomm1", "mix_placement", "multistream_dsv4_dsa_overlap"):
            if getattr(ascend, name, False):
                raise ValueError(f"Offline VQ2A8 requires {name}=False.")
        if expected_tp == 2:
            finegrained = getattr(ascend, "finegrained_tp_config", None)
            for name in (
                "oproj_tensor_parallel_size",
                "olora_tensor_parallel_size",
                "embedding_tensor_parallel_size",
                "lmhead_tensor_parallel_size",
                "mlp_tensor_parallel_size",
            ):
                if getattr(finegrained, name, 0):
                    raise ValueError(f"VQ2A8 TP2 requires {name}=0.")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._offline_loaded = False
        self._offline_trace = False
        self._offline_logits = []
        self._offline_steps = []
        self._offline_load_report = {}
        self._offline_memory_fraction = vllm_config.cache_config.gpu_memory_utilization
        self._offline_root_mode = validate_offline_config(vllm_config).get("root_linear_mode", "bf16")
        self._offline_root_verified = False
        self._v3_serving = options.get("v3_serving", False)
        self._v4_serving = options.get("v4_serving", False)
        self._v4_compute_backend = options.get("v4_compute_backend", "v1")
        self._v4_activation_reorder = options.get("v4_activation_reorder", "scalar")
        self._v4_activation_preparation = options.get("v4_activation_preparation", "rowwise")
        self._v4_device_route_decode = options.get("v4_device_route_decode", False)
        self._v4_serving_batched_ready = False
        self._v4_decode_graph = options.get("v4_decode_graph", "none")
        self._v4_decoder_metadata_mode = options.get("v4_decoder_metadata_mode", "recursive")
        self._v4_host_profile = options.get("v4_host_profile", False)
        self._v4_host_recorder = None
        self._v4_host_runner = None
        self._v4_requested_replay_stream = options.get("v4_graph_replay_stream", "owner")
        self._v4_graph_replay_stream = "owner"
        self._v4_graphs_ready = False
        self._v4_graphs_failed = False
        self._v4_graph_enabled = False
        self._v4_graph_forward_active = False
        self._v4_graph_reserve_bytes = int(options.get("cache_reserve_gib", 16.0) * 1024**3)
        self._v4_graph_kv_cache_bytes = getattr(vllm_config.cache_config, "kv_cache_memory_bytes", None)
        self._v4_graph_memory = {}
        self._v4_decoder_graph = None
        self._v4_decoder_preparing = False
        self._v4_decoder_max_model_len = vllm_config.model_config.max_model_len
        self._startup_trace_mode = options.get("v3_startup_trace", "off")

    def set_moe_parameters(self):
        # No FusedMoE allocation, expert extraction or EPLB. The TP2 runtime
        # owns routed-only reduction; shared experts remain replicated.
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
        if self._v3_serving:
            self._configure_v3_serving()
        if getattr(self, "_v4_serving", False):
            self._configure_v4_serving()
        print("MODEL_LOAD_RESULT " + json.dumps(report), flush=True)
        if self._startup_trace_mode != "off":
            from vllm_ascend.quantization.vq2a8_startup_trace import install_startup_trace

            self._vq2a8_startup_trace = install_startup_trace(self, mode=self._startup_trace_mode)
        return loaded

    def _configure_v3_serving(self):
        """Enable quiet resident execution before the worker's startup profile.

        Serving owns a persistent engine. It does not collect offline evidence,
        reset counters per request or add per-layer timing synchronization.
        """
        owner = self.model.offline_owner
        owner.measurement_mode = True
        self._measurement_valid = None
        self._measurement_forwards = 0
        for layer in owner.layers.values():
            layer.measurement_mode = True
            layer.trace_native = False

    def _configure_v4_serving(self):
        """Keep the loaded model quiet, but retain V1 startup profile geometry.

        Do not use the offline probe controller: a persistent server must not
        reset resident state or add per-request full-weight metadata scans.
        """
        owner = self.model.offline_owner
        if not self._offline_loaded or self._offline_root_mode != "bf16":
            raise ValueError("V4 serving requires strictly loaded BF16 roots.")
        if not owner.layers or any(
            getattr(layer, "execution_policy", None) != "ascendc_v4" for layer in owner.layers.values()
        ):
            raise ValueError("V4 serving requires resident V4 execution on every layer.")
        for layer in owner.layers.values():
            layer.check_resident_integrity()
        owner.measurement_mode = True
        self._measurement_valid = None
        self._measurement_forwards = 0
        for layer in owner.layers.values():
            layer.measurement_mode = True
            layer.trace_native = False

    def _enable_v4_serving_batched(self):
        """One-time switch at the first real request, after dummy/profile work."""
        if self._v4_serving_batched_ready:
            return
        from vllm_ascend.quantization.vq2a8_optimization import configure_runtime

        # V1 profile may still have device work queued. Preserve preparation
        # owners until it has completed, as the offline benchmark does.
        torch.npu.synchronize()
        layers = self.model.offline_owner.layers.values()
        for layer in layers:
            layer.check_resident_integrity()
        preset = "device_route_decode" if getattr(self, "_v4_device_route_decode", False) else "batched"
        for layer in layers:
            configure_runtime(layer, preset, profile=False)
        self._v4_serving_batched_ready = True
        stage = "MODEL_V4_RUNTIME_PREPARED" if self._v4_decode_graph != "none" else "MODEL_V4_SERVING_READY"
        print(
            f"{stage} preset={preset} compute_backend={getattr(self, '_v4_compute_backend', 'v1')} "
            f"activation_reorder={getattr(self, '_v4_activation_reorder', 'scalar')} "
            f"activation_preparation={getattr(self, '_v4_activation_preparation', 'rowwise')} "
            "expert_payload_runtime_loading=False",
            flush=True,
        )

    def prepare_v4_graphs(self, runner=None):
        """Explicit startup capture: MoE-only, or checkpointed decoder graphs.

        The historical MoE path never touches attention/cache state. The new
        decoder path requires the runner and restores mutable state after each
        position's capture and validation replay. Neither uses a real request.
        """
        if self._v4_decode_graph == "none":
            return self.v4_graph_report()
        if self._v4_decode_graph == "decoder":
            self._prepare_v4_decoder_graphs(runner)
            if getattr(self, "_v4_host_profile", False):
                from vllm_ascend.quantization.vq2a8_host_profile import attach_host_profile, wrap_host_call

                if self._v4_host_recorder is None:
                    if runner is None:
                        raise ValueError("Host profiling requires the decoder's model runner on first installation.")
                    recorder = attach_host_profile(runner)
                    self.compute_logits = wrap_host_call(recorder, "compute_logits", self.compute_logits)
                    self._v4_host_recorder = recorder
                    self._v4_host_runner = runner
                    self._v4_decoder_graph.host_profiler = recorder
                elif runner is not None and runner is not self._v4_host_runner:
                    raise ValueError("Host profiling cannot be rebound to a different model runner.")
            return self.v4_graph_report()
        if self._v4_graphs_failed or self._v4_graph_forward_active:
            raise RuntimeError("V4 graph preparation requires a healthy idle model.")
        if self._v4_graphs_ready:
            return self.v4_graph_report()
        owner = self.model.offline_owner
        if not self._offline_loaded or self._offline_root_mode != "bf16" or not self._v4_device_route_decode:
            raise ValueError("V4 MoE graphs require loaded BF16 roots and explicit device-route execution.")
        if not owner.layers or any(layer.execution_policy != "ascendc_v4" for layer in owner.layers.values()):
            raise ValueError("V4 MoE graphs require resident V4 execution on every layer.")
        if self._v4_graph_kv_cache_bytes is None:
            raise ValueError("V4 MoE graph preparation requires explicit kv_cache_memory_bytes for its memory guard.")
        try:
            torch.npu.synchronize()
            self._check_v4_graph_memory("before_capture")
            self._enable_v4_serving_batched()
            for index, layer in owner.layers.items():
                layer.set_v4_graph_phase(False)
                print(f"MODEL_V4_GRAPH_PREPARE layer={index} stage=start", flush=True)
                layer.prepare_v4_graph()
                torch.npu.synchronize()
                self._check_v4_graph_memory(f"layer:{index}")
                print(
                    "MODEL_V4_GRAPH_PREPARE "
                    + json.dumps({"layer": index, "stage": "done", **layer.v4_graph_report()}),
                    flush=True,
                )
            torch.npu.synchronize()
            self._v4_graphs_ready = True
            self.set_v4_graph_replay_stream(self._v4_requested_replay_stream)
            self.set_v4_graph_enabled(True)
        except Exception:
            self._v4_graphs_ready = False
            self._v4_graphs_failed = True
            raise
        report = self.v4_graph_report()
        print("MODEL_V4_GRAPH_READY " + json.dumps(report), flush=True)
        return report

    def _v4_decoder_compute(self, input_ids, positions):
        # No profiling counters, deferred Python validity, or nested NPUGraph
        # replay may enter capture. Pure MoE compute returns fresh device flags.
        layers = tuple(self.model.offline_owner.layers.values())
        for layer in layers:
            layer._v4_decoder_capture = True
        try:
            hidden = super().forward(input_ids, positions, None, None)
            flags = [layer._v4_decoder_valid for layer in layers]
            if any(flag is None for flag in flags):
                raise RuntimeError("Decoder capture did not execute every resident MoE layer.")
            flags.append(torch.isfinite(hidden).all())
            return hidden, torch.stack(flags).all()
        finally:
            for layer in layers:
                layer._v4_decoder_capture = False

    @torch.inference_mode()
    def _prepare_v4_decoder_graphs(self, runner):
        if self._v4_graphs_ready:
            return self.v4_graph_report()
        if self._v4_graphs_failed or self._v4_graph_forward_active or runner is None:
            raise RuntimeError("Decoder graph preparation requires the idle startup runner.")
        if (
            not self._offline_loaded
            or self._offline_root_mode != "bf16"
            or not self._v4_device_route_decode
            or self._v4_requested_replay_stream != "caller"
            or self._v4_graph_kv_cache_bytes is None
            or not getattr(self.model.offline_owner, "measurement_mode", False)
        ):
            raise ValueError("Decoder capture requires serving/device-route/BF16/caller/explicit-KV configuration.")
        bank = V4DecoderGraphBank(
            self,
            self._v4_decoder_max_model_len,
            metadata_mode=getattr(self, "_v4_decoder_metadata_mode", "recursive"),
        )
        self._v4_decoder_graph = bank
        try:
            torch.npu.synchronize()
            self._check_v4_graph_memory("before_decoder_capture")
            self._enable_v4_serving_batched()
            bank.caller_stream = torch.npu.current_stream()
            bank.capture_stream = torch.npu.Stream()
            bank.capture_state = mutable_decoder_tensors(self.model)
            bank.snapshot = DecoderStateSnapshot(bank.capture_state)
            torch.npu.synchronize()
            self._v4_decoder_preparing = True
            with torch.npu.stream(bank.capture_stream):
                for layer in self.model.offline_owner.layers.values():
                    layer.check_resident_integrity()
                    layer._v4_decoder_compute = DeviceRouteGraphCompute(layer, banks=create_device_route_banks(layer))
                    layer._v4_decoder_graph_owner = bank
                bank.computes = tuple(layer._v4_decoder_compute for layer in self.model.offline_owner.layers.values())
                for position in range(bank.max_model_len):
                    print(f"MODEL_V4_DECODER_CAPTURE position={position} stage=begin", flush=True)
                    runner._dummy_run(
                        1,
                        uniform_decode=True,
                        force_attention=True,
                        is_graph_capturing=True,
                        profile_seq_lens=position + 1,
                        vq2a8_capture_position=position,
                    )
                    self._check_v4_graph_memory(f"decoder_position:{position}")
                    print(f"MODEL_V4_DECODER_CAPTURE position={position} stage=done", flush=True)
                torch.npu.synchronize()
                bank.snapshot.restore()
                torch.npu.synchronize()
            bank.snapshot = None
            bank.ready = True
            self._v4_graphs_ready = True
            self._v4_graph_enabled = True
            self._v4_graph_replay_stream = "caller"
        except BaseException:
            bank.failed = True
            self._v4_graphs_failed = True
            raise
        finally:
            self._v4_decoder_preparing = False
        report = self.v4_graph_report()
        print("MODEL_V4_GRAPH_READY " + json.dumps(report), flush=True)
        return report

    def _check_v4_graph_memory(self, stage):
        # KV is already allocated by the worker. The original reserve includes
        # KV, so preserve only its remainder; do not charge KV twice.
        minimum_free = max(0, self._v4_graph_reserve_bytes - self._v4_graph_kv_cache_bytes)
        free, total = torch.npu.mem_get_info()
        sample = {
            "free_bytes": int(free),
            "total_bytes": int(total),
            "allocated_bytes": int(torch.npu.memory_allocated()),
            "reserved_bytes": int(torch.npu.memory_reserved()),
            "required_free_bytes": minimum_free,
        }
        self._v4_graph_memory[stage] = sample
        if free < minimum_free:
            raise RuntimeError(
                f"V4 MoE graph memory guard failed at {stage}: free={free}, required={minimum_free}; "
                "KV, reserve and expert residency are not reduced automatically."
            )

    def set_v4_graph_enabled(self, enabled):
        """Same-engine eager/graph A/B switch; never recreate banks or graphs."""
        if type(enabled) is not bool:
            raise ValueError("V4 graph enabled must be a boolean.")
        if self._v4_graphs_failed or self._v4_graph_forward_active:
            raise RuntimeError("V4 graph mode can only change on a healthy idle model.")
        if enabled and (self._v4_decode_graph == "none" or not self._v4_graphs_ready):
            raise RuntimeError("Call prepare_v4_graphs before enabling V4 MoE graphs.")
        if self._v4_decode_graph == "moe":
            torch.npu.synchronize()
            for layer in self.model.offline_owner.layers.values():
                layer.set_v4_graph_enabled(enabled)
        elif self._v4_decode_graph == "decoder":
            torch.npu.synchronize()
        self._v4_graph_enabled = enabled
        return self.v4_graph_report()

    def set_v4_graph_replay_stream(self, policy):
        """Same captured graphs, different submission stream, between requests.

        The device fence covers all layers and any graph-output consumers.
        Never switch ownership while previous static-buffer work is pending.
        """
        if policy not in ("owner", "caller"):
            raise ValueError("V4 graph replay stream must be owner or caller.")
        if self._v4_decode_graph == "decoder":
            if policy != "caller" or not self._v4_graphs_ready:
                raise ValueError("Decoder graphs require their captured caller replay policy.")
            return self.v4_graph_report()
        if self._v4_graphs_failed or self._v4_graph_forward_active:
            raise RuntimeError("V4 graph replay stream can only change on a healthy idle model.")
        if self._v4_decode_graph != "moe" or not self._v4_graphs_ready:
            raise RuntimeError("Prepare V4 MoE graphs before selecting the replay stream.")
        if policy == self._v4_graph_replay_stream:
            return self.v4_graph_report()
        try:
            torch.npu.synchronize()
            for layer in self.model.offline_owner.layers.values():
                layer.set_v4_graph_replay_stream(policy)
            self._v4_graph_replay_stream = policy
        except Exception:
            # Do not serve with a partially changed set of layer policies.
            self._v4_graphs_failed = True
            raise
        return self.v4_graph_report()

    def v4_graph_report(self):
        return {
            "requested_graph_mode": self._v4_decode_graph,
            "compute_backend": getattr(self, "_v4_compute_backend", "v1"),
            "activation_reorder": getattr(self, "_v4_activation_reorder", "scalar"),
            "activation_preparation": getattr(self, "_v4_activation_preparation", "rowwise"),
            "decoder_metadata_mode": getattr(self, "_v4_decoder_metadata_mode", "recursive"),
            "host_profile": self._v4_host_recorder.report() if getattr(self, "_v4_host_recorder", None) else None,
            "effective_graph_mode": self._v4_decode_graph if self._v4_graph_enabled else "none",
            "requested_replay_stream_policy": self._v4_requested_replay_stream,
            "replay_stream_policy": self._v4_graph_replay_stream,
            "graph_scope": (
                "V4_TP1_B1_DECODER_POSITION_SPECIALIZED"
                if self._v4_decode_graph == "decoder"
                else "V4_TP1_B1_MOE_DECODE1_ONLY"
            ),
            "baseline_mode": (
                "moe_graph_owner"
                if self._v4_decode_graph == "moe" and self._v4_requested_replay_stream == "caller"
                else "device_route_decode_eager"
            ),
            "ready": self._v4_graphs_ready,
            "failed": self._v4_graphs_failed,
            "full_model_graph_verified": False,
            "decoder": (
                self._v4_decoder_graph.report() if getattr(self, "_v4_decoder_graph", None) is not None else None
            ),
            "preparation_memory": dict(self._v4_graph_memory),
            "lowest_free_bytes": min((sample["free_bytes"] for sample in self._v4_graph_memory.values()), default=None),
            "per_layer": {
                str(index): layer.v4_graph_report() for index, layer in self.model.offline_owner.layers.items()
            }
            if self._v4_decode_graph == "moe"
            else {},
        }

    def reset_offline_trace(self):
        if getattr(self.model.offline_owner, "measurement_mode", False):
            raise ValueError("Leave measurement mode before collecting diagnostic logits.")
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

    def configure_performance_probe(self, *, measurement, compact, optimization=None, profile=False):
        """Explicit, bounded offline benchmark control; not a serving switch.

        Only the supervisor's single in-process worker calls this between
        requests. Same-stream native handlers own their tensors; legacy cache
        eviction retains its completion fence. V3 has its own immutable-bank
        and single-stream workspace contract, never a legacy cache fallback.
        """
        from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
        from vllm_ascend.quantization.vq2a8_execution import AscendCVQ2TP1MoE
        from vllm_ascend.quantization.vq2a8_optimization import (
            OptimizationOptions,
            configure_runtime,
            install_profile_hooks,
        )

        owner = self.model.offline_owner
        if not self._offline_loaded or self._offline_root_mode != "bf16":
            raise ValueError("Performance probe requires strictly loaded BF16 roots.")
        if type(measurement) is not bool or type(compact) is not bool:
            raise ValueError("Performance probe switches must be booleans.")
        if type(profile) is not bool:
            raise ValueError("Profile switch must be boolean.")
        v3 = bool(owner.layers) and all(
            getattr(layer, "execution_policy", None) == "ascendc_v3" for layer in owner.layers.values()
        )
        if v3 and optimization not in (None, "batched", "v3"):
            raise ValueError("V3 preserves the fixed resident arithmetic path; other presets require separate gates.")
        v4 = any(getattr(layer, "execution_policy", None) == "ascendc_v4" for layer in owner.layers.values())
        graph_prepared = getattr(self, "_v4_decode_graph", "none") in ("moe", "decoder")
        if getattr(self, "_v4_decode_graph", "none") == "decoder" and not measurement:
            raise ValueError("Decoder graph correctness uses the dedicated real-model probe, not Python trace capture.")
        if graph_prepared and (not self._v4_graphs_ready or self._v4_graphs_failed):
            raise RuntimeError("Prepare healthy V4 MoE graphs before configuring a performance probe.")
        if graph_prepared and (optimization != "device_route_decode" or profile):
            raise ValueError("V4 graph A/B keeps device_route_decode and requires profile=False.")
        device_route = optimization == "device_route_decode"
        if device_route and (not v4 or not getattr(self, "_v4_device_route_decode", False)):
            raise ValueError("device_route_decode requires V4 with v4_device_route_decode explicitly enabled.")
        if v4 and optimization not in (None, "batched", "device_route_decode"):
            raise ValueError(
                "V4 preserves its selected compute backend; "
                "only original/batched or opt-in device_route_decode is supported."
            )
        if optimization is not None and not v3 and not device_route:
            OptimizationOptions.preset(optimization)
        if not owner.layers or not all(isinstance(layer, AscendCVQ2TP1MoE) for layer in owner.layers.values()):
            raise ValueError("Performance probe requires the explicit AscendC backend on every layer.")
        torch.npu.synchronize()
        owner.measurement_mode = measurement
        owner.optimization_profile = profile
        install_profile_hooks(self, profile)
        self._offline_trace = False
        self._offline_logits = []
        self._offline_steps = []
        self._measurement_valid = None
        self._measurement_forwards = 0
        for layer in owner.layers.values():
            if getattr(layer, "execution_policy", None) == "ascendc_v4":
                layer.check_resident_integrity()
            layer.measurement_mode = measurement
            layer.trace_native = False
            layer.native_steps = []
            if getattr(layer, "execution_policy", None) == "ascendc_v3":
                layer.configure_v3_probe(
                    measurement=measurement, compact=compact, optimization=optimization, profile=profile
                )
                continue
            if not graph_prepared and (optimization is not None or hasattr(layer, "_optimization")):
                configure_runtime(layer, optimization, profile=profile)
            preparation = getattr(layer, "_row_preparation", None)
            if preparation is None:
                preparation = layer._row_preparation = RowwiseVQ2A8Preparation()
            if optimization is None:
                preparation.compact = compact
        return {
            "measurement": measurement,
            "compact": compact,
            "optimization": optimization,
            "scope": f"bounded_tp{getattr(owner, 'tp_size', 1)}_offline",
            **(
                {"graph": self.v4_graph_report()["per_layer"], "graph_status": self.v4_graph_report()}
                if graph_prepared
                else {}
            ),
        }

    def performance_snapshot(self):
        """Called outside timed intervals; synchronize and check finite flags."""
        torch.npu.synchronize()
        owner = self.model.offline_owner
        v3 = any(getattr(layer, "execution_policy", None) == "ascendc_v3" for layer in owner.layers.values())
        valid = self._measurement_valid
        optimization = {}
        for index, layer in owner.layers.items():
            if getattr(layer, "execution_policy", None) == "ascendc_v4":
                layer.check_resident_integrity()
            if getattr(layer, "execution_policy", None) == "ascendc_v3":
                layer.check_resident_integrity()
                v3_valid = layer.v3_validity()
                if v3_valid is not None:
                    valid = v3_valid if valid is None else valid & v3_valid
            state = getattr(layer, "_optimization", None)
            if state is not None:
                if state.valid is not None:
                    valid = state.valid if valid is None else valid & state.valid
                optimization[str(index)] = state.report()
                preparation = layer._row_preparation
                if hasattr(preparation, "report"):
                    optimization[str(index)]["graph"] = preparation.report()
        return {
            "finite": bool(valid) if valid is not None else None,
            "optimization": optimization,
            "v3": {
                str(index): layer.v3_report()
                for index, layer in owner.layers.items()
                if getattr(layer, "execution_policy", None) == "ascendc_v3"
            },
            "v4": {
                str(index): layer.v4_report()
                for index, layer in owner.layers.items()
                if getattr(layer, "execution_policy", None) == "ascendc_v4"
            },
            "forwards": getattr(self, "_measurement_forwards", 0),
            "graph": self.v4_graph_report()["per_layer"] if getattr(self, "_v4_decode_graph", "none") == "moe" else {},
            "graph_status": self.v4_graph_report()
            if getattr(self, "_v4_decode_graph", "none") != "none"
            else {"effective_graph_mode": "none"},
            "native_counter_scope": "eager_python_submissions_only; graph replays are reported separately",
            "cache": owner.cache_report(),
            "native_calls": sum(layer.native_calls for layer in owner.layers.values()),
            "native_launches": sum(layer.native_launches for layer in owner.layers.values()),
            "h2d_bytes": sum(layer.h2d_bytes for layer in owner.layers.values()),
            "host_observed_timing": None
            if v3
            else {
                key: sum(layer.timing[key] for layer in owner.layers.values())
                for key in (
                    "host_load_validate_s",
                    "host_read_s",
                    "host_validate_s",
                    "h2d_s",
                    "prepare_s",
                    "packed_projection_s",
                )
            },
            "timing_scope": "v3 host phase timers not collected; null is not zero elapsed time"
            if v3
            else "host submission/validation waits included; prepare/projection are not kernel-only",
            "allocated_bytes": torch.npu.memory_allocated(),
            "reserved_bytes": torch.npu.memory_reserved(),
            "peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "peak_reserved_bytes": torch.npu.max_memory_reserved(),
            "device_free_total_bytes": list(torch.npu.mem_get_info()),
        }

    def _retain_finite_flag(self, value):
        valid = torch.isfinite(value).all()
        self._measurement_valid = valid if self._measurement_valid is None else self._measurement_valid & valid

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if self._v4_decode_graph == "none":
            return self._forward_without_v4_graph_phase(input_ids, positions, intermediate_tensors, inputs_embeds)
        if self._v4_graphs_failed or self._v4_graph_forward_active:
            raise RuntimeError("V4 graph model is failed or already executing; no eager fallback.")
        context = get_forward_context()
        if self._v4_decode_graph == "decoder" and self._v4_decoder_preparing:
            position = getattr(context, "vq2a8_capture_position", None)
            if type(position) is not int or intermediate_tensors is not None or inputs_embeds is not None:
                raise ValueError("Decoder capture requires explicit startup position and unembedded TP1 inputs.")
            return self._v4_decoder_graph.capture(position, input_ids, positions, context, self._v4_decoder_compute)
        # Only execute_model supplies this scheduler-derived marker. Dummy
        # runs (including nonempty attention metadata) deliberately omit it.
        phase = getattr(context, "vq2a8_request_phase", "profile")
        if phase not in ("profile", "prefill", "decode"):
            raise ValueError("Invalid V4 request phase.")
        if phase != "profile" and not self._v4_graphs_ready:
            raise RuntimeError("Call prepare_v4_graphs before serving requests; lazy capture is disabled.")
        is_decode = phase == "decode" and not getattr(context, "in_profile_run", False)
        self._v4_graph_forward_active = True
        try:
            if self._v4_decode_graph == "decoder" and is_decode and self._v4_graph_enabled:
                if intermediate_tensors is not None or inputs_embeds is not None:
                    raise ValueError("Decoder replay accepts only TP1 token IDs, not intermediate tensors/embeddings.")
                hidden, valid = self._v4_decoder_graph.replay(input_ids, positions, context)
                self._measurement_valid = valid if self._measurement_valid is None else self._measurement_valid & valid
                self._measurement_forwards += 1
                return hidden
            for layer in self.model.offline_owner.layers.values():
                layer.set_v4_graph_phase(is_decode)
            return self._forward_without_v4_graph_phase(input_ids, positions, intermediate_tensors, inputs_embeds)
        except Exception:
            if phase != "profile":
                self._v4_graphs_failed = True
            raise
        finally:
            for layer in self.model.offline_owner.layers.values():
                layer.set_v4_graph_phase(False)
            self._v4_graph_forward_active = False

    def _forward_without_v4_graph_phase(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
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
        if getattr(self.model.offline_owner, "measurement_mode", False):
            context = get_forward_context()
            if self._v4_decode_graph != "none":
                real_attention = getattr(context, "vq2a8_request_phase", "profile") in (
                    "prefill",
                    "decode",
                ) and not getattr(context, "in_profile_run", False)
            else:
                real_attention = bool(context.attn_metadata)
            v4_serving = getattr(self, "_v4_serving", False)
            if not real_attention and not (getattr(self, "_v3_serving", False) or v4_serving):
                raise ValueError("Performance probes require real attention metadata.")
            if real_attention and v4_serving and not self._v4_serving_batched_ready:
                self._enable_v4_serving_batched()
            result = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
            if real_attention:
                self._retain_finite_flag(result)
                self._measurement_forwards += 1
            return result
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
        device_route = getattr(self, "_v4_device_route_decode", False)
        flags = []
        if device_route:
            # One validity decision for the complete forward, not a CPU read
            # at each MoE layer. Do not expose tokens after invalid hash IDs,
            # missing resident slots, or failed native bounds checks.
            flags = [
                state.valid
                for layer in self.model.offline_owner.layers.values()
                if (state := getattr(layer, "_optimization", None)) is not None and state.valid is not None
            ]
        if getattr(self.model.offline_owner, "measurement_mode", False):
            logits = super().compute_logits(hidden_states)
            if logits is None:
                raise ValueError("Missing model logits.")
            self._retain_finite_flag(logits)
            if device_route:
                # Include final decoder hidden states and LM-head logits as
                # well as MoE validity, before the sampler can accept tokens.
                flags.append(self._measurement_valid)
                if not bool(torch.stack(flags).all()):
                    if self._v4_decode_graph != "none":
                        self._v4_graphs_failed = True
                    raise ValueError("V4 device-route forward failed validity checks; no output tokens are accepted.")
            return logits
        if flags and not bool(torch.stack(flags).all()):
            if self._v4_decode_graph != "none":
                self._v4_graphs_failed = True
            raise ValueError("V4 device-route forward failed validity checks; no output tokens are accepted.")
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
        # Optimized row preparation records deferred device validity flags.
        # Consume them outside inference timing, including the v2 bring-up:
        # finite final logits alone cannot certify valid intermediate inputs.
        for index, layer in self.model.offline_owner.layers.items():
            if getattr(layer, "execution_policy", None) == "ascendc_v4":
                layer.check_resident_integrity()
            if getattr(layer, "execution_policy", None) == "ascendc_v3":
                layer.check_resident_integrity()
                valid = layer.v3_validity()
                if valid is None or not bool(valid):
                    raise ValueError(f"Offline v3 layer {index} has missing/failed intermediate validity.")
            state = getattr(layer, "_optimization", None)
            if state is not None and (state.valid is None or not bool(state.valid)):
                raise ValueError(f"Offline optimized layer {index} has missing/failed intermediate validity.")
        return {
            "load": self._offline_load_report,
            "steps": self._offline_steps,
            "cache": self.model.offline_owner.cache_report(),
            "logits": torch.cat(self._offline_logits),
            "peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "peak_reserved_bytes": torch.npu.max_memory_reserved(),
            "root_fp8": self.root_fp8_evidence(),
            "expert_backend": self.model.offline_owner.backend_report(),
            "v3": {
                str(index): layer.v3_report()
                for index, layer in self.model.offline_owner.layers.items()
                if getattr(layer, "execution_policy", None) == "ascendc_v3"
            },
            "v4": {
                str(index): layer.v4_report()
                for index, layer in self.model.offline_owner.layers.items()
                if getattr(layer, "execution_policy", None) == "ascendc_v4"
            },
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


class VQ2A8TP2OfflineForCausalLM(VQ2A8TP1OfflineForCausalLM):
    """Explicit TP2 eager entry; never reinterpret the TP1 architecture name."""

    _offline_tp_size = 2
