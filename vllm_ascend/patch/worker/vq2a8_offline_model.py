# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in VQ2A8 adapter for the existing Ascend DeepSeek V4 implementation.

Attention, HC and KV-cache execution remain inherited. The adapter replaces
only MoE allocation/loading and adds explicitly prepared TP1 decoder graphs.
Importing this module never monkey-patches other model instances.
"""

from pathlib import Path

import torch
from torch import nn
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.models.deepseek_v4 import AscendDeepseekV4ForCausalLM, DeepseekV2DecoderLayer, DeepseekV4Model
from vllm_ascend.quantization.vq2a8_offline import OfflineMoEOwner, validate_offline_config
from vllm_ascend.quantization.vq2a8_v4_decoder_graph import (
    DecoderStateSnapshot,
    V4DecoderGraphBank,
    mutable_decoder_tensors,
)
from vllm_ascend.quantization.vq2a8_v4_device_route import DeviceRouteGraphCompute, create_device_route_banks


class OfflineMoEAdapter(nn.Module):
    def __init__(self, owner, layer_index):
        super().__init__()
        self.owner = owner
        self.layer_index = layer_index
        self.runtime = owner.create_layer(layer_index)

    def forward(self, hidden_states, input_ids=None):
        if input_ids is None:
            raise ValueError("VQ2A8 MoE requires actual token IDs, including hash-routing layers.")
        result = self.runtime.forward(hidden_states, input_ids)
        self.owner.calls[self.layer_index] += 1
        return result


class OfflineDecoderLayer(DeepseekV2DecoderLayer):
    def __init__(self, vllm_config, prefix, topk_indices_buffer, owner):
        self._offline_owner = owner
        super().__init__(vllm_config, prefix, topk_indices_buffer=topk_indices_buffer)

    def _build_mlp(self, vllm_config, config, prefix, is_draft_layer):
        if is_draft_layer:
            raise ValueError("VQ2A8 does not support MTP/draft layers.")
        return OfflineMoEAdapter(self._offline_owner, self.layer_idx)


class OfflineDecoderModel(DeepseekV4Model):
    requires_moe_input_ids = True

    def __init__(self, *, vllm_config, prefix=""):
        options = validate_offline_config(vllm_config)
        self.offline_owner = OfflineMoEOwner(
            Path(vllm_config.model_config.model),
            options,
            torch.device("npu", torch.npu.current_device()),
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def _make_decoder_layer(self, vllm_config, prefix, topk_indices_buffer):
        return OfflineDecoderLayer(vllm_config, prefix, topk_indices_buffer, self.offline_owner)


class VQ2A8TP1OfflineForCausalLM(AscendDeepseekV4ForCausalLM):
    """Explicit TP1 entry selected through vLLM's ordinary model registry."""

    model_cls = OfflineDecoderModel

    def __init__(self, *, vllm_config, prefix=""):
        options = validate_offline_config(vllm_config)
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("VQ2A8 requires tensor_parallel_size=1.")
        ascend = get_ascend_config()
        for name in ("enable_flashcomm1", "mix_placement", "multistream_dsv4_dsa_overlap"):
            if getattr(ascend, name, False):
                raise ValueError(f"VQ2A8 requires {name}=False.")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._offline_loaded = False
        self._offline_memory_fraction = vllm_config.cache_config.gpu_memory_utilization
        self._v4_serving_batched_ready = False
        self._v4_decode_graph = "decoder"
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
        self._measurement_valid = None
        self._last_forward_is_request = False

    def set_moe_parameters(self):
        # Resident VQ2A8 experts do not allocate upstream FusedMoE or EPLB state.
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
        self.model.offline_owner.configure_cache(self._offline_memory_fraction)
        self._offline_loaded = True
        owner = self.model.offline_owner
        if not owner.layers or any(layer.execution_policy != "ascendc_v4" for layer in owner.layers.values()):
            raise ValueError("VQ2A8 requires resident V4 execution on every layer.")
        for layer in owner.layers.values():
            layer.check_resident_integrity()
            layer.measurement_mode = True
        owner.measurement_mode = True
        logger.info("VQ2A8 root weights loaded: %s", report)
        return loaded

    def _enable_v4_serving_batched(self):
        if self._v4_serving_batched_ready:
            return
        # Lazy loading keeps model registration independent of worker state.
        from vllm_ascend.quantization.vq2a8_optimization import configure_runtime

        # Startup profiling may still own queued work. Fence before changing
        # preparation owners or constructing the resident graph compute banks.
        torch.npu.synchronize()
        layers = tuple(self.model.offline_owner.layers.values())
        for layer in layers:
            layer.check_resident_integrity()
        for layer in layers:
            configure_runtime(layer, "device_route_decode", profile=False)
        self._v4_serving_batched_ready = True

    def prepare_v4_graphs(self, runner=None):
        """Capture once at worker startup, never lazily on a real request."""
        self._prepare_v4_decoder_graphs(runner)
        return self.v4_graph_report()

    def _v4_decoder_compute(self, input_ids, positions):
        # Pure MoE capture produces fresh device validity flags; no nested
        # per-layer graph replay or deferred Python validity enters capture.
        layers = tuple(self.model.offline_owner.layers.values())
        for layer in layers:
            layer._v4_decoder_capture = True
            layer._v4_decoder_valid = None
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
            return
        if self._v4_graphs_failed or self._v4_graph_forward_active or runner is None:
            raise RuntimeError("Decoder graph preparation requires the idle startup runner.")
        if not self._offline_loaded or self._v4_graph_kv_cache_bytes is None:
            raise ValueError("Decoder capture requires loaded weights and explicit KV-cache memory.")
        bank = V4DecoderGraphBank(self, self._v4_decoder_max_model_len, metadata_mode="position_template")
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
                    runner._dummy_run(
                        1,
                        uniform_decode=True,
                        force_attention=True,
                        is_graph_capturing=True,
                        profile_seq_lens=position + 1,
                        vq2a8_capture_position=position,
                    )
                    self._check_v4_graph_memory(f"decoder_position:{position}")
                torch.npu.synchronize()
                bank.snapshot.restore()
                torch.npu.synchronize()
            bank.snapshot = None
            from vllm_ascend.quantization.vq2a8_decoder_position_template import install_position_template
            from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

            if get_ascend_device_type() != AscendDeviceType.A5:
                raise ValueError("Position-template decoder metadata requires Ascend A5.")
            install_position_template(runner, bank)
            bank.ready = True
            self._v4_graphs_ready = True
            self._v4_graph_enabled = True
        except BaseException:
            bank.failed = True
            self._v4_graphs_failed = True
            raise
        finally:
            self._v4_decoder_preparing = False
        logger.info("VQ2A8 decoder graph preparation complete: %s", self.v4_graph_report())

    def _check_v4_graph_memory(self, stage):
        # KV is already allocated. The original reserve includes KV, so only
        # its remainder must remain free; do not charge the allocation twice.
        minimum_free = max(0, self._v4_graph_reserve_bytes - self._v4_graph_kv_cache_bytes)
        free, total = torch.npu.mem_get_info()
        self._v4_graph_memory[stage] = {
            "free_bytes": int(free),
            "total_bytes": int(total),
            "allocated_bytes": int(torch.npu.memory_allocated()),
            "reserved_bytes": int(torch.npu.memory_reserved()),
            "required_free_bytes": minimum_free,
        }
        if free < minimum_free:
            raise RuntimeError(
                f"VQ2A8 graph memory guard failed at {stage}: free={free}, required={minimum_free}; "
                "KV, reserve and expert residency are not reduced automatically."
            )

    def set_v4_position_template_verification(self, enabled):
        """Acceptance-only full metadata comparison, disabled during serving."""
        if type(enabled) is not bool or self._v4_graph_forward_active or self._v4_graphs_failed:
            raise ValueError("Metadata verification requires an idle healthy model and boolean mode.")
        bank = self._v4_decoder_graph
        if bank is None or bank.position_template_adapter is None:
            raise ValueError("Position-template metadata has not been prepared.")
        torch.npu.synchronize()
        bank.position_template_adapter.verify_reference = enabled
        return bank.position_template_adapter.report()

    def set_v4_graph_enabled(self, enabled):
        """Validation-only same-engine eager reference; never recaptures."""
        if type(enabled) is not bool:
            raise ValueError("Decoder graph enabled must be a boolean.")
        if self._v4_graphs_failed or self._v4_graph_forward_active or not self._v4_graphs_ready:
            raise RuntimeError("Decoder graph mode can only change on a healthy, prepared, idle model.")
        torch.npu.synchronize()
        self._v4_graph_enabled = enabled
        return self.v4_graph_report()

    def v4_graph_report(self):
        return {
            "graph_scope": "V4_TP1_B1_DECODER_POSITION_SPECIALIZED",
            "effective_graph_mode": "decoder" if self._v4_graph_enabled else "none",
            "replay_stream_policy": "caller",
            "ready": self._v4_graphs_ready,
            "failed": self._v4_graphs_failed,
            "decoder": self._v4_decoder_graph.report() if self._v4_decoder_graph is not None else None,
            "preparation_memory": dict(self._v4_graph_memory),
        }

    def _retain_finite_flag(self, value):
        valid = torch.isfinite(value).all()
        self._measurement_valid = valid if self._measurement_valid is None else self._measurement_valid & valid

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if (
            not self._offline_loaded
            or input_ids is None
            or intermediate_tensors is not None
            or inputs_embeds is not None
        ):
            raise ValueError("VQ2A8 forward requires loaded TP1 weights and actual token IDs.")
        if self._v4_graphs_failed or self._v4_graph_forward_active:
            raise RuntimeError("VQ2A8 graph model is failed or already executing; no eager fallback.")
        context = get_forward_context()
        if self._v4_decoder_preparing:
            position = getattr(context, "vq2a8_capture_position", None)
            if type(position) is not int:
                raise ValueError("Decoder capture requires an explicit startup position.")
            return self._v4_decoder_graph.capture(position, input_ids, positions, context, self._v4_decoder_compute)
        # Scheduler CPU progress distinguishes decode from one-token prefill.
        # Dummy/profile runs do not set the marker, even with attention metadata.
        phase = getattr(context, "vq2a8_request_phase", "profile")
        if phase not in ("profile", "prefill", "decode"):
            raise ValueError("Invalid VQ2A8 request phase.")
        self._last_forward_is_request = phase != "profile" and not getattr(context, "in_profile_run", False)
        if self._last_forward_is_request and not self._v4_graphs_ready:
            raise RuntimeError("Prepare decoder graphs before serving; lazy capture is disabled.")
        self._v4_graph_forward_active = True
        try:
            if phase == "decode" and self._last_forward_is_request and self._v4_graph_enabled:
                hidden, valid = self._v4_decoder_graph.replay(input_ids, positions, context)
                self._measurement_valid = valid if self._measurement_valid is None else self._measurement_valid & valid
                return hidden
            result = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
            if self._last_forward_is_request:
                self._retain_finite_flag(result)
            return result
        except Exception:
            if self._last_forward_is_request:
                self._v4_graphs_failed = True
            raise
        finally:
            self._v4_graph_forward_active = False

    def compute_logits(self, hidden_states):
        logits = super().compute_logits(hidden_states)
        if logits is None:
            raise ValueError("Missing VQ2A8 model logits.")
        self._retain_finite_flag(logits)
        flags = [self._measurement_valid]
        if self._last_forward_is_request:
            # Graph replay contributes its aggregate validity above. Eager
            # prefill/reference execution contributes each layer's fresh flag.
            # Both must be checked before the sampler can accept any token.
            for layer in self.model.offline_owner.layers.values():
                state = getattr(layer, "_optimization", None)
                if state is None or state.valid is None:
                    self._v4_graphs_failed = True
                    raise ValueError("VQ2A8 layer is missing its device validity state.")
                flags.append(state.valid)
        if not bool(torch.stack(flags).all()):
            self._v4_graphs_failed = True
            raise ValueError("VQ2A8 forward failed validity checks; no output tokens are accepted.")
        return logits
