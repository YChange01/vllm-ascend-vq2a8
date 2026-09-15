# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in M=1 device routing with unchanged V1 packed projection arithmetic.

Multiple-token prefills retain V4's batched path. The default singleton path
is eager; graph preparation/replay is separately and explicitly selected.
Neither path uses FWHT, expanded expert weights, host route IDs or per-step
pointer-table uploads. Invalid device values are checked once at the model
output boundary. Native banks own their immutable tensor storage.
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock

import torch
import torch.nn.functional as F

from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_moe import route_vq2a8
from vllm_ascend.quantization.vq2a8_optimization import FastMoEState, OptimizationOptions
from vllm_ascend.quantization.vq2a8_reference import deepseek_v4_swiglu_reference
from vllm_ascend.quantization.vq2a8_runtime import VQ2_TP1_FIELDS
from vllm_ascend.quantization.vq2a8_v4_graph import GRAPH_REPLAY_STREAM_POLICIES, V4MoEDecodeGraph

DEVICE_ROUTE_PRESET = "device_route_decode"
MAX_DEVICE_ROUTE_SLOTS = 6


def require_device_route_library():
    """Check the new ABI before loading model-sized weights; old libraries work
    unchanged when the explicit device-route option is absent.
    """
    try:
        return torch.classes.vq2a8_ascendc.ResidentBank
    except (RuntimeError, AttributeError) as error:
        raise RuntimeError(
            "V4 device routing needs ResidentBank. Rebuild tools/build_vq2a8_ascendc.py "
            "--soc Ascend950DT_9582 --build-dir build/vq2a8-ascendc-v4-device-route "
            "and select that libvq2a8_ascendc.so; the original V4 library is unchanged."
        ) from error


def initialize_device_route_banks(runtime):
    """Create small device pointer tables once, without copying expert payloads."""
    runtime._require_ready()
    runtime.check_resident_integrity()
    if getattr(runtime, "_device_route_banks", None) is not None:
        return
    runtime._device_route_banks = create_device_route_banks(runtime)


def create_device_route_banks(runtime):
    """Build metadata on the current stream, sharing every resident payload.

    Unlike initialize_device_route_banks, this does not publish or replace the
    eager banks. Graph capture owns a separate metadata-only bank on its own
    managed NPU stream; native construction-stream guards remain unchanged.
    Payload producer readiness must be established by the caller first.
    """
    runtime._require_ready()
    alternate_factory = getattr(runtime, "_create_device_route_banks", None)
    if alternate_factory is not None:
        return alternate_factory()
    if runtime.device.type != "npu":
        raise ValueError("V4 device routing requires NPU; no CPU fallback.")
    bank_type = require_device_route_library()
    count = runtime.config.num_experts
    expert_ids = tuple(runtime.layer.expert_ids)
    if not 1 <= runtime.config.top_k <= MAX_DEVICE_ROUTE_SLOTS or any(i >= count for i in expert_ids):
        raise ValueError("V4 device routing requires 1..6 routes and an in-range expert inventory.")
    # An artifact may contain a sparse subset (e.g. a hash layer). Raw router
    # IDs map to dense bank slots; absent experts stay invalid, never expert 0.
    with torch.device("cpu"):
        mapping = torch.full((count,), -1, dtype=torch.int64, device="cpu")
        for slot, expert_id in enumerate(expert_ids):
            mapping[expert_id] = slot
    banks = {"lookup": mapping.to(runtime.device), "metadata_bytes": mapping.numel() * mapping.element_size()}
    for kind in ("gate_up", "down"):
        entries = [runtime._cache[expert_id][kind] for expert_id in expert_ids]
        spec = entries[0][1]
        geometry = (spec.rows, spec.columns, spec.rht_true_columns, spec.rht_block_size)
        if any((s.rows, s.columns, s.rht_true_columns, s.rht_block_size) != geometry for _, s in entries):
            raise ValueError("Resident bank experts must share V1 projection/preparation geometry.")
        bank = bank_type(*[[p[field] for p, _ in entries] for field in VQ2_TP1_FIELDS])
        banks[kind] = (bank, spec)
        banks["metadata_bytes"] += bank.metadata()[3]
    # Publish only after both banks have been constructed. Owners survive
    # baseline/candidate switching, so A/B does not reload or duplicate weights.
    return banks


def _record_tensor_stream(tensor, stream):
    # CPU backend injection exercises protocol without pretending to implement
    # NPU allocator events. The production path always records real NPU tensors.
    if tensor.device.type == "npu":
        tensor.record_stream(stream)


class DeviceRouteDecodeState(FastMoEState):
    """Keep the top-k slots on device, preserving duplicates and reduction order."""

    def __init__(self, runtime, *, profile=False):
        if getattr(runtime, "execution_policy", None) != "ascendc_v4":
            raise ValueError("Device-route decode is only supported by the V4 resident runtime.")
        runtime._require_ready()
        if getattr(runtime, "_device_route_banks", None) is None:
            raise ValueError("Initialize V4 device-route banks outside the forward/timing window first.")
        super().__init__(runtime, OptimizationOptions.preset("batched"), profile=profile)
        self.stats.update(singleton_forwards=0, batched_prefill_forwards=0, device_select_calls=0)
        self._decode_graph = None
        self._graph_started = False
        self._graph_prepare_error = None
        self._graph_stream = self._graph_backend = self._graph_banks = None
        self._graph_compute = None
        self._graph_replay_stream_policy = "owner"
        self._graph_metadata_bytes = self._graph_stream_bridges = 0
        self._graph_lock = Lock()

    @contextmanager
    def _graph_operation(self):
        if not self._graph_lock.acquire(blocking=False):
            raise RuntimeError("V4 MoE graph cannot be used concurrently or recursively.")
        try:
            yield
        finally:
            self._graph_lock.release()

    @torch.inference_mode()
    def prepare_graph(self, runtime, *, backend=None, replay_stream_policy="owner"):
        """Explicit startup-only capture; eager routing remains the default."""
        with self._graph_operation():
            return self._prepare_graph(runtime, backend=backend, replay_stream_policy=replay_stream_policy)

    def _prepare_graph(self, runtime, *, backend=None, replay_stream_policy="owner"):
        runtime._require_ready()
        if replay_stream_policy not in GRAPH_REPLAY_STREAM_POLICIES:
            raise ValueError("V4 MoE graph replay stream policy must be owner or caller.")
        if self._graph_started:
            raise RuntimeError("V4 MoE graph preparation is single-shot; no recapture.")
        self._graph_started = True
        self._graph_replay_stream_policy = replay_stream_policy
        try:
            if backend is None and runtime.device.type != "npu":
                raise ValueError("V4 MoE decode graph requires an NPU.")
            self._graph_backend = torch.npu if backend is None else backend
            caller = self._graph_backend.current_stream(runtime.device)
            capturing = getattr(self._graph_backend, "is_current_stream_capturing", None)
            if capturing is not None and capturing():
                raise RuntimeError("Prepare V4 MoE graph outside another capture.")
            # This is the selected production policy, not a failure-triggered
            # stream switch. Original eager banks and their stream stay intact.
            self._graph_stream = self._graph_backend.Stream(device=runtime.device)
            self._graph_stream.wait_stream(caller)
            with self._graph_backend.stream(self._graph_stream):
                self._graph_banks = create_device_route_banks(runtime)
                self._graph_metadata_bytes = self._graph_banks["metadata_bytes"]
                compute = DeviceRouteGraphCompute(runtime, banks=self._graph_banks)
                self._graph_compute = compute
                graph = V4MoEDecodeGraph(runtime.device, backend=self._graph_backend, signature=compute.signature)
                # Publish before capture so a partial failure remains safely closable.
                self._decode_graph = graph
                hidden = torch.zeros((1, runtime.config.hidden_size), device=runtime.device, dtype=torch.bfloat16)
                input_ids = torch.zeros((1,), device=runtime.device, dtype=torch.int64)
                # Always bind the startup caller, including owner-mode capture,
                # so synchronized A/B policy switches need no new graph/pool.
                graph.prepare(compute, hidden, input_ids, replay_stream=caller)
            caller.wait_stream(self._graph_stream)
            return self.graph_snapshot()
        except BaseException as error:
            self._graph_prepare_error = f"{type(error).__name__}: {error}"
            if self._decode_graph is not None:
                self._decode_graph._latch(error)
            raise

    def set_graph_replay_stream(self, policy):
        """Switch the same graph only at a fenced, non-concurrent boundary.

        Model callers additionally fence the complete request before switching
        every layer. The device fence also covers retained validity produced
        by an earlier owner-mode call on a third stream; tracking an unbounded
        stream inventory is unnecessary. Owner-mode callers remain responsible
        for ordinary producer/consumer ordering between their own streams.
        These fences are never part of the per-token replay path.
        """
        with self._graph_operation():
            if policy not in GRAPH_REPLAY_STREAM_POLICIES:
                raise ValueError("V4 MoE graph replay stream policy must be owner or caller.")
            if self._decode_graph is None:
                raise RuntimeError("Prepare V4 MoE graph before selecting its replay stream.")
            graph = self._decode_graph
            graph._require_open()
            if not graph.prepared:
                raise RuntimeError("Prepare V4 MoE graph before selecting its replay stream.")
            capturing = getattr(self._graph_backend, "is_current_stream_capturing", None)
            if capturing is not None and capturing():
                raise RuntimeError("Switch V4 MoE graph replay stream outside another capture.")
            if policy != self._graph_replay_stream_policy:
                try:
                    self._graph_backend.synchronize(graph.device)
                    graph._fence_streams()
                except BaseException as error:
                    graph._latch(error)
                    raise
                self._graph_replay_stream_policy = policy
            return self.graph_snapshot()

    @torch.inference_mode()
    def forward_graph(self, runtime, hidden, input_ids):
        with self._graph_operation():
            runtime._require_ready()
            if self._decode_graph is None:
                raise RuntimeError("Prepare V4 MoE graph before decode; lazy capture is disabled.")
            graph = self._decode_graph
            graph._require_open()
            self._graph_compute.check_runtime_contract(runtime)
            graph._check_inputs(hidden, input_ids)
            caller = self._graph_backend.current_stream(runtime.device)
            capturing = getattr(self._graph_backend, "is_current_stream_capturing", None)
            if capturing is not None and capturing():
                raise RuntimeError("V4 MoE graph does not support replay inside another capture.")
            policy = self._graph_replay_stream_policy
            if policy == "caller":
                # Reject unexpected callers before any copy/submission/latch.
                # Native bank guards still run on their construction stream
                # during capture; graph replay does not re-enter those calls.
                graph._current_stream(stream_policy="caller")
            try:
                if policy == "caller":
                    # Static inputs/private pool/payloads have strong graph
                    # owners until both capture and caller streams are fenced
                    # at close. Copies, replay and escaped clones are ordered
                    # on this one caller stream, needing no per-layer bridge.
                    output, valid = graph.replay(hidden, input_ids, stream_policy="caller")
                else:
                    cross_stream = caller.npu_stream != self._graph_stream.npu_stream
                    if cross_stream:
                        self._graph_stream.wait_stream(caller)
                        _record_tensor_stream(hidden, self._graph_stream)
                        _record_tensor_stream(input_ids, self._graph_stream)
                    with self._graph_backend.stream(self._graph_stream):
                        output, valid = graph.replay(hidden, input_ids)
                    if cross_stream:
                        caller.wait_stream(self._graph_stream)
                        _record_tensor_stream(output, caller)
                        _record_tensor_stream(valid, caller)
                        self._graph_stream_bridges += 1
                # The current flag escapes the graph pool before caller-stream
                # consumption. Python native counters remain eager-only.
                self.retain(valid)
                return output
            except BaseException as error:
                graph._latch(error)
                raise

    def graph_snapshot(self):
        if self._decode_graph is None:
            return {
                "scope": "moe_decode",
                "prepared": False,
                "captures": 0,
                "replays": 0,
                "owner_replays": 0,
                "caller_replays": 0,
                "replay_stream_policy": self._graph_replay_stream_policy,
                "warmups": 0,
                "entries": 0,
                "pool_count": 0,
                "failed": self._graph_prepare_error is not None,
                "failure": self._graph_prepare_error,
                "closed": False,
                "graph_functional_verified": False,
                "graph_performance_target_met": None,
                "full_model_graph_verified": False,
            }
        return {
            **self._decode_graph.snapshot(),
            "stream_policy": (
                "dedicated_graph_owner_with_caller_event_bridges"
                if self._graph_replay_stream_policy == "owner"
                else "fixed_caller_replay_without_per_layer_event_bridges"
            ),
            "replay_stream_policy": self._graph_replay_stream_policy,
            "stream_bridges": self._graph_stream_bridges,
            "graph_metadata_bytes": self._graph_metadata_bytes,
            "graph_payload_copy_bytes": 0,
        }

    def close_graph(self):
        with self._graph_operation():
            if self._decode_graph is not None:
                self._decode_graph.close()
            elif self._graph_stream is not None:
                # Includes a failure while constructing graph metadata or RHT.
                self._graph_stream.synchronize()
            self._graph_banks = None
            self._graph_compute = None
            self._graph_stream = None

    def report(self):
        return {
            **super().report(),
            "preset": DEVICE_ROUTE_PRESET,
            "singleton_route_host_reads": 0,
            "singleton_descriptor_h2d_bytes": 0,
            "scope": "M=1 device routing; multi-token prefill retains batched V4",
            "device_execution_verified": False,  # counters alone are not NPU acceptance
        }

    def _project(self, runtime, hidden, slots, kind):
        bank, spec = runtime._device_route_banks[kind]
        with self.scope("device_select"):
            weight_scale, weight_bias, signs, valid = bank.select(slots)
            self.retain((valid != 0).all())
            self.stats["device_select_calls"] += 1
        # Slot count is static host metadata, not device route data. Each row
        # still performs the original one-row RHT/bias GEMV and FP8 conversion.
        requests = [
            (
                hidden[i : i + 1],
                {"weight_scale": weight_scale[i], "weight_bias": weight_bias[i], "rht_sign": signs[i]},
                spec,
            )
            for i in range(slots.numel())
        ]
        with self.scope("preparation"):
            prepared = runtime._row_preparation.many(requests)
            self.stats["preparation_calls"] += 1
        quantized, scale, bias = (torch.cat(values).contiguous() for values in zip(*prepared))
        with self.scope("native_projection"):
            project = getattr(runtime, "project_v4_prepared", None)
            output, valid = (
                bank.project(quantized, scale, bias, slots)
                if project is None
                else project(bank, quantized, scale, bias, slots)
            )
            self.retain((valid != 0).all() & torch.isfinite(output).all())
        runtime.native_calls += slots.numel()
        runtime.native_rows += slots.numel()
        runtime.native_launches += 1  # projection only, same coverage definition as V1
        runtime.projection_rows += slots.numel()
        runtime.prepare_batches += slots.numel()
        return output

    def forward(self, runtime, hidden, input_ids):
        if hidden.ndim != 2 or hidden.shape[0] != 1:
            self.stats["batched_prefill_forwards"] += 1
            return super().forward(runtime, hidden, input_ids)
        if (
            hidden.shape[1] != runtime.config.hidden_size
            or hidden.dtype != torch.bfloat16
            or hidden.device != runtime.device
        ):
            raise ValueError("Device-route decode requires BF16 [1,hidden_size] on the resident device.")
        with self.scope("route"):
            logits = F.linear(hidden.float(), runtime.root["gate.weight"])
            weights, ids = route_vq2a8(
                logits,
                runtime.config.top_k,
                renormalize=runtime.config.renormalize,
                correction_bias=runtime.root.get("gate.bias"),
                hash_table=runtime.root.get("gate.tid2eid"),
                input_ids=input_ids,
                validity=self.retain,
                device_only=True,
            )
            lookup = runtime._device_route_banks["lookup"]
            ids = ids.reshape(-1)
            in_range = (ids >= 0) & (ids < lookup.numel())
            mapped = lookup.index_select(0, ids.clamp(0, lookup.numel() - 1))
            slots = torch.where(in_range, mapped, -1).contiguous()
            self.retain((slots >= 0).all())
        with self.scope("gate_up"):
            gate = self._project(runtime, hidden.expand(slots.numel(), -1), slots, "gate_up")
        with self.scope("swiglu"):
            activation = deepseek_v4_swiglu_reference(gate, runtime.config.swiglu_limit)
        with self.scope("down"):
            values = self._project(runtime, activation, slots, "down")
        with self.scope("mix_shared"):
            result = (values.reshape(1, slots.numel(), hidden.shape[1]).float() * weights.unsqueeze(-1)).sum(1)
            result *= runtime.config.routed_scale
            if runtime.config.num_shared:
                result += runtime.shared(hidden).float()
            result = result.to(hidden.dtype)
            self.retain(torch.isfinite(result).all())
        runtime.native_experts += slots.numel()
        self.stats["singleton_forwards"] += 1
        self.stats["jobs"] += slots.numel()
        self.stats["rows"] += slots.numel()
        self.stats["windows"] += 1
        self.row_histogram[1] = self.row_histogram.get(1, 0) + slots.numel()
        return result


class DeviceRouteGraphCompute:
    """Fixed-structure V1-arithmetic MoE with one fresh device validity output.

    Construction is startup-only. Each projection owns a prebuilt RHT constant
    even when gate/up and down use different blocks. __call__ does not mutate
    the eager optimization state's validity, counters, or preparation cache.
    All bank/root owners remain reachable for the graph's entire lifetime.
    """

    def __init__(self, runtime, *, banks=None):
        if getattr(runtime, "execution_policy", None) != "ascendc_v4":
            raise ValueError("V4 graph compute requires the V4 resident runtime.")
        runtime._require_ready()
        if not isinstance(getattr(runtime, "_optimization", None), DeviceRouteDecodeState):
            raise ValueError("Configure device_route_decode before constructing V4 graph compute.")
        if getattr(runtime, "_device_route_banks", None) is None:
            raise ValueError("Initialize device-route banks before constructing V4 graph compute.")
        self.runtime = runtime
        self.root = dict(runtime.root)
        self.banks = dict(runtime._device_route_banks if banks is None else banks)
        self.config = runtime.config
        self.preparations = {}
        geometries = {}
        for kind in ("gate_up", "down"):
            _, spec = self.banks[kind]
            preparation = getattr(runtime, "make_v4_preparation", RowwiseVQ2A8Preparation)(compact=True)
            preparation.prepare_for_graph(runtime.device, spec.rht_block_size)
            self.preparations[kind] = preparation
            geometries[kind] = [spec.rows, spec.columns, spec.rht_true_columns, spec.rht_block_size]
        self.signature = {
            "layer": getattr(runtime, "layer_index", None),
            "compute_backend": getattr(runtime, "v4_compute_backend", "v1"),
            "activation_preparation": getattr(runtime, "v4_activation_preparation", "rowwise"),
            "activation_reorder": getattr(runtime, "v4_activation_reorder", "scalar"),
            "top_k": self.config.top_k,
            "hidden_size": self.config.hidden_size,
            "hash_route": self.root.get("gate.tid2eid") is not None,
            "num_shared": self.config.num_shared,
            "renormalize": self.config.renormalize,
            "routed_scale": self.config.routed_scale,
            "swiglu_limit": self.config.swiglu_limit,
            "projection_geometry": geometries,
        }
        self._runtime_contract = self._contract(runtime)

    @staticmethod
    def _contract(runtime):
        """Host metadata only; payload contents remain immutable by contract."""
        config = runtime.config
        return (
            id(runtime),
            getattr(runtime, "v4_compute_backend", "v1"),
            getattr(runtime, "v4_activation_preparation", "rowwise"),
            getattr(runtime, "v4_activation_reorder", "scalar"),
            id(config),
            (
                config.top_k,
                config.hidden_size,
                config.num_shared,
                config.renormalize,
                config.routed_scale,
                config.swiglu_limit,
            ),
            id(runtime._device_route_banks),
            tuple(
                (
                    kind,
                    id(runtime._device_route_banks[kind][0]),
                    tuple(
                        getattr(runtime._device_route_banks[kind][1], name)
                        for name in ("rows", "columns", "rht_true_columns", "rht_block_size")
                    ),
                )
                for kind in ("gate_up", "down")
            ),
            tuple(
                (name, id(tensor), tensor.data_ptr(), tuple(tensor.shape), tensor.dtype, tensor.device)
                for name, tensor in sorted(runtime.root.items())
            ),
        )

    def check_runtime_contract(self, runtime):
        if self._contract(runtime) != self._runtime_contract:
            raise RuntimeError("V4 MoE graph runtime/root/geometry signature changed; no implicit recapture.")

    def _project(self, hidden, slots, kind, retain):
        bank, spec = self.banks[kind]
        weight_scale, weight_bias, signs, valid = bank.select(slots)
        retain((valid != 0).all())
        requests = [
            (
                hidden[i : i + 1],
                {"weight_scale": weight_scale[i], "weight_bias": weight_bias[i], "rht_sign": signs[i]},
                spec,
            )
            for i in range(slots.numel())
        ]
        prepared = self.preparations[kind].many(requests, validity=retain)
        quantized, scale, bias = (torch.cat(values).contiguous() for values in zip(*prepared))
        project = getattr(self.runtime, "project_v4_prepared", None)
        output, valid = (
            bank.project(quantized, scale, bias, slots)
            if project is None
            else project(bank, quantized, scale, bias, slots)
        )
        retain((valid != 0).all() & torch.isfinite(output).all())
        return output

    @torch.inference_mode()
    def __call__(self, hidden, input_ids):
        if (
            hidden.shape != (1, self.config.hidden_size)
            or hidden.dtype != torch.bfloat16
            or hidden.device != self.runtime.device
            or input_ids is None
            or input_ids.shape != (1,)
            or input_ids.dtype not in (torch.int32, torch.int64)
            or input_ids.device != hidden.device
        ):
            raise ValueError("V4 graph compute requires BF16 B1 hidden and one device integer token ID.")
        # This bounded list exists only during compute/capture; no prior flag
        # is an input, so a later valid replay cannot replay stale validity.
        flags = []
        logits = F.linear(hidden.float(), self.root["gate.weight"])
        weights, ids = route_vq2a8(
            logits,
            self.config.top_k,
            renormalize=self.config.renormalize,
            correction_bias=self.root.get("gate.bias"),
            hash_table=self.root.get("gate.tid2eid"),
            input_ids=input_ids,
            validity=flags.append,
            device_only=True,
        )
        lookup = self.banks["lookup"]
        ids = ids.reshape(-1)
        in_range = (ids >= 0) & (ids < lookup.numel())
        mapped = lookup.index_select(0, ids.clamp(0, lookup.numel() - 1))
        slots = torch.where(in_range, mapped, -1).contiguous()
        flags.append((slots >= 0).all())
        gate = self._project(hidden.expand(slots.numel(), -1), slots, "gate_up", flags.append)
        activation = deepseek_v4_swiglu_reference(gate, self.config.swiglu_limit)
        values = self._project(activation, slots, "down", flags.append)
        result = (values.reshape(1, slots.numel(), hidden.shape[1]).float() * weights.unsqueeze(-1)).sum(1)
        result *= self.config.routed_scale
        if self.config.num_shared:
            result += self.runtime.shared(hidden).float()
        result = result.to(hidden.dtype)
        flags.append(torch.isfinite(result).all())
        return result, torch.stack(flags).all()
