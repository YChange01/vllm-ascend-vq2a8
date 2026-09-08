# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in TP1 experiments. The accepted reference path stays unchanged.

One host route read per layer remains necessary for the bounded lazy expert
cache. This is NOT a device-only router or a full-model graph implementation.
"""

from contextlib import nullcontext
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from vllm_ascend.quantization.vq2a8_moe import route_vq2a8
from vllm_ascend.quantization.vq2a8_reference import deepseek_v4_swiglu_reference

MAX_FORWARD_TOKENS = 128
MAX_PROJECTION_ROWS = 32
MAX_PROJECTION_JOBS = 6


@dataclass(frozen=True)
class OptimizationOptions:
    window: int = 2
    preparation: str = "rowwise"
    shared_batch: bool = False
    pipeline: bool = False
    prepare_graph: bool = False

    @classmethod
    def preset(cls, name):
        presets = {
            "fast": cls(),
            # Keep the shared-expert GEMM geometry unchanged until its profile
            # justifies a separately validated arithmetic change.
            "batched": cls(window=128),
            "fwht": cls(window=128, preparation="fwht"),
            # Isolate the native pipeline benefit even if FWHT fails the exact
            # logits gate. The final graph candidate composes both experiments.
            "pipeline": cls(window=128, pipeline=True),
            "prepare_graph": cls(window=128, preparation="fwht", pipeline=True, prepare_graph=True),
        }
        if name not in presets:
            raise ValueError(f"Unknown optimization preset: {name}")
        return presets[name]


PRESETS = ("fast", "batched", "fwht", "pipeline", "prepare_graph")


def route_plan(host_ids, available, *, max_rows=MAX_PROJECTION_ROWS):
    """Each unique (token, expert) once, retaining every duplicate top-k slot.

    Host IDs have already been materialized by the cache controller. Return
    bounded jobs, each with token rows and a mapping to flat output slots.
    """
    if type(max_rows) is not int or not 1 <= max_rows <= MAX_PROJECTION_ROWS:
        raise ValueError("Invalid projection row bound")
    if not host_ids or len(host_ids) > MAX_FORWARD_TOKENS:
        raise ValueError("Require 1..128 routed tokens")
    top_k = len(host_ids[0])
    if not 1 <= top_k <= 6 or any(len(row) != top_k for row in host_ids):
        raise ValueError("Require rectangular 1..6 routes per token")
    experts = {}
    for token, ids in enumerate(host_ids):
        for slot, expert in enumerate(ids):
            if type(expert) is not int or expert not in available:
                raise ValueError("Router selected an unavailable expert")
            experts.setdefault(expert, {}).setdefault(token, []).append(token * top_k + slot)
    jobs = []
    for expert, rows in experts.items():
        items = list(rows.items())
        for start in range(0, len(items), max_rows):
            block = items[start : start + max_rows]
            sources, destinations = [], []
            for row, (_, slots) in enumerate(block):
                sources.extend([row] * len(slots))
                destinations.extend(slots)
            jobs.append((expert, [token for token, _ in block], sources, destinations))
    return jobs


class FastMoEState:
    def __init__(self, runtime, options, *, profile=False):
        self.options = options
        self.profile = profile
        self.valid = None
        self.stats = dict(route_host_reads=0, jobs=0, rows=0, windows=0, preparation_calls=0)
        self.row_histogram = {}
        # The offline owner alone owns these immutable, strictly loaded roots.
        # Reject invalid constant roots before entering the timing window.
        # The router also retains a device validity flag on each invocation.
        bias = runtime.root.get("gate.bias")
        if bias is not None and not bool(torch.isfinite(bias).all()):
            raise ValueError("Non-finite constant router bias")

    def retain(self, valid):
        self.valid = valid if self.valid is None else self.valid & valid

    def scope(self, name):
        return torch.profiler.record_function(f"vq2a8::{name}") if self.profile else nullcontext()

    def report(self):
        return {"options": asdict(self.options), **self.stats, "row_histogram": dict(self.row_histogram)}

    def forward(self, runtime, hidden, input_ids):
        if (
            hidden.ndim != 2
            or not 1 <= hidden.shape[0] <= MAX_FORWARD_TOKENS
            or hidden.shape[1] != runtime.config.hidden_size
        ):
            raise ValueError("Fast MoE requires 1..128 tokens")
        if hidden.dtype != torch.bfloat16 or hidden.device != runtime.device:
            raise ValueError("Fast MoE requires BF16 input on the runtime device")
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
            )
            # Required for safe cache admission. Never drop bounds/membership checks.
            host_ids = ids.cpu().tolist()
            self.stats["route_host_reads"] += 1
        outputs = []
        window = self.options.window
        batch_limit = min(MAX_PROJECTION_JOBS, runtime.cache_experts)
        for start in range(0, hidden.shape[0], window):
            x = hidden[start : start + window]
            w = weights[start : start + window]
            with self.scope("plan_gather"):
                jobs = route_plan(host_ids[start : start + window], set(runtime.layer.expert_ids))
                self.stats["windows"] += 1
                self.stats["jobs"] += len(jobs)
                for _, rows, _, _ in jobs:
                    self.row_histogram[len(rows)] = self.row_histogram.get(len(rows), 0) + 1
                    self.stats["rows"] += len(rows)
                slots = torch.empty((x.shape[0] * w.shape[1], x.shape[1]), device=x.device, dtype=x.dtype)
                # One packed index upload per window, instead of per expert/slot.
                token_rows, source_rows, dest_rows, counts = [], [], [], []
                batch_row_offset = 0
                for index, (_, tokens, sources, dests) in enumerate(jobs):
                    if index % batch_limit == 0:
                        batch_row_offset = 0
                    token_rows.extend(tokens)
                    source_rows.extend(batch_row_offset + i for i in sources)
                    dest_rows.extend(dests)
                    counts.append(len(tokens))
                    batch_row_offset += len(tokens)
                all_indices = torch.tensor(token_rows + source_rows + dest_rows, dtype=torch.int64, device=x.device)
                nr, ns = len(token_rows), len(source_rows)
                selected = x.index_select(0, all_indices[:nr]).split(counts)
            slot_offset = 0
            for j in range(0, len(jobs), batch_limit):
                group = jobs[j : j + batch_limit]
                # Do not pin more experts than the cache can retain. Refill still
                # fences outstanding device work before evicting any payload.
                payloads = [runtime._get_expert(job[0]) for job in group]
                rows = selected[j : j + len(group)]
                with self.scope("gate_up"):
                    gate = runtime._projections_many([(a, *p["gate_up"]) for a, p in zip(rows, payloads)])
                with self.scope("swiglu"):
                    gate_counts = [g.shape[0] for g in gate]
                    activation = deepseek_v4_swiglu_reference(torch.cat(gate), runtime.config.swiglu_limit)
                    activation = activation.split(gate_counts)
                with self.scope("down"):
                    values = runtime._projections_many([(a, *p["down"]) for a, p in zip(activation, payloads)])
                    runtime.native_experts += len(group)
                with self.scope("slot_write"):
                    values = torch.cat(values)
                    self.retain(torch.isfinite(values).all())
                    count = sum(len(job[3]) for job in group)
                    source = all_indices[nr + slot_offset : nr + slot_offset + count]
                    dest = all_indices[nr + ns + slot_offset : nr + ns + slot_offset + count]
                    # Destinations are unique. Avoid atomic add/reordered reductions.
                    slots.index_copy_(0, dest, values.index_select(0, source))
                    slot_offset += count
                del payloads, values, gate, activation
            with self.scope("mix_shared"):
                result = (slots.reshape(x.shape[0], w.shape[1], x.shape[1]).float() * w.unsqueeze(-1)).sum(1)
                result *= runtime.config.routed_scale
                if runtime.config.num_shared:
                    if self.options.shared_batch:
                        shared = runtime.shared(x)
                    else:
                        shared = torch.cat([runtime.shared(chunk) for chunk in x.split(runtime.token_chunk)])
                    result += shared.float()
                result = result.to(hidden.dtype)
                self.retain(torch.isfinite(result).all())
                outputs.append(result)
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


def configure_runtime(runtime, preset, *, profile=False):
    """Switch outside timed work, after the caller's device completion fence.

    Retain each bounded preset's preparation cache across AB/BA trials. A
    baseline comparison must not make a graph candidate recapture every time.
    """
    from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
    from vllm_ascend.quantization.vq2a8_activation_fast import BatchedFWHTPreparation, PreparationGraph

    if not hasattr(runtime, "_baseline_preparation"):
        runtime._baseline_preparation = getattr(runtime, "_row_preparation", RowwiseVQ2A8Preparation())
        runtime._optimization_states = {}
    if preset is None:
        runtime._optimization = None
        runtime._row_preparation = runtime._baseline_preparation
        return
    options = OptimizationOptions.preset(preset)
    if preset not in runtime._optimization_states:
        state = FastMoEState(runtime, options, profile=profile)
        if options.preparation == "fwht":
            preparation = BatchedFWHTPreparation(validity=state.retain)
            if options.prepare_graph:
                preparation = PreparationGraph(preparation)
        else:
            preparation = RowwiseVQ2A8Preparation(compact=True, validity=state.retain)
        runtime._optimization_states[preset] = (state, preparation)
    state, preparation = runtime._optimization_states[preset]
    state.valid, state.profile = None, profile
    runtime._optimization, runtime._row_preparation = state, preparation


def install_profile_hooks(model, enabled):
    """Optional profiler ranges for attention and root linears, no math change."""
    for handle in getattr(model, "_optimization_profile_handles", []):
        handle.remove()
    model._optimization_profile_handles = []
    if not enabled:
        return
    for name, module in model.named_modules():
        if not (name.endswith("self_attn") or hasattr(module, "quant_method")):
            continue
        stack = []

        def before(_module, _args, *, stack=stack, name=name):
            scope = torch.profiler.record_function(f"vq2a8::module::{name}")
            scope.__enter__()
            stack.append(scope)

        def after(_module, _args, _output, *, stack=stack):
            if stack:
                stack.pop().__exit__(None, None, None)

        model._optimization_profile_handles.extend(
            (
                module.register_forward_pre_hook(before),
                module.register_forward_hook(after, always_call=True),
            )
        )
