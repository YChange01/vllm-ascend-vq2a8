# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Instance-local B1 table-upload/slot-mapping fusion, not a new model runner.

The original _prepare_inputs executes in full. Only its two block-table calls
are batched; all token preparation, CPU mirrors and request bookkeeping remain
owned by the original runner. Every invocation uploads a fresh live row snapshot.
"""

from contextlib import nullcontext
from functools import wraps
from threading import Lock

import numpy as np
import torch

INPUT_MODES = ("general", "b1_packed")
MAX_GROUPS = 16
MAX_COLUMNS = 4096
MAX_TOKENS = 4096
DISALLOWED_FLAGS = (
    "use_cp",
    "use_async_scheduling",
    "use_async_spec_decode",
    "is_multimodal_model",
    "enable_prompt_embeds",
    "supports_mm_inputs",
    "is_mm_prefix_lm",
    "enable_hamming_sparse",
    "uses_mrope",
    "uses_xdrope_dim",
    "_has_gdn",
)


def _contract(tensor):
    return (
        id(tensor),
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.dtype,
        tensor.device,
    )


def _snapshot_upload(rows, device):
    # Neither a view of the mutable runner CPU table nor a reused pinned slab.
    # torch's pinned-memory transfer owns/records the allocation until DMA ends.
    snapshot = torch.cat([row.reshape(-1) for row in rows]).pin_memory()
    return snapshot.to(device=device, non_blocking=True)


def _native_factory(tables, slots, sizes, limits):
    ops = torch.ops.vq2a8_ascendc_v4_v2
    version = ops.decoder_input_plan_version()
    if type(version) is not int or version != 1:
        raise RuntimeError("Decoder input plan requires native ABI 1; no implicit fallback")
    return torch.classes.vq2a8_ascendc_v4_v2.DecoderInputPlan(tables, slots, sizes, limits)


class DecoderInputPlan:
    def __init__(self, runner, *, native_factory=None, upload=None):
        self.runner = runner
        self.batch = runner.input_batch
        self.group = self.batch.block_table
        self.original_prepare = runner._prepare_inputs
        self.original_commit = self.group.commit_block_table
        self.original_slots = self.group.compute_slot_mapping
        self.lock = Lock()
        self.active = False
        self.enabled = True
        self.committed = False
        self.mapped = False
        self.hits = self.fallbacks = self.uploads = self.slot_calls = 0
        self.upload = _snapshot_upload if upload is None else upload
        self.blocks = tuple(self.group.block_tables)
        if not 1 <= len(self.blocks) <= MAX_GROUPS:
            raise ValueError("B1 input plan requires 1..16 KV groups")
        self.owners = []
        device = None
        spans = []
        for block in self.blocks:
            cpu, table, slots = block.block_table.cpu, block.block_table.gpu, block.slot_mapping.gpu
            if (
                cpu.device.type != "cpu"
                or cpu.dtype != torch.int32
                or cpu.ndim != 2
                or table.dtype != torch.int32
                or table.ndim != 2
                or table.shape != cpu.shape
                or not cpu.is_contiguous()
                or not table.is_contiguous()
                or slots.dtype != torch.int32
                or slots.ndim != 1
                or not slots.is_contiguous()
                or not 1 <= table.shape[1] <= MAX_COLUMNS
                or table.shape[0] < 1
                or not 1 <= block.max_num_batched_tokens <= min(MAX_TOKENS, slots.numel())
                or block.block_size <= 0
                or block.is_mamba_group
                or block.pcp_world_size != 1
                or block.dcp_world_size != 1
            ):
                raise ValueError("Unsupported B1 table/slot geometry")
            if device is not None and (table.device != device or slots.device != device):
                raise ValueError("B1 input plan requires one device")
            device = table.device
            if slots.device != device:
                raise ValueError("B1 input plan slots changed device")
            for value in (table, slots):
                start, end = value.data_ptr(), value.data_ptr() + value.numel() * value.element_size()
                if any(start < previous_end and previous_start < end for previous_start, previous_end in spans):
                    raise ValueError("B1 input plan cannot alias table/slot storage")
                spans.append((start, end))
            self.owners.append(
                (block, cpu, table, slots, _contract(cpu), _contract(table), _contract(slots), self._geometry(block))
            )
        self.device = device
        factory = _native_factory if native_factory is None else native_factory
        self.native = factory(
            [x[2] for x in self.owners],
            [x[3] for x in self.owners],
            [b.block_size for b in self.blocks],
            [b.max_num_batched_tokens for b in self.blocks],
        )

    @staticmethod
    def _geometry(block):
        return (
            block.block_size,
            block.physical_block_size,
            block.blocks_per_phys_block,
            block.use_hybrid_blocks,
            block.max_num_batched_tokens,
            block.is_mamba_group,
            block.pcp_world_size,
            block.dcp_world_size,
            block.cp_kv_cache_interleave_size,
        )

    def _phase(self, name):
        recorder = getattr(self.runner, "_vq2a8_host_profile_recorder", None)
        return nullcontext() if recorder is None else recorder.phase(name)

    def _owners_match(self):
        if self.runner.input_batch is not self.batch or self.batch.block_table is not self.group:
            return False
        if len(self.group.block_tables) != len(self.blocks):
            return False
        for index, (block, cpu, table, slots, cc, tc, sc, geometry) in enumerate(self.owners):
            if (
                self.group.block_tables[index] is not block
                or block.block_table.cpu is not cpu
                or block.block_table.gpu is not table
                or block.slot_mapping.gpu is not slots
                or self._geometry(block) != geometry
                or _contract(cpu) != cc
                or _contract(table) != tc
                or _contract(slots) != sc
            ):
                return False
        return True

    def _eligible(self, scheduler, scheduled):
        if not self.enabled:
            return False
        runner, batch = self.runner, self.runner.input_batch
        config = runner.vllm_config
        parallel = config.parallel_config
        for values in (scheduled, batch.num_computed_tokens_cpu, batch.num_prompt_tokens):
            if (
                not isinstance(values, np.ndarray)
                or values.ndim != 1
                or values.size < 1
                or values.dtype.kind not in "iu"
            ):
                return False
        if (
            any(getattr(runner, name, False) for name in DISALLOWED_FLAGS)
            or any(
                getattr(config, name, None) is not None
                for name in ("speculative_config", "lora_config", "kv_transfer_config")
            )
            or any(getattr(runner, name, None) is not None for name in ("speculative_config", "lora_config"))
            or any(
                getattr(parallel, name, 1) != 1
                for name in (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "data_parallel_size",
                    "prefill_context_parallel_size",
                    "decode_context_parallel_size",
                )
            )
            or config.scheduler_config.max_num_seqs != 1
            or config.scheduler_config.async_scheduling
            or getattr(config.cache_config, "enable_prefix_caching", False)
            or getattr(config.cache_config, "kv_sharing_fast_prefill", False)
            or getattr(runner, "num_accepted_tokens_event", None) is not None
            or batch.num_reqs != 1
            or len(batch.req_ids) != 1
            or scheduler.total_num_scheduled_tokens != 1
            or scheduled.shape != (1,)
            or int(scheduled[0]) != 1
            or scheduler.scheduled_spec_decode_tokens
            or scheduler.num_scheduled_tokens != {batch.req_ids[0]: 1}
            or batch.req_prompt_embeds
            or not self._owners_match()
        ):
            return False
        # The live CPU scheduler state, not the previous attn_state/position.
        position = int(batch.num_computed_tokens_cpu[0])
        if position <= 0 or position < int(batch.num_prompt_tokens[0]):
            return False
        return all(position // block.block_size < table.shape[1] for block, _, table, *_ in self.owners)

    def prepare(self, scheduler_output, num_scheduled_tokens):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("B1 input plan cannot overlap/reenter input preparation")
        try:
            with self._phase("input_plan_guard"):
                eligible = self._eligible(scheduler_output, num_scheduled_tokens)
            self.active, self.committed, self.mapped = eligible, False, False
            if eligible:
                self.hits += 1
            else:
                self.fallbacks += 1
            result = self.original_prepare(scheduler_output, num_scheduled_tokens)
            if eligible and not (self.committed and self.mapped):
                raise RuntimeError("Original input preparation did not execute both planned operations")
            return result
        finally:
            self.active = False
            self.lock.release()

    def commit(self, num_reqs):
        if not self.active:
            return self.original_commit(num_reqs)
        if num_reqs != 1 or self.committed or not self._owners_match():
            raise RuntimeError("B1 input commit contract changed during preparation")
        with self._phase("input_table_pack_upload"):
            packed = self.upload([cpu[0] for _, cpu, *_ in self.owners], self.device)
        with self._phase("input_table_scatter"):
            self.native.copy_rows(packed)
        self.committed = True
        self.uploads += 1

    def slots(
        self, num_reqs, query_start_loc, positions, positions_compressed_list=None, req_indices_compressed_list=None
    ):
        if not self.active:
            return self.original_slots(
                num_reqs, query_start_loc, positions, positions_compressed_list, req_indices_compressed_list
            )
        if (
            num_reqs != 1
            or not self.committed
            or self.mapped
            or positions_compressed_list is not None
            or req_indices_compressed_list is not None
            or not self._owners_match()
            or positions.shape != (1,)
            or positions.dtype != torch.int64
            or query_start_loc.shape != (2,)
            or query_start_loc.dtype != torch.int32
        ):
            raise RuntimeError("B1 slot mapping contract changed during preparation")
        with self._phase("input_grouped_slot_mapping"):
            self.native.slot_mapping(query_start_loc, positions)
        self.mapped = True
        self.slot_calls += 1

    def report(self):
        return {
            "mode": "b1_packed",
            "scope": "block_table_upload_and_grouped_slot_mapping",
            "enabled": self.enabled,
            "general_prepare_inputs_preserved": True,
            "dynamic_rows_cached": False,
            "kv_groups": len(self.blocks),
            "fastpath_calls": self.hits,
            "fallback_calls": self.fallbacks,
            "packed_uploads": self.uploads,
            "grouped_slot_calls": self.slot_calls,
            "hardware_verified": False,
        }

    def set_enabled(self, enabled):
        if type(enabled) is not bool or self.active or self.lock.locked():
            raise ValueError("B1 input toggle requires an idle plan and a bool")
        self.enabled = enabled

    def detach(self):
        if self.active or self.lock.locked():
            raise RuntimeError("Cannot detach an active B1 input plan")
        runner, group = self.runner, self.group
        if (
            runner._prepare_inputs is not self.prepare_wrapper
            or group.commit_block_table is not self.commit_wrapper
            or group.compute_slot_mapping is not self.slot_wrapper
        ):
            raise RuntimeError("B1 instance wrappers changed before detach")
        runner._prepare_inputs = self.original_prepare
        group.commit_block_table = self.original_commit
        group.compute_slot_mapping = self.original_slots
        del runner._vq2a8_decoder_input_plan


def install_decoder_input_plan(runner, mode="general", *, native_factory=None, upload=None):
    if mode not in INPUT_MODES:
        raise ValueError("decoder_input_mode must be general or b1_packed")
    if mode == "general":
        return None
    if hasattr(runner, "_vq2a8_decoder_input_plan"):
        raise RuntimeError("Decoder input plan is already installed")
    plan = DecoderInputPlan(runner, native_factory=native_factory, upload=upload)

    @wraps(plan.original_prepare)
    def prepare(*args, **kwargs):
        return plan.prepare(*args, **kwargs)

    plan.prepare_wrapper = prepare
    plan.commit_wrapper = plan.commit
    plan.slot_wrapper = plan.slots
    runner._prepare_inputs = plan.prepare_wrapper
    plan.group.commit_block_table = plan.commit_wrapper
    plan.group.compute_slot_mapping = plan.slot_wrapper
    runner._vq2a8_decoder_input_plan = plan
    return plan
