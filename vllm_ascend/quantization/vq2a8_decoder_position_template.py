# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Closed A5/B1 decoder metadata templates, never a general metadata cache.

The original builder supplies every startup template. Only position-derived
fields are retained: lengths, RoPE and SAS/QLI descriptors. Physical block
tables and slot maps are always taken from the current input batch. Prefill
and eager reference runs continue to use the original builder.
"""

from dataclasses import fields
from functools import wraps

import torch

from vllm_ascend.quantization.vq2a8_v4_decoder_graph import (
    MAX_DECODER_GRAPH_CONTEXT,
    FastPlannedDecoderMetadataBuffers,
    _is_rope_proxy,
    _tensor_contract,
    decode_position,
)

DSA_DESCRIPTOR_LENGTH = 1024

ROOT_FIELDS = frozenset(
    [
        "num_actual_tokens",
        "slot_mapping",
        "query_start_loc",
        "seq_lens",
        "block_tables",
        "sin",
        "cos",
        "num_decodes",
        "num_decode_tokens",
        "num_prefills",
        "num_input_tokens",
        "query_lens",
        "head_dim",
        "attn_mask",
        "attn_state",
        "decode",
        "prefill",
        "reshape_cache_event",
        "hadamard",
        "start_pos",
    ]
)
DECODE_FIELDS = frozenset(
    [
        "input_positions",
        "block_table",
        "seq_lens",
        "max_seqlen_kv",
        "max_seqlen_q",
        "seq_lens_list",
        "max_seq_lens",
        "slot_mapping",
        "block_size",
        "num_compressed_tokens",
        "query_start_loc",
        "query_start_loc_cpu",
        "attn_mask",
        "sin",
        "cos",
        "full_compress_sin",
        "full_compress_cos",
        "cp_seq_len",
        "batch_seq_mask",
        "start_pos",
        "num_reqs_actual",
        "sas_metadata",
        "qli_metadata",
    ]
)
BUILD_ARGUMENTS = frozenset(
    [
        "num_tokens",
        "num_reqs",
        "max_query_len",
        "num_tokens_padded",
        "num_reqs_padded",
        "ubatch_slices",
        "logits_indices",
        "use_spec_decode",
        "for_cudagraph_capture",
        "num_scheduled_tokens",
        "num_scheduled_tokens_np",
        "cascade_attn_prefix_lens",
    ]
)


def _require_schema(value, name, expected):
    if (
        type(value).__module__ != "vllm_ascend.attention.dsa_v1"
        or type(value).__name__ != name
        or frozenset(field.name for field in fields(value)) != expected
    ):
        raise ValueError(f"Position template requires the exact supported {name} schema.")


def _cpu_values(tensor, expected, name):
    # Never inspect NPU data; these are the scheduler's existing CPU buffers.
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.tolist() != expected:
        raise ValueError(f"Position template CPU {name} changed.")


def _check_schema(metadata, position):
    if type(metadata) is not dict or not metadata or any(type(key) is not str for key in metadata):
        raise ValueError("Position template requires a per-layer DSA metadata dictionary.")
    if decode_position(metadata, MAX_DECODER_GRAPH_CONTEXT) != position:
        raise ValueError("Position template metadata position changed.")
    checked = set()
    for value in metadata.values():
        if id(value) in checked:
            continue
        checked.add(id(value))
        _require_schema(value, "AscendDSAMetadata", ROOT_FIELDS)
        decode = value.decode
        _require_schema(decode, "AscendDSADecodeMetadata", DECODE_FIELDS)
        if value.num_input_tokens != 1 or getattr(value.attn_state, "name", None) != "DecodeOnly":
            raise ValueError("Position template requires unpadded DecodeOnly metadata.")
        if any(
            getattr(value, name) is not None
            for name in ("slot_mapping", "block_tables", "attn_mask", "prefill", "reshape_cache_event", "start_pos")
        ):
            raise ValueError("Position template has unsupported root metadata fields.")
        if any(getattr(decode, name) is not None for name in ("attn_mask", "cp_seq_len", "batch_seq_mask")):
            raise ValueError("Position template does not support CP/masked decode metadata.")
        if (
            decode.num_reqs_actual != 1
            or decode.num_compressed_tokens != 1
            or decode.max_seqlen_q != 1
            or decode.max_seqlen_kv != position + 1
            or decode.max_seq_lens != position + 1
        ):
            raise ValueError("Position template scalar geometry is not B1 position-specialized.")
        _cpu_values(value.query_lens, [1], "query lengths")
        _cpu_values(decode.query_start_loc_cpu, [0, 1], "query starts")
        for container in (value, decode):
            if not _is_rope_proxy(container.cos) or not _is_rope_proxy(container.sin):
                raise ValueError("Position template requires the closed DSA rotary proxy.")
        for name, shape, dtype in (
            ("input_positions", (1,), torch.int64),
            ("seq_lens", (1,), torch.int32),
            ("query_start_loc", (2,), torch.int32),
            ("start_pos", (1,), torch.int32),
            ("sas_metadata", (DSA_DESCRIPTOR_LENGTH,), torch.int32),
            ("qli_metadata", (DSA_DESCRIPTOR_LENGTH,), torch.int32),
        ):
            tensor = getattr(decode, name)
            if not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.dtype != dtype:
                raise ValueError(f"Position template unsupported tensor geometry: {name}.")
        for name in ("block_table", "slot_mapping"):
            tensor = getattr(decode, name)
            if tensor is None and name == "slot_mapping":
                continue
            shape_ok = tensor.ndim == (2 if name == "block_table" else 1) and tensor.shape[0] == 1
            if not shape_ok or tensor.dtype != torch.int32:
                raise ValueError(f"Position template requires A5 int32 B1 {name}.")


class PositionTemplateRequest(dict):
    """One-use request envelope; sources stay alive through queued copies."""

    def __init__(self, owner, sources):
        super().__init__(owner.tree)
        self.owner = owner
        self.sources = sources
        self.consumed = False


class PositionTemplateBuffers(FastPlannedDecoderMetadataBuffers):
    """Static original-builder template plus only live block/slot refreshes."""

    def __init__(self, metadata):
        self.position = decode_position(metadata, MAX_DECODER_GRAPH_CONTEXT)
        _check_schema(metadata, self.position)
        super().__init__(metadata)
        self.bindings = None
        self.requests = 0
        self.baseline_device_targets = len(
            {id(node.target) for node in self._nodes if node.kind == "tensor" and node.target.device.type != "cpu"}
        )

    def bind(self, layer_groups, sources):
        if self.bindings is not None or self.tree.keys() != layer_groups.keys():
            raise ValueError("Position template layer/KV-group mapping changed.")
        targets = {}
        for name, value in self.tree.items():
            gid, block_size = layer_groups[name]
            if value.decode.block_size != block_size:
                raise ValueError("Position template block size changed.")
            for field in ("block_table", "slot_mapping"):
                target = getattr(value.decode, field)
                if target is None:
                    continue
                source = sources[(gid, field)]
                if _tensor_contract(source) != _tensor_contract(target):
                    raise ValueError(f"Position template live {field} contract differs from startup.")
                key = (gid, field)
                previous = targets.get(id(target))
                if previous is not None and previous[1] != key:
                    raise ValueError("Position template dynamic alias topology changed.")
                targets[id(target)] = (target, key, _tensor_contract(source))
        self.bindings = tuple(targets.values())

    def request(self, sources):
        if self.bindings is None:
            raise RuntimeError("Position template has no live input binding.")
        return PositionTemplateRequest(self, sources)

    def compare_reference(self, reference, sources):
        """Acceptance-only full payload comparison; intentionally synchronizes."""
        targets = {}
        self._validate(reference, [object()] * len(self._nodes), {}, targets)
        live = {id(target): sources[key] for target, key, _ in self.bindings}
        for target, source, _ in targets.values():
            if target is not None:
                expected = live.get(id(target), target)
                torch.testing.assert_close(source, expected, rtol=0, atol=0)

    def update(self, metadata):
        if type(metadata) is not PositionTemplateRequest or metadata.owner is not self or metadata.consumed:
            raise ValueError("Position template requires its own fresh request envelope.")
        if metadata.keys() != self.tree.keys() or any(metadata[key] is not value for key, value in self.tree.items()):
            raise ValueError("Position template request tree was replaced.")
        # The compiled closed plan checks static types/constants, pointers,
        # layouts and aliases before writes. Static payload contents are owned
        # by this entry and read-only to the captured decoder.
        targets = {}
        self._validate(self.tree, [object()] * len(self._nodes), {}, targets)
        pending = []
        for target, key, expected in self.bindings:
            source = metadata.sources[key]
            if not isinstance(source, torch.Tensor) or _tensor_contract(source) != expected:
                raise ValueError("Position template live tensor contract changed.")
            pending.append((target, source))
        metadata.consumed = True
        for target, source in pending:
            target.copy_(source)
            self.copies += 1
        self.requests += 1


class PositionTemplateAdapter:
    """Instance-local producer; original preparation still computes live slots."""

    def __init__(self, runner, bank):
        self.runner, self.bank = runner, bank
        self.original = runner._build_attention_metadata
        self.hits = self.baseline_calls = 0
        self.verify_reference = False
        self.reference_builder_calls = 0
        self.reference_checks = 0
        self.reference_positions = set()
        self.layer_groups = {}
        self.block_owners = {}
        self._check_runner()
        for gid, groups in enumerate(runner.attn_groups):
            block = runner.input_batch.block_table[gid]
            self.block_owners[gid] = (
                block,
                block.get_device_tensor(),
                block.slot_mapping.gpu,
                block.block_size,
                block.physical_block_size,
            )
            for group in groups:
                builder = group.get_metadata_builder(0)
                if (
                    type(builder).__module__ != "vllm_ascend.attention.dsa_v1"
                    or type(builder).__name__ != "AscendDSAMetadataBuilder"
                ):
                    raise ValueError("Position template requires the exact DSA builder.")
                for layer in group.layer_names:
                    if layer in self.layer_groups:
                        raise ValueError("Position template layer belongs to multiple KV groups.")
                    self.layer_groups[layer] = (gid, group.kv_cache_spec.block_size)
        sources = self._sources()
        for entry in bank.entries.values():
            entry["metadata"].bind(self.layer_groups, sources)
        self.configuration_signature = self._configuration_signature()

    def _configuration_signature(self):
        groups = []
        for gid, values in enumerate(self.runner.attn_groups):
            for group in values:
                builder = group.get_metadata_builder(0)
                config = builder.model_config
                hf = config.hf_config
                groups.append(
                    (
                        gid,
                        id(group),
                        id(builder),
                        tuple(group.layer_names),
                        group.kv_cache_spec.block_size,
                        builder.compressor_ratio,
                        config.get_head_size(),
                        tuple(
                            getattr(hf, name)
                            for name in (
                                "num_attention_heads",
                                "index_topk",
                                "index_n_heads",
                                "index_head_dim",
                                "sliding_window",
                            )
                        ),
                    )
                )
        return tuple(groups)

    def _check_runner(self):
        runner = self.runner
        config = runner.vllm_config
        parallel = config.parallel_config
        if getattr(getattr(runner, "model_config", None), "enable_return_routed_experts", False):
            raise ValueError("Position template cannot skip routed-expert slot snapshot side effects.")
        if any(
            getattr(parallel, name, 1) != 1
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        ):
            raise ValueError("Position template requires single-rank execution.")
        if config.scheduler_config.max_num_seqs != 1 or config.scheduler_config.async_scheduling:
            raise ValueError("Position template requires synchronous max_num_seqs=1.")
        if any(
            getattr(config, name, None) is not None
            for name in ("speculative_config", "lora_config", "kv_transfer_config")
        ):
            raise ValueError("Position template does not support speculation, LoRA or KV transfer.")
        if any(
            getattr(runner, name, False)
            for name in (
                "use_cp",
                "use_async_scheduling",
                "use_async_spec_decode",
                "cascade_attn_enabled",
                "is_multimodal_model",
                "enable_prompt_embeds",
                "is_mm_prefix_lm",
                "enable_hamming_sparse",
            )
        ):
            raise ValueError("Position template runner has unsupported dynamic features.")
        if any(
            getattr(config.cache_config, name, False) for name in ("enable_prefix_caching", "kv_sharing_fast_prefill")
        ):
            raise ValueError("Position template does not support prefix/fast-prefill cache sharing.")

    def _sources(self):
        sources = {}
        for gid, (block, table, slots, logical, physical) in self.block_owners.items():
            current = self.runner.input_batch.block_table[gid]
            if (
                current is not block
                or current.get_device_tensor() is not table
                or current.slot_mapping.gpu is not slots
                or current.block_size != logical
                or current.physical_block_size != physical
            ):
                raise ValueError("Position template block table owner/geometry changed.")
            sources[(gid, "block_table")] = table[:1]
            sources[(gid, "slot_mapping")] = slots[:1]
        return sources

    def build(self, *args, **kwargs):
        # Eager reference and all prefill always use the original producer.
        if not self.bank.model._v4_graph_enabled or self.runner.with_prefill:
            self.baseline_calls += 1
            return self.original(*args, **kwargs)
        self._check_runner()
        if self._configuration_signature() != self.configuration_signature:
            raise ValueError("Position template attention builder/configuration changed.")
        if args or kwargs.keys() - BUILD_ARGUMENTS:
            raise ValueError("Position template builder call schema changed.")
        if any(kwargs.get(name) != 1 for name in ("num_tokens", "num_reqs", "max_query_len")):
            raise ValueError("Position template only supports B1 one-token decode.")
        if any(kwargs.get(name) not in (None, 1) for name in ("num_tokens_padded", "num_reqs_padded")):
            raise ValueError("Position template does not support padded decode.")
        if any(kwargs.get(name) for name in ("use_spec_decode", "for_cudagraph_capture")) or any(
            kwargs.get(name) is not None for name in ("ubatch_slices", "cascade_attn_prefix_lens")
        ):
            raise ValueError("Position template does not support alternate decode builders.")
        runner = self.runner
        batch = runner.input_batch
        if (
            batch.num_reqs != 1
            or len(batch.req_ids) != 1
            or kwargs.get("num_scheduled_tokens") != {batch.req_ids[0]: 1}
        ):
            raise ValueError("Position template current request/scheduler mapping changed.")
        if getattr(runner.attn_state, "name", None) != "DecodeOnly":
            raise ValueError("Position template requires the current DecodeOnly state.")
        cpu_lengths = runner.optimistic_seq_lens_cpu[:1]
        if cpu_lengths.device.type != "cpu":
            raise ValueError("Position template requires scheduler CPU lengths.")
        position = int(cpu_lengths.tolist()[0]) - 1
        if not 0 <= position < self.bank.max_model_len:
            raise ValueError("Position template position is outside captured range.")
        _cpu_values(batch.num_computed_tokens_cpu_tensor[:1], [position], "computed tokens")
        _cpu_values(runner.query_start_loc.cpu[:2], [0, 1], "query starts")
        _cpu_values(runner._dsa_positions_cpu_buf[:1], [position], "positions")
        if batch.req_prompt_embeds:
            raise ValueError("Position template does not support request prompt embeddings.")
        sources = self._sources()
        owner = self.bank.entries[position]["metadata"]
        if self.verify_reference:
            self.reference_builder_calls += 1
            reference, extra = self.original(*args, **kwargs)
            if extra is not None:
                raise ValueError("Position template reference produced unexpected auxiliary metadata.")
            owner.compare_reference(reference, sources)
            self.reference_checks += 1
            self.reference_positions.add(position)
        request = owner.request(sources)
        self.hits += 1
        return request, None

    def report(self):
        return {
            "scope": "a5_b1_position_metadata_with_live_block_tables_and_slots",
            "template_requests": self.hits,
            "original_builder_skips": self.hits - self.reference_checks,
            "original_builder_calls": self.baseline_calls + self.reference_builder_calls,
            "baseline_builder_calls": self.baseline_calls,
            "reference_builder_calls": self.reference_builder_calls,
            "dynamic_copies": sum(entry["metadata"].copies for entry in self.bank.entries.values()),
            "baseline_device_targets_per_position": {
                position: entry["metadata"].baseline_device_targets for position, entry in self.bank.entries.items()
            },
            "dynamic_targets_per_position": {
                position: len(entry["metadata"].bindings) for position, entry in self.bank.entries.items()
            },
            "input_preparation_skipped": False,
            "reference_verification_enabled": self.verify_reference,
            "reference_checks": self.reference_checks,
            "reference_positions": sorted(self.reference_positions),
            "hardware_metadata_equivalence_verified": False,
        }

    def detach(self):
        """Called only after graph completion and successful reset at close."""
        self.runner._build_attention_metadata = self.original
        del self.runner._vq2a8_position_template
        self.block_owners.clear()
        self.layer_groups.clear()
        self.runner = self.bank = self.original = None


def install_position_template(runner, bank):
    if hasattr(runner, "_vq2a8_position_template"):
        raise RuntimeError("Position template adapter is already installed.")
    adapter = PositionTemplateAdapter(runner, bank)

    @wraps(adapter.original)
    def build(*args, **kwargs):
        return adapter.build(*args, **kwargs)

    runner._build_attention_metadata = build
    runner._vq2a8_position_template = adapter
    bank.position_template_adapter = adapter
    return adapter
