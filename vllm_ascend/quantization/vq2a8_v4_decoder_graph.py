# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Position-specialized B1 decoder capture, with live DSA metadata inputs.

This deliberately does not enable vLLM's generic FULL graph dispatcher. Each
short-context position has its own graph/pool; prefill, metadata construction,
the LM head and sampling remain eager. Capture uses the real cache addresses,
but restores *all* KV/compressor/indexer state after every trial. No request is
used for warmup or capture, and no per-layer graph replay is nested inside it.
"""

from contextlib import contextmanager
from copy import copy
from dataclasses import fields, is_dataclass
from enum import Enum
from threading import Lock

import torch

MAX_DECODER_GRAPH_CONTEXT = 16
DECODER_GRAPH_WARMUPS = 2
IMMUTABLE_METADATA_FIELDS = frozenset(("hadamard", "full_compress_cos", "full_compress_sin"))


def _tensor_contract(value):
    return tuple(value.shape), tuple(value.stride()), value.dtype, value.device


def _is_rope_proxy(value):
    # Do not import rope_dsv4 here: CPU metadata tests must not initialize the
    # torch-npu runtime. This is a closed adapter for the actual DSA proxy, not
    # arbitrary __dict__ copying that could freeze a future mutable backend.
    return type(value).__module__ == "vllm_ascend.ops.rope_dsv4" and type(value).__name__ == "RopeDataProxy"


class DecoderMetadataBuffers:
    """Clone metadata, not model objects, and refresh every mutable tensor.

    CPU-side constants must match the selected position's capture contract.
    Large immutable RoPE tables are retained by address, never copied per token.
    Dataclass traversal also covers DSA slot maps, block tables, start positions,
    SAS/QLI metadata and per-layer rotary dictionaries. Unknown object types fail
    closed instead of silently freezing a newly added backend field.
    """

    def __init__(self, metadata):
        self._aliases = {}
        self.tree = self._clone(metadata, "")
        self.copies = 0

    def _clone(self, value, name):
        if _is_rope_proxy(value):
            if value.idx not in (0, 1) or not isinstance(value._data, dict):
                raise ValueError("Invalid DSA rotary proxy selector/data contract.")
            result = copy(value)
            result._data = self._clone(value._data, name)
            return result
        if isinstance(value, torch.Tensor):
            if name in IMMUTABLE_METADATA_FIELDS:
                return value
            # DSA builds many distinct Python views of the same RoPE/length
            # buffer. Share their static destination by exact storage view,
            # reducing H2D/D2D submissions; update validates alias topology.
            identity = (value.data_ptr(), _tensor_contract(value))
            if identity not in self._aliases:
                self._aliases[identity] = value.clone()
            return self._aliases[identity]
        if is_dataclass(value) and not isinstance(value, type):
            result = copy(value)
            for field in fields(value):
                setattr(result, field.name, self._clone(getattr(value, field.name), field.name))
            return result
        if isinstance(value, dict):
            return {key: self._clone(item, name) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._clone(item, name) for item in value)
        if value is None or isinstance(value, (str, int, float, bool, Enum)):
            return value
        raise TypeError(f"Unsupported decoder metadata field {name}: {type(value).__name__}.")

    def update(self, metadata):
        # Validate the complete contract before modifying any graph inputs.
        pairs = []
        self._check(self.tree, metadata, "", pairs)
        sources = {}
        for target, source in pairs:
            identity = id(target)
            storage_view = (source.data_ptr(), _tensor_contract(source))
            if identity in sources and sources[identity] != storage_view:
                raise ValueError("Decoder metadata alias topology changed; an input would otherwise be frozen.")
            sources[identity] = storage_view
        seen = set()
        for target, source in pairs:
            identity = id(target)
            if identity in seen:
                continue
            seen.add(identity)
            if target.device.type != "cpu":
                target.copy_(source)
                self.copies += 1

    def _check(self, target, source, name, pairs):
        if isinstance(target, torch.Tensor):
            if not isinstance(source, torch.Tensor) or _tensor_contract(target) != _tensor_contract(source):
                raise ValueError(f"Decoder metadata tensor contract changed: {name}.")
            if name in IMMUTABLE_METADATA_FIELDS:
                if target.data_ptr() != source.data_ptr():
                    raise ValueError(f"Decoder immutable metadata storage changed: {name}.")
            elif target.device.type == "cpu":
                if not torch.equal(target, source):
                    raise ValueError(f"Decoder CPU metadata values changed for this position: {name}.")
                pairs.append((target, source))
            else:
                pairs.append((target, source))
            return
        if type(target) is not type(source):
            raise ValueError(f"Decoder metadata type changed: {name}.")
        if _is_rope_proxy(target):
            if target.idx != source.idx:
                raise ValueError("Decoder rotary proxy cosine/sine selector changed.")
            self._check(target._data, source._data, name, pairs)
        elif is_dataclass(target) and not isinstance(target, type):
            for field in fields(target):
                self._check(getattr(target, field.name), getattr(source, field.name), field.name, pairs)
        elif isinstance(target, dict):
            if target.keys() != source.keys():
                raise ValueError(f"Decoder metadata keys changed: {name}.")
            for key in target:
                self._check(target[key], source[key], name, pairs)
        elif isinstance(target, (list, tuple)):
            if len(target) != len(source):
                raise ValueError(f"Decoder metadata length changed: {name}.")
            for old, new in zip(target, source):
                self._check(old, new, name, pairs)
        elif target != source:
            raise ValueError(f"Decoder metadata constant changed for this position: {name}.")


def decode_position(metadata, max_model_len):
    """Use existing scheduler-built CPU lengths, never positions.item()."""
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("Decoder graphs require per-layer DSA decode metadata.")
    positions = set()
    for entry in metadata.values():
        if (
            getattr(entry, "num_decodes", None) != 1
            or getattr(entry, "num_decode_tokens", None) != 1
            or getattr(entry, "num_actual_tokens", None) != 1
            or getattr(entry, "num_prefills", None) != 0
            or getattr(entry, "prefill", None) is not None
        ):
            raise ValueError("Decoder graphs support B1 one-token decode only, never prefill/padded batches.")
        decode = getattr(entry, "decode", None)
        lengths = getattr(decode, "seq_lens_list", None)
        if not isinstance(lengths, list) or len(lengths) != 1 or type(lengths[0]) is not int:
            raise ValueError("Decoder graph position requires one scheduler CPU sequence length.")
        positions.add(lengths[0] - 1)
    if len(positions) != 1:
        raise ValueError("DSA layers disagree on decoder graph position.")
    position = positions.pop()
    if not 0 <= position < max_model_len:
        raise ValueError("Decoder position exceeds the explicitly captured short-context range.")
    return position


def mutable_decoder_tensors(model):
    """Enumerate unregistered cache tensors as well as shared indexer buffers."""
    result = []
    seen = set()

    def visit(value):
        if isinstance(value, torch.Tensor):
            if value.numel() and id(value) not in seen:
                seen.add(id(value))
                result.append(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    for module in model.modules():
        for name in ("kv_cache", "topk_indices_buffer", "_mtp_hidden_buffer"):
            visit(getattr(module, name, None))
    if not result:
        raise ValueError("Decoder graph capture must follow allocation/binding of real KV cache tensors.")
    return tuple(result)


class DecoderStateSnapshot:
    """A startup-only checkpoint. Restore only after a successful device fence."""

    def __init__(self, tensors):
        self.pairs = tuple((tensor, tensor.clone()) for tensor in tensors)

    def restore(self):
        for target, saved in self.pairs:
            target.copy_(saved)

    def clear_trial(self):
        # Fresh startup caches may be uninitialized. Synthetic positions use a
        # zero history, not indeterminate data; original bytes are restored on
        # exit, including the partial compressor and indexer state caches.
        for target, _ in self.pairs:
            target.zero_()


class V4DecoderGraphBank:
    """Single owner stream, independent per-position pools, fixed caller replay."""

    def __init__(self, model, max_model_len, *, backend=None):
        if type(max_model_len) is not int or not 1 <= max_model_len <= MAX_DECODER_GRAPH_CONTEXT:
            raise ValueError("Decoder graph context must be an integer in 1..16.")
        self.model = model
        self.max_model_len = max_model_len
        self.backend = torch.npu if backend is None else backend
        self.entries = {}
        self.failed = False
        self.ready = False
        self.replays = 0
        self.capture_stream = None
        self.caller_stream = None
        self._lock = Lock()
        self.snapshot = None
        self.capture_state = ()
        self.computes = ()
        self.closed = False

    @contextmanager
    def exclusive(self):
        if self.failed or self.closed or not self._lock.acquire(blocking=False):
            raise RuntimeError("Decoder graph bank is failed, concurrent or recursively entered.")
        try:
            yield
        except BaseException:
            self.failed = True
            raise
        finally:
            self._lock.release()

    def _inputs(self, input_ids, positions):
        if (
            input_ids.shape != (1,)
            or positions.shape != (1,)
            or input_ids.dtype not in (torch.int32, torch.int64)
            or positions.dtype != torch.int64
            or input_ids.device != positions.device
        ):
            raise ValueError("Decoder graphs require one integer device token and one int64 position.")

    def capture(self, position, input_ids, positions, context, compute):
        """Called only by the startup runner with isolated mutable-state trials."""
        with self.exclusive():
            if self.ready or position in self.entries or self.snapshot is None:
                raise RuntimeError("Decoder captures are startup-only, once per position.")
            self._inputs(input_ids, positions)
            if decode_position(context.attn_metadata, self.max_model_len) != position:
                raise ValueError("Startup metadata position differs from its graph key.")
            metadata = DecoderMetadataBuffers(context.attn_metadata)
            tokens = input_ids.clone()
            static_positions = positions.clone()
            original = context.attn_metadata
            context.attn_metadata = metadata.tree
            graph = self.backend.NPUGraph()
            # Retain failed capture owners as well: queued work may still use
            # their addresses if the completion fence itself raises.
            entry = {"graph": graph, "metadata": metadata, "tokens": tokens, "positions": static_positions}
            self.entries[position] = entry
            try:
                for _ in range(DECODER_GRAPH_WARMUPS):
                    self.snapshot.clear_trial()
                    outputs = compute(tokens, static_positions)
                    self.backend.synchronize()
                reference = tuple(value.clone() for value in outputs)
                self.snapshot.clear_trial()
                self.backend.synchronize()
                with self.backend.graph(graph, stream=self.capture_stream):
                    outputs = compute(tokens, static_positions)
                if (
                    not isinstance(outputs, tuple)
                    or len(outputs) != 2
                    or outputs[0].ndim != 2
                    or outputs[0].shape[0] != 1
                    or outputs[0].dtype != torch.bfloat16
                    or outputs[1].shape != ()
                    or outputs[1].dtype != torch.bool
                    or any(value.device != tokens.device for value in outputs)
                ):
                    raise ValueError("Decoder capture must return B1 BF16 hidden and scalar device validity.")
                entry["outputs"] = outputs
                entry["output_contract"] = tuple(_tensor_contract(value) for value in outputs)
                self.backend.synchronize()
                # Capture submission is not execution evidence. Replay once
                # on an identical zero history before inspecting its outputs.
                self.snapshot.clear_trial()
                graph.replay()
                self.backend.synchronize()
                if not bool(outputs[1]):
                    raise ValueError("Synthetic decoder capture failed device validity checks.")
                torch.testing.assert_close(outputs[0], reference[0], rtol=1e-3, atol=1e-3)
            finally:
                context.attn_metadata = original
                # A failed fence aborts startup and retains all owners.
                self.backend.synchronize()
                self.snapshot.restore()
                self.backend.synchronize()
            return outputs[0]

    def replay(self, input_ids, positions, context):
        with self.exclusive():
            if not self.ready:
                raise RuntimeError("Decoder graphs must finish startup capture before requests.")
            self._inputs(input_ids, positions)
            if self.backend.is_current_stream_capturing():
                raise RuntimeError("Decoder graph replay cannot be nested in another graph.")
            if self.backend.current_stream().npu_stream != self.caller_stream.npu_stream:
                raise RuntimeError("Decoder graph caller stream changed.")
            position = decode_position(context.attn_metadata, self.max_model_len)
            entry = self.entries[position]
            if input_ids.device != entry["tokens"].device or positions.device != entry["positions"].device:
                raise ValueError("Decoder graph inputs changed devices.")
            if tuple(_tensor_contract(value) for value in entry["outputs"]) != entry["output_contract"]:
                raise ValueError("Decoder graph output buffers changed.")
            for compute in self.computes:
                compute.check_runtime_contract(compute.runtime)
            entry["metadata"].update(context.attn_metadata)
            entry["tokens"].copy_(input_ids)
            entry["positions"].copy_(positions)
            entry["graph"].replay()
            # External consumers cannot retain pool outputs across another
            # replay. Only two escapes per whole decoder, not 43 per-layer sets.
            hidden, valid = entry["outputs"]
            output = hidden.clone(), valid.clone()
            self.replays += 1
            return output

    def close(self):
        # Cleanup is allowed after a failed operation but never concurrently.
        # Any failed completion/reset keeps entries, tensors and model owners.
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Decoder graph cleanup cannot overlap execution.")
        try:
            if self.closed:
                return
            self.ready = False
            self.backend.synchronize()
            for entry in self.entries.values():
                entry["graph"].reset()
            self.entries.clear()
            self.snapshot = None
            self.capture_state = self.computes = ()
            self.model = None
            self.closed = True
        except BaseException:
            self.failed = True
            raise
        finally:
            self._lock.release()

    def report(self):
        return {
            "scope": "decoder_including_attention_hc_norm_moe_excluding_lm_head_sampler",
            "captures": len(self.entries),
            "startup_replays": len(self.entries) if self.ready else None,
            "replays": self.replays,
            "positions": sorted(self.entries),
            "independent_position_pools": True,
            "live_attention_metadata": True,
            "metadata_tensor_copies": sum(entry["metadata"].copies for entry in self.entries.values()),
            "startup_state_restored": self.ready,
            "nested_moe_graphs": False,
            "ready": self.ready,
            "failed": self.failed,
            "closed": self.closed,
            "hardware_accuracy_verified": False,
        }
