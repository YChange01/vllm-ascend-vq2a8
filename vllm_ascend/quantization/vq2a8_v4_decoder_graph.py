# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Position-specialized B1 decoder capture, with live DSA metadata inputs.

This deliberately does not enable vLLM's generic FULL graph dispatcher. Each
short-context position has its own graph/pool; prefill, the LM head and sampling
remain eager. Metadata normally uses the eager builder; the opt-in position
template producer keeps only live block/slot updates. Capture uses the real cache addresses,
but restores *all* KV/compressor/indexer state after every trial. No request is
used for warmup or capture, and no per-layer graph replay is nested inside it.
"""

from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, fields, is_dataclass
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


@dataclass(frozen=True)
class _MetadataPlanNode:
    kind: str
    target: object
    name: str
    contract: object
    children: tuple = ()


class _DecoderMetadataPlan:
    """Startup-compiled metadata DAG, with closed, transactional replay checks.

    Shared containers are cloned once and their structure is compiled into
    indexed nodes. Replay does not discover dataclass fields or rebuild a tree.
    A per-update (plan node, source identity) memo skips repeated shared paths,
    but separately validates distinct replacement objects. No source reference
    or validation result is retained across requests. Tensor aliases retain the
    recursive implementation's exact-storage-view rule.
    """

    def __init__(self, metadata):
        self._nodes = []
        self._memo = {}
        self._compiling = set()
        self._aliases = {}
        self._root = self._compile(metadata, "")
        self.tree = self._nodes[self._root].target
        self._nodes = tuple(self._nodes)
        # Startup source identities are not meaningful for subsequent requests.
        del self._memo, self._compiling, self._aliases
        self.copies = 0

    def _compile(self, value, name):
        # The same container under mutable and immutable field contexts must
        # not accidentally share a plan with different tensor ownership rules.
        key = (id(value), name in IMMUTABLE_METADATA_FIELDS)
        if key in self._compiling:
            raise TypeError("Cyclic decoder metadata is unsupported.")
        if key in self._memo:
            return self._memo[key]
        index = len(self._nodes)
        self._nodes.append(None)
        self._memo[key] = index
        self._compiling.add(key)

        if isinstance(value, torch.Tensor):
            contract = _tensor_contract(value)
            if name in IMMUTABLE_METADATA_FIELDS:
                target = value
                kind = "immutable_tensor"
            else:
                identity = (value.data_ptr(), contract)
                if identity not in self._aliases:
                    self._aliases[identity] = value.clone()
                target = self._aliases[identity]
                kind = "tensor"
            node = _MetadataPlanNode(kind, target, name, (contract, target.data_ptr()))
        elif _is_rope_proxy(value):
            if value.idx not in (0, 1) or not isinstance(value._data, dict):
                raise ValueError("Invalid DSA rotary proxy selector/data contract.")
            child = self._compile(value._data, name)
            target = copy(value)
            target._data = self._nodes[child].target
            node = _MetadataPlanNode("proxy", target, name, (type(value), value.idx), (("_data", child),))
        elif is_dataclass(value) and not isinstance(value, type):
            target = copy(value)
            children = []
            for field in fields(value):
                child = self._compile(getattr(value, field.name), field.name)
                children.append((field.name, child))
                setattr(target, field.name, self._nodes[child].target)
            node = _MetadataPlanNode(
                "dataclass", target, name, (type(value), frozenset(value.__dataclass_fields__)), tuple(children)
            )
        elif isinstance(value, dict):
            children = tuple((key, self._compile(item, name)) for key, item in value.items())
            target = {key: self._nodes[child].target for key, child in children}
            node = _MetadataPlanNode("dict", target, name, (type(value), frozenset(value)), children)
        elif isinstance(value, (list, tuple)):
            children = tuple((i, self._compile(item, name)) for i, item in enumerate(value))
            target = type(value)(self._nodes[child].target for _, child in children)
            node = _MetadataPlanNode("sequence", target, name, (type(value), len(value)), children)
        elif value is None or isinstance(value, (str, int, float, bool, Enum)):
            node = _MetadataPlanNode("constant", value, name, type(value))
        else:
            raise TypeError(f"Unsupported decoder metadata field {name}: {type(value).__name__}.")
        self._nodes[index] = node
        self._compiling.remove(key)
        return index


class DecoderMetadataBuffers(_DecoderMetadataPlan):
    """Compile metadata validation actions once; validate all inputs per update."""

    def __init__(self, metadata):
        super().__init__(metadata)
        actions = {}
        self._validate = self._compile_action(self._root, actions)

    def _compile_action(self, index, actions):
        if index in actions:
            return actions[index]
        node = self._nodes[index]
        target, name = node.target, node.name

        if node.kind in ("tensor", "immutable_tensor"):
            expected, pointer = node.contract
            target_id = id(target)
            immutable = node.kind == "immutable_tensor"
            cpu = expected[3].type == "cpu"

            def check(source, seen, views, targets):
                if seen[index] is source:
                    return
                seen[index] = source
                if not isinstance(source, torch.Tensor):
                    raise ValueError(f"Decoder metadata tensor contract changed: {name}.")
                source_id = id(source)
                source_view = views.get(source_id)
                if source_view is None:
                    # Retain the object alongside its view so even unusual
                    # accessors cannot recycle its id during this update.
                    source_view = (source, _tensor_contract(source), source.data_ptr())
                    views[source_id] = source_view
                target_view = views.get(target_id)
                if target_view is None:
                    target_view = (target, _tensor_contract(target), target.data_ptr())
                    views[target_id] = target_view
                if source_view[1] != expected or target_view[1] != expected or target_view[2] != pointer:
                    raise ValueError(f"Decoder metadata tensor contract changed: {name}.")
                if immutable:
                    if source_view[2] != pointer:
                        raise ValueError(f"Decoder immutable metadata storage changed: {name}.")
                    return
                previous = targets.get(target_id)
                if previous is not None:
                    if previous[2] != source_view[2]:
                        raise ValueError("Decoder metadata alias topology changed; an input would otherwise be frozen.")
                    return
                if cpu and not torch.equal(target, source):
                    raise ValueError(f"Decoder CPU metadata values changed for this position: {name}.")
                targets[target_id] = (None if cpu else target, source, source_view[2])

        elif node.kind == "constant":
            expected_type = node.contract

            def check(source, seen, views, targets):
                if seen[index] is source:
                    return
                seen[index] = source
                if type(source) is not expected_type:
                    raise ValueError(f"Decoder metadata type changed: {name}.")
                if source != target:
                    raise ValueError(f"Decoder metadata constant changed for this position: {name}.")

        else:
            expected_type, structure = node.contract
            children = tuple((key, self._compile_action(child, actions)) for key, child in node.children)
            if node.kind == "dict":

                def check(source, seen, views, targets):
                    if seen[index] is source:
                        return
                    seen[index] = source
                    if type(source) is not expected_type:
                        raise ValueError(f"Decoder metadata type changed: {name}.")
                    if source.keys() != structure:
                        raise ValueError(f"Decoder metadata keys changed: {name}.")
                    for key, child in children:
                        child(source[key], seen, views, targets)

            elif node.kind == "sequence":

                def check(source, seen, views, targets):
                    if seen[index] is source:
                        return
                    seen[index] = source
                    if type(source) is not expected_type:
                        raise ValueError(f"Decoder metadata type changed: {name}.")
                    if len(source) != structure:
                        raise ValueError(f"Decoder metadata length changed: {name}.")
                    for key, child in children:
                        child(source[key], seen, views, targets)

            elif node.kind == "dataclass":

                def check(source, seen, views, targets):
                    if seen[index] is source:
                        return
                    seen[index] = source
                    if type(source) is not expected_type:
                        raise ValueError(f"Decoder metadata type changed: {name}.")
                    if source.__dataclass_fields__.keys() != structure:
                        raise ValueError(f"Decoder metadata dataclass fields changed: {name}.")
                    for key, child in children:
                        child(getattr(source, key), seen, views, targets)

            else:
                # _compile has a closed set of node kinds. Only RopeDataProxy
                # can arrive here, with one validated _data child.
                child = children[0][1]

                def check(source, seen, views, targets):
                    if seen[index] is source:
                        return
                    seen[index] = source
                    if type(source) is not expected_type:
                        raise ValueError(f"Decoder metadata type changed: {name}.")
                    if source.idx != structure:
                        raise ValueError("Decoder rotary proxy cosine/sine selector changed.")
                    child(source._data, seen, views, targets)

        actions[index] = check
        return check

    def update(self, metadata):
        targets = {}
        # A one-slot identity memo per node avoids allocating (node, id) tuples
        # for shared DAG paths. A different object revalidates that node; even
        # alternating replacements are checked rather than using stale state.
        self._validate(metadata, [object()] * len(self._nodes), {}, targets)
        for target, source, _ in targets.values():
            if target is not None:
                target.copy_(source)
                self.copies += 1


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

    def __init__(self, model, max_model_len, *, backend=None, metadata_mode="position_template"):
        if type(max_model_len) is not int or not 1 <= max_model_len <= MAX_DECODER_GRAPH_CONTEXT:
            raise ValueError("Decoder graph context must be an integer in 1..16.")
        if metadata_mode != "position_template":
            raise ValueError("Decoder metadata mode must be position_template.")
        self.model = model
        self.max_model_len = max_model_len
        self.metadata_mode = metadata_mode
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
        self.position_template_adapter = None

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
            # Import lazily: the template producer shares the metadata plan.
            from vllm_ascend.quantization.vq2a8_decoder_position_template import PositionTemplateBuffers

            metadata = PositionTemplateBuffers(context.attn_metadata)
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
            # replay. Escape once per whole decoder, not once per layer.
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
            if self.position_template_adapter is not None:
                self.position_template_adapter.detach()
                self.position_template_adapter = None
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
            "metadata_mode": self.metadata_mode,
            "metadata_tensor_copies": sum(entry["metadata"].copies for entry in self.entries.values()),
            "position_template": (
                self.position_template_adapter.report() if self.position_template_adapter is not None else None
            ),
            "startup_state_restored": self.ready,
            "nested_moe_graphs": False,
            "ready": self.ready,
            "failed": self.failed,
            "closed": self.closed,
            "hardware_accuracy_verified": False,
        }
