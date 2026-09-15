# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU protocol checks only; the real-model device probe is a separate gate."""

import ast
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.validate_vq2a8_v4_decoder_graph import CASES, compare_outputs, parse_args
from vllm_ascend.quantization.vq2a8_v4_decoder_graph import (
    DecoderMetadataBuffers,
    DecoderStateSnapshot,
    V4DecoderGraphBank,
    decode_position,
    mutable_decoder_tensors,
)


@dataclass
class Decode:
    seq_lens_list: list
    input_positions: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


@dataclass
class Metadata:
    decode: Decode
    num_decodes: int = 1
    num_decode_tokens: int = 1
    num_actual_tokens: int = 1
    num_prefills: int = 0
    prefill: object = None


def metadata(position=3):
    return {
        "layer": Metadata(
            Decode([position + 1], torch.tensor([position]), torch.tensor([[2]]), torch.tensor([position]))
        )
    }


@pytest.mark.parametrize("position", (0, 1, 3, 4, 7, 8, 11, 12, 15))
def test_position_keys_cover_compressor_boundaries_without_device_item(position):
    assert decode_position(metadata(position), 16) == position


@pytest.mark.parametrize(
    "change", (lambda x: setattr(x, "num_prefills", 1), lambda x: setattr(x, "num_actual_tokens", 2))
)
def test_prefill_and_padding_never_select_decode(change):
    value = metadata()
    change(value["layer"])
    with pytest.raises(ValueError, match="B1"):
        decode_position(value, 16)


def test_position_requires_consistent_cpu_metadata_and_range():
    with pytest.raises(ValueError, match="range"):
        decode_position(metadata(16), 16)
    mixed = metadata(3)
    mixed["other"] = metadata(4)["layer"]
    with pytest.raises(ValueError, match="disagree"):
        decode_position(mixed, 16)
    mixed = metadata(3)
    mixed["layer"].decode.seq_lens_list = torch.tensor([4])
    with pytest.raises(ValueError, match="CPU"):
        decode_position(mixed, 16)


def test_metadata_clones_owners_rejects_constants_and_unknown_backend_fields():
    source = metadata()
    owner = DecoderMetadataBuffers(source)
    assert owner.tree["layer"].decode.block_table is not source["layer"].decode.block_table
    owner.update(metadata())
    changed = metadata(4)
    with pytest.raises(ValueError):
        owner.update(changed)
    with pytest.raises(TypeError, match="Unsupported"):
        DecoderMetadataBuffers({"future_event": object()})


def test_immutable_rope_owner_is_not_copied_and_replacement_is_rejected():
    @dataclass
    class Rotary:
        full_compress_cos: torch.Tensor

    tensor = torch.ones(8, 4)
    owner = DecoderMetadataBuffers(Rotary(tensor))
    assert owner.tree.full_compress_cos is tensor
    owner.update(Rotary(tensor))
    with pytest.raises(ValueError, match="immutable"):
        owner.update(Rotary(tensor.clone()))


def test_changed_alias_topology_is_rejected_before_copy():
    tensor = torch.tensor([1])
    owner = DecoderMetadataBuffers({"x": tensor, "y": tensor})
    assert owner.tree["x"] is owner.tree["y"]
    with pytest.raises(ValueError, match="alias topology"):
        owner.update({"x": tensor.clone(), "y": tensor.clone()})
    owner.update({"x": tensor, "y": tensor})


def test_explicit_rope_proxy_retains_layer_lookup_and_rejects_selector_change():
    # Load the actual backend proxy class without its torch_npu imports.
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/rope_dsv4.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RopeDataProxy")
    scope = {
        "__name__": "vllm_ascend.ops.rope_dsv4",
        "_ROPE_STATE": SimpleNamespace(layer_info={"layer0": ("config", ["default"])}),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    RopeDataProxy = scope["RopeDataProxy"]
    data = {"config": {"default": (torch.ones(1, 4), torch.zeros(1, 4))}}
    proxy = RopeDataProxy(data)
    owner = DecoderMetadataBuffers({"cos": proxy})
    assert owner.tree["cos"] is not proxy
    assert torch.equal(owner.tree["cos"]["layer0"], proxy["layer0"])
    assert owner.tree["cos"]["layer0"].data_ptr() != proxy["layer0"].data_ptr()
    owner.update({"cos": proxy})
    with pytest.raises(ValueError, match="selector"):
        owner.update({"cos": RopeDataProxy(data, is_cos=False)})


def test_mutable_state_includes_unregistered_nested_kv_and_shared_buffers():
    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.kv_cache = [[torch.arange(8)], (torch.arange(4),)]
    model.topk_indices_buffer = torch.tensor([11, 12])
    model._mtp_hidden_buffer = model.topk_indices_buffer
    values = mutable_decoder_tensors(model)
    assert len(values) == 3
    previous = [value.clone() for value in values]
    snapshot = DecoderStateSnapshot(values)
    snapshot.clear_trial()
    assert all(torch.count_nonzero(value) == 0 for value in values)
    snapshot.restore()
    assert all(torch.equal(left, right) for left, right in zip(values, previous))


class Graph:
    def __init__(self, backend):
        self.backend = backend
        self.compute = self.outputs = None
        self.resets = 0

    def replay(self):
        if self.backend.fail_replay:
            raise RuntimeError("replay fault")
        for target, source in zip(self.outputs, self.compute()):
            target.copy_(source)

    def reset(self):
        self.resets += 1


class Backend:
    def __init__(self):
        self.current = SimpleNamespace(npu_stream=19)
        self.capturing = None
        self.fail_sync = self.fail_capture = self.fail_replay = False

    def current_stream(self):
        return self.current

    def is_current_stream_capturing(self):
        return self.capturing is not None

    def synchronize(self):
        if self.fail_sync:
            raise RuntimeError("fence fault")

    def NPUGraph(self):
        return Graph(self)

    @contextmanager
    def graph(self, graph, *, stream):
        if self.fail_capture:
            raise RuntimeError("capture fault")
        assert stream is self.current
        self.capturing = graph
        try:
            yield
        finally:
            self.capturing = None


def make_bank():
    backend = Backend()
    bank = V4DecoderGraphBank(object(), 16, backend=backend)
    bank.capture_stream = bank.caller_stream = backend.current
    cache = torch.tensor([99.0])
    bank.snapshot = DecoderStateSnapshot((cache,))

    def compute(tokens, positions):
        cache.copy_(tokens.float() + positions.float())
        output = cache.repeat(2).reshape(1, 2).to(torch.bfloat16)
        result = output, (tokens >= 0).all()
        if backend.capturing is not None:
            backend.capturing.compute = lambda: compute(tokens, positions)
            backend.capturing.outputs = result
        return result

    context = SimpleNamespace(attn_metadata=metadata(3))
    bank.capture(3, torch.tensor([2]), torch.tensor([3]), context, compute)
    bank.ready = True
    return bank, backend, cache, context


def test_capture_restores_state_and_replay_refreshes_token_outputs_validity_and_escapes():
    bank, backend, cache, context = make_bank()
    assert torch.equal(cache, torch.tensor([99.0]))
    first, valid1 = bank.replay(torch.tensor([10]), torch.tensor([3]), context)
    invalid, invalid_flag = bank.replay(torch.tensor([-1]), torch.tensor([3]), context)
    last, valid2 = bank.replay(torch.tensor([20]), torch.tensor([3]), context)
    assert torch.equal(first, torch.tensor([[13, 13]], dtype=torch.bfloat16))
    assert torch.equal(last, torch.tensor([[23, 23]], dtype=torch.bfloat16))
    assert bool(valid1) and bool(valid2) and not bool(invalid_flag)
    assert bank.replays == 3
    assert first.data_ptr() != invalid.data_ptr() != last.data_ptr()
    assert context.attn_metadata["layer"].decode.seq_lens_list == [4]
    assert backend.capturing is None


def test_replay_failure_latches_but_close_fences_and_releases():
    bank, backend, _, context = make_bank()
    backend.fail_replay = True
    with pytest.raises(RuntimeError, match="fault"):
        bank.replay(torch.tensor([10]), torch.tensor([3]), context)
    with pytest.raises(RuntimeError, match="failed"):
        bank.replay(torch.tensor([10]), torch.tensor([3]), context)
    backend.fail_sync = True
    with pytest.raises(RuntimeError, match="fence"):
        bank.close()
    assert bank.entries and bank.model is not None
    backend.fail_sync = False
    bank.close()
    assert bank.closed and not bank.entries and bank.model is None


def test_replay_rejects_nested_and_wrong_caller_stream():
    bank, backend, _, context = make_bank()
    backend.current = SimpleNamespace(npu_stream=20)
    with pytest.raises(RuntimeError, match="caller"):
        bank.replay(torch.tensor([10]), torch.tensor([3]), context)
    bank, backend, _, context = make_bank()
    backend.capturing = object()
    with pytest.raises(RuntimeError, match="nested"):
        bank.replay(torch.tensor([10]), torch.tensor([3]), context)


def test_probe_cases_cover_boundaries_and_repeated_long_decode():
    decoded = {position for prompt, output in CASES for position in range(prompt, prompt + output - 1)}
    assert {3, 4, 7, 8, 11, 12}.issubset(decoded)
    assert set(range(1, 15)).issubset(decoded)
    args = parse_args(["--model", "model", "--library", "kernel.so"])
    assert args.timeout_s == 1800 and args.physical_npu == 1


def test_logprob_probe_checks_numeric_values_not_only_greedy_token_identity():
    expected = SimpleNamespace(token_ids=[3], logprobs=[{3: SimpleNamespace(logprob=-0.5)}])
    actual = SimpleNamespace(token_ids=[3], logprobs=[{3: SimpleNamespace(logprob=-0.4)}])
    with pytest.raises(AssertionError, match="probabilities"):
        compare_outputs(expected, actual)
    assert compare_outputs(expected, expected) == 0
