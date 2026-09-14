# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU capture/replay protocol tests; these do not emulate NPU graph kernels."""

from contextlib import contextmanager
from threading import Event, Thread

import pytest
import torch

from vllm_ascend.quantization.vq2a8_v3_graph import GRAPH_WARMUPS, MoEDecodeGraph


class FakeStream:
    def __init__(self, stream_id=19):
        self.npu_stream = stream_id
        self.synchronizations = 0

    def synchronize(self):
        self.synchronizations += 1


class FakeGraph:
    def __init__(self, backend):
        self.backend = backend
        self.replays = 0
        self.compute = self.output = None

    def replay(self):
        if self.backend.fail_replay:
            raise RuntimeError("injected replay failure")
        self.replays += 1
        # Recompute from captured INPUT STORAGE on every replay and overwrite
        # the same captured OUTPUT STORAGE, as a graph's device work does.
        self.output.copy_(self.compute())


class FakeBackend:
    def __init__(self):
        self.stream = FakeStream()
        self.graphs = []
        self.capturing = None
        self.capture_entries = 0
        self.fail_capture = self.fail_replay = False

    def current_stream(self, device):
        assert device == torch.device("cpu")
        return self.stream

    def NPUGraph(self):
        graph = FakeGraph(self)
        self.graphs.append(graph)
        return graph

    @contextmanager
    def graph(self, graph, stream):
        assert stream is self.stream
        self.capture_entries += 1
        if self.fail_capture:
            raise RuntimeError("injected capture failure")
        self.capturing = graph
        try:
            yield
        finally:
            self.capturing = None


class ObservedCompute:
    def __init__(self, backend):
        self.backend = backend
        self.calls = 0
        self.evaluations = []

    def evaluate(self, hidden, tokens):
        route = tokens.remainder(2).to(torch.float32)
        self.evaluations.append((hidden.clone(), tokens.clone()))
        return hidden.float() * (route[:, None] + 1) + tokens[:, None].float()

    def __call__(self, hidden, tokens):
        self.calls += 1
        output = self.evaluate(hidden, tokens)
        graph = self.backend.capturing
        if graph is not None:
            graph.output = output
            graph.compute = lambda: self.evaluate(hidden, tokens)
        return output


def _inputs(token=2):
    return torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16), torch.tensor([token], dtype=torch.int64)


def test_graph_replays_new_hidden_and_tokens_even_when_route_is_unchanged():
    assert GRAPH_WARMUPS == 2
    backend = FakeBackend()
    graph = MoEDecodeGraph("cpu", backend=backend)
    compute = ObservedCompute(backend)
    hidden, tokens = _inputs()
    first = graph.run(compute, hidden, tokens)
    assert torch.equal(first, hidden.float() + 2)
    captured_addresses = (graph.hidden.data_ptr(), graph.input_ids.data_ptr(), graph.output.data_ptr())
    hidden.add_(3)
    second = graph.run(compute, hidden, tokens)
    assert torch.equal(second, hidden.float() + 2)
    tokens.fill_(4)  # Same route (0), different hash/token-dependent input.
    third = graph.run(compute, hidden, tokens)
    assert torch.equal(third, hidden.float() + 4)
    tokens.fill_(5)  # Route changes too.
    fourth = graph.run(compute, hidden, tokens)
    assert torch.equal(fourth, hidden.float() * 2 + 5)
    assert torch.equal(first, torch.tensor([[3, 4, 5, 6]], dtype=torch.float32))
    assert len({value.data_ptr() for value in (first, second, third, fourth, graph.output)}) == 5
    assert captured_addresses == (graph.hidden.data_ptr(), graph.input_ids.data_ptr(), graph.output.data_ptr())
    assert graph.hidden.data_ptr() != hidden.data_ptr()
    assert graph.input_ids.data_ptr() != tokens.data_ptr()
    assert compute.calls == GRAPH_WARMUPS + 1
    assert len(compute.evaluations) == GRAPH_WARMUPS + 1 + 4
    assert backend.capture_entries == graph.captures == len(backend.graphs) == 1
    assert backend.stream.synchronizations == 1
    assert graph.replays == backend.graphs[0].replays == 4
    assert graph.report() == {
        "scope": "moe_decode",
        "captures": 1,
        "replays": 4,
        "entries": 1,
        "failed": False,
        "full_model_graph_verified": False,
    }


@pytest.mark.parametrize("change", ["stream", "hidden_width", "token_dtype"])
def test_graph_rejects_stream_or_signature_change_without_recapture(change):
    backend = FakeBackend()
    graph = MoEDecodeGraph("cpu", backend=backend)
    compute = ObservedCompute(backend)
    hidden, tokens = _inputs()
    graph.run(compute, hidden, tokens)
    if change == "stream":
        backend.stream = FakeStream(29)
        error, message = RuntimeError, "single owner stream"
    else:
        if change == "hidden_width":
            hidden = torch.ones((1, 8), dtype=torch.bfloat16)
        else:
            tokens = tokens.to(torch.int32)
        error, message = ValueError, "signature changed"
    with pytest.raises(error, match=message):
        graph.run(compute, hidden, tokens)
    assert graph.captures == backend.capture_entries == 1
    assert graph.replays == 1
    assert compute.calls == GRAPH_WARMUPS + 1


@pytest.mark.parametrize("change", ["batch", "dtype", "token_rank", "token_dtype", "no_tokens"])
def test_graph_rejects_invalid_inputs_before_warmup(change):
    backend = FakeBackend()
    graph = MoEDecodeGraph("cpu", backend=backend)
    compute = ObservedCompute(backend)
    hidden, tokens = _inputs()
    if change == "batch":
        hidden = hidden.repeat(2, 1)
    elif change == "dtype":
        hidden = hidden.float()
    elif change == "token_rank":
        tokens = tokens.reshape(1, 1)
    elif change == "token_dtype":
        tokens = tokens.float()
    else:
        tokens = None
    with pytest.raises(ValueError, match="B1 BF16"):
        graph.run(compute, hidden, tokens)
    assert compute.calls == backend.capture_entries == graph.captures == graph.replays == 0


@pytest.mark.parametrize("failure", ["warmup", "capture", "replay"])
def test_graph_execution_failure_latches_without_eager_fallback(failure):
    backend = FakeBackend()
    graph = MoEDecodeGraph("cpu", backend=backend)
    observed = ObservedCompute(backend)

    def compute(hidden, tokens):
        if failure == "warmup":
            raise RuntimeError("injected warmup failure")
        return observed(hidden, tokens)

    backend.fail_capture = failure == "capture"
    backend.fail_replay = failure == "replay"
    hidden, tokens = _inputs()
    with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
        graph.run(compute, hidden, tokens)
    assert graph.failed
    before = (observed.calls, backend.capture_entries, graph.captures, graph.replays)
    backend.fail_capture = backend.fail_replay = False
    with pytest.raises(RuntimeError, match="previously failed"):
        graph.run(observed, hidden, tokens)
    assert before == (observed.calls, backend.capture_entries, graph.captures, graph.replays)
    assert graph.report()["failed"]


def test_graph_rejects_recursive_use_and_releases_lock_after_failure():
    backend = FakeBackend()
    graph = MoEDecodeGraph("cpu", backend=backend)
    hidden, tokens = _inputs()

    def compute(captured_hidden, captured_tokens):
        return graph.run(compute, captured_hidden, captured_tokens)

    with pytest.raises(RuntimeError, match="concurrently or recursively"):
        graph.run(compute, hidden, tokens)
    assert graph.failed
    assert not graph._lock.locked()
    assert backend.capture_entries == 0


def test_graph_rejects_concurrent_use_without_corrupting_owner_run():
    backend = FakeBackend()
    graph = MoEDecodeGraph("cpu", backend=backend)
    compute = ObservedCompute(backend)
    hidden, tokens = _inputs()
    entered, release = Event(), Event()
    results, failures = [], []

    def blocking_compute(captured_hidden, captured_tokens):
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test owner was not released")
        return compute(captured_hidden, captured_tokens)

    def owner():
        try:
            results.append(graph.run(blocking_compute, hidden, tokens))
        except Exception as error:
            failures.append(error)

    worker = Thread(target=owner)
    worker.start()
    try:
        assert entered.wait(timeout=5)
        with pytest.raises(RuntimeError, match="concurrently or recursively"):
            graph.run(compute, hidden, tokens)
        assert not graph.failed
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert not failures
    assert len(results) == 1
    assert torch.equal(results[0], hidden.float() + 2)
    assert graph.captures == graph.replays == 1
    assert not graph._lock.locked()


def test_default_backend_does_not_silently_run_a_cpu_graph():
    with pytest.raises(ValueError, match="requires an NPU"):
        MoEDecodeGraph("cpu")
