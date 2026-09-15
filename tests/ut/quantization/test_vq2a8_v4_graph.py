# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU graph protocol and arithmetic contracts, not NPU capture evidence."""

import gc
import weakref
from contextlib import contextmanager
from copy import copy
from threading import Event, Thread

import pytest
import torch

from tests.ut.quantization.test_vq2a8_v4_device_route import make_runtime, no_host_tensor_reads
from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation
from vllm_ascend.quantization.vq2a8_optimization import configure_runtime
from vllm_ascend.quantization.vq2a8_v4_device_route import DeviceRouteGraphCompute
from vllm_ascend.quantization.vq2a8_v4_graph import GRAPH_WARMUPS, V4MoEDecodeGraph


class FakeStream:
    def __init__(self, stream_id=19):
        self.npu_stream = stream_id
        self.synchronizations = 0
        self.fail_sync = False
        self.waits = []

    def wait_stream(self, stream):
        self.waits.append(stream.npu_stream)

    def synchronize(self):
        if self.fail_sync:
            raise RuntimeError("injected synchronize failure")
        self.synchronizations += 1


class FakeGraph:
    def __init__(self, backend):
        self.backend = backend
        self.compute = self.outputs = None
        self.replays = self.resets = 0
        self.replay_streams = []

    def replay(self):
        if self.backend.fail_replay:
            raise RuntimeError("injected replay failure")
        self.replays += 1
        self.replay_streams.append(self.backend.current.npu_stream)
        # This models static device destinations, but cannot emulate native
        # queue/allocator behavior. Real NPU graph tests are a separate gate.
        for destination, value in zip(self.outputs, self.compute()):
            destination.copy_(value)

    def reset(self):
        if self.backend.fail_reset:
            raise RuntimeError("injected reset failure")
        self.resets += 1
        self.compute = self.outputs = None


class FakeBackend:
    def __init__(self):
        self.current = FakeStream()
        self.streams = [self.current]
        self.capturing = None
        self.graphs = []
        self.capture_entries = 0
        self.fail_capture = self.fail_replay = self.fail_reset = False
        self.synchronizations = 0
        self.fail_synchronize = False

    def current_stream(self, device):
        assert device == torch.device("cpu")
        return self.current

    def synchronize(self, device):
        assert device == torch.device("cpu")
        if self.fail_synchronize:
            raise RuntimeError("injected device synchronize failure")
        self.synchronizations += 1

    def Stream(self, *, device):
        assert device == torch.device("cpu")
        stream = FakeStream(19 + 10 * len(self.streams))
        self.streams.append(stream)
        return stream

    @contextmanager
    def stream(self, stream):
        previous = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = previous

    def is_current_stream_capturing(self):
        return self.capturing is not None

    def NPUGraph(self):
        graph = FakeGraph(self)
        self.graphs.append(graph)
        return graph

    @contextmanager
    def graph(self, graph, stream):
        assert stream is self.current
        self.capture_entries += 1
        if self.fail_capture:
            raise RuntimeError("injected capture failure")
        self.capturing = graph
        try:
            yield
        finally:
            self.capturing = None


class ObservedCompute:
    def __init__(self, backend, compute=None):
        self.backend = backend
        self.compute = compute or self.evaluate
        self.calls = 0

    @staticmethod
    def evaluate(hidden, tokens):
        route = tokens.remainder(2).float()
        output = (hidden.float() * (route[:, None] + 1) + tokens[:, None].float()).bfloat16()
        return output, torch.isfinite(hidden).all() & (tokens >= 0).all()

    def __call__(self, hidden, tokens):
        self.calls += 1
        outputs = self.compute(hidden, tokens)
        graph = self.backend.capturing
        if graph is not None:
            graph.outputs = outputs
            graph.compute = lambda: self.compute(hidden, tokens)
        return outputs


def inputs(token=2, dtype=torch.int64):
    return torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16), torch.tensor([token], dtype=dtype)


def prepared_graph():
    backend = FakeBackend()
    graph = V4MoEDecodeGraph("cpu", backend=backend)
    compute = ObservedCompute(backend)
    hidden, tokens = inputs()
    graph.prepare(compute, hidden, tokens)
    return backend, graph, compute


def prepared_caller_graph():
    backend = FakeBackend()
    caller = backend.current
    capture = backend.Stream(device=torch.device("cpu"))
    graph = V4MoEDecodeGraph("cpu", backend=backend)
    compute = ObservedCompute(backend)
    with backend.stream(capture):
        graph.prepare(compute, *inputs(), replay_stream=caller)
    return backend, graph, compute


def prepared_state(monkeypatch, *, policy="owner"):
    runtime, hidden = make_runtime(hash_route=True)
    configure_runtime(runtime, "device_route_decode")
    state, backend = runtime._optimization, FakeBackend()
    real_compute = DeviceRouteGraphCompute

    def wrapped_compute(current, *, banks=None):
        compute = real_compute(current, banks=banks)
        observed = ObservedCompute(backend, compute)
        observed.signature = compute.signature
        observed.check_runtime_contract = compute.check_runtime_contract
        return observed

    monkeypatch.setattr("vllm_ascend.quantization.vq2a8_v4_device_route.DeviceRouteGraphCompute", wrapped_compute)
    metadata = {**runtime._device_route_banks, "metadata_bytes": 512}
    monkeypatch.setattr("vllm_ascend.quantization.vq2a8_v4_device_route.create_device_route_banks", lambda _: metadata)
    state.prepare_graph(runtime, backend=backend, replay_stream_policy=policy)
    return runtime, hidden, state, backend


def test_explicit_prepare_only_and_dynamic_replay_preserves_old_output_and_valid():
    backend, graph, compute = prepared_graph()
    assert GRAPH_WARMUPS == 2
    assert compute.calls == GRAPH_WARMUPS + 1
    assert graph.replays == 0
    addresses = tuple(t.data_ptr() for t in (graph.hidden, graph.input_ids, graph.output, graph.valid))
    retained = []
    for token, factor, dtype in ((2, 1, torch.int64), (3, 2, torch.int32), (-1, 1, torch.int64), (2, 1, torch.int32)):
        hidden, tokens = inputs(token, dtype)
        hidden *= factor
        actual = graph.replay(hidden, tokens)
        expected = compute.evaluate(hidden, tokens)
        assert all(torch.equal(a, e) for a, e in zip(actual, expected))
        retained.append(actual)
    assert torch.equal(retained[0][0], retained[3][0])
    assert bool(retained[0][1]) and not bool(retained[2][1]) and bool(retained[3][1])
    assert len({item.data_ptr() for pair in retained for item in pair}) == 8
    assert addresses == tuple(t.data_ptr() for t in (graph.hidden, graph.input_ids, graph.output, graph.valid))
    assert graph.input_ids.dtype == torch.int64
    assert compute.calls == GRAPH_WARMUPS + 1  # Python capture callback is not replayed.
    assert backend.capture_entries == graph.captures == 1
    assert graph.replays == 4
    assert backend.current.synchronizations == 2  # No replay fence.
    report = graph.snapshot()
    assert report["pool_count"] == report["entries"] == 1
    assert not report["graph_functional_verified"]
    assert not report["full_model_graph_verified"]
    assert report["capture_peak_reserved_bytes"] is None


def test_prepare_is_single_shot_and_replay_cannot_lazy_capture():
    backend = FakeBackend()
    graph = V4MoEDecodeGraph("cpu", backend=backend)
    hidden, tokens = inputs()
    compute = ObservedCompute(backend)
    with pytest.raises(RuntimeError, match="lazy capture"):
        graph.replay(hidden, tokens)
    assert compute.calls == backend.capture_entries == 0
    graph.prepare(compute, hidden, tokens)
    with pytest.raises(RuntimeError, match="single-shot"):
        graph.prepare(compute, hidden, tokens)
    assert graph.captures == 1


@pytest.mark.parametrize("change", ["stream", "shape", "hidden_dtype", "token_dtype", "missing_token", "batch"])
def test_invalid_replay_is_rejected_before_submission_without_recapture(change):
    backend, graph, compute = prepared_graph()
    hidden, tokens = inputs()
    if change == "stream":
        backend.current = FakeStream(29)
    elif change == "shape":
        hidden = torch.ones((1, 8), dtype=torch.bfloat16)
    elif change == "hidden_dtype":
        hidden = hidden.float()
    elif change == "token_dtype":
        tokens = tokens.float()
    elif change == "missing_token":
        tokens = None
    else:
        hidden = hidden.repeat(2, 1)
    with pytest.raises((ValueError, RuntimeError)):
        graph.replay(hidden, tokens)
    assert graph.replays == graph.replay_attempts == 0
    assert graph.captures == 1 and compute.calls == GRAPH_WARMUPS + 1


@pytest.mark.parametrize("failure", ["warmup", "capture", "replay"])
def test_execution_error_is_sticky_and_close_remains_available(failure):
    backend = FakeBackend()
    graph = V4MoEDecodeGraph("cpu", backend=backend)
    observed = ObservedCompute(backend)

    def compute(hidden, tokens):
        if failure == "warmup":
            raise RuntimeError("injected warmup failure")
        return observed(hidden, tokens)

    backend.fail_capture = failure == "capture"
    backend.fail_replay = failure == "replay"
    with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
        graph.prepare(compute, *inputs())
        graph.replay(*inputs())
    assert graph.failed
    before = graph.snapshot()
    with pytest.raises(RuntimeError, match="previously failed"):
        graph.replay(*inputs())
    assert before == graph.snapshot()
    graph.close()
    assert graph.closed and graph.failed


@pytest.mark.parametrize("failure", ["synchronize", "reset"])
def test_close_failure_retains_owners_then_successful_close_releases_them(failure):
    class Owner:
        pass

    owner = Owner()
    reference = weakref.ref(owner)
    backend = FakeBackend()
    graph = V4MoEDecodeGraph("cpu", backend=backend, owners=(owner,))
    graph.prepare(ObservedCompute(backend), *inputs())
    del owner
    backend.current.fail_sync = failure == "synchronize"
    backend.fail_reset = failure == "reset"
    with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
        graph.close()
    gc.collect()
    assert reference() is not None
    assert graph.graph is not None and graph.output is not None
    assert graph.closing and not graph.closed
    with pytest.raises(RuntimeError, match="closed or closing"):
        graph.replay(*inputs())
    backend.current.fail_sync = backend.fail_reset = False
    graph.close()
    gc.collect()
    assert reference() is None
    assert graph.owners == () and graph.graph is None and graph.output is None
    graph.close()  # Idempotent only after successful teardown.
    assert backend.graphs[0].resets == 1


def test_prepare_rejects_recursive_and_concurrent_use():
    backend = FakeBackend()
    graph = V4MoEDecodeGraph("cpu", backend=backend)

    def recurse(hidden, tokens):
        graph.prepare(recurse, hidden, tokens)

    with pytest.raises(RuntimeError, match="concurrently or recursively"):
        graph.prepare(recurse, *inputs())
    assert graph.failed and not graph._lock.locked()
    graph.close()

    backend = FakeBackend()
    graph = V4MoEDecodeGraph("cpu", backend=backend)
    observed = ObservedCompute(backend)
    entered, release = Event(), Event()
    failures = []

    def compute(hidden, tokens):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("test owner not released")
        return observed(hidden, tokens)

    def owner():
        try:
            graph.prepare(compute, *inputs())
        except Exception as error:
            failures.append(error)

    worker = Thread(target=owner)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="concurrently or recursively"):
            graph.close()
        assert not graph.failed
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and not failures
    assert graph.prepared and not graph._lock.locked()


def test_nested_capture_is_rejected_and_default_cpu_backend_is_forbidden():
    backend, graph, _ = prepared_graph()
    backend.capturing = object()
    with pytest.raises(RuntimeError, match="nested capture"):
        graph.replay(*inputs())
    assert graph.replays == 0
    with pytest.raises(ValueError, match="requires an NPU"):
        V4MoEDecodeGraph("cpu")


@pytest.mark.parametrize("hash_route", [False, True])
def test_pure_compute_matches_eager_without_changing_counters_or_historical_validity(monkeypatch, hash_route):
    runtime, hidden = make_runtime(hash_route=hash_route)
    configure_runtime(runtime, "device_route_decode")
    state = runtime._optimization
    compute = DeviceRouteGraphCompute(runtime)
    original_preparation = runtime._row_preparation
    for token, factor in ((0, 1), (1, -1), (0, 0.125), (1, 2)):
        x, tokens = hidden * factor, torch.tensor([token])
        expected = state.forward(runtime, x, tokens)
        state.valid = torch.tensor(False)  # Must not contaminate pure compute.
        before = (state.report(), runtime.native_calls, runtime.native_rows, runtime.native_launches)
        with no_host_tensor_reads(monkeypatch):
            actual, valid = compute(x, tokens)
        assert torch.equal(actual, expected) and bool(valid)
        assert not bool(state.valid)
        assert before == (state.report(), runtime.native_calls, runtime.native_rows, runtime.native_launches)
        assert runtime._row_preparation is original_preparation


def test_pure_compute_invalid_then_valid_has_no_stale_flag():
    runtime, hidden = make_runtime(hash_route=True)
    configure_runtime(runtime, "device_route_decode")
    compute = DeviceRouteGraphCompute(runtime)
    assert not bool(compute(hidden, torch.tensor([-1]))[1])
    assert bool(compute(hidden, torch.tensor([0]))[1])
    invalid = hidden.clone()
    invalid[0, 0] = float("nan")
    assert not bool(compute(invalid, torch.tensor([0]))[1])
    assert bool(compute(hidden, torch.tensor([0]))[1])


def test_graph_preparation_has_separate_frozen_constants_and_no_lazy_rebuild(monkeypatch):
    runtime, hidden = make_runtime(hash_route=True)
    configure_runtime(runtime, "device_route_decode")
    # Real artifacts may use different block geometry in the two projections.
    bank, spec = runtime._device_route_banks["down"]
    spec = copy(spec)
    spec.rht_block_size = 16
    runtime._device_route_banks["down"] = bank, spec
    compute = DeviceRouteGraphCompute(runtime)
    pointers = tuple(p._hadamard.data_ptr() for p in compute.preparations.values())
    assert len(set(pointers)) == 2

    def reject(*args, **kwargs):
        pytest.fail("Hadamard must already exist before capture")

    monkeypatch.setattr("vllm_ascend.quantization.vq2a8_activation._sylvester_hadamard", reject)
    for _ in range(2):
        assert bool(compute(hidden, torch.tensor([0]))[1])
    assert pointers == tuple(p._hadamard.data_ptr() for p in compute.preparations.values())
    with pytest.raises(RuntimeError, match="geometry changed"):
        compute.preparations["gate_up"].prepare_for_graph("cpu", 16)


def test_state_graph_wrappers_keep_eager_counters_and_defer_replay_validity(monkeypatch):
    runtime, hidden = make_runtime(hash_route=True)
    configure_runtime(runtime, "device_route_decode")
    state = runtime._optimization
    backend = FakeBackend()
    real_compute = DeviceRouteGraphCompute

    # Install the fake graph recorder around real CPU tensor arithmetic.
    def wrapped_compute(current, *, banks=None):
        compute = real_compute(current, banks=banks)
        observed = ObservedCompute(backend, compute)
        observed.signature = compute.signature
        observed.check_runtime_contract = compute.check_runtime_contract
        return observed

    monkeypatch.setattr("vllm_ascend.quantization.vq2a8_v4_device_route.DeviceRouteGraphCompute", wrapped_compute)
    eager_banks = runtime._device_route_banks
    metadata_banks = {**eager_banks, "metadata_bytes": 512}
    monkeypatch.setattr(
        "vllm_ascend.quantization.vq2a8_v4_device_route.create_device_route_banks", lambda _: metadata_banks
    )
    stream_records = []
    monkeypatch.setattr(
        "vllm_ascend.quantization.vq2a8_v4_device_route._record_tensor_stream",
        lambda tensor, stream: stream_records.append((tensor.data_ptr(), stream.npu_stream)),
    )
    before = state.report(), runtime.native_calls
    state.prepare_graph(runtime, backend=backend)
    assert before == (state.report(), runtime.native_calls)
    assert state.graph_snapshot()["captures"] == 1
    assert state.valid is None
    with no_host_tensor_reads(monkeypatch):
        output = state.forward_graph(runtime, hidden, torch.tensor([0], dtype=torch.int32))
    assert output.dtype == torch.bfloat16 and bool(state.valid)
    assert before == (state.report(), runtime.native_calls)
    assert state.graph_snapshot()["replays"] == 1
    assert runtime._device_route_banks is eager_banks
    assert state._graph_banks is metadata_banks
    assert state.graph_snapshot()["graph_metadata_bytes"] == 512
    assert state.graph_snapshot()["graph_payload_copy_bytes"] == 0
    assert state.graph_snapshot()["stream_bridges"] == 1
    assert backend.current.npu_stream == 19 and state._graph_stream.npu_stream == 29
    assert state._graph_stream.waits == [19, 19]  # Initialization, then replay.
    assert backend.current.waits == [29, 29]
    assert [stream for _, stream in stream_records] == [29, 29, 19, 19]
    assert backend.current.synchronizations == 0
    state.close_graph()
    with pytest.raises(RuntimeError, match="closed or closing"):
        state.forward_graph(runtime, hidden, torch.tensor([0]))


def test_graph_compute_requires_explicit_device_route_configuration():
    runtime, _ = make_runtime()
    with pytest.raises(ValueError, match="Configure device_route_decode"):
        DeviceRouteGraphCompute(runtime)
    preparation = RowwiseVQ2A8Preparation()
    with pytest.raises(ValueError, match="power-of-two"):
        preparation.prepare_for_graph("cpu", 3)


@pytest.mark.parametrize("change", ["root", "top_k", "geometry"])
def test_runtime_signature_rejects_changed_constants_without_rebuilding(change):
    runtime, _ = make_runtime()
    configure_runtime(runtime, "device_route_decode")
    compute = DeviceRouteGraphCompute(runtime)
    compute.check_runtime_contract(runtime)
    if change == "root":
        runtime.root["gate.weight"] = runtime.root["gate.weight"].clone()
    elif change == "top_k":
        runtime.config.top_k = 1
    else:
        runtime._device_route_banks["down"][1].rht_block_size = 16
    with pytest.raises(RuntimeError, match="signature changed"):
        compute.check_runtime_contract(runtime)


def test_state_metadata_preparation_failure_is_single_shot_and_close_fences(monkeypatch):
    runtime, _ = make_runtime()
    configure_runtime(runtime, "device_route_decode")
    state, backend = runtime._optimization, FakeBackend()

    def fail(_):
        raise RuntimeError("injected metadata failure")

    monkeypatch.setattr("vllm_ascend.quantization.vq2a8_v4_device_route.create_device_route_banks", fail)
    with pytest.raises(RuntimeError, match="injected metadata failure"):
        state.prepare_graph(runtime, backend=backend)
    assert state.graph_snapshot()["failed"]
    assert state._graph_stream is not None
    with pytest.raises(RuntimeError, match="single-shot"):
        state.prepare_graph(runtime, backend=backend)
    stream = state._graph_stream
    stream.fail_sync = True
    with pytest.raises(RuntimeError, match="synchronize failure"):
        state.close_graph()
    assert state._graph_stream is stream
    stream.fail_sync = False
    state.close_graph()
    assert stream.synchronizations == 1 and state._graph_stream is None


def test_replay_has_bounded_owned_storage_under_long_cpu_protocol_stress():
    backend, graph, compute = prepared_graph()
    pointers = tuple(t.data_ptr() for t in (graph.hidden, graph.input_ids, graph.output, graph.valid))
    owners = graph.owners
    for index in range(2049):
        hidden, tokens = inputs(index % 2, torch.int32)
        output, valid = graph.replay(hidden, tokens)
        if index % 512 == 0:
            assert torch.equal(output, compute.evaluate(hidden, tokens)[0]) and bool(valid)
    assert pointers == tuple(t.data_ptr() for t in (graph.hidden, graph.input_ids, graph.output, graph.valid))
    assert graph.owners is owners
    assert graph.captures == len(backend.graphs) == 1 and graph.replays == 2049
    assert backend.current.synchronizations == 2
    graph.close()


def test_caller_replay_uses_bound_stream_preserves_outputs_and_owner_baseline():
    backend, graph, compute = prepared_caller_graph()
    addresses = tuple(t.data_ptr() for t in (graph.hidden, graph.input_ids, graph.output, graph.valid))
    retained = []
    for policy, token in (("caller", 2), ("owner", -1), ("caller", 3), ("owner", 2)):
        # Direct core API requires this boundary when changing the replay stream.
        graph._fence_streams()
        stream = graph.replay_stream if policy == "caller" else graph.stream
        hidden, tokens = inputs(token, torch.int32)
        with backend.stream(stream):
            actual = graph.replay(hidden, tokens, stream_policy=policy)
        expected = compute.evaluate(hidden, tokens)
        assert all(torch.equal(a, e) for a, e in zip(actual, expected))
        retained.append((actual, tuple(value.clone() for value in expected)))
    assert all(torch.equal(a, e) for actual, expected in retained for a, e in zip(actual, expected))
    assert addresses == tuple(t.data_ptr() for t in (graph.hidden, graph.input_ids, graph.output, graph.valid))
    report = graph.snapshot()
    assert report["owner_stream"] == 29 and report["replay_stream"] == 19
    assert report["owner_replays"] == report["caller_replays"] == 2
    assert report["replays"] == 4 and report["captures"] == 1
    assert backend.graphs[0].replay_streams == [19, 29, 19, 29]
    graph.close()


@pytest.mark.parametrize("policy", ["owner", "caller"])
def test_core_wrong_bound_replay_stream_rejects_before_submission(policy):
    backend, graph, _ = prepared_caller_graph()
    backend.current = FakeStream(77)
    before = graph.snapshot()
    with pytest.raises(RuntimeError, match=f"bound {policy} stream"):
        graph.replay(*inputs(), stream_policy=policy)
    assert before == graph.snapshot()
    assert not graph.failed and backend.graphs[0].replays == 0
    graph.close()


def test_core_rejects_unknown_replay_policy_and_wrong_device_caller():
    backend, graph, _ = prepared_caller_graph()
    with pytest.raises(ValueError, match="owner or caller"):
        graph.replay(*inputs(), stream_policy="auto")
    assert graph.replay_attempts == 0 and not graph.failed
    graph.close()
    graph = V4MoEDecodeGraph("cpu", backend=backend)
    caller = FakeStream(77)
    caller.device = torch.device("meta")
    with pytest.raises(ValueError, match="graph device"):
        graph.prepare(ObservedCompute(backend), *inputs(), replay_stream=caller)
    assert graph.graph is None and not graph.prepared


@pytest.mark.parametrize("failing_stream", ["owner", "caller"])
def test_close_fences_both_streams_and_keeps_pool_on_either_failure(failing_stream):
    backend, graph, _ = prepared_caller_graph()
    graph.replay(*inputs(), stream_policy="caller")
    target = graph.stream if failing_stream == "owner" else graph.replay_stream
    target.fail_sync = True
    owners = graph.owners
    with pytest.raises(RuntimeError, match="synchronize failure"):
        graph.close()
    assert graph.owners is owners and graph.graph is not None and graph.output is not None
    assert backend.graphs[0].resets == 0
    target.fail_sync = False
    graph.close()
    assert graph.stream.synchronizations >= 3  # Two preparation fences and close.
    assert graph.replay_stream.synchronizations >= 1
    assert backend.graphs[0].resets == 1 and graph.closed


@pytest.mark.parametrize("initial_policy", ["owner", "caller"])
def test_state_switches_same_graph_and_fences_only_policy_changes(monkeypatch, initial_policy):
    runtime, hidden, state, backend = prepared_state(monkeypatch, policy=initial_policy)
    graph = state._decode_graph
    eager_banks, captured_banks = runtime._device_route_banks, state._graph_banks
    pointers = tuple(t.data_ptr() for t in (graph.hidden, graph.input_ids, graph.output, graph.valid))
    previous = []
    records = []
    monkeypatch.setattr(
        "vllm_ascend.quantization.vq2a8_v4_device_route._record_tensor_stream",
        lambda tensor, stream: records.append(stream.npu_stream),
    )
    for policy, token in (("owner", 0), ("caller", 1), ("caller", 0), ("owner", 1)):
        old = state.graph_snapshot()["replay_stream_policy"]
        fences = graph.stream.synchronizations, graph.replay_stream.synchronizations
        device_fences = backend.synchronizations
        state.set_graph_replay_stream(policy)
        delta = int(old != policy)
        assert (graph.stream.synchronizations, graph.replay_stream.synchronizations) == tuple(f + delta for f in fences)
        assert backend.synchronizations == device_fences + delta
        expected = DeviceRouteGraphCompute(runtime)(hidden, torch.tensor([token]))[0]
        before_waits = list(graph.stream.waits), list(graph.replay_stream.waits)
        before_records = len(records)
        before_fences = graph.stream.synchronizations, graph.replay_stream.synchronizations
        before_device_fences = backend.synchronizations
        with no_host_tensor_reads(monkeypatch):
            output = state.forward_graph(runtime, hidden, torch.tensor([token], dtype=torch.int32))
        assert torch.equal(output, expected) and bool(state.valid)
        assert before_fences == (graph.stream.synchronizations, graph.replay_stream.synchronizations)
        assert before_device_fences == backend.synchronizations
        if policy == "caller":
            assert before_waits == (graph.stream.waits, graph.replay_stream.waits)
            assert before_records == len(records)
        previous.append((output, expected.clone()))
    assert all(torch.equal(actual, expected) for actual, expected in previous)
    assert pointers == tuple(t.data_ptr() for t in (graph.hidden, graph.input_ids, graph.output, graph.valid))
    assert runtime._device_route_banks is eager_banks and state._graph_banks is captured_banks
    report = state.graph_snapshot()
    assert report["captures"] == report["entries"] == report["pool_count"] == 1
    assert report["owner_replays"] == report["caller_replays"] == report["stream_bridges"] == 2
    assert backend.graphs[0].replay_streams == [29, 19, 19, 29]
    assert records == [29, 29, 19, 19] * 2
    state.close_graph()


def test_state_caller_mode_rejects_unbound_stream_without_latching(monkeypatch):
    runtime, hidden, state, backend = prepared_state(monkeypatch, policy="caller")
    caller = backend.current
    backend.current = FakeStream(77)
    before = state.graph_snapshot()
    with pytest.raises(RuntimeError, match="bound caller stream"):
        state.forward_graph(runtime, hidden, torch.tensor([0]))
    assert before == state.graph_snapshot() and not state._decode_graph.failed
    backend.current = caller
    state.forward_graph(runtime, hidden, torch.tensor([0]))
    assert state.graph_snapshot()["caller_replays"] == 1
    assert state.graph_snapshot()["stream_bridges"] == 0
    state.close_graph()


def test_state_switch_from_third_stream_owner_call_fences_device_without_rebinding(monkeypatch):
    runtime, hidden, state, backend = prepared_state(monkeypatch)
    caller = backend.current
    third_stream = FakeStream(77)
    with backend.stream(third_stream):
        state.forward_graph(runtime, hidden, torch.tensor([0]))
    assert backend.synchronizations == 0
    state.set_graph_replay_stream("caller")
    assert backend.synchronizations == 1
    assert state.graph_snapshot()["replay_stream"] == caller.npu_stream
    state.forward_graph(runtime, hidden, torch.tensor([1]))
    assert bool(state.valid)
    assert state.graph_snapshot()["stream_bridges"] == 1
    assert state.graph_snapshot()["caller_replays"] == state.graph_snapshot()["owner_replays"] == 1
    state.close_graph()


def test_state_replay_policy_rejects_before_ready_during_capture_and_after_close(monkeypatch):
    runtime, _ = make_runtime()
    configure_runtime(runtime, "device_route_decode")
    state = runtime._optimization
    with pytest.raises(RuntimeError, match="Prepare"):
        state.set_graph_replay_stream("caller")
    with pytest.raises(ValueError, match="owner or caller"):
        state.prepare_graph(runtime, backend=FakeBackend(), replay_stream_policy="auto")
    assert not state._graph_started
    _, _, state, backend = prepared_state(monkeypatch)
    with pytest.raises(ValueError, match="owner or caller"):
        state.set_graph_replay_stream("auto")
    backend.capturing = object()
    with pytest.raises(RuntimeError, match="outside another capture"):
        state.set_graph_replay_stream("caller")
    backend.capturing = None
    assert not state._decode_graph.failed
    state.close_graph()
    with pytest.raises(RuntimeError, match="closed or closing"):
        state.set_graph_replay_stream("caller")


@pytest.mark.parametrize("failing_stream", ["owner", "caller", "device"])
def test_state_policy_switch_failure_keeps_old_policy_and_every_owner(monkeypatch, failing_stream):
    _, _, state, backend = prepared_state(monkeypatch)
    graph = state._decode_graph
    target = graph.stream if failing_stream == "owner" else graph.replay_stream
    backend.fail_synchronize = failing_stream == "device"
    target.fail_sync = failing_stream != "device"
    owners, banks = graph.owners, state._graph_banks
    with pytest.raises(RuntimeError, match="synchronize failure"):
        state.set_graph_replay_stream("caller")
    assert state.graph_snapshot()["replay_stream_policy"] == "owner"
    assert graph.failed and graph.owners is owners and state._graph_banks is banks
    with pytest.raises(RuntimeError, match="previously failed"):
        state.set_graph_replay_stream("owner")
    target.fail_sync = backend.fail_synchronize = False
    state.close_graph()


def test_caller_replay_protocol_stress_keeps_static_owners_and_prior_outputs(monkeypatch):
    backend, graph, compute = prepared_caller_graph()
    owners = graph.owners
    addresses = tuple(t.data_ptr() for t in (graph.hidden, graph.input_ids, graph.output, graph.valid))
    preserved = []
    with no_host_tensor_reads(monkeypatch):
        for index in range(2049):
            hidden, tokens = inputs(index % 2, torch.int32)
            output, valid = graph.replay(hidden, tokens, stream_policy="caller")
            if index % 512 == 0:
                preserved.append(((output, valid), compute.evaluate(hidden, tokens)))
            del hidden, tokens, output, valid
    assert all(torch.equal(a, e) for actual, expected in preserved for a, e in zip(actual, expected))
    assert graph.owners is owners
    assert addresses == tuple(t.data_ptr() for t in (graph.hidden, graph.input_ids, graph.output, graph.valid))
    assert graph.caller_replays == graph.replays == 2049 and graph.owner_replays == 0
    assert graph.captures == len(backend.graphs) == 1
    assert graph.stream.synchronizations == 2 and graph.replay_stream.synchronizations == 0
    assert backend.graphs[0].replay_streams == [19] * 2049
    graph.close()
