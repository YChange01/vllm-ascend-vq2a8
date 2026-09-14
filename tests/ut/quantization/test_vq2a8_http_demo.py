# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only HTTP/ownership/contracts. No NPU inference or serving readiness claim."""

import http.client
import json
import subprocess
import sys
import threading
from contextlib import closing, contextmanager
from types import SimpleNamespace as NS

import pytest

from tools import serve_vq2a8_demo as demo


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "text",
        {},
        {"prompt": ""},
        {"prompt": ["a", "b"]},
        *({"prompt": "x", "max_tokens": value} for value in (0, -1, 128, 1.5, True, "4", None)),
        *({"prompt": "x", "temperature": value} for value in (True, 0.1, "0", None, float("nan"))),
        *({"prompt": "x", "stream": value} for value in (True, 0, None)),
        *({"prompt": "x", "n": value} for value in (0, 2, True, 1.0)),
        {"prompt": "x", "model": "other"},
        {"prompt": "x", "messages": []},
        {"prompt": "x", "ignore_eos": "false"},
        {"prompt": "x", "ignore_eos": 0},
    ],
)
def test_request_rejects_unsupported_features(payload):
    with pytest.raises(ValueError):
        demo.validate_request(payload)


def test_request_defaults_and_greedy_contract():
    assert demo.validate_request({"prompt": "hello"}) == ("hello", 32, False)
    assert demo.validate_request({"prompt": "hello", "max_tokens": 4, "ignore_eos": True}) == ("hello", 4, True)


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return NS(ids=[2] * len(text))

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "你好" + str(tokens)


def test_context_counts_bos_no_silent_truncation():
    assert demo.encode_prompt(Tokenizer(), "a" * 123, 0, 4) == [0] + [2] * 123
    with pytest.raises(ValueError, match="prompt=125"):
        demo.encode_prompt(Tokenizer(), "a" * 124, 0, 4)
    # Do not insert a second BOS if the raw prompt already encoded one.
    assert demo.encode_prompt(NS(encode=lambda *a, **kw: NS(ids=[0, 3])), "x", 0, 4) == [0, 3]


@pytest.fixture
def acceptance(tmp_path):
    repo = tmp_path / "repo"
    for name in demo.REQUIRED_SOURCES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    library = tmp_path / "kernel.so"
    library.write_bytes(b"fake-CPU-test-library-not-loaded")
    model = tmp_path / "model"
    model.mkdir()
    path = tmp_path / "report/result/summary.json"
    path.parent.mkdir(parents=True)
    report = {
        "status": "PASS",
        "optimization_sources_unchanged": True,
        "library": {"sha256": demo.digest(library)},
        "engine_options": {"model": str(model)},
        "optimization_source_sha256": {name: demo.digest(repo / name) for name in demo.REQUIRED_SOURCES},
        "candidates": [
            {
                "preset": "batched",
                "case": "p10-o4",
                "status": "PASS",
                "numerical": {"accepted": True},
                "repeat": {"accepted": True},
            }
        ],
    }
    path.write_text(json.dumps(report))
    return repo, library, model, path, report


def test_acceptance_pins_library_sources_model_and_exact_repeat(acceptance):
    repo, library, model, path, report = acceptance
    record = demo.checked_acceptance(path.parent.parent, model, library, "batched", repo=repo)
    assert record["accepted_cases"] == ["p10-o4"]
    assert record["preflight"] == path.parent.parent / "preflight/preflight.json"
    with pytest.raises(ValueError, match="preset fast"):
        demo.checked_acceptance(path, model, library, "fast", repo=repo)
    with pytest.raises(ValueError, match="Model path"):
        demo.checked_acceptance(path, model / "different", library, "batched", repo=repo)
    report["candidates"][0]["repeat"]["accepted"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="exact/repeat"):
        demo.checked_acceptance(path, model, library, "batched", repo=repo)


@pytest.mark.parametrize("kind", ["source", "missing_source", "library", "status", "source_status"])
def test_changed_or_incomplete_acceptance_rejected(acceptance, kind):
    repo, library, model, path, report = acceptance
    if kind == "source":
        (repo / demo.REQUIRED_SOURCES[0]).write_bytes(b"changed")
    elif kind == "missing_source":
        report["optimization_source_sha256"].pop(demo.REQUIRED_SOURCES[0])
    elif kind == "library":
        library.write_bytes(b"changed")
    elif kind == "status":
        report["status"] = "RUNNING"
    else:
        report["optimization_sources_unchanged"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        demo.checked_acceptance(path, model, library, "batched", repo=repo)


@pytest.fixture
def runtime(monkeypatch):
    calls = []
    engine = NS(has_unfinished_requests=lambda: False)

    def generate(prompts, params, **kwargs):
        calls.append((prompts, params, kwargs, threading.get_ident()))
        return [NS(finished=True, outputs=[NS(token_ids=[3, 4, 5, 1], finish_reason="length")])]

    monkeypatch.setitem(sys.modules, "vllm", NS(SamplingParams=lambda **kwargs: NS(**kwargs)))
    benchmark = NS(configure=lambda *a, **kw: None, snapshot=lambda llm: {"finite": True})
    monkeypatch.setitem(sys.modules, "tools.benchmark_vq2a8_offline", benchmark)
    runner = demo.DemoRuntime(
        NS(generate=generate, llm_engine=engine),
        Tokenizer(),
        {"bos_token_id": 0, "eos_token_id": 1, "vocab_size": 100},
        "batched",
        {"accepted_cases": ["p10-o4"], "library_sha256": "abc"},
    )
    return runner, calls, benchmark


def test_runtime_response_finite_usage_and_ignore_eos(runtime):
    runner, calls, _ = runtime
    result = runner.complete({"prompt": "hi", "max_tokens": 4, "ignore_eos": True})
    assert calls[0][1].stop_token_ids == []
    assert calls[0][1].ignore_eos is True
    assert calls[0][3] == runner.owner_thread
    assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    assert result["choices"][0]["text"].startswith("你好")
    assert result["vq2a8"]["finite"] is True
    runner.complete({"prompt": "hi", "max_tokens": 4})
    assert calls[1][1].stop_token_ids == [1]
    assert calls[1][1].ignore_eos is False


def test_invalid_context_does_not_poison_engine(runtime):
    runner, calls, _ = runtime
    with pytest.raises(ValueError, match="Context limit"):
        runner.complete({"prompt": "x" * 126, "max_tokens": 4})
    assert runner.healthy and not calls


@pytest.mark.parametrize("kind", ["nonfinite", "pending", "error", "incomplete", "too_many", "bad_token"])
def test_runtime_fails_closed_after_inference_failure(runtime, kind):
    runner, _, benchmark = runtime
    if kind == "nonfinite":
        benchmark.snapshot = lambda llm: {"finite": False}
    elif kind == "pending":
        runner.llm.llm_engine.has_unfinished_requests = lambda: True
    elif kind == "error":

        def fail(*a, **kw):
            raise ValueError("NPU error")

        runner.llm.generate = fail
    else:
        tokens = [2, 3] if kind == "incomplete" else [2] * 5 if kind == "too_many" else [-1] * 4
        runner.llm.generate = lambda *a, **kw: [NS(finished=True, outputs=[NS(token_ids=tokens)])]
    with pytest.raises((ValueError, RuntimeError)):
        runner.complete({"prompt": "hello", "max_tokens": 4, "ignore_eos": True})
    assert not runner.healthy
    with pytest.raises(RuntimeError, match="unhealthy"):
        runner.complete({"prompt": "hello"})


def test_runtime_rejects_foreign_thread_without_touching_engine(runtime):
    runner, calls, _ = runtime
    errors = []

    def wrong_thread():
        try:
            runner.complete({"prompt": "hello"})
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join(timeout=3)
    assert errors and "initialization thread" in errors[0]
    assert not calls


@contextmanager
def http_fixture(runtime):
    with demo.DemoServer(("127.0.0.1", 0), demo.DemoHandler) as server:
        server.runtime = runtime
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=3)


def http_request(server, method, path, body=None, headers=None):
    with closing(http.client.HTTPConnection(*server.server_address, timeout=3)) as client:
        client.request(method, path, body=body, headers=headers or {})
        response = client.getresponse()
        return response.status, json.loads(response.read())


def test_real_loopback_http_routes_validation_unicode_and_failed_health():
    calls = []

    def complete(payload):
        calls.append(payload)
        return {"choices": [{"text": "你好"}], "usage": {}, "vq2a8": {"generation_s": 0.1}}

    fake = NS(healthy=True, health=lambda: {"status": "ready"}, complete=complete)
    with http_fixture(fake) as server:
        assert http_request(server, "GET", "/health")[0] == 200
        assert http_request(server, "GET", "/v1/models")[1]["data"][0]["id"] == "vq2a8"
        assert http_request(server, "GET", "/../../config.json")[0] == 404
        status, result = http_request(
            server, "POST", "/v1/completions", json.dumps({"prompt": "hi"}), {"Content-Type": "application/json"}
        )
        assert status == 200 and result["choices"][0]["text"] == "你好"
        assert http_request(server, "POST", "/v1/chat/completions", "{}")[0] == 404
        for body, headers in [
            ("{}", {}),
            ("not json", {"Content-Type": "application/json"}),
            (json.dumps({"prompt": "hi", "stream": True}), {"Content-Type": "application/json"}),
            (" " * (demo.MAX_BODY_BYTES + 1), {"Content-Type": "application/json"}),
        ]:
            assert http_request(server, "POST", "/v1/completions", body, headers)[0] == 400
        assert len(calls) == 1
        fake.healthy = False
        assert http_request(server, "GET", "/health")[0] == 503
        assert http_request(server, "POST", "/v1/completions", "{}")[0] == 503


def test_help_without_npu_import():
    result = subprocess.run(
        [sys.executable, str(demo.REPO / "tools/serve_vq2a8_demo.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0 and "--acceptance-report" in result.stdout
