# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import json
import subprocess
import sys
from http.client import IncompleteRead
from urllib.error import HTTPError

import pytest

from tools import benchmark_vq2a8_serving as bench


def test_script_help_runs_without_importing_npu_or_shadowing_standard_library():
    result = subprocess.run([sys.executable, bench.__file__, "--help"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--base-url" in result.stdout and "--warmups" in result.stdout


def event(text="", *, usage=None, empty_choices=False):
    return {"choices": [] if empty_choices else [{"index": 0, "text": text}], "usage": usage}


def stream(*events, done=True):
    data = "".join(f"data: {json.dumps(value)}\r\n\r\n" for value in events)
    if done:
        data += "data: [DONE]\r\n\r\n"
    return io.BytesIO(data.encode())


def test_stream_uses_real_token_count_and_excludes_usage_arrival_from_timing():
    values = stream(
        event(),
        event("several tokens in one chunk"),
        event(" more tokens"),
        event(),
        event(usage={"completion_tokens": 9}, empty_choices=True),
    )
    result = bench.measure_stream(values, 10, clock=iter([10.2, 10.6]).__next__)
    assert result["ttft_ms"] == pytest.approx(200)
    assert result["tpot_ms"] == pytest.approx(50)
    assert result["completion_tokens"] == 9


def test_single_token_has_null_tpot_and_whitespace_is_generated_content():
    result = bench.measure_stream(stream(event(" ", usage={"completion_tokens": 1})), 1, clock=lambda: 1.5)
    assert result == {"ttft_ms": 500, "tpot_ms": None, "completion_tokens": 1}


def test_sse_comments_multiline_json_and_utf8():
    payload = (
        ': keepalive\n\nevent: message\ndata: {"choices": [{"text": "天空"}],\n'
        'data: "usage": {"completion_tokens": 3}}\n\ndata: [DONE]\n\n'
    )
    result = bench.measure_stream(io.BytesIO(payload.encode()), 0, clock=lambda: 1)
    assert result == {"ttft_ms": 1000, "tpot_ms": 0, "completion_tokens": 3}


@pytest.mark.parametrize(
    "events,done,match",
    [
        ([event("text")], True, "completion_tokens"),
        ([event(usage={"completion_tokens": 1}, empty_choices=True)], True, "no generated text"),
        ([event("text", usage={"completion_tokens": 1})], False, "missing \\[DONE\\]"),
        ([{"error": {"message": "engine stopped"}}], True, "engine stopped"),
        ([event("text", usage={"completion_tokens": True})], True, "integer"),
        ([event("text", usage={"completion_tokens": -1})], True, "Negative"),
        ([event("text", usage={"completion_tokens": 0})], True, "positive"),
        ([event(None, usage={"completion_tokens": 1})], True, "string"),
        ([{"choices": [{"text": "one"}, {"text": "two"}]}], True, "single"),
    ],
)
def test_invalid_or_incomplete_stream_never_fabricates_metrics(events, done, match):
    with pytest.raises(ValueError, match=match):
        bench.measure_stream(stream(*events, done=done), 0, clock=lambda: 1)


def test_truncated_event_and_invalid_json_fail():
    for raw in (b'data: {"choices": []}', b"data: {broken}\n\n"):
        with pytest.raises(ValueError):
            bench.measure_stream(io.BytesIO(raw), 0)


@pytest.mark.parametrize("base_url", ["http://127.0.0.1:8000", "http://127.0.0.1:8000/v1/"])
def test_http_request_uses_openai_streaming_usage_and_custom_prompt(base_url):
    args = bench.parse_args(["--base-url", base_url, "--prompt", "hello", "--max-tokens", "8"])

    def opener(request, *, timeout):
        assert request.full_url == "http://127.0.0.1:8000/v1/completions"
        assert request.method == "POST" and timeout == 120
        payload = json.loads(request.data)
        assert payload == dict(
            model="vq2a8",
            prompt="hello",
            max_tokens=8,
            temperature=0,
            ignore_eos=True,
            n=1,
            stream=True,
            stream_options={"include_usage": True},
        )
        response = stream(event("one two", usage={"completion_tokens": 2}))
        response.headers = {"Content-Type": "text/event-stream; charset=utf-8"}
        return response

    result = bench.request_completion(args, opener=opener, clock=iter([10, 10.2]).__next__)
    assert result["ttft_ms"] == pytest.approx(200)
    assert result["completion_tokens"] == 2


def test_http_error_and_non_stream_response_are_not_timing_successes():
    def failed(*args, **kwargs):
        raise HTTPError("http://localhost", 503, "Unavailable", {}, io.BytesIO(b"engine unavailable"))

    with pytest.raises(ValueError, match="HTTP 503: engine unavailable"):
        bench.request_completion(bench.parse_args([]), opener=failed)

    def wrong_content(*args, **kwargs):
        response = io.BytesIO(b"{}")
        response.headers = {"Content-Type": "application/json"}
        return response

    with pytest.raises(ValueError, match="text/event-stream"):
        bench.request_completion(bench.parse_args([]), opener=wrong_content)


def test_cli_warmup_is_not_printed_as_a_measurement(monkeypatch, capsys):
    calls = []

    def request(args):
        calls.append(args)
        return {"ttft_ms": 123, "tpot_ms": None, "completion_tokens": 1}

    monkeypatch.setattr(bench, "request_completion", request)
    assert bench.main(["--warmups", "1", "--repeats", "2"]) == 0
    assert len(calls) == 3
    assert capsys.readouterr().out.splitlines() == [
        "REQUEST=1 TTFT_MS=123.000 TPOT_MS=null TOKENS=1",
        "REQUEST=2 TTFT_MS=123.000 TPOT_MS=null TOKENS=1",
    ]


@pytest.mark.parametrize("error", [ValueError("missing usage"), TimeoutError("timeout"), IncompleteRead(b"partial", 2)])
def test_cli_failure_prints_error_without_metrics(monkeypatch, capsys, error):
    def request(args):
        raise error

    monkeypatch.setattr(bench, "request_completion", request)
    assert bench.main(["--warmups", "0"]) == 1
    output = capsys.readouterr()
    assert not output.out and str(error) in output.err


@pytest.mark.parametrize(
    "flags", [["--warmups", "-1"], ["--repeats", "0"], ["--timeout", "nan"], ["--max-tokens", "0"]]
)
def test_invalid_cli_arguments_fail(flags):
    with pytest.raises(SystemExit):
        bench.parse_args(flags)
