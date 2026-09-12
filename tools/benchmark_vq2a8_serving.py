#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure client-visible TTFT/TPOT from an OpenAI completions SSE stream.

TPOT uses usage.completion_tokens, never the number of SSE chunks. Transport
buffering can combine tokens, so these are HTTP delivery timings.
"""

# ruff: noqa: E402
import os
import sys

if not __package__:
    # tools/bisect would otherwise shadow Python's standard-library bisect.
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import math
import time
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def sse_data(lines):
    """Decode data events, including multiline JSON and keepalive comments."""
    data = []
    for raw in lines:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        raise ValueError("Truncated SSE event")


def measure_stream(lines, started, *, clock=time.perf_counter):
    first_content = last_content = tokens = None
    completed = False
    for data in sse_data(lines):
        if data.strip() == "[DONE]":
            completed = True
            break
        event = json.loads(data)
        if not isinstance(event, dict):
            raise ValueError("Invalid SSE completion payload")
        if event.get("error") is not None:
            raise ValueError(f"Server stream error: {event['error']}")
        choices = event.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError("Expected a single streamed completion")
        if choices:
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                raise ValueError("Unexpected streamed choice")
            content = choice.get("text", "")
            if not isinstance(content, str):
                raise ValueError("Completion text must be a string")
            if content:
                last_content = clock()
                if first_content is None:
                    first_content = last_content
        usage = event.get("usage")
        if usage is not None:
            if not isinstance(usage, dict) or type(usage.get("completion_tokens")) is not int:
                raise ValueError("Missing integer usage.completion_tokens")
            tokens = usage["completion_tokens"]
            if tokens < 0:
                raise ValueError("Negative usage.completion_tokens")
    if not completed:
        raise ValueError("Truncated stream: missing [DONE]")
    if first_content is None:
        raise ValueError("Stream contained no generated text")
    if tokens is None or tokens < 1:
        raise ValueError("Missing positive usage.completion_tokens; token count cannot be inferred from chunks")
    return {
        "ttft_ms": (first_content - started) * 1000,
        "tpot_ms": (last_content - first_content) * 1000 / (tokens - 1) if tokens >= 2 else None,
        "completion_tokens": tokens,
    }


def request_completion(args, *, opener=urlopen, clock=time.perf_counter):
    base = args.base_url.rstrip("/")
    endpoint = base + ("/completions" if base.endswith("/v1") else "/v1/completions")
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = clock()
    try:
        with opener(request, timeout=args.timeout) as response:
            if response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "text/event-stream":
                raise ValueError("Expected text/event-stream response from /v1/completions")
            return measure_stream(response, started, clock=clock)
    except HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        error.close()
        raise ValueError(f"HTTP {error.code}: {detail}") from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="vq2a8")
    parser.add_argument("--prompt", default="Explain why the sky is blue in simple terms.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP socket timeout in seconds")
    args = parser.parse_args(argv)
    url = urlsplit(args.base_url)
    if url.scheme not in ("http", "https") or not url.netloc:
        parser.error("--base-url must be an HTTP(S) server URL")
    if not args.prompt or not args.model:
        parser.error("--prompt and --model cannot be empty")
    if args.max_tokens < 1 or args.warmups < 0 or args.repeats < 1:
        parser.error("Require max-tokens >=1, warmups >=0 and repeats >=1")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        for _ in range(args.warmups):
            request_completion(args)
        for index in range(args.repeats):
            result = request_completion(args)
            tpot = "null" if result["tpot_ms"] is None else f"{result['tpot_ms']:.3f}"
            print(
                f"REQUEST={index + 1} TTFT_MS={result['ttft_ms']:.3f} "
                f"TPOT_MS={tpot} TOKENS={result['completion_tokens']}",
                flush=True,
            )
    except (HTTPException, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
