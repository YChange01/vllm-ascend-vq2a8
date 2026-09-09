#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loopback-only HTTP demo over the bounded, single-thread offline engine.

Not vllm serve, not a production/concurrency/quality certification. The model
is initialized and called on the same thread. No HTTP thread pool touches NPU.
Only non-streaming, greedy /v1/completions with a single text prompt is exposed.
"""

from __future__ import annotations

# ruff: noqa: E402
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import hashlib
import json
import platform
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL_NAME = "vq2a8"
MAX_CONTEXT = 128
MAX_BODY_BYTES = 64 * 1024
PRESETS = ("fast", "batched", "pipeline")
SCOPE = "loopback_serial_http_demo_context_le_128_not_production"
REQUIRED_SOURCES = (
    "vllm_ascend/quantization/vq2a8_optimization.py",
    "vllm_ascend/quantization/vq2a8_activation.py",
    "vllm_ascend/quantization/vq2a8_activation_fast.py",
    "vllm_ascend/quantization/vq2a8_activation_triton.py",
    "vllm_ascend/quantization/vq2a8_execution.py",
    "vllm_ascend/quantization/vq2a8_moe.py",
    "vllm_ascend/quantization/vq2a8_ascendc.py",
    "vllm_ascend/patch/worker/vq2a8_offline_model.py",
    "tools/vq2a8_optimization_report.py",
    "tools/benchmark_vq2a8_offline.py",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_acceptance(report_path, model, library, preset, *, repo=REPO):
    """Use explicit local evidence, never silently pick a newer/failed run."""
    path = report_path.resolve(strict=True)
    if path.is_dir():
        path = path / "result/summary.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("optimization_sources_unchanged") is not True:
        raise ValueError("Require a completed PASS optimization report with unchanged sources")
    candidates = [entry for entry in report.get("candidates", []) if entry.get("preset") == preset]
    if not candidates or any(
        entry.get("status") != "PASS"
        or entry.get("numerical", {}).get("accepted") is not True
        or entry.get("repeat", {}).get("accepted") is not True
        for entry in candidates
    ):
        raise ValueError(f"No complete exact/repeat PASS for preset {preset}")
    expected = report.get("library", {}).get("sha256")
    if not library.is_file() or library.suffix != ".so" or digest(library) != expected:
        raise ValueError("Library differs from the accepted optimization run")
    source_hashes = report.get("optimization_source_sha256", {})
    for name in REQUIRED_SOURCES:
        if source_hashes.get(name) != digest(repo / name):
            raise ValueError(f"Accepted source differs: {name}; rerun optimization acceptance")
    recorded_model = report.get("engine_options", {}).get("model")
    if not isinstance(recorded_model, str) or Path(recorded_model).resolve() != model.resolve():
        raise ValueError("Model path differs from the accepted run")
    return {
        "path": str(path),
        "sha256": digest(path),
        "library_sha256": expected,
        "accepted_cases": [entry["case"] for entry in candidates],
        "preflight": path.parent.parent / "preflight/preflight.json",
    }


def validate_request(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object")
    unknown = set(payload) - {"model", "prompt", "max_tokens", "temperature", "stream", "n", "ignore_eos"}
    if unknown:
        raise ValueError(f"Unsupported fields: {', '.join(sorted(unknown))}")
    if payload.get("model", MODEL_NAME) != MODEL_NAME:
        raise ValueError(f"model must be {MODEL_NAME}")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be one nonempty string; batches/chat messages are unsupported")
    count = payload.get("max_tokens", 32)
    if type(count) is not int or not 1 <= count < MAX_CONTEXT:
        raise ValueError("max_tokens must be an integer in [1,127]")
    temperature = payload.get("temperature", 0)
    if type(temperature) not in (int, float) or temperature != 0:
        raise ValueError("Only temperature=0 (greedy) is supported")
    if type(payload.get("n", 1)) is not int or payload.get("n", 1) != 1:
        raise ValueError("Only n=1 is supported")
    if payload.get("stream", False) is not False:
        raise ValueError("This demo only supports stream=false; wait for the complete response")
    if type(payload.get("ignore_eos", False)) is not bool:
        raise ValueError("ignore_eos must be a boolean")
    return prompt, count, payload.get("ignore_eos", False)


def encode_prompt(tokenizer, text, bos, max_tokens):
    tokens = tokenizer.encode(text, add_special_tokens=False).ids
    if type(bos) is int and (not tokens or tokens[0] != bos):
        tokens.insert(0, bos)
    if not tokens or len(tokens) + max_tokens > MAX_CONTEXT:
        raise ValueError(
            f"Context limit is {MAX_CONTEXT}: prompt={len(tokens)}, max_tokens={max_tokens}; shorten the request"
        )
    return tokens


class DemoRuntime:
    def __init__(self, llm, tokenizer, config, preset, acceptance):
        self.llm, self.tokenizer, self.config = llm, tokenizer, config
        self.preset, self.acceptance = preset, acceptance
        self.owner_thread = threading.get_ident()
        self.healthy = True
        eos = config.get("eos_token_id")
        self.eos = [eos] if type(eos) is int else eos if isinstance(eos, list) else []
        if any(type(token) is not int or not 0 <= token < config["vocab_size"] for token in self.eos):
            raise ValueError("Invalid eos_token_id in model config")

    def health(self):
        return {
            "status": "ready" if self.healthy else "failed_restart_required",
            "model": MODEL_NAME,
            "preset": self.preset,
            "scope": SCOPE,
            "max_context_tokens": MAX_CONTEXT,
            "streaming": False,
            "accepted_offline_cases": self.acceptance["accepted_cases"],
            "library_sha256": self.acceptance["library_sha256"],
            "production_serving_verified": False,
            "quality_verified": False,
        }

    def complete(self, payload):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("NPU engine must run on its initialization thread")
        text, count, ignore_eos = validate_request(payload)
        prompt = encode_prompt(self.tokenizer, text, self.config.get("bos_token_id"), count)
        if not self.healthy:
            raise RuntimeError("Engine is unhealthy; restart the demo")
        from vllm import SamplingParams

        from tools.benchmark_vq2a8_offline import configure, snapshot

        started = time.perf_counter()
        try:
            # Reset deferred finite flags and bounded diagnostic counters BETWEEN
            # requests. Preserve the lazy packed expert cache, but no KV prefix cache.
            configure(self.llm, measurement=True, compact=False, optimization=self.preset)
            result = self.llm.generate(
                [{"prompt_token_ids": prompt}],
                SamplingParams(
                    temperature=0,
                    max_tokens=count,
                    ignore_eos=ignore_eos,
                    stop_token_ids=[] if ignore_eos else self.eos,
                    detokenize=False,
                ),
                use_tqdm=False,
            )
            if len(result) != 1 or not result[0].finished or len(result[0].outputs) != 1:
                raise RuntimeError("Expected exactly one finished completion")
            output = result[0].outputs[0]
            tokens = list(output.token_ids)
            state = snapshot(self.llm)  # Materialize finite flags BEFORE returning text.
            if (
                not 1 <= len(tokens) <= count
                or any(type(token) is not int or not 0 <= token < self.config["vocab_size"] for token in tokens)
                or (ignore_eos and len(tokens) != count)
                or state["finite"] is not True
                or self.llm.llm_engine.has_unfinished_requests()
            ):
                raise RuntimeError("Incomplete/nonfinite result or unexpected pending request")
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception:
            # Do not continue using a possibly broken stream/cache after an NPU error.
            self.healthy = False
            raise
        elapsed = time.perf_counter() - started
        return {
            "id": "cmpl-" + uuid.uuid4().hex,
            "object": "text_completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [{"index": 0, "text": decoded, "finish_reason": output.finish_reason, "logprobs": None}],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(tokens),
                "total_tokens": len(prompt) + len(tokens),
            },
            "vq2a8": {
                "preset": self.preset,
                "generation_s": elapsed,
                "timing_scope": "configure_generate_finite_check_decode_not_http_ttft",
                "generated_token_ids": tokens,
                "finite": True,
                "scope": SCOPE,
            },
        }


class DemoServer(HTTPServer):
    """Deliberately not ThreadingHTTPServer: one owner thread for all NPU work."""

    allow_reuse_address = True
    request_queue_size = 1
    runtime = None


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "VQ2A8Demo/1"
    sys_version = ""
    timeout = 30

    def log_message(self, _format, *args):
        # Never print request bodies/prompts, headers, or untrusted URL text.
        pass

    def reply(self, code, payload):
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.close_connection = True
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass  # Generation is already finished; a disconnected client cannot corrupt it.

    def error(self, code, message):
        self.reply(code, {"error": {"message": message, "type": "vq2a8_demo_error"}})

    def do_GET(self):
        runtime = self.server.runtime
        if self.path == "/health":
            self.reply(200 if runtime.healthy else 503, runtime.health())
        elif self.path == "/v1/models":
            self.reply(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}]})
        else:
            self.error(404, "Supported routes: GET /health, GET /v1/models, POST /v1/completions")

    def read_payload(self):
        if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Encoding"):
            raise ValueError("Chunked/compressed request bodies are unsupported")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdecimal():
            raise ValueError("Require one integer Content-Length")
        length = int(lengths[0])
        if not 0 < length <= MAX_BODY_BYTES:
            raise ValueError(f"JSON request body must be 1..{MAX_BODY_BYTES} bytes")
        if self.headers.get_content_type() != "application/json":
            raise ValueError("Content-Type must be application/json")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ValueError("Incomplete request body")
        payload = json.loads(raw.decode("utf-8"))
        validate_request(payload)
        return payload

    def do_POST(self):
        if self.path != "/v1/completions":
            self.error(404, "Use POST /v1/completions with prompt; chat and streaming are not implemented")
            return
        if not self.server.runtime.healthy:
            self.error(503, "Engine failed; restart the demo and inspect its terminal traceback")
            return
        try:
            payload = self.read_payload()
        except (ValueError, TimeoutError, RecursionError) as exc:
            self.error(400, str(exc))
            return
        try:
            result = self.server.runtime.complete(payload)
        except ValueError as exc:
            if self.server.runtime.healthy:
                self.error(400, str(exc))  # Tokenized context overflow, before engine access.
                return
            traceback.print_exc()
            self.error(500, "Generation failed; inspect the server terminal and restart")
            return
        except Exception:
            self.server.runtime.healthy = False
            traceback.print_exc()
            self.error(500, "Generation failed; inspect the server terminal and restart")
            return
        self.reply(200, result)
        print(
            "HTTP_COMPLETION "
            + json.dumps({"usage": result["usage"], "generation_s": result["vq2a8"]["generation_s"]}),
            flush=True,
        )


def initialize_runtime(args, acceptance):
    # Same isolation as the tested single-process benchmark, set before imports.
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.physical_npu)
    os.environ["ASCEND_LAUNCH_BLOCKING"] = "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from tools.validate_vq2a8_v026_environment import check_scheduler_apis, require_v026_stack

    require_v026_stack()
    import torch
    import torch_npu  # noqa: F401
    from tokenizers import Tokenizer
    from vllm import LLM

    from tools.validate_vq2a8_ascendc import checked_model_preflight, require_hardware_runtime
    from tools.validate_vq2a8_qli_metadata import run_preflight
    from tools.validate_vq2a8_sas_attention import run_sas_preflight
    from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device
    from vllm_ascend.quantization.vq2a8_offline import offline_engine_options

    check_scheduler_apis()
    library = checked_model_preflight(args.library, acceptance["preflight"])
    device = _initialize_device(torch.device("npu:0"))
    if device["name"] != library["build"]["soc"]:
        raise ValueError("Actual device differs from the library build SOC; rebuild and repeat acceptance")
    require_hardware_runtime()
    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    if config.get("num_hidden_layers") != 43:
        raise ValueError("This demo is scoped to the existing 43-layer VQ2A8 checkpoint")
    run_preflight(torch.device("npu:0"), config, prompt_tokens=10)
    run_sas_preflight(torch.device("npu:0"), config, prompt_tokens=10)
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    options = offline_engine_options(
        args.model,
        args.model / "experts_vq_ascend_v2",
        execution_policy="ascendc",
        ascendc_library=library["path"],
        ascendc_sha256=library["sha256"],
    )
    options.update(max_model_len=MAX_CONTEXT, max_num_batched_tokens=MAX_CONTEXT)
    runtime = DemoRuntime(LLM(**options), tokenizer, config, args.preset, acceptance)
    # One real prefill/decode/finite check before advertising readiness. This is
    # a runtime smoke test, not a replacement for independent model quality tests.
    runtime.complete({"prompt": "The answer to 1 + 1 is", "max_tokens": 4, "ignore_eos": True})
    return runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=REPO / "build/vq2a8-ascendc-opt3/libvq2a8_ascendc.so")
    parser.add_argument(
        "--acceptance-report", type=Path, required=True, help="PASS opt3 directory or result/summary.json"
    )
    parser.add_argument("--physical-npu", type=int, default=0)
    parser.add_argument("--preset", choices=PRESETS, default="batched")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run on the Linux Ascend950 server; Windows supports only --help and CPU contract tests")
    if args.physical_npu < 0 or not 1 <= args.port <= 65535:
        parser.error("Require nonnegative physical NPU and port in [1,65535]")
    args.model, args.library = args.model.resolve(strict=True), args.library.resolve(strict=True)
    try:
        acceptance = checked_acceptance(args.acceptance_report, args.model, args.library, args.preset)
        # Bind first to reject an occupied port BEFORE loading a second model.
        # Accept requests only after initialization. Never expose 0.0.0.0.
        with DemoServer(("127.0.0.1", args.port), DemoHandler) as server:
            started = time.perf_counter()
            print(f"HTTP_DEMO_LOADING preset={args.preset} acceptance={acceptance['path']}", flush=True)
            server.runtime = initialize_runtime(args, acceptance)
            print(
                f"HTTP_DEMO_READY url=http://127.0.0.1:{args.port} startup_s={time.perf_counter() - started:.3f} "
                f"preset={args.preset} scope={SCOPE}",
                flush=True,
            )
            server.serve_forever()
    except KeyboardInterrupt:
        print("HTTP_DEMO_STOPPED", flush=True)
    except Exception:
        traceback.print_exc()
        print("HTTP_DEMO_FAILED", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
