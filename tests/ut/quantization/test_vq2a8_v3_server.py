# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quick server CLI contracts, without vLLM startup or NPU execution."""

import hashlib
import json
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from tools import serve_vq2a8_v3 as server
from vllm_ascend.quantization.vq2a8_offline import validate_offline_config


@pytest.fixture
def argv(tmp_path):
    model = tmp_path / "model with spaces"
    (model / "experts_vq_ascend_v2").mkdir(parents=True)
    library = tmp_path / "candidate.so"
    library.write_bytes(b"CPU CLI fixture, not an executable library")
    # Deliberately no manifest, receipt or acceptance directory.
    return ["--model", str(model), "--library", str(library)]


def argument(command, flag):
    return command[command.index(flag) + 1]


@pytest.mark.parametrize("preparation,graph", [("eager", "none"), ("fused", "none"), ("fused", "moe")])
def test_standard_vllm_server_command_and_offline_contract(argv, preparation, graph):
    args = server.parse_args([*argv, "--preparation", preparation, "--decode-graph", graph])
    command = server.build_command(args)
    assert command[:4] == [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve"]
    assert command[4] == str(args.model.resolve())
    assert argument(command, "--served-model-name") == "vq2a8"
    assert "--skip-tokenizer-init" not in command  # Standard text completions need a tokenizer.
    assert {"--no-async-scheduling", "--no-enable-prefix-caching", "--no-enable-chunked-prefill"} <= set(command)
    assert argument(command, "--stream-interval") == "1"
    additional = json.loads(argument(command, "--additional-config"))
    options = additional["vq2a8_offline"]
    assert options["v3_serving"] is True
    assert options["v3_preparation"] == preparation and options["v3_decode_graph"] == graph
    assert options["ascendc_v3_sha256"] == hashlib.sha256(args.library.read_bytes()).hexdigest()
    assert json.loads(argument(command, "--hf-overrides")) == {
        "architectures": ["VQ2A8TP1OfflineForCausalLM"],
        "quantization_config": None,
    }
    # Validate generated settings against the real adapter's acceptance rules.
    cfg = NS(
        additional_config=additional,
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        model_config=NS(
            enforce_eager=True,
            quantization=None,
            dtype=torch.bfloat16,
            max_model_len=int(argument(command, "--max-model-len")),
        ),
        quant_config=None,
        scheduler_config=NS(
            max_num_seqs=int(argument(command, "--max-num-seqs")),
            max_num_batched_tokens=int(argument(command, "--max-num-batched-tokens")),
            async_scheduling=False,
        ),
        compilation_config=NS(mode=0, cudagraph_mode=0),
        cache_config=NS(kv_cache_memory_bytes=int(argument(command, "--kv-cache-memory-bytes"))),
        load_config=NS(load_format=argument(command, "--load-format")),
    )
    assert validate_offline_config(cfg) == options


@pytest.mark.parametrize(
    "option,value",
    [
        ("--physical-npu", "-1"),
        ("--port", "0"),
        ("--port", "65536"),
        ("--reserve-gib", "0.5"),
        ("--reserve-gib", "nan"),
        ("--memory-fraction", "0"),
        ("--engine-memory-fraction", "1.1"),
        ("--memory-fraction", "nan"),
        ("--decode-graph", "full"),
    ],
)
def test_invalid_server_settings_fail_before_loading(argv, option, value):
    with pytest.raises(SystemExit):
        server.parse_args([*argv, option, value])


def test_server_uses_standard_engine_process_and_clears_stale_device_settings(argv, monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")
    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", "4,5")
    monkeypatch.setenv("WORLD_SIZE", "8")
    environment = server.server_environment(server.parse_args([*argv, "--physical-npu", "2"]))
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "2"
    assert environment["VLLM_ENABLE_V1_MULTIPROCESSING"] == "1"
    assert environment["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert environment["ASCEND_LAUNCH_BLOCKING"] == "0"
    assert "WORLD_SIZE" not in environment and "ASCEND_VISIBLE_DEVICES" not in environment
    assert environment["PYTHONPATH"].split(":")[0] == str(server.REPO)


def test_dry_run_does_not_start_server_or_require_acceptance(argv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["serve_vq2a8_v3.py", *argv, "--dry-run"])
    monkeypatch.setattr(server.os, "execvpe", lambda *args: pytest.fail("dry run executed vLLM"))
    assert server.main() == 0
    assert "vllm.entrypoints.cli.main serve" in capsys.readouterr().out


def test_launch_replaces_process_with_native_server(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["serve_vq2a8_v3.py", *argv])
    calls = []
    monkeypatch.setattr(server.os, "execvpe", lambda *args: calls.append(args))
    assert server.main() == 0
    assert len(calls) == 1
    executable, command, environment = calls[0]
    assert executable == command[0] == sys.executable
    assert command[3] == "serve" and environment["ASCEND_LAUNCH_BLOCKING"] == "0"
