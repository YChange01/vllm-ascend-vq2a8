# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4 server command/environment contracts; no vLLM or NPU execution."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools import serve_vq2a8_v4 as server


def assets(tmp_path):
    model = tmp_path / "model with spaces"
    model.mkdir()
    (model / "experts_vq_ascend_v2").mkdir()
    library = tmp_path / "libvq2a8_ascendc.so"
    library.write_bytes(b"CPU command fixture, not a valid native library")
    return ["--model", str(model), "--library", str(library)], model, library


def value(command, name):
    return command[command.index(name) + 1]


def test_v4_server_defaults_are_single_card_small_eager_and_v1_library():
    args = server.parse_args([])
    assert args.model == Path("/home/g00872988/vq2a8")
    assert args.physical_npu == 1 and args.host == "127.0.0.1" and args.port == 8000
    assert args.memory_fraction == args.engine_memory_fraction == 0.9
    assert args.reserve_gib == 8.0
    assert args.max_model_len == 128 and args.kv_cache_mib == 1024
    assert args.library.parent.name == "vq2a8-ascendc-v023-v1"
    assert args.library.name == "libvq2a8_ascendc.so"
    assert args.device_route_decode is False


def test_v4_server_uses_standard_cli_and_exact_tp1_contract_without_preflights(tmp_path):
    argv, model, library = assets(tmp_path)
    command = server.build_command(server.parse_args(argv))
    assert command[:5] == [server.sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model.resolve())]
    expected = {
        "--served-model-name": "vq2a8",
        "--dtype": "bfloat16",
        "--load-format": "safetensors",
        "--tensor-parallel-size": "1",
        "--pipeline-parallel-size": "1",
        "--distributed-executor-backend": "uni",
        "--max-num-seqs": "1",
        "--max-model-len": "128",
        "--max-num-batched-tokens": "128",
        "--block-size": "128",
        "--kv-cache-memory-bytes": str(1024**3),
        "--gpu-memory-utilization": "0.9",
        "--stream-interval": "1",
        "--generation-config": "vllm",
    }
    assert all(value(command, key) == expected_value for key, expected_value in expected.items())
    assert all(
        flag in command
        for flag in (
            "--enforce-eager",
            "--no-async-scheduling",
            "--no-enable-prefix-caching",
            "--no-enable-chunked-prefill",
        )
    )
    assert json.loads(value(command, "--compilation-config")) == {"mode": 0, "cudagraph_mode": "NONE"}
    assert json.loads(value(command, "--hf-overrides")) == {
        "architectures": ["VQ2A8TP1OfflineForCausalLM"],
        "quantization_config": None,
    }
    additional = json.loads(value(command, "--additional-config"))
    options = additional.pop("vq2a8_offline")
    assert all(item is False for item in additional.values())
    assert options == {
        "enabled": True,
        "artifact": str((model / "experts_vq_ascend_v2").resolve()),
        "execution_policy": "ascendc_v4",
        "ascendc_library": str(library.resolve()),
        "ascendc_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "cache_experts": 256,
        "token_chunk": 2,
        "cache_memory_fraction": 0.9,
        "cache_reserve_gib": 8.0,
        "root_linear_mode": "bf16",
        "v4_serving": True,
    }
    assert not any("v3" in item or "accept_vq2a8" in item or "preflight" in item for item in command)


def test_device_route_is_explicit_v4_only_and_preserves_prefill_and_server_contract(tmp_path):
    argv, _, _ = assets(tmp_path)
    baseline = server.build_command(server.parse_args(argv))
    candidate = server.build_command(server.parse_args([*argv, "--device-route-decode"]))
    original = json.loads(value(baseline, "--additional-config"))
    changed = json.loads(value(candidate, "--additional-config"))
    assert "v4_device_route_decode" not in original["vq2a8_offline"]
    assert changed["vq2a8_offline"].pop("v4_device_route_decode") is True
    assert changed == original
    index = candidate.index("--additional-config") + 1
    assert candidate[:index] == baseline[:index] and candidate[index + 1 :] == baseline[index + 1 :]


@pytest.mark.parametrize(
    "argv",
    [
        ["--physical-npu", "-1"],
        ["--port", "0"],
        ["--port", "65536"],
        ["--host", ""],
        ["--host", "two hosts"],
        ["--memory-fraction", "nan"],
        ["--memory-fraction", "0"],
        ["--memory-fraction", "1.1"],
        ["--engine-memory-fraction", "inf"],
        ["--engine-memory-fraction", "-1"],
        ["--reserve-gib", "nan"],
        ["--reserve-gib", "0.9"],
        ["--max-model-len", "0"],
        ["--max-model-len", "129"],
        ["--max-model-len", "16.5"],
        ["--kv-cache-mib", "0"],
        ["--kv-cache-mib", "-1"],
        ["--kv-cache-mib", "256.5"],
        ["--kv-cache-mib", "nan"],
        ["--kv-cache-mib", "2048", "--reserve-gib", "1.99"],
        ["--kv-cache-mib", "256", "--reserve-gib", "0.99"],
        ["--tensor-parallel-size", "2"],
        ["--physical-npus", "0,1"],
        ["--artifact", "/other/artifact"],
        ["--decode-graph", "moe"],
        ["--preparation", "fused"],
    ],
)
def test_v4_server_rejects_unsupported_or_invalid_parameters(argv):
    with pytest.raises(SystemExit) as exc:
        server.parse_args(argv)
    assert exc.value.code == 2


def test_v4_server_explicit_budget_and_network_options_are_forwarded(tmp_path):
    argv, _, _ = assets(tmp_path)
    args = server.parse_args(
        [
            *argv,
            "--physical-npu",
            "2",
            "--host",
            "0.0.0.0",
            "--port",
            "8123",
            "--memory-fraction",
            "1",
            "--engine-memory-fraction",
            "0.95",
            "--reserve-gib",
            "9",
        ]
    )
    command = server.build_command(args)
    options = json.loads(value(command, "--additional-config"))["vq2a8_offline"]
    assert value(command, "--host") == "0.0.0.0" and value(command, "--port") == "8123"
    assert value(command, "--gpu-memory-utilization") == "0.95"
    assert options["cache_memory_fraction"] == 1 and options["cache_reserve_gib"] == 9


def test_v4_short_text_context_and_kv_budget_do_not_change_execution_contract(tmp_path):
    argv, _, _ = assets(tmp_path)
    args = server.parse_args(
        [
            *argv,
            "--physical-npu",
            "2",
            "--max-model-len",
            "16",
            "--kv-cache-mib",
            "256",
            "--memory-fraction",
            "1",
            "--engine-memory-fraction",
            "0.9",
            "--reserve-gib",
            "3",
        ]
    )
    command = server.build_command(args)
    assert value(command, "--max-model-len") == value(command, "--max-num-batched-tokens") == "16"
    assert value(command, "--kv-cache-memory-bytes") == str(256 * 1024**2)
    assert value(command, "--gpu-memory-utilization") == "0.9"
    assert value(command, "--block-size") == "128"
    assert value(command, "--tensor-parallel-size") == value(command, "--max-num-seqs") == "1"
    assert value(command, "--dtype") == "bfloat16"
    assert all(
        flag in command
        for flag in (
            "--enforce-eager",
            "--no-async-scheduling",
            "--no-enable-prefix-caching",
            "--no-enable-chunked-prefill",
        )
    )
    options = json.loads(value(command, "--additional-config"))["vq2a8_offline"]
    assert options["cache_memory_fraction"] == 1 and options["cache_reserve_gib"] == 3
    assert options["v4_serving"] is True and options["execution_policy"] == "ascendc_v4"
    assert server.server_environment(args, {})["ASCEND_RT_VISIBLE_DEVICES"] == "2"


@pytest.mark.parametrize("length", [1, 128])
@pytest.mark.parametrize("kv_mib,reserve", [(1, 1), (256, 1), (1024, 1), (2048, 2)])
def test_v4_server_context_and_reserve_boundaries(length, kv_mib, reserve):
    args = server.parse_args(
        [
            "--max-model-len",
            str(length),
            "--kv-cache-mib",
            str(kv_mib),
            "--reserve-gib",
            str(reserve),
        ]
    )
    assert args.max_model_len == length and args.kv_cache_mib == kv_mib and args.reserve_gib == reserve


def test_v4_server_help_explains_new_context_and_kv_options(capsys):
    with pytest.raises(SystemExit) as exc:
        server.parse_args(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--max-model-len" in help_text and "--kv-cache-mib" in help_text
    assert "Total input plus output" in help_text and "MiB" in help_text


def test_v4_server_environment_is_separate_single_device_standard_engine():
    original = {
        "ASCEND_RT_VISIBLE_DEVICES": "4,5,6,7",
        "ASCEND_VISIBLE_DEVICES": "4,5,6,7",
        "NPU_VISIBLE_DEVICES": "4",
        "ASCEND_DEVICE_ID": "4",
        "DEVICE_ID": "4",
        "RANK_ID": "4",
        "LOCAL_RANK": "4",
        "RANK": "4",
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "8",
        "MASTER_ADDR": "other",
        "MASTER_PORT": "1234",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "ASCEND_LAUNCH_BLOCKING": "1",
        "PYTHONPATH": "prior-path",
        "KEEP_ME": "yes",
    }
    environment = server.server_environment(server.parse_args([]), original)
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "1"
    assert environment["ASCEND_LAUNCH_BLOCKING"] == "0"
    assert environment["VLLM_ENABLE_V1_MULTIPROCESSING"] == "1"
    assert environment["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert environment["PYTHONPATH"] == str(server.REPO) + server.os.pathsep + "prior-path"
    assert environment["KEEP_ME"] == "yes" and original["WORLD_SIZE"] == "8"
    assert not any(
        key in environment
        for key in (
            "ASCEND_VISIBLE_DEVICES",
            "NPU_VISIBLE_DEVICES",
            "RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "LOCAL_WORLD_SIZE",
            "DEVICE_ID",
        )
    )


def test_v4_server_dry_run_only_hashes_local_library_and_prints_command(monkeypatch, tmp_path, capsys):
    argv, _, _ = assets(tmp_path)
    monkeypatch.setattr(server.os, "execvpe", lambda *args: pytest.fail("dry run must not execute"))
    assert server.main([*argv, "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "V4_SERVER_DRY_RUN physical_npu=1 no_device_execution=True" in output
    assert "vllm.entrypoints.cli.main serve" in output and "ascendc_v4" in output


@pytest.mark.parametrize("argv,code", [(["--help"], 0), (["--unknown"], 2)])
def test_v4_server_help_and_unknown_exit_before_files_or_exec(monkeypatch, argv, code):
    monkeypatch.setattr(server, "build_command", lambda *args: pytest.fail("no file inspection"))
    monkeypatch.setattr(server.os, "execvpe", lambda *args: pytest.fail("no execution"))
    with pytest.raises(SystemExit) as exc:
        server.main(argv)
    assert exc.value.code == code


def test_v4_server_exec_uses_argv_not_shell_and_preserves_standard_server(monkeypatch, tmp_path):
    argv, model, _ = assets(tmp_path)
    calls = []
    monkeypatch.setattr(server.os, "execvpe", lambda *args: calls.append(args))
    assert server.main(argv) == 0
    executable, command, environment = calls[0]
    assert executable == server.sys.executable and str(model.resolve()) in command
    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "1"
    assert command[3] == "serve" and len(calls) == 1


@pytest.mark.parametrize("missing", ["model", "artifact", "library", "suffix"])
def test_v4_server_path_errors_do_not_exec(monkeypatch, tmp_path, missing):
    argv, model, library = assets(tmp_path)
    if missing == "model":
        argv[1] = str(tmp_path / "missing-model")
    elif missing == "artifact":
        (model / "experts_vq_ascend_v2").rmdir()
    elif missing == "library":
        argv[3] = str(tmp_path / "missing.so")
    else:
        invalid = tmp_path / "invalid.txt"
        invalid.write_bytes(library.read_bytes())
        argv[3] = str(invalid)
    monkeypatch.setattr(server.os, "execvpe", lambda *args: pytest.fail("invalid input must not execute"))
    assert server.main(argv) == 1


def test_v4_server_imports_only_standard_library_and_contains_no_automatic_gate():
    source = Path(server.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.module.split(".")[0] if isinstance(node, ast.ImportFrom) else name.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for name in (node.names if isinstance(node, ast.Import) else [None])
    }
    assert not {"torch", "torch_npu", "vllm", "vllm_ascend", "tools"} & imports
    assert all(
        name not in source
        for name in (
            "require_v023_stack",
            "check_runtime_environment",
            "check_python_environment",
            "load_model_preflight",
            "build_manifest",
        )
    )
