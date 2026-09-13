# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU command/math contracts; these tests do not certify NPU/HCCL execution."""

import json
import subprocess
import sys
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import validate_vq2a8_tp2_collective as tool


def launch_env(rank=0):
    return {
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "ASCEND_RT_VISIBLE_DEVICES": "3,5",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29501",
    }


def test_tp2_collective_cli_default_native_requires_explicit_pin():
    with pytest.raises(SystemExit):
        tool.parse_args([])
    args = tool.parse_args(["--library-sha256", "a" * 64])
    assert args.timeout_s == 120 and not args.communication_only
    assert args.library.as_posix().endswith("build/vq2a8-ascendc-v3/libvq2a8_ascendc_v3.so")
    command = tool.build_command(args)
    assert command[:3] == ["torchrun", "--standalone", "--nproc-per-node=2"]
    assert command[command.index("--library-sha256") + 1] == "a" * 64
    assert "--model" not in command and "--plan-only" not in command


def test_tp2_collective_communication_command_does_not_reference_library(tmp_path):
    args = tool.parse_args(["--communication-only", "--timeout-s", "31", "--report-dir", str(tmp_path)])
    command = tool.build_command(args)
    assert "--communication-only" in command and "--library" not in command and "--library-sha256" not in command
    assert command[command.index("--timeout-s") + 1] == "31"
    assert command[command.index("--report-dir") + 1] == str(tmp_path)


@pytest.mark.parametrize(
    "flags",
    [
        ("--timeout-s", "0"),
        ("--timeout-s", "-1"),
        ("--timeout-s", "nan"),
        ("--timeout-s", "1.5"),
        ("--library-sha256", "A" * 64),
        ("--library-sha256", "a" * 63),
    ],
)
def test_tp2_collective_rejects_invalid_cli(flags):
    with pytest.raises(SystemExit):
        tool.parse_args(["--communication-only", *flags])


@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_collective_launch_maps_physical_and_logical_devices(rank):
    launch = tool.launch_environment(launch_env(rank))
    assert launch == {
        "rank": rank,
        "local_rank": rank,
        "world_size": 2,
        "visible_devices": "3,5",
        "physical_npu": (3, 5)[rank],
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("WORLD_SIZE", "1"),
        ("WORLD_SIZE", "4"),
        ("LOCAL_WORLD_SIZE", "1"),
        ("RANK", "2"),
        ("LOCAL_RANK", "1"),
        ("RANK", "00"),
        ("RANK", ""),
        ("MASTER_ADDR", ""),
        ("MASTER_PORT", ""),
        ("ASCEND_RT_VISIBLE_DEVICES", ""),
        ("ASCEND_RT_VISIBLE_DEVICES", "0"),
        ("ASCEND_RT_VISIBLE_DEVICES", "0,0"),
        ("ASCEND_RT_VISIBLE_DEVICES", "0, 1"),
        ("ASCEND_RT_VISIBLE_DEVICES", "0,1,2"),
    ],
)
def test_tp2_collective_rejects_invalid_torchrun_environment(key, value):
    env = launch_env()
    env[key] = value
    with pytest.raises(ValueError):
        tool.launch_environment(env)


def fake_distributed(events, *, fail=False):
    active = []
    group = object()

    def parallel_config(**kwargs):
        events.append(("parallel_config", kwargs))
        return SimpleNamespace(**kwargs)

    def vllm_config(**kwargs):
        assert kwargs["model_config"] is None
        events.append(("vllm_config", kwargs))
        return SimpleNamespace(**kwargs)

    @contextmanager
    def current(config):
        active.append(config)
        events.append(("enter", None))
        try:
            yield
        finally:
            active.pop()
            events.append(("exit", None))

    def initialize(**kwargs):
        assert active
        events.append(("init", kwargs))

    def model_parallel(**kwargs):
        assert active
        events.append(("model_parallel", kwargs))
        if fail:
            raise RuntimeError("injected init failure")

    return (
        SimpleNamespace(ParallelConfig=parallel_config, VllmConfig=vllm_config, set_current_vllm_config=current),
        SimpleNamespace(
            init_distributed_environment=initialize,
            initialize_model_parallel=model_parallel,
            get_tp_group=lambda: group,
            destroy_model_parallel=lambda: events.append(("destroy_model", None)),
            destroy_distributed_environment=lambda: events.append(("destroy_world", None)),
        ),
        group,
    )


@pytest.mark.parametrize("fail", [False, True])
def test_tp2_collective_uses_exact_v026_initialization_and_cleanup_contract(fail):
    events = []
    config, parallel, expected_group = fake_distributed(events, fail=fail)
    launch = tool.launch_environment(launch_env(1))
    if fail:
        with (
            pytest.raises(RuntimeError, match="injected init"),
            tool.tp_environment(launch, 23, config_module=config, parallel_state=parallel),
        ):
            pytest.fail("Initialization failed, so no TP group can be yielded")
    else:
        with tool.tp_environment(launch, 23, config_module=config, parallel_state=parallel) as group:
            assert group is expected_group
    values = dict(events)
    assert values["init"] == {
        "world_size": 2,
        "rank": 1,
        "local_rank": 1,
        "distributed_init_method": "env://",
        "backend": "hccl",
        "timeout": timedelta(seconds=23),
    }
    assert values["model_parallel"] == {
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "backend": "hccl",
    }
    assert values["parallel_config"]["distributed_timeout_seconds"] == 23
    assert values["parallel_config"]["cpu_distributed_timeout_seconds"] == 23
    assert values["parallel_config"]["distributed_executor_backend"] == "external_launcher"
    assert [event for event, _ in events][-3:] == ["destroy_model", "destroy_world", "exit"]


def test_tp2_collective_group_validation_rejects_cpu_backend_or_wrong_rank():
    launch = tool.launch_environment(launch_env(1))
    group = SimpleNamespace(
        world_size=2, rank=1, local_rank=1, rank_in_group=1, ranks=[0, 1], device=SimpleNamespace(type="npu", index=1)
    )
    tool.validate_group(group, launch, backend="hccl", current_device=1)
    with pytest.raises(RuntimeError, match="HCCL"):
        tool.validate_group(group, launch, backend="gloo", current_device=1)
    with pytest.raises(RuntimeError, match="device mapping"):
        tool.validate_group(group, launch, backend="hccl", current_device=0)
    group.rank_in_group = 0
    with pytest.raises(RuntimeError, match="rank mapping"):
        tool.validate_group(group, launch, backend="hccl", current_device=1)


@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_collective_routed_fp32_sum_is_once_and_shared_is_after_sum(rank):
    calls = []

    def reduce(partial):
        assert partial.dtype == torch.float32
        assert torch.equal(partial, torch.full((1, 8), 3.0 * (rank + 1)))
        calls.append(partial.clone())
        return partial + 3.0 * (2 - rank)

    record = tool.communication_smoke(torch, SimpleNamespace(all_reduce=reduce), "cpu", rank)
    assert len(calls) == record["routed_all_reduce_calls"] == 1
    assert record["routed_sum"] == 9 and record["shared_value"] == 5 and record["final_value"] == 14
    assert record["shared_added_after_reduce"] is True


@pytest.mark.parametrize("dtype,value", [(torch.bfloat16, 9), (torch.float32, 14), (torch.float32, float("nan"))])
def test_tp2_collective_rejects_wrong_reduction_or_nonfinite(dtype, value):
    group = SimpleNamespace(all_reduce=lambda partial: torch.full_like(partial, value, dtype=dtype))
    with pytest.raises(RuntimeError):
        tool.communication_smoke(torch, group, "cpu", 0)


@pytest.mark.parametrize("kind,rank", [(kind, rank) for kind in ("gate_up", "down") for rank in (0, 1)])
def test_tp2_collective_synthetic_oracle_matches_literal_zn_decode_and_padding(kind, rank):
    with torch.device("meta"):
        payload, expected = tool.synthetic_projection(torch, kind, rank)
    x, scale, bias, packed, lut = payload
    n, k = expected.shape[1], x.shape[1]
    assert all(value.device.type == "cpu" for value in (*payload, expected))
    assert (n, k) == ((2048, 4096) if kind == "gate_up" else (4096, 2048))
    assert x.dtype == torch.float8_e4m3fn and packed.dtype == lut.dtype == torch.uint8
    if kind == "down":
        assert not bool(x.float()[:, 1024:].any())
        assert bool(lut[4:].any())  # Zero inputs, not zeroed test weights, enforce the dummy-input check.
    columns = torch.arange(k)
    decoded_lut = lut.view(torch.float8_e4m3fn).double()
    for row in (0, 1, 2, 3, 31, 32, 65, n - 1):
        pair = (row % 32) // 2
        word = packed[row // 32, columns // 16, columns % 16, pair // 2]
        codes = (word.long() >> (4 * (pair % 2))) & 15
        weights = decoded_lut[columns // 256, row // 32, codes * 2 + row % 2]
        actual = ((x.double()[0] * weights).sum() * scale.double()[0] + bias.double()[0]).to(torch.bfloat16)
        assert torch.equal(actual, expected[0, row])


@pytest.mark.parametrize("rank", [0, 1])
def test_tp2_collective_projection_loop_requires_tp2_repeat_and_one_down_sum(rank):
    calls, reduced = [], []

    def projector(inputs, *, tp_size):
        assert tp_size == 2 and len(inputs) == 1 and len(inputs[0]) == 5
        kind = "gate_up" if inputs[0][3].shape[0] * 32 == 2048 else "down"
        calls.append(kind)
        return [tool.synthetic_projection(torch, kind, rank)[1]]

    def reduce(partial):
        assert partial.dtype == torch.float32
        reduced.append(partial)
        return partial + tool.synthetic_projection(torch, "down", 1 - rank)[1].float()

    records = tool.projection_smoke(torch, SimpleNamespace(all_reduce=reduce), "cpu", rank, projector)
    assert calls == ["gate_up", "gate_up", "down", "down"]
    assert len(reduced) == 1
    assert records[0]["tp_sum_exact"] is None
    assert records[1]["tp_sum_exact"] is True and records[1]["padding_k"] == 1024


@pytest.mark.parametrize("communication_only", [False, True])
def test_tp2_collective_rank_completion_requires_both_actual_requested_checks(communication_only):
    records = [
        {"rank": rank, "communication_verified": True, "native_projection_verified": not communication_only}
        for rank in (0, 1)
    ]
    tool.validate_rank_reports(records, communication_only=communication_only)
    records[1]["native_projection_verified"] = None
    with pytest.raises(RuntimeError):
        tool.validate_rank_reports(records, communication_only=communication_only)


def test_tp2_collective_rank_completion_rejects_boolean_rank_or_missing_peer():
    with pytest.raises(RuntimeError):
        tool.validate_rank_reports([{"rank": 0}], communication_only=True)
    with pytest.raises(RuntimeError):
        tool.validate_rank_reports([{"rank": False}, {"rank": 1}], communication_only=True)


def test_tp2_collective_watchdog_is_cancelled_on_failure(monkeypatch):
    events = []

    class Timer:
        def __init__(self, seconds, callback):
            assert seconds == 7 and callable(callback)

        def start(self):
            assert self.daemon is True
            events.append("start")

        def cancel(self):
            events.append("cancel")

    monkeypatch.setattr(tool.threading, "Timer", Timer)
    with pytest.raises(RuntimeError, match="test failure"), tool.deadline(7, 0):
        raise RuntimeError("test failure")
    assert events == ["start", "cancel"]


def test_tp2_collective_plan_only_subprocess_imports_no_runtime_and_writes_nothing(tmp_path):
    script = """
import importlib.abc, runpy, sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.')
               for name in ('torch', 'torch_npu', 'vllm', 'vllm_ascend')):
            raise AssertionError('Unexpected runtime import: ' + fullname)
        return None
sys.meta_path.insert(0, BlockRuntime())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    target = tmp_path / "uncreated reports"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-X",
            "utf8",
            "-c",
            script,
            str(Path(tool.__file__)),
            "--plan-only",
            "--report-dir",
            str(target),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "PLAN_ONLY" and report["communication_verified"] is False
    assert report["native_projection_verified"] is False and report["model_loaded"] is False
    assert "ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2" in report["command"]
    assert not target.exists()


def test_tp2_collective_missing_torchrun_environment_cannot_report_pass(monkeypatch, capsys):
    monkeypatch.setattr(tool.os, "environ", {})
    assert tool.main(["--communication-only"]) == 1
    captured = capsys.readouterr()
    assert "TP2_SMOKE_STATUS=FAIL" in captured.err
    assert "TP2_SMOKE_RESULT=" not in captured.out
