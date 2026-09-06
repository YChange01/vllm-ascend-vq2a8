# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import copy
import json
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tools.validate_vq2a8_phase4_kernel import (
    benchmark,
    bitwise_equal,
    compare,
    same_fp8_oracle,
    synthetic_dense_oracle,
    synthetic_inputs,
)
from tools.validate_vq2a8_tp1_phase4 import evidence_passed, probe_list, short_report
from vllm_ascend.quantization.vq2a8_vector_gather import (
    validate_vector_gather_inputs,
    vq2a8_packed_vector_gather,
)


@pytest.mark.parametrize("rows", [1, 3, 10, 32])
def test_vector_gather_metadata_and_no_cpu_fallback(rows):
    inputs = synthetic_inputs(rows)
    shape = validate_vector_gather_inputs(*inputs)
    assert shape.size_n == 64 and shape.size_k == 512 and shape.column_tiles == 3
    with pytest.raises(ValueError, match="accelerator"):
        vq2a8_packed_vector_gather(*inputs)


@pytest.mark.parametrize("bad", ["rows", "scale", "bias", "activation_stride", "book_dtype", "tiles", "offset"])
def test_vector_gather_rejects_invalid_metadata(bad):
    inputs = list(synthetic_inputs())
    if bad == "rows":
        inputs[0] = torch.zeros((33, 512), dtype=torch.float8_e4m3fn)
    elif bad == "scale":
        inputs[1] = inputs[1][:1]
    elif bad == "bias":
        inputs[2] = inputs[2].double()
    elif bad == "activation_stride":
        inputs[0] = inputs[0].T.contiguous().T
    elif bad == "book_dtype":
        inputs[4] = inputs[4].bfloat16()
    elif bad == "tiles":
        inputs[4] = torch.zeros((33, 2, 16, 2), dtype=torch.float8_e4m3fn)
    elif bad == "offset":
        original = inputs[3]
        inputs[3] = torch.empty(original.numel() + 1, dtype=torch.int32)[1:].reshape(original.shape)
    with pytest.raises(ValueError):
        validate_vector_gather_inputs(*inputs)


def test_synthetic_patterns_cover_rows_nibbles_groups_and_tiles():
    inputs = synthetic_inputs(10, size_n=96, size_k=1024, tiles=7)
    _, _, _, packed, book, tiles = inputs
    assert bool((packed < 0).any())
    assert not torch.equal(packed[0], packed[1])
    assert len(tiles.unique()) == 7
    dense = synthetic_dense_oracle(packed, book, tiles)
    for n in (0, 1, 31, 32, 63, 64, 95):
        for k in (0, 7, 8, 511, 512, 1023):
            code = (k * 7 + (n // 2) * 3 + 3) % 16
            assert dense[n, k] == book.double()[int(tiles[k]), n // 32, code, n % 2]


def test_bitwise_evidence_distinguishes_signed_zero_and_dtype():
    a = torch.tensor([0.0, -0.0], dtype=torch.bfloat16)
    assert bitwise_equal(a, a.clone())
    assert not bitwise_equal(a, torch.zeros_like(a))
    assert not bitwise_equal(a, a.float())
    assert not bitwise_equal(a, a.reshape(1, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Developer CUDA only; not NPU acceptance")
@pytest.mark.parametrize("rows,n,k,tiles", [(1, 32, 512, 1), (3, 64, 512, 3), (10, 96, 1024, 7), (32, 64, 1024, 32)])
def test_gather_device_oracle_batch_invariance_and_determinism(rows, n, k, tiles):
    host = synthetic_inputs(rows, n, k, tiles)
    dense = synthetic_dense_oracle(*host[3:])
    expected = same_fp8_oracle(host[:3], dense)
    inputs = tuple(t.cuda() for t in host)
    actual = vq2a8_packed_vector_gather(*inputs)
    assert compare(expected, actual)["allclose"]
    separate = torch.cat(
        [
            vq2a8_packed_vector_gather(inputs[0][i : i + 1], inputs[1][i : i + 1], inputs[2][i : i + 1], *inputs[3:])
            for i in range(rows)
        ]
    )
    assert bitwise_equal(actual, separate)
    assert bitwise_equal(actual, vq2a8_packed_vector_gather(*inputs))
    with pytest.raises(ValueError, match="CPU-only"):
        synthetic_dense_oracle(*inputs[3:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Developer CUDA only; not A5 Cube/CV acceptance")
@pytest.mark.parametrize("bridge", [False, True])
def test_cube_and_vector_bridge_are_separate_synthetic_microtests(bridge):
    from tools.validate_vq2a8_phase4_kernel import native_micro

    result = native_micro(torch.device("cuda:0"), "cv_bridge" if bridge else "cube_direct")
    assert result["comparison"]["allclose"] and result["repeat_exact"] and result["synthetic_only"]
    assert result["native_fp8_expert_dot"] is False


@pytest.mark.parametrize("warmups,repeats", [(0, 10), (3, 1), (2, 9)])
def test_benchmark_rejects_smoke_as_performance(warmups, repeats):
    with pytest.raises(ValueError, match="warmups"):
        benchmark(lambda: None, torch.device("cpu"), warmups, repeats)


def test_benchmark_records_events_and_wall_separately(monkeypatch):
    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True}

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 2.0

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    calls = []
    result = benchmark(lambda: calls.append(1), torch.device("cuda:0"), 3, 10)
    assert len(calls) == 13 and result["event_ms"] == {"min": 2.0, "median": 2.0, "p95": 2.0}
    assert result["wall_ms"]["min"] > 0 and result["repeats"] == 10


def test_short_report_keeps_boundary_rows_and_full_evidence_unchanged():
    speedups = [
        {"probe": "3:127", "projection": "down", "rows": m, "event_ms_speedup": 2.5, "wall_ms_speedup": 3.0}
        for m in (1, 3, 10, 32)
    ]
    report = {
        "status": "passed",
        "scope": "STANDALONE_KERNELS",
        "device": "npu:0",
        "physical_npu": 4,
        "planned_steps": ["benchmark-3-127"],
        "results": [
            {"name": "benchmark-3-127", "passed": True, "returncode": 0, "timeout": False, "speedups": speedups}
        ],
    }
    before = copy.deepcopy(report)
    text = short_report(report)
    assert text.count("SPEEDUP=") == 1
    assert "m=1:event=2.500x,wall=3.000x" in text and "m=32:event=" in text
    assert "m=3:" not in text and "m=10:" not in text
    assert report == before and "PHASE4=INCOMPLETE" in text


def _evidence(stage="lookup", device="npu:0", probe="0:0", rows=None, cases=None):
    rows, cases = rows or [1], cases or ["deterministic"]
    report = {
        "status": "passed",
        "stage": stage,
        "device": device,
        "native_fp8_expert_dot": False,
        "model_integration_verified": False,
        "npu_verified": device.startswith("npu"),
        "requested": {"probe": probe, "rows": rows, "cases": cases},
    }
    base = {
        "passed": True,
        "repeat_exact": True,
        "oracle": {"allclose": True},
        "baseline": {"allclose": True},
        "chain_baseline": {"allclose": True},
        "row_chunk_exact": True,
        "dense_expert_weight_on_device": False,
    }
    if stage == "lookup":
        report["results"] = [
            {**copy.deepcopy(base), "rows": m, "column_tiles": t} for m in (1, 3, 32) for t in (1, 3, 16, 32)
        ]
    elif stage in ("native", "cube_direct", "cv_bridge"):
        report["results"] = [
            {
                **base,
                "synthetic_only": True,
                "comparison": {"allclose": True},
                "micro_backend": "cann" if stage == "native" and device.startswith("npu") else "triton",
            }
        ]
    else:
        sample = {
            "warmups": 3,
            "repeats": 10,
            "event_ms": {"min": 1, "median": 2, "p95": 3},
            "wall_ms": {"min": 1, "median": 2, "p95": 3},
        }
        report["results"] = [
            {
                **copy.deepcopy(base),
                "probe": probe,
                "rows": m,
                "case": c,
                "projection": kind,
                "timing": {
                    "accepted": [copy.deepcopy(sample) for _ in range(2)],
                    "candidate": [copy.deepcopy(sample) for _ in range(2)],
                },
                "event_ms_speedup": 1.0,
                "wall_ms_speedup": 1.0,
            }
            for m in rows
            for c in cases
            for kind in ("gate_up", "down")
        ]
    return report


@pytest.mark.parametrize(
    "bad", [None, "missing", "duplicate", "cpu", "baseline", "chunk", "repeat", "dense", "partial", "expert_dot"]
)
def test_evidence_requires_complete_synthetic_coverage(tmp_path, bad):
    report = _evidence()
    if bad == "missing":
        report["results"].pop()
    elif bad == "duplicate":
        report["results"][-1] = report["results"][0]
    elif bad == "cpu":
        report["device"] = "cpu"
    elif bad == "baseline":
        report["results"][0]["baseline"]["allclose"] = False
    elif bad == "chunk":
        report["results"][0]["row_chunk_exact"] = False
    elif bad == "repeat":
        report["results"][0]["repeat_exact"] = False
    elif bad == "dense":
        report["results"][0]["dense_expert_weight_on_device"] = True
    elif bad == "partial":
        report["status"] = "running"
    elif bad == "expert_dot":
        report["native_fp8_expert_dot"] = True
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(report))
    assert evidence_passed(path, "lookup", "npu:0", "0:0", [1], ["deterministic"]) is (bad is None)


@pytest.mark.parametrize("bad", [None, "missing_sample", "nan", "negative", "few_repeats", "chain"])
def test_benchmark_evidence_rejects_invalid_timing_or_chain(tmp_path, bad):
    report = _evidence("benchmark")
    sample = report["results"][0]["timing"]["candidate"][0]
    if bad == "missing_sample":
        report["results"][0]["timing"]["candidate"].pop()
    elif bad in ("nan", "negative"):
        sample["event_ms"]["median"] = float("nan") if bad == "nan" else -1
    elif bad == "few_repeats":
        sample["repeats"] = 1
    elif bad == "chain":
        report["results"][0]["chain_baseline"]["allclose"] = False
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(report))
    assert evidence_passed(path, "benchmark", "npu:0", "0:0", [1], ["deterministic"]) is (bad is None)


@pytest.mark.parametrize("fail", [None, "abort", "timeout", "missing_evidence"])
@pytest.mark.parametrize("micro_only", [False, True])
def test_phase4_supervisor_isolates_fails_closed_and_never_promotes(tmp_path, monkeypatch, fail, micro_only):
    from tools import validate_vq2a8_tp1_phase4 as driver

    model, output = tmp_path / "model", tmp_path / "report"
    (model / "experts_vq_ascend_v2").mkdir(parents=True)
    argv = ["phase4", "--model", str(model), "--output-dir", str(output), "--probes", "0:0", "--rows", "1"]
    if micro_only:
        argv.append("--micro-only")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(driver, "LiveChildLog", lambda *args: nullcontext())
    calls = []

    def run(command, **kwargs):
        stage = command[command.index("--stage") + 1]
        calls.append(stage)
        assert "--root-linear-mode" not in command and "model" not in [stage]
        assert kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "4"
        assert kwargs["env"]["ASCEND_LAUNCH_BLOCKING"] == ("0" if stage == "benchmark" else "1")
        if fail == "timeout":
            raise subprocess.TimeoutExpired(command, 1800)
        if fail == "abort":
            return SimpleNamespace(returncode=-6)
        if fail != "missing_evidence":
            cases = command[command.index("--cases") + 1 : command.index("--model")]
            path = output / (
                "lookup.json" if stage == "lookup" else "native.json" if stage == "native" else f"{stage}-0-0.json"
            )
            path.write_text(json.dumps(_evidence(stage, rows=[1], cases=cases)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(driver.subprocess, "run", run)
    assert driver.main() == int(fail is not None)
    report = json.loads((output / "phase4.json").read_text())
    assert report["phase4_complete"] is False and report["model_integration_verified"] is False
    assert report["native_fp8_expert_dot"] is False and report["serving_verified"] is False
    assert calls == (
        ["lookup"] if fail else ["lookup", "native"] if micro_only else ["lookup", "native", "packed", "benchmark"]
    )
    assert "PHASE4=INCOMPLETE" in short_report(report)


@pytest.mark.parametrize("value", ["", "0:0,0:0", "0", "-1:0", "0:0;rm", "0:1/2"])
def test_probe_arguments_cannot_be_empty_duplicated_or_paths(value):
    with pytest.raises(argparse.ArgumentTypeError):
        probe_list(value)


def test_phase4_heartbeat_recognizes_live_stage(tmp_path):
    from tools.vq2a8_live_log import LiveChildLog

    logger = LiveChildLog(tmp_path / "test.log", "phase4")
    logger._record_stage("PHASE4 stage=projection probe=3:127 kind=down m=32 case=small\n")
    assert "kind=down" in logger._last_stage
