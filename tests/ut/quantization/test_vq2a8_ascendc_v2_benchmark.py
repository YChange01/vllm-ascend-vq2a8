# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tools import benchmark_vq2a8_ascendc_v2 as bench
from vllm_ascend.quantization import vq2a8_ascendc_v2 as backend


def cli(tmp_path):
    return [
        value
        for name in ("model", "library", "preflight", "output-dir")
        for value in (f"--{name}", str(tmp_path / name))
    ]


def test_ascendc_v2_benchmark_cli_defaults_and_context(tmp_path):
    args = bench.parse_args(cli(tmp_path))
    assert args.cases == [(10, 4)] and args.preset == "batched"
    assert args.warmups == 2 and args.repeats == 5
    assert bench.parse_args(cli(tmp_path) + ["--cases", "10:4,32:32,96:32"]).cases[-1] == (96, 32)


@pytest.mark.parametrize(
    "flags",
    [
        ["--warmups", "1"],
        ["--repeats", "4"],
        ["--cases", "96:33"],
        ["--cases", "10:4,10:4"],
        ["--cases", "x"],
        ["--preset", "pipeline"],
    ],
)
def test_ascendc_v2_benchmark_rejects_invalid_cli(tmp_path, flags):
    with pytest.raises(SystemExit):
        bench.parse_args(cli(tmp_path) + flags)


@pytest.mark.parametrize("blocking", [None, "0"])
def test_ascendc_v2_benchmark_async_measurement_only(blocking):
    assert (
        bench.require_measurement_environment({"ASCEND_LAUNCH_BLOCKING": blocking, "ASCEND_RT_VISIBLE_DEVICES": "4"})[
            "physical_npu"
        ]
        == "4"
    )


@pytest.mark.parametrize(
    "env", [{"ASCEND_LAUNCH_BLOCKING": "1", "ASCEND_RT_VISIBLE_DEVICES": "0"}, {"ASCEND_RT_VISIBLE_DEVICES": "0,1"}, {}]
)
def test_ascendc_v2_benchmark_rejects_blocking_or_multiple_devices(env):
    with pytest.raises(ValueError):
        bench.require_measurement_environment(env)


def diagnostic_record(library):
    tokens = [1, 1, 1, 1]
    expected = [{"tokens": 10, "positions": list(range(10))}]
    expected.extend({"tokens": 1, "positions": [10 + i]} for i in range(3))
    return {
        "prompt": [0] * 10,
        "tokens": tokens,
        "preset": "batched",
        "evidence": {
            "steps": expected,
            "expert_backend": {
                "policy": "ascendc_v2",
                "fallback_enabled": False,
                "library": library,
                "layers": [
                    {
                        "layer": layer,
                        "steps": [
                            {
                                "tokens": s["tokens"],
                                "projection_calls": 2,
                                "expert_calls": 1,
                                "projection_rows": 2 * s["tokens"],
                                "kernel_launches": 2,
                            }
                            for s in expected
                        ],
                    }
                    for layer in range(43)
                ],
            },
            "root_fp8": {"mode": "bf16"},
            "cache": {"layer_calls": {str(i): 4 for i in range(43)}},
            "load": {"moe_layers": 43, "registered_parameters_loaded": 984},
        },
    }


def sample(kind, repeat=0, multiplier=1):
    ready = [float(multiplier * i) for i in (1, 2, 3, 4)]
    return {
        "case": "p10-o4",
        "preset": "batched",
        "kind": kind,
        "repeat": repeat,
        "tokens": [1] * 4,
        "tokens_exact": True,
        "finite": True,
        "forwards": 4,
        "native_calls": 344,
        "native_launches": 344,
        "cache_delta": {"loads": 0, "hits": 486, "evictions": 0},
        "token_ready_s": ready,
        "device_span_ms": 1.0,
        **bench.token_metrics(ready, ready[-1] + 0.5, 4),
    }


def samples():
    return (
        [sample("first_timed_after_diagnostics", multiplier=100)]
        + [sample("warmup", i, 100) for i in range(2)]
        + [sample("measured", i, i + 1) for i in range(5)]
    )


def test_ascendc_v2_benchmark_stats_exclude_first_and_warmups():
    pair = [diagnostic_record({}), diagnostic_record({})]
    result = bench.summarize_samples(samples(), [(10, 4)], "batched", 2, 5, {"p10-o4": pair})
    assert result[0]["ttft_s"] == 3 and result[0]["tpot_s"] == 3
    assert result[0]["e2e_s"] == 12.5 and result[0]["n"] == 5
    assert result[0]["hot_cache_verified"] is True


def test_ascendc_v2_benchmark_cache_misses_retained_but_not_hot():
    rows = samples()
    rows[-1]["cache_delta"].update(loads=1, evictions=1)
    result = bench.summarize_samples(rows, [(10, 4)], "batched", 2, 5, {"p10-o4": [diagnostic_record({})] * 2})
    assert result[0]["hot_cache_verified"] is False
    assert result[0]["cache_loads"] == 1 and result[0]["n"] == 5


@pytest.mark.parametrize(
    "bad",
    [
        "tokens",
        "finite",
        "forwards",
        "native_calls",
        "native_launches",
        "ttft_s",
        "tpot_s",
        "decode_intervals_s",
        "cache_delta",
    ],
)
def test_ascendc_v2_benchmark_rejects_forged_samples(bad):
    row = sample("measured")
    row[bad] = {
        "tokens": [0] * 4,
        "finite": False,
        "forwards": 3,
        "native_calls": 0,
        "native_launches": 0,
        "ttft_s": 99,
        "tpot_s": 99,
        "decode_intervals_s": [99] * 3,
        "cache_delta": {"loads": -1, "hits": 0, "evictions": 0},
    }[bad]
    with pytest.raises(ValueError):
        bench.validate_sample(row, [0] * 10, [1] * 4, "batched")


@pytest.mark.parametrize("bad", ["policy", "library", "fallback", "layers", "steps", "root", "logits"])
def test_ascendc_v2_benchmark_rejects_forged_diagnostics(tmp_path, bad):
    library = {"path": str(tmp_path / "libvq2a8_ascendc_v2.so"), "sha256": "a" * 64}
    record = diagnostic_record(library)
    evidence = copy.deepcopy(record["evidence"])
    logits = torch.tensor([[0.0, 1.0]] * 4)
    if bad == "policy":
        evidence["expert_backend"]["policy"] = "ascendc"
    elif bad == "library":
        evidence["expert_backend"]["library"]["sha256"] = "b" * 64
    elif bad == "fallback":
        evidence["expert_backend"]["fallback_enabled"] = True
    elif bad == "layers":
        evidence["expert_backend"]["layers"].pop()
    elif bad == "steps":
        evidence["steps"].pop()
    elif bad == "root":
        evidence["root_fp8"]["mode"] = "online_fp8_sm90"
    else:
        logits[0, 0] = float("nan")
    with pytest.raises(ValueError):
        bench.validate_diagnostic(evidence, logits, record["prompt"], record["tokens"], 2, library)


def report_fixture(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"num_hidden_layers": 43, "vocab_size": 2}))
    lib = tmp_path / "libvq2a8_ascendc_v2.so"
    lib.write_bytes(b"not-an-executable")
    identity = {"path": str(lib.resolve()), "sha256": bench.digest(lib), "abi_version": 1}
    monkeypatch.setattr(backend, "validate_build_manifest", lambda *args: identity)
    monkeypatch.setattr(bench, "model_identity", lambda *args: {"config": "abc"})
    monkeypatch.setattr(bench, "source_hashes", lambda: {"source": "abc"})
    pair = []
    for i in range(2):
        record = diagnostic_record(identity)
        path = tmp_path / f"diagnostic-{i}.safetensors"
        save_file({"logits": torch.tensor([[0.0, 1.0]] * 4)}, str(path))
        record.update(logits_file=str(path), logits_sha256=bench.digest(path))
        pair.append(record)
    report = {
        "status": "PASS",
        "performance_measurement_verified": True,
        "implementation": "ascendc_v2",
        "scope": bench.SCOPE,
        "preset": "batched",
        "cases": [[10, 4]],
        "warmups": 2,
        "repeats": 5,
        "model": {"config": "abc"},
        "python_source_sha256": {"source": "abc"},
        "library": identity,
        "library_unchanged": True,
        "quality_verified": False,
        "serving_verified": False,
        "performance_target_met": None,
        "measurement_environment": {"ASCEND_LAUNCH_BLOCKING": "0", "physical_npu": "0"},
        "diagnostics": {"p10-o4": pair},
        "samples": samples(),
        "hot_cache_verified": True,
    }
    report["summaries"] = bench.summarize_samples(report["samples"], [(10, 4)], "batched", 2, 5, report["diagnostics"])
    return report, identity, model


def test_ascendc_v2_benchmark_supervisor_revalidates_without_npu_environment(tmp_path, monkeypatch):
    report, identity, model = report_fixture(tmp_path, monkeypatch)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    assert bench.verify_report(report, identity, model, [(10, 4)], "batched", 2, 5) is report


@pytest.mark.parametrize(
    "bad", ["library", "source", "model", "scope", "blocking", "summary", "samples", "alias", "tensor", "status"]
)
def test_ascendc_v2_benchmark_rejects_tampered_reports(tmp_path, monkeypatch, bad):
    report, identity, model = report_fixture(tmp_path, monkeypatch)
    if bad == "library":
        report["library"] = {**identity, "sha256": "b" * 64}
    elif bad == "source":
        report["python_source_sha256"] = {}
    elif bad == "model":
        report["model"] = {}
    elif bad == "scope":
        report["scope"] = "serving"
    elif bad == "blocking":
        report["measurement_environment"]["ASCEND_LAUNCH_BLOCKING"] = "1"
    elif bad == "summary":
        report["summaries"][0]["ttft_s"] = 0.1
    elif bad == "samples":
        report["samples"].pop()
    elif bad == "alias":
        report["diagnostics"]["p10-o4"][1] = report["diagnostics"]["p10-o4"][0]
    elif bad == "tensor":
        record = report["diagnostics"]["p10-o4"][1]
        save_file({"logits": torch.tensor([[0.0, 2.0]] * 4)}, record["logits_file"])
        record["logits_sha256"] = bench.digest(Path(record["logits_file"]))
    else:
        report["status"] = "FAIL"
    with pytest.raises(ValueError):
        bench.verify_report(report, identity, model, [(10, 4)], "batched", 2, 5)


def test_ascendc_v2_benchmark_environment_failure_writes_closed_report(tmp_path, monkeypatch):
    args = bench.parse_args(cli(tmp_path))
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")
    with pytest.raises(ValueError):
        bench.run(args)
    report = json.loads((args.output_dir / "summary.json").read_text())
    assert report["status"] == "FAIL" and report["performance_measurement_verified"] is False
    assert "ASCEND_LAUNCH_BLOCKING" in report["error"]
    assert (args.output_dir / "summary.txt").exists()
