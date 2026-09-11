# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import accept_vq2a8_ascendc_v3 as accept
from tools import benchmark_vq2a8_ascendc_v3 as bench
from tools import build_vq2a8_ascendc_v3 as build
from tools import validate_vq2a8_ascendc_v3 as validate
from tools.vq2a8_perf_report import token_metrics


def cli(tmp_path):
    return [
        value
        for name in ("model", "library", "preflight", "reference-report", "output-dir")
        for value in (f"--{name}", str(tmp_path / name))
    ]


def resident(calls):
    return {
        str(i): dict(
            ready=True, full_model_graph_verified=False, route_host_reads=0, descriptor_h2d_bytes=0, decode_calls=calls
        )
        for i in range(43)
    }


def sample(kind="measured", repeat=0, count=4):
    ready = [0.1 + i * 0.02 for i in range(count)]
    events = [v * 1000 for v in ready]
    return dict(
        **token_metrics(ready, ready[-1] + 0.001, count),
        token_ready_s=ready,
        case=f"p10-o{count}",
        kind=kind,
        repeat=repeat,
        tokens=list(range(count)),
        tokens_exact=True,
        token_event_ms=events,
        device_span_ms=events[-1],
        device_decode_intervals_ms=[b - a for a, b in zip(events, events[1:])],
        finite=True,
        forwards=count,
        expert_payload_h2d_bytes=0,
        cache_delta=dict(loads=0, hits=1, evictions=0),
        native_calls=86 * count,
        native_launches=86 * count,
        v3_before=resident(2),
        v3_after=resident(2 + count - 1),
    )


def test_v3_tools_default_quick_plan_and_independent_paths(tmp_path):
    args = accept.parse_args(["--model", str(tmp_path), "--soc", "Ascend950PR_9599", "--plan-only"])
    assert args.cases == "10:4" and args.cache_reserve_gib == 16 and args.memory_fraction == 0.9
    steps = accept.commands(args, tmp_path / "output")
    assert [s[0] for s in steps] == ["environment", "build", "preflight", "v1-reference", "model-exact"]
    assert "ascendc-v3" in " ".join(steps[2][1])
    assert "--reference-only" in steps[3][1] and "--correctness-only" in steps[4][1]
    assert args.baseline_library.as_posix().endswith("vq2a8-ascendc-v026/libvq2a8_ascendc.so")
    assert bench.parse_args(cli(tmp_path)).target_tpot_ms is None


def test_v3_tools_benchmark_loads_v3_once_and_reuses_reference(tmp_path):
    args = accept.parse_args(
        [
            "--model",
            str(tmp_path),
            "--library",
            str(tmp_path / "v3.so"),
            "--benchmark",
            "--profile",
            "--cases",
            "10:64,32:64",
            "--target-tpot-ms",
            "20",
            "--reference-report",
            str(tmp_path / "reference.json"),
        ]
    )
    steps = accept.commands(args, tmp_path / "out")
    assert [name for name, _ in steps] == ["environment", "preflight", "performance"]
    assert "--target-tpot-ms" in steps[-1][1]
    assert "--profile" in steps[-1][1]
    assert "--correctness-only" not in steps[-1][1]


def test_v3_tools_profile_requires_benchmark(tmp_path):
    with pytest.raises(SystemExit):
        accept.parse_args(["--model", str(tmp_path), "--library", str(tmp_path / "v3.so"), "--profile"])
    with pytest.raises(SystemExit):
        bench.parse_args(cli(tmp_path) + ["--correctness-only", "--profile"])


def test_v3_only_plan_never_resolves_old_library_or_reference(tmp_path):
    class UnavailableBaseline:
        def resolve(self):
            raise AssertionError("v3-only must never access the old library")

    args = accept.parse_args(
        ["--model", str(tmp_path), "--library", str(tmp_path / "v3.so"), "--v3-only", "--benchmark"]
    )
    args.baseline_library = UnavailableBaseline()
    steps = accept.commands(args, tmp_path / "out")
    assert [name for name, _ in steps] == ["environment", "preflight", "performance"]
    candidate = steps[-1][1]
    assert "--v3-only" in candidate
    assert "--reference-report" not in candidate and "--reference-only" not in candidate
    assert candidate[candidate.index("--progress-interval") + 1] == "5.0"


@pytest.mark.parametrize(
    "flags",
    [
        ["--v3-only"],
        ["--v3-only", "--benchmark", "--reference-report", "unused.json"],
        ["--progress-interval", "nan"],
        ["--progress-interval", "-1"],
    ],
)
def test_v3_only_accept_rejects_conflicting_options(tmp_path, flags):
    with pytest.raises(SystemExit):
        accept.parse_args(["--model", str(tmp_path), "--library", str(tmp_path / "v3.so"), *flags])


def test_v3_only_accept_does_not_promote_unrequested_baseline(tmp_path, monkeypatch, capsys):
    args = accept.parse_args(
        [
            "--model",
            str(tmp_path),
            "--library",
            str(tmp_path / "v3.so"),
            "--v3-only",
            "--benchmark",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    monkeypatch.setattr(accept, "parse_args", lambda: args)
    monkeypatch.setattr(accept.platform, "system", lambda: "Linux")
    monkeypatch.setattr(accept, "acceptance_environment", lambda *_: {})
    monkeypatch.setattr(accept, "library_identity", lambda *_: {})
    monkeypatch.setattr(accept, "validate_receipt", lambda *_: None)
    monkeypatch.setattr(bench, "verify_report", lambda *_: None)

    def supervise(command, log, environment, timeout):
        if log.stem == "preflight":
            (args.output_dir / "preflight.json").write_text("{}", encoding="utf-8")
        if log.stem == "performance":
            child = args.output_dir / "performance"
            child.mkdir()
            (child / "summary.json").write_text(
                json.dumps(
                    {
                        "baseline_exact": None,
                        "baseline_comparison": "not_requested",
                        "performance_target_met": False,
                        "summaries": [],
                    }
                ),
                encoding="utf-8",
            )
        return dict(exit=0, timeout=False, elapsed_s=0.01, log=str(log), command=command)

    monkeypatch.setattr(accept, "supervise", supervise)
    assert accept.main() == 0
    report = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert report["baseline_exact"] is None
    assert report["baseline_comparison"] == "not_requested"
    assert report["performance_measurement_verified"] is True
    assert report["performance_target_met"] is False
    assert "BASELINE_EXACT=NOT_REQUESTED" in capsys.readouterr().out


@pytest.mark.parametrize(
    "flags",
    [
        ["--cache-reserve-gib", "nan"],
        ["--cache-budget-gib", "-1"],
        ["--memory-fraction", "1.01"],
        ["--cases", "96:64"],
        ["--cases", "10:4,10:4"],
        ["--warmups", "1"],
        ["--repeats", "4"],
        ["--target-tpot-ms", "nan"],
    ],
)
def test_v3_tools_reject_invalid_measurement_options(tmp_path, flags):
    with pytest.raises(SystemExit):
        bench.parse_args(cli(tmp_path) + flags)


def test_v3_tools_does_not_reuse_v2_cmake_target(tmp_path):
    args = build.parse_args(["--soc", "Ascend950PR_9599", "--plan-only"])
    commands = build.commands(args, tmp_path, "ascendc.cmake", "npu", "torch")
    assert commands[1][commands[1].index("--target") + 1] == "vq2a8_ascendc_v3"
    assert "vq2a8_ascendc_v2" not in " ".join(commands[0] + commands[1])
    hashes = build.source_hashes()
    assert any(k.startswith("csrc/vq2a8_ascendc_v3/") for k in hashes)
    assert any(k.startswith("csrc/vq2a8_ascendc/") for k in hashes)
    assert "tools/build_vq2a8_ascendc_v2.py" in hashes


def test_v3_tools_preflight_requires_prepared_abi_matrix():
    cases = validate.expected_cases()
    assert {f"prepared:k{k}:g{g}" for k in (512, 2048, 4096) for g in (1, 6)} <= cases
    assert len(cases) == 22
    assert "vllm_ascend/quantization/vq2a8_execution_v3.py" in validate.python_source_hashes()
    assert {
        "tools/build_vq2a8_ascendc_v2.py",
        "tools/validate_vq2a8_ascendc_v2.py",
        "vllm_ascend/quantization/vq2a8_moe.py",
    } <= validate.python_source_hashes().keys()


def test_v3_tools_token_p95_is_from_individual_tokens():
    values = [sample("warmup", i, 64) for i in range(2)] + [sample("measured", i, 64) for i in range(5)]
    diagnostics = {"p10-o64": [{"tokens": list(range(64))}]}
    rows = bench.summarize_samples(values, [(10, 64)], 2, 5, diagnostics)
    assert rows[0]["decode_token_ms"]["n"] == 5 * 63
    assert rows[0]["device_decode_token_ms"]["n"] == 5 * 63
    assert rows[0]["decode_token_ms"]["p95_nearest_rank"] == pytest.approx(20)


@pytest.mark.parametrize(
    "mutation", ["h2d", "load", "host_route", "missing_layer", "missing_event", "extra_decode", "missing_decode"]
)
def test_v3_tools_resident_or_timing_failures_do_not_pass(mutation):
    value = sample()
    if mutation == "h2d":
        value["expert_payload_h2d_bytes"] = 128
    elif mutation == "load":
        value["cache_delta"]["loads"] = 1
    elif mutation == "host_route":
        value["v3_after"]["0"]["route_host_reads"] = 1
    elif mutation == "missing_layer":
        del value["v3_after"]["0"]
    elif mutation == "missing_event":
        value["token_event_ms"].pop()
    elif mutation == "extra_decode":
        value["v3_after"]["0"]["decode_calls"] += 1
    else:
        value["v3_after"]["0"]["decode_calls"] -= 1
    with pytest.raises(ValueError):
        bench.validate_sample(value, 4, list(range(4)))


def test_v3_tools_reject_missing_measurement_repeats():
    values = [sample("warmup", i) for i in range(2)] + [sample("measured", i) for i in range(5)]
    bench.summarize_samples(values, [(10, 4)], 2, 5, {"p10-o4": [{"tokens": list(range(4))}]})
    broken = copy.deepcopy(values)
    broken[-1]["repeat"] = 0
    with pytest.raises(ValueError):
        bench.summarize_samples(broken, [(10, 4)], 2, 5, {"p10-o4": [{"tokens": list(range(4))}]})


def test_v3_tools_plan_only_never_imports_torch(tmp_path):
    root = Path(__file__).resolve().parents[3]
    command = [
        sys.executable,
        "-c",
        "import runpy,sys; "
        "sys.argv=['accept', '--model', 'missing-model', '--soc', 'Ascend950PR_9599', '--plan-only']; "
        "runpy.run_path('tools/accept_vq2a8_ascendc_v3.py',run_name='__main__')",
    ]
    # Block torch even if installed: any accidental import must make the plan fail.
    command[2] = "import sys; sys.modules['torch']=None; sys.modules['torch_npu']=None; " + command[2]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["scope"] == "plan_only_no_device_execution"
    assert plan["full_model_graph_verified"] is False
