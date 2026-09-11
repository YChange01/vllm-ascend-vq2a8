# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tools import benchmark_vq2a8_ascendc_v3 as bench
from tools.vq2a8_perf_report import token_metrics


def cli(tmp_path, *, preflight=True):
    flags = [value for name in ("model", "library", "output-dir") for value in (f"--{name}", str(tmp_path / name))]
    return flags + (["--preflight", str(tmp_path / "preflight.json")] if preflight else [])


def resident(calls):
    return {
        str(i): dict(
            ready=True, full_model_graph_verified=False, route_host_reads=0, descriptor_h2d_bytes=0, decode_calls=calls
        )
        for i in range(43)
    }


def diagnostic(tmp_path, index, library):
    prompt, tokens = list(range(10)), [1, 2, 3, 4]
    steps = [{"tokens": 10, "positions": list(range(10))}]
    steps += [{"tokens": 1, "positions": [10 + i]} for i in range(3)]
    logits = torch.zeros(4, 16, dtype=torch.float32)
    logits[torch.arange(4), torch.tensor(tokens)] = 1
    path = tmp_path / f"diag-{index}.safetensors"
    save_file({"logits": logits}, str(path))
    native_steps = [{"tokens": step["tokens"], "projection_calls": 2, "kernel_launches": 2} for step in steps]
    return dict(
        prompt=prompt,
        tokens=tokens,
        logits_file=str(path),
        logits_sha256=bench.sha256(path),
        v3_before=resident(0),
        v3=resident(3),
        evidence=dict(
            steps=steps,
            root_fp8={"mode": "bf16"},
            expert_backend=dict(
                policy="ascendc_v3",
                fallback_enabled=False,
                library=library,
                layers=[dict(layer=i, steps=copy.deepcopy(native_steps)) for i in range(43)],
            ),
            cache={"layer_calls": {str(i): 4 for i in range(43)}},
            load={"registered_parameters_loaded": 1},
        ),
    )


def sample(kind, repeat):
    ready = [0.1, 0.12, 0.14, 0.16]
    events = [100.0, 120.0, 140.0, 160.0]
    return dict(
        **token_metrics(ready, 0.17, 4),
        token_ready_s=ready,
        case="p10-o4",
        kind=kind,
        repeat=repeat,
        tokens=[1, 2, 3, 4],
        tokens_exact=True,
        token_event_ms=events,
        device_decode_intervals_ms=[20.0] * 3,
        finite=True,
        forwards=4,
        expert_payload_h2d_bytes=0,
        cache_delta=dict(loads=0, hits=1, evictions=0),
        native_calls=344,
        native_launches=344,
        v3_before=resident(0),
        v3_after=resident(3),
    )


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    args = bench.parse_args(cli(tmp_path) + ["--v3-only"])
    args.model.mkdir()
    (args.model / "config.json").write_text(json.dumps({"vocab_size": 16}), encoding="utf-8")
    args.library.write_bytes(b"v3-only-library")
    library = dict(
        path=str(args.library.resolve()), sha256=bench.sha256(args.library), namespace="vq2a8_ascendc_v3", abi_version=1
    )
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(bench, "model_identity", lambda _model: {"model": "v3-test"})
    monkeypatch.setattr(bench, "python_source_hashes", lambda: {"tool": "sha"})
    checks = []

    def preflight(path, receipt, model):
        assert path == args.library and receipt == args.preflight and model == args.model
        checks.append("v3_preflight")
        return library

    def no_reference(*_args, **_kwargs):
        raise AssertionError("v3-only accessed a v1 reference")

    monkeypatch.setattr(bench, "checked_model_preflight", preflight)
    monkeypatch.setattr(bench, "validate_reference", no_reference)
    diagnostics = {"p10-o4": [diagnostic(tmp_path, i, library) for i in range(2)]}
    samples = [sample("warmup", i) for i in range(2)] + [sample("measured", i) for i in range(5)]
    report = dict(
        schema_version=1,
        status="PASS",
        mode="performance",
        implementation="ascendc_v3",
        scope=bench.SCOPE,
        model={"model": "v3-test"},
        python_source_sha256={"tool": "sha"},
        configuration=bench.configuration(args),
        cases=[[10, 4]],
        repeat_exact=True,
        full_model_graph_verified=False,
        quality_verified=False,
        physical_npu="0",
        library=library,
        diagnostics=diagnostics,
        v3_only=True,
        baseline_exact=None,
        baseline_comparison="not_requested",
        warmups=2,
        repeats=5,
        target_tpot_ms=None,
        progress_interval_s=5.0,
        measurement_launch_blocking="0",
        samples=samples,
        summaries=bench.summarize_samples(samples, [(10, 4)], 2, 5, diagnostics),
        performance_target_met=None,
        performance_measurement_verified=True,
    )
    return args, report, checks


def test_v3_only_cli_does_not_require_reference(tmp_path):
    args = bench.parse_args(cli(tmp_path) + ["--v3-only"])
    assert args.v3_only and args.reference_report is None and args.progress_interval == 5
    assert bench.parse_args(cli(tmp_path) + ["--v3-only", "--progress-interval", "0"]).progress_interval == 0
    with pytest.raises(SystemExit):
        bench.parse_args(cli(tmp_path))
    strict = bench.parse_args(cli(tmp_path) + ["--reference-report", str(tmp_path / "reference.json")])
    assert strict.v3_only is False


@pytest.mark.parametrize(
    "extra",
    [
        ["--reference-only"],
        ["--correctness-only"],
        ["--reference-report", "old.json"],
        ["--progress-interval", "nan"],
        ["--progress-interval", "-1"],
    ],
)
def test_v3_only_rejects_conflicting_modes_and_bad_progress(tmp_path, extra):
    with pytest.raises(SystemExit):
        bench.parse_args(cli(tmp_path) + ["--v3-only", *extra])


def test_v3_only_still_requires_preflight(tmp_path):
    with pytest.raises(SystemExit):
        bench.parse_args(cli(tmp_path, preflight=False) + ["--v3-only"])


def test_v3_only_verifies_real_retained_logits_without_reference(evidence):
    args, report, checks = evidence
    assert bench.verify_report(report, args) is report
    assert checks == ["v3_preflight"]
    assert report["baseline_exact"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("baseline_exact", True),
        ("baseline_exact", False),
        ("baseline_comparison", "verified"),
        ("quality_verified", True),
        ("reference_report_sha256", "forged"),
        ("v3_only", False),
    ],
)
def test_v3_only_rejects_forged_comparison_or_quality_claim(evidence, field, value):
    args, report, _checks = evidence
    report[field] = value
    with pytest.raises(ValueError):
        bench.verify_report(report, args)


@pytest.mark.parametrize(
    "failure", ["logits_repeat", "nonfinite", "missing_layer", "host_route", "not_resident", "sample_tokens"]
)
def test_v3_only_self_checks_remain_mandatory(evidence, failure):
    args, report, _checks = evidence
    second = report["diagnostics"]["p10-o4"][1]
    if failure in ("logits_repeat", "nonfinite"):
        logits = torch.zeros(4, 16, dtype=torch.float32)
        logits[torch.arange(4), torch.tensor(second["tokens"])] = 1
        logits[0, 0] = 0.25 if failure == "logits_repeat" else float("nan")
        save_file({"logits": logits}, second["logits_file"])
        second["logits_sha256"] = bench.sha256(Path(second["logits_file"]))
    elif failure == "missing_layer":
        second["evidence"]["expert_backend"]["layers"].pop()
    elif failure == "host_route":
        second["v3"]["0"]["route_host_reads"] = 1
    elif failure == "not_resident":
        second["v3"]["0"]["ready"] = False
    else:
        report["samples"][0]["tokens"] = [1, 2, 3, 5]
    with pytest.raises(ValueError):
        bench.verify_report(report, args)


def test_v3_only_report_is_not_a_strict_comparison_receipt(evidence):
    args, report, _checks = evidence
    args.v3_only = False
    args.reference_report = args.model / "nonexistent-v1-report.json"
    with pytest.raises(ValueError, match="identity"):
        bench.verify_report(report, args)


def test_v3_only_summary_calls_baseline_not_requested(evidence, tmp_path):
    _args, report, _checks = evidence
    bench.write_report(tmp_path, report)
    assert "BASELINE_EXACT=NOT_REQUESTED REPEAT_EXACT=True" in (tmp_path / "summary.txt").read_text()
    assert json.loads((tmp_path / "summary.json").read_text())["baseline_exact"] is None
