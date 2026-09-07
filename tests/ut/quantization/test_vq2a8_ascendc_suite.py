# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import validate_vq2a8_ascendc as gate
from tools import validate_vq2a8_ascendc_suite as suite

CONFIG = {"num_hidden_layers": 43, "num_hash_layers": 3, "n_routed_experts": 256}


def evidence(stage, probe="0:0"):
    report = {
        "status": "passed",
        "implementation": "ascendc",
        "stage": stage,
        "probe": probe,
        "device": {"type": "npu", "soc": 260},
        "library": {"sha256": "abc"},
        "native_instruction_verified": False,
        "on_chip_decode_verified": False,
        "model_integration_verified": False,
        "performance_verified": False,
        "results": [],
    }
    for key in sorted(gate.expected_keys(stage)):
        row = {
            "key": key,
            "passed": True,
            "repeat_exact": True,
            "row_chunk_exact": True,
            "oracle": {"allclose": True},
            "accepted_baseline": {"allclose": True},
            "independent_chain": {"allclose": True},
        }
        if stage == "timing":
            row["timings"] = {
                "launch_blocking": False,
                "scope": "resident_prepared_projection_only",
                "wall_median_ratio_baseline_over_candidate": 2.0,
                **{
                    name: {
                        "warmups": 3,
                        "repeats": 10,
                        **{
                            clock: {"min": scale, "median": 2 * scale, "p95": 3 * scale}
                            for clock in ("event_ms", "wall_ms")
                        },
                    }
                    for name, scale in (("candidate", 1.0), ("accepted_baseline", 2.0))
                },
            }
        report["results"].append(row)
    return report


def test_all_layer_plan_is_bounded_and_controls_run_once():
    probes = suite.select_probes(CONFIG)
    assert len(probes) == len(set(probes)) == 54
    assert {int(p.split(":")[0]) for p in probes} == set(range(43))
    assert [p for p in probes if int(p.split(":")[0]) < 3] == ["0:0", "1:0", "2:0"]
    for layer in (3, 23, 42):
        assert {f"{layer}:{expert}" for expert in (0, 127, 135, 255)} <= set(probes)
    plan = suite.make_plan(probes)
    assert len(plan) == 61
    assert [p["stage"] for p in plan[:4]] == ["direct", "bridge", "fused", "boundaries"]
    assert sum(len(gate.expected_keys(p["stage"])) for p in plan if p["stage"] != "timing") == 2656
    assert len(gate.expected_keys("timing")) == 6


@pytest.mark.parametrize("requested", ["", "0", "0:0,0:0", "1:0,01:0", "43:0", "2:1", "3:256", "-1:0"])
def test_invalid_subset_rejected(requested):
    with pytest.raises(ValueError):
        suite.select_probes(CONFIG, requested)


def test_subset_and_small_model_plans():
    assert suite.select_probes(CONFIG, "3:135,42:255") == ["3:135", "42:255"]
    assert suite.select_probes({"num_hidden_layers": 1, "num_hash_layers": 1, "n_routed_experts": 256}) == ["0:0"]
    assert len(suite.make_plan(["0:0"])) == 6
    with pytest.raises(ValueError):
        suite.select_probes({**CONFIG, "num_hash_layers": 44})
    with pytest.raises(ValueError):
        suite.select_probes({**CONFIG, "num_hidden_layers": True})


@pytest.mark.parametrize("stage", ["boundaries", "timing"])
def test_new_stage_receipts_require_complete_numerics(stage, tmp_path):
    path = tmp_path / "result.json"
    original = evidence(stage)
    path.write_text(json.dumps(original))
    assert gate.evidence_passed(path, stage, "abc")
    for field in ("oracle", "accepted_baseline") + (("independent_chain",) if stage == "timing" else ()):
        changed = copy.deepcopy(original)
        changed["results"][0][field]["allclose"] = False
        path.write_text(json.dumps(changed))
        assert not gate.evidence_passed(path, stage, "abc")
    original["results"].pop()
    path.write_text(json.dumps(original))
    assert not gate.evidence_passed(path, stage, "abc")


@pytest.mark.parametrize(
    "mutation", ["missing", "blocking", "scope", "warmups", "repeats", "nan", "zero", "order", "ratio", "boolean"]
)
def test_timing_receipt_rejects_invalid_measurements(mutation, tmp_path):
    report = evidence("timing")
    timing = report["results"][0]["timings"]
    if mutation == "missing":
        del report["results"][0]["timings"]
    elif mutation == "blocking":
        timing["launch_blocking"] = True
    elif mutation == "scope":
        timing["scope"] = "full_model"
    elif mutation in ("warmups", "repeats"):
        timing["candidate"][mutation] = 1
    elif mutation in ("nan", "zero", "order"):
        timing["candidate"]["event_ms"]["median"] = {"nan": float("nan"), "zero": 0, "order": 0.5}[mutation]
    elif mutation == "ratio":
        timing["wall_median_ratio_baseline_over_candidate"] = 999
    elif mutation == "boolean":
        timing["candidate"]["event_ms"]["min"] = True
    path = tmp_path / "timing.json"
    path.write_text(json.dumps(report))
    assert not gate.evidence_passed(path, "timing", "abc")


def args_for(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(CONFIG))
    return argparse.Namespace(
        model=model,
        library=tmp_path / "native.so",
        physical_npu=4,
        probes="0:0,3:135,42:255",
        output_dir=tmp_path / "report",
        warmups=3,
        repeats=10,
        timeout=30,
    )


@pytest.mark.parametrize(
    "failure", ["numerical", "runtime", "timeout", "missing", "malformed", "device_with_assertion"]
)
def test_child_failure_classification(failure, tmp_path, monkeypatch):
    args = args_for(tmp_path)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    step = {"id": "expert-0-0", "stage": "expert", "probe": "0:0"}

    def child(command, **kwargs):
        assert kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "4"
        assert kwargs["env"]["ASCEND_LAUNCH_BLOCKING"] == "1"
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 30)
        if failure != "missing":
            error = "AssertionError: numerical mismatch" if failure != "runtime" else "RuntimeError: device"
            if failure == "malformed":
                error = {"not": "a message"}
            Path(command[command.index("--output") + 1]).write_text(json.dumps({"status": "failed", "error": error}))
        if failure == "device_with_assertion":
            kwargs["stdout"].write("aicore exception 507015\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(suite.subprocess, "run", child)
    result = suite.run_step(args, step, log_dir, "abc")
    assert not result["passed"]
    assert result["failure_kind"] == ("numerical" if failure == "numerical" else "runtime_or_unknown")


def test_timing_child_is_unblocked_and_numerically_gated(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    def child(command, **kwargs):
        assert kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "4"
        assert kwargs["env"]["ASCEND_LAUNCH_BLOCKING"] == "0"
        Path(command[command.index("--output") + 1]).write_text(json.dumps(evidence("timing")))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(suite.subprocess, "run", child)
    assert suite.run_step(args, {"id": "timing-0-0", "stage": "timing", "probe": "0:0"}, log_dir, "abc")["passed"]


@pytest.mark.parametrize("failure", [None, "numerical", "runtime_or_unknown"])
def test_campaign_continues_only_after_numeric_failures(failure, tmp_path, monkeypatch):
    args = args_for(tmp_path)
    monkeypatch.setattr(gate, "library_evidence", lambda _: {"sha256": "abc"})
    monkeypatch.setattr(suite, "collect_binary_evidence", lambda *a: {"status": "incomplete_review_pending"})
    calls = []

    def step_runner(args, step, directory, digest):
        calls.append(step["id"])
        failed = failure is not None and step["id"] == "expert-0-0"
        path = directory / f"{step['id']}.json"
        path.write_text(json.dumps(evidence(step["stage"], step["probe"])))
        return {
            **step,
            "status": "failed" if failed else "passed",
            "passed": not failed,
            "failure_kind": failure if failed else None,
            "evidence": str(path),
        }

    monkeypatch.setattr(suite, "run_step", step_runner)
    assert suite.run(args) == (1 if failure else 0)
    report = json.loads((args.output_dir / "summary.json").read_text())
    assert len(report["results"]) == len(report["plan"])
    assert all(calls.count(s) == 1 for s in ("direct", "bridge", "fused", "boundaries"))
    if failure == "runtime_or_unknown":
        assert calls[-1] == "expert-0-0"
    elif failure == "numerical":
        assert "expert-42-255" in calls and "timing-42-255" in calls
        assert "timing-0-0" not in calls
    else:
        assert report["standalone_numerics_verified"] and report["timings_collected"]
    assert report["performance_verified"] is False
    assert report["native_instruction_verified"] is False
    assert report["on_chip_decode_verified"] is False
    assert report["model_integration_verified"] is False
    assert report["all_layers_sampled"] is False  # explicit subset is labelled


@pytest.mark.parametrize("mode", ["success", "missing_tool", "timeout", "no_instructions"])
def test_binary_evidence_is_bounded_and_never_auto_certified(mode, tmp_path, monkeypatch):
    build = tmp_path / "build"
    objects = build / "auto_gen/vq2a8_ascendc_kernel"
    objects.mkdir(parents=True)
    (objects / "device.o").write_bytes(b"\x7fELFhost-test-only")
    (objects / "duplicate.o").write_bytes(b"\x7fELFhost-test-only")
    (objects / "preprocessed.o").write_text("not an ELF")
    (build / "unrelated.o").write_bytes(b"\x7fELFunrelated")
    assert len(suite.discover_device_objects(build.resolve())) == 1
    cann = tmp_path / "cann"
    tool = cann / "aarch64-linux/ccec_compiler/bin/llvm-objdump"
    if mode != "missing_tool":
        tool.parent.mkdir(parents=True)
        tool.write_bytes(b"test-tool-not-executed")

    def disassemble(command, **kwargs):
        assert command[0] == str(tool)
        if mode == "timeout":
            raise subprocess.TimeoutExpired(command, 60)
        kwargs["stdout"].write("header only\n" if mode == "no_instructions" else "0000: aa bb mad.fp8 mock\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(suite.subprocess, "run", disassemble)
    report = suite.collect_binary_evidence(
        {"path": str(build / "native.so"), "sha256": "abc", "build": {"cann": str(cann)}}, tmp_path / "audit"
    )
    assert report["status"] == ("collected_review_pending" if mode == "success" else "incomplete_review_pending")
    assert report["native_instruction_verified"] is False
    assert report["on_chip_decode_verified"] is False
    assert report["object_linkage_to_loaded_library_verified"] is False
