# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tools import validate_vq2a8_ascendc as native
from tools import validate_vq2a8_ascendc_suite as suite
from tools import validate_vq2a8_tp1_acceptance as acceptance
from vllm_ascend.quantization import vq2a8_ascendc as wrapper


def child_evidence(stage):
    stats = {
        "warmups": 3,
        "repeats": 10,
        "wall_ms": {"min": 1, "median": 2, "p95": 3},
        "event_ms": {"min": 1, "median": 2, "p95": 3},
    }
    return {
        "status": "passed",
        "implementation": "ascendc",
        "stage": stage,
        "probe": "0:0",
        "device": {"type": "npu", "soc": 260},
        "library": {"sha256": "a" * 64},
        "hardware_runtime": {"checked": True, "simulator_runtime_paths": []},
        "native_instruction_verified": False,
        "on_chip_decode_verified": False,
        "performance_verified": False,
        "model_integration_verified": False,
        "results": [
            {
                "key": key,
                "passed": True,
                "repeat_exact": True,
                "row_chunk_exact": True,
                "oracle": {"allclose": True},
                "accepted_baseline": {"allclose": True},
                "independent_chain": {"allclose": True},
                "timings": {
                    "candidate": stats,
                    "accepted_baseline": stats,
                    "wall_median_ratio_baseline_over_candidate": 1,
                    "launch_blocking": False,
                    "scope": "resident_prepared_projection_only",
                },
            }
            for key in native.expected_keys(stage)
        ],
    }


@pytest.mark.parametrize("failure", [None, "fused", "timing", "runtime", "hash"])
def test_short_model_preflight_never_runs_simulator_or_rebuild(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(native, "require_hardware_runtime", lambda: None)
    monkeypatch.setattr(native, "library_evidence", lambda p: {"sha256": "a" * 64})
    calls = []

    def run_step(args, step, directory, digest):
        calls.append(step["stage"])
        assert args.warmups == 3 and args.repeats == 10 and args.timeout == 600
        assert step["probe"] == "0:0" and digest == "a" * 64
        evidence = child_evidence(step["stage"])
        if failure == "runtime":
            evidence.pop("hardware_runtime")
        elif failure == "hash":
            evidence["library"]["sha256"] = "b" * 64
        (directory / f"{step['stage']}.json").write_text(json.dumps(evidence))
        return {"stage": step["stage"], "passed": step["stage"] != failure}

    monkeypatch.setattr(suite, "run_step", run_step)
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            native.run_model_preflight(Path("/lib.so"), tmp_path, 4, tmp_path / "preflight")
    else:
        receipt = native.run_model_preflight(Path("/lib.so"), tmp_path, 4, tmp_path / "preflight")
        assert native.checked_model_preflight(Path("/lib.so"), receipt)["sha256"] == "a" * 64
        (receipt.parent / "timing.json").write_text(json.dumps({**child_evidence("timing"), "extra": True}))
        with pytest.raises(ValueError, match="changed"):
            native.checked_model_preflight(Path("/lib.so"), receipt)
    report = json.loads((tmp_path / "preflight/preflight.json").read_text())
    assert report["status"] == ("failed" if failure else "passed")
    assert calls == (["fused"] if failure == "fused" else ["fused", "timing"])
    assert len(native.expected_keys("fused")) + len(native.expected_keys("timing")) == 34


@pytest.mark.parametrize("source", ["mapped", "config", "preload", "clean"])
def test_hardware_preflight_rejects_simulator_environment(monkeypatch, source):
    monkeypatch.setattr(native.sys, "platform", "linux")
    monkeypatch.setattr(
        Path, "read_text", lambda p: "/sim/libruntime_camodel.so" if source == "mapped" else "/cann/libruntime.so"
    )
    monkeypatch.delenv("CAMODEL_CONFIG_PATH", raising=False)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    if source == "config":
        monkeypatch.setenv("CAMODEL_CONFIG_PATH", "/private/config")
    elif source == "preload":
        monkeypatch.setenv("LD_PRELOAD", "/sim/libruntime_camodel.so")
    if source == "clean":
        assert native.require_hardware_runtime() == {"checked": True, "simulator_runtime_paths": []}
    else:
        with pytest.raises(RuntimeError, match="Simulator"):
            native.require_hardware_runtime()


@pytest.mark.parametrize("identity", ["right", "wrong", "malformed"])
def test_worker_pins_library_before_registration(tmp_path, monkeypatch, identity):
    path = tmp_path / "lib.so"
    path.write_bytes(b"mock library")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    calls = []
    monkeypatch.setattr(wrapper, "load_library", lambda p: calls.append(p))
    if identity == "right":
        assert wrapper.load_pinned_library(path, digest) == {"path": str(path.resolve()), "sha256": digest}
        assert calls == [path.resolve()]
    else:
        with pytest.raises(ValueError):
            wrapper.load_pinned_library(path, "b" * 64 if identity == "wrong" else "")
        assert calls == []


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("verbose_experts", [False, True])
def test_supervisor_preflight_precedes_model_and_failure_stops_loading(tmp_path, monkeypatch, failed, verbose_experts):
    library = tmp_path / "lib.so"
    library.touch()
    (tmp_path / "experts_vq_ascend_v2").mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "acceptance",
            "--stage",
            "model",
            "--model",
            str(tmp_path),
            "--execution-policy",
            "ascendc",
            "--ascendc-library",
            str(library),
            "--output-dir",
            str(tmp_path / "out"),
            *(["--verbose-experts"] if verbose_experts else []),
        ],
    )
    order = []

    def preflight(lib, model, physical, directory, timeout):
        order.append("preflight")
        assert lib == library and model == tmp_path and physical == 4 and timeout == 600
        if failed:
            raise RuntimeError("NPU regression failed")
        return directory / "preflight.json"

    def child(command, **kwargs):
        order.append("model")
        assert command[command.index("--execution-policy") + 1] == "ascendc"
        assert command[command.index("--ascendc-library") + 1] == str(library)
        assert "--ascendc-preflight" in command
        assert ("--verbose-experts" in command) is verbose_experts
        assert kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "4"
        # An incomplete model must still fail; never promote the preflight.
        return NS(returncode=1)

    monkeypatch.setattr(native, "run_model_preflight", preflight)
    monkeypatch.setattr(acceptance.subprocess, "run", child)
    assert acceptance.main() == 1
    assert order == (["preflight"] if failed else ["preflight", "model"])


@pytest.mark.parametrize("stage", ["expert", "moe"])
def test_expert_verbosity_is_model_only(tmp_path, monkeypatch, stage):
    monkeypatch.setattr(sys, "argv", ["acceptance", "--model", str(tmp_path), "--stage", stage, "--verbose-experts"])
    with pytest.raises(SystemExit):
        acceptance.main()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("extra", [[], ["--stage", "moe"], ["--stage", "model", "--baseline-report", "/old"]])
def test_native_supervisor_refuses_implicit_or_wrong_stage(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["acceptance", "--model", str(tmp_path), "--execution-policy", "ascendc", *extra])
    with pytest.raises(SystemExit):
        acceptance.main()
    assert not list(tmp_path.iterdir())


def test_native_model_summary_separates_execution_from_isa_and_quality():
    summary = {
        "stage": "model",
        "status": "passed",
        "execution_policy": "ascendc",
        "probes": ["full_model"],
        "results": [
            {
                "passed": True,
                "records": [
                    {
                        "type": "VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS",
                        "data": {
                            "native_fp8_dot": None,
                            "ascendc_model_execution_verified": True,
                        },
                    }
                ],
            }
        ],
    }
    text = acceptance.format_compact_summary(summary)
    assert "ASCENDC_MODEL_EXECUTION_VERIFIED=True" in text
    assert "NATIVE_FP8_DOT=unknown" in text
    assert "QUALITY_VERIFIED=False" in text and "SERVING_VERIFIED=False" in text
    summary["results"][0]["passed"] = False
    assert "ASCENDC_MODEL_EXECUTION_VERIFIED=False" in acceptance.format_compact_summary(summary)
