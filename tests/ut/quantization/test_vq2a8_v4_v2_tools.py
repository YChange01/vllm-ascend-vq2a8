# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU orchestration/oracle regressions; no CANN/NPU execution is claimed."""

import json
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import build_vq2a8_v4_v2 as build
from tools import diagnose_vq2a8_tp1_startup as startup
from tools import validate_vq2a8_v4_v2 as validate


def test_build_is_separate_and_does_not_install_or_start_model(tmp_path):
    args = build.parse_args(["--soc", "Ascend950DT_9582", "--build-dir", str(tmp_path / "new"), "--dry-run"])
    directory = build.validate_options(args)
    commands = build.cmake_commands(args, directory, "sdk/ascendc.cmake", "torch_npu", "torch/cmake")
    assert commands[1][commands[1].index("--target") + 1] == "vq2a8_ascendc_v4_v2"
    assert build.LIBRARY_NAME == "libvq2a8_ascendc_v4_v2.so"
    assert "-DSOC_VERSION=Ascend950DT_9582" in commands[0]
    flat = json.dumps(commands)
    assert "pip" not in flat and "vq2a8_ascendc_v3" not in flat
    assert not directory.exists()


@pytest.mark.parametrize("soc,jobs", [("Ascend950", 4), ("Ascend910B", 4), ("Ascend950DT_9582", 0)])
def test_build_rejects_ambiguous_target_or_jobs(tmp_path, soc, jobs):
    args = build.parse_args(["--soc", soc, "--jobs", str(jobs), "--build-dir", str(tmp_path / "new")])
    with pytest.raises(ValueError):
        build.validate_options(args)


@pytest.mark.parametrize("directory", [build.REPO, build.REPO.parent, build.SOURCE, build.SOURCE / "build"])
def test_build_never_uses_repo_or_native_source_as_output(directory):
    args = build.parse_args(["--soc", "Ascend950DT_9582", "--build-dir", str(directory)])
    with pytest.raises(ValueError, match="dedicated"):
        build.validate_options(args)


@pytest.mark.parametrize("old_file", ["libvq2a8_ascendc.so", "libvq2a8_ascendc_v2.so", "libvq2a8_ascendc_v3.so"])
def test_build_does_not_reuse_baseline_directory(tmp_path, old_file):
    (tmp_path / old_file).write_bytes(b"baseline")
    args = build.parse_args(["--soc", "Ascend950DT_9582", "--build-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="another backend"):
        build.validate_options(args)
    assert (tmp_path / old_file).read_bytes() == b"baseline"


def test_build_rejects_other_cmake_project(tmp_path):
    (tmp_path / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/other/project\n")
    args = build.parse_args(["--soc", "Ascend950DT_9582", "--build-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="another project"):
        build.validate_options(args)


def test_build_dry_run_cannot_claim_device_success(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(build, "source_hashes", lambda: {"kernel.cpp": "hash"})
    assert build.main(["--soc", "Ascend950DT_9582", "--build-dir", str(tmp_path / "out"), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "planned"
    assert not report["device_execution_verified"] and not report["model_execution_verified"]
    assert not report["performance_verified"]
    assert report["default_model_backend"] == "unchanged"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "phase,expected",
    [
        ("kernel", ["kernel"]),
        ("resident", ["kernel", "resident"]),
        ("lifetime", ["kernel", "resident", "lifetime"]),
        ("graph", ["kernel", "resident", "graph"]),
        ("all", ["kernel", "resident", "lifetime", "graph"]),
    ],
)
def test_phase_prerequisites_are_explicit(phase, expected):
    assert validate.phases_for(phase) == expected
    args = validate.parse_args(["--phase", phase])
    assert validate.validation_plan(args)["phases"] == expected
    assert args.launch_blocking == "0"


def test_default_probe_runs_kernel_only_and_preserves_card_one():
    args = validate.parse_args([])
    assert args.phase == "kernel" and args.physical_npu == 1
    assert args.library.name == build.LIBRARY_NAME
    plan = validate.validation_plan(args)
    assert plan["geometry"]["jobs"] == [1, 6]
    assert plan["geometry"]["k"] == [2048, 4096]
    for key in (
        "full_model_loaded",
        "device_execution_verified",
        "model_integration_verified",
        "performance_verified",
        "graph_verified",
        "serving_verified",
    ):
        assert plan[key] is False


@pytest.mark.parametrize(
    "extra",
    [
        ["--physical-npu", "-1"],
        ["--timeout-s", "0"],
        ["--expert", "1"],
        ["--expert", "-1:0"],
        ["--child", "--plan-only"],
        ["--phase", "lifetime", "--queue-iterations", "1"],
        ["--phase", "graph", "--queue-iterations", "2000"],
    ],
)
def test_bad_probe_arguments_rejected(extra):
    with pytest.raises(SystemExit):
        validate.parse_args(extra)


def test_child_command_propagates_real_fixture_and_bounded_options(tmp_path):
    args = validate.parse_args(
        [
            "--phase",
            "all",
            "--model",
            str(tmp_path / "model"),
            "--expert",
            "2:3",
            "--timeout-s",
            "41",
            "--queue-iterations",
            "1400",
            "--physical-npu",
            "1",
        ]
    )
    child = validate.child_command(args)
    assert "--child" in child and "--plan-only" not in child
    assert child[child.index("--timeout-s") + 1] == "41"
    assert child[child.index("--phase") + 1] == "all"
    assert child[child.index("--expert") + 1] == "2:3"
    assert child[child.index("--model") + 1] == str((tmp_path / "model").resolve())
    env = validate.child_environment(
        args, {"RANK": "3", "ASCEND_RT_VISIBLE_DEVICES": "7", "ASCEND_LAUNCH_BLOCKING": "1"}
    )
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "1" and env["ASCEND_LAUNCH_BLOCKING"] == "0"
    assert "RANK" not in env


def test_plan_only_never_queries_npu(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("plan-only must not start child or query an NPU")

    monkeypatch.setattr(validate.subprocess, "run", forbidden)
    monkeypatch.setattr(validate, "run_child", forbidden)
    assert validate.main(["--plan-only", "--phase", "all"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["phases"] == ["kernel", "resident", "lifetime", "graph"]
    assert report["device_execution_verified"] is False


def test_failed_kernel_never_constructs_resident_or_captures_graph(monkeypatch):
    calls = []

    def fail_kernel(*args):
        calls.append("kernel")
        raise RuntimeError("kernel failed")

    monkeypatch.setattr(validate, "run_kernel_checks", fail_kernel)
    for name in ("run_resident_checks", "run_lifetime_checks", "run_graph_checks"):
        monkeypatch.setattr(validate, name, lambda *args: pytest.fail("phase after kernel failure"))
    with pytest.raises(RuntimeError, match="kernel failed"):
        validate.run_phases(validate.parse_args(["--phase", "all"]), "cpu", None, None, None, {})
    assert calls == ["kernel"]


def test_failed_resident_never_starts_lifetime_or_graph(monkeypatch):
    monkeypatch.setattr(validate, "run_kernel_checks", lambda *args: ["kernel"])

    def fail_resident(*args):
        raise RuntimeError("resident failed")

    monkeypatch.setattr(validate, "run_resident_checks", fail_resident)
    monkeypatch.setattr(validate, "run_graph_checks", lambda *args: pytest.fail("captured after resident failure"))
    with pytest.raises(RuntimeError, match="resident failed"):
        validate.run_phases(validate.parse_args(["--phase", "graph"]), "cpu", None, None, None, {})


def test_stage_pass_requires_device_completion(capsys):
    seen = []
    with validate.stage_recorder("cpu_test", lambda: seen.append("sync"))("kernel"):
        seen.append("submitted")
    events = [json.loads(line.split(" ", 1)[1])["event"] for line in capsys.readouterr().out.splitlines()]
    assert events == ["BEGIN", "SUBMITTED", "PASS"]
    assert seen == ["submitted", "sync"]


def test_stage_sync_failure_is_not_pass(capsys):
    def fail():
        raise RuntimeError("device failure")

    with pytest.raises(RuntimeError, match="device failure"), validate.stage_recorder("cpu_test", fail)("kernel"):
        pass
    events = [json.loads(line.split(" ", 1)[1])["event"] for line in capsys.readouterr().out.splitlines()]
    assert events == ["BEGIN", "SUBMITTED", "FAIL"]


def test_parent_timeout_terminates_only_owned_child(monkeypatch, tmp_path):
    killed = []

    class Process:
        returncode = None

        def wait(self, timeout):
            assert timeout == 2
            raise startup.subprocess.TimeoutExpired("fake child", timeout)

    @contextmanager
    def live(*args):
        yield

    process = Process()
    monkeypatch.setattr(startup.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(startup, "LiveChildLog", live)
    monkeypatch.setattr(startup, "terminate_child", lambda child: killed.append(child) or True)
    result = validate.run_child(["fake"], {}, tmp_path / "child.log", 2)
    assert result["status"] == "TIMEOUT" and result["reaped"]
    assert killed == [process]


def test_native_oracle_catches_dtype_shape_and_numeric_errors():
    expected = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    validate.check_output(expected.clone(), expected, "oracle")
    for bad in (expected.float(), expected.reshape(-1), expected + 1, expected * float("nan")):
        with pytest.raises(AssertionError):
            validate.check_output(bad, expected, "oracle")


def test_invalid_ids_must_be_nan_and_never_report_valid():
    oracle = torch.tensor([2.0, 3.0], dtype=torch.bfloat16)
    output = torch.stack([oracle, torch.full_like(oracle, float("nan"))])
    valid = torch.tensor([1, 0], dtype=torch.int32)
    validate.check_routed((output, valid), [oracle, None], "routes")
    with pytest.raises(AssertionError, match="invalid route mask"):
        validate.check_routed((output, torch.ones_like(valid)), [oracle, None], "routes")
    with pytest.raises(AssertionError, match="must produce NaNs"):
        validate.check_routed((torch.zeros_like(output), torch.zeros_like(valid)), [None, None], "routes")


def test_synthetic_real_geometry_conversion_has_independent_oracle():
    fixture = validate.synthetic_fixture(2048, 1)
    payload = fixture["converted"]
    assert fixture["dense"].shape == (4096, 2048)
    assert payload["packed_zn"].shape == (128, 128, 16, 8)
    assert sorted(payload["activation_order"].tolist()) == list(range(2048))
    assert not torch.equal(payload["activation_order"], torch.arange(2048))
    prepared = validate.prepared_inputs(2048, 3, 7)
    expected = validate.projection_oracle(fixture, prepared)
    assert expected.shape == (3, 4096) and expected.dtype == torch.bfloat16
    inputs, oracle = validate.routed_inputs([fixture] * 3, (0, 2**32, 2), 1, 7, "cpu")
    assert inputs[0].shape == (3, 2048) and inputs[-1].dtype == torch.int64
    assert oracle[1] is None and oracle[0].shape == (4096,)


def test_graph_and_lifetime_tests_do_not_hide_fallback_or_deadlocks():
    source = Path(validate.__file__).read_text()
    assert "vq2a8_ascendc_v3" not in source
    assert "SetCustomHandler" not in source
    assert "graph.replay()" in source and "dst.copy_(src)" in source
    assert "torch.npu.graph(graph, stream=owner)" in source and "owner = torch.npu.Stream()" in source
    assert "range(7)" in source  # Valid replay after invalid IDs checks validity recovery.
    assert "lifetime_concurrent_release" in source and "executor.submit(list.clear, owners)" in source
    assert "2**63 - 1" in source and "-(2**63)" in source and "2**32" in source
    assert "model_integration_verified=False" in source and "performance_verified=False" in source


@pytest.mark.parametrize("rows", [1, 3, 32])
def test_real_reference_prepares_each_row_independently(rows):
    from vllm_ascend.quantization.vq2a8_reference import prepare_repacked_vq2a8_activation_reference

    k = 256
    hidden = ((torch.arange(rows * k).reshape(rows, k).float() % 31 - 15) / 16).bfloat16()
    payload = {
        "weight_scale": torch.ones(k),
        "weight_bias": torch.arange(k).float() / 512,
        "rht_sign": torch.ones(k, dtype=torch.int8),
    }
    actual = validate.prepare_real_rows(hidden, payload, SimpleNamespace(rht_block_size=128))
    assert [value.shape for value in actual] == [(rows, k), (rows,), (rows,)]
    for index in range(rows):
        expected = prepare_repacked_vq2a8_activation_reference(
            hidden[index : index + 1],
            payload["weight_scale"],
            payload["weight_bias"],
            payload["rht_sign"],
            128,
        )
        for batch, row in zip(actual, expected):
            assert torch.equal(batch[index : index + 1].view(torch.uint8), row.view(torch.uint8))


@pytest.mark.parametrize("corrupt", [False, True])
def test_bank_release_drops_owners_before_pressure_and_checks_real_values(monkeypatch, corrupt):
    # Small CPU test double exercises ownership and actual arithmetic; it is
    # explicitly NOT a simulation of the native kernel/allocator or a receipt.
    dense = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 8
    fixture = {
        "k": 8,
        "dense": dense,
        "converted": {name: dense if name == "packed_zn" else torch.ones(8) for name in validate.FIELDS},
    }
    owner_refs, bank_refs, input_refs, pressure_checks = [], [], [], []
    events = []

    class CpuBank:
        def __init__(self, *owners):
            self.owners = owners
            bank_refs.append(weakref.ref(self))
            owner_refs.extend(weakref.ref(tensor) for group in owners for tensor in group)

        def project(self, q, scale, bias, ids):
            input_refs.extend(weakref.ref(value) for value in (q, scale, bias, ids))
            output = (q.float() @ self.owners[0][0].T * scale[:, None] + bias[:, None]).bfloat16()
            if corrupt:
                output = output + 16
            return output, ids.eq(0).int()

    original_empty = torch.empty

    def checked_empty(*args, **kwargs):
        if args == (validate.PRESSURE_CHUNK_BYTES,):
            assert all(reference() is None for reference in bank_refs + owner_refs + input_refs)
            pressure_checks.append(True)
        return original_empty(*args, **kwargs)

    @contextmanager
    def stage(name):
        events.append(("BEGIN", name))
        yield
        events.append(("FENCE", name))

    monkeypatch.setattr(torch, "empty", checked_empty)
    if corrupt:
        with pytest.raises(AssertionError, match="oracle mismatch"):
            validate.run_bank_release_checks("cpu", CpuBank, fixture, stage)
        assert len(pressure_checks) == validate.PRESSURE_CHUNKS
    else:
        report = validate.run_bank_release_checks("cpu", CpuBank, fixture, stage)
        assert report["iterations"] == validate.BANK_RELEASE_ITERATIONS
        assert report["python_bank_and_input_owners_dropped_before_fence"]
        assert report["uploaded_python_payload_owners_dropped_before_fence"]
        assert len(pressure_checks) == validate.BANK_RELEASE_ITERATIONS * validate.PRESSURE_CHUNKS
        assert len([event for event in events if event[0] == "FENCE"]) == validate.BANK_RELEASE_ITERATIONS * 2
