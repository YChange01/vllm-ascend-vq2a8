# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
import subprocess
import sys

import pytest

from tools import accept_vq2a8_ascendc_v3 as accept
from tools import benchmark_vq2a8_ascendc_v3 as bench
from vllm_ascend.quantization.vq2a8_offline import offline_engine_options


def cli(tmp_path, tool):
    flags = ["--model", str(tmp_path / "model"), "--library", str(tmp_path / "library.so")]
    if tool is bench:
        return flags + [
            "--output-dir",
            str(tmp_path / "output"),
            "--preflight",
            str(tmp_path / "preflight.json"),
            "--v3-only",
        ]
    return flags + ["--benchmark"]


@pytest.mark.parametrize("tool", [accept, bench])
def test_v3_memory_tools_independent_defaults_and_cache_alias(tmp_path, tool):
    args = tool.parse_args(cli(tmp_path, tool))
    assert args.engine_memory_fraction == 0.98
    assert args.memory_fraction == 0.9
    config = bench.configuration(args)
    assert config["engine_memory_fraction"] == 0.98
    assert config["memory_fraction"] == 0.9
    alias = tool.parse_args(cli(tmp_path, tool) + ["--cache-memory-fraction", "1"])
    assert alias.memory_fraction == 1 and alias.engine_memory_fraction == 0.98


@pytest.mark.parametrize("tool", [accept, bench])
@pytest.mark.parametrize("flag", ["--engine-memory-fraction", "--memory-fraction", "--cache-memory-fraction"])
@pytest.mark.parametrize("value", ["0", "-0.1", "1.01", "nan", "inf", "-inf"])
def test_v3_memory_tools_reject_invalid_fractions(tmp_path, tool, flag, value):
    with pytest.raises(SystemExit):
        tool.parse_args(cli(tmp_path, tool) + [f"{flag}={value}"])


@pytest.mark.parametrize("v3_only", [False, True])
@pytest.mark.parametrize("engine_fraction,cache_fraction", [(0.98, 1.0), (1.0, 0.65)])
def test_v3_memory_tools_parent_children_and_actual_options(tmp_path, v3_only, engine_fraction, cache_fraction):
    flags = cli(tmp_path, accept) + [
        "--engine-memory-fraction",
        str(engine_fraction),
        "--memory-fraction",
        str(cache_fraction),
    ]
    args = accept.parse_args(flags + (["--v3-only"] if v3_only else []))
    steps = accept.commands(args, tmp_path / "output")
    children = [(name, command) for name, command in steps if name in ("v1-reference", "performance")]
    assert len(children) == (1 if v3_only else 2)
    for name, command in children:
        child = bench.parse_args(command[3:])
        assert child.engine_memory_fraction == engine_fraction
        assert child.memory_fraction == cache_fraction
        assert bench.configuration(child) == bench.configuration(args)
        library = {"path": str(child.library), "sha256": "a" * 64}
        options = bench.build_engine_options(child, library, offline_engine_options)
        assert options["gpu_memory_utilization"] == engine_fraction
        assert options["kv_cache_memory_bytes"] == 2**30
        offline = options["additional_config"]["vq2a8_offline"]
        assert offline["cache_memory_fraction"] == cache_fraction
        policy = "ascendc" if name == "v1-reference" else "ascendc_v3"
        assert offline["execution_policy"] == policy
        assert offline[f"{policy}_library"] == str(child.library)


@pytest.mark.parametrize("tool", [accept, bench])
def test_v3_memory_tools_plan_records_both_without_torch_or_device(tmp_path, tool):
    flags = cli(tmp_path, tool) + ["--plan-only", "--engine-memory-fraction", "0.98", "--memory-fraction", "1"]
    code = (
        "import runpy, sys; sys.modules['torch'] = None; sys.modules['torch_npu'] = None; "
        f"sys.argv = {[tool.__file__, *flags]!r}; runpy.run_path({tool.__file__!r}, run_name='__main__')"
    )
    completed = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True, check=True)
    plan = json.loads(completed.stdout)
    assert plan["scope"] == "plan_only_no_device_execution"
    assert plan["configuration"]["engine_memory_fraction"] == 0.98
    assert plan["configuration"]["memory_fraction"] == 1
    assert not (tmp_path / "output").exists()


def test_v3_memory_tools_startup_diagnostic_observes_without_clamping(tmp_path):
    args = bench.parse_args(cli(tmp_path, bench) + ["--memory-fraction", "1"])
    options = bench.build_engine_options(args, {"path": str(args.library), "sha256": "b" * 64}, offline_engine_options)
    original = copy.deepcopy(options)
    total = 80 * 2**30
    for free, fits in ((79 * 2**30, True), (77 * 2**30, False)):
        record = bench.engine_memory_diagnostic([free, total], options)
        assert record["engine_memory_fraction"] == 0.98 and record["cache_memory_fraction"] == 1
        assert record["kv_cache_memory_bytes"] == 2**30
        assert record["requested_bytes"] == int(total * 0.98)
        assert record["observed_free_bytes"] == free and record["observed_total_bytes"] == total
        assert record["observed_requested_fits_free"] is fits
        assert record["worker_check_authoritative"] is True
        assert options == original


@pytest.fixture
def configuration_evidence(tmp_path, monkeypatch):
    args = bench.parse_args(cli(tmp_path, bench) + ["--engine-memory-fraction", "0.98", "--memory-fraction", "1"])
    args.library.write_bytes(b"test-v3-library")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(bench, "model_identity", lambda _model: {"model": "test"})
    monkeypatch.setattr(bench, "python_source_hashes", lambda: {"source": "test"})
    report = dict(
        schema_version=bench.SCHEMA_VERSION,
        status="PASS",
        mode="performance",
        implementation="ascendc_v3",
        scope=bench.SCOPE,
        model={"model": "test"},
        python_source_sha256={"source": "test"},
        configuration=bench.configuration(args),
        cases=[[10, 4]],
        repeat_exact=True,
        full_model_graph_verified=False,
        v3_only=True,
        quality_verified=False,
        physical_npu="0",
        soc="Ascend950-test",
        library={"path": str(args.library.resolve()), "sha256": bench.sha256(args.library)},
    )
    return args, report


@pytest.mark.parametrize("field", ["engine_memory_fraction", "memory_fraction"])
@pytest.mark.parametrize("tamper", ["report", "request", "missing"])
def test_v3_memory_tools_verify_report_binds_each_ratio(configuration_evidence, monkeypatch, field, tamper):
    args, report = configuration_evidence

    class PassedConfigurationGate(Exception):
        pass

    def preflight(*_args):
        raise PassedConfigurationGate

    monkeypatch.setattr(bench, "checked_model_preflight", preflight)
    with pytest.raises(PassedConfigurationGate):
        bench.verify_report(report, args)
    if tamper == "report":
        report["configuration"][field] = 0.5
    elif tamper == "request":
        setattr(args, field, 0.5)
    else:
        del report["configuration"][field]
    with pytest.raises(ValueError, match="configuration"):
        bench.verify_report(report, args)


@pytest.mark.parametrize("field", ["engine_memory_fraction", "memory_fraction"])
def test_v3_memory_tools_reference_reuse_binds_both_ratios(configuration_evidence, field):
    args, report = configuration_evidence
    report.update(mode="reference", implementation="ascendc")
    assert bench.validate_reference(report, args, [(10, 4)]) is report
    report["configuration"][field] = 0.5
    with pytest.raises(ValueError, match="configuration"):
        bench.validate_reference(report, args, [(10, 4)])


def test_v3_memory_tools_accept_failure_report_retains_memory_config(tmp_path, monkeypatch, capsys):
    args = accept.parse_args(
        cli(tmp_path, accept) + ["--output-dir", str(tmp_path / "output"), "--memory-fraction", "1"]
    )
    monkeypatch.setattr(accept, "parse_args", lambda: args)
    monkeypatch.setattr(accept.platform, "system", lambda: "Linux")
    monkeypatch.setattr(accept, "acceptance_environment", lambda *_: {})
    monkeypatch.setattr(
        accept,
        "supervise",
        lambda command, log, *_: dict(exit=1, timeout=False, elapsed_s=0, log=str(log), command=command),
    )
    assert accept.main() == 1
    report = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert report["configuration"]["engine_memory_fraction"] == 0.98
    assert report["configuration"]["memory_fraction"] == 1
    assert "VQ2A8_V3_MEMORY_CONFIG=" in capsys.readouterr().out
