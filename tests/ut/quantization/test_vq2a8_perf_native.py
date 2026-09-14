# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU protocol tests; fake traces/engines below do NOT verify NPU execution."""

import ast
import copy
import csv
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tools import accept_vq2a8_release as release
from tools import profile_vq2a8_ascendc as profile
from tools import vq2a8_native_review as native
from tools import vq2a8_perf_report as perf
from vllm_ascend.quantization import vq2a8_execution as execution


def args(**updates):
    values = dict(
        cases="10:4,32:32,96:32",
        warmups=2,
        repeats=5,
        simulator_timeout_minutes=5,
        timeout_seconds=7200,
        physical_npu=0,
        cache_budget_gib=0.0,
        cache_reserve_gib=16.0,
    )
    return NS(**(values | updates))


def test_plan_only_work_packages_two_four_no_speed_target():
    result = release.plan(args())
    assert result["work_packages"] == [2, 4] and result["performance_target_met"] is None
    assert result["cases"] == [(10, 4), (32, 32), (96, 32)]
    assert "HTTP serving" in result["not_requested"]


@pytest.mark.parametrize(
    "update",
    [
        dict(cases="512:32"),
        dict(cases="10:4,10:4"),
        dict(cases="0:4"),
        dict(warmups=1),
        dict(repeats=4),
        dict(simulator_timeout_minutes=45),
        dict(cache_budget_gib=float("nan")),
        dict(cache_reserve_gib=0),
    ],
)
def test_unsupported_or_incomplete_plan_rejected(update):
    with pytest.raises(ValueError):
        release.plan(args(**update))


def test_offline_token_timing_not_generation_divided_by_decode_count():
    result = perf.token_metrics([2, 2.5, 3.5, 4], 4.2, 4)
    assert result["ttft_s"] == 2
    assert result["tpot_s"] == pytest.approx(2 / 3)
    assert result["output_tokens_per_s"] == pytest.approx(4 / 4.2)
    assert result["decode_intervals_s"] == [0.5, 1, 0.5]


@pytest.mark.parametrize(
    "ready,elapsed", [([1], 2), ([1, 3, 2, 4], 5), ([1, 2, 3, 4], 3), ([1, 2, 3, float("nan")], 5), ([1, 1, 1, 1], 5)]
)
def test_bad_token_observations_rejected(ready, elapsed):
    with pytest.raises(ValueError):
        perf.token_metrics(ready, elapsed, 4)


def samples():
    return [
        dict(
            case="p10-o4",
            variant=variant,
            kind=kind,
            repeat=i,
            finite=True,
            tokens_exact=True,
            tokens=[1, 2, 3, 4],
            forwards=4,
            token_ready_s=[1, 2, 3, 4],
            ttft_s=1.0,
            tpot_s=1.0,
            e2e_s=4.0,
            output_tokens_per_s=1.0,
            device_span_ms=3900.0,
            cache_delta=dict(loads=0, hits=10, evictions=0),
        )
        for variant in ("baseline", "compact")
        for kind, count in (("warmup", 2), ("measured", 5))
        for i in range(count)
    ]


def test_complete_measurement_has_no_slo_or_quality_claim():
    result = perf.summarize_performance(
        samples(), [(10, 4)], 5, 2, [dict(case="p10-o4", logits_exact=True, tokens_exact=True)]
    )
    assert result["performance_measurement_verified"] is True
    assert result["performance_target_met"] is None
    assert result["groups"][0]["metrics"]["e2e_s"]["tail_sample_warning"] is True
    assert result["quality_verified"] is result["serving_verified"] is False
    assert result["ratios"][0]["warm_resident_comparison"] is True


@pytest.mark.parametrize("mode", ["missing", "nonfinite", "tokens", "timing", "forwards", "regression"])
def test_missing_or_failed_samples_cannot_pass(mode):
    rows, regressions = samples(), [dict(case="p10-o4", logits_exact=True, tokens_exact=True)]
    if mode == "missing":
        rows.pop()
    elif mode == "nonfinite":
        rows[-1]["finite"] = False
    elif mode == "tokens":
        rows[-1]["tokens_exact"] = False
    elif mode == "forwards":
        rows[-1]["forwards"] = 0
    elif mode == "timing":
        rows[-1]["e2e_s"] = float("nan")
    else:
        regressions = []
    with pytest.raises(ValueError):
        perf.summarize_performance(rows, [(10, 4)], 5, 2, regressions)


def test_cache_churn_not_called_resident_compute_speedup():
    rows = samples()
    rows[-1]["cache_delta"]["loads"] = 2
    result = perf.summarize_performance(
        rows, [(10, 4)], 5, 2, [dict(case="p10-o4", logits_exact=True, tokens_exact=True)]
    )
    assert result["ratios"][0]["warm_resident_comparison"] is False


def test_summary_log_keeps_disk_protocol_but_suppresses_layer_spam(tmp_path):
    stream = io.StringIO()
    logger = release.SummaryChildLog(tmp_path / "child.log", "test", console=stream)
    logger._emit("MODEL layer=23 stage=expert_done\nPERF_SAM")
    logger._emit('PLE {"e2e_s":1}\n')
    assert stream.getvalue() == 'PERF_SAMPLE {"e2e_s":1}\n'


def test_completed_stage_resume_requires_unchanged_bytes_and_identity(tmp_path):
    (tmp_path / "summary.json").write_text('{"status":"PASS"}')
    stage = dict(status="PASS", identity_sha256="a", directory=str(tmp_path), artifacts=release.tree_hashes(tmp_path))
    assert release.reusable(stage, "a")
    assert not release.reusable(stage, "b")
    (tmp_path / "summary.json").write_text('{"status":"FAIL"}')
    assert not release.reusable(stage, "a")
    stage["status"] = "FAIL"
    assert not release.reusable(stage, "a")


def test_timeout_cleans_only_owned_group(tmp_path, monkeypatch):
    class Child:
        pid, returncode = 12345, -9

        def wait(self, timeout=None):
            if timeout:
                raise subprocess.TimeoutExpired("test", timeout)

    def spawn(command, **kwargs):
        assert kwargs["start_new_session"] is True
        return Child()

    killed = []
    monkeypatch.setattr(release.subprocess, "Popen", spawn)
    monkeypatch.setattr(release.os, "killpg", lambda pid, sig: killed.append(pid), raising=False)
    monkeypatch.setattr(release.signal, "SIGKILL", 9, raising=False)
    result = release.supervise(["test"], tmp_path / "test.log", {}, 1)
    assert result["timeout"] is True and result["exit"] == -9 and killed == [12345]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["instr", "addr", "pipe", "call_count", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def fake_traces(root):
    (root / "kernel.cpp").write_text("// synthetic test fixture, not device code\n")
    for kind in ("fused", "grouped"):
        directory = root / f"native-{kind}"
        csv_path = directory / "core.csv"
        write_csv(
            csv_path, [dict(instr="MMAD", pipe="CUBE", addr="0x1234", call_count="4", detail="dtype:E4M3E4M3XD:0")]
        )
        profile.write_json(
            directory / "summary.json",
            {
                "status": "collected_review_pending",
                "profiler": {"exit": 0, "timeout": False},
                "global_config_unchanged": True,
                "profiler_log_review": {"runtime_error_count": 0, "application_timeout_reported": False},
                "library": {
                    "sha256": "a",
                    "build": {"source_sha256": {"kernel.cpp": profile.digest(root / "kernel.cpp")}},
                },
                "application": {
                    "status": "completed",
                    "loaded_library": {"sha256": "a"},
                    "library_unchanged": True,
                    "kernel_prefix": f"vq2a8_ascendc_{kind}",
                    "logical_projections": 2 if kind == "grouped" else 1,
                    "output_shapes": [[17, 32], [1, 32]] if kind == "grouped" else [[17, 32]],
                },
                "instruction_csv": [{"path": str(csv_path), "sha256": profile.digest(csv_path)}],
            },
        )


def test_fp8_operands_observed_is_not_dataflow_verification(tmp_path):
    fake_traces(tmp_path)
    result = native.index_native_reports(tmp_path, "a")
    assert result["native_fp8_instruction_observed"] is True
    assert result["native_instruction_verified"] is result["on_chip_decode_verified"] is False
    assert result["status"] == "REVIEW_REQUIRED"


@pytest.mark.parametrize(
    "instr,pipe,dtype,count",
    [
        ("MADD", "SCALAR", "E4M3E4M3", "1"),
        ("MMAD", "CUBE", "F16F16", "1"),
        ("MMAD", "CUBE", "E4M3E4M3", "0"),
        ("MMAD", "SCALAR", "E4M3E4M3", "1"),
    ],
)
def test_fp8_false_positive_rejected(tmp_path, instr, pipe, dtype, count):
    path = tmp_path / "trace.csv"
    write_csv(path, [dict(instr=instr, pipe=pipe, detail="dtype:" + dtype, call_count=count)])
    assert native.scan_instructions(path)["fp8_mmad_calls"] == 0


@pytest.mark.parametrize("mode", ["library", "timeout", "missing_grouped", "tampered"])
def test_old_incomplete_or_changed_trace_not_approved(tmp_path, mode):
    fake_traces(tmp_path)
    path = tmp_path / "native-grouped/summary.json"
    report = json.loads(path.read_text())
    if mode == "library":
        report["library"]["sha256"] = "other"
    elif mode == "timeout":
        report["status"] = "incomplete_review_pending"
    elif mode == "missing_grouped":
        report["application"]["kernel_prefix"] = "vq2a8_ascendc_fused"
    else:
        (tmp_path / "native-grouped/core.csv").write_text("changed")
    profile.write_json(path, report)
    if mode == "tampered":
        with pytest.raises(ValueError, match="changed"):
            native.index_native_reports(tmp_path, "a")
    else:
        assert native.index_native_reports(tmp_path, "a")["native_fp8_instruction_observed"] is False


def test_human_review_requires_every_kernel_claim_and_exact_artifacts(tmp_path):
    fake_traces(tmp_path)
    index = native.index_native_reports(tmp_path, "a")
    review = {"library_sha256": "a", "reviewer": "test fixture, not a real review", "kernels": {}}
    for entry in index["kernels"]:
        path, sha = next(iter(entry["files"].items()))
        review["kernels"][entry["kernel"]] = {
            claim: dict(
                accepted=True,
                explanation="CPU test fixture only",
                references=[
                    dict(path=path, sha256=sha, location="line 2, mock PC"),
                    dict(path="kernel.cpp", sha256=index["kernel_source_sha256"], location="mock source"),
                ],
            )
            for claim in native.REVIEW_CLAIMS
        }
    assert (
        native.apply_review(index, review, tmp_path)["verification_method"] == "explicit_hash_bound_human_attestation"
    )
    for claim in native.REVIEW_CLAIMS:
        bad = copy.deepcopy(review)
        bad["kernels"]["grouped"].pop(claim)
        with pytest.raises(ValueError):
            native.apply_review(index, bad, tmp_path)
    review["library_sha256"] = "old"
    with pytest.raises(ValueError):
        native.apply_review(index, review, tmp_path)


def test_grouped_trace_command_and_geometry_include_both_halves_and_k_loop(tmp_path):
    options = NS(projection_path="grouped", soc="Ascend950DT_9574", timeout_minutes=5, library=tmp_path / "lib.so")
    command = profile.profiler_command("msprof", options, tmp_path, "a")
    assert "--kernel-name=vq2a8_ascendc_grouped" in command
    assert "--soc-version=Ascend950DT_9574" in command
    assert "--launch-count=1" in command
    assert profile.probe_shapes(options) == [(17, 32, 512, 3), (1, 32, 512, 3)]


def test_measurement_removes_only_timing_fences(monkeypatch):
    layer = execution.CachedVQ2TP1MoE.__new__(execution.CachedVQ2TP1MoE)
    layer.device = torch.device("cpu")
    calls = []
    monkeypatch.setattr(execution, "synchronize_execution", lambda device: calls.append("sync"))
    layer._timing_sync()
    layer.measurement_mode = True
    layer._timing_sync()
    layer._forward = lambda hidden, ids: hidden
    value = torch.ones(1)
    assert layer.forward(value) is value
    assert calls == ["sync"]
    tree = ast.parse(Path(execution.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CachedVQ2TP1MoE")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_expert")
    source = ast.unparse(method)
    assert source.index("synchronize_execution(self.device)") < source.index("self._cache.popitem")
    assert "non_blocking=False" in source  # no unproven asynchronous H2D ownership


def model_probe(monkeypatch):
    source = release.REPO / "vllm_ascend/patch/worker/vq2a8_offline_model.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VQ2A8TP1OfflineForCausalLM")
    methods = {
        "configure_performance_probe",
        "performance_snapshot",
        "_retain_finite_flag",
        "forward",
        "compute_logits",
    }
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    cls.bases = [ast.Name(id="Parent", ctx=ast.Load())]

    class Parent:
        def forward(self, input_ids, *args):
            return input_ids.float()

        def compute_logits(self, hidden):
            return hidden

    npu = NS(
        synchronize=lambda: None,
        memory_allocated=lambda: 1,
        memory_reserved=lambda: 2,
        max_memory_allocated=lambda: 3,
        max_memory_reserved=lambda: 4,
        mem_get_info=lambda: (100, 200),
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    namespace = {
        "Parent": Parent,
        "torch": torch,
        "ROOT_FP8_POLICY": "online_fp8_sm90",
        "get_forward_context": lambda: NS(attn_metadata={"real": True}),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), "exec"), namespace)
    model = namespace[cls.name]()
    layer = execution.AscendCVQ2TP1MoE.__new__(execution.AscendCVQ2TP1MoE)
    layer.native_calls = layer.native_launches = layer.h2d_bytes = 0
    layer.timing = dict.fromkeys(
        ("host_load_validate_s", "host_read_s", "host_validate_s", "h2d_s", "prepare_s", "packed_projection_s"), 0
    )
    model.model = NS(offline_owner=NS(layers={0: layer}, cache_report=lambda: {}))
    model._offline_loaded, model._offline_root_mode, model._offline_trace = True, "bf16", False
    return model, layer


def test_model_probe_retains_nonfinite_detection_without_hot_cpu_read(monkeypatch):
    model, layer = model_probe(monkeypatch)
    model.configure_performance_probe(measurement=True, compact=True)
    assert layer.measurement_mode and layer._row_preparation.compact
    assert model.performance_snapshot()["finite"] is None
    decisions = []
    boolean = torch.Tensor.__bool__

    def count(tensor):
        decisions.append(tensor.numel())
        return boolean(tensor)

    monkeypatch.setattr(torch.Tensor, "__bool__", count)
    model.forward(torch.ones(1), None)
    model.compute_logits(torch.tensor([float("nan")]))
    assert decisions == []
    assert model.performance_snapshot()["finite"] is False
    assert decisions == [1]
    assert model.performance_snapshot()["forwards"] == 1
    model.configure_performance_probe(measurement=False, compact=False)
    assert not layer.measurement_mode and not layer._row_preparation.compact and not model._offline_trace


def test_full_orchestrator_keeps_missing_isa_review_pending_and_reuses_perf(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    source = repo / "csrc/vq2a8_ascendc"
    source.mkdir(parents=True)
    for name in ("kernel.cpp", "layout.h", "launch.h", "torch_binding.cpp", "CMakeLists.txt"):
        (source / name).write_text("test fixture")
    model = tmp_path / "model"
    model.mkdir()
    library = tmp_path / "lib.so"
    library.write_bytes(b"fixture")
    options = args(
        model=model,
        library=library,
        plan_only=False,
        output_dir=tmp_path / "report",
        resume=None,
        review_json=None,
        cases="10:4",
    )
    lib = {"path": str(library), "sha256": "a", "build": {"cann": "/mock/cann"}}
    monkeypatch.setattr(release, "REPO", repo)
    monkeypatch.setattr(release.sys, "platform", "linux")
    monkeypatch.setattr(release, "library_evidence", lambda p: lib)
    monkeypatch.setattr(release, "fingerprint", lambda *a: {"library": lib})
    monkeypatch.setattr(release.subprocess, "run", lambda *a, **k: NS(stdout=b"fixture diff"))
    calls = []

    def child(command, log, environment, timeout):
        assert environment["ASCEND_LAUNCH_BLOCKING"] == "0"
        out = Path(command[command.index("--output-dir") + 1])
        out.mkdir()
        log.write_text("fixture only")
        kind = (
            command[command.index("--worker") + 1]
            if "--worker" in command
            else "profile"
            if "--projection-path" in command
            else "performance"
        )
        calls.append(kind)
        if kind == "environment":
            profile.write_json(
                out / "summary.json", dict(status="PASS", device_name="Ascend950DT_9574", device_properties="mock")
            )
        elif kind == "preflight":
            profile.write_json(out / "preflight.json", dict(status="passed"))
        elif kind == "performance":
            regressions = [dict(case="p10-o4", logits_exact=True, tokens_exact=True)]
            data = perf.summarize_performance(samples(), [(10, 4)], 5, 2, regressions)
            data.update(
                samples=samples(),
                cases=[[10, 4]],
                repeats=5,
                warmups=2,
                regressions=regressions,
                library=lib,
                library_unchanged=True,
            )
            profile.write_json(out / "summary.json", data)
        else:
            profile.write_json(out / "summary.json", dict(status="incomplete_review_pending"))
        return dict(exit=0, timeout=False)

    monkeypatch.setattr(release, "supervise", child)
    assert release.run(options) == 2  # not overall PASS merely because every subprocess exits zero
    report = json.loads((options.output_dir / "summary.json").read_text())
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["performance_measurement_verified"] is True
    assert report["quality"] == report["serving"] == "NOT_REQUESTED"
    assert calls == ["environment", "preflight", "performance", "binary", "profile", "profile"]
    options.resume, options.output_dir = options.output_dir, None
    calls.clear()
    assert release.run(options) == 2
    assert calls == ["environment", "binary", "profile", "profile"]  # successful costly stages are reused


def test_child_exit_zero_without_samples_never_passes(tmp_path):
    profile.write_json(tmp_path / "summary.json", {"status": "PASS", "library": {"sha256": "a"}})
    assert not release.performance_passed(tmp_path, "a")


def test_wrong_case_regression_cannot_fill_matrix():
    with pytest.raises(ValueError, match="Every case"):
        perf.summarize_performance(
            samples(), [(10, 4)], 5, 2, [dict(case="p32-o32", logits_exact=True, tokens_exact=True)]
        )


def test_child_cannot_shrink_requested_case_matrix(tmp_path):
    regressions = [dict(case="p10-o4", logits_exact=True, tokens_exact=True)]
    data = perf.summarize_performance(samples(), [(10, 4)], 5, 2, regressions)
    data.update(
        samples=samples(),
        cases=[[10, 4]],
        repeats=5,
        warmups=2,
        regressions=regressions,
        library={"sha256": "a"},
        library_unchanged=True,
    )
    profile.write_json(tmp_path / "summary.json", data)
    assert release.performance_passed(tmp_path, "a", release.plan(args(cases="10:4")))
    assert not release.performance_passed(tmp_path, "a", release.plan(args()))
