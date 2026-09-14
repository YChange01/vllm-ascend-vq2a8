# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
import sys
from types import SimpleNamespace as NS

import pytest
import torch
from safetensors.torch import save_file

from tools import accept_vq2a8_ascendc_v2 as accept
from tools import validate_vq2a8_ascendc_v2 as gate
from tools.validate_vq2a8_phase4_kernel import prepare_rows
from vllm_ascend.quantization import vq2a8_ascendc_v2 as v2
from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation


def arguments(tmp_path):
    return NS(
        model=tmp_path / "model",
        library=None,
        build_dir=tmp_path / "build",
        soc="Ascend950DT_9574",
        jobs=4,
        preflight_only=False,
        preset="batched",
    )


def test_ordered_supervisor_build_preflight_model_commands_use_only_v2(tmp_path):
    steps = accept.commands(arguments(tmp_path), tmp_path / "report")
    assert [name for name, _ in steps] == ["environment", "build", "preflight", "model"]
    assert all(cmd[:2] == [sys.executable, "-u"] for _, cmd in steps)
    model = dict(steps)["model"]
    assert model[model.index("--execution-policy") + 1] == "ascendc_v2"
    assert model[model.index("--ascendc-v2-library") + 1].endswith("libvq2a8_ascendc_v2.so")
    assert model[model.index("--ascendc-v2-preflight") + 1] == str(tmp_path / "report/preflight.json")
    assert model[model.index("--ascendc-v2-preset") + 1] == "batched"
    assert model[model.index("--root-linear-mode") + 1] == "bf16"
    assert "--ascendc-library" not in model and "--ascendc-preflight" not in model


def test_existing_library_skips_only_build_not_preflight(tmp_path):
    args = arguments(tmp_path)
    args.library = tmp_path / "selected/libvq2a8_ascendc_v2.so"
    assert [name for name, _ in accept.commands(args, tmp_path / "report")] == ["environment", "preflight", "model"]
    args.preflight_only = True
    assert [name for name, _ in accept.commands(args, tmp_path / "report")] == ["environment", "preflight"]


def test_opt_in_performance_stage_follows_model_with_v2_pinned_receipt(tmp_path):
    args = arguments(tmp_path)
    args.benchmark, args.cases, args.warmups, args.repeats = True, "10:4,32:32", 2, 5
    steps = accept.commands(args, tmp_path / "report")
    assert [name for name, _ in steps] == ["environment", "build", "preflight", "model", "performance"]
    command = dict(steps)["performance"]
    assert command[2].endswith("benchmark_vq2a8_ascendc_v2.py")
    assert command[command.index("--library") + 1].endswith("libvq2a8_ascendc_v2.so")
    assert command[command.index("--preflight") + 1] == str(tmp_path / "report/preflight.json")
    assert command[command.index("--cases") + 1] == "10:4,32:32"
    assert command[command.index("--preset") + 1] == "batched"
    assert command[command.index("--warmups") + 1] == "2"
    assert command[command.index("--repeats") + 1] == "5"
    args.preflight_only = True
    with pytest.raises(ValueError, match="full-model"):
        accept.commands(args, tmp_path / "report")


def test_only_performance_environment_disables_debug_blocking_without_mutating_parent():
    original = {"ASCEND_LAUNCH_BLOCKING": "1", "ASCEND_RT_VISIBLE_DEVICES": "4", "PYTHONPATH": "repository"}
    assert accept.stage_environment(original, "model") == original
    perf = accept.stage_environment(original, "performance")
    assert perf == {**original, "ASCEND_LAUNCH_BLOCKING": "0"}
    assert original["ASCEND_LAUNCH_BLOCKING"] == "1"


def test_supervisor_rechecks_performance_identity_and_raw_report(tmp_path, monkeypatch):
    from tools import benchmark_vq2a8_ascendc_v2 as benchmark

    args = arguments(tmp_path)
    args.physical_npu, args.cases, args.warmups, args.repeats = 4, "10:4", 2, 5
    library = tmp_path / "libvq2a8_ascendc_v2.so"
    library.write_bytes(b"non-executable test fixture")
    directory = tmp_path / "performance"
    directory.mkdir()
    raw = {"status": "PASS", "measurement_environment": {"physical_npu": "4"}}
    (directory / "summary.json").write_text(json.dumps(raw))
    calls = []

    def verify(*inputs):
        calls.append(inputs)
        return inputs[0]

    monkeypatch.setattr(benchmark, "verify_report", verify)
    assert accept.verify_performance_report(tmp_path, library, args) == raw
    assert calls == [
        (
            raw,
            {"path": str(library), "sha256": gate.sha256(library), "abi_version": 1},
            args.model,
            [(10, 4)],
            "batched",
            2,
            5,
        )
    ]
    raw["measurement_environment"]["physical_npu"] = "0"
    (directory / "summary.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="physical NPU"):
        accept.verify_performance_report(tmp_path, library, args)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "extra",
    [
        ["--benchmark", "--preflight-only"],
        ["--warmups", "1"],
        ["--repeats", "4"],
        ["--cases", "96:64"],
        ["--cases", "10:1"],
        ["--cases", "10:4,10:4"],
        ["--cases", "10:4:2"],
    ],
)
def test_supervisor_rejects_invalid_benchmark_requests_before_writes(tmp_path, monkeypatch, extra):
    output = tmp_path / "absent"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accept",
            "--model",
            str(tmp_path),
            "--soc",
            "Ascend950DT_9574",
            "--output-dir",
            str(output),
            "--plan-only",
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as exc:
        accept.main()
    assert exc.value.code == 2 and not output.exists()


def test_plan_only_does_not_create_reports_or_touch_runtime(tmp_path, monkeypatch, capsys):
    output = tmp_path / "not-created"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accept",
            "--model",
            str(tmp_path / "not-a-model"),
            "--soc",
            "Ascend950DT_9574",
            "--output-dir",
            str(output),
            "--plan-only",
        ],
    )
    assert accept.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["scope"] == "plan_only_no_device_execution" and not output.exists()


def receipt_fixture(tmp_path, monkeypatch):
    library = tmp_path / "libvq2a8_ascendc_v2.so"
    library.write_bytes(b"fake-not-executable")
    identity = {"path": str(library), "sha256": gate.sha256(library), "abi_version": 1}
    source_identity = {"test-source": "1" * 64}
    model_identity = {"path": str(tmp_path / "model"), "config_sha256": "2" * 64, "artifact_manifest_sha256": "3" * 64}
    monkeypatch.setattr(v2, "validate_build_manifest", lambda path, digest: identity)
    monkeypatch.setattr(gate, "python_source_hashes", lambda: source_identity)
    monkeypatch.setattr(gate, "model_identity", lambda model: model_identity)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    records = {}
    for key in gate.expected_cases():
        rows = (
            next(r for r in gate.SYNTHETIC_ROWS if f":g{len(r)}:m{r[0]}" in key)
            if key.startswith("synthetic:")
            else (int(key.split(":")[2][1:]),)
        )
        records[key] = {
            "passed": True,
            "repeat_exact": True,
            "grouped_exact": True,
            "current_stream_exact": True,
            "oracle_exact_required": key.startswith("synthetic:") or key.endswith(":zero"),
            "oracle": [
                {
                    "allclose": True,
                    "mismatch_count": 0,
                    "numel": m * 4096,
                    "max_abs_error": 0.0,
                    "relative_l2_error": 0.0,
                    "relative_l2_limit": 0.03,
                }
                for m in rows
            ],
        }
    report = {
        "schema_version": 1,
        "status": "passed",
        "implementation": "ascendc_v2",
        "device_execution_verified": True,
        "physical_runtime": True,
        "physical_npu": "0",
        "soc": "Ascend950DT_9574",
        "library": identity,
        "model": model_identity,
        "python_source_sha256": source_identity,
        "cases": records,
    }
    return library, identity, report


def test_complete_receipt_is_required_by_model_child(tmp_path, monkeypatch):
    library, identity, report = receipt_fixture(tmp_path, monkeypatch)
    assert len(report["cases"]) == 24
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(report))
    assert gate.checked_model_preflight(library, path, tmp_path / "model") == identity


@pytest.mark.parametrize(
    "bad",
    [
        "status",
        "simulator",
        "source",
        "library",
        "model",
        "device",
        "soc",
        "missing_case",
        "repeat",
        "grouped",
        "stream",
        "oracle",
        "oracle_error",
        "oracle_count",
        "oracle_numel",
        "exact",
        "l2",
    ],
)
def test_receipt_rejects_stale_partial_or_inconsistent_results(tmp_path, monkeypatch, bad):
    library, identity, original = receipt_fixture(tmp_path, monkeypatch)
    report = copy.deepcopy(original)
    synthetic = report["cases"]["synthetic:k2048:g1:m1"]
    if bad == "status":
        report["status"] = "running"
    elif bad == "simulator":
        report["physical_runtime"] = False
    elif bad == "source":
        report["python_source_sha256"]["test-source"] = "4" * 64
    elif bad == "library":
        report["library"]["sha256"] = "5" * 64
    elif bad == "model":
        report["model"]["artifact_manifest_sha256"] = "6" * 64
    elif bad == "device":
        report["physical_npu"] = "4"
    elif bad == "soc":
        report["soc"] = "simulator"
    elif bad == "missing_case":
        report["cases"].pop("real:down:m32:impulse")
    elif bad == "repeat":
        synthetic["repeat_exact"] = False
    elif bad == "grouped":
        synthetic["grouped_exact"] = False
    elif bad == "stream":
        synthetic["current_stream_exact"] = False
    elif bad == "oracle":
        synthetic["oracle"][0]["allclose"] = False
    elif bad == "oracle_error":
        synthetic["oracle"][0]["max_abs_error"] = 0.1
    elif bad == "oracle_count":
        synthetic["oracle"].append(copy.deepcopy(synthetic["oracle"][0]))
    elif bad == "oracle_numel":
        synthetic["oracle"][0]["numel"] = 1
    elif bad == "exact":
        synthetic["oracle_exact_required"] = False
    elif bad == "l2":
        report["cases"]["real:down:m32:impulse"]["oracle"][0]["relative_l2_error"] = 1.0
    with pytest.raises(ValueError):
        gate.validate_receipt(report, identity, tmp_path / "model", "0")
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        gate.checked_model_preflight(library, path, tmp_path / "model")


def result_fixture(tmp_path):
    library = tmp_path / "libvq2a8_ascendc_v2.so"
    library.write_bytes(b"not-executable")
    directory = tmp_path / "model-evidence"
    directory.mkdir()
    gate_record = {
        "expert_execution_policy": "ascendc_v2",
        "ascendc_v2_model_execution_verified": True,
        "offline_execution_verified": True,
        "repeat_exact": True,
        "runs": 2,
        "new_tokens": 4,
        "layers": 43,
        "root_linear_mode": "bf16",
    }
    logits = torch.zeros(4, 8)
    logits[:, 7] = 2
    result = {
        "execution_policy": "ascendc_v2",
        "finite_logits": True,
        "greedy_logits_agree": True,
        "prefill_tokens": 3,
        "layers_executed": 43,
        "decode_steps": 3,
        "generated_token_ids": [7] * 4,
        "prompt_token_ids": [0, 1, 2],
        "root_fp8": {"mode": "bf16"},
        "load": {"moe_layers": 43, "registered_parameters_loaded": 984},
        "cache": {"layer_calls": {str(i): 4 for i in range(43)}, "resident_experts": 43, "per_layer_cache_limit": 1},
        "peak_allocated_bytes": 1000,
        "peak_reserved_bytes": 2000,
        "steps": [{"tokens": 3, "positions": [0, 1, 2]}] + [{"tokens": 1, "positions": [i]} for i in (3, 4, 5)],
        "expert_backend": {
            "policy": "ascendc_v2",
            "library": {"sha256": gate.sha256(library), "abi_version": 1},
            "fallback_enabled": False,
            "layers": [
                {
                    "layer": i,
                    "steps": [
                        {
                            "tokens": m,
                            "projection_calls": 2,
                            "projection_rows": 2 * m,
                            "expert_calls": 1,
                            "kernel_launches": 2,
                        }
                        for m in (3, 1, 1, 1)
                    ],
                }
                for i in range(43)
            ],
        },
    }
    return (
        library,
        directory,
        gate_record,
        [copy.deepcopy(result), copy.deepcopy(result)],
        [logits.clone(), logits.clone()],
    )


def save_results(tmp_path, directory, marker, records, logits):
    lines = []
    for run, (result, tensor) in enumerate(zip(records, logits)):
        path = directory / f"run-{run}-logits.safetensors"
        save_file({"logits": tensor}, str(path))
        result.update(run=run, logits_sha256=gate.sha256(path), logits_file=str(path))
        (directory / f"run-{run}.json").write_text(json.dumps(result))
        lines.append("MODEL_RESULT " + json.dumps(result))
    lines.append("VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS " + json.dumps(marker))
    log = tmp_path / "model.log"
    log.write_text("\n".join(lines) + "\n")
    return log


def test_model_report_rechecks_actual_saved_logits_and_native_coverage(tmp_path):
    library, directory, marker, records, logits = result_fixture(tmp_path)
    log = save_results(tmp_path, directory, marker, records, logits)
    result = accept.verify_model_report(log, directory, library)
    assert len(result["runs"]) == 2 and result["gate"] == marker


@pytest.mark.parametrize(
    "bad",
    [
        "old_policy",
        "finite",
        "repeat_logits",
        "greedy",
        "coverage",
        "token_count",
        "roots",
        "prompt",
        "zero_layers",
        "integer_logits",
        "missing_pass",
        "duplicate_pass",
    ],
)
def test_model_report_rejects_false_pass_even_with_self_consistent_file_hashes(tmp_path, bad):
    library, directory, marker, records, logits = result_fixture(tmp_path)
    if bad == "old_policy":
        marker["expert_execution_policy"] = "ascendc"
    elif bad == "finite":
        logits[0][0, 0] = float("nan")
    elif bad == "repeat_logits":
        logits[1][0, 0] = 0.5  # Still greedy and finite; only repeat identity fails.
    elif bad == "greedy":
        logits[0][0, 6] = 3
    elif bad == "coverage":
        records[0]["expert_backend"]["layers"].pop()
    elif bad == "token_count":
        records[0]["generated_token_ids"].pop()
    elif bad == "roots":
        records[0]["root_fp8"]["mode"] = "online_fp8_sm90"
    elif bad == "prompt":
        records[1]["prompt_token_ids"] = [2, 1, 0]
    elif bad == "zero_layers":
        marker["layers"] = 0
        for result in records:
            result["layers_executed"] = result["load"]["moe_layers"] = result["cache"]["resident_experts"] = 0
            result["cache"]["layer_calls"] = {}
            result["expert_backend"]["layers"] = []
    elif bad == "integer_logits":
        logits = [tensor.to(torch.int64) for tensor in logits]
    log = save_results(tmp_path, directory, marker, records, logits)
    if bad == "missing_pass":
        log.write_text("MODEL stage=forward_done phase=profile\n")
    elif bad == "duplicate_pass":
        log.write_text(log.read_text() + "VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS " + json.dumps(marker) + "\n")
    with pytest.raises(ValueError):
        accept.verify_model_report(log, directory, library)


def test_preflight_real_preparation_preserves_metadata_before_sorted_k_gather():
    # Byte-contract CPU smoke of the exact helper sequence used in run(); the
    # source payload intentionally has nontrivial scales, biases and signs.
    prepared = gate._synthetic(3, 2048, 2)
    _, _, _, words, books, ids = prepared
    k = 2048
    host = {
        "packed_indices": words,
        "codebooks": books,
        "codebook_tile_ids": ids,
        "weight_scale": torch.linspace(0.1, 1, k),
        "weight_bias": torch.linspace(-0.2, 0.3, k),
        "rht_sign": torch.where(torch.arange(k) % 3 == 0, 1, -1).to(torch.int8),
    }
    spec = NS(columns=k, rht_true_columns=k, rht_block_size=128)
    converted = v2.convert_expert_payload(host, spec)
    hidden = (torch.arange(3 * k).reshape(3, k) % 19 - 9).to(torch.bfloat16) / 16
    actual = RowwiseVQ2A8Preparation().rows(hidden, converted, spec)
    reference = prepare_rows(hidden, host, spec)
    assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(actual, reference))
    gathered = v2.gather_prepared_activation(actual[0], converted["activation_order"])
    assert torch.equal(gathered.view(torch.uint8), reference[0].view(torch.uint8)[:, converted["activation_order"]])
    for name in ("weight_scale", "weight_bias", "rht_sign"):
        assert converted[name] is host[name]


@pytest.mark.parametrize("k", [2048, 4096])
def test_synthetic_preflight_converts_actual_metadata_on_cpu_without_requantization(k):
    inputs = gate._synthetic(3, k, 5)
    converted = gate._convert_inputs(inputs, torch.device("cpu"))
    q, scale, bias, zn, lut = converted
    order = torch.argsort(inputs[5], stable=True)
    assert q.dtype == torch.float8_e4m3fn and q.shape == (3, k)
    assert torch.equal(q.view(torch.uint8), inputs[0].view(torch.uint8).index_select(1, order))
    assert scale.dtype == bias.dtype == torch.float32
    assert torch.equal(scale, inputs[1]) and torch.equal(bias, inputs[2])
    assert zn.dtype == lut.dtype == torch.uint8
    assert zn.shape == (4096 // 32, k // 16, 16, 8)
    assert lut.shape == (k // 256, 4096 // 32, 32)
    assert all(t.device.type == "cpu" and t.is_contiguous() for t in converted)


def test_python_source_identity_covers_actual_runtime_and_numerical_oracle_files():
    hashes = gate.python_source_hashes()
    required = {
        "tools/validate_vq2a8_ascendc_v2.py",
        "tools/validate_vq2a8_phase4_kernel.py",
        "tools/validate_vq2a8_tp1_packed_kernel.py",
        "vllm_ascend/quantization/vq2a8_ascendc_v2.py",
        "vllm_ascend/quantization/vq2a8_activation.py",
        "vllm_ascend/quantization/vq2a8_repack.py",
        "vllm_ascend/quantization/vq2a8_runtime.py",
        "vllm_ascend/quantization/vq2a8_offline.py",
        "vllm_ascend/patch/worker/vq2a8_offline_model.py",
    }
    assert required.issubset(hashes)
    for relative, digest in hashes.items():
        assert len(digest) == 64 and gate.sha256(gate.REPO / relative) == digest
