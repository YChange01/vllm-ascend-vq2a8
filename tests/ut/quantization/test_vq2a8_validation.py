# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tools import validate_vq2a8_tp1_packed_kernel as gate
from tools.validate_vq2a8_tp1_acceptance import acceptance_environment, summarize_log
from vllm_ascend.quantization.vq2a8_reference import deepseek_v4_swiglu_reference
from vllm_ascend.quantization.vq2a8_validation import audit_model_storage, validate_tolerances


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
def test_invalid_tolerance_cannot_make_a_gate_pass(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_tolerances(value, 0.01)
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_tolerances(0.01, value)
    with pytest.raises(ValueError):
        deepseek_v4_swiglu_reference(torch.ones(1, 4), value)


def test_small_signal_cannot_pass_only_because_of_large_atol() -> None:
    with pytest.raises(AssertionError, match="relative_l2"):
        gate._comparison_summary(torch.full((1, 8), 1e-6), torch.zeros(1, 8), rtol=0.03, atol=0.05)


def test_passing_comparison_retains_scale_aware_metrics() -> None:
    result = gate._comparison_summary(torch.ones(1, 8), torch.full((1, 8), 1.001), rtol=0.01, atol=0.001)
    assert result["allclose"]
    assert 0 < result["relative_l2_error"] < 0.002


@pytest.mark.parametrize("returncode,timed_out,passed", [(0, False, True), (-6, False, False), (0, True, False)])
def test_supervisor_does_not_trust_pass_text_after_abort(tmp_path, returncode, timed_out, passed) -> None:
    log = tmp_path / "probe.log"
    log.write_text("KERNEL 0:0:gate_up\nVQ2A8_TP1_M1_PACKED_KERNEL_GATE=PASS {}\nerrorStr: MTE alignment\n")
    result = summarize_log(log, returncode, timed_out=timed_out)
    assert result["passed"] is passed
    assert result["last_stage"] == "KERNEL 0:0:gate_up"
    assert result["error_excerpt"] == ["errorStr: MTE alignment"]


def test_supervisor_requires_completion_marker(tmp_path) -> None:
    log = tmp_path / "probe.log"
    log.write_text('KERNEL_RESULT {"comparison": {"allclose": true}}\n')
    assert not summarize_log(log, 0)["passed"]


def test_supervisor_isolates_device_only_in_child(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5,6,7")
    child = acceptance_environment(tmp_path, 4, "npu:0")
    assert "WORLD_SIZE" not in child
    assert child["ASCEND_RT_VISIBLE_DEVICES"] == "4"
    assert os.environ["WORLD_SIZE"] == "4"


def _audit_fixture(tmp_path: Path, route_id: int) -> SimpleNamespace:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"vocab_size": 4, "num_experts_per_tok": 2}))
    save_file(
        {
            "layers.0.ffn.gate.tid2eid": torch.full((4, 2), route_id, dtype=torch.int64),
            "some_weight": torch.ones(2, 4, dtype=torch.bfloat16),
        },
        str(tmp_path / "model.safetensors"),
    )
    layer = SimpleNamespace(expert_ids=(0,), tensor_shapes={"down_packed_indices": (1, 4, 4)})
    return SimpleNamespace(
        model_config_path=config,
        layers={0: layer},
        model_layout=SimpleNamespace(num_hash_layers=1),
        manifest={"complete": True},
        layer=lambda _: layer,
    )


def test_model_inventory_checks_hash_table_values_and_counts_bytes(tmp_path) -> None:
    result = audit_model_storage(_audit_fixture(tmp_path, 0))
    assert result["root_checkpoint_tensor_bytes"] == 80
    assert result["artifact_expert_tensor_bytes"] == 64
    assert result["hash_routing"][0]["duplicate_ids_within_topk"] is True
    assert result["peak_hbm_verified"] is False


def test_hash_layer_cannot_route_to_absent_stored_expert(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing artifact experts"):
        audit_model_storage(_audit_fixture(tmp_path, 1))


def test_expert_chain_uses_actual_gate_output(monkeypatch, tmp_path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"swiglu_limit": 10}')
    cpu = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    actual = cpu + 0.001
    calls = []

    def projection(*positional, **kwargs):
        calls.append(kwargs)
        return {}, cpu, actual

    monkeypatch.setattr(gate, "_run_projection", projection)
    args = SimpleNamespace(cases=["deterministic"], warmups=0, repeats=1, rtol=0.03, atol=0.05, chain=True)
    artifact = SimpleNamespace(model_config_path=config, model_layout=SimpleNamespace(num_routed_experts=256))
    results = gate.run_probe_cases(artifact, gate.Probe(0, 0), torch.device("cpu"), args)
    assert len(calls) == 3
    assert "activation_device" not in calls[1]
    torch.testing.assert_close(calls[2]["activation_device"], deepseek_v4_swiglu_reference(actual, 10))
    torch.testing.assert_close(calls[2]["activation_cpu"], deepseek_v4_swiglu_reference(cpu, 10))
    assert results[-1]["path"] == "gate_up_swiglu_down"


def test_bad_middle_repeat_is_not_hidden_by_a_good_last_repeat(monkeypatch) -> None:
    spec = SimpleNamespace(rht_true_columns=512, columns=512, rows=32, rht_block_size=128)
    payload = {
        "weight_scale": torch.ones(512),
        "weight_bias": torch.zeros(512),
        "rht_sign": torch.ones(512, dtype=torch.int8),
        "packed_indices": torch.zeros(16, 64, dtype=torch.int32),
        "codebooks": torch.zeros(2, 1, 16, 2).to(torch.float8_e4m3fn),
        "codebook_tile_ids": torch.zeros(512, dtype=torch.uint8),
    }
    artifact = SimpleNamespace(load_expert=lambda *args: (payload, spec))
    monkeypatch.setattr(gate, "decode_repacked_vq2a8_codebook_weight", lambda *a, **kw: torch.zeros(32, 512))
    monkeypatch.setattr(gate, "vq2a8_predecoded_matmul_reference", lambda *a, **kw: torch.zeros(1, 32))
    monkeypatch.setattr(
        gate,
        "prepare_repacked_vq2a8_activation_reference",
        lambda *a: (torch.zeros(1, 512).to(torch.float8_e4m3fn), torch.ones(1), torch.zeros(1)),
    )
    monkeypatch.setattr(gate, "_synchronize", lambda *a: None)
    outputs = iter([torch.zeros(1, 32), torch.zeros(1, 32), torch.ones(1, 32), torch.zeros(1, 32)])
    monkeypatch.setattr(gate, "vq2a8_tp1_m1_packed_gemm", lambda *a: next(outputs))
    with pytest.raises(AssertionError, match="Packed kernel mismatch"):
        gate._run_projection(
            artifact, gate.Probe(0, 0), 0, "gate_up", torch.device("cpu"), warmups=0, repeats=3, rtol=0.03, atol=0.05
        )


def test_supervisor_keeps_numerical_failure_and_prepared_input_evidence(tmp_path) -> None:
    log = tmp_path / "probe.log"
    log.write_text(
        'PREPARED_INPUT_RESULT {"comparison": {"allclose": true}}\n'
        'NUMERIC_FAILURE {"allclose": false, "relative_l2_error": 0.002}\n'
    )
    result = summarize_log(log, 1)
    assert not result["passed"]
    assert [record["type"] for record in result["records"]] == ["PREPARED_INPUT_RESULT", "NUMERIC_FAILURE"]
