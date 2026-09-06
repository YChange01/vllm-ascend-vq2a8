# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the bounded A5 SWA preflight; no hardware PASS is implied."""

import ast
from pathlib import Path

import pytest
import torch

from tools.validate_vq2a8_sas_attention import DEFINED_WORDS, check_sas_metadata, expected_swa_output


def preflight_op_keywords(op_name):
    repo = Path(__file__).resolve().parents[3]
    tree = ast.parse((repo / "tools/validate_vq2a8_sas_attention.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == op_name
    ]
    assert len(calls) == 1
    assert all(keyword.arg is not None for keyword in calls[0].keywords)
    return {keyword.arg: ast.unparse(keyword.value) for keyword in calls[0].keywords}


@pytest.mark.parametrize("unsupported", ["ori_sparse_indices", "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv"])
def test_sas_compute_omits_inputs_rejected_by_a5_tiling(unsupported):
    # CheckUnrequiredParaExistence rejects these even though the torch schema
    # exposes them. Metadata and compute do not have the same input contract.
    keywords = preflight_op_keywords("npu_kv_quant_sparse_attn_sharedkv")
    assert unsupported not in keywords
    assert keywords["seqused_kv"] == "usedk"
    assert keywords["cu_seqlens_q"] == "cuq"
    assert keywords["ori_block_table"] == "table"


def test_sas_metadata_retains_cumulative_kv_lengths():
    keywords = preflight_op_keywords("npu_kv_quant_sparse_attn_sharedkv_metadata")
    assert keywords["cu_seqlens_ori_kv"] == "cuk"
    assert keywords["seqused_kv"] == "usedk"
    assert keywords["cu_seqlens_q"] == "cuq"


def test_sas_preflight_compute_keywords_match_production_a5_swa():
    # Inspect without importing the NPU/vLLM runtime on the CPU test host.
    repo = Path(__file__).resolve().parents[3]
    device_tree = ast.parse((repo / "vllm_ascend/device/device_op.py").read_text(encoding="utf-8"))
    a5 = next(node for node in device_tree.body if isinstance(node, ast.ClassDef) and node.name == "A5DeviceAdaptor")
    base = next(
        node for node in a5.body if isinstance(node, ast.FunctionDef) and node.name == "get_dsa_sparse_attn_base_kwargs"
    )
    base_kwargs = ast.literal_eval(next(node.value for node in base.body if isinstance(node, ast.Return)))
    attention_tree = ast.parse((repo / "vllm_ascend/attention/dsa_v1.py").read_text(encoding="utf-8"))
    prefill = next(
        node
        for node in ast.walk(attention_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_forward_prefill"
    )
    swa = min(
        (
            node
            for node in ast.walk(prefill)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "attn_op"
        ),
        key=lambda node: node.lineno,
    )
    keywords = preflight_op_keywords("npu_kv_quant_sparse_attn_sharedkv")
    assert set(keywords) == {keyword.arg for keyword in swa.keywords if keyword.arg is not None} | base_kwargs.keys()
    assert {key: ast.literal_eval(keywords[key]) for key in base_kwargs} == base_kwargs


def valid_metadata(heads):
    values = [0] * 1024
    row = [1, 0, 0, 0, 1, 0, 0, 0, 0 if heads == 64 else 1]
    values[:9] = row
    if heads == 128:
        values[9:18] = row
    return values


@pytest.mark.parametrize("heads", [64, 128])
def test_sas_fixed_abi_and_reserved_tail(heads):
    assert DEFINED_WORDS == 900  # QLI is 864; do not reuse its decoder.
    values = valid_metadata(heads)
    values[DEFINED_WORDS:] = [0x55555555] * (1024 - DEFINED_WORDS)
    assert check_sas_metadata(values, heads) == (1 if heads == 64 else 2)


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("index", [0, 1, 4, 7, 28 * 9, 32 * 9, 35 * 9, 36 * 9, 36 * 9 + 64 * 8, 899])
def test_sas_rejects_poison_flags_intervals_and_fd_slots(heads, index):
    values = valid_metadata(heads)
    values[index] = 0x55555555
    with pytest.raises(ValueError):
        check_sas_metadata(values, heads)


def test_sas_n128_keeps_idle_pair_barrier_counts():
    values = valid_metadata(128)
    values[2 * 9 + 8] = values[3 * 9 + 8] = 1
    assert check_sas_metadata(values, 128) == 2
    values[3 * 9 + 8] = 2
    with pytest.raises(ValueError, match="paired"):
        check_sas_metadata(values, 128)


@pytest.mark.parametrize("heads", [64, 128])
def test_sas_rejects_gaps_and_overlaps(heads):
    values = valid_metadata(heads)
    values[4] = 0
    if heads == 128:
        values[9 + 4] = 0
    with pytest.raises(ValueError):
        check_sas_metadata(values, heads)


@pytest.mark.parametrize("length,heads", [(900, 64), (1024, 32)])
def test_sas_rejects_wrong_shape_and_heads(length, heads):
    with pytest.raises(ValueError):
        check_sas_metadata([0] * length, heads)


def test_analytic_swa_oracle_has_causal_prefix_and_sink():
    result = expected_swa_output(3, 3, 64, 512)
    torch.testing.assert_close(result[:, 0, 1], torch.tensor([0.25, 1 / 3, 0.375]))
    torch.testing.assert_close(result[:, 0, 0], -result[:, 0, 1])
    assert result.shape == (3, 64, 512)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result[:, 0], result[:, -1])


def test_analytic_decode_oracle_uses_history():
    result = expected_swa_output(1, 11, 128, 512)
    assert float(result[0, 0, 1]) == pytest.approx(11 / 24)


@pytest.mark.parametrize("query_len,key_len", [(0, 1), (2, 1), (1, 129)])
def test_oracle_rejects_out_of_scope_lengths(query_len, key_len):
    with pytest.raises(ValueError):
        expected_swa_output(query_len, key_len, 64, 512)


def test_offline_runs_actual_attention_preflight_before_llm():
    repo = Path(__file__).resolve().parents[3]
    tree = ast.parse((repo / "tools/validate_vq2a8_tp1_offline.py").read_text(encoding="utf-8"))
    calls = {n.func.id: n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert calls["run_preflight"].lineno < calls["run_sas_preflight"].lineno < calls["LLM"].lineno
    assert any(k.arg == "prompt_tokens" for k in calls["run_sas_preflight"].keywords)
