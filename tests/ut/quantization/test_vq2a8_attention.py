# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the actual DSA output-projection method with CPU operator stand-ins.

These regression tests cover dispatch/layout, not CANN or full attention.
"""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


class Unquantized:
    pass


def projection_method(device_type="A5", npu=None):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/dsa_v1.py"
    method = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "_forward_o_proj"
    )
    scope = {
        "torch": torch,
        "torch_npu": npu or NS(),
        "get_ascend_device_type": lambda: device_type,
        "AscendDeviceType": NS(A5="A5"),
        "AscendUnquantizedLinearMethod": Unquantized,
        "oproj_tp_enable": lambda: False,
        "olora_tp_enable": lambda: False,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope["_forward_o_proj"]


@pytest.mark.parametrize("tokens", [1, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a5_unquantized_root_needs_no_scale_and_preserves_group_layout(tokens, dtype):
    groups, rank, hidden = 2, 3, 8
    weight = ((torch.arange(groups * rank * hidden).reshape(groups * rank, hidden) % 17) / 16).to(dtype)
    x = ((torch.arange(tokens * groups * hidden).reshape(tokens, groups, hidden) % 13) / 16).to(dtype)
    owner = NS(
        n_local_groups=groups, o_lora_rank=rank, wo_a=NS(weight=weight, quant_method=Unquantized()), wo_b=lambda x: x
    )
    assert not hasattr(owner.wo_a, "weight_scale")
    output = torch.empty(tokens, groups * rank, dtype=dtype)
    result = projection_method()(owner, x, output)
    expected = torch.cat(
        [torch.nn.functional.linear(x[:, group], weight[group * rank : (group + 1) * rank]) for group in range(groups)],
        dim=-1,
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result is output and owner.wo_a.weight is weight
    assert weight.shape == (groups * rank, hidden)


def test_unexpected_unquantized_layout_fails_before_matmul():
    owner = NS(n_local_groups=2, o_lora_rank=3, wo_a=NS(weight=torch.zeros(2, 8, 3), quant_method=Unquantized()))
    with pytest.raises(ValueError, match="A5 unquantized wo_a expects weight shape"):
        projection_method()(owner, torch.zeros(1, 2, 8), torch.empty(1, 6))


def test_existing_a5_quantized_branch_keeps_scale_and_permutation_contract():
    calls = []
    x_scale, w_scale = NS(view=lambda _: "x_scale"), NS(view=lambda _: "w_scale")
    npu = NS(
        npu_dynamic_mx_quant=lambda x, **kw: (x, x_scale),
        npu_transpose_quant_batchmatmul=lambda *a, **kw: calls.append(kw) or torch.ones(1, 2, 3),
    )
    owner = NS(
        n_local_groups=2,
        o_lora_rank=3,
        wo_a=NS(weight=torch.zeros(2, 8, 3).to(torch.float8_e4m3fn), quant_method=object(), weight_scale=w_scale),
        wo_b=lambda x: x,
    )
    projection_method(npu=npu)(owner, torch.zeros(1, 2, 8), torch.empty(1, 6))
    assert calls[0]["x1_scale"] == "x_scale" and calls[0]["x2_scale"] == "w_scale"
    assert calls[0]["group_sizes"] == (0, 0, 32)
    assert calls[0]["perm_x1"] == calls[0]["perm_y"] == (1, 0, 2)
    del owner.wo_a.weight_scale
    with pytest.raises(AttributeError, match="weight_scale"):
        projection_method(npu=npu)(owner, torch.zeros(1, 2, 8), torch.empty(1, 6))


def test_non_a5_grouped_weight_path_unchanged():
    calls = []
    npu = NS(npu_transpose_batchmatmul=lambda *a, **kw: calls.append(kw) or torch.ones(1, 2, 3))
    owner = NS(n_local_groups=2, wo_a=NS(weight=torch.zeros(2, 8, 3)), wo_b=lambda x: x)
    projection_method(device_type="A3", npu=npu)(owner, torch.zeros(1, 2, 8), torch.empty(1, 6))
    assert calls[0]["scale"] is None and calls[0]["batch_split_factor"] == 1
