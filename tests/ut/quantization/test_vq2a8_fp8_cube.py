# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host call-contract checks, NOT Ascend compiler/Cube execution tests."""

import ast
import inspect
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.quantization.vq2a8_fp8_cube as cube
from vllm_ascend.quantization.vq2a8_fused_fp8 import _cube_bridge_kernel, _vq_decode_cube_kernel
from vllm_ascend.quantization.vq2a8_phase4_micro import _fp8_cube_micro


@pytest.fixture
def scale_frontend(monkeypatch):
    """Execute the real helper body against a strict byte-scale call recorder."""
    calls = []

    def static_assert(condition):
        assert condition

    def dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, *, acc, out_dtype):
        assert isinstance(lhs_scale, torch.Tensor) and lhs_scale.dtype == torch.uint8
        assert isinstance(rhs_scale, torch.Tensor) and rhs_scale.dtype == torch.uint8
        assert lhs.dtype == rhs.dtype == torch.float8_e4m3fn
        assert lhs_format == rhs_format == "e4m3"
        assert lhs_scale.shape == (lhs.shape[0], lhs.shape[1] // 16)
        assert rhs_scale.shape == (rhs.shape[1], rhs.shape[0] // 16)
        assert out_dtype == torch.float32 and acc.dtype == torch.float32
        calls.append((lhs_scale.clone(), rhs_scale.clone()))
        # Independent E8M0 decoding: 127 -> 2**0, not scalar 127 or 2**127.
        sa = torch.exp2(lhs_scale.double() - 127).repeat_interleave(16, dim=1)
        sb = torch.exp2(rhs_scale.double() - 127).repeat_interleave(16, dim=1).T
        return ((lhs.double() * sa) @ (rhs.double() * sb) + acc.double()).float()

    language = SimpleNamespace(
        constexpr=int,
        uint8=torch.uint8,
        float8e4nv=torch.float8_e4m3fn,
        float32=torch.float32,
        static_assert=static_assert,
        full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype),
        zeros=lambda shape, dtype: torch.zeros(shape, dtype=dtype),
        trans=lambda tensor: tensor.T,
        dot_scaled=dot_scaled,
    )
    monkeypatch.setattr(cube, "tl", language)
    return language, calls


@pytest.mark.parametrize("m,n,k", [(32, 32, 128), (16, 64, 128), (64, 32, 256), (32, 32, 512)])
def test_identity_scale_shape_encoding_and_chained_accumulator(scale_frontend, m, n, k):
    _, calls = scale_frontend
    lhs = ((torch.arange(m * k).reshape(m, k) % 31 - 15) / 8).to(torch.float8_e4m3fn)
    rhs = ((torch.arange(k * n).reshape(k, n) % 29 - 14) / 8).to(torch.float8_e4m3fn)
    result = torch.full((m, n), 0.75)
    dot = lhs.double() @ rhs.double()
    for step in range(4):
        result = cube.ascend_fp8_dot_unit_scale.fn(lhs, rhs, result)
        torch.testing.assert_close(result, (0.75 + (step + 1) * dot).float(), rtol=0, atol=0)
    report = cube.ascend_fp8_unit_scale_contract(m, n, k)
    assert report["identity_byte"] == 127 and report["scale_k"] == 16
    assert report["dtype"] == "uint8" and report["format"] == "e8m0"
    assert len(calls) == 4
    for sa, sb in calls:
        assert list(sa.shape) == report["lhs_shape"]
        assert list(sb.shape) == report["rhs_shape"]
        assert torch.all(sa == 127) and torch.all(sb == 127)
    assert report["source"] == "kernel_constant"
    assert report["row_scale_bias"] == "unchanged_epilogue"


@pytest.mark.parametrize(
    "kernel,k", [(_vq_decode_cube_kernel, 128), (_cube_bridge_kernel, 128), (_fp8_cube_micro, 512)]
)
def test_every_ascend_caller_supplies_transposed_rhs_and_explicit_unit_scales(scale_frontend, kernel, k):
    # Execute the actual ASCEND branch, not a duplicate of its call. This
    # catches either former None-scale call being reintroduced. Other kernel
    # operations and backend lowering still require the hardware gates.
    language, calls = scale_frontend
    tree = ast.parse(inspect.getsource(kernel.fn))
    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "ASCEND"
    ]
    assert len(branches) == 1
    lhs = torch.ones((32, k)).to(torch.float8_e4m3fn)
    weights = (torch.arange(32 * k).reshape(32, k) % 7).to(torch.float8_e4m3fn)
    accumulator = torch.full((32, 32), 5.0)
    scope = {
        "tl": language,
        "ascend_fp8_dot_unit_scale": cube.ascend_fp8_dot_unit_scale.fn,
        "activation": lhs,
        "a": lhs,
        "weights": weights,
        "b": weights,
        "accumulator": accumulator,
    }
    exec(compile(ast.Module(body=branches[0].body, type_ignores=[]), "<ascend-call-contract>", "exec"), scope)
    actual = scope["result"] if kernel is _fp8_cube_micro else scope["accumulator"]
    expected = lhs.float() @ weights.float().T + (0 if kernel is _fp8_cube_micro else accumulator)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(calls) == 1


def test_frontend_regression_rejects_missing_scale(scale_frontend):
    language, _ = scale_frontend
    operand = torch.ones((32, 128)).to(torch.float8_e4m3fn)
    with pytest.raises(AssertionError):
        language.dot_scaled(
            operand, None, "e4m3", operand.T, None, "e4m3", acc=torch.zeros((32, 32)), out_dtype=torch.float32
        )


def test_static_guards_avoid_legacy_frontend_boolop_lowering():
    # Calling .fn executes ordinary Python and cannot expose the 3.2.2 AST
    # frontend's bool.logical_and failure. Check the actual JIT source too;
    # this is a syntax regression guard, not an Ascend compilation test.
    tree = ast.parse(inspect.getsource(cube.ascend_fp8_dot_unit_scale.fn))
    assert not any(isinstance(node, ast.BoolOp) for node in ast.walk(tree))
    guards = [
        node.args[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "static_assert"
    ]
    assert [ast.unparse(guard) for guard in guards] == [
        "lhs.dtype == tl.float8e4nv",
        "rhs.dtype == tl.float8e4nv",
        "lhs.shape[1] == rhs.shape[0]",
        "lhs.shape[1] % 64 == 0",
    ]


@pytest.mark.parametrize(
    "lhs_dtype,rhs_dtype,lhs_k,rhs_k",
    [
        (torch.float16, torch.float8_e4m3fn, 128, 128),
        (torch.bfloat16, torch.float8_e4m3fn, 128, 128),
        (torch.float8_e4m3fn, torch.float16, 128, 128),
        (torch.float8_e4m3fn, torch.bfloat16, 128, 128),
        (torch.float8_e4m3fn, torch.float8_e4m3fn, 96, 96),
        (torch.float8_e4m3fn, torch.float8_e4m3fn, 128, 192),
    ],
)
def test_unit_scaled_dot_rejects_invalid_operands_before_dot(
    scale_frontend, monkeypatch, lhs_dtype, rhs_dtype, lhs_k, rhs_k
):
    language, calls = scale_frontend

    def unexpected_dot(*args, **kwargs):
        pytest.fail("invalid operands reached dot_scaled instead of failing a static guard")

    monkeypatch.setattr(language, "dot_scaled", unexpected_dot)
    lhs = torch.ones((32, lhs_k)).to(lhs_dtype)
    rhs = torch.ones((rhs_k, 32)).to(rhs_dtype)
    with pytest.raises(AssertionError):
        cube.ascend_fp8_dot_unit_scale.fn(lhs, rhs, torch.zeros((32, 32)))
    assert not calls
