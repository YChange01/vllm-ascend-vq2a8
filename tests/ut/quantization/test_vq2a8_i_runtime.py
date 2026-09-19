# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""I runtime gates and host reporting; not hardware numerical certification."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm_ascend.quantization import vq2a8_activation_packed as packed
from vllm_ascend.quantization import vq2a8_v4_v2 as v2
from vllm_ascend.quantization.vq2a8_runtime_guard import RUNTIME_FIELDS


def native_features():
    return NS(
        activation_reorder_version=lambda: 1,
        activation_preparation_version=lambda: 1,
        activation_sign_strided_version=lambda: 1,
        select_sign_version=lambda: 1,
    )


@pytest.mark.parametrize("version", [None, True, False, 0, 2, "1", 1])
def test_i_requires_independent_abi_without_fallback(version):
    native = native_features()
    if version is not None:
        native.swiglu_select_sign_version = lambda: version
    options = dict(
        reorder="vectorized",
        preparation="sign_fused_direct",
        select_sign="fused",
        swiglu_mode="fused_select_sign",
        native_ops=native,
    )
    if type(version) is int and version == 1:
        v2.require_v4_v2_features(**options)
    else:
        with pytest.raises(RuntimeError, match="swiglu_select_sign_version"):
            v2.require_v4_v2_features(**options)


def test_i_default_does_not_require_new_native_abi():
    v2.require_v4_v2_features(
        reorder="vectorized",
        preparation="sign_fused_direct",
        select_sign="fused",
        native_ops=native_features(),
    )
    assert ("v4_swiglu_mode", "torch") in RUNTIME_FIELDS


@pytest.mark.parametrize("mode", ["torch", "fused_select_sign"])
def test_i_runtime_constructs_selected_preparation_and_reports_scope(monkeypatch, mode):
    monkeypatch.setattr(v2.AscendCV4VQ2TP1MoE, "__init__", lambda *a, **kw: None)
    runtime = v2.AscendCV4V2VQ2TP1MoE(
        v4_activation_reorder="vectorized",
        v4_activation_preparation="sign_fused_direct",
        v4_select_sign="fused",
        v4_swiglu_mode=mode,
    )
    assert runtime.v4_swiglu_mode == mode
    assert runtime.v4_swiglu_graph_build_calls == runtime.v4_swiglu_reference_calls == 0
    monkeypatch.setattr(packed, "PackedRowwiseVQ2A8Preparation", lambda **kw: kw)
    prep = runtime.make_v4_preparation(compact=True)
    assert prep["fuse_swiglu"] is (mode == "fused_select_sign")
    assert prep["fuse_select"] is True and prep["direct_output"] is True
    monkeypatch.setattr(v2.AscendCV4VQ2TP1MoE, "v4_report", lambda *a: {})
    runtime.artifact = NS(manifest={})
    runtime.timing = {}
    runtime._device_route_banks = runtime._resident_plan = None
    runtime.v4_swiglu_graph_build_calls, runtime.v4_swiglu_reference_calls = 2, 3
    report = runtime.v4_report()
    assert report["swiglu_mode"] == mode
    assert report["swiglu_graph_build_calls"] == 2
    assert report["swiglu_reference_calls"] == 3


@pytest.mark.parametrize("mode", [None, True, 1, "fused", "auto"])
def test_i_invalid_mode_fails_before_base_runtime_allocation(monkeypatch, mode):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid I mode must fail before any resident allocation")

    monkeypatch.setattr(v2.AscendCV4VQ2TP1MoE, "__init__", forbidden)
    with pytest.raises(ValueError):
        v2.AscendCV4V2VQ2TP1MoE(v4_swiglu_mode=mode)


@pytest.mark.parametrize("mode", ["torch", "fused_select_sign"])
@pytest.mark.parametrize("method", ["missing", "not_callable", "runtime_error", "present"])
def test_i_bank_gate_rejects_partial_library_only_when_selected(monkeypatch, mode, method):
    runtime = object.__new__(v2.AscendCV4V2VQ2TP1MoE)
    runtime._require_ready = lambda: None
    runtime.device = NS(type="npu")
    runtime.layer = NS(expert_ids=(0,))
    runtime.config = NS(top_k=1, num_experts=1)
    runtime.v4_activation_reorder = "vectorized"
    runtime.v4_swiglu_mode = mode
    spec = NS(rows=2048, columns=2048, rht_true_columns=2048, rht_block_size=128)
    runtime._cache = {0: {kind: ({key: None for key in v2.V4_V2_FIELDS}, spec) for kind in ("gate_up", "down")}}

    class Bank:
        def project_vectorized(self, *args):
            pytest.fail("Bank construction must not launch a projection")

        def metadata(self):
            return (0, 0, 0, v2.V4_V2_BANK_WORDS * 8)

        def __getattr__(self, name):
            if name == "swiglu_select_sign":
                if method == "runtime_error":
                    raise RuntimeError("old Torch custom class")
                if method == "not_callable":
                    return 1
                if method == "present":
                    return lambda *a: pytest.fail("Bank construction must not launch SwiGLU")
            raise AttributeError(name)

    monkeypatch.setattr(v2, "require_v4_v2_library", lambda: lambda *a: Bank())
    original_to = torch.Tensor.to

    def host_only_to(tensor, *args, **kwargs):
        # Substitute allocation transport only; this test cannot use a real NPU.
        if args and args[0] is runtime.device:
            return tensor
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", host_only_to)
    if mode == "fused_select_sign" and method != "present":
        with pytest.raises(RuntimeError, match="I SwiGLU select/sign; no fallback"):
            runtime._create_device_route_banks()
    else:
        banks = runtime._create_device_route_banks()
        assert set(banks) == {"lookup", "slots", "metadata_bytes", "gate_up", "down"}
