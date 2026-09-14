# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP2 shape contracts and opt-in real-device projection regressions.

CPU cases do not compile AscendC or certify NPU execution. Device cases require
torch_npu and a pinned V3 .so loaded with load_pinned_library before pytest.main;
they never discover/load an arbitrary local library or fall back to V2.
"""

import importlib.util
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import regex as re
import torch

from vllm_ascend.quantization import vq2a8_ascendc_v3 as binding

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "csrc/vq2a8_ascendc_v3"
TP2_SHAPES = ((2048, 4096), (4096, 2048))


def _function(source, name):
    match = re.search(rf"\b{name}\([^;{{}}]*\)\s*\{{", source)
    assert match is not None
    depth, end = 1, match.end()
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start() : end]


def _namespace(monkeypatch, capabilities=7):
    calls = []
    namespace = NS(
        resident_abi_version=lambda: 1,
        resident_capabilities=lambda: capabilities,
        grouped_projection_resident=lambda *args: calls.append(("eager", args)),
        grouped_projection_resident_out=lambda *args: calls.append(("out", args)),
    )
    monkeypatch.setattr(binding.torch.ops, "vq2a8_ascendc_v3", namespace)
    return namespace, calls


def test_tp2_native_geometry_header_compiles_without_changing_tp1_contract(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("No host C++ compiler; this is not a CANN compilation test")
    code = tmp_path / "tp2_layout.cpp"
    code.write_text(
        """#include "resident_layout.h"
#include "prepare_layout.h"
#include <cassert>
#include <initializer_list>
using namespace vq2a8_v3_resident;
int main() {
  static_assert(kAbiVersion == 1 && kJobWords == 9);
  static_assert(kTp2Projection == 4 && kCapabilities == 7);
  static_assert(kN == 128 && kAicK == 1024 && kAivK == 512);
  static_assert(kMadK == 256 && kCodebookK == 256 && kBuffers == 2);
  for (int m = -1; m <= 34; ++m) {
    bool supported = m >= 1 && m <= 32;
    assert(ValidDimensions(m, 4096, 2048) == supported);
    assert(ValidDimensions(m, 4096, 4096) == supported);
    assert(!ValidDimensions(m, 2048, 4096));
    assert(ValidTp2Dimensions(m, 2048, 4096) == supported);
    assert(ValidTp2Dimensions(m, 4096, 2048) == supported);
    assert(!ValidTp2Dimensions(m, 4096, 4096));
    for (int n : {0, 32, 128, 2048, 4096, 6144}) {
      for (int k : {0, 256, 512, 768, 1024, 1280, 1536, 1792, 2304, 3072}) {
        assert(!ValidDimensions(m, n, k) && !ValidTp2Dimensions(m, n, k));
      }
    }
  }
  for (int jobs = 1; jobs <= 6; ++jobs) {
    assert(vq2a8_v3::ValidPrepareDimensions(jobs, 2048));
    assert(vq2a8_v3::ValidPrepareDimensions(jobs, 4096));
  }
}
""",
        encoding="utf-8",
    )
    binary = tmp_path / "tp2_layout"
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(SOURCE), str(code), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_tp2_native_preserves_v2_pipeline_and_both_slot_drain_requirement():
    kernel = (SOURCE / "resident_kernel.cpp").read_text(encoding="utf-8")
    original = (REPO / "csrc/vq2a8_ascendc_v2/kernel.cpp").read_text(encoding="utf-8")
    for name in ("RunLut", "LoadVectorTile", "StoreB1", "Vector", "LoadA1", "LoadL0", "Cube"):
        assert _function(kernel, name) == _function(original, name)
    header = (SOURCE / "resident_layout.h").read_text(encoding="utf-8")
    assert "least TWO complete K1024 tiles" in header
    assert "K1024 would leave slot 1 without an acknowledgement" in header
    assert "for (uint32_t ki = 0; ki < k_ / kAicK; ++ki)" in kernel
    assert "slot < kBuffers; ++slot) CrossCoreWaitFlag" in kernel
    native = (SOURCE / "torch_binding.cpp").read_text(encoding="utf-8")
    for name in ("CheckResidentProjection", "GroupedProjectionResidentOut"):
        function = _function(native, name)
        assert "vq2a8_v3_resident::ValidDimensions(m, n, k)" in function
        assert "vq2a8_v3_resident::ValidTp2Dimensions(m, n, k)" in function
    out = _function(native, "GroupedProjectionResidentOut")
    assert "descriptors.size(1) == vq2a8_v3_resident::kJobWords" in out
    assert "RecordInputStream(owners, npuStream)" in out
    assert "RecordInputStream({descriptors}, npuStream)" in out


@pytest.mark.parametrize("n,k", TP2_SHAPES)
@pytest.mark.parametrize("jobs", [1, 6])
@pytest.mark.parametrize("cores", [1, 8, 16, 24, 32, 48])
def test_tp2_native_dynamic_n128_work_mapping_covers_every_job_once(n, k, jobs, cores):
    groups = n // 128
    blocks = min(cores, jobs * groups)
    work = [index for core in range(blocks) for index in range(core, jobs * groups, blocks)]
    assert sorted(work) == list(range(jobs * groups))
    for index in work:
        n_begin = (index % groups) * 128
        for ki in range(k // 1024):
            for half in (0, 1):
                k_begin = ki * 1024 + half * 512
                packed_begin = ((n_begin // 32) * (k // 16) + k_begin // 16) * 16 * 8
                packed_last = packed_begin + 3 * k * 8 + 512 * 8 - 1
                lut_begin = ((k_begin // 256) * (n // 32) + n_begin // 32) * 32
                lut_last = lut_begin + n + 4 * 32 - 1
                assert 0 <= packed_begin <= packed_last < n * k // 4
                assert 0 <= lut_begin <= lut_last < k // 256 * n
        assert n_begin + 128 <= n


@pytest.mark.parametrize("capabilities", [1, 3])
def test_tp2_native_old_libraries_remain_valid_for_tp1_but_reject_tp2(monkeypatch, capabilities):
    _, calls = _namespace(monkeypatch, capabilities)
    assert binding.resident_library_capabilities() == capabilities
    with pytest.raises(RuntimeError, match="TP2"):
        binding.resident_library_capabilities(require_tp2=True)
    descriptors = torch.zeros((1, 9), dtype=torch.int64)
    owner = torch.empty(1)
    binding.grouped_projection_resident_out(descriptors, [owner], jobs=1, m=1, n=4096, k=2048)
    assert calls[0][0] == "out"
    with pytest.raises(RuntimeError, match="TP2"):
        binding.grouped_projection_resident_out(descriptors, [owner], jobs=1, m=1, n=4096, k=2048, tp_size=2)
    assert len(calls) == 1


@pytest.mark.parametrize("capabilities", [True, -1, 7.0, "7"])
def test_tp2_native_capabilities_reject_non_integer_or_negative_responses(monkeypatch, capabilities):
    _namespace(monkeypatch, capabilities)
    with pytest.raises(RuntimeError, match="nonnegative integer"):
        binding.resident_library_capabilities(require_tp2=True)


@pytest.mark.parametrize("tp_size", [True, False, 0, 3, 2.0])
def test_tp2_native_wrapper_rejects_invalid_tp_sizes(monkeypatch, tp_size):
    _, calls = _namespace(monkeypatch)
    with pytest.raises(ValueError, match="tp_size"):
        binding.grouped_projection_resident([], tp_size=tp_size)
    with pytest.raises(ValueError, match="tp_size"):
        binding.grouped_projection_resident_out(None, [], jobs=1, m=1, n=2048, k=4096, tp_size=tp_size)
    assert not calls


@pytest.mark.parametrize("n,k", TP2_SHAPES)
@pytest.mark.parametrize("jobs", [1, 6])
def test_tp2_native_wrappers_dispatch_only_explicit_supported_local_shapes(monkeypatch, n, k, jobs):
    _, calls = _namespace(monkeypatch)
    assert binding.resident_library_capabilities(require_tp2=True) == 7
    x, packed = NS(ndim=2, shape=(32, k)), NS(ndim=4, shape=(n // 32, k // 16, 16, 8))
    inputs = [(x, object(), object(), packed, object())] * jobs
    binding.grouped_projection_resident(inputs, tp_size=2)
    assert calls[0][0] == "eager" and len(calls[0][1][0]) == jobs
    descriptors = torch.zeros((jobs, 9), dtype=torch.int64)
    owners = [torch.empty(1)]
    binding.grouped_projection_resident_out(descriptors, owners, jobs=jobs, m=1, n=n, k=k, tp_size=2)
    assert calls[1][0] == "out" and calls[1][1][-4:] == (jobs, 1, n, k)
    if n == 2048:
        with pytest.raises(ValueError, match="TP1"):
            binding.grouped_projection_resident(inputs)
        with pytest.raises(ValueError, match="TP1"):
            binding.grouped_projection_resident_out(descriptors, owners, jobs=jobs, m=1, n=n, k=k)


@pytest.mark.parametrize("n,k", [(4096, k) for k in (256, 512, 768, 1024, 1280, 1536, 1792, 3072)] + [(2048, 2048)])
def test_tp2_native_wrappers_reject_unpadded_or_tail_k_before_dispatch(monkeypatch, n, k):
    _, calls = _namespace(monkeypatch)
    x, packed = NS(ndim=2, shape=(1, k)), NS(ndim=4, shape=(n // 32, k // 16, 16, 8))
    with pytest.raises(ValueError, match="TP2"):
        binding.grouped_projection_resident([(x, None, None, packed, None)], tp_size=2)
    with pytest.raises(ValueError, match="TP2"):
        binding.grouped_projection_resident_out(None, [], jobs=1, m=1, n=n, k=k, tp_size=2)
    assert not calls


def _device_case(n, k, m, jobs, device, *, logical_k=None):
    """Small integer FP8 values make the full-K FP32 dot exactly representable."""
    rng = np.random.default_rng(n + k + m + jobs)
    codes = rng.integers(0, 16, (n // 2, k), dtype=np.uint8)
    zn = codes.reshape(n // 32, 16, k // 16, 16).transpose(0, 2, 3, 1)
    packed_cpu = torch.from_numpy(np.ascontiguousarray(zn[..., 0::2] | (zn[..., 1::2] << np.uint8(4))))
    books_cpu = torch.from_numpy(rng.integers(-2, 3, (k // 256, n // 32, 16, 2)).astype(np.float32))
    books_cpu = books_cpu.to(torch.float8_e4m3fn)
    ns, ks = np.arange(n)[:, None], np.arange(k)[None, :]
    raw = books_cpu.view(torch.uint8).numpy()
    dense_bytes = np.ascontiguousarray(raw[ks // 256, ns // 32, codes[ns // 2, ks], ns % 2])
    dense = torch.from_numpy(dense_bytes).view(torch.float8_e4m3fn).float()
    packed = packed_cpu.to(device)
    table = books_cpu.view(torch.uint8).reshape(k // 256, n // 32, 32).to(device)
    requests, expected = [], []
    for job in range(jobs):
        # A six-job launch exercises both AlignedM classes within one AIC loop.
        rows = m if job % 2 == 0 else 1
        x = torch.from_numpy(rng.integers(-2, 3, (rows, k)).astype(np.float32))
        if logical_k is not None:
            # Physical dummy columns are zero before the packed-K permutation.
            # Scatter them throughout both compute tiles, not just the tail.
            x[:, logical_k:] = 0
            x = x[:, torch.from_numpy(rng.permutation(k))]
        scale = torch.full((rows,), 0.5 * (job + 1), dtype=torch.float32)
        bias = torch.arange(rows, dtype=torch.float32) * 0.25 - job
        expected.append(((x @ dense.T) * scale[:, None] + bias[:, None]).to(torch.bfloat16))
        requests.append((x.to(torch.float8_e4m3fn).to(device), scale.to(device), bias.to(device), packed, table))
    return requests, expected


@pytest.fixture(scope="module")
def tp2_npu_device():
    if importlib.util.find_spec("torch_npu") is not None:
        __import__("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Device TP2 tests require Ascend NPU; CPU tests do not certify the kernel")
    namespace = torch.ops.vq2a8_ascendc_v3
    if not hasattr(namespace, "grouped_projection_resident"):
        pytest.skip("Load an explicitly SHA256-pinned V3 .so before invoking these device tests")
    binding.resident_library_capabilities(require_tp2=True)
    return torch.device("npu", torch.npu.current_device())


@pytest.mark.parametrize("n,k", TP2_SHAPES)
@pytest.mark.parametrize("jobs", [1, 6])
@pytest.mark.parametrize("m", [1, 7, 16, 17, 32])
def test_tp2_native_device_eager_exact_integer_oracle(tp2_npu_device, n, k, jobs, m):
    inputs, expected = _device_case(n, k, m, jobs, tp2_npu_device)
    for _ in range(2):
        actual = binding.grouped_projection_resident(inputs, tp_size=2)
        torch.npu.synchronize()
        assert len(actual) == jobs
        for result, reference in zip(actual, expected):
            assert torch.equal(result.cpu().view(torch.int16), reference.view(torch.int16))


@pytest.mark.parametrize("jobs", [1, 6])
@pytest.mark.parametrize("m", [1, 32])
def test_tp2_native_device_down_permuted_zero_dummy_columns(tp2_npu_device, jobs, m):
    inputs, expected = _device_case(4096, 2048, m, jobs, tp2_npu_device, logical_k=1024)
    for _ in range(2):
        actual = binding.grouped_projection_resident(inputs, tp_size=2)
        torch.npu.synchronize()
        for result, reference in zip(actual, expected):
            assert torch.equal(result.cpu().view(torch.int16), reference.view(torch.int16))


@pytest.mark.parametrize("jobs", [1, 6])
def test_tp2_native_device_fused_prepare_dummy_zero_and_all_zero_reuse(tp2_npu_device, jobs):
    # RHT/bias GEMV are deliberately outside this native operator. This checks
    # its post-RHT TP2 K2048 boundary, exact byte gather and dirty-buffer reuse.
    assert binding.resident_library_capabilities(require_tp2=True) & binding.FUSED_PREPARATION
    rng = np.random.default_rng(jobs)
    logical_k, k = 1024, 2048
    rotated_cpu = torch.zeros((jobs, k), dtype=torch.float32)
    rotated_cpu[:, :logical_k] = torch.from_numpy(rng.integers(-2, 3, (jobs, logical_k)).astype(np.float32))
    rotated_cpu[:, 0] = 448  # Exact amax makes the nonzero row scale exactly 1.
    weight_scale_cpu = torch.zeros_like(rotated_cpu)
    weight_scale_cpu[:, :logical_k] = 1
    order_cpu = torch.from_numpy(np.stack([rng.permutation(k) for _ in range(jobs)]))
    input_bias_cpu = torch.arange(jobs, dtype=torch.float32) * 0.25
    rotated, weight_scale, order, input_bias = (
        value.to(tp2_npu_device) for value in (rotated_cpu, weight_scale_cpu, order_cpu, input_bias_cpu)
    )
    quantized = torch.empty((jobs, k), dtype=torch.float8_e4m3fn, device=tp2_npu_device)
    scale, bias = (torch.empty(jobs, dtype=torch.float32, device=tp2_npu_device) for _ in range(2))
    valid = torch.empty(jobs, dtype=torch.int32, device=tp2_npu_device)
    for all_zero in (False, True):
        if all_zero:
            rotated.zero_()
            rotated_cpu.zero_()
        quantized.view(torch.uint8).fill_(127)
        scale.fill_(float("nan"))
        bias.fill_(float("nan"))
        valid.zero_()
        binding.prepare_resident_out(rotated, weight_scale, order, input_bias, quantized, scale, bias, valid)
        torch.npu.synchronize()
        expected = torch.gather(rotated_cpu.to(torch.float8_e4m3fn).view(torch.uint8), 1, order_cpu)
        actual = quantized.cpu().view(torch.uint8)
        assert torch.equal(actual, expected)
        assert bool((actual[order_cpu >= logical_k] == 0).all())
        assert torch.equal(scale.cpu(), torch.full((jobs,), 1e-12 if all_zero else 1.0))
        assert torch.equal(bias.cpu().view(torch.int32), input_bias_cpu.view(torch.int32))
        assert torch.equal(valid.cpu(), torch.ones(jobs, dtype=torch.int32))


@pytest.mark.parametrize("n,k", TP2_SHAPES)
@pytest.mark.parametrize("jobs", [1, 6])
def test_tp2_native_device_out_dirty_reuse_nondefault_stream_and_guards(tp2_npu_device, n, k, jobs):
    stream = torch.npu.Stream(device=tp2_npu_device)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        inputs, expected = _device_case(n, k, 1, jobs, tp2_npu_device)
        backing = [torch.full((n + 64,), 123, dtype=torch.bfloat16, device=tp2_npu_device) for _ in range(jobs)]
        outputs = [value[32:-32].view(1, n) for value in backing]
        owners = [value for row in inputs for value in row] + backing + outputs
        records = [[*(value.data_ptr() for value in row), out.data_ptr(), 1, n, k] for row, out in zip(inputs, outputs)]
        descriptors = torch.tensor(records, dtype=torch.int64, device=tp2_npu_device)
        for _ in range(3):
            for out in outputs:
                out.fill_(float("nan"))
            binding.grouped_projection_resident_out(descriptors, owners, jobs=jobs, m=1, n=n, k=k, tp_size=2)
            stream.synchronize()
            for out, reference, base in zip(outputs, expected, backing):
                assert torch.equal(out.cpu().view(torch.int16), reference.view(torch.int16))
                assert bool((base[:32] == 123).all()) and bool((base[-32:] == 123).all())


@pytest.mark.parametrize("k", [1024, 1280, 1536, 1792])
def test_tp2_native_device_raw_binding_rejects_unsafe_k_without_launch(tp2_npu_device, k):
    n = 4096
    x = torch.empty((1, k), dtype=torch.float8_e4m3fn, device=tp2_npu_device)
    scale = torch.ones(1, dtype=torch.float32, device=tp2_npu_device)
    packed = torch.empty((n // 32, k // 16, 16, 8), dtype=torch.uint8, device=tp2_npu_device)
    table = torch.empty((k // 256, n // 32, 32), dtype=torch.uint8, device=tp2_npu_device)
    with pytest.raises(RuntimeError, match="padded"):
        torch.ops.vq2a8_ascendc_v3.grouped_projection_resident([x], [scale], [scale], [packed], [table])
    # Raw pointer values are never consumed: shape validation must fail first.
    descriptors = torch.zeros((1, 9), dtype=torch.int64, device=tp2_npu_device)
    with pytest.raises(RuntimeError, match="padded"):
        torch.ops.vq2a8_ascendc_v3.grouped_projection_resident_out(descriptors, [x], 1, 1, n, k)
