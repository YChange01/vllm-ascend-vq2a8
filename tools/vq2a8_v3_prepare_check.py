# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mandatory real-NPU numerical preflight for the V3 fused preparation mode."""

import torch

PREPARE_PREFLIGHT_CASES = (
    "prepare_bytes_reuse_and_chunk_boundaries",
    "prepare_midpoints_negative_zero_and_bias",
    "prepare_nonfinite_and_int64_order_validation",
)


def _outputs(jobs, k, device):
    return (
        torch.empty((jobs, k), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((jobs,), dtype=torch.float32, device=device),
        torch.empty((jobs,), dtype=torch.float32, device=device),
        torch.empty((jobs,), dtype=torch.int32, device=device),
    )


def _require(condition, message):
    if not condition:
        raise RuntimeError(f"V3 fused preparation preflight failed: {message}")


def _compare(prepare, rotated, weight_scale, order, input_bias, outputs):
    transformed = rotated * weight_scale
    expected_scale = torch.clamp(transformed.abs().amax(dim=-1) / 448.0, min=1e-12)
    expected = torch.clamp(transformed / expected_scale[:, None], -448.0, 448.0).to(torch.float8_e4m3fn)
    expected_bytes = torch.gather(expected.view(torch.uint8), 1, order)
    addresses = tuple(tensor.data_ptr() for tensor in outputs)
    prepare(rotated, weight_scale, order, input_bias, *outputs)
    _require(tuple(tensor.data_ptr() for tensor in outputs) == addresses, "output storage changed")
    quantized, scale, bias, valid = outputs
    _require(torch.equal(quantized.view(torch.uint8).cpu(), expected_bytes.cpu()), "FP8 bytes differ from eager")
    _require(
        torch.equal(scale.view(torch.int32).cpu(), expected_scale.view(torch.int32).cpu()),
        "FP32 row-scale bits differ from eager",
    )
    _require(torch.equal(bias.view(torch.int32).cpu(), input_bias.view(torch.int32).cpu()), "FP32 bias bits changed")
    _require(torch.equal(valid.cpu(), torch.ones_like(valid, device="cpu")), "valid input rejected")


def check_prepare_reuse(prepare):
    generator = torch.Generator().manual_seed(1927)
    for jobs, k in ((1, 512), (6, 2048), (6, 2560), (6, 4096), (1, 65536)):
        rotated = torch.randn((jobs, k), generator=generator, dtype=torch.float32).to("npu")
        weight_scale = torch.randn((jobs, k), generator=generator, dtype=torch.float32).to("npu")
        order = torch.stack([torch.randperm(k, generator=generator) for _ in range(jobs)]).to("npu")
        input_bias = torch.randn((jobs,), generator=generator, dtype=torch.float32).to("npu")
        outputs = _outputs(jobs, k, rotated.device)
        _compare(prepare, rotated, weight_scale, order, input_bias, outputs)
        rotated.zero_()
        input_bias.zero_()
        _compare(prepare, rotated, weight_scale, order, input_bias, outputs)
        rotated.fill_(1e-16)
        _compare(prepare, rotated, weight_scale, order, input_bias, outputs)


def check_prepare_midpoints(prepare):
    # Max=448 fixes the row scale at one and isolates FP8 round-to-even ties,
    # including subnormal values and signed zero, from scale computation.
    positive = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    midpoints = (positive[:-1] + positive[1:]) / 2
    source = torch.cat((midpoints, -midpoints, torch.tensor([0.0, -0.0, 448.0, -448.0])))
    rotated = source.repeat((512 + source.numel() - 1) // source.numel())[:512].reshape(1, 512).to("npu")
    weight_scale = torch.ones_like(rotated)
    order = torch.arange(511, -1, -1, dtype=torch.int64, device="npu").reshape(1, 512)
    input_bias = torch.tensor([-0.0], dtype=torch.float32, device="npu")
    _compare(prepare, rotated, weight_scale, order, input_bias, _outputs(1, 512, rotated.device))


def check_prepare_invalid(prepare):
    rotated = torch.ones((1, 512), dtype=torch.float32, device="npu")
    weight_scale = torch.ones_like(rotated)
    order = torch.arange(512, dtype=torch.int64, device="npu").reshape(1, 512)
    input_bias = torch.zeros((1,), dtype=torch.float32, device="npu")
    outputs = _outputs(1, 512, rotated.device)
    for bad_index in (-1, 512, 2**32 + 7, -(2**32) + 7, 2**63 - 1):
        order[0, 0] = bad_index
        prepare(rotated, weight_scale, order, input_bias, *outputs)
        _require(outputs[-1].cpu().tolist() == [0], f"invalid int64 order accepted: {bad_index}")
    order[0, 0] = 0
    for tensor in (rotated, weight_scale, input_bias):
        for value in (float("inf"), -float("inf"), float("nan")):
            tensor.reshape(-1)[0] = value
            prepare(rotated, weight_scale, order, input_bias, *outputs)
            _require(outputs[-1].cpu().tolist() == [0], "nonfinite input accepted")
            tensor.reshape(-1)[0] = 1.0
    rotated[0, 0] = torch.finfo(torch.float32).max
    weight_scale[0, 0] = 2.0
    prepare(rotated, weight_scale, order, input_bias, *outputs)
    _require(outputs[-1].cpu().tolist() == [0], "FP32 product overflow accepted")


def run_prepare_preflight():
    """Run exact byte/scale/bias and safety checks; never skip or fall back.

    The caller must have loaded the explicitly pinned candidate in this process.
    Successful return proves these native preparation cases only, not full-model
    accuracy, graph safety, or latency. Comparisons deliberately synchronize.
    """
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Fused V3 preparation preflight requires real Ascend NPU hardware.")
    namespace = torch.ops.vq2a8_ascendc_v3
    if not all(hasattr(namespace, name) for name in ("prepare_out", "resident_capabilities", "resident_abi_version")):
        raise RuntimeError("Load a rebuilt, pinned V3 library with the fused preparation ABI before preflight.")
    if namespace.resident_abi_version() != 1 or namespace.resident_capabilities() & 2 != 2:
        raise RuntimeError("The pinned V3 library does not provide resident ABI 1 fused preparation.")
    cases = []
    checks = (check_prepare_reuse, check_prepare_midpoints, check_prepare_invalid)
    for name, check in zip(PREPARE_PREFLIGHT_CASES, checks):
        check(namespace.prepare_out)
        cases.append({"name": name, "passed": True, "exactbitwise": True})
    return {
        "passed": True,
        "exactbitwise": True,
        "device_execution_verified": True,
        "scope": "fused_preparation_after_rht_and_bias_gemv",
        "cases": cases,
    }
