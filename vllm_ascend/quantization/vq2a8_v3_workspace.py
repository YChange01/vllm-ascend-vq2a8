# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owned V3 workspaces for the register pair-LUT projection.

The on-disk artifact stays unchanged. Conversion happens once on the host;
decode selects device pointers and writes quantized bytes in the converted K
order. Dense RHT and bias GEMV retain their original per-row geometry.
"""

import torch

from vllm_ascend.quantization.vq2a8_ascendc_v3 import (
    RESIDENT_JOB_WORDS,
    grouped_projection_resident_out,
    prepare_resident_out,
)
from vllm_ascend.quantization.vq2a8_reference import VQ2_FP8_MIN_SCALE

RESIDENT_FIELDS = (
    ("packed_zn", torch.uint8, 1),
    ("pair_lut", torch.uint8, 1),
    ("activation_order", torch.int64, 8),
    ("weight_scale", torch.float32, 4),
    ("weight_bias", torch.float32, 4),
    ("rht_sign", torch.int8, 1),
)
RESIDENT_POINTER_FIELDS = ("packed_zn", "pair_lut")
RESIDENT_SELECTED_FIELDS = ("weight_scale", "weight_bias", "rht_sign", "activation_order")


def resident_shapes(layer, kind):
    """Validate source headers and describe the final converted bank geometry."""
    spec = layer.specs[kind]
    n, k, experts = spec.rows, spec.columns, len(layer.expert_ids)
    if n != 4096 or k not in (2048, 4096) or spec.rht_block_size != 128:
        raise ValueError("Resident V3 pair-LUT requires N4096/K2048-or-4096 and RHT128.")
    if not 0 < spec.rht_true_columns <= k:
        raise ValueError("Invalid resident true input width.")
    shapes = layer.tensor_shapes
    if shapes[f"{kind}_packed_indices"] != (experts, n // 2, k // 8):
        raise ValueError("Resident packed source header mismatch.")
    books = shapes[f"{kind}_codebooks"]
    if len(books) != 5 or books[0] != experts or not 1 <= books[1] <= 256 or books[2:] != (n // 32, 16, 2):
        raise ValueError("Resident codebook source header mismatch.")
    if any(
        shapes[f"{kind}_{field}"] != (experts, k)
        for field in ("codebook_tile_ids", "weight_scale", "weight_bias", "rht_sign")
    ):
        raise ValueError("Resident metadata source header mismatch.")
    return {
        "packed_zn": (experts, n // 32, k // 16, 16, 8),
        "pair_lut": (experts, k // 256, n // 32, 32),
        **{field: (experts, k) for field in RESIDENT_SELECTED_FIELDS},
    }


def resident_workspace_sizes(experts, jobs, n, k):
    """Allocation requests, individually rounded by the caller's budget model."""
    return (
        experts * 2 * 8,
        jobs * 2 * 8,  # pointer bank and selected pointers
        jobs * k * 2,
        jobs * k,  # BF16 hidden and permuted FP8 output
        jobs * 4,
        jobs * 4,
        jobs * n * 2,  # scale, bias, projection output
        jobs * RESIDENT_JOB_WORDS * 8,
        jobs * k * 4,
        jobs * k * 4,
        jobs * k,
        jobs * k * 8,  # selected metadata
        jobs * k * 4,
        jobs * k * 4,
        jobs * k * 4,
        jobs * k * 4,  # FP32 preparation
        jobs * 4,
        jobs * 4,  # original bias GEMV and native validity
    )


class ResidentV2ProjectionWorkspace:
    """One fixed projection bank; output is borrowed on the owner's stream."""

    def __init__(
        self,
        payloads,
        spec,
        jobs,
        preparation,
        *,
        banks,
        preparation_mode="eager",
        launcher=grouped_projection_resident_out,
        prepare_launcher=prepare_resident_out,
    ):
        if preparation_mode not in ("eager", "fused"):
            raise ValueError("V3 preparation must be eager or fused.")
        self.spec, self.jobs, self.preparation = spec, jobs, preparation
        self.launcher, self.prepare_launcher = launcher, prepare_launcher
        self.preparation_mode = preparation_mode
        self.banks, self.payloads = banks, tuple(payloads)
        self.device = banks["weight_scale"].device
        self.n, self.k = spec.rows, spec.columns
        self.selected = {
            field: torch.empty((jobs, self.k), device=self.device, dtype=banks[field].dtype)
            for field in RESIDENT_SELECTED_FIELDS
        }
        self.hidden = torch.empty((jobs, self.k), device=self.device, dtype=torch.bfloat16)
        self.x = torch.empty((jobs, self.k), device=self.device, dtype=torch.float8_e4m3fn)
        self.scale = torch.empty(jobs, device=self.device, dtype=torch.float32)
        self.bias = torch.empty_like(self.scale)
        self.input_bias = torch.empty_like(self.scale)
        self.valid = torch.ones(jobs, device=self.device, dtype=torch.int32)
        self.output = torch.empty((jobs, self.n), device=self.device, dtype=torch.bfloat16)
        self.float_hidden = torch.empty((jobs, self.k), device=self.device, dtype=torch.float32)
        self.float_sign = torch.empty_like(self.float_hidden)
        self.signed = torch.empty_like(self.float_hidden)
        self.rotated = torch.empty_like(self.float_hidden)
        self.pointer_bank = torch.tensor(
            [[payload[field].data_ptr() for field in RESIDENT_POINTER_FIELDS] for payload in payloads],
            dtype=torch.int64,
            device="cpu",
        ).to(self.device)
        self.selected_pointers = torch.empty((jobs, 2), device=self.device, dtype=torch.int64)
        records = [
            [
                self.x[job].data_ptr(),
                self.scale[job:].data_ptr(),
                self.bias[job:].data_ptr(),
                0,
                0,
                self.output[job].data_ptr(),
                1,
                self.n,
                self.k,
            ]
            for job in range(jobs)
        ]
        self.descriptors = torch.tensor(records, dtype=torch.int64, device="cpu").to(self.device)
        self.owners = (
            self.hidden,
            self.x,
            self.scale,
            self.bias,
            self.output,
            self.pointer_bank,
            self.selected_pointers,
            self.float_hidden,
            self.float_sign,
            self.signed,
            self.rotated,
            self.input_bias,
            self.valid,
            *self.banks.values(),
            *self.selected.values(),
        )

    def project(self, hidden, slots):
        if hidden.shape not in ((1, self.spec.rht_true_columns), (self.jobs, self.spec.rht_true_columns)):
            raise ValueError("Resident decode requires one input row or one per job.")
        if (
            hidden.device != self.device
            or hidden.dtype != torch.bfloat16
            or slots.shape != (self.jobs,)
            or slots.dtype != torch.int64
            or slots.device != self.device
        ):
            raise ValueError("Resident decode input/slot metadata mismatch.")
        self.hidden[:, : self.spec.rht_true_columns].copy_(hidden)
        if self.spec.rht_true_columns < self.k:
            self.hidden[:, self.spec.rht_true_columns :].zero_()
        for field in RESIDENT_SELECTED_FIELDS:
            torch.index_select(self.banks[field], 0, slots, out=self.selected[field])
        torch.index_select(self.pointer_bank, 0, slots, out=self.selected_pointers)
        self.descriptors[:, 3:5].copy_(self.selected_pointers)
        self._prepare_selected()
        self.launcher(self.descriptors, self.owners, jobs=self.jobs, m=1, n=self.n, k=self.k)
        return self.output

    def _prepare_selected(self):
        self.float_hidden.copy_(self.hidden)
        sign = self.selected["rht_sign"]
        weight_scale, weight_bias = self.selected["weight_scale"], self.selected["weight_bias"]
        valid = (
            torch.isfinite(self.float_hidden).all()
            & torch.isfinite(weight_scale).all()
            & torch.isfinite(weight_bias).all()
            & ((sign == -1) | (sign == 1)).all()
        )
        self.preparation._validate(valid)
        self.float_sign.copy_(sign)
        torch.mul(self.float_hidden, self.float_sign, out=self.signed)
        block = self.spec.rht_block_size
        self.preparation._ensure_hadamard(self.device, block)
        for index in range(self.jobs):
            # Keep the original [1,K/128,128] @ [128,128] geometry, and its
            # following one-row bias dot. Only destinations are preallocated.
            torch.matmul(
                self.signed[index : index + 1].view(1, self.k // block, block),
                self.preparation._hadamard,
                out=self.rotated[index : index + 1].view(1, self.k // block, block),
            )
            torch.mv(self.rotated[index : index + 1], weight_bias[index], out=self.input_bias[index : index + 1])
        if self.preparation_mode == "fused":
            self.prepare_launcher(
                self.rotated,
                weight_scale,
                self.selected["activation_order"],
                self.input_bias,
                self.x,
                self.scale,
                self.bias,
                self.valid,
            )
            self.preparation._validate((self.valid == 1).all())
        else:
            transformed = self.rotated * weight_scale
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
            scale = torch.clamp(transformed.abs().amax(dim=-1) / fp8_max, min=VQ2_FP8_MIN_SCALE)
            quantized = torch.clamp(transformed / scale.unsqueeze(-1), -fp8_max, fp8_max).to(torch.float8_e4m3fn)
            # A single batched byte gather, with no float8 indexing kernel or
            # intermediate destination. K order changes AFTER quantization.
            torch.gather(
                quantized.view(torch.uint8), 1, self.selected["activation_order"], out=self.x.view(torch.uint8)
            )
            self.scale.copy_(scale)
            self.bias.copy_(self.input_bias)
