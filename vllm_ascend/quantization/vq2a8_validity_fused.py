# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in exact layer-validity aggregation, separate from activation math.

Native sign/select/project already return fresh INT32 status vectors. Keep
them unreduced and scan the two projections and final combined BF16 output in
one device kernel. Routing checks and every safe gather remain unchanged.
"""

from __future__ import annotations

import torch

LAYER_VALIDITY_ABI = 1
STATUS_COUNT = 6
OUTPUT_COUNT = 3
MAX_ROUTE_FLAGS = 8
MAX_GROUPS = 6
OUTPUT_WIDTHS = (2048, 4096)


class FusedLayerValidity:
    """Native ABI gate with no host scalar read or implicit Torch fallback."""

    def __init__(self, native_ops=None):
        native = torch.ops.vq2a8_ascendc_v4_v2 if native_ops is None else native_ops
        try:
            version = native.layer_validity_version()
            operation = native.layer_validity
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError("Fused layer validity requires rebuilt ABI 1; no implicit fallback.") from error
        if type(version) is not int or version != LAYER_VALIDITY_ABI:
            raise RuntimeError(f"Unsupported layer validity ABI {version}; require {LAYER_VALIDITY_ABI}.")
        self._operation = operation

    def __call__(self, statuses, outputs, route_flags):
        validate_inputs(statuses, outputs, route_flags)
        return self._operation(list(statuses), list(outputs), list(route_flags))


def validate_inputs(statuses, outputs, route_flags):
    """Metadata checks only. Native binding also independently enforces them."""
    if len(statuses) != STATUS_COUNT or len(outputs) != OUTPUT_COUNT or len(route_flags) > MAX_ROUTE_FLAGS:
        raise ValueError("Layer validity needs six statuses, three outputs, and at most eight route flags.")
    first = statuses[0]
    if not isinstance(first, torch.Tensor) or first.ndim != 1 or not 1 <= first.numel() <= MAX_GROUPS:
        raise ValueError("Layer validity statuses require INT32[G], G 1..6.")
    groups = first.numel()
    for status in statuses:
        if (
            not isinstance(status, torch.Tensor)
            or status.dtype != torch.int32
            or status.shape != (groups,)
            or status.device != first.device
            or not status.is_contiguous()
        ):
            raise ValueError("Layer validity statuses must be matching contiguous INT32[G].")
    for index, output in enumerate(outputs):
        expected_rows = 1 if index == 2 else groups
        if (
            not isinstance(output, torch.Tensor)
            or output.dtype != torch.bfloat16
            or output.device != first.device
            or not output.is_contiguous()
            or not (output.ndim == 2 or (index != 2 and output.ndim == 3 and output.shape[1] == 1))
            or output.shape[0] != expected_rows
            or output.shape[-1] not in OUTPUT_WIDTHS
            or output.data_ptr() % 32
        ):
            raise ValueError("Layer validity outputs require aligned contiguous BF16[G,N]/[G,1,N] and result[1,H].")
    if outputs[1].shape[-1] != outputs[2].shape[-1]:
        raise ValueError("Down and combined output widths must match.")
    for flag in route_flags:
        if (
            not isinstance(flag, torch.Tensor)
            or flag.dtype != torch.bool
            or flag.ndim != 0
            or flag.device != first.device
        ):
            raise ValueError("Layer validity route flags require matching BOOL scalars.")
