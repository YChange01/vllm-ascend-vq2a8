# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact integer expert-ID to resident-slot mapping; no routing arithmetic.

Negative or out-of-range IDs map to -1 without reading outside the lookup.
Lookup contents (including duplicate, negative, or too-large slots) pass through
unchanged. The resident bank still enforces its own slot bounds downstream.
"""

from __future__ import annotations

import torch

ROUTE_MAPPING_ABI = 1
MAX_GROUPS = 6
MAX_EXPERTS = 256


def torch_route_mapping(ids, lookup):
    """The unchanged safe-gather expression, used by the default Torch path."""
    in_range = (ids >= 0) & (ids < lookup.numel())
    mapped = lookup.index_select(0, ids.clamp(0, lookup.numel() - 1))
    slots = torch.where(in_range, mapped, -1).contiguous()
    return slots, (slots >= 0).all()


class FusedRouteMapping:
    """Opt-in native ABI gate; no host scalar reads or implicit fallback."""

    def __init__(self, native_ops=None):
        native = torch.ops.vq2a8_ascendc_v4_v2 if native_ops is None else native_ops
        try:
            version = native.route_mapping_version()
            operation = native.route_mapping
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError("Fused route mapping requires rebuilt ABI 1; no implicit fallback.") from error
        if type(version) is not int or version != ROUTE_MAPPING_ABI:
            raise RuntimeError(f"Unsupported route mapping ABI {version}; require {ROUTE_MAPPING_ABI}.")
        self._operation = operation

    def __call__(self, ids, lookup):
        if (
            not isinstance(ids, torch.Tensor)
            or not isinstance(lookup, torch.Tensor)
            or ids.dtype != torch.int64
            or lookup.dtype != torch.int64
            or ids.ndim != 1
            or lookup.ndim != 1
            or not 1 <= ids.numel() <= MAX_GROUPS
            or not 1 <= lookup.numel() <= MAX_EXPERTS
            or ids.device != lookup.device
            or not ids.is_contiguous()
            or not lookup.is_contiguous()
        ):
            raise ValueError("Route mapping requires same-device contiguous INT64 ids[G] and lookup[N], G1..6/N1..256.")
        return self._operation(ids, lookup)
