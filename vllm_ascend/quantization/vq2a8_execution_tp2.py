# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-local offline VQ2 TP2 banks using the V2 core packaged in V3.

Only routed experts are tensor-sharded. Router/hash/shared roots are replicated.
One TP SUM of FP32 routed output precedes shared addition, in both prefill and
decode. Attention retains the parent's own TP collectives. No EP/cache fallback,
TP1 numerical-equivalence claim or graph-collective capture is implied.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType

import torch

from vllm_ascend.quantization.vq2a8_ascendc_v3 import (
    grouped_projection_resident,
    grouped_projection_resident_out,
    resident_library_capabilities,
)
from vllm_ascend.quantization.vq2a8_execution_v3 import AscendCV3VQ2TP1MoE
from vllm_ascend.quantization.vq2a8_v3_workspace import (
    RESIDENT_FIELDS,
    ResidentV2ProjectionWorkspace,
    resident_shapes,
)


@dataclass(frozen=True)
class TP2ComputeSpec:
    rows: int
    columns: int
    rht_true_columns: int
    rht_block_size: int = 128


@dataclass(frozen=True)
class TP2ComputeLayer:
    """Uniform launch bank geometry; original per-expert disk specs are retained."""

    layer_index: int
    expert_ids: tuple[int, ...]
    specs: Mapping[str, TP2ComputeSpec]
    tp_size: int = 2

    @classmethod
    def from_disk(cls, layer):
        expected = {
            "gate_up": TP2ComputeSpec(2048, 4096, 4096),
            "down": TP2ComputeSpec(4096, 2048, 1024),
        }
        for expert in layer.expert_ids:
            for kind, compute in expected.items():
                disk = layer.spec_for(expert, kind)
                if (
                    disk.rows != compute.rows
                    or disk.rht_true_columns != compute.rht_true_columns
                    or disk.rht_block_size != compute.rht_block_size
                    or disk.columns % 256
                    or not compute.rht_true_columns <= disk.columns <= compute.columns
                ):
                    raise ValueError(f"TP2 layer {layer.layer_index} expert {expert} {kind} is not V2-compatible.")
        result = cls(layer.layer_index, tuple(layer.expert_ids), MappingProxyType(expected))
        for kind in expected:
            resident_shapes(result, kind)
        return result


def copy_tp2_payload(destination, source, disk_spec, compute_spec):
    """Copy prepacked bytes into final banks, padding only whole K256 blocks.

    Metadata is in extended local physical-input order; activation_order maps
    it to packed K. Appended physical columns have scale=bias=0 and sign=+1,
    so RHT128, local A8 max and the down bias stay unchanged. This is not a
    dense decode/requantization, nor a second offline permutation.
    """
    n, k, disk_k = compute_spec.rows, compute_spec.columns, disk_spec.columns
    if (
        disk_spec.rows != n
        or disk_spec.rht_true_columns != compute_spec.rht_true_columns
        or disk_spec.rht_block_size != compute_spec.rht_block_size
        or compute_spec.rht_block_size != 128
        or disk_k % 256
        or not compute_spec.rht_true_columns <= disk_k <= k
    ):
        raise ValueError("Invalid TP2 disk/compute padding contract.")
    fields = {name for name, _, _ in RESIDENT_FIELDS}
    if set(source) != fields or set(destination) != fields:
        raise ValueError("TP2 resident payload requires exactly six fields.")
    shapes = {
        "packed_zn": (n // 32, k // 16, 16, 8),
        "pair_lut": (k // 256, n // 32, 32),
        **{field: (k,) for field in ("activation_order", "weight_scale", "weight_bias", "rht_sign")},
    }
    disk_shapes = dict(shapes)
    disk_shapes.update(packed_zn=(n // 32, disk_k // 16, 16, 8), pair_lut=(disk_k // 256, n // 32, 32))
    for field in ("activation_order", "weight_scale", "weight_bias", "rht_sign"):
        disk_shapes[field] = (disk_k,)
    for field, dtype, _ in RESIDENT_FIELDS:
        if (
            destination[field].shape != shapes[field]
            or destination[field].dtype != dtype
            or source[field].shape != disk_shapes[field]
            or source[field].dtype != dtype
            or source[field].device.type != "cpu"
        ):
            raise ValueError(f"TP2 resident copy shape/dtype/host mismatch: {field}.")
    # CPU loader has already validated bijection, finite values and dummy
    # metadata. Synchronous copies keep a shard alive until its H2D completes.
    for field, _, _ in RESIDENT_FIELDS:
        target = destination[field]
        if disk_k == k:
            target.copy_(source[field])
        elif field == "packed_zn":
            target.zero_()
            target[:, : disk_k // 16].copy_(source[field])
        elif field == "pair_lut":
            target.zero_()
            target[: disk_k // 256].copy_(source[field])
        elif field == "activation_order":
            target[:disk_k].copy_(source[field])
            target[disk_k:].copy_(torch.arange(disk_k, k, dtype=target.dtype, device="cpu"))
        else:
            target.fill_(1 if field == "rht_sign" else 0)
            target[:disk_k].copy_(source[field])


class AscendCV3VQ2TP2MoE(AscendCV3VQ2TP1MoE):
    """Reuse rank-local V3 lifetime/preparation guards, with explicit TP2 SUM."""

    def __init__(self, artifact, layer_index, device, *, tp_rank, tp_group, **kwargs):
        if type(tp_rank) is not int or tp_rank not in (0, 1):
            raise ValueError("TP2 requires rank 0 or 1.")
        if getattr(tp_group, "world_size", None) != 2 or getattr(tp_group, "rank_in_group", None) != tp_rank:
            raise ValueError("TP2 requires the matching initialized vLLM tensor-parallel group.")
        if not callable(getattr(tp_group, "all_reduce", None)) or getattr(artifact, "tp_rank", None) != tp_rank:
            raise ValueError("TP2 collective/artifact rank mismatch.")
        if kwargs.get("v3_decode_graph", "none") != "none" or kwargs.get("projection_kernel", "v2") != "v2":
            raise ValueError("TP2 currently requires the V2 projection core and no decode graph.")
        if kwargs.pop("tp_size", 2) != 2:
            raise ValueError("TP2 runtime cannot use a TP1 partition.")
        compute_layer = TP2ComputeLayer.from_disk(artifact.layer(layer_index))
        # Parent initializes LOCAL replicated roots and rank-local bookkeeping;
        # it owns no distributed execution. The actual TP group is checked
        # above and every native launch / reduction below explicitly uses TP2.
        super().__init__(artifact, layer_index, device, **kwargs)
        self.layer = compute_layer
        self.tp_rank, self.tp_size, self.tp_group = tp_rank, 2, tp_group
        self.tp_collectives = 0

    def _resident_capabilities(self):
        return resident_library_capabilities(require_tp2=True)

    def _resident_payloads(self):
        for shard in self.artifact.iter_rank_shards(self.layer_index, device="cpu"):
            yield from shard.items()

    def _copy_resident_payload(self, banks, row, kind, host, spec):
        payload = {field: tensor[row] for field, tensor in banks[kind].items()}
        compute = self.layer.specs[kind]
        copy_tp2_payload(payload, host, spec, compute)
        return MappingProxyType(payload), compute

    def _make_resident_workspace(self, payloads, spec, jobs, preparation, banks):
        return ResidentV2ProjectionWorkspace(
            payloads,
            spec,
            jobs,
            preparation,
            banks=banks,
            preparation_mode=self.v3_preparation,
            launcher=partial(grouped_projection_resident_out, tp_size=2),
        )

    def _launch_resident(self, inputs):
        return grouped_projection_resident(inputs, tp_size=2)

    def _reduce_routed(self, result):
        if result.dtype != torch.float32 or result.device != self.device:
            raise ValueError("TP2 SUM requires rank-local FP32 routed output on the owning NPU.")
        # Standard vLLM TP group orders the HCCL stream with its producer and
        # consumer. No CPU copies, per-expert collectives or shared double SUM.
        with self._v3_prefill_state.scope("v3_tp2_routed_sum"):
            reduced = self.tp_group.all_reduce(result.contiguous())
        if reduced.shape != result.shape or reduced.dtype != result.dtype or reduced.device != result.device:
            raise RuntimeError("TP2 collective returned incompatible routed output.")
        self.tp_collectives += 1
        return reduced

    def resident_report(self):
        return {
            **super().resident_report(),
            "tensor_parallel_size": 2,
            "tensor_parallel_rank": self.tp_rank,
            "tp_collectives": self.tp_collectives,
            "shared_policy": "replicated_added_after_routed_sum",
            "activation_quantization": "rank_local_amax",
            "tp1_exact": "not_assumed",
        }
