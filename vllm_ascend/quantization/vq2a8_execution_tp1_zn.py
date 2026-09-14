# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-width TP1 packed-zN direct residency, without conversion or collectives."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from vllm_ascend.quantization.vq2a8_execution_v3 import AscendCV3VQ2TP1MoE
from vllm_ascend.quantization.vq2a8_v3_workspace import resident_shapes
from vllm_ascend.quantization.vq2a8_zn_contract import VQ2_TP1_ZN_FORMAT


@dataclass(frozen=True)
class TP1ZNComputeLayer:
    layer_index: int
    expert_ids: tuple[int, ...]
    specs: Mapping
    tensor_shapes: Mapping
    format: str = VQ2_TP1_ZN_FORMAT
    tp_size: int = 1

    @classmethod
    def from_disk(cls, layer):
        if not layer.expert_ids:
            raise ValueError("TP1 packed-zN requires a nonempty expert bank.")
        specs, shapes = {}, {}
        for kind in ("gate_up", "down"):
            first = layer.spec_for(layer.expert_ids[0], kind)
            for expert in layer.expert_ids:
                spec = layer.spec_for(expert, kind)
                if (
                    spec.metadata.get("format") != VQ2_TP1_ZN_FORMAT
                    or spec.metadata.get("tp_size") != 1
                    or spec.tp_rank != 0
                    or spec.canonical_shape != spec.logical_shape
                    or spec.logical_shape != spec.packed_shape
                    or spec.rht_block_size != 128
                    or spec.tensor_shapes != first.tensor_shapes
                ):
                    raise ValueError("TP1 packed-zN compute bank requires full-width, unpadded rank-zero payloads.")
            specs[kind] = first
            shapes.update(
                {f"{kind}_{field}": (len(layer.expert_ids), *shape) for field, shape in first.tensor_shapes.items()}
            )
        result = cls(layer.layer_index, tuple(layer.expert_ids), MappingProxyType(specs), MappingProxyType(shapes))
        for kind in specs:
            resident_shapes(result, kind)
        return result


class AscendCV3VQ2TP1ZNMoE(AscendCV3VQ2TP1MoE):
    """Retain TP1 launch/graph/budget semantics and stream prepacked CPU shards."""

    def __init__(self, artifact, layer_index, device, *, projection_kernel="v2", **kwargs):
        if projection_kernel != "v2":
            raise ValueError("TP1 packed-zN requires the V3 V2 projection core; no legacy fallback.")
        if artifact.manifest.get("format") != VQ2_TP1_ZN_FORMAT or artifact.tp_rank != 0:
            raise ValueError("TP1 packed-zN runtime requires a validated rank-zero TP1 packed-zN artifact.")
        compute = TP1ZNComputeLayer.from_disk(artifact.layer(layer_index))
        super().__init__(artifact, layer_index, device, projection_kernel=projection_kernel, **kwargs)
        self.layer = compute

    def _resident_payloads(self):
        # One safe_open per shard. CPU value checks finish before synchronous
        # copies into final device banks; no convert_expert_payload or TP2 pad.
        for shard in self.artifact.iter_rank_shards(self.layer_index, device="cpu"):
            yield from shard.items()

    def resident_report(self):
        return {**super().resident_report(), "artifact_format": VQ2_TP1_ZN_FORMAT, "payload_load": "prepacked_direct"}
