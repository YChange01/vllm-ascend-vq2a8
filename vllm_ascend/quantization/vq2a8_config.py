# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Quantization configuration for canonical VQ2 routed experts."""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

try:
    from vllm.model_executor.layers.fused_moe import MoERunner
except ImportError:
    from vllm.model_executor.layers.fused_moe import FusedMoE as MoERunner

VQ2A8_METHOD = "vq2a8"


def _is_fused_moe_layer(layer: torch.nn.Module) -> bool:
    return isinstance(layer, MoERunner)


@register_quantization_config(VQ2A8_METHOD)
class AscendVQ2A8Config(QuantizationConfig):
    """Resolve canonical VQ2 artifacts and select compact Ascend MoE loading."""

    def __init__(
        self,
        experts_path: str,
        kernel_path: str = "experts_vq_ascend",
        allow_reference_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.experts_path = experts_path
        self.kernel_path = kernel_path
        self.allow_reference_fallback = allow_reference_fallback

    @classmethod
    def get_name(cls) -> str:
        return VQ2A8_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "Ascend hardware does not use CUDA compute capabilities."
        )

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AscendVQ2A8Config:
        experts_path = cls.get_from_keys(config, ["experts_path"])
        kernel_path = config.get("kernel_path", "experts_vq_ascend")
        allow_reference_fallback = bool(
            config.get("allow_reference_fallback", True)
        )
        return cls(
            str(experts_path),
            str(kernel_path),
            allow_reference_fallback,
        )

    def maybe_update_config(
        self,
        model_name: str,
        hf_config=None,
        revision: str | None = None,
    ) -> None:
        del hf_config, revision
        if not os.path.isabs(self.experts_path):
            self.experts_path = os.path.abspath(
                os.path.join(model_name, self.experts_path)
            )
        if not os.path.isabs(self.kernel_path):
            self.kernel_path = os.path.abspath(
                os.path.join(model_name, self.kernel_path)
            )
        if not os.path.isdir(self.experts_path):
            raise ValueError(f"VQ2 expert artifact not found: {self.experts_path}")

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> QuantizeMethodBase | None:
        if _is_fused_moe_layer(layer):
            from .vq2a8_method import AscendVQ2A8MoEMethod

            return AscendVQ2A8MoEMethod(
                layer.moe_config,
                self.kernel_path,
                prefix,
                tid2eid=tid2eid,
                allow_reference_fallback=self.allow_reference_fallback,
            )
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None
