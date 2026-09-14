# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental two-launch batched FWHT preparation, NOT MXFP8 quantization.

The row-wise FP32 activation scale and bias correction contract is preserved.
Butterfly accumulation differs from GEMM: bit-exact model regression is a gate,
not an assumption. No CPU/CUDA runtime fallback is supplied.
"""

import torch

from vllm_ascend.quantization.vq2a8_activation import RowwiseVQ2A8Preparation


def butterfly_reference(x, sign, block=128):
    """CPU-testable mathematical oracle; never selected as an NPU fallback."""
    if block <= 0 or block & (block - 1) or x.shape[-1] % block:
        raise ValueError("Invalid butterfly geometry")
    value = (x.float() * sign.float()).reshape(-1, block)
    stride = 1
    while stride < block:
        pairs = value.reshape(-1, 2, stride)
        left, right = pairs[:, 0], pairs[:, 1]
        value = torch.stack((left + right, left - right), 1).reshape(-1, block)
        stride *= 2
    return (value * block**-0.5).reshape(x.shape)


class BatchedFWHTPreparation:
    def __init__(self, *, validity):
        self.validity = validity

    def rows(self, hidden, payload, spec):
        return self.many([(hidden, payload, spec)])[0]

    def pack(self, requests):
        if not 1 <= len(requests) <= 6:
            raise ValueError("FWHT preparation requires 1..6 jobs")
        first, _, spec = requests[0]
        if first.device.type != "npu":
            raise ValueError("FWHT runtime is NPU-only")
        width, true_width = spec.columns, spec.rht_true_columns
        if spec.rht_block_size != 128 or width % 512 or not 0 < true_width <= width <= 8192:
            raise ValueError("FWHT requires RHT128, K padded to 512, and K<=8192")
        counts, hidden_rows, scales, biases, signs, row_jobs = [], [], [], [], [], []
        for index, (hidden, payload, current) in enumerate(requests):
            if (
                (current.columns, current.rht_true_columns, current.rht_block_size) != (width, true_width, 128)
                or hidden.ndim != 2
                or not 1 <= hidden.shape[0] <= 32
                or hidden.shape[1] != true_width
                or hidden.device != first.device
                or hidden.dtype != torch.bfloat16
            ):
                raise ValueError("FWHT requires matching BF16 jobs of 1..32 rows")
            padded = torch.nn.functional.pad(hidden, (0, width - true_width)) if width != true_width else hidden
            RowwiseVQ2A8Preparation._check_metadata(
                padded[:1], payload["weight_scale"], payload["weight_bias"], payload["rht_sign"], 128
            )
            counts.append(hidden.shape[0])
            hidden_rows.append(padded)
            scales.append(payload["weight_scale"])
            biases.append(payload["weight_bias"])
            signs.append(payload["rht_sign"])
            row_jobs.extend([index] * hidden.shape[0])
        # Scale/sign/bias are stacked ONCE per job, not once per token row.
        return counts, (
            torch.cat(hidden_rows),
            torch.stack(scales),
            torch.stack(biases),
            torch.stack(signs),
            torch.tensor(row_jobs, dtype=torch.int32, device=first.device),
        )

    def compute(self, packed):
        from vllm_ascend.quantization.vq2a8_activation_triton import prepare

        return prepare(*packed)

    def check(self, packed):
        x, scales, biases, signs, _ = packed
        self.validity(
            torch.isfinite(x).all()
            & torch.isfinite(scales).all()
            & torch.isfinite(biases).all()
            & ((signs == -1) | (signs == 1)).all()
        )

    def many(self, requests):
        counts, packed = self.pack(requests)
        self.check(packed)
        result = self.compute(packed)
        return list(zip(*(value.split(counts) for value in result)))


class PreparationGraph:
    """Two-entry decode-only subgraph cache; NOT a full-model decode graph.

    Own ALL captured input/output buffers; refresh every expert's metadata on
    replay. No expert payload pointer is captured. The caller must stay on one
    stream and consume outputs before the next invocation (native handlers do).
    Cache fills/graph capture belong to warmup, never accepted measured samples.
    """

    def __init__(self, preparation):
        self.preparation = preparation
        self.entries = {}
        self.captures = self.replays = self.bypasses = 0
        self.stream_id = None

    def rows(self, hidden, payload, spec):
        return self.many([(hidden, payload, spec)])[0]

    def many(self, requests):
        counts, packed = self.preparation.pack(requests)
        self.preparation.check(packed)
        current = torch.npu.current_stream()
        stream_id = current.npu_stream
        if self.stream_id is not None and self.stream_id != stream_id:
            raise RuntimeError("Preparation graph buffers require a single owner stream")
        self.stream_id = stream_id
        key = tuple((tuple(t.shape), t.dtype, t.device) for t in packed)
        if any(count != 1 for count in counts) or (key not in self.entries and len(self.entries) >= 2):
            self.bypasses += 1
            result = self.preparation.compute(packed)
        else:
            if key not in self.entries:
                static = tuple(t.clone() for t in packed)
                # Compile and allocate on a side stream before capture. Never
                # capture lazy loading, host validity decisions or route reads.
                side = torch.npu.Stream()
                side.wait_stream(current)
                with torch.npu.stream(side):
                    for _ in range(2):
                        self.preparation.compute(static)
                current.wait_stream(side)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    result = self.preparation.compute(static)
                self.entries[key] = (graph, static, result)
                self.captures += 1
            graph, static, result = self.entries[key]
            for dst, src in zip(static, packed):
                dst.copy_(src)
            graph.replay()
            self.replays += 1
        return list(zip(*(value.split(counts) for value in result)))

    def report(self):
        return dict(
            captures=self.captures,
            replays=self.replays,
            bypasses=self.bypasses,
            entries=len(self.entries),
            scope="activation_preparation_only_not_full_model",
        )
