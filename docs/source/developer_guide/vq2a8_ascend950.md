# VQ2A8 bring-up on Ascend 950

The vLLM-Ascend integration consumes the canonical VPTQ `experts_vq` files and
a tensor-parallel, rank-local derivative named `vq2a8_ascend_tp_v1`. The
canonical files remain the portable source of truth. The derivative is not a
new quantization and does not change any codebook value.

## Prepare TP-local artifacts

Run the repack on the offline Ascend host after copying the model:

```bash
python tools/repack_vq2_ascend.py \
  --input /path/to/model/experts_vq \
  --output /path/to/model/experts_vq_ascend \
  --tp-size 4
```

The command resolves the producer permutation, shards gate/up rows and down
columns with vLLM tensor-parallel semantics, packs eight 4-bit VQ indices per
`int32`, and writes one file per layer and TP rank. It operates one rank at a
time to bound host memory. The default `kernel_path` in `config.json` is
`experts_vq_ascend`.

The source `experts_vq` directory can be retained for reproducibility or removed
after a verified repack when disk capacity is more important. Keep the directory
itself present because model configuration validation uses it as the declared
canonical artifact path.

## Correctness bring-up

The quantization config accepts:

```json
{
  "quant_method": "vq2a8",
  "experts_path": "experts_vq",
  "kernel_path": "experts_vq_ascend",
  "allow_reference_fallback": true
}
```

When the compiled Ascend operators are unavailable, the reference fallback
decodes only the experts selected by the current token batch, performs the two
matmuls in BF16/FP16, and discards the decoded matrices after the layer call. It
does not materialize dense weights for all 256 experts. This path is intended
only to validate model loading and output quality; it is not a throughput
baseline. Set `allow_reference_fallback` to `false` to fail fast unless both
custom operators are installed.

Start the first 950 test with TP4, eager execution, and BF16 KV cache. FP8 KV
cache and graph capture should be enabled only after the VQ2 expert path passes
the smoke and accuracy checks.

## Ascend custom operator contract

The Python integration first looks for these operators in `torch.ops._C_ascend`:

```text
vq2a8_gate_up(
  x, expert_ids,
  packed_indices, codebooks, codebook_tile_ids,
  weight_scale, weight_bias, rht_sign,
  rht_block_size, row_group_size, activation
) -> intermediate

vq2a8_down_reduce(
  intermediate, expert_ids, token_ids, routing_weights,
  packed_indices, codebooks, codebook_tile_ids,
  weight_scale, weight_bias, rht_sign,
  rht_block_size, row_group_size, num_tokens
) -> output
```

For `M = tokens * top_k`, hidden width `H`, local intermediate width `I_tp`,
and local expert count `E`:

| Tensor | Gate/up shape | Down shape |
| --- | --- | --- |
| input | `[M, H]` | `[M, I_tp]` |
| `expert_ids` | `[M]`, int32 | `[M]`, int32 |
| `packed_indices` | `[E, I_tp, ceil(H/8)]` | `[E, H/2, ceil(I_tp/8)]` |
| `codebook_tile_ids` | `[E, H]` | `[E, I_tp]` |
| scale, bias, RHT sign | `[E, H]` | `[E, I_tp]` |
| output | `[M, I_tp]` after SwiGLU | `[tokens, H]` after routed reduction |

`codebooks` retain the canonical `[column_tile, row_tile, 16, 2]` geometry with
an expert dimension prepended. `codebook_tile_ids` maps each TP-local input
column back to its canonical codebook tile, so the device kernel does not need
the original permutation.

The optimized gate/up operator should dynamically quantize activation rows to
the selected Ascend 950 A8 representation, look up VQ2 vectors without creating
a dense weight matrix, accumulate in FP32, and fuse SwiGLU. The down operator
should use the same lookup-matmul primitive, apply routing weights, reduce rows
back to token order, and return the local TP contribution. vLLM performs the TP
all-reduce after the down operator.

The CPU reference and repack unit tests are the golden contract for the CANN
kernel. Kernel results should first be checked against them at small shapes, then
against real layer 0 and layer 3 payloads before performance tuning.
