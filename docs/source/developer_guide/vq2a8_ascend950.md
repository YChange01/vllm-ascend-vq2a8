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

The initial AscendC implementation dynamically quantizes activation rows to
FP8 E4M3, looks up VQ2 vectors without creating a dense weight matrix,
accumulates in FP32, and fuses SwiGLU. The down operator uses the same direct
lookup-matmul primitive, applies routing weights, reduces rows back to token
order, and returns the local TP contribution. vLLM performs the TP all-reduce
after the down operator. This first implementation is a correctness baseline;
replace its scalar codebook gather/dot loop with tiled Cube work only after the
fixture checks below pass.

The CPU reference and repack unit tests are the golden contract for the CANN
kernel. Kernel results should first be checked against them at small shapes, then
against real layer 0 and layer 3 payloads before performance tuning.

## Small bring-up assets

Do not use a uniformly down-sized toy matrix to tune the production kernel. Use
two complementary assets while the full checkpoint is still being transferred:

```bash
# A four-layer/eight-expert checkpoint for vLLM loading and TP4 smoke tests.
python tools/create_vq2a8_mini_checkpoint.py \
  --source-model /path/to/DeepSeek-V4-Flash-VQ2A8-32x256 \
  --output /path/to/DeepSeek-V4-Flash-VQ2A8-mini-4l-8e-tp4 \
  --layers 4 --experts 8 --tp-size 4

# One real layer-3 expert, TP4 payloads, and deterministic CPU goldens.
python tools/export_vq2a8_kernel_fixture.py \
  --input /path/to/DeepSeek-V4-Flash-VQ2A8-32x256/experts_vq \
  --output /path/to/vq2a8-layer3-expert0-tp4 \
  --layer 3 --expert 0 --tp-size 4
```

Layers 0 through 2 use hash routing and their canonical artifacts contain only
expert 0. Layer 3 is therefore the first representative 256-expert layer for
operator work. The mini checkpoint keeps layers 0 through 3 so it covers the
uncompressed, c4, and c128 attention patterns; it is a functional integration
fixture, not an accuracy model.

Start the Mini checkpoint on an Ascend 950 host with the correctness-first
settings:

```bash
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /path/to/DeepSeek-V4-Flash-VQ2A8-mini-4l-8e-tp4 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --max-model-len 128 \
  --max-num-seqs 2 \
  --additional-config '{"enable_dsa_cp": true}' \
  --enforce-eager
```

DeepSeek V4 on Ascend 950 does not support standalone TP-only attention. With
TP4 the ordinary DSA metadata builder would pass 16 rank-local query heads,
while the Ascend sparse-attention metadata operator accepts 64 or 128 heads.
FlashComm1 plus DSA-CP keeps the global 64-head attention contract and is
therefore required even for the Mini TP4 smoke test.

After installing the compiled operators, validate gate/up on one TP rank first:

```bash
python benchmarks/ops/benchmark_vq2a8_ascend950.py \
  --fixture /path/to/vq2a8-layer3-expert0-tp4 \
  --tp-size 4 --tp-rank 0 --stage gate-up --iterations 20
```

Then enable down/reduce and benchmark the complete local MoE path:

```bash
python benchmarks/ops/benchmark_vq2a8_ascend950.py \
  --fixture /path/to/vq2a8-layer3-expert0-tp4 \
  --tp-size 4 --tp-rank 0 --stage full --iterations 100 \
  --json-output vq2a8-rank0.json
```
