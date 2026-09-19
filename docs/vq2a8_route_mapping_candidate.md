# V4 v2 integer route-mapping candidate

## Measured baseline and scope

The user reported 20 sequential requests with prompt `你好`, four output tokens,
three warmups, and all of these enabled:

- `--activation-preparation sign_fused_direct`
- `--validity-mode fused`
- `--decoder-metadata-mode position_template`

Mean HTTP TPOT was **37.32485 ms**, versus **39.57920 ms** previously (5.70% lower).
Mean TTFT was 201.94210 ms; late requests had noticeably higher TTFT. These are
user-provided serving results, not new measurements performed by this patch.
The previous 41 ms configuration's profiler is not an exact breakdown of this
37 ms configuration, and must not be rescaled to produce one.

This patch adds two bounded changes:

1. Position templates validate their captured owners directly on every replay.
   This avoids the generic source/copy DAG's temporary dictionaries. Container
   schema, object/alias identity, tensor layout/pointer, scalar constants and CPU
   constant payloads remain checked **before** any live block/slot copy. It also
   rejects same-shape tensor replacement that otherwise leaves the captured
   graph reading the old allocation. No dynamic KV state is cached by position.
2. New opt-in `--route-mapping fused` combines expert-ID bounds checks, resident
   lookup, invalid-ID masking and `all(slots >= 0)` into one integer-only AIV
   kernel per decode MoE layer. Default `torch` keeps the original expression.
   Router scores/top-k/normalization, RHT, bias GEMV, quantization and grouped
   projection arithmetic are unchanged. Multi-token prefill is unchanged.

No NPU compilation, execution or performance acceptance is claimed here. The
30 ms goal is not guaranteed. A stable-view cache was evaluated but not adopted:
its complete parent/view guards were slower in the CPU prototype.

## Mapping contract

ABI 1 accepts contiguous one-dimensional INT64 `ids[G]` and `lookup[N]`, on the
same Ascend950 NPU, with G in 1..6 and N in 1..256. Naturally aligned offset views
are supported. Each result is `lookup[id]` when `0 <= id < N`, otherwise `-1`.
The entire INT64 ID is checked before indexing, including INT64_MIN/MAX.
Duplicate IDs and negative lookup values are preserved. A too-large positive
slot is also preserved: **the resident bank's existing upper-bound checks still
reject it**. The mapping validity alone is not permission to access that slot.

Outputs are fresh INT64 slots and one BOOL scalar. Inputs are not mutated;
captured tensors retain ownership through queued submission, and input storage
is recorded on the caller stream. The native launch resolves the stream before
enqueue. Missing ABI or unsupported options fail without silent fallback.

## Build only the standalone VQ2 library

Use a new build directory. Keep the current `validity-template`, `direct` and
`perf3` libraries. Do not delete the repository's `build/` tree. This patch does
not modify SAS/QLI, so the already rebuilt main package does **not** need another
uninstall/reinstall. Existing prepacked weights are reused unchanged.

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
python -u tools/build_vq2a8_v4_v2.py \
  --soc Ascend950DT_9582 \
  --cann /usr/local/Ascend/cann-9.1.0 \
  --build-dir build/vq2a8-ascendc-v4-v2-route-mapping \
  --jobs 4
```

Continue only after a successful build. Use the actual CANN installation path
if it differs. In each new terminal define:

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
VQ2_ROUTE_LIB="$PWD/build/vq2a8-ascendc-v4-v2-route-mapping/libvq2a8_ascendc_v4_v2.so"
test -f "$VQ2_ROUTE_LIB"
```

## Acceptance on idle physical NPU 1

Stop the existing server normally first. The native probe compares the new
mapper against the unchanged Torch expression and an independent literal
oracle, including invalid indices, changed graph inputs, recovery and queued
owner-release pressure. A driver/graph failure is a failure, not a reason to
skip the test or relax comparisons.

```bash
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_route_mapping.py \
  --library "$VQ2_ROUTE_LIB" \
  --physical-npu 1 --queue-lifetime --timeout-s 300
```

Require `V4_ROUTE_MAPPING=PASS`. Recheck the existing native paths in this new
binary before full-model validation:

```bash
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_validity_fused.py \
  --library "$VQ2_ROUTE_LIB" \
  --physical-npu 1 --queue-lifetime --timeout-s 300
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_activation_packed.py \
  --library "$VQ2_ROUTE_LIB" --physical-npu 1 \
  --preparation-modes sign_fused_strided sign_fused_direct \
  --queue-lifetime --timeout-s 900
```

Then validate the new template checks and fused mapping together. The decoder
validator compares eager with graph for the selected backend; both use the
selected mapper, so this does not replace the independent mapping gate above.
It retains full metadata shadow comparison, true no-shadow execution, repeated
requests and compressor-boundary coverage.

```bash
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_v4_decoder_graph.py \
  --model /home/g00872988/vq2a8 \
  --artifact /home/g00872988/vq2a8/experts_vq_v4_v2_prepacked \
  --library "$VQ2_ROUTE_LIB" \
  --physical-npu 1 --compute-backend v2 \
  --activation-reorder vectorized \
  --activation-preparation sign_fused_direct \
  --validity-mode fused --route-mapping fused \
  --decoder-metadata-mode position_template \
  --kv-cache-mib 256 --reserve-gib 3 --timeout-s 1800
```

Require `V4_DECODER_GRAPH=PASS`, with `route_mapping: fused` in both the receipt
and graph report. No NPU numerical comparison is weakened by this patch.

## Same-binary A/B serving

After all gates pass, use the same rebuilt library, code and every other setting
for both runs. First run `--route-mapping torch`, then stop the server normally
and repeat with `--route-mapping fused`. Repeat A after B if the difference is
small. This isolates mapping fusion; both runs include the owned-template guard
change. Do not use host profiling or an active profiler for latency acceptance.

```bash
TASK_QUEUE_ENABLE=1 python -u tools/serve_vq2a8_v4.py \
  --model /home/g00872988/vq2a8 \
  --artifact /home/g00872988/vq2a8/experts_vq_v4_v2_prepacked \
  --library "$VQ2_ROUTE_LIB" \
  --physical-npu 1 --compute-backend v2 \
  --activation-reorder vectorized --activation-preparation sign_fused_direct \
  --validity-mode fused --route-mapping torch \
  --device-route-decode --decode-graph decoder --graph-replay-stream caller \
  --decoder-metadata-mode position_template \
  --max-model-len 16 --kv-cache-mib 256 \
  --memory-fraction 1.0 --engine-memory-fraction 0.9 --reserve-gib 3 \
  --port 8000
```

In a second terminal, use the unchanged benchmark command (raw URL, not a
Markdown link):

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
python -u tools/benchmark_vq2a8_serving.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 \
  --prompt '你好' --max-tokens 4 --warmups 3 --repeats 20 --timeout 300
```

Send both raw 20-request results and validation summaries. Retain `torch` if
fusion is slower: fewer launches alone do not prove a speedup. For the next
optimization, obtain a fresh profile of the winning configuration with
`--host-profile` and retain its exact mode flags, library SHA256, steady decode
windows and device operator totals. The earlier profile predates fused validity
and position templates and cannot quantify their remaining hotspots.
