# V4 v2 validity and position-template candidates

These opt-in candidates preserve the measured `sign_fused_direct` +
`planned_fast` baseline, original weights and prepacked expert format.
They have **no NPU correctness or performance acceptance yet**. CPU protocol
tests do not establish a new TPOT, or guarantee the 30 ms target.

## Independent switches

| Switch | Default/reference | Candidate scope |
| --- | --- | --- |
| `--validity-mode` | `torch` | `fused`: aggregate six native INT32 status vectors, three BF16 finite scans and existing route flags in one AIV launch per MoE layer |
| `--decoder-metadata-mode` | `recursive`; measured baseline `planned_fast` | `position_template`: use startup position-owned DSA metadata, refreshing live block tables and slot maps |

Fused validity requires v2, device routing and a `sign_fused*` preparation.
It changes neither RHT, bias GEMV, FP32 quantization arithmetic nor native safe
gather checks. Routing checks and the final model-boundary error check remain.
Multi-token prefill retains its original preparation. The single-AIV scan may
be slower on some shapes: measure it independently before combining changes.

Position templates require the exact supported Ascend A5 DSA metadata schema,
TP1, synchronous B1 decode and context at most 16. Prefill and eager reference
use the original builder. Every replay obtains current physical KV block IDs
and slot mappings; cached position is not cached request state. Unknown schemas
or changed contracts fail rather than silently skipping checks. `_prepare_inputs`
still runs. The candidate does not implement speculative or asynchronous decode.

## Build separately on the NPU host

After transferring/updating the source, run from the repository root. Do not
overwrite `perf3` or `direct`; keep both available for rollback.

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
python -u tools/build_vq2a8_v4_v2.py \
  --soc Ascend950DT_9582 \
  --cann /usr/local/Ascend/cann-9.1.0 \
  --build-dir build/vq2a8-ascendc-v4-v2-validity-template \
  --jobs 4
VQ2_CANDIDATE_LIB="$PWD/build/vq2a8-ascendc-v4-v2-validity-template/libvq2a8_ascendc_v4_v2.so"
test -f "$VQ2_CANDIDATE_LIB"
```

If the installed CANN location differs, use its real path. No expert conversion
is needed; all commands below reuse `experts_vq_v4_v2_prepacked`.

## Acceptance before serving

Use an idle physical NPU 1. Stop the existing server normally before testing.
`TASK_QUEUE_ENABLE=1` must precede Python, not follow `export` on the same line.

First gate the new checker against the original Torch predicate, including
invalid/recovered graph inputs and queued owner-release pressure:

```bash
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_validity_fused.py \
  --library "$VQ2_CANDIDATE_LIB" \
  --physical-npu 1 --queue-lifetime --timeout-s 300
```

Recheck strided and direct activation preparation on the new binary. Explicitly
select these modes: the probe defaults do not cover `sign_fused_direct`.

```bash
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_activation_packed.py \
  --library "$VQ2_CANDIDATE_LIB" \
  --physical-npu 1 \
  --preparation-modes sign_fused_strided sign_fused_direct \
  --queue-lifetime --timeout-s 900
```

Then run real-model gates for each independent candidate and their combination.
Stop at any failure; do not bypass a failing gate to benchmark serving.

```bash
VQ2_GATE_ARGS=(
  --model /home/g00872988/vq2a8
  --artifact /home/g00872988/vq2a8/experts_vq_v4_v2_prepacked
  --library "$VQ2_CANDIDATE_LIB"
  --physical-npu 1 --compute-backend v2
  --activation-reorder vectorized
  --activation-preparation sign_fused_direct
  --kv-cache-mib 256 --reserve-gib 3 --timeout-s 1800
)
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_v4_decoder_graph.py \
  "${VQ2_GATE_ARGS[@]}" --validity-mode fused --decoder-metadata-mode planned_fast
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_v4_decoder_graph.py \
  "${VQ2_GATE_ARGS[@]}" --validity-mode torch --decoder-metadata-mode position_template
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_v4_decoder_graph.py \
  "${VQ2_GATE_ARGS[@]}" --validity-mode fused --decoder-metadata-mode position_template
```

The template validator enables an acceptance-only original-builder comparison:
all static tensor payloads and live block/slot fields must match exactly across
positions 1 through 14 and repeated requests. This adds synchronization and is
not a latency measurement. Each case also runs with the shadow builder disabled
and verifies real producer skips plus matching tokens/logprobs, so shadow calls
cannot conceal missing builder side effects. The token/logprob comparison is same-backend eager
versus graph, not an independent proof of native numerical accuracy. Native
predicate and activation gates are required separately.

## A/B latency

Keep every other parameter unchanged. Run one server at a time, each in a fresh
process; benchmark without `--host-profile` or an active profiler.

| Run | `--validity-mode` | `--decoder-metadata-mode` |
| --- | --- | --- |
| A: same-binary reference | `torch` | `planned_fast` |
| B: validity only | `fused` | `planned_fast` |
| C: metadata only | `torch` | `position_template` |
| D: combination | `fused` | `position_template` |

Select the two options per row. This example starts D only after all gates pass:

```bash
TASK_QUEUE_ENABLE=1 python -u tools/serve_vq2a8_v4.py \
  --model /home/g00872988/vq2a8 \
  --artifact /home/g00872988/vq2a8/experts_vq_v4_v2_prepacked \
  --library "$VQ2_CANDIDATE_LIB" \
  --physical-npu 1 --compute-backend v2 \
  --activation-reorder vectorized --activation-preparation sign_fused_direct \
  --device-route-decode --decode-graph decoder --graph-replay-stream caller \
  --validity-mode fused --decoder-metadata-mode position_template \
  --max-model-len 16 --kv-cache-mib 256 \
  --memory-fraction 1.0 --engine-memory-fraction 0.9 --reserve-gib 3 --port 8000
```

In another terminal, after service readiness:

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
python -u tools/benchmark_vq2a8_serving.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 --prompt '你好' \
  --max-tokens 4 --warmups 3 --repeats 20 --timeout 300
```

Save every raw run and the library SHA. Repeat A after D to check thermal/load
drift. The previous ~39.58 ms HTTP TPOT and ~45.28 ms profiled replay interval
are different measurements; never rescale profiler totals to HTTP TPOT.

## Quantization-tail diagnosis only

This does **not** enable the previously failing full fused quantizer. It mirrors
its arithmetic with intermediate snapshots and compares FP32 bits/hex/ULPs and
FP8 bytes at multiply, maximum, division by 448, scale clamp, normalization and
FP8 conversion. It also compares final diagnostic output with the unmodified
native quantizer, because instrumentation can change behavior.

Run separately while the server is stopped:

```bash
TASK_QUEUE_ENABLE=1 python -u tools/diagnose_vq2a8_activation_tail.py \
  --library "$VQ2_CANDIDATE_LIB" --physical-npu 1 --timeout-s 300
```

`MISMATCH` (exit 2) is a completed diagnosis finding a difference, not a
correctness PASS. Send its summary and bounded mismatch rows; no full trace or
weights are needed. `ALL_MATCH` covers only this diagnostic's cases, not all
activation shapes or full-model acceptance. Do not switch serving to
`--activation-preparation fused` based on this diagnostic alone.
