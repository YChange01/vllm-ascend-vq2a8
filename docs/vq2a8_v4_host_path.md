# V4 decoder host-path and packed activation candidates

These opt-in candidates preserve the current V4/v2 resident weights, native
projection kernel, vectorized reorder, rowwise arithmetic, and decoder graph
baseline. They are not a claim of 30 ms TPOT or of hardware validation.
No new native build or weight conversion is required: sign-only fusion uses
the existing perf3 library's `activation_sign` ABI 1, not `activation_quantize`.

## Changes and boundaries

| Option | Behavior |
| --- | --- |
| `--decoder-metadata-mode recursive` | Unchanged default recursive metadata path. |
| `--decoder-metadata-mode planned` | Compile the metadata structure at startup; deduplicate shared containers and tensor storage. Revalidate live sources on every update, then copy only after all checks pass. |
| `--activation-preparation rowwise` | Unchanged default, including prefill. |
| `--activation-preparation rowwise_packed` | Consume selected expert metadata directly in B1 decode; avoid rebuilding dictionaries, stacking selected rows, and splitting/concatenating outputs. |
| `--activation-preparation sign_fused` | Packed path plus native sign multiplication/input validation only. RHT, bias GEMVs, scaling, amax, division, clamping and FP8 conversion retain Torch arithmetic. |
| `--host-profile` | Opt-in instance-local CPU ranges and bounded inclusive wall/thread-CPU counters. No added device fences or tensor scalar reads. Disable for performance measurements. |
| `--profile-dir PATH` | Forward a Torch profiler configuration to vLLM; HTTP start/stop controls collection. Does not itself start collection. |

Both activation candidates require V4/v2 device-route decode. They support
one row per expert, 1..6 expert rows, K=2048/4096, with no padding. General
prefill still uses the inherited rowwise preparation. No implicit fallback is
allowed for unsupported geometry or a missing native ABI. The legacy `fused`
mode remains separate: its previously failing scale-byte validation is not
bypassed or treated as passing.

The planned metadata path preserves CPU constant checks, immutable pointers,
tensor shape/stride/dtype/device contracts, and captured alias topology. It
does not cache validity across requests. Runtime model contracts still run.
Packed preparation introduces small output-buffer copies: fewer Python packing
operations do not guarantee faster device execution; measure each option.

## 1. Bounded activation acceptance on the idle physical NPU 1

Stop only your own serving process before running the standalone probes. Do
not run another full model concurrently or reset other users' devices.

```bash
python -u tools/validate_vq2a8_activation_packed.py \
  --library build/vq2a8-ascendc-v4-v2-perf3/libvq2a8_ascendc_v4_v2.so \
  --physical-npu 1 --queue-lifetime --timeout-s 300
```

Require `V4_PACKED_ACTIVATION=PASS` and inspect its summary. This compares
q/scale/bias byte-for-byte with the existing rowwise path and checks graph
validity and queued lifetime. It does not load or verify the full model.
Do not relax tolerance after a failure or start the candidate as accepted.
For this isolated lifetime probe, an unset `TASK_QUEUE_ENABLE` defaults to `2`;
an explicitly inherited `0` is rejected, not silently overridden. Queue
capacity itself is not measured. The bounded queue test is not full-model
lifetime evidence.

## 2. Real decoder acceptance

First isolate the metadata change with `--activation-preparation rowwise`;
then repeat for `rowwise_packed` and `sign_fused` after the activation probe
passes. Reuse the same previously validated artifact/library in all runs.
If using preconverted experts, append `--artifact` with that existing directory
to every model command below; otherwise the model's original expert directory
is used. No artifact is overwritten.

```bash
python -u tools/validate_vq2a8_v4_decoder_graph.py \
  --model /home/g00872988/vq2a8 \
  --library build/vq2a8-ascendc-v4-v2-perf3/libvq2a8_ascendc_v4_v2.so \
  --physical-npu 1 --compute-backend v2 \
  --activation-reorder vectorized \
  --activation-preparation rowwise \
  --decoder-metadata-mode planned \
  --kv-cache-mib 256 --reserve-gib 3 --timeout-s 1800
```

Require `V4_DECODER_GRAPH=PASS`. This test compares eager and graph execution
under the *same* preparation backend across compressor boundaries and repeated
requests. Therefore it complements, not replaces, the separate byte-exact
baseline-vs-packed activation test.

## 3. Serve and measure independent A/B variants

This example isolates the metadata optimization. Keep all other launch flags,
model/library/artifact, card, request parameters and concurrency identical.

```bash
python -u tools/serve_vq2a8_v4.py \
  --model /home/g00872988/vq2a8 \
  --library build/vq2a8-ascendc-v4-v2-perf3/libvq2a8_ascendc_v4_v2.so \
  --physical-npu 1 --compute-backend v2 \
  --activation-reorder vectorized --activation-preparation rowwise \
  --device-route-decode --decode-graph decoder --graph-replay-stream caller \
  --decoder-metadata-mode planned \
  --max-model-len 16 --kv-cache-mib 256 \
  --memory-fraction 1.0 --engine-memory-fraction 0.9 --reserve-gib 3 \
  --port 8000
```

After startup, in a second shell in the same container:

```bash
python -u tools/benchmark_vq2a8_serving.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 --prompt '你好' \
  --max-tokens 4 --warmups 3 --repeats 20 --timeout 300
```

Compare baseline `recursive + rowwise`, then `planned + rowwise`, then
`planned + rowwise_packed`, then `planned + sign_fused`. Relaunch only your own
service between variants. Keep the faster, validated combination; do not infer
benefit from CPU tests or isolated operator timings. Longer decode tests must
remain within the total 16-token position-specialized context limit.

## 4. Attribute the remaining host gap

For a separate diagnostic run, append these two flags to the service command:

```bash
--host-profile --profile-dir /home/g00872988/profiler_output
```

Use the same benchmark command once to warm up before collection. Then collect
exactly three sequential four-token requests (do not benchmark unrelated
traffic in this window):

```bash
curl --noproxy '*' -fsS -X POST http://127.0.0.1:8000/start_profile
python -u tools/benchmark_vq2a8_serving.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 --prompt '你好' \
  --max-tokens 4 --warmups 0 --repeats 3 --timeout 300
curl --noproxy '*' -fsS --max-time 300 -X POST http://127.0.0.1:8000/stop_profile
```

Wait for the new rank directory's `ASCEND_PROFILER_OUTPUT/analyse.done`. Run
`tools/extract_vq2a8_profile.py` on that **new** output with
`--decode-steps-per-request 3 --expected-requests 3`; use a fresh extraction
output directory, without deleting previous results. Send its paste summary.

The added `vq2a8::host::*` scopes cover runner state/input preparation, attention
metadata building, decoder position selection, runtime contracts, metadata
updates, input copies, replay submission, output copies, LM head/validity,
sampling and bookkeeping when those runner methods are available. They are
installed on the specific runner only, after startup graph capture. They do
not instrument engine scheduling/IPC outside the runner. If a large uncovered
gap remains, Python/native sampling is still needed; an uncovered span is not
proof of CPU idle time.

Inclusive wall and calling-thread CPU counters are also available through
the model's `v4_graph_report()['host_profile']`; the real-model validation
tool includes them in its `summary.json` when passed `--host-profile`.
These counters include all calls since installation, not only the HTTP
profiler window, and are not a new HTTP endpoint. Trace annotations contain
wall spans; the extractor cannot reconstruct missing thread-CPU counters.
Nested phases, scalar waits and device work must never be added as independent
costs. Do not rescale a profiled replay interval to the unprofiled HTTP TPOT.

## Local verification limits (2026-09-17)

CPU tests exercise metadata aliasing/transactional updates, option forwarding,
real Torch packed arithmetic, both device-route call paths with an explicit
native oracle, recorder installation, and extraction. Ruff checks pass. No
Ascend compiler or NPU execution was available for this change.

The complete CPU suite also reproduced 14 pre-existing HEAD failures, left
unchanged rather than bypassed:

- 11 source-provenance tests still pin the original model-runner blob
  `70ef1d79...`; committed graph changes already changed it to `9358f3c0...`.
  This affects explicit consistency audits, not the normal V4 serve path.
- One older performance-model test fixture omits `_v4_decode_graph` by
  skipping the constructor.
- Two root-FP8 midpoint/FMA conformance expectations differ on the local
  Windows CPU Torch build; their tests and implementation are unchanged.

These are not NPU acceptance results. Keep the independent hardware checks
above and report candidate TPOT only from an unprofiled serving A/B run.
