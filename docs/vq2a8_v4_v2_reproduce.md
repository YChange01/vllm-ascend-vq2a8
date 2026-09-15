# V4 + v2 compute candidate (v023)

This opt-in backend keeps V4 full residency, device routing, the existing MoE
decode graph, and its fixed asynchronous ownership protocol. It adapts the v2
register-LUT/zN compute source, **not** the experimental v3 resident runtime.
The old V4 backend remains the default and is not overwritten.

To avoid repeating startup layout conversion, use the independent
[CPU preconverted artifact](vq2a8_v4_v2_prepacked.md) with `--artifact`.
This does not require another native build or change compute/graph options.

For the independent vector reorder, fused preparation and expanded decoder
graph candidates, see [three decode optimizations](vq2a8_v4_perf3.md). The
commands below retain the original scalar/rowwise/MoE-only baseline.

## Scope and numerical contract

- Execution policy stays `ascendc_v4`; `v4_compute_backend=v2` selects the candidate.
- Build product: `libvq2a8_ascendc_v4_v2.so`, independent namespace and ABI.
- Ascend950, TP1, BF16 roots; N4096, K2048/4096, M1..32 per projection, 1..6 jobs.
- Startup converts the original direct-TP1 artifact to compressed zN/pair-LUT
  storage. Source files are unchanged. Prefill and decode share **one** candidate
  payload representation; no second full old-layout expert copy is retained.
- Preparation metadata remains in original K order. The native adapter gathers
  the already-quantized FP8 **bytes** using the selected expert's permutation.
  Decode does not copy route IDs to CPU or upload host descriptors per step.
- Stable K regrouping and different reduction tiles change FP32 accumulation
  order. Do not reuse a V1 bit-exact acceptance receipt. Validate projections,
  same-prefix logits and generated-token behavior separately.
- Only the MoE decode subgraph is captured. Attention, KV handling and prefill
  remain eager. No claim of full-model graph support or target TPOT is made.

The original v2 cross-core/double-buffer protocol is still hardware-dependent.
The fact that v3 hung does not establish whether its cause was the host binding,
the device pipeline, or their integration. The staged probe below separates
those boundaries. CPU unit tests do not establish CANN compilation, NPU
correctness, queue-lifetime safety or performance.

## Build separately

Run in the v023 repository inside the same configured CANN/PyTorch container.
Use the actual device SoC; this example uses the user's Ascend950DT_9582.
No vLLM version migration or reinstallation is required for these Python edits
when this checkout is already installed editable. The native candidate must be
compiled; the existing V1 library is not replaced.

```bash
python -u tools/build_vq2a8_v4_v2.py \
  --soc Ascend950DT_9582 \
  --build-dir build/vq2a8-ascendc-v4-v2 \
  --jobs 4
```

## Validate before loading the full model

First confirm the selected device is available. Stop your own prior server
before these checks if it occupies that device; these tools do not stop other
jobs. Here device selector `1` is interpreted through the container's visible
device mapping, so confirm it against the process table in `npu-smi info`.

Start with the isolated compute boundary:

```bash
python -u tools/validate_vq2a8_v4_v2.py \
  --library build/vq2a8-ascendc-v4-v2/libvq2a8_ascendc_v4_v2.so \
  --physical-npu 1 --phase kernel --timeout-s 300
```

After it passes, exercise residency, dynamic routes, queue lifetime and graph
replay in the bounded staged probe:

```bash
python -u tools/validate_vq2a8_v4_v2.py \
  --library build/vq2a8-ascendc-v4-v2/libvq2a8_ascendc_v4_v2.so \
  --physical-npu 1 --phase all --timeout-s 300
```

Each stage emits its name before submission and checks device completion before
reporting PASS. A timeout/failure is not a valid performance measurement. Keep
the printed report directory and the last stage; do not proceed to the full
model after a failed stage. Synthetic PASS is not model/logit acceptance.

If the first `kernel_k2048_g1_m1` stage reports BEGIN but never SUBMITTED,
the custom operator call has not returned; the probe has not reached its
explicit device synchronization. The initial standalone binding resolved
`NPUStream::stream()` inside its queued callback. In torch_npu 2.10 that getter
drains the host queue, so its consumer can wait for its own callback to finish.
The binding now resolves the raw ACL stream on the caller before enqueueing,
matching the retained V4 implementation. Rebuild the candidate library after
updating; replacing Python files alone does not update a loaded native library.
This fixes the host self-wait hazard, not proof that the v2 device pipeline
passes. A subsequent SUBMITTED without PASS requires a separate investigation
of device completion. Keep the ordinary asynchronous probe enabled.

Also check the actual compressed gate/up and down weights of one expert before
full-model startup. This reads only that expert and compares against an
independent dense projection oracle; it does not start vLLM:

```bash
python -u tools/validate_vq2a8_v4_v2.py \
  --library build/vq2a8-ascendc-v4-v2/libvq2a8_ascendc_v4_v2.so \
  --physical-npu 1 --phase resident --timeout-s 300 \
  --model /home/g00872988/vq2a8 --expert 0:0
```

The old `accept_vq2a8_v4.py` and its V1-layout residency evidence validator
remain baseline-only. Do not use their PASS receipts to approve this candidate.

## Start V4 + v2 serving

Use a fresh process after rebuilding or switching backend. Graph captures and
payload layouts must never be changed underneath a live request.

```bash
python -u tools/serve_vq2a8_v4.py \
  --model /home/g00872988/vq2a8 \
  --compute-backend v2 \
  --library build/vq2a8-ascendc-v4-v2/libvq2a8_ascendc_v4_v2.so \
  --physical-npu 1 \
  --device-route-decode --decode-graph moe --graph-replay-stream caller \
  --max-model-len 16 --kv-cache-mib 256 \
  --memory-fraction 1.0 --engine-memory-fraction 0.9 --reserve-gib 3 \
  --port 8000
```

For the first eager-only model check, omit `--decode-graph moe` and
`--graph-replay-stream caller`. Keep `--device-route-decode` and the same library.
The launcher performs no automatic benchmark, package consistency audit or
device reset. Normal shape/ABI/ownership and memory-safety checks remain.

Confirm `compute_backend=v2` in serving readiness output and
`layout=v2_zn_pair_lut` in the residency report. The startup budget includes new
layout allocations and eager/graph bank metadata. Roots, KV, graph scratch,
activation preparation and allocator headroom still require the reserve; it is
not automatically reduced to force the model to fit.

## Measure against the retained baseline

Run the client in another terminal in the same container after readiness:

```bash
python -u tools/benchmark_vq2a8_serving.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 \
  --prompt '你好' --max-tokens 4 --warmups 3 --repeats 20 --timeout 300
```

This prompt previously tokenized to **one** input token; it is not the 10:4
workload. For a 10-token comparison, use the same independently token-counted
prompt on both backends. Compare identical inputs, sampling, graph settings and
concurrency. Collect final timing without the profiler running.

To return to baseline, restart with `--compute-backend v1` (or omit it) and your
existing fixed V4 `libvq2a8_ascendc.so`. Do not point v1 at the candidate library.
Compare in separate runs rather than keeping both full models on the same card.
Retain output/correctness evidence alongside TTFT/TPOT; different final text
alone is insufficient to diagnose the source of numerical drift.

## Local validation record (2026-09-15)

- Windows, Python 3.14.7, Torch 2.10.0+cpu, repository CPU contract harness.
- New candidate tests: 99 passed (layout/source, runtime, integration and tools).
- Related V4/offline/activation/startup regressions: 1054 passed, 1 skipped.
- Changed Python files pass Ruff lint/format, forbidden-import and boolean
  context-manager checks, and bytecode compilation; `git diff --check` passes.
- Independent build `--dry-run` and validation `--plan-only` pass. Neither
  invokes CANN or an NPU.
- The broader CPU suite still has 14 failures, independently reproduced with
  the pre-change `050169efd` model source: one old performance-test fixture,
  two root-FP8 CPU rounding tests, and eleven old framework/provenance checks.
  Those implementations and acceptance expectations were not relaxed here.

Reproduce the related CPU checks using a CPU Torch/pytest environment:

```bash
python -X utf8 tools/run_vq2a8_cpu_tests.py \
  -k 'v4 or offline or activation or full_startup_trace'
```

The harness isolates vLLM/plugin imports and rejects Triton launches. These
results do not establish native compilation, device execution or serving gains.
