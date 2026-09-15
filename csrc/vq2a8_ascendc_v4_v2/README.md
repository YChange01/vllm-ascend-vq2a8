# V4 residency with an independent V2 compute backend

This opt-in candidate keeps the working V4 baseline unchanged. It does not
import the V3 resident adapter. `kernel.cpp`, `layout.h`, and the grouped launch
entry originate from this branch's `../vq2a8_ascendc_v2/` sources, whose original
source provenance and rights remain documented in
`../vq2a8_expert_reference/README.md`. No new license grant is inferred for those
derived arithmetic sources. The new V4 adapter and integration code carry their
own SPDX headers.

The Python runtime also supports an optional
[CPU preconverted expert artifact](../../docs/vq2a8_v4_v2_prepacked.md).
It persists the current compressed payload layout and requires no native ABI change.

## Boundary

- Separate library: `libvq2a8_ascendc_v4_v2.so`.
- Separate Torch namespace/custom class: `vq2a8_ascendc_v4_v2.ResidentBank`.
- ABI version: 1; Ascend950, native 1C:2V MIX, N4096, K2048/4096,
  M1..32, and 1..6 route slots per projection.
- The V2 arithmetic functions, register pair-LUT, padded zN buffers,
  K512/K1024 pipeline, and FP32 scale/bias before one BF16 rounding are retained.
  The only grouped-work scheduling extension is an m=0 sentinel: all three
  peers skip an invalid route before loading pointers or emitting flags.
- Expert payload is converted once to compressed V2 layout; it is not expanded
  into dense GM weights. Do not keep both full old and new expert banks.

## Resident API

`ResidentBank(packed_zn[], pair_lut[], activation_order[], weight_scale[],
weight_bias[], rht_sign[])` retains the six immutable payload owners. Constructor
validation checks full K permutations once, with explicit startup-only device
synchronization, then uploads one E-by-8 int64 pointer table. `metadata()` returns
`[E, N, K, additional_persistent_bytes]`, where the last field is exactly E*64.
The already-converted payload, including int64 K permutations, is accounted for
separately by the Python resident planner.

`select(ids)` returns FP32 `[R,K]` weight scale/bias, int8 `[R,K]` signs, and
int32 `[R]` validity. Preparation metadata remains in its original K order.

`project(q, row_scale, row_bias, ids)` accepts already-prepared FP8 `[R,K]`
or `[R,M,K]` and matching FP32 `[R]` or `[R,M]` scale/bias. It returns BF16
`[R,N]` or `[R,M,N]` output and int32 `[R]` validity. Each call launches:

1. A device-only activation byte gather and descriptor construction kernel.
2. The unchanged V2 grouped arithmetic pipeline.

The gather occurs AFTER original RHT and FP8 quantization; moving the permutation
before RHT is not equivalent. Runtime route IDs never travel to the CPU, and no
per-call host descriptor or expert-payload H2D copy is used. Duplicate IDs retain
independent output slots. The complete signed int64 ID is checked before table
indexing, including values such as 2**40. Invalid rows become NaN with validity 0.

The first gather implementation uses scalar byte reads/writes split across
route/row/K chunks, deliberately isolated from the arithmetic pipeline. This is
a correctness-first adapter, NOT evidence of a speedup. Profile its cost
separately, particularly for multi-token prefill.

`project_vectorized(...)` has the same input/output contract, but replaces only
the activation gather with contiguous DMA and vector gathers in UB. The default
`project(...)` remains scalar. `prepare_vectorized(...)` exposes reordered FP8
bytes and route validity for diagnostics; bytes for invalid routes are undefined
and must not be inspected. `activation_reorder_version()` returns 1 for this
optional interface, without changing the baseline bank ABI.

The optional `activation_sign(...)` and `activation_quantize(...)` operators
fuse activation pointwise/reduction/validity work around the original one-row
RHT and bias GEMMs. They do not replace those GEMMs or change the expert layout.
`activation_preparation_version()` returns 1. Native FP8 rounding must pass the
independent fused-preparation validator; compilation is not numerical evidence.

See [the three-optimization guide](../../docs/vq2a8_v4_perf3.md) for separate
build, activation validators, short-context decoder graph validation and A/B
serving commands. Every new option is disabled by default.

Per-call scratch is reordered activation R*M*K bytes and descriptors R*72 bytes,
in addition to output R*M*N*2 and validity R*4. Scratch uses the Torch allocator;
there is no additional persistent decoded-weight cache.

## Lifetime and graph contract

The bank must be used on its construction stream. As with the current V4 bank,
the caller must establish producer readiness before construction. All indirect
payload owners are retained, their allocation streams recorded, and launch
callbacks use the V4 `RunOpApi` pattern rather than the legacy custom handler.
Runtime inputs and scratch/output owners remain strongly retained during
asynchronous submission. Same-stream graph capture can manage scratch addresses;
changing IDs and activations must be tested across repeated replays.

The independent `grouped_projection` operator takes pre-permuted activation
lists and V2 packed/LUT lists. It intentionally uses a small blocking host
descriptor upload, ONLY for the isolated compute probe. Production resident
prefill/decode must use `ResidentBank.project`, not this probe entry.

## Required verification

CPU source/layout tests do not establish CANN compilation, device completion,
deadlock freedom, numerical accuracy, or performance. This environment has no
NPU. Validate in order with bounded subprocess timeouts:

1. Standalone grouped compute, real gate/up/down shapes and M boundaries.
2. Resident fixed/dynamic IDs, full-width invalid IDs, duplicate slots, and
   original-order activations.
3. Async queue lifetime and input/output/bank owner release pressure.
4. Same-stream graph replay with changed inputs, scales, biases, and routes.
5. Full-model output/logit checks and serving TTFT/TPOT against the preserved V4.

K reordering changes FP32 accumulation order. Compare against explicit numerical
tolerances and model-quality checks; do not advertise old-baseline bitwise
identity. The shared V2 cross-core pipeline remains a hardware validation risk:
if standalone compute hangs, stop before resident or service integration.
