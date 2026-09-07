# VQ2A8 Ascend 950 TP1: scope, evidence and acceptance

## Goal and current boundary

The goal is faithful DeepSeek-V4-Flash-VQ2A8 inference on one Ascend 950,
including prefill, decode, routed and shared experts, and stable serving.
Expert weights stay packed; activation and codebook storage stay E4M3FN.
Full-model logits/token quality, peak HBM and throughput are acceptance
criteria. An HTTP 200 or one passing expert is not full-model acceptance.

**Latest milestone (2026-09-07):** phase 1 has an actual NPU bit-exact
offline regression PASS. The user also reported phase-3 operator and
FP8-root offline-model PASS at `57e48c73`, with 43 layers and two identical
runs (`/tmp/vq2a8-phase3-ej7e_5co`). Native FP8 root matmul is verified;
native FP8 expert dot, independent full-model reference and quality are not.
At the user's request, the phase-4 Vector optimization branch is paused;
development now targets a **native AscendC/C++ VQ-decode + FP8 Cube fused
projection**, described at the end of this document. The earlier Triton
fusion prototype is paused after repeated A5 compiler assertions; it is
not the implementation of the new AscendC operator. This does not change the
accepted model backend. Phase 2 stays skipped and phase 5 stays deferred.
The following sections preserve the earlier milestones as history.

Commit `533a906b8bb67735a3cca0a35a01313c05cfdfbb` passed a user-run hardware
smoke for layer 0, expert 0, separate gate_up and down projections. Both
matched the CPU oracle within tolerance and the one repeated result matched
exactly. This is an M=1 packed Vector baseline. The branch does not yet
register a VQ2A8 serving quantization method. TP4, prefill batching, MoE
routing, complete decoder execution and full-model quality were unverified
at that smoke milestone.

The subsequent user-run expanded acceptance reported 7/7 passing expert
probes, four input cases each, chained projections and at least three exact
repeats. It reported Git `bb3b4fcaea39d32cb3d178e08e4918eaf4c4a795` with a
dirty worktree. Preserve the full report's source hashes/status: that result
must not be attributed to the clean commit alone. Native FP8 dot and serving
remained false. Standalone MoE acceptance has subsequently passed on the
user's NPU as recorded below. The next stage is bounded offline model
execution, not a serving backend or a change to the accepted Vector kernel.

The CPU reference/repack tests and CUDA kernel tests exercise different code
from the Ascend-specific JIT body. Their success cannot certify NPU lowering.
The NPU Vector body is kept unchanged during the acceptance improvements.

## Available machines

| Role | Known facts | What can be established |
| --- | --- | --- |
| Local development | Windows workspace; source and Git access | Source review, code changes, reports |
| Remote NVIDIA | L20X; real source artifact under `/mnt/ACS/gyc/ascend950-transfer/DeepSeek-V4-Flash-VQ2A8-32x256` | CPU oracles, metadata/payload inspection, CUDA regression |
| User's Ascend host | No direct connection available to this workflow | User runs a bounded command and returns its report |

Do not infer Ascend access from the NVIDIA SSH connection. Hardware outcomes
are recorded as user-run evidence, not as tests executed on NVIDIA.

## Recorded Ascend environment

| Component | User-provided value |
| --- | --- |
| Device | Ascend950PR_958b, SoC 260, vLLM Ascend type A5 |
| Visible device selection | Physical 4 maps to logical `npu:0` for TP1 |
| Memory | npu-smi total 114688 MB; torch properties total 110304 MB; query free bytes at each run |
| Cores | 32 Cube, 64 Vector |
| Driver | 25.6.rc1.b196 |
| CANN | 9.1.0; `/usr/local/Ascend/cann-9.1.0` |
| Toolkit symlink | `/usr/local/Ascend/ascend-toolkit/latest` resolved to the same CANN directory |
| torch / torch-npu | 2.10.0+cpu / 2.10.0.post4 |
| Triton Ascend | 3.2.2+dev20260729205041 |
| vLLM | 0.23.0+empty |
| vLLM Ascend distribution | 0.23.1.dev30+g09353ce64, editable checkout |
| safetensors | 0.8.0 |

The editable package version is not the current Git revision. Record both.
`torch` reporting `+cpu` does not mean torch-npu cannot execute on the NPU.
The BF16 GEMM, int32 round-trip and FP8 basic gates already passed. This
does not prove every FP8 Cube operation is supported by this compiler build.
Package dependency warnings alone do not explain the observed MTE fault.

## Error history and confidence

| Failure | Evidence-supported conclusion | Not established |
| --- | --- | --- |
| TP4 layer-0 gate_up all NaN on ranks 0/1 | Non-finite output caused fail-fast, worker death and HTTP 500 | Exact first source of NaNs; hardware-vs-rank cause |
| aclInit 107001, expected device range `[0,0)` | That process had no valid runtime device mapping | Physical device failure |
| Eager RightShift 161002 | Broadcast output shape disagreed with the ACLNN operation | Packed checkpoint corruption |
| `CUBE_OR_VECTOR` assertion | Compiler pipeline failed on the mixed kernel | Unsupported FP8 hardware in general |
| Empty `hivm.hir.pointer_cast()` | Compiler emitted invalid IR for a transfer destination in that implementation | The cause of every later runtime error |
| Mixed kernel 507015 / MTE unaligned / exit 134 | Device execution aborted with an alignment fault | Exact instruction or universal plain FP8 dot failure |
| Pure Vector smoke passed | This packed decode/reduce implementation works on the tested inputs | Complete expert chain, all experts, full model or production speed |

The successful Vector alternative strengthens suspicion of the mixed
Vector/Cube memory-transfer path. It does not isolate fixpipe specifically.
A minimal FP8 GEMM test and compiled-instruction analysis are needed before
assigning the fault to a particular compiler pass. Do not edit generated IR
or keep changing unverified compiler switches as a deployment strategy.
The new input-pointer alignment checks reject unsafe contiguous views;
they do not establish that the earlier mixed-kernel MTE fault is fixed.

## Quantization and routing contract

VQ2 stores a 4-bit code per two-component vector: two index bits per scalar
weight before codebook and auxiliary overhead. It is not scalar INT2 GEMM.
Codebooks are `torch.float8_e4m3fn`; converting a decoded tile to FP32 for
Vector arithmetic does not make the stored codebooks BF16.

Let C be the repacked codebook matrix after absorbing inverse permutation,
D the RHT signs, H normalized Sylvester Hadamard, s column scales and b
column bias. The activation path is:

```text
z = x D H
u = z * s
a = max(abs(u)) / 448, clamped below at 1e-12
q = E4M3FN(clamp(u / a, -448, 448))
y = a * (q C.T) + z b
```

RHT executes once; bias correction uses z before column scaling; the
permutation has already been absorbed offline. The gate and up halves use
DeepSeek's asymmetric gate clamp and symmetric up clamp before SiLU/mul,
with BF16 boundaries preserved.

Read-only inspection of the actual NVIDIA-hosted checkpoint found that
`layers.{0,1,2}.ffn.gate.tid2eid` each has shape `[129280,6]` and contains
only zero. These exported hash layers store one expert. This is a property
of this checkpoint, not a consequence of hash routing in all models.
The audit checks route IDs against stored experts before claiming coverage.
Duplicate IDs within top-k must accumulate their routing weights correctly;
do not scatter-overwrite or silently discard duplicates. Shared experts
must execute once; routed scaling must execute once under a single owner.

## Memory and performance

The model directory's 140G includes multiple expert layouts. TP1 should load
one expert artifact plus the required root weights. The earlier TP4 figure
of 16G per rank is not a measured TP1 footprint. Root checkpoint files total
14703068868 bytes on the inspected source host; the audit reports tensor
payload bytes separately from filesystem bytes.

The combined storage budget excludes KV cache, operator workspace,
allocator reserves, load-time copies and dtype conversions. Measure actual
peak allocated/reserved HBM before selecting a serving memory fraction.
Loading all experts into FP16/BF16/FP32 would invalidate the packed-memory
design. A future Cube implementation may use bounded per-tile FP8 scratch.

The reported 26.54 ms gate_up and 6.84 ms down are one synchronized Python
call each, including launch/allocation/synchronization. They are not
device-event timings or full-model token latency. The current lookup loops
scan all column tiles and all 16 codes for each K tile: for gate_up this is
8 K tiles x 16 codebook tiles x 16 codes per output block. This correctness
implementation needs profiling before any performance promise.

## Reproducible acceptance on the disconnected NPU

Run from the checkout using the Python that imports torch-npu:

```bash
python3 tools/validate_vq2a8_tp1_acceptance.py \
  --model /home/g00872988/DeepSeek-V4-Flash-VQ2A8-32x256 \
  --physical-npu 4
```

The default probes cover hash layers 0/1/2, routed layer 3 experts 0/127/255,
and layer 42 expert 255. Each runs deterministic, zero, impulse and small inputs,
separate projections, the packed gate_up -> SwiGLU -> down chain, and every
repeat. The default model audit validates all artifact headers and the
checkpoint's hash routes. Add `--verify-tensor-hashes` for a payload hash scan
on the first child; this reads the full expert artifact once.

Each child has isolated visible-device/rank environment variables. Full
output goes to a new temporary directory. The supervisor survives a child
SIGABRT, records the stage and exit code, and stops after the first failed
probe. `summary.json` explicitly distinguishes expert acceptance from
serving readiness. Return the short `summary.txt` report; the full JSON and
logs remain available beside it. The short report is also printed on exit.

An existing long JSON report can be summarized without accessing the NPU,
rerunning tests or changing the original file:

```bash
python3 tools/validate_vq2a8_tp1_acceptance.py \
  --summarize /tmp/vq2a8-acceptance-REPLACE/summary.json
```

The short report includes run revision, input cases, one line per probe,
maximum absolute/relative-L2 errors, same-FP8-input error, checked repeats
and native-FP8/serving status. Failed probes retain a bounded error excerpt.

Add `--cases deterministic zero impulse small large` for magnitude stress.
An existing `--output-dir` is refused so earlier evidence is preserved.

Two numeric comparisons are reported: an oracle with exactly the device's
prepared FP8 inputs, and the independent CPU activation-preparation path.
The first isolates packed decode/MAC; the second covers RHT/A8 differences.
Relative L2 guards small signals where a fixed absolute tolerance alone
could accept a completely incorrect zero output. Timings are explicitly
labelled synchronized call times.

The expanded CUDA layer-0 test exposed an additional boundary: a 32x input
stress case failed the chained down comparison (38/4096 elements, maximum
absolute error 1.0, relative L2 about 0.00208), while the oracle using the
same prepared FP8 input passed. This does not certify a packed-kernel defect:
differences in activation preparation and propagated BF16/FP8 rounding must
be separated first. The failure is retained, not hidden by relaxing the
tolerance. NPU behavior on this stress case has not been measured.

Further CUDA isolation found 11 differing BF16 gate elements, identical
CPU/CUDA SwiGLU output when given the same input, and 13 differing down
activation FP8 bytes when comparing the independent chains. Preparing the
same down input on CPU/CUDA produced identical FP8 bytes (only small scale
and bias differences remained). These observations locate the amplification
at propagated rounding and requantization boundaries for this case. They
do not justify either changing the artifact or claiming full-model accuracy.

## Checks performed for this review

- 142 CPU/unit tests passed, including alignment, numerical limits, routing
  inventory, crash-summary handling and an intermittent bad middle repeat.
- On the remote NVIDIA GPU, real probes `0:0`, `3:0`, `3:127`, `3:255` passed
  deterministic, zero, impulse and small cases with independent projections,
  packed expert chains, one warmup and three checked repeats.
- The layer-0 large-input chain failure above reproduced and remains open.
- Ruff checks/formatting and markdownlint passed on the changed files.
  The repository-wide `bash format.sh ci` could not run its hooks because
  `pre-commit` was absent in the available test environment.
- No new NPU execution or full-model serving test was performed. The user
  subsequently supplied the expanded expert acceptance recorded above.

## Standalone TP1 MoE stage

`vq2a8_moe.py` adds an isolated eager MoE layer, not a model-runner patch:

- Load real root checkpoint router/shared weights and packed expert slices.
  `gate.bias` is the routing selection correction, not a linear-layer bias.
- Use FP32 sqrt-softplus scores; non-hash top-k selects by corrected scores
  with lowest-ID tie breaking, but weights use the original scores. Hash
  routing requires token IDs and retains repeated expert IDs.
- Give `mix_vq2a8_routes` sole ownership of routed scaling. Its input weights
  are unscaled; shared output is added once without routed scaling. This is
  an explicit standalone contract, not a drop-in replacement for a router
  that already scales its weights. Upstream integration must preserve the
  chosen rounding boundaries as well as the algebra.
- Evaluate each unique (token, expert) once, retain every top-k slot in FP32
  weighting/reduction, and use the accepted packed M=1 kernel for each row.
  Larger token batches are processed in bounded chunks. Grouped top-k and
  TP sizes other than one are rejected rather than silently approximated.
- Keep only a configurable LRU set of packed experts on the device. No
  dense routed-expert weights are materialized on NPU/CUDA; shared weights
  remain dense BF16. CPU-only dense decoding supplies the numeric oracle.

The gate defaults to layers 0 and 3, deterministic/zero inputs, M=1 and M=3,
a two-token chunk, a two-expert cache and three checked repeats. It compares
router IDs exactly, router weights numerically, full routed-plus-shared
outputs against CPU, and chunked output against individual-token calls.
Input amplitudes match the expert gate's cases without additional rescaling.
Repeat comparisons are exact. The gate records packed-cache bytes and
device allocated/reserved peaks, not a full-model memory estimate.

On the remote NVIDIA host, both real layers passed all four default cases.
The maximum relative-L2 output errors were about 0.00499 (layer 0) and
0.00488 (layer 3). The cache held at most two experts; layer 0 needed one.
These runs used synchronized working files in the remote test staging area,
whose Git HEAD is older and whose worktree is dirty; their reports retain
source hashes. They are CPU/CUDA evidence, not Ascend or serving acceptance.

The MoE stage regression run passed 188 CPU/unit tests across the artifact,
reference, repack, runtime, packed kernel contract, validation and MoE test
files. Added checks include duplicate slot accumulation, shared-output and
cache contracts, invalid router options, unchanged test input amplitude,
every repeat, and the isolated MoE supervisor's short/failure reports.
Ruff and Markdown checks passed; the repository-wide format script remains
unavailable because the test environment has no `pre-commit` installation.

Run the new stage on the disconnected Ascend host without repacking:

```bash
python3 tools/validate_vq2a8_tp1_acceptance.py \
  --stage moe \
  --model /home/g00872988/DeepSeek-V4-Flash-VQ2A8-32x256 \
  --physical-npu 4
```

Each layer runs in its own child process. The same short-report mechanism
prints one line per layer, including `moe_cases`, router/chunk checks,
output errors and repeat status; return `summary.txt`. Detailed root IDs,
weights, memory and failure context stay in `summary.json` and child logs.
On NPU, `NATIVE_FP8_DOT=False` and `SERVING_VERIFIED=False` remain expected.

Host routing plans, per-row launches, synchronous validation, eager RHT and
cache-miss disk/device copies are deliberate bring-up limitations. This
path is not graph-compatible or throughput-qualified. Passing it does not
validate attention, actual decoder activations, KV cache, full prefill or
token generation. Continue to retain the earlier large-input discrepancy.

## Next acceptance milestones

1. Preserve the user-reported expanded expert acceptance and dirty-worktree
   provenance; do not discard successful artifacts or rerun repack.
2. Preserve the completed standalone TP1 MoE NPU acceptance below.
3. Run the opt-in offline execution gate. Verify strict root loading,
   every layer, prefill/decode steps, finite logits and peak HBM. Then compare
   logits/tokens against a known-good execution of this checkpoint; repeat
   consistency alone cannot establish numerical correctness.
4. Start serving only after offline execution and independent numerical
   validation pass. Run repeated prompts and mixed lengths, then establish
   a quality/performance baseline.
5. Optimize decode lookup and activation preparation; evaluate minimal
   native FP8 Cube primitives independently before changing the accepted
   Vector kernel. TP4 is a later milestone with separate routing/reduction
   validation.

The available eager preparation rebuilds/transfers a Hadamard matrix and
performs finite checks that synchronize. It remains a reference path, not
the final serving hot path. Optimize it with explicit preparation/cache
ownership after integration correctness has been established.

## User-run standalone MoE acceptance

The user reported `ACCEPTANCE=PASS completed=2/2 passed=2` at
`0f2e5bd0b8adb7c62c3b8df26f30c34bf2323de4`, with a dirty worktree. The report
directory was `/tmp/vq2a8-acceptance-k7x2t_ia`. Each layer passed deterministic
and zero cases at M=1/M=3, routing and chunk comparisons, and at least three
exact repeats. Layer 0 maximum absolute/relative-L2 errors were 0.0625 and
0.00389557; layer 3 errors were 0.0625 and 0.0120408. These are user-supplied
NPU results, not reruns by the development agent. Source hashes in the full
report are required to identify the dirty working files precisely.

## Opt-in offline model execution stage

`--stage model` now selects an inheritance adapter under `patch/worker/`.
The ordinary `DeepseekV4ForCausalLM` registration and construction defaults
remain unchanged. No model-runner patch, new environment variable, repack
format, or Vector kernel change is introduced.

The gate supplies an in-memory architecture override and clears HF's root
quantization config only for this adapter. It does not edit `config.json`.
Attention, HC, embedding, head and KV-cache execution are inherited from
the Ascend implementation. Packed routed experts are explicitly owned by
the accepted `VQ2TP1MoE` runtime; each layer retains at most two packed
experts, with two-token chunks. The root shared experts remain BF16.
This avoids allocating dense placeholders for all routed experts. It is
an I/O-heavy correctness baseline, not a resident-weight performance design.

The adapter fixes three integration boundaries:

- Pass real input token IDs through the model/decoder to hash MoE layers.
- Give the accepted MoE sole ownership of routing, routed scaling and
  shared-output addition, without another FusedMoE runner/reduction.
- Strictly account for every root tensor and registered parameter; reject
  missing, duplicate, unknown, non-finite and shape/dtype-mismatched weights.
  The only permitted dtype change is exact BF16-to-FP32 widening of A5
  compressor norm weights, explicitly recorded in the load report. No FP8
  root conversion is silently performed. MTP weights alone are excluded.

Remote real-checkpoint header inspection and meta-constructor comparison
accounted for all 1199 non-MTP root tensors: 984 model parameters and 215
delegated router/shared tensors. All parameter names/shapes matched;
62 compressor norm tensors require the documented widening. This test
substitutes primitive layers/device operations with shape-only stand-ins;
it does not certify the installed NPU worker, CANN kernels or execution.
The regression suite passed 247 CPU/unit tests, including real constructor
dispatch, strict loading, invalid execution modes and gate failure reports.

The NPU command is:

```bash
cd /home/g00872988/vllm-ascend-vq2a8
git pull --ff-only origin ascend950-vq2a8
python3 tools/validate_vq2a8_tp1_acceptance.py \
  --stage model \
  --model /home/g00872988/DeepSeek-V4-Flash-VQ2A8-32x256 \
  --physical-npu 4 \
  --timeout 3600
```

It requires the complete `experts_vq_ascend_v2` artifact. The new stage uses
one in-process vLLM worker in an isolated supervised child, eager mode,
disabled compilation/graphs/async scheduling/prefix caching, max sequence
count 1, context limit 32, and an explicit 1 GiB KV allocation. Profiling
still executes; it is never reported as actual generation. The existing
vLLM multiprocessing control is disabled only inside this diagnostic child.
The worker control API is documented in
[vLLM collective_rpc](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/llm/).

Two identical greedy requests each generate four tokens. Acceptance
requires all 43 MoE layers once per forward, a real multi-token prefill,
three sequential decode steps, finite full-vocabulary logits, agreement
between greedy sampled tokens and logits, and exact token/logit repetition.
An attention call without real metadata cannot count as a generation step.
Exact-repeat failure is retained without relaxing tolerances. Full logits
and per-run details are saved under `model-evidence/`; the short report
prints stage, run/layer counts, token IDs and allocated/reserved peaks.
The default one-hour child timeout can be raised explicitly for this slow
baseline; it is not a throughput requirement.

The stage is deliberately labelled `MODEL_OFFLINE_EXECUTION`. Even a PASS
keeps `NATIVE_FP8_DOT=False`, `LOGITS_REFERENCE_VERIFIED=False`,
`QUALITY_VERIFIED=False` and `SERVING_VERIFIED=False`. It does not cover a
128-token compressor boundary, long contexts, multiple requests, all expert
routes or comparison with independent full-model logits. Those are later
gates. Full-model acceptance is not yet established; the disconnected Ascend
host must return each new hardware result.

### First offline attempt and live diagnostics

The user reported a failed offline run at
`0fed520deb72af81bf9a9ad0da54a16f25f48953` with a dirty worktree, report
`/tmp/vq2a8-acceptance-wbcyzo2w`. It reached `run=0 stage=prefill_decode`
and failed with `AscendColumnParallelLinear` missing `weight_scale`.
The summary alone does not provide the full source traceback. Source review
found an unconditional A5 FP8 output-projection branch which accesses that
attribute even when `wo_a` is unquantized. Attention profiling without real
metadata normally skips this output projection, so successful construction
and profiling cannot validate this branch.

The A5 output projection now distinguishes the actual unquantized linear
method from the existing quantized path. The BF16 root loader retains
`[G * R, K]`; views of that weight are used for grouped `torch.bmm` followed
by `wo_b`. No placeholder scales, root conversion, artifact repack or
accepted expert/MoE kernel changes are made. The existing FP8 path retains
its required MX scales. CPU regression tests execute the real projection
method with operator stand-ins, covering M=1/M=3, BF16/FP16, wrong layout,
FP8 dispatch and the non-A5 path. They do not validate CANN execution.
The updated VQ2A8 CPU/unit suite passes 263 tests, including live output
before child exit, merged stderr, Unicode/large-output draining, heartbeat,
timeout/failed-exit retention and the final-flush shutdown race.

All acceptance stages now mirror complete child stdout/stderr live to the
terminal and retain it in `probe-*.log`. Redirection remains file-backed,
so a closed terminal pipe does not block the child's disk logging. After
15 seconds without output, a terminal-only `PROBE_WAIT` line reports elapsed
time and the last observed stage; it is not evidence of completed work.
Runtime imports, per-layer root loads, root tensor counts, engine readiness,
profile/prefill/decode forwards, decoder/MoE boundaries, logits and evidence
collection have explicit flushed progress lines in the offline adapter.
These diagnostics add no extra device-to-host tensor transfers. The compact
`summary.txt` and full `summary.json` remain available after success/failure.

### Timeout diagnosis and bounded cached execution

The user run at `0e5900b05f65b1f5bd5a80c95f16939660c96dc6`
(dirty worktree, `/tmp/vq2a8-acceptance-nupgtezj`) completed the 32-token
profiling forward and finite profiling logits. Its first real 10-token
prefill reached layer 38 before the **whole child** hit its 3600-second
timeout. There is no evidence that decode started, or that generation
logits were non-finite. Incomplete finite/repetition checks now read
`unknown`; completed profile/prefill/decode counts are reported separately.

The previous policy retains only two packed experts per layer and groups
two tokens at a time. Normal routing selects six experts per token. LRU
eviction can cause repeated CPU slice validation and synchronous H2D copies,
including between profiling and the first prefill. Each projection also
prepares activation separately for every row. These are source-confirmed
overheads, not a measured explanation for every second of the NPU timeout.

The new `cached` policy changes residency, not the accepted M=1 math:

- After strict root loading, measure free/allocated/reserved HBM and plan a
  lazy, packed-only cache across all layers. Reserve 16 GiB by default for
  KV, temporary allocations and allocator overhead. The plan also respects
  the engine's 0.9 memory fraction; explicit budgets larger than the safe
  current budget fail before expert loading. A budget too small for even
  one expert per layer also fails. No full-model dense expert weights are
  allocated and no artifact repack is needed.
- Retain up to 256 experts per layer **only if the budget fits**. Single-
  expert hash layers consume one slot, not 256. A smaller budget selects a
  bounded common cap; `MODEL_CACHE_PLAN` shows the actual result. Allocation
  remains lazy, and profiling caches survive into prefill and decode.
- Keep the accepted two-token grouping and per-row RHT/dynamic-A8
  preparation and M=1 kernel calls. Router slot order, duplicate hash
  routes, shared-expert addition, routed scaling and finite-value checks
  are preserved. Batch preparation is not enabled by this policy.
- Time CPU load/validation, completed H2D transfer, activation preparation
  and completed packed projections separately. These synchronized wall
  times include Python/launch/validation overhead; they are not device-
  event kernel benchmarks. File-backed page faults may also occur during
  H2D. Expert start/load/done progress is streamed live and saved to disk.
  Compact totals cover completed MoE calls only, including profiling;
  unfinished calls remain visible in the detailed log.

The byte plan rounds each tensor allocation to 512 bytes but is not a
fragmentation guarantee. Another process can consume memory after planning.
Do not shrink the reserve to force a full cache on a busy card. Optional
`--cache-budget-gib` sets a stricter model cache bound (`0` means automatic);
`--cache-reserve-gib` changes the reserved headroom. `--execution-policy
baseline` preserves the old two-expert/two-token policy for comparison.
The one-hour timeout, eager mode and launch-blocking diagnostics stay in
place; merely increasing the timeout is not the optimization.

There is also an important dtype distinction: the root checkpoint stores
attention linears in BF16, but the original NVIDIA `VQ2A8Config` chooses
online FP8 linear methods (`wo_a`: per-block; other linears: per-tensor).
The offline Ascend adapter deliberately runs unquantized BF16 root linears.
That is a diagnostic execution fallback relative to the original online
FP8 path, not evidence that original attention inference was BF16. Routed
expert codebooks/activations remain E4M3; the NPU M=1 kernel still uses FP32
Vector MAC/reduce, with `NATIVE_FP8_DOT=False`. Restoring equivalent online
FP8 root execution is a separate, hardware-validated milestone.

Batch preparation/chunk=32 was explored locally but withdrawn before
delivery: real CUDA layer-0 M=32 deterministic inputs exceeded the existing
CPU-oracle tolerance, and new/old execution did not agree. The old policy
also exceeded the CPU oracle on that expanded case. Separately, layer-3
`small:m=10` showed CPU/CUDA router ordering differences in **both** policies.
These are retained validation limits, not passes or evidence about NPU
behavior. No comparison threshold was relaxed. The delivered cache policy
instead requires **exact** agreement with the old same-device policy,
in addition to the existing CPU-oracle and chunk-invariance checks.

The delivered policy passes 299 CPU/unit tests. On the remote NVIDIA L20X,
real layer 0 and layer 3 weights passed 12 cases (deterministic/zero inputs,
M=1/3/10), with exact old-policy agreement and three exact repeats. Layer 3
retained 56 distinct experts across those checks, with 56 loads, 636 hits
and zero evictions. These are cache observations, not NPU throughput data.
Ruff and Markdown lint pass. The repository-wide `format.sh ci` could not
run because the test environment lacks `pre-commit`.

The Ascend host is not directly accessible from the development workspace.
Run the new policy first through the independent layer gate. Only then run
the full model (which still performs its original 32-token profiling pass):

```bash
cd /home/g00872988/vllm-ascend-vq2a8
git pull --ff-only origin ascend950-vq2a8
python3 tools/validate_vq2a8_tp1_acceptance.py \
  --stage moe --execution-policy cached \
  --model /home/g00872988/DeepSeek-V4-Flash-VQ2A8-32x256 \
  --physical-npu 4 --layers 0,3 --token-counts 1 3 10 \
  --cases deterministic zero --warmups 0 --repeats 3 \
&& python3 tools/validate_vq2a8_tp1_acceptance.py \
  --stage model --execution-policy cached \
  --model /home/g00872988/DeepSeek-V4-Flash-VQ2A8-32x256 \
  --physical-npu 4 --timeout 3600
```

A standalone or full offline PASS still does not certify independent
full-model logits, original quantization equivalence, generation quality,
native FP8 Cube compute, or HTTP serving. In particular, no NPU speedup is
claimed without the new stage timings from the actual Ascend950 run.

### Phase 1: accepted offline baseline and CPU validation optimization

The Ascend950 user report `/tmp/vq2a8-acceptance-u8qzcixx` at
`f9e802da5eb14fe40f850ac5c7c2177bc9e48a3a` (dirty worktree) passes the
complete short offline gate: 43 layers, 10 prompt tokens, three decode
steps per request, two exactly repeated logits/token sequences. Generated
IDs are `[223, 20, 16, 1]`, decoded as a space and `2.` followed by EOS. This is
execution evidence, not independent original-model quality certification.

The reported completed MoE calls total 1520.473 s host load/validation,
8.980 s H2D, 29.826 s preparation and 465.214 s packed projections. These
include profiling and both requests; they overlap forward totals and
must not be added to them. There were 1514 cache loads, 7798 hits and no
evictions. Actual packed residency was 9.096 GiB; 61.541 GiB is the planned
full cache, not a measured full-residency peak. The second request took
113.086 s; the last decode forward took 8.845 s. Root linears remain the
BF16 diagnostic fallback and the NPU kernel remains pure Vector FP32 MAC
with E4M3 storage, not native FP8 Cube multiplication.

Phase 1 changes only CPU payload validation and diagnostics:

- Remove full unpacking solely to check `code >= 16`. Every four-bit
  pattern is legal for the 16-entry codebook and the unpacker already masks
  with 15; this check cannot reject a payload. Retain shape, dtype, stride,
  high padding nibble, finite codebook/normalization, tile ID population
  and sign checks. Padding validation widens only the last packed word
  per row. Repack's independent bitwise round-trip is unchanged.
- Do not change payload bytes, artifact format, cache policy, routing,
  per-row RHT/A8 geometry, shared experts, attention or either GPU kernel.
- Split `host_read_s` and `host_validate_s` inside `host_load_validate_s`.
  These are wall times, not disk counters: mapped-file page faults can
  occur in later validation/H2D. `HOST_BREAKDOWN` is a subset of the old
  total, never an additional cost. Record CPU intra/inter-op thread counts
  without changing them.
- Add a CPU-only warm-payload microbenchmark comparing retained checks
  against retained checks plus the removed full-grid work. It is explicitly
  a legacy-work replay, not a cold disk benchmark or a run of an old commit.
- Add `--baseline-report` to the model acceptance stage. Copy the old
  report/logits into a new snapshot, verify saved logits SHA256, preserve
  recorded source hashes/git status, and capture current relevant source
  files and the current tracked dirty diff. Never clean the user's tree.
  The old dirty source bytes cannot be reconstructed from hashes: capture-
  time files and small model/tokenizer/manifest identity hashes are labelled
  separately from historical run evidence. A manifest hash does not verify
  all checkpoint payloads.
- Before model construction, verify the frozen files, critical package
  versions and unchanged compute-source hashes. Compare each new run's
  prompt, execution metadata, tokens, dtype/shape and full logits against
  the corresponding old run, with exact equality. Missing evidence,
  changed identity or a mismatch fails; new logits/failure details remain
  available. This is same-implementation regression, not phase 2's
  independent reference. `LOGITS_REFERENCE_VERIFIED` remains false.

The one-command driver below preserves the baseline first, then runs a
CPU validation benchmark, the accepted 12 real-weight MoE cases, and the
complete model regression. Each step streams output and retains its log;
failure skips remaining steps. Device-child timeouts remain owned by the
acceptance supervisor, with no competing outer timeout that could orphan
its worker. No repack, TP4 or HTTP server is launched.

```bash
cd /home/g00872988/vllm-ascend-vq2a8
git pull --ff-only origin ascend950-vq2a8
python3 tools/validate_vq2a8_tp1_phase1.py --model /home/g00872988/DeepSeek-V4-Flash-VQ2A8-32x256 --physical-npu 4 --baseline-report /tmp/vq2a8-acceptance-u8qzcixx
```

Keep the previous report until snapshot completion. `PHASE1_REPORT_DIR`
contains `baseline/`, `host.json`, `moe/`, `model/`, full step logs and
`phase1.json`. The final short report must show `BASELINE_EXACT=PASS
completed=2/2`, not merely the offline execution PASS. The driver labels
success `PHASE1_REGRESSION=PASS`; performance requires reviewing host
breakdown and cold/warm timings, not inferring speed from correctness.

Development validation uses the accessible NVIDIA host, not Ascend950.
Four real CPU payloads (layers 0/3, gate-up/down, 96 CPU threads,
torch 2.13.0+cu129) retained identical bytes. Five-sample validation medians
were 1.88-4.70 ms for legacy-work replay versus 0.41-0.54 ms for retained
checks. These warm CPU measurements do not predict the user's 1520 s
Ascend host total or warm NPU decode speed. Hardware acceptance subsequently
arrived in the user report described below.

The phase-1 VQ2A8 unit suite passes 330 tests. The real CUDA checks for
layers 0/3 pass all 12 accepted deterministic/zero, M=1/3/10 cases with
exact same-device baseline agreement and three exact repeats. Full logs
are retained on the development host under
`/tmp/vq2a8-acceptance-44xgxxtr` and `/tmp/vq2a8-acceptance-1q6cxztn`.
That scratch worktree reports its old git HEAD; the synchronized source
hashes in each report identify the tested files. It is not an Ascend run.
Ruff and Markdown checks pass. The required `bash format.sh ci` was
attempted but remains unavailable because the test environment does not
have `pre-commit`; this is not reported as a repository-wide CI pass.

### Explicit remaining plan (user decision)

1. Phase 1: complete; actual Ascend950 exact regression passed. Retain the
   accepted baseline and all valid corruption checks.
2. Phase 2: **skipped by user request**. Do not generate an independent
   full-model reference or mark independent logits/quality as verified.
3. Phase 3: active by user request. Implement opt-in online FP8 root
   projections with focused operator/attention regression; real Ascend950
   acceptance is still required. Skipping phase 2 does not supply evidence
   of original full-model equivalence.
4. Phase 4, deferred: profile and optimize packed NPU execution. Keep the
   accepted Vector baseline; isolate Cube/FP8, alignment and CV/fixpipe
   experiments in microkernels before integration. No new kernel in phase 1.
5. Phase 5, deferred: broaden prompts/context/compressor boundary and
   residency tests, then normal vLLM/HTTP streaming and request lifecycle.
   Report unverified quality explicitly rather than treating service success
   as quality evidence. TP4 and further repacking are not current tasks.

### Phase 1 hardware result (2026-09-06)

The user report `/tmp/vq2a8-phase1-m2nu1ipf` at `69d5b0b3` (dirty worktree)
shows `PHASE1_REGRESSION=PASS` and `BASELINE_EXACT=PASS completed=2/2`.
Both requests retain exactly the old tokens and full logits (maximum
absolute error zero). This is not an independent reference.

Completed host load/validation fell from 1520.473 s to 115.762 s;
the new breakdown is 1.378 s read and 114.326 s validation. Total prefill
time over two requests fell from 1499.837 s to 282.465 s. However, packed
projection totals were effectively unchanged, 465.214 s versus 465.273 s,
and the last warm decode was 8.845 s versus 8.920 s. Thus phase 1 removed
cold host work, not the steady-state expert-compute bottleneck. Peak
allocated/reserved memory remained 24.955/25.113 GiB. These are the user's
measurements, not developer access to the Ascend machine.

### Phase 3: original online FP8 root projections

The target is the pinned NVIDIA source
`2d75468d44857582f9d21c983d451d69bea50ad7`, specifically its SM90/Cutlass
per-token activation policy and SM90 `wo_a` recipe `(1,128,128)`.
BF16 is the root checkpoint's storage format; online FP8 is a load-time
conversion followed by runtime activation quantization. Neither changes
the existing VQ2 expert artifact or its E4M3 codebooks.

| Root module | Weight conversion | Activation / execution |
| --- | --- | --- |
| `wq_a`, `wq_b`, `wkv`, `wo_b`, `indexer.wq_b` | E4M3, per-tensor FP32 scale | E4M3 per token; CANN FP8 matmul, BF16 output |
| `wo_a` | E4M3, 128x128 FP32 scales | FP32 inverse RoPE, 1x128 power-of-two activation scales; independent output groups |
| Compressor projections, `indexer.weights_proj`, embedding/head, norms/router | Unchanged | Not implicitly quantized by the root allowlist |

This is **not** the existing A5 MXFP8 32-element recipe. `wo_a` weight scales
stay FP32 and are not rounded to powers of two. The inverse RoPE partner
product is rounded in FP32 before `addcmul`; there is no intermediate
BF16 cast. The changed multiply-add order fixed a real CUDA
`wo_a`, M=10, small-input FP8 rounding mismatch against the original fused
inverse-RoPE kernel. No new expert kernel or CV scope experiment is used.

The native backend calls `torch_npu.npu_quant_matmul`. Its block-scale
transpose **strides** must match the transposed weight; making only the
scale contiguous is incorrect. This contract is visible in the official
[op-plugin implementation](https://github.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/QuantMatmulKernelNpuOpApi.cpp).
The documented FP32 scale shapes and G-B grouping are described in
[aclnnQuantMatmulV5](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/API/aolapi/context/aclnnQuantMatmulV5.md).
Public documentation/source is not verification of the user's exact
torch-npu/CANN wheel. The isolated native smoke gate must run on that wheel.

Implementation boundaries:

- Only `--root-linear-mode online_fp8_sm90` installs the new methods on the
  offline inheritance adapter. Default `bf16` and unrelated A5/non-A5
  attention branches are retained. The custom linear wrapper's cached
  quantization-method reference is updated too.
- Canonical BF16 parameters are allocated and strictly loaded first.
  vLLM's per-module post-load hook converts only the allowlist; FP32 scale
  buffers do not masquerade as checkpoint parameters. Cache budgeting runs
  after conversion, against actual root FP8 residency. Conversion rejects
  nonfinite, wrong-dtype, noncontiguous or unsupported block shapes.
- Forward evidence requires processed FP8 weights/FP32 scales and one
  call per main root projection per real model step, with counters reset
  after profiling. Indexer weights must be processed, but short sequences
  may not exercise index selection; the separate real indexer projection
  probe does not certify long-context indexer behavior.
- `NATIVE_FP8_ROOT_MATMUL` is separate from
  `NATIVE_FP8_EXPERT_DOT=False`. The legacy `NATIVE_FP8_DOT=False` remains
  the expert-kernel claim. No silent BF16 or eager reference fallback is
  provided for unsupported NPU FP8 matmul.
- The historical BF16 model report is preserved, not used as a bit-exact
  FP8 oracle. Full-model independent logits, quality and serving remain
  unverified because phase 2 was skipped and phase 5 remains deferred.

Operator gates compare FP8 bytes/scales between CPU and the NPU preparation
path exactly, then native output against explicit FP32 dequantized math
with one final BF16 cast. Limits are `rtol=0.01`, `atol=0.03125`, relative
L2 <= 0.01, plus three bitwise-identical output executions. These are new
root-operator limits, not relaxed expert gates or a full-model tolerance.
Cases cover deterministic, zero, impulse and small inputs at M=1/3/10/32.
The seven real projections cover layer-0 primary linears, layer-2 indexer
`wq_b`, and layer-42 `wo_a`; grouped `wo_a` exercises all eight groups.

The developer CUDA reference tool verifies referenced Python functions
against the pinned source and compares original FP8 preparation separately
from output accuracy. In the real layer-0/layer-42 `wo_a` matrices,
compiled upstream quantization and eager preparation differ at 82/147
of 33,554,432 FP8 bytes respectively, maximum quantized-value difference
0.001953125, despite identical scales. Therefore **upstream weight byte
equivalence is not claimed**. The five per-tensor probe weights match
exactly. This distinction remains visible in developer output.

Development validation passes 381 VQ2A8 unit tests and the 196-check CPU
synthetic operator gate. On the remote NVIDIA L20X, all seven real roots
pass 112 cases against original Cutlass/DeepGEMM output. FP8 activation
bytes/scales match the original per-token quantizer and fused inverse-RoPE
quantizer exactly for these cases. The successful native CUDA log is
`/tmp/vq2a8-phase3-dev-0hs3gJ/cuda-reference.log`. CPU/CUDA eager weight and
activation bytes/scales also match exactly. Block weight scale calculation
explicitly multiplies the FP32 reciprocal of 448 to reproduce the pinned
CUDA scalar-division behavior; CPU scalar division otherwise differs by
one FP32 ULP. A unit test pins this boundary. This developer check needs
the installed CUDA toolkit (`CUDA_HOME=/mnt/miniconda3/envs/gyc` on that
host); no CUDA setting belongs in the Ascend command. Ruff and Markdown
checks pass. Required `bash format.sh ci` was attempted but cannot run
without the local `pre-commit` executable, so full repository CI is not
claimed. No actual NPU phase-3 execution has been performed by the agent.

On the disconnected Ascend950, run this single command sequence; no
repack, old-report argument, TP4, HTTP server or environment upgrade is needed:

```bash
cd /home/g00872988/vllm-ascend-vq2a8
git pull --ff-only origin ascend950-vq2a8
python3 tools/validate_vq2a8_tp1_phase3.py --model /home/g00872988/DeepSeek-V4-Flash-VQ2A8-32x256 --physical-npu 4
```

The driver creates a new `/tmp/vq2a8-phase3-*` directory and runs
`smoke -> roots -> model`. Operator children have a 1800-second limit;
the model acceptance supervisor owns its existing 3600-second timeout.
Any abort, timeout, missing result or backend mismatch stops the sequence.
Exit zero alone cannot pass. Logs stream to the terminal and are also
retained in `smoke.log`, `roots.log`, `model.log` and model child logs.
`summary.txt` is short; `phase3.json`, operator JSON and model evidence
retain details. On a failure, provide the short summary and the first
error block from the failed step; another full model run is not required
to diagnose a smoke/real-projection failure.

### Phase-3 Ascend rounding failure and focused correction (2026-09-06)

The user report `/tmp/vq2a8-phase3-ij7h4_0p` stopped in real-root testing at
`layers.0.attn.wo_a.weight:10:small:g6:activation_bytes`. This is a strict
activation encoding failure, not a matmul exception, expert kernel failure,
or completed phase-3 acceptance. Earlier shown grouped outputs and repeats
passed; the model step was correctly skipped. No repack is needed.

On the accessible developer host, replacing fused inverse-RoPE FMA with
separate FP32 products/addition reproduces exactly one differing byte at
group 6, token 6, channel 970:

| Quantity | Fused | Unfused |
| --- | --- | --- |
| FP32 input bits | 885522432 | 885522433 |
| Scale | 1.862645149230957e-9 | same |
| Scaled value | 200.0 (FP8 midpoint) | 200.00001525878906 |
| E4M3 byte / decoded value | 116 / 192 | 117 / 208 |

This is a strong rounding fingerprint, **not direct proof of the installed
NPU kernel's instruction sequence**. The public
[op-plugin Addcmul implementation](https://github.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/AddcmulKernelNpuOpApi.cpp)
dispatches to `aclnnAddcmul`; its mathematical formula is not a promise
that every backend has the same single-rounding behavior as the CUDA FMA.

The correction explicitly uses
[Triton FP32 FMA](https://ascend.github.io/docs/sources/_generated/sources/triton-ascend/triton_api/Math_Ops/fma.html)
on contiguous vectors, with the partner product rounded by a separate eager
multiply before kernel launch. This small pure-Vector kernel has no FP8
loads, Cube, fixpipe, or expert logic. CUDA executes the same new kernel for
developer validation; CPU keeps the existing reference. CANN FP8 matmul,
weight/activation quantization recipes, and all acceptance limits remain
unchanged. No BF16 fallback or tolerance relaxation is introduced.

Both smoke and roots now begin with a weight-free M=10/G=8 regression.
It checks quantization of **identical CPU-prepared FP32 inputs** separately
from full inverse RoPE, then all eight groups' FP8 bytes/scales. It also
prints the old `addcmul` versus explicit-FMA boundary values on the actual
device. Failed activation checks retain mismatch counts and up to eight
coordinates with input bits, scales and decoded FP8 values in JSON and
terminal logs. Byte-space L2 is not a decoded activation norm: for example,
negative zero has byte 128 but numerical value zero.

The corrected kernel passes the original 112 real-root SM90 operator
comparisons in `/tmp/vq2a8-phase3-rounding-NdXUDU/cuda-reference.log`.
The full VQ2A8 test set passes 389 tests, including CUDA FMA broadcast/tail
checks and the exact midpoint regression; CPU smoke passes 234 checks.
Original compiled-weight byte differences remain disclosed above; they
are unrelated to this fix. NPU acceptance of this correction is pending:
rerun the same one-command phase-3 driver. It still stops before roots/model
if the new rounding preflight fails, and does not execute phases 2/4/5.

### Phase-3 actual A5 FMA failure supersedes the previous correction (2026-09-06)

The subsequent **real Ascend950 result at `14f55ce8` failed**, in
`/tmp/vq2a8-phase3-u561cz8n`, before real weights or the model were tested.
The earlier CUDA PASS did not establish A5 single-rounding behavior.
The same-input FP8 bytes/scales passed, but inverse-RoPE FP32 had 10077
different elements (maximum absolute error `5.684341886080802e-14`). At
group 6/token 6/channel 970, both explicit `tl.fma` and eager `addcmul`
returned `3.7252905826790084e-7`, whereas the fused reference returned
`3.725290298461914e-7`. This reproduces the previously identified FP8
midpoint failure. Calling an API "FMA" was not sufficient to fix it.
This evidence describes the installed build/path, not every A5 instruction.

The public
[Triton-Ascend FMA test](https://github.com/Ascend/triton-ascend/blob/main/third_party/ascend/unittest/generalization_cases/test_general_fma.py)
uses `x * y + z` with tolerance-based validation; it does not establish
bitwise single-rounding conformance for this installed A5 build.

The replacement small accelerator kernel implements single rounding with
integer significands: normalize FP32 operands (including subnormals), form
the exact 48-bit product, align product/addend with guard and sticky bits,
then round the signed sum to nearest-even. Intermediate magnitudes fit
int64. Shift counts are bounded even in unselected branches. It handles
signed zero, cancellation, subnormal rounding and overflow; NaNs are
canonicalized rather than promising reference NaN-payload identity.
It uses neither floating `tl.fma`, FP64, CPU transfers in the hot path,
nor Cube/fixpipe. This is a correctness workaround, not an acceleration
claim. Installed A5 integer lowering and performance still require testing.

Smoke and roots now start with an 11-case, weight-free primitive check
(including ties, subnormals, signed zeros and overflow cancellation),
followed by the existing exact inverse-RoPE/FP8 regression. Primitive
results are compared as bytes to preserve every FP32 bit and zero sign.
`ROOT_FP8_FMA` identifies `integer_single_rounding_rne` and prints bounded
hex encodings. All FP32/FP8 exactness checks and projection error limits
remain intact. CANN native FP8 root matmul, the online quantization policy,
BF16 historical baseline, expert artifacts and expert compute are unchanged.
No repack or blanket BF16 fallback is introduced.

Developer evidence in `/tmp/vq2a8-phase3-integer-aAG2qr`:

- All 397 VQ2A8 unit tests pass, run with
  `--confcutdir=tests/ut/quantization` to exclude NPU-wide integration fixtures
  on the NVIDIA development host. This is not the entire Ascend test suite.
- The new Triton kernel matches an independently implemented arbitrary-width
  integer oracle on 20359 edge/random triples, and CUDA hardware FMA on
  524288 random/cancellation triples. Finite results and signed zeros are
  checked bitwise; hardware NaN payload differences are not treated as bugs.
- The original 112 SM90 real-root comparisons pass; FP8 activation encodings
  are exact. Previously disclosed compiled/eager **weight** byte differences
  remain (`weights_bitwise_equal=False`); no full-model reference is claimed.
- CPU smoke passes 235 checks. Ruff and Markdown lint pass.
  `bash format.sh ci` cannot complete because local `pre-commit` is absent.

**Resource constraint and next real-device run:** this report's device is
still physical 4 / logical `npu:0`, SoC260 / `Ascend950PR_958b`, CANN9.1,
torch2.10 / torch-npu2.10.post4, Triton-Ascend3.2.2 dev20260729205041.
It reports only 7046828032 free bytes (**6.56 GiB**) of 115662127104 total;
that snapshot is insufficient for the earlier full-model footprint. It does
not identify the occupying process. Do not kill processes, reset the device,
switch cards or upgrade the environment implicitly.

Use the new operator-only mode first:

```bash
cd /home/g00872988/vllm-ascend-vq2a8
git pull --ff-only origin ascend950-vq2a8
python3 tools/validate_vq2a8_tp1_phase3.py --model /home/g00872988/DeepSeek-V4-Flash-VQ2A8-32x256 --physical-npu 4 --operators-only
```

This runs `smoke -> roots`, streams and saves logs, and never constructs
the full model. The expert artifact need not be present for these operator
checks. Success is explicitly
`PHASE3_OPERATORS=PASS PHASE3=INCOMPLETE MODEL=NOT_RUN`, with
`root_fp8_execution_verified=False`; a failed or missing child result
still stops the driver with a nonzero exit. `phase3.json` records the scope,
planned steps and operator verification separately, and operator JSON now
preserves the device memory snapshot. Once these gates pass and the device
has sufficient available HBM, the normal command without `--operators-only`
runs all three steps. No phase-3 completion is claimed before the actual
root-FP8 offline model passes. Phase 2 remains skipped; phases 4/5 remain
deferred. The Ascend950 is still not directly accessible from this workspace.

### Phase-3 operator acceptance received; model still queued (2026-09-06)

The user's report `/tmp/vq2a8-phase3-wyjoj_yc` passes smoke and all 1061
root checks on the Ascend950. The last shown layer-42 `wo_a`, M=32, small
case has relative L2 error `1.7344302472137305e-5` and exact repeats.
This supersedes the pending **operator** status above, not the pending
full-model status: `PHASE3=INCOMPLETE completed=2/3`, `MODEL=NOT_RUN`,
`ROOT_FP8_OPERATORS_VERIFIED=True`, `ROOT_FP8_EXECUTION_VERIFIED=False`.
The agent still cannot connect to this NPU. No inference about currently
available HBM is made from the earlier memory snapshot.

### Phase 4: isolated packed-kernel development while the NPU is queued

The user activated this stage before phase 3's model run became available.
Its purpose is to reduce the remaining expert-compute cost, not repeat the
already accepted host-loading optimization. Phase-1 timing showed almost
unchanged packed-projection totals and approximately 8.9 s warm decode.
Those measurements motivate optimization; they do not predict a speedup.

The implementation adds an experimental `vq2a8_vector_gather.py` kernel:

- Affine, contiguous-tail loads fetch E4M3 codebook bytes once per output
  group. Lookup uses `tl.gather` on an on-chip tensor instead of the old
  nested codebook-entry selection loops. This is not a dynamic one-byte
  gather from GM. Hardware/compiler acceptance is still required.
- Packed int32 words, four-bit indices, two-component codebook vectors,
  arbitrary validated column-tile IDs and FP8 storage are unchanged.
  Arithmetic remains **Vector FP32 MAC/reduce, not native FP8 dot**.
  The candidate does not allocate a dense expert weight matrix in GM.
- M=1 through 32 uses one host kernel launch with separate row programs.
  Each row retains K=512 reduction geometry and the accepted per-row
  RHT/dynamic-A8 preparation. It does not batch the quantization arithmetic
  or assume different reduction trees give identical FP8 encodings.
- The candidate explicitly bounds column tiles to 32. The largest logical
  lookup table is 128 KiB of FP32 byte values, before indices and other temporaries;
  this is not proof of the A5 compiler's UB allocation. Invalid shape,
  dtype, stride, device or required base alignment is rejected. Tensor
  values must come from the existing validated artifact/preparer.
- The accepted `vq2a8_triton.py`, cache/execution policy, offline adapter
  and model default backend are unchanged. No repack, environment upgrade,
  new serving registration or runtime fallback selection is introduced.

`validate_vq2a8_tp1_phase4.py` runs bounded, separate child processes:

1. `lookup`: twelve synthetic configurations cover M=1/3/32 and 1/3/16/32
   column tiles, signed packed words, every nibble/component, output groups
   and scrambled tile IDs. Compare with a literal CPU FP64 decode oracle
   and the existing kernel on identical prepared FP8 inputs.
2. `native`: a small CANN FP8 matmul capability check with **synthetic**
   32x512 operands, independent of packed expert decode.
3. `packed`: by default all seven expert probes at M=1/3/10/32, four input
   cases, both projections. Check the same-prepared-FP8 CPU FP64 oracle,
   existing per-row kernel, and complete candidate/accepted SwiGLU chains.
   Candidate row splitting and at least three repeats must be bit-exact.
   Existing prepared-input and chain error limits are retained.
4. `benchmark`: all requested experts and row counts, deterministic case,
   resident packed tensors. Each backend gets two measurement rounds in
   reversed order, at least three warmups and ten samples per round.
   JIT/first-call and activation preparation times are reported separately.

Benchmark JSON separates device-event span and synchronized wall time,
with min/median/p95. These are not a full-model tokens/s measurement.
Event spans can include gaps between host-submitted launches; wall times
also include wrapper allocation/validation, dispatch and synchronization.
The accepted baseline launches once per row, the candidate once per batch.
Reports identify the baseline backend: on CUDA it is the portable FP8-dot
kernel, **not the accepted A5 Vector body**. CUDA speedup cannot be used as
an Ascend speedup. Allocator peak deltas and input layouts are retained;
allocator counters do not establish on-chip buffer layout or compiler
spill behavior. Native expert dot remains false even if a microtest passes.

Optional `--include-cv` adds two isolated synthetic Triton tests, direct
FP8 dot followed by a Vector byte-sign-transform feeding FP8 dot. They
target the earlier Cube/CV/MTE failures without decoding real experts.
They are off by default because those compiler paths have previously
aborted. A direct or bridge micro PASS is not proof that the fused packed
kernel works. No generated IR patch, speculative synchronization switch,
device reset or other-process termination is attempted.

Each child has a default 1800-second timeout. A nonzero exit, abort,
timeout, missing/partial evidence, wrong backend/device or missing test
coverage stops remaining steps. Logs stream to the terminal and disk;
`summary.txt` shows steps and boundary-row timing ratios, while
`phase4.json` and per-child JSON preserve all row sizes, metrics, source
hashes, package versions and failure details. Even a complete standalone
PASS reports `PHASE4_KERNEL_GATES=PASS PHASE4=INCOMPLETE`:
`performance_verified=False`, `model_integration_verified=False`.
Passing numerical gates or measuring timings alone does not promote a
candidate or certify a performance improvement.

Developer evidence in `/tmp/vq2a8-phase4-dev-PoQQMZ` on the NVIDIA L20X
(torch 2.13/CUDA 12.9, Triton 3.7.1; **not** the user's Ascend wheel):

- All 452 VQ2A8 unit tests pass, including candidate CUDA numerics and
  repeat/chunk invariance, malformed evidence, coverage, timeout/abort
  handling, benchmark validation and compact reporting.
- The four synthetic driver steps pass. Direct/CV micro results establish
  only the CUDA path; CANN and A5 code generation are not executed here.
- Real experts `0:0`, `3:127` and `42:255` pass 96 projection records
  (four row counts, four input cases, both projections) with prepared-input
  CPU FP64 and accepted-chain comparisons. Development uses existing
  partial per-layer artifacts; that bypass is forbidden for NPU acceptance.
- A separate, non-overlapping benchmark session passes eight layer-0
  projection/row configurations. Ratios compare against the **CUDA**
  portable baseline and cannot predict NPU speed. The measured candidate
  allocator increment at M=32/N=4096 is 262144 bytes, the BF16 output size;
  this does not measure UB/shared memory or establish A5 spill behavior.
- Ruff format/check, Markdown lint and `git diff --check` pass.
  Required `bash format.sh ci` was attempted but local `pre-commit`
  remains absent; full repository CI and NPU tests are not claimed.

When the assigned NPU becomes available, the normal bounded command is:

```bash
cd /home/g00872988/vllm-ascend-vq2a8
git pull --ff-only origin ascend950-vq2a8
python3 tools/validate_vq2a8_tp1_phase4.py --model /home/g00872988/DeepSeek-V4-Flash-VQ2A8-32x256 --physical-npu 4
```

For a weight-free first test use `--micro-only`; only add `--include-cv`
when specifically testing the experimental compiler path. This does not
require launching the full model. Only standalone benchmark children
set `ASCEND_LAUNCH_BLOCKING=0`; correctness children retain the existing
blocking diagnostics. All remain on physical 4 / logical `npu:0`.

The remaining hardware sequence is: accept the candidate's NPU lowering
and numerics, review actual steady-state timings/layouts, then integrate
only an accepted optimization and rerun MoE and full-model regressions
after phase 3's root-FP8 model acceptance. Phase 4 is not complete until
those integration/performance gates pass. Phase 2 remains skipped and
quality is explicitly unverified; phase 5 serving work stays deferred.

## Phase 3 model: QLI AICPU available-core mismatch

The user's `/tmp/vq2a8-phase3-f4rt5hh5` run at `0aca98cf` completed
the 43-layer profile, but failed before the first real prefill forward.
Device logs identify the cause of 507018/22007 precisely:
`CheckSingleParam: Core num invalid: aic:28, aiv:64`.
The custom library and `RunCpuKernel` entry point both loaded successfully.
This is a parameter rejection, not the earlier Cube MTE alignment error,
FP8 rounding regression or an established timeout.

The host API passes `GetCubeCoreNum()` and `GetVectorCoreNum()` into the
AICPU scheduler. These runtime values need not match the earlier device
property snapshot of 32/64; why this process obtained 28 is not established.
Do not overwrite the runtime count with 32 or reset the device.

The QLI consumer maps Vector block `i` to LI slot `i / 2`. Scheduling 28
Cube slots with 64 available Vector cores has enough partners. Replace
the unnecessary divisibility check with nonzero Cube count, fixed ABI
capacity limits (36 Cube / 72 Vector), and at least two Vector cores per
scheduled Cube core. Initialize the entire defined metadata struct so
unused slots, including 28..31 in this case, are disabled rather than
uninitialized. The reserved 160-word output tail remains untouched.
Neither attention mathematics nor any FP8/VQ2 weight format changes.

`tools/validate_vq2a8_qli_metadata.py` runs a weight-free prefill/decode
metadata probe, checks defined ABI fields and three exact repetitions,
and records extension/package paths and hashes. A PASS establishes only
the tested metadata calls, not attention numerics or full-model execution.
The offline gate now also runs this preflight before constructing the LLM,
so a broken AICPU package does not first spend minutes on model profiling.

This fix changes compiled AICPU code: **git pull alone is insufficient**.
Rebuild in the user's existing CANN 9.1.0 environment, without replacing
torch, torch-npu, Triton or vLLM:

```bash
cd /home/g00872988/vllm-ascend-vq2a8
git pull --ff-only origin ascend950-vq2a8
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH"
export SOC_VERSION=ascend950pr_958b
export COMPILE_CUSTOM_KERNELS=1 MAX_JOBS=16
set -o pipefail
python3 -m pip install -v -e . --no-build-isolation --no-deps --force-reinstall 2>&1 | tee /tmp/vq2a8-qli-rebuild.log
build_status=${PIPESTATUS[0]}
echo "BUILD_EXIT=$build_status"
if [ "$build_status" -eq 0 ]; then
  python3 tools/validate_vq2a8_qli_metadata.py --physical-npu 4 2>&1 | tee /tmp/vq2a8-qli-preflight.log
  probe_status=${PIPESTATUS[0]}
  echo "QLI_EXIT=$probe_status"
fi
```

Only after `QLI_METADATA_PREFLIGHT=PASS` and `QLI_EXIT=0`, rerun the
existing phase-3 command. Its smoke/root/model requirements remain intact.
The 950 cannot be accessed from the developer workspace; the fix still
requires the user's rebuilt-package preflight and full-model acceptance.
Host tests compile the actual AICPU scheduler with a minimal CANN context
shim and undefined-behavior sanitization. They cover 28/64 and ordinary
core counts, short/zero-compressed/multi-core sequences, invalid counts,
complete nonoverlapping schedules, canaries and disabled-slot initialization.
Developer validation: all 499 VQ2A8 tests pass on the existing NVIDIA/host
test environment, including 36 actual C++ scheduler cases. Ruff check/format,
Markdown lint and `git diff --check` pass. `bash format.sh ci` was attempted
but is blocked by the local missing `pre-commit`; no CANN build or 950
hardware PASS is claimed by these development results.

## Phase 3 model: A5 KV-quant SAS metadata initialization

The user rebuilt `64b577a8` and passed all four weight-free QLI cases,
each repeated three times. That preflight reported `Ascend950PR_957d`,
28 Cube / 56 Vector cores and a different UUID from the older 958b
snapshot. Physical-device argument 4 is not a sufficient identity check
across runs. The original runtime 28/64 combination is not thereby
hardware-certified; keep the actual DEVICE/QLI_ENV records with each run.

The subsequent `/tmp/vq2a8-phase3-t426l1so` model run passed metadata
construction and entered real prefill. It failed in layer 0 at
`dsa_v1.py::_forward_prefill`, in the `compress_ratio <= 1` branch.
On A5, `DeviceOperator` selects `npu_kv_quant_sparse_attn_sharedkv` here.
Error 507015 with scalar-GM/DDR out-of-range diagnostics is different
from the previous QLI AICPU 507018/22007 rejection. The 406-second
profile did not establish that the real attention path was safe.

Source inspection and host execution of the real C++ scheduler establish
an independent defect in `KvQuantSparseAttnSharedkvMetadata::GenMetaData`:
the output is allocated with `torch::empty`, but only the runtime-count
prefixes are written. Unused flags and intervals in the fixed ABI retain
allocator contents. The A5 consumer reads FA slots by block index and
uses nonzero flags to enter the interval/address calculation. This is a
strong candidate for the observed invalid accesses, not a hardware-proven
complete diagnosis. Device-log core IDs alone do not prove logical block
indices, the actual launch envelope or platform-count consistency.

The fix initializes the entire **defined** KV-quant SAS struct (36 x 9 FA
words plus 72 x 8 FD words = 900 int32 words), then writes the schedule.
This is NOT QLI's 864-word ABI. The remaining 124 reserved output words
are untouched. For N128, retain the paired-core schedule and the idle
pair's `FA_S2_MAX_NUM` barrier counts. Do not replace the metadata tensor
with zeros in Python: doing so would also disable real attention work.
Reject zero counts and counts exceeding the fixed ABI before writing.
No weights, quantization modes, attention mathematics or launch core
counts are changed. Non-quantized SAS is outside this A5 fix's scope.

`tools/validate_vq2a8_sas_attention.py` now runs QLI followed by a
weight-free **SWA-only** attention preflight. It uses the production
packed-FP8 KV scatter, a nonzero page-table entry, the A5 SAS metadata
operator, and the actual attention operator for prefill and three decode
lengths. It checks all defined metadata words, exact repeats, finite
outputs, and an analytic oracle: Q=0, V=+/-0.5 and a zero-logit sink give
`V * visible_keys / (visible_keys + 1)`. This isolates a small nonzero
output check without depending on root/expert weights. It does not test
general attention scores, compressed DSA/indexer selection, model quality
or serving. Each substage is logged before launch. The offline validator
also invokes it before `LLM` construction/profile so failures are early.

After pulling this fix, rebuild custom ops as described above. For a clean
rebuild, only the generated `csrc/build`, `csrc/build_out`, `csrc/output`
directories need removal; never remove `csrc` or the model. The existing
build script replaces the installed `_cann_ops_custom` package after a
successful build. Run the new preflight in a fresh Python process:

```bash
(
set -euo pipefail
cd /home/g00872988/vllm-ascend-vq2a8
timeout 180s /usr/local/python3.11.10/bin/python3 -u tools/validate_vq2a8_sas_attention.py \
  --model /home/g00872988/DeepSeek-V4-Flash-VQ2A8-32x256 \
  --physical-npu 4 2>&1 | tee /tmp/vq2a8-sas-preflight.log
echo SAS_EXIT=0
)
```

Require both `SAS_ATTENTION_PREFLIGHT=PASS` and `SAS_EXIT=0` before rerunning
the full phase-3 command. Do not mark phase 3 complete from a preflight.
The earlier short/standalone successes and phase-2 skip are unchanged;
phases 4 and 5 retain their prior scope/status.

The developer cannot connect directly to the Ascend 950. A regression
test first failed against the old real C++ scheduler with a poisoned
output buffer, then passed after the fix. New host cases cover 28/64,
28/56, 32/64, 24/48, 36/72; 64/128 heads; SWA/C4/C128 schedules; prefill
and decode lengths; disabled slots, N128 pairs, fixed ABI capacity,
reserved-tail/canary preservation and exact repeats. The host harness
uses undefined-behavior sanitization; it is not a CANN or NPU simulator.
The new attention preflight still requires the user's actual NPU run.
Developer validation: all 656 VQ2A8 tests pass in the existing NVIDIA/host
environment, including 124 new real SAS C++ scheduler cases. Ruff check
and formatting, Markdown lint and `git diff --check` pass. The required
`bash format.sh ci` was attempted but cannot run without the local
`pre-commit` dependency; no complete CI or 950 build/run PASS is claimed.

## Phase 3 SAS preflight: separate metadata and compute argument contracts

The user successfully rebuilt `a7e0a7df` (`BUILD_EXIT=0`). The 957d
28/56 device passed all QLI cases again. SAS prefill completed KV scatter
and the first metadata validation (10 active FA cores), but attention
failed during **host tiling**, before device computation:
`cuSeqLensOriKv is not supported now, it must be nullptr`.
This is a bug in the new preflight's Python call, not evidence of another
failed build or a recurrence of the earlier device GM access failure.

The A5 metadata operator accepts `cu_seqlens_ori_kv`; the compute operator
does not. Its `CheckUnrequiredParaExistence` also rejects
`cu_seqlens_cmp_kv` and `ori_sparse_indices`, despite their presence in the
torch schema. The production A5 SWA call already omits these inputs.
The preflight now matches that call: retain query cumulative lengths,
`seqused_kv` and the page table, and pass cumulative KV lengths only to
metadata. Do not weaken the C++ tiling checks or change quantization.

For users who have already rebuilt `a7e0a7df`, this correction is
**Python-only**: pull the update and rerun the SAS preflight command above
in a fresh process. No C++ rebuild, cache clearing or reinstall is needed.
Both preflight PASS markers are still required before the full phase-3
model run. Attention numerics, the prior GM fault's resolution and full
phase-3 execution remain unverified until those NPU runs complete.

Regression tests check the forbidden compute inputs, retained metadata
lengths, and parity with the production A5 SWA keyword set/base settings.
Two tests fail against the old script, reproducing the argument mismatch
on the host without claiming to simulate CANN tiling.
After the Python fix, all 661 VQ2A8 tests pass in the existing NVIDIA/host
environment. Ruff, Markdown lint and `git diff --check` pass; the required
`bash format.sh ci` remains blocked by missing local `pre-commit`.

## Phase 3 accepted on the user's 950; phase 4 gather dtype correction

The user reports `PHASE3=PASS` at `57e48c73`, with evidence retained at
`/tmp/vq2a8-phase3-ej7e_5co`. All 43 layers executed for two runs of the
10-token prompt plus three decode steps. Tokens/logits repeat exactly,
root FP8 execution and native root matmul are verified by that gate, and
native expert dot remains false. Peak allocated/reserved memory was
20.453/20.500 GiB. This is operator plus short offline-execution acceptance,
not independent-reference, quality, performance or serving acceptance.
The generated IDs `223,20,201,671` differ from the historical BF16-root
baseline; preserve the new run's logits, source hashes and dirty-worktree
details as an FP8 regression baseline, not an independent numerical oracle.

The subsequent standalone phase-4 run, `/tmp/vq2a8-phase4-d8jody16`,
stopped at `lookup` before NPU execution. Its installed Triton-Ascend
frontend rejects the candidate's `tl.gather` **source** dtype `int32`.
The local version's diagnostic explicitly lists `fp32` as supported;
the index remains `int32`. Public
[gather documentation](https://triton-ascend.readthedocs.io/en/latest/python-api/generated/triton.language.gather.html)
also distinguishes the source tensor from its index tensor. Do not infer
A5 build compatibility from a different release or the CUDA backend.

Use on-chip FP32 values to carry the loaded unsigned bytes (0 through
255), gather them, convert back to uint8, then bitcast to E4M3. All byte
values are exactly representable, so this changes neither FP8 encodings
nor quantization arithmetic. The packed index words and lookup indices
remain integers. GM codebooks remain E4M3; there is no dense FP32 expert
allocation. The logical maximum table remains 128 KiB, but actual A5 UB
allocation/lowering and performance still require the hardware gate.

The phase-4 summary now says `PHASE3_MODEL=NOT_EVALUATED_BY_THIS_RUN`:
this standalone driver does not read phase-3 evidence and must not label
an already accepted model as still awaiting acceptance. It also must not
claim phase-3 PASS without evidence. The separate accepted phase-3 report
remains authoritative, and the default model backend is unchanged.

This patch is Python/Triton source only. Pull it and rerun the phase-4
driver; no C++ rebuild, reinstall, repack or cache deletion is needed.
The driver still stops on any failed child and does not enable the
experimental Cube/CV tests by default. Use one-line shell commands when
copying to avoid introducing Markdown fences into Bash continuations.

Regression checks cover all 256 byte encodings, including signed zeros
and NaN bit patterns as transport-only cases (NaN artifacts stay invalid).
A CUDA development test inspects the actual candidate's generated
`tt.gather` IR for FP32 source and int32 indices and checks its output
against the CPU oracle. It fails against the old int32-source candidate
and passes after the fix. This is not an Ascend compiler or device test.
All 663 VQ2A8 development tests pass on the existing NVIDIA/host system,
including candidate numerical, batch/chunk and repeat checks. Ruff,
Markdown lint and `git diff --check` pass. Required `bash format.sh ci`
was attempted but remains blocked by missing local `pre-commit`.

## Phase 4: preserve the typed codebook pointer through memory-scope inference

At `6b929739`, the user's `/tmp/vq2a8-phase4-cawreadd/lookup.log`
shows that the FP32 gather source passed the frontend. Compilation then
failed in `InferHIVMMemScope`, before device execution. The diagnostic
tail identifies the offending operation, not an allocation-size error:

```text
'func.func' op Failed to propagate memory scope for argument #6
'builtin.unrealized_conversion_cast' op Unsupported user for root alloc op.
```

The dumped entry function's argument #6 is the FP8 codebook pointer.
Casting it to a uint8 pointer leaves an unrealized memref conversion
that this A5 memory-scope pass rejects. Change the candidate to the
accepted Ascend kernel's loading pattern: load from the typed FP8
pointer first, bitcast the loaded values to uint8, then numerically
convert those byte values to the supported FP32 gather carrier.
The masked-load fallback is floating-point zero, avoiding an unsupported
int32-to-FP8 cast in the development frontend. FP8 encodings, codebook
layout, integer indices, K=512 reduction and output arithmetic are unchanged.
No compiler flags, accepted model backend, C++ operators or gate tolerances
are changed. The earlier int32 gather source must not be restored.

The phase-4 driver also retains a bounded, deduplicated diagnostic excerpt
in each failed step's JSON and short report, with the full log path. It
reads the existing log after child exit; it neither truncates the original
IR dump nor reruns a failed kernel. PASS still requires the original
complete numerical, coverage and execution evidence.

The actual candidate IR regression reproduces the pointer cast with the
old source and requires it to be absent after this change, while retaining
the FP32 gather check and CPU numerical oracle. NVIDIA development checks
do not establish Ascend backend support: the user's lookup and remaining
phase-4 gates must be rerun. This remains a standalone experimental kernel,
not model integration, performance acceptance or native FP8 expert dot.

This is a Python/Triton update. For the existing rebuilt installation,
pull it and rerun the same phase-4 command in a fresh process; no C++
rebuild, reinstall, repack or cache clearing is required.

All 668 VQ2A8 development tests pass on the existing NVIDIA/host system.
This includes the real candidate's compiler IR, numerical oracle, row
chunking and deterministic repeats, plus diagnostic extraction and
fail-closed supervisor regressions. Ruff check/format, Markdown lint and
`git diff --check` pass. Required `bash format.sh ci` was attempted but
remains blocked by missing local `pre-commit`. No A5 compilation or
runtime PASS is claimed for this revision before the user's rerun.

## Phase 4: reduce the gather kernel's UB workset

The user's `50ef51ee` run at `/tmp/vq2a8-phase4-w3qzncdy` passed
the previously failing memory-scope inference and stopped at
`PlanMemory (hivm-plan-memory)`. The compiler now reports a definite
on-chip UB capacity error: 2,170,880 bits (265 KiB) required versus
1,769,472 bits (216 KiB) available, a 49 KiB excess. This is not a
device-global-memory allocation failure. The excerpt does not identify
which lookup shape failed; the driver now prints each shape before its
first launch as `PHASE4 stage=lookup_start`.

Reduce the source-level workset without changing K reduction geometry:

- Retain one flat FP32 byte-carrier table instead of broadcasting it
  across output channels. Its maximum logical size is 1,024 FP32 values
  (4 KiB), down from the old `[32,1024]` (128 KiB). Flatten the indices
  for a one-dimensional `tl.gather`, then reshape its result for MAC.
- Process 16 output channels per program, with two programs sharing
  each original 32-channel codebook group. Index/weight tiles shrink
  from `[32,512]` to `[16,512]`. Packed pairs, output addresses and the
  codebook group are derived from the new program index.
- Keep full aligned 32-byte FP8 table rows, FP8 typed loads followed
  by byte bitcasts, the FP32 gather source, and 32-byte BF16 output
  stores. No scalar byte GM lookup, dense expert allocation, split-K
  reduction, atomics or new launch/compiler flags are introduced.

This follows the general workset/blocking approach in the official
[UB overflow guide](https://triton-ascend.readthedocs.io/en/latest/debug_guide/ub_overflow.html).
Neither logical tensor sizes nor CUDA compilation establish the actual
A5 buffer plan: temporaries and compiler-inserted buffers still count.
The number of programs doubles and table loads are repeated per half,
so performance must be measured rather than inferred from smaller UB use.
The accepted model backend and all phase-4 acceptance criteria remain
unchanged; no Ascend compilation or performance PASS is claimed yet.

Developer regressions inspect the actual candidate IR for a shared 1-D
FP32 source, bounded indices, retained `[16,512]` FP32 reductions and no
pointer reinterpret. The workset assertion fails against the old source.
Coverage includes 1/3/16/32 column tiles, both halves of single/multiple
codebook groups, poisoned outputs and canaries, K=4096, row chunking and
deterministic repeats. These run on the NVIDIA/host developer system,
not an Ascend simulator. The user's hardware gate remains required.

For the existing installation, pull the Python/Triton update and run
the same phase-4 command. No C++ rebuild, reinstall, repack, NPU reset
or cache clearing is needed. A failed child still stops the remaining
steps and retains its full log and diagnostic excerpt.

All 677 VQ2A8 development tests pass on the existing NVIDIA/host system,
including 71 phase-4 tests. The full CUDA `--micro-only` driver also
passes lookup (12 cases) and native micro at
`/tmp/vq2a8-phase4-f93m63k8`; this is explicitly `DEVICE=cuda:0`, not
950 acceptance or phase-4 completion. Ruff check/format, Markdown lint
and `git diff --check` pass. Required `bash format.sh ci` was attempted
but remains blocked by missing local `pre-commit`.

## Phase 4 redirected: VQ decode + native FP8 Cube prototype

The user explicitly selected this direction after the Vector-gather UB
failure. Do not spend the next hardware run on the old 16-step Vector
benchmark sequence. Keep its implementation and evidence as history and
comparison material, not as the native FP8 expert deliverable.

`vllm_ascend/quantization/vq2a8_fused_fp8.py` now contains a separate,
opt-in fused projection kernel. Its actual data path is:

```text
GM: packed int32 indices + typed E4M3 codebook + tile IDs
  -> unpack VQ codes -> shared flat byte-carrier gather
  -> E4M3 weight tile [32,128]
  -> FP8 dot with prepared E4M3 activation -> FP32 accumulator
  -> per-row activation scale + bias correction -> BF16 output
```

This is actual VQ decode feeding the dot, not the sign-flip CV microtest
renamed as a fused expert. It uses the frozen `vq2a8_direct_tp1_v1` layout;
the wrapper allocates only output, not a dense expert or a decode scratch
tensor. A8/RHT preparation, gate/up ordering, SwiGLU and the down-projection
boundary remain unchanged. This first prototype fuses **one projection's
decode and matmul**, not all expert preparation/activation/routing in one
launch. It is not registered in the MoE executor or serving path.

The Ascend tile is M=32, N=32, K=128, masking logical M=1/3/10/32 rows.
Packed-word row transfers are 64 bytes, FP8 K transfers 128 bytes,
codebook rows 32 bytes and output rows 64 bytes. The shared FP32 byte
carrier is at most 4 KiB and the decoded FP8 weight tile is 4 KiB.
These source sizes are **not** the final UB allocation: intermediate
lifetimes, compiler buffers and spills still need A5 codegen review.
The K=128 FP32 accumulation is checked numerically, not declared bit-exact
to the accepted K=512 Vector reduction. There is no FP32 Vector MAC
fallback in this prototype.

Ascend uses `tl.dot_scaled` with E4M3 operands and explicit byte-encoded
unit microscale tensors for the installed 3.2.2/A5 contract; the existing
row scale/bias is applied once after accumulation. The
[official dot_scaled reference](https://triton-ascend.readthedocs.io/en/latest/python-api/generated/triton.language.dot_scaled.html)
lists Ascend 950 FP8 support and a K multiple-of-64 restriction. That
documentation does not certify this CV pipeline on the user's installed
Triton-Ascend development build. Unsupported lowering must fail visibly.

### Isolated bring-up and evidence

Run the new supervisor, not `validate_vq2a8_tp1_phase4.py`:

```bash
python3 -u tools/validate_vq2a8_fused_fp8.py --physical-npu 4
```

It runs fresh child processes in this order and stops at the first failure:

1. `direct`: aligned FP8 dot, with the same internal K=128 tiling.
2. `bridge`: loaded FP8 weight bytes transformed on the Vector side before
   dot. This is only a CV diagnostic, not VQ fusion acceptance.
3. `fused`: real VQ unpack/lookup feeding dot; five shapes, four input cases,
   CPU same-FP8 oracle, accepted projection comparison, exact row chunking
   and three exact repeats. Warm timings are diagnostic, not a speedup gate.
4. Optional `expert`, selected by `--model`: one real gate_up/SwiGLU/down
   chain (default `--probe 0:0`), four row counts and four input cases.
   Both same-prepared-FP8 and independently prepared accepted chains are
   checked. Real dense reference weights stay on CPU only.

To include that real-expert step in the same isolated sequence:

```bash
python3 -u tools/validate_vq2a8_fused_fp8.py \
    --physical-npu 4 \
    --model /home/g00872988/DeepSeek-V4-Flash-VQ2A8-32x256
```

No C++ rebuild, reinstall, repack or Triton-cache deletion is needed for
these Python/Triton additions. The supervisor retains each child's log,
incremental JSON, source hashes, successful compiler IR/binary outputs and
compiler metadata under `FUSED_REPORT_DIR`. It does not retry, reset the
NPU, relax tolerances or switch to the accepted backend after failure.
The parent imports no accelerator runtime. Partial artifacts are rejected
on NPU; the explicit partial-artifact option is CUDA-development-only.

`FUSED_PROTOTYPE=PASS` means the requested isolated stages executed and
passed their checks. `native_instruction_verified` and
`on_chip_decode_verified` stay false pending Ascend compiler/binary and
memory-plan review. A source-level dot or even a successful launch alone
does not establish absence of FP16 fallback or GM spills. Full phase-4
completion additionally requires real-expert coverage, reviewed native
FP8/CV evidence, NPU timing/peak-memory assessment and opt-in model
integration/regression. Phase 5 is still deferred.

### Development checks versus Ascend acceptance

On the SM90 CUDA development host, M=32 source-level FP8 `tl.dot` was
observed to lower to FP16 MMA. This was caught by inspecting the actual
candidate PTX, not by a numerical test. The CUDA-only internal M tile is
therefore 64; Ascend remains 32. Unit tests and the CUDA probe now require
native E4M3 MMA in generated PTX and reject silent FP16 lowering. CUDA
codegen evidence is explicitly separate from Ascend instruction review.
The CUDA dot also bounds reduced-precision WGMMA partial accumulation to
32 elements before FP32 addition; the original unlimited partial sum
failed the same-prepared real-weight oracle. This CUDA-only setting does
not change the Ascend dot API or relax any numerical gate.

The test suite covers metadata/alignment rejection, CPU-only dense oracles,
different rows and table counts, multiple N groups and K blocks, finite
FP8 extremes/subnormals, zero/impulse/small cases, poisoned outputs and
canaries, row chunking, repeats, generated IR/PTX and fail-closed subprocess
evidence handling. Default model/Vector implementations remain unchanged.

Development result: **734 VQ2A8 tests passed**, including 57 new prototype
tests, on the NVIDIA/host development system. This is not an NPU test run.
The final synthetic-only supervisor passes all three stages at
`/tmp/vq2a8-fused-fp8-hifwkyd_`: 4 direct controls, 4 CV controls and 20
fused projection cases, with native E4M3 WGMMA found in CUDA PTX.
Ruff check/format, Markdown lint and `git diff --check` pass. Required
`bash format.sh ci` was attempted but is blocked by missing `pre-commit`.
The real expert `0:0`, M=32 deterministic chain has an explicit **FAIL** at
`/tmp/vq2a8-fused-fp8-hd9eiq2w`; its gates were not loosened:

- Gate/up same-FP8 oracle: PASS, maximum absolute error 0.03125,
  relative L2 error about 0.0006692; row chunks and repeats are exact.
- Down on the candidate's prepared input: PASS, maximum absolute error
  0.03125, relative L2 error about 0.0007150; exact chunks and repeats.
- Independently prepared accepted-versus-candidate full chain: FAIL,
  7,414/131,072 elements outside `rtol=0.03, atol=0.05`, maximum absolute
  error 0.125, relative L2 about 0.024994. Aggregate L2 alone is not PASS.

The diagnostic replay `expert-diagnostic2.json` preserves the passing
projection evidence plus `passed=false, failure_stage=chain_baseline` for
the failing down chain. Its preparation comparison has 2,226 differing
activation bytes, maximum scale difference about 1.31e-6 and maximum bias
difference about 0.01401. Inputs, prepared A8/scales/biases and outputs are
retained in the adjacent `*-chain-failure.safetensors` (about 898 KiB).
This localizes a downstream preparation difference; it does not by itself
prove the complete numerical cause or predict the Ascend result. Do not
promote this prototype into the model while that chain gate is failing.

Next hardware action is the **synthetic-only** supervisor command above:
establish Ascend direct/CV/VQ-dot execution and inspect its generated
memory plan and FP8 instructions. Real-chain numerical convergence,
broader expert coverage and opt-in model integration remain required work,
not postponed checks that may be marked PASS. Phase 2 stays skipped.

### Ascend direct control: explicit unit-scale frontend fix

The user's first fused-prototype run at `5b9a897b` failed in `direct`,
before any Cube execution, at
`/tmp/vq2a8-fused-fp8-s8zu7t1c`. Triton-Ascend
`3.2.2+dev20260729205041` rejected `lhs_scale=None` with
`lhs_scale must be int8 or uint8 tensor`. The original assumption that
this frontend accepts omitted scales was incorrect. This is not a C++
extension, packed-artifact, device-memory-capacity or numerical-gate error.

`vq2a8_fp8_cube.py` supplies both sides explicitly for all three Ascend
callers: fused VQ projection, matching-geometry direct/bridge control and
the older optional Cube microtest. The constants are produced inside the
kernel, with no extra wrapper allocation, CPU transfer or scale GM input.
For the prototype's K=128 dot tile, both scale tensors are uint8 `[32,8]`;
the older K=512 control uses `[32,32]`. The RHS scale remains N-major even
though its matrix operand is transposed to `[K,N]`.

The encoding and version-specific shape are deliberately separated:

- The [official A5 FP8 test](https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/third_party/ascend/unittest/pytest_ut/test_dot_scaled_fp4_fp8.py)
  decodes scale bytes as `2**(byte-127)`, so byte `127` is one. This is not
  the signed-exponent convention used by its older BF16 tests.
- The [Ascend API restriction](https://triton-ascend.readthedocs.io/en/latest/python-api/generated/triton.language.dot_scaled.html)
  specifies FP8 scale shapes `[M,K/16]` and `[N,K/16]` on 950. This patch
  targets the user's 3.2.2/A5 build. Newer main-branch FP8 tests use K/32;
  their layout must not be substituted without revalidating the compiler.
- The [3.2.2 frontend](https://github.com/triton-lang/triton-ascend/blob/2deb5df0254e23ec750443f175340b38a196097e/python/triton/language/semantic.py)
  requires a byte tensor for the LHS. Both scales are now explicit, without
  relying on the backend's implicit RHS handling.

These are identity controls only: packed bytes, FP8 activation and weight
values, the FP32 accumulator, A8 row-scale/bias epilogue and tolerances
remain unchanged. CUDA still uses its existing native `tl.dot` branch.
The new `FUSED_DOT_SCALE_CONTRACT` record and helper source hash are saved
even when compilation fails. Direct-control compiler artifacts are now
retained before numerical comparison, including on a subsequent oracle
failure. They are still not marked as reviewed native/CV evidence.

Host regressions execute the real helper and each actual Ascend call
branch against a strict byte-scale recorder, checking dtype, shapes,
RHS orientation, E8M0 identity and nonzero chained accumulation. They
explicitly do not execute an Ascend compiler or emulate Cube arithmetic.
An NPU rerun of the synthetic-only supervisor is still required; no
Ascend PASS, real-chain convergence or phase-4 completion is claimed.

This fix passes **746 VQ2A8 development tests** (17.67 seconds) on the
NVIDIA/host development system, including 11 new scale-contract tests and
a regression for codegen retention on direct-oracle failure. The CUDA
synthetic supervisor also passes 4 direct, 4 bridge and 20 fused cases at
`/tmp/vq2a8-fused-fp8-xknkh9yd`, with native E4M3 PTX checks intact. The
previous real-expert chain FAIL remains unresolved and is not overwritten
by these synthetic checks. Ruff check/format, Markdown lint and
`git diff --check` pass; required `bash format.sh ci` was attempted but
remains blocked by missing local `pre-commit`.

Pull this Python/Triton update and use the same synthetic-only command
above. Do not rebuild C++, repack weights, clear caches or run the full
model to diagnose this frontend failure.

### Ascend direct control: separate compile-time assertions

The next user run at `e6c95df9`, report
`/tmp/vq2a8-fused-fp8-i1sb2fr3`, fails in the helper's combined dtype
assertion with `'bool' object has no attribute 'logical_and'`. It has not
reached `dot_scaled`; the scale layout and Cube execution are still
unverified on the user's NPU. Printing the scale contract is not a PASS.

The [3.2.2 AST frontend](https://github.com/triton-lang/triton-ascend/blob/2deb5df0254e23ec750443f175340b38a196097e/python/triton/compiler/code_generator.py)
lowers `and` through `logical_and`, while dtype comparisons can produce
ordinary Python booleans. Each static assertion now contains just one
comparison. All four conditions are retained: LHS E4M3, RHS E4M3, matching
K dimensions and K divisible by 64. The scale bytes, shapes, accumulation,
epilogue, numerical gates and default model backend are unchanged.

The earlier host tests called `.fn` under ordinary Python and therefore
missed this AST-lowering incompatibility. The new regression inspects the
actual JIT source for separate guards without Boolean composition. Negative
cases independently check both operand dtypes, unequal K and unaligned K,
requiring rejection before `dot_scaled`. These tests do not emulate or
execute the Ascend compiler.

Validation: **750 VQ2A8 development tests pass** in 18.44 seconds on the
NVIDIA/host development system; the focused helper/fused set passes 73
tests. Ruff check/format, Markdown lint and `git diff --check` pass.
Required `bash format.sh ci` was attempted and remains blocked by missing
local `pre-commit`. No new NPU PASS or real-expert chain PASS is claimed.
Pull and rerun the same synthetic-only supervisor; no C++ rebuild,
reinstallation, repacking or cache clearing is required for this change.

### Ascend direct control: explicit graph synchronization solver

At `d859a86f`, the user's direct control reaches BiShengIR but aborts in
`SyncSolverIRTranslator.cpp` with
`coreType.value() != hivm::TCoreType::CUBE_OR_VECTOR`. The failed JIT run is
`/tmp/vq2a8-fused-fp8-1r9qtv06`; this is a compiler assertion, not a device
execution or numerical result.

The user replayed its cached `_cube_bridge_kernel.ttadapter` on the same
Ascend950PR_957d compiler, initially with both graph and cross-core solver
switches explicitly enabled. A subsequent graph-only replay, leaving the
cross-core switch unspecified, reports `GRAPH_ONLY_COMPILE_EXIT=0` at
`/tmp/vq2a8-graph-only.vACVZF`. It produces `kernel.o` (18,208 bytes),
`kernel_mix_aic.o` (23,808 bytes) and `kernel_mix_aiv.o` (27,768 bytes).
Its printed IR contains FP8 operands to `hivm.hir.mmadmxL1` and BF16 output
through `fixpipe`. This is direct-control compilation evidence only:
the replay does not launch a kernel, check numerical output or exercise
VQ decode. It does not identify which operation caused the earlier
ambiguous-core assertion or certify final native instructions.

Both prototype launchers now request `sync_solver=True` on NPU:
`launch_cube_control` (direct and bridge) and `launch_fused_fp8`. The
[3.2.2 A5 backend](https://github.com/triton-lang/triton-ascend/blob/2deb5df0254e23ec750443f175340b38a196097e/third_party/ascend/backend/compiler.py#L476-L479),
matching the user's printed installed code, maps this option to
`--enable-hivm-graph-sync-solver=True` only. The A2/A3 path's additional
cross-core flag must not be assumed for A5. No synchronization is disabled,
and no system compiler, global runtime setting, older Vector kernel or
default model backend is patched. CUDA keeps its previous launch options.
K/M/N geometry, unit-scale bytes, accumulation and numerical gates are
unchanged. Bridge/fused applicability still requires the NPU rerun.

The shared `fused_fp8_launch_options` helper is used by both launchers and
the child report. `FUSED_LAUNCH_OPTIONS` and JSON `requested_launch_options`
are emitted/saved before device initialization, including on failure.
These fields describe requests, not proof that a compiler honored them;
actual codegen metadata remains separately retained and unreviewed.
Host regressions record the real wrappers' keyword forwarding for NPU and
CUDA, verify options are not shared mutable state, and preserve failure
reporting without promoting execution/native/on-chip/model flags.

Validation: **764 VQ2A8 development tests pass** in 15.97 seconds on the
NVIDIA/host system, including 14 new option/report regressions. The focused
helper/fused set passes 87 tests, including existing actual CUDA kernels.
Ruff check/format, Markdown lint and `git diff --check` pass. Required
`bash format.sh ci` was attempted but remains blocked by missing local
`pre-commit`. There is no new NPU numerical PASS from the development host.

Next run the synthetic-only supervisor (no model argument):

```bash
cd /home/g00872988/vllm-ascend-vq2a8
/usr/local/python3.11.10/bin/python3 -u tools/validate_vq2a8_fused_fp8.py --physical-npu 4
```

It must pass 4 direct, 4 bridge and 20 fused cases, including unchanged
oracles and repeatability checks. Compilation alone is not a prototype
PASS. Real-expert chain convergence, on-chip/native instruction review,
model integration, performance, quality and serving remain unverified.
The earlier real-expert chain FAIL is not superseded by this compiler
replay. No C++ rebuild, editable reinstall, cache deletion or weight
repacking is required for this Python launch-option change.

## Native AscendC fused projection: implementation, not yet NPU accepted

The user explicitly requested a direct AscendC implementation after the
Triton prototype continued to fail in the A5 GraphSyncSolver. The six-way
compiler replay (`/tmp/vq2a8-compile-ab-aml5knku`) failed in every case.
The earlier one-off manual compile is not sufficient evidence of a usable
Triton fix. Pause that investigation and the old phase-4 Vector sweep.

The new implementation is in `csrc/vq2a8_ascendc/kernel.cpp`, with native
C++ Torch registration in `csrc/vq2a8_ascendc/torch_binding.cpp`. The Python
module `vq2a8_ascendc.py` only explicitly loads the built library and calls
its registered operations. It does not invoke Triton, compile on demand,
dequantize a dense expert to HBM or select a fallback. The independent
CMake target avoids enabling unrelated kernels excluded by the root A5
build. Existing package installation and model dispatch are unchanged.

### Numerical and memory design

This is a **single expert projection primitive**, not a whole fused MoE
block. Gate/up and down can each use it; routing, RHT/A8 preparation and
SwiGLU remain outside this first device kernel. It consumes the existing
frozen packed artifact without a new weight format:

- Activation: contiguous E4M3FN `[M,K]`, with FP32 row scale and bias `[M]`.
- Packed indices: int32 `[N/2,K/8]`, eight unsigned 4-bit codes per word.
- Codebooks: E4M3FN `[tiles,N/32,16,2]`; tile IDs: uint8 `[K]`.
- Output: BF16 `[M,N]`, computed as `FP32(A @ decoded_W.T) * scale + bias`.
- Scope: `1 <= M <= 32`, `N % 32 == 0`, `K % 512 == 0`, positive
  `N,K <= 65536`, and `1 <= tiles <= 256`. C++ checks metadata even if the
  Python wrapper is bypassed. Malformed tile IDs cannot index outside the
  local table: the decoder emits an FP8 NaN byte instead.

Each core group processes 32 output channels, K tiles of 128, and a padded
32-row activation tile. AIV0/AIV1 each own 16 rows of A and 16 channels of
B. Activations are padded with zero in UB without reading nonexistent GM
rows. B shares one nibble between each adjacent output pair, selecting the
two distinct codebook bytes; no numeric FP8-to-byte conversion is used.

```text
packed words + codebooks + tile IDs (GM)
  -> bounded UB decode on AIV0/AIV1
  -> shared L1 [K/32,32,32] -> FP8 L0B
activation (GM) -> zero-padded UB -> shared L1 -> FP8 L0A
  -> native Mmad -> FP32 L0C accumulation across K
  -> Fixpipe to two UBs -> FP32 row scale, then bias -> BF16 output (GM)
```

The first decoder and activation layout transform use **scalar UB
accesses**, not a tuned vector gather. This is deliberate bring-up scope;
there is no performance claim. Explicit buffer requests are 18,176 UB
bytes per AIV, 8,192 L1 bytes per core group, and 4,096 bytes each for L0A,
L0B and L0C. These are source-level allocations, not measured compiler
resource usage. Only the small tile is decoded; full expert weights remain
packed in device memory. No workspace argument is present in the launch ABI.

The implementation uses A5 mode-4 cross-core flags, following the existing
arch35 attention kernels. Both AIVs participate even when M <= 16:

| Flag | Producer -> consumer | Reuse condition |
| --- | --- | --- |
| 0 / 16 | AIV MTE3 -> AIC MTE1 | Both A/B halves have reached L1 |
| 1 / 17 | AIC MTE1 -> AIV MTE3 | Cube has copied L1 data into L0 |
| 2 / 18 | AIC FIX -> AIV V | FP32 result halves have reached UB |
| 3 / 19 | AIV MTE3 -> AIC FIX | Epilogue/output transfer has finished |

Single buffering and local pipeline fences avoid reusing UB/L1/L0 while
their previous consumers still need them. Multibuffering/overlap is not
enabled. A compile-time guard rejects CANN's 1:1 TSCM GM compatibility
route. The runtime queries the AIC count instead of assuming 16, 28 or 32
cores, and checks an Ascend950 device with a 1C:2V topology.

The FP8 operation is **unscaled `AscendC::Mmad`**, not the Triton
`dot_scaled` helper or a hidden FP16 GEMM. The official
[Mmad API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/ascendcopapi/atlasascendc_api_07_0249.html)
documents E4M3FN x E4M3FN -> FP32. Official `CANN/asc-devkit` source at
`0290f560c82a867528b8ecbdb1a20366a6760a64`,
`impl/basic_api/dav_3510/kernel_operator_mm_impl.h`, also distinguishes
plain FP8 `mad` from MX `mad_mx`. The same source audit checks the
LoadData2DParamsV2 fields, native UB->L1 copy and mode-4 synchronization.
The installed CANN 9.1 compiler remains the authoritative compatibility
check; the header audit is not device compilation or binary inspection.

### Build and first hardware run

Run on the user's Ascend950 machine. These are separate commands; stop if
the build fails. No package-wide editable reinstall, Triton cache removal,
compiler patch, OPP reinstall or model weight repack is required.

```bash
cd /home/g00872988/vllm-ascend-vq2a8
/usr/local/python3.11.10/bin/python3 -u tools/build_vq2a8_ascendc.py
/usr/local/python3.11.10/bin/python3 -u tools/validate_vq2a8_ascendc.py --physical-npu 4
```

Build options include `--cann`, `--soc`, `--jobs` and `--build-dir`.
Defaults are CANN 9.1.0 (or existing `ASCEND_HOME_PATH`),
`Ascend950PR_957d`, four build jobs and `build/vq2a8-ascendc`.
Build logs/manifests stay in that directory. Validation verifies the exact
`.so` hash and native source hashes against its successful build manifest;
stale libraries are rejected before NPU execution. For a different build
directory pass `--library /absolute/path/libvq2a8_ascendc.so` to validation.

The supervisor runs isolated children and prints progress/errors:

1. Six direct FP8 controls, beginning with M=32 before testing padding.
2. Six sign-bit-flip controls exercising Vector-produced FP8 into Cube.
3. Twenty-eight packed synthetic cases: CPU FP64 same-FP8 oracle,
   accepted Vector baseline, three bitwise repeats and exact row chunking.
4. Optional: 48 real expert gate_up/down cases with an independently
   prepared accepted chain, retaining the existing zero-exact and
   `rtol=0.03, atol=0.05` chain gates. The normalized error gate also stays.

Only after the synthetic stages pass, opt into the real expert:

```bash
/usr/local/python3.11.10/bin/python3 -u tools/validate_vq2a8_ascendc.py --physical-npu 4 --model /home/g00872988/DeepSeek-V4-Flash-VQ2A8-32x256 --probe 0:0
```

Any failure/timeout/missing or incomplete evidence stops later stages.
The accepted Triton/Vector kernel is used only as an independent comparison
in the validation harness, never as an AscendC implementation or fallback.
Reports and any chain-failure tensors remain on the machine; send printed
`ASCENDC_BUILD`, `ASCENDC_START`, `ASCENDC_RESULT` and error lines rather
than an archive. A successful build alone never prints a numerical PASS.

### Validation boundary

Development-host tests: **18 new tests pass**, including a real g++ C++17
host executable that exhaustively checks the shared packing/NZ indexing,
two-AIV L1 assembly, partial-M zero padding and core-group coverage.
The full existing VQ2A8 suite plus the new tests passes **782 tests** on the
NVIDIA/host development machine. These are not AscendC device-kernel tests.

Changed-file Ruff, clang-format, Markdown and spelling checks pass. The
required `bash format.sh ci` was attempted but could not start because
`pre-commit` is not installed in the local development shell.

No CANN installation or NPU is accessible on that development host. The
native source and build/validation tools are implemented, but **device
compilation, NPU numerical execution, generated FP8 instruction review,
on-chip-only transfer review and performance are unverified**. No prior
Triton/CUDA PASS is promoted to an AscendC PASS. Even after standalone
numerical success, instruction and on-chip flags stay false until the
generated native binary/dataflow is reviewed. Model integration, full-model
reference logits, quality, serving, and phase 5 remain separate acceptance.

### First AscendC build: scalar row clipping fix

The first CANN 9.1 build at `4b915644` completed CMake configuration and
source precompilation, then failed while compiling the AIC/AIV preprocessing
objects at `kernel.cpp:162`. The blocking error was the prototype's scalar
`Min(m_ - firstRow, kHalf)` call: the visible `AscendC::Min` overloads operate
on `LocalTensor` values, not two scalar integers. This is a source bug, not
the earlier Triton GraphSyncSolver problem. The nonfatal Kineto warning is
not the failing step.

Row clipping now uses a shared constexpr `HalfRows` helper containing only
scalar comparisons, guarded subtraction and a conditional expression. It
preserves zero rows for AIV1 when M <= 16, avoiding unsigned underflow. The
host C++ test checks all 1,089 combinations of M and starting row from 0
through 32, plus compile-time boundary assertions and activation-padding
checks using the same helper. No layouts, synchronization, FP8 arithmetic,
validation tolerances or model dispatch are changed.

The updated development-host VQ2A8 suite passes **783 tests** (including
19 AscendC host/harness tests). Changed-file lint checks pass; the unified
format script is still unavailable because local `pre-commit` is missing.

Rebuild with `tools/build_vq2a8_ascendc.py` after pulling this change; there
is no need to delete the build directory or reinstall the package. Run the
standalone validation only after `ASCENDC_BUILD=PASS`. Passing this source
regression test does not establish successful CANN compilation or NPU
execution; both remain awaiting the next hardware-machine build/run.

### AscendC hardware progress and one-command batch validation

The subsequent user-machine build reports `ASCENDC_BUILD=PASS` and produces
`build/vq2a8-ascendc/libvq2a8_ascendc.so`. The user supplied standalone
synthetic PASS evidence in `/tmp/vq2a8-ascendc-g1ewqxdd`, followed by real
selected-expert gate_up/down-chain PASS evidence in
`/tmp/vq2a8-ascendc-7ldsczv6`. Those reports include oracle/baseline checks,
bitwise repetition and row-chunk invariance. This establishes progress beyond
the earlier build failure, but not all-expert coverage or model integration.
Native instruction, on-chip decode and performance flags remain false.

To avoid another manual run/paste cycle for each probe, the batch supervisor
now schedules the remaining standalone coverage in one command:

```bash
cd /home/g00872988/vllm-ascend-vq2a8
git pull --ff-only
/usr/local/python3.11.10/bin/python3 -u tools/validate_vq2a8_ascendc_suite.py --physical-npu 4 --model /home/g00872988/DeepSeek-V4-Flash-VQ2A8-32x256
```

This update changes validation tools only: **reuse the successfully built
native library; no C++ rebuild is needed**. Library and native-source hashes
are still checked against the successful build manifest. No Triton backend
replacement, model dispatch change, compiler patch, cache removal or repack
is performed.

The default plan for the 43-layer, 3-hash-layer, 256-routed-expert checkpoint:

| Coverage | Work scheduled |
| --- | --- |
| Shared controls | Direct FP8, Vector-to-Cube sign-bit bridge and synthetic packed gates once |
| Additional boundaries | M=2/15/31, 29 output groups exceeding 28 AICs, N=65536 and K=65536 |
| Real experts | One stored expert per layer; first/middle/last routed layers also sample IDs 0/127/135/255 |
| Numerical total | 54 distinct probes, 64 shared cases plus 2592 expert projection cases: 2656 cases |
| Timing samples | First/middle/last selected probes, gate_up and down at M=1/17/32: 18 measurements |
| Native binary evidence | Bounded native-target ELF object discovery, hashes and CANN llvm-objdump output |

There are 61 isolated child processes. All 43 layers are sampled, **not all
10,243 stored experts**; this is not a 43-layer model forward. Expert cases
retain deterministic/zero/impulse/small inputs, same-FP8 CPU oracle,
accepted projection baseline, independent accepted chain, three exact
repeats and exact row-chunk checks. Existing tolerances are unchanged.
Boundary shapes are tested independently, not as a maximum-N by maximum-K
allocation. Optional `--probes 0:0,3:135,42:255` selects a smaller explicit
subset and labels its limited layer coverage in the report.

Correctness children retain launch blocking. Timing uses separate children
with `ASCEND_LAUNCH_BLOCKING=0`, at least 3 warmups and 10 repeats. It records
synchronized wall and device-event min/median/p95 for identically prepared,
resident candidate and accepted-baseline projection inputs. First projection
call time and Torch allocator peak delta are also recorded; the latter does
not include all CANN internal allocations. Preparation, expert loading,
routing and the full model are outside warm projection timings. Observed
wall-time ratios are measurements for review, not an accepted inference
speedup. `--warmups` and `--repeats` can increase the samples.

Only confirmed numerical assertion failures in an expert/timing child allow
later probes to continue. A shared-control failure, device/runtime error,
timeout or missing/incomplete evidence stops subsequent NPU work. Failed
expert probes are not timed; skipped work remains visible. Different probe
failures have distinct tensor artifact filenames. Per-child timeout defaults
to 3600 seconds and is configurable with `--timeout`; the suite has no
promised wall-clock duration.

The supervisor prints its report directory immediately, updates `summary.txt`
and `summary.json` after each child, and prints the compact summary at the end.
Send that summary, not an archive or thousands of result lines. It includes
per-stage failures and representative timings. The `binary/` subdirectory
contains native object metadata, disassembly and short instruction excerpts.
Missing disassembly tools or failed object inspection are explicitly recorded
as incomplete evidence and do not suppress numerical testing.

Instruction name matches alone never certify FP8 execution or on-chip decode:
review operand types, UB/L1/L0 transfers and object-to-loaded-library linkage.
Successful numerical/timing completion is labelled `COMPLETED_REVIEW_PENDING`,
not full acceptance. Native instruction, on-chip decode, performance, model
integration, quality and serving flags stay false pending their own evidence.
Full-model testing with this backend requires integration work first; rerunning
the old backend's phase 3 would not validate this AscendC implementation.

The new host tests exercise probe planning, complete receipts, timing validation,
failure continuation/abort rules and bounded binary collection with mocked
device subprocesses. They do not substitute for the user-machine batch run.
The updated VQ2A8 development-host suite passes **819 tests**, including 36
new batch-supervisor tests. Changed-file Ruff, Markdown and spelling checks
pass. The required `bash format.sh ci` was attempted but still cannot start
without local `pre-commit`. No new NPU execution is claimed for this harness
update; run the command above on the hardware machine.

### Hardware batch result and public objdump limitation

The user supplied `/tmp/vq2a8-ascendc-suite-qltvpdfb`: all 11 scheduled
children passed for probes `0:0,3:0,23:127,42:0`. This covers 256 numerical
cases and 18 resident-projection timing records, not every layer/expert.
For probe `0:0`, gate_up/down wall medians are 15.325/7.714 ms at M=1,
versus 39.329/9.973 ms for the accepted baseline. At M=32 they are
15.333/7.711 ms versus 1252.946/313.861 ms. The other timed probes have
similar measurements. The large batched ratios are against the existing
per-row projection baseline and are not end-to-end model speedups.

The library hash in the supplied binary report is
`c8e7dfee8676dd52c5ba1882ffd7a265b53febaa7fa483b1753dea878e446677`.
Both AIC and AIV object dumps identify `elf64-hiipu`, including fused
entry symbols, but their address lines have no instruction bodies. This is
consistent with the official CANN 9.1.0-beta.3
[BiSheng companion-tool documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/compiler/BishengCompiler/atlas_bisheng_10_0009.html):
public `llvm-objdump` exposes function names and offsets for diagnostics,
not assembly instructions. Empty bodies therefore do not prove missing
FP8 computation, a compiler failure, or an incorrect selected architecture.
The earlier instruction-text collection assumption was wrong.

The old collector incorrectly counted address-only lines as instructions:
the supplied 2494/3324 counts must be read as offsets, not decoded opcodes.
The corrected collector reports `address_lines`, `empty_address_lines`,
`undecoded_address_lines` and actual mnemonic-text `instruction_lines`
separately. Offset-only output becomes `symbols_and_offsets_only` and the
binary summary becomes `incomplete_review_pending`; partial/unknown opcode
output is also not a complete instruction-text collection. Symbol names
containing FP8/MMAD no longer count as instruction hits. None of these
text classifications automatically certifies an ISA, FP8 execution or
on-chip-only dataflow. Existing reports are not overwritten and their
numerical/timing evidence remains valid within the original scope.

Do not rerun the numerical campaign, clear caches, change `--mcpu`, or
recompile merely to get different output from this documented tool. The
next supported evidence route to evaluate is the official
[msOpProf simulator instruction timeline](https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_simulator_user_guide.md).
It can report per-core instruction/timeline data and memory-transfer
information. The installed profiler/simulator version must be checked
before constructing an invocation. Its current documentation also requires
an Ascend950 simulator `lib/config.json` logging setting of `flush_level=2`;
do not silently modify the user's global toolkit configuration. Simulation
timings must not replace the already collected hardware timings, and
simulated artifacts must be bound to the tested binary before claiming
anything about that build. Instruction types and decoded-weight transfers
still require inspection; merely seeing Cube activity is insufficient.

The source path remains an FP8 `AscendC::Mmad` and bounded UB-to-L1 handoff,
but source inspection is not machine-instruction verification. Native
instruction/on-chip flags and default model backend are unchanged. Model
integration is not advanced past the requested verification gate while
the required instruction/dataflow evidence is unavailable.

The collector correction adds nine regression cases; the development-host
VQ2A8 suite passes **828 tests**. Changed-file Ruff, Markdown and spelling
checks pass. The required `bash format.sh ci` remains blocked by missing
local `pre-commit`. No C++ kernel or model dispatch changes are included;
this correction requires neither a library rebuild nor a numerical rerun.

### Single-call simulator evidence collection

The user confirmed that `msprof op simulator --help` exposes the required
application/kernel/launch-count/timeout options. The installed
`Ascend950PR_957d` simulator resolves to `dav_3510`, and its global
`lib/config.json` has `flush_level=3`. Do not edit this global file.
The official profiler's
[CreateCamodelConfig implementation](https://github.com/Ascend/msopprof/blob/master/csrc/op_profiling/profiling/op_prof_task.cpp)
uses a private config directory via `CAMODEL_CONFIG_PATH`; its 950 branch
also changes `flush_level` from 3 to 2 there. The installed binary may differ
from that source, so the application verifies its effective config before
importing Torch or initializing a device.

Run the new minimal capture entry, using the existing successful suite:

```bash
cd /home/g00872988/vllm-ascend-vq2a8
git pull --ff-only
/usr/local/python3.11.10/bin/python3 -u tools/profile_vq2a8_ascendc.py --suite-report /tmp/vq2a8-ascendc-suite-qltvpdfb
```

This does not rebuild the library or rerun the standalone campaign. It:

- Checks the library against its build manifest and the supplied suite hash.
- Copies only the simulator config into a fresh private report directory,
  sets its `flush_level=2`, and verifies the global config hash after running.
- Launches `msprof op simulator` with the fused kernel prefix and
  `--launch-count=1`; default metrics retain synchronization-event details.
- Requires a mapped `libruntime_camodel.so` before importing Torch, refusing
  a bare-Python/hardware fallback. Device 0 here is the simulator's logical
  device, not a request to run on physical NPU 0.
- Invokes one synthetic M=32, N=32, K=512, tiles=3 projection (one AIC block,
  two AIVs, four K tiles). Both decode halves and multiple codebooks execute.
  There are no candidate warmups, repeats, dense oracles or model loads.
- Uses a five-minute simulation limit plus two minutes for parsing. The
  outer deadline terminates only the newly owned profiler process group.
- Prints bounded instruction CSV samples, including operand/transfer details,
  and an output-file inventory. Full files remain in the printed report path.

The application records the actual config path/hash, simulator runtime paths,
library hash and completion receipt. Existing numerical/timing reports are
left untouched. Simulator CSV rows may aggregate multiple instruction calls;
the summary counts rows, not dynamic instruction executions. Source maps and
hotspot information may be absent because this uses the original release
binary without adding `-g` or recompiling.

Successful collection remains `collected_review_pending`, not native or
model acceptance. Missing AIC/AIV CSVs, unsupported columns, profiler failure,
timeout or an incomplete application receipt remain incomplete. Review FP8
operand types and the decoded-weight UB/L1/L0 transfers against the tested
binary; neither CSV filenames nor Cube activity alone certify them. This small
simulation does not establish all-shape dataflow, hardware timing or complete
embedded-object linkage. No model backend or verification flag is promoted.

The development-host VQ2A8 suite passes **861 tests**, including 33 new
profiler-harness tests. Ruff, Markdown and spelling checks pass for changed
files. `bash format.sh ci` was attempted but cannot start without the local
`pre-commit` dependency. The actual CANN simulator capture remains unverified
until the user runs the command above; no new hardware result is claimed.

### Partial simulator evidence and TPipe event ownership fix

The user supplied `/tmp/vq2a8-ascendc-sim-h98hzsc6` for the same
`c8e7dfee...46677` library. The application and simulator both confirm the
private config with `flush_level=2` and the mapped `dav_3510` simulator
runtime. This resolves the earlier config-path uncertainty without changing
the global toolkit. The selected kernel is `vq2a8_ascendc_fused_2_mix_aic`.

The partial trace contains concrete native-instruction and transfer evidence:

- AIC `MMAD`, pipe `CUBE`, detail `dtype:E4M3E4M3`, address `0x10d0f480`,
  `call_count=1`: native E4M3-by-E4M3 matrix execution is observed in simulation.
  This is not the scalar integer `MADD` also present in the trace.
- Both AIVs execute `MOV_UB_TO_L1`, reading UB offset `0x1800` into L1
  offsets `0x1000` and `0x1200`. These match the source's decoded B buffer
  and the two halves of its B L1 tile. AIC `LOAD_2Dv2` reads that L1 tile
  into L0B as bytes before the FP8 MMAD.
- These observations support the first decoded K-tile handoff, not completion
  of all four K tiles, all-shape on-chip-only operation, or a new hardware
  performance result. Scalar byte loads/stores dominate the captured AIV
  samples; optimization still needs completed traces and hardware measurement.

The application never records completion. The simulator reports four
`execute_set_flag already has same set_flag` errors at `0x10d0f488`, then
hits its five-minute timeout. That PC exactly matches the post-MMAD
`SET_FLAG`, `PIPE:CUBE,TRIGGERPIPE:MTE1,FLAGID:0` in the AIC CSV.
The reported 92.96/108.49 microsecond core durations are from an incomplete
simulation, not a completed projection and not the previously measured
hardware wall times. The profiler's subsequent `All task success` describes
saved-data parsing and does not make this kernel execution successful.

The source's immediate `Fence<E>()` had hard-coded event ID 0. The official
[A5 TPipe implementation](https://gitcode.com/cann/asc-devkit/blob/0290f560c82a867528b8ecbdb1a20366a6760a64/impl/basic_api/dav_3510/kernel_tpipe_impl_c310.h)
allocates and pre-sets `M_MTE1` IDs 0, 1 and 2 during initialization, and
waits/releases them at teardown. This identifies a framework-event ownership
collision in our helper, consistent with the exact failing PC. Official
[SetFlag guidance](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0270.html)
warns against manually specified IDs and recommends obtaining them from
`TPipe` to avoid framework synchronization conflicts and hangs.

The helper now uses `GetTPipePtr()->FetchEventID(E)` for each immediately
paired SetFlag/WaitFlag. Fetch queries a free event without reserving it;
the immediate pair finishes before reuse. It does not consume TPipe's
initialization/teardown tokens. Cross-core Ready/Read/Result/Stored flag
numbers belong to a different mechanism and remain unchanged. This patch
does not disable synchronization or simulator checks, change FP8 types,
introduce an FP16 fallback, or alter model dispatch.

The profiler now records bounded runtime errors and inner application
timeouts even if the profiler exits zero after parsing. Scalar `MADD` is
no longer selected as matrix-instruction evidence. A host C++ regression
executes the actual Fence body extracted from the kernel against a model
of occupied/pre-set events, detects the old duplicate-set behavior, and
checks that repeated K-tile fences preserve framework teardown tokens.
This host model is not a CANN compilation or NPU execution test.

Rebuild into a separate directory to preserve the old tested library, then
capture only one small fused call from the changed candidate:

```bash
cd /home/g00872988/vllm-ascend-vq2a8
git pull --ff-only
/usr/local/python3.11.10/bin/python3 -u tools/build_vq2a8_ascendc.py --build-dir build/vq2a8-ascendc-syncfix
/usr/local/python3.11.10/bin/python3 -u tools/profile_vq2a8_ascendc.py --diagnostic-build --library build/vq2a8-ascendc-syncfix/libvq2a8_ascendc.so
```

Run the capture only after the new build reports PASS. `--diagnostic-build`
is explicitly separate from `--suite-report`: the changed library cannot
inherit numerical or timing acceptance from the old hash. The build manifest,
native source hashes and parent/child library hashes are still checked.
The report marks the missing matching standalone suite; all acceptance flags
remain false. Do not rerun the broad numerical campaign just to diagnose this
event conflict. After complete, error-free simulation, validate numerics and
hardware timings for the new library before proceeding to model integration.
The old standalone receipts remain historical evidence for the old binary.

The development-host VQ2A8 suite passes **867 tests** (six added regression
cases). This includes the extracted C++ Fence event-ownership test, not device
execution. Changed-file Ruff, Markdown and spelling checks pass; the required
`bash format.sh ci` was attempted but still cannot start without local
`pre-commit`. The corrected kernel still needs a CANN build and the bounded
simulator rerun on the user's machine; no new hardware PASS is claimed.

### Sync-fix build succeeds; the five-minute trace is still incomplete

The user built `build/vq2a8-ascendc-syncfix/libvq2a8_ascendc.so` successfully,
with SHA256 `022bbb8ccd39e0de0a193199027ba2e935e87967c8d49f1a9f251ddcdf6b1518`.
The capture `/tmp/vq2a8-ascendc-sim-igxi9yba` loads this exact new library
and the same isolated simulator config. No duplicate SetFlag is reported,
but this is not yet execution verification of the corrected post-MMAD fence.

The new trace records 1780/1777 executions of the decoded-byte store on the
two AIVs. The source has 16 * 128 = 2048 such stores per AIV per K tile,
and this call requires four K tiles (8192 stores per AIV). The ND word-store
sample also records only 512 calls, consistent with one activation tile.
The printed samples contain neither MMAD nor UB-to-L1 handoff. Taken together,
these observations are consistent with interruption during the first decode
tile, before the corrected fence is reached. They do not establish a new
deadlock, a hardware regression, or guaranteed completion given more time.
The 1.96/81.64/81.63 microsecond core durations are partial simulation data.

The `su_ccu_illegal_instr_t0` messages naming ZEROEXT/AND/SHL appear after
`SigIntHandler received signal: 2` and `Model is terminating`. This ordering
does not establish whether they are shutdown artifacts or independent faults.
Do not suppress the errors or change arithmetic based on their names alone.
The old log scanner missed their lowercase `[error]` tag; it now counts tags
case-insensitively and records whether each error followed a shutdown notice.
Errors in either phase still prevent successful collection status.

The initial five-minute harness cap was too restrictive for this scalar
decode instruction trace. The native contract already uses minimum N=32,
K=512; reducing M does not reduce the decoded B tile, and shortening K to
128 would change the supported contract rather than validate this binary.
The official [simulator timeout documentation](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_simulator_user_guide.md)
describes timeout as terminating simulation and parsing only the partial
data; its range is 1..2880 minutes. The harness now permits an explicit
1..60 minutes while keeping the default at five. A rough linear extrapolation
from this partial capture suggests tens of minutes, not a completion-time
guarantee. No kernel/library rebuild is part of this harness correction.

Use the same new library with one explicitly longer, bounded capture:

```bash
cd /home/g00872988/vllm-ascend-vq2a8
git pull --ff-only
/usr/local/python3.11.10/bin/python3 -u tools/profile_vq2a8_ascendc.py --diagnostic-build --library build/vq2a8-ascendc-syncfix/libvq2a8_ascendc.so --timeout-minutes 45
```

Every 30 seconds `ASCENDC_SIM_PROGRESS` prints the size and bounded last
instruction-log lines for AIC and both AIVs. It does not stream the large
CCU dumps or read device values. Changing PCs/operands provide evidence for
manual progress inspection; file growth alone cannot prove liveness and
unchanged buffered files cannot prove deadlock. The outer deadline remains
the requested simulation limit plus two parsing minutes, never reset by a
progress poll. Ctrl-C/expiry still kills only the owned profiler process group.
This adds no repeat campaign, no unbounded automatic retry, no new NPU work,
and no promotion of native/on-chip/model acceptance flags.

The development-host VQ2A8 suite passes **882 tests**, including 15 new
deadline, progress-observation and error-classification regressions. Ruff,
Markdown and repository-configured spelling checks pass for changed files.
The required `bash format.sh ci` was attempted and remains blocked by missing
local `pre-commit`. The native source and library are unchanged by this patch;
the longer simulator capture has not been run on the development host.
