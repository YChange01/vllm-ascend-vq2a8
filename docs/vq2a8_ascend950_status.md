# VQ2A8 Ascend 950 TP1: scope, evidence and acceptance

## Goal and current boundary

The goal is faithful DeepSeek-V4-Flash-VQ2A8 inference on one Ascend 950,
including prefill, decode, routed and shared experts, and stable serving.
Expert weights stay packed; activation and codebook storage stay E4M3FN.
Full-model logits/token quality, peak HBM and throughput are acceptance
criteria. An HTTP 200 or one passing expert is not full-model acceptance.

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
