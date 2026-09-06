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
routing, complete decoder execution and full-model quality remain unverified.

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
serving readiness. Return that report; full logs remain available beside it.
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
  host must run the expanded acceptance before that boundary can advance.

## Next acceptance milestones

1. Run the expanded packed expert acceptance on the user host. Keep any
   failing case, environment and artifact identity together.
2. Implement a bounded TP1 expert/MoE runtime: M>1 handling, duplicate route
   accumulation, shared experts, one routed-scale owner and finite outputs.
   Repeated M=1 calls may establish prefill correctness before optimization.
3. Register the quantization/loader integration without dense placeholders.
   Verify all weights loaded, layerwise outputs and peak HBM on a short
   offline prefill/decode, then validate logits/tokens against a known-good
   execution of this checkpoint.
4. Start serving only after the offline model gate passes. Run repeated
   prompts and mixed lengths, then establish a quality/performance baseline.
5. Optimize decode lookup and activation preparation; evaluate minimal
   native FP8 Cube primitives independently before changing the accepted
   Vector kernel. TP4 is a later milestone with separate routing/reduction
   validation.

The available eager preparation rebuilds/transfers a Hadamard matrix and
performs finite checks that synchronize. It remains a reference path, not
the final serving hot path. Optimize it with explicit preparation/cache
ownership after integration correctness has been established.
