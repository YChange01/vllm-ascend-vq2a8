# VQ2A8 ABCD candidates

## Scope and execution plan

Baseline: `a5c5a6f77`, V4/v2, `sign_fused_direct`, fused validity and route
mapping, position-template decoder, caller replay stream. Prepacked expert
weights are reused without conversion. These candidates do not enable generic
FULL graphs or alter the scheduler, RHT, bias correction, or scale calculation.

Implementation and acceptance order:

1. **A — planned runtime guards:** precompile field access, but validate the
   complete runtime/tensor metadata contract on every replay.
2. **B — vectorized validity reduction:** retain all output/status/flag checks
   and the old scalar implementation; reduce the packed finite mask on device.
3. **C — resident select + sign:** combine metadata selection and strided sign
   preparation, retaining separate selection and input-validity statuses.
4. **D — quantization tail + reorder:** retain Torch RHT, bias correction, scale
   and division; fuse only final clamp, FP8 conversion, reorder and descriptors.

Every candidate is independently opt-in. Defaults preserve the baseline. Each
stage needs CPU contracts, native numerical/graph/queue tests, real-model
eager/decoder comparison, and an unprofiled HTTP A/B/A measurement. CPU tests
and source review do not prove CANN compilation, NPU correctness or speedup.

## Switch matrix

| Stage | runtime guard | validity mode | select/sign | activation tail |
| --- | --- | --- | --- | --- |
| baseline | signature | fused | separate | torch |
| A | planned | fused | separate | torch |
| AB | planned | fused_vectorized | separate | torch |
| ABC | planned | fused_vectorized | fused | torch |
| ABCD | planned | fused_vectorized | fused | fused_reorder |

Also test B/C/D individually against baseline if a cumulative result regresses.
Never benchmark with host profiling or shadow metadata verification enabled.
Use a fresh process for each library/configuration, with only one job on card 1.

## Work status

- [x] A implementation and CPU contract tests
- [x] B implementation and native validation entry
- [x] C implementation and native validation entry
- [x] D implementation and native validation entry
- [x] Shared configuration, mode reports and regression run (known baseline failures below)
- [ ] CANN build and native/real-model correctness on card 1 (user environment)
- [ ] Independent and cumulative HTTP latency measurements (user environment)

Local checks on 2026-09-19 (Windows, Python 3.14.7, Torch 2.10.0+cpu):

- ABCD/native-probe/guard focused regression: **902 passed**.
- Full CPU regression before the final five D queue-test additions:
  **5,834 passed, 270 skipped, 14 failed**, plus 86 passing subtests.
- All 14 failures also reproduce against baseline `a5c5a6f77`: one old model
  fixture lacks `_v4_decode_graph`; two root-FP8 tests depend on CPU FMA rounding
  unavailable in this environment; eleven provenance tests expect an official
  runner hash while the baseline already contains decoder patches. These
  unrelated assertions and production guards were not weakened or bypassed.
- All changed Python files pass Ruff and formatting checks; `git diff --check`
  and the standalone builder's dry run pass. No CANN/NPU execution was performed.

## 1. Build once, without reinstalling the main package

This modifies only the standalone VQ2 library and Python adapter. Do not
uninstall vllm-ascend, delete `build/`, or rebuild SAS/QLI. Keep the known-good
perf3/direct/route-mapping libraries. The currently installed editable package
must point to this checkout. Use the actual CANN path if different.

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
python -u tools/build_vq2a8_v4_v2.py \
  --soc Ascend950DT_9582 \
  --cann /usr/local/Ascend/cann-9.1.0 \
  --build-dir build/vq2a8-ascendc-v4-v2-abcd \
  --jobs 4
```

Continue only after `V4_V2_BUILD=PASS`. All same-binary comparisons below use
this new library, including the baseline. A alone does not require the new
native ABIs, but B/C/D fail closed if their selected ABI is absent.

## 2. Per-stage switches and shared arguments

In each new validation/server terminal, run this block. Change only
`VQ2_STAGE=baseline` to `A`, `AB`, `ABC`, or `ABCD` for successive measurements.
Stages `B`, `C`, `D` are also available for isolated comparisons. This is a
Bash variable, not a new runtime environment flag.

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
VQ2_ABCD_LIB="$PWD/build/vq2a8-ascendc-v4-v2-abcd/libvq2a8_ascendc_v4_v2.so"
VQ2_STAGE=baseline
case "$VQ2_STAGE" in
  baseline) VQ2_MODES=(--runtime-guard signature --validity-mode fused --select-sign separate --activation-tail torch) ;;
  A)        VQ2_MODES=(--runtime-guard planned --validity-mode fused --select-sign separate --activation-tail torch) ;;
  AB)       VQ2_MODES=(--runtime-guard planned --validity-mode fused_vectorized --select-sign separate --activation-tail torch) ;;
  ABC)      VQ2_MODES=(--runtime-guard planned --validity-mode fused_vectorized --select-sign fused --activation-tail torch) ;;
  ABCD)     VQ2_MODES=(--runtime-guard planned --validity-mode fused_vectorized --select-sign fused --activation-tail fused_reorder) ;;
  B)        VQ2_MODES=(--runtime-guard signature --validity-mode fused_vectorized --select-sign separate --activation-tail torch) ;;
  C)        VQ2_MODES=(--runtime-guard signature --validity-mode fused --select-sign fused --activation-tail torch) ;;
  D)        VQ2_MODES=(--runtime-guard signature --validity-mode fused --select-sign separate --activation-tail fused_reorder) ;;
  *)        printf 'Unknown stage: %s\n' "$VQ2_STAGE"; return 1 2>/dev/null || exit 1 ;;
esac
VQ2_COMMON=(
  --model /home/g00872988/vq2a8
  --artifact /home/g00872988/vq2a8/experts_vq_v4_v2_prepacked
  --library "$VQ2_ABCD_LIB"
  --physical-npu 1 --compute-backend v2
  --activation-reorder vectorized --activation-preparation sign_fused_direct
  --route-mapping fused --decoder-metadata-mode position_template
  --kv-cache-mib 256 --reserve-gib 3 --engine-memory-fraction 0.9
)
printf 'STAGE=%s LIBRARY=%s\n' "$VQ2_STAGE" "$VQ2_ABCD_LIB"
test -f "$VQ2_ABCD_LIB"
```

## 3. Correctness gates, with your server stopped normally

Use idle physical card 1. Do not reset a card or stop another user's process.
Any FAIL/BLOCKED requires investigation; do not skip it or relax tolerances.
Run the existing-path gates once for the new binary:

```bash
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_activation_packed.py \
  --library "$VQ2_ABCD_LIB" --physical-npu 1 \
  --preparation-modes sign_fused_direct --queue-lifetime --timeout-s 900
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_validity_fused.py \
  --library "$VQ2_ABCD_LIB" --physical-npu 1 \
  --reduction scalar --queue-lifetime --timeout-s 600
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_route_mapping.py \
  --library "$VQ2_ABCD_LIB" --physical-npu 1 --queue-lifetime --timeout-s 300
```

Before adding B, run its standalone gate:

```bash
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_validity_fused.py \
  --library "$VQ2_ABCD_LIB" --physical-npu 1 \
  --reduction vectorized --queue-lifetime --timeout-s 600
```

Before adding C:

```bash
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_select_sign.py \
  --library "$VQ2_ABCD_LIB" --physical-npu 1 --queue-lifetime --timeout-s 900
```

Before adding D:

```bash
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_tail_reorder.py \
  --library "$VQ2_ABCD_LIB" --physical-npu 1 --queue-lifetime --timeout-s 900
```

B checks the exact validity bit and invalid/recovery behavior. C compares all
five outputs byte-for-byte against separate select/sign, including expanded
and padded BF16/FP32 inputs. D compares reordered FP8 bytes against unchanged
Torch clamp/cast plus vectorized reorder; it also checks projection output,
descriptor fields, poison rows, graph replay and queued owner lifetime. These
tests are required because a native cast's rounding/NaN behavior cannot be
certified from CPU source inspection.

For **each stage, including baseline and A**, run the real-model gate using
that stage's `VQ2_MODES`:

```bash
TASK_QUEUE_ENABLE=1 python -u tools/validate_vq2a8_v4_decoder_graph.py \
  "${VQ2_COMMON[@]}" "${VQ2_MODES[@]}" --timeout-s 1800
```

Require `V4_DECODER_GRAPH=PASS` and matching mode fields in both `summary.json`
and its graph report. This compares eager and decoder execution under the same
selected candidate, including position-template shadow/no-shadow checks and
request reuse. It does not replace the separate candidate-versus-baseline
native numerical gates, or prove equivalence for unsupported workloads.

## 4. Start one stage and measure HTTP latency

After correctness passes, use the same stage variables from section 2:

```bash
TASK_QUEUE_ENABLE=1 python -u tools/serve_vq2a8_v4.py \
  "${VQ2_COMMON[@]}" "${VQ2_MODES[@]}" \
  --device-route-decode --decode-graph decoder --graph-replay-stream caller \
  --max-model-len 16 --memory-fraction 1.0 --port 8000
```

After the server is ready, run the unchanged benchmark in another terminal.
Use the raw URL, not a Markdown link. Record the stage alongside its output.

```bash
cd /home/g00872988/vllm-ascend-vq2a8-v023
python -u tools/benchmark_vq2a8_serving.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 \
  --prompt '你好' --max-tokens 4 --warmups 3 --repeats 20 --timeout 300
```

Stop your server normally before the next validation/server process. Test
`baseline -> A -> AB -> ABC -> ABCD`; each delta is against the immediately
previous successful stage. If a stage regresses, keep its switch off and
measure later candidates independently. For differences near run-to-run noise,
repeat the previous stage after the candidate (control/candidate/control), then
increase repeats equally for both. Do not claim gains by selecting only the
fastest request. Compare mean, median and tail TPOT plus TTFT using all requests.

## Interpretation and rollback

- A still validates runtime/config/bank/root/layout on every replay. It caches
  the *check plan*, not the check result; immutable payload bytes remain under
  the existing residency contract. It adds no device synchronization.
- B retains the old scalar kernel and all checks; only packed-mask reduction
  changes. It is not permission to disable finite/status validation.
- C leaves RHT, bias GEMV, scale reduction and division untouched. It avoids a
  selected-sign intermediate, but still returns scale and bias metadata.
- D leaves the same upstream arithmetic untouched. Its fused kernel currently
  loads FP32 rows rather than FP8 rows: reduced launch count is **not** a speed
  guarantee. Keep `--activation-tail torch` if it loses.
- These changes do not make the entire serving loop a graph. The graph boundary
  remains the position-specialized decoder; scheduler/sampling are unchanged.
- For profiling, retain library SHA256, all mode flags, warm/steady windows and
  HTTP results separately. Device task sums and synchronization spans overlap;
  never add them or scale profiled replay intervals to HTTP TPOT.
- Disable all new switches using the `baseline` row to roll back without a
  weight repack. To reproduce the historical binary exactly, also select the
  previously saved library and code revision.
