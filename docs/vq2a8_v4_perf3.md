# V4/v2: three independent decode optimizations

The default V4/v1 and V4/v2 paths are preserved. These are opt-in candidates,
not a claim that TPOT reaches 30 ms. No old V3 resident implementation is used.
Compilation, numeric equivalence, graph correctness and performance must be
checked on the target Ascend950 NPU; CPU contract tests cannot certify them.

For slow repeated expert loading, [preconvert the V4/v2 payload once on CPU](vq2a8_v4_v2_prepacked.md)
and add `--artifact` to the serving or decoder-validation command. This reuses
the same compute library and leaves all three optimization choices unchanged.

## Switches and scope

| Option | Default | Candidate |
| --- | --- | --- |
| `--activation-reorder` | `scalar` | `vectorized`: contiguous activation/order DMA into UB, vector gather of FP8 bytes |
| `--activation-preparation` | `rowwise` | `fused`: native sign/input checks and scale/reduce/clamp/FP8/output checks |
| `--decode-graph` | `none` | Existing `moe`, or new position-specialized `decoder` |

The two activation candidates require `--compute-backend v2` and a newly built
`libvq2a8_ascendc_v4_v2.so`. Feature checks reject an old library before loading
full weights; they never silently substitute the baseline.

Vector reorder is **after** RHT and quantization and must preserve every FP8
byte, including signed zero and NaN encodings. It does not reorder the RHT or
allocate another full expert bank. The scalar path remains separately callable.

Fused preparation deliberately retains each original one-row RHT and bias
matrix multiplication. It is not FWHT or a batched GEMM replacement. Input and
output checks stay on device and are rejected before sampling. Native division
and FP8 conversion must pass strict byte-equivalence tests against the existing
rowwise implementation, including rounding boundaries and invalid inputs.

`decoder` captures embedding/decoder/attention/HC/norm/MoE work for TP1, B1,
BF16 roots and `--max-model-len <=16`. It requires device routing and caller
replay. Prefill, attention-metadata construction, LM head, final validity and
sampling remain eager. This does **not** enable generic vLLM whole-engine graphs:
the wrapper still passes `--enforce-eager` and engine graph mode `NONE`.

Each position has an independent graph pool so compressor branches at 3→4,
7→8 and 11→12 are not frozen to the wrong position. Replay refreshes live token,
position, slot-map, block-table and other device metadata. Capture restores KV,
compressor and indexer state before worker readiness. Unknown metadata
structures, incompatible shapes, changed immutable owners or inadequate
headroom fail explicitly; no request-time recapture or eager fallback hides
failure. More graph pools increase startup time and memory use.

## Build without overwriting a baseline

Inside the configured Linux container, from the v023 repository:

```bash
python -u tools/build_vq2a8_v4_v2.py \
  --soc Ascend950DT_9582 \
  --cann /usr/local/Ascend/cann-9.1.0 \
  --build-dir build/vq2a8-ascendc-v4-v2-perf3 \
  --jobs 4
```

An existing editable vLLM-Ascend installation uses these Python changes after
restarting the process. The new native code still requires the build above.
No package version changes or global dependency audit are necessary.

## Validate on idle card 1

Confirm the device mapping and availability with `npu-smi info`. Stop only your
own previous service if it occupies this card; these commands do not stop other
jobs. Run the stages separately and stop at the first failure.

First validate the vector adapter, its raw bytes, compute, ownership and graphs:

```bash
python -u tools/validate_vq2a8_v4_v2.py \
  --library build/vq2a8-ascendc-v4-v2-perf3/libvq2a8_ascendc_v4_v2.so \
  --physical-npu 1 \
  --phase all --activation-reorder vectorized \
  --timeout-s 300
```

Then validate fused activation independently of full-model inference:

```bash
python -u tools/validate_vq2a8_activation_fused.py \
  --library build/vq2a8-ascendc-v4-v2-perf3/libvq2a8_ascendc_v4_v2.so \
  --physical-npu 1 --queue-lifetime --timeout-s 300
```

Finally run real-model eager/decoder comparisons with both activation options.
This loads the full model once, compares greedy tokens and top-five log
probabilities, checks actual graph replay counts, crosses compression boundaries
and repeats requests to exercise cache-slot reuse:

```bash
python -u tools/validate_vq2a8_v4_decoder_graph.py \
  --model /home/g00872988/vq2a8 \
  --library build/vq2a8-ascendc-v4-v2-perf3/libvq2a8_ascendc_v4_v2.so \
  --physical-npu 1 --compute-backend v2 \
  --activation-reorder vectorized --activation-preparation fused \
  --kv-cache-mib 256 --reserve-gib 3 \
  --engine-memory-fraction 0.9 --timeout-s 1800
```

Keep each printed `SUMMARY`/`LOG` path, library SHA256 and full failure stage.
A plan-only result or a submitted asynchronous operation is not a device PASS.
If a new option fails, disable only that option to keep testing other candidates;
do not bypass the failed check or interpret incomplete timing as a speedup.

## Serve and benchmark

After target validation, start the candidate on card 1:

```bash
python -u tools/serve_vq2a8_v4.py \
  --model /home/g00872988/vq2a8 \
  --library build/vq2a8-ascendc-v4-v2-perf3/libvq2a8_ascendc_v4_v2.so \
  --physical-npu 1 --compute-backend v2 \
  --device-route-decode \
  --activation-reorder vectorized --activation-preparation fused \
  --decode-graph decoder --graph-replay-stream caller \
  --max-model-len 16 --kv-cache-mib 256 \
  --memory-fraction 1.0 --engine-memory-fraction 0.9 \
  --reserve-gib 3 --port 8000
```

Wait for `MODEL_V4_GRAPH_READY` and HTTP application startup. Confirm the report
says `compute_backend=v2`, `activation_reorder=vectorized`,
`activation_preparation=fused`, `effective_graph_mode=decoder`, `ready=true`
and `failed=false`. `full_model_graph_verified=false` remains intentional:
the LM head and sampler are outside the graph, and counters are not accuracy proof.

In another terminal in the same container:

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python -u tools/benchmark_vq2a8_serving.py \
  --base-url http://127.0.0.1:8000 --model vq2a8 \
  --prompt '你好' --max-tokens 4 \
  --warmups 3 --repeats 20 --timeout 300
```

This measures HTTP-visible TTFT/TPOT, not pure kernel time. `你好` is not fixed
10-token input. Use the same prompt/token IDs, sampling, request count and
profiling state for A/B. The useful sequence is baseline (`scalar`, `rowwise`,
`moe`/`caller`), vector only, vector+fused, then vector+fused+decoder. Restart
between configurations and retain all results; alternate A/B order to detect
the late-run slowdown seen previously. Profile separately from latency runs.

Relevant new device kernel names are `vq2a8_ascendc_v4_v2_prepare_vectorized`,
`vq2a8_v4_v2_activation_sign` and `vq2a8_v4_v2_activation_quantize` (profilers may
append suffixes). Compare both preparation and projection time, not only the
main resident kernel. Decoder graph reports should show one replay per decode
step, rather than 43 independent MoE replays.

## Local verification boundary

The targeted CPU regression command is:

```bash
python tools/run_vq2a8_cpu_tests.py \
  -k 'v4 or offline or activation or full_startup_trace or tp2_integration'
```

On the development host it passed 1,268 tests, with 1 skip. All 23 changed Python
files passed Ruff checks and formatting; Python 3.11 syntax, Markdown and diff
checks were also exercised. These include real CPU tensor arithmetic and fake
graph contract tests, not emulated NPU validation.

The full CPU suite is not green: it also exposes pre-existing root-FP8 rounding,
an older model-probe fixture and framework/provenance expectations. A new TP2
AST-fixture import failure found by that broader run was fixed and rechecked in
the targeted suite. Unrelated checks were not relaxed to obtain a green result.
No target CANN compilation, NPU correctness or latency result is claimed here.
