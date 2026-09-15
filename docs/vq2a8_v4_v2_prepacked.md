# V4 + v2: reusable CPU-preconverted expert payloads

The optional prepacked path moves the **existing CPU layout conversion** out of
model startup. It does not change quantization, resident compute, routing,
activation preparation or graph execution. The original direct-TP1 path remains
the default; V4/v1 and old V3 artifacts are not repurposed.

## Convert once on CPU

Run from the v023 repository, using Python with CPU-capable PyTorch, NumPy and
safetensors. This tool does not initialize vLLM, torch-npu or an NPU, and does not
need CANN or a native library. The runtime and exporter call the same
`convert_expert_payload` function in `vq2a8_v4_v2_layout.py`.

First inspect the storage estimate without writing anything:

```bash
python -u tools/prepack_vq2a8_v4_v2.py \
  --input /home/g00872988/vq2a8/experts_vq_ascend_v2 \
  --output /home/g00872988/vq2a8/experts_vq_v4_v2_prepacked \
  --plan-only
```

Then convert; do **not** create the output directory beforehand:

```bash
python -u tools/prepack_vq2a8_v4_v2.py \
  --input /home/g00872988/vq2a8/experts_vq_ascend_v2 \
  --output /home/g00872988/vq2a8/experts_vq_v4_v2_prepacked \
  --experts-per-shard 16 --threads 4
```

The config defaults to `INPUT/../config.json`; use `--model-config` if needed.
The tool validates the source, converts both gate/up and down for every expert,
and writes six final tensors per projection: `packed_zn`, `pair_lut`,
`activation_order`, `weight_scale`, `weight_bias`, and `rht_sign`.
Preparation metadata stays in original K order. LUT/packed data remain compressed;
there is no full dense FP8 expansion or second quantization pass.

Each shard has an explicit expert axis. CPU memory is bounded by one output
shard plus conversion temporaries, not the whole model; reduce
`--experts-per-shard` if host RAM is limited. Plan bytes describe expert payloads
only, excluding small headers/manifest, not all root model weights. Keep enough
extra disk space for an independent copy of these payloads.

Every written tensor is reopened and compared byte-for-byte with the current
converter output, including FP8 encodings and signed-zero metadata. Source,
model-config and producer hashes are recorded; source files are checked again
before publication. These full scans add one-time export cost.
After verification, a complete manifest is written and the sibling staging
directory is atomically published without replacement. Existing targets,
overlapping input/output trees and symlink aliases are rejected. Failed staging
directories are preserved with their path printed; the input is never modified.
Do not point serving at an unfinished staging directory.

Wait for `V4_V2_PREPACK_STAGE=complete` and `V4_V2_PREPACK=PASS`.
A `--plan-only` PASS is only a plan, not a converted artifact.

## Use the preconverted directory

On your existing, validated V4 + v2 service command, add:

```text
--artifact /home/g00872988/vq2a8/experts_vq_v4_v2_prepacked
```

Keep `--compute-backend v2` and your existing library, device, activation and graph
options unchanged. For example, with the previously built perf3 candidate:

```bash
python -u tools/serve_vq2a8_v4.py \
  --model /home/g00872988/vq2a8 \
  --artifact /home/g00872988/vq2a8/experts_vq_v4_v2_prepacked \
  --library build/vq2a8-ascendc-v4-v2-perf3/libvq2a8_ascendc_v4_v2.so \
  --physical-npu 1 --compute-backend v2 --device-route-decode \
  --activation-reorder vectorized --activation-preparation fused \
  --decode-graph decoder --graph-replay-stream caller \
  --max-model-len 16 --kv-cache-mib 256 \
  --memory-fraction 1.0 --engine-memory-fraction 0.9 \
  --reserve-gib 3 --port 8000
```

Do not enable unvalidated activation/graph options just to use prepacking.
This feature itself is Python-only: **no additional native compilation is
required**. The earlier perf3 native options still require their corresponding
library. Restart an existing service to load the new Python reader, after
confirming the intended card is available; these tools do not stop other jobs.

Startup validates the model binding, complete layer/expert inventory, layout
version, safetensors shapes/dtypes and small preparation metadata. It does not
read the original expert payload or call the conversion function. Full shard
SHA256 verification is available through the reader API, but is not a default
startup scan. Preserve the published artifact as immutable; same-shaped byte
corruption is not guaranteed to be detected without full hash verification.

Native banks, device pointers and graphs are rebuilt normally in each process.
They are not serialized. Layout versioning is independent of library build hash,
so a kernel rebuild with an unchanged payload contract can reuse these files.
The same model config is still required, along with normal root weights for
serving; the original direct-TP1 expert directory is not needed by this path.

Expected logs:

```text
MODEL_V4_V2_EXPERT_SOURCE ... format=vq2a8_v4_v2_prepacked_v1 ... startup_conversion=false
MODEL_V4_V2_LOAD ... preload_host_convert_s=0.0 ...
```

These logs are JSON in actual output. The per-layer load report also includes
`preload_host_read_s`, `preload_host_validate_s` and `preload_h2d_s`. Lazy mmap page
faults may be charged to validation/upload; these are code-region timings, not
isolated physical disk/bus bandwidth measurements. Compare complete layer/startup
times with the same device and cold/warm filesystem-cache conditions.

## Verification and limits

CPU regression:

```bash
python tools/run_vq2a8_cpu_tests.py -k 'v4_v2_prepack or v4_v2_integration'
```

The broader local V4/offline/activation/runtime/artifact/repack/execution
regression selection passed **1,869 tests with 7 skips**. It excludes the legacy
`v023_provenance` tests: a separate broader run still found four existing failures
from their fixed framework-source hashes. Those unrelated checks were not
relaxed or re-enabled in serving. Ruff, Python 3.11 syntax and Markdown checks
also passed. The extracted conversion function's AST is unchanged from the
previous implementation.

For one real expert's NPU projection, retain the original expert directory as an
independent oracle and run on an available card:

```bash
python -u tools/validate_vq2a8_v4_v2.py \
  --model /home/g00872988/vq2a8 \
  --artifact /home/g00872988/vq2a8/experts_vq_v4_v2_prepacked \
  --library build/vq2a8-ascendc-v4-v2-perf3/libvq2a8_ascendc_v4_v2.so \
  --physical-npu 1 --phase resident --expert 0:0 --timeout-s 300
```

The test compares stored bytes to the current converter, then actually uploads
the stored tensors and checks projection outputs against the original expert.
`tools/validate_vq2a8_v4_decoder_graph.py` also accepts `--artifact` for full-model
eager/graph comparisons with the prepacked reader.

This reduces repeated **startup conversion**, not expert transfer volume, resident
HBM usage or steady-state TPOT. CPU tests and export byte checks do not prove NPU
integration or a particular startup speedup; measure those on the target system.
