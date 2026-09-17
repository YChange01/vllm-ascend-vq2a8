# Extract an existing V4 decoder serving profile

`tools/extract_vq2a8_profile.py` is a standalone Python 3.11+ standard-library
script. Copy this one file into the offline machine. It does not import Torch,
connect to a service, initialize an NPU, install packages, or change the input.
No model restart, native build, network connection, or new capture is required.

## Inputs and command

Use one rank's `ASCEND_PROFILER_OUTPUT`, containing:

- `trace_view.json` (or `trace_view.json.gz`), required;
- `kernel_details.csv`, required for aligned device execution windows;
- `op_statistic.csv`, `api_statistic.csv`, and `analyse.done`, optional.

For the previously collected **three complete sequential requests, each with
four output tokens**, with warmup outside capture and one decoder graph replay
per decode token, there should be nine graph replays. Only in that setting use:

```bash
python -u tools/extract_vq2a8_profile.py \
  --profile-dir /home/g00872988/profiler_output/dp0_pp0_tp0_dcp0_ep0_rank0_379810_20260915231701837_ascend_pt/ASCEND_PROFILER_OUTPUT \
  --output-dir /home/g00872988/profiler_output/tpot_extract_01 \
  --decode-steps-per-request 3 --expected-requests 3
```

The output directory must be new. Omit `--output-dir` to create a timestamped
sibling directory. Paste `paste_summary.txt` (in chunks if necessary); keep the
larger `summary.json` locally. To print the saved pasteable report:

```bash
cat /home/g00872988/profiler_output/tpot_extract_01/paste_summary.txt
```

No archive or full 180 MB trace needs to leave the offline machine. Inspect the
text before sharing; names, shapes and existing call-stack excerpts can appear.

If request count/output length differs, adjust both grouping arguments. If the
capture includes partial requests, concurrent requests, MoE-only graphs, warmup,
or unknown boundaries, **omit the grouping arguments**. All adjacent windows are
then explicitly unverified candidates, not proven within-request decode steps.
The script does not reconstruct request identities from timestamps. Divisibility
and expected-count checks catch some, but not all, incorrect grouping assumptions.

## What to send back

The text includes:

1. `INVENTORY`, `GROUPING`, `CYCLE`, `GRAPH_API`, `LONG_SCALAR`: graph cadence,
   grouping assumptions and long scalar reads (not presumed source lines).
2. `WINDOW` sections for the last eligible interval, graph-to-scalar-return and
   scalar-return-to-next-graph, when a unique long scalar exists.
3. `DEVICE_COVERAGE`, `DEVICE_TYPE`, `WINDOW_MATMUL`: clipped kernel sums versus
   interval unions, and MatMul by core type, shape, dtype and format.
4. `MAIN_CPU_RECORDED_EXCLUSIVE`, `API_INCLUSIVE_NOT_ADDITIVE`,
   `HOST_SCOPE_NOT_ADDITIVE`, and uninstrumented gaps: candidates for the
   approximately 39 ms post-scalar window.
5. `GLOBAL_MATMUL_MIXED_PHASES`, `HOST_MATMUL_SAMPLE`: global shape distribution
   and existing host argument/stack excerpts. Missing shapes/stacks cannot be
   recovered if the capture never recorded them.

Start by pasting the whole text. If too long, first send `INVENTORY` through
`LONG_SCALAR`, and the `post_scalar_to_next_graph` window. Then send the other
windows and MatMul groups. `--top 5` reduces ranking lengths.

## Boundaries and optional extraction

- Default selects one late eligible cycle; this is not an average decode step.
  Use `--cycles 2` for the last two, or `--cycle-index 7` to select the interval
  starting at the eighth replay (zero-based).
- `--main-tid` selects one API thread. Multi-device CSVs are rejected.
- Only complete `X` spans are analyzed. B/E scopes are counted in inventory but
  not paired or analyzed. Crossings in nominally nested CPU spans are flagged.
- Timestamps must be microseconds and aligned between CSV and trace from the
  **same run**. Empty device-window overlap is a warning, not proof of idle NPU.
- Kernel duration sums may overlap; unions describe recorded device-task
  coverage, not hardware utilization or a proven critical path.
- CPU-op spans include waits. Subtracting child spans avoids nested double
  counting, but does not turn those spans into actual CPU busy time.
- Scalar reads, stream synchronization and device execution overlap. Their
  totals must not be added together or rescaled to the unprofiled 72 ms TPOT.
- Graph API latency is host submission, not graph device execution latency.
- Optional `--write-trace-slice` writes a short view with original timestamps,
  full intersecting X events and metadata. Outside-window flow endpoints may be
  missing; B/E scopes are omitted. This is not a complete causal trace.
- JSON is parsed incrementally in two passes. Retained complete events default
  to a 100,000-event limit, selected windows to 2,000 ms each. Exceeding a limit
  fails explicitly; it does not silently truncate successful statistics. A
  failed optional slice can remain incomplete in its new output directory.
- If existing annotations do not cover Python metadata traversal or scheduler
  work, this report can locate the gap but cannot invent function attribution.
  A later short, instrumented capture may then be necessary.

## Local tests

```bash
python tests/ut/quantization/test_vq2a8_profile_extract.py
```

These synthetic tests verify parsing, interval accounting and output behavior.
They do not establish performance on the user's unavailable real trace or NPU.
