# VQ2A8 TP1 integration

This opt-in integration consumes an existing VQ2A8 DeepSeek V4 checkpoint.
It is not a general-purpose weight quantizer or W4/W8 implementation.
Routed experts remain compressed; root and shared-expert weights use BF16.

## Scope

- Linux, Ascend950/A5, a matching CANN and torch_npu installation.
- TP/PP/DP = 1, one request, no expert/context parallelism.
- Expert projection N = 4096, K = 2048 or 4096, at most 256 experts per layer.
- Maximum context length 16, block size 128, explicit KV-cache allocation.
- Eager prefill and explicitly prepared, position-specific decoder graphs.
- Planned runtime guards, fused vectorized validity, fused resident selection/sign,
  vectorized activation reorder, original Torch activation tail and SwiGLU.

Other execution policies, FP8 root conversion, speculative decoding, async
scheduling, prefix caching, LoRA, offloading and experimental operator modes are
not supported. Invalid settings fail rather than selecting a fallback.

The extracted build needs fresh NPU validation. Prior results from the original
experimental binary do not establish correctness or speed of a rebuilt binary.
The native source attribution in `csrc/vq2a8_ascendc_v4_v2/NOTICE` also requires
contribution-rights clearance before this code can be submitted upstream.

## Build

Use the normal vLLM Ascend installation procedure for the matching vLLM version.
Source the CANN environment and set `SOC_VERSION` to the exact local Ascend950
variant before building. The opt-in build variables are:

```bash
export COMPILE_CUSTOM_KERNELS=1
export VLLM_ASCEND_BUILD_VQ2A8=1
python -m pip install --no-build-isolation -e .
```

`VLLM_ASCEND_BUILD_VQ2A8` defaults to `0` and is not sensitive. When enabled,
the normal CMake build installs `libvq2a8_ascendc_v4_v2.so` in `vllm_ascend`.
Runtime discovers that installed library; no `tools/` directory is required.
An explicit `ascendc_library` override must include `ascendc_sha256` and should
be used only with the matching validated binary in a fresh process.

## Checkpoint input

Keep the canonical root `config.json` and BF16/F32 root safetensors unchanged.
The runtime accepts either a complete direct-TP1 `experts_vq_ascend_v2` artifact
or its compressed V4/v2 prepacked form. Model configuration, shapes, manifests
and tensor contents are checked; a canonical `experts_vq` directory is not
silently treated as a direct or prepacked artifact.

For an existing direct-TP1 artifact, static compressed conversion is available
as an installed package module:

```bash
python -m vllm_ascend.quantization.vq2a8_prepack \
    --input /path/to/direct-tp1 \
    --model-config /path/to/model/config.json \
    --output /path/to/new-prepacked-directory --plan-only
```

Remove `--plan-only` to convert. Output must not exist or overlap the input.
Publication is atomic and non-overwriting, with serialized bytes and source
hashes checked before publication. Failed conversion preserves a quarantined
partial directory. This conversion does not validate device arithmetic or
model accuracy. Creating a direct-TP1 artifact from an arbitrary checkpoint
is outside this integration's command-line interface.

## Model integration

The ordinary vLLM model registry registers `VQ2A8TP1OfflineForCausalLM` as an
adapter inheriting the existing Ascend DeepSeek V4 model. It does not replace
the default DeepSeek V4 registration or monkey-patch other model instances.

`vllm_ascend.quantization.vq2a8_offline.offline_engine_options(model_root,
artifact)` constructs the supported vLLM engine options, including
`hf_overrides`, `additional_config.vq2a8_offline`, disabled outer graph
compilation and a 1 GiB KV allocation. It defaults to token-ID input
(`skip_tokenizer_init=True`). A serving deployment must enable tokenizer
initialization and preserve the other validated limits.

Minimal VQ2-specific additional configuration is
`{"vq2a8_offline": {"enabled": true, "artifact": "/path/to/artifact"}}`.
This fragment alone is not a complete engine configuration: the context,
scheduler, BF16, parallelism and memory constraints above also apply.

Per-layer validity and final logits validity are checked before publishing a
token. Graph capture is performed at worker startup, with mutable decoder/KV
state restored; no capture is triggered by a live request.

## Validation

Focused unit tests live in `tests/ut/quantization/test_vq2a8*.py`; they cover
schema, compressed conversion, CPU reference arithmetic, supported settings,
ownership/graph protocols and build contracts. CPU tests and source checks do
not execute CANN kernels. A bounded native smoke suite is also available after
building the optional library on Ascend950:

```bash
TASK_QUEUE_ENABLE=1 python -m pytest -v -rs \
    tests/e2e/pull_request/one_card/test_vq2a8_native.py
```

Select the intended device using the deployment's normal visibility settings.
Missing hardware or an unbuilt optional library is reported as skipped, not
passed. The smoke suite checks synthetic selection, exact zero projection,
graph replay and bounded asynchronous ownership; it is not general numeric
or model acceptance. Before upstream review, validate the standard build,
native numeric and queue-lifetime behavior, repeated prefill/decode equivalence
and serving latency on the target NPU using the same model artifact.
