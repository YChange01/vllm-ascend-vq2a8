# VQ2A8 migration to vLLM Ascend 0.26

## Scope and status

The migration branch is `ascend950-vq2a8-v026`, based on official
`vllm-ascend v0.26.0rc1` (`f2f74a16c`). Its vLLM counterpart is **v0.26.0**.
Ascend's tag is a release candidate, not a final v0.26.0 release.
The original `ascend950-vq2a8` branch at `f9f49e316` is preserved for rollback.
Publishing the new branch and installing it on a server are separate steps;
a local migration commit does not update the server.

This ports the existing opt-in offline architecture, packed expert runtime,
AscendC kernel/binding, validation tools, and custom QLI/SAS metadata fixes.
It preserves upstream 0.26 model registrations and Eagle auxiliary-state
collection. Default upstream models are not switched to VQ2A8.
The direct CLI entry points avoid the new `tools/bisect` package shadowing
Python's standard-library module.

The packed format remains `vq2a8_direct_tp1_v1`. **Do not repack or retransmit
the model merely for this migration.** The existing `experts_vq_ascend_v2`
directory remains the input. Kernel math, routing and checkpoint format have
not been redesigned as part of this version migration.

This is source/CPU validation, **not** a new NPU, quality, native-instruction,
performance or serving PASS. TP1 offline inference remains the supported
experimental entry point; this does not add a `vllm serve` quantization method.

## Matched environment

Use Linux with Python >=3.10,<3.13; Python 3.11 matches the previous server.
Use a separate virtual environment and checkout, keeping the old environment
and libraries available until the new stack passes hardware acceptance.

| Component | Migration target |
| --- | --- |
| vLLM | `0.26.0`, source install with `VLLM_TARGET_DEVICE=empty` |
| vLLM Ascend | This branch on `v0.26.0rc1`; local development suffix is normal |
| PyTorch | `2.10.0` (CPU wheel used by torch-npu) |
| torch-npu | `2.10.0.post4` |
| Transformers | `5.14.1` |
| Triton Ascend | `3.2.2` |
| FastAPI | `>=0.133.0,<0.137.0` |
| CANN / NNAL | `9.1.0`, with compatible driver/firmware |

The official Ascend rc1 requirements still cap FastAPI below 0.124.0, while
vLLM 0.26.0 requires >=0.133.0,<0.137.0. This branch updates **both** runtime
and build requirements to the latter interval; installing with `--no-deps`
to hide this conflict is not the migration procedure.

For Ascend 950, the official rc1 `Dockerfile.a5` additionally installs
`cann-950-ops-transformer 9.2.0-beta.2` on its CANN 9.1.0 base. Check this
operator package when preparing a new 950 machine; this is not an instruction
to replace the entire CANN installation with 9.2. Old custom `.so` files do
not substitute for the new stack's dependencies.

Sources: [release installation requirements](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/installation.html),
[Ascend rc1 requirements](https://github.com/vllm-project/vllm-ascend/blob/v0.26.0rc1/requirements.txt),
[vLLM 0.26 requirements](https://github.com/vllm-project/vllm/blob/v0.26.0/requirements/common.txt),
[950 image recipe](https://github.com/vllm-project/vllm-ascend/blob/v0.26.0rc1/Dockerfile.a5).

## Install and rebuild on the NPU server

Commands below assume this migration branch is already present in a **new**
Linux checkout. Do not execute them in the old checkout/venv. Adjust the CANN
environment-script paths to the installed locations and confirm `npu-smi info`
works first. No model files or old build directories need to be deleted.

```bash
set -e
# Keep upstream's release tag available for setuptools-scm version detection.
# A branch fetched from a fork does not necessarily bring that tag with it.
git fetch https://github.com/vllm-project/vllm-ascend.git tag v0.26.0rc1
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
/usr/local/python3.11.10/bin/python3 -m venv .venv-v026
source .venv-v026/bin/activate
python -m pip install --upgrade pip

# From the migration checkout; install Ascend's runtime dependencies first.
python -m pip install -r requirements.txt \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --extra-index-url https://download.pytorch.org/whl/cpu

# Empty-device vLLM uses common requirements, not CUDA/CPU torch 2.11 pins.
# Supply its build tools explicitly when disabling build isolation.
python -m pip install 'cmake>=3.26.1' ninja 'packaging>=24.2' \
  'setuptools>=77.0.3,<81' 'setuptools-scm>=8' \
  'setuptools-rust>=1.9' wheel jinja2
git clone --branch v0.26.0 --depth 1 \
  https://github.com/vllm-project/vllm.git ../vllm-v026
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation -e ../vllm-v026

# Rebuild this branch's Python extension AND custom AICPU/ACLNN operators.
# Leave SOC_VERSION unset for setup.py's npu-smi auto-detection.
env -u SOC_VERSION COMPILE_CUSTOM_KERNELS=1 python -m pip install -e . \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip check
python tools/validate_vq2a8_v026_environment.py
```

If automatic SOC detection fails, use the **lowercase CANN target accepted by
the toolkit** for `SOC_VERSION` in the main package build. Do not guess a board
suffix. The standalone build's `--soc` is separate and accepts the chip name
reported by torch-npu, for example `Ascend950PR_958b` on the old 198 machine.
Do not let the standalone script's historical 957d default select a wrong chip.

```bash
VQ2_SOC=$(python -c 'import torch, torch_npu; print(torch.npu.get_device_name(0))')
printf 'Standalone target: %s\n' "$VQ2_SOC"
python -u tools/build_vq2a8_ascendc.py \
  --soc "$VQ2_SOC" --build-dir build/vq2a8-ascendc-v026 --jobs 4
```

Build isolation for vLLM itself would download its torch 2.11 build dependency;
the empty-device command above builds against the prepared Ascend environment.
Do not install the generic CUDA vLLM wheel over this source install. If an
existing environment contains both generic `triton` and `triton-ascend`, repair
the overlapping installation before proceeding; using a fresh venv avoids
inheriting that collision. The preflight checks exact package versions and
real imports, but does not certify compiled operators or execute model kernels.

## One bounded NPU acceptance command

After environment validation and both builds pass, run from the new checkout:

```bash
python -u tools/validate_vq2a8_tp1_acceptance.py \
  --stage model \
  --model /home/g00872988/DeepSeek-V4-Flash-VQ2A8-32x256 \
  --artifact /home/g00872988/DeepSeek-V4-Flash-VQ2A8-32x256/experts_vq_ascend_v2 \
  --physical-npu 0 --device npu:0 \
  --execution-policy ascendc --root-linear-mode bf16 \
  --ascendc-library build/vq2a8-ascendc-v026/libvq2a8_ascendc.so
```

Adjust only model/artifact paths and the physical device if needed. This entry
point runs the short native-library hardware preflight before the offline
model test, then checks metadata, real routing, all 43 layers, short prefill,
decode and two-run repeatability. Per-expert verbose logging is opt-in.
Keep the emitted report and new library SHA; a successful build alone is not
model execution evidence. The diagnostic timing is not serving throughput.

Do not pass the old 0.23 report as `--baseline-report`: its same-environment
guard deliberately rejects a changed vLLM package/attention source. Do not
weaken that guard to manufacture an exact PASS. Preserve old logits as a
cross-version comparison reference, investigate differences explicitly, and
freeze a new 0.26 baseline only after NPU execution is validated.
Native FP8 instruction provenance, on-chip decode, independent logits quality,
longer workloads, performance and serving still require their own evidence.
There is no need to repeat the old 45-minute simulator timeout as an
installation prerequisite.

## Local validation record, 2026-09-08

Local hardware: Windows x64, Python 3.14.7, real torch 2.10.0+cpu and
Transformers 5.14.1. This Python version is used only for CPU checks/repacking,
not certified as an Ascend runtime.

- CPU suite: 879 passed, 211 skipped, 2 failed. The two failures are the same
  existing FP8/FMA rounding tests reproduced on the preserved old branch in
  the same Windows environment. Their tolerances/expectations were not relaxed.
- New migration contracts: 32 passed (included above). Covers dependency
  rejection, timeout reporting, direct CLI launches, model registration,
  real token-ID forwarding and upstream auxiliary-state collection.
- Ruff format/check passed for the changed Python files. The full
  `bash format.sh ci` command could not start because `pre-commit` is absent
  from this PC's shell; this is not a full repository lint/CI PASS.
- Hardware/Triton tests remain skipped. CPU contract collection uses a
  task-local namespace/JIT shim that raises if a device kernel is launched;
  it is not a Triton compiler or an NPU emulator. It runs without the repo's
  NPU-dependent global pytest configuration.
- Real model header audit against vLLM 0.26's `DeepseekV4Config`: 1199 root
  tensors, 984 parameter mappings, 43 MoE layers and 43 repacked layers match.
  Constructor auditing uses meta tensors/device stand-ins, not an actual
  instantiated vLLM engine. The actual new EngineArgs field names were checked.
- Two known failing tests: `test_inverse_rope_fp8_midpoint_regression_and_bounded_diagnostics`
  and `test_rounding_probe_requires_exact_inputs_scales_and_fp32_before_weight_gates`
  in `tests/ut/quantization/test_vq2a8_root_fp8.py`.

NPU compilation/imports/execution, cross-version logit agreement, quality,
latency and serving have **not** been validated on this branch locally.
Do not carry any old binary or model PASS flag forward automatically.
