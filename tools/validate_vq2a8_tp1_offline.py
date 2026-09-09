#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded TP1 model execution gate. No HTTP server or quality certification.

Run under acceptance.py --stage model for device isolation and abort reporting.
The two greedy runs retain full logits on disk, but print only short metrics.
"""

from __future__ import annotations

# Direct scripts must not put tools/bisect ahead of the stdlib bisect module.
# ruff: noqa: E402
import os as _bootstrap_os
import sys as _bootstrap_sys

if not __package__:
    _bootstrap_sys.path[0] = _bootstrap_os.path.dirname(
        _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))
    )

import argparse
import hashlib
import inspect
import json
import os
import time
from pathlib import Path


def reset_worker_trace(worker):
    worker.get_model().reset_offline_trace()
    return os.getpid()


def capture_worker_trace(worker):
    return worker.get_model().offline_evidence()


def configure_v2_worker(worker, preset):
    return worker.get_model().configure_performance_probe(measurement=False, compact=False, optimization=preset)


def single_worker_result(results):
    if len(results) != 1:
        raise ValueError("The offline gate requires exactly one worker result.")
    return results[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-report", type=Path, help="Frozen report from the acceptance supervisor.")
    parser.add_argument("--device", choices=["npu:0"], default="npu:0")
    parser.add_argument("--audit-model", action="store_true")
    parser.add_argument("--verify-tensor-hashes", action="store_true")
    parser.add_argument("--verbose-experts", action="store_true", help="Print per-expert load/execution diagnostics.")
    parser.add_argument("--execution-policy", choices=["baseline", "cached", "ascendc", "ascendc_v2"], default="cached")
    parser.add_argument("--ascendc-library", type=Path)
    parser.add_argument("--ascendc-preflight", type=Path)
    parser.add_argument("--ascendc-v2-library", type=Path)
    parser.add_argument("--ascendc-v2-preflight", type=Path)
    parser.add_argument("--ascendc-v2-preset", choices=["fast", "batched"], default=None)
    parser.add_argument("--root-linear-mode", choices=["bf16", "online_fp8_sm90"], default="bf16")
    parser.add_argument("--cache-budget-gib", type=float, default=0.0, help="0: auto budget after root loading.")
    parser.add_argument(
        "--cache-reserve-gib", type=float, default=16.0, help="Reserve for KV, workspace and allocator."
    )
    args = parser.parse_args()
    if args.execution_policy == "ascendc":
        if args.ascendc_library is None or args.ascendc_preflight is None:
            parser.error("AscendC requires --ascendc-library and the short --ascendc-preflight receipt.")
    elif args.ascendc_library or args.ascendc_preflight:
        parser.error("Native library/preflight options require execution-policy ascendc.")
    if args.execution_policy == "ascendc_v2":
        if args.ascendc_v2_library is None or args.ascendc_v2_preflight is None:
            parser.error("V2 requires --ascendc-v2-library and --ascendc-v2-preflight.")
        if args.root_linear_mode != "bf16":
            parser.error("V2 bring-up requires BF16 roots.")
    elif args.ascendc_v2_library or args.ascendc_v2_preflight or args.ascendc_v2_preset:
        parser.error("V2 options require explicit execution-policy ascendc_v2.")
    from tools.validate_vq2a8_v026_environment import check_scheduler_apis, require_v026_stack

    print("MODEL_V026_ENVIRONMENT " + json.dumps(require_v026_stack()), flush=True)
    if args.baseline_report and args.root_linear_mode != "bf16":
        parser.error("The phase-1 BF16 baseline is not an exact oracle for online FP8; do not mix these gates.")
    model_root, artifact_root = args.model.resolve(strict=True), args.artifact.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    native_library = None
    if args.execution_policy == "ascendc":
        from tools.validate_vq2a8_ascendc import checked_model_preflight, require_hardware_runtime

        require_hardware_runtime()
        native_library = checked_model_preflight(args.ascendc_library, args.ascendc_preflight)
        print("MODEL_ASCENDC_LIBRARY " + json.dumps(native_library), flush=True)
    elif args.execution_policy == "ascendc_v2":
        from tools.validate_vq2a8_ascendc import require_hardware_runtime
        from tools.validate_vq2a8_ascendc_v2 import checked_model_preflight

        require_hardware_runtime()
        if artifact_root != model_root / "experts_vq_ascend_v2":
            raise ValueError("V2 preflight and model must use the same canonical artifact.")
        native_library = checked_model_preflight(args.ascendc_v2_library, args.ascendc_v2_preflight, model_root)
        print("MODEL_ASCENDC_V2_LIBRARY " + json.dumps(native_library), flush=True)
    # Existing vLLM diagnostic control: the single worker stays in this
    # supervised process. A timeout/abort must not leave an EngineCore orphan.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Lazy device imports: --help is usable on a host without torch/NPU.
    startup_start = time.perf_counter()
    print("MODEL stage=import_runtime_start", flush=True)
    import torch
    import torch_npu  # noqa: F401
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs

    from tools.validate_vq2a8_qli_metadata import run_preflight
    from tools.validate_vq2a8_sas_attention import run_sas_preflight
    from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device, environment_report
    from tools.vq2a8_baseline import compare_baseline_run, load_baseline
    from vllm_ascend.quantization.vq2a8_offline import (
        OFFLINE_CONTEXT_LIMIT,
        OFFLINE_NEW_TOKENS,
        OFFLINE_RUNS,
        audit_offline_root,
        offline_engine_options,
        validate_offline_evidence,
    )
    from vllm_ascend.quantization.vq2a8_runtime import open_vq2a8_tp1_artifact
    from vllm_ascend.quantization.vq2a8_validation import audit_model_storage

    environment = environment_report()
    print("ENVIRONMENT " + json.dumps(environment), flush=True)
    print("MODEL_V026_SCHEDULER_PREFLIGHT " + json.dumps(check_scheduler_apis()), flush=True)
    previous_runs, previous_logits = None, None
    if args.baseline_report:
        print("MODEL stage=baseline_preflight", flush=True)
        previous_runs, previous_logits = load_baseline(
            args.baseline_report, environment, model_root, artifact_root, execution_policy=args.execution_policy
        )
        print("BASELINE_PREFLIGHT=PASS INDEPENDENT_REFERENCE=False", flush=True)
    print("MODEL stage=device_init_start", flush=True)
    print("DEVICE " + json.dumps(_initialize_device(torch.device(args.device))), flush=True)
    if native_library:
        require_hardware_runtime()
    print("MODEL stage=artifact_and_root_audit", flush=True)
    artifact = open_vq2a8_tp1_artifact(
        artifact_root,
        model_root / "config.json",
        require_complete=True,
        require_reference_identity=True,
        verify_tensor_hashes=args.verify_tensor_hashes,
    )
    print(
        "ARTIFACT_RESULT "
        + json.dumps(
            {
                "root": str(artifact_root),
                "complete": True,
                "layers": len(artifact.layers),
                "root_tensors": len(audit_offline_root(model_root)),
            }
        ),
        flush=True,
    )
    if args.audit_model:
        print("MODEL_AUDIT " + json.dumps(audit_model_storage(artifact)), flush=True)
    config = json.loads((model_root / "config.json").read_text())
    tokenizer = Tokenizer.from_file(str(model_root / "tokenizer.json"))
    prompt_text = "The answer to 1 + 1 is"
    prompt = tokenizer.encode(prompt_text, add_special_tokens=False).ids
    bos = config.get("bos_token_id")
    if isinstance(bos, int) and (not prompt or prompt[0] != bos):
        prompt.insert(0, bos)
    if not 2 <= len(prompt) <= OFFLINE_CONTEXT_LIMIT - OFFLINE_NEW_TOKENS:
        raise ValueError(f"Prompt does not fit the short execution gate: {len(prompt)} tokens.")
    print("MODEL stage=qli_metadata_preflight", flush=True)
    run_preflight(torch.device(args.device), config, prompt_tokens=len(prompt))
    print("MODEL stage=sas_attention_preflight", flush=True)
    run_sas_preflight(torch.device(args.device), config, prompt_tokens=len(prompt))
    options = offline_engine_options(
        model_root,
        artifact_root,
        execution_policy=args.execution_policy,
        cache_budget_gib=args.cache_budget_gib,
        cache_reserve_gib=args.cache_reserve_gib,
        root_linear_mode=args.root_linear_mode,
        ascendc_library=native_library["path"] if args.execution_policy == "ascendc" else None,
        ascendc_sha256=native_library["sha256"] if args.execution_policy == "ascendc" else None,
        ascendc_v2_library=native_library["path"] if args.execution_policy == "ascendc_v2" else None,
        ascendc_v2_sha256=native_library["sha256"] if args.execution_policy == "ascendc_v2" else None,
        verbose_experts=args.verbose_experts,
    )
    missing = set(options) - set(inspect.signature(EngineArgs).parameters)
    if missing or not hasattr(LLM, "collective_rpc"):
        raise RuntimeError(f"Installed vLLM lacks required offline gate APIs/options: {sorted(missing)}.")
    print("MODEL_PLAN " + json.dumps(options), flush=True)
    print("MODEL stage=construct_load_profile_kv_cache", flush=True)
    engine_start = time.perf_counter()
    llm = LLM(**options)
    if args.ascendc_v2_preset is not None:
        single_worker_result(llm.collective_rpc(configure_v2_worker, args=(args.ascendc_v2_preset,)))
    print("MODEL stage=engine_ready", flush=True)
    print(
        "MODEL_STARTUP_TIMING "
        + json.dumps(
            {
                "startup_s": time.perf_counter() - startup_start,
                "construct_load_profile_kv_s": time.perf_counter() - engine_start,
                "execution_policy": args.execution_policy,
                "expert_device": args.device,
                "expert_load_path": "cpu_validate_then_device_cache",
                "root_linear_execution": args.root_linear_mode,
                "native_fp8_dot": None if native_library else False,
            }
        ),
        flush=True,
    )
    samples = SamplingParams(temperature=0, max_tokens=OFFLINE_NEW_TOKENS, ignore_eos=True, detokenize=False)
    baseline_logits, baseline_tokens = None, None
    for run in range(OFFLINE_RUNS):
        pid = single_worker_result(llm.collective_rpc(reset_worker_trace))
        if pid != os.getpid():
            raise RuntimeError("Worker isolation contract failed: worker is outside the supervised process.")
        print(f"MODEL run={run} stage=prefill_decode", flush=True)
        run_start = time.perf_counter()
        generated = llm.generate([{"prompt_token_ids": prompt}], samples, use_tqdm=False)
        generation_s = time.perf_counter() - run_start
        print(f"MODEL run={run} stage=collect_validate_evidence", flush=True)
        if len(generated) != 1 or not generated[0].finished or len(generated[0].outputs) != 1:
            raise ValueError("The short offline request did not finish normally.")
        tokens = list(generated[0].outputs[0].token_ids)
        evidence = single_worker_result(llm.collective_rpc(capture_worker_trace))
        if evidence.get("root_fp8", {}).get("mode", "bf16") != args.root_linear_mode:
            raise ValueError("Requested root mode was not installed on the executing model.")
        result = validate_offline_evidence(
            evidence,
            prompt,
            tokens,
            config["num_hidden_layers"],
            config["vocab_size"],
            execution_policy=args.execution_policy,
            ascendc_sha256=native_library["sha256"] if native_library else None,
            ascendc_v2_sha256=native_library["sha256"] if args.execution_policy == "ascendc_v2" else None,
        )
        logits = evidence["logits"]
        path = output / f"run-{run}-logits.safetensors"
        save_file({"logits": logits.contiguous()}, str(path))
        result.update(
            {
                "run": run,
                "generation_s": generation_s,
                "execution_policy": args.execution_policy,
                "logits_file": str(path),
                "logits_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "prompt_text": prompt_text,
                "prompt_token_ids": prompt,
                "generated_text": tokenizer.decode(tokens, skip_special_tokens=False),
            }
        )
        # Retain every run before checking reproducibility, including failures.
        (output / f"run-{run}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if previous_runs is not None:
            comparison = compare_baseline_run(previous_runs[run], previous_logits[run], result, logits)
            (output / f"run-{run}-baseline.json").write_text(json.dumps(comparison, indent=2) + "\n")
            print("MODEL_BASELINE_RESULT " + json.dumps(comparison), flush=True)
            if not comparison["baseline_exact"]:
                raise AssertionError(
                    "Offline logits/tokens differ from the frozen same-policy baseline; evidence retained."
                )
        if baseline_logits is not None:
            if tokens != baseline_tokens or not torch.equal(logits, baseline_logits):
                print(
                    "MODEL_REPEAT_FAILURE "
                    + json.dumps(
                        {
                            "tokens_equal": tokens == baseline_tokens,
                            "max_abs_error": (logits - baseline_logits).abs().max().item(),
                        }
                    ),
                    flush=True,
                )
                raise AssertionError(
                    "Repeated offline logits/tokens are not exactly reproducible; both runs are saved."
                )
        else:
            baseline_logits, baseline_tokens = logits.clone(), tokens
        print("MODEL_RESULT " + json.dumps(result, allow_nan=False), flush=True)
    print(
        "VQ2A8_TP1_OFFLINE_EXECUTION_GATE=PASS "
        + json.dumps(
            {
                "runs": OFFLINE_RUNS,
                "new_tokens": OFFLINE_NEW_TOKENS,
                "layers": len(artifact.layers),
                "native_fp8_dot": None if native_library else False,
                "native_instruction_verified": False,
                "repeat_exact": True,
                "root_linear_mode": args.root_linear_mode,
                "native_fp8_root_matmul": args.root_linear_mode == "online_fp8_sm90",
                "root_fp8_execution_verified": args.root_linear_mode == "online_fp8_sm90",
                "baseline_exact": True if previous_runs is not None else None,
                "offline_execution_verified": True,
                "expert_execution_policy": args.execution_policy,
                "ascendc_model_execution_verified": args.execution_policy == "ascendc",
                "ascendc_v2_model_execution_verified": args.execution_policy == "ascendc_v2",
                "on_chip_decode_verified": False,
                "logits_reference_verified": False,
                "quality_verified": False,
                "serving_integration_verified": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
