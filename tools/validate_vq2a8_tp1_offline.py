#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded TP1 model execution gate. No HTTP server or quality certification.

Run under acceptance.py --stage model for device isolation and abort reporting.
The two greedy runs retain full logits on disk, but print only short metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path


def reset_worker_trace(worker):
    worker.get_model().reset_offline_trace()
    return os.getpid()


def capture_worker_trace(worker):
    return worker.get_model().offline_evidence()


def single_worker_result(results):
    if len(results) != 1:
        raise ValueError("The offline gate requires exactly one worker result.")
    return results[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=["npu:0"], default="npu:0")
    parser.add_argument("--audit-model", action="store_true")
    parser.add_argument("--verify-tensor-hashes", action="store_true")
    args = parser.parse_args()
    model_root, artifact_root = args.model.resolve(strict=True), args.artifact.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Existing vLLM diagnostic control: the single worker stays in this
    # supervised process. A timeout/abort must not leave an EngineCore orphan.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Lazy device imports: --help is usable on a host without torch/NPU.
    print("MODEL stage=import_runtime_start", flush=True)
    import torch
    import torch_npu  # noqa: F401
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs

    from tools.validate_vq2a8_tp1_packed_kernel import _initialize_device, environment_report
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

    print("ENVIRONMENT " + json.dumps(environment_report()), flush=True)
    print("MODEL stage=device_init_start", flush=True)
    print("DEVICE " + json.dumps(_initialize_device(torch.device(args.device))), flush=True)
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
    options = offline_engine_options(model_root, artifact_root)
    missing = set(options) - set(inspect.signature(EngineArgs).parameters)
    if missing or not hasattr(LLM, "collective_rpc"):
        raise RuntimeError(f"Installed vLLM lacks required offline gate APIs/options: {sorted(missing)}.")
    print("MODEL_PLAN " + json.dumps(options), flush=True)
    print("MODEL stage=construct_load_profile_kv_cache", flush=True)
    llm = LLM(**options)
    print("MODEL stage=engine_ready", flush=True)
    samples = SamplingParams(temperature=0, max_tokens=OFFLINE_NEW_TOKENS, ignore_eos=True, detokenize=False)
    baseline_logits, baseline_tokens = None, None
    for run in range(OFFLINE_RUNS):
        pid = single_worker_result(llm.collective_rpc(reset_worker_trace))
        if pid != os.getpid():
            raise RuntimeError("Worker isolation contract failed: worker is outside the supervised process.")
        print(f"MODEL run={run} stage=prefill_decode", flush=True)
        generated = llm.generate([{"prompt_token_ids": prompt}], samples, use_tqdm=False)
        print(f"MODEL run={run} stage=collect_validate_evidence", flush=True)
        if len(generated) != 1 or not generated[0].finished or len(generated[0].outputs) != 1:
            raise ValueError("The short offline request did not finish normally.")
        tokens = list(generated[0].outputs[0].token_ids)
        evidence = single_worker_result(llm.collective_rpc(capture_worker_trace))
        result = validate_offline_evidence(evidence, prompt, tokens, config["num_hidden_layers"], config["vocab_size"])
        logits = evidence["logits"]
        path = output / f"run-{run}-logits.safetensors"
        save_file({"logits": logits.contiguous()}, str(path))
        result.update(
            {
                "run": run,
                "logits_file": str(path),
                "logits_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "prompt_text": prompt_text,
                "prompt_token_ids": prompt,
                "generated_text": tokenizer.decode(tokens, skip_special_tokens=False),
            }
        )
        # Retain every run before checking reproducibility, including failures.
        (output / f"run-{run}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
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
                "native_fp8_dot": False,
                "repeat_exact": True,
                "offline_execution_verified": True,
                "logits_reference_verified": False,
                "quality_verified": False,
                "serving_integration_verified": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
