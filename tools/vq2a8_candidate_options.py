# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load host-only CLI contracts without executing the vLLM plugin initializer."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "vllm_ascend/quantization/vq2a8_abcd.py"
_SPEC = importlib.util.spec_from_file_location("vq2a8_abcd_cli_contract", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
add_candidate_arguments = _MODULE.add_candidate_arguments
validate_candidates = _MODULE.validate_candidates
validate_candidate_args = _MODULE.validate_candidate_args
