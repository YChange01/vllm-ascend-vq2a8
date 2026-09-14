# TP2 local projection contract

The V3 resident projection still uses the original V2 computation: N128,
AIC K1024, AIV K512, Mmad K256, and two pipeline slots. No V2 source or
V3 resident compute instruction was changed for TP2.

| Projection | Local N | Runtime packed K |
| --- | ---: | ---: |
| TP1 gate/up | 4096 | 4096 |
| TP1 down | 4096 | 2048 |
| TP2 gate/up | 2048 | 4096 |
| TP2 down | 4096 | 2048 |

The eager input ABI supports M1..32 and 1..6 jobs. The prepared descriptor out
ABI remains M1-only, 1..6 jobs, with nine int64 words per job:
`x, scale, bias, packed_zn, pair_lut, output, M, N, K`.
All jobs in a launch share N and K. Descriptor pointers are a trusted internal
workspace ABI, not an interface for untrusted file/RPC pointer values.

TP2 down has 1024 logical input columns. Its artifact can omit empty source
codebooks, but the runtime must extend the artifact to packed K2048 before
launch. Appended physical activation columns are zero; their weight scale and
bias are zero and RHT signs are +1. RHT/A8 precede the byte permutation.

K256 describes the LUT population, not a supported compute tail. The current
kernel drains both K1024 slots unconditionally: K1024 would leave the second
slot without an acknowledgement, and non-K1024 tails are not computed. Neither
case is accepted by the native binding. N2048 consists of sixteen complete
N128 tiles and needs no change to the arithmetic or pipeline.

Resident ABI version stays 1. Capability bit 4 (`RESIDENT_TP2_PROJECTION`) is
required for TP2; the new capability mask is 7. Old masks 1/3 remain accepted
for TP1 but are rejected for TP2. A matching newly compiled V3 library is
required; a Python-only update cannot grant an old library this capability.

## Validation boundary

The CPU tests check geometry, unchanged V2 arithmetic/pipeline source, work
coverage, address bounds and wrapper capability rejection. They do not compile
AscendC or establish device correctness or performance.

After building the library on an Ascend950/CANN host, run the device cases in a
fresh process with its absolute path and SHA256 supplied explicitly:

```bash
python - /absolute/path/libvq2a8_ascendc_v3.so 'REPLACE_WITH_LIBRARY_SHA256' <<'PY'
import sys
import pytest
import torch_npu
from vllm_ascend.quantization.vq2a8_ascendc_v3 import load_pinned_library

load_pinned_library(sys.argv[1], sys.argv[2])
raise SystemExit(pytest.main([
    "tests/ut/quantization/test_vq2a8_tp2_native.py", "-q", "-k", "device",
]))
PY
```

These local-kernel cases cover both TP2 shapes, M boundaries, mixed-M six-job
launches, repeated eager/out execution, a non-default stream, output guards and
raw-binding rejection of unsafe K values. Dedicated down cases scatter zero
dummy columns through both compute tiles; fused preparation checks dummy byte
gather and all-zero reuse of dirty outputs. Their small integer FP8 inputs allow
an exact CPU FP32/BF16 projection oracle. They are not two-card communication,
full-model numerical or TPOT acceptance tests; those remain separate checks.
