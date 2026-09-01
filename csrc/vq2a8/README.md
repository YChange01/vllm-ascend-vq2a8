# Ascend 950 VQ2A8 kernels

This directory contains the Ascend 950 direct-kernel implementations of
`vq2a8_gate_up` and `vq2a8_down_reduce`. The shared preparation path applies
the routed expert's RHT/sign transform, performs per-row FP8 E4M3 dynamic
quantization, and produces the bias correction. Both projection kernels then
consume packed uint4 indices and codebook vectors directly; they never create
a dense expert weight tensor.

Their Python dispatch, fallback, tensor schema, and golden reference are
defined in:

- `vllm_ascend/quantization/vq2a8_ops.py`
- `vllm_ascend/quantization/vq2a8_format.py`
- `docs/source/developer_guide/vq2a8_ascend950.md`
- `tests/ut/quantization/test_vq2a8_ops.py`

Do not change the `vq2a8_ascend_tp_v1` serialized format in a device-only
kernel change. Introduce a new format version and keep the old loader when an
incompatible layout is required.
