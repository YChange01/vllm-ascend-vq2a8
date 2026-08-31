# Ascend 950 VQ2A8 kernels

This directory is reserved for the CANN implementations of
`vq2a8_gate_up` and `vq2a8_down_reduce`. Their executable Python dispatch,
fallback, tensor schema, and golden reference are defined in:

- `vllm_ascend/quantization/vq2a8_ops.py`
- `vllm_ascend/quantization/vq2a8_format.py`
- `docs/source/developer_guide/vq2a8_ascend950.md`
- `tests/ut/quantization/test_vq2a8_ops.py`

Do not change the `vq2a8_ascend_tp_v1` serialized format in a device-only
kernel change. Introduce a new format version and keep the old loader when an
incompatible layout is required.
