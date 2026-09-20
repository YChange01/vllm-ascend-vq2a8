# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded Ascend950 smoke tests for the installed VQ2A8 native extension.

These tests exercise synthetic selection, exact zero projection, asynchronous
owners and graph replay. They do not certify general projection numerics,
model correctness, queue saturation or serving performance. No tools module,
model checkpoint, tolerance relaxation or CPU implementation fallback is used.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.timeout(120)

EXPERTS = 3
OUTPUT_WIDTH = 4096
QUEUE_ITERATIONS = 8


@pytest.fixture(scope="module")
def native():
    pytest.importorskip("torch_npu", reason="VQ2A8 native smoke requires real torch_npu and Ascend950 hardware")
    if not torch.npu.is_available():
        pytest.skip("No available NPU; VQ2A8 device execution was not tested")
    device = torch.device("npu", torch.npu.current_device())
    name = torch.npu.get_device_name(device)
    if not name.startswith("Ascend950"):
        pytest.skip(f"VQ2A8 kernels require Ascend950, not {name}; device execution was not tested")

    import vllm_ascend
    from vllm_ascend.quantization.vq2a8_v4_v2 import load_v4_v2_library, require_v4_v2_features

    library = Path(vllm_ascend.__file__).resolve().parent / "libvq2a8_ascendc_v4_v2.so"
    if not library.is_file():
        pytest.skip(
            "Optional VQ2A8 extension is not installed. Build vllm-ascend with "
            "VLLM_ASCEND_BUILD_VQ2A8=1 on Ascend950; device execution was not tested"
        )
    # Missing optional build skips; a present but unloadable/wrong-ABI build fails.
    load_v4_v2_library()
    require_v4_v2_features()
    yield SimpleNamespace(device=device, bank_type=torch.classes.vq2a8_ascendc_v4_v2.ResidentBank)
    torch.npu.synchronize()


def _make_bank(native, width, *, book_byte=0):
    packed = torch.zeros((OUTPUT_WIDTH // 32, width // 16, 16, 8), dtype=torch.uint8, device=native.device)
    books = torch.full((width // 256, OUTPUT_WIDTH // 32, 32), book_byte, dtype=torch.uint8, device=native.device)
    # A nonidentity order also exercises the native byte-gather metadata.
    order = torch.arange(width, dtype=torch.int64).flip(0).to(native.device)
    columns = torch.arange(width)
    scales = [((columns % 17).float() - 8) / 16 + expert / 128 for expert in range(EXPERTS)]
    biases = [((columns % 13).float() - 6) / 8 - expert / 64 for expert in range(EXPERTS)]
    signs = [torch.where(columns % 2 == expert % 2, -1, 1).to(torch.int8) for expert in range(EXPERTS)]
    owners = (
        [packed] * EXPERTS,
        [books] * EXPERTS,
        [order] * EXPERTS,
        [value.to(native.device) for value in scales],
        [value.to(native.device) for value in biases],
        [value.to(native.device) for value in signs],
    )
    return native.bank_type(*owners), owners, (scales, biases, signs)


def _hidden(width, dtype, layout, groups=EXPERTS):
    rows = 1 if layout == "expanded" else groups
    columns = width + 32 if layout == "padded" else width
    owner = (((torch.arange(rows * columns).reshape(rows, columns) % 127).float() - 63) / 32).to(dtype)
    owner[:, 16 if layout == "padded" else 0] = 0.0
    owner[:, 17 if layout == "padded" else 1] = -0.0
    return owner


def _view(owner, width, layout, groups=EXPERTS):
    if layout == "expanded":
        return owner.expand(groups, width)
    if layout == "padded":
        return owner[:, 16 : 16 + width]
    return owner


def _select_sign_oracle(hidden, slots, metadata):
    """Independent CPU formula; never invokes native select/preparation ops."""
    scales, biases, signs = metadata
    groups, width = hidden.shape
    selected_scale = torch.full((groups, width), 0x7FC00000, dtype=torch.int32).view(torch.float32)
    selected_bias = selected_scale.clone()
    selected_sign = torch.zeros((groups, width), dtype=torch.int8)
    selected = torch.zeros(groups, dtype=torch.int32)
    for row, slot in enumerate(slots):
        if 0 <= slot < len(scales):
            selected_scale[row].copy_(scales[slot])
            selected_bias[row].copy_(biases[slot])
            selected_sign[row].copy_(signs[slot])
            selected[row] = 1
    source = hidden.float()
    valid = (
        (
            torch.isfinite(source)
            & torch.isfinite(selected_scale)
            & torch.isfinite(selected_bias)
            & (selected_sign.abs() == 1)
        )
        .all(dim=1)
        .to(torch.int32)
    )
    return source * selected_sign.float(), selected_scale, selected_bias, selected, valid


def _assert_bytes(actual, expected):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert got.is_contiguous()
        assert torch.equal(got.detach().cpu().view(torch.uint8), want.contiguous().view(torch.uint8))


@pytest.mark.parametrize("width", [2048, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["contiguous", "expanded", "padded"])
def test_select_sign_exact_bytes_and_invalid_slot_recovery(native, width, dtype, layout):
    bank, owners, metadata = _make_bank(native, width)
    cpu_owner = _hidden(width, dtype, layout)
    owner = cpu_owner.to(native.device)
    hidden = _view(owner, width, layout)
    cpu_hidden = _view(cpu_owner, width, layout)
    slots = torch.zeros(EXPERTS, dtype=torch.int64, device=native.device)
    before = tuple(value.clone() for values in metadata for value in values)
    for ids in ([0, 1, 2], [-1, EXPERTS, -(1 << 63)], [2, 0, 1]):
        slots.copy_(torch.tensor(ids, dtype=torch.int64))
        actual = bank.select_sign(hidden, slots)
        _assert_bytes(actual, _select_sign_oracle(cpu_hidden, ids, metadata))
    _assert_bytes([owner], [cpu_owner])
    _assert_bytes([value for values in owners[3:] for value in values], before)


@pytest.mark.parametrize("width", [2048, 4096])
@pytest.mark.parametrize("zero_input", [True, False])
def test_projection_exact_zero_cases_and_invalid_slot_recovery(native, width, zero_input):
    # 0x38 encodes exactly 1.0 in e4m3fn. Zero input with nonzero books,
    # or nonzero input with zero books, must produce exact +0 with zero bias.
    bank, owners, _ = _make_bank(native, width, book_byte=0x38 if zero_input else 0)
    activation = torch.full((EXPERTS, width), 0.0 if zero_input else 1.0).to(torch.float8_e4m3fn).to(native.device)
    scale = torch.ones(EXPERTS, dtype=torch.float32, device=native.device)
    bias = torch.zeros_like(scale)
    slots = torch.zeros(EXPERTS, dtype=torch.int64, device=native.device)
    for ids in ([0, 1, 2], [0, -1, EXPERTS], [2, 1, 0]):
        slots.copy_(torch.tensor(ids, dtype=torch.int64))
        output, status = bank.project_vectorized(activation, scale, bias, slots)
        expected_bits = torch.zeros((EXPERTS, OUTPUT_WIDTH), dtype=torch.int16)
        selected = torch.tensor([int(0 <= slot < EXPERTS) for slot in ids], dtype=torch.int32)
        for row, valid in enumerate(selected):
            if not valid:
                expected_bits[row].fill_(0x7FC0)
        _assert_bytes((output, status), (expected_bits.view(torch.bfloat16), selected))
    # Keep payload owners explicitly alive until the final native result fence.
    assert len(owners) == 6


def test_select_sign_graph_live_inputs_invalid_and_recovered(native):
    width, dtype = 2048, torch.bfloat16
    stream = torch.npu.Stream(device=native.device)
    graph = None
    try:
        with torch.npu.stream(stream):
            bank, owners, metadata = _make_bank(native, width)
            cpu_hidden = _hidden(width, dtype, "contiguous")
            hidden = cpu_hidden.to(native.device)
            slots = torch.tensor([0, 1, 2], dtype=torch.int64, device=native.device)
            for _ in range(2):
                bank.select_sign(hidden, slots)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            actual = bank.select_sign(hidden, slots)
        for offset, ids in enumerate(([0, 1, 2], [0, -1, 2], [2, 1, 0])):
            changed = (cpu_hidden.float() + offset).to(dtype)
            with torch.npu.stream(stream):
                hidden.copy_(changed)
                slots.copy_(torch.tensor(ids, dtype=torch.int64))
                graph.replay()
            torch.npu.synchronize()
            _assert_bytes(actual, _select_sign_oracle(changed, ids, metadata))
        assert len(owners) == 6
    finally:
        # Never reset/free capture pools while native queue work is pending.
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()


def test_bounded_async_owner_release_and_ordinary_operation(native):
    # This is a small lifetime/interoperation smoke, not queue-depth coverage.
    pending = []
    for index in range(QUEUE_ITERATIONS):
        bank, owners, metadata = _make_bank(native, 2048)
        cpu_hidden = (_hidden(2048, torch.float32, "contiguous") + index).contiguous()
        hidden = cpu_hidden.to(native.device)
        ids = [0, -1, 2] if index % 2 else [2, 1, 0]
        slots = torch.tensor(ids, dtype=torch.int64, device=native.device)
        result = bank.select_sign(hidden, slots)
        ordinary = hidden + 2.0
        pending.append((result, _select_sign_oracle(cpu_hidden, ids, metadata), ordinary, cpu_hidden + 2.0))
        # Queued native callbacks must retain bank metadata and input storage.
        del bank, owners, hidden, slots
    torch.npu.synchronize()
    for result, expected, ordinary, ordinary_expected in pending:
        _assert_bytes(result, expected)
        _assert_bytes((ordinary,), (ordinary_expected,))
