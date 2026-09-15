# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source/layout contracts only: these tests do not establish NPU correctness."""

from pathlib import Path

import pytest
import regex as re

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "csrc/vq2a8_ascendc_v4_v2"
V2_SOURCE = REPO / "csrc/vq2a8_ascendc_v2"


def function(source, name):
    match = re.search(rf"\b{re.escape(name)}\([^;{{}}]*\)\s*(?:const\s*)?\{{", source)
    assert match is not None, name
    start = match.start()
    begin = source.index("{", start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


@pytest.mark.parametrize("name", ["Init", "RunLut", "LoadVectorTile", "StoreB1", "Vector", "LoadA1", "LoadL0", "Cube"])
def test_computation_is_preserved_from_v2_not_the_v3_resident_port(name):
    baseline = (V2_SOURCE / "kernel.cpp").read_text()
    candidate = (SOURCE / "kernel.cpp").read_text()
    assert function(candidate, name) == function(baseline, name)
    assert '"resident_kernel.cpp"' not in candidate


def test_unique_library_namespace_and_all_cann_sources_are_built():
    cmake = (SOURCE / "CMakeLists.txt").read_text()
    assert "project(vq2a8_ascendc_v4_v2 LANGUAGES CXX)" in cmake
    assert "STATIC kernel.cpp resident_select.cpp resident_prepare.cpp" in cmake
    assert "SHARED torch_binding.cpp grouped_binding.cpp" in cmake
    for path in SOURCE.glob("*.cpp"):
        source = path.read_text()
        assert "namespace vq2a8_ascendc_v4_v2" in source
        assert "namespace vq2a8_ascendc_v3" not in source
        assert "SetCustomHandler" not in source


def test_bank_hotpath_has_no_host_descriptor_or_value_transfer():
    source = (SOURCE / "torch_binding.cpp").read_text()
    for name in ("Select", "Project", "CheckIds"):
        body = function(source, name)
        assert not any(
            token in body for token in (".cpu(", ".item(", ".to(", "at::kCPU", "synchronize", "at::sort", ".equal(")
        )
    project = function(source, "Project")
    assert "RunOpApi(" in project
    assert project.index("LaunchResidentPrepare(") < project.index("LaunchGrouped(")
    assert "RecordInputs({x, scale, bias, ids}" in project
    assert "[state, x, scale, bias, ids, reordered, descriptors, output, valid" in project
    assert "return {output, valid}" in project


def test_pointer_owners_and_permutation_are_validated_once_before_upload():
    source = (SOURCE / "torch_binding.cpp").read_text()
    constructor = function(source, "ResidentBank")
    assert "std::get<0>(at::sort(permutation)).equal(expectedOrder)" in constructor
    assert constructor.index("permutation of 0..K-1") < constructor.index("state->table = host.to(")
    assert constructor.count("host.to(") == 1
    for field in ("packed", "books", "order", "weight_scale", "weight_bias", "signs"):
        assert f"RecordInputs(state->{field}, stream)" in constructor
    assert "getCurrentNPUStream().stream() == state_->stream" in function(source, "CheckIds")
    assert "state_->table.nbytes()" in function(source, "Metadata")
    assert "constexpr uint32_t kBankWords = 8;" in (SOURCE / "resident_layout.h").read_text()


def test_invalid_routes_skip_on_both_core_kinds_before_any_indirect_load_or_flag():
    source = (SOURCE / "kernel.cpp").read_text()
    body = function(source, "Process")
    assert body.index("m_ = records.GetValue(base + kRows)") < body.index("if (m_ == 0) continue;")
    assert body.index("if (m_ == 0) continue;") < body.index("x_.SetGlobalBuffer")
    assert body.index("if (m_ == 0) continue;") < body.index("Cube()")
    layout = (SOURCE / "resident_layout.h").read_text()
    assert "slot >= 0 && static_cast<uint64_t>(slot) < experts" in layout
    for filename in ("resident_select.cpp", "resident_prepare.cpp"):
        body = function((SOURCE / filename).read_text(), "Process")
        assert "const int64_t expert" in body
        assert body.index("ValidResidentSlot(expert, experts_)") < body.index("static_cast<uint32_t>(expert)")
    prepare = (SOURCE / "resident_prepare.cpp").read_text()
    assert "Duplicate(outputUb_.Get<uint16_t>(), kInvalidBf16" in prepare
    assert "record.SetValue(field, uint64_t(0))" in prepare


def test_standalone_probe_is_separate_and_uses_safe_owner_callback():
    source = (SOURCE / "grouped_binding.cpp").read_text()
    assert "Standalone KERNEL PROBE ONLY" in source
    assert "RunOpApi(" in source
    assert "command.Run()" not in source
    assert "x, scale, bias, packed, table, output]" in source
    assert 'm.def("abi_version() -> int"' in source


@pytest.mark.parametrize(
    "routes,m,k", [(1, 1, 2048), (6, 1, 4096), (6, 15, 2048), (6, 16, 4096), (6, 17, 2048), (6, 32, 4096)]
)
def test_prepare_work_partition_has_no_missing_or_overlapping_route_row_chunks(routes, m, k):
    # Mirror only launch-domain/index arithmetic; this is not an AscendC emulator.
    n, columns = 4096, 256
    chunks = max(n, k) // columns
    blocks = min(64, routes * m * chunks)
    visits = []
    descriptors = []
    for core in range(blocks):
        for work in range(core, routes * m * chunks, blocks):
            route, row, column = work // (m * chunks), (work // chunks) % m, work % chunks * columns
            visits.append((route, row, column))
            if row == 0 and column == 0:
                descriptors.append(route)
    assert len(visits) == len(set(visits)) == routes * m * chunks
    assert sorted(descriptors) == list(range(routes))
    for route in range(routes):
        for row in range(m):
            assert sorted(column for r, mr, column in visits if r == route and mr == row and column < k) == list(
                range(0, k, columns)
            )


def test_valid_dimension_and_layout_contract_match_v2():
    original = (V2_SOURCE / "layout.h").read_text()
    candidate = (SOURCE / "layout.h").read_text()
    assert candidate.replace("vq2a8_ascendc_v4_v2", "vq2a8_ascendc_v2") == original
    assert re.search(r"kMaxM\s*=\s*32", candidate)
    assert re.search(r"kMaxJobs\s*=\s*6", candidate)
