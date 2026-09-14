# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts only; the separate synthetic CLI supplies NPU evidence."""

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import validate_vq2a8_v4_device_route as tool

REPO = Path(__file__).resolve().parents[3]


def test_device_route_gate_defaults_are_small_and_separate_from_v1():
    args = tool.parse_args([])
    assert args.physical_npu == 2 and args.timeout_s == 180
    assert args.launch_blocking == "0" and not args.allow_busy
    assert args.library == REPO / "build/vq2a8-ascendc-v4-device-route/libvq2a8_ascendc.so"
    assert not hasattr(args, "model") and not hasattr(args, "artifact")
    assert tool.EXPERTS == 4 and tool.OUTPUT_COLUMNS == 64 and tool.REDUCTIONS == (512, 1024)
    assert {len(routes) for routes in tool.VALID_ROUTES} == {1, 2, 6}
    assert any(len(set(routes)) == 1 and len(routes) == 6 for routes in tool.VALID_ROUTES)


@pytest.mark.parametrize(
    "argv",
    [
        ["--physical-npu", "-1"],
        ["--physical-npu", "0,1"],
        ["--timeout-s", "0"],
        ["--launch-blocking", "2"],
        ["--model", "/unused"],
        ["--artifact", "/unused"],
        ["--child", "--plan-only"],
    ],
)
def test_device_route_gate_rejects_ambiguous_or_unbounded_inputs(argv):
    with pytest.raises(SystemExit):
        tool.parse_args(argv)


def test_device_route_gate_plan_does_not_import_accelerator_or_vllm(tmp_path):
    script = REPO / "tools/validate_vq2a8_v4_device_route.py"
    code = (
        "import builtins,runpy,sys\n"
        "original=builtins.__import__\n"
        "def guarded(name,*args,**kwargs):\n"
        "    if name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend'):\n"
        "        raise AssertionError('plan imported runtime: '+name)\n"
        "    return original(name,*args,**kwargs)\n"
        "builtins.__import__=guarded\n"
        f"sys.argv=[{str(script)!r},'--plan-only','--report-dir',{str(tmp_path / 'must-not-exist')!r}]\n"
        f"runpy.run_path({str(script)!r},run_name='__main__')\n"
    )
    process = subprocess.run([sys.executable, "-c", code], cwd=REPO, text=True, capture_output=True, timeout=30)
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["physical_npu"] == 2
    for key in (
        "model_weights_loaded",
        "full_model_verified",
        "graph_verified",
        "performance_verified",
        "timing_valid",
    ):
        assert report[key] is False
    assert report["device_execution_verified"] is False
    assert "--child" in report["command"]
    assert not (tmp_path / "must-not-exist").exists()


def test_device_route_gate_child_selection_isolated_without_new_environment_switches():
    args = tool.parse_args(["--physical-npu", "5", "--launch-blocking", "1"])
    original = {"RANK": "8", "WORLD_SIZE": "16", "ASCEND_RT_VISIBLE_DEVICES": "0,1", "OTHER": "keep"}
    result = tool.child_environment(args, original)
    assert result["ASCEND_RT_VISIBLE_DEVICES"] == "5"
    assert result["ASCEND_LAUNCH_BLOCKING"] == "1"
    assert "RANK" not in result and "WORLD_SIZE" not in result
    assert result["OTHER"] == "keep" and original["RANK"] == "8"
    command = tool.child_command(args)
    assert command[:2] == [sys.executable, "-u"]
    assert command[command.index("--physical-npu") + 1] == "5"
    assert "--model" not in command and "--artifact" not in command


def test_library_identity_records_bytes_without_manifest_or_version_audit(tmp_path):
    library = tmp_path / "libvq2a8_ascendc.so"
    library.write_bytes(b"not-native-cpu-contract-only")
    # An old/incompatible manifest is not a new consistency gate.
    (tmp_path / "build-manifest.json").write_text("not json")
    evidence = tool.library_identity(library)
    assert evidence == {"path": str(library.resolve()), "sha256": hashlib.sha256(library.read_bytes()).hexdigest()}


def test_missing_resident_abi_requests_explicit_new_build_not_fallback():
    fake = SimpleNamespace(classes=SimpleNamespace(vq2a8_ascendc=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="ResidentBank ABI.*build/vq2a8-ascendc-v4-device-route.*fresh process"):
        tool.resident_bank_class(fake)
    constructor = object()
    fake.classes.vq2a8_ascendc.ResidentBank = constructor
    assert tool.resident_bank_class(fake) is constructor


@pytest.mark.parametrize("reduction", tool.REDUCTIONS)
def test_synthetic_resident_geometry_has_distinct_expert_metadata_and_tiles(reduction):
    payloads = tool.synthetic_experts(reduction)
    assert len(payloads) == 4
    assert [payload["codebooks"].shape[0] for payload in payloads] == [1, 3, 7, 4]
    for index, payload in enumerate(payloads):
        assert set(payload) == set(tool.PAYLOAD_FIELDS)
        assert all(value.device.type == "cpu" and value.is_contiguous() for value in payload.values())
        assert payload["packed_indices"].shape == (32, reduction // 8)
        assert payload["packed_indices"].dtype == torch.int32
        assert payload["codebooks"].shape[1:] == (2, 16, 2)
        assert payload["codebooks"].dtype == torch.float8_e4m3fn
        assert payload["codebook_tile_ids"].dtype == torch.uint8
        assert int(payload["codebook_tile_ids"].max()) < payload["codebooks"].shape[0]
        assert (
            payload["weight_scale"].shape == payload["weight_bias"].shape == payload["rht_sign"].shape == (reduction,)
        )
        assert payload["rht_sign"].dtype == torch.int8
        if index:
            assert not torch.equal(payload["weight_scale"], payloads[0]["weight_scale"])
            assert not torch.equal(payload["rht_sign"], payloads[0]["rht_sign"])


@pytest.mark.parametrize(
    "kwargs", [{"reduction": 256}, {"reduction": 512, "experts": 256}, {"reduction": 512, "columns": 4096}]
)
def test_fixture_cannot_accidentally_become_full_model_allocation(kwargs):
    with pytest.raises(ValueError, match="bounded"):
        tool.synthetic_experts(**kwargs)


def test_byte_comparison_does_not_hide_signed_zero_or_dtype_changes():
    with pytest.raises(AssertionError, match="bitwise"):
        tool.exact_tensor(torch.tensor([0.0]), torch.tensor([-0.0]), "zeros")
    with pytest.raises(AssertionError, match="shape/dtype"):
        tool.exact_tensor(torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.bfloat16), "dtype")


def cpu_contract_functions(corruption=None):
    """Real CPU arithmetic oracle; never used by the production CLI."""
    from tools.validate_vq2a8_phase4_kernel import same_fp8_oracle, synthetic_dense_oracle

    dense_cache = {}

    def projection(x, scale, bias, packed, books, tile_ids):
        key = tuple(value.data_ptr() for value in (packed, books, tile_ids))
        if key not in dense_cache:
            # Keep owners: later fixtures must not reuse a pointer with stale data.
            dense_cache[key] = ((packed, books, tile_ids), synthetic_dense_oracle(packed, books, tile_ids))
        return same_fp8_oracle((x, scale, bias), dense_cache[key][1])

    class Bank:
        def __init__(self, *fields):
            self.payloads = [dict(zip(tool.PAYLOAD_FIELDS, values)) for values in zip(*fields)]
            self.k = self.payloads[0]["weight_scale"].numel()
            self.n = self.payloads[0]["packed_indices"].shape[0] * 2

        def select(self, ids):
            rows = []
            for expert in ids.tolist():
                if 0 <= expert < len(self.payloads):
                    rows.append([self.payloads[expert][name] for name in tool.PAYLOAD_FIELDS[3:]])
                else:
                    rows.append(
                        [
                            torch.full((self.k,), float("nan")),
                            torch.full((self.k,), float("nan")),
                            torch.zeros(self.k, dtype=torch.int8),
                        ]
                    )
            result = [torch.stack([row[index] for row in rows]) for index in range(3)]
            if corruption == "metadata":
                result[0][0, 0] += 1
            valid = ((ids >= 0) & (ids < len(self.payloads))).int()
            return [*result, valid]

        def project(self, x, scale, bias, ids):
            rows = []
            for row, expert in enumerate(ids.tolist()):
                if 0 <= expert < len(self.payloads):
                    rows.append(
                        projection(
                            x[row : row + 1],
                            scale[row : row + 1],
                            bias[row : row + 1],
                            *(self.payloads[expert][name] for name in tool.PAYLOAD_FIELDS[:3]),
                        )
                    )
                else:
                    rows.append(torch.full((1, self.n), float("nan"), dtype=torch.bfloat16))
            output = torch.cat(rows)
            if corruption == "projection":
                output[0, 0] += 16
            valid = ((ids >= 0) & (ids < len(self.payloads))).int()
            if corruption == "invalid_flags":
                valid.fill_(1)
            return [output, valid]

    return Bank, projection


def test_complete_synthetic_route_suite_runs_real_cpu_oracles_without_hardware_claim():
    bank, projection = cpu_contract_functions()
    checks = tool.run_synthetic_checks(
        torch.device("cpu"), bank_factory=bank, projection=projection, synchronize=lambda: None
    )
    assert len(checks) == len(tool.REDUCTIONS) * (len(tool.VALID_ROUTES) + 1)
    assert all(check["exact"] for check in checks)
    assert sum(check.get("invalid_ids_safe", False) for check in checks) == 2
    assert not any("device_execution_verified" in check for check in checks)


@pytest.mark.parametrize("corruption", ["metadata", "projection", "invalid_flags"])
def test_synthetic_gate_rejects_wrong_native_outputs(corruption):
    bank, projection = cpu_contract_functions(corruption)
    with pytest.raises(AssertionError):
        tool.run_synthetic_checks(
            torch.device("cpu"), bank_factory=bank, projection=projection, synchronize=lambda: None
        )


def cpp_function(source, signature):
    """Extract one balanced C++ body for narrow native source contracts."""
    start = source.index(signature)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_v4_resident_preserves_original_v1_cube_and_vector_arithmetic():
    source = (REPO / "csrc/vq2a8_ascendc/kernel.cpp").read_text()
    # Existing V1 bodies before ProcessResident: arithmetic/masks/core ordering,
    # not generated binaries or a build/version acceptance gate.
    for name, digest in {
        "Vector": "413f086bfb7a1504edefb31ee8eef58b020eb02a63566d356551f669c861ef33",
        "Cube": "6665d2794a92d11486ce121ea6e4d274d80a751ccfc44638f5de7ada591e4342",
    }.items():
        body = cpp_function(source, f"__aicore__ inline void {name}(")
        assert hashlib.sha256(body.encode()).hexdigest() == digest, f"Original V1 {name} changed"


def test_resident_projection_bounds_before_any_pointer_read_and_cross_core_work():
    source = (REPO / "csrc/vq2a8_ascendc/kernel.cpp").read_text()
    body = cpp_function(source, "__aicore__ inline void ProcessResident(")
    assert "const int64_t expert = selected.GetValue(route);" in body
    assert body.index("ValidResidentSlot(expert, experts)") < body.index("if (!inRange)")
    assert body.index("if (!inRange)") < body.index("continue;") < body.index("records.GetValue(")
    assert body.index("continue;") < body.index("Cube();") < body.index("Vector<2>(group);")
    invalid = cpp_function(body, "if (!inRange)")
    assert "WriteResidentInvalid(output, route, group, n)" in invalid
    assert "continue;" in invalid
    assert "static_cast<uint32_t>(expert)" not in body[: body.index("continue;")]
    assert "n, k, records.GetValue(base + kBankTiles)" in body


def test_metadata_selector_only_gathers_three_v1_vectors_and_handles_invalid_ids():
    source = (REPO / "csrc/vq2a8_ascendc/resident_select.cpp").read_text()
    body = cpp_function(source, "__aicore__ inline void Process()")
    assert "const int64_t expert = routeIds_.GetValue(route);" in body
    assert body.index("ValidResidentSlot(expert, experts_)") < body.index("if (valid)") < body.index("bank_.GetValue(")
    for field in ("kBankScale", "kBankBias", "kBankSign"):
        assert f"bank_.GetValue(record + {field})" in body
    for forbidden in ("kBankPacked", "kBankBook", "kBankTileIds", "Mmad(", "Matmul("):
        assert forbidden not in body
    assert body.count("0x7fc00000") == 2
    assert "Duplicate(signUb_.Get<int16_t>(), int16_t(0)" in body
    assert "KERNEL_TYPE_AIV_ONLY" in source
    assert "FetchEventID(E)" in source


def test_resident_native_abi_owns_all_weights_and_has_no_per_call_host_staging():
    source = (REPO / "csrc/vq2a8_ascendc/resident_binding.cpp").read_text()
    assert 'm.class_<vq2a8_ascendc::ResidentBank>("ResidentBank")' in source
    assert "Tensors packed, books, tile_ids, weight_scale, weight_bias, signs;" in source
    assert "std::shared_ptr<ResidentBankState> state_" in source
    assert source.count("host.to(") == 1
    for name in ("Select", "Project"):
        body = cpp_function(source, f"Tensors {name}(")
        assert "const auto state = state_;" in body and "CheckIds(ids);" in body
        assert "[state," in body
        for forbidden in ("host.to(", "at::kCPU", ".cpu()", ".item", "synchronize(", "aclrtSynchronize"):
            assert forbidden not in body
    checker = cpp_function(source, "void CheckIds(")
    assert 'at::kLong, 1, "resident_ids", sizeof(int64_t)' in checker
    assert "getCurrentNPUStream().stream() == state_->stream" in checker
    cmake = (REPO / "csrc/vq2a8_ascendc/CMakeLists.txt").read_text()
    assert "STATIC kernel.cpp resident_select.cpp" in cmake
    assert "SHARED torch_binding.cpp resident_binding.cpp" in cmake


@pytest.mark.parametrize(
    ("method", "op_name", "captures"),
    [
        ("Select", "Vq2a8AscendCResidentSelect", "state, ids, scale, bias, sign, valid, routes, blocks"),
        ("Project", "Vq2a8AscendCResidentProjection", "state, x, scale, bias, ids, output, valid, routes, blocks"),
    ],
)
def test_resident_callback_uses_opapi_release_queue_without_dropping_tensor_owners(method, op_name, captures):
    # Source contract only. The NPU queue-lifetime probe exercises slot reuse;
    # a short, synchronized numeric test cannot reproduce this deadlock.
    source = (REPO / "csrc/vq2a8_ascendc/resident_binding.cpp").read_text()
    body = cpp_function(source, f"Tensors {method}(")
    normalized = " ".join(body.split())
    assert body.count("at_npu::native::OpCommand::RunOpApi(") == 1
    assert f'"{op_name}", [{captures}]() -> int' in normalized
    assert "}, false);" in normalized
    assert "SetCustomHandler" not in body and "command.Run(" not in body
    assert "RecordResidentInputs(" in body
    assert "return 0;" in body


@pytest.mark.parametrize(
    ("signature", "op_name"),
    [
        ("at::Tensor Run(", "Vq2a8AscendCProjection"),
        ("std::vector<at::Tensor> GroupedProjection(", "Vq2a8AscendCGroupedProjection"),
    ],
)
def test_v1_callbacks_share_resident_opapi_release_queue_without_forced_sync(signature, op_name):
    # A ResidentBank-only migration leaves the old V1 reference/prefill callback
    # in reusable producer slots. Mixing the paths must not restore that route.
    # This checks source dispatch, not live mutex ownership or device completion.
    source = (REPO / "csrc/vq2a8_ascendc/torch_binding.cpp").read_text()
    body = cpp_function(source, signature)
    normalized = " ".join(body.split())
    assert body.count("at_npu::native::OpCommand::RunOpApi(") == 1
    assert f'RunOpApi( "{op_name}",' in normalized
    assert "}, false);" in normalized
    assert "SetCustomHandler" not in body and "command.Run(" not in body
    assert "getCurrentNPUStream().stream()" in body
    assert "return 0;" in body and "return output;" in body
    for forbidden in ("aclrtSynchronize", "synchronize(", ".cpu()", ".item("):
        assert forbidden not in body


def test_standalone_native_sources_have_no_legacy_tensor_callback_submission():
    for path in (REPO / "csrc/vq2a8_ascendc").glob("*.cpp"):
        source = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.DOTALL)
        assert not re.search(r"\b\w+\s*\.\s*(?:SetCustomHandler|Run)\s*\(", source), path.name


def test_v1_opapi_migration_preserves_tensor_owners_and_original_launch_arguments():
    source = (REPO / "csrc/vq2a8_ascendc/torch_binding.cpp").read_text()
    run = " ".join(cpp_function(source, "at::Tensor Run(").split())
    grouped = " ".join(cpp_function(source, "std::vector<at::Tensor> GroupedProjection(").split())
    # The value capture retains actual tensors, including synthetic dense/scale/
    # bias temporaries; replacing them with raw pointers would defeat ownership.
    assert "[=]() -> int" in run
    assert (
        "Launch(stream, blocks, x.data_ptr(), scale.data_ptr(), bias.data_ptr(), "
        "packed.defined() ? packed.data_ptr() : nullptr, book.defined() ? book.data_ptr() : nullptr, "
        "ids.defined() ? ids.data_ptr() : nullptr, dense.defined() ? dense.data_ptr() : nullptr, "
        "output.data_ptr(), x.size(0), n, x.size(1), tiles, mode);"
    ) in run
    assert "[stream, blocks, descriptors, jobs, groups, x, scale, bias, packed, book, ids, output]() -> int" in grouped
    for owner in ("x", "scale", "bias", "packed", "book", "ids", "output"):
        assert f"(void){owner};" in grouped
    assert "auto descriptors = host.to(x[0].device(), at::kLong, false, true);" in grouped
    assert "if constexpr (Pipeline)" in grouped
    for launcher in ("LaunchGroupedPipeline", "LaunchGrouped"):
        assert f"{launcher}(stream, blocks, descriptors.data_ptr(), static_cast<uint32_t>(jobs), groups);" in grouped
    for name, variant in (("grouped_projection", "false"), ("grouped_projection_pipeline", "true")):
        assert f'm.impl("{name}", &vq2a8_ascendc::GroupedProjection<{variant}>);' in source


def test_resident_stream_ownership_records_all_payloads_only_at_construction():
    source = (REPO / "csrc/vq2a8_ascendc/resident_binding.cpp").read_text()
    record = cpp_function(source, "void RecordResidentInputs(")
    assert "NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream)" in record
    constructor = cpp_function(source, "ResidentBank(Tensors packed")
    for name in ("packed", "books", "tile_ids", "weight_scale", "weight_bias", "signs"):
        assert f"RecordResidentInputs(state->{name}, constructionStream)" in constructor
    select = cpp_function(source, "Tensors Select(")
    project = cpp_function(source, "Tensors Project(")
    assert "RecordResidentInputs({ids}, c10_npu::getCurrentNPUStream())" in select
    assert "RecordResidentInputs({x, scale, bias, ids}, c10_npu::getCurrentNPUStream())" in project
    for body in (select, project):
        assert body.count("RecordResidentInputs(") == 1
        assert "RecordResidentInputs(state->" not in body


def test_resident_id_bounds_helper_checks_entire_signed_int64_on_host(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("Host C++ compiler unavailable; not an Ascend kernel execution test")
    source = tmp_path / "resident_bounds.cpp"
    source.write_text(
        '#include "csrc/vq2a8_ascendc/resident_layout.h"\n'
        "#include <cassert>\n#include <cstdint>\n#include <limits>\n"
        "using vq2a8_ascendc::ValidResidentSlot;\n"
        "static_assert(!ValidResidentSlot(-1, 256));\n"
        "static_assert(!ValidResidentSlot(INT64_MIN, 256));\n"
        "static_assert(!ValidResidentSlot(INT64_MAX, 256));\n"
        "static_assert(!ValidResidentSlot(int64_t(1) << 40, 256));\n"
        "static_assert(!ValidResidentSlot(int64_t(1) << 32, 256));\n"
        "static_assert(!ValidResidentSlot(256, 256));\n"
        "static_assert(ValidResidentSlot(0, 1));\n"
        "static_assert(ValidResidentSlot(255, 256));\n"
        "int main() { for (uint32_t n=1; n<=256; ++n) {\n"
        "  assert(!ValidResidentSlot(-1,n)); assert(!ValidResidentSlot(n,n));\n"
        "  assert(ValidResidentSlot(n-1,n));\n"
        "} }\n"
    )
    executable = tmp_path / "resident_bounds"
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(REPO), str(source), "-o", str(executable)],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(executable)], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
