# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU control/arithmetic tests; none certifies native graph execution."""

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from tools import validate_vq2a8_v4_graph as tool

REPO = Path(__file__).resolve().parents[3]


class CpuBank:
    """Deterministic CPU tensor oracle, never selected by the production CLI."""

    def __init__(self, *fields):
        self.metadata = [torch.stack(values) for values in fields[3:]]
        self.experts = len(fields[0])
        self.columns = fields[0][0].shape[0] * 2

    def select(self, ids):
        valid = (ids >= 0) & (ids < self.experts)
        safe = ids.clamp(0, self.experts - 1)
        result = [value.index_select(0, safe) for value in self.metadata]
        result = [
            torch.where(valid[:, None], value, 0 if index == 2 else float("nan")) for index, value in enumerate(result)
        ]
        return (*result, valid.int())

    def project(self, hidden, scale, bias, ids):
        valid = (ids >= 0) & (ids < self.experts)
        value = (hidden.float().sum(1) * scale + bias + ids.float())[:, None]
        output = value.expand(-1, self.columns).contiguous().bfloat16()
        return torch.where(valid[:, None], output, float("nan")), valid.int()


class CpuCapture:
    """Recomputes Python, deliberately unlike NPUGraph; tests orchestration only."""

    def __init__(self, compute, inputs, *, synchronize):
        self.compute = compute
        self.inputs = tuple(value.clone() for value in inputs)
        self.outputs = tuple(compute(*self.inputs))
        self.replays = 0
        self.closed = False
        self.synchronize = synchronize

    def replay(self, *inputs):
        for static, current in zip(self.inputs, inputs):
            static.copy_(current)
        self.outputs = tuple(self.compute(*self.inputs))
        self.replays += 1
        return tuple(value.clone() for value in self.outputs)

    def snapshot(self):
        return {
            "captures": 1,
            "replays": self.replays,
            "entries": int(not self.closed),
            "static_input_pointers": [value.data_ptr() for value in self.inputs],
        }

    def close(self):
        self.synchronize()
        self.closed = True


def test_graph_probe_defaults_and_child_command_are_small_and_isolated():
    args = tool.parse_args([])
    assert args.physical_npu == 1 and args.timeout_s == 300
    assert args.launch_blocking == "0" and not args.queue_lifetime
    assert args.queue_iterations == 2049
    command = tool.child_command(args)
    assert command[:2] == [sys.executable, "-u"] and "--child" in command
    for forbidden in ("--model", "--artifact", "--compare-v1", "--build", "--allow-busy"):
        assert forbidden not in command
    lifetime = tool.parse_args(["--queue-lifetime", "--queue-iterations", "8192"])
    assert tool.child_command(lifetime)[-3:] == ["--queue-lifetime", "--queue-iterations", "8192"]


@pytest.mark.parametrize(
    "args",
    [
        ["--physical-npu", "-1"],
        ["--timeout-s", "0"],
        ["--timeout-s", "3601"],
        ["--launch-blocking", "1"],
        ["--allow-busy"],
        ["--queue-iterations", "2048"],
        ["--queue-iterations", "8193"],
        ["--queue-iterations", "2050"],
        ["--child", "--plan-only"],
    ],
)
def test_probe_rejects_unsupported_inputs_before_execution(args):
    with pytest.raises(SystemExit) as error:
        tool.parse_args(args)
    assert error.value.code == 2


def test_plan_only_never_imports_torch_or_vllm_or_creates_reports(tmp_path):
    script = REPO / "tools/validate_vq2a8_v4_graph.py"
    target = tmp_path / "must-not-exist"
    code = (
        "import builtins,runpy,sys\n"
        "original=builtins.__import__\n"
        "def guarded(name,*args,**kwargs):\n"
        " if name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend'):\n"
        "  raise AssertionError('unexpected runtime import: '+name)\n"
        " return original(name,*args,**kwargs)\n"
        "builtins.__import__=guarded\n"
        f"sys.argv=[{str(script)!r},'--plan-only','--queue-lifetime','--report-dir',{str(target)!r}]\n"
        f"runpy.run_path({str(script)!r},run_name='__main__')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["physical_npu"] == 1 and report["queue_lifetime"]["iterations_per_phase"] == 2049
    for name in (
        "device_execution_verified",
        "graph_functional_verified",
        "full_model_graph_verified",
        "model_weights_loaded",
        "full_model_verified",
        "serving_verified",
        "performance_verified",
        "timing_valid",
    ):
        assert report[name] is False
    assert not target.exists()


def test_probe_environment_uses_existing_single_device_nonblocking_contract():
    args = tool.parse_args(["--physical-npu", "3"])
    original = {"RANK": "4", "WORLD_SIZE": "8", "ASCEND_RT_VISIBLE_DEVICES": "0,1", "OTHER": "keep"}
    actual = tool.child_environment(args, original)
    assert actual["ASCEND_RT_VISIBLE_DEVICES"] == "3" and actual["ASCEND_LAUNCH_BLOCKING"] == "0"
    assert "RANK" not in actual and "WORLD_SIZE" not in actual and actual["OTHER"] == "keep"
    assert original["RANK"] == "4"


def test_real_capture_rejects_cpu_even_when_a_fake_npu_backend_exists():
    with pytest.raises(ValueError, match="native NPU inputs"):
        tool.CapturedTuple(lambda value: (value,), (torch.zeros(1),), synchronize=lambda: None)


@pytest.mark.parametrize("k,routes", [(256, 1), (512, 2), (1024, 7)])
def test_synthetic_geometry_cannot_expand_into_model_allocations(k, routes):
    with pytest.raises(ValueError, match="bounded"):
        tool.dynamic_inputs(torch.device("cpu"), k, routes)


@pytest.mark.parametrize("routes", [1, 6])
def test_dynamic_cases_cover_same_hidden_ids_changes_invalid_recovery_and_nan(routes):
    cases = tool.dynamic_inputs(torch.device("cpu"), 512, routes)
    assert len(cases) == 11
    assert all(hidden.shape == (1, 512) and ids.shape == (routes,) for _, hidden, ids, _ in cases)
    assert cases[0][1] is cases[1][1] and cases[0][2] is cases[3][2]
    assert cases[0][2] is cases[2][2] is cases[4][2]
    assert [valid for _, _, _, valid in cases][-6:] == [False, True, False, True, False, True]
    assert cases[7][2][0] == 2**40


def test_all_select_pipeline_cases_run_cpu_oracles_without_hardware_claim(monkeypatch):
    monkeypatch.setattr(tool, "REDUCTIONS", (512,))
    rows = tool.run_synthetic_checks(
        torch.device("cpu"), bank_factory=CpuBank, synchronize=lambda: None, capture_factory=CpuCapture
    )
    assert len(rows) == 4
    assert {(row["routes"], row["kind"]) for row in rows} == {
        (1, "select"),
        (1, "pipeline"),
        (6, "select"),
        (6, "pipeline"),
    }
    for row in rows:
        assert row["graph"]["captures"] == 1 and row["graph"]["replays"] == 11
        assert row["same_static_addresses"] and row["previous_outputs_preserved"]
        assert all(check["bit_exact"] for check in row["checks"])
        assert "device_execution_verified" not in row


@pytest.mark.parametrize("corruption", ["stale_inputs", "validity", "alias_outputs", "recapture"])
def test_dynamic_graph_checks_reject_stale_inputs_flags_aliases_and_recapture(corruption):
    class Broken(CpuCapture):
        def replay(self, *inputs):
            if corruption == "stale_inputs":
                self.replays += 1
                return tuple(value.clone() for value in self.outputs)
            previous = self.outputs
            result = super().replay(*inputs)
            if corruption == "validity":
                result[-1].fill_(True)
            if corruption == "alias_outputs":
                for old, new in zip(previous, result):
                    old.copy_(new)
                self.outputs = previous
                return previous
            return result

        def snapshot(self):
            result = super().snapshot()
            if corruption == "recapture" and self.replays:
                result["captures"] += 1
            return result

    payloads = tool.synthetic_experts(512)
    bank = CpuBank(*([payload[key] for payload in payloads] for key in tool.PAYLOAD_FIELDS))
    spec = type("Spec", (), {"columns": 512, "rht_true_columns": 512, "rht_block_size": 128})()
    with pytest.raises(AssertionError):
        tool.exercise_graph(
            tool.pipeline_compute(bank, spec),
            tool.dynamic_inputs(torch.device("cpu"), 512, 1),
            synchronize=lambda: None,
            capture_factory=Broken,
        )


@pytest.mark.parametrize("iterations", [False, 0, 2048, 8193])
def test_lifetime_bounds_fail_before_any_allocation(iterations):
    with pytest.raises(ValueError, match="bounded"):
        tool.run_queue_lifetime_checks(
            torch.device("cpu"),
            bank_factory=None,
            projection=None,
            grouped_projection=None,
            grouped_projection_pipeline=None,
            synchronize=None,
            iterations=iterations,
        )


def test_lifetime_requires_real_eager_entries_before_any_graph_or_bank_creation():
    with pytest.raises(ValueError, match="no fallback"):
        tool.run_queue_lifetime_checks(
            torch.device("cpu"),
            bank_factory=None,
            projection=lambda: None,
            grouped_projection=None,
            grouped_projection_pipeline=None,
            synchronize=None,
        )


def test_lifetime_cpu_orchestration_has_no_loop_host_reads_or_fences(monkeypatch):
    # Run the minimum production iteration bound with tiny tensor operations,
    # not a 2049-iteration CPU simulation of native packed matrix arithmetic.
    active = {"loop": False, "calls": 0, "syncs": 0}

    @contextmanager
    def stage(name):
        active["loop"] = name != "graph_lifetime_setup"
        yield
        active["loop"] = False

    def sync():
        assert not active["loop"], "No explicit iteration fence is permitted"
        active["syncs"] += 1

    def projection(hidden, scale, bias, *payload):
        return torch.ones((hidden.shape[0], 64), dtype=torch.bfloat16)

    def grouped(jobs):
        active["calls"] += 1
        return [projection(*job) for job in jobs]

    def tiny_compute(bank, spec):
        def compute(hidden, ids):
            q = hidden.to(torch.float8_e4m3fn)
            one = torch.ones(1)
            out = torch.ones((1, 64), dtype=torch.bfloat16)
            return (one, one, one, one.int(), q, one, one, out, one.int(), torch.tensor(True))

        return compute

    for name in ("cpu", "tolist", "item", "__bool__"):
        original = getattr(torch.Tensor, name)

        def guarded(value, *args, _original=original, **kwargs):
            assert not active["loop"], "No device value may be read on the host in the lifetime loop"
            return _original(value, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, guarded)
    monkeypatch.setattr(tool, "pipeline_compute", tiny_compute)
    result = tool.run_queue_lifetime_checks(
        torch.device("cpu"),
        bank_factory=CpuBank,
        projection=projection,
        grouped_projection=grouped,
        grouped_projection_pipeline=grouped,
        synchronize=sync,
        stage=stage,
        capture_factory=CpuCapture,
        observe=lambda _: {},
    )
    assert result["graph"]["captures"] == 1 and result["graph"]["replays"] == 4098
    assert active["calls"] == 4098 and active["syncs"] == 4
    assert result["retained_output_samples"] == 4 and result["all_iteration_checks_passed"]
    assert result["explicit_per_iteration_synchronize"] is False
    assert result["replays_are_not_queue_slot_measurements"] is True
    assert result["performance_verified"] is False


def receipt():
    low = [
        {
            "k": k,
            "routes": routes,
            "kind": kind,
            "graph": {"captures": 1, "entries": 1, "replays": len(tool.DYNAMIC_CASE_NAMES)},
            "same_static_addresses": True,
            "previous_outputs_preserved": True,
            "checks": [
                {
                    "case": name,
                    "bit_exact": True,
                    "valid": name not in {"invalid_id", "invalid_int64_id"}
                    and not (kind == "pipeline" and name == "nan_hidden"),
                }
                for name in tool.DYNAMIC_CASE_NAMES
            ],
        }
        for k in tool.REDUCTIONS
        for routes in tool.ROUTE_COUNTS
        for kind in ("select", "pipeline")
    ]
    moe = []
    for hashed, top_k in ((True, 1), (True, 6), (False, 1)):
        names = ["a", "b", "a_again", "nan", "valid_after_nan"]
        if hashed:
            names[2:2] = ["same_hidden_token_b", "same_token_hidden_b"]
            names += ["invalid_token", "recover_token", "missing_slot", "recover_slot"]
        moe.append(
            {
                "hash_route": hashed,
                "top_k": top_k,
                "graph": {
                    "captures": 1,
                    "entries": 1,
                    "replays": len(names),
                    "stream_bridges": len(names),
                    "graph_payload_copy_bytes": 0,
                },
                "production_stream_bridge_verified": True,
                "checks": [
                    {"case": name, "bit_exact": True, "valid": name not in {"nan", "invalid_token", "missing_slot"}}
                    for name in names
                ],
            }
        )
    return {
        "status": "PASS",
        "exit_code": 0,
        "events": [
            {
                "case": tool.CASE,
                "event": "CASE_PASS",
                "scope": tool.SCOPE,
                "device_execution_verified": True,
                "graph_functional_verified": True,
                "checks": low,
                "moe_checks": moe,
                **{
                    key: False
                    for key in (
                        "model_weights_loaded",
                        "full_model_verified",
                        "full_model_graph_verified",
                        "serving_verified",
                        "performance_verified",
                        "timing_valid",
                    )
                },
            }
        ],
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "case",
        "no_native",
        "no_graph",
        "no_moe",
        "exit",
        "timeout",
        "duplicate",
        "geometry",
        "stale_replay",
        "bridges",
        "overclaim",
    ],
)
def test_receipt_rejects_nonmatching_or_incomplete_hardware_claim(mutation):
    result = receipt()
    event = result["events"][0]
    if mutation == "case":
        event["case"] = "different_probe"
    elif mutation == "no_native":
        event["device_execution_verified"] = False
    elif mutation == "no_graph":
        event["graph_functional_verified"] = False
    elif mutation == "no_moe":
        event["moe_checks"] = []
    elif mutation == "exit":
        result["exit_code"] = 1
    elif mutation == "timeout":
        result["status"] = "TIMEOUT"
    elif mutation == "geometry":
        event["checks"].pop()
    elif mutation == "stale_replay":
        event["checks"][0]["graph"]["replays"] = 0
    elif mutation == "bridges":
        event["moe_checks"][0]["graph"]["stream_bridges"] = 0
    elif mutation == "overclaim":
        event["full_model_verified"] = True
    else:
        result["events"] *= 2
    with pytest.raises(ValueError):
        tool.validate_receipt(result)


def test_matching_receipt_contains_no_model_or_performance_inference():
    actual = tool.validate_receipt(receipt())
    assert actual["graph_functional_verified"]
    assert actual["model_weights_loaded"] is False and actual["performance_verified"] is False


def test_requested_lifetime_cannot_pass_on_short_receipt():
    with pytest.raises(ValueError, match="lifetime"):
        tool.validate_receipt(receipt(), require_lifetime=True)


def test_moe_fixture_uses_original_packed_layout_and_bounded_sparse_lookup():
    runtime = tool.make_moe_runtime(torch.device("cpu"), bank_factory=CpuBank, hash_route=True, top_k=6)
    assert runtime.config.hidden_size == 512 and runtime.config.num_experts == 8
    assert runtime._device_route_banks["lookup"].tolist() == [0, -1, 1, -1, -1, 2, -1, 3]
    assert runtime._device_route_banks["gate_up"][1].rows == 1024
    assert runtime._device_route_banks["down"][1].rows == 512
    assert len(runtime._synthetic_payload_owners) == 8
    assert all(payload["packed_indices"].dtype == torch.int32 for payload in runtime._synthetic_payload_owners)
    with pytest.raises(ValueError, match="bounded"):
        tool.make_moe_runtime(torch.device("cpu"), bank_factory=CpuBank, hash_route=False, top_k=6)


def test_moe_probe_uses_production_entry_points_with_actual_cpu_tensor_comparison(monkeypatch):
    from vllm_ascend.quantization.vq2a8_v4_device_route import DeviceRouteDecodeState, DeviceRouteGraphCompute

    def prepare(state, runtime):
        state._test_compute = DeviceRouteGraphCompute(runtime)
        state._test_replays = 0

    def replay(state, runtime, hidden, ids):
        output, valid = state._test_compute(hidden, ids)
        state.retain(valid)
        state._test_replays += 1
        return output.clone()

    def snapshot(state):
        return {
            "captures": 1,
            "entries": 1,
            "replays": state._test_replays,
            "stream_bridges": state._test_replays,
            "graph_payload_copy_bytes": 0,
        }

    monkeypatch.setattr(DeviceRouteDecodeState, "prepare_graph", prepare)
    monkeypatch.setattr(DeviceRouteDecodeState, "forward_graph", replay)
    monkeypatch.setattr(DeviceRouteDecodeState, "graph_snapshot", snapshot)
    monkeypatch.setattr(DeviceRouteDecodeState, "close_graph", lambda state: None)
    rows = tool.run_moe_checks(torch.device("cpu"), bank_factory=CpuBank, synchronize=lambda: None)
    assert [(row["hash_route"], row["top_k"]) for row in rows] == [(True, 1), (True, 6), (False, 1)]
    for row in rows:
        assert row["shared_expert_arithmetic"] == "original_gate_up_swiglu_down"
        assert row["graph"]["stream_bridges"] == len(row["checks"])
        assert all(check["bit_exact"] for check in row["checks"])
        assert "device_execution_verified" not in row


@pytest.mark.parametrize("bad", ["output", "validity", "bridges", "memory", None])
def test_production_bridge_lifetime_is_checked_at_boundaries(monkeypatch, bad):
    active = {"loop": False}

    @contextmanager
    def stage(name):
        active["loop"] = True
        yield
        active["loop"] = False

    class State:
        replays = 0
        valid = None

        def graph_snapshot(self):
            return {
                "captures": 1,
                "entries": 1,
                "replays": self.replays,
                "stream_bridges": self.replays - int(bad == "bridges" and self.replays > 0),
            }

        def forward_graph(self, runtime, hidden, ids):
            self.replays += 1
            self.valid = torch.tensor(bad != "validity")
            return hidden + int(bad == "output")

    def sync():
        assert not active["loop"]

    for name in ("cpu", "item", "tolist", "__bool__"):
        original = getattr(torch.Tensor, name)

        def guarded(value, *args, _original=original, **kwargs):
            assert not active["loop"]
            return _original(value, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, guarded)
    reads = []

    def observe(device):
        reads.append(1)
        return {"allocated_bytes": tool.MEMORY_GROWTH_TOLERANCE_BYTES + 1 if bad == "memory" and len(reads) > 1 else 0}

    template = (torch.ones(1, 4), torch.zeros(1, dtype=torch.int64), torch.ones(1, 4))
    runtime = type("Runtime", (), {"device": torch.device("cpu")})()
    if bad is None:
        result = tool.exercise_moe_lifetime(
            State(), runtime, [template, template], synchronize=sync, iterations=2049, stage=stage, observe=observe
        )
        assert result["all_iteration_checks_passed"] and result["retained_output_samples"] == 2
        assert result["explicit_per_iteration_synchronize"] is False
    else:
        with pytest.raises(AssertionError):
            tool.exercise_moe_lifetime(
                State(), runtime, [template, template], synchronize=sync, iterations=2049, stage=stage, observe=observe
            )
