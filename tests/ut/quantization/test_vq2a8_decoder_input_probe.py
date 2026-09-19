# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import deepcopy

import pytest

from tools import validate_vq2a8_decoder_input_plan as probe


def evidence(queue=True):
    values = {
        "numeric": probe.case_names(),
        "native_contract": probe.CONTRACT_CASES,
        "defensive_device_inputs": list(probe.DEFENSIVE_CASES),
    }
    if queue:
        values["queue"] = {
            "iterations": 513,
            "input_and_plan_owners_released": True,
            "per_iteration_synchronize": False,
            "runtime_queue_slots_measured": False,
        }
    return {
        "status": "PASS",
        "exit_code": 0,
        "reaped": True,
        "events": [
            {"event": "PASS", "stage": "final_sync"},
            {
                "event": "CASE_PASS",
                "case": probe.CASE,
                "native_abi": 1,
                "library": {"path": f"/tmp/{probe.LIBRARY_NAME}", "sha256": "a" * 64},
                "device_execution_verified": True,
                "model_integration_verified": False,
                "performance_verified": False,
                "results": values,
            },
        ],
    }


def test_input_probe_evidence_requires_exact_cases_queue_and_final_sync():
    valid = evidence()
    probe.validate_child_evidence(valid, True)
    for change in ("status", "exit", "reaped", "numeric", "native", "queue", "device", "abi", "sync", "fail"):
        bad = deepcopy(valid)
        final = bad["events"][-1]
        if change == "status":
            bad["status"] = "FAIL"
        elif change == "exit":
            bad["exit_code"] = 1
        elif change == "reaped":
            bad["reaped"] = False
        elif change == "numeric":
            final["results"]["numeric"].pop()
        elif change == "native":
            final["results"]["native_contract"] = 0
        elif change == "queue":
            final["results"]["queue"]["per_iteration_synchronize"] = True
        elif change == "device":
            final["device_execution_verified"] = False
        elif change == "abi":
            final["native_abi"] = True
        elif change == "sync":
            bad["events"].pop(0)
        else:
            bad["events"].insert(0, {"event": "FAIL"})
        with pytest.raises(ValueError):
            probe.validate_child_evidence(bad, True)


def test_input_probe_oracle_has_padding_offsets_and_int32_wrap():
    rows = probe.expected_owners(1, 1, 7, 0)
    # block_size2: table[3] = INT32_MAX, slot=INT32_MAX*2+1 -> -1.
    assert rows == [
        (
            [probe.SENTINEL, 2, 11, -2, 2147483647] + [probe.SENTINEL] * 5,
            [probe.SENTINEL] + [-1] * 5 + [probe.SENTINEL] * 4,
        )
    ]
    assert len(probe.case_names()) == len(set(probe.case_names())) == 48


def test_input_probe_without_queue_does_not_claim_queue_coverage():
    valid = evidence(False)
    probe.validate_child_evidence(valid, False)
    with pytest.raises(ValueError):
        probe.validate_child_evidence(valid, True)


def test_input_probe_plan_only_is_nonhardware(capsys):
    assert probe.main(["--library", "/tmp/libvq2a8_ascendc_v4_v2.so", "--plan-only"]) == 0
    value = capsys.readouterr().out
    assert '"status": "PLANNED"' in value
    assert '"device_execution_verified": false' in value


@pytest.mark.parametrize(
    "argv", [["--timeout-s", "0"], ["--physical-npu", "-1"], ["--child", "--plan-only"], ["--timeout-s", "7201"]]
)
def test_input_probe_rejects_invalid_control_arguments(argv):
    with pytest.raises(SystemExit):
        probe.parse_args(["--library", "x", *argv])
