# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only warning/failure reporting contracts; no child/NPU execution."""

import json
import sys

import pytest

from tools import accept_vq2a8_optimizations as runner


def environment_log(path, **report):
    path.write_text(runner.ENVIRONMENT_PREFIX + json.dumps(report) + "\n", encoding="utf-8")
    return path


def test_reports_environment_errors_and_failed_pip_check_only(tmp_path):
    log = environment_log(
        tmp_path / "environment.log",
        status="failed",
        errors=["vllm==0.23.0 required, found 0.26.0.", "pip check failed."],
        packages={"private_package": "DO_NOT_DISPLAY_PACKAGE_VALUE"},
        environment={"TOKEN": "DO_NOT_DISPLAY_ENV_VALUE"},
        pip_check={"exit": 1, "output": "alpha 1.0 requires beta>=2, but beta 1.0 is installed.\n"},
    )
    causes = runner.failure_causes(log)
    assert causes == [
        "vllm==0.23.0 required, found 0.26.0.",
        "pip check failed.",
        "pip check: alpha 1.0 requires beta>=2, but beta 1.0 is installed.",
    ]
    assert "DO_NOT_DISPLAY" not in str(causes)


def test_does_not_display_successful_pip_output_or_unselected_fields(tmp_path):
    log = environment_log(
        tmp_path / "environment.log",
        errors=["Runtime import failed: missing ABI symbol"],
        pip_check={"exit": 0, "output": "DO_NOT_DISPLAY_SUCCESS_OUTPUT"},
        runtime_imports={"exception": "DO_NOT_DISPLAY_ARBITRARY_VALUE"},
    )
    assert runner.failure_causes(log) == ["Runtime import failed: missing ABI symbol"]


def test_new_pip_report_selects_blocking_issues_not_accepted_differences(tmp_path):
    log = environment_log(
        tmp_path / "environment.log",
        errors=["pip check has blocking dependency issues."],
        warnings=["APPROVED_DIFFERENCE"],
        pip_check={
            "exit": 1,
            "output": "APPROVED_DIFFERENCE\nDO_NOT_DISPLAY_RAW_OUTPUT",
            "accepted_issues": ["APPROVED_DIFFERENCE"],
            "blocking_issues": ["alpha requires beta>=2; TOKEN=SecretValue"],
        },
    )
    causes = runner.failure_causes(log)
    assert causes == [
        "pip check has blocking dependency issues.",
        "pip check: alpha requires beta>=2; TOKEN=[REDACTED]",
    ]
    assert "APPROVED_DIFFERENCE" not in str(causes)
    assert "DO_NOT_DISPLAY" not in str(causes)
    assert "SecretValue" not in str(causes)


@pytest.mark.parametrize("blocking", [[], None, "invalid", {}, [None, {}, ""]])
def test_present_pip_blocking_field_never_falls_back_to_raw_output(tmp_path, blocking):
    log = environment_log(
        tmp_path / "environment.log",
        errors=["Another gate failed."],
        pip_check={"exit": 1, "output": "APPROVED_DIFFERENCE", "blocking_issues": blocking},
    )
    assert runner.failure_causes(log) == ["Another gate failed."]


def test_new_pip_blocking_causes_are_bounded_and_sanitized(tmp_path):
    log = environment_log(
        tmp_path / "environment.log",
        pip_check={
            "exit": 1,
            "blocking_issues": ["https://private:SecretValue@proxy.local --token TokenValue " + "x" * 1000] * 12,
        },
    )
    causes = runner.failure_causes(log)
    assert len(causes) == 1
    assert len(causes[0]) <= runner.MAX_FAILURE_CAUSE_CHARS
    assert "SecretValue" not in causes[0] and "TokenValue" not in causes[0]


def test_environment_warnings_select_only_bounded_sanitized_strings(tmp_path):
    log = environment_log(
        tmp_path / "environment.log",
        warnings=[
            "Approved torch-npu difference; not NPU verification; TOKEN=SecretValue",
            "duplicate",
            "duplicate",
            None,
            {"token": "SecretObject"},
            "https://user:PasswordValue@proxy.local/path --password 'Cli Secret' " + "x" * 1000,
            "OUTSIDE_WARNING_LIMIT",
        ],
        errors=["DO_NOT_DISPLAY_ERROR"],
        packages={"package": "DO_NOT_DISPLAY_PACKAGE"},
        pip_check={"exit": 1, "output": "DO_NOT_DISPLAY_PIP_OUTPUT"},
    )
    warnings = runner.environment_warnings(log)
    assert len(warnings) == 3
    assert "not NPU verification" in warnings[0]
    assert warnings.count("duplicate") == 1
    assert all(len(warning) <= runner.MAX_FAILURE_CAUSE_CHARS for warning in warnings)
    assert all(
        value not in str(warnings)
        for value in ("SecretValue", "SecretObject", "PasswordValue", "Cli Secret", "OUTSIDE", "DO_NOT_DISPLAY")
    )


@pytest.mark.parametrize("warnings", [None, {}, "invalid", [None, {}, ""]])
def test_invalid_environment_warnings_are_not_forwarded(tmp_path, warnings):
    log = environment_log(tmp_path / "environment.log", warnings=warnings)
    assert runner.environment_warnings(log) == []


def test_missing_or_malformed_warning_log_has_no_raw_fallback(tmp_path):
    log = tmp_path / "environment.log"
    assert runner.environment_warnings(log) == []
    log.write_text('VQ2A8_V023_ENVIRONMENT {"warnings": ["RawSecret"\n', encoding="utf-8")
    assert runner.environment_warnings(log) == []


def test_latest_environment_warning_report_wins(tmp_path):
    log = environment_log(tmp_path / "environment.log", warnings=["earlier warning"])
    with log.open("a", encoding="utf-8") as out:
        out.write(runner.ENVIRONMENT_PREFIX + json.dumps({"warnings": ["latest warning"]}) + "\n")
    assert runner.environment_warnings(log) == ["latest warning"]


@pytest.mark.parametrize(
    ("text", "secrets"),
    [
        ("failed https://user:PasswordValue@proxy.local/path?token=QuerySecret", ["PasswordValue", "QuerySecret"]),
        ("HTTPS_PROXY=http://proxy.local:1234", ["proxy.local"]),
        ("http_proxy=ProxySecret", ["ProxySecret"]),
        ("password=PasswordValue", ["PasswordValue"]),
        ("passwd: PasswordValue", ["PasswordValue"]),
        ("'token': 'Token Secret Value'", ["Token Secret Value"]),
        ('"api_key": "ApiSecretValue"', ["ApiSecretValue"]),
        ("AWS_SECRET_ACCESS_KEY=AwsSecretValue", ["AwsSecretValue"]),
        ("--access-token CliSecretValue", ["CliSecretValue"]),
        ("--password 'Cli Secret Value'", ["Cli Secret Value"]),
        ("Authorization: Bearer BearerSecretValue", ["BearerSecretValue"]),
        ("Authorization=Basic BasicSecretValue==", ["BasicSecretValue"]),
        ("error hf_FakeTokenValue123", ["hf_FakeTokenValue123"]),
        ("error sk-proj-FakeTokenValue123", ["sk-proj-FakeTokenValue123"]),
        ("error ghp_FakeTokenValue123", ["ghp_FakeTokenValue123"]),
        ("error github_pat_FakeTokenValue123", ["github_pat_FakeTokenValue123"]),
    ],
)
def test_selected_errors_redact_common_credentials(tmp_path, text, secrets):
    log = environment_log(tmp_path / "environment.log", errors=[text], pip_check={"exit": 1, "output": text})
    result = " ".join(runner.failure_causes(log))
    assert "REDACTED" in result
    assert all(secret not in result for secret in secrets)


@pytest.mark.parametrize(
    "text",
    [
        "Traceback: DO_NOT_DISPLAY_RAW_TEXT\nRuntimeError: password=DoNotDisplayEither\n",
        'VQ2A8_V023_ENVIRONMENT {"errors": ["MalformedSecret"\n',
        "VQ2A8_V023_ENVIRONMENT []\n",
        'VQ2A8_V023_ENVIRONMENT {"errors": {"token": "Secret"}}\n',
        'VQ2A8_V023_ENVIRONMENT {"pip_check": {"exit": "1", "output": "Secret"}}\n',
        'VQ2A8_V026_ENVIRONMENT {"errors": ["WrongVersionSecret"]}\n',
        "\xff\x00unstructured invalid content\n",
    ],
)
def test_malformed_or_unrecognized_logs_use_fixed_safe_fallback(tmp_path, text):
    log = tmp_path / "failed.log"
    log.write_bytes(text.encode("latin-1"))
    assert runner.failure_causes(log) == [
        "No recognized environment cause in the bounded log tail; inspect the full log."
    ]


def test_missing_log_and_timeout_have_safe_explicit_reasons(tmp_path):
    causes = runner.failure_causes(tmp_path / "missing.log", timed_out=True)
    assert causes == ["Child exceeded its stage timeout.", "Child log is unavailable; no diagnostic text was read."]


def test_latest_structured_report_wins_without_forwarding_earlier_errors(tmp_path):
    log = environment_log(tmp_path / "environment.log", errors=["earlier wrong cause"])
    with log.open("a", encoding="utf-8") as out:
        out.write(runner.ENVIRONMENT_PREFIX + json.dumps({"errors": ["latest actual cause"]}) + "\n")
    assert runner.failure_causes(log) == ["latest actual cause"]


def test_output_is_bounded_deduplicated_and_single_line(tmp_path):
    log = environment_log(
        tmp_path / "environment.log",
        errors=["\x1b[31m版本错误\x1b[0m\nTOKEN=HiddenValue\r\nnext", "duplicate", "duplicate"]
        + [f"error-{i}: " + "x" * 1000 for i in range(12)],
        pip_check={"exit": 1, "output": "\n".join("dependency " + "x" * 1000 for _ in range(12))},
    )
    causes = runner.failure_causes(log)
    assert len(causes) <= runner.MAX_FAILURE_CAUSES
    assert all(len(cause) <= runner.MAX_FAILURE_CAUSE_CHARS for cause in causes)
    assert all("\n" not in cause and "\r" not in cause and "\x1b" not in cause for cause in causes)
    assert "HiddenValue" not in str(causes)
    assert causes.count("duplicate") == 1
    assert "版本错误" in causes[0]


def test_log_tail_and_individual_line_limits_reject_unbounded_payload(tmp_path):
    log = environment_log(tmp_path / "environment.log", errors=["old error outside tail"])
    with log.open("a", encoding="utf-8") as out:
        out.write("x" * (runner.MAX_FAILURE_LOG_BYTES + 1) + "\n")
        out.write(runner.ENVIRONMENT_PREFIX + json.dumps({"errors": ["a" * runner.MAX_FAILURE_LINE_BYTES]}) + "\n")
    assert runner.failure_causes(log) == [
        "No recognized environment cause in the bounded log tail; inspect the full log."
    ]


def _main_arguments(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accept_vq2a8_optimizations.py",
            "--model",
            str(tmp_path / "unused-model"),
            "--library",
            str(tmp_path / "unused.so"),
            "--output-dir",
            str(tmp_path / "report"),
        ],
    )
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")


@pytest.mark.parametrize("failed_stage", ["environment", "preflight", "performance"])
@pytest.mark.parametrize("timed_out", [False, True])
def test_failed_stage_prints_safe_cause_and_still_stops_immediately(
    monkeypatch, tmp_path, capsys, failed_stage, timed_out
):
    from tools import accept_vq2a8_release as release

    _main_arguments(monkeypatch, tmp_path)
    calls = []

    def supervise(_command, log, _env, _timeout):
        name = log.stem
        calls.append(name)
        failing = name == failed_stage
        environment_log(log, errors=["torch version mismatch; TOKEN=SecretValue"] if failing else [])
        return {"exit": 0 if timed_out or not failing else 3, "timeout": timed_out and failing, "log": str(log)}

    monkeypatch.setattr(release, "supervise", supervise)
    assert runner.main() == 1
    order = ["environment", "preflight", "performance"]
    assert calls == order[: order.index(failed_stage) + 1]
    stdout = capsys.readouterr().out
    assert "torch version mismatch" in stdout and "SecretValue" not in stdout
    assert "OPTIMIZATION_STATUS=FAIL" in stdout
    report = json.loads((tmp_path / "report/run.json").read_text())
    assert report["status"] == "FAIL" and not report["device_execution_verified"]
    assert report["stages"][-1]["failure_causes"]
    assert "SecretValue" not in json.dumps(report)
    if timed_out:
        assert "stage timeout" in stdout


def test_success_does_not_read_or_print_failure_logs(monkeypatch, tmp_path):
    from tools import accept_vq2a8_release as release

    _main_arguments(monkeypatch, tmp_path)
    calls = []

    def supervise(_command, log, _env, _timeout):
        calls.append(log.stem)
        if log.stem == "performance":
            result = log.parent / "result"
            result.mkdir()
            (result / "summary.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        return {"exit": 0, "timeout": False, "log": str(log)}

    monkeypatch.setattr(release, "supervise", supervise)
    monkeypatch.setattr(runner, "failure_causes", lambda *a, **k: pytest.fail("success must not read failure logs"))
    assert runner.main() == 0
    assert calls == ["environment", "preflight", "performance"]
    report = json.loads((tmp_path / "report/run.json").read_text())
    assert report["status"] == "PASS"
    assert all("failure_causes" not in stage for stage in report["stages"])


@pytest.mark.parametrize("preflight_fails", [False, True])
def test_successful_environment_prints_and_persists_warnings_without_verifying_device(
    monkeypatch, tmp_path, capsys, preflight_fails
):
    from tools import accept_vq2a8_release as release

    _main_arguments(monkeypatch, tmp_path)
    calls = []

    def supervise(_command, log, _env, _timeout):
        calls.append(log.stem)
        if log.stem == "environment":
            environment_log(
                log,
                warnings=["Approved torch-npu 2.10.0.post4.dev20260715 difference; no NPU proof; TOKEN=SecretValue"],
                pip_check={"exit": 1, "output": "DO_NOT_DISPLAY_ACCEPTED_PIP", "blocking_issues": []},
            )
        else:
            environment_log(log, warnings=["DO_NOT_DISPLAY_NON_ENVIRONMENT_WARNING"])
        if log.stem == "performance":
            result = log.parent / "result"
            result.mkdir()
            (result / "summary.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
        return {"exit": int(preflight_fails and log.stem == "preflight"), "timeout": False, "log": str(log)}

    monkeypatch.setattr(release, "supervise", supervise)
    assert runner.main() == (1 if preflight_fails else 0)
    assert calls == (["environment", "preflight"] if preflight_fails else ["environment", "preflight", "performance"])
    stdout = capsys.readouterr().out
    assert stdout.count("OPTIMIZATION_WARNING=") == 1
    assert "torch-npu 2.10.0.post4.dev20260715" in stdout
    assert "no NPU proof" in stdout and "SecretValue" not in stdout
    assert "DO_NOT_DISPLAY" not in stdout
    report = json.loads((tmp_path / "report/run.json").read_text())
    assert not report["device_execution_verified"]
    assert report["stages"][0]["warnings"] == [
        "Approved torch-npu 2.10.0.post4.dev20260715 difference; no NPU proof; TOKEN=[REDACTED]"
    ]
    assert "SecretValue" not in json.dumps(report) and "DO_NOT_DISPLAY" not in json.dumps(report)
    assert all("warnings" not in stage for stage in report["stages"][1:])


@pytest.mark.parametrize("timed_out", [False, True])
def test_approved_pip_issue_does_not_bypass_failed_child_or_become_failure_cause(
    monkeypatch, tmp_path, capsys, timed_out
):
    from tools import accept_vq2a8_release as release

    _main_arguments(monkeypatch, tmp_path)
    calls = []

    def supervise(_command, log, _env, _timeout):
        calls.append(log.stem)
        environment_log(
            log,
            warnings=["APPROVED_DIFFERENCE"],
            pip_check={"exit": 1, "output": "APPROVED_DIFFERENCE", "blocking_issues": []},
        )
        return {"exit": 0 if timed_out else 1, "timeout": timed_out, "log": str(log)}

    monkeypatch.setattr(release, "supervise", supervise)
    assert runner.main() == 1
    assert calls == ["environment"]
    stdout = capsys.readouterr().out
    assert "OPTIMIZATION_WARNING=" not in stdout
    assert "APPROVED_DIFFERENCE" not in stdout
    assert "OPTIMIZATION_STATUS=FAIL" in stdout
    report = json.loads((tmp_path / "report/run.json").read_text())
    assert not report["device_execution_verified"]
    assert "APPROVED_DIFFERENCE" not in json.dumps(report)
    assert "warnings" not in report["stages"][0]
