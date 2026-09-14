# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contracts for reading startup reports, never starting a model."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import summarize_vq2a8_full_startup as summary


def _event(kind, span, parent=None, stage="model.forward", **extra):
    return {
        "event": kind,
        "span_id": span,
        "parent_id": parent,
        "seq": span,
        "stage": stage,
        "pid": 42,
        "time": "2026-09-14T01:02:03+00:00",
        "elapsed_s": 1.0,
        "mode": "sync",
        "device_completion": "not_verified",
        "main_thread_id": 33,
        **extra,
    }


def _write_events(directory, events, tail=""):
    path = directory / "events.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events) + tail, encoding="utf-8")
    return path


def test_latest_empty_report_does_not_fall_back_to_old_complete_report(tmp_path):
    old = tmp_path / "vq2-full-startup-old"
    new = tmp_path / "vq2-full-startup-new"
    old.mkdir()
    _write_events(old, [_event("PASS", 1)])
    new.mkdir()
    os.utime(old, (100, 100))
    os.utime(new, (200, 200))
    assert summary.select_report(root=tmp_path) == new.resolve()
    output = "\n".join(summary.summarize(new))
    assert "TRACE_NOT_READY" in output
    assert "no older report was substituted" in output
    assert "STACK_STATUS=MISSING" in output


def test_explicit_missing_report_is_not_replaced(tmp_path):
    (tmp_path / "vq2-full-startup-other").mkdir()
    with pytest.raises(FileNotFoundError):
        summary.select_report(tmp_path / "missing", root=tmp_path)
    with pytest.raises(FileNotFoundError):
        summary.select_report(root=tmp_path / "empty")


def test_partial_last_record_does_not_close_pending_span(tmp_path):
    path = _write_events(tmp_path, [_event("BEGIN", 1)], json.dumps(_event("PASS", 1)))
    events = summary.read_events(path)
    assert events["count"] == 1
    assert events["partial_tail"] is True
    assert 1 in events["active"]


def test_deepest_open_span_survives_completed_sibling(tmp_path):
    _write_events(
        tmp_path,
        [
            _event("BEGIN", 1),
            _event("BEGIN", 2, 1, "decoder.0.forward"),
            _event("BEGIN", 3, 2, "decoder.0.hc_pre.attention"),
            _event("PASS", 3, 2, "decoder.0.hc_pre.attention"),
            _event("BEGIN", 4, 2, "decoder.0.attention"),
            _event("SUBMITTED", 4, 2, "decoder.0.attention"),
        ],
    )
    output = summary.summarize(tmp_path)
    pending = next(line for line in output if line.startswith("DEEPEST_OPEN"))
    assert "depth=2" in pending
    assert "span=4" in pending
    assert "stage=decoder.0.attention" in pending
    assert any("call returned; synchronization has not completed" in line for line in output)


def test_terminal_failure_closes_span_and_error_is_one_bounded_line(tmp_path):
    _write_events(
        tmp_path,
        [
            _event("BEGIN", 1),
            _event("FAIL", 1, error_type="RuntimeError", error="failure\n" * 1000),
        ],
    )
    output = summary.summarize(tmp_path)
    assert "DEEPEST_OPEN=NONE" in output
    errors = [line for line in output if line.startswith("LAST_ERROR")]
    assert len(errors) == 1 and len(errors[0]) <= 500
    assert "\n" not in errors[0]


def test_deepest_failure_is_preserved_during_outer_unwind(tmp_path):
    _write_events(
        tmp_path,
        [
            _event("BEGIN", 1),
            _event("BEGIN", 2, 1, "decoder.0.forward"),
            _event("BEGIN", 3, 2, "moe.0.bind_stream"),
            _event("FAIL", 3, 2, "moe.0.bind_stream", error="inner failure"),
            _event("FAIL", 2, 1, "decoder.0.forward", error="inner failure"),
            _event("FAIL", 1, error="inner failure"),
        ],
    )
    output = summary.summarize(tmp_path)
    assert any(line.startswith("LAST_ERROR stage=model.forward") for line in output)
    assert any(line.startswith("DEEPEST_ERROR stage=moe.0.bind_stream") for line in output)
    assert "DEEPEST_OPEN=NONE" in output


def test_new_independent_failure_replaces_previous_deepest_failure(tmp_path):
    path = _write_events(
        tmp_path,
        [
            _event("BEGIN", 1),
            _event("BEGIN", 2, 1, "old.inner"),
            _event("FAIL", 2, 1, "old.inner"),
            _event("FAIL", 1),
            _event("BEGIN", 3, stage="new.forward"),
            _event("FAIL", 3, stage="new.forward"),
        ],
    )
    assert summary.read_events(path)["deepest_error"]["stage"] == "new.forward"


def test_ready_event_supplies_main_thread_without_model_spans(tmp_path):
    _write_events(tmp_path, [_event("READY", None, stage="startup.awaiting_worker_profile")])
    (tmp_path / "stacks.log").write_text(
        'Timeout (0:00:30)!\nThread 0x21 (most recent call first):\n  File "worker.py", line 12 in wait\n',
        encoding="utf-8",
    )
    lines = summary.summarize(tmp_path)
    assert "DEEPEST_OPEN=NONE (hooks ready; no model span yet)" in lines
    assert "STACK_STATUS=LATEST_MAIN_THREAD_DUMP" in lines
    assert any("worker.py" in line for line in lines)
    assert any("event=READY" in line for line in lines)
    assert not any("TRACE_NOT_READY" in line for line in lines)


def test_malformed_complete_events_are_ignored(tmp_path):
    path = _write_events(tmp_path, [_event("BEGIN", 1)])
    with path.open("a", encoding="utf-8") as stream:
        stream.write('not JSON\n{"event": []}\n[]\n')
        stream.write(json.dumps(_event("PASS", 1)) + "\n")
    events = summary.read_events(path)
    assert events["count"] == 2
    assert events["ignored"] == 3
    assert not events["active"]


def test_stack_matches_main_thread_id_not_last_thread(tmp_path):
    stacks = tmp_path / "stacks.log"
    stacks.write_text(
        "Timeout (0:00:30)!\nThread 0x00000021 (most recent call first):\n"
        '  File "main.py", line 11 in forward\n\n'
        "Thread 0x00000022 (most recent call first):\n"
        '  File "background.py", line 99 in wait\n',
        encoding="utf-8",
    )
    status, lines = summary.read_main_stack(stacks, 33)
    assert status == "LATEST_MAIN_THREAD_DUMP"
    assert any("main.py" in line for line in lines)
    assert not any("background.py" in line for line in lines)


def test_incomplete_latest_dump_does_not_reuse_old_main_stack(tmp_path):
    stacks = tmp_path / "stacks.log"
    stacks.write_text(
        "Timeout (0:00:30)!\nThread 0x21 (most recent call first):\n"
        '  File "old.py", line 1 in stale\n\n'
        "Timeout (0:00:30)!\nThread 0x22 (most recent call first):\n"
        '  File "other.py", line 2 in wait\n',
        encoding="utf-8",
    )
    status, lines = summary.read_main_stack(stacks, 33)
    assert status == "LATEST_DUMP_MAIN_THREAD_NOT_YET_PRESENT"
    assert lines == []
    assert summary.read_main_stack(stacks, None) == ("MAIN_THREAD_ID_UNKNOWN", [])


def test_output_is_bounded_and_does_not_expand_tensor_metadata(tmp_path):
    _write_events(
        tmp_path,
        [_event("BEGIN", span, metadata={"tensor": "DO_NOT_SHOW" * 1000}) for span in range(1, 101)],
    )
    (tmp_path / "stacks.log").write_text(
        "Timeout (0:00:30)!\nThread 0x21 (most recent call first):\n"
        + "".join(f'  File "model.py", line {index} in forward\n' for index in range(100)),
        encoding="utf-8",
    )
    lines = summary.summarize(tmp_path)
    assert len(lines) < 60
    assert sum(line.startswith("seq=") for line in lines) == 12
    assert "DO_NOT_SHOW" not in "\n".join(lines)
    assert len(summary.read_main_stack(tmp_path / "stacks.log", 33)[1]) == 24


def test_cli_is_read_only_and_imports_no_torch_or_vllm(tmp_path):
    _write_events(tmp_path, [_event("BEGIN", 1)])
    script = Path(summary.__file__).resolve()
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    code = (
        "import runpy,sys; "
        "sys.modules['torch']=None; sys.modules['torch_npu']=None; sys.modules['vllm']=None; "
        f"sys.argv=[{str(script)!r}, {str(tmp_path)!r}]; "
        f"runpy.run_path({str(script)!r}, run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "NPU_INITIALIZED=False" in result.stdout
    assert "DEEPEST_OPEN" in result.stdout
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
