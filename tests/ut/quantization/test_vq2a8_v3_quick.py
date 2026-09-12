# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import quick_benchmark_vq2a8_v3 as quick


def test_v3_quick_defaults_are_bounded_engine_timing(tmp_path):
    args = quick.parse_args(["--model", str(tmp_path)])
    assert args.model == tmp_path
    assert args.library.as_posix().endswith("build/vq2a8-ascendc-v3/libvq2a8_ascendc_v3.so")
    assert args.physical_npu == 0
    assert args.prompt_tokens == 10 and args.output_tokens == 32
    assert args.warmups == 1 and args.repeats == 3
    assert args.engine_memory_fraction == 0.98 and args.cache_memory_fraction == 1
    assert args.cache_reserve_gib == 3


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--physical-npu", "-1"),
        ("--prompt-tokens", "0"),
        ("--output-tokens", "1"),
        ("--warmups", "-1"),
        ("--repeats", "0"),
        ("--cache-reserve-gib", "0.99"),
        ("--cache-reserve-gib", "nan"),
        ("--cache-reserve-gib", "inf"),
        *[
            (flag, value)
            for flag in ("--engine-memory-fraction", "--cache-memory-fraction")
            for value in ("0", "-0.1", "1.01", "nan", "inf", "-inf")
        ],
    ],
)
def test_v3_quick_rejects_invalid_cli(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        quick.parse_args(["--model", str(tmp_path), f"{flag}={value}"])


def test_v3_quick_context_and_repeat_boundaries(tmp_path):
    args = quick.parse_args(
        ["--model", str(tmp_path), "--prompt-tokens", "126", "--output-tokens", "2", "--warmups", "0", "--repeats", "1"]
    )
    assert args.prompt_tokens + args.output_tokens == 128 and args.warmups == 0 and args.repeats == 1
    with pytest.raises(SystemExit):
        quick.parse_args(["--model", str(tmp_path), "--prompt-tokens", "127", "--output-tokens", "2"])


@pytest.mark.parametrize("engine_fraction,cache_fraction", [(0.98, 1.0), (1.0, 0.7)])
def test_v3_quick_build_options_keeps_engine_and_cache_separate(tmp_path, engine_fraction, cache_fraction):
    args = quick.parse_args(
        [
            "--model",
            str(tmp_path),
            "--engine-memory-fraction",
            str(engine_fraction),
            "--cache-memory-fraction",
            str(cache_fraction),
        ]
    )
    library = {"path": str(tmp_path / "v3.so"), "sha256": "a" * 64}
    calls = []

    def factory(*positional, **kwargs):
        calls.append((positional, kwargs))
        return {"gpu_memory_utilization": 0.9, "kv_cache_memory_bytes": 2**30}

    options = quick.build_options(args, library, factory)
    assert len(calls) == 1
    positional, kwargs = calls[0]
    assert positional == (args.model, args.model / "experts_vq_ascend_v2")
    assert kwargs["execution_policy"] == "ascendc_v3" and kwargs["root_linear_mode"] == "bf16"
    assert kwargs["cache_budget_gib"] == 0 and kwargs["cache_reserve_gib"] == 3
    assert kwargs["cache_memory_fraction"] == cache_fraction
    assert kwargs["ascendc_v3_library"] == library["path"] and kwargs["ascendc_v3_sha256"] == library["sha256"]
    assert "ascendc_library" not in kwargs
    assert options["gpu_memory_utilization"] == engine_fraction
    assert options["kv_cache_memory_bytes"] == 2**30
    assert options["max_model_len"] == options["max_num_batched_tokens"] == args.prompt_tokens + args.output_tokens


def request_output(tokens, *, request_id="quick-test", finished=False, sequences=1):
    return SimpleNamespace(
        request_id=request_id,
        outputs=[SimpleNamespace(token_ids=list(tokens)) for _ in range(sequences)],
        finished=finished,
    )


class ScriptedEngine:
    def __init__(self, steps, *, residual=False, initially_busy=False):
        self.steps = list(steps)
        self.residual = residual
        self.initially_busy = initially_busy
        self.added = []
        self.events = []
        self.now = 0.0
        self.index = 0

    def add_request(self, *args, **kwargs):
        self.added.append((args, kwargs))
        self.events.append("add_request")

    def step(self):
        self.events.append("step")
        if self.index >= len(self.steps):
            return []
        self.now, output = self.steps[self.index]
        self.index += 1
        return output

    def has_unfinished_requests(self):
        if not self.added:
            return self.initially_busy
        return self.index < len(self.steps) or self.residual

    def synchronize(self):
        self.events.append("sync")

    def clock(self):
        return self.now


def measure(engine, count=4):
    return quick.measure_request(
        engine,
        [10, 11],
        SimpleNamespace(temperature=0),
        "quick-test",
        count,
        synchronize=engine.synchronize,
        clock=engine.clock,
    )


def test_v3_quick_tpot_uses_first_last_token_not_bulk_e2e(capsys):
    engine = ScriptedEngine(
        [
            (1.0, [request_output([1])]),
            (1.2, [request_output([1, 2])]),
            (1.8, [request_output([1, 2, 3])]),
            (2.5, [request_output([1, 2, 3, 4], finished=True)]),
        ]
    )
    result = measure(engine)
    assert result["ttft_ms"] == pytest.approx(1000)
    assert result["tpot_ms"] == pytest.approx(500)
    assert result["e2e_s"] == pytest.approx(2.5)
    assert result["output_tokens"] == 4
    assert engine.events == ["sync", "add_request", "step", "step", "step", "step", "sync"]
    assert len(engine.added) == 1
    assert capsys.readouterr().out == ""


def test_v3_quick_allows_zero_token_progress_without_counting_it():
    engine = ScriptedEngine(
        [
            (0.2, []),
            (1.0, [request_output([1])]),
            (1.1, [request_output([1])]),
            (1.3, [request_output([1, 2], finished=True)]),
        ]
    )
    result = measure(engine, 2)
    assert result["ttft_ms"] == pytest.approx(1000)
    assert result["tpot_ms"] == pytest.approx(300)
    assert result["output_tokens"] == 2


def test_v3_quick_final_synchronization_does_not_inflate_tpot():
    engine = ScriptedEngine([(1.0, [request_output([1])]), (1.3, [request_output([1, 2], finished=True)])])

    def synchronize():
        engine.synchronize()
        if engine.added:
            engine.now += 0.5

    result = quick.measure_request(engine, [10], object(), "quick-test", 2, synchronize=synchronize, clock=engine.clock)
    assert result["tpot_ms"] == pytest.approx(300)
    assert result["e2e_s"] == pytest.approx(1.8)


@pytest.mark.parametrize(
    "steps,count",
    [
        ([(1.0, [request_output([1, 2], finished=True)])], 2),
        ([(1.0, [request_output([1])]), (1.3, [request_output([1, 2, 3], finished=True)])], 3),
        ([(1.0, [request_output([1])]), (1.3, [request_output([9, 2], finished=True)])], 2),
        ([(1.0, [request_output([1])]), (1.3, [request_output([], finished=True)])], 2),
        ([(1.0, [request_output([1], finished=True)])], 2),
        ([(1.0, [request_output([1])]), (1.3, [request_output([1, 2])])], 2),
        ([(1.0, [request_output([1], request_id="other", finished=True)])], 2),
        ([(1.0, [request_output([1], sequences=2, finished=True)])], 2),
        ([(1.0, [request_output([1]), request_output([1], request_id="other")])], 2),
        (
            [
                (1.0, [request_output([1])]),
                (1.3, [request_output([1, 2])]),
                (1.5, [request_output([1, 2, 3], finished=True)]),
            ],
            2,
        ),
    ],
)
def test_v3_quick_rejects_invalid_engine_output(steps, count):
    with pytest.raises((ValueError, RuntimeError)):
        measure(ScriptedEngine(steps), count)


def test_v3_quick_rejects_residual_work_and_preexisting_requests():
    steps = [(1.0, [request_output([1])]), (1.2, [request_output([1, 2], finished=True)])]
    for options in ({"residual": True}, {"initially_busy": True}):
        with pytest.raises((ValueError, RuntimeError)):
            measure(ScriptedEngine(steps, **options), 2)


def test_v3_quick_rejects_two_cumulative_results_from_one_step():
    engine = ScriptedEngine([(1.0, [request_output([1]), request_output([1, 2], finished=True)])])

    def advancing_clock():
        engine.now += 0.001
        return engine.now

    with pytest.raises(ValueError):
        quick.measure_request(
            engine, [10], object(), "quick-test", 2, synchronize=engine.synchronize, clock=advancing_clock
        )


def test_v3_quick_samples_exclude_warmups_and_print_outside_request(tmp_path, monkeypatch, capsys):
    args = quick.parse_args(["--model", str(tmp_path)])
    calls = []
    engine, params = object(), object()

    def fake_measure(actual_engine, prompt, actual_params, request_id, output_tokens, *, synchronize):
        assert actual_engine is engine and actual_params is params and prompt == [10, 11]
        assert output_tokens == args.output_tokens
        assert capsys.readouterr().out.splitlines()[-1].startswith("QUICK_V3_START=")
        result = {"ttft_ms": 100.0, "tpot_ms": float(len(calls)), "e2e_s": 1.0, "output_tokens": output_tokens}
        calls.append((request_id, result))
        return result

    monkeypatch.setattr(quick, "measure_request", fake_measure)
    samples = quick.run_samples(engine, [10, 11], params, args, lambda: None)
    assert [request_id for request_id, _ in calls] == [
        "quick-v3-warmup-0",
        "quick-v3-measured-0",
        "quick-v3-measured-1",
        "quick-v3-measured-2",
    ]
    assert samples == [result for _, result in calls[1:]]
    assert "QUICK_V3_SAMPLE=measured" in capsys.readouterr().out


def test_v3_quick_help_does_not_import_torch_or_npu():
    code = (
        "import runpy, sys; sys.modules['torch'] = None; sys.modules['torch_npu'] = None; "
        f"sys.argv = {[quick.__file__, '--help']!r}; runpy.run_path({quick.__file__!r}, run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", code], text=True, capture_output=True, check=True)
    assert "--model" in result.stdout and "--output-tokens" in result.stdout


def test_v3_quick_structure_keeps_engine_timing_free_of_diagnostic_work():
    source = Path(quick.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert {"LLM", "measure_request", "add_request", "step"} <= called
    assert any(isinstance(node, ast.Attribute) and node.attr == "llm_engine" for node in ast.walk(tree))
    assert not called & {
        "checked_model_preflight",
        "run_preflight",
        "run_sas_preflight",
        "diagnostic",
        "collect_profile",
        "capture_worker_trace",
        "grouped_projection_v3",
        "grouped_projection_out",
        "cube_control",
    }
    timing = ast.parse(inspect.getsource(quick.measure_request))
    for loop in (node for node in ast.walk(timing) if isinstance(node, (ast.For, ast.While))):
        calls = {
            node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(loop)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
        }
        assert not calls & {"synchronize", "print", "Event", "record", "elapsed_time", "item"}
