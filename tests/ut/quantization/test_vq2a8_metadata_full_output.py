# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks of the opt-in full-output gates; not hardware evidence."""

import ast
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools import validate_vq2a8_qli_metadata as qli
from tools import validate_vq2a8_sas_attention as sas

CASES = ((qli, None, 864), (sas, 64, 900), (sas, 128, 900))


def valid_words(module, heads):
    values = [0] * 1024
    if module is qli:
        values[:8] = [1, 0, 0, 0, 1, 0, 0, 0]
    else:
        row = [1, 0, 0, 0, 1, 0, 0, 0, int(heads == 128)]
        values[:9] = row
        if heads == 128:
            values[9:18] = row
    return values


def check(module, heads, words, **kwargs):
    if module is qli:
        return qli.check_metadata(words, **kwargs)
    return sas.check_sas_metadata(words, heads, **kwargs)


@pytest.mark.parametrize("module,heads,defined", CASES)
def test_zero_tail_passes_full_and_legacy_checks(module, heads, defined):
    words = valid_words(module, heads)
    assert defined == module.DEFINED_WORDS
    assert check(module, heads, words, require_full_output=True) == (2 if heads == 128 else 1)
    assert check(module, heads, words) == check(module, heads, words, require_full_output=False)


@pytest.mark.parametrize("module,heads,defined", CASES)
@pytest.mark.parametrize("offset", (0, 1, -1))
@pytest.mark.parametrize("poison", (1, -1, 0x7FFFFFFF))
def test_any_reserved_tail_poison_fails_only_full_mode(module, heads, defined, offset, poison):
    words = valid_words(module, heads)
    words[1023 if offset == -1 else defined + offset] = poison
    assert check(module, heads, words) in (1, 2)
    assert check(module, heads, words, require_full_output=False) in (1, 2)
    with pytest.raises(ValueError, match="reserved metadata tail must be zero"):
        check(module, heads, words, require_full_output=True)


@pytest.mark.parametrize("module,heads,defined", CASES)
@pytest.mark.parametrize("require_full_output", (False, True))
def test_defined_domain_and_shape_stay_strict(module, heads, defined, require_full_output):
    words = valid_words(module, heads)
    words[defined - 1] = 1
    with pytest.raises(ValueError):
        check(module, heads, words, require_full_output=require_full_output)
    for size in (defined, 1023, 1025):
        with pytest.raises(ValueError, match="1024"):
            check(module, heads, [0] * size, require_full_output=require_full_output)


@pytest.mark.parametrize("function", (qli.run_preflight, sas.run_sas_preflight))
def test_run_interface_keeps_existing_callers_compatible(function):
    signature = inspect.signature(function)
    option = signature.parameters["require_full_output"]
    assert option.default is False
    assert option.kind is inspect.Parameter.KEYWORD_ONLY
    signature.bind("npu:0", {}, 10)
    signature.bind("npu:0", {}, prompt_tokens=10)


@pytest.mark.parametrize(
    "module,function_name,current_name,values_name",
    (
        (qli, "run_preflight", "current", "values"),
        (sas, "run_sas_preflight", "current_meta", "words"),
    ),
)
def test_run_selects_all_words_and_reports_exact_comparison_scope(module, function_name, current_name, values_name):
    # Execute the real loop's small selection/receipt expressions without NPU
    # imports; do not duplicate the selection policy in a fake implementation.
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    selections = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == current_name for target in node.targets)
    ]
    assert len(selections) == 1
    record = next(
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "record" for target in node.targets)
    )
    fields = {keyword.arg: keyword.value for keyword in record.keywords}
    for full in (False, True):
        namespace = dict(vars(module), require_full_output=full, **{values_name: list(range(1024))})
        selected = eval(compile(ast.Expression(selections[0]), "<metadata-selection>", "eval"), namespace)
        assert selected == list(range(1024 if full else module.DEFINED_WORDS))
        receipt = {
            key: eval(compile(ast.Expression(fields[key]), "<metadata-receipt>", "eval"), namespace)
            for key in (
                "full_output_verified",
                "compared_metadata_words",
                "defined_metadata_words",
                "total_metadata_words",
            )
        }
        assert receipt == {
            "full_output_verified": full,
            "compared_metadata_words": 1024 if full else module.DEFINED_WORDS,
            "defined_metadata_words": module.DEFINED_WORDS,
            "total_metadata_words": 1024,
        }


@pytest.mark.parametrize("module,callees", ((qli, ("run_preflight",)), (sas, ("run_preflight", "run_sas_preflight"))))
def test_cli_forwards_full_output_to_every_preflight(module, callees):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    flag = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and any(isinstance(arg, ast.Constant) and arg.value == "--require-full-output" for arg in node.args)
    )
    assert any(keyword.arg == "action" and ast.literal_eval(keyword.value) == "store_true" for keyword in flag.keywords)
    for callee in callees:
        call = next(
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == callee
        )
        option = next(keyword for keyword in call.keywords if keyword.arg == "require_full_output")
        assert ast.unparse(option.value) == "args.require_full_output"


@pytest.mark.parametrize("full,poison", ((False, False), (False, True), (True, False), (True, True)))
def test_qli_actual_preflight_receipts_and_tail_failure(monkeypatch, tmp_path, capsys, full, poison):
    """Run the Python preflight itself with explicitly fake native execution."""

    class FakeTensor:
        dtype = "int32"
        device = "npu:0"

        def __init__(self, values):
            self.values = list(values)
            self.shape = (len(values),)

        def clone(self):
            return FakeTensor(self.values)

        def cpu(self):
            return self

        def tolist(self):
            return list(self.values)

    calls = []

    def native(**kwargs):
        calls.append(kwargs)
        values = valid_words(qli, None)
        if poison:
            # Vary every invocation, proving legacy comparisons still exclude
            # the undefined tail while the opt-in check rejects it immediately.
            values[-1] = len(calls)
        return FakeTensor(values)

    fake_torch = ModuleType("torch")
    fake_torch.int32 = "int32"
    fake_torch.tensor = lambda values, **kwargs: FakeTensor(values)
    fake_torch.npu = SimpleNamespace(synchronize=lambda: None)
    fake_torch.ops = SimpleNamespace(_C_ascend=SimpleNamespace(npu_vllm_quant_lightning_indexer_metadata=native))
    utils = ModuleType("vllm_ascend.utils")
    utils.bootstrap_custom_op_env = lambda: None
    extension = ModuleType("vllm_ascend.vllm_ascend_C")
    binary = tmp_path / "fake_extension.so"
    binary.write_bytes(b"CPU test fixture, not an NPU binary")
    extension.__file__ = str(binary)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, extension.__name__, extension)
    config = {"index_n_heads": 64, "index_head_dim": 128, "index_topk": 512}
    if full and poison:
        with pytest.raises(ValueError, match="reserved metadata tail"):
            qli.run_preflight("npu:0", config, require_full_output=True)
        assert len(calls) == 1
        assert "QLI_METADATA_PREFLIGHT=PASS" not in capsys.readouterr().out
    else:
        results = qli.run_preflight("npu:0", config, require_full_output=full)
        assert len(results) == 4 and len(calls) == 12
        assert all(record["full_output_verified"] is full for record in results)
        assert all(record["compared_metadata_words"] == (1024 if full else 864) for record in results)
        assert capsys.readouterr().out.count("QLI_RESULT ") == 4
