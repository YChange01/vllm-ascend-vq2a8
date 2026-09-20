# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU protocol tests, not a substitute for NPU graph/serving acceptance."""

import ast
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
GRAPH_PATH = ROOT / "vllm_ascend/quantization/vq2a8_v4_decoder_graph.py"
MODEL_PATH = ROOT / "vllm_ascend/patch/worker/vq2a8_offline_model.py"


def _load_graph():
    name = "vq2a8_clean_model_graph_under_test"
    spec = importlib.util.spec_from_file_location(name, GRAPH_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GRAPH = _load_graph()


def _metadata(position=2):
    entry = SimpleNamespace(
        num_decodes=1,
        num_decode_tokens=1,
        num_actual_tokens=1,
        num_prefills=0,
        prefill=None,
        decode=SimpleNamespace(seq_lens_list=[position + 1]),
    )
    return {"layer": entry}


def _model(context):
    """Exercise the real adapter class with a CPU-only inherited model stub."""

    class BaseModel:
        def forward(self, *args):
            self.eager_calls += 1
            return torch.ones((1, 4), dtype=torch.bfloat16)

        def compute_logits(self, hidden):
            return self.logits

    tree = ast.parse(MODEL_PATH.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VQ2A8TP1OfflineForCausalLM"
    )
    namespace = {
        "torch": torch,
        "AscendDeepseekV4ForCausalLM": BaseModel,
        "OfflineDecoderModel": object,
        "get_forward_context": lambda: context,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(MODEL_PATH), "exec"), namespace)
    adapter = namespace[cls.name]
    model = object.__new__(adapter)
    model.model = SimpleNamespace(
        offline_owner=SimpleNamespace(
            layers={0: SimpleNamespace(_optimization=SimpleNamespace(valid=torch.tensor(True)))}
        )
    )
    model._offline_loaded = True
    model._v4_graphs_ready = True
    model._v4_graphs_failed = False
    model._v4_graph_forward_active = False
    model._v4_decoder_preparing = False
    model._v4_graph_enabled = True
    model._measurement_valid = None
    model._last_forward_is_request = False
    model.eager_calls = 0
    model.logits = torch.ones((1, 8), dtype=torch.bfloat16)
    return model


def test_one_token_prefill_is_eager_not_shape_selected_decode():
    context = SimpleNamespace(vq2a8_request_phase="prefill")
    model = _model(context)
    model._v4_decoder_graph = SimpleNamespace(replay=lambda *args: pytest.fail("Prefill entered decoder replay"))
    hidden = model.forward(torch.tensor([7]), torch.tensor([0]))
    assert model.eager_calls == 1
    assert model.compute_logits(hidden).shape == (1, 8)


def test_decode_uses_graph_and_rejects_invalid_device_flag_before_tokens():
    context = SimpleNamespace(vq2a8_request_phase="decode")
    model = _model(context)
    model._v4_decoder_graph = SimpleNamespace(
        replay=lambda *args: (torch.ones((1, 4), dtype=torch.bfloat16), torch.tensor(False))
    )
    hidden = model.forward(torch.tensor([7]), torch.tensor([2]))
    assert model.eager_calls == 0
    with pytest.raises(ValueError, match="no output tokens"):
        model.compute_logits(hidden)
    assert model._v4_graphs_failed
    with pytest.raises(RuntimeError, match="failed"):
        model.forward(torch.tensor([7]), torch.tensor([3]))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_logits_are_rejected(invalid):
    model = _model(SimpleNamespace(vq2a8_request_phase="prefill"))
    hidden = model.forward(torch.tensor([1]), torch.tensor([0]))
    model.logits[0, 0] = invalid
    with pytest.raises(ValueError, match="validity checks"):
        model.compute_logits(hidden)


def test_missing_layer_validity_is_not_accepted_as_finite_logits():
    model = _model(SimpleNamespace(vq2a8_request_phase="prefill"))
    hidden = model.forward(torch.tensor([1]), torch.tensor([0]))
    model.model.offline_owner.layers[0]._optimization.valid = None
    with pytest.raises(ValueError, match="missing its device validity"):
        model.compute_logits(hidden)


def test_request_cannot_lazily_capture_or_fall_back_before_ready():
    model = _model(SimpleNamespace(vq2a8_request_phase="prefill"))
    model._v4_graphs_ready = False
    with pytest.raises(RuntimeError, match="lazy capture"):
        model.forward(torch.tensor([1]), torch.tensor([0]))
    assert model.eager_calls == 0


def test_capture_cannot_reuse_stale_layer_validity():
    model = _model(SimpleNamespace())
    layer = model.model.offline_owner.layers[0]
    layer._v4_decoder_valid = torch.tensor(True)
    with pytest.raises(RuntimeError, match="every resident MoE layer"):
        model._v4_decoder_compute(torch.tensor([1]), torch.tensor([0]))
    assert layer._v4_decoder_valid is None
    assert layer._v4_decoder_capture is False


def test_metadata_revalidates_replacements_and_constants_each_update():
    original = {"length": torch.tensor([3]), "position": 2}
    buffers = GRAPH.DecoderMetadataBuffers(original)
    buffers.update({"length": original["length"].clone(), "position": 2})
    with pytest.raises(ValueError, match="CPU metadata values changed"):
        buffers.update({"length": torch.tensor([4]), "position": 2})
    with pytest.raises(ValueError, match="constant changed"):
        buffers.update({"length": torch.tensor([3]), "position": 3})
    assert buffers.copies == 0


def test_metadata_rejects_broken_aliases_and_replaced_capture_storage():
    tensor = torch.tensor([3])
    buffers = GRAPH.DecoderMetadataBuffers({"left": tensor, "right": tensor.view(1)})
    assert buffers.tree["left"] is buffers.tree["right"]
    with pytest.raises(ValueError, match="alias topology"):
        buffers.update({"left": tensor, "right": tensor.clone()})
    buffers.tree["left"].resize_(2)
    with pytest.raises(ValueError, match="tensor contract"):
        buffers.update({"left": tensor, "right": tensor})


def test_snapshot_restores_all_mutable_state():
    tensors = (torch.arange(4), torch.tensor([9], dtype=torch.int32))
    saved = tuple(value.clone() for value in tensors)
    snapshot = GRAPH.DecoderStateSnapshot(tensors)
    snapshot.clear_trial()
    assert all(torch.count_nonzero(value) == 0 for value in tensors)
    snapshot.restore()
    assert all(torch.equal(value, before) for value, before in zip(tensors, saved))


@pytest.mark.parametrize("valid", [True, False])
def test_capture_restores_kv_state_even_after_invalid_replay(monkeypatch, valid):
    # Inject only the metadata producer; test the actual capture/fence protocol.
    template = SimpleNamespace(PositionTemplateBuffers=lambda metadata: SimpleNamespace(tree=dict(metadata)))
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization.vq2a8_decoder_position_template", template)
    calls = []
    graph = SimpleNamespace(replay=lambda: calls.append("replay"))
    backend = SimpleNamespace(
        NPUGraph=lambda: graph,
        graph=lambda *args, **kwargs: nullcontext(),
        synchronize=lambda: calls.append("fence"),
    )
    bank = GRAPH.V4DecoderGraphBank(object(), 4, backend=backend)
    cache = torch.tensor([9, 8])
    saved = cache.clone()
    bank.snapshot = GRAPH.DecoderStateSnapshot((cache,))
    context = SimpleNamespace(attn_metadata=_metadata())
    original_metadata = context.attn_metadata

    def compute(tokens, positions):
        assert torch.count_nonzero(cache) == 0
        cache.fill_(5)
        return torch.ones((1, 4), dtype=torch.bfloat16), torch.tensor(valid)

    if valid:
        bank.capture(2, torch.tensor([7]), torch.tensor([2]), context, compute)
        assert not bank.failed
    else:
        with pytest.raises(ValueError, match="device validity"):
            bank.capture(2, torch.tensor([7]), torch.tensor([2]), context, compute)
        assert bank.failed
    assert context.attn_metadata is original_metadata
    assert torch.equal(cache, saved)
    assert "replay" in calls
    assert bank.entries[2]["graph"] is graph


def _replay_bank():
    calls = []
    stream = SimpleNamespace(npu_stream=17)
    backend = SimpleNamespace(
        current_stream=lambda: stream,
        is_current_stream_capturing=lambda: False,
        synchronize=lambda: calls.append("fence"),
    )
    bank = GRAPH.V4DecoderGraphBank(object(), 4, backend=backend)
    bank.ready = True
    bank.caller_stream = stream
    outputs = (torch.ones((1, 4), dtype=torch.bfloat16), torch.tensor(True))
    bank.entries[2] = {
        "tokens": torch.tensor([0]),
        "positions": torch.tensor([2]),
        "outputs": outputs,
        "output_contract": tuple(GRAPH._tensor_contract(value) for value in outputs),
        "metadata": SimpleNamespace(update=lambda metadata: calls.append("metadata")),
        "graph": SimpleNamespace(replay=lambda: calls.append("replay"), reset=lambda: calls.append("reset")),
    }
    bank.computes = (SimpleNamespace(runtime=object(), check_runtime_contract=lambda runtime: calls.append("guard")),)
    return bank, backend, calls


def test_replay_checks_before_submission_and_escapes_static_outputs():
    bank, _, calls = _replay_bank()
    returned = bank.replay(torch.tensor([8]), torch.tensor([2]), SimpleNamespace(attn_metadata=_metadata()))
    assert calls == ["guard", "metadata", "replay"]
    assert bank.entries[2]["tokens"].tolist() == [8]
    assert all(
        actual.data_ptr() != captured.data_ptr() for actual, captured in zip(returned, bank.entries[2]["outputs"])
    )
    bank.entries[2]["outputs"][0].zero_()
    assert torch.all(returned[0] == 1)


def test_changed_stream_fails_without_submission_and_latches_failure():
    bank, backend, calls = _replay_bank()
    backend.current_stream = lambda: SimpleNamespace(npu_stream=18)
    with pytest.raises(RuntimeError, match="caller stream changed"):
        bank.replay(torch.tensor([8]), torch.tensor([2]), SimpleNamespace(attn_metadata=_metadata()))
    assert bank.failed
    assert not calls


def test_failed_cleanup_fence_retains_graph_and_tensor_owners():
    bank, backend, calls = _replay_bank()
    original_entries = dict(bank.entries)

    def failed_fence():
        raise RuntimeError("device fence failed")

    backend.synchronize = failed_fence
    with pytest.raises(RuntimeError, match="device fence failed"):
        bank.close()
    assert bank.entries == original_entries
    assert bank.computes
    assert bank.failed and not bank.closed
    assert not calls


def test_successful_cleanup_fences_before_graph_reset():
    bank, _, calls = _replay_bank()
    bank.close()
    assert calls == ["fence", "reset"]
    assert bank.closed and not bank.entries and not bank.computes


def test_registration_keeps_official_models_and_only_tp1_adapter():
    source = (ROOT / "vllm_ascend/models/__init__.py").read_text(encoding="utf-8")
    assert '"DeepseekV4ForCausalLM"' in source
    assert '"DeepSeekV4MTPModel"' in source
    assert '"VQ2A8TP1OfflineForCausalLM"' in source
    assert "VQ2A8TP2" not in source


def test_no_removed_experiment_or_tool_imports_in_model_graph():
    paths = [MODEL_PATH, GRAPH_PATH, ROOT / "vllm_ascend/quantization/vq2a8_decoder_position_template.py"]
    forbidden = ("tools.", "vq2a8_root_fp8", "vq2a8_host_profile", "vq2a8_startup_trace", "vq2a8_decoder_input_plan")
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not any(name in (node.module or "") for name in forbidden)


@pytest.mark.parametrize(
    "directory,filename,constant",
    [
        ("kv_quant_sparse_attn_sharedkv_metadata", "kv_quant_sparse_attn_sharedkv_metadata_aicpu.cpp", "SAS_META_SIZE"),
        ("vllm_quant_lightning_indexer_metadata", "vllm_quant_lightning_indexer_metadata_aicpu.cpp", "QLI_META_SIZE"),
    ],
)
def test_a5_metadata_producers_bound_and_initialize_full_descriptor(directory, filename, constant):
    source = (ROOT / "csrc/attention" / directory / "op_kernel_aicpu" / filename).read_text(encoding="utf-8")
    assert "aicCoreNum_ > AIC_CORE_NUM" in source
    assert "aivCoreNum_ > AIV_CORE_NUM" in source
    assert f"i < {constant}" in source
    assert "outputWords[i] = 0;" in source
    assert "GetDataSize() < outputBytes" in source
