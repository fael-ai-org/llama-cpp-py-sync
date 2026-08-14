from __future__ import annotations

import inspect
from types import MethodType

import pytest

from llama_cpp_py_sync._cffi_bindings import ffi
from llama_cpp_py_sync.embeddings import normalize_embedding
from llama_cpp_py_sync.llama import (
    POOLING_TYPES,
    Llama,
    TranscriptionResult,
    _clean_asr_text,
)


def test_upstream_embedding_controls_are_present_in_the_generated_cdef():
    bindings = __import__(
        "llama_cpp_py_sync._cffi_bindings", fromlist=["_LLAMA_H_CDEF"]
    )._LLAMA_H_CDEF
    assert "enum llama_pooling_type      pooling_type" in bindings
    assert "llama_get_embeddings_ith" in bindings
    assert "llama_get_embeddings_seq" in bindings
    assert "llama_batch_init" in bindings
    assert "bool offload_kqv" in bindings
    assert "bool op_offload" in bindings


def test_upstream_tts_input_matches_the_available_helper_controls():
    bindings = __import__(
        "llama_cpp_py_sync._cffi_bindings", fromlist=["_LLAMA_H_CDEF"]
    )._LLAMA_H_CDEF
    start = bindings.index("struct mtmd_helper_gen_audio_inp {")
    end = bindings.index("};", start)
    audio_input = bindings[start:end]
    assert "speaker_ref" in audio_input
    assert "lang" in audio_input
    assert "instruct" not in audio_input
    assert "mtmd_gen_audio_process" in bindings
    assert "mtmd_helper_gen_audio_step_gen" in bindings
    assert "mtmd_helper_gen_audio_get_output" in bindings


def test_common_embedding_normalization_modes_match_upstream():
    assert normalize_embedding([3.0, 4.0], mode="none") == [3.0, 4.0]
    assert normalize_embedding([3.0, 4.0], mode="l1") == pytest.approx([0.4285714, 0.5714286])
    assert normalize_embedding([3.0, 4.0], mode="l2") == pytest.approx([0.6, 0.8])
    assert normalize_embedding([3.0, 4.0], mode=3) == pytest.approx(
        [3.0 / 91.0 ** (1.0 / 3.0), 4.0 / 91.0 ** (1.0 / 3.0)]
    )


def test_asr_wrappers_are_structured_and_do_not_leak():
    text = _clean_asr_text("<asr_text><|en|> hello world </asr_text>")
    assert text == "hello world"
    result = TranscriptionResult(text=text)
    assert result.text == "hello world"
    assert not hasattr(result, "language")
    assert not hasattr(result, "segments")


def test_public_audio_signatures_only_expose_upstream_controls():
    for method_name in ("generate_audio", "generate_audio_stream"):
        parameters = inspect.signature(getattr(Llama, method_name)).parameters
        assert "speaker" not in parameters
        assert "instruct" not in parameters


def test_capabilities_do_not_advertise_unavailable_modalities_or_operations():
    model = object.__new__(Llama)
    model._lib = type("Lib", (), {})()
    model._embedding = False
    model._pooling_type_requested = None
    model._n_seq_max = 1

    capabilities = model.capabilities
    assert "stt_capabilities" not in capabilities
    assert "tts_capabilities" not in capabilities
    assert capabilities["embedding_capabilities"]["normalization"] == {
        "upstream_c_api": False,
        "python_postprocess": True,
        "modes": ["none", "max_abs", "taxicab", "euclidean", "p-norm"],
    }


def test_projector_context_is_reused_across_turns(monkeypatch):
    import llama_cpp_py_sync.multimodal as multimodal

    created = []
    closed = []

    class _FakeContext:
        def __init__(self, *_args, **kwargs):
            created.append(kwargs)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(multimodal, "MultimodalContext", _FakeContext)
    model = object.__new__(Llama)
    model._multimodal_context = None
    model._multimodal_context_options = None

    first = model._get_multimodal_context(
        "projector.gguf", discover_projector=True, use_gpu=True, warmup=False
    )
    second = model._get_multimodal_context(
        "projector.gguf", discover_projector=True, use_gpu=True, warmup=True
    )

    assert first is second
    assert len(created) == 1
    model.close()
    assert closed == [True]


class _FakeEmbeddingLib:
    def __init__(self, pooling_type: int):
        self.pooling = pooling_type
        self.calls = []
        self._pointers = []

    def llama_pooling_type(self, _ctx):
        return self.pooling

    def llama_n_seq_max(self, _ctx):
        return 2

    def llama_model_n_embd_out(self, _model):
        return 3

    def llama_model_has_encoder(self, _model):
        return False

    def llama_batch_init(self, n_tokens, _embd, n_seq_max):
        batch = type("Batch", (), {})()
        batch.n_tokens = n_tokens
        batch.token = ffi.new("llama_token[]", n_tokens)
        batch.pos = ffi.new("llama_pos[]", n_tokens)
        batch.n_seq_id = ffi.new("int32_t[]", n_tokens)
        batch.seq_id = ffi.new("llama_seq_id *[]", n_tokens)
        batch.logits = ffi.new("int8_t[]", n_tokens)
        batch._seq_buffers = []
        for i in range(n_tokens):
            batch._seq_buffers.append(ffi.new("llama_seq_id[]", n_seq_max))
            batch.seq_id[i] = batch._seq_buffers[-1]
        return batch

    def llama_batch_free(self, _batch):
        pass

    def llama_free(self, _ctx):
        pass

    def llama_model_free(self, _model):
        pass

    def llama_decode(self, _ctx, batch):
        self.calls.append(
            [
                (int(batch.token[i]), int(batch.pos[i]), int(batch.seq_id[i][0]), int(batch.logits[i]))
                for i in range(batch.n_tokens)
            ]
        )
        return 0

    def llama_get_embeddings_seq(self, _ctx, seq_id):
        pointer = ffi.new("float[]", [float(seq_id + 1), 2.0, 3.0])
        self._pointers.append(pointer)
        return pointer

    def llama_get_embeddings_ith(self, _ctx, index):
        pointer = ffi.new("float[]", [float(index), 10.0, 20.0])
        self._pointers.append(pointer)
        return pointer


def _fake_embedding_model(fake_lib, *, pooling_type: int):
    model = object.__new__(Llama)
    model._lib = fake_lib
    model._ffi = ffi
    model._model = object()
    model._ctx = object()
    model._sampler = None
    model._embedding = True
    model._n_batch = 8
    model._n_ctx = 64
    model._n_seq_max = 2
    model._pooling_type_requested = pooling_type
    model.tokenize = MethodType(lambda _self, text, add_special=True, parse_special=False: [1, 2] if text == "a" else [3, 4], model)
    return model


def test_embeddings_use_one_native_multi_sequence_batch():
    fake = _FakeEmbeddingLib(POOLING_TYPES["mean"])
    model = _fake_embedding_model(fake, pooling_type=POOLING_TYPES["mean"])
    values = model.get_embeddings_batch(["a", "b"], normalize=None)
    assert values == [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]]
    assert len(fake.calls) == 1
    assert {row[2] for row in fake.calls[0]} == {0, 1}


def test_per_token_embeddings_require_none_pooling():
    fake = _FakeEmbeddingLib(POOLING_TYPES["none"])
    model = _fake_embedding_model(fake, pooling_type=POOLING_TYPES["none"])
    values = model.get_embeddings("a", normalize=None, per_token=True)
    assert values == [[0.0, 10.0, 20.0], [1.0, 10.0, 20.0]]

    fake = _FakeEmbeddingLib(POOLING_TYPES["mean"])
    model = _fake_embedding_model(fake, pooling_type=POOLING_TYPES["mean"])
    with pytest.raises(ValueError, match="per_token"):
        model.get_embeddings("a", per_token=True)
