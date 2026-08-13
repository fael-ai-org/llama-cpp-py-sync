"""Optional upstream-model integration tests configured entirely by paths."""

import os

import pytest

from llama_cpp_py_sync import Llama


@pytest.mark.integration
def test_upstream_embedding_model():
    path = os.environ.get("LLAMA_TEST_EMBEDDING_MODEL")
    if not path:
        pytest.skip("set LLAMA_TEST_EMBEDDING_MODEL to an upstream-supported GGUF")
    with Llama(path, embedding=True, n_ctx=512, n_batch=512, n_gpu_layers=99) as model:
        vector = model.get_embeddings("binding integration test")
        assert len(vector) == model.n_embd
        assert any(vector)


@pytest.mark.integration
def test_upstream_asr_model():
    model_path = os.environ.get("LLAMA_TEST_ASR_MODEL")
    projector = os.environ.get("LLAMA_TEST_ASR_PROJECTOR")
    audio = os.environ.get("LLAMA_TEST_ASR_AUDIO")
    if not all((model_path, projector, audio)):
        pytest.skip("set LLAMA_TEST_ASR_MODEL, LLAMA_TEST_ASR_PROJECTOR, and LLAMA_TEST_ASR_AUDIO")
    with Llama(model_path, n_ctx=4096, n_gpu_layers=99) as model:
        assert model.transcribe(audio, projector_path=projector, max_tokens=64).strip()


@pytest.mark.integration
def test_upstream_tts_model():
    model_path = os.environ.get("LLAMA_TEST_TTS_MODEL")
    projector = os.environ.get("LLAMA_TEST_TTS_PROJECTOR")
    if not model_path or not projector:
        pytest.skip("set LLAMA_TEST_TTS_MODEL and LLAMA_TEST_TTS_PROJECTOR")
    with Llama(model_path, n_ctx=2048, n_gpu_layers=99) as model:
        result = model.generate_audio(
            "This is an integration test.", projector_path=projector, max_frames=8, seed=1
        )
        assert result.data
        assert result.sample_rate > 0
