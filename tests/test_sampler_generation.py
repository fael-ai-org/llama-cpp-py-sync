from __future__ import annotations

import os
from pathlib import Path

import pytest

from llama_cpp_py_sync._cffi_bindings import get_ffi, get_lib
from llama_cpp_py_sync.llama import Llama


def _get_native_library_or_skip():
    """Load llama.cpp when this test environment provides a native build."""
    try:
        return get_lib()
    except RuntimeError as error:
        pytest.skip(f"Native llama.cpp library is unavailable in this test environment: {error}")


def test_native_sampler_skips_when_no_native_library_is_available(monkeypatch):
    """Source-only CI must not require a platform-specific native build."""
    def missing_library():
        raise RuntimeError("native test library missing")

    monkeypatch.setitem(_get_native_library_or_skip.__globals__, "get_lib", missing_library)

    with pytest.raises(pytest.skip.Exception):
        _get_native_library_or_skip()


@pytest.mark.native
def test_penalties_sampler_uses_current_upstream_signature():
    """The generated CFFI ABI requires n_vocab as the first argument."""
    lib = _get_native_library_or_skip()
    ffi = get_ffi()

    sampler = lib.llama_sampler_init_penalties(128, 64, 1.1, 0.0, 0.0)
    try:
        assert sampler != ffi.NULL
    finally:
        if sampler != ffi.NULL:
            lib.llama_sampler_free(sampler)


@pytest.mark.integration
def test_real_generation_constructs_sampler_and_decodes():
    """Run one real decode when a caller supplies a local GGUF test model."""
    model_value = os.environ.get("LLAMA_TEST_MODEL", "").strip()
    if not model_value:
        pytest.skip("Set LLAMA_TEST_MODEL to run the real generation regression test")

    model_path = Path(model_value)
    if not model_path.is_file():
        pytest.fail(f"LLAMA_TEST_MODEL does not point to a file: {model_path}")

    with Llama(
        str(model_path),
        n_ctx=256,
        n_batch=32,
        n_threads=2,
        n_gpu_layers=int(os.environ.get("LLAMA_TEST_GPU_LAYERS", "0")),
        verbose=False,
    ) as model:
        output = model.generate(
            "Reply with the single word: hello",
            max_tokens=8,
            temperature=0.0,
            seed=123,
        )

    assert isinstance(output, str)
    assert output.strip()
