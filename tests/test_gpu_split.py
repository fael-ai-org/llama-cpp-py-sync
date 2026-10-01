from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from llama_cpp_py_sync._cffi_bindings import get_ffi, get_lib
from llama_cpp_py_sync.llama import Llama, _configure_gpu_split


@pytest.fixture
def native_params():
    ffi = get_ffi()
    params = ffi.new("struct llama_model_params *")
    params.split_mode = 1
    return ffi, SimpleNamespace(llama_max_devices=lambda: 4), params


def test_preserves_upstream_defaults(native_params):
    ffi, lib, params = native_params
    assert _configure_gpu_split(ffi, lib, params, None, None, None) is None
    assert params.split_mode == 1
    assert params.tensor_split == ffi.NULL


@pytest.mark.parametrize("mode,value", [("none", 0), ("layer", 1), ("row", 2), ("tensor", 3), (1, 1)])
def test_placement_and_full_capacity_pointer(native_params, mode, value):
    ffi, lib, params = native_params
    buffer = _configure_gpu_split(ffi, lib, params, mode, 1, [3, 1])
    assert params.split_mode == value
    assert params.main_gpu == 1
    assert params.tensor_split == buffer
    assert list(buffer) == pytest.approx([3, 1, 0, 0])


@pytest.mark.parametrize("proportions", [[], [0, 0], [-1, 1], [float("nan")], [float("inf")], [1] * 5, [1e39], [1e-300], "3,1"])
def test_rejects_unsafe_proportions(native_params, proportions):
    ffi, lib, params = native_params
    with pytest.raises(ValueError):
        _configure_gpu_split(ffi, lib, params, "layer", 0, proportions)


@pytest.mark.parametrize("mode,gpu", [("invalid", 0), (4, 0), (True, 0), (1, -1), (1, 4), (1, True), (1.5, 0), (1, 0.5)])
def test_rejects_invalid_placement(native_params, mode, gpu):
    ffi, lib, params = native_params
    with pytest.raises((TypeError, ValueError)):
        _configure_gpu_split(ffi, lib, params, mode, gpu, None)


@pytest.mark.native
def test_native_default_params_accept_split_controls():
    ffi = get_ffi()
    try:
        lib = get_lib()
    except RuntimeError as error:
        pytest.skip(f"Native llama.cpp library unavailable: {error}")
    params = lib.llama_model_default_params()
    buffer = _configure_gpu_split(ffi, lib, params, "layer", 0, [1])
    assert params.split_mode == lib.LLAMA_SPLIT_MODE_LAYER
    assert params.tensor_split[0] == 1.0
    assert len(buffer) == lib.llama_max_devices()


@pytest.mark.integration
def test_real_generation_with_explicit_layer_split():
    """Single-device smoke; this alone does not establish multi-GPU execution."""
    model_path = os.environ.get("LLAMA_TEST_MODEL")
    if not model_path:
        pytest.skip("Set LLAMA_TEST_MODEL to run native inference")
    gpu_layers = int(os.environ.get("LLAMA_TEST_GPU_LAYERS", "0"))
    if gpu_layers and not get_lib().llama_supports_gpu_offload():
        pytest.fail("GPU smoke requested but the native library has no GPU offload")
    with Llama(
        model_path,
        n_ctx=256,
        n_batch=32,
        n_threads=2,
        n_gpu_layers=gpu_layers,
        split_mode="layer",
        main_gpu=0,
        tensor_split=[1],
    ) as model:
        # Prove the constructor retains the array consumed by native loading.
        assert model._tensor_split_buffer[0] == 1.0
        output = model.generate("Say hello.", max_tokens=4, temperature=0.0, seed=123)
        assert isinstance(output, str) and output.strip()
