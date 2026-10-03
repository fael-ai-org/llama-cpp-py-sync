from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llama_cpp_py_sync._cffi_bindings import ffi
from llama_cpp_py_sync.llama import Llama
from llama_cpp_py_sync.multimodal import MultimodalContext


@pytest.fixture
def native(monkeypatch, tmp_path):
    path = tmp_path / "mmproj.gguf"
    path.write_bytes(b"fixture")
    params = ffi.new("struct mtmd_context_params *")[0]
    params.batch_max_tokens = 1024
    handle = ffi.cast("ggml_backend_dev_t", 1)
    backend = SimpleNamespace(ggml_backend_dev_by_name=Mock(return_value=handle))
    lib = SimpleNamespace(
        mtmd_get_cap_from_file=Mock(return_value=SimpleNamespace(inp_audio=True, inp_vision=False)),
        mtmd_context_params_default=Mock(return_value=params),
        mtmd_init_from_file=Mock(return_value=ffi.cast("struct mtmd_context *", 2)),
        mtmd_support_vision=Mock(return_value=False),
        mtmd_support_audio=Mock(return_value=True),
        mtmd_free=Mock(),
    )
    monkeypatch.setattr("llama_cpp_py_sync.multimodal.get_ffi", lambda: ffi)
    monkeypatch.setattr("llama_cpp_py_sync.multimodal.get_mtmd_lib", lambda: lib)
    monkeypatch.setattr("llama_cpp_py_sync.multimodal.get_backend_lib", lambda: backend)
    monkeypatch.setattr(MultimodalContext, "capabilities", property(lambda self: {}))
    model = SimpleNamespace(_model=ffi.NULL, _ctx=ffi.NULL)
    return path, model, params, backend, lib, handle


def test_named_device_reaches_native_projector_parameters(native):
    path, model, params, backend, lib, handle = native
    with MultimodalContext(model, path, device="RPC_named_device", warmup=False):
        assert params.device == handle
        assert params.use_gpu
        backend.ggml_backend_dev_by_name.assert_called_once_with(b"RPC_named_device")
        assert lib.mtmd_init_from_file.call_args.args[2].device == handle
    lib.mtmd_free.assert_called_once()


def test_default_keeps_native_device_selection(native):
    path, model, params, backend, _, _ = native
    with MultimodalContext(model, path):
        assert params.device == ffi.NULL
    backend.ggml_backend_dev_by_name.assert_not_called()


@pytest.mark.parametrize("device,use_gpu", [("", True), ("x\0y", True), (3, True), ("GPU0", False)])
def test_invalid_selection_does_not_load_native_projector(native, device, use_gpu):
    path, model, _, _, lib, _ = native
    with pytest.raises(ValueError):
        MultimodalContext(model, path, device=device, use_gpu=use_gpu)
    lib.mtmd_init_from_file.assert_not_called()


def test_missing_device_does_not_fall_back_to_automatic_selection(native):
    path, model, _, backend, lib, _ = native
    backend.ggml_backend_dev_by_name.return_value = ffi.NULL
    with pytest.raises(ValueError, match="Unknown native projector device"):
        MultimodalContext(model, path, device="missing")
    lib.mtmd_init_from_file.assert_not_called()


def test_cached_projector_identity_includes_device(monkeypatch):
    created = []
    def create(*args, **kwargs):
        result = SimpleNamespace(close=Mock(), device=kwargs["device"])
        created.append(result)
        return result
    monkeypatch.setattr("llama_cpp_py_sync.multimodal.MultimodalContext", create)
    model = Llama.__new__(Llama)
    model._multimodal_context = None
    model._multimodal_context_options = None
    first = model._get_multimodal_context("mmproj.gguf", discover_projector=False, use_gpu=True, device="GPU0")
    assert model._get_multimodal_context("mmproj.gguf", discover_projector=False, use_gpu=True, device="GPU0") is first
    second = model._get_multimodal_context("mmproj.gguf", discover_projector=False, use_gpu=True, device="GPU1")
    assert second.device == "GPU1"
    first.close.assert_called_once()
    assert len(created) == 2
    model.close()
