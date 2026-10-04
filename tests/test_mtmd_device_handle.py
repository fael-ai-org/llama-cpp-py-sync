from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llama_cpp_py_sync import multimodal as module
from llama_cpp_py_sync._cffi_bindings import ffi


@pytest.fixture
def native_context(tmp_path, monkeypatch):
    projector = tmp_path / "projector.gguf"
    projector.write_bytes(b"projector")
    params = SimpleNamespace()
    lib = SimpleNamespace(mtmd_context_params_default=lambda: params,
        mtmd_get_cap_from_file=lambda _: SimpleNamespace(inp_vision=True, inp_audio=False),
        mtmd_init_from_file=Mock(return_value=ffi.cast("mtmd_context *", 2)),
        mtmd_support_vision=lambda _: True, mtmd_free=Mock())
    device = ffi.cast("ggml_backend_dev_t", 1)
    name = ffi.new("char[]", b"supplied-device")
    backend = SimpleNamespace(ggml_backend_dev_by_name=Mock(return_value=device))
    monkeypatch.setattr(module, "get_mtmd_lib", lambda: lib)
    monkeypatch.setattr(module, "get_backend_lib", lambda: backend)
    monkeypatch.setattr(module, "get_backend_base_lib", lambda: SimpleNamespace(ggml_backend_dev_name=lambda _: name))
    monkeypatch.setattr(module.MultimodalContext, "capabilities", property(lambda self: {
        "projector_offload": {"requested_device": self.device}}))
    model = SimpleNamespace(_model=ffi.cast("llama_model *", 3), _ctx=ffi.cast("llama_context *", 4))
    return projector, params, lib, device, backend, model


def test_supplied_native_device_is_forwarded_without_global_registration(native_context):
    projector, params, lib, device, backend, model = native_context
    context = module.MultimodalContext(model, projector, device=device)
    assert params.device == device
    assert context._device_handle == device
    assert model._multimodal_capabilities["projector_offload"]["requested_device"] == "supplied-device"
    backend.ggml_backend_dev_by_name.assert_not_called()
    context.close()
    lib.mtmd_free.assert_called_once()


def test_named_native_device_keeps_existing_lookup(native_context):
    projector, params, _, device, backend, model = native_context
    context = module.MultimodalContext(model, projector, device="CUDA0")
    assert params.device == device and context.device == "CUDA0"
    backend.ggml_backend_dev_by_name.assert_called_once_with(b"CUDA0")
    context.close()


@pytest.mark.parametrize("device", [1, True, "", "bad\0name", ffi.NULL,
    ffi.cast("ggml_backend_dev_t", 0), ffi.cast("llama_model *", 1)])
def test_invalid_handle_never_reaches_native_initialization(native_context, device):
    projector, _, lib, _, _, model = native_context
    with pytest.raises(ValueError):
        module.MultimodalContext(model, projector, device=device)
    lib.mtmd_init_from_file.assert_not_called()


def test_device_selection_cannot_silently_change_cpu_request(native_context):
    projector, _, lib, device, _, model = native_context
    with pytest.raises(ValueError, match="use_gpu"):
        module.MultimodalContext(model, projector, device=device, use_gpu=False)
    lib.mtmd_init_from_file.assert_not_called()
