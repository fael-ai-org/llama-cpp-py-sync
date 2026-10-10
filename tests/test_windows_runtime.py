import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llama_cpp_py_sync import _windows_runtime as runtime


def test_missing_bundled_openmp_does_not_load_library(monkeypatch, tmp_path):
    loader = Mock()
    monkeypatch.setattr(ctypes, "WinDLL", loader, raising=False)
    runtime.retain_openmp_runtime(tmp_path)
    loader.assert_not_called()


def test_bundled_openmp_is_pinned_once_by_absolute_path(monkeypatch, tmp_path):
    path = tmp_path / "vcomp140.dll"
    path.touch()
    pin = Mock(return_value=1)
    owner = object()
    loader = Mock(side_effect=[owner, SimpleNamespace(GetModuleHandleExW=pin)])
    monkeypatch.setattr(ctypes, "WinDLL", loader, raising=False)
    monkeypatch.setattr(runtime, "_retained_runtimes", {})
    runtime.retain_openmp_runtime(tmp_path)
    runtime.retain_openmp_runtime(tmp_path)
    assert loader.call_args_list[0].args == (str(path.resolve()),)
    assert pin.call_args.args[:2] == (1, str(path.resolve()))
    pin.assert_called_once()
    assert runtime._retained_runtimes[path.resolve()] is owner


def test_failed_pin_is_not_recorded_as_retained(monkeypatch, tmp_path):
    (tmp_path / "vcomp140.dll").touch()
    pin = Mock(return_value=0)
    monkeypatch.setattr(ctypes, "WinDLL", Mock(side_effect=[object(),
        SimpleNamespace(GetModuleHandleExW=pin)]), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    monkeypatch.setattr(ctypes, "WinError", lambda code: OSError(code, "pin failed"), raising=False)
    monkeypatch.setattr(runtime, "_retained_runtimes", {})
    with pytest.raises(OSError):
        runtime.retain_openmp_runtime(tmp_path)
    assert not runtime._retained_runtimes
