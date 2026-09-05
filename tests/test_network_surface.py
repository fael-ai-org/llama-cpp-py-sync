import pytest

from scripts.validate_llama_exports import (
    _require_rpc_disabled,
    forbidden_native_artifacts,
    is_forbidden_native_artifact,
)


def test_rpc_and_server_binaries_are_forbidden() -> None:
    assert is_forbidden_native_artifact("libggml-rpc.so.0")
    assert is_forbidden_native_artifact("ggml-rpc.dll")
    assert is_forbidden_native_artifact("llama-server")
    assert is_forbidden_native_artifact("llama-server.exe")
    assert is_forbidden_native_artifact("rpc-server")


def test_inference_libraries_are_allowed() -> None:
    assert not is_forbidden_native_artifact("libllama.so.0")
    assert not is_forbidden_native_artifact("llama.dll")
    assert not is_forbidden_native_artifact("libggml-cuda.so.0")
    assert not is_forbidden_native_artifact("libmtmd.dylib")
    assert not is_forbidden_native_artifact("_cffi_bindings.py")


def test_forbidden_native_artifacts_reports_only_matches() -> None:
    names = [
        "src/llama_cpp_py_sync/libllama.so.0",
        "src/llama_cpp_py_sync/libggml-rpc.so.0",
        "llama-server.exe",
    ]

    assert forbidden_native_artifacts(names) == [
        "libggml-rpc.so.0",
        "llama-server.exe",
    ]


def test_require_rpc_disabled_accepts_false() -> None:
    class Library:
        def llama_supports_rpc(self) -> bool:
            return False

    _require_rpc_disabled(Library())


def test_require_rpc_disabled_rejects_true() -> None:
    class Library:
        def llama_supports_rpc(self) -> bool:
            return True

    with pytest.raises(RuntimeError, match="GGML_RPC"):
        _require_rpc_disabled(Library())
