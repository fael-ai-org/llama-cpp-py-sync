from llama_cpp_py_sync import backends


def test_packaged_vulkan_library_is_detected(tmp_path, monkeypatch) -> None:
    package_dir = tmp_path / "llama_cpp_py_sync"
    package_dir.mkdir()
    fake_module = package_dir / "backends.py"
    fake_module.touch()
    (package_dir / "libggml-vulkan.dylib").touch()
    monkeypatch.setattr(backends, "__file__", str(fake_module))

    detected = backends._check_bundled_backend_libraries()

    assert detected == {
        "cuda": False,
        "metal": False,
        "vulkan": True,
        "rocm": False,
    }
