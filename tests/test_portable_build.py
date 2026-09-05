from pathlib import Path

from scripts.build_llama_cpp import (
    _codesign_macos_package_dylibs,
    _copy_linux_dependency_sos,
    _copy_linux_runtime_so,
    _preferred_linux_library_name,
    get_cmake_args,
)


def test_all_wheel_builds_disable_native_cpu_optimization(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    backends = {
        "cuda": (False, None),
        "rocm": (False, None),
        "vulkan": (False, None),
        "metal": (False, None),
        "blas": (False, None),
    }

    args = get_cmake_args(backends)

    assert args.count("-DGGML_NATIVE=OFF") == 1


def test_wheel_builds_disable_server_curl_and_rpc(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    backends = {
        "cuda": (False, None),
        "rocm": (False, None),
        "vulkan": (False, None),
        "metal": (False, None),
        "blas": (False, None),
    }

    args = get_cmake_args(backends)

    assert "-DLLAMA_BUILD_SERVER=OFF" in args
    assert "-DLLAMA_CURL=OFF" in args
    assert "-DGGML_RPC=OFF" in args


def test_macos_dylibs_are_resigned_after_loader_path_changes(tmp_path, monkeypatch) -> None:
    first = tmp_path / "libFirst.dylib"
    second = tmp_path / "libSecond.dylib"
    first.touch()
    second.touch()
    calls = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(args, **kwargs):
        calls.append(args)
        return Result()

    monkeypatch.setattr("scripts.build_llama_cpp.subprocess.run", fake_run)

    _codesign_macos_package_dylibs(Path(tmp_path))

    assert calls == [
        ["codesign", "--force", "--sign", "-", str(first)],
        ["codesign", "--force", "--sign", "-", str(second)],
    ]


def test_linux_runtime_symlink_is_materialized_once_under_soname(tmp_path) -> None:
    source_dir = tmp_path / "cuda"
    package_dir = tmp_path / "package"
    source_dir.mkdir()
    package_dir.mkdir()
    versioned = source_dir / "libcublas.so.12.8.5.5"
    versioned.write_bytes(b"cuda-library")
    soname = source_dir / "libcublas.so.12"
    soname.symlink_to(versioned.name)

    assert _copy_linux_runtime_so(soname, package_dir, runtime_name="libcublas.so.12")

    packaged = list(package_dir.iterdir())
    assert [path.name for path in packaged] == ["libcublas.so.12"]
    assert packaged[0].read_bytes() == b"cuda-library"
    assert not packaged[0].is_symlink()


def test_linux_library_fallback_prefers_abi_name() -> None:
    aliases = [
        Path("libggml-cuda.so.0.19.0"),
        Path("libggml-cuda.so"),
        Path("libggml-cuda.so.0"),
    ]

    assert _preferred_linux_library_name(aliases) == "libggml-cuda.so.0"


def test_linux_build_aliases_produce_one_soname_copy(tmp_path, monkeypatch) -> None:
    build_dir = tmp_path / "build" / "bin"
    package_dir = tmp_path / "package"
    build_dir.mkdir(parents=True)
    package_dir.mkdir()

    llama = build_dir / "libllama.so"
    llama.write_bytes(b"llama")
    (build_dir / "libllama.so.0").symlink_to(llama.name)
    (build_dir / "libllama.so.0.0.1").symlink_to(llama.name)

    cuda = build_dir / "libggml-cuda.so.0.19.0"
    cuda.write_bytes(b"large-cuda-library")
    (build_dir / "libggml-cuda.so.0").symlink_to(cuda.name)
    (build_dir / "libggml-cuda.so").symlink_to("libggml-cuda.so.0")

    sonames = {
        llama.resolve(): "libllama.so.0",
        cuda.resolve(): "libggml-cuda.so.0",
    }
    monkeypatch.setattr(
        "scripts.build_llama_cpp._linux_shared_library_soname",
        lambda path: sonames[path],
    )

    _copy_linux_dependency_sos(llama, package_dir)

    assert sorted(path.name for path in package_dir.iterdir()) == [
        "libggml-cuda.so.0",
        "libllama.so.0",
    ]
