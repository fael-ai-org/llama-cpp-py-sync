from pathlib import Path

from scripts.build_llama_cpp import _codesign_macos_package_dylibs, get_cmake_args


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
