from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "build.yml"


def test_workflow_uses_one_broad_cuda_128_configuration() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert workflow.count("cuda: '12.8.0'") == 2
    assert "cuda: '12.2.0'" not in workflow
    assert "cuda: '12.4.1'" not in workflow
    assert workflow.count("CMAKE_CUDA_ARCHITECTURES: '50;52;60;61;70;75;80;86;89;90;100;120'") == 2
    assert "1cu128" in workflow


def test_windows_cuda_runtime_dlls_are_discovered_by_pattern() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    for pattern in (
        "cudart64_*.dll",
        "cublas64_*.dll",
        "cublasLt64_*.dll",
        "nvrtc64_*.dll",
        "nvJitLink64_*.dll",
    ):
        assert pattern in workflow


def test_macos_arm64_vulkan_wheel_is_built_and_released() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "build-macos-arm64-vulkan:" in workflow
    assert "runs-on: macos-14" in workflow
    assert "uses: jakoch/install-vulkan-sdk-action@v1" in workflow
    assert "vulkan_version: '1.4.335.0'" in workflow
    assert "brew install cmake ninja" in workflow
    assert "macosx_14_0_arm64 --build 1vulkan" in workflow
    assert "name: wheel-macos-arm64-vulkan" in workflow
    assert "build-macos-arm64-vulkan, build-macos-x86_64" in workflow


def test_vulkan_smoke_tests_do_not_require_runner_gpu_hardware() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert workflow.count("assert info.vulkan, info") == 4
    assert "info.vulkan and info.gpu_offload" not in workflow
