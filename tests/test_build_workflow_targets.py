from fnmatch import fnmatch
from pathlib import Path

WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "build.yml"


def test_manual_build_targets_are_selectable_and_publishable():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    targets = [
        "linux-x86_64-cpu",
        "linux-x86_64-cuda",
        "linux-x86_64-vulkan",
        "macos-arm64-metal",
        "macos-arm64-vulkan",
        "macos-x86_64-cpu",
        "macos-x86_64-vulkan",
        "windows-x64-cpu",
        "windows-x64-cuda",
        "windows-x64-vulkan",
    ]

    assert "build_target:" in workflow
    assert "release_tag:" in workflow
    for target in targets:
        assert f"- {target}" in workflow
        assert f"build_target == '{target}'" in workflow

    assert "tag_name: ${{ needs.check-upstream.outputs.release_tag }}" in workflow
    assert "overwrite_files: true" in workflow
    assert "if: always() && !failure() && !cancelled()" in workflow


def test_partial_rebuilds_do_not_publish_to_pypi():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    publish_section = workflow.split("- name: Publish to PyPI", 1)[1]

    assert "startsWith(github.ref, 'refs/tags/')" in publish_section
    assert "build_target == 'all'" in publish_section
    assert "github.event_name == 'push'" not in publish_section

    pypi_artifact_condition = (
        "if: startsWith(github.ref, 'refs/tags/') && "
        "needs.check-upstream.outputs.build_target == 'all'"
    )
    assert workflow.split("- name: Publish to PyPI", 1)[0].count(pypi_artifact_condition) == 2


def test_release_assets_exclude_ci_test_signing_files():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    release_section = workflow.split("- name: Create Release", 1)[1].split(
        "- name: Publish to PyPI", 1
    )[0]
    pattern = next(
        line.strip().removeprefix("files: ")
        for line in release_section.splitlines()
        if line.strip().startswith("files: ")
    )
    for variant in ("cpu", "cu128", "vulkan", "metal"):
        assert fnmatch(f"dist/llama_cpp_py_sync-0.11227-1{variant}-py3-none-platform.whl", pattern)
    for name in (
        "LLaMA-Linux-CPU-Test-Signing.asc",
        "LLaMA-Linux-CUDA-Test-Signing.asc",
        "LLaMA-Linux-Vulkan-Test-Signing.asc",
        "llama_cpp_py_sync-0.11227-1cpu-py3-none-manylinux2014_x86_64.whl.asc",
        "SHA256SUMS-linux-x86_64-cpu",
        "SHA256SUMS.asc",
    ):
        assert not fnmatch(f"dist/{name}", pattern)


def test_linux_jobs_build_and_attest_without_test_signing():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert not (WORKFLOW.parents[2] / "scripts" / "sign_linux_artifacts.py").exists()
    for job in ("build-linux-x86_64", "build-linux-x86_64-cuda", "build-linux-x86_64-vulkan"):
        section = workflow.split(f"  {job}:\n", 1)[1].split("\n  build-", 1)[0]
        assert "sign_linux_artifacts" not in section
        assert "gnupg" not in section
        assert "SHA256SUMS" not in section
        assert "Test-Signing" not in section
        assert "Smoke test installed wheel" in section
        assert "uses: actions/attest@v4" in section
        assert "subject-path: dist/*.whl" in section
