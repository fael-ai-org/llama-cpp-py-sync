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

    assert "github.event_name == 'push'" in publish_section
    assert "build_target == 'all'" in publish_section
