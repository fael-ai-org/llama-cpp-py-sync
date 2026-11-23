"""Upstream tag to package-version mapping regressions."""

import sys

from scripts import auto_version
from scripts.auto_version import _version_from_upstream_tag, update_version_file


def test_stable_upstream_tags_keep_the_same_numeric_package_version():
    assert _version_from_upstream_tag("v1.0.0") == "1.0.0"
    assert _version_from_upstream_tag("v1.2.3") == "1.2.3"


def test_legacy_upstream_build_tags_keep_the_existing_mapping():
    assert _version_from_upstream_tag("b12284") == "0.12284"


def test_version_file_records_the_exact_upstream_tag_safely(tmp_path):
    update_version_file(
        tmp_path,
        "1.0.0",
        llama_commit="b1b770d90d94cad9e9ea5b993bdbc383a522fb28",
        llama_tag='v1.0.0"\n__version__ = "0.0.0',
    )

    namespace = {}
    version_file = tmp_path / "src/llama_cpp_py_sync/_version.py"
    exec(version_file.read_text(encoding="utf-8"), namespace)
    assert namespace["__version__"] == "1.0.0"
    assert namespace["__llama_cpp_commit__"] == "b1b770d90d94cad9e9ea5b993bdbc383a522fb28"
    assert namespace["__llama_cpp_tag__"] == 'v1.0.0"\n__version__ = "0.0.0'


def test_cli_uses_the_explicit_upstream_tag_on_a_branch_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "auto_version.py",
            "--project-root",
            str(tmp_path),
            "--update",
            "--upstream-tag",
            "v1.0.0",
        ],
    )
    monkeypatch.setenv("GITHUB_REF_TYPE", "branch")
    monkeypatch.setenv("GITHUB_REF_NAME", "master")

    auto_version.main()

    namespace = {}
    version_file = tmp_path / "src/llama_cpp_py_sync/_version.py"
    exec(version_file.read_text(encoding="utf-8"), namespace)
    assert namespace["__version__"] == "1.0.0"
    assert namespace["__llama_cpp_tag__"] == "v1.0.0"
