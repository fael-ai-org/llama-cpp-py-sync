"""Keep legacy Linux interpreters installable without narrowing the CI matrix."""

import ast
import itertools
import json
import re
from pathlib import Path


def test_runner_overrides_preserve_every_configured_os_python_pair():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/test.yml").read_text(encoding="utf-8")
    job = workflow.split("\n  test:\n", 1)[1]
    matrix = job.split("      matrix:\n", 1)[1].split("\n    steps:", 1)[0]
    platforms = re.search(r"os: \[([^\]]+)\]", matrix).group(1).split(", ")
    config = (root / "pyproject.toml").read_text(encoding="utf-8")
    versions = ast.literal_eval(re.search(r"test_pythons\s*=\s*(\[[^\]]+\])", config).group(1))
    pairs = set(itertools.product(platforms, versions))
    overrides = [json.loads(line.strip()[2:]) for line in matrix.splitlines()
                 if line.strip().startswith('- {')]

    assert "${{ fromJson(needs.config.outputs.test_pythons) }}" in matrix
    assert "runs-on: ${{ matrix.runner || matrix.os }}" in job
    assert "fail-fast: false" in job
    assert "max-parallel: 2" in job
    assert "exclude:" not in matrix
    assert len(overrides) == 2
    assert {(item["os"], item["python-version"]) for item in overrides} == {
        ("ubuntu-latest", "3.8"), ("ubuntu-latest", "3.9"),
    }
    for item in overrides:
        assert (item["os"], item["python-version"]) in pairs
    for platform, version in pairs:
        matches = [item for item in overrides
                   if (item["os"], item["python-version"]) == (platform, version)]
        runner = matches[0]["runner"] if matches else platform
        expected = "ubuntu-22.04" if platform == "ubuntu-latest" and version in ("3.8", "3.9") else platform
        assert runner == expected
