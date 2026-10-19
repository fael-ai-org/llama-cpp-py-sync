"""Pinned synchronization and publication-order regressions."""
import json
import subprocess
from pathlib import Path

import pytest

from scripts import sync_upstream


def test_pinned_sync_records_requested_commit_instead_of_newer_head(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    origin.mkdir()
    subprocess.run(["git", "init", str(origin)], check=True, capture_output=True)

    def commit(text):
        (origin / "fixture.txt").write_text(text)
        subprocess.run(["git", "-C", str(origin), "add", "fixture.txt"], check=True)
        subprocess.run(["git", "-C", str(origin), "-c", "user.name=Test",
                        "-c", "user.email=test@example.invalid", "commit", "-m", text],
                       check=True, capture_output=True)
        return subprocess.check_output(["git", "-C", str(origin), "rev-parse", "HEAD"], text=True).strip()

    pinned = commit("requested release")
    newer = commit("newer upstream HEAD")
    monkeypatch.setattr(sync_upstream, "UPSTREAM_REPO", str(origin))
    project = tmp_path / "project"
    project.mkdir()
    assert sync_upstream.sync(project, force=True, revision=pinned)
    assert sync_upstream.get_current_sha(project) == pinned != newer
    assert json.loads((project / sync_upstream.SYNC_STATE_FILE).read_text())["last_sync_sha"] == pinned


def test_failed_pinned_fetch_does_not_record_success(tmp_path, monkeypatch):
    (tmp_path / sync_upstream.VENDOR_DIR).mkdir(parents=True)
    monkeypatch.setattr(sync_upstream, "get_local_head_sha", lambda _path: "a" * 40)
    monkeypatch.setattr(sync_upstream, "run_git_command", lambda *_args, **_kwargs: (1, "", "fetch rejected"))
    assert not sync_upstream.sync(tmp_path, force=True, revision="b" * 40)
    assert not (tmp_path / sync_upstream.SYNC_STATE_FILE).exists()


@pytest.mark.parametrize("revision", ["master", "--upload-pack=bad", "a" * 39, "A" * 40])
def test_revision_must_be_an_exact_commit(tmp_path, revision):
    with pytest.raises(ValueError, match="full lowercase"):
        sync_upstream.sync(tmp_path, revision=revision)


def test_release_is_published_and_dispatched_only_after_validated_commit():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/sync.yml").read_text()
    ordered = ["Resolve latest upstream tag", "Sync upstream", "Regenerate bindings",
               "Validate regenerated CFFI against llama.cpp headers",
               "Validate high-level Python against regenerated CFFI", "Commit changes",
               "Publish validated upstream tag", "Trigger Build Wheels for mirrored tag"]
    positions = [workflow.index("- name: " + name) for name in ordered]
    assert positions == sorted(positions)
    assert '--revision "${{ steps.check.outputs.upstream_sha }}"' in workflow
    assert 'git rev-parse "${LATEST_UPSTREAM_TAG_REF}^{commit}"' in workflow
    assert "curl --fail-with-body" in workflow
    assert '|| echo "No changes to commit"' not in workflow
    build = (root / ".github/workflows/build.yml").read_text()
    validation = build.split("  validate-bindings:", 1)[1].split("  build-linux-x86_64:", 1)[0]
    assert "UPSTREAM_SHA: ${{ needs.check-upstream.outputs.upstream_sha }}" in validation
    assert "state.read_text" not in validation
