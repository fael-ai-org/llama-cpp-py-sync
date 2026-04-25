#!/usr/bin/env python3
"""
Sync upstream llama.cpp repository.

This script manages synchronization with the upstream llama.cpp repository.
It checks for updates, clones or updates the vendor directory, and tracks
the current commit SHA.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

UPSTREAM_REPO = "https://github.com/ggml-org/llama.cpp.git"
VENDOR_DIR = "vendor/llama.cpp"
SYNC_STATE_FILE = ".llama_cpp_sync_state.json"


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


def run_git_command(args: list, cwd: Optional[Path] = None) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -1, "", "Git not found"


def get_remote_head_sha() -> Optional[str]:
    """Get the latest commit SHA from upstream remote."""
    code, stdout, stderr = run_git_command([
        "ls-remote", UPSTREAM_REPO, "HEAD"
    ])

    if code != 0:
        print(f"Error fetching remote HEAD: {stderr}", file=sys.stderr)
        return None

    if stdout:
        return stdout.split()[0]
    return None


def get_local_head_sha(vendor_path: Path) -> Optional[str]:
    """Get the current HEAD SHA of the local vendor directory."""
    if not vendor_path.exists():
        return None

    code, stdout, stderr = run_git_command(["rev-parse", "HEAD"], cwd=vendor_path)

    if code != 0:
        return None

    return stdout


def load_sync_state(project_root: Path) -> dict:
    """Load the sync state from file."""
    state_file = project_root / SYNC_STATE_FILE

    if state_file.exists():
        try:
            with open(state_file) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    return {
        "last_sync_sha": None,
        "last_sync_time": None,
        "sync_count": 0
    }


def save_sync_state(project_root: Path, state: dict):
    """Save the sync state to file."""
    state_file = project_root / SYNC_STATE_FILE

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def clone_upstream(vendor_path: Path, shallow: bool = True) -> bool:
    """Clone the upstream repository."""
    vendor_path.parent.mkdir(parents=True, exist_ok=True)

    args = ["clone"]
    if shallow:
        args.extend(["--depth", "1"])
    args.extend([UPSTREAM_REPO, str(vendor_path)])

    print(f"Cloning llama.cpp to {vendor_path}...")
    code, stdout, stderr = run_git_command(args)

    if code != 0:
        print(f"Error cloning repository: {stderr}", file=sys.stderr)
        return False

    print("Clone successful!")
    return True


def update_upstream(vendor_path: Path) -> bool:
    """Update the existing vendor directory."""
    print(f"Updating llama.cpp in {vendor_path}...")

    code, stdout, stderr = run_git_command(["fetch", "--depth", "1", "origin", "master"], cwd=vendor_path)
    if code != 0:
        print(f"Error fetching updates: {stderr}", file=sys.stderr)
        return False

    code, stdout, stderr = run_git_command(["reset", "--hard", "origin/master"], cwd=vendor_path)
    if code != 0:
        print(f"Error resetting to origin/master: {stderr}", file=sys.stderr)
        return False

    print("Update successful!")
    return True


def check_update_needed(project_root: Path) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Check if an update is needed.

    Returns:
        Tuple of (update_needed, local_sha, remote_sha)
    """
    vendor_path = project_root / VENDOR_DIR

    remote_sha = get_remote_head_sha()
    if remote_sha is None:
        print("Could not fetch remote SHA", file=sys.stderr)
        return False, None, None

    local_sha = get_local_head_sha(vendor_path)

    if local_sha is None:
        print("No local clone found, update needed")
        return True, None, remote_sha

    if local_sha != remote_sha:
        print(f"Update available: {local_sha[:8]} -> {remote_sha[:8]}")
        return True, local_sha, remote_sha

    print(f"Already up to date: {local_sha[:8]}")
    return False, local_sha, remote_sha


def sync(project_root: Path, force: bool = False) -> bool:
    """
    Synchronize with upstream.

    Args:
        project_root: Project root directory.
        force: Force sync even if already up to date.

    Returns:
        True if sync was performed, False otherwise.
    """
    vendor_path = project_root / VENDOR_DIR
    state = load_sync_state(project_root)

    update_needed, local_sha, remote_sha = check_update_needed(project_root)

    if not update_needed and not force:
        return False

    if not vendor_path.exists():
        success = clone_upstream(vendor_path)
    else:
        success = update_upstream(vendor_path)

    if success:
        new_sha = get_local_head_sha(vendor_path)
        state["last_sync_sha"] = new_sha
        state["last_sync_time"] = datetime.utcnow().isoformat()
        state["sync_count"] = state.get("sync_count", 0) + 1
        save_sync_state(project_root, state)

        print(f"Synced to: {new_sha[:8] if new_sha else 'unknown'}")

    return success


def get_current_sha(project_root: Path) -> Optional[str]:
    """Get the current llama.cpp commit SHA."""
    vendor_path = project_root / VENDOR_DIR
    return get_local_head_sha(vendor_path)


def main():
    parser = argparse.ArgumentParser(
        description="Sync upstream llama.cpp repository"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check if update is available, don't sync"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force sync even if already up to date"
    )
    parser.add_argument(
        "--sha",
        action="store_true",
        help="Print current commit SHA and exit"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory (default: auto-detect)"
    )

    args = parser.parse_args()

    project_root = args.project_root or get_project_root()

    if args.sha:
        sha = get_current_sha(project_root)
        if sha:
            print(sha)
            sys.exit(0)
        else:
            print("No llama.cpp clone found", file=sys.stderr)
            sys.exit(1)

    if args.check:
        update_needed, local_sha, remote_sha = check_update_needed(project_root)
        if update_needed:
            print("UPDATE_NEEDED")
            sys.exit(0)
        else:
            print("UP_TO_DATE")
            sys.exit(0)

    success = sync(project_root, force=args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
