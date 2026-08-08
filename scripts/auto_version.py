#!/usr/bin/env python3
"""
Automatic version generation for llama-cpp-py-sync.

This script generates version strings based on:
- Date (CalVer style: YYYY.MM.DD)
- Commit count
- llama.cpp upstream commit SHA
"""

import argparse
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple


def get_project_root() -> Path:
    return Path(__file__).parent.parent.resolve()

def run_git_command(args: list, cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    try:
        result = subprocess.run(
            ["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=30
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return -1, "", "Git not available"


def get_commit_count(project_root: Path) -> int:
    code, stdout, _ = run_git_command(["rev-list", "--count", "HEAD"], cwd=project_root)
    if code == 0 and stdout.isdigit():
        return int(stdout)
    return 0


def get_current_commit_sha(project_root: Path, short: bool = True) -> Optional[str]:
    args = ["rev-parse", "--short" if short else "", "HEAD"]
    args = [a for a in args if a]
    code, stdout, _ = run_git_command(args, cwd=project_root)
    return stdout if code == 0 and stdout else None


def get_llama_cpp_commit(project_root: Path) -> Optional[str]:
    vendor_path = project_root / "vendor" / "llama.cpp"
    if not vendor_path.exists():
        return None
    code, stdout, _ = run_git_command(["rev-parse", "--short", "HEAD"], cwd=vendor_path)
    return stdout if code == 0 and stdout else None


def generate_version(project_root: Path, include_dev: bool = False) -> str:
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y.%m.%d")
    commit_count = get_commit_count(project_root)
    version = f"{date_str}.{commit_count}"
    if include_dev:
        version += ".dev0"
    return version


_PLAIN_NUMERIC_VERSION_RE = re.compile(r"^\d+(?:\.\d+)*$")


def _version_from_upstream_tag(tag: str) -> Optional[str]:
    # llama.cpp historically uses tags like b4512. Map these to a PEP 440 version.
    if tag.startswith("b"):
        num = tag[1:]
        if _PLAIN_NUMERIC_VERSION_RE.match(num):
            return f"0.{num}"
        return None

    # Semver-like tags are often prefixed with "v".
    if tag.startswith("v") and _PLAIN_NUMERIC_VERSION_RE.match(tag[1:] or ""):
        return tag[1:]

    # Plain numeric dotted versions already satisfy PEP 440 for our purposes.
    if _PLAIN_NUMERIC_VERSION_RE.match(tag):
        return tag

    return None


def update_version_file(
    project_root: Path,
    version: str,
    llama_commit: Optional[str] = None,
    llama_tag: Optional[str] = None,
):
    version_file = project_root / "src" / "llama_cpp_py_sync" / "_version.py"
    content = f'''"""Version information for llama-cpp-py-sync."""

__version__ = "{version}"
__llama_cpp_commit__ = "{llama_commit or 'unknown'}"
__llama_cpp_tag__ = "{llama_tag or ''}"
'''
    version_file.parent.mkdir(parents=True, exist_ok=True)
    with open(version_file, "w") as f:
        f.write(content)
    print(f"Updated {version_file}")
    print(f"  Version: {version}")
    print(f"  llama.cpp commit: {llama_commit or 'unknown'}")
    if llama_tag:
        print(f"  llama.cpp tag: {llama_tag}")


def main():
    parser = argparse.ArgumentParser(description="Generate version for llama-cpp-py-sync")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--dev", action="store_true", help="Add .dev0 suffix")
    parser.add_argument("--update", action="store_true", help="Update _version.py file")
    parser.add_argument("--print", action="store_true", dest="print_version", help="Print version")
    args = parser.parse_args()

    project_root = args.project_root or get_project_root()

    ref_type = os.environ.get("GITHUB_REF_TYPE")
    ref_name = os.environ.get("GITHUB_REF_NAME")
    llama_tag = None

    version = None
    if ref_type == "tag" and ref_name:
        llama_tag = ref_name
        mapped = _version_from_upstream_tag(ref_name)
        if mapped is not None:
            version = mapped

    if version is None:
        version = generate_version(project_root, include_dev=args.dev)

    llama_commit = get_llama_cpp_commit(project_root)

    if args.update:
        update_version_file(project_root, version, llama_commit, llama_tag)

    if args.print_version or not args.update:
        print(version)


if __name__ == "__main__":
    main()
