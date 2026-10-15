"""Apply one committed, hash-guarded Python development fix over an installed wheel.

Run with the target environment's interpreter. Native libraries, generated CFFI,
version and manifest remain unchanged. This is development evidence, not wheel
or release acceptance. The caller must review both normalized source hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from importlib.metadata import distribution
from pathlib import Path


def checksum(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def dependencies(package: Path) -> dict[str, str]:
    return {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in package.rglob("*") if path.is_file() and (
                path.name in {"_cffi_bindings.py", "_version.py", "native_manifest.json"}
                or path.suffix in {".dll", ".so", ".dylib", ".pyd"} or ".so." in path.name)}


def install(source: Path, target: Path, expected_old: str, expected_new: str,
            receipt: Path, revision: str) -> dict:
    if source.is_symlink() or target.is_symlink() or source.resolve() == target.resolve():
        raise ValueError("Source and installed regular file must be distinct")
    data, old = source.read_bytes(), target.read_bytes()
    if checksum(data) != expected_new or checksum(old) not in {expected_old, expected_new}:
        raise ValueError("Reviewed Python source hashes do not match")
    compile(data, str(source), "exec")
    before = dependencies(target.parent)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    backup = receipt.parent / ("original-" + hashlib.sha256(old).hexdigest() + ".py")
    if backup.exists() and backup.read_bytes() != old:
        raise ValueError("Existing backup differs")
    backup.write_bytes(old)
    if checksum(old) != expected_new:
        descriptor, temporary = tempfile.mkstemp(prefix=".python-fix-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, target.stat().st_mode & 0o777)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    if checksum(target.read_bytes()) != expected_new or dependencies(target.parent) != before:
        raise RuntimeError("Installed source or preserved native dependencies changed")
    result = {"kind": "development-python-source-fix", "source_revision": revision,
              "baseline_sha256": expected_old, "source_sha256": expected_new,
              "installed_sha256": checksum(target.read_bytes()), "dependencies": before,
              "native_dependencies_unchanged": True, "release_acceptance": False}
    receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-installed-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "src/llama_cpp_py_sync/llama.py"
    committed = subprocess.check_output(["git", "show", "HEAD:src/llama_cpp_py_sync/llama.py"], cwd=root)
    if checksum(committed) != checksum(source.read_bytes()):
        raise ValueError("Commit the reviewed Python fix before installation")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    target = Path(distribution("llama-cpp-py-sync").locate_file("llama_cpp_py_sync/llama.py"))
    try:
        target.resolve().relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("Use the installed runtime environment, not an editable source checkout")
    print(json.dumps(install(source, target, args.expected_installed_sha256,
                             args.expected_source_sha256, args.receipt, revision)))


if __name__ == "__main__":
    main()
