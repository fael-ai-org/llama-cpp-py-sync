import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "source_fix", Path(__file__).resolve().parents[1] / "scripts/install_python_source_fix.py")
FIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIX)


def test_guarded_fix_preserves_native_files_backup_and_repeat_install(tmp_path):
    package = tmp_path / "installed"
    package.mkdir()
    native = package / "libllama.so"
    native.write_bytes(b"pinned-native")
    target = package / "llama.py"
    target.write_bytes(b"value = 1\r\n")
    source = tmp_path / "new.py"
    source.write_bytes(b"value = 2\n")
    receipt = tmp_path / "evidence/install.json"
    old, new = FIX.checksum(target.read_bytes()), FIX.checksum(source.read_bytes())
    result = FIX.install(source, target, old, new, receipt, "reviewed")
    assert target.read_bytes() == source.read_bytes()
    assert native.read_bytes() == b"pinned-native"
    assert next(receipt.parent.glob("original-*.py")).read_bytes() == b"value = 1\r\n"
    assert result["native_dependencies_unchanged"]
    assert not result["release_acceptance"]
    FIX.install(source, target, old, new, receipt, "reviewed")


def test_unreviewed_installed_source_is_preserved(tmp_path):
    source, target = tmp_path / "new.py", tmp_path / "installed.py"
    source.write_bytes(b"value = 2\n")
    target.write_bytes(b"unexpected = 3\n")
    with pytest.raises(ValueError, match="hashes"):
        FIX.install(source, target, "0" * 64, FIX.checksum(source.read_bytes()), tmp_path / "receipt.json", "reviewed")
    assert target.read_bytes() == b"unexpected = 3\n"
