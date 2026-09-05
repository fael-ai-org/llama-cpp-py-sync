import argparse
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from shutil import rmtree


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


# Wheels and other throwaway build outputs (not pytest's ``tests/`` package).
_ARTIFACTS_DIR_NAME = "test_temp"

# Load only ``_cffi_bindings`` without importing ``llama_cpp_py_sync.__init__`` (which pulls
# numpy/embeddings). Smoke uses ``pip install --no-deps`` so pip does not reinstall NumPy
# from cache (can trigger Windows Defender / Application Control on .pyd under strict policies).
_SMOKE_LOAD_NATIVE = """
import importlib.util
import site
from pathlib import Path

def _main() -> None:
    roots = list(site.getsitepackages())
    u = getattr(site, "getusersitepackages", lambda: None)()
    if u:
        roots.append(u)
    for sp in roots:
        p = Path(sp) / "llama_cpp_py_sync" / "_cffi_bindings.py"
        if p.is_file():
            spec = importlib.util.spec_from_file_location(
                "llama_cpp_py_sync._cffi_bindings",
                p,
            )
            mod = importlib.util.module_from_spec(spec)
            loader = spec.loader
            if loader is None:
                raise SystemExit("smoke: invalid module spec")
            loader.exec_module(mod)
            llama = mod.get_lib()
            mod.get_mtmd_lib()
            if llama.llama_supports_rpc():
                raise SystemExit("smoke: llama_supports_rpc() is true")
            print("smoke-ok")
            return
    raise SystemExit("smoke: llama_cpp_py_sync/_cffi_bindings.py not found under site-packages")

_main()
"""


def _artifacts_dir(root: Path) -> Path:
    """Ephemeral wheels live under ``test_temp/dist`` at repo root; cleaned after runs."""
    d = root / _ARTIFACTS_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _repo_venv_python(repo_root: Path) -> Path | None:
    """Use the same interpreter as dev: active venv if it lives under the repo, else ``./venv`` / ``./.venv``."""
    try:
        root = repo_root.resolve()
        cur = Path(sys.executable).resolve()
        cur.relative_to(root)
        return cur
    except ValueError:
        pass
    for name in ("venv", ".venv"):
        py = _venv_python(repo_root / name)
        if py.exists():
            return py
    return None


def _clean_test_temp_dist(repo_root: Path) -> None:
    dist = repo_root / _ARTIFACTS_DIR_NAME / "dist"
    if not dist.is_dir():
        return
    for p in dist.iterdir():
        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                rmtree(p, ignore_errors=True)
        except OSError:
            pass


def _run(cmd: list[str], cwd: Path) -> int:
    print(f"\n$ {' '.join(cmd)}")
    sys.stdout.flush()
    proc = subprocess.run(cmd, cwd=str(cwd))
    return int(proc.returncode)


def _require_module(module: str, install_hint: str) -> bool:
    if find_spec(module) is not None:
        return True
    print(f"Missing dependency: {module}")
    print("Install it with:")
    print(f"  {install_hint}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local checks before pushing changes.")
    parser.add_argument("--skip-wheel", action="store_true", help="Skip building a wheel.")
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip wheel smoke test (install wheel into repo venv, load native lib).",
    )
    parser.add_argument(
        "--skip-bindings-check",
        action="store_true",
        help="Skip validate_cffi_surface.py and validate_high_level_api.py checks.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()

    python_exe = sys.executable
    if os.environ.get("VIRTUAL_ENV") is None:
        print("Warning: VIRTUAL_ENV is not set. Consider activating venv before running this script.")
        print(f"Using python: {python_exe}")

    pip_install = f"{python_exe} -m pip install -r requirements.txt"

    if not _require_module("ruff", pip_install):
        return 1
    if not _require_module("pytest", pip_install):
        return 1
    if not args.skip_wheel and not _require_module("build", pip_install):
        return 1

    checks: list[list[str]] = [
        [python_exe, "-m", "ruff", "check", "."],
        [python_exe, "-m", "pytest"],
    ]

    if not args.skip_bindings_check:
        checks.append(
            [
                python_exe,
                str(repo_root / "scripts" / "validate_cffi_surface.py"),
                "--check-structs",
                "--check-enums",
                "--check-signatures",
            ]
        )
        checks.append(
            [
                python_exe,
                str(repo_root / "scripts" / "validate_high_level_api.py"),
            ]
        )

    if not args.skip_wheel:
        wheel_out = _artifacts_dir(repo_root) / "dist"
        wheel_out.mkdir(parents=True, exist_ok=True)
        checks.append(
            [
                python_exe,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(wheel_out),
            ]
        )

    for cmd in checks:
        rc = _run(cmd, cwd=repo_root)
        if rc != 0:
            print(f"\nFailed: {' '.join(cmd)}")
            _clean_test_temp_dist(repo_root)
            return rc

    if not args.skip_smoke:
        dist_dir = _artifacts_dir(repo_root) / "dist"
        wheels = sorted(dist_dir.glob("*.whl"))
        if not wheels:
            print(
                f"\nNo wheels found in {_ARTIFACTS_DIR_NAME}/dist/. "
                "Run without --skip-wheel or build a wheel first."
            )
            _clean_test_temp_dist(repo_root)
            return 1

        vpy = _repo_venv_python(repo_root)
        if vpy is None:
            print(
                "\nWheel smoke test needs a repo virtualenv at ./venv or ./.venv "
                "(same environment as normal development).\n"
                "Create one: python -m venv venv\n"
                "Or pass --skip-smoke."
            )
            _clean_test_temp_dist(repo_root)
            return 1

        wheel_path = str(wheels[-1])
        try:
            rc = _run(
                [
                    str(vpy),
                    "-m",
                    "pip",
                    "install",
                    "--force-reinstall",
                    "--no-deps",
                    wheel_path,
                ],
                cwd=repo_root,
            )
            if rc != 0:
                print("\nWheel install into repo venv failed.")
                return rc

            rc = _run([str(vpy), "-c", _SMOKE_LOAD_NATIVE.strip()], cwd=repo_root)
            if rc != 0:
                print("\nWheel smoke test failed (native library load).")
                return rc
        finally:
            _run([str(vpy), "-m", "pip", "install", "-e", "."], cwd=repo_root)
            _clean_test_temp_dist(repo_root)

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
