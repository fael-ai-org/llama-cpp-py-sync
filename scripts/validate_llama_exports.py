#!/usr/bin/env python3

import argparse
import ctypes
import platform
import re
import shutil
import subprocess
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_library_path(project_root: Path) -> Path:
    pkg_dir = project_root / "src" / "llama_cpp_py_sync"
    system = platform.system().lower()

    if system == "windows":
        names = ["llama.dll", "libllama.dll"]
    elif system == "darwin":
        names = ["libllama.dylib", "libllama.so"]
    else:
        names = ["libllama.so"]

    for name in names:
        p = pkg_dir / name
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not locate built llama.cpp shared library in the package directory. "
        "Expected it next to the Python package under src/llama_cpp_py_sync/."
    )


def _default_mtmd_library_path(project_root: Path) -> Path:
    pkg_dir = project_root / "src" / "llama_cpp_py_sync"
    system = platform.system().lower()
    if system == "windows":
        names = ["mtmd.dll", "libmtmd.dll"]
    elif system == "darwin":
        names = ["libmtmd.dylib", "libmtmd.so"]
    else:
        names = ["libmtmd.so"]
    for name in names:
        path = pkg_dir / name
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate the packaged mtmd shared library")


def _load_library(lib_path: Path) -> ctypes.CDLL:
    system = platform.system().lower()

    if system == "windows":
        try:
            if hasattr(ctypes, "WinDLL"):
                return ctypes.WinDLL(str(lib_path))
        except Exception:
            pass

    return ctypes.CDLL(str(lib_path))


def _windows_dumpbin_exports_text(lib_path: Path) -> str | None:
    if platform.system().lower() != "windows":
        return None

    dumpbin = shutil.which("dumpbin")
    if not dumpbin:
        return None

    try:
        proc = subprocess.run(
            [dumpbin, "/exports", str(lib_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return None

    return proc.stdout


def _require_symbol(lib: ctypes.CDLL, name: str) -> None:
    try:
        getattr(lib, name)
    except AttributeError as e:
        raise RuntimeError(f"Missing required export: {name}") from e


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate llama.cpp shared library exports")
    parser.add_argument(
        "--lib",
        type=Path,
        default=None,
        help="Path to the built llama shared library (defaults to src/llama_cpp_py_sync/*llama*)",
    )
    parser.add_argument(
        "--mtmd-lib",
        type=Path,
        default=None,
        help="Path to the built mtmd shared library",
    )

    args = parser.parse_args()

    root = _project_root()
    lib_path = args.lib or _default_library_path(root)

    required = [
        "llama_model_default_params",
        "llama_context_default_params",
        "llama_model_load_from_file",
        "llama_model_free",
    ]

    dumpbin_text = _windows_dumpbin_exports_text(lib_path)
    if dumpbin_text is not None:
        missing = [
            sym
            for sym in required
            if re.search(rf"\b{re.escape(sym)}\b", dumpbin_text) is None
        ]
        if missing:
            raise RuntimeError(f"Missing required export(s): {', '.join(missing)}")
        mtmd_path = args.mtmd_lib or _default_mtmd_library_path(root)
        mtmd_dumpbin = _windows_dumpbin_exports_text(mtmd_path)
        if mtmd_dumpbin is not None:
            missing_mtmd = [
                sym
                for sym in [
                    "mtmd_context_params_default",
                    "mtmd_init_from_file",
                    "mtmd_free",
                    "mtmd_tokenize",
                    "mtmd_helper_bitmap_init_from_buf",
                    "mtmd_helper_eval_chunk_single",
                ]
                if re.search(rf"\b{re.escape(sym)}\b", mtmd_dumpbin) is None
            ]
            if missing_mtmd:
                raise RuntimeError(f"Missing required mtmd export(s): {', '.join(missing_mtmd)}")
            return 0

    lib = _load_library(lib_path)
    for sym in required:
        _require_symbol(lib, sym)

    mtmd_path = args.mtmd_lib or _default_mtmd_library_path(root)
    mtmd = _load_library(mtmd_path)
    for sym in [
        "mtmd_context_params_default",
        "mtmd_init_from_file",
        "mtmd_free",
        "mtmd_tokenize",
        "mtmd_helper_bitmap_init_from_buf",
        "mtmd_helper_eval_chunk_single",
    ]:
        _require_symbol(mtmd, sym)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
