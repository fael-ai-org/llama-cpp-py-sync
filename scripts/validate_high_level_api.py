#!/usr/bin/env python3
"""Validate that high-level Python uses only names that exist in the CFFI cdef.

After llama.cpp sync, struct fields or function names in ``llama.h`` / ``mtmd``
headers can change. ``validate_cffi_surface.py`` checks header vs cdef; this
script checks that high-level modules only reference:

- ``ctx_params.<field>`` where ``<field>`` exists on ``struct llama_context_params``
  in the cdef
- ``model_params.<field>`` on ``struct llama_model_params``
- ``self._lib.<fn>`` / ``lib.<fn>`` where ``<fn>`` appears as a function in the cdef
- Direct ``_lib.<fn>(...)`` calls that pass at least as many positional arguments
  as the cdef signature (starred calls such as ``fn(*args)`` are skipped)

This does not require a vendored ``llama.h`` — the cdef is the ABI contract for
the Python package. Run with the same tree you ship (committed ``_cffi_bindings.py``).

Usage:
    python scripts/validate_high_level_api.py
    python scripts/validate_high_level_api.py --module src/llama_cpp_py_sync/embeddings.py
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import sys
from pathlib import Path


_LIB_NAME_RE = r"(?:llama|mtmd)_[a-zA-Z0-9_]+"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_validate_cffi():
    path = Path(__file__).resolve().parent / "validate_cffi_surface.py"
    spec = importlib.util.spec_from_file_location("validate_cffi_surface", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _extract_lib_api_names(py_source: str) -> set[str]:
    """Collect llama_* / mtmd_* API names referenced from Python source."""
    names: set[str] = set()

    for m in re.finditer(rf"\bself\._lib\.({_LIB_NAME_RE})\b", py_source):
        names.add(m.group(1))

    for m in re.finditer(rf"\blib\.({_LIB_NAME_RE})\b", py_source):
        names.add(m.group(1))

    for m in re.finditer(
        rf"\b(?:getattr|hasattr)\s*\(\s*[^,]+,\s*[\"']({_LIB_NAME_RE})[\"']\s*\)",
        py_source,
    ):
        names.add(m.group(1))

    for m in re.finditer(
        rf"\bget_\w+\s*=\s*getattr\s*\(\s*[^,]+,\s*[\"']({_LIB_NAME_RE})[\"']",
        py_source,
    ):
        names.add(m.group(1))

    return names


def _extract_ctx_param_fields(py_source: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"\bctx_params\.([a-zA-Z0-9_]+)\b", py_source)}


def _extract_model_param_fields(py_source: str) -> set[str]:
    return {m.group(1) for m in re.finditer(r"\bmodel_params\.([a-zA-Z0-9_]+)\b", py_source)}


def _count_c_args(arglist: str) -> int:
    text = " ".join(arglist.strip().split())
    if not text or text == "void":
        return 0
    depth = 0
    count = 1
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            count += 1
    return count


def _extract_cdef_function_arities(vcs, cdef_text: str) -> dict[str, int]:
    cdef_text = vcs._strip_c_comments(cdef_text)
    arities: dict[str, int] = {}
    for stmt in vcs._iter_c_statements(cdef_text):
        source = stmt.strip()
        if not source or "(" not in source or ")" not in source:
            continue
        lowered = source.lstrip().lower()
        if lowered.startswith("typedef "):
            continue
        if "{" in source or "}" in source:
            continue
        paren = source.find("(")
        before = " ".join(source[:paren].split())
        idents = vcs._IDENT_RE.findall(before)
        if not idents:
            continue
        close = source.rfind(")")
        if close <= paren:
            continue
        arities[idents[-1]] = _count_c_args(source[paren + 1 : close])
    return arities


def _attr_chain(node: ast.AST) -> str | None:
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _direct_lib_call_arities(py_source: str) -> dict[str, list[int]]:
    tree = ast.parse(py_source)
    found: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            continue
        chain = _attr_chain(node.func)
        if not chain:
            continue
        parts = chain.split(".")
        if len(parts) < 2 or parts[-2] not in {"_lib", "lib"}:
            continue
        name = parts[-1]
        if not re.fullmatch(_LIB_NAME_RE, name):
            continue
        found.setdefault(name, []).append(len(node.args))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate high-level Python against CFFI cdef struct fields and functions"
    )
    parser.add_argument(
        "--bindings",
        type=Path,
        default=None,
        help="Path to _cffi_bindings.py (default: src/.../_cffi_bindings.py)",
    )
    parser.add_argument(
        "--module",
        action="append",
        default=None,
        help="Extra Python file to scan (repeatable). llama.py, embeddings.py, and multimodal.py are always scanned.",
    )
    args = parser.parse_args()

    root = _project_root()
    vcs = _load_validate_cffi()
    bindings_path = args.bindings or (root / "src" / "llama_cpp_py_sync" / "_cffi_bindings.py")
    bindings_text = bindings_path.read_text(encoding="utf-8", errors="ignore")
    cdef_text = vcs._extract_cdef_text(bindings_text)
    cdef_funcs = vcs._extract_cdef_functions(cdef_text)
    cdef_structs = vcs._extract_structs(cdef_text)
    cdef_arities = _extract_cdef_function_arities(vcs, cdef_text)

    modules = [
        root / "src" / "llama_cpp_py_sync" / "llama.py",
        root / "src" / "llama_cpp_py_sync" / "embeddings.py",
        root / "src" / "llama_cpp_py_sync" / "multimodal.py",
    ]
    if args.module:
        for item in args.module:
            path = Path(item) if Path(item).is_absolute() else root / item
            if path not in modules:
                modules.append(path)

    combined = "\n\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in modules)

    ctx_used = _extract_ctx_param_fields(combined)
    model_used = _extract_model_param_fields(combined)
    lib_used = _extract_lib_api_names(combined)
    direct_calls = _direct_lib_call_arities(combined)

    ctx_fields = cdef_structs.get("llama_context_params", set())
    model_fields = cdef_structs.get("llama_model_params", set())

    bad_ctx = sorted(ctx_used - ctx_fields)
    bad_model = sorted(model_used - model_fields)
    bad_lib = sorted(lib_used - cdef_funcs)
    bad_arity: list[str] = []
    for name, counts in sorted(direct_calls.items()):
        expected = cdef_arities.get(name)
        if expected is None:
            continue
        shortest = min(counts)
        if shortest < expected:
            bad_arity.append(f"{name}: Python passes {shortest} args, cdef expects {expected}")

    print(f"Scanned {len(modules)} module(s); cdef functions: {len(cdef_funcs)}")
    print(f"ctx_params.* fields used: {sorted(ctx_used)}")
    print(f"model_params.* fields used: {sorted(model_used)}")
    print(f"lib API symbols used: {len(lib_used)}")

    ok = True
    if bad_ctx:
        ok = False
        print("\nERROR: ctx_params fields not in cdef struct llama_context_params:")
        for item in bad_ctx:
            print(f"  - {item}")
    if bad_model:
        ok = False
        print("\nERROR: model_params fields not in cdef struct llama_model_params:")
        for item in bad_model:
            print(f"  - {item}")
    if bad_lib:
        ok = False
        print("\nERROR: llama_*/mtmd_* calls not found as cdef functions:")
        for item in bad_lib:
            print(f"  - {item}")
    if bad_arity:
        ok = False
        print("\nERROR: high-level native calls pass fewer arguments than the cdef:")
        for item in bad_arity:
            print(f"  - {item}")

    if ok:
        print("\nOK: high-level references match cdef.")
        return 0
    print(
        "\nFix: sync/regenerate _cffi_bindings.py or update the high-level wrapper to match the cdef.\n"
        "Also run: python scripts/validate_cffi_surface.py --check-structs ... against vendor llama.h",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
