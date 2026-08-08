#!/usr/bin/env python3

"""Validate that the Python CFFI surface matches llama.cpp's public C API.

What it checks:
- Public API (LLAMA_API) functions in vendor llama.h are present in the CFFI cdef.
- Optionally, that those functions are also exported by the built shared library.

This is intended to be run after syncing/updating vendor/llama.cpp and/or
regenerating bindings.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# Keep these in sync with the rewrites in scripts/gen_bindings.py preprocess_header().
# (Currently we aim to preserve types and resolve missing definitions from vendored headers.)
_SIG_TYPE_REWRITES: list[tuple[str, str]] = []


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_header_path(project_root: Path) -> Path:
    p = project_root / "vendor" / "llama.cpp" / "include" / "llama.h"
    if not p.exists():
        raise FileNotFoundError(f"Could not find llama.h at: {p}")
    return p


def _default_mtmd_header_paths(project_root: Path) -> list[Path]:
    mtmd_dir = project_root / "vendor" / "llama.cpp" / "tools" / "mtmd"
    return [
        path for path in (mtmd_dir / "mtmd.h", mtmd_dir / "mtmd-helper.h") if path.exists()
    ]


def _default_cffi_bindings_path(project_root: Path) -> Path:
    p = project_root / "src" / "llama_cpp_py_sync" / "_cffi_bindings.py"
    if not p.exists():
        raise FileNotFoundError(f"Could not find bindings file at: {p}")
    return p


def _default_library_path(project_root: Path) -> Path | None:
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

    return None


def _default_mtmd_library_path(project_root: Path) -> Path | None:
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
    return None


def _extract_cdef_text(bindings_py: str) -> str:
    # Keep this strict: if generation changes the marker, we want to notice.
    m = re.search(r"^_LLAMA_H_CDEF\s*=\s*\"\"\"\n(.*?)\n\"\"\"\s*$", bindings_py, re.DOTALL | re.MULTILINE)
    if not m:
        raise RuntimeError("Could not locate _LLAMA_H_CDEF triple-quoted string")
    return m.group(1)


def _strip_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    return text


def _strip_preprocessor_lines(text: str) -> str:
    return re.sub(r"^\s*#.*$", "", text, flags=re.MULTILINE)


def _slice_c_api_region(text: str) -> str:
    # Restrict each public header to its C API and stop before the C++ wrappers
    # that follow mtmd's extern-C block. This also supports concatenated headers.
    marker = 'extern "C"'
    regions: list[str] = []
    search_from = 0
    while True:
        idx = text.find(marker, search_from)
        if idx == -1:
            break
        end = text.find("#ifdef __cplusplus", idx + len(marker))
        regions.append(text[idx:end] if end != -1 else text[idx:])
        if end == -1:
            break
        search_from = end
    return "\n".join(regions) if regions else text


def _iter_named_blocks(text: str, start_re: re.Pattern[str]) -> Iterable[tuple[str, str]]:
    """Yield (name, body) for blocks like `keyword name { ... }`.

    `start_re` must have a single capturing group for the name, and match up to the '{'.
    """
    for m in start_re.finditer(text):
        name = m.group(1)
        brace_open = text.find("{", m.end() - 1)
        if brace_open < 0:
            continue

        depth = 0
        j = brace_open
        while j < len(text):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j >= len(text):
            continue

        body = text[brace_open + 1 : j]
        yield name, body


def _extract_enums(text: str) -> dict[str, set[str]]:
    text = _strip_preprocessor_lines(_strip_c_comments(text))
    enums: dict[str, set[str]] = {}

    enum_start = re.compile(r"\benum\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{")
    for name, body in _iter_named_blocks(text, enum_start):
        members: set[str] = set()
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # Remove trailing commas and strip initializers.
            line = line.split("//", 1)[0].strip()
            line = re.sub(r"\s*=\s*[^,}]+", "", line).strip()
            line = line.rstrip(",").strip()
            if not line:
                continue
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\b", line)
            if m:
                members.add(m.group(1))

        enums[name] = members

    return enums


def _extract_struct_fields_from_body(body: str) -> set[str]:
    # Extract field names from top-level ';'-terminated declarations.
    fields: set[str] = set()

    stmt = ""
    paren_depth = 0
    brace_depth = 0

    for ch in body:
        stmt += ch
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            if paren_depth > 0:
                paren_depth -= 1
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            if brace_depth > 0:
                brace_depth -= 1

        if ch == ";" and paren_depth == 0 and brace_depth == 0:
            s = " ".join(_strip_c_comments(stmt).split())
            stmt = ""
            if not s:
                continue
            # Skip nested type declarations.
            if "{" in s or "}" in s:
                continue
            lowered = s.lstrip().lower()
            # Skip nested type definitions (they contain '{'); keep members like
            # `enum llama_pooling_type pooling_type;` (no '{' in the declaration).
            if lowered.startswith(("typedef ", "struct ", "enum ", "union ")):
                if "{" in s:
                    continue

            # Function pointer field: capture `(*name)`
            m_fp = re.search(r"\(\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", s)
            if m_fp:
                fields.add(m_fp.group(1))
                continue

            # Otherwise, last identifier before ';' (after stripping arrays/bitfields).
            s2 = s.rstrip(";")
            s2 = re.sub(r"\[[^\]]*\]", "", s2)  # arrays
            s2 = re.sub(r"\s*:\s*\d+\s*$", "", s2)  # bitfields
            idents = _IDENT_RE.findall(s2)
            if idents:
                fields.add(idents[-1])

    return fields


def _extract_structs(text: str) -> dict[str, set[str]]:
    text = _strip_preprocessor_lines(_strip_c_comments(text))
    structs: dict[str, set[str]] = {}

    # Match `struct name { ... };`
    struct_start = re.compile(r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{")
    for name, body in _iter_named_blocks(text, struct_start):
        structs[name] = _extract_struct_fields_from_body(body)

    # Match `typedef struct name { ... } alias;` and `typedef struct { ... } alias;`
    i = 0
    needle = "typedef struct"
    while True:
        start = text.find(needle, i)
        if start < 0:
            break

        brace_open = text.find("{", start)
        if brace_open < 0:
            break

        depth = 0
        j = brace_open
        while j < len(text):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1

        if j >= len(text):
            break

        brace_close = j
        semi = text.find(";", brace_close)
        if semi < 0:
            break

        after = text[brace_close + 1 : semi]
        m_name = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*$", after.strip())
        if m_name:
            typedef_name = m_name.group(1)
            body = text[brace_open + 1 : brace_close]
            structs[typedef_name] = _extract_struct_fields_from_body(body)

        i = semi + 1

    return structs


def _iter_public_api_function_decls(header_text: str) -> Iterable[str]:
    # We intentionally scan from LLAMA_API occurrences, because the header uses
    # conditional wrappers (e.g. DEPRECATED(...)). This keeps it robust.
    i = 0
    n = len(header_text)
    while True:
        matches = [
            match
            for match in (
                header_text.find("LLAMA_API", i),
                header_text.find("MTMD_API", i),
            )
            if match >= 0
        ]
        if not matches:
            return
        idx = min(matches)

        j = idx
        paren_depth = 0
        saw_paren = False

        while j < n:
            ch = header_text[j]
            if ch == "(":
                paren_depth += 1
                saw_paren = True
            elif ch == ")":
                if paren_depth > 0:
                    paren_depth -= 1
            elif ch == ";" and saw_paren and paren_depth == 0:
                yield header_text[idx : j + 1]
                i = j + 1
                break
            j += 1

        # Safety: if we failed to find a terminator, move forward.
        if j >= n:
            i = idx + 8


def _decl_to_func_name(decl: str) -> str | None:
    decl_one_line = " ".join(decl.replace("\r", "").split())

    # Ignore non-function statements.
    if "(" not in decl_one_line or ")" not in decl_one_line:
        return None

    # Some declarations are wrapped like:
    #   LLAMA_API DEPRECATED(ret_t func(args), "hint");
    # In that case, the first '(' is the DEPRECATED macro, not the function params.
    if re.search(r"\bDEPRECATED\s*\(", decl_one_line):
        # If the token after LLAMA_API is DEPRECATED, parse the first macro argument.
        after_api = decl_one_line
        if "LLAMA_API" in after_api:
            after_api = after_api.split("LLAMA_API", 1)[1].strip()
        if after_api.startswith("DEPRECATED"):
            m = re.search(r"\bDEPRECATED\s*\(", decl_one_line)
            if m:
                k = m.end()  # position after 'DEPRECATED('
                depth = 0
                arg_start = k
                while k < len(decl_one_line):
                    ch = decl_one_line[k]
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        if depth > 0:
                            depth -= 1
                    elif ch == "," and depth == 0:
                        break
                    k += 1
                first_arg = decl_one_line[arg_start:k].strip()
                if "(" in first_arg:
                    p = first_arg.find("(")
                    left = first_arg[:p]
                    idents = _IDENT_RE.findall(left)
                    if idents:
                        return idents[-1]

    paren = decl_one_line.find("(")
    before = decl_one_line[:paren]

    # If present, keep only the segment after LLAMA_API (e.g. DEPRECATED(LLAMA_API ...)
    # or plain LLAMA_API ...).
    if "LLAMA_API" in before:
        before = before.split("LLAMA_API", 1)[1].strip()

    # Drop leading wrappers like 'DEPRECATED(' that might remain.
    before = before.lstrip("(").strip()

    idents = _IDENT_RE.findall(before)
    if not idents:
        return None

    # The function name is the last identifier before '('.
    return idents[-1]


def _extract_header_public_functions(header_text: str) -> set[str]:
    header_text = _slice_c_api_region(_strip_c_comments(header_text))
    header_text = _strip_preprocessor_lines(header_text)
    funcs: set[str] = set()

    for decl in _iter_public_api_function_decls(header_text):
        if "std::" in decl or "::" in decl:
            continue
        name = _decl_to_func_name(decl)
        if name:
            funcs.add(name)

    return funcs


def _iter_c_statements(text: str) -> Iterable[str]:
    # Splits into ';'-terminated statements, tracking parentheses so we don't split
    # inside parameter lists, and tracking braces so we don't treat struct fields
    # as top-level statements.
    i = 0
    n = len(text)
    stmt_start = 0
    paren_depth = 0
    brace_depth = 0

    while i < n:
        ch = text[i]
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            if paren_depth > 0:
                paren_depth -= 1
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            if brace_depth > 0:
                brace_depth -= 1
        elif ch == ";" and paren_depth == 0 and brace_depth == 0:
            yield text[stmt_start : i + 1]
            stmt_start = i + 1
        i += 1


def _extract_cdef_functions(cdef_text: str) -> set[str]:
    cdef_text = _strip_c_comments(cdef_text)
    funcs: set[str] = set()

    for stmt in _iter_c_statements(cdef_text):
        s = stmt.strip()
        if not s or "(" not in s or ")" not in s:
            continue

        lowered = s.lstrip().lower()
        if lowered.startswith("typedef "):
            continue
        if "{" in s or "}" in s:
            continue

        paren = s.find("(")
        before = " ".join(s[:paren].split())
        idents = _IDENT_RE.findall(before)
        if not idents:
            continue

        name = idents[-1]
        # Filter out likely macro remnants or invalid names.
        if name in {"if", "for", "while"}:
            continue

        funcs.add(name)

    return funcs


def _normalize_c_type(text: str) -> str:
    text = " ".join(text.strip().split())
    # Normalize pointer spacing.
    text = re.sub(r"\s*\*\s*", "*", text)
    # Normalize commas.
    text = re.sub(r"\s*,\s*", ",", text)
    return text


def _strip_param_name(param: str) -> str:
    p = " ".join(param.strip().split())
    if not p or p == "void" or p == "...":
        return p

    # Remove the identifier in function-pointer parameters: `(*name)` -> `(*)`.
    p = re.sub(r"\(\s*\*\s*[A-Za-z_][A-Za-z0-9_]*\s*\)", "(*)", p)
    if "(*)" in p:
        return p

    # Remove trailing array dimensions but keep them attached to the type.
    array_suffix = ""
    m_arr = re.search(r"(\s*\[[^\]]*\]\s*)+$", p)
    if m_arr:
        array_suffix = _normalize_c_type(m_arr.group(0))
        p = p[: m_arr.start()].rstrip()

    # Heuristic: drop the last identifier as the parameter name when it looks like
    # `type name` (common for C prototypes).
    m = re.match(r"^(?P<type>.+?)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*$", p)
    if m and not m.group("type").rstrip().endswith(")"):
        p = m.group("type")

    return _normalize_c_type(p + array_suffix)


def _split_params(param_list: str) -> list[str]:
    params: list[str] = []
    cur = ""
    paren_depth = 0
    brace_depth = 0

    for ch in param_list:
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            if paren_depth > 0:
                paren_depth -= 1
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            if brace_depth > 0:
                brace_depth -= 1

        if ch == "," and paren_depth == 0 and brace_depth == 0:
            params.append(cur)
            cur = ""
        else:
            cur += ch

    if cur.strip():
        params.append(cur)

    return [p.strip() for p in params if p.strip()]


def _unwrap_deprecated_decl(decl_one_line: str) -> str:
    # If the declaration is wrapped like `DEPRECATED(<decl>, "hint")`, keep only <decl>.
    m = re.search(r"\bDEPRECATED\s*\(", decl_one_line)
    if not m:
        return decl_one_line

    k = m.end()
    depth = 0
    arg_start = k
    while k < len(decl_one_line):
        ch = decl_one_line[k]
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth > 0:
                depth -= 1
        elif ch == "," and depth == 0:
            break
        k += 1
    first_arg = decl_one_line[arg_start:k].strip()
    return first_arg if first_arg else decl_one_line


def _signature_from_prototype(proto: str) -> tuple[str, str] | None:
    # Returns (name, normalized_signature).
    s = " ".join(proto.replace("\r", "").split()).strip().rstrip(";")
    if not s or "(" not in s or ")" not in s:
        return None

    # Find function name as the last identifier before the first '('.
    paren = s.find("(")
    before = s[:paren]
    idents = _IDENT_RE.findall(before)
    if not idents:
        return None
    name = idents[-1]

    # Extract return type portion.
    m_name = re.search(rf"\b{re.escape(name)}\s*$", before)
    if not m_name:
        return None
    ret = before[: m_name.start()].strip()
    ret = _normalize_c_type(ret)

    # Extract parameter list, respecting nested parentheses.
    i = s.find("(", m_name.end())
    if i < 0:
        return None
    depth = 0
    j = i
    while j < len(s):
        ch = s[j]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        j += 1
    if j >= len(s):
        return None
    params_raw = s[i + 1 : j].strip()

    params = []
    if params_raw and params_raw != "void":
        for p in _split_params(params_raw):
            params.append(_strip_param_name(p))

    sig = f"{ret}({','.join(params)})"
    return name, sig


def _extract_header_function_signatures(header_text: str) -> dict[str, str]:
    header_text = _slice_c_api_region(_strip_c_comments(header_text))
    header_text = _strip_preprocessor_lines(header_text)
    sigs: dict[str, str] = {}

    for decl in _iter_public_api_function_decls(header_text):
        if "std::" in decl or "::" in decl:
            continue
        decl_one_line = " ".join(decl.replace("\r", "").split())
        decl_one_line = _unwrap_deprecated_decl(decl_one_line)
        decl_one_line = (
            decl_one_line.replace("LLAMA_API ", "")
            .replace(" LLAMA_API ", " ")
            .replace("MTMD_API ", "")
            .replace(" MTMD_API ", " ")
        )

        for pattern, repl in _SIG_TYPE_REWRITES:
            decl_one_line = re.sub(pattern, repl, decl_one_line)

        parsed = _signature_from_prototype(decl_one_line)
        if parsed:
            name, sig = parsed
            sigs[name] = sig

    return sigs


def _extract_cdef_function_signatures(cdef_text: str) -> dict[str, str]:
    cdef_text = _strip_c_comments(cdef_text)
    sigs: dict[str, str] = {}

    for stmt in _iter_c_statements(cdef_text):
        s = stmt.strip()
        if not s or "(" not in s or ")" not in s:
            continue

        lowered = s.lstrip().lower()
        if lowered.startswith("typedef "):
            continue
        if "{" in s or "}" in s:
            continue

        parsed = _signature_from_prototype(s)
        if parsed:
            name, sig = parsed
            sigs[name] = sig

    return sigs


def _load_library(lib_path: Path) -> ctypes.CDLL:
    system = platform.system().lower()

    if system == "windows":
        lib_dir = str(lib_path.parent)
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(lib_dir)
        except Exception:
            pass
        try:
            os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass

        try:
            if hasattr(ctypes, "WinDLL"):
                return ctypes.WinDLL(str(lib_path))
        except Exception:
            pass

    return ctypes.CDLL(str(lib_path))


def _exported_symbol_exists(lib: ctypes.CDLL, name: str) -> bool:
    try:
        getattr(lib, name)
        return True
    except AttributeError:
        return False


@dataclass
class Report:
    header_function_count: int
    cdef_function_count: int
    missing_in_cdef: list[str]
    extra_in_cdef: list[str]
    missing_exports: list[str]
    function_signature_mismatches: dict[str, dict[str, str]]
    header_enum_count: int
    cdef_enum_count: int
    missing_enums: list[str]
    enum_member_mismatches: dict[str, dict[str, list[str]]]
    header_struct_count: int
    cdef_struct_count: int
    missing_structs: list[str]
    struct_field_mismatches: dict[str, dict[str, list[str]]]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate that llama.cpp public C API matches the Python CFFI surface"
    )
    parser.add_argument(
        "--header",
        type=Path,
        default=None,
        help="Path to vendor llama.h (default: vendor/llama.cpp/include/llama.h)",
    )
    parser.add_argument(
        "--mtmd-header",
        action="append",
        type=Path,
        default=None,
        help="Path to an mtmd public header (repeat for mtmd.h and mtmd-helper.h)",
    )
    parser.add_argument(
        "--bindings",
        type=Path,
        default=None,
        help="Path to src/llama_cpp_py_sync/_cffi_bindings.py",
    )
    parser.add_argument(
        "--check-exports",
        action="store_true",
        help="Also validate that functions exist in the built shared library",
    )
    parser.add_argument(
        "--check-signatures",
        action="store_true",
        help="Also validate function signatures (return + parameter types) match between header and cdef",
    )
    parser.add_argument(
        "--check-enums",
        action="store_true",
        help="Also validate enum names and members match between header and cdef",
    )
    parser.add_argument(
        "--check-structs",
        action="store_true",
        help="Also validate struct names and top-level fields match between header and cdef",
    )
    parser.add_argument(
        "--lib",
        type=Path,
        default=None,
        help="Path to the built shared library (default: src/llama_cpp_py_sync/*llama*)",
    )
    parser.add_argument(
        "--mtmd-lib",
        type=Path,
        default=None,
        help="Path to the built mtmd shared library",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if CFFI contains extra functions not found in the header",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write a JSON report to this path",
    )

    args = parser.parse_args()

    root = _project_root()
    header_path = args.header or _default_header_path(root)
    bindings_path = args.bindings or _default_cffi_bindings_path(root)

    header_paths = [header_path]
    header_paths.extend(args.mtmd_header or _default_mtmd_header_paths(root))
    header_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in header_paths if path.exists()
    )
    bindings_text = bindings_path.read_text(encoding="utf-8", errors="ignore")
    cdef_text = _extract_cdef_text(bindings_text)

    header_funcs = _extract_header_public_functions(header_text)
    cdef_funcs = _extract_cdef_functions(cdef_text)

    missing_in_cdef = sorted(header_funcs - cdef_funcs)
    extra_in_cdef = sorted(cdef_funcs - header_funcs)

    missing_exports: list[str] = []
    if args.check_exports:
        lib_path = args.lib or _default_library_path(root)
        mtmd_lib_path = args.mtmd_lib or _default_mtmd_library_path(root)
        if lib_path is None or not lib_path.exists():
            raise FileNotFoundError(
                "--check-exports was set, but no built library was found. "
                "Build first (scripts/build_llama_cpp.py) or pass --lib."
            )
        if mtmd_lib_path is None or not mtmd_lib_path.exists():
            raise FileNotFoundError(
                "--check-exports was set, but no built mtmd library was found. "
                "Build first (scripts/build_llama_cpp.py) or pass --mtmd-lib."
            )
        lib = _load_library(lib_path)
        mtmd_lib = _load_library(mtmd_lib_path)
        missing_exports = sorted(
            [
                fn
                for fn in header_funcs
                if not _exported_symbol_exists(lib if fn.startswith("llama_") else mtmd_lib, fn)
            ]
        )

    function_signature_mismatches: dict[str, dict[str, str]] = {}
    if args.check_signatures:
        header_sigs = _extract_header_function_signatures(header_text)
        cdef_sigs = _extract_cdef_function_signatures(cdef_text)
        for name, hsig in header_sigs.items():
            if name not in cdef_sigs:
                continue
            csig = cdef_sigs[name]
            if hsig != csig:
                function_signature_mismatches[name] = {"header": hsig, "cdef": csig}

    header_enums: dict[str, set[str]] = {}
    cdef_enums: dict[str, set[str]] = {}
    missing_enums: list[str] = []
    enum_member_mismatches: dict[str, dict[str, list[str]]] = {}
    if args.check_enums:
        header_enums = _extract_enums(_slice_c_api_region(header_text))
        cdef_enums = _extract_enums(cdef_text)
        missing_enums = sorted(set(header_enums.keys()) - set(cdef_enums.keys()))
        for name, members in header_enums.items():
            if name not in cdef_enums:
                continue
            missing_members = sorted(members - cdef_enums[name])
            # Extra members in cdef can appear due to parsing differences; missing is what breaks coverage.
            if missing_members:
                enum_member_mismatches[name] = {
                    "missing": missing_members,
                }

    header_structs: dict[str, set[str]] = {}
    cdef_structs: dict[str, set[str]] = {}
    missing_structs: list[str] = []
    struct_field_mismatches: dict[str, dict[str, list[str]]] = {}
    if args.check_structs:
        header_structs = _extract_structs(_slice_c_api_region(header_text))
        cdef_structs = _extract_structs(cdef_text)
        missing_structs = sorted(set(header_structs.keys()) - set(cdef_structs.keys()))
        for name, fields in header_structs.items():
            if name not in cdef_structs:
                continue
            missing_fields = sorted(fields - cdef_structs[name])
            # Extra fields in cdef can appear due to parsing differences; missing is what breaks ABI usability.
            if missing_fields:
                struct_field_mismatches[name] = {
                    "missing": missing_fields,
                }

    report = Report(
        header_function_count=len(header_funcs),
        cdef_function_count=len(cdef_funcs),
        missing_in_cdef=missing_in_cdef,
        extra_in_cdef=extra_in_cdef,
        missing_exports=missing_exports,
        function_signature_mismatches=function_signature_mismatches,
        header_enum_count=len(header_enums),
        cdef_enum_count=len(cdef_enums),
        missing_enums=missing_enums,
        enum_member_mismatches=enum_member_mismatches,
        header_struct_count=len(header_structs),
        cdef_struct_count=len(cdef_structs),
        missing_structs=missing_structs,
        struct_field_mismatches=struct_field_mismatches,
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.__dict__, indent=2) + "\n", encoding="utf-8")

    print(f"Header public functions: {report.header_function_count}")
    print(f"CFFI cdef functions:     {report.cdef_function_count}")

    if args.check_enums:
        print(f"Header enums:           {report.header_enum_count}")
        print(f"CFFI cdef enums:        {report.cdef_enum_count}")

    if args.check_structs:
        print(f"Header structs:         {report.header_struct_count}")
        print(f"CFFI cdef structs:      {report.cdef_struct_count}")

    if report.missing_in_cdef:
        print(f"\nMissing in CFFI ({len(report.missing_in_cdef)}):")
        for name in report.missing_in_cdef[:200]:
            print(f"  - {name}")
        if len(report.missing_in_cdef) > 200:
            print(f"  ... ({len(report.missing_in_cdef) - 200} more)")

    if report.extra_in_cdef:
        print(f"\nExtra in CFFI ({len(report.extra_in_cdef)}):")
        for name in report.extra_in_cdef[:200]:
            print(f"  - {name}")
        if len(report.extra_in_cdef) > 200:
            print(f"  ... ({len(report.extra_in_cdef) - 200} more)")

    if report.missing_exports:
        print(f"\nMissing exports in shared library ({len(report.missing_exports)}):")
        for name in report.missing_exports[:200]:
            print(f"  - {name}")
        if len(report.missing_exports) > 200:
            print(f"  ... ({len(report.missing_exports) - 200} more)")

    if args.check_signatures and report.function_signature_mismatches:
        print(f"\nFunction signature mismatches ({len(report.function_signature_mismatches)}):")
        for name in sorted(report.function_signature_mismatches.keys())[:50]:
            print(f"  - {name}")
        if len(report.function_signature_mismatches) > 50:
            print(f"  ... ({len(report.function_signature_mismatches) - 50} more)")

    if args.check_enums:
        if report.missing_enums:
            print(f"\nMissing enums in CFFI ({len(report.missing_enums)}):")
            for name in report.missing_enums[:200]:
                print(f"  - {name}")
            if len(report.missing_enums) > 200:
                print(f"  ... ({len(report.missing_enums) - 200} more)")

        if report.enum_member_mismatches:
            print(f"\nEnum member mismatches ({len(report.enum_member_mismatches)}):")
            for name, diff in list(report.enum_member_mismatches.items())[:50]:
                miss = diff.get("missing", [])
                extra = diff.get("extra", [])
                print(f"  - {name}: missing={len(miss)} extra={len(extra)}")
            if len(report.enum_member_mismatches) > 50:
                print(f"  ... ({len(report.enum_member_mismatches) - 50} more)")

    if args.check_structs:
        if report.missing_structs:
            print(f"\nMissing structs in CFFI ({len(report.missing_structs)}):")
            for name in report.missing_structs[:200]:
                print(f"  - {name}")
            if len(report.missing_structs) > 200:
                print(f"  ... ({len(report.missing_structs) - 200} more)")

        if report.struct_field_mismatches:
            print(f"\nStruct field mismatches ({len(report.struct_field_mismatches)}):")
            for name, diff in list(report.struct_field_mismatches.items())[:50]:
                miss = diff.get("missing", [])
                extra = diff.get("extra", [])
                print(f"  - {name}: missing={len(miss)} extra={len(extra)}")
            if len(report.struct_field_mismatches) > 50:
                print(f"  ... ({len(report.struct_field_mismatches) - 50} more)")

    ok = True
    if report.missing_in_cdef:
        ok = False
    if report.missing_exports:
        ok = False
    if args.strict and report.extra_in_cdef:
        ok = False
    if args.check_signatures and report.function_signature_mismatches:
        ok = False
    if args.check_enums:
        if report.missing_enums or report.enum_member_mismatches:
            ok = False
    if args.check_structs:
        if report.missing_structs or report.struct_field_mismatches:
            ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
