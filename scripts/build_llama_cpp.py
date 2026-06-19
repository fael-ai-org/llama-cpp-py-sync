#!/usr/bin/env python3
"""
Build llama.cpp shared library.

This script compiles llama.cpp into a shared library that can be used
by the Python bindings. It supports multiple backends including CPU,
CUDA, ROCm, Vulkan, Metal, and various BLAS implementations.
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HOMEBREW_INSTALL_URL = "https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh"


def _copy_runtime_dll(src: Path, dst_dir: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst = dst_dir / src.name
    try:
        shutil.copy2(src, dst)
    except Exception:
        return False
    return True


def _copy_msvc_openmp_runtimes(package_dir: Path) -> int:
    candidates: List[Path] = []

    vs_roots: List[Path] = []
    for env_key in ["VCToolsInstallDir", "VCINSTALLDIR", "VSINSTALLDIR", "VSCMD_ARG_VCVARS"]:
        val = os.environ.get(env_key)
        if val:
            vs_roots.append(Path(val))

    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\\Program Files (x86)")
    try:
        vs_roots.extend(
            list(
                Path(program_files_x86).glob(
                    "Microsoft Visual Studio/*/*/VC/Redist/MSVC/*/x64/*"
                )
            )
        )
    except Exception:
        pass

    vc_patterns = [
        "Microsoft.VC*CRT/vcruntime140*.dll",
        "Microsoft.VC*CRT/msvcp140*.dll",
        "Microsoft.VC*CRT/concrt140*.dll",
        "Microsoft.VC*OpenMP/vcomp140*.dll",
    ]

    for root in vs_roots:
        try:
            if root.is_dir():
                for pattern in vc_patterns:
                    for p in root.glob(pattern):
                        if p.is_file():
                            candidates.append(p)
        except Exception:
            pass

    system32 = Path(r"C:\\Windows\\System32")
    for p in [
        system32 / "vcruntime140.dll",
        system32 / "vcruntime140_1.dll",
        system32 / "msvcp140.dll",
        system32 / "msvcp140_1.dll",
        system32 / "msvcp140_atomic_wait.dll",
        system32 / "concrt140.dll",
        system32 / "vcomp140.dll",
    ]:
        if p.exists():
            candidates.append(p)

    copied = 0
    seen = set()
    for src in candidates:
        key = src.name.lower()
        if key in seen:
            continue
        seen.add(key)
        if _copy_runtime_dll(src, package_dir):
            copied += 1
    return copied


def _copy_runtime_dylib(src: Path, dst_dir: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst = dst_dir / src.name
    try:
        shutil.copy2(src, dst)
    except Exception:
        return False
    return True


def _brew_prefix(formula: str) -> Optional[Path]:
    brew = _find_brew()
    if brew is None:
        return None
    try:
        result = subprocess.run(
            [brew, "--prefix", formula],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    prefix = result.stdout.strip()
    if not prefix:
        return None
    path = Path(prefix)
    return path if path.exists() else None


def _macos_vulkan_library_dirs() -> list[Path]:
    dirs: list[Path] = []

    vulkan_sdk = os.environ.get("VULKAN_SDK")
    if vulkan_sdk:
        sdk_root = Path(vulkan_sdk)
        for subdir in ["lib", "Lib", "MoltenVK/macOS", "macOS/lib"]:
            _append_unique_dir(dirs, sdk_root / subdir)

    for prefix_formula in ["vulkan-loader", "molten-vk"]:
        prefix = _brew_prefix(prefix_formula)
        if prefix is not None:
            _append_unique_dir(dirs, prefix / "lib")

    for lib_dir in [Path("/usr/local/lib"), Path("/opt/homebrew/lib")]:
        _append_unique_dir(dirs, lib_dir)

    return dirs


def _macos_vulkan_icd_candidates() -> list[Path]:
    candidates: list[Path] = []

    vulkan_sdk = os.environ.get("VULKAN_SDK")
    if vulkan_sdk:
        sdk_root = Path(vulkan_sdk)
        for subdir in [
            "share/vulkan/icd.d/MoltenVK_icd.json",
            "MoltenVK/icd/MoltenVK_icd.json",
            "macOS/share/vulkan/icd.d/MoltenVK_icd.json",
        ]:
            path = sdk_root / subdir
            if path.exists():
                candidates.append(path)

    for prefix_formula in ["molten-vk", "vulkan-loader"]:
        prefix = _brew_prefix(prefix_formula)
        if prefix is not None:
            icd = prefix / "share" / "vulkan" / "icd.d" / "MoltenVK_icd.json"
            if icd.exists():
                candidates.append(icd)

    for icd in [
        Path("/usr/local/share/vulkan/icd.d/MoltenVK_icd.json"),
        Path("/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json"),
    ]:
        if icd.exists():
            candidates.append(icd)

    return candidates


def _macos_vulkan_dylib_patterns() -> list[str]:
    return ["libvulkan*.dylib", "libMoltenVK*.dylib"]


def _copy_vulkan_runtime_dlls(package_dir: Path) -> int:
    copied = 0
    vulkan_sdk = os.environ.get("VULKAN_SDK")
    vulkan_bin_dir = Path(vulkan_sdk) / "Bin" if vulkan_sdk else None

    if vulkan_bin_dir is not None and vulkan_bin_dir.exists():
        src = vulkan_bin_dir / "vulkan-1.dll"
        if _copy_runtime_dll(src, package_dir):
            copied += 1

    if not (package_dir / "vulkan-1.dll").exists():
        for sys_path in [
            Path(r"C:\\Windows\\System32\\vulkan-1.dll"),
            Path(r"C:\\Windows\\SysWOW64\\vulkan-1.dll"),
        ]:
            if _copy_runtime_dll(sys_path, package_dir):
                copied += 1
                break

    return copied


def _copy_vulkan_runtime_dylibs(package_dir: Path) -> int:
    copied = 0
    seen: set[str] = set()

    for lib_dir in _macos_vulkan_library_dirs():
        for pattern in _macos_vulkan_dylib_patterns():
            for src in lib_dir.glob(pattern):
                if not src.is_file():
                    continue
                key = src.name.lower()
                if key in seen:
                    continue
                seen.add(key)
                if _copy_runtime_dylib(src, package_dir):
                    copied += 1
                    print(f"Bundled Vulkan runtime: {src} -> {package_dir / src.name}")

    return copied


def _bundle_macos_vulkan_icd(package_dir: Path) -> bool:
    dest = package_dir / "MoltenVK_icd.json"
    if dest.exists():
        return True

    moltenvk_name = ""
    for candidate in sorted(package_dir.glob("libMoltenVK*.dylib")):
        moltenvk_name = candidate.name
        break

    if not moltenvk_name:
        return False

    icd_payload = {
        "file_format_version": "1.0.0",
        "ICD": {
            "library_path": f"./{moltenvk_name}",
            "api_version": "1.2.0",
            "is_portability_driver": True,
        },
    }

    for src in _macos_vulkan_icd_candidates():
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
            icd = data.get("ICD")
            if isinstance(icd, dict):
                icd["library_path"] = f"./{moltenvk_name}"
            icd_payload = data
            break
        except Exception:
            continue

    try:
        dest.write_text(json.dumps(icd_payload, indent=4) + "\n", encoding="utf-8")
        print(f"Bundled MoltenVK ICD: {dest}")
        return True
    except Exception:
        return False


def _otool_load_commands(dylib: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["otool", "-L", str(dylib)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []

    if result.returncode != 0:
        return []

    deps: list[str] = []
    for line in (result.stdout or "").splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        dep = line.split("(", 1)[0].strip()
        if dep.startswith("@loader_path/") or dep.startswith("@rpath/"):
            continue
        deps.append(dep)
    return deps


def _resolve_bundled_dylib_name(package_dir: Path, dep_path: str) -> Optional[str]:
    dep_name = Path(dep_path).name
    direct = package_dir / dep_name
    if direct.exists():
        return dep_name

    stem = dep_name
    if stem.endswith(".dylib"):
        stem = stem[: -len(".dylib")]
    matches = sorted(package_dir.glob(f"{stem}*.dylib"))
    if matches:
        return matches[0].name
    return None


def _patch_macos_package_dylibs(package_dir: Path) -> None:
    dylibs = sorted(p for p in package_dir.glob("*.dylib") if p.is_file())
    if not dylibs:
        return

    for dylib in dylibs:
        _run_install_name_tool(["-id", f"@loader_path/{dylib.name}", str(dylib)])
        _run_install_name_tool(["-add_rpath", "@loader_path", str(dylib)])

    for dylib in dylibs:
        for dep_path in _otool_load_commands(dylib):
            bundled_name = _resolve_bundled_dylib_name(package_dir, dep_path)
            if bundled_name is None:
                continue
            _run_install_name_tool(
                [
                    "-change",
                    dep_path,
                    f"@loader_path/{bundled_name}",
                    str(dylib),
                ]
            )


def _bundle_macos_vulkan_runtime(
    package_dir: Path,
    backends: Dict[str, Tuple[bool, Optional[str]]],
    enable_vulkan: bool,
) -> None:
    if not enable_vulkan or not backends.get("vulkan", (False, None))[0]:
        return

    vulkan_count = _copy_vulkan_runtime_dylibs(package_dir)
    icd_ok = _bundle_macos_vulkan_icd(package_dir)
    _patch_macos_package_dylibs(package_dir)

    if vulkan_count == 0:
        print(
            "Warning: Vulkan backend was enabled but no macOS Vulkan runtime dylibs were bundled. "
            "Install the Vulkan SDK (set VULKAN_SDK) or `brew install molten-vk vulkan-loader` before building."
        )
    elif not icd_ok:
        print(
            "Warning: Vulkan runtime dylibs were bundled but MoltenVK_icd.json could not be created. "
            "GPU offload may fail unless VK_ICD_FILENAMES is set at runtime."
        )


def _copy_cuda_runtime_dlls(package_dir: Path) -> int:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        return 0
    cuda_bin = Path(cuda_home) / "bin"
    if not cuda_bin.exists():
        return 0

    required = [
        "cudart64_*.dll",
        "cublas64_*.dll",
        "cublasLt64_*.dll",
        "cusparse64_*.dll",
        "cusolver64_*.dll",
        "curand64_*.dll",
    ]

    copied = 0
    seen = set()
    for pattern in required:
        for src in cuda_bin.glob(pattern):
            key = src.name.lower()
            if key in seen:
                continue
            seen.add(key)
            if _copy_runtime_dll(src, package_dir):
                copied += 1
    return copied


def _copy_runtime_so(src: Path, dst_dir: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst = dst_dir / src.name
    try:
        shutil.copy2(src, dst)
    except Exception:
        return False
    return True


def _ensure_linux_symlink(dst: Path, target_name: str) -> bool:
    try:
        if dst.exists() or dst.is_symlink():
            return True
        os.symlink(target_name, dst)
        return True
    except Exception:
        return False


def _copy_linux_runtime_so(src: Path, dst_dir: Path) -> bool:
    if not src.exists() or not (src.is_file() or src.is_symlink()):
        return False

    dst = dst_dir / src.name
    if dst.exists() or dst.is_symlink():
        return True

    if src.is_symlink():
        try:
            target_path = src.resolve(strict=True)
        except Exception:
            return False

        if not target_path.is_file():
            return False

        if not _copy_linux_runtime_so(target_path, dst_dir):
            return False

        return _ensure_linux_symlink(dst, target_path.name)

    try:
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def _append_unique_dir(dirs: list[Path], path: Path) -> None:
    if not path.exists() or not path.is_dir():
        return
    if path in dirs:
        return
    dirs.append(path)


def _linux_cuda_library_dirs() -> list[Path]:
    dirs: list[Path] = []

    for env_key in ["LLAMA_CPP_CUDA_LIB_DIRS", "LLAMA_CPP_EXTRA_LIB_DIRS", "LD_LIBRARY_PATH"]:
        for raw_path in os.environ.get(env_key, "").split(os.pathsep):
            if raw_path:
                _append_unique_dir(dirs, Path(raw_path))

    cuda_roots: list[Path] = []
    for raw_root in [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")]:
        if raw_root:
            cuda_roots.append(Path(raw_root))

    nvcc = shutil.which("nvcc")
    if nvcc:
        cuda_roots.append(Path(nvcc).resolve().parent.parent)

    cuda_roots.extend([Path("/usr/local/cuda"), Path("/usr/cuda")])

    seen_roots: list[Path] = []
    for root in cuda_roots:
        if root in seen_roots:
            continue
        seen_roots.append(root)
        for subdir in ["lib64", "targets/x86_64-linux/lib", "lib"]:
            _append_unique_dir(dirs, root / subdir)

    for raw_path in sys.path:
        if not raw_path:
            continue
        base = Path(raw_path)
        if not base.exists() or not base.is_dir():
            continue
        nvidia_root = base / "nvidia"
        if not nvidia_root.exists() or not nvidia_root.is_dir():
            continue
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            _append_unique_dir(dirs, lib_dir)

    return dirs


def _copy_linux_cuda_runtime_sos(package_dir: Path) -> int:
    ggml_cuda = package_dir / "libggml-cuda.so.0"
    if not ggml_cuda.exists():
        return 0

    try:
        proc = subprocess.run(
            ["ldd", str(ggml_cuda)],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return 0

    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")

    # Keep this restricted to CUDA/NVIDIA user-mode libraries. The NVIDIA driver library
    # (libcuda.so.1) is provided by the host and must not be bundled.
    wanted_prefixes = ("libcu", "libnv")
    copied = 0
    seen: set[str] = set()

    for line in output.splitlines():
        m = re.match(r"\s*(?P<name>[^\s]+)\s+=>\s+(?P<target>[^\s]+)", line)
        if not m:
            continue

        name = m.group("name").strip()
        target = m.group("target").strip()

        if not name.startswith(wanted_prefixes):
            continue
        if name == "libcuda.so.1":
            continue

        # ldd unresolved entry looks like: "libcudart.so.12 => not found"
        # Our regex captures "not" as the target; treat it as unresolved.
        if target == "not":
            continue

        src = Path(target)
        if not src.exists():
            continue

        key = src.name.lower()
        if key not in seen:
            seen.add(key)
            if _copy_linux_runtime_so(src, package_dir):
                copied += 1

        # Ensure the dependency name exists in the package dir as a symlink when it differs.
        if name != src.name:
            _ensure_linux_symlink(package_dir / name, src.name)

    return copied


def _bundle_linux_runtime_sos(
    package_dir: Path,
    backends: Dict[str, Tuple[bool, Optional[str]]],
    enable_cuda: bool,
) -> None:
    cuda_count = 0

    if enable_cuda and backends.get("cuda", (False, None))[0]:
        cuda_count = _copy_linux_cuda_runtime_sos(package_dir)
        if cuda_count > 0:
            _patch_linux_rpath(package_dir)

    if enable_cuda and backends.get("cuda", (False, None))[0] and cuda_count == 0:
        print(
            "Warning: CUDA backend was enabled but no Linux CUDA runtime shared libraries were bundled. "
            "If you see missing libcudart/libcublas errors, ensure CUDA libraries are installed in a standard CUDA location, "
            "available on LD_LIBRARY_PATH, inside the active Python environment's nvidia/*/lib packages, "
            "or set LLAMA_CPP_CUDA_LIB_DIRS before building."
        )


def _bundle_windows_runtime_dlls(
    package_dir: Path,
    backends: Dict[str, Tuple[bool, Optional[str]]],
    enable_cuda: bool,
    enable_vulkan: bool,
) -> None:
    msvc_count = _copy_msvc_openmp_runtimes(package_dir)
    cuda_count = 0
    vulkan_count = 0

    if enable_cuda and backends.get("cuda", (False, None))[0]:
        cuda_count = _copy_cuda_runtime_dlls(package_dir)

    if enable_vulkan and backends.get("vulkan", (False, None))[0]:
        vulkan_count = _copy_vulkan_runtime_dlls(package_dir)

    if msvc_count == 0:
        print(
            "Warning: No MSVC/OpenMP runtime DLLs were bundled. If you see Windows error 0x7e on a clean machine, install VC++ 2015-2022 x64 redistributable."
        )
    if enable_cuda and backends.get("cuda", (False, None))[0] and cuda_count == 0:
        print(
            "Warning: CUDA backend was enabled but no CUDA runtime DLLs were bundled. If you see Windows error 0x7e on a clean machine, ensure CUDA runtime DLLs are present or set CUDA_PATH when building."
        )

    if enable_vulkan and backends.get("vulkan", (False, None))[0] and vulkan_count == 0:
        print(
            "Warning: Vulkan backend was enabled but vulkan-1.dll could not be bundled. Ensure Vulkan is installed (GPU drivers) or set VULKAN_SDK when building."
        )


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


def _read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _ensure_vendor_llama_cpp(project_root: Path, vendor_path: Path) -> None:
    if vendor_path.exists():
        return

    git = shutil.which("git")
    if git is None:
        raise RuntimeError(
            f"Vendor directory not found: {vendor_path}. "
            "Git is required to auto-fetch llama.cpp (install git or provide --vendor-path)."
        )

    vendor_path.parent.mkdir(parents=True, exist_ok=True)

    # Otherwise clone from upstream.
    repo_url = os.environ.get("LLAMA_CPP_VENDOR_REPO", "https://github.com/ggml-org/llama.cpp")
    cmd = [git, "clone", "--depth", "1", repo_url, str(vendor_path)]
    print(f"Vendor llama.cpp missing; cloning: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(project_root))
    if res.returncode != 0 or not vendor_path.exists():
        raise RuntimeError(
            f"Failed to clone llama.cpp into {vendor_path}. "
            "You can clone it manually or set LLAMA_CPP_VENDOR_REPO / use --vendor-path."
        )


def _is_windows() -> bool:
    return platform.system().lower() == "windows"


def _run_and_capture_env(cmd: list[str], cwd: Path | None = None) -> dict[str, str]:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        msg = stderr or stdout or f"Command failed: {' '.join(cmd)}"
        raise RuntimeError(msg)

    env: dict[str, str] = {}
    for line in (result.stdout or "").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k] = v
    return env


def _try_import_windows_toolchain_env(project_root: Path) -> bool:
    if not _is_windows():
        return True

    setup_script = project_root / "scripts" / "setup_windows_toolchain.ps1"
    if not setup_script.exists():
        return False

    # Run the toolchain script in a child PowerShell, then print the resulting environment
    # so we can import it into this Python process.
    ps_cmd = (
        "& { "
        f". '{str(setup_script)}' -Quiet; "
        "Get-ChildItem Env:* | ForEach-Object { \"$($_.Name)=$($_.Value)\" } "
        "}"
    )

    try:
        env = _run_and_capture_env(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                ps_cmd,
            ],
            cwd=project_root,
        )
    except Exception:
        return False

    for k, v in env.items():
        os.environ[k] = v

    return shutil.which("cl") is not None


def _vswhere_path() -> Optional[Path]:
    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", ""))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe",
        Path(os.environ.get("ProgramFiles", ""))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _prepend_path(dir_path: Path) -> None:
    if not dir_path.exists():
        return
    p = str(dir_path)
    cur = os.environ.get("PATH", "")
    if cur.lower().startswith(p.lower() + os.pathsep):
        return
    os.environ["PATH"] = p + os.pathsep + cur


def _try_add_vs_cmake_ninja(install_path: str) -> None:
    # Visual Studio bundles CMake and Ninja under the IDE directory.
    # This keeps setup minimal for Windows users.
    root = Path(install_path)
    cmake_bin = root / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "CMake" / "bin"
    ninja_bin = root / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "Ninja"
    _prepend_path(cmake_bin)
    _prepend_path(ninja_bin)


def _try_load_msvc_env(project_root: Optional[Path] = None) -> bool:
    if not _is_windows():
        return True

    if shutil.which("cl") is not None:
        return True

    root = project_root or get_project_root()
    if _try_import_windows_toolchain_env(root):
        return True

    vswhere = _vswhere_path()
    if vswhere is None:
        return False

    try:
        install_path = subprocess.check_output(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            text=True,
        ).strip()
    except Exception:
        return False

    if not install_path:
        return False

    _try_add_vs_cmake_ninja(install_path)

    vcvars64 = Path(install_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars64.exists():
        return False

    try:
        env = _run_and_capture_env(["cmd", "/c", f'"{vcvars64}" && set'])
    except Exception:
        return False

    for k, v in env.items():
        os.environ[k] = v

    return shutil.which("cl") is not None


def _cmake_generator() -> Optional[str]:
    if not _is_windows():
        return "Ninja"

    if shutil.which("ninja") is not None:
        return "Ninja"
    if shutil.which("nmake") is not None:
        return "NMake Makefiles"
    if shutil.which("cl") is not None:
        return "NMake Makefiles"
    return None


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _find_brew() -> Optional[str]:
    brew = shutil.which("brew")
    if brew:
        return brew
    for candidate in ["/opt/homebrew/bin/brew", "/usr/local/bin/brew"]:
        if Path(candidate).exists():
            _prepend_path(Path(candidate).parent)
            return candidate
    return None


def _install_homebrew() -> bool:
    if not _is_macos():
        return False
    if _find_brew() is not None:
        return True
    print("Homebrew not found. Installing Homebrew (this may prompt for your password)...")
    cmd = ["/bin/bash", "-c", f'"$(curl -fsSL {_HOMEBREW_INSTALL_URL})"']
    try:
        res = subprocess.run(cmd)
    except FileNotFoundError:
        return False
    return res.returncode == 0 and _find_brew() is not None


def _brew_install(packages: list[str]) -> bool:
    brew = _find_brew()
    if brew is None:
        return False
    cmd = [brew, "install", *packages]
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    return res.returncode == 0


def _maybe_install_macos_build_tools(auto_install: bool) -> None:
    if not _is_macos():
        return

    missing: list[str] = []
    if shutil.which("cmake") is None:
        missing.append("cmake")
    if shutil.which("ninja") is None:
        missing.append("ninja")

    if not missing:
        return

    if not auto_install:
        raise RuntimeError(
            "Missing build tools: "
            + ", ".join(missing)
            + ". Install Homebrew and run `brew install cmake ninja`, "
            "or rerun this script with --install-macos-deps to attempt installing them automatically."
        )

    if _find_brew() is None:
        if not _install_homebrew():
            raise RuntimeError(
                "Homebrew was not found and could not be installed automatically. "
                "Install Homebrew from https://brew.sh and retry."
            )

    if missing and not _brew_install(missing):
        raise RuntimeError(
            "Failed to install required build tools via Homebrew. "
            "Try running `brew install cmake ninja` manually."
        )


def _require_build_tools(*, auto_install_macos_deps: bool = False) -> None:
    if _is_macos():
        _maybe_install_macos_build_tools(auto_install_macos_deps)

    if shutil.which("cmake") is None:
        raise RuntimeError(
            "CMake was not found in PATH. Install CMake and ensure `cmake` is available, "
            "or install a prebuilt wheel that bundles the llama shared library."
        )

    if _is_windows():
        if not _try_load_msvc_env(get_project_root()):
            raise RuntimeError(
                "No usable C/C++ toolchain was detected. Install 'Visual Studio Build Tools' (MSVC) with the C++ workload, "
                "or run scripts/setup_windows_toolchain.ps1, then retry."
            )
        gen = _cmake_generator()
        if gen is None:
            raise RuntimeError(
                "No usable C/C++ build toolchain was detected. "
                "Install 'Visual Studio Build Tools' (MSVC) or add a supported generator to PATH (Ninja/NMake)."
            )
    else:
        if shutil.which("ninja") is None:
            raise RuntimeError(
                "Ninja was not found in PATH. Install Ninja and ensure `ninja` is available. "
                "(This project uses Ninja for builds.)"
            )


def detect_cuda() -> Tuple[bool, Optional[str]]:
    """Detect if CUDA is available and return version."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

    if cuda_home and Path(cuda_home).exists():
        nvcc_path = Path(cuda_home) / "bin" / ("nvcc.exe" if platform.system() == "Windows" else "nvcc")
        if nvcc_path.exists():
            try:
                result = subprocess.run([str(nvcc_path), "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "release" in line.lower():
                            return True, line.strip()
            except Exception:
                pass
            return True, "unknown version"

    for path in ["/usr/local/cuda", "/usr/cuda", "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"]:
        if Path(path).exists():
            return True, "detected"

    return False, None


def detect_rocm() -> Tuple[bool, Optional[str]]:
    """Detect if ROCm is available."""
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")

    if Path(rocm_path).exists():
        hipcc_path = Path(rocm_path) / "bin" / "hipcc"
        if hipcc_path.exists():
            try:
                result = subprocess.run([str(hipcc_path), "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    return True, result.stdout.split("\n")[0]
            except Exception:
                pass
            return True, "detected"

    return False, None


def detect_vulkan() -> Tuple[bool, Optional[str]]:
    """Detect if Vulkan SDK is available."""
    vulkan_sdk = os.environ.get("VULKAN_SDK")

    if vulkan_sdk and Path(vulkan_sdk).exists():
        return True, vulkan_sdk

    if platform.system() == "Linux":
        if Path("/usr/include/vulkan/vulkan.h").exists():
            return True, "system"

    if platform.system() == "Darwin":
        header_candidates = [
            Path("/usr/local/include/vulkan/vulkan.h"),
            Path("/opt/homebrew/include/vulkan/vulkan.h"),
        ]
        for header in header_candidates:
            if header.exists():
                return True, str(header.parent.parent)

        for prefix_formula in ["vulkan-loader", "vulkan-headers"]:
            prefix = _brew_prefix(prefix_formula)
            if prefix is not None and (prefix / "include" / "vulkan" / "vulkan.h").exists():
                return True, str(prefix)

    return False, None


def detect_metal() -> Tuple[bool, Optional[str]]:
    """Detect if Metal is available (macOS only)."""
    if platform.system() != "Darwin":
        return False, None

    try:
        result = subprocess.run(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
    except Exception:
        pass

    return False, None


def detect_blas() -> Tuple[bool, str]:
    """Detect available BLAS implementation."""
    if platform.system() == "Darwin":
        return True, "accelerate"

    openblas_paths = [
        "/usr/include/openblas",
        "/usr/local/include/openblas",
        "/opt/OpenBLAS/include",
    ]
    for path in openblas_paths:
        if Path(path).exists():
            return True, "openblas"

    mkl_root = os.environ.get("MKLROOT")
    if mkl_root and Path(mkl_root).exists():
        return True, "mkl"

    return False, "none"


def detect_backends() -> Dict[str, Tuple[bool, Optional[str]]]:
    """Detect all available backends."""
    return {
        "cuda": detect_cuda(),
        "rocm": detect_rocm(),
        "vulkan": detect_vulkan(),
        "metal": detect_metal(),
        "blas": detect_blas(),
    }


def get_cmake_args(
    backends: Dict[str, Tuple[bool, Optional[str]]],
    enable_cuda: bool = True,
    enable_rocm: bool = True,
    enable_vulkan: bool = True,
    enable_metal: bool = True,
    enable_blas: bool = True,
) -> List[str]:
    """Get CMake configuration arguments based on detected backends."""
    args = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_CURL=OFF",
    ]

    if platform.system() == "Linux":
        args.append("-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON")

    # When producing distributable wheels in CI, never compile with -march=native
    # (GGML_NATIVE=ON). GitHub runners may support instructions (e.g. AVX512) that
    # are not available on end-user CPUs, leading to runtime "Illegal instruction"
    # crashes.
    if os.environ.get("GITHUB_ACTIONS") == "true" and not any(
        a.startswith("-DGGML_NATIVE=") for a in args
    ):
        args.append("-DGGML_NATIVE=OFF")

    if enable_cuda and backends["cuda"][0]:
        args.append("-DGGML_CUDA=ON")
        if not any(a.startswith("-DCMAKE_CUDA_ARCHITECTURES=") for a in args):
            cuda_archs = os.environ.get("CMAKE_CUDA_ARCHITECTURES")
            if not cuda_archs:
                cuda_archs = "75;80;86"
            args.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}")
        print(f"  CUDA: enabled ({backends['cuda'][1]})")
    else:
        args.append("-DGGML_CUDA=OFF")

    if enable_rocm and backends["rocm"][0]:
        args.append("-DGGML_HIP=ON")
        print(f"  ROCm: enabled ({backends['rocm'][1]})")
    else:
        args.append("-DGGML_HIP=OFF")

    if enable_vulkan and backends["vulkan"][0]:
        args.append("-DGGML_VULKAN=ON")
        print(f"  Vulkan: enabled ({backends['vulkan'][1]})")
    else:
        args.append("-DGGML_VULKAN=OFF")

    if enable_metal and backends["metal"][0]:
        args.append("-DGGML_METAL=ON")
        print(f"  Metal: enabled ({backends['metal'][1]})")
    else:
        args.append("-DGGML_METAL=OFF")

    if enable_blas and backends["blas"][0]:
        blas_type = backends["blas"][1]
        if blas_type == "accelerate":
            args.append("-DGGML_ACCELERATE=ON")
        elif blas_type == "openblas":
            args.append("-DGGML_BLAS=ON")
            args.append("-DGGML_BLAS_VENDOR=OpenBLAS")
        elif blas_type == "mkl":
            args.append("-DGGML_BLAS=ON")
            args.append("-DGGML_BLAS_VENDOR=Intel10_64lp")
        print(f"  BLAS: enabled ({blas_type})")
    else:
        args.append("-DGGML_BLAS=OFF")

    return args


def run_cmake_configure(
    source_dir: Path,
    build_dir: Path,
    cmake_args: List[str],
    *,
    auto_install_macos_deps: bool = False,
) -> bool:
    """Run CMake configuration."""
    _require_build_tools(auto_install_macos_deps=auto_install_macos_deps)
    build_dir.mkdir(parents=True, exist_ok=True)

    gen = _cmake_generator()
    if gen is None:
        raise RuntimeError("Unable to select a CMake generator for this platform.")
    cmd = ["cmake", "-G", gen, str(source_dir)] + cmake_args

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=build_dir)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Failed to run CMake. Ensure CMake is installed and available on PATH."
        ) from e
    return result.returncode == 0


def run_cmake_build(build_dir: Path, parallel: int = 0, target: Optional[str] = None) -> bool:
    """Run CMake build."""
    _require_build_tools(auto_install_macos_deps=False)
    cmd = ["cmake", "--build", str(build_dir)]

    if target:
        cmd.extend(["--target", target])

    if parallel > 0:
        cmd.extend(["--parallel", str(parallel)])
    else:
        cmd.extend(["--parallel"])

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Failed to run CMake build. Ensure CMake is installed and available on PATH."
        ) from e
    return result.returncode == 0


def find_built_library(build_dir: Path) -> Optional[Path]:
    """Find the built shared library."""
    system = platform.system().lower()

    if system == "windows":
        patterns = [
            "**/libllama.dll",
            "**/Release/libllama.dll",
            "**/bin/libllama.dll",
            "**/llama.dll",
            "**/Release/llama.dll",
            "**/bin/llama.dll",
        ]
    elif system == "darwin":
        patterns = ["**/libllama.dylib", "**/lib/libllama.dylib"]
    else:
        patterns = ["**/libllama.so", "**/lib/libllama.so"]

    for pattern in patterns:
        matches = list(build_dir.glob(pattern))
        if matches:
            return matches[0]

    return None


def _copy_windows_dependency_dlls(lib_path: Path, package_dir: Path) -> None:
    lib_dir = lib_path.parent
    patterns = ["ggml*.dll"]

    copied_any = False
    for pattern in patterns:
        for dep_path in lib_dir.glob(pattern):
            if dep_path.name.lower() == lib_path.name.lower():
                continue

            dest_path = package_dir / dep_path.name
            shutil.copy2(dep_path, dest_path)
            copied_any = True
            print(f"Copied {dep_path} to {dest_path}")

    if not copied_any:
        print(
            "Note: no ggml*.dll dependencies were found next to the built llama DLL. "
            "If you still see Windows error 0x7e at runtime, the missing dependency is likely a system/runtime DLL."
        )


def _run_install_name_tool(args: list[str]) -> None:
    try:
        subprocess.run(["install_name_tool", *args], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return


def _run_patchelf(args: list[str]) -> bool:
    patchelf = shutil.which("patchelf")
    if not patchelf:
        for candidate in [
            Path(sys.executable).parent / "patchelf",
            Path(sys.executable).resolve().parent / "patchelf",
        ]:
            if candidate.exists():
                patchelf = str(candidate)
                break
    if not patchelf:
        return False

    try:
        result = subprocess.run([patchelf, *args], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return False

    return result.returncode == 0


def _copy_macos_dependency_dylibs(lib_path: Path, package_dir: Path) -> None:
    lib_dir = lib_path.parent
    patterns = ["libllama*.dylib", "libggml*.dylib"]

    # Some builds (notably Metal) may place dependent dylibs in a nearby output
    # directory rather than next to libllama.dylib.
    candidate_dirs: list[Path] = [
        lib_dir,
        lib_dir.parent,
        lib_dir.parent / "lib",
        lib_dir.parent / "bin",
    ]
    candidate_dirs = [p for p in candidate_dirs if p.exists()]

    copied: list[Path] = []
    for pattern in patterns:
        for search_dir in candidate_dirs:
            for dep_path in search_dir.glob(pattern):
                dest_path = package_dir / dep_path.name
                if dest_path.exists():
                    continue
                shutil.copy2(dep_path, dest_path)
                copied.append(dest_path)
                print(f"Copied {dep_path} to {dest_path}")

    # Always patch the main dylib in the package. Previously we only patched
    # dylibs that were freshly copied, which left libllama.dylib still pointing
    # at @rpath dependencies inside built wheels.
    primary = package_dir / lib_path.name
    dylibs_to_patch: list[Path] = []
    if primary.exists():
        dylibs_to_patch.append(primary)
    dylibs_to_patch.extend(copied)

    bundled_deps: list[Path] = []
    for pattern in patterns:
        bundled_deps.extend(package_dir.glob(pattern))

    for dylib_path in dylibs_to_patch:
        _run_install_name_tool(["-id", f"@loader_path/{dylib_path.name}", str(dylib_path)])
        _run_install_name_tool(["-add_rpath", "@loader_path", str(dylib_path)])

    for dylib_path in dylibs_to_patch:
        for dep in bundled_deps:
            _run_install_name_tool(
                [
                    "-change",
                    f"@rpath/{dep.name}",
                    f"@loader_path/{dep.name}",
                    str(dylib_path),
                ]
            )


def _copy_linux_dependency_sos(lib_path: Path, package_dir: Path) -> None:
    lib_dir = lib_path.parent
    patterns = ["libggml*.so*"]

    candidate_dirs: list[Path] = [
        lib_dir,
        lib_dir.parent,
        lib_dir.parent / "lib",
        lib_dir.parent / "bin",
    ]
    candidate_dirs = [p for p in candidate_dirs if p.exists()]

    copied_any = False
    for pattern in patterns:
        for search_dir in candidate_dirs:
            for dep_path in search_dir.glob(pattern):
                if dep_path.name == lib_path.name:
                    continue

                dest_path = package_dir / dep_path.name
                if dest_path.exists():
                    continue

                shutil.copy2(dep_path, dest_path)
                copied_any = True
                print(f"Copied {dep_path} to {dest_path}")

    if not copied_any:
        print(
            "Note: no libggml*.so* dependencies were found next to the built llama shared library. "
            "If the packaged libllama.so fails to load on Linux, ensure the build outputs include the ggml shared libraries."
        )


def _patch_linux_rpath(package_dir: Path) -> None:
    shared_objects = sorted(path for path in package_dir.glob("*.so*") if path.is_file())
    if not shared_objects:
        return

    patched_any = False
    for so_path in shared_objects:
        if _run_patchelf(["--set-rpath", "$ORIGIN", str(so_path)]):
            patched_any = True
            print(f"Patched RUNPATH on {so_path} to $ORIGIN")
        else:
            print(
                "Warning: failed to set Linux RUNPATH with patchelf for "
                f"{so_path}. Bundled shared libraries may not load unless patchelf is installed."
            )

    if not patched_any:
        print(
            "Warning: patchelf was not available, so Linux bundled shared libraries were not patched with $ORIGIN RUNPATH."
        )


def copy_library_to_package(lib_path: Path, package_dir: Path) -> Path:
    """Copy the built library to the package directory."""
    package_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ["*.dylib", "*.so", "*.so.*", "*.dll", "MoltenVK_icd.json"]:
        for artifact in package_dir.glob(pattern):
            if artifact.is_file():
                artifact.unlink()

    dest_path = package_dir / lib_path.name
    shutil.copy2(lib_path, dest_path)

    print(f"Copied {lib_path} to {dest_path}")

    if platform.system().lower() == "windows" and lib_path.suffix.lower() == ".dll":
        _copy_windows_dependency_dlls(lib_path, package_dir)

    if platform.system().lower() == "darwin" and lib_path.suffix.lower() == ".dylib":
        _copy_macos_dependency_dylibs(lib_path, package_dir)

    if platform.system().lower() == "linux" and ".so" in lib_path.name:
        _copy_linux_dependency_sos(lib_path, package_dir)
        _patch_linux_rpath(package_dir)

    return dest_path


def build_llama_cpp(
    vendor_path: Path,
    output_dir: Path,
    enable_cuda: bool = True,
    enable_rocm: bool = True,
    enable_vulkan: bool = True,
    enable_metal: bool = True,
    enable_blas: bool = True,
    parallel: int = 0,
    clean: bool = False,
    fetch_vendor: bool = True,
    bundle_runtime_dlls: bool = True,
    project_root: Optional[Path] = None,
    auto_install_macos_deps: bool = False,
) -> Optional[Path]:
    """
    Build llama.cpp and return path to the built library.

    Args:
        vendor_path: Path to the llama.cpp source directory.
        output_dir: Directory to place the built library.
        enable_*: Enable specific backends if available.
        parallel: Number of parallel build jobs (0 for auto).
        clean: Clean build directory before building.

    Returns:
        Path to the built library, or None if build failed.
    """
    if not vendor_path.exists():
        if fetch_vendor:
            try:
                _ensure_vendor_llama_cpp(project_root or get_project_root(), vendor_path)
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                return None
        else:
            print(f"Error: Vendor directory not found: {vendor_path}", file=sys.stderr)
            return None

    build_dir = vendor_path / "build"

    if clean and build_dir.exists():
        print(f"Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir)

    print("Detecting available backends...")
    backends = detect_backends()

    print("\nConfiguring build...")
    cmake_args = get_cmake_args(
        backends,
        enable_cuda=enable_cuda,
        enable_rocm=enable_rocm,
        enable_vulkan=enable_vulkan,
        enable_metal=enable_metal,
        enable_blas=enable_blas,
    )

    if not run_cmake_configure(
        vendor_path,
        build_dir,
        cmake_args,
        auto_install_macos_deps=auto_install_macos_deps,
    ):
        print("Error: CMake configuration failed", file=sys.stderr)
        return None

    print("\nBuilding...")
    if not run_cmake_build(build_dir, parallel=parallel, target="llama"):
        print("Error: Build failed", file=sys.stderr)
        return None

    print("\nLocating built library...")
    lib_path = find_built_library(build_dir)

    if lib_path is None:
        print("Error: Could not find built library", file=sys.stderr)
        return None

    print(f"Found library: {lib_path}")

    dest_path = copy_library_to_package(lib_path, output_dir)

    if _is_windows() and bundle_runtime_dlls:
        _bundle_windows_runtime_dlls(
            output_dir,
            backends,
            enable_cuda=enable_cuda,
            enable_vulkan=enable_vulkan,
        )

    if platform.system().lower() == "linux":
        _bundle_linux_runtime_sos(
            output_dir,
            backends,
            enable_cuda=enable_cuda,
        )

    if _is_macos() and enable_vulkan:
        _bundle_macos_vulkan_runtime(
            output_dir,
            backends,
            enable_vulkan=enable_vulkan,
        )

    return dest_path


def main():
    parser = argparse.ArgumentParser(
        description="Build llama.cpp shared library"
    )
    parser.add_argument(
        "--vendor-path",
        type=Path,
        default=None,
        help="Path to vendor/llama.cpp directory"
    )
    parser.add_argument(
        "--no-fetch-vendor",
        action="store_true",
        help="Do not auto-fetch vendor/llama.cpp when missing"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for built library"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory"
    )

    parser.add_argument(
        "--backend",
        choices=["auto", "cpu", "cuda", "vulkan", "rocm", "metal"],
        default="auto",
        help=(
            "Select a single backend to build (default: auto). "
            "This is a convenience flag that sets the --no-* toggles for you; "
            "explicit --no-* flags still override."
        ),
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA even if available"
    )
    parser.add_argument(
        "--no-rocm",
        action="store_true",
        help="Disable ROCm even if available"
    )
    parser.add_argument(
        "--no-vulkan",
        action="store_true",
        help="Disable Vulkan even if available"
    )
    parser.add_argument(
        "--no-metal",
        action="store_true",
        help="Disable Metal even if available"
    )
    parser.add_argument(
        "--no-blas",
        action="store_true",
        help="Disable BLAS even if available"
    )
    parser.add_argument(
        "--parallel", "-j",
        type=int,
        default=0,
        help="Number of parallel build jobs (0 for auto)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directory before building"
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only detect backends, don't build"
    )

    parser.add_argument(
        "--no-bundle-runtime-dlls",
        action="store_true",
        help="Do not bundle Windows runtime DLLs (CUDA/MSVC/OpenMP) next to the built library",
    )

    parser.add_argument(
        "--install-macos-deps",
        action="store_true",
        help=(
            "macOS only: attempt to install missing build tools (Homebrew + cmake + ninja) automatically. "
            "If you prefer manual setup, install Homebrew then run `brew install cmake ninja`."
        ),
    )

    args = parser.parse_args()

    project_root = args.project_root or get_project_root()
    vendor_path = args.vendor_path or (project_root / "vendor" / "llama.cpp")
    output_dir = args.output_dir or (project_root / "src" / "llama_cpp_py_sync")

    if args.detect_only:
        print("Detecting available backends...")
        backends = detect_backends()
        print("\nBackend Detection Results:")
        print(f"  CUDA:   {'✓' if backends['cuda'][0] else '✗'} {backends['cuda'][1] or ''}")
        print(f"  ROCm:   {'✓' if backends['rocm'][0] else '✗'} {backends['rocm'][1] or ''}")
        print(f"  Vulkan: {'✓' if backends['vulkan'][0] else '✗'} {backends['vulkan'][1] or ''}")
        print(f"  Metal:  {'✓' if backends['metal'][0] else '✗'} {backends['metal'][1] or ''}")
        print(f"  BLAS:   {'✓' if backends['blas'][0] else '✗'} {backends['blas'][1] or ''}")
        return

    # Backend selection convenience. Defaults to existing behavior (auto).
    enable_cuda = not args.no_cuda
    enable_rocm = not args.no_rocm
    enable_vulkan = not args.no_vulkan
    enable_metal = not args.no_metal
    enable_blas = not args.no_blas

    if args.backend != "auto":
        enable_cuda = args.backend == "cuda"
        enable_rocm = args.backend == "rocm"
        enable_vulkan = args.backend == "vulkan"
        enable_metal = args.backend == "metal"
        # For a predictable single-backend build, keep BLAS off by default.
        enable_blas = False

    # Allow explicit no-* flags to override backend convenience.
    if args.no_cuda:
        enable_cuda = False
    if args.no_rocm:
        enable_rocm = False
    if args.no_vulkan:
        enable_vulkan = False
    if args.no_metal:
        enable_metal = False
    if args.no_blas:
        enable_blas = False

    lib_path = build_llama_cpp(
        vendor_path=vendor_path,
        output_dir=output_dir,
        enable_cuda=enable_cuda,
        enable_rocm=enable_rocm,
        enable_vulkan=enable_vulkan,
        enable_metal=enable_metal,
        enable_blas=enable_blas,
        parallel=args.parallel,
        clean=args.clean,
        fetch_vendor=not args.no_fetch_vendor,
        bundle_runtime_dlls=(not args.no_bundle_runtime_dlls) if _is_windows() else False,
        project_root=project_root,
        auto_install_macos_deps=args.install_macos_deps,
    )

    if lib_path:
        print("\nBuild successful!")
        print(f"Library: {lib_path}")
        sys.exit(0)
    else:
        print("\nBuild failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
