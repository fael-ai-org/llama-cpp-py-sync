"""Lifetime management for the bundled Microsoft OpenMP runtime."""

import ctypes
from pathlib import Path

_retained_runtimes = {}


def retain_openmp_runtime(library_dir):
    """Keep OpenMP worker code mapped through native-library finalization.

    Microsoft OpenMP workers can outlive CFFI's DLL handles at interpreter
    shutdown. Pin only the bundled runtime, leaving model cleanup unchanged.
    """
    path = Path(library_dir).resolve() / "vcomp140.dll"
    if not path.is_file():
        return
    if path in _retained_runtimes:
        return
    runtime = ctypes.WinDLL(str(path))
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    pin = kernel.GetModuleHandleExW
    pin.argtypes = [ctypes.c_ulong, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p)]
    pin.restype = ctypes.c_int
    module = ctypes.c_void_p()
    if not pin(1, str(path), ctypes.byref(module)):
        raise ctypes.WinError(ctypes.get_last_error())
    _retained_runtimes[path] = runtime
