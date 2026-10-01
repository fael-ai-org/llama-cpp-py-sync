"""Explicit access to the native RPC backend.

RPC traffic is unauthenticated and unencrypted. Use trusted endpoints or an
authenticated tunnel. Run blocking workers in a dedicated supervised process.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Sequence

from llama_cpp_py_sync._cffi_bindings import (
    get_backend_base_lib,
    get_backend_lib,
    get_ffi,
    get_lib,
    get_rpc_lib,
)


def _validate_endpoint(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+:[0-9]{1,5}", endpoint):
        raise ValueError("RPC endpoint must be an IPv4 address or hostname followed by :port")
    host, port = endpoint.rsplit(":", 1)
    if not 1 <= int(port) <= 65535:
        raise ValueError("RPC port must be between 1 and 65535")
    return f"{host}:{int(port)}"


def _require_rpc():
    core = get_lib()
    if not core.llama_supports_rpc():
        raise RuntimeError("The native library was built without RPC support")
    core.llama_backend_init()
    return get_ffi(), get_backend_lib(), get_rpc_lib()


def rpc_devices(endpoints: Sequence[str], *, include_local: bool = True):
    """Return device handles for explicitly selected RPC servers.

    Local GPU/IGPU devices can precede remote devices. Native registrations are
    process-wide and remain alive until process exit. Unreachable endpoints can
    terminate the native process; use a dedicated model process.
    """
    if isinstance(endpoints, (str, bytes)):
        raise ValueError("RPC endpoints must be a sequence, not a string")
    normalized = [_validate_endpoint(endpoint) for endpoint in endpoints]
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("RPC endpoints must be nonempty and unique")
    ffi, backend, rpc = _require_rpc()
    if len(normalized) > rpc.GGML_RPC_MAX_SERVERS:
        raise ValueError("Too many RPC servers")
    devices = []
    base = get_backend_base_lib()
    if include_local:
        for index in range(backend.ggml_backend_dev_count()):
            device = backend.ggml_backend_dev_get(index)
            name = ffi.string(base.ggml_backend_dev_name(device)).decode("utf-8")
            kind = base.ggml_backend_dev_type(device)
            if not name.startswith("RPC") and kind in (
                base.GGML_BACKEND_DEVICE_TYPE_GPU, base.GGML_BACKEND_DEVICE_TYPE_IGPU
            ):
                devices.append(device)
    for endpoint in normalized:
        registration = rpc.ggml_backend_rpc_add_server(endpoint.encode("ascii"))
        if registration == ffi.NULL:
            raise RuntimeError("RPC server exposes no devices")
        backend.ggml_backend_register(registration)
        devices.extend(
            base.ggml_backend_reg_dev_get(registration, index)
            for index in range(base.ggml_backend_reg_dev_count(registration))
        )
    if len(devices) > get_lib().llama_max_devices():
        raise ValueError("Selected devices exceed the native model device capacity")
    return devices


def start_rpc_server(
    endpoint: str = "127.0.0.1:50052", *, devices: Sequence[str], n_threads: int = 4
) -> None:
    """Serve named native devices; blocks until the worker process is stopped.

    No worker starts on import. The default listener is loopback-only. This
    function provides no authentication, encryption or in-process stop API.
    """
    endpoint = _validate_endpoint(endpoint)
    if isinstance(devices, (str, bytes)) or not devices:
        raise ValueError("devices must be a nonempty sequence of native device names")
    if any(not isinstance(name, str) or not name or "\0" in name for name in devices):
        raise ValueError("Invalid native device name")
    if len(set(devices)) != len(devices):
        raise ValueError("Native device names must be unique")
    if isinstance(n_threads, bool) or not 1 <= operator.index(n_threads) <= 2147483647:
        raise ValueError("n_threads must be a positive native integer")
    ffi, backend, rpc = _require_rpc()
    handles = [backend.ggml_backend_dev_by_name(name.encode("utf-8")) for name in devices]
    if any(handle == ffi.NULL for handle in handles):
        raise ValueError("Unknown native device name")
    buffer = ffi.new("ggml_backend_dev_t[]", handles)
    rpc.ggml_backend_rpc_start_server(
        endpoint.encode("ascii"), ffi.NULL, n_threads, len(handles), buffer
    )
    raise RuntimeError("Native RPC worker stopped")
