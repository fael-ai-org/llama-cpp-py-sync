"""Explicit access to the native RPC backend.

RPC traffic is unauthenticated and unencrypted. Use trusted endpoints or an
authenticated tunnel. Run blocking workers in a dedicated supervised process.
"""

from __future__ import annotations

import operator
import re
import uuid
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


_registered_streams = {}


class RPCStream:
    """Blocking caller-owned stream; authenticate/encrypt it before native use.

    The transport supplies sendall, recv_into and close. No listener or RDMA
    upgrade is created. Close models/backends before closing a registered
    stream. Use in a supervised process: upstream RPC errors can abort it.
    """

    def __init__(self, transport):
        if any(not callable(getattr(transport, name, None)) for name in ("sendall", "recv_into", "close")):
            raise ValueError("RPC transport requires sendall, recv_into and close")
        self.transport = transport
        self.name = uuid.uuid4().hex
        self._mode = "new"
        self._transport_closed = False
        self.error = None
        self._devices = None
        self._ffi = get_ffi()
        ffi = self._ffi

        @ffi.callback("bool(void *, const void *, size_t)", error=False)
        def send(_context, data, size):
            try:
                for offset in range(0, size, 1024 * 1024):
                    transport.sendall(ffi.buffer(ffi.cast("char *", data) + offset, min(size - offset, 1024 * 1024)))
                return True
            except Exception as exc:
                self.error = exc
                return False

        @ffi.callback("bool(void *, void *, size_t)", error=False)
        def recv(_context, data, size):
            try:
                offset = 0
                while offset < size:
                    buffer = ffi.buffer(ffi.cast("char *", data) + offset, min(size - offset, 1024 * 1024))
                    count = transport.recv_into(buffer)
                    if type(count) is not int or not 0 < count <= len(buffer):
                        return False
                    offset += count
                return True
            except Exception as exc:
                self.error = exc
                return False

        @ffi.callback("void(void *)")
        def close(_context):
            self._close_transport()

        self._callbacks = (send, recv, close)
        self._native = ffi.new("ggml_rpc_stream *", {"send": send, "recv": recv, "close": close})

    def _close_transport(self):
        if not self._transport_closed:
            self._transport_closed = True
            try:
                self.transport.close()
            except Exception as exc:
                self.error = exc

    def devices(self):
        """Register once and return its native devices; retain until close."""
        if self._mode == "registered" and not self._transport_closed:
            return list(self._devices)
        if self._mode != "new" or self._transport_closed:
            raise RuntimeError("RPC stream is not available for registration")
        ffi, backend, native = _require_rpc()
        if not hasattr(native, "ggml_backend_rpc_add_stream"):
            raise RuntimeError("Native RPC stream support requires a rebuilt library")
        self._mode = "registered"
        _registered_streams[self.name] = self
        registration = native.ggml_backend_rpc_add_stream(self.name.encode("ascii"), self._native)
        if registration == ffi.NULL:
            self.close()
            raise RuntimeError("RPC stream exposes no devices")
        # Explicit model device handles avoid publishing a closed stream in
        # the process-wide native device registry, which has no unregister API.
        base = get_backend_base_lib()
        self._devices = [base.ggml_backend_reg_dev_get(registration, index)
                         for index in range(base.ggml_backend_reg_dev_count(registration))]
        return list(self._devices)

    def serve(self, *, devices: Sequence[str], n_threads: int = 4):
        """Serve one supplied connection, returning after disconnect cleanup."""
        if self._mode != "new" or self._transport_closed:
            raise RuntimeError("RPC stream is not available for serving")
        if isinstance(devices, (str, bytes)) or not devices or any(
            not isinstance(name, str) or not name or "\0" in name for name in devices
        ) or len(set(devices)) != len(devices):
            raise ValueError("Invalid native device selection")
        if isinstance(n_threads, bool) or not 1 <= operator.index(n_threads) <= 2147483647:
            raise ValueError("Invalid native thread count")
        ffi, backend, native = _require_rpc()
        if not hasattr(native, "ggml_backend_rpc_serve_stream"):
            raise RuntimeError("Native RPC stream support requires a rebuilt library")
        handles = [backend.ggml_backend_dev_by_name(name.encode("utf-8")) for name in devices]
        if any(handle == ffi.NULL for handle in handles):
            raise ValueError("Unknown native device name")
        self._mode = "serving"
        try:
            if not native.ggml_backend_rpc_serve_stream(self._native, ffi.NULL, n_threads,
                    len(handles), ffi.new("ggml_backend_dev_t[]", handles)):
                raise RuntimeError("Native RPC stream initialization failed")
        finally:
            self._mode = "closed"
            self._close_transport()

    def close(self):
        """Release after native consumers have stopped; never during inference."""
        if self._mode == "serving":
            raise RuntimeError("Stop the serving connection before closing its owner")
        if self._mode == "registered":
            if not get_rpc_lib().ggml_backend_rpc_remove_stream(self.name.encode("ascii")):
                raise RuntimeError("RPC stream still has native consumers")
            _registered_streams.pop(self.name, None)
        self._mode = "closed"
        self._close_transport()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def rpc_stream_devices(streams: Sequence[RPCStream], *, include_local: bool = True):
    """Select caller-owned streams without connecting to network endpoints."""
    if isinstance(streams, (str, bytes)) or not streams or any(not isinstance(stream, RPCStream) for stream in streams):
        raise ValueError("RPC streams must be a nonempty sequence of RPCStream owners")
    if len({id(stream) for stream in streams}) != len(streams):
        raise ValueError("RPC streams must be unique")
    ffi, backend, native = _require_rpc()
    if len(streams) > native.GGML_RPC_MAX_SERVERS:
        raise ValueError("Too many RPC streams")
    base = get_backend_base_lib()
    devices = []
    if include_local:
        for index in range(backend.ggml_backend_dev_count()):
            device = backend.ggml_backend_dev_get(index)
            name = ffi.string(base.ggml_backend_dev_name(device)).decode("utf-8")
            if not name.startswith("RPC") and base.ggml_backend_dev_type(device) in (
                base.GGML_BACKEND_DEVICE_TYPE_GPU, base.GGML_BACKEND_DEVICE_TYPE_IGPU
            ):
                devices.append(device)
    for stream in streams:
        devices.extend(stream.devices())
    if len(devices) > get_lib().llama_max_devices():
        raise ValueError("Selected devices exceed the native model device capacity")
    return devices
