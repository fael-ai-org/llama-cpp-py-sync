"""Isolated native worker for opt-in caller-owned TLS stream tests."""
import socket
import ssl
import sys

from llama_cpp_py_sync.rpc import RPCStream

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.minimum_version = ssl.TLSVersion.TLSv1_3
context.verify_mode = ssl.CERT_REQUIRED
context.load_cert_chain(sys.argv[1], sys.argv[2])
context.load_verify_locations(sys.argv[1])
with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    listener.listen(3)
    listener.settimeout(30)
    print(listener.getsockname()[1], flush=True)
    while True:
        connection, _ = listener.accept()
        connection.settimeout(180)
        try:
            secured = context.wrap_socket(connection, server_side=True)
            break
        except ssl.SSLError:
            connection.close()
            print("unauthenticated-peer-rejected", file=sys.stderr, flush=True)
assert secured.version() == "TLSv1.3"
owner = RPCStream(secured)
owner.serve(devices=["CPU"], n_threads=2)
print("stream-worker-stopped", flush=True)
