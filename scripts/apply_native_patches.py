"""Apply the reviewed RPC stream extension, rejecting incompatible sources."""
import argparse
import subprocess
from pathlib import Path


def apply(vendor_path: Path):
    patch = Path(__file__).resolve().parent.parent / "patches" / "rpc-stream.patch"
    contents = patch.read_bytes()
    source = (vendor_path / "ggml/src/ggml-rpc/ggml-rpc.cpp")
    if source.is_file() and "bool rpc_dispatcher::start(" in source.read_text(encoding="utf-8"):
        # Upstream made startup fallible. Keep its return value and avoid
        # registering an unstarted dispatcher or retaining its reserved name.
        contents = contents.replace(b"    void start(", b"    bool start(")
        contents = contents.replace(b"void rpc_dispatcher::start(", b"bool rpc_dispatcher::start(")
        contents = contents.replace(
            b"+    dispatcher->start(endpoint, sock);",
            b"+    if (!dispatcher->start(endpoint, sock)) {\n"
            b"+        std::lock_guard<std::mutex> lock(stream_mutex);\n"
            b"+        stream_names.erase(endpoint);\n"
            b"+        return nullptr;\n"
            b"+    }",
        )
    args = ["git", "-C", str(vendor_path), "apply", "--recount"]
    if subprocess.run(args + ["--reverse", "--check", "-"], input=contents, capture_output=True).returncode == 0:
        return
    # Keep failed forward checks visible in CI, including the exact conflicting
    # file and hunk. Only the expected negative reverse probe is quiet.
    subprocess.run(args + ["--check", "-"], input=contents, check=True)
    subprocess.run(args + ["-"], input=contents, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vendor_path", type=Path)
    apply(parser.parse_args().vendor_path)
