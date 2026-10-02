"""Apply the reviewed RPC stream extension, rejecting incompatible sources."""
import argparse
import subprocess
from pathlib import Path


def apply(vendor_path: Path):
    patch = Path(__file__).resolve().parent.parent / "patches" / "rpc-stream.patch"
    args = ["git", "-C", str(vendor_path), "apply"]
    if subprocess.run(args + ["--reverse", "--check", str(patch)], capture_output=True).returncode == 0:
        return
    subprocess.run(args + ["--check", str(patch)], check=True, capture_output=True)
    subprocess.run(args + [str(patch)], check=True, capture_output=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vendor_path", type=Path)
    apply(parser.parse_args().vendor_path)
