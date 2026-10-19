"""Apply the reviewed RPC stream extension, rejecting incompatible sources."""
import argparse
import subprocess
from pathlib import Path


def apply(vendor_path: Path):
    patch = Path(__file__).resolve().parent.parent / "patches" / "rpc-stream.patch"
    args = ["git", "-C", str(vendor_path), "apply"]
    if subprocess.run(args + ["--reverse", "--check", str(patch)], capture_output=True).returncode == 0:
        return
    # Keep failed forward checks visible in CI, including the exact conflicting
    # file and hunk. Only the expected negative reverse probe is quiet.
    subprocess.run(args + ["--check", str(patch)], check=True)
    subprocess.run(args + [str(patch)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vendor_path", type=Path)
    apply(parser.parse_args().vendor_path)
