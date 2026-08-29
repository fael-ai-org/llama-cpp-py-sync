"""Create test-only detached signatures for final Linux release artifacts.

Linux shared objects have no Authenticode-equivalent trust path. This helper
therefore signs the final wheel(s) and a SHA-256 manifest with a short-lived
GPG key, exports only the public key, and deletes the private key with its
temporary GNUPGHOME when the process exits.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

KEY_UID = "Ephemeral CI Test Key (TEST ONLY)"


def run(command: list[str], *, homedir: Path, output: Path | None = None) -> None:
    full_command = ["gpg", "--batch", "--homedir", str(homedir), *command]
    if output is None:
        subprocess.run(full_command, check=True)
        return
    with output.open("wb") as stream:
        subprocess.run(full_command, check=True, stdout=stream)


def fingerprint(homedir: Path) -> str:
    result = subprocess.run(
        ["gpg", "--batch", "--homedir", str(homedir), "--with-colons", "--list-secret-keys"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        fields = line.split(":")
        if fields and fields[0] == "fpr" and len(fields) > 9 and fields[9]:
            return fields[9]
    raise RuntimeError("GPG did not return the test signing-key fingerprint")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Directory containing final Linux artifacts")
    parser.add_argument("--public-key", type=Path, required=True, help="Output path for the public test key")
    parser.add_argument("--manifest", type=Path, required=True, help="Output path for the SHA-256 manifest")
    args = parser.parse_args()

    root = args.root.resolve()
    public_key = args.public_key.resolve()
    manifest = args.manifest.resolve()
    if not root.is_dir():
        raise SystemExit(f"Artifact directory does not exist: {root}")
    if shutil.which("gpg") is None:
        raise SystemExit("gpg is required for Linux detached signatures")

    artifacts = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".whl", ".tar", ".gz", ".zip", ".deb", ".rpm"}
    )
    if not artifacts:
        raise SystemExit(f"No final Linux artifacts found under {root}")

    public_key.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fael-linux-signing-") as temporary:
        homedir = Path(temporary)
        os.chmod(homedir, 0o700)
        try:
            run(
                [
                    "--pinentry-mode",
                    "loopback",
                    "--passphrase",
                    "",
                    "--quick-gen-key",
                    KEY_UID,
                    "ed25519",
                    "sign",
                    "1d",
                ],
                homedir=homedir,
            )
            key_id = fingerprint(homedir)

            manifest.write_text(
                "".join(f"{sha256(path)}  {path.name}\n" for path in artifacts),
                encoding="utf-8",
            )
            for artifact in artifacts:
                signature = artifact.with_name(f"{artifact.name}.asc")
                run(
                    ["--local-user", key_id, "--armor", "--detach-sign", "--output", str(signature), str(artifact)],
                    homedir=homedir,
                )
                print(f"Signed: {artifact.name}")

            run(
                ["--local-user", key_id, "--armor", "--detach-sign", "--output", str(manifest.with_name("SHA256SUMS.asc")), str(manifest)],
                homedir=homedir,
            )
            run(["--armor", "--export", key_id], homedir=homedir, output=public_key)
            print(f"GPG test-key fingerprint: {key_id}")
            print(f"Public key: {public_key}")

            for artifact in artifacts:
                signature = artifact.with_name(f"{artifact.name}.asc")
                run(["--verify", str(signature), str(artifact)], homedir=homedir)
            run(
                ["--verify", str(manifest.with_name("SHA256SUMS.asc")), str(manifest)],
                homedir=homedir,
            )
        finally:
            gpgconf = shutil.which("gpgconf")
            if gpgconf:
                subprocess.run(
                    [gpgconf, "--homedir", str(homedir), "--kill", "gpg-agent"],
                    capture_output=True,
                    text=True,
                    check=False,
                )

    print("Removed the ephemeral Linux private key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
