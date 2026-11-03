"""Fetch selected draft assets for the hosted verification job using HTTPS."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def open_https(url):
    opener = build_opener(NoRedirect())
    for _ in range(4):
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or parsed.hostname not in {
                "api.github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}
                or parsed.username or parsed.password or parsed.port not in (None, 443)):
            raise RuntimeError("Release asset URL is not approved")
        headers = {"Accept": "application/octet-stream", "User-Agent": "release-asset-verification"}
        if parsed.hostname == "api.github.com":
            headers["Authorization"] = "Bearer " + os.environ["GH_TOKEN"]
            if "/releases/assets/" not in parsed.path:
                headers["Accept"] = "application/vnd.github+json"
        try:
            return opener.open(Request(url, headers=headers), timeout=120)
        except HTTPError as error:
            try:
                if error.code not in (301, 302, 303, 307, 308):
                    raise RuntimeError("GitHub release transfer failed") from None
                url = error.headers.get("Location", "")
            finally:
                error.close()
    raise RuntimeError("Release transfer exceeded its redirect limit")


def api_json(endpoint):
    with open_https("https://api.github.com" + endpoint) as response:
        raw = response.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise RuntimeError("GitHub release metadata exceeds its limit")
    return json.loads(raw)


def fetch(mode):
    repository = os.environ["GITHUB_REPOSITORY"]
    release_id = os.environ["RELEASE_ID"]
    if repository not in {"fael-ai-org/llama-cpp-py-sync", "fael-ai-org/python-builds"} or not re.fullmatch(r"[0-9]{1,20}", release_id):
        raise RuntimeError("Release identity is not approved")
    prefix = f"/repos/{repository}/releases/{release_id}"
    release = api_json(prefix)
    if release.get("id") != int(release_id) or release.get("draft") is not True or release.get("immutable"):
        raise RuntimeError("Only the selected mutable draft can be verified")
    root = Path("signed-release")
    root.mkdir(exist_ok=True)
    if mode == "metadata":
        names = ["release-signing.json", "release-signing.sig.json"]
    elif mode == "files":
        raw = (root / "release-signing.json").read_bytes()
        if len(raw) > 1024 * 1024:
            raise RuntimeError("Release manifest exceeds its limit")
        manifest = json.loads(raw)
        if manifest.get("tag") != release["tag_name"]:
            raise RuntimeError("Release tag changed after manifest authentication")
        names = [item["name"] for item in manifest["files"]]
        if len(names) > 1000 or len(set(names)) != len(names):
            raise RuntimeError("Release file inventory is invalid")
    else:
        raise RuntimeError("Unknown release verification download phase")
    assets = []
    for page in range(1, 101):
        values = api_json(prefix + f"/assets?per_page=100&page={page}")
        if not isinstance(values, list):
            raise RuntimeError("Release asset inventory is invalid")
        assets.extend(values)
        if len(values) < 100:
            break
    else:
        raise RuntimeError("Release asset inventory exceeds its limit")
    for name in names:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,219}", name) or name.endswith("."):
            raise RuntimeError("Release asset filename is invalid")
        matches = [asset for asset in assets if asset["name"] == name]
        if len(matches) != 1:
            raise RuntimeError("Release asset is missing or ambiguous")
        asset = matches[0]
        if (type(asset["id"]) is not int or asset["id"] <= 0 or type(asset["size"]) is not int
                or not 0 < asset["size"] <= (1024 * 1024 if mode == "metadata" else 2 * 1024**3)):
            raise RuntimeError("Release asset identity or size is invalid")
        count = 0
        with open_https(f"https://api.github.com/repos/{repository}/releases/assets/{asset['id']}") as incoming:
            with (root / name).open("xb") as output:
                while chunk := incoming.read(1024 * 1024):
                    count += len(chunk)
                    if count > asset["size"]:
                        raise RuntimeError("Release asset exceeds its declared size")
                    output.write(chunk)
        if count != asset["size"]:
            raise RuntimeError("Release asset was truncated")


if __name__ == "__main__":
    fetch(sys.argv[1])
