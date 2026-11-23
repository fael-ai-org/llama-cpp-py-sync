"""Validate GitHub's latest-release response and return its stable version tag."""

import json
import re
import sys
from typing import Any

_STABLE_VERSION_TAG = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z")


def resolve_release_tag(release: Any) -> str:
    """Return a published, stable ``vMAJOR.MINOR.PATCH`` release tag."""
    if not isinstance(release, dict):
        raise ValueError("release response must be a JSON object")
    if release.get("draft") is not False:
        raise ValueError("latest release must not be a draft")
    if release.get("prerelease") is not False:
        raise ValueError("latest release must not be a prerelease")
    published_at = release.get("published_at")
    if not isinstance(published_at, str) or not published_at.strip():
        raise ValueError("latest release must have a publication timestamp")
    tag = release.get("tag_name")
    if not isinstance(tag, str) or _STABLE_VERSION_TAG.fullmatch(tag) is None:
        raise ValueError("latest release tag must use vMAJOR.MINOR.PATCH versioning")
    return tag


def main() -> int:
    try:
        release = json.load(sys.stdin)
        tag = resolve_release_tag(release)
    except (json.JSONDecodeError, ValueError) as error:
        print(f"Could not resolve latest upstream version release: {error}", file=sys.stderr)
        return 1
    print(tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
