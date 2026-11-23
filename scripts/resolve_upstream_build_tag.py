"""Select the newest numbered upstream build tag from fetched Git refs."""

import re
import sys
from typing import Iterable, List, Tuple

_BUILD_TAG = re.compile(r"b(0|[1-9][0-9]*)\Z")


def resolve_latest_build_tag(refs: Iterable[str]) -> str:
    """Return the numerically newest ``b<number>`` tag from full refs or tag names."""
    candidates: List[Tuple[int, str]] = []
    for ref in refs:
        tag = ref.strip().rsplit("/", 1)[-1]
        match = _BUILD_TAG.fullmatch(tag)
        if match is not None:
            candidates.append((int(match.group(1)), tag))
    if not candidates:
        raise ValueError("no upstream build tags matching b<number> were found")
    return max(candidates)[1]


def main() -> int:
    try:
        tag = resolve_latest_build_tag(sys.stdin)
    except ValueError as error:
        print(f"Could not resolve latest upstream build tag: {error}", file=sys.stderr)
        return 1
    print(tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
