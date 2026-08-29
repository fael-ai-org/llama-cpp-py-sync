#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <artifact-root>" >&2
  exit 2
fi

root="$1"
if [[ ! -d "$root" ]]; then
  echo "Artifact directory does not exist: $root" >&2
  exit 1
fi

# Ad-hoc signing is deliberately test-only. It seals each LLaMA-owned Mach-O
# file but does not establish an Apple-trusted Developer ID publisher.
owned_name() {
  local name
  name="$(basename "$1" | tr '[:upper:]' '[:lower:]')"
  case "$name" in
    llama*|libllama*|ggml*|libggml*|mtmd*|libmtmd*|*.so)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

signed=0
while IFS= read -r -d '' candidate; do
  if ! owned_name "$candidate"; then
    continue
  fi
  if ! file -b "$candidate" | grep -q 'Mach-O'; then
    continue
  fi

  codesign --force --sign - --timestamp=none "$candidate"
  codesign --verify --strict --verbose=2 "$candidate"
  echo "Ad-hoc signed: $candidate"
  signed=$((signed + 1))
done < <(find "$root" -type f -print0)

if [[ "$signed" -eq 0 ]]; then
  echo "No LLaMA-owned Mach-O files were found under $root." >&2
  exit 1
fi

echo "Ad-hoc signed and verified $signed LLaMA-owned Mach-O file(s)."
