#!/usr/bin/env bash
# Fetch the pretrained MOS checkpoint from the GitHub release into the path the
# configs expect (runs/consistency_ft/best.pt). Verifies SHA-256; safe to re-run.
set -euo pipefail

URL="https://github.com/subhransu10/sparseunet4d/releases/tag/v2.0"
SHA="sha256:1852e83806c30ce1eae392c3105e051e8fcefb16c739f60cb4d362b70685a214"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # deploy/ -> repo root
DEST_DIR="$REPO/runs/consistency_ft"
DEST="$DEST_DIR/best.pt"
mkdir -p "$DEST_DIR"

if [ -f "$DEST" ] && echo "$SHA  $DEST" | sha256sum -c - >/dev/null 2>&1; then
  echo "checkpoint already present & verified: $DEST"
  exit 0
fi

echo "downloading MOS checkpoint (~29 MB) ..."
if command -v wget >/dev/null; then wget -O "$DEST" "$URL"
else curl -L -o "$DEST" "$URL"; fi

echo "$SHA  $DEST" | sha256sum -c - && echo "OK -> $DEST"
