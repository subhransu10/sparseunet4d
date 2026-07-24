#!/usr/bin/env bash
# Fetch the pretrained MOS checkpoint from the GitHub release into the path the
# configs expect (runs/consistency_ft/best.pt). Verifies SHA-256; safe to re-run.
set -euo pipefail

URL="https://github.com/subhransu10/sparseunet4d/releases/download/v1.0/consistency_ft_best.pt"
SHA="ee16661acb2590c3008af8705a053357a871401355782cbda85afb8da03f68b7"

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
