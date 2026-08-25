#!/usr/bin/env bash
# Download the published checkpoint and verify it before installation.
set -euo pipefail

ASSET="best.pt"
URL="https://github.com/subhransu10/sparseunet4d/releases/latest/download/${ASSET}"
SHA256="65f7525f00a4a490df30ec91b5db713d865f30dffd76b4c7f9dfcbc353e31f1c"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$REPO/checkpoints/sparseunet4d_semantickitti"
DEST="$DEST_DIR/best.pt"
TMP="$DEST.part"
mkdir -p "$DEST_DIR"

verify() {
  echo "$SHA256  $1" | sha256sum -c - >/dev/null 2>&1
}

if [[ -f "$DEST" ]] && verify "$DEST"; then
  echo "Checkpoint already present and verified: $DEST"
  exit 0
fi

rm -f "$TMP"
echo "Downloading $ASSET ..."
if command -v curl >/dev/null 2>&1; then
  curl --fail --location --retry 3 --output "$TMP" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget --tries=3 --output-document="$TMP" "$URL"
else
  echo "ERROR: install curl or wget and rerun this script." >&2
  exit 1
fi

if ! verify "$TMP"; then
  rm -f "$TMP"
  echo "ERROR: checkpoint checksum mismatch; partial file removed." >&2
  exit 1
fi

mv "$TMP" "$DEST"
echo "Checkpoint installed and verified: $DEST"
