#!/bin/bash
# ============================================================
# Download TeCoA / FARE checkpoints for Table 6 (AFT superposition).
#
# These two .pth files (~1.4GB total) are NOT in the git repo
# (they exceed GitHub's 100MB file limit). They must be uploaded
# to a GitHub Release (or any HTTP host) first, then this script
# fetches them into checkpoints/.
#
# Usage:
#   bash scripts/download_checkpoints.sh                 # default GitHub Releases
#   TAG=v1.0 bash scripts/download_checkpoints.sh        # custom release tag
#   BASE_URL=https://example.com/weights bash scripts/download_checkpoints.sh
#
# Files: checkpoints/fare_vit_b32.pth, checkpoints/tecoa_vit_b32.pth
# ============================================================
set -e

TAG=${TAG:-v1.0}
BASE_URL=${BASE_URL:-"https://github.com/1011245276/NAC/releases/download/${TAG}"}

FILES=("fare_vit_b32.pth" "tecoa_vit_b32.pth")
mkdir -p checkpoints

for f in "${FILES[@]}"; do
  dest="checkpoints/$f"
  if [ -s "$dest" ]; then
    echo "[skip] $dest already exists ($(du -h "$dest" | cut -f1))"
    continue
  fi
  echo "[download] $f -> $dest"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "$dest" "$BASE_URL/$f"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$dest" "$BASE_URL/$f"
  else
    echo "[error] neither curl nor wget found. Install one or download manually to $dest" >&2
    exit 1
  fi
  echo "  saved: $(du -h "$dest" | cut -f1)"
done

echo "[OK] Checkpoints ready in checkpoints/"
echo "Table 6 (AFT superposition) will now run via scripts/reproduce_aft.sh"
