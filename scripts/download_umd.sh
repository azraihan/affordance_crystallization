#!/usr/bin/env bash
# download_umd.sh — download the UMD Part Affordance Dataset (tools split).
# The "tools" archive contains all 17 object categories used in the paper.
# Correct URL discovered by following the redirect from the original ~fer/ URL.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="data/umd"
mkdir -p "$DATA_DIR"

ARCHIVE="$DATA_DIR/part-affordance-dataset-tools.tar.gz"
URL="https://obj.umiacs.umd.edu/part-affordance/part-affordance-dataset-tools.tar.gz"

echo "=== Downloading UMD Part Affordance Dataset (tools split) ==="

if [ -f "$ARCHIVE" ]; then
    echo "[INFO] Archive already exists at $ARCHIVE — skipping download."
else
    echo "Downloading from: $URL"
    if curl -L --retry 3 --retry-delay 5 -o "$ARCHIVE" "$URL"; then
        echo "Download complete."
    else
        echo "ERROR: curl failed. Try:"
        echo "  wget -O $ARCHIVE '$URL'"
        exit 1
    fi
fi

echo "=== Extracting archive ==="
tar -xzf "$ARCHIVE" -C "$DATA_DIR"
echo "Extraction complete. Contents of $DATA_DIR:"
ls "$DATA_DIR"

echo ""
echo "=== Done. Run: python scripts/verify_umd.py ==="
