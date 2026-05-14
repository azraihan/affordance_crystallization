#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

TEST_FLAG=""
if [[ "${1:-}" == "--test" ]]; then
    TEST_FLAG="--test"
    echo "=== Feature Extraction [TEST MODE] ==="
else
    echo "=== Feature Extraction (all models) ==="
fi

for MODEL in vjepa2 vjepa2_1 dinov2 random_vit; do
    echo ""
    echo "--- Extracting: $MODEL ---"
    python src/extract_features.py --model "$MODEL" $TEST_FLAG
done
echo ""
echo "=== Extraction complete. ==="
