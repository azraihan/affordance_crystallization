#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

TEST_FLAG=""
if [[ "${1:-}" == "--test" ]]; then
    TEST_FLAG="--test"
    echo "=== Layer-wise Probing [TEST MODE] ==="
else
    echo "=== Layer-wise Probing (all models) ==="
fi

for MODEL in vjepa2 vjepa2_1 dinov2 random_vit; do
    echo ""
    echo "--- Probing: $MODEL ---"
    python src/probe.py --model "$MODEL" $TEST_FLAG
done
echo ""
echo "=== Probing complete. Results in results/probing/ ==="
