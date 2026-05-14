#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

TEST_FLAG=""
if [[ "${1:-}" == "--test" ]]; then
    TEST_FLAG="--test"
    echo "=== Head Ablation [TEST MODE] ==="
else
    echo "=== Head Ablation (JEPA + DINOv2 models) ==="
fi

for MODEL in vjepa2 vjepa2_1 dinov2; do
    echo ""
    echo "--- Ablating: $MODEL ---"
    python src/ablate.py --model "$MODEL" $TEST_FLAG
done
echo ""
echo "=== Ablation complete. Results in results/ablation/ ==="
