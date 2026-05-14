#!/usr/bin/env bash
# run_all_experiments.sh — full pipeline from extraction to figures.
# Usage:
#   bash scripts/run_all_experiments.sh           # full run
#   bash scripts/run_all_experiments.sh --test    # smoke-test (tiny data, fast)
set -euo pipefail
cd "$(dirname "$0")/.."

TEST_FLAG=""
if [[ "${1:-}" == "--test" ]]; then
    TEST_FLAG="--test"
    echo "========================================"
    echo "  Full Pipeline  [TEST MODE]"
    echo "========================================"
else
    echo "========================================"
    echo "  Full Experiment Pipeline"
    echo "========================================"
fi

echo ""
echo "[Step 1/4] Feature Extraction"
bash scripts/run_extraction.sh $TEST_FLAG

echo ""
echo "[Step 2/4] Layer-wise Probing"
bash scripts/run_probing.sh $TEST_FLAG

echo ""
echo "[Step 3/4] Head Ablation"
bash scripts/run_ablation.sh $TEST_FLAG

echo ""
echo "[Step 4/4] Figures and Tables"
python src/generate_figures.py
python src/generate_tables.py

echo ""
echo "========================================"
echo "  Done."
echo "  Figures  → figures/"
echo "  Tables   → tables/"
echo "  Raw data → figures/data/  (for restyling figures)"
echo "========================================"
