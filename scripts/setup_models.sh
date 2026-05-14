#!/usr/bin/env bash
# setup_models.sh — clone the facebookresearch/vjepa2 repo and download checkpoints.
# Run once before any experiment.
set -euo pipefail
cd "$(dirname "$0")/.."   # run from project root

REPO_DIR="third_party/vjepa2"
CKPT_DIR="checkpoints"
mkdir -p "$CKPT_DIR"

# ── 1. Clone V-JEPA 2 repo ────────────────────────────────────────────────────
if [ -d "$REPO_DIR/.git" ]; then
    echo "[INFO] $REPO_DIR already exists — skipping clone."
else
    echo "=== Cloning facebookresearch/vjepa2 ==="
    git clone --depth 1 https://github.com/facebookresearch/vjepa2.git "$REPO_DIR"
fi

# ── 2. Download V-JEPA 2 ViT-L checkpoint (~4.8 GB) ──────────────────────────
VJEPA2_CKPT="$CKPT_DIR/vjepa2_vitl.pt"
if [ -f "$VJEPA2_CKPT" ]; then
    echo "[INFO] $VJEPA2_CKPT already exists — skipping download."
else
    echo "=== Downloading V-JEPA 2 ViT-L checkpoint (~4.8 GB) ==="
    curl -L --retry 3 --retry-delay 5 \
        -o "$VJEPA2_CKPT" \
        "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt"
    echo "Downloaded: $VJEPA2_CKPT"
fi

# ── 3. Download V-JEPA 2.1 ViT-L checkpoint (~4.8 GB) ────────────────────────
VJEPA21_CKPT="$CKPT_DIR/vjepa2_1_vitl.pt"
if [ -f "$VJEPA21_CKPT" ]; then
    echo "[INFO] $VJEPA21_CKPT already exists — skipping download."
else
    echo "=== Downloading V-JEPA 2.1 ViT-L checkpoint (~4.8 GB) ==="
    curl -L --retry 3 --retry-delay 5 \
        -o "$VJEPA21_CKPT" \
        "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
    echo "Downloaded: $VJEPA21_CKPT"
fi

echo ""
echo "=== Setup complete. ==="
echo "  Repo:        $REPO_DIR"
echo "  V-JEPA 2:    $VJEPA2_CKPT"
echo "  V-JEPA 2.1:  $VJEPA21_CKPT"
echo ""
echo "Next: bash scripts/download_umd.sh"
