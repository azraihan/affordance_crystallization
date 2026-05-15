#!/usr/bin/env python3
"""
Verifies UMD dataset structure and prints statistics.
Exits 0 if OK, 1 if broken.

Dataset layout:
  tools/<category>/<stem>_rgb.jpg
  tools/<category>/<stem>_label.mat   (gt_label key, uint8 H×W, values 0–7)
"""
import sys
from pathlib import Path
import numpy as np
import scipy.io
from PIL import Image

DATA_ROOT = Path("data/umd")
AFFORDANCE_CLASSES = ["grasp", "cut", "scoop", "wrap-grasp", "poke", "support", "contain"]


def find_root(base: Path) -> Path:
    if (base / "tools").is_dir():
        return base
    for candidate in sorted(base.rglob("tools")):
        if candidate.is_dir():
            return candidate.parent
    raise FileNotFoundError(
        f"'tools/' not found under {base}. Run download_umd.sh first."
    )


def main():
    try:
        root = find_root(DATA_ROOT)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    rgb_files = sorted(root.glob("tools/*/*_rgb.jpg")) + \
                sorted(root.glob("tools/*/*_rgb.png"))

    paired, missing_label = [], []
    for rgb_p in rgb_files:
        label_stem = rgb_p.stem.replace("_rgb", "_label")
        label_p = rgb_p.parent / (label_stem + ".mat")
        if label_p.exists():
            paired.append((rgb_p, label_p))
        else:
            missing_label.append(rgb_p)

    print(f"Dataset root   : {root}")
    print(f"RGB images     : {len(rgb_files)}")
    print(f"Paired samples : {len(paired)}")

    if not rgb_files:
        print("ERROR: No RGB images found.")
        sys.exit(1)

    if missing_label:
        print(f"WARNING: {len(missing_label)} RGB images have no matching label .mat")

    # Spot-check first 5 images
    for rgb_p, _ in paired[:5]:
        try:
            Image.open(rgb_p).verify()
        except Exception as e:
            print(f"ERROR: Cannot open {rgb_p}: {e}")
            sys.exit(1)

    categories = sorted({rgb_p.parent.name for rgb_p, _ in paired})
    print(f"Object categories ({len(categories)}): {categories}")

    _, sample_label_p = paired[0]
    mat = scipy.io.loadmat(str(sample_label_p))
    mask = mat["gt_label"]
    unique_vals = np.unique(mask)
    print(f"Sample label pixel values: {unique_vals}  (expected 0–7)")

    print("\nVerification PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
