#!/usr/bin/env python3
"""
Verifies UMD dataset structure and prints statistics.
Exits 0 if OK, 1 if broken.
"""
import sys
from pathlib import Path
import numpy as np
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

    rgb_files   = sorted(root.glob("tools/*/rgb/*.jpg")) + \
                  sorted(root.glob("tools/*/rgb/*.png"))
    label_files = sorted(root.glob("tools/*/label/*.png"))

    print(f"Dataset root : {root}")
    print(f"RGB images   : {len(rgb_files)}")
    print(f"Label masks  : {len(label_files)}")

    if not rgb_files:
        print("ERROR: No RGB images found.")
        sys.exit(1)

    if len(rgb_files) != len(label_files):
        print(f"WARNING: Mismatch — {len(rgb_files)} RGB vs {len(label_files)} labels.")

    # Spot-check first 5 images
    for f in rgb_files[:5]:
        try:
            Image.open(f).verify()
        except Exception as e:
            print(f"ERROR: Cannot open {f}: {e}")
            sys.exit(1)

    categories = sorted({f.parent.parent.name for f in rgb_files})
    print(f"Object categories ({len(categories)}): {categories}")

    sample_label = np.array(Image.open(label_files[0]))
    if sample_label.ndim == 3:
        sample_label = sample_label[:, :, 0]
    unique_vals = np.unique(sample_label)
    print(f"Sample label pixel values: {unique_vals}  (expected 0–7)")

    print("\nVerification PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
