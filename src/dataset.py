"""
PyTorch Dataset for the UMD Part Affordance Dataset (tools split).
Returns (image_tensor, label_vector) pairs.
label_vector: float32 tensor of shape (7,), binary multi-label.

Dataset layout (actual):
  tools/<category>/<stem>_rgb.jpg
  tools/<category>/<stem>_label.mat   # gt_label key, uint8 H×W, values 0–7
"""
import random
import numpy as np
import scipy.io
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from config import (DATA_ROOT, AFFORDANCE_CLASSES, NUM_AFFORDANCES,
                    AFFORDANCE_THRESHOLD, IMG_RESIZE, SEED,
                    TRAIN_SPLIT, VAL_SPLIT, SUBSAMPLE_N)


def find_dataset_root(base: Path) -> Path:
    """Find the subdirectory that directly contains the 'tools/' folder."""
    if (base / "tools").is_dir():
        return base
    for candidate in sorted(base.rglob("tools")):
        if candidate.is_dir():
            return candidate.parent
    raise FileNotFoundError(
        f"'tools/' directory not found under {base}. "
        "Run scripts/download_umd.sh first."
    )


def build_label_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    mask: H×W uint8 array, values 0..7 (0=background, 1..7=affordance class).
    Returns binary float32 vector of shape (NUM_AFFORDANCES,).
    Affordance i is positive if pixels labeled (i+1) cover > AFFORDANCE_THRESHOLD of the image.
    """
    total = mask.size
    label = np.zeros(NUM_AFFORDANCES, dtype=np.float32)
    for i in range(NUM_AFFORDANCES):
        frac = (mask == i + 1).sum() / total
        label[i] = float(frac > AFFORDANCE_THRESHOLD)
    return label


def collect_samples(root: Path) -> List[Tuple[Path, Path]]:
    """
    Walk tools/<category>/*_rgb.jpg and pair with same-directory *_label.mat.
    Skips any image whose label file is missing.
    """
    rgb_paths = sorted(root.glob("tools/*/*_rgb.jpg")) + \
                sorted(root.glob("tools/*/*_rgb.png"))
    samples = []
    for rgb_p in rgb_paths:
        label_stem = rgb_p.stem.replace("_rgb", "_label")
        label_p = rgb_p.parent / (label_stem + ".mat")
        if label_p.exists():
            samples.append((rgb_p, label_p))
    return samples


def split_samples(
    samples: List[Tuple[Path, Path]],
    seed: int = SEED,
    max_total: Optional[int] = None,
) -> Tuple[List, List, List]:
    """
    Stratified split by object category.
    If max_total is set, sample that many items proportionally per category (for test mode).
    """
    random.seed(seed)
    np.random.seed(seed)

    by_cat: dict = defaultdict(list)
    for rgb_p, label_p in samples:
        cat = rgb_p.parent.name   # tools/<category>/img_rgb.jpg
        by_cat[cat].append((rgb_p, label_p))

    if max_total is not None:
        # Proportional allocation: each category keeps (cat_size / total) * max_total items.
        # Use floor first, then distribute the leftover slots to categories with the largest
        # fractional remainders, so the final count is exactly max_total.
        total_pool = len(samples)
        frac = max_total / total_pool
        cats = list(by_cat.keys())

        floors = {cat: max(1, int(len(by_cat[cat]) * frac)) for cat in cats}
        leftover = max_total - sum(floors.values())
        # Sort by fractional remainder descending to allocate leftover slots fairly
        remainders = sorted(
            cats,
            key=lambda c: (len(by_cat[c]) * frac - floors[c]),
            reverse=True,
        )
        for cat in remainders[:max(0, leftover)]:
            floors[cat] += 1

        for cat in cats:
            random.shuffle(by_cat[cat])
            by_cat[cat] = by_cat[cat][: floors[cat]]

    train, val, test = [], [], []
    for cat, items in by_cat.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * TRAIN_SPLIT)
        n_val   = int(n * VAL_SPLIT)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    return train, val, test


def get_transform(img_size: int = IMG_RESIZE) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


class UMDAffordanceDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, Path]], transform=None):
        self.samples   = samples
        self.transform = transform or get_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_p, label_p = self.samples[idx]

        img = Image.open(rgb_p).convert("RGB")
        img_tensor = self.transform(img)   # (3, H, W)

        mask = scipy.io.loadmat(str(label_p))["gt_label"]   # uint8 H×W, values 0–7
        label = build_label_from_mask(mask)

        return img_tensor, torch.from_numpy(label)


def get_datasets(data_root: Path = DATA_ROOT, max_total: Optional[int] = SUBSAMPLE_N):
    """Returns (train_dataset, val_dataset, test_dataset).
    max_total: if set, subsample to this many images total (for --test mode).
    """
    root = find_dataset_root(data_root)
    samples = collect_samples(root)
    if not samples:
        raise RuntimeError(f"No samples found at {root}. Run download_umd.sh first.")
    print(f"Total samples found: {len(samples)}")

    train_s, val_s, test_s = split_samples(samples, max_total=max_total)
    print(f"Split — train: {len(train_s)}, val: {len(val_s)}, test: {len(test_s)}")

    return (
        UMDAffordanceDataset(train_s),
        UMDAffordanceDataset(val_s),
        UMDAffordanceDataset(test_s),
    )
