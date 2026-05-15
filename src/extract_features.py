"""
Feature extraction: iterates the UMD dataset, runs each frozen model,
saves per-layer features as compressed .npz files.

Output layout (per model):
  features/<model_key>/{train,val,test}_layer{i:02d}.npz
    - 'features': float32 (N, D)
    - 'labels':   float32 (N, 7)
  features/<model_key>/{train,val,test}_labels.npy  (convenience copy)

Layer index convention (consistent across all models):
  layer 0  = patch embedding output (before any transformer block)
  layer 1  = output of transformer block 1
  ...
  layer 24 = output of transformer block 24
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (MODELS, FEATURE_DIR, BATCH_SIZE, NUM_WORKERS, SEED,
                    TEST_N_SAMPLES, TEST_BATCH_SIZE)
from dataset import get_datasets
from models import FeatureExtractor

torch.manual_seed(SEED)
np.random.seed(SEED)


def extract_and_save(
    model_key: str,
    split: str,
    dataset,
    extractor: FeatureExtractor,
    batch_size: int,
):
    if len(dataset) == 0:
        print(f"  Skipping {split} split (empty — normal in test mode with many categories).")
        return

    out_dir = FEATURE_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    )

    # Accumulate per-layer: {layer_idx: [batch_arrays...]}
    per_layer: dict = {}
    all_labels: list = []

    for images, labels in tqdm(loader, desc=f"[{model_key}|{split}]"):
        feat_dict = extractor.extract(images)   # {layer_idx: (B, D)}
        for layer_idx, feat in feat_dict.items():
            per_layer.setdefault(layer_idx, []).append(feat)
        all_labels.append(labels.numpy())

    labels_np = np.concatenate(all_labels, axis=0)   # (N, 7)
    np.save(out_dir / f"{split}_labels.npy", labels_np)

    for layer_idx in sorted(per_layer.keys()):
        feats_np = np.concatenate(per_layer[layer_idx], axis=0)   # (N, D)
        out_path = out_dir / f"{split}_layer{layer_idx:02d}.npz"
        np.savez_compressed(out_path, features=feats_np, labels=labels_np)
        print(f"  Saved {out_path} — shape {feats_np.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract per-layer features from a frozen model.")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode: tiny dataset, small batch, fast smoke-test.")
    args = parser.parse_args()

    model_key  = args.model
    is_test    = args.test
    batch_size = TEST_BATCH_SIZE if is_test else BATCH_SIZE
    max_total  = TEST_N_SAMPLES  if is_test else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | test_mode: {is_test}")

    print(f"\n=== Loading model: {model_key} ===")
    extractor = FeatureExtractor(model_key, device)

    print("\n=== Loading dataset ===")
    train_ds, val_ds, test_ds = get_datasets(max_total=max_total)

    for split_name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        extract_and_save(model_key, split_name, ds, extractor, batch_size)

    print(f"\n=== Done. Features saved to {FEATURE_DIR / model_key} ===")


if __name__ == "__main__":
    main()
