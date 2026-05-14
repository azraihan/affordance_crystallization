"""
Layer-wise linear probing.

For each layer of each model: fits a multi-label OneVsRest LogisticRegression on
training features and evaluates mean Average Precision (mAP) on the test split.

Output: results/probing/<model_key>_probe_results.json
  {
    "layer_0":  {"mAP": float, "per_class_AP": {"grasp": float, ...}},
    ...
    "layer_24": {...},
    "peak_layer": int,
    "peak_mAP":   float
  }
"""
import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from config import (MODELS, FEATURE_DIR, RESULT_DIR, SEED,
                    PROBE_MAX_ITER, PROBE_C, PROBE_SOLVER,
                    AFFORDANCE_CLASSES, TEST_PROBE_LAYERS)


def load_split(model_key: str, split: str, layer_idx: int):
    path = FEATURE_DIR / model_key / f"{split}_layer{layer_idx:02d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run extract_features.py first.")
    data = np.load(path)
    return data["features"], data["labels"]   # (N, D), (N, 7)


def fit_and_evaluate(X_train, y_train, X_test, y_test) -> dict:
    """Fit OneVsRest logistic regression, return mAP and per-class AP."""
    clf = OneVsRestClassifier(
        LogisticRegression(
            C=PROBE_C, max_iter=PROBE_MAX_ITER,
            solver=PROBE_SOLVER, random_state=SEED, n_jobs=-1,
        )
    )
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)   # (N, 7)

    per_class_ap = {}
    valid_aps    = []
    for i, cls in enumerate(AFFORDANCE_CLASSES):
        if y_test[:, i].sum() == 0:
            per_class_ap[cls] = None   # class absent in test split
        else:
            ap = float(average_precision_score(y_test[:, i], y_score[:, i]))
            per_class_ap[cls] = ap
            valid_aps.append(ap)

    mAP = float(np.mean(valid_aps)) if valid_aps else 0.0
    return {"mAP": mAP, "per_class_AP": per_class_ap}


def probe_model(model_key: str, test_mode: bool = False) -> dict:
    num_layers  = MODELS[model_key]["num_layers"]
    layer_range = range(num_layers + 1)   # 0 through num_layers inclusive
    if test_mode:
        layer_range = [l for l in TEST_PROBE_LAYERS if l <= num_layers]

    results = {}
    maps    = {}

    for layer_idx in tqdm(layer_range, desc=f"Probing {model_key}"):
        try:
            X_train, y_train = load_split(model_key, "train", layer_idx)
            X_test,  y_test  = load_split(model_key, "test",  layer_idx)
        except FileNotFoundError as e:
            print(f"  Skipping layer {layer_idx}: {e}")
            continue

        # z-score normalise using training statistics
        mu    = X_train.mean(axis=0, keepdims=True)
        sigma = X_train.std(axis=0, keepdims=True) + 1e-8
        X_train_n = (X_train - mu) / sigma
        X_test_n  = (X_test  - mu) / sigma

        res = fit_and_evaluate(X_train_n, y_train, X_test_n, y_test)
        results[f"layer_{layer_idx}"] = res
        maps[layer_idx] = res["mAP"]
        print(f"  Layer {layer_idx:2d}: mAP = {res['mAP']:.4f}")

    if not maps:
        raise RuntimeError("No layers probed — check that features exist.")

    peak_layer = max(maps, key=maps.get)
    results["peak_layer"] = peak_layer
    results["peak_mAP"]   = maps[peak_layer]
    print(f"\nPeak layer: {peak_layer}  (mAP = {maps[peak_layer]:.4f})")

    out_dir  = RESULT_DIR / "probing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_key}_probe_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Layer-wise linear probing for affordance classification.")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--test", action="store_true",
                        help="Probe only a subset of layers (fast smoke-test).")
    args = parser.parse_args()
    probe_model(args.model, test_mode=args.test)


if __name__ == "__main__":
    main()
