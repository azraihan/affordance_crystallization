"""
Attention head ablation study at each model's peak layer (from probing results).

Ablation mechanism (Michel et al. 2019):
  For head h at the peak layer, we zero the h*head_dim:(h+1)*head_dim columns of the
  attention value output BEFORE the output projection Linear. This removes head h's
  contribution to the attention sub-layer without affecting the MLP or other heads.
  Implemented as a register_forward_pre_hook on block.attn.proj (V-JEPA) or
  block.attention.output.dense (DINOv2 HuggingFace).

After ablation, features are mean-pooled from the (ablated) peak-layer block output,
then probed with a linear classifier to measure the mAP drop.

Output: results/ablation/<model_key>_ablation_results.json
  {
    "peak_layer":    int,
    "baseline_mAP":  float,
    "per_head":      {"head_0": {"mAP": float, "drop": float, "importance": float}, ...},
    "head_ranking":  [int, ...],          # descending importance
    "keep_top_k":    {"k1": {"mAP": ..., "drop": ..., "pct_heads_kept": ...}, ...}
  }
"""
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from config import (MODELS, FEATURE_DIR, RESULT_DIR, SEED,
                    PROBE_MAX_ITER, PROBE_C, PROBE_SOLVER,
                    AFFORDANCE_CLASSES, KEEP_TOP_K_HEADS, TEST_KEEP_TOP_K)
from dataset import get_datasets
from models import FeatureExtractor, make_ablation_hooks, run_forward_pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_peak_layer(model_key: str) -> int:
    path = RESULT_DIR / "probing" / f"{model_key}_probe_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Run probe.py --model {model_key} first.")
    with open(path) as f:
        return int(json.load(f)["peak_layer"])


def load_saved_features(model_key: str, split: str, layer_idx: int):
    path = FEATURE_DIR / model_key / f"{split}_layer{layer_idx:02d}.npz"
    data = np.load(path)
    return data["features"], data["labels"]


def _quick_mAP(X_train, y_train, X_test, y_test) -> float:
    """Normalise, fit probe, return mAP."""
    mu    = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True) + 1e-8
    Xtr   = (X_train - mu) / sigma
    Xte   = (X_test  - mu) / sigma

    clf = OneVsRestClassifier(
        LogisticRegression(C=PROBE_C, max_iter=PROBE_MAX_ITER,
                           solver=PROBE_SOLVER, random_state=SEED, n_jobs=-1)
    )
    clf.fit(Xtr, y_train)
    y_score = clf.predict_proba(Xte)

    aps = []
    for c in range(y_train.shape[1]):
        if y_test[:, c].sum() > 0:
            aps.append(average_precision_score(y_test[:, c], y_score[:, c]))
    return float(np.mean(aps)) if aps else 0.0


def _extract_ablated_features(
    extractor: FeatureExtractor,
    loader: DataLoader,
    peak_layer: int,
    heads_to_zero: list,
) -> np.ndarray:
    """
    Run a forward pass with specified heads zeroed at peak_layer.
    Returns (N, D) array of mean-pooled features from that layer.
    """
    cfg      = extractor.cfg
    model    = extractor.model
    device   = extractor.device
    head_dim = cfg["hidden_dim"] // cfg["num_heads"]

    all_feats = []
    for images, _ in tqdm(loader, desc=f"    ablate heads={heads_to_zero[:3]}{'...' if len(heads_to_zero)>3 else ''}",
                          leave=False):
        buffer = []
        handles = make_ablation_hooks(
            model        = model,
            model_type   = extractor.model_type,
            peak_layer   = peak_layer,
            heads_to_zero= heads_to_zero,
            head_dim     = head_dim,
            collection_buffer = buffer,
        )
        run_forward_pass(model, extractor.model_type, images, device,
                         video_frames=cfg.get("video_frames", 8))
        for h in handles:
            h.remove()

        if not buffer:
            raise RuntimeError(
                f"Ablation collection hook captured nothing at layer {peak_layer}. "
                "Check block indexing in models.make_ablation_hooks."
            )
        all_feats.append(np.concatenate(buffer, axis=0))   # (B, D)

    return np.concatenate(all_feats, axis=0)   # (N, D)


# ── Main ablation logic ───────────────────────────────────────────────────────

def ablate_model(model_key: str, test_mode: bool = False):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor  = FeatureExtractor(model_key, device)
    peak_layer = get_peak_layer(model_key)
    num_heads  = MODELS[model_key]["num_heads"]

    print(f"\n=== Ablating {model_key} | peak_layer={peak_layer} | num_heads={num_heads} ===")

    # Dataset must use the same split as feature extraction so that y_train / y_test
    # from saved features align with the ablation loader's sample order and count.
    if test_mode:
        train_ds, _, test_ds = get_datasets(max_total=200)
    else:
        train_ds, _, test_ds = get_datasets()

    b  = 4 if test_mode else 16
    nw = 0 if test_mode else 4
    train_loader = DataLoader(train_ds, batch_size=b, shuffle=False, num_workers=nw)
    test_loader  = DataLoader(test_ds,  batch_size=b, shuffle=False, num_workers=nw)

    # Baseline: use saved features (no ablation)
    X_train_base, y_train = load_saved_features(model_key, "train", peak_layer)
    X_test_base,  y_test  = load_saved_features(model_key, "test",  peak_layer)
    baseline_mAP = _quick_mAP(X_train_base, y_train, X_test_base, y_test)
    print(f"Baseline mAP at layer {peak_layer}: {baseline_mAP:.4f}")

    results = {
        "peak_layer":   peak_layer,
        "baseline_mAP": baseline_mAP,
        "per_head":     {},
        "head_ranking": [],
        "keep_top_k":   {},
    }

    # ── Per-head ablation ─────────────────────────────────────────────────────
    head_importances = []
    for head_idx in tqdm(range(num_heads), desc="Per-head ablation"):
        X_tr = _extract_ablated_features(extractor, train_loader, peak_layer, [head_idx])
        X_te = _extract_ablated_features(extractor, test_loader,  peak_layer, [head_idx])
        mAP  = _quick_mAP(X_tr, y_train, X_te, y_test)
        drop = baseline_mAP - mAP
        head_importances.append(drop)
        results["per_head"][f"head_{head_idx}"] = {
            "mAP": mAP, "drop": drop, "importance": drop,
        }
        print(f"  Head {head_idx:2d}: mAP={mAP:.4f}  drop={drop:+.4f}")

    # Rank heads: highest drop = most important
    head_ranking = sorted(range(num_heads), key=lambda h: head_importances[h], reverse=True)
    results["head_ranking"] = head_ranking

    # ── Keep top-k heads (pruning efficiency) ─────────────────────────────────
    k_vals = TEST_KEEP_TOP_K if test_mode else KEEP_TOP_K_HEADS
    for k in k_vals:
        k = min(k, num_heads)
        top_k      = head_ranking[:k]
        heads_zero = [h for h in range(num_heads) if h not in top_k]

        X_tr = _extract_ablated_features(extractor, train_loader, peak_layer, heads_zero)
        X_te = _extract_ablated_features(extractor, test_loader,  peak_layer, heads_zero)
        mAP  = _quick_mAP(X_tr, y_train, X_te, y_test)
        drop = baseline_mAP - mAP
        results["keep_top_k"][f"k{k}"] = {
            "mAP": mAP, "drop": drop,
            "pct_heads_kept": k / num_heads,
        }
        print(f"  Keep top-{k:2d} heads: mAP={mAP:.4f}  drop={drop:+.4f}  "
              f"({k}/{num_heads} heads = {k/num_heads*100:.0f}%)")

    out_dir  = RESULT_DIR / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_key}_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Attention head ablation at each model's peak layer.")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--test", action="store_true",
                        help="Run with small dataset and abbreviated k sweep.")
    args = parser.parse_args()
    ablate_model(args.model, test_mode=args.test)


if __name__ == "__main__":
    main()
