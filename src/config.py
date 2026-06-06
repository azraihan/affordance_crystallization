"""
Single source of truth for all hyperparameters and paths.
Modify only this file to change experimental settings.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
DATA_ROOT     = ROOT / "data" / "umd"
FEATURE_DIR   = ROOT / "features"
RESULT_DIR    = ROOT / "results"
FIGURE_DIR    = ROOT / "figures"
TABLE_DIR     = ROOT / "tables"

# Path to the cloned facebookresearch/vjepa2 repo (populated by setup_models.sh)
VJEPA2_REPO_PATH = ROOT / "third_party" / "vjepa2"

# ── Checkpoint download URLs (direct from Meta CDN) ────────────────────────────
CHECKPOINT_URLS = {
    "vjepa2":   "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt",
    "vjepa2_1": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
}
CHECKPOINT_DIR = ROOT / "checkpoints"

# ── Models ─────────────────────────────────────────────────────────────────────
# Both JEPA models use ViT-L (the smallest model Meta released for V-JEPA 2).
# V-JEPA 2 ViT-B was never released publicly.
# DINOv2 ViT-L used for fair architectural comparison (24 layers, same as ViT-L JEPA).
MODELS = {
    "vjepa2": {
        "type":         "vjepa",
        "checkpoint":   str(CHECKPOINT_DIR / "vjepa2_vitl.pt"),
        "checkpoint_key": "target_encoder",   # key inside the .pt file
        "model_name":   "vit_large",          # function in vjepa2 src/models/vision_transformer.py
        "model_kwargs": {
            "patch_size":           16,
            "tubelet_size":         2,
            "uniform_power":        True,
            "use_rope":             True,
            "img_temporal_dim_size": None,
        },
        "num_layers":   24,
        "num_heads":    16,
        "hidden_dim":   1024,
        "video_frames": 8,    # replicate static image N times; official eval uses 16 but 8 saves VRAM on P100
        "img_size":     224,
    },
    "vjepa2_1": {
        "type":         "vjepa",
        "checkpoint":   str(CHECKPOINT_DIR / "vjepa2_1_vitl.pt"),
        "checkpoint_key": "ema_encoder",
        "model_name":   "vit_large",
        "model_kwargs": {
            "patch_size":           16,
            "tubelet_size":         2,
            "uniform_power":        True,
            "use_rope":             True,
            "img_temporal_dim_size": 1,   # V-JEPA 2.1 uses a separate image patch embed
        },
        "num_layers":   24,
        "num_heads":    16,
        "hidden_dim":   1024,
        "video_frames": 8,
        "img_size":     224,
    },
    "dinov2": {
        "type":         "dinov2",
        "hf_id":        "facebook/dinov2-large",   # ViT-L: 24 layers, matches JEPA models
        "num_layers":   24,
        "num_heads":    16,
        "hidden_dim":   1024,
        "img_size":     224,
    },
    "random_vit": {
        "type":         "random",
        "hf_id":        "facebook/dinov2-large",   # same architecture, random weights
        "num_layers":   24,
        "num_heads":    16,
        "hidden_dim":   1024,
        "img_size":     224,
    },
}

# ── Dataset ────────────────────────────────────────────────────────────────────
AFFORDANCE_CLASSES   = ["grasp", "cut", "scoop", "wrap-grasp", "poke", "support", "contain"]
NUM_AFFORDANCES      = len(AFFORDANCE_CLASSES)    # 7
AFFORDANCE_THRESHOLD = 0.001  # label is positive if >0.1% of pixels belong to that class (~307 px in 480×640)

IMG_RESIZE   = 224
BATCH_SIZE   = 16     # ViT-L is large; reduce to 8 if VRAM is tight
NUM_WORKERS  = 4
SEED         = 42
TRAIN_SPLIT  = 0.80
VAL_SPLIT    = 0.10
# test = 1 - 0.80 - 0.10 = 0.10

# ── Subsampling ────────────────────────────────────────────────────────────────
# Stratified by object category so every category keeps proportional representation.
# Set to None to use the full ~10 K image dataset (~4 h extraction per model on T4).
# 2 000 images: ~19 per category, stable mAP estimates, ~15 min extraction per model.
SUBSAMPLE_N  = 2_000

# ── Probing ────────────────────────────────────────────────────────────────────
PROBE_MAX_ITER = 1000
PROBE_C        = 1.0
PROBE_SOLVER   = "lbfgs"

# ── Head Ablation ──────────────────────────────────────────────────────────────
KEEP_TOP_K_HEADS = [1, 2, 3, 4, 6, 9, 12, 16]   # k values to sweep (up to num_heads=16)

# ── Test mode ─────────────────────────────────────────────────────────────────
# Pass --test to any script to run a tiny end-to-end smoke test.
TEST_N_SAMPLES   = 200                         # total samples to use (stratified subset)
TEST_BATCH_SIZE  = 4
TEST_PROBE_LAYERS = [0, 8, 16, 24]             # probe only these layer indices
TEST_KEEP_TOP_K  = [1, 3]                      # abbreviated k sweep

# ── Figures ────────────────────────────────────────────────────────────────────
FIGURE_DPI  = 300
FIGURE_EXT  = "pdf"
MODEL_COLORS = {
    "vjepa2":    "#E05C5C",
    "vjepa2_1":  "#5C8FE0",
    "dinov2":    "#5CBF7A",
    "random_vit":"#AAAAAA",
}
MODEL_LABELS = {
    "vjepa2":    "V-JEPA 2 (ViT-L)",
    "vjepa2_1":  "V-JEPA 2.1 (ViT-L)",
    "dinov2":    "DINOv2 (ViT-L)",
    "random_vit":"Random ViT-L",
}
