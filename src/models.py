"""
Model loading and layer-wise feature extraction for all four models.

V-JEPA 2 / V-JEPA 2.1 (ViT-L):
  - Loaded from official Meta checkpoints via the cloned facebookresearch/vjepa2 repo.
  - Checkpoint keys: 'target_encoder' (V-JEPA 2), 'ema_encoder' (V-JEPA 2.1).
  - Input: (B, C, T, H, W) video. Static images are replicated T=video_frames times.
  - Layer 0 = patch embedding output; layers 1-24 = after each of 24 transformer blocks.
  - Representation: mean over all spatio-temporal tokens (no CLS token in JEPA ViTs).

DINOv2 ViT-L / Random ViT-L:
  - Loaded via HuggingFace transformers.
  - Layer 0 = patch embedding; layers 1-24 = hidden states after each block.
  - Representation: CLS token (index 0).

Head ablation (used in ablate.py):
  - Hooks block.attn.proj (V-JEPA) or block.attention.output.dense (DINOv2) with a
    forward pre-hook that zeros the `h*head_dim:(h+1)*head_dim` columns for each
    ablated head h. This zeros head h's contribution before the output projection,
    which is the principled approach from Michel et al. (2019).
"""
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from config import MODELS, VJEPA2_REPO_PATH

logger = logging.getLogger(__name__)


# ── V-JEPA model loading ───────────────────────────────────────────────────────

def _add_vjepa2_to_path():
    """Temporarily add the cloned vjepa2 repo to sys.path so its src/ is importable."""
    repo = str(VJEPA2_REPO_PATH)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _load_vjepa_model(cfg: dict) -> nn.Module:
    """
    Build a V-JEPA ViT-L and load pretrained weights from a .pt checkpoint.
    Follows the same logic as evals/image_classification_frozen/modelcustom/vit_encoder.py.
    """
    if not VJEPA2_REPO_PATH.exists():
        raise RuntimeError(
            f"V-JEPA 2 repo not found at {VJEPA2_REPO_PATH}. "
            "Run scripts/setup_models.sh first."
        )
    checkpoint_path = cfg["checkpoint"]
    if not Path(checkpoint_path).exists():
        raise RuntimeError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run scripts/setup_models.sh first."
        )

    _add_vjepa2_to_path()
    import src.models.vision_transformer as vjepa_vit  # from vjepa2 repo

    model_fn   = vjepa_vit.__dict__[cfg["model_name"]]
    model_kwargs = dict(cfg["model_kwargs"])
    num_frames = cfg["video_frames"]
    model = model_fn(num_frames=num_frames, **model_kwargs)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    key  = cfg["checkpoint_key"]
    if key not in ckpt:
        available = list(ckpt.keys())
        raise KeyError(
            f"Checkpoint key '{key}' not found. Available keys: {available}"
        )
    state = ckpt[key]
    # Strip DDP / module wrappers
    state = {k.replace("module.", "").replace("backbone.", ""): v
             for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ── DINOv2 / Random ViT loading ───────────────────────────────────────────────

def _load_dinov2(hf_id: str, random_init: bool = False) -> nn.Module:
    from transformers import AutoModel
    model = AutoModel.from_pretrained(hf_id)
    if random_init:
        model.apply(_reset_params)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _reset_params(module: nn.Module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_dinov2_layer_features(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> Dict[int, np.ndarray]:
    """
    images: (N, 3, H, W)
    Returns: {layer_idx: (N, D)} where layer 0 = patch embeddings, 1-24 = block outputs.
    Uses the CLS token (position 0) as the image representation.
    """
    images = images.to(device)
    with torch.no_grad():
        out = model(pixel_values=images, output_hidden_states=True)
    # out.hidden_states: tuple of len=num_layers+1; [0]=patch embed, [i]=after block i-1
    result: Dict[int, np.ndarray] = {}
    for layer_idx, hs in enumerate(out.hidden_states):
        cls = hs[:, 0, :].cpu().numpy()   # CLS token: (N, D)
        result[layer_idx] = cls
    return result


def extract_vjepa_layer_features(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    video_frames: int,
) -> Dict[int, np.ndarray]:
    """
    images: (N, 3, H, W)
    Returns: {layer_idx: (N, D)} where layer 0 = patch embeddings, 1-24 = block outputs.
    Uses mean over all spatio-temporal tokens as the representation.

    Hooks are registered on model.blocks (ModuleList of 24 blocks) for layers 1-24,
    and on model.patch_embed for layer 0.
    """
    N = images.shape[0]
    # (N, 3, T, H, W) — replicate the static image across T frames
    video = images.unsqueeze(2).repeat(1, 1, video_frames, 1, 1).to(device)

    activations: Dict[int, np.ndarray] = {}
    hooks = []

    def _mean_pool_hook(layer_idx: int):
        def hook(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            activations[layer_idx] = hs.mean(dim=1).detach().cpu().numpy()  # (N, D)
        return hook

    # Layer 0: patch embedding output
    h0 = model.patch_embed.register_forward_hook(_mean_pool_hook(0))
    hooks.append(h0)

    # Layers 1..num_blocks: each transformer block
    for i, blk in enumerate(model.blocks):
        h = blk.register_forward_hook(_mean_pool_hook(i + 1))
        hooks.append(h)

    with torch.no_grad():
        model(video)   # forward; masks=None (no masking at inference)

    for h in hooks:
        h.remove()

    return activations


# ── Unified extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Unified wrapper.
    Usage:
        extractor = FeatureExtractor("vjepa2", device)
        layer_features = extractor.extract(image_batch)  # {layer_idx: (N, D)}
    """
    def __init__(self, model_key: str, device: torch.device):
        self.model_key  = model_key
        self.device     = device
        cfg             = MODELS[model_key]
        self.cfg        = cfg
        self.model_type = cfg["type"]

        if self.model_type == "dinov2":
            self.model = _load_dinov2(cfg["hf_id"]).to(device)
        elif self.model_type == "random":
            self.model = _load_dinov2(cfg["hf_id"], random_init=True).to(device)
        elif self.model_type == "vjepa":
            self.model = _load_vjepa_model(cfg).to(device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.eval()

    @property
    def num_layers(self) -> int:
        return self.cfg["num_layers"]

    def extract(self, images: torch.Tensor) -> Dict[int, np.ndarray]:
        """images: (N, 3, H, W). Returns {layer_idx: (N, D)}."""
        if self.model_type in ("dinov2", "random"):
            return extract_dinov2_layer_features(self.model, images, self.device)
        else:
            return extract_vjepa_layer_features(
                self.model, images, self.device,
                video_frames=self.cfg["video_frames"],
            )


# ── Head ablation utilities (used by ablate.py) ───────────────────────────────

def _find_attn_proj(block: nn.Module, model_type: str) -> Optional[nn.Module]:
    """
    Return the attention output-projection Linear layer inside a transformer block.
    For V-JEPA: block.attn.proj
    For DINOv2 (HuggingFace): block.attention.output.dense
    Returns None if the submodule cannot be located.
    """
    # V-JEPA path
    if hasattr(block, "attn") and hasattr(block.attn, "proj"):
        return block.attn.proj
    # HuggingFace DINOv2 path
    try:
        return block.attention.output.dense
    except AttributeError:
        pass
    return None


def make_ablation_hooks(
    model: nn.Module,
    model_type: str,
    peak_layer: int,
    heads_to_zero: List[int],
    head_dim: int,
    collection_buffer: List[np.ndarray],
):
    """
    Register two hooks at peak_layer:
      1. A forward pre-hook on the attention output projection that zeros specified heads
         BEFORE the projection mixes them (principled head ablation, Michel et al. 2019).
      2. A forward hook on the block itself that mean-pools the (ablated) output
         and appends it to collection_buffer.

    Returns a list of hook handles (call .remove() on each when done).
    """
    handles = []
    num_layers = len(model.blocks) if hasattr(model, "blocks") else len(model.encoder.layer)

    # Identify the block at peak_layer.
    # layer 0 is patch embed, so block index = peak_layer - 1.
    block_idx = peak_layer - 1
    if block_idx < 0:
        raise ValueError("Cannot ablate layer 0 (patch embedding has no attention heads).")

    if hasattr(model, "blocks"):
        block = model.blocks[block_idx]
    else:
        block = model.encoder.layer[block_idx]

    # ── 1. Attention projection pre-hook (zeros head columns before projection) ──
    attn_proj = _find_attn_proj(block, model_type)

    if attn_proj is not None and heads_to_zero:
        def proj_pre_hook(module, args):
            x = args[0].clone()   # (N, seq_len, D)
            for h in heads_to_zero:
                x[:, :, h * head_dim:(h + 1) * head_dim] = 0.0
            return (x,)
        handles.append(attn_proj.register_forward_pre_hook(proj_pre_hook))
    else:
        # Fallback: zero head columns in the block output (less principled but functional)
        logger.warning(
            f"Could not locate attn output projection for {model_type} block {block_idx}. "
            "Falling back to block-level output zeroing."
        )

    # ── 2. Block output collection hook ───────────────────────────────────────
    def collect_hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        # If no principled projection hook, apply fallback zeroing here
        if attn_proj is None and heads_to_zero:
            hs = hs.clone()
            for h in heads_to_zero:
                hs[:, :, h * head_dim:(h + 1) * head_dim] = 0.0
        # Match the representation used during feature extraction:
        # DINOv2 uses CLS token (index 0); V-JEPA uses mean over all tokens.
        if model_type in ("dinov2", "random"):
            rep = hs[:, 0, :].detach().cpu().numpy()       # CLS token: (N, D)
        else:
            rep = hs.mean(dim=1).detach().cpu().numpy()    # mean pool: (N, D)
        collection_buffer.append(rep)

    handles.append(block.register_forward_hook(collect_hook))
    return handles


def run_forward_pass(
    model: nn.Module,
    model_type: str,
    images: torch.Tensor,
    device: torch.device,
    video_frames: int = 8,
):
    """Run a forward pass appropriate for the model type."""
    images = images.to(device)
    with torch.no_grad():
        if model_type in ("dinov2", "random"):
            model(pixel_values=images, output_hidden_states=False)
        else:
            video = images.unsqueeze(2).repeat(1, 1, video_frames, 1, 1)
            model(video)
