"""
Generates all paper figures from saved results JSON files.
Also saves the plot-ready data as JSON to figures/data/ so figures can be
restyled later by rerunning this script without re-running any experiments.

Figure 1 — fig_layer_probe:
  Layer-wise mAP curves for all 4 models.

Figure 2 — fig_head_heatmap:
  Per-head importance heatmap at peak layer for V-JEPA 2, V-JEPA 2.1, DINOv2.

Figure 3 — fig_topk_efficiency:
  Accuracy-efficiency tradeoff when keeping only top-k heads.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from config import (RESULT_DIR, FIGURE_DIR, FIGURE_DPI, FIGURE_EXT,
                    MODEL_COLORS, MODEL_LABELS, MODELS, AFFORDANCE_CLASSES)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = FIGURE_DIR / "data"   # underlying plot data
DATA_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi":      FIGURE_DPI,
})


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_probe_results(model_key: str) -> dict:
    path = RESULT_DIR / "probing" / f"{model_key}_probe_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run probe.py for {model_key}.")
    with open(path) as f:
        return json.load(f)


def load_ablation_results(model_key: str) -> dict:
    path = RESULT_DIR / "ablation" / f"{model_key}_ablation_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run ablate.py for {model_key}.")
    with open(path) as f:
        return json.load(f)


def save_figure(fig, name: str):
    pdf = FIGURE_DIR / f"{name}.pdf"
    png = FIGURE_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"  Saved: {pdf.name}, {png.name}")


def save_plot_data(name: str, data: dict):
    """Persist the numeric data behind a figure so it can be re-plotted anytime."""
    path = DATA_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Plot data: {path.name}")


# ── Figure 1: Layer-wise probing curves ───────────────────────────────────────

def figure1_layer_probe():
    plot_data = {}

    # Three-panel broken y-axis:
    #   top = 96–100.5% (plateau, zoomed)
    #   mid = 80–96%    (early rise)
    #   bot = 0–80%     (random baseline, compressed)
    fig = plt.figure(figsize=(6.5, 6.0))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1], hspace=0.08)
    ax_top = fig.add_subplot(gs[0])   # 0.96–1.005
    ax_mid = fig.add_subplot(gs[1])   # 0.80–0.96
    ax_bot = fig.add_subplot(gs[2])   # 0.00–0.80

    pct_fmt = mticker.FuncFormatter(lambda x, _: f"{x*100:.0f}")

    for model_key in MODELS:
        try:
            data = load_probe_results(model_key)
        except FileNotFoundError as e:
            print(f"  [WARN] Skipping {model_key}: {e}")
            continue

        layers, maps = [], []
        for key, val in data.items():
            if key.startswith("layer_") and isinstance(val, dict):
                layer_idx = int(key.split("_")[1])
                layers.append(layer_idx)
                maps.append(val["mAP"])

        order  = np.argsort(layers)
        layers = np.array(layers)[order].tolist()
        maps   = np.array(maps)[order].tolist()

        peak = data.get("peak_layer")
        plot_data[model_key] = {
            "layers": layers, "maps": maps,
            "peak_layer": peak, "peak_mAP": data.get("peak_mAP"),
        }

        color = MODEL_COLORS[model_key]
        plot_kw = dict(marker="o", markersize=4, linewidth=1.8, color=color)
        ax_top.plot(layers, maps, label=MODEL_LABELS[model_key], **plot_kw)
        ax_mid.plot(layers, maps, label="_nolegend_", **plot_kw)
        ax_bot.plot(layers, maps, label="_nolegend_", **plot_kw)

        if peak is not None and model_key != "random_vit":
            peak_map = data[f"layer_{peak}"]["mAP"]
            for ax in (ax_top, ax_mid, ax_bot):
                ax.axvline(x=peak, color=color, linestyle="--",
                           linewidth=0.8, alpha=0.6)
            y_offset = 0.004 if model_key == "vjepa2" else 0.001
            ax_top.annotate(
                f"L{peak}",
                xy=(peak, peak_map),
                xytext=(peak + 0.3, peak_map + y_offset),
                color=color, fontsize=7,
            )

    # Y-axis limits
    ax_top.set_ylim(0.960, 1.005)
    ax_mid.set_ylim(0.800, 0.960)
    ax_bot.set_ylim(0.000, 0.800)

    # Y-axis tick formatting
    for ax in (ax_top, ax_mid, ax_bot):
        ax.yaxis.set_major_formatter(pct_fmt)
    ax_top.yaxis.set_major_locator(mticker.MultipleLocator(0.01))   # 1% ticks in plateau
    ax_mid.yaxis.set_major_locator(mticker.MultipleLocator(0.04))   # 4% ticks in mid
    ax_bot.set_yticks([0.0, 0.2, 0.4, 0.6])

    # Broken-axis spine cosmetics
    ax_top.spines["bottom"].set_visible(False)
    ax_mid.spines["top"].set_visible(False)
    ax_mid.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_mid.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_bot.xaxis.tick_bottom()

    # Diagonal break marks at top/mid boundary
    d = 0.02
    bkw = dict(color="k", clip_on=False, linewidth=0.8)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **bkw)
    ax_top.plot((1-d, 1+d), (-d, +d), transform=ax_top.transAxes, **bkw)
    ax_mid.plot((-d, +d), (1-d, 1+d), transform=ax_mid.transAxes, **bkw)
    ax_mid.plot((1-d, 1+d), (1-d, 1+d), transform=ax_mid.transAxes, **bkw)
    # Diagonal break marks at mid/bot boundary
    ax_mid.plot((-d, +d), (-d, +d), transform=ax_mid.transAxes, **bkw)
    ax_mid.plot((1-d, 1+d), (-d, +d), transform=ax_mid.transAxes, **bkw)
    ax_bot.plot((-d, +d), (1-d, 1+d), transform=ax_bot.transAxes, **bkw)
    ax_bot.plot((1-d, 1+d), (1-d, 1+d), transform=ax_bot.transAxes, **bkw)

    # Labels
    ax_bot.set_xlabel("Transformer Layer")
    ax_top.set_title("Layer-wise Affordance Probe Accuracy")
    fig.text(0.02, 0.5, "mAP (%)", va="center", rotation="vertical", fontsize=11)
    handles, labels = ax_top.get_legend_handles_labels()
    ax_top.legend(handles, labels, loc="upper left")

    for ax in (ax_top, ax_mid, ax_bot):
        ax.set_xlim(-0.5, 25)
        ax.grid(True, alpha=0.3)

    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.08)

    save_plot_data("fig_layer_probe", plot_data)
    save_figure(fig, "fig_layer_probe")
    plt.close()


# ── Figure 2: Head importance heatmap ─────────────────────────────────────────

def figure2_head_heatmap():
    model_keys = ["vjepa2", "vjepa2_1", "dinov2"]
    available, data_list = [], []
    for mk in model_keys:
        try:
            data_list.append(load_ablation_results(mk))
            available.append(mk)
        except FileNotFoundError as e:
            print(f"  [WARN] Skipping {mk}: {e}")

    if not available:
        print("  [WARN] No ablation results — skipping Figure 2.")
        return

    plot_data = {}
    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(6.5, 1.3 * n + 0.8))
    if n == 1:
        axes = [axes]

    for ax, mk, data in zip(axes, available, data_list):
        num_heads   = MODELS[mk]["num_heads"]
        importances = np.array([
            data["per_head"].get(f"head_{h}", {}).get("importance", 0.0)
            for h in range(num_heads)
        ])
        imp_norm = importances / importances.max() if importances.max() > 0 else importances

        plot_data[mk] = {
            "num_heads":       num_heads,
            "importances":     importances.tolist(),
            "importances_norm":imp_norm.tolist(),
            "head_ranking":    data.get("head_ranking", []),
            "peak_layer":      data.get("peak_layer"),
        }

        im = ax.imshow(imp_norm[np.newaxis, :], aspect="auto",
                       cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_yticks([0])
        ax.set_yticklabels([MODEL_LABELS[mk]], fontsize=9)
        ax.set_xticks(range(num_heads))
        ax.set_xticklabels([str(h) for h in range(num_heads)], fontsize=8)

        # Highlight top-3 heads
        top3 = data.get("head_ranking", [])[:3]
        for h in top3:
            ax.add_patch(plt.Rectangle(
                (h - 0.5, -0.5), 1, 1,
                fill=False, edgecolor="blue", linewidth=1.5,
            ))

        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    axes[-1].set_xlabel("Head Index")
    fig.suptitle(
        "Per-Head Affordance Importance at Peak Layer\n(Blue box = top-3 heads)",
        fontsize=11,
    )
    plt.tight_layout()

    save_plot_data("fig_head_heatmap", plot_data)
    save_figure(fig, "fig_head_heatmap")
    plt.close()


# ── Figure 3: Top-k efficiency tradeoff ───────────────────────────────────────

def figure3_topk_efficiency():
    plot_data = {}

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for mk in ["vjepa2", "vjepa2_1", "dinov2"]:
        try:
            data = load_ablation_results(mk)
        except FileNotFoundError as e:
            print(f"  [WARN] Skipping {mk}: {e}")
            continue

        baseline = data["baseline_mAP"]
        k_vals, map_vals = [], []
        for key, val in data.get("keep_top_k", {}).items():
            k_vals.append(int(key[1:]))
            map_vals.append(val["mAP"])

        if not k_vals:
            continue

        order    = np.argsort(k_vals)
        k_vals   = np.array(k_vals)[order].tolist()
        map_vals = np.array(map_vals)[order].tolist()

        plot_data[mk] = {
            "k_vals":   k_vals,
            "map_vals": map_vals,
            "baseline": baseline,
        }

        ax.plot(k_vals, map_vals, marker="o", markersize=5, linewidth=1.8,
                color=MODEL_COLORS[mk], label=MODEL_LABELS[mk])
        ax.axhline(y=baseline, color=MODEL_COLORS[mk],
                   linestyle=":", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Number of Heads Kept (k)")
    ax.set_ylabel("mAP (Affordance Classification)")
    ax.set_title("Efficiency–Accuracy Tradeoff: Top-k Head Pruning")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    save_plot_data("fig_topk_efficiency", plot_data)
    save_figure(fig, "fig_topk_efficiency")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=== Figure 1: Layer-wise probe curves ===")
    figure1_layer_probe()
    print("\n=== Figure 2: Head importance heatmap ===")
    figure2_head_heatmap()
    print("\n=== Figure 3: Top-k efficiency tradeoff ===")
    figure3_topk_efficiency()
    print(f"\nAll figures → {FIGURE_DIR}")
    print(f"Plot data   → {DATA_DIR}")


if __name__ == "__main__":
    main()
