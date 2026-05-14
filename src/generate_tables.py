"""
Generates LaTeX table source code from results JSON files.
Writes .tex files to tables/ so the paper can \input{} them directly.

Table 1 — table_main_results.tex:
  Per-class AP and mAP at each model's peak layer.

Table 2 — table_ablation_efficiency.tex:
  mAP vs. k heads kept at peak layer, with approximate attention speedup.
"""
import json
import numpy as np
from pathlib import Path

from config import (RESULT_DIR, TABLE_DIR, MODELS, MODEL_LABELS, AFFORDANCE_CLASSES)

TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt(val, decimals=1) -> str:
    """Format a probability as a percentage string. None / NaN → '--'."""
    if val is None:
        return "--"
    if isinstance(val, float) and np.isnan(val):
        return "--"
    return f"{val * 100:.{decimals}f}"


def generate_table1():
    rows = []
    for mk in MODELS:
        path = RESULT_DIR / "probing" / f"{mk}_probe_results.json"
        if not path.exists():
            print(f"  [WARN] Skipping {mk} — no probe results")
            continue
        data       = _load(path)
        peak       = data["peak_layer"]
        peak_data  = data[f"layer_{peak}"]
        rows.append((mk, peak, peak_data["mAP"], peak_data["per_class_AP"]))

    if not rows:
        print("  [WARN] No probe results — table1 not generated.")
        return

    # Find column bests
    best = {cls: 0.0 for cls in AFFORDANCE_CLASSES}
    best["mAP"] = 0.0
    for _, _, mAP, pc in rows:
        for cls in AFFORDANCE_CLASSES:
            v = pc.get(cls) or 0
            if v > best[cls]:
                best[cls] = v
        if mAP > best["mAP"]:
            best["mAP"] = mAP

    class_cols = " & ".join(f"\\textit{{{c}}}" for c in AFFORDANCE_CLASSES)
    ncols      = 2 + len(AFFORDANCE_CLASSES) + 1   # model + peak + classes + mAP
    col_spec   = "l c " + "c " * len(AFFORDANCE_CLASSES) + "c"

    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Layer-wise linear probe at each model's peak layer. "
        "Values are Average Precision (\\%). Bold = best per column.}\n"
        "\\label{tab:main_results}\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"Model & Peak Layer & {class_cols} & mAP \\\\\n"
        "\\midrule\n"
    )

    body = ""
    for mk, peak, mAP, pc in rows:
        label = MODEL_LABELS.get(mk, mk)
        cells = []
        for cls in AFFORDANCE_CLASSES:
            v = pc.get(cls)
            s = fmt(v)
            if v is not None and abs(v - best[cls]) < 1e-4:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        mAP_s = fmt(mAP)
        if abs(mAP - best["mAP"]) < 1e-4:
            mAP_s = f"\\textbf{{{mAP_s}}}"
        body += f"{label} & {peak} & {' & '.join(cells)} & {mAP_s} \\\\\n"

    footer = "\\bottomrule\n\\end{tabular}}\n\\end{table}\n"

    path = TABLE_DIR / "table_main_results.tex"
    path.write_text(header + body + footer)
    print(f"  Saved: {path.name}")


def generate_table2():
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Head pruning efficiency at the peak layer. "
        "mAP (\\%) when keeping only top-$k$ affordance heads. "
        "Approx.\\ speedup $\\approx H/k$ where $H=16$ heads.}\n"
        "\\label{tab:ablation_efficiency}\n"
        "\\begin{tabular}{l c c c c}\n"
        "\\toprule\n"
        "Model & $k$ heads & mAP & Drop & Approx.\\ Speedup \\\\\n"
        "\\midrule\n"
    )

    body = ""
    for mk in ["vjepa2", "vjepa2_1", "dinov2"]:
        path = RESULT_DIR / "ablation" / f"{mk}_ablation_results.json"
        if not path.exists():
            print(f"  [WARN] Skipping {mk} — no ablation results")
            continue
        data     = _load(path)
        label    = MODEL_LABELS.get(mk, mk)
        num_heads = MODELS[mk]["num_heads"]
        for key, val in sorted(data.get("keep_top_k", {}).items(),
                               key=lambda x: int(x[0][1:])):
            k       = int(key[1:])
            speedup = f"{num_heads / k:.1f}$\\times$"
            body   += (
                f"{label} & {k} & {fmt(val['mAP'])} & "
                f"{fmt(val['drop'], 2)} & {speedup} \\\\\n"
            )
        body += "\\midrule\n"

    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    path = TABLE_DIR / "table_ablation_efficiency.tex"
    path.write_text(header + body + footer)
    print(f"  Saved: {path.name}")


def main():
    print("=== Table 1: Main probe results ===")
    generate_table1()
    print("\n=== Table 2: Ablation efficiency ===")
    generate_table2()
    print(f"\nAll tables → {TABLE_DIR}")


if __name__ == "__main__":
    main()
