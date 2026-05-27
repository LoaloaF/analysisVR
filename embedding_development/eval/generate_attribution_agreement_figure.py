#!/usr/bin/env python3
"""
generate_attribution_agreement_figure.py

Cross-model attribution agreement:
  Left panel:  MLP GPV vs TempConv-Cont GPV (per valid pair, aggregated by feature group)
  Right panel: MLP IG  vs TempConv-Cont IG  (same pairs)

One point per (feature group × valid pair). Diagonal = y=x.
Spearman ρ shown as plain text in upper-left.

Output: attribution_agreement_scatter.png  (FIG.FULL = 9.5 × 4.2")
"""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.figure_style import (
    FIG, DPI, FONT, FEATURE_NAMES_SHORT, AXIS_LABELS,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

# ─── PATHS ────────────────────────────────────────────────────────────────────
base     = os.path.dirname(os.path.abspath(__file__))
root     = os.path.join(base, "..")
mlp_dir  = os.path.join(root, "outputs", "mlps", "ensembles_multiseed")
cont_dir = os.path.join(root, "outputs", "cebra_eval",      "ensembles")

OUT_DIRS = [mlp_dir, '/mnt/c/Users/amits/Desktop']

# ─── LOAD ─────────────────────────────────────────────────────────────────────
gpv_mlp  = np.load(os.path.join(mlp_dir,  "importance_global_pv_semantic.npy"))  # (29,23,11)
gpv_cont = np.load(os.path.join(cont_dir, "importance_global_pv_semantic.npy"))
ig_mlp   = np.load(os.path.join(mlp_dir,  "importance_ig_semantic.npy"))
ig_cont  = np.load(os.path.join(cont_dir, "importance_ig_semantic.npy"))

with open(os.path.join(mlp_dir, "semantic_groups.pkl"), "rb") as f:
    semantic_groups = pickle.load(f)
group_names = [g[0] for g in semantic_groups]   # raw column-name keys
n_groups    = len(group_names)

# ─── SHARED VALID PAIRS (must have attribution in BOTH models) ────────────────
valid_gpv = ~np.all(np.isnan(gpv_mlp), axis=-1) & ~np.all(np.isnan(gpv_cont), axis=-1)
valid_ig  = ~np.all(np.isnan(ig_mlp),  axis=-1) & ~np.all(np.isnan(ig_cont),  axis=-1)

n_gpv = valid_gpv.sum()
n_ig  = valid_ig.sum()
print(f"Pairs with GPV in both models: {n_gpv}")
print(f"Pairs with IG  in both models: {n_ig}")

# Flatten over (session, ensemble) for valid pairs
gpv_mlp_flat  = gpv_mlp[ valid_gpv, :]   # (n_gpv, 11)
gpv_cont_flat = gpv_cont[valid_gpv, :]
ig_mlp_flat   = ig_mlp[  valid_ig,  :]
ig_cont_flat  = ig_cont[ valid_ig,  :]

# ─── AGGREGATE TO GROUP MEANS ─────────────────────────────────────────────────
# One point per feature group: mean across all valid pairs
gpv_mlp_mean  = np.nanmean(gpv_mlp_flat,  axis=0)   # (11,)
gpv_cont_mean = np.nanmean(gpv_cont_flat, axis=0)
ig_mlp_mean   = np.nanmean(ig_mlp_flat,   axis=0)
ig_cont_mean  = np.nanmean(ig_cont_flat,  axis=0)

rho_gpv, _  = spearmanr(gpv_mlp_mean, gpv_cont_mean)
rho_ig,  _  = spearmanr(ig_mlp_mean,  ig_cont_mean)
print(f"Spearman ρ (GPV): {rho_gpv:.3f}")
print(f"Spearman ρ (IG):  {rho_ig:.3f}")

# ─── COLOURS PER FEATURE GROUP ───────────────────────────────────────────────
FEAT_COLORS = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78',
    '#2ca02c', '#98df8a', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
]

# ─── FIGURE ───────────────────────────────────────────────────────────────────
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig, [ax_l, ax_r])

for (ax, x_vals, y_vals, x_lab, y_lab, rho, panel_label) in [
    (ax_l,
     gpv_mlp_mean, gpv_cont_mean,
     f"MLP {AXIS_LABELS['r2_drop']}",
     f"TempConv-Cont {AXIS_LABELS['r2_drop']}",
     rho_gpv, 'A'),
    (ax_r,
     ig_mlp_mean, ig_cont_mean,
     f"MLP {AXIS_LABELS['ig']}",
     f"TempConv-Cont {AXIS_LABELS['ig']}",
     rho_ig, 'B'),
]:
    lim = max(np.nanmax(x_vals), np.nanmax(y_vals)) * 1.18
    lim = max(lim, 1e-4)
    ax.plot([0, lim], [0, lim], 'k--', lw=0.9, zorder=1)

    for i, (gn, gx, gy) in enumerate(zip(group_names, x_vals, y_vals)):
        if not (np.isfinite(gx) and np.isfinite(gy)):
            continue
        color = FEAT_COLORS[i % len(FEAT_COLORS)]
        ax.scatter(gx, gy, s=90, color=color, zorder=3, edgecolors='none')
        label = FEATURE_NAMES_SHORT.get(gn, gn)
        ax.annotate(label, (gx, gy), fontsize=7,
                    xytext=(4, 2), textcoords='offset points', color=color)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(x_lab, fontsize=FONT.LABEL - 1)
    ax.set_ylabel(y_lab, fontsize=FONT.LABEL - 1)

    ax.text(0.04, 0.97, f"ρ = {rho:.2f}",
            transform=ax.transAxes, ha='left', va='top',
            fontsize=FONT.ANNOTATION, color='dimgray')

    add_panel_label(ax, panel_label)

add_footnote(fig,
    f"One point per semantic feature group (mean across valid pairs); "
    f"GPV: {n_gpv} shared pairs; IG: {n_ig} shared pairs; diagonal = y=x")

savefig_manifest(fig, "attribution_agreement_scatter.png", OUT_DIRS)
print("Generated attribution_agreement_scatter.png")
