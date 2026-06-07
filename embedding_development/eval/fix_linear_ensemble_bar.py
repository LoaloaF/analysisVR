#!/usr/bin/env python3
"""
Regenerate the linear ensemble R² bar with the same format as the MLP bar
(mako_r colormap, top-N bold labels, shared y-axis with MLP).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use(['science', 'no-latex'])

root      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
lin_path  = os.path.join(root, "outputs", "linear", "ensembles_multiseed", "all_r2.npy")
mlp_path  = os.path.join(root, "outputs", "mlps",   "ensembles_multiseed", "all_r2.npy")
out_dir   = os.path.join(root, "outputs", "linear", "ensembles_multiseed")

lin_r2 = np.load(lin_path)  # (seeds, sessions, ensembles)
mlp_r2 = np.load(mlp_path)

num_sessions = lin_r2.shape[1]
num_ensembles = lin_r2.shape[2]
n_seeds = lin_r2.shape[0]

TOP_LABEL = 5

def _bar(all_r2, out_path, model_name, ymax):
    med   = np.nanmedian(all_r2, axis=0)   # (sessions, ensembles)
    means = np.nanmean(med, axis=0)         # (ensembles,)
    order = np.argsort(means)              # ascending
    means_sorted  = np.clip(means[order], 0, None)
    labels_sorted = [f"E{i+1:02d}" for i in order]

    pal = sns.color_palette("mako_r", as_cmap=True)(np.linspace(0.15, 0.85, num_ensembles))

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(num_ensembles), means_sorted, width=0.7, color=pal, zorder=3, linewidth=0)
    ax.axhline(0, color='firebrick', linestyle='--', linewidth=1)

    for pos in range(num_ensembles - TOP_LABEL, num_ensembles):
        ax.text(pos, means_sorted[pos] + ymax * 0.02,
                labels_sorted[pos],
                ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=90)

    ax.set_xticks(range(0, num_ensembles, 3))
    ax.set_xticklabels(labels_sorted[::3], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Mean R²")
    ax.set_title(f"Per-Ensemble Mean R² Across {n_seeds} Seeds × {num_sessions} Sessions")
    ax.set_ylim(0, ymax)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

# Shared y-axis: max across both models
_bar(lin_r2, os.path.join(out_dir, "r2_bar.png"), "Linear", 0.2)
print("Done.")
