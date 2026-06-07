#!/usr/bin/env python3
"""
regen_consistency_comparison.py

Regenerate outputs/cebra_comparison/embedding_consistency_comparison.png
with Arial font, matching the original layout:
  - Top: 3 consistency heatmaps (MLP | CEBRA-Contrast | CEBRA-Pred)
    with diagonal hatching on NaN cells
  - Bottom: violin + jitter for all 3 models
  - Metric: 1-CV (already in matrices as NaN = invalid)

Data: uses per-model NaN structure — no external R² filter applied,
so counts match: MLP 74, CEBRA-Contrast 73, CEBRA-Pred 94.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as _fm
import seaborn as sns

for _fp in ['/mnt/c/Windows/Fonts/arial.ttf', '/mnt/c/Windows/Fonts/arialbd.ttf']:
    if os.path.exists(_fp): _fm.fontManager.addfont(_fp)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.spines.top': False,
    'axes.spines.right': False,
})

base    = os.path.dirname(os.path.abspath(__file__))
root    = os.path.join(base, "..")
out_dir = os.path.join(root, "outputs", "cebra_comparison")

# ── Load data ──────────────────────────────────────────────────────────────────
mlp_mat = np.load(os.path.join(root, "outputs", "mlps", "ensembles_multiseed",
                               "consistency_matrix_mlp.npy"))
cc_mat  = np.load(os.path.join(out_dir, "consistency_matrix_cebra_contrast.npy"))
cp_mat  = np.load(os.path.join(out_dir, "consistency_matrix_cebra_pred.npy"))

models = [
    ("MLP",            mlp_mat, '#4472C4'),
    ("CEBRA-Contrast", cc_mat,  '#ED7D31'),
    ("CEBRA-Pred",     cp_mat,  '#70AD47'),
]

n_sess = mlp_mat.shape[0]
sess_labels = [f"S{i+1:02d}" for i in range(n_sess)]

# Top 14 ensembles by MLP mean consistency
ens_mean = np.nanmean(mlp_mat, axis=0)
TOP_N    = 14
top_ens  = np.argsort(ens_mean)[::-1][:TOP_N]
ens_labels = [f"E{i+1:02d}" for i in top_ens]

NO_DATA = '#bbbbbb'
cmap    = plt.cm.Blues.copy()
cmap.set_bad(color=NO_DATA)

fig = plt.figure(figsize=(14, 8))
gs  = fig.add_gridspec(2, 1, height_ratios=[1.8, 1.0],
                       left=0.07, right=0.97, top=0.95, bottom=0.08,
                       hspace=0.45)
gs_top = gs[0].subgridspec(1, 4, wspace=0.08, width_ratios=[1, 1, 1, 0.06])

# ── Top row: heatmaps ──────────────────────────────────────────────────────────
for col, (label, mat, color) in enumerate(models):
    ax = fig.add_subplot(gs_top[col])
    sub = mat[:, top_ens].T        # (TOP_N, n_sess) — use native NaN structure
    masked = np.ma.array(sub, mask=np.isnan(sub))

    # seaborn heatmap for the blue-scale cells
    sns.heatmap(sub, ax=ax, cmap=cmap, vmin=0, vmax=1,
                xticklabels=sess_labels, yticklabels=ens_labels if col == 0 else [],
                cbar=False, linewidths=0, linecolor='none')

    # Hatching for NaN cells (matches original style)
    for (i, j) in zip(*np.where(np.isnan(sub))):
        ax.add_patch(plt.Rectangle([j, i], 1, 1, fill=True, facecolor=NO_DATA,
                                   hatch='////', edgecolor='#888888', lw=0.5, zorder=2))

    ax.set_title(label, fontsize=12, fontweight='bold', pad=6)
    ax.set_xlabel('Session', fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    if col == 0:
        ax.set_ylabel(f'Top {TOP_N} ensembles\n(ranked by MLP consistency)', fontsize=9)
        ax.set_yticklabels(ens_labels, fontsize=8)
    else:
        ax.set_ylabel('')

# Shared colorbar
cax = fig.add_subplot(gs_top[3])
norm = mcolors.Normalize(vmin=0, vmax=1)
sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, cax=cax)
cb.set_label('1-CV', fontsize=10)
cb.ax.tick_params(labelsize=9)

# ── Bottom row: violin + jitter ────────────────────────────────────────────────
ax_v = fig.add_subplot(gs[1])
rng  = np.random.default_rng(0)

flat_data, flat_labels, flat_colors = [], [], []
for label, mat, color in models:
    vals = mat.ravel()
    vals = vals[np.isfinite(vals)]    # no extra R² filter — matrix encodes validity
    flat_data.append(vals)
    flat_labels.append(label)
    flat_colors.append(color)

parts = ax_v.violinplot(flat_data, positions=[0, 1, 2],
                        showmedians=True, showextrema=True, widths=0.6)
for pc, c in zip(parts['bodies'], flat_colors):
    pc.set_facecolor(c); pc.set_alpha(0.45)
for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
    parts[part].set_color('k'); parts[part].set_linewidth(0.9)

for i, (vals, label, color) in enumerate(zip(flat_data, flat_labels, flat_colors)):
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    ax_v.scatter(i + jitter, vals, s=14, color=color, alpha=0.65, linewidths=0, zorder=3)
    med = float(np.median(vals))
    ax_v.text(i, med + 0.012, f'{med:.2f}', ha='center', va='bottom',
              fontsize=10, fontweight='bold')
    ax_v.text(i, -0.06, f'n={len(vals)}', ha='center', va='top',
              fontsize=9, color='#555555', transform=ax_v.get_xaxis_transform())

ax_v.set_xticks([0, 1, 2])
ax_v.set_xticklabels(flat_labels, fontsize=11)
ax_v.set_ylabel('Consistency (1-CV)', fontsize=10)
ax_v.set_ylim(0.55, 1.12)
ax_v.axhline(0.9, color='k', linestyle=':', lw=0.9, label='0.9 threshold')
ax_v.legend(fontsize=9, frameon=False, loc='lower right')
ax_v.set_title('Distribution of consistency scores (R²≥0.1)', fontsize=11)
ax_v.spines[['top', 'right']].set_visible(False)

# ── Save ───────────────────────────────────────────────────────────────────────
out_paths = [
    os.path.join(out_dir, "embedding_consistency_comparison.png"),
    '/mnt/c/Users/amits/Desktop/embedding_consistency_comparison.png',
]
for p in out_paths:
    fig.savefig(p, dpi=150, bbox_inches='tight')
    print(f"Saved {p}")
plt.close()
print("Done.")
