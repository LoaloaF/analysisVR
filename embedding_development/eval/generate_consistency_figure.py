#!/usr/bin/env python3
"""
generate_consistency_figure.py

Generate r2_consistency_bar.png from pre-computed consistency_matrix_mlp.npy.
Shows mean cross-seed Pearson r per ensemble, sorted by R².
"""
import os, sys
import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.figure_style import (
    FIG, DPI, FONT, AXIS_LABELS,
    apply_style, add_footnote, savefig_manifest,
)

base   = os.path.dirname(os.path.abspath(__file__))
root   = os.path.join(base, "..")
mdir   = os.path.join(root, "outputs", "mlps", "ensembles_multiseed")

OUT_DIRS = [mdir, '/mnt/c/Users/amits/Desktop']

# ─── LOAD ─────────────────────────────────────────────────────────────────────
consistency = np.load(os.path.join(mdir, "consistency_matrix_mlp.npy"))  # (29, 23)
all_r2      = np.load(os.path.join(mdir, "all_r2.npy"))                  # (5, 29, 23)

n_sessions, n_ensembles = consistency.shape

# Ensemble-level mean Pearson r (over sessions)
ens_pearson_r = np.nanmean(consistency, axis=0)   # (23,) — NaN where no overlap

# Ensemble-level mean R² (to sort by)
mean_r2 = np.nanmean(all_r2, axis=0)              # (29, 23)
ens_mean_r2 = np.nanmean(mean_r2, axis=0)         # (23,)

# Sort ensembles by mean R² (ascending → highest at right)
order = np.argsort(ens_mean_r2)

x   = np.arange(n_ensembles)
pal = sns.color_palette("mako_r", as_cmap=True)(np.linspace(0.15, 0.85, n_ensembles))

n_valid = int(np.sum(~np.isnan(ens_pearson_r)))
print(f"Ensembles with consistency data: {n_valid}/{n_ensembles}")
print(f"Mean Pearson r: {np.nanmean(ens_pearson_r):.3f}")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
apply_style(fig, ax)

r_vals = ens_pearson_r[order]
has_data = ~np.isnan(r_vals)

ax.bar(x[has_data],  r_vals[has_data],  width=0.7,
       color=[pal[i] for i in np.where(has_data)[0]], zorder=3, linewidth=0)
ax.bar(x[~has_data], np.zeros(has_data.size - has_data.sum()), width=0.7,
       color='#dddddd', zorder=3, linewidth=0, label='no overlap')

ax.axhline(0, color='#888888', lw=0.8, linestyle='--')
ax.set_ylim(0, 1.05)
ax.set_xticks(x[::3])
ax.set_xticklabels([f"E{order[i]+1:02d}" for i in range(0, n_ensembles, 3)],
                   rotation=45, ha='right', fontsize=FONT.TICK - 1)
ax.set_ylabel(AXIS_LABELS['pearson_r'], fontsize=FONT.LABEL)
ax.set_xlabel(AXIS_LABELS['ensemble'], fontsize=FONT.LABEL)

if has_data.sum() < n_ensembles:
    ax.legend(fontsize=FONT.LEGEND - 1, frameon=False)

add_footnote(fig,
    f"MLP; cross-seed Pearson r on held-out trials present in ≥2 seeds; "
    f"{n_valid}/{n_ensembles} ensembles have overlapping test trials")

savefig_manifest(fig, "r2_consistency_bar.png", OUT_DIRS)
print("Generated r2_consistency_bar.png")
