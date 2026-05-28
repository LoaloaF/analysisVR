#!/usr/bin/env python3
"""
generate_case_studies.py

Case study figure comparing attribution profiles for two contrasting ensembles:
  E07 (index 6)  — speed-dominated, moderate head angle
  E23 (index 22) — head angle dominated, some speed contribution

Two-panel figure:
  Left: E07 — per-feature GPV bar chart (mean ± SD across valid sessions)
  Right: E23 — same

Output: case_studies_e07_e23.png  (FIG.FULL = 9.5 × 4.2")
"""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.figure_style import (
    FIG, DPI, FONT, FEATURE_NAMES_SHORT, AXIS_LABELS,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ENSEMBLES = {'E07': 6, 'E23': 22}

base   = os.path.dirname(os.path.abspath(__file__))
root   = os.path.join(base, "..")
mdir   = os.path.join(root, "outputs", "mlps", "ensembles_multiseed")
OUT_DIRS = [mdir, '/mnt/c/Users/amits/Desktop']

# ─── LOAD ─────────────────────────────────────────────────────────────────────
gpv = np.load(os.path.join(mdir, "importance_global_pv_semantic.npy"))  # (29, 23, 11)
cpv = np.load(os.path.join(mdir, "importance_cond_pv_semantic.npy"))

with open(os.path.join(mdir, "semantic_groups.pkl"), "rb") as f:
    sg = pickle.load(f)
group_names = [g[0] for g in sg]   # raw keys
n_groups    = len(group_names)

ytick_labels = [FEATURE_NAMES_SHORT.get(g, g) for g in group_names]

# ─── FIGURE ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=FIG.FULL, sharey=False)
apply_style(fig, list(axes))

panel_labels = ['A', 'B']
x = np.arange(n_groups)
width = 0.35

for ax, (en_name, en_idx), pl in zip(axes, ENSEMBLES.items(), panel_labels):
    g_gpv = gpv[:, en_idx, :]   # (29, 11) — sessions × groups
    g_cpv = cpv[:, en_idx, :]

    valid = ~np.all(np.isnan(g_gpv), axis=-1)   # sessions where this ensemble is valid
    n_sess = valid.sum()

    gpv_mean = np.nanmean(g_gpv[valid], axis=0)   # (11,)
    gpv_std  = np.nanstd( g_gpv[valid], axis=0)
    cpv_mean = np.nanmean(g_cpv[valid], axis=0)
    cpv_std  = np.nanstd( g_cpv[valid], axis=0)

    ek = dict(ecolor='k', lw=0.7, capsize=3)
    ax.bar(x - width/2, gpv_mean, width, yerr=gpv_std,
           color='#5b9bd5', alpha=0.85, label='Global PV', error_kw=ek)
    ax.bar(x + width/2, cpv_mean, width, yerr=cpv_std,
           color='#ed7d31', alpha=0.85, label='Cond. PV', error_kw=ek)

    ax.set_xticks(x)
    ax.set_xticklabels(ytick_labels, rotation=90, ha='center',
                       fontsize=max(6, FONT.TICK - 3))
    # Show y-axis label and tick numbers only on panel A to avoid the panel
    # letter 'B' and the rotated y-label competing in the same left-margin space.
    if pl == 'A':
        ax.set_ylabel(AXIS_LABELS['r2_drop'], fontsize=FONT.LABEL - 1)
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])
    ax.axhline(0, color='#888', lw=0.7, linestyle='--')

    ax.text(0.97, 0.97,
            f"{en_name}  (n={n_sess} sessions)",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FONT.ANNOTATION, color='dimgray')

    if pl == 'A':
        ax.legend(fontsize=FONT.LEGEND - 1, frameon=False, loc='upper left')

    add_panel_label(ax, pl)

add_footnote(fig,
    "Mean ± SD across valid sessions (R²≥0.01, 5 seeds); "
    "Global PV = collinearity-blind; Cond. PV = k-NN conditioned on other features")

savefig_manifest(fig, "case_studies_e07_e23.png", OUT_DIRS)
print("Generated case_studies_e07_e23.png")
