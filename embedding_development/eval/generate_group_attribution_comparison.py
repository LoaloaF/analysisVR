#!/usr/bin/env python3
"""
generate_group_attribution_comparison.py

Group-level GPV attribution bar chart: MLP vs TempConv-Cont, side by side per
semantic feature group.  Answers: do the two model architectures agree on which
behavioral features drive neural ensemble activity?

Output: group_attribution_comparison.png  (FIG.FULL = 9.5 × 4.2")
"""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import (
    FIG, DPI, FONT, FEATURE_NAMES_SHORT, AXIS_LABELS, MODEL_COLORS,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

# ─── PATHS ────────────────────────────────────────────────────────────────────
base     = os.path.dirname(os.path.abspath(__file__))
root     = os.path.join(base, '..')
mlp_dir  = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
ceb_dir  = os.path.join(root, 'outputs', 'cebra_eval', 'ensembles')
OUT_DIRS = [mlp_dir, '/mnt/c/Users/amits/Desktop']

# ─── LOAD ─────────────────────────────────────────────────────────────────────
gpv_mlp  = np.load(os.path.join(mlp_dir, 'importance_global_pv_semantic.npy'))  # (29,23,11)
gpv_ceb  = np.load(os.path.join(ceb_dir, 'importance_global_pv_semantic.npy'))
ig_mlp   = np.load(os.path.join(mlp_dir, 'importance_ig_semantic.npy'))
ig_ceb   = np.load(os.path.join(ceb_dir, 'importance_ig_semantic.npy'))
all_r2_mlp = np.load(os.path.join(mlp_dir, 'all_r2.npy'))   # (5,29,23)
all_r2_ceb = np.load(os.path.join(ceb_dir, 'all_r2.npy'))

with open(os.path.join(mlp_dir, 'semantic_groups.pkl'), 'rb') as f:
    semantic_groups = pickle.load(f)
group_names  = [g[0] for g in semantic_groups]
n_groups     = len(group_names)
short_labels = [FEATURE_NAMES_SHORT.get(g, g) for g in group_names]

# ─── VALIDITY MASKS ───────────────────────────────────────────────────────────
# Use nanmean >= 0.01 for TempConv-Cont (some seeds have NaN due to CEBRA convergence)
R2_THR  = 0.01
mean_mlp = np.nanmean(all_r2_mlp, axis=0)
mean_ceb = np.nanmean(all_r2_ceb, axis=0)

valid_mlp = (~np.any(np.isnan(all_r2_mlp), axis=0)) & (mean_mlp >= R2_THR)
valid_ceb = ~np.all(np.isnan(gpv_ceb), axis=-1)   # use GPV NaN pattern as validity

n_mlp = valid_mlp.sum()
n_ceb = valid_ceb.sum()
print(f'Valid pairs — MLP: {n_mlp}  TempConv-Cont: {n_ceb}')

# ─── GROUP MEANS ──────────────────────────────────────────────────────────────
mlp_gpv_mean = np.nanmean(gpv_mlp[valid_mlp], axis=0)   # (11,)
ceb_gpv_mean = np.nanmean(gpv_ceb[valid_ceb], axis=0)
mlp_gpv_sem  = np.nanstd(gpv_mlp[valid_mlp],  axis=0) / np.sqrt(n_mlp)
ceb_gpv_sem  = np.nanstd(gpv_ceb[valid_ceb],  axis=0) / np.sqrt(n_ceb)

mlp_ig_mean  = np.nanmean(ig_mlp[valid_mlp],  axis=0)
ceb_ig_mean  = np.nanmean(ig_ceb[valid_ceb],  axis=0)
mlp_ig_sem   = np.nanstd(ig_mlp[valid_mlp],   axis=0) / np.sqrt(n_mlp)
ceb_ig_sem   = np.nanstd(ig_ceb[valid_ceb],   axis=0) / np.sqrt(n_ceb)

print('Group GPV (MLP vs TempConv-Cont):')
for g, m, c in zip(short_labels, mlp_gpv_mean, ceb_gpv_mean):
    print(f'  {g:15s}  MLP={m:.4f}  Cont={c:.4f}')

# ─── FIGURE ───────────────────────────────────────────────────────────────────
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig, [ax_l, ax_r])

x   = np.arange(n_groups)
w   = 0.35
ek  = dict(ecolor='k', lw=0.7, capsize=3)
c_mlp = MODEL_COLORS['MLP']
c_ceb = MODEL_COLORS['TempConv-Cont']

for ax, mlp_vals, mlp_sems, ceb_vals, ceb_sems, ylabel, panel in [
    (ax_l, mlp_gpv_mean, mlp_gpv_sem, ceb_gpv_mean, ceb_gpv_sem,
     AXIS_LABELS['r2_drop'], 'A'),
    (ax_r, mlp_ig_mean,  mlp_ig_sem,  ceb_ig_mean,  ceb_ig_sem,
     AXIS_LABELS['ig'],    'B'),
]:
    ax.bar(x - w/2, mlp_vals, w, yerr=mlp_sems,
           color=c_mlp, alpha=0.85, label='MLP', error_kw=ek, linewidth=0)
    ax.bar(x + w/2, ceb_vals, w, yerr=ceb_sems,
           color=c_ceb, alpha=0.85, label='TempConv-Cont', error_kw=ek, linewidth=0)

    ax.axhline(0, color='#888', lw=0.7, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha='right',
                       fontsize=max(6, FONT.TICK - 3))
    ax.set_ylabel(ylabel, fontsize=FONT.LABEL - 1)
    if panel == 'A':
        ax.legend(fontsize=FONT.LEGEND - 1, frameon=False)
    add_panel_label(ax, panel)

add_footnote(fig,
    f'Mean ± SEM across valid pairs: MLP n={n_mlp}, TempConv-Cont n={n_ceb}; '
    f'R² ≥ {R2_THR}; IG = mean |integrated gradient|')
savefig_manifest(fig, 'group_attribution_comparison.png', OUT_DIRS)
print('Saved group_attribution_comparison.png')
