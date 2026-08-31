#!/usr/bin/env python3
"""
eval_cebra_heatmaps.py

Side-by-side attribution heatmaps and bar charts for all three model types:
  MLP, TempConv-Cont, TempConv-Pred.

Generates plots that match the MLP-only figures already in outputs/mlps/.

Figures:
  fig1_attr_bars.png       — mean |attribution| per semantic group (all 6 method/attr combos)
  fig2_gpv_heatmap.png     — sessions × groups heatmap of GPV for all 3 models
  fig3_ig_heatmap.png      — sessions × groups heatmap of IG for all 3 models
  fig4_cross_model_ig.png  — scatter MLP IG vs CEB-Cont IG vs PRED IG per valid pair

Outputs: outputs/cebra_comparison/ and Desktop
"""

import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import spearmanr

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'utils'))
from figure_style import (
    FIG, DPI, FONT, LINE, MARKER,
    MODEL_COLORS, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

# ─── paths ────────────────────────────────────────────────────────────────────
mlp_dir   = 'outputs/mlps/ensembles_multiseed'
ceb_dir   = 'outputs/cebra_eval/ensembles'
pred_dir  = 'outputs/cebra_pred_eval/ensembles'
out_dir   = 'outputs/cebra_comparison'
desk_dir  = '/mnt/c/Users/amits/Desktop/cebra_comparison'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(desk_dir, exist_ok=True)

R2_THRESH = 0.01

# ─── load ─────────────────────────────────────────────────────────────────────
with open('outputs/mlps/ensembles_multiseed/semantic_groups.pkl', 'rb') as f:
    sg = pickle.load(f)
group_names = [g[0] for g in sg]
n_groups    = len(group_names)

# Shortened display labels
SHORT = {
    'frame_raw_500msMedian':                   'Speed',
    'frame_raw_abs_acc_500msMedian':            'Speed acc.',
    'frame_YawPitch_abs_vel_sum_500msMedian':   'Rot. vel.',
    'frame_YawPitch_abs_acc_sum_500msMedian':   'Rot. acc.',
    'head_angle_vel':                           'Head vel.',
    'head_angle':                               'Head angle',
    'frame_position':                           'Position',
    'cue_visible':                              'Cue vis.',
    'upcoming_choice':                          'Up. choice',
    'reward_window':                            'Reward win.',
    'lick_detected':                            'Lick det.',
}
glabels = [SHORT.get(g, g) for g in group_names]

# validity masks
def load_valid(r2_path):
    r2 = np.load(r2_path)   # (5, 29, 23)
    return (~np.all(np.isnan(r2), axis=0)) & (np.nanmean(r2, axis=0) >= R2_THRESH)

mlp_valid  = load_valid(f'{mlp_dir}/all_r2.npy')   # (29, 23)
ceb_valid  = load_valid(f'{ceb_dir}/all_r2.npy')
pred_valid = load_valid(f'{pred_dir}/all_r2.npy')

def load_attr(path, valid_mask):
    arr = np.load(path)   # (29, 23, 11)
    out = arr.copy()
    out[~valid_mask] = np.nan
    return out

mlp_gpv  = load_attr(f'{mlp_dir}/importance_global_pv_semantic.npy',  mlp_valid)
mlp_ig   = load_attr(f'{mlp_dir}/importance_ig_semantic.npy',          mlp_valid)
ceb_gpv  = load_attr(f'{ceb_dir}/importance_global_pv_semantic.npy',  ceb_valid)
ceb_ig   = load_attr(f'{ceb_dir}/importance_ig_semantic.npy',          ceb_valid)
pred_gpv = load_attr(f'{pred_dir}/importance_global_pv_semantic.npy', pred_valid)
pred_ig  = load_attr(f'{pred_dir}/importance_ig_semantic.npy',         pred_valid)

n_sessions, n_ens = mlp_gpv.shape[:2]

METHODS = [
    ('MLP',              mlp_gpv,  mlp_ig,  '#1565C0'),
    ('TempConv-Cont',    ceb_gpv,  ceb_ig,  '#6A1B9A'),
    ('TempConv-Pred',    pred_gpv, pred_ig, '#E65100'),
]


def savefig(name):
    savefig_manifest(plt.gcf(), name, [out_dir, desk_dir])
    print(f'Saved {name}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — attribution bar chart
# Rows: GPV, IG   |   Columns: MLP, TempConv-Cont, TempConv-Pred
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
apply_style(fig, axes.flatten())

for col, (label, gpv, ig, color) in enumerate(METHODS):
    for row, (attr, attr_name) in enumerate([(gpv, 'GPV'), (ig, 'IG')]):
        ax = axes[row, col]
        # mean over valid pairs
        means = np.nanmean(attr, axis=(0, 1))    # (11,)
        sems  = np.nanstd(attr, axis=(0, 1)) / np.sqrt(
            np.sum(~np.isnan(attr[:, :, 0])))
        x = np.arange(n_groups)
        bars = ax.bar(x, means, color=color, alpha=0.8, width=0.7)
        ax.errorbar(x, means, yerr=sems, fmt='none', color='black',
                    capsize=3, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(glabels, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'{label} — {attr_name}', fontsize=10)
        if col == 0:
            ax.set_ylabel('Mean attribution', fontsize=9)

plt.suptitle('Feature attribution by semantic group — all models', fontsize=12)
plt.tight_layout()
savefig('fig1_attr_bars.png')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — GPV heatmap (sessions × groups, averaged over ensembles)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
apply_style(fig, axes)

for ax, (label, gpv, ig, color) in zip(axes, METHODS):
    # mean over ensembles, NaN where no valid pairs exist
    heat = np.nanmean(gpv, axis=1)   # (29, 11)
    vmax = np.nanpercentile(heat, 95)
    im = ax.imshow(heat, aspect='auto', cmap='viridis',
                   vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_title(f'{label} — GPV', fontsize=11)
    ax.set_xlabel('Feature group', fontsize=9)
    ax.set_ylabel('Session' if ax is axes[0] else '')
    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels(glabels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(np.arange(n_sessions))
    ax.set_yticklabels([f'S{i:02d}' for i in range(n_sessions)], fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.7, label='GPV')

plt.suptitle('GPV attribution heatmap — sessions × feature groups', fontsize=12)
plt.tight_layout()
savefig('fig2_gpv_heatmap.png')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — IG heatmap (same format)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
apply_style(fig, axes)

for ax, (label, gpv, ig, color) in zip(axes, METHODS):
    heat = np.nanmean(ig, axis=1)   # (29, 11)
    vmax = np.nanpercentile(heat, 95)
    im = ax.imshow(heat, aspect='auto', cmap='plasma',
                   vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_title(f'{label} — IG', fontsize=11)
    ax.set_xlabel('Feature group', fontsize=9)
    ax.set_ylabel('Session' if ax is axes[0] else '')
    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels(glabels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(np.arange(n_sessions))
    ax.set_yticklabels([f'S{i:02d}' for i in range(n_sessions)], fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.7, label='|IG|')

plt.suptitle('IG attribution heatmap — sessions × feature groups', fontsize=12)
plt.tight_layout()
savefig('fig3_ig_heatmap.png')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Cross-model IG scatter (shared valid pairs only)
# ══════════════════════════════════════════════════════════════════════════════
shared_valid = mlp_valid & ceb_valid & pred_valid   # (29, 23)

group_colors = plt.cm.tab10(np.linspace(0, 1, n_groups))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
apply_style(fig, axes)

for ax, (x_arr, x_lab, y_arr, y_lab) in [
    (axes[0], (mlp_ig, 'MLP IG', ceb_ig,  'TempConv-Cont IG')),
    (axes[1], (mlp_ig, 'MLP IG', pred_ig, 'TempConv-Pred IG')),
]:
    for g_idx in range(n_groups):
        mask = shared_valid
        x_vals = x_arr[mask, g_idx]
        y_vals = y_arr[mask, g_idx]
        ok = ~(np.isnan(x_vals) | np.isnan(y_vals))
        ax.scatter(x_vals[ok], y_vals[ok], color=group_colors[g_idx],
                   s=18, alpha=0.6, label=glabels[g_idx])

    # diagonal
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], 'k--', linewidth=0.8, alpha=0.5)

    # overall Spearman
    xs = x_arr[shared_valid].flatten()
    ys = y_arr[shared_valid].flatten()
    ok = ~(np.isnan(xs) | np.isnan(ys))
    rho, p = spearmanr(xs[ok], ys[ok])
    ax.set_xlabel(x_lab, fontsize=10)
    ax.set_ylabel(y_lab, fontsize=10)
    ax.set_title(f'rho={rho:+.3f}  p={p:.1e}  n={ok.sum()}', fontsize=9)

axes[0].legend(loc='upper left', fontsize=6, ncol=2, framealpha=0.7,
               title='Feature group', title_fontsize=7)
plt.suptitle('Cross-model IG agreement per (session, ensemble, group)', fontsize=11)
plt.tight_layout()
savefig('fig4_cross_model_ig.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Ranked attribution bars per feature group (overlay all models)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
apply_style(fig, axes)

width = 0.25
x = np.arange(n_groups)
colors_m  = ['#1565C0', '#6A1B9A', '#E65100']
labels_m  = ['MLP', 'TempConv-Cont', 'TempConv-Pred']

for ax, arr_list, attr_name in [
    (axes[0], [mlp_gpv, ceb_gpv, pred_gpv], 'GPV'),
    (axes[1], [mlp_ig,  ceb_ig,  pred_ig],  'IG'),
]:
    for i, (arr, label, color) in enumerate(zip(arr_list, labels_m, colors_m)):
        means = np.nanmean(arr, axis=(0, 1))
        ax.bar(x + (i - 1) * width, means, width=width,
               color=color, alpha=0.85, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(glabels, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel(f'Mean {attr_name}', fontsize=10)
    ax.set_title(f'{attr_name} — model comparison', fontsize=11)
    ax.legend(fontsize=9)

plt.suptitle('Attribution by feature group — model comparison', fontsize=12)
plt.tight_layout()
savefig('fig5_grouped_bars.png')

print('\nAll figures saved.')
