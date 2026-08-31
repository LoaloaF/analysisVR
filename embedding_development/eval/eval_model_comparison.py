#!/usr/bin/env python3
"""
eval_model_comparison.py

Cross-model attribution comparison: MLP vs TempConv-Cont vs TempConv-Pred.

Questions:
  1. Do GPV rankings agree across models? (scatter: MLP vs TempConv per pair)
  2. Does IG agree with GPV within each model? (within-model scatter)
  3. Which semantic groups do different models emphasise? (group-level bar)
  4. Where do MLP and TempConv DISAGREE? (pairs with large IG delta)

Inputs:
  outputs/mlps/ensembles_multiseed/importance_global_pv_semantic.npy   (29,23,11)
  outputs/mlps/ensembles_multiseed/importance_ig_semantic.npy          (29,23,11)
  outputs/mlps/ensembles_multiseed/all_r2.npy                          (5,29,23)
  outputs/cebra_eval/ensembles/importance_global_pv_semantic.npy       (29,23,11)
  outputs/cebra_eval/ensembles/importance_ig_semantic.npy              (29,23,11)
  outputs/cebra_eval/ensembles/all_r2.npy                              (5,29,23)
  outputs/cebra_pred_eval/ensembles/importance_global_pv_semantic.npy  (29,23,11)
  outputs/cebra_pred_eval/ensembles/importance_ig_semantic.npy         (29,23,11)  [optional]
  outputs/cebra_pred_eval/ensembles/all_r2.npy                         (5,29,23)
"""

import os, shutil, pickle, warnings
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'utils'))
from figure_style import (
    FIG, DPI, FONT, LINE, MARKER,
    MODEL_COLORS, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)
import pandas as pd

warnings.filterwarnings('ignore')

# ─── paths ───────────────────────────────────────────────────────────────────
mlp_dir      = './outputs/mlps/ensembles_multiseed'
ceb_dir      = './outputs/cebra_eval/ensembles'
pred_dir     = './outputs/cebra_pred_eval/ensembles'
output_dir   = './outputs/model_comparison'
desktop_dir  = '/mnt/c/Users/amits/Desktop/model_comparison'

R2_THRESHOLD = 0.01

for d in (output_dir, desktop_dir):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

with open(os.path.join(mlp_dir, 'semantic_groups.pkl'), 'rb') as f:
    semantic_groups = pickle.load(f)
group_names = [g[0] for g in semantic_groups]
n_groups    = len(group_names)
short_names = {
    'frame_raw_500msMedian':                    'speed',
    'frame_raw_abs_acc_500msMedian':            'acc',
    'frame_YawPitch_abs_vel_sum_500msMedian':   'rot_vel',
    'frame_YawPitch_abs_acc_sum_500msMedian':   'rot_acc',
    'head_angle_vel':                           'head_vel',
    'head_angle':                               'head_ang',
    'frame_position':                           'position',
    'cue_visible':                              'cue_vis',
    'upcoming_choice':                          'up_choice',
    'reward_window':                            'reward',
    'lick_detected':                            'lick',
}
glabels = [short_names.get(g, g) for g in group_names]


def load_arr(path):
    return np.load(path) if os.path.exists(path) else None


def validity_mask(all_r2_path):
    arr = load_arr(all_r2_path)
    if arr is None:
        return None
    mean_r2 = np.nanmean(arr, axis=0)
    return (~np.all(np.isnan(arr), axis=0)) & (mean_r2 >= R2_THRESHOLD)


def savefig(fname):
    savefig_manifest(plt.gcf(), fname, [output_dir, desktop_dir])


# ─── load all arrays ─────────────────────────────────────────────────────────
mlp_gpv  = load_arr(os.path.join(mlp_dir,  'importance_global_pv_semantic.npy'))
mlp_ig   = load_arr(os.path.join(mlp_dir,  'importance_ig_semantic.npy'))
mlp_r2   = load_arr(os.path.join(mlp_dir,  'all_r2.npy'))
mlp_valid = validity_mask(os.path.join(mlp_dir, 'all_r2.npy'))

ceb_gpv  = load_arr(os.path.join(ceb_dir,  'importance_global_pv_semantic.npy'))
ceb_ig   = load_arr(os.path.join(ceb_dir,  'importance_ig_semantic.npy'))
ceb_r2   = load_arr(os.path.join(ceb_dir,  'all_r2.npy'))
ceb_valid = validity_mask(os.path.join(ceb_dir, 'all_r2.npy'))

pred_gpv  = load_arr(os.path.join(pred_dir, 'importance_global_pv_semantic.npy'))
pred_ig   = load_arr(os.path.join(pred_dir, 'importance_ig_semantic.npy'))
pred_r2   = load_arr(os.path.join(pred_dir, 'all_r2.npy'))
pred_valid = validity_mask(os.path.join(pred_dir, 'all_r2.npy'))

print(f"MLP    valid pairs: {mlp_valid.sum() if mlp_valid is not None else 'N/A'}")
print(f"TempConv-Cont valid pairs: {ceb_valid.sum() if ceb_valid is not None else 'N/A'}")
print(f"TempConv-Pred valid pairs: {pred_valid.sum() if pred_valid is not None else 'N/A'}")


# ─── shared valid mask (intersection of all models) ──────────────────────────
# For fair per-pair comparisons we need pairs valid in BOTH models being compared
def shared_mask(*masks):
    result = np.ones(masks[0].shape, dtype=bool)
    for m in masks:
        if m is not None:
            result &= m
    return result

both_mlp_ceb  = shared_mask(mlp_valid, ceb_valid)
both_mlp_pred = shared_mask(mlp_valid, pred_valid) if pred_valid is not None else None
all_three     = shared_mask(mlp_valid, ceb_valid, pred_valid) if pred_valid is not None else None

print(f"Shared MLP∩TempConv-Cont: {both_mlp_ceb.sum()}")
if both_mlp_pred is not None:
    print(f"Shared MLP∩TempConv-Pred: {both_mlp_pred.sum()}")
if all_three is not None:
    print(f"Shared all three:   {all_three.sum()}")


# ─── Figure 1: Mean R² per model ─────────────────────────────────────────────
models_r2 = {}
if mlp_r2 is not None:
    m = np.nanmean(mlp_r2, axis=0)
    models_r2['MLP'] = m[mlp_valid]
if ceb_r2 is not None:
    m = np.nanmean(ceb_r2, axis=0)
    models_r2['TempConv-Cont'] = m[ceb_valid]
if pred_r2 is not None:
    m = np.nanmean(pred_r2, axis=0)
    models_r2['TempConv-Pred'] = m[pred_valid]

colors = {'MLP': '#4CAF50', 'TempConv-Cont': '#2196F3', 'TempConv-Pred': '#FF9800'}

fig, ax = plt.subplots(figsize=(6, 4))
apply_style(fig, ax)
names = list(models_r2.keys())
means = [v.mean() for v in models_r2.values()]
sems  = [v.std() / np.sqrt(len(v)) for v in models_r2.values()]
bars  = ax.bar(names, means, yerr=sems, capsize=4,
               color=[colors[n] for n in names], alpha=0.85, width=0.5)
ax.set_ylabel('Mean R² (test set, valid pairs only)')
ax.set_title('Prediction performance across models')
for bar, m, s, vals in zip(bars, means, sems, models_r2.values()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.002,
            f'{m:.3f}\n(n={len(vals)})', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
savefig('fig1_r2_comparison.png')
print('Saved fig1_r2_comparison.png')


# ─── Figure 2: Group-level mean attribution bars (all models) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
apply_style(fig, axes)

# GPV panel
ax = axes[0]
x  = np.arange(n_groups)
w  = 0.28
offset = -w
for arm, gpv, valid, color in [
    ('MLP',        mlp_gpv,  mlp_valid,  '#4CAF50'),
    ('TempConv-Cont', ceb_gpv,  ceb_valid,  '#2196F3'),
    ('TempConv-Pred', pred_gpv, pred_valid, '#FF9800'),
]:
    if gpv is None or valid is None:
        offset += w; continue
    vals = np.nanmean(gpv[valid], axis=0)   # (11,) mean over valid pairs
    ax.bar(x + offset, vals, w, label=arm, color=color, alpha=0.85)
    offset += w

ax.set_xticks(x)
ax.set_xticklabels(glabels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Mean GPV attribution')
ax.set_title('GPV: which features matter?')
ax.legend(fontsize=9)

# IG panel
ax = axes[1]
offset = -w
for arm, ig, valid, color in [
    ('MLP',        mlp_ig,  mlp_valid,  '#4CAF50'),
    ('TempConv-Cont', ceb_ig,  ceb_valid,  '#2196F3'),
    ('TempConv-Pred', pred_ig, pred_valid, '#FF9800'),
]:
    if ig is None or valid is None:
        offset += w; continue
    vals = np.nanmean(ig[valid], axis=0)
    ax.bar(x + offset, vals, w, label=arm, color=color, alpha=0.85)
    offset += w

ax.set_xticks(x)
ax.set_xticklabels(glabels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Mean IG attribution')
ax.set_title('IG: which features matter?')
ax.legend(fontsize=9)

plt.suptitle('Feature importance by semantic group — all models', fontsize=11)
plt.tight_layout()
savefig('fig2_group_attribution_bars.png')
print('Saved fig2_group_attribution_bars.png')


# ─── Figure 3: Per-pair scatter — MLP GPV vs TempConv-Cont GPV ──────────────────
if mlp_gpv is not None and ceb_gpv is not None:
    mask2d = both_mlp_ceb
    x_vals = mlp_gpv[mask2d].flatten()     # (n_pairs × 11,)
    y_vals = ceb_gpv[mask2d].flatten()

    ok = ~(np.isnan(x_vals) | np.isnan(y_vals))
    x_ok, y_ok = x_vals[ok], y_vals[ok]
    rho, p = spearmanr(x_ok, y_ok)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    apply_style(fig, axes)

    ax = axes[0]
    ax.scatter(x_ok, y_ok, s=6, alpha=0.3, color='steelblue')
    lim = max(x_ok.max(), y_ok.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='y=x')
    ax.set_xlabel('MLP GPV')
    ax.set_ylabel('TempConv-Cont GPV')
    ax.set_title(f'GPV agreement: MLP vs TempConv-Cont\n(all valid pairs × all groups)\nρ={rho:.3f}, p={p:.2e}')
    ax.legend(fontsize=8)

    # IG agreement
    if mlp_ig is not None and ceb_ig is not None:
        ax = axes[1]
        xi = mlp_ig[mask2d].flatten()
        yi = ceb_ig[mask2d].flatten()
        ok2 = ~(np.isnan(xi) | np.isnan(yi))
        xi, yi = xi[ok2], yi[ok2]
        rho2, p2 = spearmanr(xi, yi)
        ax.scatter(xi, yi, s=6, alpha=0.3, color='coral')
        lim2 = max(xi.max(), yi.max()) * 1.05
        ax.plot([0, lim2], [0, lim2], 'k--', lw=1, label='y=x')
        ax.set_xlabel('MLP IG')
        ax.set_ylabel('TempConv-Cont IG')
        ax.set_title(f'IG agreement: MLP vs TempConv-Cont\nρ={rho2:.3f}, p={p2:.2e}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    savefig('fig3_mlp_vs_tempconv_scatter.png')
    print('Saved fig3_mlp_vs_tempconv_scatter.png')


# ─── Figure 4: Within-model IG vs GPV (does IG agree with GPV?) ──────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
apply_style(fig, axes)

for ax, arm, gpv, ig, valid, color in [
    (axes[0], 'MLP',        mlp_gpv,  mlp_ig,  mlp_valid,  '#4CAF50'),
    (axes[1], 'TempConv-Cont', ceb_gpv,  ceb_ig,  ceb_valid,  '#2196F3'),
    (axes[2], 'TempConv-Pred', pred_gpv, pred_ig, pred_valid, '#FF9800'),
]:
    if gpv is None or ig is None or valid is None:
        ax.set_title(f'{arm} (no data)')
        continue
    x_vals = gpv[valid].flatten()
    y_vals = ig[valid].flatten()
    ok = ~(np.isnan(x_vals) | np.isnan(y_vals))
    xo, yo = x_vals[ok], y_vals[ok]
    if ok.sum() < 4:
        ax.set_title(f'{arm} (too few points)')
        continue
    rho, p = spearmanr(xo, yo)
    ax.scatter(xo, yo, s=6, alpha=0.3, color=color)
    ax.set_xlabel('GPV attribution')
    ax.set_ylabel('IG attribution')
    ax.set_title(f'{arm}: GPV vs IG\nρ={rho:.3f}, p={p:.2e}')

plt.suptitle('Within-model: do GPV and IG agree?', fontsize=11)
plt.tight_layout()
savefig('fig4_gpv_vs_ig_within_model.png')
print('Saved fig4_gpv_vs_ig_within_model.png')


# ─── Figure 5: Disagreement map — where MLP and TempConv diverge ────────────────
if mlp_ig is not None and ceb_ig is not None:
    mask2d = both_mlp_ceb

    # Normalise each model's IG per-pair to [0,1] range (so magnitudes are comparable)
    mlp_ig_norm  = mlp_ig.copy()
    ceb_ig_norm  = ceb_ig.copy()
    for s in range(mlp_ig.shape[0]):
        for n in range(mlp_ig.shape[1]):
            if mask2d[s, n]:
                mx = np.nanmax(mlp_ig[s, n])
                if mx > 0: mlp_ig_norm[s, n] /= mx
                mx = np.nanmax(ceb_ig[s, n])
                if mx > 0: ceb_ig_norm[s, n] /= mx

    # Delta: (TempConv - MLP) per group, averaged over shared valid pairs
    delta = np.nanmean((ceb_ig_norm - mlp_ig_norm)[mask2d], axis=0)   # (11,)

    fig, ax = plt.subplots(figsize=(9, 4))
    apply_style(fig, ax)
    clr = ['#E53935' if d > 0 else '#1565C0' for d in delta]
    ax.bar(np.arange(n_groups), delta, color=clr, alpha=0.85)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels(glabels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('TempConv-Cont IG − MLP IG  (normalised)')
    ax.set_title('Where TempConv-Cont and MLP disagree\n'
                 'Red = TempConv emphasises more, Blue = MLP emphasises more')
    plt.tight_layout()
    savefig('fig5_disagreement_bar.png')
    print('Saved fig5_disagreement_bar.png')

    # Group-level breakdown: rank correlation per session
    rho_per_sess = []
    n_sess = mlp_ig.shape[0]
    for s in range(n_sess):
        valid_n = mask2d[s]
        if valid_n.sum() < 3:
            continue
        m_vals = np.nanmean(mlp_ig_norm[s, valid_n], axis=0)   # (11,)
        c_vals = np.nanmean(ceb_ig_norm[s, valid_n], axis=0)
        ok = ~(np.isnan(m_vals) | np.isnan(c_vals))
        if ok.sum() < 4:
            continue
        rho, _ = spearmanr(m_vals[ok], c_vals[ok])
        rho_per_sess.append(rho)

    if rho_per_sess:
        rho_arr = np.array(rho_per_sess)
        print(f'\nPer-session group-rank agreement (MLP vs TempConv-Cont IG):')
        print(f'  median ρ = {np.median(rho_arr):.3f}  '
              f'[{np.percentile(rho_arr, 25):.3f}, {np.percentile(rho_arr, 75):.3f}]')


# ─── Figure 6: Predictive vs Contrastive (if pred IG available) ───────────────
if pred_ig is not None and ceb_ig is not None:
    mask2d_pc = shared_mask(ceb_valid, pred_valid)
    xi = ceb_ig[mask2d_pc].flatten()
    yi = pred_ig[mask2d_pc].flatten()
    ok = ~(np.isnan(xi) | np.isnan(yi))
    xo, yo = xi[ok], yi[ok]
    if ok.sum() >= 4:
        rho, p = spearmanr(xo, yo)
        fig, ax = plt.subplots(figsize=(5, 5))
        apply_style(fig, ax)
        ax.scatter(xo, yo, s=6, alpha=0.3, color='mediumpurple')
        lim = max(xo.max(), yo.max()) * 1.05
        ax.plot([0, lim], [0, lim], 'k--', lw=1, label='y=x')
        ax.set_xlabel('TempConv-Cont IG')
        ax.set_ylabel('TempConv-Pred IG')
        ax.set_title(f'TempConv-Cont vs TempConv-Pred IG\n'
                     f'Do training objectives change feature emphasis?\nρ={rho:.3f}, p={p:.2e}')
        ax.legend(fontsize=8)
        plt.tight_layout()
        savefig('fig6_tempconv_cont_vs_pred_scatter.png')
        print('Saved fig6_tempconv_cont_vs_pred_scatter.png')

        # Group-level delta: pred - cont
        pred_ig_norm = pred_ig.copy(); ceb_ig_norm2 = ceb_ig.copy()
        for s in range(pred_ig.shape[0]):
            for n in range(pred_ig.shape[1]):
                if mask2d_pc[s, n]:
                    mx = np.nanmax(pred_ig[s, n]);
                    if mx > 0: pred_ig_norm[s, n] /= mx
                    mx = np.nanmax(ceb_ig[s, n]);
                    if mx > 0: ceb_ig_norm2[s, n] /= mx
        delta_pc = np.nanmean((pred_ig_norm - ceb_ig_norm2)[mask2d_pc], axis=0)
        fig, ax = plt.subplots(figsize=(9, 4))
        apply_style(fig, ax)
        clr = ['#E53935' if d > 0 else '#1565C0' for d in delta_pc]
        ax.bar(np.arange(n_groups), delta_pc, color=clr, alpha=0.85)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(np.arange(n_groups))
        ax.set_xticklabels(glabels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('TempConv-Pred IG − TempConv-Cont IG  (normalised)')
        ax.set_title('TempConv-Cont vs TempConv-Pred: feature emphasis shift\n'
                     'Red = TempConv-Pred emphasises more')
        plt.tight_layout()
        savefig('fig6b_tempconv_pred_vs_cont_delta.png')
        print('Saved fig6b_tempconv_pred_vs_cont_delta.png')


# ─── print summary table ─────────────────────────────────────────────────────
print('\n' + '='*70)
print('Group-level attribution summary (mean over valid pairs, normalised to sum=1)')
print('='*70)
row_data = []
for i, g in enumerate(group_names):
    row = {'group': glabels[i]}
    for arm, gpv, ig, valid in [
        ('MLP',        mlp_gpv,  mlp_ig,  mlp_valid),
        ('TempConv-Cont',    ceb_gpv,  ceb_ig,  ceb_valid),
        ('TempConv-Pred',    pred_gpv, pred_ig, pred_valid),
    ]:
        if gpv is not None and valid is not None:
            v = np.nanmean(gpv[valid, i])
            row[f'{arm}_gpv'] = v
        if ig is not None and valid is not None:
            v = np.nanmean(ig[valid, i])
            row[f'{arm}_ig'] = v
    row_data.append(row)

df = pd.DataFrame(row_data)
print(df.to_string(index=False, float_format='{:.4f}'.format))
df.to_csv(os.path.join(output_dir, 'attribution_summary.csv'), index=False)
print('\nSaved attribution_summary.csv')

print('\nDone.')
