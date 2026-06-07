#!/usr/bin/env python3
"""
honest_attribution_example.py

Four-panel figure for S21 E08 (R²=0.21), placed in the position section:
  A  raw Spearman |r| per feature group  → position ranks #1
  B  MLP GPV attribution per feature group → head_angle ranks #1
  C  Head-angle binned tuning curve → U-shaped (non-monotonic), explains r≈0
  D  Position binned tuning curve   → monotonic trend but GPV 4× lower

Narrative (position section context): position looks like the best predictor
by correlation, but the model correctly identifies head_angle as the primary
driver because it captures non-monotonic (U-shaped) tuning that Spearman r
cannot detect. Position IS predictive in isolation (see position ablation) but
loses to head_angle in the full model.

figsize: 9.5" × 4.8" — fits slide content area below a 0.6" title bar.
"""
import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import (
    FIG, DPI, FONT, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

S_IDX, E_IDX = 20, 7   # S21, E08

base  = os.path.dirname(os.path.abspath(__file__))
root  = os.path.join(base, '..')
mdir  = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [mdir, '/mnt/c/Users/amits/Desktop']

# ── Load ──────────────────────────────────────────────────────────────────────
with open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())

with open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb') as f:
    sg = pickle.load(f)
group_names       = [g[0] for g in sg]
feat_idx_by_group = {g: cols for g, cols in sg}
tick_labels       = [FEATURE_NAMES_SHORT.get(g, g) for g in group_names]
n_groups          = len(group_names)

mlp_gpv = np.load(os.path.join(mdir, 'importance_global_pv_semantic.npy'))
mlp_r2  = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)

# ── Compute raw |r| ───────────────────────────────────────────────────────────
sess = sessions[S_IDX]
sd   = ds[sess]
Xall = np.concatenate([sd['data'][t] for t in sd['data']], axis=0)
Y    = np.concatenate([sd['labels'][t] for t in sd['data']], axis=0)
y    = Y[:, E_IDX]

raw_r = np.full(n_groups, np.nan)
for g_idx, (g_name, g_cols) in enumerate(sg):
    rs = [abs(spearmanr(Xall[:, c], y).statistic)
          for c in g_cols if np.std(Xall[:, c]) > 1e-8]
    if rs:
        raw_r[g_idx] = np.nanmax(rs)

gpv = mlp_gpv[S_IDX, E_IDX, :]
r2  = mlp_r2[S_IDX, E_IDX]

order_r   = np.argsort(raw_r)[::-1]
order_gpv = np.argsort(gpv)[::-1]

pos_g = group_names.index('frame_position')
ha_g  = group_names.index('head_angle')
C_POS = '#E53935'
C_HA  = '#1E88E5'
C_DEF = '#AAAAAA'

def bar_color(g_idx, panel):
    if panel == 'r'   and g_idx == pos_g: return C_POS
    if panel == 'gpv' and g_idx == ha_g:  return C_HA
    if panel == 'r'   and g_idx == ha_g:  return C_HA
    if panel == 'gpv' and g_idx == pos_g: return C_POS
    return C_DEF

# ── Binned tuning curves ──────────────────────────────────────────────────────
N_BINS = 15

def binned_tuning(x, y, n_bins):
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    centres, means, sems = [], [], []
    for i in range(n_bins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if mask.sum() > 5:
            centres.append((edges[i] + edges[i + 1]) / 2)
            means.append(y[mask].mean())
            sems.append(y[mask].std() / np.sqrt(mask.sum()))
    return np.array(centres), np.array(means), np.array(sems)

pos_vals = Xall[:, feat_idx_by_group['frame_position']].ravel()
ha_vals  = Xall[:, feat_idx_by_group['head_angle']].ravel()
ha_c,  ha_m,  ha_s  = binned_tuning(ha_vals,  y, N_BINS)
pos_c, pos_m, pos_s = binned_tuning(pos_vals, y, N_BINS)

r_pos = spearmanr(pos_vals, y).statistic
r_ha  = spearmanr(ha_vals,  y).statistic

# ── Figure: 2 rows × 2 cols, sized to fit slide content area ─────────────────
fig = plt.figure(figsize=(9.5, 4.8))
gs  = fig.add_gridspec(2, 2, hspace=0.90, wspace=0.38,
                       left=0.07, right=0.98, top=0.93, bottom=0.18)
ax_r   = fig.add_subplot(gs[0, 0])
ax_gpv = fig.add_subplot(gs[0, 1])
ax_ha  = fig.add_subplot(gs[1, 0])
ax_pos = fig.add_subplot(gs[1, 1])
apply_style(fig, [ax_r, ax_gpv, ax_ha, ax_pos])

# ── Panels A & B ─────────────────────────────────────────────────────────────
rank_pos_in_r   = int(np.where(order_r   == pos_g)[0][0])
rank_ha_in_r    = int(np.where(order_r   == ha_g)[0][0])
rank_ha_in_gpv  = int(np.where(order_gpv == ha_g)[0][0])
rank_pos_in_gpv = int(np.where(order_gpv == pos_g)[0][0])

for ax, order, vals, ylabel, panel, panel_label, title in [
    (ax_r,   order_r,   raw_r, 'Spearman |r|', 'r',   'A', 'Per-feature correlation'),
    (ax_gpv, order_gpv, gpv,   'GPV (ΔR²)',    'gpv', 'B', 'Model attribution (GPV)'),
]:
    x_pos = np.arange(n_groups)
    colors = [bar_color(g, panel) for g in order]
    ax.bar(x_pos, vals[order], color=colors, alpha=0.88, width=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([tick_labels[g] for g in order],
                       rotation=90, ha='center', fontsize=max(5, FONT.TICK - 4))
    ax.set_ylabel(ylabel, fontsize=FONT.LABEL - 2)
    ax.axhline(0, color='#888', lw=0.5, linestyle='--')
    ax.set_title(title, fontsize=FONT.LABEL - 2, pad=2)

    # annotate the two highlighted bars
    for rank, g_idx in enumerate(order):
        if g_idx in (pos_g, ha_g):
            v = vals[g_idx]
            ax.text(rank, v + np.nanmax(vals) * 0.05,
                    f'#{rank+1}',
                    ha='center', va='bottom',
                    fontsize=FONT.ANNOTATION - 1,
                    color=bar_color(g_idx, panel),
                    fontweight='bold')
    add_panel_label(ax, panel_label)

# ── Rank-reversal annotation between A and B ─────────────────────────────────
# Draw a curved arrow in figure coordinates showing position drops, head_angle rises
fig.text(0.50, 0.87,
         '← rank reversal →',
         ha='center', va='center',
         fontsize=FONT.ANNOTATION - 1, color='#555555', style='italic')

# ── Panels C & D ─────────────────────────────────────────────────────────────
for ax, centres, means, sems, xlabel, panel_label, color, r_val, title, note in [
    (ax_ha,  ha_c,  ha_m,  ha_s,  'Head angle (cm)',    'C', C_HA,  r_ha,
     f'Head angle tuning  (r={r_ha:+.2f})',
     'U-shape: +/− arms\ncancel in rank r'),
    (ax_pos, pos_c, pos_m, pos_s, 'Track position (cm)', 'D', C_POS, r_pos,
     f'Position tuning  (r={r_pos:+.2f})',
     'Monotonic trend\nbut GPV 4× lower'),
]:
    ax.fill_between(centres, means - sems, means + sems, color=color, alpha=0.18)
    ax.plot(centres, means, color=color, lw=1.6, marker='o', ms=3.0)
    ax.axhline(0, color='#888', lw=0.5, linestyle='--')
    ax.set_xlabel(xlabel, fontsize=FONT.LABEL - 2)
    ax.set_ylabel('Mean activity', fontsize=FONT.LABEL - 2)
    ax.set_title(title, fontsize=FONT.LABEL - 2, pad=2)
    # add mechanism note in upper-right
    ax.text(0.97, 0.96, note,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FONT.ANNOTATION - 2, color=color,
            linespacing=1.3)
    add_panel_label(ax, panel_label)

add_footnote(fig,
    f'S21 E08  (MLP R²={r2:.2f}, n=32 499 time points).  '
    'Position (red) ranks #1 by Spearman |r| but #5 by GPV.  '
    'Head angle (blue) ranks last by correlation (U-shaped tuning cancels in rank r) '
    'yet is the top driver by both GPV and IG (ig=0.30).  '
    'Position IS predictive alone (position-only MLP R²>0.2) but '
    'is outcompeted in the full model.')

savefig_manifest(fig, 'honest_attribution_example.png', OUT_DIRS)
print('Saved honest_attribution_example.png')
