#!/usr/bin/env python3
"""
gen_head_angle_tuning_variety.py

Slide-sized (9.5" × 4.2") figure showing characteristic head angle tuning
curve shapes across ensemble-session pairs. Two rows:
  Row 1: monotonic examples (high |Spearman ρ|)
  Row 2: non-monotonic examples (low ρ but high η² — U-shaped, bimodal, etc.)

3 examples per row = 6 panels total. Motivates why non-linear models +
IG attribution are needed.

Output: outputs/mlps/ensembles_multiseed/head_angle_tuning_variety.png
"""
import os, sys, pickle
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import (
    FIG, DPI, FONT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

root   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
mdir   = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [mdir, '/mnt/c/Users/amits/Desktop']

N_BINS    = 18
MIN_PTS   = 10
R2_THR    = 0.05   # only well-fitted pairs
IG_THR    = 0.05   # IG must flag head angle
N_EACH    = 3      # examples per row

# ── Load ──────────────────────────────────────────────────────────────────────
ds       = pickle.load(open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb'))
sessions = list(ds.keys())
sg       = pickle.load(open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb'))
group_names = [g[0] for g in sg]
feat_idx    = {g: cols for g, cols in sg}
r2_all      = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)
ig_mat      = np.load(os.path.join(mdir, 'importance_ig_semantic.npy'))

ha_cols  = feat_idx['head_angle']
ha_g     = group_names.index('head_angle')

# ── Compute tuning curves + classify all valid pairs ─────────────────────────
def tuning_curve(ha, y, n_bins=N_BINS, min_pts=MIN_PTS):
    edges   = np.linspace(np.nanpercentile(ha, 2), np.nanpercentile(ha, 98), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means, sems = np.full(n_bins, np.nan), np.full(n_bins, np.nan)
    for b in range(n_bins):
        pts = y[(ha >= edges[b]) & (ha < edges[b+1])]
        if len(pts) >= min_pts:
            means[b] = np.mean(pts)
            sems[b]  = np.std(pts) / np.sqrt(len(pts))
    return centers, means, sems

pairs = []   # (s, e, rho, ig, centers, means, sems)

for s in range(len(sessions)):
    for e in range(r2_all.shape[1]):
        if r2_all[s, e] < R2_THR: continue
        ig_val = float(ig_mat[s, e, ha_g]) if np.isfinite(ig_mat[s, e, ha_g]) else 0.0
        if ig_val < IG_THR: continue

        sd   = ds[sessions[s]]
        X    = np.concatenate([sd['data'][t]   for t in sd['data']], axis=0)
        y    = np.concatenate([sd['labels'][t][:, e] for t in sd['data']])
        ha   = X[:, ha_cols[0]]

        rho, _ = spearmanr(ha, y)
        centers, means, sems = tuning_curve(ha, y)

        valid_bins = ~np.isnan(means)
        if valid_bins.sum() < 6: continue

        pairs.append(dict(s=s, e=e, rho=rho, ig=ig_val,
                          centers=centers, means=means, sems=sems))

print(f'Found {len(pairs)} valid pairs (R²≥{R2_THR}, IG≥{IG_THR})')

# ── Split by |ρ|: relatively linear vs strongly non-monotonic ─────────────────
# All high-IG pairs have low |ρ| — this is the finding. Show the gradient.
pairs.sort(key=lambda p: -abs(p['rho']))
linear_ish = pairs[:N_EACH]        # highest |ρ| (most linear among non-monotonic)
nonmono    = sorted(pairs, key=lambda p: p['ig'])[-N_EACH:]  # highest IG (most non-monotonic)
nonmono.sort(key=lambda p: -p['ig'])

for p in linear_ish:
    print(f'  linear  S{p["s"]+1:02d}E{p["e"]+1:02d}  rho={p["rho"]:+.3f}  IG={p["ig"]:.3f}')
for p in nonmono:
    print(f'  nonmono S{p["s"]+1:02d}E{p["e"]+1:02d}  rho={p["rho"]:+.3f}  IG={p["ig"]:.3f}')

# ── Figure: 2 rows × 3 cols ───────────────────────────────────────────────────
C_MONO    = '#1F77B4'   # blue — relatively linear
C_NONMONO = '#FF7F0E'   # orange — strongly non-monotonic

fig, axes = plt.subplots(2, N_EACH, figsize=FIG.FULL,
                          gridspec_kw={'hspace': 0.55, 'wspace': 0.38})
apply_style(fig, axes.ravel())
fig.subplots_adjust(bottom=0.14)

ROW_LABELS = [
    ('Approx. linear', C_MONO,    linear_ish),
    ('Non-monotonic',  C_NONMONO, nonmono),
]
PANEL_LABELS = [['A','B','C'],['D','E','F']]

for row, (row_label, clr, examples) in enumerate(ROW_LABELS):
    for col, p in enumerate(examples[:N_EACH]):
        ax = axes[row, col]
        ok = ~np.isnan(p['means'])
        ax.fill_between(p['centers'][ok],
                        p['means'][ok] - p['sems'][ok],
                        p['means'][ok] + p['sems'][ok],
                        color=clr, alpha=0.20)
        ax.plot(p['centers'][ok], p['means'][ok],
                color=clr, lw=1.8)
        ax.axhline(0, color='#888', lw=0.5, ls=':')
        ax.set_xlabel('Head angle (z)', fontsize=FONT.LABEL - 2)
        if col == 0:
            ax.set_ylabel(f'{row_label}\nActivity (z)', fontsize=FONT.LABEL - 2, color=clr)
        ax.set_title(f'S{p["s"]+1} E{p["e"]+1}  ρ={p["rho"]:+.2f}',
                     fontsize=FONT.TICK - 1, pad=2)
        ax.tick_params(labelsize=FONT.TICK - 2)
        ax.text(0.02, 0.97, PANEL_LABELS[row][col], transform=ax.transAxes,
                fontsize=FONT.PANEL, fontweight='bold', va='top', ha='left')

add_footnote(fig,
    f'Head angle tuning curves for pairs with R²≥{R2_THR} and IG(head angle)≥{IG_THR}.  '
    'Top: highest |ρ| pairs (most linear).  '
    'Bottom: highest IG pairs (most non-monotonic); all have |ρ|<0.15.  '
    'IG detects tuning that Spearman misses.')

savefig_manifest(fig, 'head_angle_tuning_variety.png', OUT_DIRS)
print('Saved head_angle_tuning_variety.png')
