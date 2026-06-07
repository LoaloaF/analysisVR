#!/usr/bin/env python3
"""
eval_e07_cue_zone_tuning.py

E07 position tuning restricted to the cue-visible track segment.
We train with cue_visible (one-hot), so the feature is only meaningful when
the cue is actually visible — here we filter to exactly those timepoints.

Shows position-binned activity for Cue 1 vs Cue 2, top 4 sessions by Cohen's d,
using only the position range within the cue zone.

Output: outputs/ablation_vs_attribution/e07_cue_zone_tuning.png
"""
import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import (
    FIG, DPI, FONT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, '..')
mdir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [
    os.path.join(root, 'outputs', 'ablation_vs_attribution'),
    '/mnt/c/Users/amits/Desktop',
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

E_IDX    = 6
N_PANELS = 4
N_BINS   = 20
MIN_PTS  = 8

CUE_COLORS = {1: '#FF7F0E', 2: '#D62728'}
CUE_LABELS = {1: 'Cue 1', 2: 'Cue 2'}

# ── Load ──────────────────────────────────────────────────────────────────────
ds          = pickle.load(open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb'))
sessions    = list(ds.keys())
sg          = pickle.load(open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb'))
feat_idx    = {g: cols for g, cols in sg}
r2_all      = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)

pos_cols = feat_idx['frame_position']
cue_cols = feat_idx['cue_visible']


def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ps = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return abs(np.mean(a) - np.mean(b)) / ps if ps > 1e-10 else 0.0


# ── Select top-N sessions by Cohen's d (Cue 1 vs Cue 2, cue-zone only) ───────
session_stats = []
for s_idx in range(len(sessions)):
    if r2_all[s_idx, E_IDX] < 0.01:
        continue
    sd   = ds[sessions[s_idx]]
    X    = np.concatenate([sd['data'][t]              for t in sd['data']], axis=0)
    y    = np.concatenate([sd['labels'][t][:, E_IDX]  for t in sd['data']])
    cond = np.argmax(X[:, cue_cols], axis=1)

    # Filter to cue zone only
    cue_mask = cond > 0
    if cue_mask.sum() < 40:
        continue
    X_cz   = X[cue_mask]
    y_cz   = y[cue_mask]
    cond_cz = cond[cue_mask]

    if np.unique(cond_cz).size < 2:
        continue

    y1 = y_cz[cond_cz == 1]
    y2 = y_cz[cond_cz == 2]
    cd = cohen_d(y1, y2)
    session_stats.append(dict(s=s_idx, cd=cd, X_cz=X_cz, y_cz=y_cz, cond_cz=cond_cz))

session_stats.sort(key=lambda x: -x['cd'])
selected = session_stats[:N_PANELS]
print('Top sessions (Cue 1 vs Cue 2, cue zone only):')
for r in selected:
    print(f'  S{r["s"]+1:02d}  d={r["cd"]:.3f}  n={len(r["y_cz"])}')

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, N_PANELS, figsize=FIG.FULL, sharey=False)
apply_style(fig, list(axes))
PANEL_LABELS = ['A', 'B', 'C', 'D']

for col, (r, ax) in enumerate(zip(selected, axes)):
    pos_cz   = r['X_cz'][:, pos_cols[0]]
    y_cz     = r['y_cz']
    cond_cz  = r['cond_cz']

    # Position bins within the cue zone
    edges   = np.linspace(np.nanpercentile(pos_cz, 2),
                          np.nanpercentile(pos_cz, 98), N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    for c in [1, 2]:
        mask  = cond_cz == c
        means = np.full(N_BINS, np.nan)
        sems  = np.full(N_BINS, np.nan)
        for b in range(N_BINS):
            pts = y_cz[mask & (pos_cz >= edges[b]) & (pos_cz < edges[b + 1])]
            if len(pts) >= MIN_PTS:
                means[b] = np.mean(pts)
                sems[b]  = np.std(pts) / np.sqrt(len(pts))

        ok  = ~np.isnan(means)
        clr = CUE_COLORS[c]
        ax.fill_between(centers[ok], means[ok] - sems[ok], means[ok] + sems[ok],
                        color=clr, alpha=0.20)
        ax.plot(centers[ok], means[ok], color=clr, lw=1.8,
                label=CUE_LABELS[c])

    ax.axhline(0, color='#888', lw=0.5, ls=':')
    ax.set_xlabel('Position in cue zone (z)', fontsize=FONT.LABEL - 2)
    if col == 0:
        ax.set_ylabel('z-scored activity', fontsize=FONT.LABEL - 2)
    ax.set_title(f'S{r["s"]+1:02d}  d={r["cd"]:.2f}', fontsize=FONT.LABEL - 1, pad=3)
    ax.tick_params(labelsize=FONT.TICK - 2)
    if col == 0:
        ax.legend(fontsize=FONT.LEGEND - 1, frameon=False, loc='upper left')
    add_panel_label(ax, PANEL_LABELS[col])

add_footnote(fig,
    'E07 (ensemble 6).  Only timepoints where cue_visible in {Cue 1, Cue 2} (cue zone).  '
    'Position bins restricted to the cue-zone track segment.  '
    'Sessions selected by max Cohen\'s d (Cue 1 vs Cue 2).')

savefig_manifest(fig, 'e07_cue_zone_tuning.png', OUT_DIRS)
print('Saved e07_cue_zone_tuning.png')
