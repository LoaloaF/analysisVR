#!/usr/bin/env python3
"""
eval_e07_joint_tuning.py

E07 (ensemble 6): joint position × cue tuning.
Shows position-binned activity split by cue condition for the 4 sessions with
highest Cohen's d for cue_visible. Illustrates why marginal GPV misses joint encoding.

Output: outputs/ablation_vs_attribution/e07_joint_tuning.png
"""
import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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

# ── Load ──────────────────────────────────────────────────────────────────────
ds       = pickle.load(open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb'))
sessions = list(ds.keys())
sg       = pickle.load(open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb'))
group_names = [g[0] for g in sg]
feat_idx    = {g: cols for g, cols in sg}
gpv    = np.load(os.path.join(mdir, 'importance_global_pv_semantic.npy'))
r2_all = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)

E_IDX    = 6       # ensemble 07
N_PANELS = 4       # sessions to show
N_BINS   = 20
MIN_PTS  = 15

pos_cols = feat_idx['frame_position']   # single column [6]
cue_cols = feat_idx['cue_visible']      # one-hot [7, 8, 9]

# Cue condition colours: 0=no cue, 1=cue A, 2=cue B
CUE_COLORS = {0: '#AAAAAA', 1: '#FF7F0E', 2: '#D62728'}
CUE_LABELS = {0: 'No cue', 1: 'Cue 1', 2: 'Cue 2'}

# ── Select top-N sessions by Cohen's d for cue_visible ────────────────────────
def cd_categorical(X_oh, y):
    cond = np.argmax(X_oh, axis=1); best = 0.0
    for i in range(X_oh.shape[1]):
        for j in range(i + 1, X_oh.shape[1]):
            a, b = y[cond == i], y[cond == j]
            if len(a) < 2 or len(b) < 2: continue
            ps = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            if ps > 1e-10: best = max(best, abs(np.mean(a) - np.mean(b)) / ps)
    return best

session_stats = []
for s_idx in range(len(sessions)):
    if r2_all[s_idx, E_IDX] < 0.01: continue
    sd  = ds[sessions[s_idx]]
    X   = np.concatenate([sd['data'][t]   for t in sd['data']], axis=0)
    y   = np.concatenate([sd['labels'][t][:, E_IDX] for t in sd['data']])
    cond = np.argmax(X[:, cue_cols], axis=1)
    if len(np.unique(cond)) < 2: continue
    cd  = cd_categorical(X[:, cue_cols], y)
    gv_cue = float(gpv[s_idx, E_IDX, group_names.index('cue_visible')])
    gv_pos = float(gpv[s_idx, E_IDX, group_names.index('frame_position')])
    session_stats.append(dict(s=s_idx, cd=cd, gv_cue=gv_cue, gv_pos=gv_pos, X=X, y=y))

session_stats.sort(key=lambda x: -x['cd'])
selected = session_stats[:N_PANELS]
print('Selected sessions (by Cohen\'s d for cue):')
for r in selected:
    print(f'  S{r["s"]+1:02d}  d={r["cd"]:.3f}  GPV(cue)={r["gv_cue"]:.4f}  GPV(pos)={r["gv_pos"]:.4f}')

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, N_PANELS, figsize=FIG.FULL)
apply_style(fig, list(axes))

PANEL_LABELS = ['A', 'B', 'C', 'D']

for col, (r, ax) in enumerate(zip(selected, axes)):
    X = r['X']; y = r['y']
    pos  = X[:, pos_cols[0]]
    cond = np.argmax(X[:, cue_cols], axis=1)
    unique_conds = sorted(np.unique(cond))

    edges   = np.linspace(np.nanpercentile(pos, 2), np.nanpercentile(pos, 98), N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    for c in unique_conds:
        mask   = cond == c
        means  = []
        sems   = []
        valid  = []
        for b in range(N_BINS):
            pts = y[mask & (pos >= edges[b]) & (pos < edges[b + 1])]
            if len(pts) >= MIN_PTS:
                means.append(np.mean(pts))
                sems.append(np.std(pts) / np.sqrt(len(pts)))
                valid.append(b)
            else:
                means.append(np.nan)
                sems.append(np.nan)
                valid.append(b)

        means = np.array(means); sems = np.array(sems)
        ok    = ~np.isnan(means)
        clr   = CUE_COLORS[c]
        ax.fill_between(centers[ok], means[ok] - sems[ok], means[ok] + sems[ok],
                        color=clr, alpha=0.18)
        ax.plot(centers[ok], means[ok], color=clr, lw=1.6,
                label=CUE_LABELS[c])

    ax.axhline(0, color='#888', lw=0.5, ls=':')
    ax.set_xlabel('Track position (z)', fontsize=FONT.LABEL - 2)
    if col == 0:
        ax.set_ylabel('z-scored activity', fontsize=FONT.LABEL - 2)
    ax.set_title(f'S{r["s"]+1:02d}  d={r["cd"]:.2f}', fontsize=FONT.LABEL - 1, pad=3)
    ax.tick_params(labelsize=FONT.TICK - 2)
    if col == 0:
        ax.legend(fontsize=FONT.LEGEND - 1, frameon=False, loc='upper left')
    add_panel_label(ax, PANEL_LABELS[col])

add_footnote(fig,
    'E07 (ensemble 6): position-binned activity split by cue condition (3-way one-hot).  '
    'Marginal GPV permutes cue OR position separately — misses interactions between them.  '
    'Grouped GPV (permute cue + position jointly) would capture the full joint contribution.')

savefig_manifest(fig, 'e07_joint_tuning.png', OUT_DIRS)
print('Saved e07_joint_tuning.png')
