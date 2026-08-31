#!/usr/bin/env python3
"""
eval_e07_position_cue_tuning.py

E07 position tuning averaged across all valid sessions, split by cue condition.
Shows that the yellow peak (high activity at cue zone) is cue-condition-specific.

Single panel: mean ± SEM activity vs track position, one line per cue condition.
Averaged across all 19 valid sessions (R²≥0.01).

Output: outputs/ablation_vs_attribution/e07_position_cue_tuning.png
"""
import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import (
    FIG, DPI, FONT,
    apply_style, add_footnote, savefig_manifest,
)

root   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
mdir   = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [
    os.path.join(root, 'outputs', 'ablation_vs_attribution'),
    '/mnt/c/Users/amits/Desktop',
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

E_IDX     = 6
R2_THRESH = 0.01
N_BINS    = 30
MIN_PTS   = 8

CUE_COLORS = {0: '#AAAAAA', 1: '#FF7F0E', 2: '#D62728'}
CUE_LABELS = {0: 'No cue', 1: 'Cue 1', 2: 'Cue 2'}

# ── Load ──────────────────────────────────────────────────────────────────────
ds       = pickle.load(open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb'))
sessions = list(ds.keys())
sg       = pickle.load(open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb'))
group_names = [g[0] for g in sg]
feat_idx    = {g: cols for g, cols in sg}
r2_all      = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)

cue_cols = feat_idx['cue_visible']
pos_cols = feat_idx['frame_position']

valid = [s for s in range(len(sessions)) if r2_all[s, E_IDX] >= R2_THRESH]
print(f'E07: {len(valid)} valid sessions')

# ── Per-session, per-condition tuning curves ──────────────────────────────────
# Use a shared position grid across all sessions (z-scored, so comparable)
all_curves = {c: [] for c in [0, 1, 2]}  # cond → list of (means array)

# Global position edges from all sessions pooled
all_pos_data = []
for s in valid:
    sd = ds[sessions[s]]
    X  = np.concatenate([sd['data'][t] for t in sd['data']], axis=0)
    all_pos_data.append(X[:, pos_cols[0]])
all_pos = np.concatenate(all_pos_data)
edges   = np.linspace(np.nanpercentile(all_pos, 2),
                      np.nanpercentile(all_pos, 98), N_BINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])

for s in valid:
    sd   = ds[sessions[s]]
    X    = np.concatenate([sd['data'][t]              for t in sd['data']], axis=0)
    y    = np.concatenate([sd['labels'][t][:, E_IDX]  for t in sd['data']])
    pos  = X[:, pos_cols[0]]
    cond = np.argmax(X[:, cue_cols], axis=1)

    for c in [0, 1, 2]:
        mask = cond == c
        if mask.sum() < 20:
            continue
        means = np.full(N_BINS, np.nan)
        for b in range(N_BINS):
            pts = y[mask & (pos >= edges[b]) & (pos < edges[b+1])]
            if len(pts) >= MIN_PTS:
                means[b] = np.mean(pts)
        all_curves[c].append(means)

# ── Aggregate: mean ± SEM across sessions ─────────────────────────────────────
fig, ax = plt.subplots(figsize=FIG.FULL)
apply_style(fig, ax)

for c in [0, 1, 2]:
    mats = np.array(all_curves[c])         # (n_sessions, N_BINS)
    if len(mats) == 0:
        continue
    # Per-bin mean and SEM across sessions, ignoring NaN
    m    = np.nanmean(mats, axis=0)
    sem  = np.nanstd(mats, axis=0) / np.sqrt(np.sum(~np.isnan(mats), axis=0).clip(1))
    ok   = ~np.isnan(m)
    clr  = CUE_COLORS[c]
    n_s  = len(mats)

    ax.fill_between(centers[ok], m[ok] - sem[ok], m[ok] + sem[ok],
                    color=clr, alpha=0.18)
    ax.plot(centers[ok], m[ok], color=clr, lw=2.0,
            label=f'{CUE_LABELS[c]}  (n={n_s})')

ax.axhline(0, color='#888', lw=0.6, ls=':')
ax.set_xlabel('Track position (z-scored)', fontsize=FONT.LABEL)
ax.set_ylabel('z-scored ensemble activity', fontsize=FONT.LABEL)
ax.set_title('E07 — position tuning by cue condition', fontsize=FONT.LABEL)
ax.legend(fontsize=FONT.LEGEND, frameon=False, loc='upper right')
ax.tick_params(labelsize=FONT.TICK)

add_footnote(fig,
    f'E07 (ensemble 6).  Mean ± SEM across {len(valid)} sessions (R²≥{R2_THRESH}).  '
    f'{N_BINS} position bins.  '
    'Peak activity at cue zone position is specific to cue conditions 1 and 2.')

savefig_manifest(fig, 'e07_position_cue_tuning.png', OUT_DIRS)
print('Saved e07_position_cue_tuning.png')
