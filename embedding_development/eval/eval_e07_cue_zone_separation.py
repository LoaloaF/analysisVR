#!/usr/bin/env python3
"""
eval_e07_cue_zone_separation.py  —  S69

Best-session deep dive: S20 (d=0.21, highest Cue1/Cue2 discriminability in E07).
Shows both actual and MLP-predicted activity in the cue zone, split by cue condition.

Panel A: actual ensemble activity binned by position in cue zone, Cue 1 vs Cue 2.
Panel B: MLP predicted activity — same layout.
         If the model captures the difference, predictions should mirror actuals.

Output: outputs/ablation_vs_attribution/e07_cue_zone_separation.png
"""
import os, sys, pickle
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.models import MLP
from utils.figure_style import (
    FIG, DPI, FONT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

root   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
mdir   = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [
    os.path.join(root, 'outputs', 'ablation_vs_attribution'),
    '/mnt/c/Users/amits/Desktop',
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

SEED   = 42
E_IDX  = 6
N_BINS = 20
MIN_PTS = 5

C1 = '#FF7F0E'   # Cue 1
C2 = '#D62728'   # Cue 2
device = torch.device('cpu')

ds       = pickle.load(open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb'))
sessions = list(ds.keys())
sg       = pickle.load(open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb'))
feat_idx = {g: cols for g, cols in sg}
r2_all   = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)

cue_cols = feat_idx['cue_visible']
pos_cols = feat_idx['frame_position']


def cohen_d_cz(s_idx):
    sd   = ds[sessions[s_idx]]
    X    = np.concatenate([sd['data'][t]             for t in sd['data']], axis=0)
    y    = np.concatenate([sd['labels'][t][:, E_IDX] for t in sd['data']])
    cond = np.argmax(X[:, cue_cols], axis=1)
    cz   = cond > 0
    if cz.sum() < 30:
        return 0.0
    y1 = y[cz & (cond == 1)]; y2 = y[cz & (cond == 2)]
    if len(y1) < 2 or len(y2) < 2:
        return 0.0
    ps = np.sqrt((np.var(y1, ddof=1) + np.var(y2, ddof=1)) / 2)
    return abs(np.mean(y1) - np.mean(y2)) / ps if ps > 1e-10 else 0.0

valid = [s for s in range(len(sessions)) if r2_all[s, E_IDX] >= 0.01]
best_s = max(valid, key=cohen_d_cz)
best_d = cohen_d_cz(best_s)
print(f'Best session: S{best_s+1:02d}  d={best_d:.3f}')

# ── Load all data for best session ────────────────────────────────────────────
sd   = ds[sessions[best_s]]
X    = np.concatenate([sd['data'][t]             for t in sd['data']], axis=0).astype(float)
y    = np.concatenate([sd['labels'][t][:, E_IDX] for t in sd['data']]).astype(float)
cond = np.argmax(X[:, cue_cols], axis=1)
pos  = X[:, pos_cols[0]]

cz      = cond > 0
X_cz    = X[cz]; y_cz = y[cz]; cond_cz = cond[cz]; pos_cz = pos[cz]

# ── Load MLP and predict ──────────────────────────────────────────────────────
path = os.path.join(root, 'models', 'mlps', 'ensembles',
                    f'seed{SEED}', f'session_{best_s:02d}_neuron_{E_IDX:02d}.pt')
sd_ck = torch.load(path, map_location=device)
state = sd_ck['model_state_dict'] if isinstance(sd_ck, dict) and 'model_state_dict' in sd_ck else sd_ck
h, nin = state['fc.0.weight'].shape
mlp = MLP(nin, h, 2, 1).to(device); mlp.load_state_dict(state); mlp.eval()

with torch.no_grad():
    out   = mlp(torch.tensor(X_cz, dtype=torch.float32))
    y_hat = (out[1] if isinstance(out, tuple) else out).cpu().numpy().ravel()

# ── Position bins (cue zone only) ─────────────────────────────────────────────
edges   = np.linspace(np.nanpercentile(pos_cz, 2),
                      np.nanpercentile(pos_cz, 98), N_BINS + 1)
centers = 0.5 * (edges[:-1] + edges[1:])


def bin_by_pos(vals, cond_arr, pos_arr, c):
    mask = cond_arr == c
    m = np.full(N_BINS, np.nan); sem = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        pts = vals[mask & (pos_arr >= edges[b]) & (pos_arr < edges[b + 1])]
        if len(pts) >= MIN_PTS:
            m[b] = np.mean(pts); sem[b] = np.std(pts) / np.sqrt(len(pts))
    return m, sem


# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig, [ax_l, ax_r])

for ax, vals, panel, ttl in [
        (ax_l, y_cz,   'A', 'Actual activity'),
        (ax_r, y_hat,  'B', 'MLP predicted'),
]:
    for c, clr, lbl in [(1, C1, 'Cue 1'), (2, C2, 'Cue 2')]:
        m, sem = bin_by_pos(vals, cond_cz, pos_cz, c)
        ok = ~np.isnan(m)
        ax.fill_between(centers[ok], m[ok] - sem[ok], m[ok] + sem[ok],
                        color=clr, alpha=0.22)
        ax.plot(centers[ok], m[ok], color=clr, lw=2.0, label=lbl)

    ax.axhline(0, color='#888', lw=0.5, ls=':')
    ax.set_xlabel('Position in cue zone (z-scored)', fontsize=FONT.LABEL - 1)
    ax.set_ylabel('z-scored activity', fontsize=FONT.LABEL - 1)
    ax.set_title(f'S{best_s+1:02d} E07 — {ttl}\n(Cue 1 vs Cue 2, cue zone, d={best_d:.2f})',
                 fontsize=FONT.LABEL - 1, pad=3)
    ax.legend(fontsize=FONT.LEGEND, frameon=False)
    ax.tick_params(labelsize=FONT.TICK)
    add_panel_label(ax, panel)

add_footnote(fig,
    f'Session S{best_s+1:02d} (highest Cohen\'s d={best_d:.2f} for Cue 1 vs Cue 2 in E07).  '
    f'Cue-zone timepoints only.  {N_BINS} position bins, min {MIN_PTS} pts/bin.  '
    'MLP predictions from same checkpoint (seed 42).')

savefig_manifest(fig, 'e07_cue_zone_separation.png', OUT_DIRS)
print('Saved e07_cue_zone_separation.png')
