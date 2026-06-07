#!/usr/bin/env python3
"""
eval_e07_assembly_heatmap.py

Reproduces the supervisor's Assembly007 × position heatmap (Cue 2 trials),
adding the MLP model prediction alongside to show the model captures the signal.

Layout: 2-column figure per session
  Left:  actual activity binned by position, one row per trial
  Right: MLP-predicted activity, same layout

Picks the 2 sessions with highest Cohen's d for cue_visible (E07).
"""
import os, sys, pickle
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

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

SEED      = 42
E_IDX     = 6       # E07
N_SESSIONS = 2      # show top N by Cohen's d
CUE_TARGET  = 2     # show trials where cue 2 was present
MIN_TRIALS  = 5
N_POS_BINS  = 40    # finer position resolution (was 25)
device    = torch.device('cpu')

# ── Load ──────────────────────────────────────────────────────────────────────
ds       = pickle.load(open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb'))
sessions = list(ds.keys())
sg       = pickle.load(open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb'))
group_names = [g[0] for g in sg]
feat_idx    = {g: cols for g, cols in sg}
r2_all      = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)

cue_cols = feat_idx['cue_visible']
pos_cols = feat_idx['frame_position']   # single column

# ── Select sessions ───────────────────────────────────────────────────────────
def cd_cat(X_oh, y):
    cond = np.argmax(X_oh, axis=1); best = 0.0
    for i in range(X_oh.shape[1]):
        for j in range(i+1, X_oh.shape[1]):
            a, b = y[cond==i], y[cond==j]
            if len(a)<2 or len(b)<2: continue
            ps = np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2)
            if ps>1e-10: best = max(best, abs(np.mean(a)-np.mean(b))/ps)
    return best

session_scores = []
for s in range(len(sessions)):
    if r2_all[s, E_IDX] < 0.01: continue
    sd = ds[sessions[s]]
    X  = np.concatenate([sd['data'][t] for t in sd['data']], axis=0)
    y  = np.concatenate([sd['labels'][t][:, E_IDX] for t in sd['data']])
    cond = np.argmax(X[:, cue_cols], axis=1)
    if CUE_TARGET not in np.unique(cond): continue
    cd = cd_cat(X[:, cue_cols], y)
    session_scores.append((s, cd))

session_scores.sort(key=lambda x: -x[1])
selected = [s for s, _ in session_scores[:N_SESSIONS]]
print(f'Selected: {[f"S{s+1:02d} (d={d:.3f})" for s,d in session_scores[:N_SESSIONS]]}')


# ── Load MLP ──────────────────────────────────────────────────────────────────
def load_mlp(s):
    path = os.path.join(root, 'models', 'mlps', 'ensembles',
                        f'seed{SEED}', f'session_{s:02d}_neuron_{E_IDX:02d}.pt')
    if not os.path.exists(path): return None
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        w = sd['model_state_dict']['fc.0.weight']
        h, nin = w.shape
        m = MLP(nin, h, 2, 1).to(device)
        m.load_state_dict(sd['model_state_dict'])
    else:
        w = sd['fc.0.weight']; h, nin = w.shape
        m = MLP(nin, h, 2, 1).to(device); m.load_state_dict(sd)
    m.eval(); return m


def predict(mlp, X):
    with torch.no_grad():
        out = mlp(torch.tensor(X.astype(np.float32)))
        pred = out[1] if isinstance(out, tuple) else out
    return pred.cpu().numpy().ravel()


# ── Per-trial position heatmap ────────────────────────────────────────────────
def make_heatmap(trials_data, pos_edges):
    """trials_data: list of (pos_array, act_array) per trial. Returns (n_trials, n_bins) matrix."""
    n_bins = len(pos_edges) - 1
    mat = np.full((len(trials_data), n_bins), np.nan)
    for ti, (pos, act) in enumerate(trials_data):
        for b in range(n_bins):
            mask = (pos >= pos_edges[b]) & (pos < pos_edges[b+1])
            if mask.sum() >= 2:
                mat[ti, b] = np.mean(act[mask])
    return mat


# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(N_SESSIONS, 2,
                          figsize=(9.50, 4.20),
                          gridspec_kw={'wspace': 0.10, 'hspace': 0.45})
if N_SESSIONS == 1:
    axes = axes[np.newaxis, :]
apply_style(fig, axes.ravel())
fig.subplots_adjust(bottom=0.14)

PANEL_LABELS = [['A', 'B'], ['C', 'D']]

for row, s_idx in enumerate(selected):
    sd    = ds[sessions[s_idx]]
    all_t = list(sd['data'].keys())
    mlp   = load_mlp(s_idx)

    # Use FULL trial trajectories for rows where cue 2 was present at any point.
    # This gives complete position coverage per row (matches supervisor's figure).
    cue2_trials = []
    for t in sorted(all_t):
        X_t = sd['data'][t].astype(float)
        y_t = sd['labels'][t][:, E_IDX].astype(float)
        cond_t = np.argmax(X_t[:, cue_cols], axis=1)
        if not np.any(cond_t == CUE_TARGET):
            continue   # skip trials that never show cue 2
        # Use ALL timepoints from this trial (full track traversal)
        pos_t  = X_t[:, pos_cols[0]]
        act_t  = y_t
        pred_t = predict(mlp, X_t) if mlp is not None else np.full(len(y_t), np.nan)
        cue2_trials.append((pos_t, act_t, pred_t))

    if len(cue2_trials) < MIN_TRIALS:
        print(f'  S{s_idx+1:02d}: only {len(cue2_trials)} Cue-2 trials, skipping')
        continue

    print(f'  S{s_idx+1:02d}: {len(cue2_trials)} Cue-2 trials')

    # Global position edges from all Cue-2 data
    all_pos = np.concatenate([p for p, _, _ in cue2_trials])
    pos_edges = np.linspace(np.nanpercentile(all_pos, 2),
                            np.nanpercentile(all_pos, 98), N_POS_BINS + 1)
    bin_centers = 0.5 * (pos_edges[:-1] + pos_edges[1:])

    act_mat  = make_heatmap([(p, a) for p, a, _ in cue2_trials], pos_edges)
    pred_mat = make_heatmap([(p, pr) for p, _, pr in cue2_trials], pos_edges)

    # Clip negatives to 0 (match supervisor's non-negative projection scale)
    act_mat  = np.clip(act_mat,  0, None)
    pred_mat = np.clip(pred_mat, 0, None)

    vmin = 0
    vmax = np.nanpercentile(act_mat, 97)

    for col, (mat, title) in enumerate([
            (act_mat,  f'Actual — S{s_idx+1:02d}'),
            (pred_mat, f'MLP prediction — S{s_idx+1:02d}'),
    ]):
        ax = axes[row, col]
        im = ax.imshow(mat, aspect='auto', origin='upper',
                       cmap='viridis', vmin=vmin, vmax=vmax,
                       extent=[bin_centers[0], bin_centers[-1], len(cue2_trials), 0],
                       interpolation='nearest')
        ax.set_xlabel('Position (z-scored)', fontsize=FONT.LABEL - 2)
        if col == 0:
            ax.set_ylabel('Trial index', fontsize=FONT.LABEL - 2)
        ax.set_title(title, fontsize=FONT.LABEL - 1, pad=2)
        ax.tick_params(labelsize=FONT.TICK - 2)
        add_panel_label(ax, PANEL_LABELS[row][col])

        # Colorbar on right panel only
        if col == 1:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label('z-scored activity', fontsize=FONT.TICK - 1)
            cb.ax.tick_params(labelsize=FONT.TICK - 2)

add_footnote(fig,
    f'E07 (ensemble 6), Cue {CUE_TARGET} trials only.  '
    'Left: actual z-scored ensemble activity binned by track position per trial.  '
    'Right: MLP predicted activity, same layout.  '
    'Yellow peak at cue zone position should appear in both panels.')

savefig_manifest(fig, 'e07_assembly_heatmap.png', OUT_DIRS)
print('Saved e07_assembly_heatmap.png')
