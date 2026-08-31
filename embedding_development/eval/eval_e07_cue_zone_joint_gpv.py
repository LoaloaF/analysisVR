#!/usr/bin/env python3
"""
eval_e07_cue_zone_joint_gpv.py

Story: in the cue zone (cue_visible != 0), does the cue identity actually
modulate position tuning?  Sessions where it does should also show high
joint GPV — the model should capture this cue × position interaction.

Panel A: Mean position tuning curves per cue condition (Cue 1 vs Cue 2),
         averaged across valid sessions.  Shows the phenomenon.

Panel B: Scatter of pos-tuning cue-difference vs joint GPV per session.
         Shows the model captures the interaction.

Output: outputs/ablation_vs_attribution/e07_cue_zone_joint_gpv.png
"""
import os, sys, pickle
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.models import MLP
from utils.figure_style import (
    FIG, DPI, FONT, MODEL_COLORS,
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

SEED      = 42
N_PERMS   = 10
E_IDX     = 6
R2_THRESH = 0.01
MIN_CZ    = 30
N_POS_BINS = 8
device    = torch.device('cpu')

# ── Load ──────────────────────────────────────────────────────────────────────
ds       = pickle.load(open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb'))
sessions = list(ds.keys())
sg       = pickle.load(open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb'))
feat_idx = {g: cols for g, cols in sg}
r2_all   = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)
tidx_map = np.load(os.path.join(root, 'splits', f'split_seed{SEED}.npy'),
                   allow_pickle=True).item()

CUE_COLS   = feat_idx['cue_visible']
POS_COLS   = feat_idx['frame_position']
JOINT_COLS = CUE_COLS + POS_COLS

valid_sessions = [s for s in range(len(sessions)) if r2_all[s, E_IDX] >= R2_THRESH]
print(f'E07: {len(valid_sessions)} valid sessions (R²≥{R2_THRESH})')


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_mlp(s):
    path = os.path.join(root, 'models', 'mlps', 'ensembles',
                        f'seed{SEED}', f'session_{s:02d}_neuron_{E_IDX:02d}.pt')
    if not os.path.exists(path):
        return None
    sd = torch.load(path, map_location=device)
    state = sd['model_state_dict'] if isinstance(sd, dict) and 'model_state_dict' in sd else sd
    h, nin = state['fc.0.weight'].shape
    m = MLP(nin, h, 2, 1).to(device)
    m.load_state_dict(state)
    m.eval()
    return m


def predict(mlp, X):
    with torch.no_grad():
        out = mlp(torch.tensor(X, dtype=torch.float32))
        pred = out[1] if isinstance(out, tuple) else out
        return pred.cpu().numpy().ravel()


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0


def gpv_permuted(mlp, X, y, cols, rng, n_perms):
    r2s = []
    for _ in range(n_perms):
        X_p = X.copy()
        idx = rng.permutation(len(X_p))
        X_p[:, cols] = X_p[idx][:, cols]
        r2s.append(r2(y, predict(mlp, X_p)))
    return float(np.mean(r2s))


def pos_tuning_diff(Xte_cz, yte_cz, cue_id):
    """Mean |tuning_cue1 - tuning_cue2| over shared position bins."""
    if len(POS_COLS) > 1:
        # one-hot encoded position bins
        pos_bin = np.argmax(Xte_cz[:, POS_COLS], axis=1)
        n_bins  = len(POS_COLS)
    else:
        # continuous position — bin manually
        pv = Xte_cz[:, POS_COLS[0]]
        edges  = np.linspace(np.nanmin(pv), np.nanmax(pv), N_POS_BINS + 1)
        pos_bin = np.digitize(pv, edges[1:-1])
        n_bins  = N_POS_BINS

    means1 = np.array([np.nanmean(yte_cz[(pos_bin == b) & (cue_id == 1)])
                       for b in range(n_bins)])
    means2 = np.array([np.nanmean(yte_cz[(pos_bin == b) & (cue_id == 2)])
                       for b in range(n_bins)])
    valid = np.isfinite(means1) & np.isfinite(means2)
    if not valid.any():
        return np.nan, means1, means2, n_bins
    return float(np.nanmean(np.abs(means1[valid] - means2[valid]))), means1, means2, n_bins


# ── Per-session computation ────────────────────────────────────────────────────
rng  = np.random.default_rng(0)
rows = []

# Collect tuning curves for Panel A (averaged across sessions)
all_means1 = []
all_means2 = []
n_bins_global = None

for s_idx in valid_sessions:
    sess_id = sessions[s_idx]
    sd      = ds[sess_id]
    test_t  = [t for t in tidx_map.get(sess_id, []) if t in sd['data']]
    if not test_t:
        continue

    Xte = np.concatenate([sd['data'][t]             for t in test_t]).astype(float)
    yte = np.concatenate([sd['labels'][t][:, E_IDX] for t in test_t]).astype(float)

    cond = np.argmax(Xte[:, CUE_COLS], axis=1)
    cz   = cond > 0
    if cz.sum() < MIN_CZ:
        continue

    Xte_cz = Xte[cz]
    yte_cz = yte[cz]
    cue_id = cond[cz]

    # Skip sessions with only one cue condition
    if len(np.unique(cue_id)) < 2:
        continue

    mlp = load_mlp(s_idx)
    if mlp is None:
        continue

    r2_base  = r2(yte_cz, predict(mlp, Xte_cz))
    r2_joint = gpv_permuted(mlp, Xte_cz, yte_cz, JOINT_COLS, rng, N_PERMS)
    gpv_j    = r2_base - r2_joint

    diff, m1, m2, n_bins = pos_tuning_diff(Xte_cz, yte_cz, cue_id)
    if n_bins_global is None:
        n_bins_global = n_bins

    rows.append(dict(
        s=s_idx, r2_base=r2_base, n_cz=int(cz.sum()),
        gpv_joint=gpv_j, pos_cue_diff=diff,
    ))
    all_means1.append(m1)
    all_means2.append(m2)

    print(f'  S{s_idx+1:02d}  R²={r2_base:.3f}  GPV(joint)={gpv_j:.4f}  '
          f'pos_diff={diff:.4f}  n_cz={cz.sum()}')

if not rows:
    print('No results'); raise SystemExit(1)

n_bins_global = n_bins_global or N_POS_BINS

print(f'\n{len(rows)} sessions computed')
print(f'  Mean GPV(joint) = {np.nanmean([r["gpv_joint"] for r in rows]):.4f}')
print(f'  Mean pos_cue_diff = {np.nanmean([r["pos_cue_diff"] for r in rows]):.4f}')

# Average position tuning curves across sessions (nan-safe)
avg_means1 = np.nanmean([m for m in all_means1 if len(m) == n_bins_global], axis=0)
avg_means2 = np.nanmean([m for m in all_means2 if len(m) == n_bins_global], axis=0)

# ── Figure ────────────────────────────────────────────────────────────────────
C_CUE1 = '#FF7F0E'
C_CUE2 = '#1F77B4'
C_JOINT = '#9467BD'

fig, (ax_tun, ax_scat) = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig, [ax_tun, ax_scat])

# Panel A: position tuning curves per cue condition (averaged across sessions)
x_bins = np.arange(n_bins_global)
ax_tun.plot(x_bins, avg_means1, '-o', color=C_CUE1, lw=1.8, markersize=5,
            label='Cue 1')
ax_tun.plot(x_bins, avg_means2, '-o', color=C_CUE2, lw=1.8, markersize=5,
            label='Cue 2')
ax_tun.fill_between(x_bins, avg_means1, avg_means2, alpha=0.15, color='#888')
ax_tun.set_xlabel('Position bin', fontsize=FONT.LABEL)
ax_tun.set_ylabel('Mean activity (z-sc.)', fontsize=FONT.LABEL)
ax_tun.set_xticks(x_bins)
ax_tun.legend(fontsize=FONT.LEGEND, frameon=False)
ax_tun.text(0.03, 0.97, 'A', transform=ax_tun.transAxes,
            fontsize=FONT.PANEL, fontweight='bold', va='top', ha='left')

# Panel B: pos_cue_diff vs joint GPV per session
pos_diffs  = np.array([r['pos_cue_diff'] for r in rows])
joint_vals = np.array([r['gpv_joint']    for r in rows])
s_idxs     = [r['s'] for r in rows]

valid = np.isfinite(pos_diffs) & np.isfinite(joint_vals)
ax_scat.scatter(pos_diffs[valid], joint_vals[valid],
                c=C_JOINT, s=55, alpha=0.85, zorder=3,
                edgecolors='white', linewidths=0.4)
for pd_, jv, si in zip(pos_diffs[valid], joint_vals[valid],
                        np.array(s_idxs)[valid]):
    ax_scat.annotate(f'S{si+1:02d}', (pd_, jv), fontsize=5.5,
                     xytext=(3, 3), textcoords='offset points', color='#444')

# Trend line
if valid.sum() >= 3:
    m, b = np.polyfit(pos_diffs[valid], joint_vals[valid], 1)
    xf = np.linspace(pos_diffs[valid].min(), pos_diffs[valid].max(), 50)
    ax_scat.plot(xf, m * xf + b, '--', color='#555', lw=1.0, alpha=0.7)

ax_scat.set_xlabel('Pos. tuning diff. (|Cue1 - Cue2|)', fontsize=FONT.LABEL)
ax_scat.set_ylabel('Joint GPV (ΔR²)', fontsize=FONT.LABEL)
ax_scat.text(0.03, 0.97, 'B', transform=ax_scat.transAxes,
             fontsize=FONT.PANEL, fontweight='bold', va='top', ha='left')

add_footnote(fig,
    f'E07 (E06), {len(rows)} sessions; cue-zone only; '
    f'joint GPV: n_perms={N_PERMS}; pos_diff: mean |Cue1-Cue2| across {n_bins_global} bins')

savefig_manifest(fig, 'e07_cue_zone_joint_gpv.png', OUT_DIRS)
print('\nSaved e07_cue_zone_joint_gpv.png')
