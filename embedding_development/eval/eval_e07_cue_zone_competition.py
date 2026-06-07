#!/usr/bin/env python3
"""
eval_e07_cue_zone_competition.py  —  S70

Shows why the joint cue+position GPV method (S69) still under-attributes:
speed and head angle already have LARGER GPV in the cue zone than
cue+position jointly, so the model has routed the signal through them.

Single figure: one bar per permutation group —
  Cue (marginal), Position (marginal), Cue+Pos (joint),
  Speed, Head Angle, Rot. Vel.
A reference line at GPV(cue+pos joint) makes the competition explicit.

Output: outputs/ablation_vs_attribution/e07_cue_zone_competition.png
"""
import os, sys, pickle
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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
N_PERMS   = 10
E_IDX     = 6
R2_THRESH = 0.01
MIN_CZ    = 30
device    = torch.device('cpu')

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
SPD_COLS   = feat_idx['frame_raw_500msMedian']
HA_COLS    = feat_idx['head_angle']
RV_COLS    = feat_idx['frame_YawPitch_abs_vel_sum_500msMedian']

valid = [s for s in range(len(sessions)) if r2_all[s, E_IDX] >= R2_THRESH]


def load_mlp(s):
    path = os.path.join(root, 'models', 'mlps', 'ensembles',
                        f'seed{SEED}', f'session_{s:02d}_neuron_{E_IDX:02d}.pt')
    if not os.path.exists(path):
        return None
    sd    = torch.load(path, map_location=device)
    state = sd['model_state_dict'] if isinstance(sd, dict) and 'model_state_dict' in sd else sd
    h, nin = state['fc.0.weight'].shape
    m = MLP(nin, h, 2, 1).to(device); m.load_state_dict(state); m.eval()
    return m


def predict(mlp, X):
    with torch.no_grad():
        out = mlp(torch.tensor(X, dtype=torch.float32))
        return (out[1] if isinstance(out, tuple) else out).cpu().numpy().ravel()


def r2(yt, yp):
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0


def gpv_mean(mlp, X, y, cols, rng, n=N_PERMS):
    r2s = []
    for _ in range(n):
        Xp = X.copy(); idx = rng.permutation(len(Xp))
        Xp[:, cols] = Xp[idx][:, cols]
        r2s.append(r2(y, predict(mlp, Xp)))
    return float(np.mean(r2s))


rng  = np.random.default_rng(0)
gpvs = {k: [] for k in ['cue', 'pos', 'joint', 'speed', 'head_angle', 'rot_vel']}

for s_idx in valid:
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
    Xte = Xte[cz]; yte = yte[cz]

    mlp = load_mlp(s_idx)
    if mlp is None:
        continue

    r2b = r2(yte, predict(mlp, Xte))
    gpvs['cue'].append(       r2b - gpv_mean(mlp, Xte, yte, CUE_COLS,   rng))
    gpvs['pos'].append(       r2b - gpv_mean(mlp, Xte, yte, POS_COLS,   rng))
    gpvs['joint'].append(     r2b - gpv_mean(mlp, Xte, yte, JOINT_COLS, rng))
    gpvs['speed'].append(     r2b - gpv_mean(mlp, Xte, yte, SPD_COLS,   rng))
    gpvs['head_angle'].append(r2b - gpv_mean(mlp, Xte, yte, HA_COLS,    rng))
    gpvs['rot_vel'].append(   r2b - gpv_mean(mlp, Xte, yte, RV_COLS,    rng))
    print(f'  S{s_idx+1:02d} done')

means = {k: float(np.mean(v)) if v else 0.0 for k, v in gpvs.items()}
n_s   = max(len(v) for v in gpvs.values())

print('\nGPV in cue zone (mean across sessions):')
for k, v in means.items():
    print(f'  {k:15s}: {v:.4f}')

# ── Figure ────────────────────────────────────────────────────────────────────
BARS = [
    ('Cue\n(marginal)',          'cue',        '#FF7F0E'),
    ('Position\n(marginal)',     'pos',        '#8C564B'),
    ('Cue + Pos\n(joint)',       'joint',      '#9467BD'),
    ('Fwd Speed',                'speed',      '#1F77B4'),
    ('Head Angle',               'head_angle', '#2CA02C'),
    ('Rot. Vel.',                'rot_vel',    '#7F7F7F'),
]

labels = [b[0] for b in BARS]
vals   = [means[b[1]] for b in BARS]
colors = [b[2] for b in BARS]
x      = np.arange(len(BARS))

fig, ax = plt.subplots(figsize=FIG.FULL)
apply_style(fig, ax)

bars = ax.bar(x, vals, color=colors, alpha=0.82, width=0.58)

# Value labels above each bar
for xi, v in zip(x, vals):
    if abs(v) > 0.0001:
        ax.text(xi, max(v, 0) + max(vals) * 0.015, f'{v:.4f}',
                ha='center', va='bottom', fontsize=FONT.ANNOTATION - 1, color='#333')

# Reference line at GPV(cue+pos joint)
joint_val = means['joint']
ax.axhline(joint_val, color='#9467BD', lw=1.2, ls='--', alpha=0.70,
           label=f'GPV(cue+pos jointly) = {joint_val:.4f}')

# Annotate the joint bar bracket
ax.axhline(0, color='#555', lw=0.7, ls='-')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=FONT.TICK)
ax.set_ylabel('GPV (ΔR²) in cue zone', fontsize=FONT.LABEL)
ax.set_title(
    'Cue+position GPV is outcompeted by head angle and speed',
    fontsize=FONT.LABEL, pad=4)
ax.legend(fontsize=FONT.LEGEND, frameon=False, loc='upper left')
ax.tick_params(labelsize=FONT.TICK)

add_footnote(fig,
    f'E07 (E06), {n_s} sessions; cue-zone timepoints; '
    f'GPV = mean over {N_PERMS} perms; dashed line = joint cue+position GPV')

savefig_manifest(fig, 'e07_cue_zone_competition.png', OUT_DIRS)
print('\nSaved e07_cue_zone_competition.png')
