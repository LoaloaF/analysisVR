#!/usr/bin/env python3
"""
eval_e07_cue_zone_attribution.py

Within the cue zone, compute two things for each feature group:
  1. eta² vs cue condition (Cue 1 vs Cue 2) — how much does this feature
     co-vary with which cue is on screen?
  2. GPV within the cue zone — how much does the model rely on it there?

Shows that speed (and sometimes head angle) co-varies with cue identity
in the cue zone, explaining why the model uses those proxies rather than
cue_visible or frame_position directly.

Output: outputs/ablation_vs_attribution/e07_cue_zone_attribution.png
"""
import os, sys, pickle
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.models import MLP
from utils.figure_style import (
    FIG, DPI, FONT, FEATURE_NAMES_SHORT,
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

# ── Load ──────────────────────────────────────────────────────────────────────
ds       = pickle.load(open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb'))
sessions = list(ds.keys())
sg       = pickle.load(open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb'))
group_names = [g[0] for g in sg]
feat_idx    = {g: cols for g, cols in sg}
r2_all      = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)
tidx_map    = np.load(os.path.join(root, 'splits', f'split_seed{SEED}.npy'),
                      allow_pickle=True).item()

CUE_COLS = feat_idx['cue_visible']
valid_sessions = [s for s in range(len(sessions)) if r2_all[s, E_IDX] >= R2_THRESH]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_mlp(s):
    path = os.path.join(root, 'models', 'mlps', 'ensembles',
                        f'seed{SEED}', f'session_{s:02d}_neuron_{E_IDX:02d}.pt')
    if not os.path.exists(path):
        return None
    sd    = torch.load(path, map_location=device)
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


def eta_sq_binary(feat_vals, cond_binary):
    """eta² of feat_vals vs binary cue condition (0=Cue1, 1=Cue2).
    Uses mean of each column (for multi-column groups, average eta²)."""
    etas = []
    for col in feat_vals.T:
        grand = np.mean(col)
        groups = [col[cond_binary == c] for c in np.unique(cond_binary)]
        ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
        ss_total   = np.sum((col - grand) ** 2)
        etas.append(ss_between / ss_total if ss_total > 1e-10 else 0.0)
    return float(np.mean(etas))


# ── Per-session computation ────────────────────────────────────────────────────
rng  = np.random.default_rng(0)
# Per-group results: dict of group_name → list of (eta², gpv)
group_eta  = {g: [] for g in group_names}
group_gpv  = {g: [] for g in group_names}

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
    cond_cz = cond[cz]          # values in {1, 2}
    cond_bin = (cond_cz == 2).astype(int)   # 0=Cue1, 1=Cue2

    mlp = load_mlp(s_idx)
    if mlp is None:
        continue

    r2_base = r2(yte_cz, predict(mlp, Xte_cz))
    if r2_base < 1e-6:
        continue

    for gname in group_names:
        cols = feat_idx[gname]
        # eta²
        eta = eta_sq_binary(Xte_cz[:, cols], cond_bin)
        group_eta[gname].append(eta)
        # GPV
        r2_perm = gpv_permuted(mlp, Xte_cz, yte_cz, cols, rng, N_PERMS)
        gpv_val = r2_base - r2_perm
        group_gpv[gname].append(gpv_val)

    print(f'  S{s_idx+1:02d} done (n_cz={cz.sum()})')

# ── Aggregate ─────────────────────────────────────────────────────────────────
mean_eta = {g: np.mean(v) if v else 0.0 for g, v in group_eta.items()}
mean_gpv = {g: np.mean(v) if v else 0.0 for g, v in group_gpv.items()}

# Sort by eta² descending
order = sorted(group_names, key=lambda g: -mean_eta[g])
print('\nFeature co-variation with cue condition in cue zone:')
for g in order:
    short = FEATURE_NAMES_SHORT.get(g, g)
    print(f'  {short:20s}  eta²={mean_eta[g]:.4f}  GPV={mean_gpv[g]:.4f}')

# ── Figure ────────────────────────────────────────────────────────────────────
short_names = [FEATURE_NAMES_SHORT.get(g, g) for g in order]
eta_vals    = [mean_eta[g] for g in order]
gpv_vals    = [mean_gpv[g] for g in order]

fig, (ax_eta, ax_gpv) = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig, [ax_eta, ax_gpv])

y_pos = np.arange(len(order))

# Panel A — eta² (co-variation with cue condition)
C_ETA = '#1F77B4'
bars_a = ax_eta.barh(y_pos, eta_vals, color=C_ETA, alpha=0.80, height=0.55)
ax_eta.set_yticks(y_pos)
ax_eta.set_yticklabels(short_names, fontsize=FONT.TICK - 1)
ax_eta.set_xlabel('η² (cue condition: Cue 1 vs Cue 2)', fontsize=FONT.LABEL - 1)
ax_eta.set_title('Feature co-variation with cue\nin the cue zone', fontsize=FONT.LABEL - 1, pad=3)
ax_eta.axvline(0, color='#555', lw=0.7)
for i, v in enumerate(eta_vals):
    if v > 0.001:
        ax_eta.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=FONT.ANNOTATION - 2)
add_panel_label(ax_eta, 'A')

# Panel B — GPV within cue zone
C_GPV = '#2CA02C'
bars_b = ax_gpv.barh(y_pos, gpv_vals, color=C_GPV, alpha=0.80, height=0.55)
ax_gpv.set_yticks(y_pos)
ax_gpv.set_yticklabels(short_names, fontsize=FONT.TICK - 1)
ax_gpv.set_xlabel('GPV (ΔR²) in the cue zone', fontsize=FONT.LABEL - 1)
ax_gpv.set_title('Model attribution within\nthe cue zone', fontsize=FONT.LABEL - 1, pad=3)
ax_gpv.axvline(0, color='#555', lw=0.7)
for i, v in enumerate(gpv_vals):
    if v > 0.0001:
        ax_gpv.text(v + max(gpv_vals) * 0.01, i, f'{v:.4f}',
                    va='center', fontsize=FONT.ANNOTATION - 2)
add_panel_label(ax_gpv, 'B')

n_sess_used = max(len(v) for v in group_eta.values() if v)
add_footnote(fig,
    f'E07 (ensemble 6), {n_sess_used} sessions with cue-zone data.  '
    'Panel A: η² of each feature vs binary cue condition (Cue 1 / Cue 2).  '
    'Panel B: GPV = R²_base − R²_permuted within cue-zone timepoints.  '
    f'n_perms={N_PERMS}.  Features sorted by η².')

savefig_manifest(fig, 'e07_cue_zone_attribution.png', OUT_DIRS)
print('\nSaved e07_cue_zone_attribution.png')
