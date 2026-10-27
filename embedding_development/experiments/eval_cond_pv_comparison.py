#!/usr/bin/env python3
"""
eval_cond_pv_comparison.py

Global vs Conditional Permutation Variance for E23 (idx 22) and E07 (idx 6),
for upcoming_choice_1 and cue_visible_2.

Session selection: qualify sessions where the range of group means of
z-scored ensemble activations across upcoming_choice categories {-1, 0, 1}
is >= MEAN_DIFF_THRESHOLD (replicates notebook Cell 8).

y_choice is reconstructed from the z-scored one-hot features in X (as in the
notebook), not from df_full, to stay in the training cache's data space.
"""
import os, sys, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from models import MLP
from figure_style import (
    FIG, DPI, FONT, LINE, MARKER,
    MODEL_COLORS, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TARGET_ASSEMBLIES    = {'E23': 22}
MEAN_DIFF_THRESHOLD  = 0.2
MIN_GROUP_SIZE       = 10

# E23 encodes upcoming_choice — filter to frame_position > 100 (z>0) for stop/skip
ASSEMBLY_GROUPS = {
    'E23': {'var': 'choice', 'groups': [-1, 0, 1]},
}

NON_CAT = ['frame_raw_500msMedian', 'frame_raw_abs_acc_500msMedian',
           'frame_YawPitch_abs_vel_sum_500msMedian', 'frame_YawPitch_abs_acc_sum_500msMedian',
           'head_angle_vel', 'head_angle', 'frame_position']
CAT_VARS = ['cue_visible', 'upcoming_choice', 'reward_window', 'lick_detected']

N_HA_BINS      = 5
N_COND_PERMS   = 200
N_GLOBAL_PERMS = 50
SEED           = 42

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, "..")

# ─── BUILD FEATURE COLUMNS (to get column indices) ───────────────────────────
import pandas as pd
data_dir = os.path.join(root, "outputs", "glm_input_data")
beh_cols = np.load(os.path.join(data_dir, "behavior_glm_input_columns.npy"), allow_pickle=True)
beh_vals = np.load(os.path.join(data_dir, "behavior_glm_input.npy"),         allow_pickle=True)
beh_idx  = np.load(os.path.join(data_dir, "behavior_glm_input_index.npy"),   allow_pickle=True)
df_full  = pd.DataFrame(beh_vals, index=pd.Index(beh_idx), columns=beh_cols)

zone_onehot = []
for col in CAT_VARS:
    col_vals = sorted(df_full[col].dropna().unique().astype(int))
    if len(col_vals) == 2:
        zone_onehot.append(f'{col}_{col_vals[-1]}')
    else:
        zone_onehot.extend([f'{col}_{v}' for v in col_vals])

ALL_FEAT_COLS  = NON_CAT + zone_onehot
N_FEATS        = len(ALL_FEAT_COLS)
HEAD_ANGLE_IDX = ALL_FEAT_COLS.index('head_angle')

# Indices for reconstructing condition labels from z-scored one-hot X
UC_1_IDX    = ALL_FEAT_COLS.index('upcoming_choice_1')
UC_NEG1_IDX = ALL_FEAT_COLS.index('upcoming_choice_-1')
CV_1_IDX    = ALL_FEAT_COLS.index('cue_visible_1')
CV_2_IDX    = ALL_FEAT_COLS.index('cue_visible_2')
FRAME_POS_IDX = ALL_FEAT_COLS.index('frame_position')

COND_PV_FEATS = {
    'upcoming_choice_1': UC_1_IDX,
    'cue_visible_2':     CV_2_IDX,
}
print(f"{N_FEATS} features  head_angle={HEAD_ANGLE_IDX}  frame_pos={FRAME_POS_IDX}  "
      f"UC_1={UC_1_IDX}  UC_neg1={UC_NEG1_IDX}  CV_1={CV_1_IDX}  CV_2={CV_2_IDX}")

# ─── LOAD SESSION DATASET & BUILD FLAT ARRAYS ────────────────────────────────
cache_path = os.path.join(root, "outputs", "session_dataset_ensembles.pkl")
with open(cache_path, "rb") as f:
    ds = pickle.load(f)

# Preserve session order from beh_idx (same as notebook Cell 3)
_seen, session_ids = set(), []
for elem in beh_idx:
    sid = elem[0]
    if sid not in _seen and sid in ds:
        _seen.add(sid); session_ids.append(sid)
num_sessions = len(session_ids)

# Flatten cache → per-session X/Y arrays and position map (all trials)
session_X    = {}   # sid → (n_timepoints, N_FEATS)
session_Y    = {}   # sid → (n_timepoints, 23)
session_pos  = {}   # sid → slice/array of global indices (not used here but mirrors notebook)

for sid in session_ids:
    sd   = ds[sid]
    tids = sorted(sd['data'].keys())
    Xs   = np.concatenate([sd['data'][t]   for t in tids]).astype(np.float32)
    Ys   = np.concatenate([sd['labels'][t] for t in tids]).astype(np.float32)
    session_X[sid] = Xs
    session_Y[sid] = Ys

tidx_map = np.load(os.path.join(root, "splits", f"split_seed{SEED}.npy"),
                   allow_pickle=True).item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}  sessions={num_sessions}")

# ─── LOAD MLP HELPER ──────────────────────────────────────────────────────────
def _load_mlp(s_idx, n_idx):
    path = os.path.join(root, "models", "mlps", "ensembles",
                        f"seed{SEED}", f"session_{s_idx:02d}_neuron_{n_idx:02d}.pt")
    if not os.path.exists(path):
        return None
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        h = sd['model_state_dict']['fc.0.weight'].shape[0]
        m = MLP(N_FEATS, h, 2, 1).to(device)
        m.load_state_dict(sd['model_state_dict'])
    else:
        h = sd['fc.0.weight'].shape[0]
        m = MLP(N_FEATS, h, 2, 1).to(device)
        m.load_state_dict(sd)
    m.eval()
    return m

# ─── SESSION SELECTION ────────────────────────────────────────────────────────
def _encoding_diff(sid, en_idx, en_name):
    """
    Range of group means of z-scored ensemble activation across the
    assembly-specific condition variable (ens_encodings.ipynb approach).

    E23 — upcoming_choice {-1, 0, 1}:
        stop/skip groups filtered to frame_position > 0 z-score (≈raw>100);
        baseline (0) uses all positions.
    E07 — cue_visible {0, 1, 2}:
        all positions, no filter.
    """
    Xs = session_X[sid]   # (T, N_FEATS), z-scored
    Ys = session_Y[sid]   # (T, 23)

    y_ens = Ys[:, en_idx].astype(np.float64)
    std   = y_ens.std()
    if std < 1e-8:
        return np.nan
    y_z = (y_ens - y_ens.mean()) / std

    cfg = ASSEMBLY_GROUPS[en_name]

    if cfg['var'] == 'choice':
        # Reconstruct upcoming_choice from z-scored one-hots
        y_cond = np.where(Xs[:, UC_1_IDX]    > 0,  1,
                 np.where(Xs[:, UC_NEG1_IDX] > 0, -1, 0)).astype(np.int32)
        pos_high = Xs[:, FRAME_POS_IDX] > 0   # z>0 ≈ frame_position > median (~100)
        # baseline (0): all positions; stop (+1) / skip (-1): position-filtered
        masks = {
            -1: (y_cond == -1) & pos_high,
             0:  y_cond == 0,
             1: (y_cond ==  1) & pos_high,
        }
    else:  # 'cue'
        y_cond = np.where(Xs[:, CV_2_IDX] > 0, 2,
                 np.where(Xs[:, CV_1_IDX] > 0, 1, 0)).astype(np.int32)
        masks = {g: (y_cond == g) for g in cfg['groups']}

    groups = [y_z[masks[g]] for g in cfg['groups']]
    if any(len(g) < MIN_GROUP_SIZE for g in groups):
        return np.nan

    gm = np.array([g.mean() for g in groups])
    return float(gm.max() - gm.min())

# ─── COMPUTE GLOBAL + CONDITIONAL PV ─────────────────────────────────────────
global_pv_sess = {en: {fn: [] for fn in COND_PV_FEATS} for en in TARGET_ASSEMBLIES}
cond_pv_sess   = {en: {fn: [] for fn in COND_PV_FEATS} for en in TARGET_ASSEMBLIES}
diff_sess      = {en: [] for en in TARGET_ASSEMBLIES}
n_qualifying   = {en: 0 for en in TARGET_ASSEMBLIES}

rng = np.random.default_rng(SEED)

for en_name, en_idx in TARGET_ASSEMBLIES.items():
    print(f"\n=== {en_name} (idx {en_idx}) ===")
    for s_idx, sess_id in enumerate(session_ids):

        diff = _encoding_diff(sess_id, en_idx, en_name)
        diff_str = f"{diff:.3f}" if not np.isnan(diff) else "nan"
        if np.isnan(diff) or diff < MEAN_DIFF_THRESHOLD:
            print(f"  S{s_idx+1}: skip (choice_range={diff_str})")
            continue

        # PV computation uses TEST trials only
        test_trials = tidx_map.get(sess_id, [])
        valid_t = [t for t in test_trials if t in ds[sess_id]["data"]]
        if not valid_t:
            print(f"  S{s_idx+1}: skip (no test trials, choice_range={diff_str})")
            continue

        Xnp = np.concatenate([ds[sess_id]["data"][t]   for t in valid_t], 0).astype(np.float32)
        Ynp = np.concatenate([ds[sess_id]["labels"][t] for t in valid_t], 0)
        y_s = Ynp[:, en_idx].astype(np.float32)

        model = _load_mlp(s_idx, en_idx)
        if model is None:
            print(f"  S{s_idx+1}: skip (no model)")
            continue

        with torch.no_grad():
            _, yb = model(torch.tensor(Xnp, device=device))
        r2_base = float(r2_score(y_s, yb.squeeze(-1).cpu().numpy()))

        n_qualifying[en_name] += 1
        diff_sess[en_name].append(diff)
        print(f"  S{s_idx+1}: choice_range={diff_str}  R²={r2_base:.3f} ✓", flush=True)

        for feat_name, feat_idx in COND_PV_FEATS.items():
            # Global PV
            gpv_drops = []
            for _ in range(N_GLOBAL_PERMS):
                perm = Xnp.copy()
                perm[:, feat_idx] = Xnp[rng.permutation(len(Xnp)), feat_idx]
                with torch.no_grad():
                    _, yp = model(torch.tensor(perm, device=device))
                gpv_drops.append(r2_base - float(r2_score(y_s, yp.squeeze(-1).cpu().numpy())))
            global_pv_sess[en_name][feat_name].append(float(np.mean(gpv_drops)))

            # Conditional PV (permute within head_angle bins)
            ha    = Xnp[:, HEAD_ANGLE_IDX]
            edges = np.quantile(ha, np.linspace(0, 1, N_HA_BINS + 1))
            bids  = np.digitize(ha, edges[1:-1])
            cpv_drops = []
            for _ in range(N_COND_PERMS):
                Xp = Xnp.copy()
                for b in range(N_HA_BINS):
                    bmask = (bids == b)
                    if bmask.sum() < 2:
                        continue
                    bi = np.where(bmask)[0]
                    Xp[bmask, feat_idx] = Xp[rng.permutation(bi), feat_idx]
                with torch.no_grad():
                    _, yp = model(torch.tensor(Xp, device=device))
                cpv_drops.append(r2_base - float(r2_score(y_s, yp.squeeze(-1).cpu().numpy())))
            cond_pv_sess[en_name][feat_name].append(float(np.mean(cpv_drops)))

        del model; torch.cuda.empty_cache()

print("\n── Qualifying sessions ──")
for en_name in TARGET_ASSEMBLIES:
    diffs = diff_sess[en_name]
    md_str = f"{np.mean(diffs):.3f}" if diffs else "N/A"
    print(f"  {en_name}: {n_qualifying[en_name]}/{num_sessions}  mean_choice_range={md_str}")

# ─── PLOT ─────────────────────────────────────────────────────────────────────
# One panel per feature; within each panel: one pair of bars (Global / Conditional).
en_name = list(TARGET_ASSEMBLIES.keys())[0]   # 'E23'
n_q     = n_qualifying[en_name]

feat_names = list(COND_PV_FEATS.keys())
fig, axes = plt.subplots(1, len(feat_names), figsize=(6.54, 4.32), sharey=False)
apply_style(fig, axes if hasattr(axes, '__iter__') else [axes])
if len(feat_names) == 1:
    axes = [axes]

for ax, feat_name in zip(axes, feat_names):
    gvals = global_pv_sess[en_name][feat_name]
    cvals = cond_pv_sess[en_name][feat_name]

    gmean = float(np.mean(gvals)) if gvals else 0
    gsem  = float(np.std(gvals) / max(1, len(gvals)) ** 0.5) if gvals else 0
    cmean = float(np.mean(cvals)) if cvals else 0
    csem  = float(np.std(cvals) / max(1, len(cvals)) ** 0.5) if cvals else 0
    ratio = cmean / (gmean + 1e-8)

    ek = dict(ecolor='k', lw=0.8, capsize=4)
    ax.bar(0, gmean, 0.4, color='#aec7e8', label='Global PV',      alpha=0.9, yerr=gsem,  error_kw=ek)
    ax.bar(1, cmean, 0.4, color='#1f77b4', label='Conditional PV', alpha=0.9, yerr=csem, error_kw=ek)

    ymax = max(gmean + gsem, cmean + csem) * 1.45 or 0.05
    ax.set_ylim(0, ymax)
    ax.text(1, cmean + csem + ymax * 0.04,
            f'×{ratio:.1f}', ha='center', fontsize=9,
            color='#c0392b' if ratio < 0.5 else 'k', fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Global PV', 'Conditional PV'], fontsize=9)
    ax.set_ylabel('PV (R² drop)', fontsize=9)
    ax.set_title(f'{en_name}  —  {feat_name.replace("_", " ")}\n(n={n_q} sessions)', fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)

plt.suptitle(f'Global vs Conditional PV  (choice selectivity ≥{MEAN_DIFF_THRESHOLD}σ)\n'
             '×ratio < 1 → collinearity suppresses global PV',
             fontsize=9, y=1.02)

out_dir = os.path.join(root, "outputs", "residual_choice")
os.makedirs(out_dir, exist_ok=True)
savefig_manifest(fig, "v3_cond_pv_comparison.png", [out_dir])
print(f"\nSaved {os.path.join(out_dir, 'v3_cond_pv_comparison.png')}")
