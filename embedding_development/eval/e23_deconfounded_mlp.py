#!/usr/bin/env python3
"""
Retrain MLP for E23 x upcoming_choice with the 5 collinear feature groups removed.
Collinear groups (mean Cohen d > 0.5 split by upcoming_choice condition):
  frame_raw_500msMedian, frame_YawPitch_abs_vel_sum_500msMedian,
  frame_position, head_angle, lick_detected

Remaining groups used as input:
  frame_raw_abs_acc_500msMedian, frame_YawPitch_abs_acc_sum_500msMedian,
  head_angle_vel, cue_visible, reward_window, upcoming_choice

Then compute GPV for upcoming_choice in this deconfounded model and check
whether it now correlates with per-session Cohen's d.
"""

import os, sys, pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

SEEDS      = [42, 43, 44, 45, 46]
N_EPOCHS   = 150
LR         = 1e-3
HIDDEN     = 64
N_LAYERS   = 2
BATCH_SIZE = 512
R2_THRESH  = 0.01
ENS_IDX    = 22
G_NAME     = 'upcoming_choice'

COLLINEAR_GROUPS = {
    'frame_raw_500msMedian',
    'frame_YawPitch_abs_vel_sum_500msMedian',
    'frame_position',
    'head_angle',
    'lick_detected',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

output_dir  = '../outputs/ablation_vs_attribution'
desktop_dir = '/mnt/c/Users/amits/Desktop/ablation_vs_attribution'
model_dir   = '../outputs/ablation_vs_attribution/deconfounded_models'
os.makedirs(model_dir, exist_ok=True)

with open('../outputs/mlps/ensembles_multiseed/semantic_groups.pkl', 'rb') as f:
    sg = pickle.load(f)
group_names       = [g[0] for g in sg]
feat_idx_by_group = {g: cols for g, cols in sg}

# build kept feature columns and group index within kept features
kept_groups  = [(g, cols) for g, cols in sg if g not in COLLINEAR_GROUPS]
kept_cols    = np.concatenate([cols for _, cols in kept_groups]).tolist()
choice_cols_kept = feat_idx_by_group[G_NAME]   # original indices
# find upcoming_choice indices within kept_cols
choice_local = [kept_cols.index(c) for c in choice_cols_kept]

print(f'\nKept feature groups ({len(kept_groups)}):')
for g, cols in kept_groups:
    marker = '  <-- target' if g == G_NAME else ''
    print(f'  {g}  ({len(cols)} cols){marker}')
print(f'Total kept features: {len(kept_cols)}')
print(f'upcoming_choice local indices: {choice_local}')

with open('../outputs/session_dataset_ensembles.pkl', 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())

mlp_r2_all = np.load('../outputs/mlps/ensembles_multiseed/all_r2.npy')
mlp_valid  = (~np.all(np.isnan(mlp_r2_all), axis=0)) & (np.nanmean(mlp_r2_all, axis=0) >= R2_THRESH)


def make_mlp(in_size):
    layers, prev = [], in_size
    for _ in range(N_LAYERS):
        layers += [nn.Linear(prev, HIDDEN), nn.ReLU()]
        prev = HIDDEN
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers).to(device)


def get_arrays(sess, trial_ids):
    sd = ds[sess]
    vt = [t for t in trial_ids if t in sd['data']]
    X  = np.concatenate([sd['data'][t][:, kept_cols]  for t in vt], axis=0).astype(np.float32)
    y  = np.concatenate([sd['labels'][t][:, ENS_IDX]  for t in vt], axis=0).astype(np.float32)
    return X, y


def train_and_eval(X_tr, y_tr, X_te, y_te, seed, model_path=None):
    if model_path and os.path.exists(model_path):
        net = make_mlp(X_tr.shape[1])
        net.load_state_dict(torch.load(model_path, map_location=device))
    else:
        torch.manual_seed(seed)
        net     = make_mlp(X_tr.shape[1])
        opt     = torch.optim.Adam(net.parameters(), lr=LR)
        loss_fn = nn.MSELoss()
        Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
        yt = torch.tensor(y_tr, dtype=torch.float32, device=device).unsqueeze(1)
        for _ in range(N_EPOCHS):
            perm = torch.randperm(len(Xt), device=device)
            for b in range(0, len(Xt), BATCH_SIZE):
                idx = perm[b:b+BATCH_SIZE]
                opt.zero_grad()
                loss_fn(net(Xt[idx]), yt[idx]).backward()
                opt.step()
        if model_path:
            torch.save(net.state_dict(), model_path)

    net.eval()
    Xte_t = torch.tensor(X_te, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = net(Xte_t).squeeze().cpu().numpy()
    r2 = float(r2_score(y_te, pred)) if np.var(y_te) > 1e-8 else np.nan

    # GPV for upcoming_choice: permute its columns, measure R² drop
    X_perm = X_te.copy()
    rng = np.random.default_rng(seed)
    X_perm[:, choice_local] = rng.permutation(X_perm[:, choice_local])
    with torch.no_grad():
        pred_perm = net(torch.tensor(X_perm, dtype=torch.float32, device=device)).squeeze().cpu().numpy()
    r2_perm = float(r2_score(y_te, pred_perm)) if np.var(y_te) > 1e-8 else np.nan
    gpv = r2 - r2_perm

    return r2, gpv


def cohens_d_raw(sess):
    sd = ds[sess]
    ch_cols = feat_idx_by_group[G_NAME]
    X = np.concatenate([sd['data'][t][:, ch_cols] for t in sd['data']], axis=0)
    y = np.concatenate([sd['labels'][t][:, ENS_IDX] for t in sd['data']], axis=0)
    cond = np.argmax(X, axis=1)
    groups = [y[cond==c] for c in range(3) if (cond==c).sum()>1]
    if len(groups)<2: return np.nan
    best=0.0
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            a,b=groups[i],groups[j]
            s=np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2)
            if s>1e-10: best=max(best,abs(a.mean()-b.mean())/s)
    return best


# ── main loop ─────────────────────────────────────────────────────────────────
valid_sessions = [s for s in range(len(sessions)) if mlp_valid[s, ENS_IDX]]
print(f'\nValid sessions for E23: {len(valid_sessions)}')

rows = []
for s_idx in valid_sessions:
    sess = sessions[s_idx]
    r2s, gpvs = [], []

    for seed in SEEDS:
        split_map    = np.load(f'../splits/split_seed{seed}.npy', allow_pickle=True).item()
        test_trials  = [int(i) for i in split_map[sess]]
        all_trials   = list(ds[sess]['data'].keys())
        train_trials = [t for t in all_trials if t not in test_trials]

        X_tr, y_tr = get_arrays(sess, train_trials)
        X_te, y_te = get_arrays(sess, test_trials)

        if len(X_tr) < 50 or len(X_te) < 20:
            continue

        mpath = os.path.join(model_dir, f'E23_deconf_S{s_idx:02d}_seed{seed}.pt')
        r2, gpv = train_and_eval(X_tr, y_tr, X_te, y_te, seed, mpath)
        if not np.isnan(r2):
            r2s.append(r2)
            gpvs.append(gpv)

    if not r2s:
        continue

    cd = cohens_d_raw(sess)
    rows.append(dict(
        s_idx=s_idx, session=sess,
        r2=float(np.median(r2s)),
        gpv=float(np.median(gpvs)),
        cohen_d=cd,
        n_seeds=len(r2s),
    ))
    print(f'  S{s_idx:02d}  r2={rows[-1]["r2"]:.4f}  gpv={rows[-1]["gpv"]:.5f}  '
          f'cohen_d={cd:.3f}')

df = pd.DataFrame(rows)
df.to_csv(os.path.join(output_dir, 'E23_deconfounded_results.csv'), index=False)

print(f'\nResults ({len(df)} sessions):')
print(f'  Median deconfounded R²: {df.r2.median():.4f}')
print(f'  Median GPV (upcoming_choice): {df.gpv.median():.5f}')

sub = df[['gpv', 'cohen_d']].dropna()
rho, p = spearmanr(sub['gpv'], sub['cohen_d'])
print(f'\n  GPV vs Cohen d: rho={rho:+.3f}  p={p:.3f}  n={len(sub)}')

# also check vs original full-model GPV
orig_gpv = np.load('../outputs/mlps/ensembles_multiseed/importance_global_pv_semantic.npy')
g_idx = group_names.index(G_NAME)
df['orig_gpv'] = [orig_gpv[r.s_idx, ENS_IDX, g_idx] for _, r in df.iterrows()]
sub2 = df[['orig_gpv', 'cohen_d']].dropna()
rho2, p2 = spearmanr(sub2['orig_gpv'], sub2['cohen_d'])
print(f'  Original GPV vs Cohen d: rho={rho2:+.3f}  p={p2:.3f}  n={len(sub2)}')

# scatter
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, col, label, color in [
    (axes[0], 'orig_gpv', 'Original GPV\n(all features)', '#1B5E20'),
    (axes[1], 'gpv',      'Deconfounded GPV\n(collinear features removed)', '#E65100'),
]:
    sub = df[[col, 'cohen_d']].dropna()
    rho, p = spearmanr(sub[col], sub['cohen_d'])
    ax.scatter(sub[col], sub['cohen_d'], color=color, s=65, alpha=0.85, zorder=3)
    for _, row in sub.iterrows():
        ax.annotate(f"S{int(df.loc[row.name,'s_idx']):02d}",
                    (row[col], row['cohen_d']),
                    fontsize=6, xytext=(3,2), textcoords='offset points')
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Cohen's d (upcoming_choice)" if ax is axes[0] else '')
    ax.set_title(f'rho={rho:+.3f}  p={p:.3f}', fontsize=10)

plt.suptitle("E23 × upcoming_choice — does removing collinear features\nimprove GPV correlation with Cohen's d?", fontsize=11)
plt.tight_layout()
for root in (output_dir, desktop_dir):
    plt.savefig(os.path.join(root, 'E23_deconfounded_gpv_vs_cohend.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved E23_deconfounded_gpv_vs_cohend.png')
