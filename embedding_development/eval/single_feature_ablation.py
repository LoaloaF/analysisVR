#!/usr/bin/env python3
"""
Single-feature ablation test.

For a set of (session, ensemble, feature) triples, trains three models
using ONLY that feature as input and checks test R²:

  1. Linear     — OLS on the single feature (point-in-time)
  2. MLP        — 2-hidden-layer MLP on the single feature (point-in-time)
  3. Ridge-win  — Ridge on a 10-step window of the single feature (temporal)

If R² > null (shuffled) and > linear, there is a non-trivial functional
mapping that the MLP/temporal model has found.

Pairs tested
------------
  head_angle_vel:
    S12 E13  (TempConv-Pred GPV=0.450 — top hit)
    S09 E09  (TempConv-Pred GPV=0.075)
    S11 E09  (TempConv-Pred GPV=0.053)
    S08 E09  (TempConv-Pred GPV=0.052)
    S10 E09  (TempConv-Pred GPV=0.040)

  rot_acc (frame_YawPitch_abs_acc_sum):
    S01 E05  (TempConv GPV ≈ 0.001 — expected null)
    S11 E07  (TempConv GPV ≈ 0.003 — expected null)
"""

import os, sys, pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from scipy.stats import ttest_1samp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from load_encoder import build_windows   # reuse window builder

# ── config ────────────────────────────────────────────────────────────────────
SEEDS       = [42, 43, 44, 45, 46]
N_EPOCHS    = 200
LR          = 1e-3
HIDDEN      = 64
N_LAYERS    = 2
WIN_LEN     = 10
N_SHUFFLE   = 20    # shuffled-label replicates for null distribution
BATCH_SIZE  = 512
RIDGE_ALPHA = 1.0

# ── load data ─────────────────────────────────────────────────────────────────
with open('../outputs/mlps/ensembles_multiseed/semantic_groups.pkl', 'rb') as f:
    sg = pickle.load(f)
group_names       = [g[0] for g in sg]
feat_idx_by_group = {g: cols for g, cols in sg}

with open('../outputs/session_dataset_ensembles.pkl', 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())

hvel_col = feat_idx_by_group['head_angle_vel'][0]   # single column index
rota_col = feat_idx_by_group['frame_YawPitch_abs_acc_sum_500msMedian'][0]

PAIRS = [
    ('head_vel S12E13', 12, 13, hvel_col),
    ('head_vel S09E09',  9,  9, hvel_col),
    ('head_vel S11E09', 11,  9, hvel_col),
    ('head_vel S08E09',  8,  9, hvel_col),
    ('head_vel S10E09', 10,  9, hvel_col),
    ('rot_acc  S01E05',  1,  5, rota_col),
    ('rot_acc  S11E07', 11,  7, rota_col),
]

# ── model ─────────────────────────────────────────────────────────────────────
def make_mlp(in_size):
    layers = []
    prev = in_size
    for _ in range(N_LAYERS):
        layers += [nn.Linear(prev, HIDDEN), nn.ReLU()]
        prev = HIDDEN
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def train_mlp(X_tr, y_tr, X_te, y_te, seed):
    torch.manual_seed(seed)
    net = make_mlp(X_tr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    for _ in range(N_EPOCHS):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), BATCH_SIZE):
            idx = perm[b:b + BATCH_SIZE]
            opt.zero_grad()
            loss_fn(net(Xt[idx]), yt[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(X_te, dtype=torch.float32)).squeeze().numpy()
    return float(r2_score(y_te, pred))


# ── helpers ───────────────────────────────────────────────────────────────────
def get_arrays(sess, trial_ids, feat_col, n_idx):
    sd = ds[sess]
    vt = [t for t in trial_ids if t in sd['data']]
    X  = np.concatenate([sd['data'][t][:, [feat_col]] for t in vt], axis=0).astype(np.float32)
    y  = np.concatenate([sd['labels'][t][:, n_idx]    for t in vt], axis=0).astype(np.float32)
    return X, y


def windowed_single(X_1d):
    """Build (T, WIN_LEN) windows from a (T, 1) feature array."""
    arr2d  = np.concatenate([X_1d, np.zeros_like(X_1d)], axis=1)[:, :1]  # (T,1)
    wins   = build_windows(arr2d.reshape(-1, 1), window_len=WIN_LEN)       # (T, 1, WIN_LEN)
    return wins.reshape(len(wins), WIN_LEN)                                # (T, WIN_LEN)


# ── main loop ─────────────────────────────────────────────────────────────────
print(f'{"Pair":<20}  {"lin_r2":>8}  {"mlp_r2":>8}  {"win_r2":>8}  '
      f'{"null_r2":>8}  {"mlp>null":>9}  {"win>null":>9}')
print('-' * 85)

summary = []

for label, s_idx, n_idx, feat_col in PAIRS:
    sess = sessions[s_idx]

    lin_r2s, mlp_r2s, win_r2s, null_r2s = [], [], [], []

    for seed in SEEDS:
        split_map    = np.load(f'../splits/split_seed{seed}.npy', allow_pickle=True).item()
        test_trials  = [int(i) for i in split_map[sess]]
        all_trials   = list(ds[sess]['data'].keys())
        train_trials = [t for t in all_trials if t not in test_trials]

        X_tr, y_tr = get_arrays(sess, train_trials, feat_col, n_idx)
        X_te, y_te = get_arrays(sess, test_trials,  feat_col, n_idx)

        if len(X_tr) < 50 or len(X_te) < 20 or np.var(y_te) < 1e-8:
            continue

        # 1. Linear
        lr = LinearRegression()
        lr.fit(X_tr, y_tr)
        lin_r2s.append(float(r2_score(y_te, lr.predict(X_te))))

        # 2. MLP (point-in-time, single feature)
        mlp_r2s.append(train_mlp(X_tr, y_tr, X_te, y_te, seed))

        # 3. Ridge on windowed single feature
        W_tr = windowed_single(X_tr)
        W_te = windowed_single(X_te)
        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(W_tr, y_tr)
        win_r2s.append(float(r2_score(y_te, ridge.predict(W_te))))

        # 4. Null: shuffle labels N_SHUFFLE times
        rng = np.random.default_rng(seed)
        for _ in range(N_SHUFFLE):
            y_shuf = rng.permutation(y_te)
            null_r2s.append(float(r2_score(y_shuf, ridge.predict(W_te))))

    if not mlp_r2s:
        print(f'{label:<20}  (no valid splits)')
        continue

    lin_med  = float(np.median(lin_r2s))
    mlp_med  = float(np.median(mlp_r2s))
    win_med  = float(np.median(win_r2s))
    null_med = float(np.median(null_r2s))

    # one-sample t-test: is win_r2 significantly above null?
    null_arr = np.array(null_r2s)
    mlp_arr  = np.array(mlp_r2s)
    win_arr  = np.array(win_r2s)
    _, p_mlp = ttest_1samp(mlp_arr - null_arr.mean(), 0, alternative='greater')
    _, p_win = ttest_1samp(win_arr  - null_arr.mean(), 0, alternative='greater')

    sig_mlp = '***' if p_mlp < 0.001 else ('**' if p_mlp < 0.01 else ('*' if p_mlp < 0.05 else 'ns'))
    sig_win = '***' if p_win < 0.001 else ('**' if p_win < 0.01 else ('*' if p_win < 0.05 else 'ns'))

    print(f'{label:<20}  {lin_med:8.4f}  {mlp_med:8.4f}  {win_med:8.4f}  '
          f'{null_med:8.4f}  {sig_mlp:>9}  {sig_win:>9}')

    summary.append(dict(label=label, lin=lin_med, mlp=mlp_med,
                        win=win_med, null=null_med,
                        p_mlp=p_mlp, p_win=p_win))

print()
print('win_r2  = Ridge on 10-step window of single feature (temporal)')
print('mlp_r2  = 2-hidden-layer MLP on single feature (point-in-time)')
print('null_r2 = shuffled labels (expected ≈ 0)')
print('sig     = t-test vs null distribution (* p<0.05, ** p<0.01, *** p<0.001)')
