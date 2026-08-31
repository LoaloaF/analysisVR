#!/usr/bin/env python3
"""
Train a TempConv-Pred (MSE, offset10-model-mse) model using ONLY a single
feature column as input, for the pairs where TempConv IG flags that feature.

Architecture: (T, 1, 10) windows -> CNN encoder (32 units, 8-dim) -> Ridge probe -> R²
This is identical to the full TempConv-Pred pipeline, just with num_neurons=1.

Tests:
  rot_acc pairs  (TempConv IG high but GPV~0 — expected null)
  head_vel pairs (TempConv IG high AND GPV high — expected signal)
  head_ang pairs (baseline, both models agree — expected signal)
"""

import os, sys, pickle
import numpy as np
import torch
import torch.nn as nn
import cebra.models
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from load_encoder import build_windows

# ── config ────────────────────────────────────────────────────────────────────
SEEDS        = [42, 43, 44, 45, 46]
N_EPOCHS     = 300
LR           = 3e-4
BATCH_SIZE   = 512
RIDGE_ALPHA  = 1.0
N_SHUFFLE    = 20
EMBED_DIM    = 8
N_UNITS      = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── data ──────────────────────────────────────────────────────────────────────
with open('../outputs/mlps/ensembles_multiseed/semantic_groups.pkl', 'rb') as f:
    sg = pickle.load(f)
feat_idx_by_group = {g: cols for g, cols in sg}

with open('../outputs/session_dataset_ensembles.pkl', 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())

rota_col = feat_idx_by_group['frame_YawPitch_abs_acc_sum_500msMedian'][0]
hvel_col = feat_idx_by_group['head_angle_vel'][0]
hang_col = feat_idx_by_group['head_angle'][0]

# pairs: (label, s_idx, n_idx, feat_col)
# rot_acc — the CEBRA-IG-exclusive pairs where GPV~0
# head_vel — the CEBRA-IG-exclusive pairs where GPV is real
# head_ang — baseline (both models agree, known signal)
PAIRS = [
    ('rot_acc  S01E05',  1,  5, rota_col, 'rota'),
    ('rot_acc  S11E07', 11,  7, rota_col, 'rota'),
    ('rot_acc  S09E07',  9,  7, rota_col, 'rota'),
    ('rot_acc  S24E16', 24, 16, rota_col, 'rota'),
    ('head_vel S12E13', 12, 13, hvel_col, 'hvel'),
    ('head_vel S10E09', 10,  9, hvel_col, 'hvel'),
    ('head_vel S11E09', 11,  9, hvel_col, 'hvel'),
    ('head_ang S00E02',  0,  2, hang_col, 'base'),
    ('head_ang S01E02',  1,  2, hang_col, 'base'),
]


# ── helpers ───────────────────────────────────────────────────────────────────
def get_arrays(sess, trial_ids, feat_col, n_idx):
    sd = ds[sess]
    vt = [t for t in trial_ids if t in sd['data']]
    X  = np.concatenate([sd['data'][t][:, [feat_col]] for t in vt], axis=0).astype(np.float32)
    y  = np.concatenate([sd['labels'][t][:, n_idx]    for t in vt], axis=0).astype(np.float32)
    return X, y   # X: (T, 1)


def make_encoder():
    enc = cebra.models.init('offset10-model-mse',
                             num_neurons=1, num_units=N_UNITS, num_output=EMBED_DIM)
    return enc.to(device)


def embed(encoder, X_1feat):
    """X_1feat: (T, 1) numpy -> (T, EMBED_DIM) numpy."""
    wins = build_windows(X_1feat, window_len=10).astype(np.float32)  # (T, 1, 10)
    encoder.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(wins), 512):
            chunk = torch.tensor(wins[b:b+512], device=device)
            z = encoder(chunk)
            if z.dim() == 3: z = z.squeeze(-1)
            out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0)


def train_cebra_pred(X_tr, y_tr, seed):
    """Train encoder + linear head end-to-end with MSE on train set."""
    torch.manual_seed(seed)
    encoder = make_encoder()
    head    = nn.Linear(EMBED_DIM, 1).to(device)
    params  = list(encoder.parameters()) + list(head.parameters())
    opt     = torch.optim.Adam(params, lr=LR)
    loss_fn = nn.MSELoss()

    wins = build_windows(X_tr, window_len=10).astype(np.float32)  # (T, 1, 10)
    Wt   = torch.tensor(wins,  device=device)
    yt   = torch.tensor(y_tr,  device=device).unsqueeze(1)

    encoder.train(); head.train()
    for _ in range(N_EPOCHS):
        perm = torch.randperm(len(Wt), device=device)
        for b in range(0, len(Wt), BATCH_SIZE):
            idx = perm[b:b + BATCH_SIZE]
            opt.zero_grad()
            z    = encoder(Wt[idx])
            if z.dim() == 3: z = z.squeeze(-1)
            loss_fn(head(z), yt[idx]).backward()
            opt.step()

    encoder.eval()
    return encoder


def eval_with_ridge(encoder, X_tr, y_tr, X_te, y_te, seed, n_shuffle=N_SHUFFLE):
    """Fit Ridge on train embeddings, evaluate on test. Also compute null."""
    emb_tr = embed(encoder, X_tr)
    emb_te = embed(encoder, X_te)

    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(emb_tr, y_tr)
    pred   = ridge.predict(emb_te)
    r2     = float(r2_score(y_te, pred))

    rng    = np.random.default_rng(seed)
    nulls  = [float(r2_score(rng.permutation(y_te), pred)) for _ in range(n_shuffle)]
    return r2, float(np.median(nulls))


# ── main ──────────────────────────────────────────────────────────────────────
from scipy.stats import ttest_1samp

print(f'Device: {device}')
print()
hdr = f'{"Pair":<22}  {"cebra_r2":>9}  {"null_r2":>8}  {"delta":>7}  sig'
print(hdr)
print('-' * len(hdr))

for label, s_idx, n_idx, feat_col, cat in PAIRS:
    sess = sessions[s_idx]
    r2s, nulls = [], []

    for seed in SEEDS:
        split_map    = np.load(f'../splits/split_seed{seed}.npy', allow_pickle=True).item()
        test_trials  = [int(i) for i in split_map[sess]]
        all_trials   = list(ds[sess]['data'].keys())
        train_trials = [t for t in all_trials if t not in test_trials]

        X_tr, y_tr = get_arrays(sess, train_trials, feat_col, n_idx)
        X_te, y_te = get_arrays(sess, test_trials,  feat_col, n_idx)

        if len(X_tr) < 50 or len(X_te) < 20 or np.var(y_te) < 1e-8:
            continue

        encoder = train_cebra_pred(X_tr, y_tr, seed)
        r2, null_med = eval_with_ridge(encoder, X_tr, y_tr, X_te, y_te, seed)
        r2s.append(r2)
        nulls.append(null_med)

    if not r2s:
        print(f'{label:<22}  (no valid splits)')
        continue

    r2_med   = float(np.median(r2s))
    null_med = float(np.median(nulls))
    delta    = r2_med - null_med

    # t-test: each seed's r2 vs its own null
    deltas_per_seed = np.array(r2s) - np.array(nulls)
    _, p = ttest_1samp(deltas_per_seed, 0, alternative='greater')
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))

    print(f'{label:<22}  {r2_med:9.4f}  {null_med:8.4f}  {delta:7.4f}  {sig}  '
          f'[{" ".join(f"{v:.3f}" for v in r2s)}]')

print()
print('cebra_r2 = Ridge probe on CNN encoder trained end-to-end (MSE) with 1 feature')
print('null_r2  = shuffled test labels through same encoder+Ridge')
