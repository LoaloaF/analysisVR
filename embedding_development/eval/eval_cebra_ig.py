#!/usr/bin/env python3
"""
eval_cebra_ig.py

Integrated Gradients attribution for TempConv encoders.

Pipeline:
  behavioral state-actions (T, 17)
    → sliding 10-step windows (T, 17, 10)
    → TempConv encoder  → embedding (T, 8)
    → Ridge probe    → predicted neural activity (T,)

IG is applied end-to-end: we compose encoder + Ridge as a differentiable
function and run the standard IG algorithm with a zero baseline.

IG attribution shape: (T, 17, 10)
  → sum |IG| over window dim → (T, 17)
  → mean over T              → (17,)
  → aggregate by semantic group → (11,)

Saves: outputs/{arm}_eval/ensembles/importance_ig_semantic.npy  (29, 23, 11)

Usage:
    python eval/eval_cebra_ig.py --arm cebra
    python eval/eval_cebra_ig.py --arm cebra_pred
    python eval/eval_cebra_ig.py --arm cebra --force_recompute
"""
import os
import sys
import argparse
import pickle
import warnings

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from load_encoder import build_windows, load_encoder

# ─── config ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--arm', type=str, default='cebra',
                    choices=['cebra', 'cebra_pred'],
                    help='Which TempConv arm to compute IG for')
parser.add_argument('--models_dir', type=str, default=None,
                    help='Override model checkpoint directory')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Override output directory')
parser.add_argument('--force_recompute', action='store_true',
                    help='Ignore per-seed checkpoints and recompute')
args = parser.parse_args()

ARM    = args.arm
SEEDS  = [42, 43, 44, 45, 46]
IG_STEPS     = 50
RIDGE_ALPHA  = 1.0
R2_THRESHOLD = 0.01
IG_BATCH     = 256    # timesteps per forward pass during IG (controls GPU/CPU memory)

splits_dir   = './splits'
models_root  = args.models_dir if args.models_dir else f'./models/{ARM}/ensembles'
cache_path   = './outputs/session_dataset_ensembles.pkl'
output_dir   = args.output_dir if args.output_dir else f'./outputs/{ARM}_eval/ensembles'
r2_path      = os.path.join(output_dir, 'all_r2.npy')
os.makedirs(output_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"ARM={ARM}  device={device}  IG_STEPS={IG_STEPS}")

# ─── load shared data ────────────────────────────────────────────────────────
with open(cache_path, 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())
num_sessions = len(sessions)

# Semantic groups — load from the MLP output (same feature space)
with open('./outputs/mlps/ensembles_multiseed/semantic_groups.pkl', 'rb') as f:
    semantic_groups = pickle.load(f)
n_groups  = len(semantic_groups)
group_names = [g[0] for g in semantic_groups]
print(f"Loaded {num_sessions} sessions, {n_groups} semantic groups")

# num_neurons from first session
num_neurons = next(iter(ds.values()))['labels'][
    next(iter(next(iter(ds.values()))['labels']))
].shape[1]
print(f"num_neurons (ensembles) = {num_neurons}")

# Validity mask: valid if mean R² (over seeds) >= threshold and no all-NaN seed
all_r2   = np.load(r2_path)                          # (5, 29, 23)
mean_r2  = np.nanmean(all_r2, axis=0)                 # (29, 23)
any_seed_all_nan = np.all(np.isnan(all_r2), axis=0)  # (29, 23)
valid = (~any_seed_all_nan) & (mean_r2 >= R2_THRESHOLD)
print(f"Valid (session, ensemble) pairs: {valid.sum()}")


# ─── helpers ─────────────────────────────────────────────────────────────────
def get_arrays(sess, trial_ids):
    sd = ds[sess]
    valid_t = [t for t in trial_ids if t in sd['data']]
    if not valid_t:
        return np.zeros((0, 17), dtype=np.float32), np.zeros((0, num_neurons), dtype=np.float32)
    X = np.concatenate([sd['data'][t]   for t in valid_t], axis=0).astype(np.float32)
    Y = np.concatenate([sd['labels'][t] for t in valid_t], axis=0).astype(np.float32)
    return X, Y


class EncoderRidge(torch.nn.Module):
    """Differentiable composition of TempConv encoder and a fitted Ridge probe."""
    def __init__(self, encoder, coef, bias):
        super().__init__()
        self.encoder = encoder
        # Register as buffers so they move with .to(device)
        self.register_buffer('coef', torch.tensor(coef, dtype=torch.float32))
        self.register_buffer('bias', torch.tensor(float(bias), dtype=torch.float32))

    def forward(self, x):
        # x: (B, 17, 10)
        z = self.encoder(x)          # (B, 8) or (B, 8, 1)
        if z.dim() == 3:
            z = z.squeeze(-1)        # (B, 8)
        return z @ self.coef + self.bias  # (B,)


def compute_cebra_ig(model, windows_np, steps=IG_STEPS):
    """
    Returns:
      ig_sem  (n_groups,)  — sum of mean |IG| per feature within each semantic group
      abs_col (17,)        — mean |IG| per input feature (summed over window dim)

    Processes all T test timesteps at once per alpha step (one forward-backward pass
    per alpha), which is much faster than batching by timestep.
    Memory: T × 17 × 10 × 4 bytes ≈ 3 MB for T=4500 — well within budget.
    """
    T       = len(windows_np)
    n_feats = windows_np.shape[1]   # 17
    win_len = windows_np.shape[2]   # 10

    test_t  = torch.tensor(windows_np, dtype=torch.float32, device=device)
    alphas  = np.linspace(0.0, 1.0, steps + 1)[1:]   # skip alpha=0 (baseline is zeros)

    grads_sum = np.zeros((T, n_feats, win_len), dtype=np.float64)

    for alpha in alphas:
        # baseline is zero, so interpolated input = alpha * test
        x_int = (float(alpha) * test_t).detach().requires_grad_(True)
        with torch.enable_grad():
            pred = model(x_int)        # (T,)
            pred.sum().backward()
        grads_sum += x_int.grad.detach().cpu().numpy()
        del x_int, pred

    mean_grads  = grads_sum / steps              # (T, 17, 10)
    ig_attr     = windows_np * mean_grads        # (T, 17, 10)  IG attributions

    # Reduce: sum |IG| over window dim, then mean over T
    ig_per_feat = np.sum(np.abs(ig_attr), axis=2)   # (T, 17)
    abs_col     = np.mean(ig_per_feat, axis=0)       # (17,)

    ig_sem = np.array([np.sum(abs_col[g_cols]) for _, g_cols in semantic_groups])
    return ig_sem, abs_col


# ─── main loop ───────────────────────────────────────────────────────────────
ig_sem_out_path = os.path.join(output_dir, 'importance_ig_semantic.npy')
ig_col_out_path = os.path.join(output_dir, 'importance_ig_per_feature.npy')

all_ig_sem = np.full((len(SEEDS), num_sessions, num_neurons, n_groups), np.nan)
all_ig_col = np.full((len(SEEDS), num_sessions, num_neurons, 17),       np.nan)

for seed_idx, seed in enumerate(SEEDS):
    ckpt_path = os.path.join(output_dir, f'ig_cebra_checkpoint_seed{seed}.npz')

    if os.path.exists(ckpt_path) and not args.force_recompute:
        d = np.load(ckpt_path)
        all_ig_sem[seed_idx] = d['sem']
        all_ig_col[seed_idx] = d['col']
        print(f"Seed {seed}: loaded from checkpoint.")
        continue

    split_map = np.load(
        os.path.join(splits_dir, f'split_seed{seed}.npy'), allow_pickle=True
    ).item()

    for s_idx, sess in enumerate(sessions):
        test_trials  = [int(i) for i in split_map[sess]]
        all_trials   = list(ds[sess]['data'].keys())
        train_trials = [t for t in all_trials if t not in test_trials]

        Xtr, Ytr = get_arrays(sess, train_trials)
        Xte, Yte = get_arrays(sess, test_trials)
        if len(Xtr) == 0 or len(Xte) == 0:
            continue

        # Precompute train/test windows
        wins_tr = build_windows(Xtr).astype(np.float32)   # (T_tr, 17, 10)
        wins_te = build_windows(Xte).astype(np.float32)   # (T_te, 17, 10)

        for n_idx in range(num_neurons):
            if not valid[s_idx, n_idx]:
                continue

            mpath = os.path.join(models_root, f'seed{seed}',
                                 f'session_{s_idx:02d}_neuron_{n_idx:02d}.pt')
            if not os.path.exists(mpath):
                continue

            encoder, _, _ = load_encoder(mpath, device=device)
            encoder.eval()

            # Fit Ridge probe on train embeddings
            with torch.no_grad():
                wins_tr_t = torch.tensor(wins_tr, dtype=torch.float32, device=device)
                emb_tr = []
                for b in range(0, len(wins_tr_t), IG_BATCH):
                    z = encoder(wins_tr_t[b:b + IG_BATCH])
                    if z.dim() == 3:
                        z = z.squeeze(-1)
                    emb_tr.append(z.cpu().numpy())
                emb_tr = np.concatenate(emb_tr, axis=0)

            y_tr = Ytr[:, n_idx]
            ridge = Ridge(alpha=RIDGE_ALPHA)
            ridge.fit(emb_tr, y_tr)

            # Build differentiable model
            model_ig = EncoderRidge(encoder, ridge.coef_, ridge.intercept_).to(device)
            model_ig.eval()

            ig_sem, abs_col = compute_cebra_ig(model_ig, wins_te)
            all_ig_sem[seed_idx, s_idx, n_idx] = ig_sem
            all_ig_col[seed_idx, s_idx, n_idx] = abs_col

            del encoder, model_ig, emb_tr
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        print(f"  Seed {seed}  S{s_idx+1:02d}/{num_sessions}  "
              f"valid={(~np.isnan(all_ig_sem[seed_idx, s_idx, :, 0])).sum()}")

    np.savez(ckpt_path, sem=all_ig_sem[seed_idx], col=all_ig_col[seed_idx])
    print(f"Seed {seed} done — checkpoint saved to {ckpt_path}")

# ─── aggregate over seeds ─────────────────────────────────────────────────────
with np.errstate(all='ignore'):
    ig_sem_final = np.nanmedian(all_ig_sem, axis=0)   # (29, 23, 11)
    ig_col_final = np.nanmedian(all_ig_col, axis=0)   # (29, 23, 17)

np.save(ig_sem_out_path, ig_sem_final)
np.save(ig_col_out_path, ig_col_final)
print(f"\nSaved:\n  {ig_sem_out_path}  shape={ig_sem_final.shape}"
      f"\n  {ig_col_out_path}  shape={ig_col_final.shape}")

# ─── quick sanity check ───────────────────────────────────────────────────────
n_valid = np.sum(~np.isnan(ig_sem_final[:, :, 0]) & valid)
print(f"\nNon-NaN entries in ig_sem_final: {n_valid} / {valid.sum()} valid pairs")

top_group = np.nanargmax(np.nanmean(ig_sem_final, axis=(0, 1)))
print(f"Top semantic group (mean attribution): {group_names[top_group]} (idx {top_group})")
print("Done.")
