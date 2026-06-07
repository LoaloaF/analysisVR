#!/usr/bin/env python3
"""
eval_position_ablation.py

For the same top-4 (session, ensemble) pairs shown in position_tuning.png,
train a position-only MLP (1 input feature: track position) and overlay
its predicted tuning curve on the actual binned data.

Shows: even though the full model attributes little to position, a position-only
MLP still recovers the tuning curve shape — confirming that position IS
predictive, it's just masked by co-varying features in the full model.

Figure layout: 2 rows × 4 panels
  Row 1: actual tuning curves (data, mean ± SEM in red) overlaid with the
          position-only MLP prediction (dark line) and full-model prediction
          from binned test-set outputs (dashed)
  Row 2: per-pair R² bars — full model vs position-only vs shuffled null
"""
import os, sys, pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import (
    FIG, DPI, FONT, AXIS_LABELS,
    apply_style, add_footnote, savefig_manifest,
)

# ── Config ─────────────────────────────────────────────────────────────────────
SEEDS      = [42, 43, 44]   # 3 seeds — enough for variance estimate
N_PAIRS    = 4
N_BINS     = 20
MIN_BIN_PTS = 10
POS_IDX    = 6          # frame_position in the 17-feature input
N_EPOCHS   = 300        # full-batch: fast even on CPU
LR         = 3e-3
HIDDEN     = [32, 32]
N_SHUFFLE  = 5
BATCH      = 512        # unused in full-batch mode

base   = os.path.dirname(os.path.abspath(__file__))
root   = os.path.join(base, '..')
mdir   = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [mdir, '/mnt/c/Users/amits/Desktop']

# Cache key encodes all hyperparameters that affect the results
CACHE_KEY  = f'np{N_PAIRS}_s{"_".join(map(str,SEEDS))}_ep{N_EPOCHS}_h{"_".join(map(str,HIDDEN))}_sh{N_SHUFFLE}'
CACHE_FILE = os.path.join(mdir, f'position_ablation_cache_{CACHE_KEY}.pkl')

# ── Load data ──────────────────────────────────────────────────────────────────
with open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb') as f:
    ds = pickle.load(f)
session_ids = list(ds.keys())

all_r2  = np.load(os.path.join(mdir, 'all_r2.npy'))   # (5, 29, 23)
mean_r2 = np.nanmean(all_r2, axis=0)                  # (29, 23)
n_sessions, n_ensembles = mean_r2.shape

# Same pair selection as eval_position_tuning.py: top N by mean R²
flat_order = np.argsort(mean_r2.ravel())[::-1]
top_pairs  = []
for fi in flat_order:
    s, e = fi // n_ensembles, fi % n_ensembles
    if np.isfinite(mean_r2[s, e]) and mean_r2[s, e] >= 0.01:
        top_pairs.append((s, e))
    if len(top_pairs) == N_PAIRS:
        break

print("Selected pairs:")
for s, e in top_pairs:
    print(f"  S{s+1:02d} E{e+1:02d}  R²={mean_r2[s, e]:.3f}")

# Full-batch on CPU is ~9× faster than mini-batch CUDA for small data (22k pts)
device = torch.device('cpu')
print(f"device={device} (full-batch mode)")

# ── MLP ────────────────────────────────────────────────────────────────────────
class SmallMLP(nn.Module):
    def __init__(self, in_dim=1, hidden=None, out_dim=1):
        super().__init__()
        if hidden is None:
            hidden = [64, 64]
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


def train_position_mlp(pos_train, y_train, pos_test, y_test, seed):
    torch.manual_seed(seed)
    model = SmallMLP(in_dim=1, hidden=HIDDEN, out_dim=1).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    Xtr = torch.tensor(pos_train[:, None], dtype=torch.float32, device=device)
    Ytr = torch.tensor(y_train[:, None],   dtype=torch.float32, device=device)
    Xte = torch.tensor(pos_test[:, None],  dtype=torch.float32, device=device)

    for ep in range(N_EPOCHS):
        loss = loss_fn(model(Xtr), Ytr)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xte).cpu().numpy().ravel()
    r2 = float(r2_score(y_test, pred))
    return model, r2, pred


def tuning_curve(pos, act, n_bins=N_BINS, min_pts=MIN_BIN_PTS):
    edges   = np.linspace(np.nanpercentile(pos, 1), np.nanpercentile(pos, 99), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means, sems = np.full(n_bins, np.nan), np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = (pos >= edges[b]) & (pos < edges[b + 1])
        pts = act[m]
        if len(pts) >= min_pts:
            means[b] = np.mean(pts)
            sems[b]  = np.std(pts) / np.sqrt(len(pts))
    return centers, means, sems, edges


# ── Per-pair analysis (with cache) ────────────────────────────────────────────
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'rb') as f:
        results = pickle.load(f)
    print(f'Loaded results from cache: {CACHE_FILE}')
else:
    results = []

# Only compute pairs not already in cache
cached_pairs = {(r['s'], r['e']) for r in results if r is not None}
missing_pairs = [p for p in top_pairs if p not in cached_pairs]
if missing_pairs:
    print(f'Training {len(missing_pairs)} pair(s) not in cache...')

for s_idx, e_idx in missing_pairs:
    sess_id = session_ids[s_idx]
    sd      = ds[sess_id]
    all_t   = list(sd['data'].keys())

    # Load split for seed 42
    split_path = os.path.join(root, 'splits', f'split_seed42.npy')
    split = np.load(split_path, allow_pickle=True).item()
    test_trials  = [t for t in split.get(sess_id, []) if t in sd['data']]
    train_trials = [t for t in all_t if t not in test_trials]

    def get_arrays(trial_list):
        if not trial_list:
            return np.array([]), np.array([])
        Xs = np.concatenate([sd['data'][t]   for t in trial_list]).astype(float)
        Ys = np.concatenate([sd['labels'][t] for t in trial_list]).astype(float)
        return Xs[:, POS_IDX], Ys[:, e_idx]

    pos_tr, y_tr = get_arrays(train_trials)
    pos_te, y_te = get_arrays(test_trials)

    if len(pos_tr) < 50 or len(pos_te) < 20:
        print(f"  S{s_idx+1:02d} E{e_idx+1:02d}: insufficient data, skipping")
        results.append(None); continue

    # Train position-only MLPs across seeds
    pos_r2s = []
    best_model, best_r2, best_pred = None, -np.inf, None
    for seed in SEEDS:
        # use all trials for training, leave test fixed from seed 42
        model, r2, pred = train_position_mlp(pos_tr, y_tr, pos_te, y_te, seed)
        pos_r2s.append(r2)
        if r2 > best_r2:
            best_r2, best_model, best_pred = r2, model, pred

    pos_r2_mean = float(np.mean(pos_r2s))
    pos_r2_std  = float(np.std(pos_r2s))

    # Null distribution (shuffle labels)
    null_r2s = []
    rng = np.random.default_rng(0)
    for _ in range(N_SHUFFLE):
        y_sh = rng.permutation(y_tr)
        _, r2_sh, _ = train_position_mlp(pos_tr, y_sh, pos_te, y_te, 42)
        null_r2s.append(r2_sh)
    null_r2_mean = float(np.mean(null_r2s))

    # Full-session actual tuning curve
    pos_all, y_all = get_arrays(all_t)
    centers, means, sems, edges = tuning_curve(pos_all, y_all)

    # Predicted tuning curve from best position-only model
    grid_pos = torch.tensor(centers[:, None], dtype=torch.float32, device=device)
    best_model.eval()
    with torch.no_grad():
        pred_curve = best_model(grid_pos).cpu().numpy().ravel()

    full_r2 = mean_r2[s_idx, e_idx]
    print(f"  S{s_idx+1:02d} E{e_idx+1:02d}  full={full_r2:.3f}  "
          f"pos-only={pos_r2_mean:.3f}±{pos_r2_std:.3f}  null={null_r2_mean:.3f}")

    results.append(dict(
        s=s_idx, e=e_idx,
        centers=centers, means=means, sems=sems,
        pred_curve=pred_curve,
        full_r2=full_r2,
        pos_r2=pos_r2_mean, pos_r2_std=pos_r2_std,
        null_r2=null_r2_mean,
    ))

# Save cache after any new training
if missing_pairs:
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved cache → {CACHE_FILE}')

# Re-order to match top_pairs (cache may have different order)
pair_to_result = {(r['s'], r['e']): r for r in results if r is not None}
results = [pair_to_result.get(p, None) for p in top_pairs]

# ── Figure ─────────────────────────────────────────────────────────────────────
valid_res = [r for r in results if r is not None]
N = len(valid_res)

fig, axes = plt.subplots(2, N, figsize=FIG.FULL,
                          gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.55,
                                       'wspace': 0.40})
apply_style(fig, axes.ravel())

DATA_CLR = '#d62728'   # actual data (tuning curve)
PRED_CLR = '#1f77b4'   # position-only MLP (shared between curve and bar)
FULL_CLR = '#555555'   # full model R² reference bar (not shown as curve)
NULL_CLR = '#aaaaaa'

for col, r in enumerate(valid_res):
    # ── Row 0: tuning curves ────────────────────────────────────────────────
    ax = axes[0, col]
    valid = ~np.isnan(r['means'])

    ax.fill_between(r['centers'][valid],
                    r['means'][valid] - r['sems'][valid],
                    r['means'][valid] + r['sems'][valid],
                    color=DATA_CLR, alpha=0.25)
    ax.plot(r['centers'][valid], r['means'][valid],
            color=DATA_CLR, lw=1.8, label='Actual')
    ax.plot(r['centers'][valid], r['pred_curve'][valid],
            color=PRED_CLR, lw=1.6, linestyle='--', label='Position-only MLP')
    ax.axhline(0, color='#888', lw=0.5, linestyle=':')

    ax.set_xlabel(AXIS_LABELS['position'], fontsize=FONT.LABEL - 2)
    if col == 0:
        ax.set_ylabel('z-scored activity', fontsize=FONT.LABEL - 2)
    ax.tick_params(axis='x', labelsize=FONT.TICK - 2, rotation=30)
    ax.tick_params(axis='y', labelsize=FONT.TICK - 2)
    ax.text(0.97, 0.97, f"S{r['s']+1:02d} E{r['e']+1:02d}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FONT.ANNOTATION - 1, color='dimgray')
    if col == 0:
        ax.legend(fontsize=FONT.LEGEND - 1, frameon=False, loc='lower left')

    # ── Row 1: R² bars ──────────────────────────────────────────────────────
    ax = axes[1, col]
    bars = ax.bar([0, 1, 2],
                  [r['full_r2'], r['pos_r2'], r['null_r2']],
                  color=[FULL_CLR, PRED_CLR, NULL_CLR],
                  width=0.55, alpha=0.85)
    ax.errorbar(1, r['pos_r2'], yerr=r['pos_r2_std'],
                fmt='none', color='black', capsize=3, lw=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Full', 'Pos.\nonly', 'Null'],
                       fontsize=FONT.TICK - 1)
    ax.set_ylim(bottom=min(0, r['null_r2'] - 0.02))
    if col == 0:
        ax.set_ylabel('R²', fontsize=FONT.LABEL - 2)
    ax.tick_params(axis='y', labelsize=FONT.TICK - 2)

fig.subplots_adjust(bottom=0.22)
add_footnote(fig,
    f"Same top {N_PAIRS} pairs as position_tuning.png; "
    f"position-only MLP: {len(SEEDS)} seeds × {N_EPOCHS} epochs; "
    f"null: {N_SHUFFLE} shuffles")

savefig_manifest(fig, 'position_ablation.png', OUT_DIRS)
print("Generated position_ablation.png")
