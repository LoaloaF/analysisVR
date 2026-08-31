#!/usr/bin/env python3
"""
eval_cross_ensemble_prediction.py

Asks: if the model was trained to predict ensemble e_i, how well does its
embedding predict ensemble e_j's neural activity?

For each (session, source_ensemble, model_type) on seed=42:
  1. Load model, embed X_test  → H shape (T, 64)
  2. Ridge-regress H → Y_test[:, e_j] for every target ensemble e_j
  3. Record R² per (session, source_ens, target_ens, model)

The diagonal (e_i == e_j) is the within-target R² (reference).
The off-diagonal is cross-target generalization.

Outputs:
  outputs/mlps/ensembles_multiseed/cross_ensemble_prediction.npy  (records list)
  outputs/mlps/ensembles_multiseed/cross_ensemble_prediction.png
"""
import os, sys, pickle, warnings
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.models import MLP
from utils.load_encoder import build_windows, embed_windows, load_encoder
from utils.figure_style import FIG, DPI, FONT, apply_style, add_footnote, savefig_manifest

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, '..')
mdir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
cdir = os.path.join(root, 'outputs', 'cebra_comparison')

REF_SEED    = 42
R2_THR      = 0.1
RIDGE_ALPHA = 1.0
CV_FOLDS    = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

# ── Load data ──────────────────────────────────────────────────────────────────
with open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb') as f:
    ds = pickle.load(f)
session_ids = list(ds.keys())
n_sess      = len(session_ids)

splits = np.load(os.path.join(root, 'splits', f'split_seed{REF_SEED}.npy'),
                 allow_pickle=True).item()

mlp_r2 = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)
cc_r2  = np.nanmean(np.load(os.path.join(root, 'outputs', 'cebra_64d_eval',
                                          'ensembles', 'all_r2.npy')), axis=0)
cp_r2  = np.nanmean(np.load(os.path.join(root, 'outputs', 'cebra_pred_64d_eval',
                                          'ensembles', 'all_r2.npy')), axis=0)
n_ens = mlp_r2.shape[1]
R2_MATS = {'mlp': mlp_r2, 'cc': cc_r2, 'cp': cp_r2}


def get_test_Xy(sess_id):
    sd     = ds[sess_id]
    trials = sorted(t for t in splits.get(sess_id, []) if t in sd['data'])
    if not trials:
        return None, None
    X = np.concatenate([sd['data'][t]   for t in trials], axis=0).astype(np.float32)
    Y = np.concatenate([sd['labels'][t] for t in trials], axis=0).astype(np.float32)
    return X, Y   # (T,17), (T,23)


def mlp_embed(s_idx, e_idx, X):
    path = os.path.join(root, 'models', 'mlps', 'ensembles',
                        f'seed{REF_SEED}',
                        f'session_{s_idx:02d}_neuron_{e_idx:02d}.pt')
    if not os.path.exists(path):
        return None
    model = MLP(17, 64, 2, 1).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        H = model.embed(torch.tensor(X, device=device, dtype=torch.float32)).cpu().numpy()
    del model; torch.cuda.empty_cache()
    return H


def tc_embed(arm, s_idx, e_idx, X):
    path = os.path.join(root, 'models', f'{arm}_64d', 'ensembles',
                        f'seed{REF_SEED}',
                        f'session_{s_idx:02d}_neuron_{e_idx:02d}.pt')
    if not os.path.exists(path):
        return None
    encoder, _, _ = load_encoder(path, device=str(device))
    wins = build_windows(X)
    H    = embed_windows(encoder, wins, device=str(device))
    del encoder; torch.cuda.empty_cache()
    return None if np.isnan(H).any() else H


def ridge_r2(H, y):
    """5-fold CV R² of ridge regression H → y (scalar target)."""
    if H is None or len(H) < CV_FOLDS * 2:
        return np.nan
    if np.std(y) < 1e-8:
        return np.nan
    scores = cross_val_score(Ridge(RIDGE_ALPHA), H, y, cv=CV_FOLDS, scoring='r2')
    return float(np.mean(np.clip(scores, 0, 1)))


# ══════════════════════════════════════════════════════════════════════════════
# Main loop (with cache)
# ══════════════════════════════════════════════════════════════════════════════
CACHE_FILE = os.path.join(mdir, 'cross_ensemble_prediction.npy')

if os.path.exists(CACHE_FILE):
    print(f'Loading cached results from {CACHE_FILE}', flush=True)
    records = list(np.load(CACHE_FILE, allow_pickle=True))
    print(f'Loaded {len(records)} records.', flush=True)
else:
    records = []
    for s_idx, sess_id in enumerate(session_ids):
        print(f'Session {s_idx+1}/{n_sess}', flush=True)
        X, Y = get_test_Xy(sess_id)
        if X is None or len(X) < CV_FOLDS * 2:
            continue

        for src_e in range(n_ens):
            for mk, arm in [('mlp', None), ('cc', 'cebra'), ('cp', 'cebra_pred')]:
                if R2_MATS[mk][s_idx, src_e] < R2_THR:
                    continue

                H = mlp_embed(s_idx, src_e, X) if mk == 'mlp' else tc_embed(arm, s_idx, src_e, X)
                if H is None:
                    continue

                T = min(len(H), len(Y))
                H_aligned = H[:T]
                Y_aligned = Y[:T]

                for tgt_e in range(n_ens):
                    r2 = ridge_r2(H_aligned, Y_aligned[:, tgt_e])
                    if np.isfinite(r2):
                        records.append(dict(
                            s_idx=s_idx, src_e=src_e, tgt_e=tgt_e,
                            model=mk, r2=r2,
                            is_self=(src_e == tgt_e),
                        ))

    np.save(CACHE_FILE, records, allow_pickle=True)
    print(f'\nSaved {len(records)} records.', flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd
df = pd.DataFrame(list(records))

# Require target ensemble to also pass R² threshold (fair comparison)
tgt_r2_ok = df.apply(
    lambda row: R2_MATS[row['model']][int(row['s_idx']), int(row['tgt_e'])] >= R2_THR,
    axis=1)
df = df[tgt_r2_ok].reset_index(drop=True)
print(f'After target R² filter: {len(df)} records', flush=True)

print('\n── Cross-ensemble prediction R² summary ──')
for mk in ['mlp', 'cc', 'cp']:
    sub  = df[df.model == mk]
    self_ = sub[sub.is_self].r2
    cross = sub[~sub.is_self].r2
    print(f'  {mk:4s}  self: n={len(self_):4d}  median={self_.median():.3f}  '
          f'| cross: n={len(cross):5d}  median={cross.median():.3f}  '
          f'| ratio={cross.median()/self_.median():.2f}')

# ══════════════════════════════════════════════════════════════════════════════
# Figure: violin — self vs cross-ensemble R² by architecture
# ══════════════════════════════════════════════════════════════════════════════
MODEL_COLORS = {'mlp': '#4CAF50', 'cc': '#2196F3', 'cp': '#FF9800'}
MODEL_LABELS = {'mlp': 'MLP', 'cc': 'TCC', 'cp': 'TCP'}

fig, ax = plt.subplots(figsize=(9.5, 4.2))
apply_style(fig, ax)

rng = np.random.default_rng(0)
positions = []
labels    = []
colors    = []
data_arrs = []
x = 0.0
for mk in ['mlp', 'cc', 'cp']:
    sub = df[df.model == mk]
    for tag, is_self in [('self', True), ('cross', False)]:
        arr = sub[sub.is_self == is_self].r2.values
        positions.append(x)
        labels.append(f'{MODEL_LABELS[mk]}\n{tag}')
        colors.append(MODEL_COLORS[mk])
        data_arrs.append(arr)
        x += 1.4
    x += 1.0  # gap between model groups

vp = ax.violinplot(data_arrs, positions=positions,
                   showmedians=True, showextrema=True, widths=0.65)
for pc, c in zip(vp['bodies'], colors):
    pc.set_facecolor(c); pc.set_alpha(0.4)
for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
    vp[part].set_color('#333'); vp[part].set_linewidth(0.8)

for arr, c, p in zip(data_arrs, colors, positions):
    if len(arr) == 0: continue
    jitter = rng.uniform(-0.1, 0.1, size=min(len(arr), 300))
    sample = arr if len(arr) <= 300 else rng.choice(arr, 300, replace=False)
    ax.scatter(p + jitter, sample, s=3, color=c, alpha=0.25, linewidths=0, zorder=3)
    med = float(np.nanmedian(arr))
    ax.text(p, med + 0.03, f'{med:.2f}', ha='center', va='bottom',
            fontsize=FONT.ANNOTATION - 1, fontweight='bold')

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=FONT.TICK, ha='center')
ax.set_ylabel('R² (embed. to neural)', fontsize=FONT.LABEL)
ax.set_ylim(0.0, 1.1)
ax.set_title('Self vs Cross-Ensemble Prediction', fontsize=FONT.LABEL)

add_footnote(fig,
    f'Seed={REF_SEED} test set; ridge regression (5-fold CV, α=1.0); '
    f'source model threshold R²≥{R2_THR}; TC models use 64d encoder')

OUT_DIRS = [mdir, cdir, '/mnt/c/Users/amits/Desktop']
savefig_manifest(fig, 'cross_ensemble_prediction.png', OUT_DIRS)
print('\nDone. Generated cross_ensemble_prediction.png')
