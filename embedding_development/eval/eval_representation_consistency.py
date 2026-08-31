#!/usr/bin/env python3
"""
eval_representation_consistency.py

Measures whether models hold the SAME information in their embeddings
(last hidden layer before the output linear projection).

Metric: for each (session, ensemble, model, seed), fit 11 ridge probes
from the embedding to each behavioral feature group → 11-dim R² profile
("decodability profile").  Two conditions are consistent if their profiles
agree: Spearman ρ close to 1 means both embeddings encode the same features.

Four comparison types, all using the same metric:
  cross-seed    — same model, same (s,e), different random seed
  cross-ensemble — same model, session, seed; different target ensembles
  cross-session  — same model, ensemble, seed; different recording sessions
  cross-model   — same (s,e,seed); MLP vs TempConv-Cont / TempConv-Pred

MLP embedding  : 64-dim last hidden layer  (model.embed())
TempConv embed : 8-dim encoder output      (load_encoder + embed_windows)

Outputs (all in outputs/mlps/ensembles_multiseed/):
  decodability_profiles_mlp.npy          shape (5, 29, 23, 11)  seeds×sess×ens×groups
  decodability_profiles_cebra_cont.npy   shape (5, 29, 23, 11)
  decodability_profiles_cebra_pred.npy   shape (5, 29, 23, 11)
  representation_consistency.png
"""
import os, sys, pickle, warnings
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.models import MLP
from utils.load_encoder import build_windows, embed_windows, load_encoder
from utils.figure_style import FIG, DPI, FONT, apply_style, add_footnote, savefig_manifest

base  = os.path.dirname(os.path.abspath(__file__))
root  = os.path.join(base, '..')
mdir  = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
cdir  = os.path.join(root, 'outputs', 'cebra_comparison')

SEEDS  = [42, 43, 44, 45, 46]
R2_THR = 0.1
CV_FOLDS = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Data ──────────────────────────────────────────────────────────────────────
with open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb') as f:
    ds = pickle.load(f)
session_ids = list(ds.keys())
n_sess      = len(session_ids)

with open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb') as f:
    sg = pickle.load(f)
group_names = [g for g, _ in sg]
group_cols  = [cols for _, cols in sg]
n_groups    = len(sg)

splits = {
    s: np.load(os.path.join(root, 'splits', f'split_seed{s}.npy'),
               allow_pickle=True).item()
    for s in SEEDS
}

mlp_r2 = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)   # (29,23)
cc_r2  = np.nanmean(np.load(os.path.join(root, 'outputs', 'cebra_64d_eval',
                                          'ensembles', 'all_r2.npy')), axis=0)
cp_r2  = np.nanmean(np.load(os.path.join(root, 'outputs', 'cebra_pred_64d_eval',
                                          'ensembles', 'all_r2.npy')), axis=0)

n_ens = mlp_r2.shape[1]

MODEL_SPECS = [
    ('mlp',  os.path.join(root, 'models', 'mlps',          'ensembles'),
     mlp_r2, 'decodability_profiles_mlp.npy'),
    ('cc',   os.path.join(root, 'models', 'cebra_64d',     'ensembles'),
     cc_r2,  'decodability_profiles_cebra_cont_64d.npy'),
    ('cp',   os.path.join(root, 'models', 'cebra_pred_64d','ensembles'),
     cp_r2,  'decodability_profiles_cebra_pred_64d.npy'),
]


def _get_test_data(sess_id, seed):
    """Return (X_test, Y_test) concatenated over test trials for this seed."""
    sd     = ds[sess_id]
    trials = splits[seed].get(sess_id, [])
    valid  = [t for t in trials if t in sd['data']]
    if not valid:
        return None, None
    X = np.concatenate([sd['data'][t]   for t in valid], axis=0).astype(np.float32)
    Y = np.concatenate([sd['labels'][t] for t in valid], axis=0).astype(np.float32)
    return X, Y


def _mlp_embed(model_dir, seed, s_idx, e_idx, X):
    """Forward pass through MLP → (T, 64) last hidden layer."""
    path = os.path.join(model_dir, f'seed{seed}',
                        f'session_{s_idx:02d}_neuron_{e_idx:02d}.pt')
    if not os.path.exists(path):
        return None
    model = MLP(X.shape[1], 64, 2, 1).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        H = model.embed(torch.tensor(X, device=device)).cpu().numpy()
    del model
    return H   # (T, 64)


def _tc_embed(model_dir, seed, s_idx, e_idx, X):
    """Forward pass through CEBRA encoder → (T, 64) embedding."""
    path = os.path.join(model_dir, f'seed{seed}',
                        f'session_{s_idx:02d}_neuron_{e_idx:02d}.pt')
    if not os.path.exists(path):
        return None
    encoder, _, _ = load_encoder(path, device=str(device))
    windows = build_windows(X)                 # (T, 17, 10)
    H = embed_windows(encoder, windows, device=str(device))  # (T, 64)
    del encoder
    torch.cuda.empty_cache()
    if np.isnan(H).any():
        return None
    return H


def _decodability_profile(H, X):
    """
    Fit ridge CV probes: H → each of 11 behavioral feature groups.
    Returns 11-dim R² vector (NaN for groups with zero variance).
    """
    profile = np.full(n_groups, np.nan)
    for g, cols in enumerate(group_cols):
        # For multi-column groups, predict the first column only
        # (categorical one-hot; all cols are informative but first is simplest)
        y = X[:, cols[0]]
        if np.std(y) < 1e-8:
            continue
        try:
            scores = cross_val_score(
                Ridge(alpha=1.0), H, y,
                cv=CV_FOLDS, scoring='r2',
            )
            profile[g] = float(np.mean(np.clip(scores, 0, 1)))
        except Exception:
            pass
    return profile


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Compute or load cached decodability profiles
# ══════════════════════════════════════════════════════════════════════════════

all_profiles = {}  # key: model_key → (5, 29, 23, 11) array

for model_key, model_dir, r2_mat, cache_name in MODEL_SPECS:
    cache_path = os.path.join(mdir, cache_name)
    if os.path.exists(cache_path):
        prof = np.load(cache_path)
        print(f'Loaded cached {cache_name}  valid={np.isfinite(prof).any(axis=-1).sum()}')
    else:
        prof = np.full((len(SEEDS), n_sess, n_ens, n_groups), np.nan)
        print(f'\nComputing {model_key} profiles …')
        embed_fn = _mlp_embed if model_key == 'mlp' else _tc_embed
        for si, seed in enumerate(SEEDS):
            print(f'  seed {seed}', end='', flush=True)
            for s_idx, sess_id in enumerate(session_ids):
                X_test, Y_test = _get_test_data(sess_id, seed)
                if X_test is None or len(X_test) < 20:
                    continue
                for e_idx in range(n_ens):
                    if r2_mat[s_idx, e_idx] < R2_THR:
                        continue
                    H = embed_fn(model_dir, seed, s_idx, e_idx, X_test)
                    if H is None or len(H) < 20:
                        continue
                    prof[si, s_idx, e_idx] = _decodability_profile(H, X_test)
                print('.', end='', flush=True)
            print()
        np.save(cache_path, prof)
        print(f'  Saved {cache_path}')
    all_profiles[model_key] = prof

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Compute Spearman ρ for each comparison type
# ══════════════════════════════════════════════════════════════════════════════

def _both_valid(p1, p2):
    """Both profiles have ≥4 non-NaN values for a stable Spearman ρ."""
    ok = np.isfinite(p1) & np.isfinite(p2)
    return ok.sum() >= 4, ok

def _rho(p1, p2):
    ok = np.isfinite(p1) & np.isfinite(p2)
    if ok.sum() < 4:
        return np.nan
    r, _ = spearmanr(p1[ok], p2[ok])
    return r


rho_results = {
    'Cross-seed':      [],
    'Cross-ensemble':  [],
    'Cross-session':   [],
    'Cross-model\n(MLP × TC-Cont)':  [],
    'Cross-model\n(MLP × TC-Pred)':  [],
}

prof_mlp = all_profiles['mlp']   # (5, 29, 23, 11)
prof_cc  = all_profiles['cc']
prof_cp  = all_profiles['cp']

# Cross-seed: same (s,e,model), different seeds
for model_key, prof in [('mlp', prof_mlp), ('cc', prof_cc), ('cp', prof_cp)]:
    for s_idx in range(n_sess):
        for e_idx in range(n_ens):
            for si in range(len(SEEDS)):
                for sj in range(si + 1, len(SEEDS)):
                    r = _rho(prof[si, s_idx, e_idx], prof[sj, s_idx, e_idx])
                    if np.isfinite(r):
                        rho_results['Cross-seed'].append(r)

# Cross-ensemble: same (session, model, seed), different ensembles
for model_key, prof in [('mlp', prof_mlp), ('cc', prof_cc), ('cp', prof_cp)]:
    for si in range(len(SEEDS)):
        for s_idx in range(n_sess):
            valid_ens = [e for e in range(n_ens) if np.any(np.isfinite(prof[si, s_idx, e]))]
            for i in range(len(valid_ens)):
                for j in range(i + 1, len(valid_ens)):
                    r = _rho(prof[si, s_idx, valid_ens[i]], prof[si, s_idx, valid_ens[j]])
                    if np.isfinite(r):
                        rho_results['Cross-ensemble'].append(r)

# Cross-session: same (ensemble, model, seed), different sessions
for model_key, prof in [('mlp', prof_mlp), ('cc', prof_cc), ('cp', prof_cp)]:
    for si in range(len(SEEDS)):
        for e_idx in range(n_ens):
            valid_sess = [s for s in range(n_sess) if np.any(np.isfinite(prof[si, s, e_idx]))]
            for i in range(len(valid_sess)):
                for j in range(i + 1, len(valid_sess)):
                    r = _rho(prof[si, valid_sess[i], e_idx], prof[si, valid_sess[j], e_idx])
                    if np.isfinite(r):
                        rho_results['Cross-session'].append(r)

# Cross-model: MLP vs TempConv-Cont / TempConv-Pred (seed-matched)
for si in range(len(SEEDS)):
    for s_idx in range(n_sess):
        for e_idx in range(n_ens):
            r = _rho(prof_mlp[si, s_idx, e_idx], prof_cc[si, s_idx, e_idx])
            if np.isfinite(r):
                rho_results['Cross-model\n(MLP × TC-Cont)'].append(r)
            r = _rho(prof_mlp[si, s_idx, e_idx], prof_cp[si, s_idx, e_idx])
            if np.isfinite(r):
                rho_results['Cross-model\n(MLP × TC-Pred)'].append(r)

for label, vals in rho_results.items():
    arr = np.array(vals)
    print(f'{label.replace(chr(10)," ")}: n={len(arr)} '
          f'median={np.nanmedian(arr):.3f} mean={np.nanmean(arr):.3f}')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Figure
# ══════════════════════════════════════════════════════════════════════════════

labels  = list(rho_results.keys())
data    = [np.array(rho_results[l]) for l in labels]
colors  = ['#4CAF50', '#9C27B0', '#FF5722', '#2196F3', '#FF9800']

fig, ax = plt.subplots(figsize=FIG.FULL)
apply_style(fig, ax)

rng = np.random.default_rng(0)
positions = list(range(len(labels)))

vp = ax.violinplot(data, positions=positions,
                   showmedians=True, showextrema=True, widths=0.6)
for pc, c in zip(vp['bodies'], colors):
    pc.set_facecolor(c); pc.set_alpha(0.45)
for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
    vp[part].set_color('#333'); vp[part].set_linewidth(0.9)

for i, (vals, c) in enumerate(zip(data, colors)):
    jitter = rng.uniform(-0.09, 0.09, size=len(vals))
    ax.scatter(i + jitter, vals, s=5, color=c, alpha=0.35, linewidths=0, zorder=3)
    med = float(np.median(vals))
    ax.text(i, med + 0.025, f'{med:.2f}', ha='center', va='bottom',
            fontsize=FONT.ANNOTATION - 1, fontweight='bold')
    ax.text(i, -0.07, f'n={len(vals)}', ha='center', va='top',
            fontsize=FONT.TICK - 2, color='#555',
            transform=ax.get_xaxis_transform())

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=FONT.TICK - 1)
ax.set_ylabel('Spearman ρ of behavioral decodability profiles', fontsize=FONT.LABEL)
ax.axhline(0,   color='#aaa', lw=0.7, linestyle=':')
ax.axhline(0.9, color='#555', lw=0.9, linestyle='--', label='ρ = 0.9')
ax.legend(fontsize=FONT.LEGEND, frameon=False)
ax.set_ylim(-0.15, 1.1)

add_footnote(fig,
    f'Decodability profile: 5-fold CV R² of ridge probe from embedding to each of '
    f'{n_groups} behavioral feature groups; pairs where both conditions R²≥{R2_THR}; '
    f'MLP embedding = 64-dim last hidden layer; TempConv embedding = 64-dim encoder output')

OUT_DIRS = [mdir, cdir, '/mnt/c/Users/amits/Desktop']
savefig_manifest(fig, 'representation_consistency.png', OUT_DIRS)
print('\nGenerated representation_consistency.png')
