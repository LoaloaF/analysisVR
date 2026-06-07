#!/usr/bin/env python3
"""
eval_hidden_probe.py

Intervention-based hidden-layer probe.

Simply decoding from the unmodified hidden layer is trivially high because
embed() sees all 17 inputs (including one-hot columns) — any linear probe
can read them back directly.

The correct test zeros out the target one-hot columns in X *before* calling
embed(), then tries to decode the class from the resulting hidden layer.

  acc_full    = decode(h(X_full))       — sanity check, expected ~1.0
  acc_ablated = decode(h(X_zeroed))     — the real test:
                    HIGH (>>1/3):  model routes class through kinematics
                    LOW  (≈1/3):   model does not encode class at all

GPV cross-check: GPV measures R² drop when one-hot is zeroed at the *output*.
  GPV≈0 + acc_ablated HIGH  -> model routes via kinematics, attribution misses it
  GPV≈0 + acc_ablated CHANCE -> model never encoded the class at all

Pairs:
  E07 x cue_visible     (control: GPV is high, expect acc_ablated to drop)
  E23 x upcoming_choice (puzzle:  GPV~0, ablation reveals mechanism)
"""

import os, sys, shutil, pickle, warnings
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
import pandas as pd
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from models import MLP
from figure_style import (
    FIG, DPI, FONT, LINE, MARKER,
    MODEL_COLORS, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

# ─── config ──────────────────────────────────────────────────────────────────
SEEDS             = [42, 43, 44, 45, 46]
R2_THRESHOLD      = 0.01
N_FOLDS           = 5
HIDDEN_SIZE       = 64
NUM_HIDDEN_LAYERS = 2
OUTPUT_SIZE       = 1
CONTINUOUS_IDXS   = list(range(7))   # indices 0-6: continuous kinematic features

PAIRS = [
    {
        'name':        'E07_cue_visible',
        'label':       'E07 x cue_visible',
        'ens_idx':     6,
        'feat_idxs':   [7, 8, 9],
        'class_names': ['no_cue', 'cue1', 'cue2'],
    },
    {
        'name':        'E23_upcoming_choice',
        'label':       'E23 x upcoming_choice',
        'ens_idx':     22,
        'feat_idxs':   [10, 11, 12],
        'class_names': ['skip', 'baseline', 'stop'],
    },
]

models_root = "./models/mlps/ensembles"
cache_path  = "./outputs/session_dataset_ensembles.pkl"
all_r2_path = "./outputs/mlps/ensembles_multiseed/all_r2.npy"
gpv_path    = "./outputs/mlps/ensembles_multiseed/importance_global_pv_semantic.npy"

output_dir  = "./outputs/mlps/hidden_probe"
desktop_dir = "/mnt/c/Users/amits/Desktop/hidden_probe"
for d in (output_dir, desktop_dir):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

device = torch.device('cpu')

# ─── helpers ─────────────────────────────────────────────────────────────────
def load_model(s_idx, n_idx, seed):
    mpath = os.path.join(models_root, f"seed{seed}",
                         f"session_{s_idx:02d}_neuron_{n_idx:02d}.pt")
    if not os.path.exists(mpath):
        return None
    sd = torch.load(mpath, map_location=device)
    in_size = sd['fc.0.weight'].shape[1]
    m = MLP(in_size, HIDDEN_SIZE, NUM_HIDDEN_LAYERS, OUTPUT_SIZE).to(device)
    m.load_state_dict(sd)
    m.eval()
    return m


def get_embeddings(models, X_np):
    """Average embed() across seeds -> (T, 64)."""
    X_t = torch.tensor(X_np, dtype=torch.float32)
    embeds = []
    with torch.no_grad():
        for m in models:
            embeds.append(m.embed(X_t).numpy())
    return np.mean(embeds, axis=0)


def decode_cv(h, y, n_folds=N_FOLDS):
    """Balanced accuracy via stratified k-fold logistic regression."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < n_folds:
        return np.nan
    scaler = StandardScaler()
    h_s = scaler.fit_transform(h)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0,
                             solver='lbfgs')
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(h_s, y):
        clf.fit(h_s[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(h_s[te])))
    return float(np.mean(accs))


def cohens_d_max(y_neural, y_cat):
    """Max pairwise Cohen's d over class means of neural activity."""
    classes = np.unique(y_cat)
    d_max = 0.0
    for i, ca in enumerate(classes):
        for cb in classes[i + 1:]:
            a = y_neural[y_cat == ca]
            b = y_neural[y_cat == cb]
            pooled = np.sqrt((np.var(a) + np.var(b)) / 2.0)
            if pooled > 0:
                d_max = max(d_max, abs(a.mean() - b.mean()) / pooled)
    return d_max


def savefig(fname):
    savefig_manifest(plt.gcf(), fname, [output_dir, desktop_dir])


# ─── load shared data ────────────────────────────────────────────────────────
all_r2  = np.load(all_r2_path)      # (5, 29, 23)
mean_r2 = all_r2.mean(axis=0)       # (29, 23)
valid   = (~np.any(np.isnan(all_r2), axis=0)) & (mean_r2 >= R2_THRESHOLD)

gpv = np.load(gpv_path)             # (29, 23, 11) — group importance per session/ensemble
# semantic group indices: 7=cue_visible, 8=upcoming_choice

with open(cache_path, 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())
print(f"Loaded {len(sessions)} sessions\n")

# ─── main analysis loop ───────────────────────────────────────────────────────
all_results = {}

for pair in PAIRS:
    name      = pair['name']
    label     = pair['label']
    ens_idx   = pair['ens_idx']
    feat_idxs = pair['feat_idxs']
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    rows = []
    for s_idx, sess in enumerate(sessions):
        if not valid[s_idx, ens_idx]:
            continue

        # Build full-session arrays
        X_list, cat_list, act_list = [], [], []
        for t in ds[sess]['data']:
            X = ds[sess]['data'][t]                         # (T, 17)
            y_cat = np.argmax(X[:, feat_idxs], axis=1)     # 0/1/2
            y_act = ds[sess]['labels'][t][:, ens_idx]
            X_list.append(X)
            cat_list.append(y_cat)
            act_list.append(y_act)

        X_full   = np.concatenate(X_list,   axis=0)  # (T_total, 17)
        y_cat    = np.concatenate(cat_list, axis=0)  # (T_total,)
        y_act    = np.concatenate(act_list, axis=0)  # (T_total,)

        classes, counts = np.unique(y_cat, return_counts=True)
        if len(classes) < 2 or counts.min() < N_FOLDS:
            print(f"  S{s_idx:02d}: skipped — fewer than {N_FOLDS} samples in a class")
            continue

        # Build ablated input: zero out the target one-hot columns
        X_ablated = X_full.copy()
        X_ablated[:, feat_idxs] = 0.0

        # Load models
        mdls = [load_model(s_idx, ens_idx, seed) for seed in SEEDS]
        mdls = [m for m in mdls if m is not None]
        if not mdls:
            continue

        # Hidden representations
        h_full    = get_embeddings(mdls, X_full)    # (T, 64) — all inputs
        h_ablated = get_embeddings(mdls, X_ablated) # (T, 64) — one-hot zeroed

        # Cohen's d (data effect size: E neural activity by class)
        cd = cohens_d_max(y_act, y_cat)

        # GPV for this (session, ensemble, group) — semantic group idx same as pair order
        # cue_visible = group 7, upcoming_choice = group 8
        grp_idx = 7 if 'cue' in name else 8
        gpv_val = float(gpv[s_idx, ens_idx, grp_idx])

        # Decode accuracies
        acc_full    = decode_cv(h_full,    y_cat)   # sanity check (trivially high)
        acc_ablated = decode_cv(h_ablated, y_cat)   # key test: does model route through kin?

        n_cls = {c: int((y_cat == c).sum()) for c in [0, 1, 2]}
        print(f"  S{s_idx:02d}  d={cd:.3f}  gpv={gpv_val:.4f}  "
              f"full={acc_full:.3f}  ablated={acc_ablated:.3f}  n={n_cls}")

        rows.append({
            's_idx':       s_idx,
            'session':     sess,
            'cohens_d':    cd,
            'gpv':         gpv_val,
            'acc_full':    acc_full,
            'acc_ablated': acc_ablated,
            'n_class0':    n_cls[0],
            'n_class1':    n_cls[1],
            'n_class2':    n_cls[2],
        })

    df = pd.DataFrame(rows).sort_values('cohens_d').reset_index(drop=True)
    df.to_csv(os.path.join(output_dir, f'{name}_results.csv'), index=False)
    all_results[name] = df
    print(f"  -> {len(df)} valid sessions")

    # ── Plot 1: acc_ablated vs Cohen's d (per session bar chart) ───────────
    n = len(df)
    if n == 0:
        continue
    chance = 1 / 3
    fig, ax = plt.subplots(figsize=(max(8, n * 0.55), 4))
    apply_style(fig, ax)
    x = np.arange(n)
    w = 0.35

    ax.bar(x - w/2, df['acc_ablated'], w,
           label='Hidden (one-hot ablated)', color='#E53935', alpha=0.85)
    ax.bar(x + w/2, df['acc_full'],    w,
           label='Hidden (full input)',      color='#90A4AE', alpha=0.6)
    ax.axhline(chance, color='gray', lw=1.2, ls='--', label=f'Chance ({chance:.2f})')

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{r['s_idx']:02d}\nd={r['cohens_d']:.2f}"
                        for _, r in df.iterrows()], fontsize=7)
    ax.set_ylabel('Balanced accuracy (3-class)')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{label} — hidden-layer probe after ablating one-hot input')
    ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    savefig(f'{name}_bars.png')

    # ── Plot 2: scatter — Cohen's d vs acc_ablated, coloured by GPV ────────
    fig, ax = plt.subplots(figsize=(6, 5))
    apply_style(fig, ax)
    sc = ax.scatter(df['cohens_d'], df['acc_ablated'],
                    c=df['gpv'], cmap='viridis', s=80, zorder=3,
                    vmin=0, vmax=df['gpv'].max() if df['gpv'].max() > 0 else 1)
    plt.colorbar(sc, ax=ax, label='GPV attribution')
    ax.axhline(chance, color='gray', lw=1.2, ls='--', label='Chance')

    # Add session labels
    for _, r in df.iterrows():
        ax.annotate(f"S{r['s_idx']:02d}",
                    (r['cohens_d'], r['acc_ablated']),
                    fontsize=7, ha='left', va='bottom',
                    xytext=(2, 2), textcoords='offset points')

    # Spearman rho
    mask = df['acc_ablated'].notna()
    if mask.sum() >= 4:
        rho, p = spearmanr(df.loc[mask, 'cohens_d'], df.loc[mask, 'acc_ablated'])
        ax.text(0.05, 0.95, f'rho = {rho:.2f}, p = {p:.3f}',
                transform=ax.transAxes, fontsize=9, va='top')

    ax.set_xlabel("Cohen's d  (data: E neural activity by class)")
    ax.set_ylabel('Balanced acc. after ablating one-hot (3-class)')
    ax.set_title(f'{label}: does hidden layer route class via kinematics?')
    ax.legend(fontsize=8)
    plt.tight_layout()
    savefig(f'{name}_scatter.png')

    print(f"  -> saved {name}_bars.png, {name}_scatter.png")


# ─── comparison plot: E07 vs E23 side by side ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
apply_style(fig, axes)
chance = 1 / 3

for ax, pair in zip(axes, PAIRS):
    df = all_results.get(pair['name'], pd.DataFrame())
    if df.empty:
        ax.set_title(pair['label'] + ' (no valid sessions)')
        continue

    sc = ax.scatter(df['cohens_d'], df['acc_ablated'],
                    c=df['gpv'], cmap='viridis', s=70, zorder=3,
                    vmin=0, vmax=max(df['gpv'].max(), 0.001))
    plt.colorbar(sc, ax=ax, label='GPV')
    ax.axhline(chance, color='gray', lw=1.1, ls='--', label='Chance (1/3)')

    for _, r in df.iterrows():
        ax.annotate(f"S{r['s_idx']:02d}",
                    (r['cohens_d'], r['acc_ablated']),
                    fontsize=7, ha='left', va='bottom',
                    xytext=(2, 2), textcoords='offset points')

    ax.set_xlabel("Cohen's d")
    ax.set_ylabel('Acc. after ablating one-hot')
    ax.set_title(pair['label'])
    ax.set_ylim(0.1, 1.05)
    ax.legend(fontsize=8)

plt.suptitle("Does the hidden layer route class through kinematics? (ablation probe)", fontsize=10)
plt.tight_layout()
savefig('comparison_scatter.png')
print("\nSaved comparison_scatter.png")

# ─── summary table ────────────────────────────────────────────────────────────
for pair in PAIRS:
    df = all_results.get(pair['name'], pd.DataFrame())
    if df.empty:
        continue
    print(f"\n{pair['label']}")
    print(df[['s_idx', 'cohens_d', 'gpv', 'acc_full', 'acc_ablated']].to_string(index=False))

print("\nDone.")
