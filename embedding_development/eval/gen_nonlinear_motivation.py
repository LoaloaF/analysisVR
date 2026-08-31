"""
Simple motivation figure: best nonlinear (η²) vs best linear (Pearson r²) per unit.

One point per (session, ensemble) unit. For each unit we take the best-case
linear association (max Pearson r², over continuous features) and the best-case
nonlinear association (max η², decile bins, over continuous features), both on
the *same* response scale (fraction of ensemble-activity variance). Per feature
η² >= r² by construction (correlation ratio >= correlation coefficient), so the
informative quantity is the *magnitude* of the gap η² - r² (the nonlinear bonus),
not merely the fraction of points above the diagonal.

Pearson r² is computed here from the raw data on the identical (x, y) used for
η² in eval_ml_vs_naive.py (x = first column of each continuous group; y = the
ensemble activity), so it is directly comparable to the loaded η² array.

Output: nonlinear_motivation.png  (no caption / footnote)
"""
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import FIG, apply_style, savefig_manifest

# ─── PATHS ────────────────────────────────────────────────────────────────────
base     = os.path.dirname(os.path.abspath(__file__))
root     = os.path.join(base, '..')
mv_dir   = os.path.join(root, 'outputs', 'mlps', 'ml_vs_naive')
attr_dir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [mv_dir, '/mnt/c/Users/amits/Desktop']

# ─── LOAD ─────────────────────────────────────────────────────────────────────
eta2       = np.load(os.path.join(mv_dir, 'eta2.npy'))               # (29,23,11) η²
all_r2_mlp = np.load(os.path.join(attr_dir, 'all_r2.npy'))          # (5,29,23)

with open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())
with open(os.path.join(attr_dir, 'semantic_groups.pkl'), 'rb') as f:
    semantic_groups = pickle.load(f)               # list of (name, [col_indices])

R2_THRESHOLD = 0.01
mean_r2_mlp  = all_r2_mlp.mean(axis=0)                               # (29,23)
valid        = (~np.any(np.isnan(all_r2_mlp), axis=0)) & (mean_r2_mlp >= R2_THRESHOLD)

CONT_GROUPS = list(range(7))   # continuous features only
n_sess, n_ens = valid.shape

# ─── Pearson r² per unit/feature, on the SAME (x, y) η² uses ───────────────────
# (fraction of ensemble-activity variance explained by a single-feature linear fit)
pearson_r2 = np.full((n_sess, n_ens, len(CONT_GROUPS)), np.nan)
for s_idx, sess in enumerate(sessions):
    X_full = np.concatenate([ds[sess]['data'][t]   for t in ds[sess]['data']], axis=0)
    Y_full = np.concatenate([ds[sess]['labels'][t] for t in ds[sess]['data']], axis=0)
    for n_idx in range(n_ens):
        if not valid[s_idx, n_idx]:
            continue
        y = Y_full[:, n_idx].astype(np.float64)
        if np.std(y) < 1e-12:
            continue
        for gi, g_idx in enumerate(CONT_GROUPS):
            x = X_full[:, semantic_groups[g_idx][1][0]].astype(np.float64)
            if np.std(x) < 1e-12:
                continue
            r = np.corrcoef(x, y)[0, 1]
            pearson_r2[s_idx, n_idx, gi] = r * r

# best-per-unit over continuous features (both share Var(y) as denominator)
eta2_c    = eta2[:, :, CONT_GROUPS]
best_r2   = np.nanmax(pearson_r2, axis=2)          # (29,23)
best_eta2 = np.nanmax(eta2_c,     axis=2)

sel = valid & ~np.isnan(best_r2) & ~np.isnan(best_eta2)
x = best_r2[sel]
y = best_eta2[sel]
# Reported statistic: among signal-bearing units (either metric > 0.05), the
# median ratio of nonlinear η² to linear r². Median (not mean) because the ratio
# blows up where r² ≈ 0, which is common (linear captures almost nothing).
big   = (x > 0.05) | (y > 0.05)
ratio = y[big] / x[big]
print(f'{sel.sum()} units total;  {big.sum()} signal-bearing (either > 0.05)')
print(f'  among signal-bearing: nonlinear η² is a median {np.median(ratio):.1f}x '
      f'the linear r²  (median gap η²-r² = {np.median(y[big] - x[big]):.3f})')

# ─── PLOT ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=FIG.SQUARE)
apply_style(fig, [ax])

lim = max(x.max(), y.max()) * 1.06
ax.scatter(x, y, s=12, alpha=0.5, color='#2CA02C', linewidths=0, zorder=3)
ax.plot([0, lim], [0, lim], 'k--', lw=1.0, alpha=0.6, zorder=2)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_aspect('equal')
ax.set_xlabel('best linear  r²')
ax.set_ylabel('best nonlinear  η²')

savefig_manifest(fig, 'nonlinear_motivation.png', OUT_DIRS)
print('  Saved nonlinear_motivation.png')
