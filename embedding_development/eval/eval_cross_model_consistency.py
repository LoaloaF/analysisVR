#!/usr/bin/env python3
"""
eval_cross_model_consistency.py

Two figures:

1. embedding_consistency_comparison.png  (replaces regen_consistency_comparison.py)
   Each model's cross-seed consistency heatmap uses its OWN valid ensembles
   (sorted by that model's mean consistency) instead of MLP's top-14.
   Shows ALL 23 ensembles with NaN hatching so per-model coverage is visible.
   Bottom violin uses each model's own valid pairs (already correct in data).

2. cross_model_attribution_consistency.png  (NEW)
   For every (session, ensemble) pair where BOTH models reach R²≥0.1,
   compute Spearman ρ between the two models' GPV attribution profiles
   (across the 11 semantic feature groups).
   Pairs: MLP×TempConv-Cont, MLP×TempConv-Pred.
   Presented as violin+jitter, same visual style as cross_ensemble_consistency.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import (
    FIG, DPI, FONT, apply_style, add_footnote, savefig_manifest,
)

root    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
mdir    = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
cdir    = os.path.join(root, 'outputs', 'cebra_comparison')
ccdir   = os.path.join(root, 'outputs', 'cebra_eval', 'ensembles')
cpdir   = os.path.join(root, 'outputs', 'cebra_pred_eval', 'ensembles')
OUT_DIRS = [mdir, cdir, '/mnt/c/Users/amits/Desktop']

R2_THRESH = 0.1

# ── Load R² and consistency matrices ──────────────────────────────────────────
mlp_r2 = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)   # (29,23)
cc_r2  = np.nanmean(np.load(os.path.join(ccdir, 'all_r2.npy')), axis=0)
cp_r2  = np.nanmean(np.load(os.path.join(cpdir, 'all_r2.npy')), axis=0)

mlp_cons = np.load(os.path.join(mdir, 'consistency_matrix_mlp.npy'))
cc_cons  = np.load(os.path.join(cdir, 'consistency_matrix_cebra_contrast.npy'))
cp_cons  = np.load(os.path.join(cdir, 'consistency_matrix_cebra_pred.npy'))

# ── Load GPV attribution arrays for cross-model ρ ─────────────────────────────
mlp_gpv = np.load(os.path.join(mdir,  'importance_global_pv_semantic.npy'))  # (29,23,11)
cc_gpv  = np.load(os.path.join(ccdir, 'importance_global_pv_semantic.npy'))
cp_gpv  = np.load(os.path.join(cpdir, 'importance_global_pv_semantic.npy'))

n_sess, n_ens = mlp_r2.shape
sess_labels   = [f'S{i+1:02d}' for i in range(n_sess)]
ens_labels    = [f'E{i+1:02d}' for i in range(n_ens)]

MODELS = [
    ('MLP',           mlp_cons, mlp_r2, '#4CAF50'),
    ('TempConv-Cont', cc_cons,  cc_r2,  '#2196F3'),
    ('TempConv-Pred', cp_cons,  cp_r2,  '#FF9800'),
]

NO_DATA = '#bbbbbb'
cmap    = plt.cm.Blues.copy()
cmap.set_bad(color=NO_DATA)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Cross-seed embedding consistency comparison
# Each model uses its OWN ensemble ordering (by that model's mean consistency)
# All 23 ensembles shown; NaN cells hatched
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(FIG.FULL[0], 7.5))
gs  = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0],
                       left=0.07, right=0.97, top=0.96, bottom=0.06,
                       hspace=0.42)
gs_top = gs[0].subgridspec(1, 4, wspace=0.06, width_ratios=[1, 1, 1, 0.05])

for col, (label, cons, r2, color) in enumerate(MODELS):
    ax = fig.add_subplot(gs_top[col])

    # Sort ensembles by THIS model's mean consistency (own ordering)
    own_mean  = np.nanmean(cons, axis=0)          # (23,)
    own_order = np.argsort(own_mean)[::-1]        # highest first
    sub       = cons[:, own_order].T              # (23, n_sess)

    masked = np.ma.array(sub, mask=np.isnan(sub))
    sns.heatmap(sub, ax=ax, cmap=cmap, vmin=0, vmax=1,
                xticklabels=sess_labels,
                yticklabels=[f'E{own_order[i]+1:02d}' for i in range(n_ens)] if col == 0 else [],
                cbar=False, linewidths=0, linecolor='none')

    for (i, j) in zip(*np.where(np.isnan(sub))):
        ax.add_patch(plt.Rectangle([j, i], 1, 1, fill=True, facecolor=NO_DATA,
                                   hatch='////', edgecolor='#999', lw=0.4, zorder=2))

    n_valid = int(np.isfinite(cons).sum())
    ax.set_title(f'{label}\n(n={n_valid} valid pairs)', fontsize=FONT.LABEL - 2,
                 fontweight='bold', pad=4)
    ax.set_xlabel('Session', fontsize=FONT.TICK - 1)
    # Show every other session to avoid label crowding across 29 sessions
    xt = ax.get_xticks()
    ax.set_xticks(xt[::2])
    ax.set_xticklabels([sess_labels[int(t)] for t in xt[::2] if int(t) < n_sess],
                       rotation=45, ha='right', fontsize=6)
    if col == 0:
        ax.set_ylabel('Ensembles (sorted by own consistency)', fontsize=FONT.TICK - 1)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    else:
        ax.set_ylabel('')

# Shared colorbar
cax  = fig.add_subplot(gs_top[3])
norm = mcolors.Normalize(vmin=0, vmax=1)
sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, cax=cax)
cb.set_label('Cross-seed Pearson r', fontsize=FONT.TICK - 1)
cb.ax.tick_params(labelsize=FONT.TICK - 2)

# Bottom violin
ax_v = fig.add_subplot(gs[1])
apply_style(fig, ax_v)
rng  = np.random.default_rng(0)

flat_data, flat_labels, flat_colors = [], [], []
for label, cons, r2, color in MODELS:
    vals = cons.ravel()
    vals = vals[np.isfinite(vals)]
    flat_data.append(vals)
    flat_labels.append(label)
    flat_colors.append(color)

parts = ax_v.violinplot(flat_data, positions=[0, 1, 2],
                        showmedians=True, showextrema=True, widths=0.55)
for pc, c in zip(parts['bodies'], flat_colors):
    pc.set_facecolor(c); pc.set_alpha(0.45)
for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
    parts[part].set_color('#333'); parts[part].set_linewidth(0.9)

for i, (vals, label, color) in enumerate(zip(flat_data, flat_labels, flat_colors)):
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    ax_v.scatter(i + jitter, vals, s=10, color=color, alpha=0.55, linewidths=0, zorder=3)
    med = float(np.median(vals))
    ax_v.text(i, med + 0.015, f'{med:.2f}', ha='center', va='bottom',
              fontsize=FONT.ANNOTATION - 1, fontweight='bold')
    ax_v.text(i, -0.04, f'n={len(vals)}', ha='center', va='top',
              fontsize=FONT.TICK - 2, color='#555',
              transform=ax_v.get_xaxis_transform())

ax_v.set_xticks([0, 1, 2])
ax_v.set_xticklabels(flat_labels, fontsize=FONT.TICK)
ax_v.set_ylabel('Cross-seed Pearson r', fontsize=FONT.LABEL - 1)
ax_v.set_ylim(0.5, 1.12)
ax_v.axhline(0.9, color='#555', linestyle=':', lw=0.9, label='r = 0.9')
ax_v.legend(fontsize=FONT.LEGEND - 1, frameon=False, loc='lower right')

add_footnote(fig,
    f'Cross-seed consistency: mean pairwise Pearson r on overlapping test trials; '
    f'each model uses own R²≥{R2_THRESH} mask; hatching = insufficient data')

savefig_manifest(fig, 'embedding_consistency_comparison.png', OUT_DIRS,
                 skip_tight_layout=True)
print('Generated embedding_consistency_comparison.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Cross-model attribution profile consistency
# Spearman ρ between GPV profiles (11 feature groups) for each pair where
# BOTH models have R²≥0.1
# ══════════════════════════════════════════════════════════════════════════════

cross_model_pairs = [
    ('MLP × TempConv-Cont', mlp_gpv, mlp_r2, cc_gpv,  cc_r2,  '#4CAF50', '#2196F3'),
    ('MLP × TempConv-Pred', mlp_gpv, mlp_r2, cp_gpv,  cp_r2,  '#4CAF50', '#FF9800'),
]

def _cross_rhos(gpv_a, r2_a, gpv_b, r2_b):
    rhos = []
    for s in range(n_sess):
        for e in range(n_ens):
            if r2_a[s, e] < R2_THRESH or r2_b[s, e] < R2_THRESH:
                continue
            va = gpv_a[s, e]
            vb = gpv_b[s, e]
            ok = np.isfinite(va) & np.isfinite(vb)
            if ok.sum() < 4:
                continue
            r, _ = spearmanr(va[ok], vb[ok])
            rhos.append(r)
    return np.array(rhos)

rho_data = []
for label, ga, ra, gb, rb, ca, cb in cross_model_pairs:
    rhos = _cross_rhos(ga, ra, gb, rb)
    rho_data.append((label, rhos, ca, cb))
    print(f'{label}: n={len(rhos)}  median ρ={np.median(rhos):.3f}  '
          f'mean ρ={np.mean(rhos):.3f}')

fig2, ax2 = plt.subplots(figsize=(FIG.HALF[0], FIG.HALF[1]))
apply_style(fig2, ax2)
rng2 = np.random.default_rng(1)

positions = list(range(len(rho_data)))
flat2, labels2, colors2 = zip(*[(rd[1], rd[0], rd[2]) for rd in rho_data])

vp = ax2.violinplot(flat2, positions=positions,
                    showmedians=True, showextrema=True, widths=0.55)
for pc, (_, _, ca, _) in zip(vp['bodies'], rho_data):
    pc.set_facecolor(ca); pc.set_alpha(0.5)
for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
    vp[part].set_color('#333'); vp[part].set_linewidth(0.9)

for i, (label, rhos, ca, cb) in enumerate(rho_data):
    jitter = rng2.uniform(-0.08, 0.08, size=len(rhos))
    ax2.scatter(i + jitter, rhos, s=10, color=ca, alpha=0.5, linewidths=0, zorder=3)
    med = float(np.median(rhos))
    ax2.text(i, med + 0.02, f'{med:.2f}', ha='center', va='bottom',
             fontsize=FONT.ANNOTATION, fontweight='bold')
    ax2.text(i, -0.04, f'n={len(rhos)}', ha='center', va='top',
             fontsize=FONT.TICK - 1, color='#555',
             transform=ax2.get_xaxis_transform())

ax2.set_xticks(positions)
ax2.set_xticklabels([rd[0].replace(' × ', '\n× ') for rd in rho_data],
                    fontsize=FONT.TICK)
ax2.set_ylabel('Spearman ρ of GPV profiles', fontsize=FONT.LABEL)
ax2.axhline(0, color='#999', lw=0.7, linestyle=':')
ax2.axhline(0.9, color='#555', lw=0.9, linestyle='--', label='ρ = 0.9')
ax2.legend(fontsize=FONT.LEGEND, frameon=False)

add_footnote(fig2,
    f'GPV = Global Permutation Variance across 11 feature groups; '
    f'pairs where both models R²≥{R2_THRESH}; Spearman ρ per pair')

savefig_manifest(fig2, 'cross_model_attribution_consistency.png', OUT_DIRS)
print('Generated cross_model_attribution_consistency.png')
