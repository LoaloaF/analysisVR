#!/usr/bin/env python3
"""
gen_new_analysis_figures.py

Generates four new figures from pre-computed data:
  1. joint_effect_scatter.png     — ML-exclusive pairs (high R², low max|ρ|)
  2. tempconv_delta_r2.png        — ΔR² distribution TempConv-Pred vs MLP
  3. r2_distribution_cdf.png      — R² CDF replacing grand-mean framing
  4. cross_ensemble_consistency.png — within vs across session attribution ρ
"""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import (
    FIG, DPI, FONT, apply_style, add_footnote, savefig_manifest,
)

base  = os.path.dirname(os.path.abspath(__file__))
root  = os.path.join(base, '..')
mdir  = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUTS  = [mdir, '/mnt/c/Users/amits/Desktop']

# ── Load shared data ───────────────────────────────────────────────────────────
all_r2  = np.load(os.path.join(mdir, 'all_r2.npy'))
mean_r2 = np.nanmean(all_r2, axis=0)   # (29, 23)
gpv     = np.load(os.path.join(mdir, 'importance_global_pv_semantic.npy'))
valid   = (~np.all(np.isnan(gpv), axis=-1)) & (mean_r2 >= 0.01)

with open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb') as f:
    sg = pickle.load(f)

pred_r2 = np.nanmean(np.load(os.path.join(root, 'outputs', 'cebra_pred_eval',
                                            'ensembles', 'all_r2.npy')), axis=0)


def _eta_sq_binned(x, y, n_bins=10):
    """η² from binned one-way ANOVA — captures linear and non-linear associations."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < n_bins * 3:
        return np.nan
    x_m, y_m = x[mask], y[mask]
    boundaries = np.quantile(x_m, np.linspace(0, 1, n_bins + 1))
    boundaries[-1] += 1e-10
    labels = np.searchsorted(boundaries[1:], x_m).clip(0, n_bins - 1)
    grand_mean = y_m.mean()
    ss_total = float(np.sum((y_m - grand_mean) ** 2))
    if ss_total < 1e-12:
        return 0.0
    ss_between = sum(
        len(y_m[labels == k]) * (float(y_m[labels == k].mean()) - grand_mean) ** 2
        for k in np.unique(labels) if len(y_m[labels == k]) > 0
    )
    return ss_between / ss_total


# ── Pre-compute max univariate η² and |ρ| per pair ────────────────────────────
max_eta_path = '/tmp/max_eta_sq.npy'
max_rho_path = '/tmp/max_rho.npy'

need_compute = not os.path.exists(max_eta_path) or not os.path.exists(max_rho_path)

if not need_compute:
    max_eta_sq = np.load(max_eta_path)
    max_rho    = np.load(max_rho_path)
    print("Loaded cached max_eta_sq and max_rho")
else:
    with open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb') as f:
        ds = pickle.load(f)
    session_ids = list(ds.keys())
    n_s, n_e = mean_r2.shape
    max_eta_sq = np.full((n_s, n_e), np.nan)
    max_rho    = np.full((n_s, n_e), np.nan)
    print("Computing max univariate η² and |ρ|...")
    for s_idx, sid in enumerate(session_ids):
        sd = ds[sid]; all_t = list(sd['data'].keys())
        if not all_t: continue
        Xs = np.concatenate([sd['data'][t] for t in all_t]).astype(float)
        Ys = np.concatenate([sd['labels'][t] for t in all_t]).astype(float)
        for e_idx in range(n_e):
            if not valid[s_idx, e_idx]: continue
            y = Ys[:, e_idx]
            etas, rhos = [], []
            for _, cols in sg:
                x = Xs[:, cols[0]]
                etas.append(_eta_sq_binned(x, y))
                rhos.append(abs(spearmanr(x, y, nan_policy='omit')[0]))
            max_eta_sq[s_idx, e_idx] = np.nanmax(etas)
            max_rho[s_idx, e_idx]    = max(rhos)
        print(f"  session {s_idx+1}/{n_s} done")
    np.save(max_eta_path, max_eta_sq)
    np.save(max_rho_path, max_rho)
    print("Saved max_eta_sq and max_rho caches")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Joint-effect scatter
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 4.2))
apply_style(fig, ax)

ETA_THRESH = 0.05   # same as MLP R² threshold — stricter than ρ<0.15
v_mask = valid & np.isfinite(max_eta_sq)
r2_v   = mean_r2[v_mask]
eta_v  = max_eta_sq[v_mask]

# Colour by zone: ML-exclusive = MLP R²≥0.05 but max η²<threshold
ml_excl = (r2_v >= 0.05) & (eta_v < ETA_THRESH)
ax.scatter(eta_v[~ml_excl], r2_v[~ml_excl], s=6, color='#aaaaaa',
           alpha=0.4, linewidths=0, rasterized=True, label='Other valid pairs')
ax.scatter(eta_v[ml_excl],  r2_v[ml_excl],  s=10, color='#d62728',
           alpha=0.75, linewidths=0, rasterized=True,
           label=f'ML-exclusive ({ml_excl.sum()} pairs)')

ax.axvline(ETA_THRESH, color='#d62728', lw=0.9, linestyle='--', alpha=0.6)
ax.axhline(0.05, color='#d62728', lw=0.9, linestyle='--', alpha=0.6)

pct = 100 * ml_excl.sum() / max((r2_v >= 0.05).sum(), 1)
ax.text(0.02, 0.92,
        f"{ml_excl.sum()} pairs ({pct:.0f}% of R²≥0.05)\n"
        f"MLP captures interaction effects\nno single feature explains ≥5% variance",
        transform=ax.transAxes, fontsize=FONT.ANNOTATION,
        color='#d62728', va='top')

ax.set_xlabel('Max univariate η² (binned ANOVA, 10 bins)', fontsize=FONT.LABEL)
ax.set_ylabel('MLP mean R²', fontsize=FONT.LABEL)
ax.legend(fontsize=FONT.LEGEND, frameon=False, loc='lower right')
add_footnote(fig, f"Valid pairs (R²≥0.01): {v_mask.sum()}; "
             f"η² = binned-ANOVA variance explained; ML-exclusive: R²≥0.05 and η²<{ETA_THRESH}")
savefig_manifest(fig, 'joint_effect_scatter.png', OUTS)
print("Generated joint_effect_scatter.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — TempConv-Pred vs MLP ΔR² distribution
# ═══════════════════════════════════════════════════════════════════════════════
both  = np.isfinite(mean_r2) & np.isfinite(pred_r2) & (mean_r2 >= 0.01)
delta = pred_r2[both] - mean_r2[both]

fig, ax = plt.subplots(figsize=(5.5, 4.2))
apply_style(fig, ax)

bins = np.linspace(-0.25, 0.25, 41)
ax.hist(delta[delta <= 0],  bins=bins, color='#2166ac', alpha=0.75,
        label=f'MLP better ({(delta<0).sum()} pairs)')
ax.hist(delta[delta > 0],   bins=bins, color='#d6604d', alpha=0.75,
        label=f'TempConv-Pred better ({(delta>0).sum()} pairs)')
ax.axvline(0,            color='black',   lw=1.0, linestyle='-')
ax.axvline(delta.mean(), color='#555555', lw=1.2, linestyle='--',
           label=f'Mean ΔR² = {delta.mean():.3f}')
ax.axvline(0.05, color='#d6604d', lw=0.8, linestyle=':')
ax.text(0.06, 0.82,
        f">{0.05} ΔR²:\n{(delta>0.05).sum()} pairs\n({100*(delta>0.05).mean():.1f}%)",
        transform=ax.transAxes, fontsize=FONT.ANNOTATION, color='#d6604d')

ax.set_xlabel('ΔR² (TempConv-Pred minus MLP)', fontsize=FONT.LABEL)
ax.set_ylabel('Number of pairs', fontsize=FONT.LABEL)
ax.legend(fontsize=FONT.LEGEND, frameon=False)
add_footnote(fig, f"n={len(delta)} pairs where both models valid (MLP R²≥0.01); "
             f"median ΔR²={np.median(delta):.3f}")
savefig_manifest(fig, 'tempconv_delta_r2.png', OUTS)
print("Generated tempconv_delta_r2.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — R² distribution CDF
# ═══════════════════════════════════════════════════════════════════════════════
r2_all = mean_r2[valid].ravel()
r2_all = r2_all[np.isfinite(r2_all)]
r2_sorted = np.sort(r2_all)
cdf = np.arange(1, len(r2_sorted)+1) / len(r2_sorted)

fig, ax = plt.subplots(figsize=(5.5, 4.2))
apply_style(fig, ax)

ax.plot(r2_sorted, 1 - cdf, color='#2166ac', lw=2)
thresholds = [0.05, 0.10, 0.20]
colors_t   = ['#4dac26', '#d7191c', '#7b3294']
for t, c in zip(thresholds, colors_t):
    frac = (r2_all >= t).mean()
    ax.axvline(t, color=c, lw=1.0, linestyle='--')
    ax.text(t + 0.003, 0.65 - thresholds.index(t)*0.12,
            f"≥{t}: {frac:.0%}",
            color=c, fontsize=FONT.ANNOTATION, va='top')

ax.set_xlabel('MLP mean R²', fontsize=FONT.LABEL)
ax.set_ylabel('Fraction of pairs exceeding threshold', fontsize=FONT.LABEL)
ax.set_xlim(0, max(r2_sorted) * 1.05)
ax.set_ylim(0, 1.02)
add_footnote(fig, f"n={len(r2_all)} valid pairs (R²≥0.01 across 29 sessions × 23 ensembles)")
savefig_manifest(fig, 'r2_distribution_cdf.png', OUTS)
print("Generated r2_distribution_cdf.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Cross-ensemble attribution consistency
# ═══════════════════════════════════════════════════════════════════════════════
R2_THRESH = 0.05
n_s, n_e = mean_r2.shape

within_r, across_r = [], []
for s in range(n_s):
    ens = [e for e in range(n_e)
           if mean_r2[s,e] >= R2_THRESH and not np.all(np.isnan(gpv[s,e]))]
    for i in range(len(ens)):
        for j in range(i+1, len(ens)):
            v = np.isfinite(gpv[s,ens[i]]) & np.isfinite(gpv[s,ens[j]])
            if v.sum() >= 4:
                r, _ = spearmanr(gpv[s,ens[i]][v], gpv[s,ens[j]][v])
                within_r.append(r)

for e in range(n_e):
    sess = [s for s in range(n_s)
            if mean_r2[s,e] >= R2_THRESH and not np.all(np.isnan(gpv[s,e]))]
    for i in range(len(sess)):
        for j in range(i+1, len(sess)):
            v = np.isfinite(gpv[sess[i],e]) & np.isfinite(gpv[sess[j],e])
            if v.sum() >= 4:
                r, _ = spearmanr(gpv[sess[i],e][v], gpv[sess[j],e][v])
                across_r.append(r)

ws = np.array(within_r)
xs = np.array(across_r)

fig, ax = plt.subplots(figsize=(5.0, 4.2))
apply_style(fig, ax)

vp = ax.violinplot([ws, xs], positions=[0, 1],
                   showmedians=True, showextrema=True, widths=0.5)
colors_v = ['#2166ac', '#d6604d']
for pc, c in zip(vp['bodies'], colors_v):
    pc.set_facecolor(c); pc.set_alpha(0.5)
for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
    vp[part].set_color('black'); vp[part].set_linewidth(1.0)

rng = np.random.default_rng(0)
for i, (vals, c) in enumerate([(ws, colors_v[0]), (xs, colors_v[1])]):
    jit = rng.uniform(-0.07, 0.07, size=len(vals))
    ax.scatter(i + jit, vals, s=3, color=c, alpha=0.25, linewidths=0, rasterized=True)
    med = np.median(vals)
    ax.text(i, med + 0.03, f'{med:.2f}', ha='center', fontsize=FONT.ANNOTATION,
            fontweight='bold')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Within session\n(different ensembles)',
                    'Across sessions\n(same ensemble)'], fontsize=FONT.TICK)
ax.set_ylabel('Spearman ρ of attribution profiles', fontsize=FONT.LABEL)
ax.axhline(0, color='#888', lw=0.6, linestyle=':')
add_footnote(fig,
    f"GPV attribution profile correlation (R²≥{R2_THRESH}); "
    f"within: n={len(ws)} pairs; across: n={len(xs)} pairs")
savefig_manifest(fig, 'cross_ensemble_consistency.png', OUTS)
print("Generated cross_ensemble_consistency.png")
print("\nAll done.")
