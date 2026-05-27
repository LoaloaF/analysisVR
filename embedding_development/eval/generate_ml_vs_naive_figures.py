#!/usr/bin/env python3
"""
generate_ml_vs_naive_figures.py

Generates presentation-ready figures from pre-computed ml_vs_naive arrays.
Run eval_ml_vs_naive.py first to populate outputs/mlps/ml_vs_naive/.

Outputs (all manifest-sized):
  ml_vs_naive_scatter.png      FIG.FULL  — Part 1: ML attribution vs naive effect size
  nonlinearity_advantage.png   FIG.FULL  — Part 2: η² vs |ρ|, GPV tracks nonlinear bonus
  mlp_vs_linear_r2.png         FIG.FULL  — Part 3: MLP outperforms GLM
  ablation_proof.png           FIG.FULL  — Part 4: single-feature MLP proves functional mapping
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import (
    FIG, DPI, FONT, FEATURE_NAMES_SHORT, AXIS_LABELS, MODEL_COLORS,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

# ─── PATHS ────────────────────────────────────────────────────────────────────
base     = os.path.dirname(os.path.abspath(__file__))
root     = os.path.join(base, '..')
mv_dir   = os.path.join(root, 'outputs', 'mlps', 'ml_vs_naive')
attr_dir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
lin_dir  = os.path.join(root, 'outputs', 'linear', 'ensembles_multiseed')
OUT_DIRS = [mv_dir, '/mnt/c/Users/amits/Desktop']

# ─── LOAD ─────────────────────────────────────────────────────────────────────
naive   = np.load(os.path.join(mv_dir, 'naive_importance.npy'))   # (29,23,11) |ρ| or d
eta2    = np.load(os.path.join(mv_dir, 'eta2.npy'))               # (29,23,11) η²
gpv     = np.load(os.path.join(attr_dir, 'importance_global_pv_semantic.npy'))
ig      = np.load(os.path.join(attr_dir, 'importance_ig_semantic.npy'))
all_r2_mlp = np.load(os.path.join(attr_dir, 'all_r2.npy'))        # (5,29,23)
all_r2_lin = np.load(os.path.join(lin_dir, 'all_r2.npy'))

with open(os.path.join(attr_dir, 'semantic_groups.pkl'), 'rb') as f:
    semantic_groups = pickle.load(f)
group_names = [g[0] for g in semantic_groups]

df_abl = pd.read_csv(os.path.join(mv_dir, 'ablation_results.csv'))

R2_THRESHOLD = 0.01
mean_r2_mlp = all_r2_mlp.mean(axis=0)   # (29,23)
mean_r2_lin = all_r2_lin.mean(axis=0)
valid = (~np.any(np.isnan(all_r2_mlp), axis=0)) & (mean_r2_mlp >= R2_THRESHOLD)

n_sess, n_ens, n_grp = naive.shape
CONT_GROUPS = list(range(7))
CAT_GROUPS  = list(range(7, 11))

# ─── COMMON MASKS ─────────────────────────────────────────────────────────────
valid3d = np.broadcast_to(valid[:, :, np.newaxis], (n_sess, n_ens, n_grp))
# triples where valid and ALL of naive, gpv, ig are non-NaN
both_ok  = valid3d & ~np.isnan(naive) & ~np.isnan(gpv) & ~np.isnan(ig)
naive_f  = naive[both_ok]
gpv_f    = gpv[both_ok]
ig_f     = ig[both_ok]

# feature type per triple (continuous / categorical)
grp_type = np.array(['continuous'] * 7 + ['categorical'] * 4)
gtype3d  = np.broadcast_to(grp_type[np.newaxis, np.newaxis, :],
                            (n_sess, n_ens, n_grp))
gtype_f  = gtype3d[both_ok]

C_CONT = '#1565C0'
C_CAT  = '#C62828'
col_f  = np.where(gtype_f == 'continuous', C_CONT, C_CAT)

n_pairs = valid.sum()
print(f'Valid pairs: {n_pairs}   triples with both naive+attribution: {both_ok.sum()}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Part 1: ML attribution vs naive data effect size
# ═══════════════════════════════════════════════════════════════════════════════
print('Figure 1: ml_vs_naive_scatter.png...')

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig, [ax_l, ax_r])

for ax, attr_vals, attr_name, panel in [
    (ax_l, gpv_f, 'GPV', 'A'),
    (ax_r, ig_f,  'IG',  'B'),
]:
    rho_sp, p_sp = spearmanr(naive_f, attr_vals)

    ax.scatter(naive_f[gtype_f == 'continuous'], attr_vals[gtype_f == 'continuous'],
               c=C_CONT, s=6, alpha=0.30, zorder=2, label='Continuous', linewidths=0)
    ax.scatter(naive_f[gtype_f == 'categorical'], attr_vals[gtype_f == 'categorical'],
               c=C_CAT, s=6, alpha=0.45, zorder=2, label='Categorical',
               marker='D', linewidths=0)

    # trend line
    z  = np.polyfit(naive_f, attr_vals, 1)
    xs = np.linspace(naive_f.min(), naive_f.max(), 60)
    ax.plot(xs, np.polyval(z, xs), 'k--', lw=1.2, alpha=0.7)

    ax.set_xlabel('Naive data effect  (|ρ| or Cohen\'s d)', fontsize=FONT.LABEL - 1)
    ylab = AXIS_LABELS['r2_drop'] if attr_name == 'GPV' else AXIS_LABELS['ig']
    ax.set_ylabel(f'ML {attr_name} attribution', fontsize=FONT.LABEL - 1)

    ax.text(0.04, 0.97, f'ρ = {rho_sp:.2f}',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=FONT.ANNOTATION, color='dimgray')

    if panel == 'A':
        ax.legend(fontsize=FONT.LEGEND - 1, frameon=False, markerscale=2)
    add_panel_label(ax, panel)

add_footnote(fig,
    f'One point per (session, ensemble, feature group) triple; {both_ok.sum()} valid triples; '
    f'pairs: MLP R² ≥ {R2_THRESHOLD}')
savefig_manifest(fig, 'ml_vs_naive_scatter.png', OUT_DIRS)
print('  Saved ml_vs_naive_scatter.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Part 2: Nonlinearity advantage
# ═══════════════════════════════════════════════════════════════════════════════
print('Figure 2: nonlinearity_advantage.png...')

# continuous features only
cont3d  = np.zeros((n_sess, n_ens, n_grp), dtype=bool)
cont3d[:, :, CONT_GROUPS] = True
sel_c   = valid3d & cont3d & ~np.isnan(eta2) & ~np.isnan(naive)
rho_f2  = naive[sel_c]    # |ρ| for continuous (same as naive for cont groups)
eta2_f2 = eta2[sel_c]
gpv_f2  = gpv[sel_c]
nl_bonus = eta2_f2 - rho_f2**2

rho_nb, p_nb = spearmanr(nl_bonus, gpv_f2)

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig, [ax_l, ax_r])

# Left: |ρ| vs η², colored by GPV
vmax_gpv = float(np.nanpercentile(gpv_f2, 95))
sc = ax_l.scatter(rho_f2, eta2_f2, c=gpv_f2, cmap='YlOrRd',
                  s=8, alpha=0.45, zorder=3, vmin=0, vmax=vmax_gpv,
                  linewidths=0)
cbar = plt.colorbar(sc, ax=ax_l, shrink=0.85)
cbar.set_label(AXIS_LABELS['r2_drop'], fontsize=FONT.LEGEND)
cbar.ax.tick_params(labelsize=FONT.TICK - 2)

lim = max(rho_f2.max(), eta2_f2.max()) * 1.06
ax_l.plot([0, lim], [0, lim], 'k--', lw=1.0, alpha=0.5, label='η² = ρ²  (linear)')
ax_l.fill_between([0, lim], [0, lim], lim, alpha=0.05, color='red')
ax_l.set_xlim(0, lim); ax_l.set_ylim(0, lim)
ax_l.set_xlabel('Spearman |ρ|  (linear)', fontsize=FONT.LABEL - 1)
ax_l.set_ylabel('η²  (nonlinear, decile bins)', fontsize=FONT.LABEL - 1)
ax_l.legend(fontsize=FONT.LEGEND - 1, frameon=False)
add_panel_label(ax_l, 'A')

# Right: nonlinear bonus vs GPV
ax_r.scatter(nl_bonus, gpv_f2, s=8, alpha=0.40, color='#C62828', zorder=3,
             linewidths=0)
z  = np.polyfit(nl_bonus, gpv_f2, 1)
xs = np.linspace(nl_bonus.min(), nl_bonus.max(), 60)
ax_r.plot(xs, np.polyval(z, xs), 'k--', lw=1.2, alpha=0.7)
ax_r.axvline(0, color='#888', lw=0.7, linestyle=':')
ax_r.set_xlabel('Nonlinear bonus  (η² − ρ²)', fontsize=FONT.LABEL - 1)
ax_r.set_ylabel(AXIS_LABELS['r2_drop'], fontsize=FONT.LABEL - 1)
ax_r.text(0.04, 0.97, f'ρ = {rho_nb:.2f}',
          transform=ax_r.transAxes, ha='left', va='top',
          fontsize=FONT.ANNOTATION, color='dimgray')
add_panel_label(ax_r, 'B')

add_footnote(fig,
    f'Continuous features only ({sel_c.sum()} triples); '
    f'η² via 10-bin equal-frequency partition; shaded region = nonlinear advantage (η² > ρ²)')
savefig_manifest(fig, 'nonlinearity_advantage.png', OUT_DIRS)
print('  Saved nonlinearity_advantage.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Part 3: MLP vs Linear R²
# ═══════════════════════════════════════════════════════════════════════════════
print('Figure 3: mlp_vs_linear_r2.png...')

both_valid = valid & ~np.isnan(mean_r2_lin)
mlp_v    = mean_r2_mlp[both_valid]
lin_v    = mean_r2_lin[both_valid]
delta    = mlp_v - lin_v
max_gpv  = np.nanmax(gpv, axis=2)
max_gpv_v = max_gpv[both_valid]
rho_dlt, p_dlt = spearmanr(max_gpv_v, delta)

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig, [ax_l, ax_r])

lim = max(mlp_v.max(), lin_v.max()) * 1.06
sc = ax_l.scatter(lin_v, mlp_v, c=max_gpv_v, cmap='YlOrRd',
                  s=14, alpha=0.60, zorder=3, vmin=0,
                  vmax=float(np.nanpercentile(max_gpv_v, 95)), linewidths=0)
cbar_l = plt.colorbar(sc, ax=ax_l, shrink=0.85)
cbar_l.set_label('Max GPV  (strongest group)', fontsize=FONT.LEGEND)
cbar_l.ax.tick_params(labelsize=FONT.TICK - 2)

ax_l.plot([0, lim], [0, lim], 'k--', lw=0.9, alpha=0.5)
ax_l.set_xlim(0, lim); ax_l.set_ylim(0, lim)
ax_l.set_xlabel('Linear R²  (GLM baseline)', fontsize=FONT.LABEL - 1)
ax_l.set_ylabel('MLP R²', fontsize=FONT.LABEL - 1)
ax_l.text(0.04, 0.97,
          f'n = {both_valid.sum()} pairs\nMean: GLM {lin_v.mean():.3f}  MLP {mlp_v.mean():.3f}',
          transform=ax_l.transAxes, ha='left', va='top',
          fontsize=FONT.ANNOTATION, color='dimgray')
add_panel_label(ax_l, 'A')

ax_r.scatter(max_gpv_v, delta, s=14, alpha=0.50, color='#1565C0', zorder=3,
             linewidths=0)
z  = np.polyfit(max_gpv_v, delta, 1)
xs = np.linspace(max_gpv_v.min(), max_gpv_v.max(), 60)
ax_r.plot(xs, np.polyval(z, xs), 'k--', lw=1.2, alpha=0.7)
ax_r.axhline(0, color='#888', lw=0.7, linestyle='--')
ax_r.set_xlabel('Max GPV  (strongest attributed feature)', fontsize=FONT.LABEL - 1)
ax_r.set_ylabel('MLP R² − Linear R²  (advantage)', fontsize=FONT.LABEL - 1)
ax_r.text(0.04, 0.97, f'ρ = {rho_dlt:.2f}',
          transform=ax_r.transAxes, ha='left', va='top',
          fontsize=FONT.ANNOTATION, color='dimgray')
add_panel_label(ax_r, 'B')

add_footnote(fig,
    f'{both_valid.sum()} pairs with valid Linear R²; MLP validity threshold R² ≥ {R2_THRESHOLD}; '
    f'5-seed mean R² plotted; diagonal = parity')
savefig_manifest(fig, 'mlp_vs_linear_r2.png', OUT_DIRS)
print('  Saved mlp_vs_linear_r2.png')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Part 4: Ablation proof
# ═══════════════════════════════════════════════════════════════════════════════
print('Figure 4: ablation_proof.png...')

if df_abl.empty:
    print('  No ablation pairs — skipping.')
else:
    n_abl = len(df_abl)
    fig, ax = plt.subplots(figsize=FIG.FULL)
    apply_style(fig, ax)

    x = np.arange(n_abl)
    w = 0.22
    ek = dict(ecolor='k', lw=0.7, capsize=3)

    ax.bar(x - w,   df_abl['lin_r2_mean'],  w,
           yerr=df_abl['lin_r2_sem'], color='#90A4AE', alpha=0.9,
           label='Single-feature Linear', error_kw=ek, linewidth=0)
    ax.bar(x,        df_abl['mlp_r2_mean'],  w,
           yerr=df_abl['mlp_r2_sem'], color='#C62828', alpha=0.9,
           label='Single-feature MLP', error_kw=ek, linewidth=0)
    ax.bar(x + w,    df_abl['full_mlp_r2'],  w,
           color='#1565C0', alpha=0.85,
           label='Full MLP (all features)', linewidth=0)

    ax.axhline(0, color='#888888', lw=0.8, linestyle='--')

    # annotate η² and |ρ| above bars
    ymax_all = max(df_abl['mlp_r2_mean'].max(),
                   df_abl['full_mlp_r2'].max()) * 1.02
    for i, row in df_abl.iterrows():
        ax.text(i, ymax_all + 0.002,
                f"η²={row['eta2']:.2f}\n|ρ|={row['rho']:.2f}",
                ha='center', va='bottom', fontsize=FONT.ANNOTATION - 2, color='dimgray')

    # prettier x-tick labels — canonical short name + session
    def _short_label(row):
        raw = row['label'].split('×')[-1].split('\n')[0]
        short = FEATURE_NAMES_SHORT.get(raw, raw[:10])
        sess_s = row['label'].split('\n')[-1] if '\n' in row['label'] else ''
        return f"E{row['n_i']+1:02d} {short}\n{sess_s}"

    if 'n_i' in df_abl.columns:
        xlabels = [_short_label(r) for _, r in df_abl.iterrows()]
    else:
        xlabels = df_abl['label'].str.replace('\\n', '\n')

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=FONT.TICK - 2, ha='center')
    ax.set_ylabel('Test R²  (5-seed mean ± SEM)', fontsize=FONT.LABEL - 1)
    ax.legend(fontsize=FONT.LEGEND - 1, frameon=False)

    add_footnote(fig,
        'Pairs selected: GPV > 0.008, Spearman |ρ| < 0.25, η² > 0.01; '
        '5 seeds × single-feature MLP (64-unit, 2 hidden) trained for 150 epochs')
    savefig_manifest(fig, 'ablation_proof.png', OUT_DIRS)
    print('  Saved ablation_proof.png')

print('\nAll ml_vs_naive figures done.')
