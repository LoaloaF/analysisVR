"""
Honest summary figure: what TempConv's temporal window does and does not explain.

4 panels:
  A  R² by model (Linear / MLP / TempConv-Cont / TempConv-Pred) — shared valid pairs
  B  Mean η² per pair: advantage pairs vs background (TempConv advantage not from temporal signal)
  C  Shape ρ(IG window, lagged η²): advantage vs background — no significant difference
  D  Head angle case study: lagged |ρ|, lagged η², TempConv IG — IG agrees with nonlinear analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os, sys
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from figure_style import (
    FIG, DPI, FONT, LINE, MARKER,
    MODEL_COLORS, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

# ── paths ──────────────────────────────────────────────────────────────────────
MLP_R2   = 'outputs/mlps/ensembles_multiseed/all_r2.npy'
LIN_R2   = 'outputs/linear/ensembles_multiseed/all_r2.npy'
CEBC_R2  = 'outputs/cebra_eval/ensembles/all_r2.npy'
CEBP_R2  = 'outputs/cebra_pred_eval/ensembles/all_r2.npy'
ETA2_F   = 'outputs/temporal_advantage/lagged_eta2.npy'
RHO_F    = 'outputs/temporal_advantage/lagged_rho.npy'
ADV_IG_F = 'outputs/temporal_advantage/adv_ig_profiles.npy'
BG_IG_F  = 'outputs/temporal_advantage/bg_ig_profiles.npy'
SG_F     = 'outputs/mlps/ensembles_multiseed/semantic_groups.pkl'

OUT_LOCAL  = 'outputs/temporal_advantage/fig_honest_summary.png'
OUT_DESK   = '/mnt/c/Users/amits/Desktop/fig_honest_summary.png'

LAGS       = list(range(-5, 5))
R2_THR     = 0.01
ADV_THR    = 0.015
TOP_N      = 20

HEAD_ANGLE_IDX  = 5   # group index in eta2 (8 continuous groups)
HEAD_ANGLE_FEAT = [5]  # feature column in 17-dim input

# ── load R² arrays ─────────────────────────────────────────────────────────────
mlp_r2  = np.nanmean(np.load(MLP_R2),  axis=0)   # (29, 23)
lin_r2  = np.nanmean(np.load(LIN_R2),  axis=0)
cebc_r2 = np.nanmean(np.load(CEBC_R2), axis=0)
cebp_r2 = np.nanmean(np.load(CEBP_R2), axis=0)

eta2 = np.load(ETA2_F)  # (29, 23, 8, 10)
rho  = np.load(RHO_F)   # (29, 23, 8, 10)
adv_ig = np.load(ADV_IG_F, allow_pickle=True).item()
bg_ig  = np.load(BG_IG_F,  allow_pickle=True).item()

sg = pickle.load(open(SG_F, 'rb'))

# ── shared valid mask (all three main models) ──────────────────────────────────
shared_valid = (mlp_r2 > R2_THR) & (cebc_r2 > R2_THR)

# ── advantage pairs — use the same pairs as the cached IG dicts ────────────────
top_pairs = list(adv_ig.keys())   # 20 advantage pairs with IG computed
bg_pairs  = list(bg_ig.keys())    # 20 background pairs with IG computed

# ── Panel A: R² per model on shared valid pairs ────────────────────────────────
model_vals = {
    'Linear':           lin_r2[shared_valid],
    'MLP':              mlp_r2[shared_valid],
    'TempConv\nCont':   cebc_r2[shared_valid],
    'TempConv\nPred':   cebp_r2[shared_valid],
}
# clip negatives to 0 for display (Linear can be slightly negative)
model_vals = {k: np.clip(v, 0, None) for k, v in model_vals.items()}
MODEL_COLORS = ['#888888', '#4878d0', '#ee8866', '#aa3377']

# ── Panel B: mean η² per pair ──────────────────────────────────────────────────
def mean_eta2_per_pair(pairs):
    return np.array([np.nanmean(eta2[s, n, :, :]) for s, n in pairs])

adv_eta2_mean = mean_eta2_per_pair(top_pairs)
bg_eta2_mean  = mean_eta2_per_pair(bg_pairs)
_, p_eta2 = mannwhitneyu(adv_eta2_mean, bg_eta2_mean, alternative='greater')

# ── Panel C: shape ρ(IG window, lagged η²) per pair ───────────────────────────
def shape_rho_per_pair(ig_dict, pairs):
    results = []
    for s, n in pairs:
        if (s, n) not in ig_dict:
            continue
        ig17 = ig_dict[(s, n)]  # (17, 10)
        pair_rhos = []
        for g_idx, (_, feat_idxs) in enumerate(sg[:8]):
            ig_g  = np.abs(ig17[feat_idxs, :]).mean(axis=0)
            eta_g = eta2[s, n, g_idx, :]
            if np.isnan(eta_g).all() or ig_g.std() < 1e-9 or np.nanstd(eta_g) < 1e-9:
                continue
            valid = ~np.isnan(eta_g)
            if valid.sum() < 4:
                continue
            rv, _ = spearmanr(ig_g[valid], eta_g[valid])
            pair_rhos.append(rv)
        if pair_rhos:
            results.append(np.mean(pair_rhos))
    return np.array(results)

adv_shape_rho = shape_rho_per_pair(adv_ig, top_pairs)
bg_shape_rho  = shape_rho_per_pair(bg_ig,  bg_pairs)
if len(adv_shape_rho) > 0 and len(bg_shape_rho) > 0:
    _, p_shape = mannwhitneyu(adv_shape_rho, bg_shape_rho, alternative='greater')
else:
    p_shape = np.nan

# ── Panel D: Head angle temporal profiles ─────────────────────────────────────
ha_rho  = np.nanmean(np.abs(rho[:, :, HEAD_ANGLE_IDX, :]), axis=(0, 1))   # (10,)
ha_eta2 = np.nanmean(eta2[:, :, HEAD_ANGLE_IDX, :],        axis=(0, 1))   # (10,)

all_ig = {**adv_ig, **bg_ig}
ig_ha_list = [np.abs(ig17[HEAD_ANGLE_FEAT, :]).mean(axis=0)
              for ig17 in all_ig.values()]
ha_ig = np.mean(ig_ha_list, axis=0)  # (10,)

def norm01(x):
    mn, mx = np.nanmin(x), np.nanmax(x)
    return (x - mn) / (mx - mn) if mx - mn > 1e-12 else x * 0

lag_labels = [f't{l:+d}' for l in LAGS]

# ── figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
apply_style(fig)
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35,
                      left=0.07, right=0.97, top=0.92, bottom=0.08)

# ── A: R² by model ─────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
labels = list(model_vals.keys())
positions = np.arange(len(labels))

for i, (lbl, v, col) in enumerate(zip(labels, model_vals.values(), MODEL_COLORS)):
    if len(v) == 0:
        continue
    parts = ax_a.violinplot(v, positions=[i], widths=0.6,
                             showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(col); pc.set_alpha(0.55)
    med = np.median(v)
    ax_a.hlines(med, i - 0.25, i + 0.25, colors=col, linewidths=2.5)
    ax_a.text(i, -0.015, f'{med:.3f}', ha='center', va='top',
              fontsize=8.5, color=col, fontweight='bold')

ax_a.set_xticks(positions)
ax_a.set_xticklabels(labels, fontsize=9)
ax_a.set_ylabel('R²', fontsize=10)
ax_a.set_title('A  Model R² comparison\n(shared valid pairs, n=363)', fontsize=10, fontweight='bold', loc='left')
ax_a.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
ax_a.spines[['top', 'right']].set_visible(False)

# ── B: mean η² per pair ────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
cols_b = ['#ee8866', '#4878d0']
lbls_b = [f'TempConv-advantage\n(n={len(adv_eta2_mean)})',
          f'Background\n(n={len(bg_eta2_mean)})']

for i, (d, c) in enumerate(zip([adv_eta2_mean, bg_eta2_mean], cols_b)):
    if len(d) == 0:
        continue
    parts = ax_b.violinplot(d, positions=[i], widths=0.55,
                             showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(c); pc.set_alpha(0.55)
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(d))
    ax_b.scatter(i + jitter, d, color=c, s=22, alpha=0.7, zorder=3)
    ax_b.hlines(np.median(d), i - 0.22, i + 0.22, colors=c, linewidths=2.5)

p_str  = f'p={p_eta2:.3f}' if p_eta2 >= 0.001 else 'p<0.001'
sig_col = '#228833' if p_eta2 < 0.05 else '#888888'
ax_b.text(0.5, 0.96, f'Mann-Whitney {p_str}', transform=ax_b.transAxes,
          ha='center', va='top', fontsize=9, color=sig_col, fontstyle='italic')
ax_b.set_xticks([0, 1]); ax_b.set_xticklabels(lbls_b, fontsize=9)
ax_b.set_ylabel('Mean η² (all features × lags)', fontsize=9)
ax_b.set_title('B  Advantage pairs have stronger\nbehavioral signal (η²)', fontsize=10, fontweight='bold', loc='left')
ax_b.spines[['top', 'right']].set_visible(False)

# ── C: shape ρ violin ──────────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])

for i, (d, c) in enumerate(zip([adv_shape_rho, bg_shape_rho], cols_b)):
    if len(d) == 0:
        continue
    parts = ax_c.violinplot(d, positions=[i], widths=0.55,
                             showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(c); pc.set_alpha(0.55)
    jitter = np.random.default_rng(1).uniform(-0.12, 0.12, len(d))
    ax_c.scatter(i + jitter, d, color=c, s=22, alpha=0.7, zorder=3)
    ax_c.hlines(np.median(d), i - 0.22, i + 0.22, colors=c, linewidths=2.5)

if not np.isnan(p_shape):
    p_str_c  = f'p={p_shape:.3f}' if p_shape >= 0.001 else 'p<0.001'
    sig_col_c = '#228833' if p_shape < 0.05 else '#888888'
    ax_c.text(0.5, 0.96, f'Mann-Whitney {p_str_c}', transform=ax_c.transAxes,
              ha='center', va='top', fontsize=9, color=sig_col_c, fontstyle='italic')
ax_c.axhline(0, color='k', lw=0.8, ls='--', alpha=0.3)
ax_c.set_xticks([0, 1]); ax_c.set_xticklabels(lbls_b, fontsize=9)
ax_c.set_ylabel('Shape ρ (IG window vs lagged η²)', fontsize=9)
ax_c.set_title('C  IG–η² alignment not higher in\nadvantage pairs — temporal not the driver',
               fontsize=10, fontweight='bold', loc='left')
ax_c.spines[['top', 'right']].set_visible(False)

# ── D: Head angle temporal profiles ───────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
x = np.arange(len(LAGS))
ax_d.plot(x, norm01(ha_rho),  color='#4878d0', lw=2.0, marker='o', ms=4,
          label='Lagged |ρ| (linear)')
ax_d.plot(x, norm01(ha_eta2), color='#228833', lw=2.0, ls='-.', marker='s', ms=4,
          label='Lagged η² (nonlinear)')
ax_d.plot(x, norm01(ha_ig),   color='#aa3377', lw=2.0, ls='--', marker='^', ms=4,
          label='TempConv IG window')

for arr, col in [(norm01(ha_rho), '#4878d0'),
                 (norm01(ha_eta2), '#228833'),
                 (norm01(ha_ig), '#aa3377')]:
    pk = np.argmax(arr)
    ax_d.axvline(pk, color=col, lw=0.8, ls=':', alpha=0.6)

ax_d.set_xticks(x)
ax_d.set_xticklabels(lag_labels, fontsize=8, rotation=45)
ax_d.set_ylabel('Normalised value', fontsize=9)
ax_d.set_title('D  Head angle: TempConv IG agrees with\nnonlinear analysis; linear ρ disagrees',
               fontsize=10, fontweight='bold', loc='left')
ax_d.legend(fontsize=8.5, framealpha=0.6, loc='lower right')
ax_d.spines[['top', 'right']].set_visible(False)

fig.suptitle('TempConv temporal advantage: what the data shows',
             fontsize=13, fontweight='bold', y=0.98)

out_dir_local = os.path.dirname(OUT_LOCAL)
out_dir_desk  = os.path.dirname(OUT_DESK)
savefig_manifest(fig, os.path.basename(OUT_LOCAL), [out_dir_local, out_dir_desk])
print('Done.')
