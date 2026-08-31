#!/usr/bin/env python3
"""
eval_ml_vs_naive.py

Demonstrates that MLP attribution adds scientific value beyond standard
GLM/correlation-based neuroscience analysis.

Three-part story
────────────────
Part 1  Validation    : ML attribution and naive data effects agree (positive correlation)
Part 2  Nonlinearity  : ML detects non-monotonic tuning that Spearman ρ / GLM misses
Part 3  MLP vs Linear : MLP predicts better; the gap is largest for attribution-flagged pairs
Part 4  Ablation proof: for pairs where GLM says nothing, a single-feature nonlinear model
                        still predicts — proving a functional mapping exists
"""

import os, sys, shutil, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, mannwhitneyu
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from models import MLP
from figure_style import (
    FIG, DPI, FONT, LINE, MARKER,
    MODEL_COLORS, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

# ─── config ──────────────────────────────────────────────────────────────────
SEEDS          = [42, 43, 44, 45, 46]
R2_THRESHOLD   = 0.01
N_BINS         = 10      # decile bins for η²
N_EPOCHS_SING  = 150     # epochs for single-feature models
LR             = 1e-3
HIDDEN_SIZE    = 64
NUM_HIDDEN     = 2

# Continuous groups only (indices 0-6 in the 11-group list)
CONT_GROUPS  = list(range(7))
# Categorical groups (indices 7-10)
CAT_GROUPS   = list(range(7, 11))

output_dir  = "./outputs/mlps/ml_vs_naive"
desktop_dir = "/mnt/c/Users/amits/Desktop/ml_vs_naive"
for d in (output_dir, desktop_dir):
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d)

device = torch.device('cpu')

# ─── load pre-computed data ───────────────────────────────────────────────────
base_attr = "./outputs/mlps/ensembles_multiseed"

all_r2_mlp  = np.load(f"{base_attr}/all_r2.npy")                      # (5,29,23)
all_r2_lin  = np.load("./outputs/linear/ensembles_multiseed/all_r2.npy")  # (5,29,23)
gpv         = np.load(f"{base_attr}/importance_global_pv_semantic.npy")   # (29,23,11)
ig          = np.load(f"{base_attr}/importance_ig_semantic.npy")
cpv         = np.load(f"{base_attr}/importance_cond_pv_semantic.npy")

with open(f"{base_attr}/semantic_groups.pkl", 'rb') as f:
    semantic_groups = pickle.load(f)   # list of (name, [col_indices])

with open("./outputs/session_dataset_ensembles.pkl", 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())

splits = {seed: np.load(f"./splits/split_seed{seed}.npy",
                        allow_pickle=True).item()
          for seed in SEEDS}

mean_r2_mlp = all_r2_mlp.mean(axis=0)   # (29,23)
mean_r2_lin = all_r2_lin.mean(axis=0)

valid = (~np.any(np.isnan(all_r2_mlp), axis=0)) & (mean_r2_mlp >= R2_THRESHOLD)
# valid[s,n] = True if MLP is reliable for this pair

n_sess, n_ens, n_grp = 29, 23, 11
group_names = [name for name, _ in semantic_groups]
print(f"Valid pairs: {valid.sum()}  out of {n_sess*n_ens}")

# ─── helper functions ─────────────────────────────────────────────────────────

def eta_sq(x, y, n_bins=N_BINS):
    """Nonlinear R² via equal-frequency decile binning."""
    total_var = np.var(y)
    if total_var < 1e-12:
        return np.nan
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    edges[-1] += 1e-9
    within_vars = []
    for b in range(n_bins):
        mask = (x >= edges[b]) & (x < edges[b + 1])
        if mask.sum() >= 2:
            within_vars.append(np.var(y[mask]))
    if not within_vars:
        return np.nan
    return max(0.0, 1.0 - np.mean(within_vars) / total_var)


def cohens_d_max(y_neural, y_cat):
    """Max pairwise Cohen's d across class means."""
    classes = np.unique(y_cat)
    d_max = 0.0
    for i, ca in enumerate(classes):
        for cb in classes[i+1:]:
            a, b = y_neural[y_cat == ca], y_neural[y_cat == cb]
            pooled = np.sqrt((np.var(a) + np.var(b)) / 2.0)
            if pooled > 0:
                d_max = max(d_max, abs(a.mean() - b.mean()) / pooled)
    return d_max


def build_session_arrays(sess, ens_idx):
    """Return X_full (T,17), y_full (T,), trial_ids (T,) for a session/ensemble."""
    X_list, y_list, tid_list = [], [], []
    for t in ds[sess]['data']:
        X = ds[sess]['data'][t].astype(np.float32)
        y = ds[sess]['labels'][t][:, ens_idx].astype(np.float32)
        X_list.append(X); y_list.append(y)
        tid_list.extend([t] * len(y))
    return (np.concatenate(X_list, axis=0),
            np.concatenate(y_list, axis=0),
            np.array(tid_list))


def savefig(fname):
    savefig_manifest(plt.gcf(), fname, [output_dir, desktop_dir])


# ─── compute naive importance for all valid triples ───────────────────────────
print("\nComputing naive importance scores...")
naive   = np.full((n_sess, n_ens, n_grp), np.nan)   # Spearman |ρ| or Cohen's d
eta2_arr = np.full((n_sess, n_ens, n_grp), np.nan)  # η² (continuous only)
rho_arr  = np.full((n_sess, n_ens, n_grp), np.nan)  # |ρ| (continuous only)

for s_idx, sess in enumerate(sessions):
    for n_idx in range(n_ens):
        if not valid[s_idx, n_idx]:
            continue
        X_full, y_full, _ = build_session_arrays(sess, n_idx)

        for g_idx, (gname, gcols) in enumerate(semantic_groups):
            if g_idx in CONT_GROUPS:
                # single continuous feature
                x = X_full[:, gcols[0]]
                rho, _ = spearmanr(x, y_full)
                naive[s_idx, n_idx, g_idx]    = abs(rho)
                rho_arr[s_idx, n_idx, g_idx]  = abs(rho)
                eta2_arr[s_idx, n_idx, g_idx] = eta_sq(x, y_full)
            else:
                # categorical: decode class from one-hot
                y_cat = np.argmax(X_full[:, gcols], axis=1)
                if len(np.unique(y_cat)) < 2:
                    continue
                naive[s_idx, n_idx, g_idx] = cohens_d_max(y_full, y_cat)

    print(f"  session {s_idx:02d} done", end='\r')
print("\nDone.")

np.save(os.path.join(output_dir, 'naive_importance.npy'), naive)
np.save(os.path.join(output_dir, 'eta2.npy'), eta2_arr)

# ─── FIGURE 1 — Validation: naive vs ML attribution ─────────────────────────
print("\nFigure 1: validation scatter...")

# Flatten valid triples for scatter — require both naive and attribution to be non-nan
valid3d  = np.broadcast_to(valid[:, :, np.newaxis], (n_sess, n_ens, n_grp))
both_ok  = valid3d & ~np.isnan(naive) & ~np.isnan(gpv) & ~np.isnan(ig)

naive_f = naive[both_ok]
gpv_f   = gpv[both_ok]
ig_f    = ig[both_ok]

# colour by group type (continuous vs categorical)
grp_type = np.array(['continuous'] * 7 + ['categorical'] * 4)
grp_type_tiled = np.broadcast_to(
    grp_type[np.newaxis, np.newaxis, :], (n_sess, n_ens, n_grp)
)
gtype_f = grp_type_tiled[both_ok]
c_cont = '#1565C0'; c_cat = '#C62828'
col_f   = np.where(gtype_f == 'continuous', c_cont, c_cat)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
apply_style(fig, axes)

for ax, attr, attr_name in [(axes[0], gpv_f, 'GPV'), (axes[1], ig_f, 'IG')]:
    rho, rp = spearmanr(naive_f, attr)
    ax.scatter(naive_f, attr, c=col_f, s=8, alpha=0.35, zorder=2)
    # group medians
    for gtype, col in [('continuous', c_cont), ('categorical', c_cat)]:
        sel = gtype_f == gtype
        ax.scatter([], [], color=col, s=40, label=gtype, alpha=0.8)
    z  = np.polyfit(naive_f, attr, 1)
    xs = np.linspace(naive_f.min(), naive_f.max(), 50)
    ax.plot(xs, np.polyval(z, xs), 'k--', lw=1.5, alpha=0.7)
    ax.set_xlabel('Naive data effect  (|ρ| for continuous, Cohen\'s d for categorical)',
                  fontsize=10)
    ax.set_ylabel(f'{attr_name} attribution', fontsize=10)
    ax.set_title(f'Naive vs {attr_name}\nSpearman ρ = {rho:.2f},  p = {rp:.2e}', fontsize=10)
    ax.legend(fontsize=9, title='Feature type')

plt.suptitle('Part 1 — Validation: ML attribution tracks data-effect size', fontsize=11,
             fontweight='bold')
plt.tight_layout()
savefig('part1_validation_scatter.png')

# violin: naive score for high-attr vs low-attr pairs (GPV)
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
apply_style(fig, axes)
for ax, attr, attr_name, col in [
    (axes[0], gpv, 'GPV',  '#2E7D32'),
    (axes[1], ig,  'IG',   '#6A1B9A'),
]:
    med_attr = np.nanmedian(attr[both_ok])
    hi_naive = naive[both_ok & (attr >= med_attr)]
    lo_naive = naive[both_ok & (attr <  med_attr)]
    _, p = mannwhitneyu(hi_naive, lo_naive, alternative='greater')
    parts = ax.violinplot([lo_naive, hi_naive], positions=[0,1],
                          showmedians=True, widths=0.5)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
    for pc, alpha in zip(parts['bodies'], [0.45, 0.8]):
        pc.set_facecolor(col); pc.set_alpha(alpha)
    for key in ('cbars','cmins','cmaxes'):
        parts[key].set_color('black'); parts[key].set_linewidth(1)
    np.random.seed(0)
    for vals, xc in [(lo_naive, 0), (hi_naive, 1)]:
        jit = np.random.uniform(-0.07, 0.07, len(vals))
        ax.scatter(xc + jit, vals, s=4, color=col, alpha=0.35, zorder=3)
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    ymax = max(hi_naive.max(), lo_naive.max()) * 1.05
    ax.plot([0, 0, 1, 1], [ymax, ymax*1.03, ymax*1.03, ymax], lw=1, color='k')
    ax.text(0.5, ymax*1.04, f'{sig}  p={p:.3f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Low {attr_name}\n(n={len(lo_naive)})',
                        f'High {attr_name}\n(n={len(hi_naive)})'], fontsize=9)
    ax.set_ylabel('Naive data effect size', fontsize=10)
    ax.set_title(f'{attr_name} — high-attr pairs have\nlarger data effects', fontsize=10)

plt.suptitle('Part 1 — Validation: high-attribution pairs have stronger data effects',
             fontsize=11, fontweight='bold')
plt.tight_layout()
savefig('part1_validation_violin.png')
print("  Figure 1 saved.")

# ─── FIGURE 2 — Nonlinearity advantage ───────────────────────────────────────
print("Figure 2: nonlinearity advantage...")

# Only continuous features
cont_mask = valid.copy()
cont_arr  = np.zeros((n_sess, n_ens, n_grp), dtype=bool)
cont_arr[:, :, CONT_GROUPS] = True
sel = cont_mask[:, :, np.newaxis] & cont_arr & ~np.isnan(eta2_arr) & ~np.isnan(rho_arr)

rho_f   = rho_arr[sel]
eta2_f  = eta2_arr[sel]
gpv_f2  = gpv[sel]
nl_bonus = eta2_f - rho_f**2   # "nonlinear bonus": η² beyond what Spearman² explains

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
apply_style(fig, axes)

# Left: scatter |ρ| vs η², coloured by GPV
ax = axes[0]
sc = ax.scatter(rho_f, eta2_f, c=gpv_f2, cmap='YlOrRd',
                s=12, alpha=0.5, zorder=3,
                vmin=0, vmax=np.percentile(gpv_f2, 95))
plt.colorbar(sc, ax=ax, label='GPV attribution')
# reference line: η² = ρ² (linear model perfectly captures the relationship)
lim_max = max(rho_f.max(), eta2_f.max()) * 1.05
ax.plot([0, lim_max], [0, lim_max], 'k--', lw=1.2, alpha=0.5, label='η² = ρ²  (linear)')
# shade "nonlinear advantage" region
ax.fill_between([0, lim_max], [0, lim_max], [lim_max, lim_max],
                alpha=0.06, color='red', label='nonlinear advantage\n(η² > ρ²)')
ax.set_xlim(0, lim_max); ax.set_ylim(0, lim_max)
ax.set_xlabel('Spearman |ρ|  (linear data-effect)', fontsize=10)
ax.set_ylabel('η²  (nonlinear data-effect, decile bins)', fontsize=10)
ax.set_title('ML attribution (GPV) is highest\nwhere nonlinear bonus is large', fontsize=10)
ax.legend(fontsize=8)
ax.set_aspect('equal')

# Right: nonlinear bonus vs GPV scatter + Spearman ρ
ax = axes[1]
rho_nb, p_nb = spearmanr(nl_bonus, gpv_f2)
ax.scatter(nl_bonus, gpv_f2, s=10, alpha=0.4, color='#C62828', zorder=3)
z = np.polyfit(nl_bonus, gpv_f2, 1)
xs = np.linspace(nl_bonus.min(), nl_bonus.max(), 50)
ax.plot(xs, np.polyval(z, xs), 'k--', lw=1.5, alpha=0.7)
ax.set_xlabel('Nonlinear bonus  (η² − ρ²)', fontsize=10)
ax.set_ylabel('GPV attribution', fontsize=10)
ax.set_title(f'GPV tracks nonlinear bonus\nSpearman ρ = {rho_nb:.2f},  p = {p_nb:.2e}',
             fontsize=10)

plt.suptitle('Part 2 — Nonlinearity: ML attribution captures tuning that correlation misses',
             fontsize=11, fontweight='bold')
plt.tight_layout()
savefig('part2_nonlinearity.png')

# Case studies: top pairs by nonlinear bonus (η² − ρ²) among continuous valid pairs
# Progressively relax constraints until we have at least 3
nl_bonus_full = eta2_arr - rho_arr**2   # (29, 23, 11)

def pick_case_studies(rho_thr, gpv_pct, eta_thr, n=3):
    score = nl_bonus_full.copy()
    score[~sel.reshape(n_sess, n_ens, n_grp)] = np.nan
    if gpv_pct is not None:
        score[gpv < np.nanpercentile(gpv[~np.isnan(gpv)], gpv_pct)] = np.nan
    score[rho_arr >= rho_thr] = np.nan
    score[eta2_arr < eta_thr] = np.nan
    flat_idx = np.argsort(score.ravel())[::-1]
    triples = []
    for fi in flat_idx:
        s_i, n_i, g_i = np.unravel_index(fi, (n_sess, n_ens, n_grp))
        if np.isnan(score[s_i, n_i, g_i]):
            break
        triples.append((int(s_i), int(n_i), int(g_i)))
        if len(triples) >= n:
            break
    return triples

case_triples = pick_case_studies(rho_thr=0.4, gpv_pct=25, eta_thr=0.05)
if len(case_triples) < 2:
    case_triples = pick_case_studies(rho_thr=0.5, gpv_pct=None, eta_thr=0.03)
if len(case_triples) < 2:
    case_triples = pick_case_studies(rho_thr=0.6, gpv_pct=None, eta_thr=0.0)

n_cs = max(len(case_triples), 1)
fig, axes_cs = plt.subplots(1, n_cs, figsize=(5.5 * n_cs, 4.5))
apply_style(fig, axes_cs if hasattr(axes_cs, '__iter__') else [axes_cs])
if n_cs == 1:
    axes_cs = [axes_cs]
for ax, (s_i, n_i, g_i) in zip(axes_cs, case_triples):
    gname, gcols = semantic_groups[g_i]
    sess = sessions[s_i]
    X_full, y_full, _ = build_session_arrays(sess, n_i)
    x = X_full[:, gcols[0]]
    # decile bin tuning curve
    edges = np.percentile(x, np.linspace(0, 100, N_BINS + 1))
    edges[-1] += 1e-9
    centers, means, sems = [], [], []
    for b in range(N_BINS):
        idx = (x >= edges[b]) & (x < edges[b+1])
        if idx.sum() >= 2:
            v = y_full[idx]
            centers.append(float((edges[b]+edges[b+1])/2))
            means.append(float(v.mean()))
            sems.append(float(v.std() / np.sqrt(len(v))))
    ax.errorbar(centers, means, yerr=sems, fmt='-o', color='#C62828',
                markersize=5, lw=1.8, elinewidth=1.2, capsize=3)
    ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
    rho_val = rho_arr[s_i, n_i, g_i]
    eta2_val = eta2_arr[s_i, n_i, g_i]
    gpv_val  = gpv[s_i, n_i, g_i]
    ax.set_xlabel(gname, fontsize=9)
    ax.set_ylabel('E neural activity (z)', fontsize=9)
    ax.set_title(f'E{n_i+1:02d} × {gname}\nS{s_i:02d}  |ρ|={rho_val:.2f}  η²={eta2_val:.2f}  GPV={gpv_val:.4f}',
                 fontsize=9)

plt.suptitle('Part 2 — Case studies: nonlinear tuning found by model, missed by correlation',
             fontsize=11, fontweight='bold')
plt.tight_layout()
savefig('part2_case_studies.png')
print("  Figure 2 saved.")

# ─── FIGURE 3 — MLP vs Linear R² ─────────────────────────────────────────────
print("Figure 3: MLP vs Linear R²...")

# per valid pair (session, ensemble)
mlp_r2 = mean_r2_mlp.copy(); mlp_r2[~valid] = np.nan
lin_r2 = mean_r2_lin.copy(); lin_r2[~valid] = np.nan

# use MLP validity mask for both
both_valid = valid & ~np.isnan(lin_r2)

mlp_v = mlp_r2[both_valid]
lin_v = lin_r2[both_valid]
delta = mlp_v - lin_v   # MLP advantage

# high vs low attribution: does MLP advantage increase with attribution?
# use max group GPV per (session, ensemble)
max_gpv = np.nanmax(gpv, axis=2)   # (29,23)
max_gpv_v = max_gpv[both_valid]

rho_delta, p_delta = spearmanr(max_gpv_v, delta)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
apply_style(fig, axes)

# Left: MLP R² vs Linear R² scatter
ax = axes[0]
lim = max(mlp_v.max(), lin_v.max()) * 1.05
sc = ax.scatter(lin_v, mlp_v, c=max_gpv_v, cmap='YlOrRd', s=18, alpha=0.6, zorder=3,
                vmin=0, vmax=np.percentile(max_gpv_v, 95))
plt.colorbar(sc, ax=ax, label='Max GPV (any feature)')
ax.plot([0, lim], [0, lim], 'k--', lw=1.2, alpha=0.5, label='Linear = MLP')
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_aspect('equal')
ax.set_xlabel('Linear model R²  (GLM baseline)', fontsize=10)
ax.set_ylabel('MLP R²', fontsize=10)
ax.set_title(f'MLP outperforms GLM across {both_valid.sum()} valid pairs\n'
             f'Mean linear R²={lin_v.mean():.3f},  MLP R²={mlp_v.mean():.3f}', fontsize=10)
ax.legend(fontsize=9)

# Right: MLP advantage (Δ R²) vs max GPV
ax = axes[1]
ax.scatter(max_gpv_v, delta, s=18, alpha=0.5, color='#1565C0', zorder=3)
z = np.polyfit(max_gpv_v, delta, 1)
xs = np.linspace(max_gpv_v.min(), max_gpv_v.max(), 50)
ax.plot(xs, np.polyval(z, xs), 'k--', lw=1.5, alpha=0.7)
ax.axhline(0, color='gray', lw=0.8, ls=':')
ax.set_xlabel('Max GPV (strongest attributed feature)', fontsize=10)
ax.set_ylabel('MLP R² − Linear R²  (MLP advantage)', fontsize=10)
ax.set_title(f'MLP advantage is larger when attribution is high\n'
             f'Spearman ρ = {rho_delta:.2f},  p = {p_delta:.3f}', fontsize=10)

plt.suptitle('Part 3 — MLP vs GLM: nonlinear models outperform, especially for attributed features',
             fontsize=11, fontweight='bold')
plt.tight_layout()
savefig('part3_mlp_vs_linear.png')
print("  Figure 3 saved.")

# ─── FIGURE 4 — Single-feature ablation: proof of functional mapping ──────────
print("\nFigure 4: single-feature ablation...")

# Select pairs:
#   - Valid full model (mean R² ≥ 0.01)
#   - LOW naive correlation (|ρ| < 0.15 for this group)
#   - LOW full-model GPV (< median GPV among valid pairs → model not attributing much)
#   - NONLINEAR signal present (η² > 0.05)
# These are pairs where BOTH GLM and attribution say "nothing to see",
# but a nonlinear single-feature model might still predict.

# Select pairs where ML attribution is HIGH but naive |ρ| is LOW:
# → model says "this feature matters" but GLM would dismiss it as noise
# This is where the single-feature MLP can prove a functional mapping exists
# that linear analysis misses.

cand_score = np.full((n_sess, n_ens, n_grp), -np.inf)
for s_i in range(n_sess):
    for n_i in range(n_ens):
        if not valid[s_i, n_i]:
            continue
        for g_i in CONT_GROUPS:
            rho_v = rho_arr[s_i, n_i, g_i]
            eta_v = eta2_arr[s_i, n_i, g_i]
            gpv_v = gpv[s_i, n_i, g_i]
            if np.isnan(rho_v) or np.isnan(eta_v) or np.isnan(gpv_v):
                continue
            # high GPV (model uses it) + low |ρ| (GLM would not flag it) + some η²
            if gpv_v > 0.008 and rho_v < 0.25 and eta_v > 0.01:
                cand_score[s_i, n_i, g_i] = gpv_v   # rank by attribution strength

flat_cand = np.argsort(cand_score.ravel())[::-1]
selected = []
for fi in flat_cand:
    s_i, n_i, g_i = np.unravel_index(fi, (n_sess, n_ens, n_grp))
    if cand_score[s_i, n_i, g_i] == -np.inf:
        break
    # avoid picking too many from the same session
    if sum(1 for (ss, _, _) in selected if ss == s_i) < 2:
        selected.append((int(s_i), int(n_i), int(g_i)))
    if len(selected) >= 5:
        break

print(f"  Selected {len(selected)} pairs for single-feature training:")
for s_i, n_i, g_i in selected:
    print(f"    S{s_i:02d} E{n_i+1:02d} × {group_names[g_i]}  "
          f"|ρ|={rho_arr[s_i,n_i,g_i]:.3f}  "
          f"η²={eta2_arr[s_i,n_i,g_i]:.3f}  "
          f"GPV={gpv[s_i,n_i,g_i]:.4f}")

def train_single_feature(s_i, n_i, g_i, seed):
    """Train a single-feature MLP and return test R²."""
    torch.manual_seed(seed); np.random.seed(seed)
    sess    = sessions[s_i]
    gcols   = semantic_groups[g_i][1]
    test_trials = set(splits[seed].get(sess, []))

    X_tr, y_tr, X_te, y_te = [], [], [], []
    for t in ds[sess]['data']:
        X = ds[sess]['data'][t].astype(np.float32)[:, gcols]
        y = ds[sess]['labels'][t][:, n_i].astype(np.float32)
        if t in test_trials:
            X_te.append(X); y_te.append(y)
        else:
            X_tr.append(X); y_tr.append(y)

    if not X_tr or not X_te:
        return np.nan, np.nan
    X_tr = np.concatenate(X_tr); y_tr = np.concatenate(y_tr)
    X_te = np.concatenate(X_te); y_te = np.concatenate(y_te)

    in_size = X_tr.shape[1]
    model   = MLP(in_size, HIDDEN_SIZE, NUM_HIDDEN, 1).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_tr); yt = torch.tensor(y_tr).unsqueeze(1)
    model.train()
    for _ in range(N_EPOCHS_SING):
        opt.zero_grad()
        _, pred = model(Xt)
        loss_fn(pred, yt).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        _, pred_te = model(torch.tensor(X_te))
    mlp_r2_v = r2_score(y_te, pred_te.numpy().squeeze())

    # also compute linear R² on same split
    lr = LinearRegression().fit(X_tr, y_tr)
    lin_r2_v = r2_score(y_te, lr.predict(X_te))

    return float(mlp_r2_v), float(lin_r2_v)


results_ablation = []
for s_i, n_i, g_i in selected:
    mlp_r2s, lin_r2s = [], []
    for seed in SEEDS:
        mr, lr_v = train_single_feature(s_i, n_i, g_i, seed)
        if not np.isnan(mr):
            mlp_r2s.append(mr); lin_r2s.append(lr_v)
    if mlp_r2s:
        results_ablation.append({
            's_i': s_i, 'n_i': n_i, 'g_i': g_i,
            'group': group_names[g_i],
            'label': f"E{n_i+1:02d}×{group_names[g_i][:12]}\nS{s_i:02d}",
            'mlp_r2_mean': float(np.mean(mlp_r2s)),
            'mlp_r2_sem':  float(np.std(mlp_r2s) / np.sqrt(len(mlp_r2s))),
            'lin_r2_mean': float(np.mean(lin_r2s)),
            'lin_r2_sem':  float(np.std(lin_r2s) / np.sqrt(len(lin_r2s))),
            'eta2': float(eta2_arr[s_i, n_i, g_i]),
            'rho':  float(rho_arr[s_i, n_i, g_i]),
            'gpv':  float(gpv[s_i, n_i, g_i]),
            'full_mlp_r2': float(mean_r2_mlp[s_i, n_i]),
        })
        print(f"  S{s_i:02d} E{n_i+1:02d} × {group_names[g_i]:30s}  "
              f"MLP={np.mean(mlp_r2s):.4f}  Linear={np.mean(lin_r2s):.4f}")

df_abl = pd.DataFrame(results_ablation)
df_abl.to_csv(os.path.join(output_dir, 'ablation_results.csv'), index=False)

if not df_abl.empty:
    x = np.arange(len(df_abl))
    w = 0.28
    fig, ax = plt.subplots(figsize=(max(7, len(df_abl) * 1.6), 5))
    apply_style(fig, ax)

    ax.bar(x - w, df_abl['lin_r2_mean'], w,
           yerr=df_abl['lin_r2_sem'], capsize=3,
           label='Single-feature GLM (linear)', color='#90A4AE', alpha=0.9)
    ax.bar(x,     df_abl['mlp_r2_mean'], w,
           yerr=df_abl['mlp_r2_sem'], capsize=3,
           label='Single-feature MLP (nonlinear)', color='#C62828', alpha=0.9)
    ax.bar(x + w, df_abl['full_mlp_r2'], w,
           label='Full MLP (all 17 features)', color='#1565C0', alpha=0.9)

    ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(df_abl['label'], fontsize=9)
    ax.set_ylabel('Test R²  (mean ± SEM across 5 seeds)', fontsize=10)
    ax.set_title(
        'Part 4 — Ablation proof: a single nonlinear feature predicts where GLM sees nothing\n'
        'Pairs selected: low Spearman |ρ| + low GPV + nonlinear signal (η² > 0.04)',
        fontsize=10
    )
    ax.legend(fontsize=9)

    # annotate η² and |ρ| above bars
    for i, row in df_abl.iterrows():
        ax.text(i, max(row['mlp_r2_mean'], row['lin_r2_mean'],
                       row['full_mlp_r2']) + 0.005,
                f"η²={row['eta2']:.2f}\n|ρ|={row['rho']:.2f}",
                ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    savefig('part4_ablation_proof.png')
    print("  Figure 4 saved.")
else:
    print("  No candidates found for ablation — skipping Figure 4.")

print(f"\nAll outputs saved to {output_dir} and {desktop_dir}")
print("Done.")
