#!/usr/bin/env python3
"""
eval_cebra_compare.py

Comparison plots: TempConv-Cont vs TempConv-Pred vs MLP.

Loads:
  - outputs/cebra_eval/ensembles/all_r2.npy          (5, 29, 23)
  - outputs/cebra_pred_eval/ensembles/all_r2.npy     (5, 29, 23)
  - outputs/mlps/ensembles_multiseed/all_r2.npy      (5, 29, 23)
  - importance_global_pv_semantic.npy for each arm

Produces:
  - r2_comparison_bar.png      — per-ensemble mean R² bar for all 3 models
  - r2_comparison_scatter.png  — scatter: TempConv vs MLP per (session, ensemble)
  - importance_comparison.png  — side-by-side feature importance bars
"""
import os
import pickle
import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'utils'))
from figure_style import (
    FIG, DPI, FONT, LINE, MARKER,
    MODEL_COLORS, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

# ═══════════════════════════════════ PATHS ════════════════════════════════════
cebra_dir      = "./outputs/cebra_64d_eval/ensembles"
cebra_pred_dir = "./outputs/cebra_pred_64d_eval/ensembles"
mlp_dir        = "./outputs/mlps/ensembles_multiseed"
out_dir        = "./outputs/cebra_comparison"
os.makedirs(out_dir, exist_ok=True)

SEEDS            = [42, 43, 44, 45, 46]
R2_THRESHOLD     = 0.01
GOOD_THRESHOLD   = 0.1    # threshold for "good" ensemble in per-session count
GOOD_THRESHOLD2  = 0.05   # second threshold for two-threshold scatter
TOP_LABEL        = 5
prefix_name      = "E"


DESK = '/mnt/c/Users/amits/Desktop/'

def savefig(name):
    savefig_manifest(plt.gcf(), name, [out_dir, DESK])
    print(f"Saved {os.path.join(out_dir, name)}")


def load_r2(directory, filename="all_r2.npy"):
    p = os.path.join(directory, filename)
    if not os.path.exists(p):
        print(f"  WARNING: {p} not found — skipping")
        return None
    return np.load(p)


def mean_r2_valid(all_r2):
    """(sessions, neurons) mean R² ignoring NaN."""
    with np.errstate(all='ignore'):
        return np.nanmean(all_r2, axis=0)


# ═══════════════════════════ LOAD DATA ════════════════════════════════════════
r2_cebra  = load_r2(cebra_dir)
r2_cpred  = load_r2(cebra_pred_dir)
r2_mlp    = load_r2(mlp_dir)

available = {
    "TempConv-Cont":  r2_cebra,
    "TempConv-Pred":  r2_cpred,
    "MLP":         r2_mlp,
}
available = {k: v for k, v in available.items() if v is not None}
print(f"Loaded: {list(available.keys())}")

if len(available) == 0:
    print("No data available. Run eval_cebra_seeds.py first.")
    raise SystemExit(0)

# Get common shape
shapes = [v.shape for v in available.values()]
num_sessions = shapes[0][1]
num_neurons  = shapes[0][2]

mean_r2s = {k: mean_r2_valid(v) for k, v in available.items()}
masks    = {k: np.all(np.isnan(v), axis=0) for k, v in available.items()}

# ══════════════════════ GRAND-MEAN R² SUMMARY BAR ════════════════════════════
model_names  = list(available.keys())
grand_means  = []
grand_stds   = []
COLORS = {'TempConv-Cont': '#2196F3', 'TempConv-Pred': '#FF9800', 'MLP': '#4CAF50'}

for name in model_names:
    m  = mean_r2s[name]
    mk = masks[name]
    vals = m[~mk]
    vals = np.clip(vals, 0, 1)
    grand_means.append(np.nanmean(vals))
    grand_stds.append(np.nanstd(vals))

fig, ax = plt.subplots(figsize=(6, 4))
apply_style(fig, ax)
x = np.arange(len(model_names))
bars = ax.bar(x, grand_means,
              color=[COLORS.get(n, 'steelblue') for n in model_names],
              alpha=0.85, width=0.5)
for bar, gm in zip(bars, grand_means):
    ax.text(bar.get_x() + bar.get_width() / 2, gm + 0.005,
            f'{gm:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylabel("Grand Mean R² (valid session-ensemble pairs)", fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
savefig("r2_comparison_grand_mean.png")

# ══════════════════ PER-ENSEMBLE R² BAR (one figure per model) ════════════════
for name, all_r2 in available.items():
    mr2  = mean_r2s[name]
    mk   = masks[name]
    masked_arr = ma.array(np.clip(mr2, 0, 1), mask=mk)
    n_mean = masked_arr.mean(axis=0).filled(np.nan)
    n_std  = masked_arr.std(axis=0).filled(np.nan)
    order  = np.argsort(n_mean)
    x      = np.arange(num_neurons)
    pal    = sns.color_palette("mako_r", as_cmap=True)(np.linspace(0.15, 0.85, num_neurons))

    ymax = max(float(np.nanmax(np.clip(n_mean, 0, None))) * 1.25, 0.05)
    if not np.isfinite(ymax):
        ymax = 0.20

    fig, ax = plt.subplots(figsize=(16, 5))
    apply_style(fig, ax)
    ax.bar(x, n_mean[order], width=0.7, color=pal, zorder=3, linewidth=0)
    ax.axhline(0, color='firebrick', linestyle='--', linewidth=1)
    ax.set_xticks(x[::3])
    ax.set_xticklabels([f"{prefix_name}{order[i]+1:02d}" for i in range(0, num_neurons, 3)],
                       rotation=45, ha='right', fontsize=8)
    for pos in range(num_neurons - TOP_LABEL, num_neurons):
        if np.isfinite(n_mean[order[pos]]):
            ax.text(pos, n_mean[order[pos]] + ymax * 0.02,
                    f"{prefix_name}{order[pos]+1:02d}",
                    ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Mean R²")
    ax.set_xlabel(f"Ensemble (sorted by mean R²)")
    ax.set_title(f"{name} — Per-Ensemble Ridge Probe R²\n"
                 f"(mean ± SD across {len(SEEDS)} seeds × {num_sessions} sessions  |  "
                 f"grand mean = {np.nanmean(n_mean):.3f})")
    valid_counts = (~mk).sum(axis=0)
    ax.annotate(
        f"n ensembles = {num_neurons}  |  "
        f"median sessions per ensemble = {int(np.median(valid_counts))}  |  "
        f"grand mean R² = {np.nanmean(n_mean):.3f}",
        xy=(0.01, 0.97), xycoords='axes fraction',
        va='top', ha='left', fontsize=9, color='dimgray',
    )
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fname = f"r2_bar_{name.lower().replace('-', '_').replace(' ', '_')}.png"
    savefig(fname)
    print(f"  {name}: grand mean R² = {np.nanmean(n_mean):.3f}")

# ══════════════════════ SCATTER: TempConv vs MLP per pair ══════════════════════
if 'TempConv-Cont' in available and 'MLP' in available:
    mr2_c = mean_r2s['TempConv-Cont']
    mr2_m = mean_r2s['MLP']
    mk_c  = masks['TempConv-Cont']
    mk_m  = masks['MLP']
    both_valid = ~mk_c & ~mk_m

    c_vals = np.clip(mr2_c[both_valid], 0, 1)
    m_vals = np.clip(mr2_m[both_valid], 0, 1)

    if len(c_vals) > 1:
        r_val, _ = pearsonr(c_vals, m_vals)
        diff     = c_vals - m_vals
        pct_wins = (diff > 0).mean() * 100

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        apply_style(fig, axes)

        ax = axes[0]
        lim = max(c_vals.max(), m_vals.max()) * 1.08
        ax.scatter(m_vals, c_vals, alpha=0.25, s=10, color='steelblue')
        ax.plot([0, lim], [0, lim], 'k--', lw=0.9, label='y = x')
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        ax.set_xlabel("MLP R²")
        ax.set_ylabel("TempConv-Cont Ridge Probe R²")
        ax.set_title(f"TempConv-Cont vs MLP\n(n={len(c_vals)} pairs, r={r_val:.3f})")
        ax.annotate(f"TempConv wins: {pct_wins:.0f}%\nMLP mean = {m_vals.mean():.3f}\n"
                    f"TempConv mean = {c_vals.mean():.3f}",
                    xy=(0.05, 0.95), xycoords='axes fraction', va='top', fontsize=9)
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

        ax = axes[1]
        ax.hist(diff, bins=40, color='mediumpurple', edgecolor='white', alpha=0.85)
        ax.axvline(0,          color='black', lw=0.9, ls='--')
        ax.axvline(diff.mean(), color='red',  lw=1.2, label=f'mean diff = {diff.mean():.3f}')
        ax.set_xlabel("TempConv R² − MLP R²")
        ax.set_ylabel("Count")
        ax.set_title(f"TempConv-Cont improvement over MLP\n({pct_wins:.0f}% of pairs)")
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

        plt.suptitle("TempConv-Cont vs MLP — R² comparison", fontweight='bold')
        plt.tight_layout()
        savefig("r2_scatter_tempconv_vs_mlp.png")

if 'TempConv-Pred' in available and 'MLP' in available:
    mr2_p = mean_r2s['TempConv-Pred']
    mr2_m = mean_r2s['MLP']
    mk_p  = masks['TempConv-Pred']
    mk_m  = masks['MLP']
    both_valid = ~mk_p & ~mk_m

    p_vals = np.clip(mr2_p[both_valid], 0, 1)
    m_vals = np.clip(mr2_m[both_valid], 0, 1)

    if len(p_vals) > 1:
        r_val, _ = pearsonr(p_vals, m_vals)
        diff     = p_vals - m_vals
        pct_wins = (diff > 0).mean() * 100

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        apply_style(fig, axes)

        ax = axes[0]
        lim = max(p_vals.max(), m_vals.max()) * 1.08
        ax.scatter(m_vals, p_vals, alpha=0.25, s=10, color='darkorange')
        ax.plot([0, lim], [0, lim], 'k--', lw=0.9, label='y = x')
        ax.set_xlabel("MLP R²")
        ax.set_ylabel("TempConv-Pred Ridge Probe R²")
        ax.set_title(f"TempConv-Pred vs MLP\n(n={len(p_vals)} pairs, r={r_val:.3f})")
        ax.annotate(f"TempConv-Pred wins: {pct_wins:.0f}%\nMLP mean = {m_vals.mean():.3f}\n"
                    f"TempConv-Pred mean = {p_vals.mean():.3f}",
                    xy=(0.05, 0.95), xycoords='axes fraction', va='top', fontsize=9)
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

        ax = axes[1]
        ax.hist(diff, bins=40, color='darkorange', edgecolor='white', alpha=0.85)
        ax.axvline(0,          color='black', lw=0.9, ls='--')
        ax.axvline(diff.mean(), color='red',  lw=1.2, label=f'mean diff = {diff.mean():.3f}')
        ax.set_xlabel("TempConv-Pred R² − MLP R²")
        ax.set_ylabel("Count")
        ax.set_title(f"TempConv-Pred improvement over MLP\n({pct_wins:.0f}% of pairs)")
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

        plt.suptitle("TempConv-Pred vs MLP — R² comparison", fontweight='bold')
        plt.tight_layout()
        savefig("r2_scatter_tempconvpred_vs_mlp.png")

# ═══════════════════ TempConv-ContONT vs TempConv-PredRED scatter ════════════════════════
if 'TempConv-Cont' in available and 'TempConv-Pred' in available:
    mr2_c = mean_r2s['TempConv-Cont']
    mr2_p = mean_r2s['TempConv-Pred']
    mk_c  = masks['TempConv-Cont']
    mk_p  = masks['TempConv-Pred']
    both  = ~mk_c & ~mk_p

    c_vals = np.clip(mr2_c[both], 0, 1)
    p_vals = np.clip(mr2_p[both], 0, 1)

    if len(c_vals) > 1:
        r_val, _ = pearsonr(c_vals, p_vals)
        diff     = p_vals - c_vals
        pct_pred_wins = (diff > 0).mean() * 100

        fig, ax = plt.subplots(figsize=(6, 5))
        apply_style(fig, ax)
        lim = max(c_vals.max(), p_vals.max()) * 1.08
        ax.scatter(c_vals, p_vals, alpha=0.25, s=10, color='mediumpurple')
        ax.plot([0, lim], [0, lim], 'k--', lw=0.9, label='y = x')
        ax.set_xlabel("TempConv-Cont Ridge Probe R²")
        ax.set_ylabel("TempConv-Pred Ridge Probe R²")
        ax.set_title(f"TempConv-Cont vs TempConv-Pred\n(n={len(c_vals)} pairs, r={r_val:.3f})")
        ax.annotate(f"Pred wins: {pct_pred_wins:.0f}%\n"
                    f"Cont mean = {c_vals.mean():.3f}\nPred mean = {p_vals.mean():.3f}",
                    xy=(0.05, 0.95), xycoords='axes fraction', va='top', fontsize=9)
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        savefig("r2_scatter_cont_vs_pred.png")

# ═══════════════════ FEATURE IMPORTANCE COMPARISON ═══════════════════════════
imp_paths = {
    'TempConv-Cont': os.path.join(cebra_dir,      "importance_global_pv_semantic.npy"),
    'TempConv-Pred': os.path.join(cebra_pred_dir, "importance_global_pv_semantic.npy"),
    'MLP':        os.path.join(mlp_dir,         "importance_global_pv_semantic.npy"),
}
groups_path = os.path.join(cebra_dir, "semantic_groups.pkl")

imp_available = {k: np.load(v) for k, v in imp_paths.items()
                 if os.path.exists(v)}
print(f"Importance data available: {list(imp_available.keys())}")

if imp_available and os.path.exists(groups_path):
    with open(groups_path, 'rb') as f:
        semantic_groups = pickle.load(f)
    group_names = [g[0] for g in semantic_groups]
    n_groups    = len(group_names)

    fig, axes = plt.subplots(1, len(imp_available), figsize=(6 * len(imp_available), 5),
                             sharey=False)
    apply_style(fig, axes if hasattr(axes, '__iter__') else [axes])
    if len(imp_available) == 1:
        axes = [axes]

    for ax, (name, imp_arr) in zip(axes, imp_available.items()):
        # imp_arr shape: (sessions, neurons, groups) — already masked
        flat     = imp_arr.reshape(-1, n_groups)
        g_mean   = np.nanmean(flat, axis=0)
        g_std    = np.nanstd( flat, axis=0)
        order    = np.argsort(g_mean)[::-1]

        ax.bar(np.arange(n_groups), g_mean[order], yerr=g_std[order],
               color=COLORS.get(name, 'steelblue'), capsize=4, alpha=0.8)
        ax.set_xticks(np.arange(n_groups))
        ax.set_xticklabels([group_names[i] for i in order], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean R² drop')
        ax.set_title(f'{name}')
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle("Semantic Feature Importance Comparison", fontsize=13, fontweight='bold')
    plt.tight_layout()
    savefig("importance_comparison.png")

    # Also: overlaid bar chart
    if len(imp_available) >= 2:
        fig, ax = plt.subplots(figsize=(12, 5))
        apply_style(fig, ax)
        x    = np.arange(n_groups)
        w    = 0.25
        offsets = np.linspace(-(len(imp_available)-1)/2, (len(imp_available)-1)/2, len(imp_available)) * w

        for (name, imp_arr), offset in zip(imp_available.items(), offsets):
            flat   = imp_arr.reshape(-1, n_groups)
            g_mean = np.nanmean(flat, axis=0)
            g_std  = np.nanstd( flat, axis=0)
            ax.bar(x + offset, g_mean, width=w, yerr=g_std,
                   label=name, color=COLORS.get(name, 'steelblue'),
                   capsize=3, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(group_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean R² drop')
        ax.set_title('Semantic Feature Importance — All Models Overlaid', fontsize=13)
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        savefig("importance_comparison_overlaid.png")

# ══════════ PER-SESSION GOOD ENSEMBLE COUNT (S64) ════════════════════════════
# Count ensembles per session with mean R² ≥ GOOD_THRESHOLD for each model.
if len(available) >= 2:
    sess_idx = np.arange(num_sessions)
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    apply_style(fig, ax)
    w_bar   = 0.8 / len(available)
    offsets = np.linspace(-(len(available)-1)/2, (len(available)-1)/2, len(available)) * w_bar
    for (name, all_r2), off in zip(available.items(), offsets):
        mr2  = mean_r2s[name]
        mk   = masks[name]
        vals = np.clip(mr2, 0, 1)
        vals[mk] = np.nan
        counts = np.sum(vals >= GOOD_THRESHOLD, axis=1)  # (sessions,)
        ax.bar(sess_idx + off, counts, width=w_bar,
               color=COLORS.get(name, 'steelblue'), label=name, alpha=0.85)
    ax.set_xticks(sess_idx)
    ax.set_xticklabels([f'S{s+1}' for s in sess_idx], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(f'# ensembles with R²≥{GOOD_THRESHOLD}', fontsize=9)
    ax.set_title(f'Good Ensembles per Session (R²≥{GOOD_THRESHOLD})', fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    savefig("per_session_good_ensembles.png")

# ══════════ MLP vs TempConv-PRED SCATTER (S65) ═══════════════════════════════════
if 'TempConv-Pred' in available and 'MLP' in available:
    mr2_p  = mean_r2s['TempConv-Pred']
    mr2_m  = mean_r2s['MLP']
    mk_p   = masks['TempConv-Pred']
    mk_m   = masks['MLP']
    both   = ~mk_p & ~mk_m
    p_vals = np.clip(mr2_p[both], 0, 1)
    m_vals = np.clip(mr2_m[both], 0, 1)
    if len(p_vals) > 1:
        r_val, _ = pearsonr(p_vals, m_vals)
        fig, ax = plt.subplots(figsize=(9.5, 4.5))
        apply_style(fig, ax)
        lim = max(p_vals.max(), m_vals.max()) * 1.08
        ax.scatter(m_vals, p_vals, alpha=0.25, s=14, color='darkorange', linewidths=0)
        ax.plot([0, lim], [0, lim], 'k--', lw=0.9, label='y = x')
        pct_pred = (p_vals > m_vals).mean() * 100
        ax.annotate(f'TempConv-Pred wins: {pct_pred:.0f}%\nMLP mean = {m_vals.mean():.3f}\n'
                    f'TempConv-Pred mean = {p_vals.mean():.3f}\nr = {r_val:.3f}',
                    xy=(0.04, 0.96), xycoords='axes fraction', va='top', fontsize=8)
        ax.set_xlabel('MLP R²', fontsize=9)
        ax.set_ylabel('TempConv-Pred R²', fontsize=9)
        ax.set_title('MLP vs TempConv-Pred — Per (Session × Ensemble) Pair', fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        savefig("scatter_mlp_vs_tempconv_pred.png")

# ══════════ TWO-THRESHOLD SCATTER (S66) ═══════════════════════════════════════
if 'MLP' in available:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    apply_style(fig, axes)
    for ax, thr, col in zip(axes,
                            [GOOD_THRESHOLD2, GOOD_THRESHOLD],
                            ['steelblue', 'darkorange']):
        x_vals, y_vals, labels = [], [], []
        for name, all_r2 in available.items():
            mr2 = np.clip(mean_r2s[name], 0, 1)
            mk  = masks[name]
            mr2[mk] = np.nan
            counts = np.nansum(mr2 >= thr, axis=1)
            x_vals.append(counts)
            labels.append(name)
        x = np.arange(num_sessions)
        w_b = 0.8 / len(x_vals)
        offs = np.linspace(-(len(x_vals)-1)/2, (len(x_vals)-1)/2, len(x_vals)) * w_b
        for counts, nm, off in zip(x_vals, labels, offs):
            ax.bar(x + off, counts, width=w_b,
                   color=COLORS.get(nm, col), label=nm, alpha=0.85)
        ax.set_xticks(x[::5])
        ax.set_xticklabels([f'S{x[i]+1}' for i in range(0, len(x), 5)], rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(f'# ensembles ≥ {thr}', fontsize=8)
        ax.set_title(f'R² ≥ {thr}', fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    plt.suptitle('Model Comparison at Two R² Thresholds', fontsize=10)
    plt.tight_layout()
    savefig("two_thresholds_scatter.png")

print(f"\nAll comparison outputs saved to: {out_dir}")
