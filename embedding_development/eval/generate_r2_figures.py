#!/usr/bin/env python3
"""
generate_r2_figures.py

Load pre-computed R² arrays and generate manifest-compliant figures:
  - r2_bar_mlp.png         FIG.FULL  (9.5, 4.2) — per-ensemble MLP bar
  - r2_bar_linear.png      FIG.HALF  (4.5, 4.2) — per-ensemble Linear bar
  - r2_grand_mean_bars.png (4.0,4.0)             — 4-model grand mean
  - two_thresholds_scatter.png (5.0, 4.0)        — counts at 2 thresholds

Run after training and evaluation scripts have produced all_r2.npy files.
"""
import os, sys
import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.figure_style import (
    FIG, DPI, FONT, MODEL_COLORS, AXIS_LABELS,
    apply_style, add_footnote, savefig_manifest,
)

# ─── PATHS ────────────────────────────────────────────────────────────────────
base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, "..")

PATHS = {
    'Linear':        os.path.join(root, "outputs", "linear",           "ensembles_multiseed", "all_r2.npy"),
    'MLP':           os.path.join(root, "outputs", "mlps",             "ensembles_multiseed", "all_r2.npy"),
    'TempConv-Cont': os.path.join(root, "outputs", "cebra_eval",       "ensembles",           "all_r2.npy"),
    'TempConv-Pred': os.path.join(root, "outputs", "cebra_pred_eval",  "ensembles",           "all_r2.npy"),
}

OUT_DIRS = [
    os.path.join(root, "outputs", "cebra_comparison"),
    '/mnt/c/Users/amits/Desktop',
]
os.makedirs(OUT_DIRS[0], exist_ok=True)

SEEDS      = [42, 43, 44, 45, 46]
R2_THR_LOW  = 0.05
R2_THR_HIGH = 0.10
PREFIX      = "E"
TOP_LABEL   = 5

# ─── LOAD ─────────────────────────────────────────────────────────────────────
data = {}
for name, path in PATHS.items():
    if os.path.exists(path):
        data[name] = np.load(path)   # (5, 29, 23)
        print(f"Loaded {name}: {data[name].shape}")
    else:
        print(f"MISSING: {path}")

if not data:
    raise SystemExit("No R² data found. Run training and eval scripts first.")

n_seeds, n_sessions, n_ensembles = next(iter(data.values())).shape

def _mean_and_mask(all_r2):
    """Return (mean_r2 (29,23), invalid_mask (29,23))."""
    mean = np.nanmean(all_r2, axis=0)
    # A pair is invalid only if ALL seeds are NaN
    invalid = np.all(np.isnan(all_r2), axis=0)
    return mean, invalid

stats = {name: _mean_and_mask(v) for name, v in data.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: per-ensemble bar chart
# ═══════════════════════════════════════════════════════════════════════════════
def _per_ensemble_bar(name, figsize, filename, y_ceil=None):
    mean_r2, invalid = stats[name]
    masked  = ma.array(np.clip(mean_r2, 0, 1), mask=invalid)
    n_mean  = masked.mean(axis=0).filled(np.nan)
    order   = np.argsort(n_mean)                          # ascending → highest at right
    x       = np.arange(n_ensembles)

    pal = sns.color_palette("mako_r", as_cmap=True)(np.linspace(0.15, 0.85, n_ensembles))

    ymax = float(np.nanmax(np.clip(n_mean, 0, None))) * 1.25
    if not np.isfinite(ymax) or ymax == 0:
        ymax = 0.20
    if y_ceil is not None:
        ymax = y_ceil

    grand_mean = float(np.nanmean(n_mean))

    fig, ax = plt.subplots(figsize=figsize)
    apply_style(fig, ax)

    ax.bar(x, n_mean[order], width=0.7, color=pal, zorder=3, linewidth=0)
    ax.axhline(0, color='#888888', linestyle='--', linewidth=0.8)

    # Label top-5 ensembles
    for pos in range(n_ensembles - TOP_LABEL, n_ensembles):
        ens_idx = order[pos]
        if np.isfinite(n_mean[ens_idx]):
            ax.text(pos, n_mean[ens_idx] + ymax * 0.025,
                    f"{PREFIX}{ens_idx+1:02d}",
                    ha='center', va='bottom', fontsize=max(6, FONT.TICK - 3),
                    fontweight='bold', rotation=90)

    step = max(1, n_ensembles // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([f"{PREFIX}{order[i]+1:02d}" for i in range(0, n_ensembles, step)],
                       rotation=45, ha='right', fontsize=FONT.TICK - 2)
    ax.set_ylabel(AXIS_LABELS['r2'], fontsize=FONT.LABEL)
    ax.set_xlabel(AXIS_LABELS['ensemble'], fontsize=FONT.LABEL)
    ax.set_ylim(0, ymax)

    n_valid = int((~invalid).sum())
    add_footnote(fig,
        f"{name}; {len(SEEDS)} seeds × {n_sessions} sessions; "
        f"{n_valid} valid pairs; grand mean R² = {grand_mean:.3f}")

    savefig_manifest(fig, filename, OUT_DIRS)


# ─── MLP bar (FIG.FULL) ───────────────────────────────────────────────────────
if 'MLP' in data:
    _per_ensemble_bar('MLP', FIG.FULL, 'r2_bar_mlp.png')
    print("Generated r2_bar_mlp.png")

# ─── Linear bar (FIG.HALF, y-ceiling 0.15 to show near-zero) ─────────────────
if 'Linear' in data:
    _per_ensemble_bar('Linear', FIG.HALF, 'r2_bar_linear.png', y_ceil=0.15)
    print("Generated r2_bar_linear.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Grand mean bars — all 4 models on one chart
# ═══════════════════════════════════════════════════════════════════════════════
model_order = [n for n in ['Linear', 'MLP', 'TempConv-Cont', 'TempConv-Pred'] if n in data]
grand_means, grand_sds = [], []
for name in model_order:
    mean_r2, invalid = stats[name]
    vals = np.clip(mean_r2[~invalid], 0, 1)
    grand_means.append(np.nanmean(vals))
    grand_sds.append(np.nanstd(vals))

fig, ax = plt.subplots(figsize=(4.0, 4.0))
apply_style(fig, ax)

x     = np.arange(len(model_order))
colors = [MODEL_COLORS.get(n, '#888') for n in model_order]
ek    = dict(ecolor='k', lw=0.8, capsize=5)
bars  = ax.bar(x, grand_means, yerr=grand_sds,
               color=colors, width=0.55, alpha=0.9, error_kw=ek)

for bar, gm in zip(bars, grand_means):
    ax.text(bar.get_x() + bar.get_width() / 2,
            gm + 0.004,
            f'{gm:.3f}',
            ha='center', va='bottom',
            fontsize=FONT.TICK - 2, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(model_order, fontsize=FONT.TICK - 1, rotation=15, ha='right')
ax.set_ylabel(AXIS_LABELS['r2'], fontsize=FONT.LABEL)
ax.set_ylim(0, max(grand_means) * 1.35)

n_valid_mlp = int((~stats['MLP'][1]).sum()) if 'MLP' in data else 0
add_footnote(fig, f"Mean ± SD across valid session-ensemble pairs (R²≥0.01, {len(SEEDS)} seeds)")

savefig_manifest(fig, "r2_grand_mean_bars.png", OUT_DIRS)
print("Generated r2_grand_mean_bars.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Two-threshold bar (count of ensembles exceeding threshold per session)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(5.0, 4.0))
apply_style(fig, list(axes))

for ax, thr in zip(axes, [R2_THR_LOW, R2_THR_HIGH]):
    w_bar  = 0.8 / len(model_order)
    offsets = np.linspace(-(len(model_order)-1)/2,
                           (len(model_order)-1)/2,
                           len(model_order)) * w_bar
    x = np.arange(n_sessions)
    for name, off in zip(model_order, offsets):
        mean_r2, invalid = stats[name]
        vals = np.clip(mean_r2, 0, 1)
        vals[invalid] = np.nan
        counts = np.nansum(vals >= thr, axis=1)   # (n_sessions,)
        ax.bar(x + off, counts, width=w_bar,
               color=MODEL_COLORS.get(name, '#888'),
               label=name, alpha=0.85)

    ax.set_xticks(x[::4])
    ax.set_xticklabels([f'S{s+1}' for s in x[::4]],
                       rotation=45, ha='right', fontsize=FONT.TICK - 3)
    ax.set_ylabel(f'# ensembles ≥ {thr}', fontsize=FONT.LABEL - 2)
    ax.set_xlabel(AXIS_LABELS['session'], fontsize=FONT.LABEL - 2)

axes[0].legend(fontsize=FONT.LEGEND - 2, frameon=False, loc='upper right')
add_footnote(fig, f"Left: R²≥{R2_THR_LOW}; right: R²≥{R2_THR_HIGH}; per-session count across {n_ensembles} ensembles")
fig.tight_layout()
savefig_manifest(fig, "two_thresholds_scatter.png", OUT_DIRS)
print("Generated two_thresholds_scatter.png")
print("All R² manifest figures done.")
