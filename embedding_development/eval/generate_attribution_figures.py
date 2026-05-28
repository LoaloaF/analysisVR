#!/usr/bin/env python3
"""
generate_attribution_figures.py

Load pre-computed attribution arrays and generate manifest-compliant figures:
  - gpv_group_ensemble_heatmap.png  (FIG.FULL)
  - ig_per_ensemble_heatmap.png     (FIG.FULL)
  - global_vs_cond_pv_scatter.png   ((6.0, 4.2))

Run after eval_mlp_attribution.py has computed the .npy files.
"""
import os, sys, pickle
import numpy as np
import numpy.ma as ma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.figure_style import (
    FIG, DPI, FONT, FEATURE_NAMES, FEATURE_NAMES_SHORT, AXIS_LABELS,
    apply_style, add_footnote, savefig_manifest,
)

# ─── PATHS ────────────────────────────────────────────────────────────────────
base   = os.path.dirname(os.path.abspath(__file__))
root   = os.path.join(base, "..")
mdir   = os.path.join(root, "outputs", "mlps", "ensembles_multiseed")

OUT_DIRS = [
    os.path.join(root, "outputs", "mlps", "ensembles_multiseed"),
    '/mnt/c/Users/amits/Desktop',
]

# ─── LOAD ARRAYS ──────────────────────────────────────────────────────────────
all_r2   = np.load(os.path.join(mdir, "all_r2.npy"))            # (seeds, 29, 23)
gpv      = np.load(os.path.join(mdir, "importance_global_pv_semantic.npy"))  # (29, 23, 11)
cpv      = np.load(os.path.join(mdir, "importance_cond_pv_semantic.npy"))    # (29, 23, 11)
ig       = np.load(os.path.join(mdir, "importance_ig_semantic.npy"))         # (29, 23, 11)

with open(os.path.join(mdir, "semantic_groups.pkl"), "rb") as f:
    semantic_groups = pickle.load(f)

group_names = [g[0] for g in semantic_groups]   # raw column-name keys
n_groups    = len(group_names)
n_sessions, n_ensembles = gpv.shape[:2]

# ─── VALIDITY MASK ────────────────────────────────────────────────────────────
# Attribution was computed only for pairs where: not (any-seed NaN) AND R² >= 0.01.
# Use the attribution NaN pattern as the definitive valid indicator.
valid_mask = ~np.all(np.isnan(gpv), axis=-1)      # (29, 23) True = has attribution
mean_r2    = np.nanmean(all_r2, axis=0)           # (29, 23)
mean_r2_v  = np.where(valid_mask, mean_r2, np.nan)

# Ensemble-level mean R² (average across sessions where valid)
ensemble_mean_r2 = np.nanmean(mean_r2_v, axis=0)   # (23,)

# ─── CANONICAL FEATURE LABELS ─────────────────────────────────────────────────
# Map raw group_names → short canonical labels for heatmap y-axis
ytick_labels = [FEATURE_NAMES_SHORT.get(g, g) for g in group_names]

print(f"Shapes: gpv={gpv.shape}  cpv={cpv.shape}  ig={ig.shape}")
print(f"Valid pairs: {valid_mask.sum()}/{valid_mask.size}")
print(f"Ensemble mean R²: min={np.nanmin(ensemble_mean_r2):.3f}  "
      f"max={np.nanmax(ensemble_mean_r2):.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — GPV per ensemble heatmap
# x-axis: ensembles sorted by mean R²
# y-axis: semantic feature groups (canonical short names)
# cell: mean GPV across sessions (NaN = no valid sessions)
# ═══════════════════════════════════════════════════════════════════════════════
def _make_attribution_heatmap(attr, title_fallback, filename,
                              cbar_label, cmap='Blues'):
    """
    attr: (29, 23, 11) attribution array, NaN for invalid pairs.
    Generates heatmap: (11 groups) × (ensembles sorted by mean R²).

    Both GPV and IG heatmaps use an explicit fixed-position colorbar so the
    layout is pixel-identical regardless of colorbar label length.
    """
    import matplotlib.cm as mcm

    ens_mean = np.nanmean(attr, axis=0)

    r2_order      = np.argsort(ensemble_mean_r2)[::-1]
    ens_sorted    = ens_mean[r2_order, :]
    hm            = ens_sorted.T   # (11, 23) — features × ensembles
    xlabels       = [f"E{orig_idx+1:02d}" for orig_idx in r2_order]

    vmax = float(np.nanpercentile(hm[~np.isnan(hm)], 97)) if np.any(~np.isnan(hm)) else 1.0

    fig, ax = plt.subplots(figsize=FIG.FULL)
    apply_style(fig, ax)

    cmap_obj = mcm.get_cmap(cmap)
    cmap_obj.set_bad('#dddddd')

    # cbar=False — colorbar is added manually below at a fixed position so both
    # GPV and IG heatmaps are layout-identical regardless of label string length.
    sns.heatmap(hm, ax=ax, cmap=cmap_obj,
                vmin=0, vmax=vmax,
                xticklabels=xlabels,
                yticklabels=ytick_labels,
                cbar=False)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=max(6, FONT.TICK - 4),
                       rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT.TICK)
    ax.set_xlabel(AXIS_LABELS['ensemble'], fontsize=FONT.LABEL)

    # Fixed margins — same for every call to this function.
    fig.subplots_adjust(left=0.18, right=0.84, top=0.96, bottom=0.22)

    # Colorbar at fixed figure-fraction coordinates.
    cax  = fig.add_axes([0.86, 0.22, 0.018, 0.72])
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    sm   = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=FONT.TICK)
    cbar.set_label(cbar_label, fontsize=FONT.LABEL)

    n_valid = valid_mask.sum()
    add_footnote(fig,
        f"{n_valid} valid (session, ensemble) pairs (R² ≥ 0.01, 5 seeds); "
        f"sorted by ensemble mean R²")

    savefig_manifest(fig, filename, OUT_DIRS, skip_tight_layout=True)


_make_attribution_heatmap(
    gpv, "GPV per ensemble", "gpv_group_ensemble_heatmap.png",
    cbar_label=AXIS_LABELS['r2_drop'],
    cmap='Blues',
)
print("Generated gpv_group_ensemble_heatmap.png")

_make_attribution_heatmap(
    ig, "IG per ensemble", "ig_per_ensemble_heatmap.png",
    cbar_label=AXIS_LABELS['ig'],
    cmap='Blues',
)
print("Generated ig_per_ensemble_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Global PV vs Conditional PV scatter
# One point per semantic feature group (mean across all valid pairs).
# Diagonal = y=x; points below diagonal = collinearity inflated the global score.
# ═══════════════════════════════════════════════════════════════════════════════
# Apply valid mask: NaN pairs don't contribute
gpv_m = np.where(valid_mask[:, :, np.newaxis], gpv, np.nan)
cpv_m = np.where(valid_mask[:, :, np.newaxis], cpv, np.nan)

gpv_flat = gpv_m.reshape(-1, n_groups)   # (pairs, groups)
cpv_flat = cpv_m.reshape(-1, n_groups)

gpv_mean = np.nanmean(gpv_flat, axis=0)  # (11,)
cpv_mean = np.nanmean(cpv_flat, axis=0)

lim = max(np.nanmax(gpv_mean), np.nanmax(cpv_mean)) * 1.18
lim = max(lim, 0.01)

# Color points by feature type: continuous vs categorical
FEAT_COLORS = {
    'frame_raw_500msMedian':                   '#1f77b4',
    'frame_raw_abs_acc_500msMedian':           '#aec7e8',
    'frame_YawPitch_abs_vel_sum_500msMedian':  '#ff7f0e',
    'frame_YawPitch_abs_acc_sum_500msMedian':  '#ffbb78',
    'head_angle_vel':                          '#2ca02c',
    'head_angle':                              '#98df8a',
    'frame_position':                          '#d62728',
    'cue_visible':                             '#9467bd',
    'upcoming_choice':                         '#8c564b',
    'reward_window':                           '#e377c2',
    'lick_detected':                           '#7f7f7f',
}
colors = [FEAT_COLORS.get(g, '#333333') for g in group_names]

fig, ax = plt.subplots(figsize=(6.0, 4.2))
apply_style(fig, ax)

ax.plot([0, lim], [0, lim], 'k--', lw=0.9, zorder=1, label='y = x')
for i, (gn, gx, gy, gc) in enumerate(zip(group_names, gpv_mean, cpv_mean, colors)):
    if not (np.isfinite(gx) and np.isfinite(gy)):
        continue
    ax.scatter(gx, gy, s=70, color=gc, zorder=3, edgecolors='none')
    # Label with canonical short name; offset to avoid overlap
    label = FEATURE_NAMES_SHORT.get(gn, gn)
    ax.annotate(label, (gx, gy), fontsize=8,
                xytext=(4, 2), textcoords='offset points', color=gc)

ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel(AXIS_LABELS['r2_drop'], fontsize=FONT.LABEL)
ax.set_ylabel(AXIS_LABELS['cond_r2_drop'], fontsize=FONT.LABEL)
ax.legend(fontsize=FONT.LEGEND, frameon=False)

add_footnote(fig,
    f"{valid_mask.sum()} valid pairs; one point per semantic feature group; "
    f"points below diagonal → collinearity")

savefig_manifest(fig, "global_vs_cond_pv_scatter.png", OUT_DIRS)
print("Generated global_vs_cond_pv_scatter.png")
print("All attribution manifest figures done.")
