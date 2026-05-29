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
import matplotlib.ticker as mticker
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

    Uses ax.imshow(aspect='auto') instead of seaborn.heatmap.  seaborn's
    internal set_aspect('equal') constraint fights fig.subplots_adjust and
    resolves differently on the first vs second call (different figure state),
    producing inconsistent layouts between the GPV and IG slides.  imshow with
    aspect='auto' gives matplotlib no aspect-ratio constraints to fight, so
    fig.subplots_adjust is the sole layout authority and the two slides are
    pixel-identical.
    """
    import matplotlib.cm as mcm

    ens_mean = np.nanmean(attr, axis=0)

    r2_order      = np.argsort(ensemble_mean_r2)[::-1]
    ens_sorted    = ens_mean[r2_order, :]
    hm            = ens_sorted.T   # (n_groups, n_ensembles)
    xlabels       = [f"E{orig_idx+1:02d}" for orig_idx in r2_order]
    n_rows, n_cols = hm.shape

    vmax = float(np.nanpercentile(hm[~np.isnan(hm)], 97)) if np.any(~np.isnan(hm)) else 1.0
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=FIG.FULL)
    apply_style(fig, ax)
    ax.yaxis.grid(False)   # disable background grid that would show through cells

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad('#dddddd')
    masked_hm = np.ma.array(hm, mask=np.isnan(hm))

    im = ax.imshow(masked_hm, aspect='auto', cmap=cmap_obj, norm=norm,
                   interpolation='nearest')

    # Major ticks at cell centers — outward, zero length (labels only; grid
    # provides the visual structure).  Must be set explicitly here because
    # scienceplots' rcParams were already overridden in apply_style, but imshow
    # may reset internal axis state between the first and second call.
    ax.tick_params(which='major', direction='out', length=0,
                   top=False, right=False, bottom=True, left=True)
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(xlabels, fontsize=max(6, FONT.TICK - 4), rotation=45, ha='right')
    ax.set_yticklabels(ytick_labels, fontsize=FONT.TICK)
    ax.set_xlabel(AXIS_LABELS['ensemble'], fontsize=FONT.LABEL)

    # White cell-divider lines via minor ticks — zero length on all sides so
    # only the grid line is drawn, no tick marks inside the cells.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    ax.tick_params(which='minor', length=0,
                   top=False, right=False, bottom=False, left=False)

    # Fixed margins and colorbar — same coordinates for GPV and IG.
    CBAR_L, CBAR_B, CBAR_W, CBAR_H = 0.86, 0.22, 0.018, 0.72
    fig.subplots_adjust(left=0.18, right=0.84, top=0.96, bottom=CBAR_B)
    cax  = fig.add_axes([CBAR_L, CBAR_B, CBAR_W, CBAR_H])
    cbar = fig.colorbar(im, cax=cax)
    # Force exactly 5 equally-spaced ticks (0 … vmax) on both GPV and IG so
    # the topmost tick lands at the same relative position on both colorbars.
    # Without this, auto-selected tick counts differ per vmax and the top tick
    # is at 94.7 % on S13 but 90.1 % on S14 — visibly different spacing.
    # MaxNLocator gives round-number ticks (e.g. 0, 0.005, 0.010) rather than
    # the non-round values from LinearLocator on an arbitrary vmax.
    cbar.locator = mticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=FONT.TICK)

    # Place the label as a title above the colorbar — avoids overlap with tick
    # numbers on the right side of the colorbar axis.
    cbar.ax.set_title(cbar_label, fontsize=FONT.LABEL, pad=5)

    n_valid = valid_mask.sum()
    add_footnote(fig,
        f"{n_valid} valid (session, ensemble) pairs (R² ≥ 0.01, 5 seeds); "
        f"sorted by ensemble mean R²")

    savefig_manifest(fig, filename, OUT_DIRS, skip_tight_layout=True)


# Use short equal-length colorbar labels so the rotated text spans the same
# pixel height on both colorbars — a longer label would start higher on the
# colorbar axis and make S13/S14 look mismatched.
_make_attribution_heatmap(
    gpv, "GPV per ensemble", "gpv_group_ensemble_heatmap.png",
    cbar_label='GPV (ΔR²)',
    cmap='Blues',
)
print("Generated gpv_group_ensemble_heatmap.png")

_make_attribution_heatmap(
    ig, "IG per ensemble", "ig_per_ensemble_heatmap.png",
    cbar_label='Mean |IG|',
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
valid_pts = [(gn, gx, gy, gc)
             for gn, gx, gy, gc in zip(group_names, gpv_mean, cpv_mean, colors)
             if np.isfinite(gx) and np.isfinite(gy)]

for gn, gx, gy, gc in valid_pts:
    ax.scatter(gx, gy, s=70, color=gc, zorder=3, edgecolors='none')

ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel('GPV — permutation importance (ΔR²)', fontsize=FONT.LABEL - 1)
ax.set_ylabel('Cond. PV (ΔR²)', fontsize=FONT.LABEL)
ax.legend(fontsize=FONT.LEGEND, frameon=False)

# Greedy label placement: render each annotation, check its pixel bbox against
# all previously placed labels, and drop it if they overlap.
# Points are processed most-isolated-first so densely-packed clusters get
# fewer labels rather than arbitrarily dropping isolated ones.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

min_sep = {i: min(
    (np.hypot(gx - gx2, gy - gy2)
     for j, (_, gx2, gy2, _) in enumerate(valid_pts) if j != i),
    default=float('inf')
) for i, (_, gx, gy, _) in enumerate(valid_pts)}

sorted_pts = sorted(enumerate(valid_pts), key=lambda t: min_sep[t[0]], reverse=True)

placed_bboxes = []
for _, (gn, gx, gy, gc) in sorted_pts:
    label = FEATURE_NAMES_SHORT.get(gn, gn)
    ann = ax.annotate(label, (gx, gy), fontsize=8,
                      xytext=(4, 2), textcoords='offset points', color=gc)
    fig.canvas.draw()
    bb = ann.get_window_extent(renderer)
    # 3-px padding so labels don't just touch
    padded = bb.expanded(1.06, 1.06)
    if any(padded.overlaps(prev) for prev in placed_bboxes):
        ann.remove()
    else:
        placed_bboxes.append(bb)

add_footnote(fig,
    f"{valid_mask.sum()} valid pairs; one point per semantic feature group; "
    f"points below diagonal = collinearity-inflated global score")

savefig_manifest(fig, "global_vs_cond_pv_scatter.png", OUT_DIRS)
print("Generated global_vs_cond_pv_scatter.png")
print("All attribution manifest figures done.")
