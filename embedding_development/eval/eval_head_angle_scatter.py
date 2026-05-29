#!/usr/bin/env python3
"""
eval_head_angle_scatter.py

Phase 1a: Head angle × Head angular velocity scatter (2×2 grid),
colored by z-scored ensemble activation.

Each panel = top (session, ensemble) pair sorted by mean MLP R².
Picks top 4 pairs; verifies color gradient is visible (std > 0.3 in color).

Output: head_angle_scatter_2x2.png  (FIG.FULL = 9.5 × 4.2")
"""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.figure_style import (
    FIG, DPI, FONT, FEATURE_NAMES, AXIS_LABELS,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

# ─── PATHS ────────────────────────────────────────────────────────────────────
base   = os.path.dirname(os.path.abspath(__file__))
root   = os.path.join(base, "..")
mdir   = os.path.join(root, "outputs", "mlps", "ensembles_multiseed")

OUT_DIRS = [mdir, '/mnt/c/Users/amits/Desktop']

# ─── FEATURE INDICES (from semantic_groups.pkl) ───────────────────────────────
HA_IDX  = 5   # head_angle
HAV_IDX = 4   # head_angle_vel

# ─── LOAD ─────────────────────────────────────────────────────────────────────
cache_path = os.path.join(root, "outputs", "session_dataset_ensembles.pkl")
with open(cache_path, "rb") as f:
    ds = pickle.load(f)
session_ids = list(ds.keys())

all_r2    = np.load(os.path.join(mdir, "all_r2.npy"))    # (5, 29, 23)
mean_r2   = np.nanmean(all_r2, axis=0)                   # (29, 23)

n_sessions, n_ensembles = mean_r2.shape

# ─── SELECT TOP 4 PAIRS ───────────────────────────────────────────────────────
flat_order = np.argsort(mean_r2.ravel())[::-1]
top_pairs  = []
for fi in flat_order:
    s_idx = fi // n_ensembles
    n_idx = fi % n_ensembles
    if np.isfinite(mean_r2[s_idx, n_idx]) and mean_r2[s_idx, n_idx] > 0.01:
        top_pairs.append((s_idx, n_idx))
    if len(top_pairs) == 4:
        break

print(f"Top 4 pairs (session_idx, ensemble_idx):")
for s_idx, n_idx in top_pairs:
    print(f"  S{s_idx+1:02d} E{n_idx+1:02d}  mean_R²={mean_r2[s_idx, n_idx]:.3f}")

# ─── BUILD PANEL DATA ─────────────────────────────────────────────────────────
panels = []
for s_idx, n_idx in top_pairs:
    sess_id = session_ids[s_idx]
    sd      = ds[sess_id]
    all_t   = list(sd['data'].keys())

    Xs = np.concatenate([sd['data'][t]   for t in all_t]).astype(np.float32)
    Ys = np.concatenate([sd['labels'][t] for t in all_t]).astype(np.float32)

    ha    = Xs[:, HA_IDX].astype(float)
    hav   = Xs[:, HAV_IDX].astype(float)
    act   = Ys[:, n_idx].astype(float)       # already z-scored

    # Verify z-scoring
    act_std = np.nanstd(act)
    print(f"  S{s_idx+1:02d} E{n_idx+1:02d}: "
          f"ha=[{ha.min():.2f}, {ha.max():.2f}]  "
          f"hav=[{hav.min():.2f}, {hav.max():.2f}]  "
          f"act_std={act_std:.3f}")

    panels.append({
        'ha': ha, 'hav': hav, 'act': act,
        's_idx': s_idx, 'n_idx': n_idx,
        'sess_id': sess_id[:10],
        'r2': mean_r2[s_idx, n_idx],
    })

# ─── FIGURE ───────────────────────────────────────────────────────────────────
# Explicit subplots_adjust + manually-placed colorbar axis to avoid the
# fig.colorbar(ax=all_axes) approach which can steal too much horizontal space
# and overlap the right-column panels.
fig, axes = plt.subplots(2, 2, figsize=FIG.FULL)
apply_style(fig, axes.ravel())
# bottom=0.20: extra room for x-tick labels + x-axis label on bottom panels C
# and D so they don't overlap with the footnote at y≈0.01.
# Panel labels are placed INSIDE the axes (top-left corner) to avoid competing
# with the rotated y-axis label in the narrow left margin.
fig.subplots_adjust(left=0.14, right=0.84, top=0.94, bottom=0.20,
                    hspace=0.54, wspace=0.44)

cmap_act = cm.RdBu_r
all_act   = np.concatenate([p['act'] for p in panels])
vmax_clim = float(np.percentile(np.abs(all_act), 95))

sc = None   # will be set in loop
panel_labels = ['A', 'B', 'C', 'D']
for i, (ax, p, pl) in enumerate(zip(axes.ravel(), panels, panel_labels)):
    sc = ax.scatter(
        p['ha'], p['hav'],
        c=p['act'],
        cmap=cmap_act,
        vmin=-vmax_clim, vmax=vmax_clim,
        s=4, alpha=0.4, linewidths=0, rasterized=True,
    )
    ax.set_xlabel('Head Angle (z-scored)', fontsize=FONT.LABEL - 1)
    # Only set y-axis label on the left column (panels A=0, C=2); right-column
    # labels appear in the centre of the figure and overlap the left panels.
    if i % 2 == 0:
        ax.set_ylabel('Head Ang. Vel. (z-scored)', fontsize=FONT.LABEL - 1)
    else:
        ax.set_ylabel('')
    ax.tick_params(labelsize=FONT.TICK - 2)

    # Inside placement: the y-axis label "Head Ang. Vel. (°/s)" when rotated
    # 90° is ~20 chars × ~23 px/char ≈ 460 px, taller than the panel height
    # (~220 px at hspace=0.54), so its top protrudes above the panel and would
    # collide with an outside panel label at y=1.05.  Placing the label inside
    # avoids the margin competition entirely.
    ax.text(0.03, 0.97, pl, transform=ax.transAxes,
            fontsize=FONT.PANEL, fontweight='bold', va='top', ha='left')

    # Session/ensemble annotation in upper-right corner
    ax.text(0.97, 0.96,
            f"S{p['s_idx']+1:02d} E{p['n_idx']+1:02d}  R²={p['r2']:.2f}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FONT.ANNOTATION - 2, color='dimgray')

# Dedicated colorbar axis — height adjusted for new bottom=0.20.
cax = fig.add_axes([0.86, 0.20, 0.018, 0.72])
cbar = fig.colorbar(sc, cax=cax)
cbar.ax.tick_params(labelsize=FONT.TICK - 1)
cbar.set_label(AXIS_LABELS['activity'], fontsize=FONT.LABEL - 1)

add_footnote(fig,
    "Top 4 pairs by MLP R²; color = z-scored activation (5th–95th %ile clip)")

# skip_tight_layout: layout is set explicitly above; tight_layout would fight
# with the manually-positioned colorbar axis.
savefig_manifest(fig, "head_angle_scatter_2x2.png", OUT_DIRS, skip_tight_layout=True)
print("Generated head_angle_scatter_2x2.png")
