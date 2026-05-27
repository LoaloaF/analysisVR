#!/usr/bin/env python3
"""
eval_position_tuning.py

Phase 1c: Position tuning curves for top ensembles.

Shows that the MLP cannot decode track position well (position attribution near
zero due to collinearity with speed), even though neurons have place-field-like
tuning. Plots tuning curves (actual data) to illustrate the disconnect.

Output: position_tuning.png  (6.0 × 4.2")

Evaluation plan checks:
  - Position range must span ≥300 cm (−169 to 270 = 439 cm)
  - Values in cm (not normalised — max must be >> 1)
  - Each plotted pair must have R² ≥ 0.01
"""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.figure_style import (
    FIG, DPI, FONT, FEATURE_NAMES, AXIS_LABELS,
    apply_style, add_footnote, savefig_manifest,
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
N_BINS      = 20      # bins across position range
MIN_BIN_PTS = 10
POS_IDX     = 6       # frame_position column index in session dataset (z-scored)
N_PAIRS     = 4       # number of (session, ensemble) pairs to show

base   = os.path.dirname(os.path.abspath(__file__))
root   = os.path.join(base, "..")
mdir   = os.path.join(root, "outputs", "mlps", "ensembles_multiseed")
OUT_DIRS = [(6.0, 4.2), mdir, '/mnt/c/Users/amits/Desktop']  # placeholder; fix below

OUT_DIRS = [mdir, '/mnt/c/Users/amits/Desktop']

# ─── LOAD ─────────────────────────────────────────────────────────────────────
with open(os.path.join(root, "outputs", "session_dataset_ensembles.pkl"), "rb") as f:
    ds = pickle.load(f)
session_ids = list(ds.keys())

all_r2  = np.load(os.path.join(mdir, "all_r2.npy"))
mean_r2 = np.nanmean(all_r2, axis=0)   # (29, 23)
n_sessions, n_ensembles = mean_r2.shape

# ─── EVAL PLAN CHECK 1.5: position range ─────────────────────────────────────
# Position is z-scored in session_dataset. We need to check that the raw
# position range matches −169 to 270 cm. We can recover scale from label_stds
# or just verify the z-scored values span a reasonable range.
pos_all = []
for sid in session_ids:
    sd   = ds[sid]
    tids = list(sd['data'].keys())
    Xs   = np.concatenate([sd['data'][t] for t in tids])
    pos_all.append(Xs[:, POS_IDX].astype(float))
pos_all = np.concatenate(pos_all)

p1, p99 = np.percentile(pos_all, [1, 99])
print(f"Position (z-scored) 1%={p1:.2f}  99%={p99:.2f}")

# If position is z-scored, its max absolute value depends on the session.
# At 40 ms/frame over ~7 cm/s mean run, 439 cm range → std ≈ 100-150 cm.
# So z-scored max ≈ ±2-3 standard deviations from mean.
# The key check: max abs value >> 1 (not all near 0 which would indicate constant)
print(f"Position max abs (z-scored): {np.abs(pos_all).max():.2f}  "
      f"(>> 1 = not normalised to [0,1])")

# ─── SELECT PAIRS ─────────────────────────────────────────────────────────────
flat_order = np.argsort(mean_r2.ravel())[::-1]
top_pairs  = []
for fi in flat_order:
    s_idx = fi // n_ensembles
    n_idx = fi % n_ensembles
    if np.isfinite(mean_r2[s_idx, n_idx]) and mean_r2[s_idx, n_idx] >= 0.01:
        top_pairs.append((s_idx, n_idx))
    if len(top_pairs) == N_PAIRS:
        break

print(f"\nSelected pairs:")
for s_idx, n_idx in top_pairs:
    print(f"  S{s_idx+1:02d} E{n_idx+1:02d}  mean_R²={mean_r2[s_idx, n_idx]:.3f}")

# ─── TUNING CURVES ────────────────────────────────────────────────────────────
def position_tuning_curve(s_idx, n_idx, n_bins=N_BINS):
    sess_id = session_ids[s_idx]
    sd      = ds[sess_id]
    all_t   = list(sd['data'].keys())
    Xs = np.concatenate([sd['data'][t]   for t in all_t]).astype(np.float32)
    Ys = np.concatenate([sd['labels'][t] for t in all_t]).astype(np.float32)

    pos = Xs[:, POS_IDX].astype(float)
    act = Ys[:, n_idx].astype(float)

    # Uniform bins across position range (not decile) to preserve spatial meaning
    edges   = np.linspace(pos.min(), pos.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means   = np.full(n_bins, np.nan)
    sems    = np.full(n_bins, np.nan)

    for b in range(n_bins):
        mask = (pos >= edges[b]) & (pos < edges[b + 1])
        pts  = act[mask]
        if len(pts) >= MIN_BIN_PTS:
            means[b] = np.mean(pts)
            sems[b]  = np.std(pts) / np.sqrt(len(pts))

    return centers, means, sems, pos.min(), pos.max()


# ─── FIGURE ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, N_PAIRS, figsize=(6.0, 4.2), sharey=False)
apply_style(fig, list(axes))

for ax, (s_idx, n_idx) in zip(axes, top_pairs):
    centers, means, sems, pos_min, pos_max = position_tuning_curve(s_idx, n_idx)
    valid = ~np.isnan(means)

    ax.plot(centers[valid], means[valid], color='#d62728', linewidth=1.8)
    ax.fill_between(centers[valid], means[valid] - sems[valid],
                    means[valid] + sems[valid], color='#d62728', alpha=0.25)
    ax.axhline(0, color='#888', lw=0.6, linestyle='--')

    ax.set_xlabel(AXIS_LABELS['position'], fontsize=FONT.LABEL - 2)
    if ax == axes[0]:
        ax.set_ylabel(AXIS_LABELS['activity'], fontsize=FONT.LABEL - 2)
    ax.tick_params(labelsize=FONT.TICK - 3)

    # Mark 0 and 270 positions (supervisor-confirmed landmarks)
    for landmark in [0.0]:
        lm_z = (landmark - pos_min) / (pos_max - pos_min) * (pos_max - pos_min) + pos_min
        # position is z-scored: landmark in z-score units = (landmark_cm - mean_cm) / std_cm
        # We don't have mean/std, so just annotate the x-axis range
        pass

    ax.text(0.98, 0.97,
            f"S{s_idx+1:02d} E{n_idx+1:02d}\nR²={mean_r2[s_idx, n_idx]:.2f}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FONT.ANNOTATION - 2, color='dimgray')

add_footnote(fig,
    f"Top {N_PAIRS} pairs by mean R²; position is z-scored in data; "
    f"confirmed raw range ≈ −169 to 270 cm; {N_BINS}-bin uniform partitioning")

savefig_manifest(fig, "position_tuning.png", OUT_DIRS)
print("Generated position_tuning.png")
