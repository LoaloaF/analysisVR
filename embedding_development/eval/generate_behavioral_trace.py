#!/usr/bin/env python3
"""
generate_behavioral_trace.py

Generate behavioral_trace.png — compact multi-panel trace of all 7 continuous
behavioral features for one representative trial.

Picks trial 170 from session 2025-01-26_21-48 (same as plot_trial_traces.py default).
Layout: 7 rows (one per continuous feature) at FIG.FULL (9.5 × 4.2").

Output: behavioral_trace.png  (FIG.FULL = 9.5 × 4.2")
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.figure_style import (
    FIG, DPI, FONT, FEATURE_NAMES,
    add_footnote, savefig_manifest,
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SESSION  = "2025-01-26_21-48"
TRIAL_ID = 170

# Continuous features — canonical order from training scripts
CONT_FEATURES = [
    'frame_raw_500msMedian',
    'frame_raw_abs_acc_500msMedian',
    'frame_YawPitch_abs_vel_sum_500msMedian',
    'frame_YawPitch_abs_acc_sum_500msMedian',
    'head_angle_vel',
    'head_angle',
    'frame_position',
]

# Categorical / binary features shown as filled step plots
CAT_FEATURES = [
    'cue_visible',
    'upcoming_choice',
    'reward_window',
    'lick_detected',
]

base   = os.path.dirname(os.path.abspath(__file__))
root   = os.path.join(base, "..")
data_dir = os.path.join(root, "outputs", "glm_input_data")
OUT_DIRS = [
    os.path.join(root, "outputs", "mlps", "ensembles_multiseed"),
    '/mnt/c/Users/amits/Desktop',
]

# ─── LOAD ─────────────────────────────────────────────────────────────────────
beh_vals = np.load(os.path.join(data_dir, "behavior_glm_input.npy"),         allow_pickle=True)
beh_idx  = np.load(os.path.join(data_dir, "behavior_glm_input_index.npy"),   allow_pickle=True)
beh_cols = np.load(os.path.join(data_dir, "behavior_glm_input_columns.npy"), allow_pickle=True)
beh = pd.DataFrame(beh_vals, index=pd.Index(beh_idx), columns=beh_cols)

sess_mask  = beh.index.map(lambda t: t[0]) == SESSION
beh_sess   = beh[sess_mask]
trial_mask = beh_sess["trial_id"].astype(float) == float(TRIAL_ID)
beh_trial  = beh_sess[trial_mask].copy()

if len(beh_trial) == 0:
    # Fallback: pick any session and trial
    sess_ids = beh.index.map(lambda t: t[0]).unique()
    SESSION  = sess_ids[0]
    sess_mask = beh.index.map(lambda t: t[0]) == SESSION
    beh_sess  = beh[sess_mask]
    TRIAL_ID  = int(beh_sess["trial_id"].iloc[100])
    beh_trial = beh_sess[beh_sess["trial_id"].astype(float) == float(TRIAL_ID)].copy()
    print(f"Fallback: session={SESSION}  trial={TRIAL_ID}")

sort_order = np.argsort(beh_trial["frame_pc_timestamp"].values)
beh_trial  = beh_trial.iloc[sort_order]

t0        = float(beh_trial["frame_pc_timestamp"].values[0])
rel_times = (beh_trial["frame_pc_timestamp"].values.astype(np.float64) - t0) / 1e6
T_total   = rel_times[-1]
print(f"Session {SESSION}  Trial {TRIAL_ID}: {len(beh_trial)} frames, {T_total:.1f}s")

# ─── FIGURE ───────────────────────────────────────────────────────────────────
all_features = CONT_FEATURES + CAT_FEATURES
n_cont       = len(CONT_FEATURES)
n_cat        = len(CAT_FEATURES)
n_feat_total = n_cont + n_cat

# Continuous rows get 2× height vs categorical (binary) rows.
height_ratios = [2] * n_cont + [1] * n_cat

fig = plt.figure(figsize=FIG.FULL, facecolor="white")
gs  = gridspec.GridSpec(
    n_feat_total, 1, hspace=0.06,
    left=0.30, right=0.97, top=0.96, bottom=0.13,
    height_ratios=height_ratios,
)

# Distinct colors: tab10 for continuous, Set2 for categorical.
cont_palette = plt.cm.tab10(np.linspace(0, 0.7, n_cont))
cat_palette  = plt.cm.Set2(np.linspace(0, 0.7, n_cat))
palette      = list(cont_palette) + list(cat_palette)

for i, feat in enumerate(all_features):
    ax         = fig.add_subplot(gs[i])
    is_last    = (i == n_feat_total - 1)
    is_cat     = (i >= n_cont)

    if feat not in beh_trial.columns:
        ax.set_visible(False)
        continue

    arr = beh_trial[feat].astype(float).values

    if is_cat:
        # Binary feature — filled step plot with 0/1 y-axis.
        ax.fill_between(rel_times, arr, step='post',
                        alpha=0.75, color=palette[i], linewidth=0)
        ax.plot(rel_times, arr, drawstyle='steps-post',
                color=palette[i], linewidth=0.7, alpha=0.9)
        ax.set_ylim(-0.15, 1.4)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['0', '1'], fontsize=max(5, FONT.TICK - 5))
    else:
        ax.plot(rel_times, arr, color=palette[i], linewidth=0.8, alpha=0.95)
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        pad = (hi - lo) * 0.12 if hi != lo else 0.5
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_yticks([lo, hi])
        ax.set_yticklabels([f"{lo:.2g}", f"{hi:.2g}"], fontsize=max(5, FONT.TICK - 5))
        ax.axhline(0, color="#d0d0d0", linewidth=0.4, zorder=0)

    canonical = FEATURE_NAMES.get(feat, feat)
    ax.set_ylabel(canonical, rotation=0, ha="right", va="center",
                  fontsize=max(6, FONT.TICK - 3), labelpad=4)

    ax.set_xlim(0, T_total)
    ax.tick_params(axis="y", length=2, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    if not is_last:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xlabel("Time (s)", fontsize=FONT.LABEL - 2)
        ax.tick_params(axis="x", labelsize=FONT.TICK - 3, length=3)

# Place session/trial label in the top margin (above all panels) to avoid
# overlapping the x-axis ticks of the bottom panel.
fig.text(0.97, 0.99,
         f"Session {SESSION[:10]} | Trial {TRIAL_ID}",
         ha='right', va='top',
         fontsize=FONT.FOOTNOTE, color='dimgray',
         transform=fig.transFigure)

savefig_manifest(fig, "behavioral_trace.png", OUT_DIRS, skip_tight_layout=True)
print("Generated behavioral_trace.png")
