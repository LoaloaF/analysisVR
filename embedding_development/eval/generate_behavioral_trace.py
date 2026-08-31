#!/usr/bin/env python3
"""
generate_behavioral_trace.py

Generate behavioral_trace.png — multi-panel trace of all 11 behavioral
features (7 continuous + 4 categorical) for one representative trial.

Y-axes are scaled to the GLOBAL min/max across all sessions so the amplitude
of each feature is directly interpretable, not just relative to this trial.

Categorical features (which can take 3 or more discrete values) are shown as
step line plots only — no shading — so individual level changes are clear.

Output: behavioral_trace.png  (9.5 × 5.5")
"""
import os, sys
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

# Categorical features shown as step line plots (no fill)
CAT_FEATURES = [
    'cue_visible',
    'upcoming_choice',
    'reward_window',
    'lick_detected',
]

base     = os.path.dirname(os.path.abspath(__file__))
root     = os.path.join(base, "..")
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

# ─── GLOBAL FEATURE RANGES (across all sessions and trials) ───────────────────
# Using global min/max so y-axes are interpretable regardless of which trial
# is shown — the scale reflects the full behavioural envelope.
global_lo, global_hi = {}, {}
for feat in CONT_FEATURES:
    if feat in beh.columns:
        vals = beh[feat].dropna().astype(float).values
        global_lo[feat] = float(np.nanmin(vals))
        global_hi[feat] = float(np.nanmax(vals))

# Unique discrete levels for each categorical feature (global, for ytick labels)
cat_levels = {}
for feat in CAT_FEATURES:
    if feat in beh.columns:
        cat_levels[feat] = sorted(beh[feat].dropna().unique().astype(float))

# ─── SELECT TRIAL ─────────────────────────────────────────────────────────────
sess_mask  = beh.index.map(lambda t: t[0]) == SESSION
beh_sess   = beh[sess_mask]
trial_mask = beh_sess["trial_id"].astype(float) == float(TRIAL_ID)
beh_trial  = beh_sess[trial_mask].copy()

if len(beh_trial) == 0:
    sess_ids  = beh.index.map(lambda t: t[0]).unique()
    SESSION   = sess_ids[0]
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

# Continuous rows taller than categorical rows; taller overall canvas (5.5")
# to accommodate all 11 features without crowding.
height_ratios = [2] * n_cont + [1] * n_cat

fig = plt.figure(figsize=(FIG.FULL[0], 5.5), facecolor="white")
gs  = gridspec.GridSpec(
    n_feat_total, 1, hspace=0.05,
    left=0.30, right=0.97, top=0.96, bottom=0.09,
    height_ratios=height_ratios,
)

# Distinct palettes: tab10 for continuous features, Set2 for categorical.
cont_palette = plt.cm.tab10(np.linspace(0, 0.7, n_cont))
cat_palette  = plt.cm.Set2(np.linspace(0, 0.7, n_cat))
palette      = list(cont_palette) + list(cat_palette)

for i, feat in enumerate(all_features):
    ax      = fig.add_subplot(gs[i])
    is_last = (i == n_feat_total - 1)
    is_cat  = (i >= n_cont)

    if feat not in beh_trial.columns:
        ax.set_visible(False)
        continue

    arr = beh_trial[feat].astype(float).values

    if is_cat:
        # Step line only — no shading.  Categorical features can have 3 levels
        # (e.g. upcoming_choice ∈ {−1, 0, 1}) so a fill from 0 would be
        # ambiguous.
        levels = cat_levels.get(feat, sorted(set(arr)))
        ax.plot(rel_times, arr, drawstyle='steps-post',
                color=palette[i], linewidth=1.0)
        lo_c = min(levels) - 0.3
        hi_c = max(levels) + 0.3
        ax.set_ylim(lo_c, hi_c)
        # Show every distinct level as a tick
        tick_vals = [v for v in levels if lo_c <= v <= hi_c]
        ax.set_yticks(tick_vals)
        ax.set_yticklabels([f"{int(v)}" if v == int(v) else f"{v:.1f}"
                            for v in tick_vals],
                           fontsize=max(5, FONT.TICK - 5))
    else:
        ax.plot(rel_times, arr, color=palette[i], linewidth=0.8, alpha=0.95)
        lo = global_lo.get(feat, np.nanmin(arr))
        hi = global_hi.get(feat, np.nanmax(arr))
        pad = (hi - lo) * 0.06 if hi != lo else 0.5
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_yticks([lo, hi])
        ax.set_yticklabels([f"{lo:.2g}", f"{hi:.2g}"],
                           fontsize=max(5, FONT.TICK - 5))
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

# Session/trial label in the top margin — clear of the bottom-panel x-ticks.
fig.text(0.97, 0.99,
         f"Session {SESSION[:10]} | Trial {TRIAL_ID}",
         ha='right', va='top',
         fontsize=FONT.FOOTNOTE, color='dimgray',
         transform=fig.transFigure)

savefig_manifest(fig, "behavioral_trace.png", OUT_DIRS, skip_tight_layout=True)
print("Generated behavioral_trace.png")
