#!/usr/bin/env python3
"""
plot_trial_traces.py — Save behavioral feature traces for a single trial as a static figure.

Usage:
    python plot_trial_traces.py
    python plot_trial_traces.py --trial_id 170 --session 2025-01-26_21-48
    python plot_trial_traces.py --out trial_traces.png --dpi 200
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser()
parser.add_argument("--trial_id", type=int,   default=170)
parser.add_argument("--session",  type=str,   default="2025-01-26_21-48")
parser.add_argument("--base",     type=str,   default="./outputs/glm_input_data")
parser.add_argument("--out",      type=str,   default=None)
parser.add_argument("--dpi",      type=int,   default=150)
args = parser.parse_args()

# action_enc_cols then state_enc_cols, matching training scripts
BEHAVIOR_FEATURES = [
    "frame_raw_500msMedian",
    "frame_raw_abs_acc_500msMedian",
    "frame_YawPitch_abs_vel_sum_500msMedian",
    "frame_YawPitch_abs_acc_sum_500msMedian",
    "head_angle_vel",
    "head_angle",
    "frame_position",
    "cue_visible",
    "upcoming_choice",
    "reward_window",
    "lick_detected",
]

PRETTY_NAMES = {
    "frame_raw_500msMedian":                  "Forward Velocity",
    "frame_raw_abs_acc_500msMedian":          "Forward Acceleration",
    "frame_YawPitch_abs_vel_sum_500msMedian": "Off-Rotation Velocity",
    "frame_YawPitch_abs_acc_sum_500msMedian": "Off-Rotation Acceleration",
    "head_angle_vel":                         "Head Angle Velocity",
    "head_angle":                             "Head Angle",
    "frame_position":                         "Position",
    "cue_visible":                            "Cue Visible",
    "upcoming_choice":                        "Upcoming Choice",
    "reward_window":                          "Reward Window",
    "lick_detected":                          "Lick Detected",
}

BINARY_FEATURES = {"lick_detected"}
STEP_FEATURES   = {"cue_visible", "upcoming_choice", "reward_window"}

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data …")
base = args.base
beh_vals = np.load(f"{base}/behavior_glm_input.npy", allow_pickle=True)
beh_idx  = np.load(f"{base}/behavior_glm_input_index.npy", allow_pickle=True)
beh_cols = np.load(f"{base}/behavior_glm_input_columns.npy", allow_pickle=True)
beh = pd.DataFrame(beh_vals, index=pd.Index(beh_idx), columns=beh_cols)

sess_mask  = beh.index.map(lambda t: t[0]) == args.session
beh_sess   = beh[sess_mask]
trial_mask = beh_sess["trial_id"].astype(float) == float(args.trial_id)
beh_trial  = beh_sess[trial_mask].copy()

sort_order = np.argsort(beh_trial["frame_pc_timestamp"].values)
beh_trial  = beh_trial.iloc[sort_order]

t0        = float(beh_trial["frame_pc_timestamp"].values[0])
rel_times = (beh_trial["frame_pc_timestamp"].values.astype(np.float64) - t0) / 1e6
T_total   = rel_times[-1]
print(f"Trial {args.trial_id}: {len(beh_trial)} frames, {T_total:.1f}s")

# Pre-compute global levels for categorical features from the full dataset
GLOBAL_LEVELS = {}
for feat in STEP_FEATURES | BINARY_FEATURES:
    if feat in beh.columns:
        vals = beh[feat].dropna()
        if vals.dtype == object or str(vals.dtype) == "category":
            GLOBAL_LEVELS[feat] = sorted(vals.unique())
        else:
            GLOBAL_LEVELS[feat] = sorted(vals.astype(float).unique())

# ── Figure ────────────────────────────────────────────────────────────────────
n_feat = len(BEHAVIOR_FEATURES)
fig = plt.figure(figsize=(10, n_feat * 0.8), facecolor="white")
gs  = gridspec.GridSpec(
    n_feat, 1,
    hspace=0.35,
    left=0.22, right=0.97, top=0.95, bottom=0.05,
)

# Color cycle — use a perceptually distinct, print-friendly palette
colors = plt.cm.tab10(np.linspace(0, 1, n_feat))

axes = []
for i, feat in enumerate(BEHAVIOR_FEATURES):
    ax  = fig.add_subplot(gs[i])
    raw = beh_trial[feat]
    color = colors[i]

    # label-encode string columns so they can be plotted numerically;
    # use global levels so the encoding is consistent with the full dataset
    if raw.dtype == object or str(raw.dtype) == "category":
        global_cats = GLOBAL_LEVELS.get(feat, sorted(raw.dropna().unique()))
        cat_map    = {c: idx for idx, c in enumerate(global_cats)}
        arr        = raw.map(cat_map).values.astype(float)
        tick_labels = {idx: c for c, idx in cat_map.items()}
    else:
        arr = raw.astype(float).values
        tick_labels = None

    if feat in BINARY_FEATURES:
        ax.step(rel_times, arr, color=color, linewidth=0.9, alpha=0.9, where="post")
        levels = GLOBAL_LEVELS.get(feat, [0, 1])
        ax.set_yticks(levels)
        ax.set_yticklabels([str(int(v)) for v in levels], fontsize=6)
        ax.set_ylim(min(levels) - 0.2, max(levels) + 0.2)
    elif feat in STEP_FEATURES:
        ax.step(rel_times, arr, color=color, linewidth=0.9, alpha=0.9, where="post")
        levels = GLOBAL_LEVELS.get(feat, sorted(np.unique(arr[~np.isnan(arr)]).tolist()))
        if tick_labels:
            numeric_levels = [cat_map[c] for c in levels]
            ax.set_yticks(numeric_levels)
            ax.set_yticklabels([str(c) for c in levels], fontsize=6)
            ax.set_ylim(min(numeric_levels) - 0.3, max(numeric_levels) + 0.3)
        else:
            ax.set_yticks(levels)
            ax.set_yticklabels([str(int(v)) for v in levels], fontsize=6)
            ax.set_ylim(min(levels) - 0.3, max(levels) + 0.3)
    else:
        ax.plot(rel_times, arr, color=color, linewidth=0.9, alpha=0.9)
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        ax.set_yticks([lo, hi])
        ax.set_yticklabels([f"{lo:.2g}", f"{hi:.2g}"], fontsize=6)
        ax.axhline(0, color="#cccccc", linewidth=0.5, zorder=0)

    ax.set_xlim(0, T_total)
    ax.set_ylabel(
        PRETTY_NAMES[feat],
        rotation=0, ha="right", va="center",
        fontsize=7.5, labelpad=6,
    )
    ax.tick_params(axis="y", length=2, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#aaaaaa")
    ax.spines["bottom"].set_color("#aaaaaa")
    ax.set_facecolor("white")

    if i < n_feat - 1:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(axis="x", labelsize=7, length=3)

    axes.append(ax)

fig.suptitle(
    f"Behavioral signals — Session {args.session}, Trial {args.trial_id}",
    fontsize=10, fontweight="bold", y=0.97,
)

out_path = args.out or f"trial_{args.trial_id}_traces.png"
fig.savefig(out_path, dpi=args.dpi, facecolor="white", bbox_inches="tight")
print(f"Saved → {out_path}")
