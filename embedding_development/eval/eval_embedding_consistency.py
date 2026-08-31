#!/usr/bin/env python3
"""
eval_embedding_consistency.py

Cross-seed embedding consistency using per-seed test splits.

For each session, identifies trials that appear in >= 2 seeds' test sets
(the only trials where we can compare predictions across seeds without
any seed having trained on that trial).

Produces two figures:
  1. Overlap heatmap  — for each session, how many trials are 2-way / 3-way / 4-way
  2. Consistency heatmap — mean pairwise Pearson r of predictions for overlapping
                           test trials, per (session, ensemble)
"""
import os, sys, itertools
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from models import MLP
from figure_style import (
    FIG, DPI, FONT, LINE, MARKER,
    MODEL_COLORS, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS       = [42, 43, 44, 45, 46]
R2_THRESHOLD = 0.1
MIN_TIMEPOINTS = 5   # minimum frames per trial to compute Pearson r

base        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
data_dir    = os.path.join(base, "outputs", "glm_input_data")
models_root = os.path.join(base, "models", "mlps", "ensembles")
splits_dir  = os.path.join(base, "splits")
output_dir  = os.path.join(base, "outputs", "mlps", "ensembles_multiseed")
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load raw data ─────────────────────────────────────────────────────────────
print("Loading data …")
beh_vals = np.load(os.path.join(data_dir, "behavior_glm_input.npy"),       allow_pickle=True)
beh_idx  = np.load(os.path.join(data_dir, "behavior_glm_input_index.npy"), allow_pickle=True)
beh_cols = np.load(os.path.join(data_dir, "behavior_glm_input_columns.npy"),allow_pickle=True)
spk_vals = np.load(os.path.join(data_dir, "fr_full.npy"))
spk_idx  = np.load(os.path.join(data_dir, "fr_full_index.npy"),            allow_pickle=True)
ensembles_values = np.load(os.path.join(data_dir, "ensembles.npy"))

try:
    beh_index = pd.MultiIndex.from_tuples(beh_idx)
except Exception:
    beh_index = pd.Index(beh_idx)
try:
    spk_index = pd.MultiIndex.from_tuples(spk_idx)
except Exception:
    spk_index = pd.Index(spk_idx)

behavior_glm_loaded = pd.DataFrame(beh_vals, index=beh_index, columns=beh_cols)
spikes_loaded       = pd.DataFrame(spk_vals, index=spk_index)

spikes_loaded.index = pd.Index(
    [( t[0], t[1] // 40000 - 1) for t in spikes_loaded.index]
)
spikes_unique_trials = set(idx[0] for idx in spikes_loaded.index)
behavior_glm_loaded  = behavior_glm_loaded[
    behavior_glm_loaded.index.map(lambda t: t[0] in spikes_unique_trials)
]
non_nan_rows        = behavior_glm_loaded.index[~behavior_glm_loaded.isna().any(axis=1)]
behavior_glm_loaded = behavior_glm_loaded.loc[non_nan_rows]
spikes_loaded       = spikes_loaded.loc[non_nan_rows]

session_ids = behavior_glm_loaded.index.map(lambda x: x[0]).unique()
n_sessions  = len(session_ids)
print(f"{n_sessions} sessions")

# ── Feature engineering (mirrors train_mlp.py) ────────────────────────────────
non_categorical_cols = [
    'frame_raw_500msMedian',
    'frame_raw_abs_acc_500msMedian',
    'frame_YawPitch_abs_vel_sum_500msMedian',
    'frame_YawPitch_abs_acc_sum_500msMedian',
    'head_angle_vel',
    'head_angle',
    'frame_position',
]
categorical_variables = ['cue_visible', 'upcoming_choice', 'reward_window', 'lick_detected']

zone_onehot_cols = []
for col in categorical_variables:
    col_vals = sorted(behavior_glm_loaded[col].dropna().unique().astype(int))
    for v in col_vals:
        behavior_glm_loaded[f'{col}_{v}'] = (behavior_glm_loaded[col] == v).astype(float)
    if len(col_vals) == 2:
        zone_onehot_cols.append(f'{col}_{col_vals[-1]}')
    else:
        zone_onehot_cols.extend([f'{col}_{v}' for v in col_vals])

input_size  = len(non_categorical_cols) + len(zone_onehot_cols)
n_ensembles = ensembles_values.shape[1]
hidden_size, num_hidden_layers, output_size = 64, 2, 1

# ── Build per-session datasets ────────────────────────────────────────────────
print("Building session datasets …")
session_dataset = {}
for session_id in session_ids:
    mask_s    = behavior_glm_loaded.index.map(lambda x: x[0]) == session_id
    beh       = behavior_glm_loaded[mask_s].copy()
    spk_sess  = spikes_loaded[mask_s].astype(float)

    for col in non_categorical_cols:
        if col in beh.columns:
            beh[col] = beh[col].astype(float)
            mu, sig  = beh[col].mean(), beh[col].std() + 1e-8
            beh[col] = (beh[col] - mu) / sig

    spk_vals_s = spk_sess.values @ ensembles_values
    spk_df = pd.DataFrame(spk_vals_s, index=spk_sess.index)
    for col in spk_df.columns:
        mu, sig = spk_df[col].mean(), spk_df[col].std() + 1e-8
        if sig > 0:
            spk_df[col] = (spk_df[col] - mu) / sig

    data_by_trial, labels_by_trial = {}, {}
    for trial_id in beh["trial_id"].unique():
        tm = beh["trial_id"] == trial_id
        data_by_trial[trial_id]   = beh.loc[tm, non_categorical_cols + zone_onehot_cols].values.astype(np.float32)
        labels_by_trial[trial_id] = spk_df.loc[tm].values.astype(np.float32)

    session_dataset[session_id] = {"data": data_by_trial, "labels": labels_by_trial}

# ── Load per-seed splits ──────────────────────────────────────────────────────
splits = {
    seed: np.load(os.path.join(splits_dir, f"split_seed{seed}.npy"), allow_pickle=True).item()
    for seed in SEEDS
}

# ── Compute trial overlap per session ─────────────────────────────────────────
# overlap_map[session][trial_id] = list of seeds that have it in test
overlap_map = {}
for sess in session_ids:
    trial_seeds = {}
    for seed in SEEDS:
        for trial in splits[seed].get(sess, []):
            trial_seeds.setdefault(trial, []).append(seed)
    overlap_map[sess] = trial_seeds

# Overlap count distribution: sessions x k-way (2, 3, 4, 5)
K_LEVELS = list(range(2, len(SEEDS) + 1))
overlap_counts = np.zeros((len(K_LEVELS), n_sessions), dtype=int)
for s_idx, sess in enumerate(session_ids):
    for trial, seed_list in overlap_map[sess].items():
        k = len(seed_list)
        if k >= 2:
            overlap_counts[K_LEVELS.index(k), s_idx] += 1

# ── Compute consistency matrix ────────────────────────────────────────────────
# consistency[s_idx, e_idx] = mean pairwise Pearson r across overlapping test trials
all_r2 = np.load(os.path.join(output_dir, "all_r2.npy"))  # (seeds, sessions, ensembles)
mean_r2 = np.nanmean(all_r2, axis=0)                       # (sessions, ensembles)

consistency_matrix = np.full((n_sessions, n_ensembles), np.nan)

for s_idx, sess in enumerate(session_ids):
    sdata = session_dataset[sess]

    # Only consider trials with >= 2 seeds in test
    multi_seed_trials = {
        trial: seed_list
        for trial, seed_list in overlap_map[sess].items()
        if len(seed_list) >= 2 and trial in sdata["data"]
    }

    if not multi_seed_trials:
        print(f"  S{s_idx+1} ({sess}): no overlapping test trials, skipping")
        continue

    # Cache per-seed predictions: {seed: {trial: preds_np}}
    seed_preds = {seed: {} for seed in SEEDS}

    for seed in SEEDS:
        mdir = os.path.join(models_root, f"seed{seed}")
        # Collect all trials this seed needs to predict
        trials_needed = [t for t, sl in multi_seed_trials.items() if seed in sl]
        if not trials_needed:
            continue

        for e_idx in range(n_ensembles):
            if mean_r2[s_idx, e_idx] < R2_THRESHOLD:
                continue
            mpath = os.path.join(mdir, f"session_{s_idx:02d}_neuron_{e_idx:02d}.pt")
            if not os.path.exists(mpath):
                continue

            model = MLP(input_size, hidden_size, num_hidden_layers, output_size).to(device)
            model.load_state_dict(torch.load(mpath, map_location=device, weights_only=True))
            model.eval()

            for trial in trials_needed:
                x = torch.tensor(sdata["data"][trial], dtype=torch.float32).to(device)
                if x.shape[0] < MIN_TIMEPOINTS:
                    continue
                with torch.no_grad():
                    _, pred = model(x)
                seed_preds[seed].setdefault(trial, {})[e_idx] = pred.squeeze().cpu().numpy()

            del model
        torch.cuda.empty_cache()

    # Compute pairwise Pearson r for each ensemble
    for e_idx in range(n_ensembles):
        if mean_r2[s_idx, e_idx] < R2_THRESHOLD:
            continue

        pairwise_rs = []
        for trial, seed_list in multi_seed_trials.items():
            # Gather predictions from all seeds that have this trial in test
            trial_preds = []
            for seed in seed_list:
                p = seed_preds[seed].get(trial, {}).get(e_idx)
                if p is not None and len(p) >= MIN_TIMEPOINTS:
                    trial_preds.append(p)

            if len(trial_preds) < 2:
                continue

            for pi, pj in itertools.combinations(trial_preds, 2):
                if len(pi) != len(pj):
                    n = min(len(pi), len(pj))
                    pi, pj = pi[:n], pj[:n]
                if np.std(pi) < 1e-8 or np.std(pj) < 1e-8:
                    continue
                r, _ = pearsonr(pi, pj)
                pairwise_rs.append(r)

        if pairwise_rs:
            consistency_matrix[s_idx, e_idx] = np.mean(pairwise_rs)

    n_valid = sum(1 for sl in multi_seed_trials.values() if len(sl) >= 2)
    print(f"  S{s_idx+1} ({sess}): {n_valid} overlapping trials")

# ── Figure 1: Overlap heatmap ─────────────────────────────────────────────────
sess_labels = [f"S{i+1}" for i in range(n_sessions)]
kway_labels = [f"{k}-way" for k in K_LEVELS]

fig1, ax1 = plt.subplots(figsize=(max(6, n_sessions * 0.38), 2.4))
apply_style(fig1, ax1)
cmap1 = plt.cm.YlOrRd.copy()
cmap1.set_bad("#eeeeee")
data_for_plot = overlap_counts.astype(float)
data_for_plot[data_for_plot == 0] = np.nan

sns.heatmap(
    data_for_plot,
    ax=ax1,
    cmap=cmap1,
    xticklabels=sess_labels,
    yticklabels=kway_labels,
    annot=True, fmt=".0f", annot_kws={"size": 7},
    linewidths=0.3, linecolor="#cccccc",
    cbar_kws={"label": "# trials"},
)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=8)
ax1.set_title("Test-trial overlap across seeds (# trials per session × k-way level)", fontsize=9, pad=6)
ax1.set_xlabel("Session", fontsize=8)
ax1.set_ylabel("Overlap level", fontsize=8)
savefig_manifest(fig1, "consistency_overlap_counts.png", [output_dir])
print(f"Saved {os.path.join(output_dir, 'consistency_overlap_counts.png')}")

# ── Figure 2: Consistency heatmap ─────────────────────────────────────────────
# Show top ensembles by mean consistency (same layout as old script)
ens_mean = np.nanmean(consistency_matrix, axis=0)    # (ensembles,)
valid_idx = np.where(np.isfinite(ens_mean))[0]
TOP_N     = min(14, len(valid_idx))
top_order = valid_idx[np.argsort(ens_mean[valid_idx])[::-1][:TOP_N]]

cons_show  = consistency_matrix[:, top_order].T       # (TOP_N, sessions)
ens_labels = [f"E{i+1:02d}" for i in top_order]

NO_DATA_COLOR = "#bbbbbb"
cmap2 = plt.cm.Blues.copy()
cmap2.set_bad(color=NO_DATA_COLOR)

fig2, axes = plt.subplots(1, 2, figsize=(7.5, 4.4),
                           gridspec_kw={"width_ratios": [3, 1]})
apply_style(fig2, axes)

sns.heatmap(cons_show, cmap=cmap2, vmin=0, vmax=1,
            xticklabels=sess_labels, yticklabels=ens_labels,
            ax=axes[0], cbar_kws={"label": "Mean pairwise Pearson r (test-only)"})
for (i, j) in zip(*np.where(np.isnan(cons_show))):
    axes[0].add_patch(plt.Rectangle([j, i], 1, 1, fill=True, facecolor=NO_DATA_COLOR,
                                     hatch="////", edgecolor="#888888", lw=0.5, zorder=2))
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right", fontsize=8)
axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=8)
axes[0].set_title(
    f"Cross-seed consistency — overlapping test trials only\n"
    f"(Top {TOP_N} ensembles; R²≥{R2_THRESHOLD}; grey=silent/insufficient overlap)",
    fontsize=8.5, pad=6,
)
axes[0].set_xlabel("Session", fontsize=8)
axes[0].set_ylabel("Ensemble (sorted by mean consistency)", fontsize=8)

bar_mean  = ens_mean[valid_idx]
bar_std   = np.nanstd(consistency_matrix[:, valid_idx], axis=0)
bar_order = np.argsort(bar_mean)
xn        = np.arange(len(valid_idx))
axes[1].barh(xn, bar_mean[bar_order], xerr=bar_std[bar_order],
             color="steelblue", alpha=0.7, capsize=2, height=0.7)
axes[1].set_yticks(xn[::2])
axes[1].set_yticklabels([f"E{valid_idx[bar_order[i]]+1:02d}"
                          for i in range(0, len(xn), 2)], fontsize=8)
axes[1].axvline(0.9, color="k", linestyle=":", lw=0.8, label="0.9")
axes[1].set_xlabel("Mean Pearson r", fontsize=8)
axes[1].set_title(f"All {len(valid_idx)}\nwith data", fontsize=8)
axes[1].legend(fontsize=7, frameon=False)
axes[1].spines[["top", "right"]].set_visible(False)

TOP_LABEL = 5
for pos in range(len(xn) - TOP_LABEL, len(xn)):
    axes[1].text(
        bar_mean[bar_order[pos]] + bar_std[bar_order[pos]] + 0.01,
        xn[pos], f"E{valid_idx[bar_order[pos]]+1:02d}",
        va="center", fontsize=6, fontweight="bold",
    )

savefig_manifest(fig2, "cross_seed_consistency.png", [output_dir])
print(f"Saved {os.path.join(output_dir, 'cross_seed_consistency.png')}")

np.save(os.path.join(output_dir, "consistency_matrix_mlp.npy"), consistency_matrix)
print(f"Saved consistency_matrix_mlp.npy")

# ── Summary ───────────────────────────────────────────────────────────────────
total_overlap = overlap_counts.sum(axis=1)
for k, cnt in zip(K_LEVELS, total_overlap):
    print(f"Total {k}-way trials across all sessions: {cnt}")
print(f"Valid (session, ensemble) cells: "
      f"{np.isfinite(consistency_matrix).sum()} / {consistency_matrix.size}")
