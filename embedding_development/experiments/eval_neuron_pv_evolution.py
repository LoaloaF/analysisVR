#!/usr/bin/env python3
"""
eval_neuron_pv_evolution.py

Compute Global Permutation Variance for the top-K single-unit (spikes) MLP models
and generate the evolution heatmap (for S21) and evolution lineplot (for S22).

Only Global PV is computed (no IG, no cond-PV).
Uses seed 42 only for speed; labels output as median over 1 seed.
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from models import MLP
from figure_style import (
    FIG, DPI, FONT, LINE, MARKER,
    MODEL_COLORS, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
SEEDS         = [42, 43, 44, 45, 46]
TOP_K         = 6      # top neurons by R² to include in evolution heatmap
N_PERM        = 5      # permutation repeats per group
R2_THRESHOLD  = 0.01
NO_DATA_COLOR = '#bbbbbb'

base  = os.path.dirname(os.path.abspath(__file__))
root  = os.path.join(base, "..")
out   = os.path.join(root, "outputs", "mlps", "spikes_multiseed")
os.makedirs(out, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

# ─── LOAD SESSION DATASET ──────────────────────────────────────────────────────
cache = os.path.join(root, "outputs", "session_dataset_spikes.pkl")
with open(cache, "rb") as f:
    ds = pickle.load(f)
session_ids = list(ds.keys())
num_sessions = len(session_ids)
print(f"{num_sessions} sessions from cache")

# Get number of neurons from cache
sample_key  = session_ids[0]
num_neurons = ds[sample_key]["labels"][list(ds[sample_key]["labels"].keys())[0]].shape[1]
print(f"{num_neurons} neurons")

# ─── FEATURE / SEMANTIC GROUP DEFINITIONS ─────────────────────────────────────
data_dir = os.path.join(root, "outputs", "glm_input_data")
beh_cols = np.load(os.path.join(data_dir, "behavior_glm_input_columns.npy"), allow_pickle=True)

non_cat = [
    'frame_raw_500msMedian',
    'frame_raw_abs_acc_500msMedian',
    'frame_YawPitch_abs_vel_sum_500msMedian',
    'frame_YawPitch_abs_acc_sum_500msMedian',
    'head_angle_vel',
    'head_angle',
    'frame_position',
]
cat_vars = ['cue_visible', 'upcoming_choice', 'reward_window', 'lick_detected']
beh_vals = np.load(os.path.join(data_dir, "behavior_glm_input.npy"), allow_pickle=True)
beh_idx  = np.load(os.path.join(data_dir, "behavior_glm_input_index.npy"), allow_pickle=True)
df_full  = pd.DataFrame(beh_vals, index=pd.Index(beh_idx), columns=beh_cols)

all_feat_cols = list(non_cat)
for col in cat_vars:
    col_vals = sorted(df_full[col].dropna().unique().astype(int))
    if len(col_vals) == 2:
        all_feat_cols.append(f"{col}_{col_vals[-1]}")
    else:
        all_feat_cols.extend([f"{col}_{v}" for v in col_vals])

feat_idx = {c: i for i, c in enumerate(all_feat_cols)}

semantic_groups = []
group_names     = []
for name in non_cat:
    if name in feat_idx:
        semantic_groups.append((name, [feat_idx[name]]))
        group_names.append(name)
for col in cat_vars:
    idxs = [feat_idx[f"{col}_{v}"]
            for v in sorted(df_full[col].dropna().unique().astype(int))
            if f"{col}_{v}" in feat_idx]
    if idxs:
        semantic_groups.append((col, idxs))
        group_names.append(col)

n_groups  = len(semantic_groups)
n_feats   = len(all_feat_cols)
print(f"{n_groups} semantic groups, {n_feats} features")

# ─── TOP-K NEURONS BY MEAN R² ─────────────────────────────────────────────────
r2_path = os.path.join(out, "all_r2.npy")
all_r2  = np.load(r2_path)                          # (seeds, sessions, neurons)
mean_r2 = np.nanmean(np.nanmedian(all_r2, axis=0), axis=0)  # (neurons,)
top_k_idx = np.argsort(mean_r2)[::-1][:TOP_K]
print(f"Top {TOP_K} neurons: {['U'+str(n+1) for n in top_k_idx]}")
print(f"Their mean R²: {[round(float(mean_r2[n]),4) for n in top_k_idx]}")

# ─── LOAD MLP HELPER ──────────────────────────────────────────────────────────
def _load_mlp(path):
    sd = torch.load(path, map_location=device)
    # checkpoint is raw state dict; infer hidden_size from first layer weight
    h  = sd["fc.0.weight"].shape[0]
    m  = MLP(n_feats, h, 2, 1).to(device)
    m.load_state_dict(sd)
    m.eval()
    return m

# ─── COMPUTE GLOBAL PV PER SEED → MEDIAN OVER SEEDS ──────────────────────────
# Shape: (seeds, sessions, TOP_K, n_groups)
all_gpv = np.full((len(SEEDS), num_sessions, TOP_K, n_groups), np.nan)
splits_dir = os.path.join(root, "splits")

for seed_idx, seed in enumerate(SEEDS):
    ckpt_path = os.path.join(out, f"neuron_gpv_seed{seed}.npy")
    if os.path.exists(ckpt_path):
        ckpt = np.load(ckpt_path)
        all_gpv[seed_idx] = ckpt[:, :TOP_K, :]   # slice if checkpoint has more neurons
        print(f"Seed {seed}: loaded checkpoint")
        continue

    tidx_map = np.load(os.path.join(splits_dir, f"split_seed{seed}.npy"),
                       allow_pickle=True).item()
    mdir = os.path.join(root, "models", "mlps", "spikes", f"seed{seed}")

    for s_idx, sess_id in enumerate(session_ids):
        test_trials = tidx_map.get(sess_id, [])
        valid_t     = [t for t in test_trials if t in ds[sess_id]["data"]]
        if not valid_t:
            continue
        Xnp = np.concatenate([ds[sess_id]["data"][t]   for t in valid_t], 0).astype(np.float32)
        Ynp = np.concatenate([ds[sess_id]["labels"][t] for t in valid_t], 0)

        for k, n_idx in enumerate(top_k_idx):
            mpath = os.path.join(mdir, f"session_{s_idx:02d}_neuron_{n_idx:02d}.pt")
            if not os.path.exists(mpath):
                continue
            model   = _load_mlp(mpath)
            Xt      = torch.tensor(Xnp, dtype=torch.float32, device=device)
            with torch.no_grad():
                base_pred = model(Xt)[1].squeeze().cpu().numpy()
            base_r2 = r2_score(Ynp[:, n_idx], base_pred)
            if base_r2 < R2_THRESHOLD:
                del model; torch.cuda.empty_cache(); continue

            for g_idx, (_, g_cols) in enumerate(semantic_groups):
                r2_perms = []
                for _ in range(N_PERM):
                    perm = Xnp.copy()
                    pidx = np.random.permutation(len(Xnp))
                    perm[:, g_cols] = Xnp[pidx][:, g_cols]
                    with torch.no_grad():
                        pp = model(torch.tensor(perm, dtype=torch.float32, device=device)
                                   )[1].squeeze().cpu().numpy()
                    r2_perms.append(r2_score(Ynp[:, n_idx], pp))
                all_gpv[seed_idx, s_idx, k, g_idx] = base_r2 - np.mean(r2_perms)

            del model; torch.cuda.empty_cache()
        print(f"  Seed {seed}  Session {s_idx+1}/{num_sessions} done", flush=True)

    np.save(ckpt_path, all_gpv[seed_idx])
    print(f"Seed {seed}: checkpoint saved")

# Median over seeds
gpv = np.nanmedian(all_gpv, axis=0)   # (sessions, TOP_K, n_groups)
np.save(os.path.join(out, "neuron_gpv_top12.npy"), gpv)
print("Saved neuron_gpv_top12.npy")

# ─── EVOLUTION HEATMAP (S21) ──────────────────────────────────────────────────
_nm_max = np.nanmax(np.where(
    np.isnan(np.nanmedian(all_r2, axis=0)),
    np.nan,
    np.nanmedian(all_r2, axis=0)), axis=0)  # (neurons,) — per-neuron max R² across sessions

_all_valid = gpv[~np.isnan(gpv)].ravel()
_floor_vmax = float(2.0 * np.nanmedian(gpv.reshape(-1, n_groups), axis=0).max()) * 3
_vmax = max(float(np.nanpercentile(_all_valid, 98)) if len(_all_valid) else 0.01,
            _floor_vmax, 0.01)

ncols = min(3, TOP_K)
nrows = (TOP_K + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(ncols * 7, nrows * max(3.5, n_groups * 0.35)),
                         squeeze=False)
apply_style(fig, axes.flatten())
_af   = axes.flatten()
cmap_ = plt.cm.Blues.copy()
cmap_.set_bad(NO_DATA_COLOR)

for k, n_idx in enumerate(top_k_idx):
    ax    = _af[k]
    data  = gpv[:, k, :].T.copy()   # (n_groups, sessions)
    sns.heatmap(data, ax=ax, cmap=cmap_, vmin=0, vmax=_vmax,
                xticklabels=[f"S{s+1}" for s in range(num_sessions)],
                yticklabels=group_names,
                cbar_kws={'label': f'(scale max={_vmax:.3f})'})
    for j in range(num_sessions):
        if np.all(np.isnan(data[:, j])):
            ax.add_patch(plt.Rectangle([j, 0], 1, n_groups, fill=True,
                                       facecolor=NO_DATA_COLOR,
                                       hatch='////', edgecolor='#888888', lw=0.5, zorder=2))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_title(f"U{n_idx+1:02d} — max R²={_nm_max[n_idx]:.3f}", fontsize=9)
    ax.set_xlabel('Session', fontsize=8)

for k in range(TOP_K, len(_af)):
    _af[k].set_visible(False)

plt.suptitle(f'Global PV Evolution — Top {TOP_K} Neurons\n'
             f'(shared color scale = 98th pctile; grey = R²<{R2_THRESHOLD})',
             fontsize=13, y=1.01)
savefig_manifest(fig, "evolution_heatmap_gpv.png", [out])
print(f"Saved {os.path.join(out, 'evolution_heatmap_gpv.png')}")

# ─── EVOLUTION LINEPLOT (S22) ─────────────────────────────────────────────────
# Pick diverse interesting neuron-feature pairs for lineplots
# (high temporal CV + mean above noise floor)
mean_gpv  = np.nanmean(gpv, axis=0)                    # (TOP_K, n_groups)
floor_g   = 2.0 * np.nanmedian(gpv.reshape(-1, n_groups), axis=0)
cv_gpv    = np.nanstd(gpv, axis=0) / (np.nanmean(gpv, axis=0) + 1e-9)  # (TOP_K, n_groups)
above     = (mean_gpv > floor_g[np.newaxis, :]) & (mean_gpv > 0)
cv_masked = np.where(above, cv_gpv, 0)

# Pick up to max_panels neuron-feature pairs
max_panels  = 9
max_per_neu = 2
max_per_grp = 3
cnt_n, cnt_g = {}, {}
pairs = []
for fi in np.argsort(cv_masked.ravel())[::-1]:
    if len(pairs) >= max_panels:
        break
    k, gi = fi // n_groups, fi % n_groups
    if cv_masked[k, gi] <= 0:
        break
    if cnt_n.get(k, 0) >= max_per_neu:
        continue
    if cnt_g.get(gi, 0) >= max_per_grp:
        continue
    pairs.append((k, gi))
    cnt_n[k] = cnt_n.get(k, 0) + 1
    cnt_g[gi] = cnt_g.get(gi, 0) + 1

# Load per-seed gpv for IQR shading
seed_stack = []
for seed in SEEDS:
    p = os.path.join(out, f"neuron_gpv_seed{seed}.npy")
    if os.path.exists(p):
        seed_stack.append(np.load(p))

sess_x      = np.arange(1, num_sessions + 1)
GROUP_COLORS = sns.color_palette("tab20", max(n_groups, 20))

from collections import OrderedDict
by_neu = OrderedDict()
for k, gi in pairs:
    by_neu.setdefault(k, []).append(gi)

neurons_list = list(by_neu.keys())
ncols_ = min(3, len(neurons_list))
nrows_ = (len(neurons_list) + ncols_ - 1) // ncols_
fig2, axes2 = plt.subplots(nrows_, ncols_,
                            figsize=(ncols_ * 5.5, nrows_ * 3.8),
                            sharey=True, squeeze=False)
apply_style(fig2, axes2.flatten())
distinct = sns.color_palette("tab10", 10)

for pi, k in enumerate(neurons_list):
    ax_ = axes2[pi // ncols_][pi % ncols_]
    for li, gi in enumerate(by_neu[k]):
        series = gpv[:, k, gi]
        col    = GROUP_COLORS[gi % len(GROUP_COLORS)]
        local_used = [GROUP_COLORS[g % len(GROUP_COLORS)] for g in by_neu[k][:li]]
        if col in local_used:
            col = distinct[li % len(distinct)]
        if seed_stack:
            ss = np.stack([sv[:, k, gi] for sv in seed_stack], axis=0)
            lo = np.nanpercentile(ss, 25, axis=0)
            hi = np.nanpercentile(ss, 75, axis=0)
            ok = ~(np.isnan(lo) | np.isnan(hi))
            if ok.any():
                ax_.fill_between(sess_x[ok], lo[ok], hi[ok], alpha=0.20, color=col)
        ax_.plot(sess_x, series, '-o', color=col, markersize=4, lw=1.8,
                 label=f"{group_names[gi]}  (μ={mean_gpv[k, gi]:.3f})")
    ax_.axhline(0, color='grey', linestyle='--', lw=0.7)
    ax_.set_title(f"U{top_k_idx[k]+1:02d}", fontsize=11, fontweight='bold')
    ax_.set_xticks(sess_x)
    ax_.set_xticklabels([f"S{s}" for s in sess_x], rotation=60, ha='right', fontsize=7)
    ax_.set_ylabel('Permutation importance (R² drop)', fontsize=9)
    ax_.legend(fontsize=8, loc='upper right', framealpha=0.7)
    ax_.spines[['top', 'right']].set_visible(False)

for pi in range(len(neurons_list), nrows_ * ncols_):
    axes2[pi // ncols_][pi % ncols_].set_visible(False)

fig2.suptitle('Feature Importance Evolution — Global PV (Neurons)\n'
              'Diverse interesting pairs (high temporal variability, noise-floored)\n'
              'Shaded band = IQR across seeds',
              fontsize=12, y=1.01)
savefig_manifest(fig2, "evolution_lineplot_gpv.png", [out])
print(f"Saved {os.path.join(out, 'evolution_lineplot_gpv.png')}")
print("All done.")
