#!/usr/bin/env python3
"""
eval_cebra_seeds.py

Evaluates CEBRA contrastive or CEBRA predictive encoders using a Ridge linear
probe on learned embeddings. Mirrors eval_mlp_attribution.py structure.

Methods:
  1. Multi-seed R²/MSE evaluation via Ridge probe
  2. Global semantic permutation importance (per-seed checkpointed)

Usage:
    python eval_cebra_seeds.py --arm cebra
    python eval_cebra_seeds.py --arm cebra_pred
"""
import os
import argparse
import pickle
import numpy as np
import numpy.ma as ma
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from load_encoder import build_windows, load_encoder

# ═══════════════════════════════════ CONFIG ═══════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--arm",   type=str, default="cebra",
                    choices=["cebra", "cebra_pred"],
                    help="Which model arm to evaluate")
parser.add_argument("--no_eval",   action="store_true", help="Skip R² eval (load from disk)")
parser.add_argument("--no_imp",    action="store_true", help="Skip importance (load from disk)")
args = parser.parse_args()

ARM              = args.arm
SEEDS            = [42, 43, 44, 45, 46]
N_PERM_REPEATS   = 5
R2_THRESHOLD     = 0.01
RIDGE_ALPHA      = 1.0

RUN_EVAL         = not args.no_eval
RUN_GLOBAL_PV    = not args.no_imp

mode_str    = "ensembles"
prefix_name = "E"
models_root = f"./models/{ARM}/ensembles"
splits_dir  = "./splits"
output_dir  = f"./outputs/{ARM}_eval/ensembles"
cache_path  = "./outputs/session_dataset_ensembles.pkl"
os.makedirs(output_dir, exist_ok=True)

unit_label    = "ensemble"
unit_label_pl = "ensembles"
arm_label     = "CEBRA-Contrastive" if ARM == "cebra" else "CEBRA-Predictive"

def savefig(name):
    path = os.path.join(output_dir, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")

print(f"ARM={ARM} ({arm_label})  seeds={SEEDS}")
print(f"RUN_EVAL={RUN_EVAL}  RUN_GLOBAL_PV={RUN_GLOBAL_PV}")

# ══════════════════════════════ LOAD RAW DATA ════════════════════════════════
base = "./outputs/glm_input_data/"

beh_vals = np.load(os.path.join(base, "behavior_glm_input.npy"),         allow_pickle=True)
beh_idx  = np.load(os.path.join(base, "behavior_glm_input_index.npy"),   allow_pickle=True)
beh_cols = np.load(os.path.join(base, "behavior_glm_input_columns.npy"), allow_pickle=True)
spk_vals = np.load(os.path.join(base, "fr_full.npy"))
spk_idx  = np.load(os.path.join(base, "fr_full_index.npy"),              allow_pickle=True)
spk_cols = np.load(os.path.join(base, "fr_full_columns.npy"),            allow_pickle=True)
ensembles_values = np.load(os.path.join(base, "ensembles.npy"))

behavior_glm_loaded = pd.DataFrame(beh_vals, index=pd.Index(beh_idx), columns=beh_cols)
spikes_loaded       = pd.DataFrame(spk_vals, index=pd.Index(spk_idx),  columns=spk_cols)
print(f"behavior {behavior_glm_loaded.shape}  spikes {spikes_loaded.shape}")

# ══════════════════════════════ PREPROCESSING ════════════════════════════════
spikes_loaded.index = pd.MultiIndex.from_tuples(
    spikes_loaded.index.map(lambda t: (t[0], t[1] // 40000 - 1))
)
spikes_unique_trials = set(idx[0] for idx in spikes_loaded.index)
behavior_glm_loaded  = behavior_glm_loaded[
    behavior_glm_loaded.index.map(lambda t: t[0] in spikes_unique_trials)
]
non_nan_rows        = behavior_glm_loaded.index[~behavior_glm_loaded.isna().any(axis=1)]
behavior_glm_loaded = behavior_glm_loaded.loc[non_nan_rows]
spikes_loaded       = spikes_loaded.loc[non_nan_rows]
session_ids         = behavior_glm_loaded.index.map(lambda x: x[0]).unique()
print(f"{len(session_ids)} sessions")

# ══════════════════════════════ FEATURE COLUMNS ══════════════════════════════
non_categorical_cols = [
    'frame_raw_500msMedian',
    'frame_raw_abs_acc_500msMedian',
    'frame_YawPitch_abs_vel_sum_500msMedian',
    'frame_YawPitch_abs_acc_sum_500msMedian',
    'head_angle_vel',
    'head_angle',
    'frame_position',
]
categorical_variables = [
    'cue_visible', 'upcoming_choice', 'reward_window', 'lick_detected',
]
zone_onehot_cols = []
for col in categorical_variables:
    col_vals = sorted(behavior_glm_loaded[col].dropna().unique().astype(int))
    for v in col_vals:
        behavior_glm_loaded[f'{col}_{v}'] = (behavior_glm_loaded[col] == v).astype(float)
    if len(col_vals) == 2:
        zone_onehot_cols.append(f'{col}_{col_vals[-1]}')
    else:
        zone_onehot_cols.extend([f'{col}_{v}' for v in col_vals])

all_feat_cols = non_categorical_cols + zone_onehot_cols
n_feats       = len(all_feat_cols)
print(f"{n_feats} input features")

# ══════════════════════════ SEMANTIC GROUPS ══════════════════════════════════
semantic_groups: list = []
for col in non_categorical_cols:
    semantic_groups.append((col, [all_feat_cols.index(col)]))
for cat_var in categorical_variables:
    col_indices = [i for i, c in enumerate(all_feat_cols) if c.startswith(f'{cat_var}_')]
    if col_indices:
        semantic_groups.append((cat_var, col_indices))

n_groups    = len(semantic_groups)
group_names = [g[0] for g in semantic_groups]
print(f"\nSemantic groups ({n_groups}):")
for name, cols in semantic_groups:
    print(f"  {name}: {[all_feat_cols[i] for i in cols]}")

with open(os.path.join(output_dir, "semantic_groups.pkl"), 'wb') as f:
    pickle.dump(semantic_groups, f)

# ══════════════════════════ SESSION DATASET ══════════════════════════════════
# Reuse the shared cache built by eval_mlp_attribution.py (same feature space).
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        session_dataset_singles = pickle.load(f)
    session_ids = pd.Index(list(session_dataset_singles.keys()))
    print(f"\nLoaded dataset cache: {cache_path}")
else:
    print(f"Cache not found at {cache_path}; building dataset…")
    session_dataset_singles = {}
    for session_id in session_ids:
        sm  = behavior_glm_loaded.index.map(lambda x: x[0]) == session_id
        beh = behavior_glm_loaded[sm].copy()
        spk = spikes_loaded[sm].copy()
        for col in non_categorical_cols:
            if col in beh.columns:
                beh.loc[:, col] = beh[col].astype(float)
                std_v = beh[col].std() + 1e-8
                beh.loc[:, col] = (beh[col] - beh[col].mean()) / std_v
        spk = spk.astype(float)
        sv    = spk.values
        new_s = pd.DataFrame(index=spk.index)
        for i in range(ensembles_values.shape[1]):
            new_s[f'ensemble_{i}'] = sv @ ensembles_values[:, i]
        spk = new_s
        label_stds = []
        for c in spk.columns:
            std_v = spk[c].std() + 1e-8
            label_stds.append(std_v)
            spk[c] = (spk[c] - spk[c].mean()) / std_v
        data_by_trial, labels_by_trial = {}, {}
        for tid in beh["trial_id"].unique():
            tm                   = beh["trial_id"] == tid
            data_by_trial[tid]   = beh.loc[tm, all_feat_cols].values.astype(np.float32)
            labels_by_trial[tid] = spk.loc[tm].values.astype(np.float32)
        session_dataset_singles[session_id] = {
            "data": data_by_trial, "labels": labels_by_trial, "label_stds": label_stds,
        }
    with open(cache_path, 'wb') as f:
        pickle.dump(session_dataset_singles, f)
    print(f"Dataset built and cached to {cache_path}")

# ══════════════════════════ MODEL CONFIG ════════════════════════════════════
num_sessions = len(session_ids)
num_neurons  = len(next(iter(session_dataset_singles.values()))['label_stds'])
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nsessions={num_sessions}, {unit_label_pl}={num_neurons}, device={device}")


def _get_arrays(session_id, trial_ids):
    """Concatenate data/labels for a list of trial IDs, skipping missing trials."""
    sd       = session_dataset_singles[session_id]
    valid    = [t for t in trial_ids if t in sd["data"]]
    X_chunks = [sd["data"][t]   for t in valid]
    Y_chunks = [sd["labels"][t] for t in valid]
    if not X_chunks:
        return np.zeros((0, n_feats), dtype=np.float32), np.zeros((0, num_neurons), dtype=np.float32)
    return (np.concatenate(X_chunks, axis=0).astype(np.float32),
            np.concatenate(Y_chunks, axis=0).astype(np.float32))


def _embed(encoder, X_np):
    """Sliding-window encode X_np → (T, embed_dim) numpy on CPU."""
    wins   = build_windows(X_np)                                            # (T, n_feats, 10)
    wins_t = torch.tensor(wins, dtype=torch.float32, device=device)
    with torch.no_grad():
        z = encoder(wins_t)
        if z.dim() == 3:
            z = z.squeeze(-1)
    return z.cpu().numpy()


# ══════════════════════════ EVALUATION (R²/MSE) ══════════════════════════════
all_r2_path  = os.path.join(output_dir, "all_r2.npy")
all_mse_path = os.path.join(output_dir, "all_mse.npy")

if RUN_EVAL:
    all_r2  = np.full((len(SEEDS), num_sessions, num_neurons), np.nan)
    all_mse = np.full((len(SEEDS), num_sessions, num_neurons), np.nan)

    for seed_idx, seed in enumerate(SEEDS):
        test_idx_map = np.load(os.path.join(splits_dir, f"split_seed{seed}.npy"),
                               allow_pickle=True).item()
        mdir = os.path.join(models_root, f"seed{seed}")

        for s_idx, session_id in enumerate(session_ids):
            test_trials  = [int(i) for i in test_idx_map[session_id]]
            all_trials   = list(session_dataset_singles[session_id]["data"].keys())
            train_trials = [t for t in all_trials if t not in test_trials]

            Xtr, Ytr = _get_arrays(session_id, train_trials)
            Xte, Yte = _get_arrays(session_id, test_trials)
            if len(Xtr) == 0 or len(Xte) == 0:
                continue

            for n_idx in range(num_neurons):
                mpath = os.path.join(mdir, f"session_{s_idx:02d}_neuron_{n_idx:02d}.pt")
                if not os.path.exists(mpath):
                    continue

                encoder, _, _ = load_encoder(mpath, device=device)
                encoder.eval()

                emb_tr = _embed(encoder, Xtr)
                emb_te = _embed(encoder, Xte)

                y_tr = Ytr[:, n_idx]
                y_te = Yte[:, n_idx]

                ridge = Ridge(alpha=RIDGE_ALPHA)
                ridge.fit(emb_tr, y_tr)
                y_pred = ridge.predict(emb_te)

                all_mse[seed_idx, s_idx, n_idx] = mean_squared_error(y_te, y_pred)
                if np.var(y_te) > 1e-8:
                    all_r2[seed_idx, s_idx, n_idx] = r2_score(y_te, y_pred)

                del encoder
                torch.cuda.empty_cache()

            print(f"  Seed {seed} S{s_idx+1:02d}/{num_sessions}  "
                  f"mean_R²={np.nanmean(all_r2[seed_idx, s_idx]):.3f}")

        # Checkpoint after each seed so work isn't lost
        np.save(all_r2_path,  all_r2)
        np.save(all_mse_path, all_mse)
        print(f"Seed {seed} eval complete. Checkpointed.")

    print("Evaluation complete.\n")

all_r2  = np.load(all_r2_path)
all_mse = np.load(all_mse_path)

# ═══════════════════════ MASKS & AGGREGATION ════════════════════════════════
mask_3d = np.isnan(all_r2)
mask    = np.any(mask_3d, axis=0)   # (sessions, neurons)

with np.errstate(all='ignore'):
    mean_r2  = np.nanmean(all_r2,  axis=0)
    mean_mse = np.nanmean(all_mse, axis=0)
    std_r2   = np.nanstd(all_r2,   axis=0)
    std_mse  = np.nanstd(all_mse,  axis=0)

low_r2_mask = mean_r2 < R2_THRESHOLD
print(f"Valid pairs: {(~mask).sum()}")
print(f"Pairs R²≥{R2_THRESHOLD}: {(~mask & ~low_r2_mask).sum()}")
print(f"Grand mean R²: {np.nanmean(mean_r2[~mask]):.3f} ± {np.nanstd(mean_r2[~mask]):.3f}")


def _apply_imp_mask(arr):
    return np.where((mask | low_r2_mask)[:, :, np.newaxis], np.nan, arr)


# ═══════════════════════════ R² HEATMAP ══════════════════════════════════════
NO_DATA_COLOR = '#bbbbbb'
cmap_r2 = plt.cm.Blues.copy()
cmap_r2.set_bad(color=NO_DATA_COLOR)

r2_c      = np.clip(mean_r2, 0.0, 1.0)
masked_r2 = ma.array(r2_c, mask=mask)
n_order   = np.argsort(masked_r2.mean(axis=0).filled(np.nan))[::-1]

r2_plot = r2_c[:, n_order].astype(float)
r2_plot[mask[:, n_order]] = np.nan

cell_h = 0.45
fig, ax = plt.subplots(figsize=(14, max(8, num_neurons * cell_h)))
sns.heatmap(r2_plot.T, cmap=cmap_r2, vmin=0, vmax=1, ax=ax,
            xticklabels=[f"S{j+1}" for j in range(num_sessions)],
            yticklabels=[f"{prefix_name}{n_order[i]+1:02d}" for i in range(num_neurons)])
ax.set_title(f'{arm_label} Ridge Probe — Mean R² ({len(SEEDS)} seeds)',
             fontsize=14, pad=10)
ax.set_xlabel('Session')
ax.set_ylabel(f'{unit_label_pl.capitalize()} (sorted by mean R²)')
plt.tight_layout()
savefig("r2_heatmap.png")

# ═══════════════════════════ MSE HEATMAP ═════════════════════════════════════
cmap_m = plt.cm.viridis.copy()
cmap_m.set_bad(color=NO_DATA_COLOR)

mse_c    = np.clip(mean_mse, 0, 2.0)
mse_plot = mse_c[:, n_order].astype(float)
mse_plot[mask[:, n_order]] = np.nan

fig, ax = plt.subplots(figsize=(14, max(8, num_neurons * cell_h)))
sns.heatmap(mse_plot.T, cmap=cmap_m, ax=ax,
            xticklabels=[f"S{j+1}" for j in range(num_sessions)],
            yticklabels=[f"{prefix_name}{n_order[i]+1:02d}" for i in range(num_neurons)])
ax.set_title(f'{arm_label} Ridge Probe — Mean MSE ({len(SEEDS)} seeds)', fontsize=14, pad=10)
ax.set_xlabel('Session')
ax.set_ylabel(f'{unit_label_pl.capitalize()} (sorted by mean R²)')
plt.tight_layout()
savefig("mse_heatmap.png")

# ════════════════════════════ R² BAR CHART ════════════════════════════════════
neuron_mean_r2 = masked_r2.mean(axis=0).filled(np.nan)
neuron_std_r2  = masked_r2.std(axis=0).filled(np.nan)
order_bar      = np.argsort(neuron_mean_r2)
x              = np.arange(num_neurons)
pal            = sns.color_palette("mako_r", as_cmap=True)(np.linspace(0.15, 0.85, num_neurons))

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(x, neuron_mean_r2[order_bar], width=0.7, color=pal, zorder=3, linewidth=0)
ax.errorbar(x, neuron_mean_r2[order_bar], yerr=neuron_std_r2[order_bar],
            fmt='none', ecolor='gray', elinewidth=0.8, capsize=3, alpha=0.7)
ax.axhline(0, color='firebrick', linestyle='--', linewidth=1)
ax.set_xticks(x[::3])
ax.set_xticklabels([f"{prefix_name}{order_bar[i]+1:02d}" for i in range(0, num_neurons, 3)],
                   rotation=45, ha='right', fontsize=8)
ax.set_ylabel("Mean R²")
ax.set_xlabel(f"{unit_label.capitalize()} (sorted by mean R²)")
ax.set_title(f"Per-{unit_label.capitalize()} Ridge Probe R² — {arm_label}\n"
             f"(mean ± SD across {len(SEEDS)} seeds × {num_sessions} sessions)",
             fontsize=13, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
valid_counts = (~mask).sum(axis=0)
ax.annotate(
    f"n {unit_label_pl} = {num_neurons}  |  "
    f"median sessions per {unit_label} = {int(np.median(valid_counts))}  |  "
    f"grand mean R² = {np.nanmean(neuron_mean_r2):.3f}",
    xy=(0.01, 0.97), xycoords='axes fraction',
    va='top', ha='left', fontsize=9, color='dimgray',
)
plt.tight_layout()
savefig("r2_bar.png")

# ════════════════════════════ MSE BAR CHART ═══════════════════════════════════
masked_mse      = ma.array(mean_mse, mask=mask)
neuron_mean_mse = masked_mse.mean(axis=0).filled(np.nan)
neuron_std_mse  = masked_mse.std(axis=0).filled(np.nan)
order_mse       = np.argsort(neuron_mean_mse)

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(x, neuron_mean_mse[order_mse], width=0.7, color=pal, zorder=3, linewidth=0)
ax.errorbar(x, neuron_mean_mse[order_mse], yerr=neuron_std_mse[order_mse],
            fmt='none', ecolor='gray', elinewidth=0.8, capsize=3, alpha=0.7)
ax.set_xticks(x[::3])
ax.set_xticklabels([f"{prefix_name}{order_mse[i]+1:02d}" for i in range(0, num_neurons, 3)],
                   rotation=45, ha='right', fontsize=8)
ax.set_ylabel("Mean MSE  (z-scored ensemble activity)")
ax.set_xlabel(f"{unit_label.capitalize()} (sorted by mean MSE)")
ax.set_title(f"Per-{unit_label.capitalize()} Ridge Probe MSE — {arm_label}\n"
             f"(mean ± SD across {len(SEEDS)} seeds × {num_sessions} sessions)",
             fontsize=13, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
ax.annotate(
    f"grand mean MSE = {np.nanmean(neuron_mean_mse):.3f}",
    xy=(0.01, 0.97), xycoords='axes fraction',
    va='top', ha='left', fontsize=9, color='dimgray',
)
plt.tight_layout()
savefig("mse_bar.png")

# ════════════════════ VARIANCE vs R² / MSE SCATTERS ═════════════════════════
# Use seed-42 split for test-set variance (fast, representative)
test_idx_map_42 = np.load(os.path.join(splits_dir, "split_seed42.npy"),
                          allow_pickle=True).item()

variance_vals, r2_flat, mse_flat = [], [], []
for s_idx, session_id in enumerate(session_ids):
    test_trials = [int(i) for i in test_idx_map_42[session_id]]
    _, Yte      = _get_arrays(session_id, test_trials)
    if len(Yte) == 0:
        continue
    for n_idx in range(num_neurons):
        if mask[s_idx, n_idx]:
            continue
        variance_vals.append(float(np.var(Yte[:, n_idx])))
        r2_flat.append(float(np.clip(mean_r2[s_idx, n_idx], 0, 1)))
        mse_flat.append(float(mean_mse[s_idx, n_idx]))

variance_vals = np.array(variance_vals)
r2_flat       = np.array(r2_flat)
mse_flat      = np.array(mse_flat)

for metric, vals, ylabel, fname in [
    ("R²",  r2_flat,  "Ridge Probe R² (mean over seeds)",  "variance_vs_r2.png"),
    ("MSE", mse_flat, "Ridge Probe MSE (mean over seeds)", "variance_vs_mse.png"),
]:
    if len(variance_vals) < 2:
        continue
    r_corr = np.corrcoef(variance_vals, vals)[0, 1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(variance_vals, vals, alpha=0.15, s=8, color='steelblue')
    m, b = np.polyfit(variance_vals, vals, 1)
    xl   = np.linspace(variance_vals.min(), variance_vals.max(), 200)
    ax.plot(xl, m * xl + b, color='firebrick', linewidth=1.5, label=f'r = {r_corr:.2f}')
    ax.set_xlabel('Test-set Variance (z-scored ensemble activity)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Ensemble Variance vs {metric} — {arm_label}')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    savefig(fname)

# ═══════════════════════ TRIAL PREDICTION TRACES ═════════════════════════════
# For each session: show the 3 ensembles with highest mean R² — one random test trial
import random
random.seed(42)

for s_idx, session_id in enumerate(session_ids):
    test_trials = [int(i) for i in test_idx_map_42[session_id]]
    valid       = [t for t in test_trials
                   if t in session_dataset_singles[session_id]["data"]]
    if not valid:
        continue

    r2_sess   = np.clip(mean_r2[s_idx, :], 0, 1)
    valid_ens = np.where(~mask[s_idx, :])[0]
    if len(valid_ens) == 0:
        continue
    top_ens   = sorted(valid_ens, key=lambda n: r2_sess[n], reverse=True)[:3]

    trial_id     = random.choice(valid)
    trial_data   = session_dataset_singles[session_id]["data"][trial_id]    # (T, n_feats)
    trial_labels = session_dataset_singles[session_id]["labels"][trial_id]  # (T, num_neurons)

    # We need one encoder per ensemble to get predictions — use seed-42 models
    mdir42 = os.path.join(models_root, "seed42")

    # Also need train embeddings to fit probe
    train_trials = [t for t in session_dataset_singles[session_id]["data"]
                    if t not in test_trials]
    Xtr, Ytr = _get_arrays(session_id, train_trials)
    if len(Xtr) == 0:
        continue

    fig, axes = plt.subplots(len(top_ens), 1, figsize=(14, 3 * len(top_ens)))
    if len(top_ens) == 1:
        axes = [axes]

    for plot_idx, n_idx in enumerate(top_ens):
        mpath = os.path.join(mdir42, f"session_{s_idx:02d}_neuron_{n_idx:02d}.pt")
        if not os.path.exists(mpath):
            continue
        encoder, _, _ = load_encoder(mpath, device=device)
        encoder.eval()

        emb_tr   = _embed(encoder, Xtr)
        emb_trial = _embed(encoder, trial_data)

        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(emb_tr, Ytr[:, n_idx])
        pred = ridge.predict(emb_trial)

        ax = axes[plot_idx]
        t  = np.arange(len(trial_labels))
        ax.plot(t, trial_labels[:, n_idx], 'o-', label='Actual',    alpha=0.7, markersize=3)
        ax.plot(t, pred,                   's-', label='Predicted', alpha=0.7, markersize=3)
        ax.set_ylabel(f'{prefix_name}{n_idx+1:02d} (z-scored)')
        ax.set_title(f'Session {session_id} — {prefix_name}{n_idx+1:02d} — R²={r2_sess[n_idx]:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if plot_idx == len(top_ens) - 1:
            ax.set_xlabel('Time steps')

        del encoder
        torch.cuda.empty_cache()

    plt.suptitle(f'{arm_label} — Session {session_id} — Trial {trial_id}',
                 fontweight='bold')
    plt.tight_layout()
    savefig(f"trial_traces_session_{s_idx:02d}.png")

print("Trial trace plots saved.\n")

# ══════════════════════════════════════════════════════════════════════════════
#            GLOBAL SEMANTIC PERMUTATION IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
_gpv_sem_path = os.path.join(output_dir, "importance_global_pv_semantic.npy")

if RUN_GLOBAL_PV:
    all_gpv = np.full((len(SEEDS), num_sessions, num_neurons, n_groups), np.nan)

    for seed_idx, seed in enumerate(SEEDS):
        ckpt_path = os.path.join(output_dir, f"importance_global_pv_semantic_seed{seed}.npy")
        if os.path.exists(ckpt_path):
            all_gpv[seed_idx] = np.load(ckpt_path)
            print(f"Seed {seed} importance: loaded checkpoint.")
            continue

        test_idx_map = np.load(os.path.join(splits_dir, f"split_seed{seed}.npy"),
                               allow_pickle=True).item()
        mdir = os.path.join(models_root, f"seed{seed}")

        for s_idx, session_id in enumerate(session_ids):
            test_trials  = [int(i) for i in test_idx_map[session_id]]
            all_trials   = list(session_dataset_singles[session_id]["data"].keys())
            train_trials = [t for t in all_trials if t not in test_trials]

            Xtr, Ytr = _get_arrays(session_id, train_trials)
            Xte, Yte = _get_arrays(session_id, test_trials)
            if len(Xtr) == 0 or len(Xte) == 0:
                continue

            for n_idx in range(num_neurons):
                base_r2 = all_r2[seed_idx, s_idx, n_idx]
                if np.isnan(base_r2) or base_r2 < R2_THRESHOLD:
                    continue

                mpath = os.path.join(mdir, f"session_{s_idx:02d}_neuron_{n_idx:02d}.pt")
                if not os.path.exists(mpath):
                    continue

                encoder, _, _ = load_encoder(mpath, device=device)
                encoder.eval()

                # Fit Ridge on training embeddings
                emb_tr = _embed(encoder, Xtr)
                emb_te = _embed(encoder, Xte)
                y_tr   = Ytr[:, n_idx]
                y_te   = Yte[:, n_idx]

                ridge = Ridge(alpha=RIDGE_ALPHA)
                ridge.fit(emb_tr, y_tr)

                # Permute each semantic group on the test features, re-embed, re-predict
                for g_idx, (g_name, g_cols) in enumerate(semantic_groups):
                    r2_perms = []
                    for _ in range(N_PERM_REPEATS):
                        perm_Xte = Xte.copy()
                        pidx     = np.random.permutation(len(Xte))
                        perm_Xte[:, g_cols] = Xte[pidx][:, g_cols]
                        emb_perm = _embed(encoder, perm_Xte)
                        y_pred_p = ridge.predict(emb_perm)
                        if np.var(y_te) > 1e-8:
                            r2_perms.append(r2_score(y_te, y_pred_p))
                    if r2_perms:
                        all_gpv[seed_idx, s_idx, n_idx, g_idx] = base_r2 - np.mean(r2_perms)

                del encoder
                torch.cuda.empty_cache()

            print(f"  Seed {seed} S{s_idx+1:02d}/{num_sessions} importance done")

        np.save(ckpt_path, all_gpv[seed_idx])
        print(f"Seed {seed} importance checkpointed.")

    with np.errstate(all='ignore'):
        global_pv_sem = np.nanmedian(all_gpv, axis=0)   # (sessions, neurons, groups)
    np.save(_gpv_sem_path, global_pv_sem)
    print("Saved importance_global_pv_semantic.npy")
else:
    global_pv_sem = np.load(_gpv_sem_path)
    print(f"Loaded {_gpv_sem_path}")

global_pv_sem_m = _apply_imp_mask(global_pv_sem)

# ════════════════════ IMPORTANCE BAR CHART ════════════════════════════════════
feat_mean = np.nanmean(global_pv_sem_m.reshape(-1, n_groups), axis=0)
feat_std  = np.nanstd( global_pv_sem_m.reshape(-1, n_groups), axis=0)
order_imp = np.argsort(feat_mean)[::-1]

fig, ax = plt.subplots(figsize=(10, 5))
x_imp = np.arange(n_groups)
ax.bar(x_imp, feat_mean[order_imp], yerr=feat_std[order_imp],
       color='steelblue', capsize=4, alpha=0.8)
ax.set_xticks(x_imp)
ax.set_xticklabels([group_names[i] for i in order_imp], rotation=45, ha='right')
ax.set_ylabel('Mean R² drop')
ax.set_title(f'Semantic Permutation Importance — {arm_label}\n'
             f'(median over {len(SEEDS)} seeds, valid session-{unit_label} pairs)')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
savefig("importance_bar.png")

# ════════════════════ IMPORTANCE HEATMAP (ensemble × group) ══════════════════
ens_importance = np.nanmean(global_pv_sem_m, axis=0)   # (n_neurons, n_groups)
valid_ens      = ~np.all(np.isnan(ens_importance), axis=1)
hm_data        = ens_importance[valid_ens, :].T         # (n_groups, n_valid_ens)
ens_labels     = [f"{prefix_name}{n+1:02d}" for n in np.where(valid_ens)[0]]
order_ens      = np.argsort(np.nanmean(hm_data, axis=0))[::-1]
hm_data        = hm_data[:, order_ens]
ens_labels     = [ens_labels[i] for i in order_ens]

fig, ax = plt.subplots(figsize=(max(8, len(ens_labels) * 0.55), 5))
sns.heatmap(hm_data, xticklabels=ens_labels, yticklabels=group_names,
            cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Mean R² drop'})
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.title(f'Permutation Importance by {unit_label.capitalize()} — {arm_label}\n'
          f'(mean over sessions with mean R² ≥ {R2_THRESHOLD})',
          fontsize=12)
plt.tight_layout()
savefig("importance_heatmap_by_ensemble.png")

# ════════════════════ IMPORTANCE HEATMAP (session-ensemble × group) ══════════
pair_labels, pair_rows = [], []
for s in range(num_sessions):
    for n in range(num_neurons):
        row = global_pv_sem_m[s, n, :]
        if not np.all(np.isnan(row)):
            pair_labels.append(f"S{s+1:02d}·{prefix_name}{n+1:02d}")
            pair_rows.append(row)

if pair_rows:
    hm2      = np.array(pair_rows).T                   # (n_groups, n_pairs)
    ord2     = np.argsort(np.nanmean(hm2, axis=0))[::-1]
    hm2      = hm2[:, ord2]
    plabels2 = [pair_labels[i] for i in ord2]

    fig, ax = plt.subplots(figsize=(max(12, len(plabels2) * 0.3), 5))
    sns.heatmap(hm2, xticklabels=plabels2, yticklabels=group_names,
                cmap='YlOrRd', ax=ax, cbar_kws={'label': 'R² drop'})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5)
    plt.title(f'Permutation Importance — {arm_label}\n'
              f'{len(plabels2)} pairs with mean R² ≥ {R2_THRESHOLD}')
    plt.tight_layout()
    savefig("importance_heatmap_pairs.png")

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY PRINT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"SUMMARY — {arm_label}")
print("="*60)
print(f"  Sessions:         {num_sessions}")
print(f"  Ensembles:        {num_neurons}")
print(f"  Seeds:            {len(SEEDS)}")
print(f"  Valid pairs:      {(~mask).sum()}")
print(f"  R²≥{R2_THRESHOLD} pairs: {(~mask & ~low_r2_mask).sum()}")
print(f"  Grand mean R²:    {np.nanmean(mean_r2[~mask]):.4f}")
print(f"  Grand mean MSE:   {np.nanmean(mean_mse[~mask]):.4f}")
if pair_rows:
    top_group = group_names[order_imp[0]]
    print(f"  Top feature:      {top_group} (mean R² drop = {feat_mean[order_imp[0]]:.4f})")
print(f"\nAll outputs in: {output_dir}")
print("="*60)
