#!/usr/bin/env python3
"""
eval_mlp_attribution.py

MLP evaluation with attribution methods operating at the semantic-variable level.

Methods:
  1. Global PV (semantic)   – permute all columns of each semantic group jointly
  2. Cond-PV (semantic)     – k-NN on features excluding the entire group
  3. Integrated Gradients   – |IG| summed to semantic groups; signed IG saved
  4. Supergroup PV          – Ward clusters over semantic-variable distance matrix
  5. Per-categorical-level  – signed IG and R² broken down by level

Old per-column .npy files on disk are preserved; new outputs use _semantic suffix.
"""
import os
import pickle
import numpy as np
import numpy.ma as ma
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram as _dg
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from models import MLP

# ═══════════════════════════════════ CONFIG ═══════════════════════════════════
USE_ENSEMBLES    = True
SEEDS            = [42, 43, 44, 45, 46]
N_PERM_REPEATS   = 5
R2_THRESHOLD     = 0.01

RUN_EVAL         = False   # recompute R² metrics
RUN_GLOBAL_PV    = True    # method 1 — semantic groups
RUN_COND_PV      = True    # method 2 — semantic groups
RUN_IG           = True    # method 3 — semantic + signed
RUN_GROUPED_PV   = True    # method 4 — supergroups of semantic vars
RUN_PER_LEVEL    = True    # method 5 — per-categorical-level decomposition
RUN_DRY_RUN      = True    # verification checks before full run
RUN_EVOLUTION    = True    # section A — temporal encoding evolution across sessions
RUN_DISCOVERY    = True    # section B — automated discovery of interesting pairs

K_NEIGHBORS      = 10
IG_STEPS         = 50
CLUSTER_DIST_THR = 0.5
MIN_LEVEL_TRIALS = 20      # flag levels with fewer active trials

SAVE_PLOTS       = True
SHOW_PLOTS       = False

mode_str    = "ensembles" if USE_ENSEMBLES else "spikes"
prefix_name = "E"         if USE_ENSEMBLES else "U"
models_root = f"./models/mlps/{mode_str}"
splits_dir  = "./splits"
output_dir  = f"./outputs/mlps/{mode_str}_multiseed"
cache_path  = f"./outputs/session_dataset_{mode_str}.pkl"
os.makedirs(output_dir, exist_ok=True)

unit_label    = "ensemble" if USE_ENSEMBLES else "neuron"
unit_label_pl = unit_label + "s"


def savefig(name):
    if SAVE_PLOTS:
        plt.savefig(os.path.join(output_dir, name), dpi=150, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close()


print(f"mode={mode_str}  seeds={SEEDS}")
print(f"RUN_EVAL={RUN_EVAL}  RUN_GLOBAL_PV={RUN_GLOBAL_PV}  "
      f"RUN_COND_PV={RUN_COND_PV}  RUN_IG={RUN_IG}  "
      f"RUN_GROUPED_PV={RUN_GROUPED_PV}  RUN_PER_LEVEL={RUN_PER_LEVEL}")
print(f"RUN_EVOLUTION={RUN_EVOLUTION}  RUN_DISCOVERY={RUN_DISCOVERY}")

# ══════════════════════════════ LOAD RAW DATA ═════════════════════════════════
base     = "./outputs/glm_input_data/"
beh_vals = np.load(os.path.join(base, "behavior_glm_input.npy"),         allow_pickle=True)
beh_idx  = np.load(os.path.join(base, "behavior_glm_input_index.npy"),   allow_pickle=True)
beh_cols = np.load(os.path.join(base, "behavior_glm_input_columns.npy"), allow_pickle=True)
spk_vals = np.load(os.path.join(base, "fr_full.npy"))
spk_idx  = np.load(os.path.join(base, "fr_full_index.npy"),              allow_pickle=True)
ensembles_values = np.load(os.path.join(base, "ensembles.npy"))

beh_index = pd.Index(beh_idx)
spk_index = pd.Index(spk_idx)

behavior_glm_loaded = pd.DataFrame(beh_vals, index=beh_index, columns=beh_cols)
spk_cols = np.load(os.path.join(base, "fr_full_columns.npy"), allow_pickle=True)
spikes_loaded = pd.DataFrame(spk_vals, index=spk_index, columns=spk_cols)
print(f"behavior_glm_loaded {behavior_glm_loaded.shape}  spikes_loaded {spikes_loaded.shape}")

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
behavior_glm_loaded = behavior_glm_loaded.drop("track_zone", axis=1)
session_ids         = behavior_glm_loaded.index.map(lambda x: x[0]).unique()
print(f"{len(session_ids)} sessions")

# ══════════════════════════════ FEATURE COLUMNS ═══════════════════════════════
non_categorical_cols = [
    'frame_raw_500msMedian',
    'frame_raw_abs_acc_500msMedian',
    'frame_YawPitch_abs_vel_sum_500msMedian',
    'frame_YawPitch_abs_acc_sum_500msMedian',
    'head_angle_vel',
    'head_angle',
    'movement_energy_smooth5',
]
categorical_variables = [
    'track_zone_int', 'cue_visible', 'upcoming_choice', 'reward_window', 'lick_detected',
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

# ══════════════════════ SEMANTIC GROUPS (Section 1) ═══════════════════════════
# Each entry: (group_name, [column_indices_in_all_feat_cols])
# Continuous variables → singleton groups; categorical → one group per variable.
semantic_groups: list = []

for col in non_categorical_cols:
    semantic_groups.append((col, [all_feat_cols.index(col)]))

for cat_var in categorical_variables:
    col_indices = [i for i, c in enumerate(all_feat_cols) if c.startswith(f'{cat_var}_')]
    if col_indices:
        semantic_groups.append((cat_var, col_indices))

n_groups    = len(semantic_groups)
group_names = [g[0] for g in semantic_groups]

print(f"\nSemantic groups ({n_groups} total):")
for name, cols in semantic_groups:
    col_names = [all_feat_cols[i] for i in cols]
    print(f"  {name}: {col_names}")

with open(os.path.join(output_dir, "semantic_groups.pkl"), 'wb') as _f:
    pickle.dump(semantic_groups, _f)
print("Saved semantic_groups.pkl\n")

# Multi-level categorical info for per-level decomposition.
# Skip singletons (2-level categoricals where only 1 one-hot was kept).
multilevel_cat_info = []  # list of (var_name, level_value, col_idx_in_all_feat_cols)
for name, col_indices in semantic_groups:
    if name in categorical_variables and len(col_indices) >= 2:
        for col_idx in col_indices:
            col_name  = all_feat_cols[col_idx]
            # e.g. "upcoming_choice_-1" → split on '_' → last token = '-1'
            level_val = int(col_name.split('_')[-1])
            multilevel_cat_info.append((name, level_val, col_idx))

n_level_cols    = len(multilevel_cat_info)
cat_level_index = [(v, l) for v, l, _ in multilevel_cat_info]

with open(os.path.join(output_dir, "categorical_level_index.pkl"), 'wb') as _f:
    pickle.dump(cat_level_index, _f)

print(f"Multi-level categorical columns ({n_level_cols}):")
for v, l, ci in multilevel_cat_info:
    print(f"  {all_feat_cols[ci]}  ({v}, level {l})")
print()

# ══════════════════════════ SESSION DATASET ═══════════════════════════════════
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        session_dataset_singles = pickle.load(f)
    session_ids = pd.Index(list(session_dataset_singles.keys()))
    print(f"Loaded dataset from cache: {cache_path}")
else:
    session_dataset_singles = {}
    for session_id in session_ids:
        sm   = behavior_glm_loaded.index.map(lambda x: x[0]) == session_id
        beh  = behavior_glm_loaded[sm]
        spk  = spikes_loaded[sm]
        for col in non_categorical_cols:
            if col in beh.columns:
                beh.loc[:, col] = beh[col].astype(float)
                std_v = beh[col].std() + 1e-8
                beh.loc[:, col] = (beh[col] - beh[col].mean()) / std_v
        spk = spk.astype(float)
        if USE_ENSEMBLES:
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
            data_by_trial[tid]   = beh.loc[tm, all_feat_cols].values.astype(np.float16)
            labels_by_trial[tid] = spk.loc[tm].values.astype(np.float16)
        session_dataset_singles[session_id] = {
            "data": data_by_trial, "labels": labels_by_trial, "label_stds": label_stds,
        }
    with open(cache_path, 'wb') as f:
        pickle.dump(session_dataset_singles, f)
    print(f"Dataset built and cached to {cache_path}")

# ══════════════════════════ MODEL CONFIG ══════════════════════════════════════
input_size        = n_feats
hidden_size       = 64
num_hidden_layers = 2
output_size       = 1
num_sessions      = len(session_ids)
num_neurons       = len(next(iter(session_dataset_singles.values()))['label_stds'])
device            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"input_size={input_size}, sessions={num_sessions}, "
      f"{unit_label_pl}={num_neurons}, device={device}")


def _load_model(mpath):
    m = MLP(input_size, hidden_size, num_hidden_layers, output_size).to(device)
    m.load_state_dict(torch.load(mpath, map_location=device))
    m.eval()
    return m


def _test_arrays(session_id, test_indices):
    sd  = session_dataset_singles[session_id]
    Xnp = np.concatenate([sd["data"][i]   for i in test_indices], axis=0).astype(np.float32)
    Ynp = np.concatenate([sd["labels"][i] for i in test_indices], axis=0)
    return Xnp, Ynp


# ══════════════════════════ EVALUATION (R²) ══════════════════════════════════
if RUN_EVAL:
    all_mse = np.full((len(SEEDS), num_sessions, num_neurons), np.nan)
    all_r2  = np.full((len(SEEDS), num_sessions, num_neurons), np.nan)
    for seed_idx, seed in enumerate(SEEDS):
        test_idx_map = np.load(os.path.join(splits_dir, f"split_seed{seed}.npy"),
                               allow_pickle=True).item()
        mdir = os.path.join(models_root, f"seed{seed}")
        for s_idx, sess in enumerate(session_ids):
            Xnp, Ynp = _test_arrays(sess, test_idx_map[sess])
            Xt = torch.tensor(Xnp, dtype=torch.float32).to(device)
            for n_idx in range(num_neurons):
                mpath = os.path.join(mdir, f"session_{s_idx:02d}_neuron_{n_idx:02d}.pt")
                if not os.path.exists(mpath):
                    continue
                model = _load_model(mpath)
                with torch.no_grad():
                    _, preds = model(Xt)
                pnp    = preds.squeeze().cpu().numpy()
                actual = Ynp[:, n_idx]
                all_mse[seed_idx, s_idx, n_idx] = mean_squared_error(actual, pnp)
                if np.var(actual) > 1e-8:
                    all_r2[seed_idx, s_idx, n_idx] = r2_score(actual, pnp)
                del model
            del Xt
            torch.cuda.empty_cache()
        print(f"Seed {seed} eval done.")
    np.save(os.path.join(output_dir, "all_mse.npy"), all_mse)
    np.save(os.path.join(output_dir, "all_r2.npy"),  all_r2)
    print("Saved all_mse.npy and all_r2.npy")

all_mse = np.load(os.path.join(output_dir, "all_mse.npy"))
all_r2  = np.load(os.path.join(output_dir, "all_r2.npy"))

# ═══════════════════════ MASKS & AGGREGATION ═════════════════════════════════
mask_3d = np.isnan(all_r2)
mask    = np.any(mask_3d, axis=0)   # (sessions, neurons)

all_r2_m  = np.where(mask_3d, np.nan, all_r2.astype(float))
all_mse_m = np.where(mask_3d, np.nan, all_mse.astype(float))

with np.errstate(all='ignore'):
    mean_r2  = np.nanmean(all_r2_m,  axis=0)
    std_r2   = np.nanstd(all_r2_m,   axis=0)
    mean_mse = np.nanmean(all_mse_m, axis=0)

print(f"Valid pairs: {(~mask).sum()}")
print(f"Mean R² (seed-avg): {np.nanmean(mean_r2[~mask]):.3f} ± {np.nanstd(mean_r2[~mask]):.3f}")

# Per-session mask: exclude (session, neuron) pairs where mean R² < threshold
low_r2_session_mask = mean_r2 < R2_THRESHOLD   # (sessions, neurons)
valid_neurons       = list(range(num_neurons))  # all neurons included
_n_valid_pairs      = (~mask & ~low_r2_session_mask).sum()
print(f"Session-ensemble pairs above R²≥{R2_THRESHOLD}: {_n_valid_pairs}")


def _apply_imp_mask(arr):
    """NaN out session-ensemble pairs with no data or below per-session R² threshold."""
    return np.where(
        (mask | low_r2_session_mask)[:, :, np.newaxis], np.nan, arr
    )


# ═══════════ FEATURE CORRELATION MATRIX (used by dendrogram + supergroups) ═══
_X_chunks = []
for sid in session_ids:
    sd = session_dataset_singles[sid]["data"]
    _X_chunks.append(np.concatenate(list(sd.values()), axis=0).astype(np.float32))
X_all = np.concatenate(_X_chunks, axis=0)
del _X_chunks

feat_corr = np.corrcoef(X_all.T)   # (n_feats, n_feats)
del X_all

# ══════════════════════ DRY-RUN VERIFICATION (Section VERIFY) ════════════════
if RUN_DRY_RUN:
    print("\n" + "="*60)
    print("DRY-RUN VERIFICATION")
    print("="*60)

    _dry_seed   = 42
    _dry_s_idx  = 0
    _dry_sess   = session_ids[_dry_s_idx]
    _dry_n_idx  = valid_neurons[0]
    _dry_tidx   = np.load(os.path.join(splits_dir, f"split_seed{_dry_seed}.npy"),
                          allow_pickle=True).item()
    _dry_mdir   = os.path.join(models_root, f"seed{_dry_seed}")
    _dry_mpath  = os.path.join(_dry_mdir,
                               f"session_{_dry_s_idx:02d}_neuron_{_dry_n_idx:02d}.pt")

    print(f"\nCheck 1 — Semantic groups: {n_groups} groups, {n_feats} total columns")
    for name, cols in semantic_groups:
        print(f"  {name:45s} {len(cols):2d} col(s)  indices={cols}")

    if os.path.exists(_dry_mpath):
        _dry_Xnp, _dry_Ynp = _test_arrays(_dry_sess, _dry_tidx[_dry_sess])
        _dry_model = _load_model(_dry_mpath)
        _dry_base_r2 = all_r2[SEEDS.index(_dry_seed), _dry_s_idx, _dry_n_idx]

        print(f"\nCheck 2 — Semantic Global PV (session {_dry_s_idx}, "
              f"{unit_label} {_dry_n_idx}, seed {_dry_seed}, base R²={_dry_base_r2:.4f})")
        _dry_sem_pv = {}
        for g_name, g_cols in semantic_groups:
            _r2s = []
            for _ in range(N_PERM_REPEATS):
                _perm               = _dry_Xnp.copy()
                _pidx               = np.random.permutation(len(_dry_Xnp))
                _perm[:, g_cols]    = _dry_Xnp[_pidx][:, g_cols]
                with torch.no_grad():
                    _, _pp = _dry_model(torch.tensor(_perm, dtype=torch.float32).to(device))
                _r2s.append(r2_score(_dry_Ynp[:, _dry_n_idx], _pp.squeeze().cpu().numpy()))
            _dry_sem_pv[g_name] = _dry_base_r2 - np.mean(_r2s)
            print(f"  {g_name:45s}  sem_PV={_dry_sem_pv[g_name]:.5f}")

        # Compare categorical group vs its individual columns (if old file exists)
        _old_gpv_path = os.path.join(output_dir, "importance_global_pv.npy")
        if os.path.exists(_old_gpv_path):
            _old_gpv = np.load(_old_gpv_path)
            print("\n  Categorical group vs max individual column PV (old per-column):")
            for g_name, g_cols in semantic_groups:
                if g_name in categorical_variables and len(g_cols) >= 2:
                    _old_max = np.nanmax(_old_gpv[_dry_s_idx, _dry_n_idx, g_cols])
                    _sem_val = _dry_sem_pv[g_name]
                    flag = "OK ✓" if _sem_val > _old_max else "FAIL ✗"
                    print(f"  {g_name}: sem={_sem_val:.5f}  max_col={_old_max:.5f}  {flag}")
        else:
            print("  (Old global_pv.npy not found — skipping column comparison)")

        print(f"\nCheck 3 — Semantic Cond-PV k-NN neighborhood size")
        for g_name, g_cols in semantic_groups[:3]:  # spot-check first 3
            _other = [c for c in range(n_feats) if c not in g_cols]
            print(f"  {g_name}: other_cols={len(_other)}  "
                  f"(expected {n_feats}-{len(g_cols)}={n_feats - len(g_cols)})  "
                  f"{'OK ✓' if len(_other) == n_feats - len(g_cols) else 'FAIL ✗'}")

        print(f"\nCheck 4 — IG shapes (n_feats={n_feats}, n_groups={n_groups}, "
              f"n_level_cols={n_level_cols})")
        # Run one IG pass manually
        _n_samp   = len(_dry_Xnp)
        _test_t   = torch.tensor(_dry_Xnp, dtype=torch.float32, device=device)
        _base_t   = torch.zeros(1, n_feats, dtype=torch.float32, device=device)
        _alphas   = np.linspace(0.0, 1.0, IG_STEPS + 1)[1:]
        _grads    = np.zeros((_n_samp, n_feats), dtype=np.float32)
        for _alpha in _alphas[:3]:  # just 3 steps for speed
            _xi = (_base_t + float(_alpha) * (_test_t - _base_t)).detach().requires_grad_(True)
            with torch.enable_grad():
                _, _pred = _dry_model(_xi)
                _pred.sum().backward()
            _grads += _xi.grad.detach().cpu().numpy()
            del _xi
        _grads /= 3
        _ig_attr   = _dry_Xnp * _grads
        _abs_ig    = np.mean(np.abs(_ig_attr), axis=0)
        _signed_ig = np.mean(_ig_attr, axis=0)
        _ig_sem    = np.array([np.sum(_abs_ig[g_cols]) for _, g_cols in semantic_groups])
        _level_ig  = np.full(n_level_cols, np.nan)
        for _li, (_, _, _ci) in enumerate(multilevel_cat_info):
            _active = _dry_Xnp[:, _ci] > 0.5
            if _active.sum() >= MIN_LEVEL_TRIALS:
                _level_ig[_li] = _ig_attr[_active, _ci].mean()
        print(f"  abs_ig shape:    {_abs_ig.shape}   (expected ({n_feats},))")
        print(f"  signed_ig shape: {_signed_ig.shape} (expected ({n_feats},))")
        print(f"  ig_sem shape:    {_ig_sem.shape}   (expected ({n_groups},))")
        print(f"  level_ig shape:  {_level_ig.shape} (expected ({n_level_cols},))")

        print(f"\nCheck 5 — Per-level signed IG (NaN for inactive levels)")
        for _li, (v, l, ci) in enumerate(multilevel_cat_info):
            _active_count = (_dry_Xnp[:, ci] > 0.5).sum()
            _val = _level_ig[_li]
            _flag = "NaN (inactive)" if np.isnan(_val) else f"{_val:.5f}"
            print(f"  {all_feat_cols[ci]:35s}  active={_active_count:5d}  "
                  f"level_IG={_flag}")

        del _dry_model, _dry_Xnp, _dry_Ynp
        torch.cuda.empty_cache()
    else:
        print(f"  Model not found at {_dry_mpath} — skipping checks 2-5")

    print("\nDry-run complete. Proceeding to full pipeline.\n" + "="*60 + "\n")

# ═══════════════════════════════ R² PLOTS ════════════════════════════════════
valid_n = ~np.all(mask, axis=0)
valid_s = ~np.all(mask, axis=1)
n_vn    = valid_n.sum()
n_vs    = valid_s.sum()

r2_c    = np.clip(mean_r2, 0.0, 1.0)
r2_f    = r2_c[np.ix_(np.where(valid_s)[0], np.where(valid_n)[0])]
mask_f  = mask[np.ix_(np.where(valid_s)[0], np.where(valid_n)[0])]
n_mean  = ma.array(r2_f, mask=mask_f).mean(axis=0).filled(np.nan)
n_order = np.argsort(n_mean)[::-1]
valid_n_idx = np.where(valid_n)[0]
valid_s_idx = np.where(valid_s)[0]
N_SHOW      = min(30, n_vn)
show_idx    = n_order[:N_SHOW]
sorted_n    = valid_n_idx[show_idx]

r2_show  = r2_f[:, show_idx].T
m_show   = mask_f[:, show_idx].T
r2_plot  = r2_show.astype(float)
r2_plot[m_show] = np.nan

NO_DATA_COLOR = '#bbbbbb'
cmap_r2 = plt.cm.Blues.copy(); cmap_r2.set_bad(color=NO_DATA_COLOR)

cell_h, cell_w = 0.45, 0.50
fig, ax = plt.subplots(figsize=(max(14, n_vs * cell_w), max(8, N_SHOW * cell_h)))
sns.heatmap(r2_plot, cmap=cmap_r2, vmin=0, vmax=1,
            xticklabels=[f"S{j+1}" for j in valid_s_idx],
            yticklabels=[f"{prefix_name}{sorted_n[i]+1:02d}" for i in range(N_SHOW)],
            ax=ax)
for (i, j) in zip(*np.where(m_show)):
    ax.add_patch(plt.Rectangle([j, i], 1, 1, fill=True, facecolor=NO_DATA_COLOR,
                                hatch='////', edgecolor='#888888', lw=0.5, zorder=2))
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=13)
ax.set_title(f'MLP Mean R² Across {len(SEEDS)} Seeds\n'
             f'(Top {N_SHOW} {unit_label_pl.capitalize()} sorted by mean R²)', fontsize=16, pad=12)
plt.tight_layout()
savefig("r2_heatmap.png")

masked_r2      = ma.array(np.clip(mean_r2, 0, 1), mask=mask)
neuron_mean_r2 = masked_r2.mean(axis=0).filled(np.nan)
order_bar      = np.argsort(neuron_mean_r2)
x              = np.arange(num_neurons)
labels_bar     = [f"{prefix_name}{i+1:02d}" for i in order_bar]
means_bar      = neuron_mean_r2[order_bar]
pal            = sns.color_palette("mako_r", as_cmap=True)(np.linspace(0.15, 0.85, num_neurons))

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(x, means_bar, width=0.7, color=pal, zorder=3, linewidth=0)
ax.axhline(0, color='firebrick', linestyle='--', linewidth=1)
for pos in range(num_neurons - 5, num_neurons):
    ax.text(pos, means_bar[pos] + 0.005, labels_bar[pos],
            ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=90)
ax.set_xticks(x[::3])
ax.set_xticklabels(labels_bar[::3], rotation=45, ha='right', fontsize=8)
ax.set_ylabel("Mean R²")
ax.set_title(f"Per-{unit_label.capitalize()} Mean R² Across {len(SEEDS)} Seeds × {num_sessions} Sessions")
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
savefig("r2_bar.png")

# ══════════════════════════════════════════════════════════════════════════════
#                  METHOD 1: GLOBAL PV — SEMANTIC GROUPS (Section 2)
# ══════════════════════════════════════════════════════════════════════════════
# Permute all columns in each group with the same row index. This preserves
# within-group joint structure (e.g. one-hot mutual exclusivity) while
# destroying the group's relationship with the target.
_gpv_sem_path = os.path.join(output_dir, "importance_global_pv_semantic.npy")

if RUN_GLOBAL_PV:
    all_gpv = np.full((len(SEEDS), num_sessions, num_neurons, n_groups), np.nan)
    for seed_idx, seed in enumerate(SEEDS):
        ckpt = os.path.join(output_dir, f"importance_global_pv_semantic_seed{seed}.npy")
        if os.path.exists(ckpt):
            all_gpv[seed_idx] = np.load(ckpt)
            print(f"Seed {seed} global-PV-sem: loaded checkpoint.")
            continue
        tidx_map = np.load(os.path.join(splits_dir, f"split_seed{seed}.npy"),
                           allow_pickle=True).item()
        mdir = os.path.join(models_root, f"seed{seed}")
        for s_idx, sess in enumerate(session_ids):
            Xnp, Ynp = _test_arrays(sess, tidx_map[sess])
            n_samp    = len(Xnp)
            for n_idx in range(num_neurons):
                if mask[s_idx, n_idx] or low_r2_session_mask[s_idx, n_idx]:
                    continue
                mpath = os.path.join(mdir, f"session_{s_idx:02d}_neuron_{n_idx:02d}.pt")
                if not os.path.exists(mpath):
                    continue
                model   = _load_model(mpath)
                base_r2 = all_r2[seed_idx, s_idx, n_idx]
                for g_idx, (_, g_cols) in enumerate(semantic_groups):
                    r2_perm = []
                    for _ in range(N_PERM_REPEATS):
                        perm             = Xnp.copy()
                        pidx             = np.random.permutation(n_samp)
                        perm[:, g_cols]  = Xnp[pidx][:, g_cols]
                        with torch.no_grad():
                            _, pp = model(torch.tensor(perm, dtype=torch.float32).to(device))
                        r2_perm.append(r2_score(Ynp[:, n_idx], pp.squeeze().cpu().numpy()))
                    all_gpv[seed_idx, s_idx, n_idx, g_idx] = base_r2 - np.mean(r2_perm)
                del model
                torch.cuda.empty_cache()
            del Xnp, Ynp
        np.save(ckpt, all_gpv[seed_idx])
        print(f"Seed {seed} global-PV-sem done, checkpoint saved.")
    with np.errstate(all='ignore'):
        global_pv_sem = np.nanmedian(all_gpv, axis=0)
    np.save(_gpv_sem_path, global_pv_sem)
    print("Saved importance_global_pv_semantic.npy")
else:
    global_pv_sem = np.load(_gpv_sem_path)
    print(f"Loaded {_gpv_sem_path}")

global_pv_sem_m = _apply_imp_mask(global_pv_sem)

# ══════════════════════════════════════════════════════════════════════════════
#           METHOD 2: COND-PV — SEMANTIC GROUPS (Section 3)
# ══════════════════════════════════════════════════════════════════════════════
# k-NN neighbourhood built on all columns EXCLUDING the entire group, so
# one-hot partners are not in the conditioning set. Joint structure preserved
# by using the same neighbour row for all columns in the group.
_cpv_sem_path = os.path.join(output_dir, "importance_cond_pv_semantic.npy")

if RUN_COND_PV:
    all_cpv = np.full((len(SEEDS), num_sessions, num_neurons, n_groups), np.nan)
    for seed_idx, seed in enumerate(SEEDS):
        ckpt = os.path.join(output_dir, f"importance_cond_pv_semantic_seed{seed}.npy")
        if os.path.exists(ckpt):
            all_cpv[seed_idx] = np.load(ckpt)
            print(f"Seed {seed} cond-PV-sem: loaded checkpoint.")
            continue
        tidx_map = np.load(os.path.join(splits_dir, f"split_seed{seed}.npy"),
                           allow_pickle=True).item()
        mdir = os.path.join(models_root, f"seed{seed}")
        for s_idx, sess in enumerate(session_ids):
            Xnp, Ynp = _test_arrays(sess, tidx_map[sess])
            n_samp    = len(Xnp)

            # Pre-compute one k-NN index per group, excluding all group columns.
            nn_nb = {}
            for g_idx, (_, g_cols) in enumerate(semantic_groups):
                other = [c for c in range(n_feats) if c not in g_cols]
                nn    = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, n_jobs=1)
                nn.fit(Xnp[:, other])
                _, idx_nb    = nn.kneighbors(Xnp[:, other])
                nn_nb[g_idx] = idx_nb[:, 1:]  # (n_samp, K) — exclude self

            for n_idx in range(num_neurons):
                if mask[s_idx, n_idx] or low_r2_session_mask[s_idx, n_idx]:
                    continue
                mpath = os.path.join(mdir, f"session_{s_idx:02d}_neuron_{n_idx:02d}.pt")
                if not os.path.exists(mpath):
                    continue
                model   = _load_model(mpath)
                base_r2 = all_r2[seed_idx, s_idx, n_idx]
                for g_idx, (_, g_cols) in enumerate(semantic_groups):
                    nb_idx  = nn_nb[g_idx]
                    r2_perm = []
                    for _ in range(N_PERM_REPEATS):
                        perm    = Xnp.copy()
                        chosen  = nb_idx[np.arange(n_samp),
                                         np.random.randint(0, K_NEIGHBORS, n_samp)]
                        perm[:, g_cols] = Xnp[chosen][:, g_cols]
                        with torch.no_grad():
                            _, pp = model(torch.tensor(perm, dtype=torch.float32).to(device))
                        r2_perm.append(r2_score(Ynp[:, n_idx], pp.squeeze().cpu().numpy()))
                    all_cpv[seed_idx, s_idx, n_idx, g_idx] = base_r2 - np.mean(r2_perm)
                del model
                torch.cuda.empty_cache()
            del Xnp, Ynp, nn_nb
        np.save(ckpt, all_cpv[seed_idx])
        print(f"Seed {seed} cond-PV-sem done, checkpoint saved.")
    with np.errstate(all='ignore'):
        cond_pv_sem = np.nanmedian(all_cpv, axis=0)
    np.save(_cpv_sem_path, cond_pv_sem)
    print("Saved importance_cond_pv_semantic.npy")
else:
    cond_pv_sem = np.load(_cpv_sem_path)
    print(f"Loaded {_cpv_sem_path}")

cond_pv_sem_m = _apply_imp_mask(cond_pv_sem)

# ══════════════════════════════════════════════════════════════════════════════
#               METHOD 3: INTEGRATED GRADIENTS (Section 4)
# ══════════════════════════════════════════════════════════════════════════════
def compute_ig(model, Xnp, steps, dev):
    """
    Returns:
      abs_ig     (n_feats,)      – mean |IG| per column
      signed_ig  (n_feats,)      – mean signed IG per column (all trials)
      ig_sem     (n_groups,)     – sum of abs_ig within each semantic group
      level_ig   (n_level_cols,) – mean signed IG per categorical level
                                   (subset to active trials; NaN if inactive)
    """
    n_samp  = len(Xnp)
    test_t  = torch.tensor(Xnp,           dtype=torch.float32, device=dev)
    base_t  = torch.zeros(1, n_feats,     dtype=torch.float32, device=dev)
    alphas  = np.linspace(0.0, 1.0, steps + 1)[1:]

    grads_acc = np.zeros((n_samp, n_feats), dtype=np.float32)
    for alpha in alphas:
        x_int = (base_t + float(alpha) * (test_t - base_t)).detach().requires_grad_(True)
        with torch.enable_grad():
            _, pred = model(x_int)
            pred.sum().backward()
        grads_acc += x_int.grad.detach().cpu().numpy()
        del x_int

    grads_acc /= steps
    ig_attr    = Xnp * grads_acc   # (n_samp, n_feats)

    abs_ig    = np.mean(np.abs(ig_attr), axis=0)
    signed_ig = np.mean(ig_attr,         axis=0)
    ig_sem    = np.array([np.sum(abs_ig[g_cols]) for _, g_cols in semantic_groups])

    level_ig = np.full(n_level_cols, np.nan)
    for li, (_, _, col_idx) in enumerate(multilevel_cat_info):
        active = Xnp[:, col_idx] > 0.5
        if active.sum() >= MIN_LEVEL_TRIALS:
            level_ig[li] = ig_attr[active, col_idx].mean()

    return abs_ig, signed_ig, ig_sem, level_ig


_ig_sem_path      = os.path.join(output_dir, "importance_ig_semantic.npy")
_signed_col_path  = os.path.join(output_dir, "signed_ig_per_column.npy")
_signed_lvl_path  = os.path.join(output_dir, "signed_ig_per_level.npy")

if RUN_IG:
    all_ig_abs_col  = np.full((len(SEEDS), num_sessions, num_neurons, n_feats),      np.nan)
    all_ig_sem      = np.full((len(SEEDS), num_sessions, num_neurons, n_groups),     np.nan)
    all_signed_col  = np.full((len(SEEDS), num_sessions, num_neurons, n_feats),      np.nan)
    all_level_ig    = np.full((len(SEEDS), num_sessions, num_neurons, n_level_cols), np.nan)

    for seed_idx, seed in enumerate(SEEDS):
        ckpt = os.path.join(output_dir, f"ig_checkpoint_seed{seed}.npz")
        if os.path.exists(ckpt):
            _d = np.load(ckpt)
            all_ig_abs_col[seed_idx] = _d['abs_col']
            all_ig_sem[seed_idx]     = _d['sem']
            all_signed_col[seed_idx] = _d['signed_col']
            all_level_ig[seed_idx]   = _d['level']
            print(f"Seed {seed} IG: loaded checkpoint.")
            continue
        tidx_map = np.load(os.path.join(splits_dir, f"split_seed{seed}.npy"),
                           allow_pickle=True).item()
        mdir = os.path.join(models_root, f"seed{seed}")
        for s_idx, sess in enumerate(session_ids):
            Xnp, _ = _test_arrays(sess, tidx_map[sess])
            for n_idx in range(num_neurons):
                if mask[s_idx, n_idx] or low_r2_session_mask[s_idx, n_idx]:
                    continue
                mpath = os.path.join(mdir, f"session_{s_idx:02d}_neuron_{n_idx:02d}.pt")
                if not os.path.exists(mpath):
                    continue
                model = _load_model(mpath)
                abs_ig, signed_ig, ig_sem, level_ig = compute_ig(model, Xnp, IG_STEPS, device)
                all_ig_abs_col[seed_idx, s_idx, n_idx, :]  = abs_ig
                all_ig_sem[seed_idx,     s_idx, n_idx, :]  = ig_sem
                all_signed_col[seed_idx, s_idx, n_idx, :]  = signed_ig
                all_level_ig[seed_idx,   s_idx, n_idx, :]  = level_ig
                del model
                torch.cuda.empty_cache()
            del Xnp
        np.savez(ckpt,
                 abs_col=all_ig_abs_col[seed_idx],
                 sem=all_ig_sem[seed_idx],
                 signed_col=all_signed_col[seed_idx],
                 level=all_level_ig[seed_idx])
        print(f"Seed {seed} IG done, checkpoint saved.")

    with np.errstate(all='ignore'):
        ig_sem     = np.nanmedian(all_ig_sem,     axis=0)
        signed_col = np.nanmedian(all_signed_col, axis=0)
        level_ig   = np.nanmedian(all_level_ig,   axis=0)

    np.save(_ig_sem_path,     ig_sem)
    np.save(_signed_col_path, signed_col)
    np.save(_signed_lvl_path, level_ig)
    print("Saved importance_ig_semantic.npy, signed_ig_per_column.npy, signed_ig_per_level.npy")
else:
    ig_sem     = np.load(_ig_sem_path)
    signed_col = np.load(_signed_col_path)
    level_ig   = np.load(_signed_lvl_path)
    print(f"Loaded IG semantic files.")
    # Load per-seed arrays for per-level R² section
    all_ig_sem     = None
    all_signed_col = None
    all_level_ig   = None

ig_sem_m = _apply_imp_mask(ig_sem)
_ig_max  = np.nanmax(ig_sem_m, axis=-1, keepdims=True)
ig_sem_m_norm = ig_sem_m / (_ig_max + 1e-12)

# ══════════════════════════════════════════════════════════════════════════════
#         METHOD 4: SUPERGROUP PV — WARD CLUSTERS OF SEMANTIC VARS (Section 5)
# ══════════════════════════════════════════════════════════════════════════════
# Build n_groups × n_groups distance matrix at the semantic-variable level.
# dist(gi, gj) = 1 - max(|corr(col_a, col_b)|) over all column pairs.
# max |corr| is the most conservative choice: if any pair is highly correlated,
# the groups are considered close.
sem_dist = np.ones((n_groups, n_groups))
np.fill_diagonal(sem_dist, 0.0)
for i, (_, gi_cols) in enumerate(semantic_groups):
    for j, (_, gj_cols) in enumerate(semantic_groups):
        if i == j:
            continue
        sub = np.abs(feat_corr[np.ix_(gi_cols, gj_cols)])
        sem_dist[i, j] = 1.0 - float(sub.max())

# Enforce exact symmetry and clip to valid range
sem_dist = (sem_dist + sem_dist.T) / 2
sem_dist = np.clip(sem_dist, 0.0, 1.0)
np.fill_diagonal(sem_dist, 0.0)

Z_sem               = linkage(squareform(sem_dist), method='ward')
supergroup_labels   = fcluster(Z_sem, t=CLUSTER_DIST_THR, criterion='distance')
n_supergroups       = int(supergroup_labels.max())

supergroup_names = []
for sgid in range(1, n_supergroups + 1):
    members = [group_names[i] for i in range(n_groups) if supergroup_labels[i] == sgid]
    supergroup_names.append(' | '.join(members))

print(f"\nSupergroups (dist_thr={CLUSTER_DIST_THR}): {n_supergroups} supergroups")
for sgid, name in enumerate(supergroup_names, 1):
    print(f"  SG{sgid:02d}: {name}")

np.save(os.path.join(output_dir, "supergroup_labels.npy"), supergroup_labels)

_sgpv_path = os.path.join(output_dir, "importance_supergrouped_pv.npy")

if RUN_GROUPED_PV:
    all_sgpv = np.full((len(SEEDS), num_sessions, num_neurons, n_supergroups), np.nan)
    for seed_idx, seed in enumerate(SEEDS):
        ckpt = os.path.join(output_dir, f"importance_supergrouped_pv_seed{seed}.npy")
        if os.path.exists(ckpt):
            all_sgpv[seed_idx] = np.load(ckpt)
            print(f"Seed {seed} supergroup-PV: loaded checkpoint.")
            continue
        tidx_map = np.load(os.path.join(splits_dir, f"split_seed{seed}.npy"),
                           allow_pickle=True).item()
        mdir = os.path.join(models_root, f"seed{seed}")
        for s_idx, sess in enumerate(session_ids):
            Xnp, Ynp = _test_arrays(sess, tidx_map[sess])
            n_samp    = len(Xnp)
            for n_idx in range(num_neurons):
                if mask[s_idx, n_idx] or low_r2_session_mask[s_idx, n_idx]:
                    continue
                mpath = os.path.join(mdir, f"session_{s_idx:02d}_neuron_{n_idx:02d}.pt")
                if not os.path.exists(mpath):
                    continue
                model   = _load_model(mpath)
                base_r2 = all_r2[seed_idx, s_idx, n_idx]
                for sgid in range(1, n_supergroups + 1):
                    # Collect all columns from all semantic groups in this supergroup
                    sg_cols = []
                    for g_idx in range(n_groups):
                        if supergroup_labels[g_idx] == sgid:
                            sg_cols.extend(semantic_groups[g_idx][1])
                    r2_perm = []
                    for _ in range(N_PERM_REPEATS):
                        perm             = Xnp.copy()
                        pidx             = np.random.permutation(n_samp)
                        perm[:, sg_cols] = Xnp[pidx][:, sg_cols]
                        with torch.no_grad():
                            _, pp = model(torch.tensor(perm, dtype=torch.float32).to(device))
                        r2_perm.append(r2_score(Ynp[:, n_idx], pp.squeeze().cpu().numpy()))
                    all_sgpv[seed_idx, s_idx, n_idx, sgid - 1] = base_r2 - np.mean(r2_perm)
                del model
                torch.cuda.empty_cache()
            del Xnp, Ynp
        np.save(ckpt, all_sgpv[seed_idx])
        print(f"Seed {seed} supergroup-PV done, checkpoint saved.")
    with np.errstate(all='ignore'):
        supergrouped_pv = np.nanmedian(all_sgpv, axis=0)
    np.save(_sgpv_path, supergrouped_pv)
    print("Saved importance_supergrouped_pv.npy")
else:
    supergrouped_pv = np.load(_sgpv_path)
    print(f"Loaded {_sgpv_path}")

supergrouped_pv_m = _apply_imp_mask(supergrouped_pv)

# ══════════════════════════════════════════════════════════════════════════════
#             METHOD 5: PER-CATEGORICAL-LEVEL DECOMPOSITION (Section 6)
# ══════════════════════════════════════════════════════════════════════════════
_r2_lvl_path     = os.path.join(output_dir, "r2_per_level.npy")
_count_lvl_path  = os.path.join(output_dir, "trial_counts_per_level.npy")

if RUN_PER_LEVEL:
    all_r2_lvl    = np.full((len(SEEDS), num_sessions, num_neurons, n_level_cols), np.nan)
    all_count_lvl = np.full((len(SEEDS), num_sessions, n_level_cols), 0, dtype=np.int32)

    for seed_idx, seed in enumerate(SEEDS):
        ckpt_r2  = os.path.join(output_dir, f"r2_level_seed{seed}.npy")
        ckpt_cnt = os.path.join(output_dir, f"counts_level_seed{seed}.npy")
        if os.path.exists(ckpt_r2) and os.path.exists(ckpt_cnt):
            all_r2_lvl[seed_idx]    = np.load(ckpt_r2)
            all_count_lvl[seed_idx] = np.load(ckpt_cnt)
            print(f"Seed {seed} per-level R²: loaded checkpoint.")
            continue
        tidx_map = np.load(os.path.join(splits_dir, f"split_seed{seed}.npy"),
                           allow_pickle=True).item()
        mdir = os.path.join(models_root, f"seed{seed}")
        for s_idx, sess in enumerate(session_ids):
            Xnp, Ynp = _test_arrays(sess, tidx_map[sess])
            # Count active trials per level (same across neurons)
            for li, (_, _, col_idx) in enumerate(multilevel_cat_info):
                all_count_lvl[seed_idx, s_idx, li] = int((Xnp[:, col_idx] > 0.5).sum())

            Xt = torch.tensor(Xnp, dtype=torch.float32).to(device)
            for n_idx in range(num_neurons):
                if mask[s_idx, n_idx] or low_r2_session_mask[s_idx, n_idx]:
                    continue
                mpath = os.path.join(mdir, f"session_{s_idx:02d}_neuron_{n_idx:02d}.pt")
                if not os.path.exists(mpath):
                    continue
                model = _load_model(mpath)
                with torch.no_grad():
                    _, preds = model(Xt)
                pnp = preds.squeeze().cpu().numpy()
                for li, (_, _, col_idx) in enumerate(multilevel_cat_info):
                    active = Xnp[:, col_idx] > 0.5
                    if active.sum() < MIN_LEVEL_TRIALS:
                        continue
                    actual_sub = Ynp[active, n_idx]
                    if np.var(actual_sub) < 1e-8:
                        continue
                    all_r2_lvl[seed_idx, s_idx, n_idx, li] = r2_score(
                        actual_sub, pnp[active]
                    )
                del model
                torch.cuda.empty_cache()
            del Xt, Xnp, Ynp
        np.save(ckpt_r2,  all_r2_lvl[seed_idx])
        np.save(ckpt_cnt, all_count_lvl[seed_idx])
        print(f"Seed {seed} per-level R² done, checkpoint saved.")

    with np.errstate(all='ignore'):
        r2_per_level    = np.nanmedian(all_r2_lvl,    axis=0)
        counts_per_lvl  = np.nanmedian(all_count_lvl, axis=0).astype(int)

    np.save(_r2_lvl_path,    all_r2_lvl)
    np.save(_count_lvl_path, all_count_lvl)
    print("Saved r2_per_level.npy and trial_counts_per_level.npy")
else:
    all_r2_lvl    = np.load(_r2_lvl_path)
    all_count_lvl = np.load(_count_lvl_path)
    with np.errstate(all='ignore'):
        r2_per_level   = np.nanmedian(all_r2_lvl,    axis=0)
        counts_per_lvl = np.nanmedian(all_count_lvl, axis=0).astype(int)
    print("Loaded per-level R² files.")

# signed IG per level (from IG run above)
level_ig_m = np.where(
    (mask | low_r2_session_mask)[:, :, np.newaxis], np.nan, level_ig
)

# ══════════════════════════════════════════════════════════════════════════════
#                    VISUALISATION — INDIVIDUAL METHOD PLOTS
# ══════════════════════════════════════════════════════════════════════════════
TOP_K       = 6
_nm         = np.nanmax(np.where(mask | low_r2_session_mask, np.nan, mean_r2), axis=0)
top_neurons = np.argsort(_nm)[::-1][:TOP_K]


def _imp_bar(imp_flat, feat_labels, title, fname, ylabel='Mean R² drop'):
    feat_mean  = np.nanmean(imp_flat, axis=0)
    feat_std   = np.nanstd(imp_flat,  axis=0)
    feat_order = np.argsort(feat_mean)[::-1]
    x = np.arange(len(feat_labels))
    fig, ax = plt.subplots(figsize=(max(8, len(feat_labels) * 0.5), 5))
    ax.bar(x, feat_mean[feat_order], yerr=feat_std[feat_order],
           color='steelblue', capsize=4, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([feat_labels[i] for i in feat_order], rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    savefig(fname)


def _imp_heatmap(imp_masked, feat_labels, title, fname, cbar_label='importance'):
    neuron_imp = np.nanmean(imp_masked, axis=0)
    valid_n_hm = ~np.all(np.isnan(neuron_imp), axis=1)
    hm         = neuron_imp[valid_n_hm, :].T
    orig_idx   = np.where(valid_n_hm)[0]
    order_n    = np.argsort(np.nanmean(hm, axis=0))[::-1]
    hm         = hm[:, order_n]
    sorted_orig = orig_idx[order_n]
    xtick_lab  = [f"{prefix_name}{sorted_orig[i]+1:02d}" for i in range(len(sorted_orig))]
    valid_vals = hm[~np.isnan(hm)]
    vmax = float(np.nanpercentile(valid_vals, 98)) if len(valid_vals) > 0 else 1.0
    n_cols = len(sorted_orig)
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 0.65), max(4, len(feat_labels) * 0.45)))
    sns.heatmap(hm, xticklabels=xtick_lab, yticklabels=feat_labels,
                cmap='Blues', ax=ax, vmin=0, vmax=vmax,
                cbar_kws={'label': f'{cbar_label} (max={vmax:.3f})'})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    savefig(fname)


# ── Global PV (semantic) ────────────────────────────────────────────────────
_imp_bar(
    global_pv_sem_m.reshape(-1, n_groups), group_names,
    f'Global PV (semantic) — Mean R² Drop  ({len(SEEDS)} seeds, session R² ≥ {R2_THRESHOLD})',
    'global_pv_sem_bar.png'
)
_imp_heatmap(
    global_pv_sem_m, group_names,
    f'Global PV (semantic) — Group × {unit_label.capitalize()} (mean over sessions)',
    'global_pv_sem_heatmap.png'
)

# ── Cond-PV (semantic) ──────────────────────────────────────────────────────
_imp_bar(
    cond_pv_sem_m.reshape(-1, n_groups), group_names,
    f'Cond-PV (semantic, k-NN k={K_NEIGHBORS}) — Mean R² Drop',
    'cond_pv_sem_bar.png'
)
_imp_heatmap(
    cond_pv_sem_m, group_names,
    f'Cond-PV (semantic, k-NN) — Group × {unit_label.capitalize()}',
    'cond_pv_sem_heatmap.png'
)

# ── IG (semantic, normalised) ───────────────────────────────────────────────
_imp_bar(
    ig_sem_m_norm.reshape(-1, n_groups), group_names,
    f'Integrated Gradients (semantic) — Mean |Attribution| (normalised)',
    'ig_sem_bar.png', ylabel='Mean |IG| summed (normalised)'
)
_imp_heatmap(
    ig_sem_m_norm, group_names,
    f'Integrated Gradients (semantic) — Group × {unit_label.capitalize()}',
    'ig_sem_heatmap.png', cbar_label='Mean |IG| (norm)'
)

# ── Supergroup PV ───────────────────────────────────────────────────────────
_imp_bar(
    supergrouped_pv_m.reshape(-1, n_supergroups), supergroup_names,
    f'Supergroup PV — R² Drop  (dist_thr={CLUSTER_DIST_THR})',
    'supergrouped_pv_bar.png'
)
_imp_heatmap(
    supergrouped_pv_m, supergroup_names,
    f'Supergroup PV — Supergroup × {unit_label.capitalize()}',
    'supergrouped_pv_heatmap.png'
)

# ══════════════════════════════════════════════════════════════════════════════
#            PER-CATEGORICAL-LEVEL FIGURES (Section 6.3)
# ══════════════════════════════════════════════════════════════════════════════
# Determine which categorical variables have ≥2 levels
_lvl_vars = []
_seen     = set()
for v, l, ci in multilevel_cat_info:
    if v not in _seen:
        _lvl_vars.append(v)
        _seen.add(v)

# ── 6.3a: Signed IG per level ───────────────────────────────────────────────
# For each categorical variable: one cluster of bars, one per level.
# level_ig_m shape: (n_sessions, n_neurons, n_level_cols)
level_ig_flat = level_ig_m.reshape(-1, n_level_cols)   # (pairs, n_level_cols)
level_ig_mean = np.nanmean(level_ig_flat, axis=0)
level_ig_std  = np.nanstd(level_ig_flat,  axis=0)

# Build cluster positions
_var_to_levels = {}
for li, (v, l, ci) in enumerate(multilevel_cat_info):
    _var_to_levels.setdefault(v, []).append((l, li))

fig, ax = plt.subplots(figsize=(max(10, n_level_cols * 0.6 + 2), 5))
_x      = 0
_xticks, _xlabels = [], []
_cluster_gap = 0.4
for var in _lvl_vars:
    levels = _var_to_levels[var]
    for l, li in levels:
        val = level_ig_mean[li]
        err = level_ig_std[li]
        col = 'firebrick' if val >= 0 else 'steelblue'
        ax.bar(_x, val, width=0.7, yerr=err, color=col, alpha=0.8, capsize=4,
               ecolor='black')
        _xticks.append(_x)
        _xlabels.append(f"{l}")
        _x += 1
    # Add variable label between clusters
    _mid = _x - len(levels) / 2 - 0.5
    ax.text(_mid, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.0005,
            var, ha='center', va='top', fontsize=8, color='dimgray',
            transform=ax.get_xaxis_transform())
    _x += _cluster_gap

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(_xticks)
ax.set_xticklabels(_xlabels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Mean signed IG (active trials only)')
ax.set_title('Signed IG per categorical level\n(red = positive drive, blue = negative, bars = levels within each variable)')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
savefig('signed_ig_per_level.png')
print("Saved signed_ig_per_level.png")

# ── 6.3b: R² per level ──────────────────────────────────────────────────────
# r2_per_level shape: (n_sessions, n_neurons, n_level_cols)
r2_lvl_flat = np.where(
    (mask | low_r2_session_mask)[:, :, np.newaxis], np.nan,
    r2_per_level
).reshape(-1, n_level_cols)

r2_lvl_mean = np.nanmean(r2_lvl_flat, axis=0)
r2_lvl_std  = np.nanstd(r2_lvl_flat,  axis=0)
cnt_mean    = np.nanmean(counts_per_lvl.reshape(-1, n_level_cols), axis=0).astype(int)

fig, ax = plt.subplots(figsize=(max(10, n_level_cols * 0.6 + 2), 5))
_x = 0
_xticks, _xlabels = [], []
for var in _lvl_vars:
    levels = _var_to_levels[var]
    for l, li in levels:
        val  = r2_lvl_mean[li]
        err  = r2_lvl_std[li]
        low  = cnt_mean[li] < MIN_LEVEL_TRIALS
        col  = '#aaaaaa' if low else 'steelblue'
        hatch = '////' if low else None
        ax.bar(_x, val, width=0.7, yerr=err, color=col, alpha=0.8, capsize=4,
               ecolor='black', hatch=hatch)
        ax.text(_x, (val or 0) + (err or 0) + 0.002, f"n={cnt_mean[li]}",
                ha='center', va='bottom', fontsize=6, rotation=90)
        _xticks.append(_x)
        _xlabels.append(f"{l}")
        _x += 1
    _mid = _x - len(levels) / 2 - 0.5
    ax.text(_mid, -0.005, var, ha='center', va='top', fontsize=8, color='dimgray',
            transform=ax.get_xaxis_transform())
    _x += _cluster_gap

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(_xticks)
ax.set_xticklabels(_xlabels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Mean R² (active trials only)')
ax.set_title(f'R² per categorical level  (hatched = fewer than {MIN_LEVEL_TRIALS} trials)')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
savefig('r2_per_level.png')
print("Saved r2_per_level.png")

# ══════════════════════════════════════════════════════════════════════════════
#             CROSS-METHOD COMPARISON AT SEMANTIC-GROUP LEVEL (Section 7)
# ══════════════════════════════════════════════════════════════════════════════
# Feature correlation dendrogram over semantic variables
fig, ax = plt.subplots(figsize=(max(10, n_groups * 0.6), 4))
_dg(Z_sem, labels=group_names, ax=ax, leaf_rotation=45,
    color_threshold=CLUSTER_DIST_THR, above_threshold_color='grey')
ax.axhline(CLUSTER_DIST_THR, color='firebrick', linestyle='--', lw=1.2,
           label=f'cut = {CLUSTER_DIST_THR}')
ax.set_title('Semantic-variable dendrogram (Ward, max |corr| distance)')
ax.legend(fontsize=9)
plt.tight_layout()
savefig('sem_var_dendrogram.png')
print("Saved sem_var_dendrogram.png")

# Aggregate importance to (n_groups,) for scatter/rank plots
flat_gpv_s = global_pv_sem_m.reshape(-1, n_groups)
flat_cpv_s = cond_pv_sem_m.reshape(-1, n_groups)
flat_ig_s  = ig_sem_m_norm.reshape(-1, n_groups)
valid_rows  = (np.any(np.isfinite(flat_gpv_s), axis=1) &
               np.any(np.isfinite(flat_cpv_s), axis=1) &
               np.any(np.isfinite(flat_ig_s),  axis=1))

gpv_mean_s = np.nanmean(flat_gpv_s[valid_rows], axis=0)
cpv_mean_s = np.nanmean(flat_cpv_s[valid_rows], axis=0)
ig_mean_s  = np.nanmean(flat_ig_s[valid_rows],  axis=0)

# ── Global PV vs Cond-PV scatter (semantic) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(gpv_mean_s, cpv_mean_s, s=80,
           c=np.arange(n_groups), cmap='tab20', zorder=3)
for i, name in enumerate(group_names):
    ax.annotate(name[:18], (gpv_mean_s[i], cpv_mean_s[i]),
                fontsize=7, textcoords='offset points', xytext=(4, 2))
lim = max(gpv_mean_s.max(), cpv_mean_s.max()) * 1.15
ax.plot([0, lim], [0, lim], 'k--', lw=0.8, label='y=x')
ax.set_xlabel('Global PV (mean R² drop, semantic)')
ax.set_ylabel('Cond-PV (mean R² drop, semantic)')
ax.set_title('Global PV vs Cond-PV — semantic groups\n'
             '(points below diagonal = collinearity-inflated global score)')
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(fontsize=9)
plt.tight_layout()
savefig('global_vs_cond_pv_scatter.png')
print("Saved global_vs_cond_pv_scatter.png")

# ── Global PV vs IG scatter (semantic) ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(gpv_mean_s, ig_mean_s, s=80,
           c=np.arange(n_groups), cmap='tab20', zorder=3)
for i, name in enumerate(group_names):
    ax.annotate(name[:18], (gpv_mean_s[i], ig_mean_s[i]),
                fontsize=7, textcoords='offset points', xytext=(4, 2))
ax.set_xlabel('Global PV (mean R² drop, semantic)')
ax.set_ylabel('Integrated Gradients (mean |IG| summed, norm, semantic)')
ax.set_title('Global PV vs Integrated Gradients — semantic groups\n'
             '(disagreement = saturation or threshold-gating)')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
savefig('global_pv_vs_ig_scatter.png')
print("Saved global_pv_vs_ig_scatter.png")

# ── NEW: Supergroup PV vs IG scatter ────────────────────────────────────────
# Supergroup IG = sum of constituent semantic-group IG means
sg_ig_mean = np.array([
    np.sum([ig_mean_s[g_idx]
            for g_idx in range(n_groups) if supergroup_labels[g_idx] == sgid])
    for sgid in range(1, n_supergroups + 1)
])
sg_pv_mean = np.nanmean(supergrouped_pv_m.reshape(-1, n_supergroups), axis=0)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(sg_pv_mean, sg_ig_mean, s=80, c=np.arange(n_supergroups), cmap='tab20', zorder=3)
for i, name in enumerate(supergroup_names):
    ax.annotate(name[:22], (sg_pv_mean[i], sg_ig_mean[i]),
                fontsize=7, textcoords='offset points', xytext=(4, 2))
ax.set_xlabel('Supergroup PV (mean R² drop)')
ax.set_ylabel('Sum of semantic-group Global PV')
ax.set_title('Supergroup PV vs summed semantic-group PV\n'
             '(locomotion redundancy diagnostic)')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
savefig('supergrouped_pv_vs_ig_scatter.png')
print("Saved supergrouped_pv_vs_ig_scatter.png")

# ── Spearman rank-correlation matrix (semantic) ──────────────────────────────
_vectors_s = {
    'Global PV': gpv_mean_s,
    'Cond-PV':   cpv_mean_s,
    'IG (norm)': ig_mean_s,
}
method_names_s = list(_vectors_s.keys())
n_m  = len(method_names_s)
rho_m = np.eye(n_m)
for i, mi in enumerate(method_names_s):
    for j, mj in enumerate(method_names_s):
        if i != j:
            rho_m[i, j] = spearmanr(_vectors_s[mi], _vectors_s[mj]).statistic

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(rho_m, annot=True, fmt='.2f', xticklabels=method_names_s,
            yticklabels=method_names_s, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            ax=ax, cbar_kws={'label': 'Spearman ρ'})
ax.set_title(f'Cross-method ranking agreement — {n_groups} semantic groups\n'
             f'(Spearman ρ)')
plt.tight_layout()
savefig('method_rank_correlation.png')
print("Saved method_rank_correlation.png")

# ── Side-by-side ranking bars (semantic) ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, (mname, mvec) in zip(axes, _vectors_s.items()):
    order_m = np.argsort(mvec)[::-1]
    ax.barh(np.arange(n_groups), mvec[order_m], color='steelblue', alpha=0.8)
    ax.set_yticks(np.arange(n_groups))
    ax.set_yticklabels([group_names[i] for i in order_m], fontsize=9)
    ax.set_title(mname, fontsize=12)
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Semantic group (ranked)')
fig.suptitle('Feature importance rankings — semantic groups', fontsize=14, y=1.01)
plt.tight_layout()
savefig('method_comparison_ranked_bars.png')
print("Saved method_comparison_ranked_bars.png")

# ── Supergroup vs sum-of-semantic-group PV ────────────────────────────────────
# "sum of individual" = sum of each constituent semantic-group's mean PV
flat_sgpv   = supergrouped_pv_m.reshape(-1, n_supergroups)
sg_sum_ind  = np.array([
    np.sum([gpv_mean_s[g_idx]
            for g_idx in range(n_groups) if supergroup_labels[g_idx] == sgid])
    for sgid in range(1, n_supergroups + 1)
])
sg_grp = np.nanmean(flat_sgpv, axis=0)

fig, ax = plt.subplots(figsize=(max(8, n_supergroups * 0.9), 5))
x = np.arange(n_supergroups)
w = 0.35
ax.bar(x - w/2, sg_sum_ind, w, label='Sum of semantic-group PV', color='steelblue', alpha=0.8)
ax.bar(x + w/2, sg_grp,     w, label='Supergroup PV (joint)',    color='coral',     alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"SG{i+1}" for i in range(n_supergroups)], rotation=45, ha='right')
ax.set_ylabel('Mean R² drop')
ax.set_title('Sum-of-semantic-group vs supergroup PV\n'
             '(joint > sum → locomotion-level collinearity across variables)')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
savefig('supergrouped_vs_individual_pv.png')
print("Saved supergrouped_vs_individual_pv.png")

# ══════════════════════════════════════════════════════════════════════════════
#         SECTION 8 — APPENDIX: PER-COLUMN GLOBAL PV (motivational figure)
# ══════════════════════════════════════════════════════════════════════════════
# Shows why semantic grouping is needed: naive per-column PV massively
# underestimates one-hot categorical importance. Load old file if available.
_old_gpv_path = os.path.join(output_dir, "importance_global_pv.npy")
_old_gpv_avail = os.path.exists(_old_gpv_path)
if not _old_gpv_avail:
    # Fall back to legacy name
    _legacy = os.path.join(output_dir, "importance_matrix.npy")
    if os.path.exists(_legacy):
        _old_gpv_path  = _legacy
        _old_gpv_avail = True

if _old_gpv_avail:
    _old_gpv = np.load(_old_gpv_path)
    if _old_gpv.shape[-1] != n_feats:
        _old_gpv_avail = False
        print(f"Skipping legacy per-column PV plot: shape {_old_gpv.shape} != n_feats={n_feats}")
if _old_gpv_avail:
    _old_gpv_m = _apply_imp_mask(_old_gpv)
    _old_flat  = _old_gpv_m.reshape(-1, n_feats)
    _old_mean  = np.nanmean(_old_flat, axis=0)
    _old_std   = np.nanstd(_old_flat,  axis=0)
    _order_old = np.argsort(_old_mean)[::-1]

    # Color: orange for one-hot categorical columns, steelblue for continuous
    _cat_cols_set = set(zone_onehot_cols)
    _bar_colors   = ['#e07b3a' if all_feat_cols[i] in _cat_cols_set else 'steelblue'
                     for i in _order_old]

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(n_feats)
    ax.bar(x, _old_mean[_order_old], yerr=_old_std[_order_old],
           color=_bar_colors, capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([all_feat_cols[i] for i in _order_old],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean R² drop (per column)')
    ax.set_title('Appendix — Naive per-column Global PV\n'
                 '(orange = one-hot categorical; blue = continuous)\n'
                 'One-hot columns appear near-zero because categorical importance is '
                 'split across sibling columns')
    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='steelblue', label='Continuous'),
                        Patch(color='#e07b3a',   label='One-hot categorical')],
              fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    savefig('appendix_per_column_pv.png')
    print("Saved appendix_per_column_pv.png")
else:
    print("Old per-column global_pv.npy not found — skipping appendix figure.")

print("\nDone. Outputs saved to:", output_dir)

# ══════════════════════════════════════════════════════════════════════════════
#       SECTION A — TEMPORAL ENCODING EVOLUTION ACROSS SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
# All computation reads existing masked arrays — no model loading, no GPU.
# Treats session_ids index (0…N-1) as recording order.

# Initialise outputs to None so Section C can check safely
_evo_stab_gpv      = None   # mean cross-session cosine similarity, shape (num_neurons,)
_evo_stab_ig       = None
_evo_cluster_labels = None  # Ward k=4, shape (num_neurons,)
_evo_top_feat_idx  = None   # argmax group per session, shape (sessions, neurons)

if RUN_EVOLUTION:
    print("\n" + "="*60)
    print("SECTION A — TEMPORAL ENCODING EVOLUTION")
    print("="*60)

    # Pre-flight: shapes and NaN counts
    print(f"  global_pv_sem_m  {global_pv_sem_m.shape}  "
          f"NaN={np.isnan(global_pv_sem_m).sum()}")
    print(f"  ig_sem_m         {ig_sem_m.shape}  "
          f"NaN={np.isnan(ig_sem_m).sum()}")
    print(f"  mean_r2          {mean_r2.shape}  "
          f"NaN={np.isnan(mean_r2).sum()}")
    print(f"  session ordering: treating index 0..{num_sessions-1} as recording order")
    if not all(str(session_ids[i]) < str(session_ids[i+1])
               for i in range(len(session_ids) - 1)):
        print("  WARNING: session_ids are not in ascending order — "
              "temporal analyses assume index = recording order")

    # ── A.2 Per-ensemble top-feature trajectories ─────────────────────────
    _evo_top_feat_idx = np.full((num_sessions, num_neurons), -1, dtype=np.int16)
    _evo_top_feat_mag = np.full((num_sessions, num_neurons), np.nan)
    for s in range(num_sessions):
        for n in range(num_neurons):
            row = global_pv_sem_m[s, n, :]
            if not np.all(np.isnan(row)):
                _evo_top_feat_idx[s, n] = int(np.nanargmax(row))
                _evo_top_feat_mag[s, n] = float(np.nanmax(row))

    np.save(os.path.join(output_dir, "top_features_per_session.npy"), _evo_top_feat_idx)
    np.save(os.path.join(output_dir, "top_feature_magnitudes.npy"),   _evo_top_feat_mag)
    print("Saved top_features_per_session.npy, top_feature_magnitudes.npy")

    # ── A.3 Encoding stability: pairwise cosine similarity across sessions ─
    # sim[n, s1, s2] = cos_sim( importance_vector(s1,n), importance_vector(s2,n) )
    _evo_sim_gpv = np.full((num_neurons, num_sessions, num_sessions), np.nan)
    _evo_sim_ig  = np.full((num_neurons, num_sessions, num_sessions), np.nan)

    for n in range(num_neurons):
        for s1 in range(num_sessions):
            for s2 in range(s1, num_sessions):
                v1g = global_pv_sem_m[s1, n, :]
                v2g = global_pv_sem_m[s2, n, :]
                v1i = ig_sem_m[s1, n, :]
                v2i = ig_sem_m[s2, n, :]
                for sim_arr, v1, v2 in [(_evo_sim_gpv, v1g, v2g), (_evo_sim_ig, v1i, v2i)]:
                    u1 = np.nan_to_num(v1)
                    u2 = np.nan_to_num(v2)
                    denom = np.linalg.norm(u1) * np.linalg.norm(u2)
                    if denom > 1e-12:
                        cs = float(np.dot(u1, u2) / denom)
                        sim_arr[n, s1, s2] = cs
                        sim_arr[n, s2, s1] = cs

    np.save(os.path.join(output_dir, "session_similarity_global_pv.npy"), _evo_sim_gpv)
    np.save(os.path.join(output_dir, "session_similarity_ig.npy"),         _evo_sim_ig)
    print("Saved session_similarity_global_pv.npy, session_similarity_ig.npy")

    # Mean off-diagonal cosine similarity per neuron (= encoding stability scalar)
    _evo_stab_gpv = np.full(num_neurons, np.nan)
    _evo_stab_ig  = np.full(num_neurons, np.nan)
    _off_diag_mask = ~np.eye(num_sessions, dtype=bool)
    for n in range(num_neurons):
        od_g = _evo_sim_gpv[n][_off_diag_mask]
        od_i = _evo_sim_ig[n][_off_diag_mask]
        if not np.all(np.isnan(od_g)):
            _evo_stab_gpv[n] = float(np.nanmean(od_g))
        if not np.all(np.isnan(od_i)):
            _evo_stab_ig[n]  = float(np.nanmean(od_i))

    # ── A.4 Ensemble signature clustering (Ward k=4) ──────────────────────
    _evo_sig        = np.nanmean(global_pv_sem_m, axis=0)   # (neurons, n_groups)
    _evo_valid_sig  = ~np.all(np.isnan(_evo_sig), axis=1)
    _evo_sig_clean  = np.nan_to_num(_evo_sig[_evo_valid_sig])
    _evo_k          = min(4, int(_evo_valid_sig.sum()))

    _evo_cluster_labels = np.full(num_neurons, -1, dtype=np.int8)
    _evo_cluster_centroids = np.full((_evo_k, n_groups), np.nan)

    if len(_evo_sig_clean) >= _evo_k:
        _Z_ens = linkage(_evo_sig_clean, method='ward')
        _evo_cl_valid = fcluster(_Z_ens, t=_evo_k, criterion='maxclust')
        _evo_cluster_labels[_evo_valid_sig] = _evo_cl_valid
        for k in range(1, _evo_k + 1):
            members = np.where(_evo_cluster_labels == k)[0]
            if len(members) > 0:
                _evo_cluster_centroids[k-1] = np.nanmean(_evo_sig[members], axis=0)

    np.save(os.path.join(output_dir, "ensemble_cluster_labels.npy"),    _evo_cluster_labels)
    np.save(os.path.join(output_dir, "ensemble_cluster_centroids.npy"), _evo_cluster_centroids)
    print(f"Saved ensemble_cluster_labels.npy  (Ward k={_evo_k})")

    # ── A.5 Load per-seed checkpoints for std shading ─────────────────────
    _seed_gpv_stack = []
    for _s in SEEDS:
        _p = os.path.join(output_dir, f"importance_global_pv_semantic_seed{_s}.npy")
        if os.path.exists(_p):
            _seed_gpv_stack.append(_apply_imp_mask(np.load(_p)))
    _evo_gpv_std = (np.nanstd(np.stack(_seed_gpv_stack, axis=0), axis=0)
                    if _seed_gpv_stack else np.zeros_like(global_pv_sem_m))

    # ── A.5b Behavioral performance correlation ────────────────────────────
    if 'reward_window' in behavior_glm_loaded.columns:
        _sess_reward = []
        for _sid in session_ids:
            _sm = behavior_glm_loaded.index.map(lambda x: x[0]) == _sid
            _rw = behavior_glm_loaded.loc[_sm, 'reward_window']
            _sess_reward.append(float((_rw == 1).mean()))
        _sess_reward   = np.array(_sess_reward)
        _sess_mean_r2  = np.nanmean(np.where(mask, np.nan, mean_r2), axis=1)
        _rho_perf, _p_perf = spearmanr(_sess_reward, _sess_mean_r2, nan_policy='omit')
        print(f"  Behavioral performance vs mean R²: "
              f"Spearman ρ={_rho_perf:.3f}, p={_p_perf:.3f}")
    else:
        _sess_reward = None

    # ── FIGURES ───────────────────────────────────────────────────────────

    # Fig A1: top feature per session heatmap (neurons × sessions)
    _vn_top = [n for n in valid_neurons
               if not np.all(_evo_top_feat_idx[:, n] == -1)]
    if _vn_top:
        _hm_top = _evo_top_feat_idx[:, _vn_top].T.astype(float)
        _hm_top[_hm_top < 0] = np.nan
        fig, ax = plt.subplots(
            figsize=(max(8, num_sessions * 0.45), max(4, len(_vn_top) * 0.38)))
        _cmap_top = plt.cm.get_cmap('tab20', n_groups)
        _cmap_top.set_bad(NO_DATA_COLOR)
        _im = ax.imshow(_hm_top, aspect='auto', cmap=_cmap_top,
                        vmin=-0.5, vmax=n_groups - 0.5, interpolation='nearest')
        _cbar = plt.colorbar(_im, ax=ax, ticks=range(n_groups))
        _cbar.set_ticklabels(group_names, fontsize=7)
        ax.set_xlabel('Session (recording order)')
        ax.set_ylabel(f'{unit_label.capitalize()}')
        ax.set_yticks(range(len(_vn_top)))
        ax.set_yticklabels([f"{prefix_name}{n+1:02d}" for n in _vn_top], fontsize=8)
        ax.set_title('Top feature per session (Global PV)\n(color = semantic group)')
        plt.tight_layout()
        savefig('evolution_top_feature.png')
        print("Saved evolution_top_feature.png")

    # Fig A2: encoding stability bar chart
    _vn_stab = [n for n in valid_neurons if not np.isnan(_evo_stab_gpv[n])]
    if _vn_stab:
        _order_stab = sorted(_vn_stab, key=lambda n: _evo_stab_gpv[n], reverse=True)
        fig, ax = plt.subplots(figsize=(max(8, len(_vn_stab) * 0.42), 4))
        _xs = np.arange(len(_vn_stab))
        ax.bar(_xs - 0.2, [_evo_stab_gpv[n] for n in _order_stab], 0.38,
               label='Global PV', color='steelblue', alpha=0.82)
        ax.bar(_xs + 0.2, [_evo_stab_ig[n]  for n in _order_stab], 0.38,
               label='IG',        color='coral',     alpha=0.82)
        ax.axhline(0, color='black', lw=0.7)
        ax.set_xticks(_xs)
        ax.set_xticklabels([f"{prefix_name}{n+1:02d}" for n in _order_stab],
                           rotation=45, ha='right')
        ax.set_ylabel('Mean cross-session cosine similarity')
        ax.set_title('Encoding stability by ensemble\n'
                     '(high = consistent feature tuning across sessions)')
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        savefig('evolution_stability_bar.png')
        print("Saved evolution_stability_bar.png")

    # Fig A3: session pairwise similarity matrix averaged across valid neurons
    _mean_sim_gpv = np.nanmean(_evo_sim_gpv[valid_neurons], axis=0)
    _mean_sim_ig  = np.nanmean(_evo_sim_ig[valid_neurons],  axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for _ax, _mat, _title in zip(axes,
                                  [_mean_sim_gpv, _mean_sim_ig],
                                  ['Global PV',  'IG']):
        _cmap_s = plt.cm.RdYlGn.copy()
        _cmap_s.set_bad(NO_DATA_COLOR)
        _im2 = _ax.imshow(np.ma.masked_invalid(_mat), cmap=_cmap_s, vmin=0, vmax=1,
                          aspect='auto')
        plt.colorbar(_im2, ax=_ax, label='Cosine similarity')
        _ax.set_xlabel('Session')
        _ax.set_ylabel('Session')
        _ax.set_title(f'Mean encoding stability — {_title}')
    plt.tight_layout()
    savefig('evolution_similarity_matrix.png')
    print("Saved evolution_similarity_matrix.png")

    # Fig A4: ensemble signature clusters
    _vn_sig_idx = np.where(_evo_valid_sig)[0]
    if len(_vn_sig_idx) >= 2:
        _cl_labels_valid = _evo_cluster_labels[_evo_valid_sig]
        _order_cl = np.argsort(_cl_labels_valid)
        _sig_hm   = _evo_sig_clean[_order_cl]   # (valid_neurons, n_groups)
        _xlabels_cl = [
            f"{prefix_name}{_vn_sig_idx[i]+1:02d}\nC{_cl_labels_valid[i]}"
            for i in _order_cl
        ]
        fig, ax = plt.subplots(
            figsize=(max(8, len(_vn_sig_idx) * 0.45), max(4, n_groups * 0.42)))
        sns.heatmap(_sig_hm.T, xticklabels=_xlabels_cl, yticklabels=group_names,
                    cmap='Blues', ax=ax, cbar_kws={'label': 'Mean importance'},
                    linewidths=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_title(f'Ensemble signature clusters (Ward k={_evo_k})\n'
                     f'Sorted by cluster; importance averaged across sessions')
        plt.tight_layout()
        savefig('evolution_signature_clusters.png')
        print("Saved evolution_signature_clusters.png")

    # Fig A5 (optional): behavioral performance vs encoding quality
    if _sess_reward is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        _sess_mean_r2_plot = np.nanmean(np.where(mask, np.nan, mean_r2), axis=1)
        ax.scatter(_sess_reward, _sess_mean_r2_plot, alpha=0.75, color='steelblue', s=55)
        ax.set_xlabel('Mean reward rate (session)')
        ax.set_ylabel(f'Mean R² (valid {unit_label_pl})')
        ax.set_title(f'Behavioral performance vs encoding quality\n'
                     f'Spearman ρ={_rho_perf:.3f}, p={_p_perf:.3f}')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        savefig('evolution_performance_correlation.png')
        print("Saved evolution_performance_correlation.png")

    # ── Helper: select diverse, temporally-interesting pairs ─────────────
    def _diverse_interesting_pairs(imp, max_pairs=12, max_per_neuron=2, max_per_group=2,
                                    n_min_above=2, above_thr=0.1):
        """Pick pairs split equally between stable strong encoders (top mean) and
        temporally dynamic pairs (top CV).
        Pairs must have >= n_min_above sessions above above_thr to qualify.
        Diversity caps are shared globally across both phases so dominant groups
        are hard-capped at max_per_group total, not per-phase."""
        mean_ng = np.nanmean(imp, axis=0)          # (neurons, groups)
        std_ng  = np.nanstd(imp,  axis=0)

        # qualify: >= n_min_above sessions clearly above threshold
        n_above_thr = np.array([
            [(~np.isnan(imp[:, ni, gi]) & (imp[:, ni, gi] >= above_thr)).sum()
             for gi in range(n_groups)]
            for ni in range(imp.shape[1])
        ])                                          # (neurons, groups)
        qualifies = n_above_thr >= n_min_above

        cv = np.where((mean_ng > 1e-10) & qualifies, std_ng / (mean_ng + 1e-12), 0.0)
        cv = np.nan_to_num(cv, nan=0.0)
        mean_floored = np.where(qualifies, mean_ng, 0.0)
        mean_floored = np.nan_to_num(mean_floored, nan=0.0)

        n_anchor  = max_pairs // 2
        n_dynamic = max_pairs - n_anchor

        # shared caps — dominant groups cannot exceed max_per_group across both phases
        _cnt_n, _cnt_g = {}, {}

        def _pick(score, budget, exclude=None):
            order = np.argsort(score.ravel())[::-1]
            picked = []
            for fi in order:
                if len(picked) >= budget:
                    break
                ni, gi = fi // n_groups, fi % n_groups
                if score[ni, gi] <= 0:
                    break
                if exclude and (ni, gi) in exclude:
                    continue
                if _cnt_n.get(ni, 0) >= max_per_neuron:
                    continue
                if _cnt_g.get(gi, 0) >= max_per_group:
                    continue
                picked.append((ni, gi))
                _cnt_n[ni] = _cnt_n.get(ni, 0) + 1
                _cnt_g[gi]  = _cnt_g.get(gi,  0) + 1
            return picked

        anchors  = _pick(mean_floored, n_anchor)
        dynamics = _pick(cv, n_dynamic, exclude=set(anchors))
        return anchors + dynamics

    # ── Load per-seed IG for shading ──────────────────────────────────────
    _seed_ig_stack = []
    for _s in SEEDS:
        _pig = os.path.join(output_dir, f"ig_checkpoint_seed{_s}.npz")
        if os.path.exists(_pig):
            _d = np.load(_pig)
            _seed_ig_stack.append(_apply_imp_mask(_d['sem']))
    # _seed_ig_stack entries shape: (sessions, neurons, n_groups)

    _sess_x = np.arange(1, num_sessions + 1)
    _GROUP_COLORS = sns.color_palette("tab20", max(n_groups, 20))

    def _lineplot_pairs(imp, seed_stack, pairs, ylabel, suptitle, fname):
        mean_ng = np.nanmean(imp, axis=0)
        # group pairs by ensemble so each gets its own subplot
        from collections import OrderedDict
        _by_neuron = OrderedDict()
        for _ni, _gi in pairs:
            _by_neuron.setdefault(_ni, []).append(_gi)
        _neurons = list(_by_neuron.keys())
        _ncols = min(3, len(_neurons))
        _nrows = int(np.ceil(len(_neurons) / _ncols))
        # locally-distinct colors per subplot: reassign from a fixed palette
        # so that lines within the same panel are never the same hue
        _distinct = sns.color_palette("tab10", 10)
        fig, axes = plt.subplots(_nrows, _ncols,
                                 figsize=(_ncols * 5.5, _nrows * 3.8),
                                 sharey=True, squeeze=False)
        for _pi, _ni in enumerate(_neurons):
            _ax = axes[_pi // _ncols][_pi % _ncols]
            for _li, _gi in enumerate(_by_neuron[_ni]):
                _series = imp[:, _ni, _gi]
                # global color by group for cross-panel consistency,
                # but override with a locally-distinct color if groups in
                # this panel would otherwise share a hue
                _col = _GROUP_COLORS[_gi % len(_GROUP_COLORS)]
                _local_cols_used = [_GROUP_COLORS[g % len(_GROUP_COLORS)]
                                    for g in _by_neuron[_ni][:_li]]
                if _col in _local_cols_used:
                    _col = _distinct[_li % len(_distinct)]
                if seed_stack:
                    _ss = np.stack([_sv[:, _ni, _gi] for _sv in seed_stack], axis=0)
                    _lo = np.nanpercentile(_ss, 25, axis=0)
                    _hi = np.nanpercentile(_ss, 75, axis=0)
                    _ok = ~(np.isnan(_lo) | np.isnan(_hi))
                    if _ok.any():
                        _ax.fill_between(_sess_x[_ok], _lo[_ok], _hi[_ok],
                                         alpha=0.20, color=_col)
                _ax.plot(_sess_x, _series, '-o', color=_col, markersize=4, lw=1.8,
                         label=f"{group_names[_gi]}  (μ={mean_ng[_ni, _gi]:.3f})")
            _ax.axhline(0, color='grey', linestyle='--', lw=0.7)
            _ax.set_title(f"{prefix_name}{_ni+1:02d}", fontsize=11, fontweight='bold')
            _ax.set_xticks(_sess_x)
            _ax.set_xticklabels([f"S{s}" for s in _sess_x],
                                rotation=60, ha='right', fontsize=7)
            _ax.set_ylabel(ylabel, fontsize=9)
            _ax.legend(fontsize=8, loc='upper right', framealpha=0.7)
            _ax.spines[['top', 'right']].set_visible(False)
        # hide unused subplots
        for _pi in range(len(_neurons), _nrows * _ncols):
            axes[_pi // _ncols][_pi % _ncols].set_visible(False)
        fig.suptitle(suptitle, fontsize=12, y=1.01)
        plt.tight_layout()
        savefig(fname)
        print(f"Saved {fname}")

    _pairs_gpv = _diverse_interesting_pairs(global_pv_sem_m)
    _pairs_ig  = _diverse_interesting_pairs(ig_sem_m)

    _lineplot_pairs(global_pv_sem_m, _seed_gpv_stack, _pairs_gpv,
                    'Permutation importance (R² drop)',
                    'Feature Importance Evolution — Global PV\n'
                    'Diverse interesting pairs (high temporal variability, noise-floored)\n'
                    'Shaded band = IQR across seeds',
                    'evolution_lineplot_gpv.png')

    _lineplot_pairs(ig_sem_m, _seed_ig_stack, _pairs_ig,
                    'Mean |IG| attribution',
                    'Feature Importance Evolution — Integrated Gradients\n'
                    'Diverse interesting pairs (high temporal variability, noise-floored)',
                    'evolution_lineplot_ig.png')

    # ── Fig A7: Per-top-ensemble heatmap, both metrics, per-panel scale ───
    # vmax is set per-neuron so the full dynamic range is visible.
    # Values < 2× median across all valid pairs are treated as noise floor.
    TOP_K_EVO_HM = 6
    _nm_max      = np.nanmax(np.where(mask, np.nan, mean_r2), axis=0)

    # Select top ensembles by composite: peak R² × encoding breadth.
    # Breadth = number of feature groups whose mean importance exceeds the
    # per-group noise floor — rewards ensembles encoding multiple features.
    # Require at least MIN_EVO_SESSIONS valid sessions so a single lucky
    # session can't inflate breadth and produce a nearly-empty heatmap.
    MIN_EVO_SESSIONS = 5
    _gpv_n_valid  = (~np.isnan(global_pv_sem_m[:, :, 0])).sum(axis=0)  # (ensembles,)
    _gpv_mean_ens = np.nanmean(global_pv_sem_m, axis=0)   # (ensembles, groups)
    _gpv_floor_g_pre = 2.0 * np.nanmedian(
        global_pv_sem_m.reshape(-1, n_groups), axis=0)
    _breadth = ((_gpv_mean_ens > _gpv_floor_g_pre[np.newaxis, :]) &
                (_gpv_n_valid >= MIN_EVO_SESSIONS)[:, np.newaxis]
                ).sum(axis=1).astype(float)           # (ensembles,)
    _qualifies_evo = _gpv_n_valid >= MIN_EVO_SESSIONS
    _evo_score   = (np.nan_to_num(_nm_max, nan=-1) * np.maximum(_breadth, 1)
                    * _qualifies_evo)
    _top_n_evo   = np.argsort(_evo_score)[::-1][:TOP_K_EVO_HM]

    # noise floor per group — any value below this is effectively noise
    _gpv_floor_g  = 2.0 * np.nanmedian(
        global_pv_sem_m.reshape(-1, n_groups), axis=0)  # (n_groups,)
    _ig_floor_g   = 2.0 * np.nanmedian(
        ig_sem_m.reshape(-1, n_groups), axis=0)
    _cond_floor_g = 2.0 * np.nanmedian(
        cond_pv_sem_m.reshape(-1, n_groups), axis=0)

    def _neuron_heatmaps(imp, floor_g, top_neurons, ylabel, suptitle, fname):
        _ncols = min(3, len(top_neurons))
        _nrows = (len(top_neurons) + _ncols - 1) // _ncols
        fig, axes = plt.subplots(
            _nrows, _ncols,
            figsize=(_ncols * 7, _nrows * max(3.5, n_groups * 0.35)),
            squeeze=False)
        _af    = axes.flatten()
        _cmap  = plt.cm.Blues.copy()
        _cmap.set_bad(NO_DATA_COLOR)

        # Shared vmax across all panels in this figure
        _all_valid = np.concatenate([
            imp[:, _n, :][~np.isnan(imp[:, _n, :])].ravel()
            for _n in top_neurons
        ]) if len(top_neurons) > 0 else np.array([0.01])
        _floor_vmax = float(np.nanmax(floor_g)) * 3
        _vmax = max(float(np.nanpercentile(_all_valid, 98)) if len(_all_valid) > 0 else 0.01,
                    _floor_vmax, 0.01)

        for _k, _n_idx in enumerate(top_neurons):
            _ax   = _af[_k]
            _data = imp[:, _n_idx, :].T.copy()   # (n_groups, sessions)

            sns.heatmap(_data, ax=_ax, cmap=_cmap, vmin=0, vmax=_vmax,
                        xticklabels=[f"S{s+1}" for s in range(num_sessions)],
                        yticklabels=group_names,
                        cbar_kws={'label': f'(scale max={_vmax:.3f})'})
            for _j in range(num_sessions):
                if np.all(np.isnan(_data[:, _j])):
                    _ax.add_patch(plt.Rectangle(
                        [_j, 0], 1, n_groups, fill=True,
                        facecolor=NO_DATA_COLOR,
                        hatch='////', edgecolor='#888888', lw=0.5, zorder=2))
            _ax.set_xticklabels(_ax.get_xticklabels(),
                                rotation=45, ha='right', fontsize=8)
            _ax.set_yticklabels(_ax.get_yticklabels(), fontsize=9)
            _ax.set_title(
                f"{prefix_name}{_n_idx+1:02d} — max R²={_nm_max[_n_idx]:.3f}",
                fontsize=12)
            _ax.set_xlabel('Session', fontsize=10)

        for _k in range(len(top_neurons), len(_af)):
            _af[_k].set_visible(False)

        plt.suptitle(suptitle, fontsize=13, y=1.01)
        plt.tight_layout()
        savefig(fname)
        print(f"Saved {fname}")

    _neuron_heatmaps(
        global_pv_sem_m, _gpv_floor_g, _top_n_evo,
        'R² drop',
        f'Global PV Evolution — Top {TOP_K_EVO_HM} {unit_label_pl.capitalize()}\n'
        f'(shared color scale = 98th pctile across panels; grey = R²<{R2_THRESHOLD})',
        'evolution_heatmap_gpv.png')

    _neuron_heatmaps(
        ig_sem_m, _ig_floor_g, _top_n_evo,
        '|IG|',
        f'IG Evolution — Top {TOP_K_EVO_HM} {unit_label_pl.capitalize()}\n'
        f'(shared color scale = 98th pctile across panels; grey = R²<{R2_THRESHOLD})',
        'evolution_heatmap_ig.png')

    _neuron_heatmaps(
        cond_pv_sem_m, _cond_floor_g, _top_n_evo,
        'R² drop (cond)',
        f'Cond-PV Evolution — Top {TOP_K_EVO_HM} {unit_label_pl.capitalize()}\n'
        f'(shared color scale = 98th pctile across panels; grey = R²<{R2_THRESHOLD})',
        'evolution_heatmap_cond_pv.png')

# ══════════════════════════════════════════════════════════════════════════════
#       SECTION B — AUTOMATED DISCOVERY OF INTERESTING NEURON-FEATURE PAIRS
# ══════════════════════════════════════════════════════════════════════════════
# Reads global_pv_sem_m and ig_sem_m. No model loading.

# Initialise to None so Section C can check safely
_disc_trend_gpv  = None
_disc_disagree   = None
_disc_comp_gpv   = None

if RUN_DISCOVERY:
    print("\n" + "="*60)
    print("SECTION B — AUTOMATED DISCOVERY")
    print("="*60)
    print(f"  global_pv_sem_m  {global_pv_sem_m.shape}  "
          f"NaN={np.isnan(global_pv_sem_m).sum()}")
    print(f"  ig_sem_m         {ig_sem_m.shape}  "
          f"NaN={np.isnan(ig_sem_m).sum()}")

    # ── B.1 Per-(neuron, group) metrics ───────────────────────────────────

    def _compute_ng_metrics(imp):
        """imp: (sessions, neurons, groups)
        Returns trend (Spearman ρ), step (max consecutive diff), cv (coeff variation).
        All outputs shape (neurons, groups), NaN where < 3 valid sessions.
        """
        n_s, n_n, n_g = imp.shape
        trend = np.full((n_n, n_g), np.nan)
        step  = np.full((n_n, n_g), np.nan)
        cv    = np.full((n_n, n_g), np.nan)
        _idx  = np.arange(n_s)
        for n in range(n_n):
            for g in range(n_g):
                v = imp[:, n, g]
                ok = ~np.isnan(v)
                if ok.sum() < 3:
                    continue
                v_ok = v[ok]
                s_ok = _idx[ok]
                rho, _ = spearmanr(s_ok, v_ok)
                trend[n, g] = rho
                diffs = np.abs(np.diff(v_ok))
                if len(diffs) > 0:
                    step[n, g] = float(diffs.max())
                _m = float(np.mean(v_ok))
                _s = float(np.std(v_ok))
                if _m > 1e-10:
                    cv[n, g] = _s / _m
        return trend, step, cv

    _disc_trend_gpv, _disc_step_gpv, _disc_cv_gpv = _compute_ng_metrics(global_pv_sem_m)
    _disc_trend_ig,  _disc_step_ig,  _disc_cv_ig  = _compute_ng_metrics(ig_sem_m)

    # Noise floor: 2× median importance across valid (neuron, group) pairs
    _gpv_floor = 2.0 * np.nanmedian(
        global_pv_sem_m.reshape(-1, n_groups), axis=0)  # (n_groups,)
    _ig_floor  = 2.0 * np.nanmedian(
        ig_sem_m.reshape(-1, n_groups), axis=0)

    _gpv_mean_ng   = np.nanmean(global_pv_sem_m, axis=0)   # (neurons, groups)
    _ig_mean_ng    = np.nanmean(ig_sem_m,        axis=0)

    _gpv_above     = _gpv_mean_ng > _gpv_floor[np.newaxis, :]
    _ig_above      = _ig_mean_ng  > _ig_floor[np.newaxis, :]

    # Cross-method disagreement: abs diff of z-scored mean importance
    def _row_zscore(arr):
        _m = np.nanmean(arr)
        _s = np.nanstd(arr)
        return (arr - _m) / (_s + 1e-10)

    _disc_disagree = np.abs(_row_zscore(_gpv_mean_ng) - _row_zscore(_ig_mean_ng))

    # Composite interestingness: z(|trend|) + z(step) + z(disagree)
    def _global_zscore(arr):
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return np.zeros_like(arr)
        return (arr - float(np.nanmean(finite))) / (float(np.nanstd(finite)) + 1e-10)

    _disc_comp_gpv = (_global_zscore(np.abs(_disc_trend_gpv))
                      + _global_zscore(_disc_step_gpv)
                      + _global_zscore(_disc_disagree))
    _disc_comp_ig  = (_global_zscore(np.abs(_disc_trend_ig))
                      + _global_zscore(_disc_step_ig)
                      + _global_zscore(_disc_disagree))

    # Apply noise floor (below-floor pairs → NaN so they don't rank)
    _disc_comp_gpv = np.where(_gpv_above, _disc_comp_gpv, np.nan)
    _disc_comp_ig  = np.where(_ig_above,  _disc_comp_ig,  np.nan)

    # ── B.2 interesting_pairs.csv ─────────────────────────────────────────
    _TOP_INT = 20

    def _top_pairs_df(comp, method, mean_ng, trend, step, disagree):
        rows = []
        flat_order = np.argsort(np.nan_to_num(comp, nan=-1e9).ravel())[::-1]
        for fi in flat_order:
            if len(rows) >= _TOP_INT:
                break
            n_i = fi // n_groups
            g_i = fi %  n_groups
            v   = comp[n_i, g_i]
            if not np.isfinite(v):
                continue
            rows.append({
                'method':     method,
                'ensemble_id':  f"{prefix_name}{n_i+1:02d}",
                'neuron_idx': int(n_i),
                'group':      group_names[g_i],
                'group_idx':  int(g_i),
                'mean_imp':   round(float(mean_ng[n_i, g_i]), 6),
                'trend_rho':  round(float(trend[n_i, g_i]),   4)
                              if np.isfinite(trend[n_i, g_i]) else np.nan,
                'max_step':   round(float(step[n_i, g_i]),    6)
                              if np.isfinite(step[n_i, g_i])  else np.nan,
                'disagree':   round(float(disagree[n_i, g_i]), 4),
                'composite':  round(float(v), 4),
            })
        return pd.DataFrame(rows)

    _df_gpv = _top_pairs_df(_disc_comp_gpv, 'global_pv',
                            _gpv_mean_ng, _disc_trend_gpv, _disc_step_gpv, _disc_disagree)
    _df_ig  = _top_pairs_df(_disc_comp_ig,  'ig',
                            _ig_mean_ng,  _disc_trend_ig,  _disc_step_ig,  _disc_disagree)
    _df_int = pd.concat([_df_gpv, _df_ig], ignore_index=True)
    _df_int = _df_int.sort_values('composite', ascending=False).reset_index(drop=True)
    _df_int.to_csv(os.path.join(output_dir, "interesting_pairs.csv"), index=False)
    print(f"Saved interesting_pairs.csv ({len(_df_int)} rows)")
    if len(_df_int) > 0:
        print(_df_int[['method','ensemble_id','group','trend_rho','composite']].head(10).to_string(
            index=False))

    # ── FIGURES ───────────────────────────────────────────────────────────
    _hm_xticks = [f"{prefix_name}{i+1:02d}" for i in range(num_neurons)]

    for _sfx, _trend, _step, _comp, _mname in [
        ('gpv', _disc_trend_gpv, _disc_step_gpv, _disc_comp_gpv, 'Global PV'),
        ('ig',  _disc_trend_ig,  _disc_step_ig,  _disc_comp_ig,  'IG'),
    ]:
        _fw = max(8, num_neurons * 0.42)
        _fh = max(4, n_groups * 0.42)

        # Trend
        fig, ax = plt.subplots(figsize=(_fw, _fh))
        sns.heatmap(_trend.T, xticklabels=_hm_xticks, yticklabels=group_names,
                    cmap='RdBu_r', center=0, ax=ax,
                    cbar_kws={'label': 'Spearman ρ'},
                    mask=np.isnan(_trend.T))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_title(f'Session trend (Spearman ρ) — {_mname}\n'
                     f'red=increasing, blue=decreasing across sessions')
        plt.tight_layout()
        savefig(f'discovery_trend_{_sfx}.png')

        # Max step
        fig, ax = plt.subplots(figsize=(_fw, _fh))
        sns.heatmap(_step.T, xticklabels=_hm_xticks, yticklabels=group_names,
                    cmap='YlOrRd', ax=ax,
                    cbar_kws={'label': 'Max consecutive step'},
                    mask=np.isnan(_step.T))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_title(f'Max consecutive step — {_mname}')
        plt.tight_layout()
        savefig(f'discovery_step_{_sfx}.png')

        # Composite score
        fig, ax = plt.subplots(figsize=(_fw, _fh))
        sns.heatmap(_comp.T, xticklabels=_hm_xticks, yticklabels=group_names,
                    cmap='viridis', ax=ax,
                    cbar_kws={'label': 'Composite score'},
                    mask=np.isnan(_comp.T))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_title(f'Interestingness composite score — {_mname}\n'
                     f'z(|trend|) + z(step) + z(disagree)  [noise-floored]')
        plt.tight_layout()
        savefig(f'discovery_composite_{_sfx}.png')

        print(f"Saved 3 discovery figures for {_mname}")

    # Cross-method disagreement panel
    fig, ax = plt.subplots(figsize=(max(8, num_neurons * 0.42), max(4, n_groups * 0.42)))
    sns.heatmap(_disc_disagree.T, xticklabels=_hm_xticks, yticklabels=group_names,
                cmap='Oranges', ax=ax,
                cbar_kws={'label': '|z(GPV) − z(IG)|'},
                mask=np.isnan(_disc_disagree.T))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_title('Cross-method disagreement (Global PV vs IG)\n'
                 'High = methods disagree on feature importance')
    plt.tight_layout()
    savefig('discovery_disagreement.png')
    print("Saved discovery_disagreement.png")

# ══════════════════════════════════════════════════════════════════════════════
#       SECTION C — PER-ENSEMBLE NARRATIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
# Always runs. Reads variables from earlier sections if available (else NaN/N/A).

print("\n" + "="*60)
print("SECTION C — PER-ENSEMBLE NARRATIVE SUMMARY")
print("="*60)

_summary_rows = []
for _n in range(num_neurons):
    _sess_ok      = ~mask[:, _n] & ~low_r2_session_mask[:, _n]
    _valid_n      = bool(_sess_ok.any())
    _r2_vals      = mean_r2[:, _n]
    _r2_masked    = _r2_vals[_sess_ok]
    _max_r2       = float(np.nanmax(_r2_masked))   if len(_r2_masked) > 0 else np.nan
    _mean_r2_n    = float(np.nanmean(_r2_masked))  if len(_r2_masked) > 0 else np.nan
    _n_valid_sess = int(_sess_ok.sum())

    # Top feature by Global PV (mean over sessions)
    _gpv_mn = np.nanmean(global_pv_sem_m[:, _n, :], axis=0)
    _top_gpv_g = int(np.nanargmax(_gpv_mn)) if not np.all(np.isnan(_gpv_mn)) else -1
    _top_gpv   = group_names[_top_gpv_g]   if _top_gpv_g >= 0 else 'N/A'

    # Top feature by IG
    _ig_mn = np.nanmean(ig_sem_m[:, _n, :], axis=0)
    _top_ig_g = int(np.nanargmax(_ig_mn)) if not np.all(np.isnan(_ig_mn)) else -1
    _top_ig   = group_names[_top_ig_g]   if _top_ig_g >= 0 else 'N/A'

    # Encoding stability (from Section A)
    _stab = (float(_evo_stab_gpv[_n])
             if _evo_stab_gpv is not None and not np.isnan(_evo_stab_gpv[_n])
             else np.nan)

    # Cluster (from Section A)
    _cluster = (int(_evo_cluster_labels[_n])
                if _evo_cluster_labels is not None and _evo_cluster_labels[_n] >= 0
                else -1)

    # Most dynamically-tuned feature (from Section B)
    _most_dyn  = 'N/A'
    _max_trend = np.nan
    if _disc_trend_gpv is not None:
        _tr = _disc_trend_gpv[_n, :]
        if not np.all(np.isnan(_tr)):
            _g_dyn     = int(np.nanargmax(np.abs(_tr)))
            _most_dyn  = group_names[_g_dyn]
            _max_trend = round(float(np.abs(_tr[_g_dyn])), 4)

    # Cross-method agreement on top feature
    _agree = (_top_gpv_g >= 0 and _top_ig_g >= 0 and _top_gpv_g == _top_ig_g)

    # Disagreement magnitude (from Section B, mean over groups)
    _disagree_mean = (float(np.nanmean(_disc_disagree[_n]))
                      if _disc_disagree is not None else np.nan)

    _summary_rows.append({
        'ensemble_id':             f"{prefix_name}{_n+1:02d}",
        'neuron_idx':            _n,
        'is_valid':              _valid_n,
        'n_valid_sessions':      _n_valid_sess,
        'max_r2':                round(_max_r2,    4) if not np.isnan(_max_r2)    else np.nan,
        'mean_r2':               round(_mean_r2_n, 4) if not np.isnan(_mean_r2_n) else np.nan,
        'top_feature_global_pv': _top_gpv,
        'top_feature_ig':        _top_ig,
        'methods_agree':         _agree,
        'cross_method_disagree': round(_disagree_mean, 4) if not np.isnan(_disagree_mean) else np.nan,
        'encoding_stability':    round(_stab, 4) if not np.isnan(_stab) else np.nan,
        'cluster':               _cluster,
        'most_dynamic_feature':  _most_dyn,
        'max_trend_rho':         _max_trend,
    })

_df_summary = pd.DataFrame(_summary_rows)
_df_summary.to_csv(os.path.join(output_dir, "ensemble_summary.csv"), index=False)
print(f"Saved ensemble_summary.csv  ({len(_df_summary)} rows)")
_print_cols = ['ensemble_id', 'max_r2', 'top_feature_global_pv', 'top_feature_ig',
               'methods_agree', 'encoding_stability', 'cluster', 'most_dynamic_feature']
print(_df_summary[_print_cols].to_string(index=False))

print("\nAll sections complete. Outputs in:", output_dir)
