#!/usr/bin/env python3
"""
head_angle_analysis.py

Deep dive into why head_angle is the dominant attribution feature across MLP ensembles.

Sections:
  1. Head angle characterization — distribution, temporal structure, correlation
     with other behavioral variables across sessions
  2. Head angle - ensemble attribution ranking — which ensembles are most
     head-angle tuned, per method (GPV, IG, CPV)
  3. Head angle - ensemble scatter showcase — for top ensembles, visualize
     ensemble activity vs head_angle at the per-session level
  4. Zero-ablation test — run existing MLP with only head_angle feature active
     (all others zeroed): R² compared to full model
  5. Single-feature MLP training — train a small MLP using only head_angle
     as input for top (session, ensemble) pairs; compare R² to full model
  6. Confound analysis — partial correlation: does head_angle's strength
     survive partialling out movement_energy and track_zone?
"""
import os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from models import MLP

# ═══════════════════════════════════ CONFIG ═══════════════════════════════════
USE_ENSEMBLES   = True
SEEDS           = [42, 43, 44, 45, 46]
R2_THRESHOLD    = 0.01
TOP_SCATTER     = 9       # top (session, ensemble) pairs for scatter showcase
TOP_ABLATION    = 5       # pairs for ablation/training analysis
N_SINGLE_SEEDS  = 3       # replicate seeds for single-feature MLP training
SINGLE_LR       = 3e-3    # learning rate for single-feature MLP
SINGLE_EPOCHS   = 400     # training epochs for single-feature MLP

RUN_CHAR        = True    # head angle characterization
RUN_RANKING     = True    # attribution ranking
RUN_SCATTER     = True    # scatter showcase (no model loading)
RUN_ABLATION    = True    # zero-ablation test (loads models)
RUN_SINGLEFEAT  = True    # train single-feature MLPs (slow; trains new models)
RUN_CONFOUND    = True    # confound / partial correlation analysis

mode_str     = "ensembles" if USE_ENSEMBLES else "spikes"
prefix_name  = "E"         if USE_ENSEMBLES else "U"
models_root  = f"./models/mlps/{mode_str}"
splits_dir   = "./splits"
attr_dir     = f"./outputs/mlps/{mode_str}_multiseed"

ts           = datetime.now().strftime('%Y%m%d_%H%M')
output_dir   = f"./outputs/mlps/head_angle_analysis_{ts}"
desktop_dir  = f"/mnt/c/Users/amits/Desktop/head_angle_analysis_{ts}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(desktop_dir, exist_ok=True)

print(f"Output: {output_dir}")
print(f"Desktop: {desktop_dir}")


def savefig(name):
    for d in (output_dir, desktop_dir):
        plt.savefig(os.path.join(d, name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {name}")


# ══════════════════════════════ LOAD RAW DATA ═════════════════════════════════
base             = "./outputs/glm_input_data/"
beh_vals         = np.load(os.path.join(base, "behavior_glm_input.npy"),         allow_pickle=True)
beh_idx          = np.load(os.path.join(base, "behavior_glm_input_index.npy"),   allow_pickle=True)
beh_cols         = np.load(os.path.join(base, "behavior_glm_input_columns.npy"), allow_pickle=True)
spk_vals         = np.load(os.path.join(base, "fr_full.npy"))
spk_idx          = np.load(os.path.join(base, "fr_full_index.npy"),              allow_pickle=True)
ensembles_values = np.load(os.path.join(base, "ensembles.npy"))

behavior_glm_loaded = pd.DataFrame(beh_vals, index=pd.Index(beh_idx), columns=beh_cols)
spk_cols            = np.load(os.path.join(base, "fr_full_columns.npy"), allow_pickle=True)
spikes_loaded       = pd.DataFrame(spk_vals, index=pd.Index(spk_idx), columns=spk_cols)

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
# Keep original head_angle BEFORE dropping track_zone
ha_raw_col   = behavior_glm_loaded['head_angle'].values.astype(float)
ha_vel_col   = behavior_glm_loaded['head_angle_vel'].values.astype(float)
session_col  = behavior_glm_loaded.index.map(lambda x: x[0]).values
behavior_glm_loaded = behavior_glm_loaded.drop("track_zone", axis=1)
session_ids  = behavior_glm_loaded.index.map(lambda x: x[0]).unique()

# ── Feature columns ───────────────────────────────────────────────────────────
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

# ── Semantic groups ───────────────────────────────────────────────────────────
semantic_groups: list = []
for col in non_categorical_cols:
    semantic_groups.append((col, [all_feat_cols.index(col)]))
for cat_var in categorical_variables:
    col_indices = [i for i, c in enumerate(all_feat_cols) if c.startswith(f'{cat_var}_')]
    if col_indices:
        semantic_groups.append((cat_var, col_indices))

n_groups    = len(semantic_groups)
group_names = [g[0] for g in semantic_groups]

# Feature indices of interest
ha_g_idx  = group_names.index('head_angle')
ha_f_idx  = all_feat_cols.index('head_angle')
hav_f_idx = all_feat_cols.index('head_angle_vel')
me_f_idx  = all_feat_cols.index('movement_energy_smooth5')
tz_cols   = [i for i, c in enumerate(all_feat_cols) if c.startswith('track_zone_int_')]

# ── Session dataset (cached) ──────────────────────────────────────────────────
cache_path = f"./outputs/session_dataset_{mode_str}.pkl"
with open(cache_path, 'rb') as f:
    session_dataset_singles = pickle.load(f)
session_ids   = pd.Index(list(session_dataset_singles.keys()))
num_sessions  = len(session_ids)
num_ensembles = len(next(iter(session_dataset_singles.values()))['label_stds'])
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size    = n_feats
hidden_size   = 64
num_hidden_layers = 2
output_size   = 1

print(f"sessions={num_sessions}  ensembles={num_ensembles}  groups={n_groups}  device={device}")

# ── Precomputed attributions ──────────────────────────────────────────────────
all_r2 = np.load(os.path.join(attr_dir, "all_r2.npy"))           # (seeds, sess, ens)
gpv    = np.load(os.path.join(attr_dir, "importance_global_pv_semantic.npy"))
ig_mat = np.load(os.path.join(attr_dir, "importance_ig_semantic.npy"))
cpv    = np.load(os.path.join(attr_dir, "importance_cond_pv_semantic.npy"))

mask_3d     = np.isnan(all_r2)
mask        = np.any(mask_3d, axis=0)
mean_r2     = np.nanmean(np.where(mask_3d, np.nan, all_r2.astype(float)), axis=0)
low_r2_mask = mean_r2 < R2_THRESHOLD
valid_mask  = ~mask & ~low_r2_mask

def _apply_imp_mask(arr):
    return np.where((mask | low_r2_mask)[:, :, np.newaxis], np.nan, arr)

gpv_m  = _apply_imp_mask(gpv)
ig_m   = _apply_imp_mask(ig_mat)
cpv_m  = _apply_imp_mask(cpv)
max_r2_ens = np.nanmax(np.where(mask, np.nan, mean_r2), axis=0)

# ── Model helpers ──────────────────────────────────────────────────────────────
_test_idx_map = np.load(os.path.join(splits_dir, "split_seed42.npy"),
                        allow_pickle=True).item()

def _test_arrays(sess):
    sd     = session_dataset_singles[sess]
    trials = _test_idx_map[sess]
    X = np.concatenate([sd['data'][t]   for t in trials], axis=0).astype(np.float32)
    Y = np.concatenate([sd['labels'][t] for t in trials], axis=0).astype(np.float32)
    return X, Y

def _load_model(s_idx, n_idx, seed=42):
    mpath = os.path.join(models_root, f"seed{seed}",
                         f"session_{s_idx:02d}_neuron_{n_idx:02d}.pt")
    if not os.path.exists(mpath):
        return None
    m = MLP(input_size, hidden_size, num_hidden_layers, output_size).to(device)
    m.load_state_dict(torch.load(mpath, map_location=device))
    m.eval()
    return m

print(f"Valid pairs: {valid_mask.sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: HEAD ANGLE CHARACTERIZATION
# ══════════════════════════════════════════════════════════════════════════════

if RUN_CHAR:
    print("\n" + "="*60)
    print("SECTION 1 — HEAD ANGLE CHARACTERIZATION")
    print("="*60)

    # Per-session head_angle statistics
    sess_ha_stats = []
    for sid in session_ids:
        sm   = session_col == sid
        ha_s = ha_raw_col[sm]
        sess_ha_stats.append({
            'session_id': sid,
            'mean':    float(np.nanmean(ha_s)),
            'std':     float(np.nanstd(ha_s)),
            'range':   float(np.nanmax(ha_s) - np.nanmin(ha_s)),
            'p5':      float(np.nanpercentile(ha_s, 5)),
            'p95':     float(np.nanpercentile(ha_s, 95)),
        })
    df_ha = pd.DataFrame(sess_ha_stats)
    df_ha.to_csv(os.path.join(output_dir, 'head_angle_session_stats.csv'), index=False)
    print(df_ha[['session_id', 'mean', 'std', 'range']].to_string(index=False))

    # Figure 1a: violin/box plot of head_angle distribution per session
    ha_data   = []
    sess_labs = []
    for _i, sid in enumerate(session_ids):
        sm   = session_col == sid
        ha_s = ha_raw_col[sm]
        ha_data.append(ha_s)
        sess_labs.append(f"S{_i+1}")

    fig, ax = plt.subplots(figsize=(max(10, num_sessions * 0.6), 4))
    ax.violinplot(ha_data, positions=range(num_sessions), showmedians=True, widths=0.7)
    ax.set_xticks(range(num_sessions))
    ax.set_xticklabels(sess_labs, fontsize=8, rotation=45)
    ax.set_ylabel('Head angle (raw units)', fontsize=10)
    ax.set_title('Head Angle Distribution per Session\n'
                 '(each violin = full session; horizontal line = median)', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    savefig('ha_distribution_per_session.png')

    # Figure 1b: correlation of head_angle with other behavioral vars (global)
    ha_all   = ha_raw_col[~np.isnan(ha_raw_col)]
    corr_vars = {}
    for col in non_categorical_cols:
        if col in behavior_glm_loaded.columns:
            x = behavior_glm_loaded[col].values.astype(float)
            ok = ~(np.isnan(ha_raw_col) | np.isnan(x))
            if ok.sum() > 100:
                rho, p = spearmanr(ha_raw_col[ok], x[ok])
                corr_vars[col] = (rho, p)
    # Also check categorical vars (one-hot columns)
    for cat_var in categorical_variables:
        cat_c = f'{cat_var}'
        if cat_c in behavior_glm_loaded.columns:
            x = behavior_glm_loaded[cat_c].values.astype(float)
            ok = ~(np.isnan(ha_raw_col) | np.isnan(x))
            if ok.sum() > 100:
                rho, p = spearmanr(ha_raw_col[ok], x[ok])
                corr_vars[cat_c] = (rho, p)

    print("\nHead angle correlation with other behavioral variables:")
    for k, (rho, p) in sorted(corr_vars.items(), key=lambda x: abs(x[1][0]), reverse=True):
        print(f"  {k:45s}  ρ={rho:.3f}  p={p:.4f}")

    # Figure 1c: correlation bar chart
    fig, ax = plt.subplots(figsize=(max(8, len(corr_vars) * 0.6), 4))
    _names = list(corr_vars.keys())
    _rhos  = [corr_vars[n][0] for n in _names]
    _pvals = [corr_vars[n][1] for n in _names]
    _colors = ['steelblue' if r >= 0 else 'coral' for r in _rhos]
    ax.bar(range(len(_names)), _rhos, color=_colors, alpha=0.85)
    ax.axhline(0, color='black', lw=0.7)
    for _i, (r, p) in enumerate(zip(_rhos, _pvals)):
        if p < 0.001:
            ax.text(_i, r + 0.01 * np.sign(r), '***', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(_names)))
    ax.set_xticklabels([n[:20] for n in _names], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Spearman ρ with head_angle', fontsize=10)
    ax.set_title('Head Angle Correlations with Other Behavioral Variables\n'
                 '(global, all sessions pooled)', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    savefig('ha_correlations.png')

    # Figure 1d: temporal autocorrelation of head_angle within a session
    _sid0   = session_ids[0]
    _sm0    = session_col == _sid0
    _ha0    = ha_raw_col[_sm0]
    _ha0    = _ha0[~np.isnan(_ha0)]
    _maxlag = min(100, len(_ha0) // 4)
    _acf    = [float(np.corrcoef(_ha0[:-k], _ha0[k:])[0, 1]) if k > 0 else 1.0
               for k in range(_maxlag)]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(range(_maxlag), _acf, color='steelblue', lw=1.5)
    ax.axhline(0, color='black', lw=0.7)
    ax.axhline(0.1,  color='grey', lw=0.7, linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag (timepoints)', fontsize=10)
    ax.set_ylabel('Autocorrelation', fontsize=10)
    ax.set_title(f'Head Angle Temporal Autocorrelation (Session 1, n={len(_ha0)} pts)\n'
                 f'High autocorrelation = slow-varying signal', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    savefig('ha_autocorrelation.png')

    # Figure 1e: head_angle range vs mean R² per session
    _sess_mean_r2 = np.nanmean(np.where(mask, np.nan, mean_r2), axis=1)  # (sessions,)
    _sess_ha_range = df_ha['range'].values
    _rho_r2_ha, _p_r2_ha = spearmanr(_sess_ha_range, _sess_mean_r2, nan_policy='omit')
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(_sess_ha_range, _sess_mean_r2, alpha=0.75, color='steelblue', s=55)
    for _i, (_r, _m) in enumerate(zip(_sess_ha_range, _sess_mean_r2)):
        ax.annotate(f'S{_i+1}', (_r, _m), fontsize=7, ha='left', va='bottom')
    ax.set_xlabel('Head angle range per session (90th pctile)', fontsize=9)
    ax.set_ylabel('Mean R² across valid ensembles', fontsize=9)
    ax.set_title(f'Head Angle Range vs Encoding Quality\nSpearman ρ={_rho_r2_ha:.3f}  p={_p_r2_ha:.3f}',
                 fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    savefig('ha_range_vs_r2.png')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: HEAD ANGLE - ENSEMBLE ATTRIBUTION RANKING
# ══════════════════════════════════════════════════════════════════════════════

if RUN_RANKING:
    print("\n" + "="*60)
    print("SECTION 2 — HEAD ANGLE ATTRIBUTION RANKING")
    print("="*60)

    # For each ensemble: mean head_angle attribution per method, averaged across valid sessions
    ha_gpv = np.nanmean(gpv_m[:, :, ha_g_idx], axis=0)   # (ensembles,)
    ha_ig  = np.nanmean(ig_m[:, :, ha_g_idx],  axis=0)
    ha_cpv = np.nanmean(cpv_m[:, :, ha_g_idx], axis=0)

    # Also compute head_angle attribution as fraction of total attribution
    total_gpv = np.nansum(gpv_m, axis=2)  # (sessions, ensembles) - total attribution
    ha_gpv_frac = np.nanmean(gpv_m[:, :, ha_g_idx] / (total_gpv + 1e-12), axis=0)

    # Sort ensembles by head_angle GPV
    order = np.argsort(ha_gpv)[::-1]
    labels = [f"{prefix_name}{i+1:02d}" for i in order]

    print("\nHead angle GPV ranking (top 10):")
    for _k, _n in enumerate(order[:10]):
        print(f"  {prefix_name}{_n+1:02d}  GPV={ha_gpv[_n]:.4f}  IG={ha_ig[_n]:.4f}  "
              f"CPV={ha_cpv[_n]:.4f}  frac={ha_gpv_frac[_n]:.3f}  maxR²={max_r2_ens[_n]:.3f}")

    # Figure 2a: grouped bar chart — head_angle attribution per ensemble
    fig, ax = plt.subplots(figsize=(max(10, num_ensembles * 0.65), 4.5))
    _x  = np.arange(num_ensembles)
    _w  = 0.25
    ax.bar(_x - _w, ha_gpv[order], _w, label='Global PV',  color='steelblue', alpha=0.85)
    ax.bar(_x,      ha_ig[order],  _w, label='IG',         color='coral',     alpha=0.85)
    ax.bar(_x + _w, ha_cpv[order], _w, label='Cond-PV',    color='forestgreen', alpha=0.85)
    ax.axhline(0, color='black', lw=0.7)
    ax.set_xticks(_x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean head_angle attribution (averaged across sessions)', fontsize=9)
    ax.set_title(f'Head Angle Attribution per Ensemble (all 3 methods)\n'
                 f'Sorted by Global PV; head_angle dominates many ensembles', fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    savefig('ha_ranking_bar.png')

    # Figure 2b: head_angle vs all-features attribution comparison (each ensemble)
    mean_all_gpv = np.nanmean(np.nanmean(gpv_m, axis=0), axis=1)  # (ensembles,) - mean over all groups
    fig, ax = plt.subplots(figsize=(max(8, num_ensembles * 0.55), 4))
    _x = np.arange(num_ensembles)
    order2 = np.argsort(mean_all_gpv)[::-1]
    ax.bar(_x, mean_all_gpv[order2], label='All features (mean GPV)', color='grey', alpha=0.6)
    ax.bar(_x, ha_gpv[order2],       label='head_angle only',         color='steelblue', alpha=0.85)
    ax.set_xticks(_x)
    ax.set_xticklabels([f"{prefix_name}{i+1:02d}" for i in order2], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean GPV attribution', fontsize=9)
    ax.set_title('Head Angle Attribution vs Mean Over All Features\n'
                 'Sorted by overall GPV; head_angle fraction visible', fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    savefig('ha_fraction_of_total.png')

    # Save head_angle attribution summary
    np.save(os.path.join(output_dir, 'ha_gpv_per_ensemble.npy'), ha_gpv)
    np.save(os.path.join(output_dir, 'ha_ig_per_ensemble.npy'),  ha_ig)
    np.save(os.path.join(output_dir, 'ha_cpv_per_ensemble.npy'), ha_cpv)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SCATTER SHOWCASE — ENSEMBLE ACTIVITY vs HEAD ANGLE
# ══════════════════════════════════════════════════════════════════════════════

if RUN_SCATTER:
    print("\n" + "="*60)
    print("SECTION 3 — SCATTER SHOWCASE")
    print("="*60)

    # Select top pairs by head_angle GPV × R²
    ha_gpv_pair  = gpv_m[:, :, ha_g_idx]               # (sessions, ensembles)
    score_scatter = np.nan_to_num(ha_gpv_pair * mean_r2, nan=-1.0)
    score_scatter = np.where(valid_mask, score_scatter, -1.0)

    # Diversity selection: max 2 per ensemble
    cnt_ens = {}
    top_pairs = []
    for fi in np.argsort(score_scatter.ravel())[::-1]:
        if len(top_pairs) >= TOP_SCATTER:
            break
        s_i = fi // num_ensembles
        n_i = fi  % num_ensembles
        if score_scatter[s_i, n_i] <= 0:
            break
        if not valid_mask[s_i, n_i]:
            continue
        if cnt_ens.get(n_i, 0) >= 2:
            continue
        top_pairs.append((s_i, n_i))
        cnt_ens[n_i] = cnt_ens.get(n_i, 0) + 1

    print(f"Selected {len(top_pairs)} scatter pairs:")
    for s_i, n_i in top_pairs:
        print(f"  S{s_i+1:2d}  {prefix_name}{n_i+1:02d}  "
              f"ha_GPV={gpv[s_i, n_i, ha_g_idx]:.4f}  R²={mean_r2[s_i, n_i]:.3f}")

    # Figure 3a: 3×3 scatter grid
    _nr = int(np.ceil(len(top_pairs) / 3))
    _nc = min(3, len(top_pairs))
    fig, axes = plt.subplots(_nr, _nc, figsize=(_nc * 5, _nr * 4.5), squeeze=False)
    _af = axes.flatten()

    for _k, (s_i, n_i) in enumerate(top_pairs):
        sess = session_ids[s_i]
        X_test, Y_test = _test_arrays(sess)
        ha_test = X_test[:, ha_f_idx]
        y_ens   = Y_test[:, n_i]
        rho_v, _ = spearmanr(ha_test, y_ens)

        _ax = _af[_k]
        _n_sc = min(3000, len(y_ens))
        _ax.scatter(ha_test[:_n_sc], y_ens[:_n_sc], alpha=0.25, s=8,
                    c='steelblue', rasterized=True)
        _ax.set_xlabel('head_angle (z-scored)', fontsize=9)
        _ax.set_ylabel(f'{prefix_name}{n_i+1:02d} activity (z)', fontsize=9)
        _ax.set_title(f'{prefix_name}{n_i+1:02d} — Session {s_i+1}\n'
                      f'ρ={rho_v:.3f}  R²={mean_r2[s_i, n_i]:.3f}  GPV={gpv[s_i, n_i, ha_g_idx]:.4f}',
                      fontsize=9)
        _ax.spines[['top', 'right']].set_visible(False)

    for _k in range(len(top_pairs), len(_af)):
        _af[_k].set_visible(False)
    plt.suptitle(f'Ensemble Activity vs Head Angle — Top Pairs (seed-42 test data)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    savefig('ha_scatter_grid.png')

    # Figure 3b: per-session trend for the single strongest ensemble
    best_ens = top_pairs[0][1]
    _rhos_by_sess = []
    _valid_sess   = []
    for s_i, sess in enumerate(session_ids):
        if not valid_mask[s_i, best_ens]:
            _rhos_by_sess.append(np.nan)
        else:
            X_t, Y_t = _test_arrays(sess)
            rho_s, _ = spearmanr(X_t[:, ha_f_idx], Y_t[:, best_ens])
            _rhos_by_sess.append(rho_s)
        _valid_sess.append(s_i + 1)

    fig, ax = plt.subplots(figsize=(max(8, num_sessions * 0.55), 4))
    _colors = ['steelblue' if not np.isnan(r) else '#bbbbbb' for r in _rhos_by_sess]
    ax.bar(range(num_sessions), _rhos_by_sess, color=_colors)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(range(num_sessions))
    ax.set_xticklabels([f"S{j+1}" for j in range(num_sessions)], rotation=45, fontsize=9)
    ax.set_ylabel('Spearman ρ (head_angle vs ensemble)', fontsize=9)
    ax.set_title(f'Head Angle Correlation with {prefix_name}{best_ens+1:02d} across Sessions\n'
                 f'(grey = R²<{R2_THRESHOLD} threshold)', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    savefig(f'ha_corr_over_sessions_E{best_ens+1:02d}.png')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: ZERO-ABLATION TEST
# ══════════════════════════════════════════════════════════════════════════════
# Run the trained MLP with:
#   (a) all features → full R²
#   (b) only head_angle active (others zeroed) → single-feature R²
#   (c) head_angle zeroed (kept all others) → ablated R²
# Loads the seed-42 model; tests on seed-42 test set.

if RUN_ABLATION:
    print("\n" + "="*60)
    print("SECTION 4 — ZERO-ABLATION TEST")
    print("="*60)

    # Select top TOP_ABLATION pairs by head_angle attribution × R²
    _score_abl = np.where(valid_mask, ha_gpv_pair * mean_r2 if 'ha_gpv_pair' in dir() else
                           gpv_m[:, :, ha_g_idx] * mean_r2, -1.0)
    _abl_pairs = []
    _cnt_abl   = {}
    for fi in np.argsort(np.nan_to_num(_score_abl, nan=-1.0).ravel())[::-1]:
        if len(_abl_pairs) >= TOP_ABLATION:
            break
        s_i = fi // num_ensembles
        n_i = fi  % num_ensembles
        if _score_abl[s_i, n_i] <= 0:
            break
        if not valid_mask[s_i, n_i]:
            continue
        if _cnt_abl.get(n_i, 0) >= 1:
            continue
        _abl_pairs.append((s_i, n_i))
        _cnt_abl[n_i] = _cnt_abl.get(n_i, 0) + 1

    ablation_rows = []
    print(f"Ablation pairs: {[f'S{s+1} {prefix_name}{n+1}' for s, n in _abl_pairs]}")

    for s_i, n_i in _abl_pairs:
        sess = session_ids[s_i]
        model = _load_model(s_i, n_i, seed=42)
        if model is None:
            print(f"  Model not found for S{s_i+1} {prefix_name}{n_i+1} — skipping")
            continue

        X_test, Y_test = _test_arrays(sess)
        y_true = Y_test[:, n_i]
        Xt = torch.tensor(X_test, device=device, dtype=torch.float32)

        # (a) Full model
        with torch.no_grad():
            _, pf = model(Xt)
        r2_full = r2_score(y_true, pf.squeeze().cpu().numpy())

        # (b) Only head_angle
        X_ha_only = np.zeros_like(X_test)
        X_ha_only[:, ha_f_idx] = X_test[:, ha_f_idx]
        with torch.no_grad():
            _, ph = model(torch.tensor(X_ha_only, device=device, dtype=torch.float32))
        r2_ha_only = r2_score(y_true, ph.squeeze().cpu().numpy())

        # (c) head_angle zeroed
        X_no_ha = X_test.copy()
        X_no_ha[:, ha_f_idx] = 0.0
        with torch.no_grad():
            _, pn = model(torch.tensor(X_no_ha, device=device, dtype=torch.float32))
        r2_no_ha = r2_score(y_true, pn.squeeze().cpu().numpy())

        # (d) head_angle + head_angle_vel only
        X_hav_only = np.zeros_like(X_test)
        X_hav_only[:, ha_f_idx]  = X_test[:, ha_f_idx]
        X_hav_only[:, hav_f_idx] = X_test[:, hav_f_idx]
        with torch.no_grad():
            _, phv = model(torch.tensor(X_hav_only, device=device, dtype=torch.float32))
        r2_hav_only = r2_score(y_true, phv.squeeze().cpu().numpy())

        ablation_rows.append({
            'session': s_i + 1, 'ensemble': f"{prefix_name}{n_i+1:02d}",
            'r2_full': round(r2_full, 4),
            'r2_ha_only': round(r2_ha_only, 4),
            'r2_ha_vel_only': round(r2_hav_only, 4),
            'r2_no_ha': round(r2_no_ha, 4),
            'ha_gpv': round(float(gpv[s_i, n_i, ha_g_idx]), 4),
        })
        print(f"  S{s_i+1} {prefix_name}{n_i+1:02d}: full={r2_full:.3f}  "
              f"ha_only={r2_ha_only:.3f}  ha+vel_only={r2_hav_only:.3f}  no_ha={r2_no_ha:.3f}")

        del model
        torch.cuda.empty_cache()

    if ablation_rows:
        df_abl = pd.DataFrame(ablation_rows)
        df_abl.to_csv(os.path.join(output_dir, 'ablation_results.csv'), index=False)
        print(f"\nSaved ablation_results.csv")

        # Figure 4: grouped bar chart of ablation R²
        _n_pairs = len(df_abl)
        _labels  = [f"S{r['session']} {r['ensemble']}" for _, r in df_abl.iterrows()]
        _x  = np.arange(_n_pairs)
        _w  = 0.20
        fig, ax = plt.subplots(figsize=(max(8, _n_pairs * 1.2), 4.5))
        ax.bar(_x - 1.5*_w, df_abl['r2_full'],     _w, label='Full model',           color='#2196F3', alpha=0.85)
        ax.bar(_x - 0.5*_w, df_abl['r2_ha_only'],  _w, label='Only head_angle',      color='#FF9800', alpha=0.85)
        ax.bar(_x + 0.5*_w, df_abl['r2_ha_vel_only'], _w, label='head_angle + vel',  color='#FF5722', alpha=0.85)
        ax.bar(_x + 1.5*_w, df_abl['r2_no_ha'],    _w, label='Without head_angle',   color='#9E9E9E', alpha=0.85)
        ax.axhline(0, color='black', lw=0.7)
        ax.set_xticks(_x)
        ax.set_xticklabels(_labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('R² (seed-42 test set)', fontsize=10)
        ax.set_title('Zero-Ablation: How Much Does head_angle Alone Explain?\n'
                     '(features not selected are set to zero; uses existing trained model)',
                     fontsize=11)
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        savefig('ablation_r2_bars.png')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: SINGLE-FEATURE MLP TRAINING
# ══════════════════════════════════════════════════════════════════════════════
# Train a fresh small MLP using only head_angle as input.
# Compare R² to the full-model R² to quantify how much variance head_angle
# alone can explain when the model is optimised for it (not for all features).

if RUN_SINGLEFEAT:
    print("\n" + "="*60)
    print("SECTION 5 — SINGLE-FEATURE MLP TRAINING")
    print("="*60)

    class SmallMLP(nn.Module):
        def __init__(self, n_in=1, n_hid=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, n_hid), nn.ReLU(),
                nn.Linear(n_hid, n_hid), nn.ReLU(),
                nn.Linear(n_hid, 1)
            )
        def forward(self, x):
            return self.net(x)

    def _train_single(X_tr, y_tr, X_te, y_te, n_in=1, seed_t=0):
        torch.manual_seed(seed_t)
        mdl = SmallMLP(n_in=n_in, n_hid=32).to(device)
        opt = torch.optim.Adam(mdl.parameters(), lr=SINGLE_LR)
        Xtr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
        ytr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
        for _ in range(SINGLE_EPOCHS):
            mdl.train()
            opt.zero_grad()
            loss = nn.functional.mse_loss(mdl(Xtr_t).squeeze(), ytr_t)
            loss.backward()
            opt.step()
        mdl.eval()
        with torch.no_grad():
            preds = mdl(torch.tensor(X_te, dtype=torch.float32, device=device)).squeeze().cpu().numpy()
        return r2_score(y_te, preds)

    # Select TOP_ABLATION pairs (same as ablation)
    _sf_score  = np.where(valid_mask, gpv_m[:, :, ha_g_idx] * mean_r2, -1.0)
    _sf_pairs  = []
    _cnt_sf    = {}
    for fi in np.argsort(np.nan_to_num(_sf_score, nan=-1.0).ravel())[::-1]:
        if len(_sf_pairs) >= TOP_ABLATION:
            break
        s_i = fi // num_ensembles
        n_i = fi  % num_ensembles
        if _sf_score[s_i, n_i] <= 0:
            break
        if not valid_mask[s_i, n_i]:
            continue
        if _cnt_sf.get(n_i, 0) >= 1:
            continue
        _sf_pairs.append((s_i, n_i))
        _cnt_sf[n_i] = _cnt_sf.get(n_i, 0) + 1

    sf_rows = []
    print(f"Single-feature pairs: {[f'S{s+1} {prefix_name}{n+1}' for s, n in _sf_pairs]}")

    for s_i, n_i in _sf_pairs:
        sess = session_ids[s_i]
        sd   = session_dataset_singles[sess]
        test_trials  = _test_idx_map[sess]
        train_trials = [t for t in sorted(sd['data'].keys()) if t not in test_trials]

        X_tr = np.concatenate([sd['data'][t]   for t in train_trials], axis=0).astype(np.float32)
        Y_tr = np.concatenate([sd['labels'][t] for t in train_trials], axis=0).astype(np.float32)
        X_te = np.concatenate([sd['data'][t]   for t in test_trials],  axis=0).astype(np.float32)
        Y_te = np.concatenate([sd['labels'][t] for t in test_trials],  axis=0).astype(np.float32)

        y_tr = Y_tr[:, n_i]
        y_te = Y_te[:, n_i]

        # Train N_SINGLE_SEEDS replicates with head_angle only (1 feature)
        r2_ha_seeds = [_train_single(
            X_tr[:, ha_f_idx:ha_f_idx+1],
            y_tr,
            X_te[:, ha_f_idx:ha_f_idx+1],
            y_te, n_in=1, seed_t=_s
        ) for _s in range(N_SINGLE_SEEDS)]

        # Also train with head_angle + head_angle_vel (2 features)
        ha_cols = np.array([ha_f_idx, hav_f_idx])
        r2_hav_seeds = [_train_single(
            X_tr[:, ha_cols],
            y_tr,
            X_te[:, ha_cols],
            y_te, n_in=2, seed_t=_s
        ) for _s in range(N_SINGLE_SEEDS)]

        r2_full = float(mean_r2[s_i, n_i])  # from precomputed all_r2 (seed-avg)

        sf_rows.append({
            'session':          s_i + 1,
            'ensemble':         f"{prefix_name}{n_i+1:02d}",
            'r2_full_seedavg':  round(r2_full, 4),
            'r2_ha_mean':       round(np.mean(r2_ha_seeds), 4),
            'r2_ha_std':        round(np.std(r2_ha_seeds), 4),
            'r2_hav_mean':      round(np.mean(r2_hav_seeds), 4),
            'r2_hav_std':       round(np.std(r2_hav_seeds), 4),
            'ha_explains_pct':  round(100 * np.mean(r2_ha_seeds) / max(r2_full, 1e-6), 1),
            'ha_gpv':           round(float(gpv[s_i, n_i, ha_g_idx]), 4),
        })
        print(f"  S{s_i+1} {prefix_name}{n_i+1:02d}: full_R²={r2_full:.3f}  "
              f"ha_only_R²={np.mean(r2_ha_seeds):.3f}±{np.std(r2_ha_seeds):.3f}  "
              f"ha+vel_R²={np.mean(r2_hav_seeds):.3f}  "
              f"({round(100*np.mean(r2_ha_seeds)/max(r2_full,1e-6), 1)}% explained by ha alone)")

    if sf_rows:
        df_sf = pd.DataFrame(sf_rows)
        df_sf.to_csv(os.path.join(output_dir, 'single_feature_mlp_results.csv'), index=False)
        print(f"\nSaved single_feature_mlp_results.csv")

        # Figure 5: scatter — head-angle-only R² vs full model R²
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(df_sf['r2_full_seedavg'], df_sf['r2_ha_mean'],
                   s=80, color='steelblue', zorder=5, label='head_angle only')
        ax.scatter(df_sf['r2_full_seedavg'], df_sf['r2_hav_mean'],
                   s=80, color='coral',     zorder=5, marker='^', label='head_angle + vel')
        # Error bars for HA only
        for _, row in df_sf.iterrows():
            ax.errorbar(row['r2_full_seedavg'], row['r2_ha_mean'],
                        yerr=row['r2_ha_std'], fmt='none', color='steelblue', alpha=0.5)
        _lim = max(df_sf['r2_full_seedavg'].max(), df_sf['r2_ha_mean'].max()) * 1.05
        ax.plot([0, _lim], [0, _lim], 'k--', lw=0.8, label='y=x (equal)')
        for _, row in df_sf.iterrows():
            ax.annotate(f"S{row['session']} {row['ensemble']}",
                        (row['r2_full_seedavg'], row['r2_ha_mean']),
                        fontsize=7, ha='left', va='bottom')
        ax.set_xlabel('Full model R² (seed-avg)', fontsize=10)
        ax.set_ylabel('Single-feature MLP R² (head_angle)', fontsize=10)
        ax.set_title('How Well Does head_angle Alone Predict Ensemble?\n'
                     'Points above diagonal = head_angle alone matches full model', fontsize=10)
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        savefig('single_feat_mlp_scatter.png')

        # Figure 5b: bar chart percentage explained
        fig, ax = plt.subplots(figsize=(max(7, len(df_sf) * 1.2), 4))
        _lbs = [f"S{r['session']} {r['ensemble']}" for _, r in df_sf.iterrows()]
        ax.bar(range(len(df_sf)), df_sf['ha_explains_pct'], color='steelblue', alpha=0.85)
        ax.axhline(100, color='coral', linestyle='--', lw=1.2, label='100% (= full model)')
        ax.set_xticks(range(len(df_sf)))
        ax.set_xticklabels(_lbs, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('head_angle-only R² as % of full model R²', fontsize=9)
        ax.set_title('head_angle Alone Explains __% of Full Model Performance', fontsize=11)
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        savefig('single_feat_pct_explained.png')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: CONFOUND ANALYSIS — PARTIAL CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
# Is head_angle's attribution strength driven by its unique information, or
# is it a proxy for movement/position? We check:
#   (a) Partial correlation: head_angle ~ ensemble, partialling out movement_energy
#   (b) How much does head_angle attribution change after conditioning on track_zone?
#   (c) head_angle distribution across track zones (is it zone-specific?)

if RUN_CONFOUND:
    print("\n" + "="*60)
    print("SECTION 6 — CONFOUND / PARTIAL CORRELATION ANALYSIS")
    print("="*60)

    def partial_correlation(x, y, z):
        """Partial correlation of x and y, controlling for z (all 1D arrays)."""
        from numpy.linalg import lstsq
        _x_res = x - lstsq(z[:, None], x, rcond=None)[0][0] * z
        _y_res = y - lstsq(z[:, None], y, rcond=None)[0][0] * z
        r, p   = pearsonr(_x_res, _y_res)
        return r, p

    # For each session: compute partial correlation between ensemble activity
    # and head_angle, partialling out movement_energy
    partial_corr_rows = []
    for s_idx, sess in enumerate(session_ids):
        sd = session_dataset_singles[sess]
        trial_keys = sorted(sd['data'].keys())
        X_all = np.concatenate([sd['data'][t]   for t in trial_keys], axis=0).astype(float)
        Y_all = np.concatenate([sd['labels'][t] for t in trial_keys], axis=0).astype(float)

        ha  = X_all[:, ha_f_idx]
        me  = X_all[:, me_f_idx]
        ok  = ~(np.isnan(ha) | np.isnan(me))
        ha_ok = ha[ok]
        me_ok = me[ok]

        for n_i in range(num_ensembles):
            if not valid_mask[s_idx, n_i]:
                continue
            y   = Y_all[ok, n_i]
            # Raw Spearman
            rho_raw, _ = spearmanr(ha_ok, y)
            # Partial pearson (controlling for movement_energy)
            r_part, p_part = partial_correlation(ha_ok, y, me_ok)
            partial_corr_rows.append({
                'session': s_idx + 1, 'ensemble': f"{prefix_name}{n_i+1:02d}",
                'raw_spearman_ha': round(rho_raw, 4),
                'partial_r_ha_ctrl_me': round(r_part, 4),
                'partial_p': round(p_part, 6),
            })

    df_partial = pd.DataFrame(partial_corr_rows)
    df_partial.to_csv(os.path.join(output_dir, 'partial_correlation.csv'), index=False)

    # Summary: is partial correlation still strong?
    print(f"\nPartial correlation (head_angle ~ ensemble, controlling for movement_energy):")
    print(f"  Mean raw |ρ|:     {df_partial['raw_spearman_ha'].abs().mean():.3f}")
    print(f"  Mean partial |r|: {df_partial['partial_r_ha_ctrl_me'].abs().mean():.3f}")

    # Figure 6a: raw vs partial correlation scatter
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df_partial['raw_spearman_ha'].abs(),
               df_partial['partial_r_ha_ctrl_me'].abs(),
               alpha=0.35, s=12, color='steelblue', rasterized=True)
    _mx = max(df_partial['raw_spearman_ha'].abs().max(),
              df_partial['partial_r_ha_ctrl_me'].abs().max()) * 1.05
    ax.plot([0, _mx], [0, _mx], 'k--', lw=0.8, label='y=x')
    ax.set_xlabel('|Raw Spearman ρ| (head_angle ~ ensemble)', fontsize=9)
    ax.set_ylabel('|Partial r| (controlling for movement_energy)', fontsize=9)
    ax.set_title('Does head_angle Uniquely Predict Ensemble Activity?\n'
                 'Points on diagonal = movement_energy is not a confound', fontsize=10)
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    savefig('partial_corr_scatter.png')

    # Figure 6b: head_angle distribution per track zone (using raw behavioral data)
    if 'track_zone_int' in behavior_glm_loaded.columns:
        tz_vals = behavior_glm_loaded['track_zone_int'].values.astype(float)
        ha_vals = behavior_glm_loaded['head_angle'].values.astype(float)
        ok_tz   = ~(np.isnan(tz_vals) | np.isnan(ha_vals))
        tz_vals_ok = tz_vals[ok_tz]
        ha_vals_ok = ha_vals[ok_tz]
        unique_tz  = sorted(set(tz_vals_ok.astype(int)))

        fig, ax = plt.subplots(figsize=(max(7, len(unique_tz) * 0.8), 4))
        data_tz = [ha_vals_ok[tz_vals_ok.astype(int) == tz] for tz in unique_tz]
        ax.violinplot(data_tz, positions=unique_tz, showmedians=True, widths=0.7)
        # Spearman ρ between head_angle and track_zone
        rho_tz, p_tz = spearmanr(tz_vals_ok, ha_vals_ok)
        ax.set_xlabel('Track zone', fontsize=10)
        ax.set_ylabel('Head angle (raw)', fontsize=10)
        ax.set_title(f'Head Angle Distribution by Track Zone\nSpearman ρ={rho_tz:.3f}  p={p_tz:.3f}',
                     fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        savefig('ha_by_track_zone.png')
        print(f"\n  head_angle ~ track_zone: ρ={rho_tz:.3f}  p={p_tz:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("DONE — Head Angle Analysis Complete")
print("="*60)
print(f"\nAll outputs in: {output_dir}")
print(f"Desktop copy:   {desktop_dir}")
