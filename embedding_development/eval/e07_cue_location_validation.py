#!/usr/bin/env python3
"""
e07_cue_location_validation.py

Validation case study for E07 across two features:
  Row A — cue_visible  (categorical): Cohen's d (max pairwise between cue conditions)
  Row B — frame_position (continuous): R² of linear regression position → neural activity

Each row has 6 columns: MLP IG, MLP GPV, TC-Cont IG, TC-Cont GPV, TC-Pred IG, TC-Pred GPV.
X-axis = attribution score, Y-axis = independent neural measure.
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

R2_THRESH = 0.01
ENS_IDX   = 6   # E07

output_dir  = '../outputs/ablation_vs_attribution'
desktop_dir = '/mnt/c/Users/amits/Desktop/ablation_vs_attribution'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(desktop_dir, exist_ok=True)

# ── Load semantic groups ───────────────────────────────────────────────────────
with open('../outputs/mlps/ensembles_multiseed/semantic_groups.pkl', 'rb') as f:
    sg = pickle.load(f)
group_names       = [g[0] for g in sg]
feat_idx_by_group = {g: cols for g, cols in sg}

CUE_NAME  = 'cue_visible'
LOC_NAME  = 'frame_position'
cue_cols  = feat_idx_by_group[CUE_NAME]    # [7, 8, 9]  one-hot
loc_cols  = feat_idx_by_group[LOC_NAME]    # [6]        continuous
cue_g_idx = group_names.index(CUE_NAME)
loc_g_idx = group_names.index(LOC_NAME)

# ── Load data ─────────────────────────────────────────────────────────────────
with open('../outputs/session_dataset_ensembles.pkl', 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())

mlp_gpv  = np.load('../outputs/mlps/ensembles_multiseed/importance_global_pv_semantic.npy')
mlp_ig   = np.load('../outputs/mlps/ensembles_multiseed/importance_ig_semantic.npy')
ceb_gpv  = np.load('../outputs/cebra_eval/ensembles/importance_global_pv_semantic.npy')
ceb_ig   = np.load('../outputs/cebra_eval/ensembles/importance_ig_semantic.npy')
pred_gpv = np.load('../outputs/cebra_pred_eval/ensembles/importance_global_pv_semantic.npy')
pred_ig  = np.load('../outputs/cebra_pred_eval/ensembles/importance_ig_semantic.npy')

mlp_r2_all  = np.load('../outputs/mlps/ensembles_multiseed/all_r2.npy')
ceb_r2_all  = np.load('../outputs/cebra_eval/ensembles/all_r2.npy')
pred_r2_all = np.load('../outputs/cebra_pred_eval/ensembles/all_r2.npy')

mlp_valid  = (~np.all(np.isnan(mlp_r2_all),  axis=0)) & (np.nanmean(mlp_r2_all,  axis=0) >= R2_THRESH)
ceb_valid  = (~np.all(np.isnan(ceb_r2_all),  axis=0)) & (np.nanmean(ceb_r2_all,  axis=0) >= R2_THRESH)
pred_valid = (~np.all(np.isnan(pred_r2_all), axis=0)) & (np.nanmean(pred_r2_all, axis=0) >= R2_THRESH)

# ── Cohen's d helper (cue) ────────────────────────────────────────────────────
def cohens_d_max(y_by_cond):
    conds = [c for c in y_by_cond if len(c) > 1]
    if len(conds) < 2:
        return np.nan
    best = 0.0
    for i in range(len(conds)):
        for j in range(i + 1, len(conds)):
            a, b = conds[i], conds[j]
            pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            if pooled_sd < 1e-10:
                continue
            best = max(best, abs(np.mean(a) - np.mean(b)) / pooled_sd)
    return best

# ── Build rows ────────────────────────────────────────────────────────────────
valid_sessions = [s for s in range(len(sessions)) if mlp_valid[s, ENS_IDX]]

rows = []
for s_idx in valid_sessions:
    sess = sessions[s_idx]
    sd   = ds[sess]
    all_trials = [t for t in sd['data']]

    X_cue, X_loc, y_all = [], [], []
    for t in all_trials:
        X_cue.append(sd['data'][t][:, cue_cols])
        X_loc.append(sd['data'][t][:, loc_cols])
        y_all.append(sd['labels'][t][:, ENS_IDX])

    X_cue = np.concatenate(X_cue, axis=0)
    X_loc = np.concatenate(X_loc, axis=0).ravel()   # (T,)
    y     = np.concatenate(y_all, axis=0)            # (T,)

    # Cohen's d for cue (one-hot → argmax → group)
    cond     = np.argmax(X_cue, axis=1)
    y_by_cond = [y[cond == c] for c in range(X_cue.shape[1])]
    cue_cd   = cohens_d_max(y_by_cond)

    # R² for location (linear regression)
    if np.std(X_loc) < 1e-10 or np.std(y) < 1e-10:
        loc_r2 = np.nan
    else:
        loc_r2 = r2_score(y, LinearRegression().fit(X_loc[:, None], y).predict(X_loc[:, None]))

    rows.append(dict(
        s_idx   = s_idx,
        session = sess,
        cue_cd  = cue_cd,
        loc_r2  = loc_r2,
        mlp_ig_cue   = mlp_ig[s_idx,  ENS_IDX, cue_g_idx],
        mlp_gpv_cue  = mlp_gpv[s_idx, ENS_IDX, cue_g_idx],
        mlp_ig_loc   = mlp_ig[s_idx,  ENS_IDX, loc_g_idx],
        mlp_gpv_loc  = mlp_gpv[s_idx, ENS_IDX, loc_g_idx],
        ceb_ig_cue   = ceb_ig[s_idx,  ENS_IDX, cue_g_idx]  if ceb_valid[s_idx,  ENS_IDX] else np.nan,
        ceb_gpv_cue  = ceb_gpv[s_idx, ENS_IDX, cue_g_idx]  if ceb_valid[s_idx,  ENS_IDX] else np.nan,
        ceb_ig_loc   = ceb_ig[s_idx,  ENS_IDX, loc_g_idx]  if ceb_valid[s_idx,  ENS_IDX] else np.nan,
        ceb_gpv_loc  = ceb_gpv[s_idx, ENS_IDX, loc_g_idx]  if ceb_valid[s_idx,  ENS_IDX] else np.nan,
        pred_ig_cue  = pred_ig[s_idx,  ENS_IDX, cue_g_idx] if pred_valid[s_idx, ENS_IDX] else np.nan,
        pred_gpv_cue = pred_gpv[s_idx, ENS_IDX, cue_g_idx] if pred_valid[s_idx, ENS_IDX] else np.nan,
        pred_ig_loc  = pred_ig[s_idx,  ENS_IDX, loc_g_idx] if pred_valid[s_idx, ENS_IDX] else np.nan,
        pred_gpv_loc = pred_gpv[s_idx, ENS_IDX, loc_g_idx] if pred_valid[s_idx, ENS_IDX] else np.nan,
    ))

df = pd.DataFrame(rows)
print(f'Valid sessions: {len(df)}')

# ── Plot ──────────────────────────────────────────────────────────────────────
METHODS = [
    ('mlp_ig',   'MLP IG',           '#4CAF50', 'o'),
    ('mlp_gpv',  'MLP GPV',          '#1B5E20', 's'),
    ('ceb_ig',   'TC-Cont IG',       '#2196F3', 'o'),
    ('ceb_gpv',  'TC-Cont GPV',      '#0D47A1', 's'),
    ('pred_ig',  'TC-Pred IG',       '#FF9800', 'o'),
    ('pred_gpv', 'TC-Pred GPV',      '#E65100', 's'),
]

ROWS = [
    ('cue',  'cue_cd',  "Cohen's d (cue_visible)",  'cue_visible attribution'),
    ('loc',  'loc_r2',  "R² (position → activity)", 'frame_position attribution'),
]

fig, axes = plt.subplots(2, 6, figsize=(26, 9))

print('\nE07 validation:')
for row_i, (feat_sfx, y_col, y_label, x_label_base) in enumerate(ROWS):
    print(f'\n  Feature: {y_col}')
    print(f'  {"Method":<20}  {"rho":>6}  {"p":>6}  {"n":>4}')
    for col_i, (m_sfx, m_label, color, marker) in enumerate(METHODS):
        ax    = axes[row_i, col_i]
        x_col = f'{m_sfx}_{feat_sfx}'
        sub   = df[[y_col, x_col]].dropna()

        if len(sub) < 4:
            ax.set_title(f'{m_label}\n(n<4)', fontsize=9)
            ax.set_visible(True)
            continue

        rho, p = spearmanr(sub[x_col], sub[y_col])
        print(f'  {m_label:<20}  {rho:+.3f}  {p:.3f}  {len(sub):>4}')

        ax.scatter(sub[x_col], sub[y_col],
                   color=color, marker=marker, s=60, alpha=0.8, zorder=3)
        for _, r in sub.iterrows():
            sidx = int(df.loc[r.name, 's_idx'])
            ax.annotate(f'S{sidx:02d}', (r[x_col], r[y_col]),
                        fontsize=6, xytext=(3, 2), textcoords='offset points')

        ax.set_xlabel(m_label, fontsize=9)
        if col_i == 0:
            ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(f'ρ={rho:+.3f}  p={p:.3f}', fontsize=9)

plt.suptitle('E07 — cue_visible (Cohen\'s d) and frame_position (R²) vs attribution scores',
             fontsize=11)
plt.tight_layout()

fname = 'E07_cue_location_validation.png'
for d in (output_dir, desktop_dir):
    plt.savefig(os.path.join(d, fname), dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved {fname}')
