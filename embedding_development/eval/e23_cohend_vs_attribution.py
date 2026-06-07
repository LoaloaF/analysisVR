#!/usr/bin/env python3
"""
For E23 x upcoming_choice: scatter per-session Cohen's d (neural activity split by
condition) against each attribution score (MLP IG/GPV, TempConv-Cont IG/GPV,
TempConv-Pred IG/GPV).
"""

import os, sys, pickle, shutil
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

SEEDS     = [42, 43, 44, 45, 46]
R2_THRESH = 0.01
ENS_IDX   = 22
G_NAME    = 'upcoming_choice'

output_dir  = '../outputs/ablation_vs_attribution'
desktop_dir = '/mnt/c/Users/amits/Desktop/ablation_vs_attribution'

with open('../outputs/mlps/ensembles_multiseed/semantic_groups.pkl', 'rb') as f:
    sg = pickle.load(f)
group_names       = [g[0] for g in sg]
feat_idx_by_group = {g: cols for g, cols in sg}
feat_cols = feat_idx_by_group[G_NAME]
g_idx     = group_names.index(G_NAME)

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
            d = abs(np.mean(a) - np.mean(b)) / pooled_sd
            best = max(best, d)
    return best


rows = []
valid_sessions = [s for s in range(len(sessions)) if mlp_valid[s, ENS_IDX]]

for s_idx in valid_sessions:
    sess = sessions[s_idx]
    sd   = ds[sess]
    all_trials = list(sd['data'].keys())

    # compute Cohen's d on test trials only (median across seeds)
    cd_per_seed = []
    for seed in SEEDS:
        split_map   = np.load(f'../splits/split_seed{seed}.npy', allow_pickle=True).item()
        test_trials = [int(i) for i in split_map[sess]]
        vt = [t for t in test_trials if t in sd['data']]
        if len(vt) == 0:
            continue
        X = np.concatenate([sd['data'][t][:, feat_cols] for t in vt], axis=0)
        y = np.concatenate([sd['labels'][t][:, ENS_IDX] for t in vt], axis=0)
        cond = np.argmax(X, axis=1)
        y_by_cond = [y[cond == c] for c in range(X.shape[1])]
        cd_per_seed.append(cohens_d_max(y_by_cond))

    cd = float(np.nanmedian(cd_per_seed)) if cd_per_seed else np.nan

    rows.append(dict(
        s_idx=s_idx, session=sess,
        cohen_d=cd,
        mlp_ig   = mlp_ig[s_idx,  ENS_IDX, g_idx],
        mlp_gpv  = mlp_gpv[s_idx, ENS_IDX, g_idx],
        ceb_ig   = ceb_ig[s_idx,  ENS_IDX, g_idx]  if ceb_valid[s_idx,  ENS_IDX] else np.nan,
        ceb_gpv  = ceb_gpv[s_idx, ENS_IDX, g_idx]  if ceb_valid[s_idx,  ENS_IDX] else np.nan,
        pred_ig  = pred_ig[s_idx,  ENS_IDX, g_idx] if pred_valid[s_idx, ENS_IDX] else np.nan,
        pred_gpv = pred_gpv[s_idx, ENS_IDX, g_idx] if pred_valid[s_idx, ENS_IDX] else np.nan,
    ))

df = pd.DataFrame(rows).dropna(subset=['cohen_d'])
print(f'Sessions with valid Cohen d: {len(df)}')

methods = [
    ('mlp_ig',   'MLP IG',          '#4CAF50', 'o'),
    ('mlp_gpv',  'MLP GPV',         '#1B5E20', 's'),
    ('ceb_ig',   'TempConv-Cont IG',   '#2196F3', 'o'),
    ('ceb_gpv',  'TempConv-Cont GPV',  '#0D47A1', 's'),
    ('pred_ig',  'TempConv-Pred IG',   '#FF9800', 'o'),
    ('pred_gpv', 'TempConv-Pred GPV',  '#E65100', 's'),
]

print(f'\nE23 x upcoming_choice — Spearman rho with Cohen d:')
print(f'  {"Method":<20}  {"rho":>6}  {"p":>6}  {"n":>4}')
for col, label, _, _ in methods:
    sub = df[['cohen_d', col]].dropna()
    if len(sub) < 4:
        print(f'  {label:<20}  (n<4)')
        continue
    rho, p = spearmanr(sub[col], sub['cohen_d'])
    print(f'  {label:<20}  {rho:+.3f}  {p:.3f}  {len(sub):>4}')

# scatter plot
fig, axes = plt.subplots(1, 6, figsize=(26, 4.5))
for ax, (col, label, color, marker) in zip(axes, methods):
    sub = df[['cohen_d', col]].dropna()
    if len(sub) < 4:
        ax.set_title(f'{label}\n(n<4)')
        continue
    rho, p = spearmanr(sub[col], sub['cohen_d'])
    ax.scatter(sub[col], sub['cohen_d'], color=color, marker=marker,
               s=60, alpha=0.8, zorder=3)
    for _, row in sub.iterrows():
        sidx = int(df.loc[row.name, 's_idx'])
        ax.annotate(f'S{sidx:02d}', (row[col], row['cohen_d']),
                    fontsize=6, xytext=(3, 2), textcoords='offset points')
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Cohen's d (upcoming_choice)" if ax is axes[0] else '')
    ax.set_title(f'{label}\nρ={rho:+.3f}  p={p:.3f}', fontsize=10)

plt.suptitle("E23 × upcoming_choice — attribution vs per-session Cohen's d", fontsize=11)
plt.tight_layout()
for root in (output_dir, desktop_dir):
    plt.savefig(os.path.join(root, 'E23_attribution_vs_cohend.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\nSaved E23_attribution_vs_cohend.png')
