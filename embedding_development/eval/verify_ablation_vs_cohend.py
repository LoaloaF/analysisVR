#!/usr/bin/env python3
"""
For E07 x cue_visible and E23 x upcoming_choice:
  - Load the per-session ablation R² already computed
  - Compute per-session Cohen's d of neural activity split by condition
  - Check Spearman rho between Cohen's d and ablation R²

If ablation R² doesn't track Cohen's d, it's the wrong metric for categorical features.
"""

import os, sys, pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))

with open('../outputs/mlps/ensembles_multiseed/semantic_groups.pkl', 'rb') as f:
    sg = pickle.load(f)
feat_idx_by_group = {g: cols for g, cols in sg}

with open('../outputs/session_dataset_ensembles.pkl', 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())

PAIRS = [
    dict(name='E07_cue_visible',    ens_idx=6,  feat_cols=feat_idx_by_group['cue_visible'],
         csv='../outputs/ablation_vs_attribution/E07_cue_visible_ablation.csv'),
    dict(name='E23_upcoming_choice', ens_idx=22, feat_cols=feat_idx_by_group['upcoming_choice'],
         csv='../outputs/ablation_vs_attribution/E23_upcoming_choice_ablation.csv'),
]


def cohens_d_max(y_by_cond):
    """Max pairwise Cohen's d across conditions."""
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


for pair in PAIRS:
    df = pd.read_csv(pair['csv'])
    ens_idx   = pair['ens_idx']
    feat_cols = pair['feat_cols']

    cohen_ds = []
    for _, row in df.iterrows():
        sess = row['session']
        sd   = ds[sess]
        all_trials = list(sd['data'].keys())

        # collect all timepoints, split by one-hot condition
        X_list, y_list = [], []
        for t in all_trials:
            if t not in sd['data']:
                continue
            X_list.append(sd['data'][t][:, feat_cols])
            y_list.append(sd['labels'][t][:, ens_idx])

        X = np.concatenate(X_list, axis=0)  # (T, 3)
        y = np.concatenate(y_list, axis=0)  # (T,)

        cond = np.argmax(X, axis=1)         # 0, 1, or 2
        y_by_cond = [y[cond == c] for c in range(X.shape[1])]
        cohen_ds.append(cohens_d_max(y_by_cond))

    df['cohen_d'] = cohen_ds

    rho, p = spearmanr(df['cohen_d'], df['abl_r2'])

    print(f'\n{pair["name"]}')
    print(f'  Sessions: {len(df)}')
    print(f'  Cohen d range:   {df["cohen_d"].min():.3f} – {df["cohen_d"].max():.3f}  '
          f'(mean {df["cohen_d"].mean():.3f})')
    print(f'  Ablation R² range: {df["abl_r2"].min():.4f} – {df["abl_r2"].max():.4f}  '
          f'(mean {df["abl_r2"].mean():.4f})')
    print(f'  Spearman rho(cohen_d, abl_r2) = {rho:+.3f}  p={p:.3f}')
    print()
    print(f'  {"session":<30}  {"cohen_d":>8}  {"abl_r2":>8}')
    for _, row in df.sort_values('cohen_d', ascending=False).iterrows():
        print(f'  {row["session"]:<30}  {row["cohen_d"]:8.3f}  {row["abl_r2"]:8.4f}')
