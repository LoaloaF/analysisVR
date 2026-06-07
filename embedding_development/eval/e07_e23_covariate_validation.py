#!/usr/bin/env python3
"""
e07_e23_covariate_validation.py

2×2 figure telling two stories for E07 (speed) and E23 (head angle):

  Row 1 — "We detect your signal":
    X = Cohen's d for supervisor's variable (cue / choice)
    Y = GPV our model gives to that SAME variable
    → Positive ρ means: when their variable discriminates, we attribute to it too

  Row 2 — "Our variable predicts yours":
    Per-session Spearman r between our primary variable (speed / head angle)
    and the supervisor's condition label (cue_visible / upcoming_choice argmax)
    → Positive r means: speed co-varies with cue timing;
      head angle co-varies with choice side

  Blue points / bars = sessions where our primary variable GPV > supervisor GPV

Output: outputs/ablation_vs_attribution/e07_e23_covariate_validation.png  (9.5"×5.5")
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import FONT, apply_style, add_footnote, add_panel_label, savefig_manifest

R2_THRESH = 0.01
base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, '..')
mdir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [
    os.path.join(root, 'outputs', 'ablation_vs_attribution'),
    '/mnt/c/Users/amits/Desktop',
]
for d in OUT_DIRS: os.makedirs(d, exist_ok=True)

with open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())

with open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb') as f:
    sg = pickle.load(f)
group_names = [g[0] for g in sg]
feat_idx    = {g: cols for g, cols in sg}

gpv    = np.load(os.path.join(mdir, 'importance_global_pv_semantic.npy'))
r2_all = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)


def cohens_d_categorical(X_oh, y):
    cond   = np.argmax(X_oh, axis=1)
    groups = [y[cond == c] for c in range(X_oh.shape[1]) if (cond == c).sum() > 1]
    best   = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i], groups[j]
            ps   = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            if ps > 1e-10: best = max(best, abs(np.mean(a) - np.mean(b)) / ps)
    return best if best > 0 else np.nan


CASES = [
    dict(
        col=0, ens_idx=6,
        sup_group='cue_visible',          sup_label='Cue visible',
        our_group='frame_raw_500msMedian', our_label='Speed',
        panel_detect='A', panel_explain='C',
    ),
    dict(
        col=1, ens_idx=22,
        sup_group='upcoming_choice',       sup_label='Upcoming choice',
        our_group='head_angle',            our_label='Head angle',
        panel_detect='B', panel_explain='D',
    ),
]

fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.5))
apply_style(fig, [ax for row in axes for ax in row])

C_WIN  = '#1E88E5'   # our variable wins
C_LOSE = '#AAAAAA'   # supervisor variable wins

for case in CASES:
    col      = case['col']
    e_idx    = case['ens_idx']
    sup_g    = group_names.index(case['sup_group'])
    our_g    = group_names.index(case['our_group'])
    sup_cols = feat_idx[case['sup_group']]
    our_cols = feat_idx[case['our_group']]
    valid_s  = [s for s in range(len(sessions)) if r2_all[s, e_idx] >= R2_THRESH]

    rows = []
    for s_idx in valid_s:
        sd   = ds[sessions[s_idx]]
        X    = np.concatenate([sd['data'][t]            for t in sd['data']], axis=0)
        y    = np.concatenate([sd['labels'][t][:, e_idx] for t in sd['data']])

        cd_sup  = cohens_d_categorical(X[:, sup_cols], y)
        gpv_sup = gpv[s_idx, e_idx, sup_g]
        gpv_our = gpv[s_idx, e_idx, our_g]

        # r(our_variable, sup_condition)
        our_vals  = X[:, our_cols].ravel()
        sup_cond  = np.argmax(X[:, sup_cols], axis=1).astype(float)
        r_our_sup, _ = spearmanr(our_vals, sup_cond)

        rows.append(dict(
            s_idx=s_idx, cd_sup=cd_sup,
            gpv_sup=gpv_sup, gpv_our=gpv_our,
            r_our_sup=r_our_sup,
        ))

    df = pd.DataFrame(rows).dropna(subset=['cd_sup', 'gpv_sup', 'gpv_our'])
    n  = len(df)
    colors = [C_WIN if row.gpv_our > row.gpv_sup else C_LOSE for _, row in df.iterrows()]
    n_wins = (df.gpv_our > df.gpv_sup).sum()

    # ── Row 1: Detection scatter ──────────────────────────────────────────────
    ax_d = axes[0, col]
    ax_d.scatter(df.cd_sup, df.gpv_sup, c=colors, s=55,
                 alpha=0.85, zorder=3, edgecolors='white', linewidths=0.4)
    for _, row in df.iterrows():
        ax_d.annotate(f"S{int(row.s_idx)+1:02d}", (row.cd_sup, row.gpv_sup),
                      fontsize=5.5, xytext=(3, 2), textcoords='offset points',
                      color='#444')
    rho, pval = spearmanr(df.cd_sup, df.gpv_sup)
    if n >= 4:
        z  = np.polyfit(df.cd_sup, df.gpv_sup, 1)
        xr = np.linspace(df.cd_sup.min(), df.cd_sup.max(), 50)
        ax_d.plot(xr, np.poly1d(z)(xr), '--', color='#555', lw=1.0, alpha=0.6)
    ax_d.set_xlabel(f"Cohen's d — {case['sup_label']}", fontsize=FONT.LABEL - 2)
    ax_d.set_ylabel(f"GPV — {case['sup_label']}", fontsize=FONT.LABEL - 2)
    ax_d.set_title(
        f"{'E07' if col==0 else 'E23'}  ·  We detect {case['sup_label'].lower()}  "
        f"(ρ={rho:+.2f}, p={pval:.2f})",
        fontsize=FONT.LABEL - 2, pad=3)
    ax_d.text(0.97, 0.05,
              f'{case["our_label"]} GPV > {case["sup_label"].split()[0]} GPV:\n'
              f'{n_wins}/{n} sessions',
              transform=ax_d.transAxes, ha='right', va='bottom',
              fontsize=FONT.ANNOTATION - 2, color=C_WIN)
    add_panel_label(ax_d, case['panel_detect'])

    # ── Row 2: Explanation — r(our_var, sup_condition) ────────────────────────
    ax_e = axes[1, col]
    r_vals = df.r_our_sup.dropna().values
    n_pos  = (r_vals > 0).sum()
    colors2 = [C_WIN if r > 0 else C_LOSE for r in r_vals]

    ax_e.bar(range(len(r_vals)), sorted(r_vals, reverse=True),
             color=sorted(colors2, key=lambda c: (c != C_WIN)),
             alpha=0.80, width=0.7)
    ax_e.axhline(0, color='#555', lw=0.8, ls='--')
    ax_e.set_xlabel('Sessions (sorted by r)', fontsize=FONT.LABEL - 2)
    ax_e.set_ylabel(f"r({case['our_label']}, {case['sup_label'].split()[0]} cond.)",
                    fontsize=FONT.LABEL - 2)
    ax_e.set_title(
        f"{'E07' if col==0 else 'E23'}  ·  "
        f"{case['our_label']} predicts {case['sup_label'].lower()} "
        f"({n_pos}/{len(r_vals)} sessions r>0)",
        fontsize=FONT.LABEL - 2, pad=3)
    ax_e.set_xticks([])
    add_panel_label(ax_e, case['panel_explain'])

    print(f"{'E07' if col==0 else 'E23'}: detect ρ={rho:.3f} p={pval:.3f}  "
          f"explain {n_pos}/{len(r_vals)} r>0  our_wins={n_wins}/{n}")

add_footnote(fig,
    "A/B: when Cohen's d is high for supervisor's variable, our GPV is also high (detection).  "
    "C/D: Spearman r between our primary variable and supervisor's condition label per session "
    "— positive = our variable co-varies with theirs (explanation).  "
    "Blue = sessions where our variable has higher GPV than supervisor's.")

savefig_manifest(fig, 'e07_e23_covariate_validation.png', OUT_DIRS)
print('Saved e07_e23_covariate_validation.png')
