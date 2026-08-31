"""
E07/E23 validation, |IG| version (replaces the GPV/R2 per-session figure).

For each case (E07 x cue_visible, E23 x upcoming_choice):
  - analysis set = sessions where the supervisor variable has a behavioral
    effect (Cohen's d >= 0.1), sorted by d
  - bar = mean |IG| attribution to the supervisor feature group for that session
  - blue if |IG| >= IG_THRESH (model attributes to it directly), else grey

Output: e07_e23_bucket_ig.png
"""
import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, '..')
sys.path.insert(0, root)
from utils.figure_style import FIG, FONT, apply_style, add_footnote, add_panel_label, savefig_manifest

mdir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [os.path.join(root, 'outputs', 'ablation_vs_attribution'),
            '/mnt/c/Users/amits/Desktop']

with open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())
with open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb') as f:
    sg = pickle.load(f)
group_names = [g[0] for g in sg]
feat_idx    = {g: cols for g, cols in sg}

ig_all = np.load(os.path.join(mdir, 'importance_ig_semantic.npy'))
r2_all = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)

D_THRESH  = 0.1
IG_THRESH = 0.05

def cd_categorical(X_oh, y):
    cond = np.argmax(X_oh, axis=1); best = 0.0
    for i in range(X_oh.shape[1]):
        for j in range(i + 1, X_oh.shape[1]):
            a, b = y[cond == i], y[cond == j]
            if len(a) < 2 or len(b) < 2: continue
            ps = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            if ps > 1e-10:
                best = max(best, abs(np.mean(a) - np.mean(b)) / ps)
    return best

CASES = [
    dict(ens_idx=6,  label='E07', sup_group='cue_visible',     sup_label='Cue Visible',   panel='C', color='#1F77B4'),
    dict(ens_idx=22, label='E23', sup_group='upcoming_choice', sup_label='Up. Choice',    panel='D', color='#1F77B4'),
]

fig, axes = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig, list(axes))

for case, ax in zip(CASES, axes):
    e_idx    = case['ens_idx']
    sup_g    = group_names.index(case['sup_group'])
    sup_cols = feat_idx[case['sup_group']]
    valid    = [s for s in range(len(sessions)) if r2_all[s, e_idx] >= 0.01]

    rows = []
    for s_idx in valid:
        sd = ds[sessions[s_idx]]
        X  = np.concatenate([sd['data'][t]             for t in sd['data']], axis=0)
        y  = np.concatenate([sd['labels'][t][:, e_idx] for t in sd['data']])
        cd = cd_categorical(X[:, sup_cols], y)
        ig = float(ig_all[s_idx, e_idx, sup_g])
        rows.append((s_idx, cd, ig))

    analysis = sorted([r for r in rows if r[1] >= D_THRESH], key=lambda r: -r[1])
    igs   = np.array([r[2] for r in analysis])
    n_hi  = int((igs >= IG_THRESH).sum())
    colors = [case['color'] if v >= IG_THRESH else '#CCCCCC' for v in igs]
    x = np.arange(len(analysis))

    ax.bar(x, igs, color=colors, alpha=0.85, width=0.7, zorder=3)
    ax.axhline(IG_THRESH, color='#444', lw=1.0, ls='--', alpha=0.8, zorder=2,
               label=f'|IG| = {IG_THRESH}')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{analysis[i][0]+1:02d}' for i in range(len(analysis))],
                       rotation=90, fontsize=FONT.TICK - 3)
    ax.set_ylabel(f'|IG|({case["sup_label"]})', fontsize=FONT.LABEL - 1)
    ax.set_title(f"{case['label']} ({case['sup_label']}): {n_hi}/{len(analysis)} "
                 rf"$\geq$ {IG_THRESH}", fontsize=FONT.LABEL - 1)
    ax.legend(fontsize=FONT.LEGEND - 1, frameon=False, loc='upper right')
    add_panel_label(ax, case['panel'])
    print(f"{case['label']}: analysis set n={len(analysis)}, |IG|>={IG_THRESH}: {n_hi}")

add_footnote(fig,
    'Analysis set: sessions with Cohen\'s d >= 0.1 (supervisor variable has a behavioral effect), '
    'sorted by d. Blue: mean |IG| to the supervisor feature >= 0.05 (model attributes directly).')
savefig_manifest(fig, 'e07_e23_bucket_ig.png', OUT_DIRS)
print('Saved e07_e23_bucket_ig.png')
