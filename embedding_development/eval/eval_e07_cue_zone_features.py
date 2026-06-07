#!/usr/bin/env python3
"""
eval_e07_cue_zone_features.py  —  S70

In the cue zone, what behavioral features actually differ between Cue 1 and Cue 2?
Shows violin plots (pooled data + per-session means) for the three strongest
co-varying features: Fwd Speed, Head Angle, and Rot. Vel.

This explains WHY the model uses these proxies: they co-vary with cue identity
in the cue zone, so the model can exploit them instead of cue_visible or position.

Output: outputs/ablation_vs_attribution/e07_cue_zone_features.png
"""
import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import (
    FIG, DPI, FONT, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
mdir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [
    os.path.join(root, 'outputs', 'ablation_vs_attribution'),
    '/mnt/c/Users/amits/Desktop',
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

E_IDX     = 6
R2_THRESH = 0.01

C1 = '#FF7F0E'   # Cue 1
C2 = '#D62728'   # Cue 2

ds       = pickle.load(open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb'))
sessions = list(ds.keys())
sg       = pickle.load(open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb'))
feat_idx = {g: cols for g, cols in sg}
r2_all   = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)

cue_cols = feat_idx['cue_visible']

valid = [s for s in range(len(sessions)) if r2_all[s, E_IDX] >= R2_THRESH]

# Top 3 features by eta² (from eval_e07_cue_zone_attribution results)
FEATURES = [
    ('frame_raw_500msMedian',           'Fwd Speed'),
    ('head_angle',                       'Head Angle'),
    ('frame_YawPitch_abs_vel_sum_500msMedian', 'Rot. Vel.'),
]

# ── Collect pooled data and per-session means ──────────────────────────────────
pool  = {f: {1: [], 2: []} for f, _ in FEATURES}   # f → cue → list of values
smean = {f: [] for f, _ in FEATURES}                # f → list of (m1, m2) per session

for s in valid:
    sd   = ds[sessions[s]]
    X    = np.concatenate([sd['data'][t] for t in sd['data']], axis=0)
    cond = np.argmax(X[:, cue_cols], axis=1)
    cz   = cond > 0
    if cz.sum() < 30:
        continue
    X_cz = X[cz]; cond_cz = cond[cz]

    for fname, _ in FEATURES:
        col   = feat_idx[fname][0]
        vals  = X_cz[:, col]
        v1    = vals[cond_cz == 1]
        v2    = vals[cond_cz == 2]
        if len(v1) < 5 or len(v2) < 5:
            continue
        pool[fname][1].extend(v1.tolist())
        pool[fname][2].extend(v2.tolist())
        smean[fname].append((np.mean(v1), np.mean(v2)))

# ── Figure: 3 panels ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIG.FULL)
apply_style(fig, list(axes))

rng = np.random.default_rng(0)

for ax, (fname, label) in zip(axes, FEATURES):
    d1 = np.array(pool[fname][1])
    d2 = np.array(pool[fname][2])

    # Violin
    parts = ax.violinplot([d1, d2], positions=[0, 1],
                          showmedians=True, showextrema=False, widths=0.55)
    for i, (pc, clr) in enumerate(zip(parts['bodies'], [C1, C2])):
        pc.set_facecolor(clr); pc.set_alpha(0.45)
    parts['cmedians'].set_color('#333'); parts['cmedians'].set_linewidth(1.2)

    # Jittered per-session means
    sm = smean[fname]
    if sm:
        sm = np.array(sm)
        jit1 = rng.uniform(-0.08, 0.08, len(sm))
        jit2 = rng.uniform(-0.08, 0.08, len(sm))
        ax.scatter(0 + jit1, sm[:, 0], s=18, color=C1, alpha=0.70, zorder=3,
                   edgecolors='white', linewidths=0.3)
        ax.scatter(1 + jit2, sm[:, 1], s=18, color=C2, alpha=0.70, zorder=3,
                   edgecolors='white', linewidths=0.3)
        # Connecting lines (faint)
        for m1, m2, j1, j2 in zip(sm[:, 0], sm[:, 1], jit1, jit2):
            ax.plot([0 + j1, 1 + j2], [m1, m2], color='#999', lw=0.5, alpha=0.50)

        # t-test on session means
        t, p = stats.ttest_rel(sm[:, 1], sm[:, 0])
        star = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else np.max([d1.max(), d2.max()])
        ax.text(0.5, 0.93, star, transform=ax.transAxes, ha='center',
                fontsize=FONT.ANNOTATION, color='#222')
        ax.text(0.5, 0.86, f'p={p:.3f}', transform=ax.transAxes, ha='center',
                fontsize=FONT.ANNOTATION - 2, color='#555')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Cue 1', 'Cue 2'], fontsize=FONT.TICK)
    ax.set_ylabel(f'{label} (z-scored)', fontsize=FONT.LABEL - 1)
    ax.set_title(f'{label} in the cue zone', fontsize=FONT.LABEL - 1, pad=3)
    ax.tick_params(labelsize=FONT.TICK)

add_panel_label(axes[0], 'A')
add_panel_label(axes[1], 'B')
add_panel_label(axes[2], 'C')

n_sess = max(len(smean[f]) for f, _ in FEATURES)
add_footnote(fig,
    f'E07 (ensemble 6), {n_sess} sessions.  Cue-zone timepoints only.  '
    'Violins: pooled data across sessions.  Dots: per-session means.  '
    'Lines connect session means across conditions.  Stats: paired t-test on session means.')

savefig_manifest(fig, 'e07_cue_zone_features.png', OUT_DIRS)
print('Saved e07_cue_zone_features.png')
