#!/usr/bin/env python3
"""
generate_gpv_task_ensembles.py

Two-panel figure for S38 supplement:
  Left:  GPV attribution profiles for the 4 ensembles most sensitive to
         cue_visible + upcoming_choice (bar chart, task variables highlighted)
  Right: Full GPV heatmap with those 4 ensembles outlined in orange

Addresses supervisor feedback: heatmap nearly white because task variables
have low absolute GPV; this figure makes the task-sensitive ensembles explicit.
"""
import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import (FONT, apply_style, add_footnote,
                                 savefig_manifest, FEATURE_NAMES_SHORT)

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, '..')
mdir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
cdir = os.path.join(root, 'outputs', 'cebra_comparison')

gpv = np.load(os.path.join(mdir, 'importance_global_pv_semantic_seed42.npy'),
              allow_pickle=True)   # (29, 23, 11)
r2  = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)  # (29, 23)

with open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb') as f:
    sem = pickle.load(f)
group_names  = [g[0] for g in sem]
ytick_labels = [FEATURE_NAMES_SHORT.get(g, g) for g in group_names]

ens_mean_gpv    = np.nanmean(gpv, axis=0)                                    # (23, 11)
ensemble_mean_r2 = np.nanmean(np.where(r2 >= 0.01, r2, np.nan), axis=0)     # (23,)

CUE_IDX, CHOICE_IDX = 7, 8
task_score = ens_mean_gpv[:, CUE_IDX] + ens_mean_gpv[:, CHOICE_IDX]
top4 = np.argsort(np.where(~np.isnan(task_score), task_score, -1))[::-1][:4]

print(f'Top 4 task-sensitive ensembles (0-indexed): {top4}')
for e in top4:
    print(f'  E{e+1:02d}: cue={ens_mean_gpv[e,CUE_IDX]:.4f}  '
          f'choice={ens_mean_gpv[e,CHOICE_IDX]:.4f}')

# ── Figure — heatmap only ─────────────────────────────────────────────────────
from utils.figure_style import FIG
fig, ax_heat = plt.subplots(1, 1, figsize=FIG.FULL)
apply_style(fig, ax_heat)

r2_order = np.argsort(ensemble_mean_r2)[::-1]
hm       = ens_mean_gpv[r2_order, :].T        # (11, 23)
xlabels  = [f'E{i+1:02d}' for i in r2_order]
vmax     = float(np.nanpercentile(hm[~np.isnan(hm)], 97)) if np.any(~np.isnan(hm)) else 1.0

im = ax_heat.imshow(hm, aspect='auto', cmap='Blues',
                    norm=mcolors.Normalize(0, vmax), interpolation='nearest')
ax_heat.set_xticks(np.arange(23))
ax_heat.set_yticks(np.arange(11))
ax_heat.set_xticklabels(xlabels, fontsize=5, rotation=45, ha='right')
ax_heat.set_yticklabels(ytick_labels, fontsize=FONT.TICK - 1)
ax_heat.set_xlabel('Ensemble (sorted by R²)', fontsize=FONT.LABEL)
ax_heat.set_title('GPV Heatmap (sorted by R²)', fontsize=FONT.LABEL)

plt.colorbar(im, ax=ax_heat, label='GPV  (ΔR²)', fraction=0.02, pad=0.01)

add_footnote(fig,
    'Mean GPV per (feature group, ensemble) across all sessions with R²≥0.01.  '
    'Sorted by ensemble mean R².  '
    'Low absolute GPV for task features (cue, choice) reflects sparse event density.')

OUT_DIRS = [mdir, cdir, '/mnt/c/Users/amits/Desktop']
savefig_manifest(fig, 'gpv_task_ensembles.png', OUT_DIRS)
print('Done.')
