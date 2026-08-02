#!/usr/bin/env python3
"""
gen_head_angle_attribution_summary.py

Single-panel figure: mean GPV and mean |IG| per feature group,
averaged across all valid (R²≥0.05) MLP pairs (29 sessions × 23 ensembles).
Sorted by GPV descending. Head angle is #1 by both metrics.

Placed as S51 in the head angle section to motivate focusing on
head angle as the case study.

Output: head_angle_attribution_summary.png  (9.5" × 3.5")
"""
import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import (
    FIG, DPI, FONT, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, savefig_manifest,
)

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, '..')
mdir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [mdir, '/mnt/c/Users/amits/Desktop/']

with open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb') as f:
    sg = pickle.load(f)
group_names = [g[0] for g in sg]
tick_labels = [FEATURE_NAMES_SHORT.get(g, g) for g in group_names]
n_groups    = len(group_names)

ig     = np.load(os.path.join(mdir, 'importance_ig_semantic.npy'))
r2_all = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)    # (29,23)

# Valid pairs: R²≥0.05
valid     = r2_all >= 0.05                          # (29,23) bool
valid_exp = valid[:, :, np.newaxis]                 # broadcast over groups
ig_masked = np.where(valid_exp, ig, np.nan)

ig_mean = np.nanmean(ig_masked.reshape(-1, n_groups), axis=0)

n_valid = valid.sum()

# Sort by IG descending
order = np.argsort(ig_mean)[::-1]

ha_g = group_names.index('head_angle')
C_IG = '#FF7F0E'   # orange for IG bars

width = 0.55
x     = np.arange(n_groups)

fig, ax = plt.subplots(figsize=(9.5, 3.5))
apply_style(fig, ax)

ax.bar(x, ig_mean[order], width, color=C_IG, alpha=0.90, label='Mean |IG|')

# Box around head angle
ha_pos = int(np.where(order == ha_g)[0][0])
pad    = 0.05
y_top  = ig_mean[ha_g] * 1.18
y_bot  = min(0, ig_mean[ha_g]) - 0.002
from matplotlib.patches import FancyBboxPatch
ax.add_patch(FancyBboxPatch(
    (ha_pos - width/2 - pad, y_bot),
    width + 2 * pad, y_top - y_bot,
    boxstyle='round,pad=0.01', linewidth=1.5,
    edgecolor='#333333', facecolor='none', zorder=5))

ax.set_xticks(x)
ax.set_xticklabels([tick_labels[g] for g in order],
                   rotation=60, ha='right', fontsize=FONT.TICK - 2)
ax.set_ylabel('Mean |IG| (valid pairs)', fontsize=FONT.LABEL - 1)
ax.axhline(0, color='#888', lw=0.6, linestyle='--')
ax.legend(fontsize=FONT.LEGEND, frameon=False, loc='upper right')

add_footnote(fig,
    f'Mean across {n_valid} valid pairs (R²≥0.05, MLP).  '
    'Head angle (boxed) is the top-attributed feature by IG.')

savefig_manifest(fig, 'head_angle_attribution_summary.png', OUT_DIRS)
print('Saved head_angle_attribution_summary.png')
