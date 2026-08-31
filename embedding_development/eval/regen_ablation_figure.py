#!/usr/bin/env python3
"""
regen_ablation_figure.py

Regenerate combined_attribution_vs_ablation_r2.png from saved CSVs
(no model retraining needed).  Outputs at FIG.FULL = 9.5 × 4.2" at DPI=200
with proper FONT constants — replacing the old 12×10" figure that was
unreadable when scaled into a slide.

Layout: 1 row × 2 panels
  Left  — E07 × cue_visible    (IG attribution vs Ablation R²)
  Right — E23 × upcoming_choice (IG attribution vs Ablation R²)
Each panel overlays MLP, TempConv-Cont, TempConv-Pred as different markers.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import (
    FIG, DPI, FONT, apply_style, add_footnote, savefig_manifest,
)

base    = os.path.dirname(os.path.abspath(__file__))
root    = os.path.join(base, '..')
abl_dir = os.path.join(root, 'outputs', 'ablation_vs_attribution')
mdir    = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [mdir, '/mnt/c/Users/amits/Desktop']

df_e07 = pd.read_csv(os.path.join(abl_dir, 'E07_cue_visible_ablation.csv'))
df_e23 = pd.read_csv(os.path.join(abl_dir, 'E23_upcoming_choice_ablation.csv'))

PAIRS = [
    ('E07 × cue_visible',     df_e07),
    ('E23 × upcoming_choice', df_e23),
]

MODEL_STYLES = [
    ('mlp',  'MLP',           '#4CAF50', 'o', 0.00),
    ('ceb',  'TempConv-Cont', '#2196F3', 's', 0.12),
    ('pred', 'TempConv-Pred', '#FF9800', '^', 0.24),
]

TARGET_COL   = 'abl_r2'
TARGET_LABEL = 'Ablation R²'
ATTR_COL     = 'ig'
ATTR_LABEL   = 'IG attribution'

fig, axes = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig, axes)

for ax, (title, df) in zip(axes, PAIRS):
    for model, label, mcolor, mk, yoff in MODEL_STYLES:
        col = f'{model}_{ATTR_COL}'
        if col not in df.columns:
            continue
        sub = df[[TARGET_COL, col]].dropna()
        if len(sub) < 3:
            continue
        rho, p = spearmanr(sub[col], sub[TARGET_COL])
        ax.scatter(sub[col], sub[TARGET_COL],
                   color=mcolor, marker=mk, s=55, alpha=0.85,
                   zorder=3, label=label)
        ax.text(0.05, 0.96 - yoff,
                f'{label}: ρ={rho:+.3f}  p={p:.2f}',
                transform=ax.transAxes,
                fontsize=FONT.ANNOTATION - 1,
                color=mcolor, va='top')

    ax.axhline(0, color='#888', lw=0.8, ls='--')
    ax.set_xlabel(f'{ATTR_LABEL}', fontsize=FONT.LABEL - 1)
    ax.set_ylabel(TARGET_LABEL,    fontsize=FONT.LABEL - 1)
    ax.set_title(title, fontsize=FONT.LABEL - 1, fontweight='bold', pad=5)

axes[0].legend(fontsize=FONT.LEGEND - 1, frameon=False, loc='lower right')

add_footnote(fig,
    f"Ablation R²: MLP trained only on target feature's one-hot columns; "
    f"attribution = IG from full MLP/TempConv-Cont/TempConv-Pred models")

savefig_manifest(fig, 'combined_attribution_vs_ablation_r2.png', OUT_DIRS)
print('Generated combined_attribution_vs_ablation_r2.png')
