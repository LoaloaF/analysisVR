#!/usr/bin/env python3
"""
eval_e07_joint_captured.py

Shows that even though the model does not directly attribute to the joint
position × cue signal in E07, it captures the same information through speed
and head angle — which co-vary with the cue condition.

Two-panel figure:
  A — GPV comparison: cue, position, joint (cue+pos), speed, head angle
      Shows speed & head angle dominate; joint GPV ≈ additive (no synergy)
  B — Scatter: η²(speed, cue condition) vs GPV(speed)/R² per session
      Same format as S64 bucket scatter — both high → model captures the signal
"""
import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import (
    FIG, DPI, FONT, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

root   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
mdir   = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [
    os.path.join(root, 'outputs', 'ablation_vs_attribution'),
    '/mnt/c/Users/amits/Desktop',
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

JOINT_GPV_MEAN = 0.0082   # pre-computed by eval_e07_joint_gpv.py
ADDITIVE_MEAN  = 0.0083

# ── Load ──────────────────────────────────────────────────────────────────────
ds       = pickle.load(open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb'))
sessions = list(ds.keys())
sg       = pickle.load(open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb'))
group_names = [g[0] for g in sg]
feat_idx    = {g: cols for g, cols in sg}
gpv    = np.load(os.path.join(mdir, 'importance_global_pv_semantic.npy'))
r2_all = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)

E_IDX    = 6
R2_THRESH = 0.01
valid    = [s for s in range(len(sessions)) if r2_all[s, E_IDX] >= R2_THRESH]

cue_g   = group_names.index('cue_visible')
pos_g   = group_names.index('frame_position')
spd_g   = group_names.index('frame_raw_500msMedian')
ang_g   = group_names.index('head_angle')
cue_cols = feat_idx['cue_visible']
spd_cols = feat_idx['frame_raw_500msMedian']

def eta_sq(vals, X_oh):
    cond = np.argmax(X_oh, axis=1); gm = vals.mean()
    ss_tot = ((vals - gm) ** 2).sum()
    if ss_tot < 1e-12: return 0.0
    return float(sum(len(vals[cond==c]) * (vals[cond==c].mean() - gm)**2
                     for c in np.unique(cond) if (cond==c).sum() > 0) / ss_tot)

# ── Per-session data ───────────────────────────────────────────────────────────
rows = []
for s in valid:
    sd  = ds[sessions[s]]
    X   = np.concatenate([sd['data'][t] for t in sd['data']], axis=0)
    r2  = r2_all[s, E_IDX]
    gv  = gpv[s, E_IDX]
    spd = X[:, spd_cols].ravel()
    e2  = eta_sq(spd, X[:, cue_cols])
    rows.append(dict(
        s=s, r2=r2,
        gpv_cue=gv[cue_g], gpv_pos=gv[pos_g],
        gpv_spd=gv[spd_g], gpv_ang=gv[ang_g],
        gpv_spd_frac=gv[spd_g] / r2 if r2 > 0 else 0,
        eta2_spd_cue=e2,
    ))

# Aggregate means
mean_cue = np.mean([r['gpv_cue'] for r in rows])
mean_pos = np.mean([r['gpv_pos'] for r in rows])
mean_spd = np.mean([r['gpv_spd'] for r in rows])
mean_ang = np.mean([r['gpv_ang'] for r in rows])

# ── Figure ────────────────────────────────────────────────────────────────────
C_CUE   = '#FF7F0E'
C_POS   = '#8C564B'
C_JOINT = '#9467BD'
C_ADD   = '#CCCCCC'
C_SPD   = '#2CA02C'
C_ANG   = '#1F77B4'

fig, (ax_bar, ax_scat) = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig, [ax_bar, ax_scat])

# ── Panel A: GPV bar chart ─────────────────────────────────────────────────────
bar_labels = ['Cue', 'Position', 'Joint GPV', 'Fwd Speed', 'Head Angle']
bar_vals   = [mean_cue, mean_pos, JOINT_GPV_MEAN, mean_spd, mean_ang]
bar_colors = [C_CUE,    C_POS,    C_JOINT,        C_SPD,    C_ANG]
x = np.arange(len(bar_labels))

ax_bar.barh(x, bar_vals, color=bar_colors, alpha=0.85, height=0.6)
ax_bar.axvline(0, color='#555', lw=0.7, ls='--')
ax_bar.text(JOINT_GPV_MEAN * 1.06, 2, '≈ cue+pos\n(no synergy)',
            va='center', fontsize=FONT.ANNOTATION - 2, color='#666')

ax_bar.set_yticks(x)
ax_bar.set_yticklabels(bar_labels, fontsize=FONT.TICK - 1)
ax_bar.invert_yaxis()
ax_bar.set_xlabel('Mean GPV (ΔR²)', fontsize=FONT.LABEL - 1)
ax_bar.set_title(f'E07 — GPV comparison  (n={len(rows)} sessions)',
                 fontsize=FONT.LABEL - 1, pad=3)
add_panel_label(ax_bar, 'A')

# ── Panel B: scatter η²(speed, cue) vs GPV(speed)/R² ─────────────────────────
xs = np.array([r['eta2_spd_cue']   for r in rows])
ys = np.array([r['gpv_spd_frac']   for r in rows])
s_idxs = [r['s'] for r in rows]

ax_scat.scatter(xs, ys, c=C_SPD, s=65, alpha=0.85, zorder=3,
                edgecolors='white', linewidths=0.4)
for xi, yi, si in zip(xs, ys, s_idxs):
    ax_scat.annotate(f'S{si+1:02d}', (xi, yi), fontsize=5.5,
                     xytext=(3, 3), textcoords='offset points', color='#444')

# Reference lines
xlim = max(xs) * 1.18
ylim = max(ys) * 1.12
ax_scat.axhline(0.10, color='#444', lw=0.9, ls='--', alpha=0.6)
ax_scat.axvline(0.05, color='#888', lw=0.9, ls=':', alpha=0.5)
ax_scat.text(xlim * 0.02, 0.10 + ylim * 0.02, 'GPV/R² = 10%',
             fontsize=FONT.ANNOTATION - 2, color='#444', va='bottom')
ax_scat.text(0.05 + xlim * 0.01, ylim * 0.02, 'η² = 0.05',
             fontsize=FONT.ANNOTATION - 2, color='#888', va='bottom')
ax_scat.set_xlim(0, xlim); ax_scat.set_ylim(0, ylim)
ax_scat.set_xlabel('η²(Fwd Speed, cue condition)', fontsize=FONT.LABEL - 1)
ax_scat.set_ylabel('GPV(Fwd Speed) / R²', fontsize=FONT.LABEL - 1)
ax_scat.set_title('Speed co-varies with cue AND model attributes to it',
                  fontsize=FONT.LABEL - 1, pad=3)
add_panel_label(ax_scat, 'B')

add_footnote(fig,
    'E07 (ensemble 6).  '
    'A: mean GPV across all valid sessions.  Joint GPV pre-computed by eval_e07_joint_gpv.py.  '
    'B: per-session scatter — speed co-varies with cue condition (x) and the model attributes to it (y).  '
    'Upper-right: signal captured through speed despite no direct cue/position attribution.')

savefig_manifest(fig, 'e07_joint_captured.png', OUT_DIRS)
print('Saved e07_joint_captured.png')
