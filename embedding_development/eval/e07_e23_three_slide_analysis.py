#!/usr/bin/env python3
"""
e07_e23_three_slide_analysis.py

Generates three figures for the E07/E23 case study slides:

  Fig 1: supervisors_signal.png
    "The supervisor's variable does have a neural signal in some sessions"
    Bar chart of Cohen's d per session for cue_visible (E07) and upcoming_choice (E23)

  Fig 2: gpv_comparison.png
    "Our model detects it too — and our primary variable has far higher attribution"
    Per-session bar pairs: GPV(supervisor's var) vs GPV(our var)

  Fig 3: covariate_explanation.png
    "Why: our variable co-varies with theirs (E07) / has independent stronger tuning (E23)"
    E07: per-session r(speed, cue_condition) — always positive (median=0.39)
    E23: per-session r(head_angle, choice_condition) — weak, shown alongside
         GPV(head_angle) dominating GPV(choice) 30×

All figures: 9.5" × 4.2" (FIG.FULL), two panels each (E07 left, E23 right).
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

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, '..')
mdir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [
    os.path.join(root, 'outputs', 'ablation_vs_attribution'),
    '/mnt/c/Users/amits/Desktop',
]
for d in OUT_DIRS: os.makedirs(d, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
with open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())

with open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb') as f:
    sg = pickle.load(f)
group_names = [g[0] for g in sg]
feat_idx    = {g: cols for g, cols in sg}

gpv    = np.load(os.path.join(mdir, 'importance_global_pv_semantic.npy'))
r2_all = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)

CASES = [
    dict(col=0, ens_idx=6,  panel='A',
         sup_group='cue_visible',          sup_label='Cue visible',
         our_group='frame_raw_500msMedian', our_label='Speed',
         C_OUR='#2CA02C', C_SUP='#FF7F0E'),
    dict(col=1, ens_idx=22, panel='B',
         sup_group='upcoming_choice',       sup_label='Upcoming choice',
         our_group='head_angle',            our_label='Head angle',
         C_OUR='#1F77B4', C_SUP='#FF7F0E'),
]

def cd_categorical(X_oh, y):
    cond = np.argmax(X_oh, axis=1); best = 0.0
    for i in range(X_oh.shape[1]):
        for j in range(i + 1, X_oh.shape[1]):
            a, b = y[cond == i], y[cond == j]
            if len(a) < 2 or len(b) < 2: continue
            ps = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            if ps > 1e-10: best = max(best, abs(np.mean(a) - np.mean(b)) / ps)
    return best


# ── Collect data ──────────────────────────────────────────────────────────────
case_data = []
for case in CASES:
    e_idx    = case['ens_idx']
    sup_g    = group_names.index(case['sup_group'])
    our_g    = group_names.index(case['our_group'])
    sup_cols = feat_idx[case['sup_group']]
    our_cols = feat_idx[case['our_group']]
    valid    = [s for s in range(len(sessions)) if r2_all[s, e_idx] >= 0.01]

    rows = []
    for s_idx in valid:
        sd   = ds[sessions[s_idx]]
        X    = np.concatenate([sd['data'][t]             for t in sd['data']], axis=0)
        y    = np.concatenate([sd['labels'][t][:, e_idx] for t in sd['data']])
        cd   = cd_categorical(X[:, sup_cols], y)
        gv_s = gpv[s_idx, e_idx, sup_g]
        gv_o = gpv[s_idx, e_idx, our_g]
        cond = np.argmax(X[:, sup_cols], axis=1).astype(float)
        r, _ = spearmanr(X[:, our_cols].ravel(), cond)
        rows.append(dict(s_idx=s_idx, cd=cd, gpv_sup=gv_s, gpv_our=gv_o, r_explain=r))

    case_data.append(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Supervisor's signal: Cohen's d per session
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig1, list(axes1))

for case, rows, ax in zip(CASES, case_data, axes1):
    cds     = np.array([r['cd'] for r in rows])
    s_idxs  = [r['s_idx'] for r in rows]
    order   = np.argsort(cds)[::-1]
    x       = np.arange(len(rows))
    colors  = [case['C_SUP']] * len(rows)

    ax.bar(x, cds[order], color=colors, alpha=0.80, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s_idxs[i]+1:02d}' for i in order],
                       rotation=90, fontsize=max(5, FONT.TICK - 4))
    ax.set_ylabel("Cohen's d", fontsize=FONT.LABEL - 1)
    ax.set_title(
        f"{'E07' if case['col']==0 else 'E23'}  —  {case['sup_label']} signal "
        f"(median={np.median(cds):.2f}, max={cds.max():.2f})",
        fontsize=FONT.LABEL - 1, pad=3)
    ax.axhline(np.median(cds), color='#555', lw=0.8, ls='--', alpha=0.7,
               label=f'median = {np.median(cds):.2f}')
    ax.legend(fontsize=FONT.LEGEND - 1, frameon=False)
    add_panel_label(ax, case['panel'])

add_footnote(fig1,
    "Cohen's d computed as max pairwise effect between conditions.  "
    "Sorted descending per ensemble.  "
    "Dashed = session median.")
savefig_manifest(fig1, 'e07_e23_supervisors_signal.png', OUT_DIRS)
print('Saved e07_e23_supervisors_signal.png')
plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — GPV comparison: our variable vs supervisor's variable
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig2, list(axes2))

for case, rows, ax in zip(CASES, case_data, axes2):
    cds      = np.array([r['cd']      for r in rows])
    gpv_sups = np.array([r['gpv_sup'] for r in rows])
    gpv_ours = np.array([r['gpv_our'] for r in rows])
    s_idxs   = [r['s_idx'] for r in rows]

    # Sort by Cohen's d descending to align with Fig 1
    order = np.argsort(cds)[::-1]
    x     = np.arange(len(rows))
    w     = 0.38

    ax.bar(x - w/2, gpv_sups[order], w, color=case['C_SUP'], alpha=0.80,
           label=case['sup_label'])
    ax.bar(x + w/2, gpv_ours[order], w, color=case['C_OUR'], alpha=0.80,
           label=case['our_label'])
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s_idxs[i]+1:02d}' for i in order],
                       rotation=90, fontsize=max(5, FONT.TICK - 4))
    ax.set_ylabel('GPV (ΔR²)', fontsize=FONT.LABEL - 1)
    ratio = np.mean(gpv_ours) / np.mean(gpv_sups) if np.mean(gpv_sups) > 0 else float('inf')
    ax.set_title(
        f"{'E07' if case['col']==0 else 'E23'}  —  "
        f"{case['our_label']} GPV is {ratio:.0f}× higher on average",
        fontsize=FONT.LABEL - 1, pad=3)
    ax.legend(fontsize=FONT.LEGEND - 1, frameon=False, loc='upper right')
    add_panel_label(ax, case['panel'])

add_footnote(fig2,
    'Sessions sorted by supervisor\'s Cohen\'s d (same order as Fig 1).  '
    'Our variable (green/blue) has consistently higher GPV than the supervisor\'s variable (orange).')
savefig_manifest(fig2, 'e07_e23_gpv_comparison.png', OUT_DIRS)
print('Saved e07_e23_gpv_comparison.png')
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Covariate explanation
#   E07: r(speed, cue_condition) per session — speed always co-varies with cue
#   E23: r(head_angle, choice_cond) per session — weaker, shown with GPV ratio
# ══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig3, list(axes3))

for case, rows, ax in zip(CASES, case_data, axes3):
    r_vals = np.array([r['r_explain'] for r in rows])
    s_idxs = [r['s_idx'] for r in rows]
    order  = np.argsort(r_vals)[::-1]
    x      = np.arange(len(rows))
    colors = [case['C_OUR'] if r > 0 else '#CCCCCC' for r in r_vals[order]]
    n_pos  = (r_vals > 0).sum()

    ax.bar(x, r_vals[order], color=colors, alpha=0.85, width=0.7)
    ax.axhline(0, color='#555', lw=0.8, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s_idxs[i]+1:02d}' for i in order],
                       rotation=90, fontsize=max(5, FONT.TICK - 4))
    ax.set_ylabel(f"r({case['our_label']}, {case['sup_label'].split()[0]} condition)",
                  fontsize=FONT.LABEL - 1)

    med_r = np.nanmedian(r_vals)
    ax.set_title(
        f"{'E07' if case['col']==0 else 'E23'}  —  "
        f"{case['our_label']} predicts {case['sup_label'].lower()}: "
        f"{n_pos}/{len(r_vals)} sessions r>0  (median={med_r:.2f})",
        fontsize=FONT.LABEL - 1, pad=3)
    ax.axhline(med_r, color=case['C_OUR'], lw=0.9, ls=':', alpha=0.8,
               label=f'median r={med_r:.2f}')
    ax.legend(fontsize=FONT.LEGEND - 1, frameon=False)
    add_panel_label(ax, case['panel'])

add_footnote(fig3,
    'Spearman r between our primary variable and the supervisor\'s condition label per session.  '
    'E07: speed co-varies with cue timing in 18/19 sessions (median r=0.39) '
    '→ the cue signal is captured through speed changes.  '
    'E23: head angle has weaker correlation with choice side (median r=0.03).')
savefig_manifest(fig3, 'e07_e23_covariate_explanation.png', OUT_DIRS)
print('Saved e07_e23_covariate_explanation.png')
plt.close(fig3)
