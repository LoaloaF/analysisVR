#!/usr/bin/env python3
"""
e07_e23_bucket_analysis.py

Three-figure set for the E07/E23 case study (bucket structure):

  Fig 1: "Where the supervisor has a point"
    Cohen's d per session for cue (E07) and choice (E23). Sessions above d=0.1 form the analysis set.

  Fig 2: "Bucket 1 — our model detects it directly"
    GPV(sup_var) per session (analysis set only). Sessions with GPV>=0.005 = direct detection.
    E07: 5/9. E23: 0/19 (all go to Bucket 2).

  Fig 3: "Bucket 2 — captured through co-variation (η²)"
    For sessions where GPV(sup_var)<0.005: η² of the best co-varying feature vs supervisor's condition.
    E07: 4 sessions (speed η²≈0.22-0.34). E23: 19 sessions (speed/head-vel η²≈0.03-0.30).
"""
import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
with open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())

with open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb') as f:
    sg = pickle.load(f)
group_names = [g[0] for g in sg]
feat_idx    = {g: cols for g, cols in sg}

ig_all = np.load(os.path.join(mdir, 'importance_ig_semantic.npy'))
r2_all = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)

# ── Helpers ───────────────────────────────────────────────────────────────────
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

def eta_sq(vals, X_oh):
    """η² of categorical condition (one-hot) predicting continuous variable."""
    cond    = np.argmax(X_oh, axis=1)
    gm      = vals.mean()
    ss_tot  = ((vals - gm) ** 2).sum()
    if ss_tot < 1e-12: return 0.0
    ss_bet  = sum(
        len(vals[cond == c]) * (vals[cond == c].mean() - gm) ** 2
        for c in np.unique(cond) if (cond == c).sum() > 0
    )
    return float(ss_bet / ss_tot)

SHORT = FEATURE_NAMES_SHORT  # canonical display names
D_THRESH  = 0.1     # "supervisor has a point" (Cohen's d)
IG_THRESH = 0.05    # "high IG attribution": absolute IG value per semantic group

# ── Case definitions ──────────────────────────────────────────────────────────
CASES = [
    dict(col=0, ens_idx=6,  label='E07', panel1='A', panel2='C', panel3='E',
         sup_group='cue_visible',          sup_label='Cue Visible',
         our_group='frame_raw_500msMedian', our_label='Fwd Speed',
         C_SUP='#FF7F0E', C_B1='#1F77B4', C_OUR='#2CA02C'),
    dict(col=1, ens_idx=22, label='E23', panel1='B', panel2='D', panel3='F',
         sup_group='upcoming_choice',      sup_label='Upcoming Choice',
         our_group='head_angle',           our_label='Head Angle',
         C_SUP='#FF7F0E', C_B1='#1F77B4', C_OUR='#1F77B4'),
]

# ── Collect data ──────────────────────────────────────────────────────────────
case_data = []
for case in CASES:
    e_idx    = case['ens_idx']
    sup_g    = group_names.index(case['sup_group'])
    sup_cols = feat_idx[case['sup_group']]
    valid    = [s for s in range(len(sessions)) if r2_all[s, e_idx] >= 0.01]

    our_g    = group_names.index(case['our_group'])
    rows = []
    for s_idx in valid:
        sd   = ds[sessions[s_idx]]
        X    = np.concatenate([sd['data'][t]              for t in sd['data']], axis=0)
        y    = np.concatenate([sd['labels'][t][:, e_idx]  for t in sd['data']])

        r2      = float(r2_all[s_idx, e_idx])
        cd      = cd_categorical(X[:, sup_cols], y)
        ig_sup  = float(ig_all[s_idx, e_idx, sup_g])
        ig_our  = float(ig_all[s_idx, e_idx, our_g])

        eta2s = {}
        for g_name in group_names:
            vals = X[:, feat_idx[g_name]].mean(axis=1)
            eta2s[g_name] = eta_sq(vals, X[:, sup_cols])

        ig_by_group = {g: float(ig_all[s_idx, e_idx, gi])
                       for gi, g in enumerate(group_names)}

        joint_best_g    = max(group_names,
                              key=lambda g: eta2s[g] * ig_by_group[g])
        joint_best_eta2 = eta2s[joint_best_g]
        joint_best_ig   = ig_by_group[joint_best_g]

        rows.append(dict(s_idx=s_idx, r2=r2, cd=cd,
                         ig_sup=ig_sup, ig_our=ig_our,
                         eta2s=eta2s, ig_by_group=ig_by_group,
                         joint_best_g=joint_best_g,
                         joint_best_eta2=joint_best_eta2,
                         joint_best_frac=joint_best_ig))

    case_data.append(rows)

    above = [r for r in rows if r['cd'] >= D_THRESH]
    b1    = [r for r in above if r['ig_sup'] >= IG_THRESH]
    b2    = [r for r in above if r['ig_sup'] <  IG_THRESH]
    print(f"{case['label']}: {len(rows)} sessions, {len(above)} d>={D_THRESH}, "
          f"high_ig={len(b1)}, low_ig={len(b2)}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Supervisor's variable: Cohen's d per session
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig1, list(axes1))

_all_cds = [r['cd'] for rows in case_data for r in rows]
shared_cd_ymax = max(_all_cds) * 1.15 if _all_cds else 1.0

for case, rows, ax in zip(CASES, case_data, axes1):
    rows_s = sorted(rows, key=lambda r: -r['cd'])
    cds    = np.array([r['cd']   for r in rows_s])
    s_idxs = [r['s_idx']         for r in rows_s]
    colors = [case['C_SUP'] if d >= D_THRESH else '#CCCCCC' for d in cds]
    x      = np.arange(len(rows_s))

    ax.bar(x, cds, color=colors, alpha=0.85, width=0.7)
    ax.axhline(D_THRESH, color='#444', lw=1.0, ls='--', alpha=0.8,
               label=f'd = {D_THRESH}  (analysis threshold)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s_idxs[i]+1:02d}' for i in range(len(rows_s))],
                       rotation=90, fontsize=max(5, FONT.TICK - 3))
    ax.set_ylim(0, shared_cd_ymax)
    ax.set_ylabel("Cohen's d", fontsize=FONT.LABEL - 1)
    n_above = sum(1 for d in cds if d >= D_THRESH)
    sup_short = SHORT.get(case['sup_group'], case['sup_label'])
    ax.set_title(
        f"{case['label']} ({sup_short}): d, "
        f"{n_above}/{len(rows_s)} ≥ {D_THRESH}",
        fontsize=FONT.LABEL - 1, pad=3)
    ax.legend(fontsize=FONT.LEGEND - 1, frameon=False)

add_footnote(fig1,
    f"Cohen's d: max pairwise effect between conditions.  "
    f"Orange bars (d ≥ {D_THRESH}) form the analysis set.  "
    f"Sorted descending.")
savefig_manifest(fig1, 'e07_e23_bucket_signal.png', OUT_DIRS)
print('Saved e07_e23_bucket_signal.png')
plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Pie chart: 3-way attribution breakdown per ensemble
# Cat 1: GPV(sup_var)/R² ≥ GPV_THRESH  (direct IG attribution)
# Cat 2: GPV(best co-var)/R² ≥ GPV_THRESH  AND  η²(best co-var, condition) ≥ ETA2_THRESH
# Cat 3: low both
# ══════════════════════════════════════════════════════════════════════════════
ETA2_THRESH = 0.05

CAT_COLORS = ['#1F77B4', '#2CA02C', '#CCCCCC', '#E08080']
CAT_NAMES  = [
    'High IG to\ntarget feature',
    'High IG to\nco-varying feature',
    'Valid model,\nunexplained',
    'Model not valid\n(R² < 0.05)',
]

fig2, axes2 = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig2, list(axes2))

for case, rows, ax in zip(CASES, case_data, axes2):
    n1 = n2 = n3 = n4 = 0
    for r in rows:
        if r['cd'] < D_THRESH:
            continue
        if r['r2'] < 0.05:
            n4 += 1
        elif r['ig_sup'] >= IG_THRESH:
            n1 += 1
        elif r['joint_best_frac'] >= IG_THRESH and r['joint_best_eta2'] >= ETA2_THRESH:
            n2 += 1
        else:
            n3 += 1

    n_above = n1 + n2 + n3 + n4
    sizes  = [n1, n2, n3, n4]
    labels = [f'{name}\n(n={n})' for name, n in zip(CAT_NAMES, sizes)]
    nz     = [(s, l, c) for s, l, c in zip(sizes, labels, CAT_COLORS) if s > 0]
    if nz:
        sz, lb, cl = zip(*nz)
        wedges, texts, autotexts = ax.pie(
            sz, labels=lb, colors=cl, autopct='%1.0f%%',
            startangle=90, textprops={'fontsize': FONT.LABEL - 2},
            wedgeprops={'linewidth': 0.6, 'edgecolor': 'white'})
        for at in autotexts:
            at.set_fontsize(FONT.LABEL - 1)
            at.set_fontweight('bold')

    sup_short = SHORT.get(case['sup_group'], case['sup_label'])
    ax.set_title(
        f"{case['label']} × {sup_short}\nn = {n_above} sessions (d ≥ {D_THRESH})",
        fontsize=FONT.LABEL - 1, pad=8)

savefig_manifest(fig2, 'e07_e23_attribution_pie.png', OUT_DIRS)
print('Saved e07_e23_attribution_pie.png')
plt.close(fig2)
