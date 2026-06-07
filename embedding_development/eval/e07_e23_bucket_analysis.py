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

gpv    = np.load(os.path.join(mdir, 'importance_global_pv_semantic.npy'))
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
D_THRESH   = 0.1    # "supervisor has a point" (Cohen's d)
GPV_THRESH = 0.10   # "direct detection": GPV(sup_var)/R² >= 10% of ensemble's explained variance

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

        cd       = cd_categorical(X[:, sup_cols], y)
        r2       = float(r2_all[s_idx, e_idx])
        gv_sup   = float(gpv[s_idx, e_idx, sup_g])
        gv_our   = float(gpv[s_idx, e_idx, our_g])
        frac_sup = gv_sup / r2 if r2 > 0 else 0.0
        frac_our = gv_our / r2 if r2 > 0 else 0.0

        eta2s = {}
        for g_name in group_names:
            vals = X[:, feat_idx[g_name]].mean(axis=1)
            eta2s[g_name] = eta_sq(vals, X[:, sup_cols])

        # GPV/R² for every feature group
        frac_all = {g: float(gpv[s_idx, e_idx, gi]) / r2 if r2 > 0 else 0.0
                    for gi, g in enumerate(group_names)}

        # best joint feature: argmax η²(feat, sup_cond) × GPV(feat)/R²
        joint_best_g    = max(group_names,
                              key=lambda g: eta2s[g] * frac_all[g])
        joint_best_eta2 = eta2s[joint_best_g]
        joint_best_frac = frac_all[joint_best_g]

        rows.append(dict(s_idx=s_idx, cd=cd, r2=r2,
                         gpv_sup=gv_sup, frac=frac_sup,
                         gpv_our=gv_our, frac_our=frac_our,
                         eta2s=eta2s, frac_all=frac_all,
                         joint_best_g=joint_best_g,
                         joint_best_eta2=joint_best_eta2,
                         joint_best_frac=joint_best_frac))

    case_data.append(rows)

    above = [r for r in rows if r['cd'] >= D_THRESH]
    b1    = [r for r in above if r['frac'] >= GPV_THRESH]
    b2    = [r for r in above if r['frac'] <  GPV_THRESH]
    print(f"{case['label']}: {len(rows)} sessions, {len(above)} d>={D_THRESH}, "
          f"B1={len(b1)}, B2={len(b2)}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Supervisor's variable: Cohen's d per session
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig1, list(axes1))

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
    ax.set_ylabel("Cohen's d", fontsize=FONT.LABEL - 1)
    n_above = sum(1 for d in cds if d >= D_THRESH)
    sup_short = SHORT.get(case['sup_group'], case['sup_label'])
    ax.set_title(
        f"{case['label']} ({sup_short}): d, "
        f"{n_above}/{len(rows_s)} ≥ {D_THRESH}",
        fontsize=FONT.LABEL - 1, pad=3)
    ax.legend(fontsize=FONT.LEGEND - 1, frameon=False)
    ax.text(0.02, 0.97, case['panel1'], transform=ax.transAxes,
            fontsize=FONT.PANEL, fontweight='bold', va='top', ha='left')

add_footnote(fig1,
    f"Cohen's d: max pairwise effect between conditions.  "
    f"Orange bars (d ≥ {D_THRESH}) form the analysis set.  "
    f"Sorted descending.")
savefig_manifest(fig1, 'e07_e23_bucket_signal.png', OUT_DIRS)
print('Saved e07_e23_bucket_signal.png')
plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Bucket 1: direct GPV detection
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig2, list(axes2))

# Compute shared y-axis max across both cases
_all_fracs = [r['frac'] for rows in case_data
              for r in rows if r['cd'] >= D_THRESH]
shared_ymax = max(_all_fracs) * 1.25 if _all_fracs else 1.0

for case, rows, ax in zip(CASES, case_data, axes2):
    above  = sorted([r for r in rows if r['cd'] >= D_THRESH], key=lambda r: -r['cd'])
    fracs  = np.array([r['frac'] for r in above])
    s_idxs = [r['s_idx'] for r in above]
    n_b1   = (fracs >= GPV_THRESH).sum()
    colors = [case['C_B1'] if f >= GPV_THRESH else '#CCCCCC' for f in fracs]
    x      = np.arange(len(above))

    ax.bar(x, fracs, color=colors, alpha=0.85, width=0.7)
    ax.axhline(GPV_THRESH, color='#444', lw=1.0, ls='--', alpha=0.8,
               label=f'GPV/R² = {GPV_THRESH}  (10%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s_idxs[i]+1:02d}' for i in range(len(above))],
                       rotation=90, fontsize=max(5, FONT.TICK - 3))
    ax.set_ylabel(f'GPV({case["sup_label"]}) / R²', fontsize=FONT.LABEL - 1)
    ax.set_ylim(0, shared_ymax)
    ax.set_title(
        f"{case['label']} ({SHORT.get(case['sup_group'], case['sup_label'])}): "
        f"{n_b1}/{len(above)} ≥ 10%",
        fontsize=FONT.LABEL - 1, pad=3)
    ax.legend(fontsize=FONT.LEGEND - 1, frameon=False)
    ax.text(0.02, 0.97, case['panel2'], transform=ax.transAxes,
            fontsize=FONT.PANEL, fontweight='bold', va='top', ha='left')

add_footnote(fig2,
    f"Analysis set: sessions with d ≥ {D_THRESH}, sorted by Cohen's d.  "
    f"Blue: GPV(feature) ≥ 10% of ensemble R² (model attributes directly).  "
    f"Grey: Bucket 2.")
savefig_manifest(fig2, 'e07_e23_bucket_gpv.png', OUT_DIRS)
print('Saved e07_e23_bucket_gpv.png')
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Bucket 2: scatter η²(best feature, sup_cond) vs GPV(best feature)/R²
# Best feature = argmax η² × GPV/R² (highest joint co-variation + attribution score)
# ══════════════════════════════════════════════════════════════════════════════
FEAT_COLORS = {
    'frame_raw_500msMedian':                  '#2CA02C',
    'frame_YawPitch_abs_vel_sum_500msMedian': '#9467BD',
    'head_angle':                             '#1F77B4',
    'frame_position':                         '#8C564B',
    'head_angle_vel':                         '#E377C2',
}
DEFAULT_COLOR = '#888888'

fig3, axes3 = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig3, list(axes3))

# Pre-compute shared axis limits across both panels
all_b2 = [r for rows in case_data for r in rows
          if r['cd'] >= D_THRESH and r['frac'] < GPV_THRESH]
_xs = [r['joint_best_eta2'] for r in all_b2]
_ys = [r['joint_best_frac'] for r in all_b2]
X_MAX = max(_xs) * 1.18 if _xs else 0.35
Y_MAX = max(_ys) * 1.12 if _ys else 2.0

for case, rows, ax in zip(CASES, case_data, axes3):
    b2 = [r for r in rows if r['cd'] >= D_THRESH and r['frac'] < GPV_THRESH]
    if not b2:
        ax.text(0.5, 0.5, 'No Bucket 2 sessions', ha='center', va='center',
                transform=ax.transAxes, fontsize=FONT.LABEL)
        ax.set_title(f"{case['label']}: Bucket 2 (none)",
                     fontsize=FONT.LABEL - 1)
        add_panel_label(ax, case['panel3'])
        continue

    xs      = np.array([r['joint_best_eta2'] for r in b2])
    ys      = np.array([r['joint_best_frac'] for r in b2])
    feat_gs = [r['joint_best_g']             for r in b2]
    s_idxs  = [r['s_idx']                    for r in b2]
    colors  = [FEAT_COLORS.get(g, DEFAULT_COLOR) for g in feat_gs]

    ax.scatter(xs, ys, c=colors, s=65, alpha=0.85, zorder=3,
               edgecolors='white', linewidths=0.4)
    for xi, yi, si in zip(xs, ys, s_idxs):
        ax.annotate(f'S{si+1:02d}', (xi, yi), fontsize=5.5,
                    xytext=(3, 3), textcoords='offset points', color='#444')

    ax.axhline(GPV_THRESH, color='#444', lw=0.9, ls='--', alpha=0.6,
               label=f'GPV/R² = {GPV_THRESH}')
    ax.axvline(0.05, color='#888', lw=0.9, ls=':', alpha=0.5,
               label='η² = 0.05')

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, Y_MAX)

    ax.set_xlabel(f'η²(feature, {case["sup_label"]} condition)',
                  fontsize=FONT.LABEL - 1)
    ax.set_ylabel('GPV / R²  (same feature)', fontsize=FONT.LABEL - 1)
    ax.set_title(
        f"{case['label']}: {len(b2)} Bucket 2 sessions",
        fontsize=FONT.LABEL - 1, pad=3)
    ax.text(0.02, 0.97, case['panel3'], transform=ax.transAxes,
            fontsize=FONT.PANEL, fontweight='bold', va='top', ha='left')

add_footnote(fig3,
    f'Bucket 2: GPV(feature)/R² < {GPV_THRESH} despite d ≥ {D_THRESH}.  '
    'Feature = argmax η²(feat, condition) × GPV(feat)/R² per session.  '
    'X: co-variation with the condition label.  Y: model attribution to that feature.  '
    'Colours: ■ green = Fwd Speed  ■ blue = Head Angle  ■ purple = Rot. Vel.')
savefig_manifest(fig3, 'e07_e23_bucket_covariation.png', OUT_DIRS)
print('Saved e07_e23_bucket_covariation.png')
plt.close(fig3)
