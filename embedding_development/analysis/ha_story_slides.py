#!/usr/bin/env python3
"""
ha_story_slides.py

Three publication-ready figures for the head-angle story:

  fig1_ha_distribution.png   — What is head angle and how is it distributed?
  fig2_attribution_outlier.png — Head angle is a dominant attribution outlier
  fig3_genuine_predictor.png — HA is a genuine predictor; nonlinear metrics needed
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from datetime import datetime

sys.path.insert(0, '.')

# ─────────────────────────────── CONFIG ─────────────────────────────────────
TUNING_CSV   = "outputs/mlps/head_angle_tuning_20260517_1634/head_angle_tuning_table.csv"
ABLATION_CSV = "outputs/mlps/head_angle_analysis_20260517_1529/ablation_results.csv"
ATTR_DIR     = "outputs/mlps/ensembles_multiseed"
DATASET_PATH = "outputs/session_dataset_ensembles.pkl"
GLM_BASE     = "outputs/glm_input_data/"
HA_FEAT_IDX  = 5
N_BINS       = 10

# Example pairs for tuning curves
MONO_PAIR    = (27, 17)   # s_i=27, n_i=17 → session 28, E18
NONMONO_PAIR = (19, 22)   # s_i=19, n_i=22 → session 20, E23

ts         = datetime.now().strftime('%Y%m%d_%H%M')
OUT_DIR    = f"outputs/mlps/ha_story_slides_{ts}"
DESK_DIR   = f"/mnt/c/Users/amits/Desktop/ha_story_slides_{ts}"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DESK_DIR, exist_ok=True)
print(f"Output: {OUT_DIR}")

PALETTE = {
    'mono_correct':    '#2196F3',
    'nonmono_correct': '#FF9800',
    'genuine_overattr':'#F44336',
    'under_attr':      '#9E9E9E',
    'other':           '#CCCCCC',
}
LABELS = {
    'mono_correct':    'Mono-tuned',
    'nonmono_correct': 'Non-mono-tuned',
    'genuine_overattr':'Genuine over-attr',
    'under_attr':      'Under-attr',
    'other':           'Other',
}
CATS = ['mono_correct', 'nonmono_correct', 'genuine_overattr', 'under_attr']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def savefig(name):
    for d in (OUT_DIR, DESK_DIR):
        plt.savefig(os.path.join(d, name), dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved {name}")


def compute_tuning_curve(ha, activity, n_bins=N_BINS):
    """Bin ha into n_bins equal-frequency bins; return (centers, means, stds)."""
    edges = np.percentile(ha, np.linspace(0, 100, n_bins + 1))
    centers, means, stds = [], [], []
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (ha >= edges[i]) & (ha < edges[i + 1])
        else:
            mask = (ha >= edges[i]) & (ha <= edges[i + 1])
        if mask.sum() < 3:
            continue
        centers.append((edges[i] + edges[i + 1]) / 2)
        means.append(activity[mask].mean())
        stds.append(activity[mask].std())
    return np.array(centers), np.array(means), np.array(stds)


def compute_eta2(ha, activity, n_bins=N_BINS):
    edges = np.percentile(ha, np.linspace(0, 100, n_bins + 1))
    overall_var = np.var(activity)
    within_vars = []
    for i in range(n_bins):
        mask = (ha >= edges[i]) & (ha < edges[i + 1]) if i < n_bins - 1 else \
               (ha >= edges[i]) & (ha <= edges[i + 1])
        if mask.sum() > 2:
            within_vars.append(np.var(activity[mask]))
    return 1 - np.mean(within_vars) / (overall_var + 1e-12)


# ─────────────────────────────── LOAD DATA ──────────────────────────────────
print("Loading data...")

# Raw behavior (physical degrees)
beh_vals = np.load(os.path.join(GLM_BASE, "behavior_glm_input.npy"), allow_pickle=True)
beh_idx  = np.load(os.path.join(GLM_BASE, "behavior_glm_input_index.npy"), allow_pickle=True)
beh_cols = np.load(os.path.join(GLM_BASE, "behavior_glm_input_columns.npy"), allow_pickle=True)
beh_df   = pd.DataFrame(beh_vals, columns=beh_cols)
ha_raw   = beh_df['head_angle'].dropna().astype(float).values   # physical degrees
session_col = np.array([t[0] for t in beh_idx])
# Restrict to rows without NaN in head_angle
valid_mask = ~beh_df['head_angle'].isna().values
ha_raw_all  = ha_raw
session_all = session_col[valid_mask]

# Session dataset (z-scored features + ensemble labels)
with open(DATASET_PATH, 'rb') as f:
    dataset = pickle.load(f)
session_keys = sorted(dataset.keys())

# Per-feature IG matrix (29 sessions × 23 ensembles × 26 features)
ig_col = np.load(os.path.join(ATTR_DIR, "signed_ig_per_column.npy"))   # (29, 23, 26)
abs_ig  = np.abs(ig_col)
mean_ig_feat = np.nanmean(abs_ig.reshape(-1, 26), axis=0)               # (26,)

# Tuning CSV
df_tuning = pd.read_csv(TUNING_CSV)
df_tuning['s_i'] = df_tuning['session'] - 1
df_tuning['n_i'] = df_tuning['ensemble'].str.replace('E', '').astype(int) - 1

# Ablation CSV
df_abl = pd.read_csv(ABLATION_CSV)
df_abl['s_i'] = df_abl['session'] - 1
df_abl['n_i'] = df_abl['ensemble'].str.replace('E', '').astype(int) - 1

# Example tuning curve data
def get_session_xy(s_i):
    sk = session_keys[s_i]
    v  = dataset[sk]
    X  = np.concatenate([v['data'][t]   for t in sorted(v['data'])],   axis=0)
    Y  = np.concatenate([v['labels'][t] for t in sorted(v['labels'])], axis=0)
    return X, Y

print("Loading example pair data...")
X_mono,    Y_mono    = get_session_xy(MONO_PAIR[0])
X_nonmono, Y_nonmono = get_session_xy(NONMONO_PAIR[0])

ha_mono     = X_mono[:, HA_FEAT_IDX]
act_mono    = Y_mono[:, MONO_PAIR[1]]
ha_nonmono  = X_nonmono[:, HA_FEAT_IDX]
act_nonmono = Y_nonmono[:, NONMONO_PAIR[1]]

rho_mono,    _ = spearmanr(ha_mono,    act_mono)
rho_nonmono, _ = spearmanr(ha_nonmono, act_nonmono)
eta2_mono      = compute_eta2(ha_mono,    act_mono)
eta2_nonmono   = compute_eta2(ha_nonmono, act_nonmono)

c_mono,    m_mono,    s_mono    = compute_tuning_curve(ha_mono,    act_mono)
c_nonmono, m_nonmono, s_nonmono = compute_tuning_curve(ha_nonmono, act_nonmono)

print(f"Mono pair:    Spearman={rho_mono:.3f}, eta2={eta2_mono:.3f}")
print(f"Nonmono pair: Spearman={rho_nonmono:.3f}, eta2={eta2_nonmono:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Head angle distribution
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Head angle distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel A: Histogram of raw HA (degrees) pooled
ax = axes[0]
ax.hist(ha_raw_all, bins=80, color='#455A64', alpha=0.85, edgecolor='none', density=True)
ax.axvline(ha_raw_all.mean(), color='#E53935', lw=2, ls='--', label=f'Mean = {ha_raw_all.mean():.0f}°')
ax.axvspan(ha_raw_all.mean() - ha_raw_all.std(),
           ha_raw_all.mean() + ha_raw_all.std(),
           alpha=0.15, color='#E53935', label=f'±1 SD = {ha_raw_all.std():.0f}°')
ax.set_xlabel('Head angle (degrees)')
ax.set_ylabel('Density')
ax.set_title('Head angle distribution\n(all sessions, all trials)')
ax.legend(fontsize=9)
ax.text(0.97, 0.97,
        f'n = {len(ha_raw_all):,} time bins\nRange: [{ha_raw_all.min():.0f}°, {ha_raw_all.max():.0f}°]',
        transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#CCC', alpha=0.9))

# Panel B: Per-session HA distributions (z-scored) - violin
ax2 = axes[1]
session_ids = sorted(beh_df['session'].dropna().unique()) if 'session' in beh_df.columns else []

# Use the z-scored HA from session dataset
ha_per_session = []
s_labels = []
for s_i, sk in enumerate(session_keys):
    v = dataset[sk]
    X = np.concatenate([v['data'][t] for t in sorted(v['data'])], axis=0)
    ha_z = X[:, HA_FEAT_IDX]
    ha_per_session.append(ha_z)
    s_labels.append(f's{s_i+1:02d}')

positions = np.arange(1, len(ha_per_session) + 1)
vp = ax2.violinplot(ha_per_session, positions=positions,
                    showmedians=True, showextrema=False, widths=0.8)
for body in vp['bodies']:
    body.set_facecolor('#607D8B')
    body.set_alpha(0.6)
vp['cmedians'].set_color('#E53935')
vp['cmedians'].set_linewidth(1.5)

ax2.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax2.set_xlabel('Session')
ax2.set_ylabel('Head angle (z-scored per session)')
ax2.set_title('Consistent HA range across sessions\n(z-scored; all sessions)')
ax2.set_xticks(positions[::3])
ax2.set_xticklabels(s_labels[::3], rotation=30, ha='right', fontsize=8)
ax2.text(0.97, 0.03,
         'Each session z-scored independently\nbefore model input',
         transform=ax2.transAxes, ha='right', va='bottom', fontsize=8,
         color='#555', style='italic')

plt.tight_layout()
savefig("fig1_ha_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Attribution outlier
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2: Attribution outlier...")

feat_names_short = [
    'Camera\nmotion', 'Camera\nacc', 'Head\nYaw vel', 'Head\nYaw acc',
    'Head angle\nvel', 'HEAD\nANGLE', 'Movement\nenergy',
    'Zone 0', 'Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5', 'Zone 6', 'Zone 7', 'Zone 8',
    'Cue 0', 'Cue 1', 'Cue 2',
    'Choice 0', 'Choice 1', 'Choice 2',
    'Reward 0', 'Reward 1', 'Reward 2',
    'Lick',
]

# Semantic group coloring
group_colors = {
    'motion':  '#78909C',   # feats 0-3
    'head':    '#E53935',   # feats 4-6 (head angle vel, head angle, movement)
    'zone':    '#7E57C2',   # feats 7-15
    'task':    '#43A047',   # feats 16-25
}
feat_colors = []
for i in range(26):
    if i in [0, 1, 2, 3]:
        feat_colors.append(group_colors['motion'])
    elif i in [4, 5, 6]:
        feat_colors.append(group_colors['head'])
    elif 7 <= i <= 15:
        feat_colors.append(group_colors['zone'])
    else:
        feat_colors.append(group_colors['task'])

# Sort by importance
order = np.argsort(mean_ig_feat)[::-1]

fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.bar(np.arange(26), mean_ig_feat[order],
              color=[feat_colors[i] for i in order],
              alpha=0.85, edgecolor='none', width=0.7)

# Highlight bar 0 (head angle, rank 1)
bars[0].set_edgecolor('#C62828')
bars[0].set_linewidth(2)
bars[0].set_alpha(1.0)

ax.set_xticks(np.arange(26))
ax.set_xticklabels([feat_names_short[i] for i in order],
                   rotation=45, ha='right', fontsize=7.5)
ax.set_ylabel('Mean |IG| attribution')
ax.set_title('Head angle dominates integrated-gradient attribution\n'
             '(mean |IG| per feature across all session–ensemble pairs)')

# Ratio annotation
ratio = mean_ig_feat[order[0]] / mean_ig_feat[order[1]]
ax.annotate(f'{ratio:.1f}× next feature',
            xy=(0, mean_ig_feat[order[0]]),
            xytext=(3, mean_ig_feat[order[0]] * 0.85),
            fontsize=9, color='#C62828',
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))

# Legend
legend_elems = [
    Line2D([0], [0], color=group_colors['motion'], lw=6, label='Camera / body motion', alpha=0.85),
    Line2D([0], [0], color=group_colors['head'],   lw=6, label='Head orientation',     alpha=0.85),
    Line2D([0], [0], color=group_colors['zone'],   lw=6, label='Track zone (location)',alpha=0.85),
    Line2D([0], [0], color=group_colors['task'],   lw=6, label='Task state (cue/choice/reward/lick)', alpha=0.85),
]
ax.legend(handles=legend_elems, loc='upper right', fontsize=9)

plt.tight_layout()
savefig("fig2_attribution_outlier.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Genuine predictor; nonlinear metrics needed
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 3: Genuine predictor...")

fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.55,
                       left=0.06, right=0.97, top=0.95, bottom=0.12)

ax_sc       = fig.add_subplot(gs[:, 0])   # Spearman vs eta2 scatter (full height)
ax_mono     = fig.add_subplot(gs[0, 1])   # tuning curve — monotonic
ax_nm       = fig.add_subplot(gs[1, 1])   # tuning curve — non-monotonic
ax_abl_top  = fig.add_subplot(gs[0, 2])   # ablation: HA contribution to R²
ax_abl_bot  = fig.add_subplot(gs[1, 2])   # ablation: standalone HA vs HA-vel

# ── Panel A: Spearman vs η² scatter ──────────────────────────────────────────
# Plot 'other' first (grey, small)
df_other = df_tuning[df_tuning['category'] == 'other']
ax_sc.scatter(np.abs(df_other['spearman_rho']), df_other['eta2'],
              c=PALETTE['other'], s=12, alpha=0.4, label='Other', zorder=1)

# Plot each category
for cat in CATS:
    sub = df_tuning[df_tuning['category'] == cat]
    ax_sc.scatter(np.abs(sub['spearman_rho']), sub['eta2'],
                  c=PALETTE[cat], s=28, alpha=0.8, label=LABELS[cat], zorder=3)

# Threshold lines (median of CATS-only pairs)
df_cats = df_tuning[df_tuning['category'].isin(CATS)]
spear_med = np.median(np.abs(df_cats['spearman_rho']))
eta2_med  = np.median(df_cats['eta2'])
ax_sc.axvline(spear_med, color='black', lw=1, ls='--', alpha=0.5)
ax_sc.axhline(eta2_med,  color='black', lw=1, ls='--', alpha=0.5)

# Quadrant labels
xmax = ax_sc.get_xlim()[1] if ax_sc.get_xlim()[1] > 0 else 0.5
ax_sc.text(0.02, 0.99, 'Non-monotonic\ntuned', transform=ax_sc.transAxes,
           ha='left', va='top', fontsize=7.5, color='#FF9800',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF3E0', edgecolor='none'))
ax_sc.text(0.58, 0.99, 'Monotonic\ntuned', transform=ax_sc.transAxes,
           ha='left', va='top', fontsize=7.5, color='#2196F3',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#E3F2FD', edgecolor='none'))
ax_sc.text(0.58, 0.01, 'Spearman\nmisses tuning', transform=ax_sc.transAxes,
           ha='left', va='bottom', fontsize=7.5, color='#9E9E9E')

ax_sc.set_xlabel('|Spearman ρ|  (monotonic correlation)')
ax_sc.set_ylabel('η²  (nonlinear ANOVA)')
ax_sc.set_title('Two metrics for HA tuning\nSpearman misses non-monotonic curves')
ax_sc.legend(fontsize=8, loc='lower right', markerscale=1.4)

# ── Panels B & C: Tuning curves ───────────────────────────────────────────────
def _plot_curve(ax, centers, means, stds, rho, eta2, title, color):
    ax.fill_between(centers, means - stds, means + stds,
                    alpha=0.25, color=color)
    ax.plot(centers, means, 'o-', color=color, lw=1.8, ms=5, zorder=3)
    ax.set_xlabel('Head angle (z-scored)', fontsize=9)
    ax.set_ylabel('Ensemble activity', fontsize=9)
    ax.set_title(title, fontsize=9.5)
    stats_txt = f'|ρ| = {abs(rho):.3f}   η² = {eta2:.3f}'
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#BBB', alpha=0.9))

_plot_curve(ax_mono, c_mono, m_mono, s_mono,
            rho_mono, eta2_mono,
            title=f'Monotonic tuning\n(Session {MONO_PAIR[0]+1}, E{MONO_PAIR[1]+1:02d})',
            color=PALETTE['mono_correct'])

_plot_curve(ax_nm, c_nonmono, m_nonmono, s_nonmono,
            rho_nonmono, eta2_nonmono,
            title=f'Non-monotonic tuning\n(Session {NONMONO_PAIR[0]+1}, E{NONMONO_PAIR[1]+1:02d})',
            color=PALETTE['nonmono_correct'])

# Add panel label explaining the contrast
ax_mono.text(0.02, 0.04, 'Spearman ✓  η² ✓', transform=ax_mono.transAxes,
             fontsize=8, color=PALETTE['mono_correct'], va='bottom')
ax_nm.text(0.02, 0.04, 'Spearman ✗  η² ✓', transform=ax_nm.transAxes,
           fontsize=8, color=PALETTE['nonmono_correct'], va='bottom')

# ── Panel D (top): HA contribution to R² — stacked bars ──────────────────────
df_abl_m = df_abl.copy()
df_abl_m['delta_ha']  = df_abl_m['r2_full'] - df_abl_m['r2_no_ha']   # HA's unique contribution
df_abl_m['delta_vel'] = df_abl_m['r2_full'] - (df_abl_m['r2_full']   # approximate vel contribution
                         - df_abl_m['r2_ha_vel_only'] + df_abl_m['r2_no_ha'])

n_pairs = len(df_abl_m)
x = np.arange(n_pairs)
pair_labels = [f"s{r['session']}\n{r['ensemble']}" for _, r in df_abl_m.iterrows()]

# Stacked: base (r2_no_ha, grey) + HA contribution (red, on top)
base   = df_abl_m['r2_no_ha'].values
delta  = df_abl_m['delta_ha'].values

# Positive and negative base need separate handling
base_pos = np.clip(base, 0, None)
base_neg = np.clip(base, None, 0)

ax_abl_top.bar(x, base_pos, color='#90A4AE', alpha=0.85, label='R² without HA', zorder=2)
ax_abl_top.bar(x, base_neg, color='#90A4AE', alpha=0.85, zorder=2)
ax_abl_top.bar(x, delta,   bottom=base, color='#E53935', alpha=0.85, label="HA's contribution (ΔR²)", zorder=3)

ax_abl_top.axhline(0, color='black', lw=0.8, alpha=0.5)
ax_abl_top.set_ylabel('R²', fontsize=9)
ax_abl_top.set_title("HA's unique contribution to model R²", fontsize=9.5)
ax_abl_top.set_xticks(x)
ax_abl_top.set_xticklabels(pair_labels, fontsize=8)
ax_abl_top.legend(fontsize=7.5, loc='upper right')

# Annotate ΔR² percentage of total
for xi, (b, d, full) in enumerate(zip(base, delta, df_abl_m['r2_full'].values)):
    pct = 100 * d / (full + 1e-8)
    ax_abl_top.text(xi, max(full, 0) + 0.01, f'{pct:.0f}%', ha='center', va='bottom',
                    fontsize=7.5, color='#B71C1C', fontweight='bold')

# ── Panel D (bot): Standalone HA vs HA-vel ────────────────────────────────────
w = 0.32
ax_abl_bot.bar(x - w/2, df_abl_m['r2_ha_only'],    w, label='HA alone',     color='#E53935', alpha=0.85)
ax_abl_bot.bar(x + w/2, df_abl_m['r2_ha_vel_only'],w, label='HA-vel alone', color='#FF7043', alpha=0.85)

ax_abl_bot.axhline(0, color='black', lw=0.8, alpha=0.5)
ax_abl_bot.set_ylabel('R²', fontsize=9)
ax_abl_bot.set_title('Standalone predictive power\n(single-feature ablation)', fontsize=9.5)
ax_abl_bot.set_xticks(x)
ax_abl_bot.set_xticklabels(pair_labels, fontsize=8)
ax_abl_bot.legend(fontsize=7.5, loc='upper right')

savefig("fig3_genuine_predictor.png")

print("\nDone. All figures saved.")
