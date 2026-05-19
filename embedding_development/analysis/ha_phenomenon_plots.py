#!/usr/bin/env python3
"""
ha_phenomenon_plots.py

Clean summary figures for the η²/Spearman/IG head_angle phenomenon:
  Fig 1 — Tuning curve example: inverted-U (E23, S20) showing why Spearman fails
  Fig 2 — η² vs |Spearman| scatter colored by IG (across 198 pairs)
  Fig 3 — Alignment bar: IG/GPV alignment with Spearman vs η² as ground truth
  Fig 4 — Baseline-distance effect: IG vs head_angle session std (before/after η² control)
  Fig 5 — Pair-category counts (mono correct / nonmono correct / overattr / under-attr)
"""
import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, pearsonr
from scipy import stats as scipy_stats
from datetime import datetime

# ──────────────────────────── CONFIG ────────────────────────────
TUNING_CSV   = "outputs/mlps/head_angle_tuning_20260517_1634/head_angle_tuning_table.csv"
ATTR_DIR     = "outputs/mlps/ensembles_multiseed"
DATASET_PATH = "outputs/session_dataset_ensembles.pkl"

HA_FEAT_IDX  = 5     # head_angle column in 26-feature input
HA_GRP_IDX   = 5     # head_angle group index in semantic groups
HAV_GRP_IDX  = 4     # head_angle_vel group index
E23_IDX      = 22    # ensemble E23 (0-indexed)
N_BINS       = 10
MIN_BIN_PTS  = 5

ts          = datetime.now().strftime('%Y%m%d_%H%M')
OUT_DIR     = f"outputs/mlps/ha_phenomenon_{ts}"
DESK_DIR    = f"/mnt/c/Users/amits/Desktop/ha_phenomenon_{ts}"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DESK_DIR, exist_ok=True)
print(f"Output: {OUT_DIR}")
print(f"Desktop: {DESK_DIR}")

PALETTE = {
    'mono_correct':    '#2196F3',  # blue
    'nonmono_correct': '#FF9800',  # orange
    'genuine_overattr':'#F44336',  # red
    'under_attr':      '#9E9E9E',  # grey
    'other':           '#E0E0E0',  # light grey
}
LABELS = {
    'mono_correct':    'Monotonic\n(correctly attributed)',
    'nonmono_correct': 'Non-monotonic\n(rescued by η²)',
    'genuine_overattr':'Genuine\nover-attribution',
    'under_attr':      'Under-\nattributed',
}

def savefig(name):
    for d in (OUT_DIR, DESK_DIR):
        plt.savefig(os.path.join(d, name), dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Saved {name}")


# ──────────────────────────── LOAD DATA ─────────────────────────
print("Loading data...")
df = pd.read_csv(TUNING_CSV)

attr_r2  = np.load(os.path.join(ATTR_DIR, "all_r2.npy"))
ig_mat   = np.load(os.path.join(ATTR_DIR, "importance_ig_semantic.npy"))
gpv_mat  = np.load(os.path.join(ATTR_DIR, "importance_global_pv_semantic.npy"))
with open(os.path.join(ATTR_DIR, "semantic_groups.pkl"), 'rb') as f:
    semantic_groups = pickle.load(f)

# Valid mask (same as main analysis)
R2_THRESHOLD = 0.01
mask_3d     = np.isnan(attr_r2)
mean_r2_mat = np.nanmean(np.where(mask_3d, np.nan, attr_r2.astype(float)), axis=0)
low_r2_mask = mean_r2_mat < R2_THRESHOLD
valid_mask  = ~np.any(mask_3d, axis=0) & ~low_r2_mask

with open(DATASET_PATH, 'rb') as f:
    dataset = pickle.load(f)
session_keys = sorted(dataset.keys())

# Reconstruct per-session head_angle std (for baseline-distance analysis)
ha_stds = {}
for s_i, sk in enumerate(session_keys):
    trials = dataset[sk]['data']
    X = np.concatenate([trials[t] for t in sorted(trials.keys())], axis=0).astype(np.float32)
    ha_stds[s_i] = float(np.std(X[:, HA_FEAT_IDX]))

# valid_idx_arr: list of (s_i, n_i) for all valid pairs
valid_idx_arr = [(s, n) for s in range(attr_r2.shape[1]) for n in range(attr_r2.shape[2])
                 if valid_mask[s, n]]

# Build per-pair table with session std
df_rows = []
for row in df.itertuples():
    # match to s_i by session number (session col is 1-indexed session number)
    s_num = int(row.session)  # 1-indexed session stored in CSV? check
    # The CSV stores 0-indexed session based on valid_idx_arr
    pass

# Rebuild valid pairs with both η² and IG (the 198-pair table already has this)
# session col is actually s_i (0-indexed)
df['ha_std'] = df['session'].apply(lambda s: ha_stds.get(int(s), np.nan))
df_valid = df[df['category'] != 'other'].copy()
print(f"Valid pairs (non-other): {len(df_valid)}")
print(f"Category counts:\n{df_valid['category'].value_counts()}")

# ══════════════════════════════════════════════════════════════════
# FIG 1 — Inverted-U tuning curve: E23, Session 20
# ══════════════════════════════════════════════════════════════════
print("\n── Fig 1: Tuning curve example (E23, S20) ──")

# Also add a second panel with a monotonic example (E15, S20)
examples = [
    (20, 22, 'E23 · S20',  'Non-monotonic\n(inverted-U)', PALETTE['nonmono_correct']),
    (20, 14, 'E15 · S20',  'Monotonic',                   PALETTE['mono_correct']),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.patch.set_facecolor('white')

for ax, (s_i, n_i, title, subtitle, color) in zip(axes, examples):
    sk = session_keys[s_i]
    trials = dataset[sk]['data']
    Y_trials = dataset[sk]['labels']
    X = np.concatenate([trials[t] for t in sorted(trials.keys())], axis=0).astype(np.float32)
    Y = np.concatenate([Y_trials[t] for t in sorted(Y_trials.keys())], axis=0).astype(np.float32)
    ha   = X[:, HA_FEAT_IDX]
    ens  = Y[:, n_i]

    # Decile bins
    bin_edges = np.percentile(ha, np.linspace(0, 100, N_BINS + 1))
    bin_edges[-1] += 1e-6
    centers, means, sems = [], [], []
    for b in range(N_BINS):
        mask_b = (ha >= bin_edges[b]) & (ha < bin_edges[b + 1])
        if mask_b.sum() >= MIN_BIN_PTS:
            centers.append(float(np.mean(ha[mask_b])))
            means.append(float(np.mean(ens[mask_b])))
            sems.append(float(np.std(ens[mask_b]) / np.sqrt(mask_b.sum())))

    centers = np.array(centers); means = np.array(means); sems = np.array(sems)

    # Stats
    rho_val, _  = spearmanr(ha, ens)
    total_var = np.var(ens)
    within_vars = []
    for b in range(N_BINS):
        mask_b = (ha >= bin_edges[b]) & (ha < bin_edges[b + 1])
        if mask_b.sum() >= MIN_BIN_PTS:
            within_vars.append(float(np.var(ens[mask_b])))
    eta2_val = max(0.0, 1.0 - np.mean(within_vars) / total_var) if total_var > 0 else 0

    ax.fill_between(centers, means - sems, means + sems, alpha=0.25, color=color)
    ax.plot(centers, means, 'o-', color=color, linewidth=2, markersize=6, zorder=3)
    ax.axhline(0, color='#888888', linewidth=0.8, linestyle='--')

    # Annotation box
    textstr = f"Spearman ρ = {rho_val:+.3f}\nη² = {eta2_val:.3f}"
    ax.text(0.97, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=color, linewidth=1.5))

    ax.set_xlabel("Head angle (z-score)", fontsize=12)
    ax.set_ylabel("Ensemble activity (z-score)", fontsize=12)
    ax.set_title(f"{title}\n{subtitle}", fontsize=12, color=color, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle("Tuning curves: why Spearman ρ misses non-monotonic relationships",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
savefig("fig1_tuning_curve_examples.png")


# ══════════════════════════════════════════════════════════════════
# FIG 2 — η² vs |Spearman| scatter, colored by IG
# ══════════════════════════════════════════════════════════════════
print("── Fig 2: η² vs |Spearman| scatter ──")

df_all = df[df['eta2'].notna() & df['ig_attr'].notna()].copy()
df_all['abs_rho'] = df_all['spearman_rho'].abs()

fig, ax = plt.subplots(figsize=(6.5, 5.5))
fig.patch.set_facecolor('white')

sc = ax.scatter(df_all['abs_rho'], df_all['eta2'],
                c=df_all['ig_attr'], cmap='plasma',
                s=30, alpha=0.7, linewidths=0, zorder=3)
cb = plt.colorbar(sc, ax=ax)
cb.set_label("IG attribution (head_angle)", fontsize=11)

# Reference line: if tuning were perfectly monotonic, η² ≈ ρ²
x_ref = np.linspace(0, df_all['abs_rho'].max(), 100)
ax.plot(x_ref, x_ref**2, 'k--', linewidth=1.5, alpha=0.6, label="η² = ρ² (monotonic)")

# Median split lines
med_eta2 = df_all['eta2'].median()
med_rho  = df_all['abs_rho'].median()
ax.axhline(med_eta2, color='#888', linewidth=0.8, linestyle=':')
ax.axvline(med_rho,  color='#888', linewidth=0.8, linestyle=':')

ax.text(med_rho + 0.01, med_eta2 + 0.005, "η² > ρ²\n(non-monotonic\ngap)", fontsize=9, color='#555')

ax.set_xlabel("|Spearman ρ|  (monotonic tuning)", fontsize=12)
ax.set_ylabel("η²  (nonlinear tuning)", fontsize=12)
ax.set_title("η² vs |Spearman ρ| for head_angle, colored by IG attribution\n"
             "Points above y = ρ² line have nonlinear gap captured by IG", fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
savefig("fig2_eta2_vs_spearman_scatter.png")


# ══════════════════════════════════════════════════════════════════
# FIG 3 — Alignment bar chart
# ══════════════════════════════════════════════════════════════════
print("── Fig 3: Alignment comparison bar chart ──")

# Values from the completed analysis
# (hard-coded from the printed output of head_angle_tuning_analysis.py
#  and find_feature_correspondences.py)
data_align = {
    'head_angle\n(IG)':  {'Spearman ρ ground truth': 0.055, 'η² ground truth': 0.669},
    'head_angle\n(GPV)': {'Spearman ρ ground truth': 0.268, 'η² ground truth': 0.774},
    'head_angle_vel\n(IG)':  {'Spearman ρ ground truth': 0.700, 'η² ground truth': 0.700},
    'head_angle_vel\n(GPV)': {'Spearman ρ ground truth': 0.674, 'η² ground truth': 0.674},
}

methods = list(data_align.keys())
gt_types = ['Spearman ρ ground truth', 'η² ground truth']
bar_colors = ['#90CAF9', '#1565C0']   # light / dark blue

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('white')

for j, (gt, color) in enumerate(zip(gt_types, bar_colors)):
    vals = [data_align[m][gt] for m in methods]
    bars = ax.bar(x + (j - 0.5) * width, vals, width, label=gt,
                  color=color, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f"{val:.3f}", ha='center', va='bottom', fontsize=8.5, color='#333')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylabel("Alignment (Spearman ρ with behavioral correlation)", fontsize=11)
ax.set_ylim(-0.1, 1.0)
ax.set_title("Attribution alignment: which ground truth reveals the correct picture?\n"
             "IG alignment jumps 0.055 → 0.669 when η² replaces Spearman ρ as ground truth",
             fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.spines[['top', 'right']].set_visible(False)

# Add vertical separator between ha and ha_vel
ax.axvline(1.5, color='#CCCCCC', linewidth=1.5, linestyle='--')
ax.text(0.5, 0.96, "head_angle", transform=ax.transAxes, ha='center',
        fontsize=10, color='#555', style='italic')
ax.text(0.5, 0.96, "     ← head_angle       head_angle_vel →",
        transform=ax.transAxes, ha='center', fontsize=9, color='#888')

plt.tight_layout()
savefig("fig3_alignment_bar.png")


# ══════════════════════════════════════════════════════════════════
# FIG 4 — Baseline-distance confound: IG vs session head_angle std
# ══════════════════════════════════════════════════════════════════
print("── Fig 4: Baseline-distance effect ──")

df_v = df_valid.copy()
df_v['ha_std'] = df_v['session'].apply(lambda s: ha_stds.get(int(s), np.nan))
df_v = df_v.dropna(subset=['ha_std', 'ig_attr', 'eta2'])

# Partial correlation: IG vs ha_std controlling for eta2
def partial_corr(x, y, z):
    """Spearman partial correlation of x,y controlling for z."""
    rx_z = scipy_stats.spearmanr(x, z)[0]
    ry_z = scipy_stats.spearmanr(y, z)[0]
    rx_y = scipy_stats.spearmanr(x, y)[0]
    num  = rx_y - rx_z * ry_z
    den  = np.sqrt((1 - rx_z**2) * (1 - ry_z**2))
    return num / den if den > 0 else np.nan

rho_raw    = scipy_stats.spearmanr(df_v['ha_std'], df_v['ig_attr'])[0]
rho_partial = partial_corr(df_v['ha_std'].values, df_v['ig_attr'].values, df_v['eta2'].values)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
fig.patch.set_facecolor('white')

cat_order = ['mono_correct', 'nonmono_correct', 'genuine_overattr', 'under_attr']

for ax, (xcol, xlabel, title_suffix, rho_show) in zip(axes, [
    ('ha_std', 'Head angle σ (session z-score)', 'raw', rho_raw),
    ('eta2',   'η² (nonlinear tuning strength)',  'controlling for η²', rho_partial),
]):
    for cat in cat_order:
        sub = df_v[df_v['category'] == cat]
        if len(sub) == 0:
            continue
        ax.scatter(sub[xcol], sub['ig_attr'],
                   color=PALETTE[cat], label=LABELS[cat].replace('\n', ' '),
                   s=30, alpha=0.7, linewidths=0, zorder=3)

    # Regression line
    xv = df_v[xcol].values; yv = df_v['ig_attr'].values
    m, b = np.polyfit(xv, yv, 1)
    xfit = np.linspace(xv.min(), xv.max(), 100)
    ax.plot(xfit, m * xfit + b, 'k--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("IG attribution (head_angle)", fontsize=11)
    ax.set_title(f"IG vs {xlabel}\nρ = {rho_show:.3f} ({title_suffix})", fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)

axes[0].legend(fontsize=8, loc='upper right', framealpha=0.8)
fig.suptitle("Baseline-distance confound: larger head_angle range → longer IG integration path",
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
savefig("fig4_baseline_distance_confound.png")


# ══════════════════════════════════════════════════════════════════
# FIG 5 — Pair category counts + summary
# ══════════════════════════════════════════════════════════════════
print("── Fig 5: Category summary ──")

cat_counts = df_valid['category'].value_counts()
cats_ordered = ['mono_correct', 'nonmono_correct', 'genuine_overattr', 'under_attr']
counts  = [cat_counts.get(c, 0) for c in cats_ordered]
colors  = [PALETTE[c] for c in cats_ordered]
labels_short = [
    'Monotonic\n(correct)',
    'Non-monotonic\n(correct via η²)',
    'Genuine\nover-attribution',
    'Under-\nattributed',
]

total_high_ig = cat_counts.get('mono_correct', 0) + cat_counts.get('nonmono_correct', 0) + cat_counts.get('genuine_overattr', 0)
rescued_pct   = 100 * cat_counts.get('nonmono_correct', 0) / max(1, cat_counts.get('nonmono_correct', 0) + cat_counts.get('genuine_overattr', 0))

fig = plt.figure(figsize=(11, 4.5))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.4], wspace=0.35)

# Left: bar chart
ax_bar = fig.add_subplot(gs[0])
bars = ax_bar.barh(range(len(cats_ordered)), counts, color=colors,
                   edgecolor='white', linewidth=0.8, height=0.6)
for bar, n in zip(bars, counts):
    ax_bar.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(n), va='center', ha='left', fontsize=11, fontweight='bold')
ax_bar.set_yticks(range(len(cats_ordered)))
ax_bar.set_yticklabels(labels_short, fontsize=10)
ax_bar.set_xlabel("Number of (session, ensemble) pairs", fontsize=11)
ax_bar.set_title("Head_angle pair classification\n(median η² and IG splits)", fontsize=11)
ax_bar.spines[['top', 'right']].set_visible(False)
ax_bar.set_xlim(0, max(counts) * 1.25)

# Right: stacked bar showing the "apparent over-attribution" breakdown
ax_r = fig.add_subplot(gs[1])
apparent_overattr   = cat_counts.get('nonmono_correct', 0) + cat_counts.get('genuine_overattr', 0)
actual_nonmono      = cat_counts.get('nonmono_correct', 0)
actual_genuine      = cat_counts.get('genuine_overattr', 0)

# Two rows: all high-IG pairs breakdown
categories_stacked  = ['High-IG pairs\n(n={})'.format(total_high_ig),
                        '"Apparent"\nover-attr\n(low Spearman,\nhigh IG)']
stacks = [
    [cat_counts.get('mono_correct', 0), apparent_overattr],  # mono correct
    [actual_nonmono, actual_genuine],                         # nonmono
]
# Simpler: stacked horizontal bars
ax_r.barh(0, total_high_ig, color='white', edgecolor='#CCC', linewidth=1, height=0.5)
ax_r.barh(0, cat_counts.get('mono_correct', 0),
          color=PALETTE['mono_correct'], height=0.5, label='Monotonic (correct)')
ax_r.barh(0, actual_nonmono,
          left=cat_counts.get('mono_correct', 0),
          color=PALETTE['nonmono_correct'], height=0.5, label='Non-monotonic (rescued)')
ax_r.barh(0, actual_genuine,
          left=cat_counts.get('mono_correct', 0) + actual_nonmono,
          color=PALETTE['genuine_overattr'], height=0.5, label='Genuine over-attribution')

ax_r.barh(-0.8, apparent_overattr, color='white', edgecolor='#CCC', linewidth=1, height=0.5)
ax_r.barh(-0.8, actual_nonmono,
          color=PALETTE['nonmono_correct'], height=0.5)
ax_r.barh(-0.8, actual_genuine,
          left=actual_nonmono,
          color=PALETTE['genuine_overattr'], height=0.5)

ax_r.set_yticks([0, -0.8])
ax_r.set_yticklabels([f'All high-IG pairs\n(n={total_high_ig})',
                       f'"Apparent" over-attr\n(low Spearman, high IG)\n(n={apparent_overattr})'],
                     fontsize=10)
ax_r.set_xlabel("Number of pairs", fontsize=11)
ax_r.set_title(f"Of 'apparent over-attribution':\n{rescued_pct:.0f}% is genuine non-monotonic tuning", fontsize=11)
ax_r.legend(fontsize=9, loc='lower right', framealpha=0.9)
ax_r.spines[['top', 'right']].set_visible(False)

fig.suptitle("IG over-attribution to head_angle: mostly correct detection of non-monotonic tuning",
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
savefig("fig5_category_summary.png")


# ══════════════════════════════════════════════════════════════════
# FIG 6 — 2×3 conceptual summary (one figure to rule them all)
# ══════════════════════════════════════════════════════════════════
print("── Fig 6: Combined summary panel ──")

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38,
                       left=0.07, right=0.97, top=0.92, bottom=0.08)

# ─── Panel A: tuning curve (E23 S20) ───
ax_a = fig.add_subplot(gs[0, 0])
s_i, n_i = 19, 22   # session index 19 = S20 (0-indexed), ensemble 22 = E23
sk = session_keys[s_i]
X_s = np.concatenate([dataset[sk]['data'][t] for t in sorted(dataset[sk]['data'])], axis=0).astype(np.float32)
Y_s = np.concatenate([dataset[sk]['labels'][t] for t in sorted(dataset[sk]['labels'])], axis=0).astype(np.float32)
ha   = X_s[:, HA_FEAT_IDX]
ens  = Y_s[:, n_i]
bin_edges = np.percentile(ha, np.linspace(0, 100, N_BINS + 1)); bin_edges[-1] += 1e-6
centers_a, means_a, sems_a = [], [], []
for b in range(N_BINS):
    mask_b = (ha >= bin_edges[b]) & (ha < bin_edges[b + 1])
    if mask_b.sum() >= MIN_BIN_PTS:
        centers_a.append(float(np.mean(ha[mask_b])))
        means_a.append(float(np.mean(ens[mask_b])))
        sems_a.append(float(np.std(ens[mask_b]) / np.sqrt(mask_b.sum())))
centers_a = np.array(centers_a); means_a = np.array(means_a); sems_a = np.array(sems_a)
rho_a, _ = spearmanr(ha, ens)
eta2_a = df[(df['session'] == 20) & (df['ensemble'] == 'E23')]['eta2'].values[0]
ig_a   = df[(df['session'] == 20) & (df['ensemble'] == 'E23')]['ig_attr'].values[0]

ax_a.fill_between(centers_a, means_a - sems_a, means_a + sems_a,
                   alpha=0.25, color=PALETTE['nonmono_correct'])
ax_a.plot(centers_a, means_a, 'o-', color=PALETTE['nonmono_correct'],
           linewidth=2.5, markersize=6, zorder=3)
ax_a.axhline(0, color='#AAAAAA', linewidth=0.7, linestyle='--')
ax_a.text(0.97, 0.97,
           f"ρ = {rho_a:+.3f}  →  miss\nη² = {eta2_a:.3f}  →  hit\nIG = {ig_a:.3f}  →  hit",
           transform=ax_a.transAxes, fontsize=8.5, va='top', ha='right',
           bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=PALETTE['nonmono_correct'], lw=1.2))
ax_a.set_xlabel("Head angle (z-score)", fontsize=10)
ax_a.set_ylabel("Ensemble activity (z-score)", fontsize=10)
ax_a.set_title("(A) Inverted-U tuning — E23, S20", fontsize=10, fontweight='bold',
                color=PALETTE['nonmono_correct'])
ax_a.spines[['top', 'right']].set_visible(False)

# ─── Panel B: η² vs |Spearman| scatter ───
ax_b = fig.add_subplot(gs[0, 1])
df_all_b = df[df['eta2'].notna() & df['ig_attr'].notna()].copy()
df_all_b['abs_rho'] = df_all_b['spearman_rho'].abs()
cat_col = df_all_b['category'].map(PALETTE).fillna(PALETTE['other'])
ax_b.scatter(df_all_b['abs_rho'], df_all_b['eta2'],
             c=cat_col, s=22, alpha=0.65, linewidths=0, zorder=3)
x_ref = np.linspace(0, df_all_b['abs_rho'].max(), 100)
ax_b.plot(x_ref, x_ref**2, 'k--', linewidth=1.5, alpha=0.6, label="y = x² (monotonic)")
ax_b.fill_between(x_ref, x_ref**2, x_ref**2 + 0.25, alpha=0.06, color='orange')
ax_b.text(0.35, 0.70, "nonlinear\ngap", fontsize=8.5, color='#c06000',
           transform=ax_b.transAxes, style='italic')
ax_b.set_xlabel("|Spearman ρ|", fontsize=10)
ax_b.set_ylabel("η²", fontsize=10)
ax_b.set_title("(B) η² vs |Spearman ρ|", fontsize=10, fontweight='bold')
legend_els = [Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE[c], markersize=7, label=LABELS[c].split('\n')[0])
              for c in cats_ordered if c in PALETTE]
legend_els.append(Line2D([0], [0], linestyle='--', color='k', label="monotonic"))
ax_b.legend(handles=legend_els, fontsize=7.5, loc='upper left', framealpha=0.9)
ax_b.spines[['top', 'right']].set_visible(False)

# ─── Panel C: Alignment bar ───
ax_c = fig.add_subplot(gs[0, 2])
methods_c  = ['IG\nhead_angle', 'GPV\nhead_angle', 'IG\nha_vel', 'GPV\nha_vel']
vals_spear = [0.055, 0.268, 0.700, 0.674]
vals_eta2  = [0.669, 0.774, 0.700, 0.674]
xc = np.arange(len(methods_c)); w = 0.38
ax_c.bar(xc - w/2, vals_spear, w, color='#90CAF9', label='Spearman ρ ground truth',
         edgecolor='white')
ax_c.bar(xc + w/2, vals_eta2,  w, color='#1565C0', label='η² ground truth',
         edgecolor='white')
for xi, (vs, ve) in enumerate(zip(vals_spear, vals_eta2)):
    ax_c.text(xi - w/2, vs + 0.02, f"{vs:.2f}", ha='center', va='bottom', fontsize=7.5, color='#333')
    ax_c.text(xi + w/2, ve + 0.02, f"{ve:.2f}", ha='center', va='bottom', fontsize=7.5, color='#003c8f')
ax_c.axvline(1.5, color='#CCCCCC', linewidth=1.2, linestyle='--')
ax_c.set_xticks(xc); ax_c.set_xticklabels(methods_c, fontsize=9)
ax_c.set_ylabel("Alignment ρ", fontsize=10)
ax_c.set_ylim(-0.1, 1.0)
ax_c.set_title("(C) Alignment: Spearman vs η² ground truth", fontsize=10, fontweight='bold')
ax_c.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax_c.spines[['top', 'right']].set_visible(False)
ax_c.axhline(0, color='black', linewidth=0.7)

# ─── Panel D: Baseline-distance (IG vs ha_std) ───
ax_d = fig.add_subplot(gs[1, 0])
df_v2 = df_valid.dropna(subset=['ha_std', 'ig_attr', 'eta2']).copy()
for cat in cats_ordered:
    sub = df_v2[df_v2['category'] == cat]
    ax_d.scatter(sub['ha_std'], sub['ig_attr'],
                 color=PALETTE[cat], s=25, alpha=0.7, linewidths=0, zorder=3,
                 label=LABELS[cat].replace('\n', ' '))
m_d, b_d = np.polyfit(df_v2['ha_std'].values, df_v2['ig_attr'].values, 1)
xfit_d   = np.linspace(df_v2['ha_std'].min(), df_v2['ha_std'].max(), 100)
ax_d.plot(xfit_d, m_d * xfit_d + b_d, 'k--', linewidth=1.5, alpha=0.7)
rho_d, _ = scipy_stats.spearmanr(df_v2['ha_std'], df_v2['ig_attr'])
ax_d.text(0.97, 0.97, f"ρ = {rho_d:.3f}", transform=ax_d.transAxes,
           fontsize=9, va='top', ha='right',
           bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#888', lw=0.8))
ax_d.set_xlabel("Head angle σ per session", fontsize=10)
ax_d.set_ylabel("IG attribution (head_angle)", fontsize=10)
ax_d.set_title("(D) Baseline-distance effect:\nlarger σ → longer integration path → higher IG",
               fontsize=10, fontweight='bold')
ax_d.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
ax_d.spines[['top', 'right']].set_visible(False)

# ─── Panel E: Category counts (horizontal bar) ───
ax_e = fig.add_subplot(gs[1, 1])
bars_e = ax_e.barh(range(len(cats_ordered)), counts, color=colors,
                    edgecolor='white', linewidth=0.8, height=0.55)
for bar, n, pct in zip(bars_e, counts, [c/sum(counts)*100 for c in counts]):
    ax_e.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
              f"{n}  ({pct:.0f}%)", va='center', ha='left', fontsize=9, fontweight='bold')
ax_e.set_yticks(range(len(cats_ordered)))
ax_e.set_yticklabels(labels_short, fontsize=9)
ax_e.set_xlabel("Number of valid pairs", fontsize=10)
ax_e.set_title(f"(E) Pair classifications\n(n={sum(counts)} total)", fontsize=10, fontweight='bold')
ax_e.set_xlim(0, max(counts) * 1.35)
ax_e.spines[['top', 'right']].set_visible(False)

# ─── Panel F: Key message text ───
ax_f = fig.add_subplot(gs[1, 2])
ax_f.axis('off')
summary_text = (
    "Summary\n\n"
    "IG appears to 'over-attribute' to head_angle\n"
    "because Spearman ρ is the wrong baseline:\n\n"
    "  • Many ensembles show non-monotonic\n"
    "    (inverted-U) head angle tuning curves\n\n"
    "  • Spearman ρ ≈ 0 for these (cancellation),\n"
    "    but η² is large (clear functional relationship)\n\n"
    f"  • {rescued_pct:.0f}% of 'apparent over-attribution'\n"
    "    is actually correct detection of non-\n"
    "    monotonic tuning\n\n"
    "  • Remaining genuine over-attribution: driven\n"
    "    by large head_angle range (baseline-distance\n"
    "    effect: IG integrates from baseline=0)\n\n"
    "  • head_angle_vel (monotonic) shows equally\n"
    "    high IG alignment with both metrics"
)
ax_f.text(0.05, 0.97, summary_text, transform=ax_f.transAxes,
           fontsize=9.5, va='top', ha='left',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='#F5F5F5', edgecolor='#BBBBBB', lw=1.2),
           family='monospace')
ax_f.set_title("(F) Key take-away", fontsize=10, fontweight='bold')

fig.suptitle("Head angle attribution: IG's apparent over-attribution is mostly correct non-monotonic tuning detection",
             fontsize=13, fontweight='bold')
savefig("fig6_combined_summary.png")

print(f"\nDone. All figures in:\n  {OUT_DIR}\n  {DESK_DIR}")
