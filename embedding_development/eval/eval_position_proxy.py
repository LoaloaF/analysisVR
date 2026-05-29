#!/usr/bin/env python3
"""
eval_position_proxy.py

"Our models don't directly attribute to position — but that's because
 position variation in sensitive ensembles is captured by task-event and
 kinematic features that ARE highly attributed."

Figure: two-panel
  Left  — Scatter: for each non-position feature, plot its mean GPV attribution
           (for position-sensitive pairs) vs its mean Spearman |ρ| with position.
           Labeled points; task-event features expected top-right.
  Right — Position-binned mean of cue_visible and upcoming_choice for a
           representative session: shows task events fire at specific positions,
           explaining why attribution to those features proxies for position.

Position-sensitive pairs: R²≥0.01, position-neural |ρ| ≥ 0.15.
"""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
from scipy.stats import spearmanr

for _fp in ['/mnt/c/Windows/Fonts/arial.ttf', '/mnt/c/Windows/Fonts/arialbd.ttf']:
    if os.path.exists(_fp): _fm.fontManager.addfont(_fp)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import (
    FIG, DPI, FONT, FEATURE_NAMES_SHORT, AXIS_LABELS,
    apply_style, add_footnote, savefig_manifest,
)

base  = os.path.dirname(os.path.abspath(__file__))
root  = os.path.join(base, '..')
mdir  = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')

OUT_DIRS = [mdir, '/mnt/c/Users/amits/Desktop']

# ── Load ──────────────────────────────────────────────────────────────────────
with open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb') as f:
    ds = pickle.load(f)
session_ids = list(ds.keys())

with open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb') as f:
    sg = pickle.load(f)
group_names = [g[0] for g in sg]
feat_idx    = {g: cols[0] for g, cols in sg}   # first col index per group
POS_IDX     = feat_idx['frame_position']        # 6

all_r2   = np.load(os.path.join(mdir, 'all_r2.npy'))        # (5, 29, 23)
gpv      = np.load(os.path.join(mdir, 'importance_global_pv_semantic.npy'))  # (29, 23, 11)
mean_r2  = np.nanmean(all_r2, axis=0)                        # (29, 23)
valid    = (~np.all(np.isnan(gpv), axis=-1)) & (mean_r2 >= 0.01)

n_sessions, n_ensembles, n_groups = gpv.shape
short = [FEATURE_NAMES_SHORT.get(g, g) for g in group_names]

# ── Per-session: position-feature correlations and position-neural correlations
POS_RHO_THRESH = 0.15   # |ρ| between position and neural activity to call "position-sensitive"

feat_pos_rho   = np.full((n_sessions, n_groups), np.nan)   # feature-position |ρ|
pair_pos_rho   = np.full((n_sessions, n_ensembles), np.nan)  # position-neural |ρ|

for s_idx, sess_id in enumerate(session_ids):
    sd      = ds[sess_id]
    all_t   = list(sd['data'].keys())
    if not all_t:
        continue
    Xs = np.concatenate([sd['data'][t]   for t in all_t]).astype(float)  # (T, 17)
    Ys = np.concatenate([sd['labels'][t] for t in all_t]).astype(float)  # (T, 23)

    pos = Xs[:, POS_IDX]
    if np.nanstd(pos) < 1e-6:
        continue

    # Feature-position correlations
    for g_idx, (gname, cols) in enumerate(sg):
        feat_vals = Xs[:, cols[0]]
        if np.nanstd(feat_vals) < 1e-6:
            continue
        rho, _ = spearmanr(feat_vals, pos, nan_policy='omit')
        feat_pos_rho[s_idx, g_idx] = abs(rho)

    # Position-neural correlations
    for e_idx in range(n_ensembles):
        if not valid[s_idx, e_idx]:
            continue
        neural = Ys[:, e_idx]
        if np.nanstd(neural) < 1e-6:
            continue
        rho, _ = spearmanr(neural, pos, nan_policy='omit')
        pair_pos_rho[s_idx, e_idx] = abs(rho)

# ── Select position-sensitive pairs ──────────────────────────────────────────
pos_sensitive = valid & (pair_pos_rho >= POS_RHO_THRESH)
print(f"Position-sensitive pairs (|ρ|≥{POS_RHO_THRESH}, R²≥0.01): {pos_sensitive.sum()}")

# Mean GPV over position-sensitive pairs
gpv_ps = np.where(pos_sensitive[:, :, np.newaxis], gpv, np.nan)
mean_gpv = np.nanmean(gpv_ps.reshape(-1, n_groups), axis=0)  # (11,)

# Mean feature-position |ρ| over sessions that contribute at least one sensitive pair
has_pair = pos_sensitive.any(axis=1)   # (29,) sessions
mean_fpos = np.nanmean(feat_pos_rho[has_pair], axis=0)   # (11,)

# Representative session for scatter inset: pick the session with the most sensitive pairs
best_sess = int(np.argmax(pos_sensitive.sum(axis=1)))
sd_rep    = ds[session_ids[best_sess]]
all_t     = list(sd_rep['data'].keys())
Xs_rep    = np.concatenate([sd_rep['data'][t] for t in all_t]).astype(float)
pos_rep   = Xs_rep[:, POS_IDX]
spd_rep   = Xs_rep[:, feat_idx['frame_raw_500msMedian']]
ha_rep    = Xs_rep[:, feat_idx['head_angle']]
rho_spd_pos, _ = spearmanr(spd_rep, pos_rep, nan_policy='omit')
rho_ha_pos,  _ = spearmanr(ha_rep,  pos_rep, nan_policy='omit')

# Position-binned task-event means for representative session.
# For each categorical group pick the one-hot column with highest variance
# in the representative session (avoids the constant "not-active" column).
def _best_col_for_group(gname, Xs):
    """Return the column index and display label for the most variable one-hot column."""
    cols = [c for g, c in [(g, c) for g, cols in sg for c in cols] if g == gname]
    # Re-extract correctly: (gname, cols) tuples
    cols = [c for g, cs in sg if g == gname for c in cs]
    variances = [Xs[:, c].var() for c in cols]
    return cols[int(np.argmax(variances))]

# Pick the two most variable categorical features in the representative session
cat_names = [g for g, _ in sg if g in ('cue_visible', 'upcoming_choice', 'reward_window', 'lick_detected')]
cat_vars = []
for gname in cat_names:
    best_c = _best_col_for_group(gname, Xs_rep)
    var = Xs_rep[:, best_c].var()
    cat_vars.append((gname, best_c, var))
# Sort by variance descending, pick top 2
cat_vars.sort(key=lambda x: x[2], reverse=True)
cat_vars = cat_vars[:2]

n_bins = 20
pos_bins    = np.linspace(np.nanpercentile(pos_rep, 1), np.nanpercentile(pos_rep, 99), n_bins + 1)
bin_centers = 0.5 * (pos_bins[:-1] + pos_bins[1:])
binned_vars = []
for gname, cidx, _ in cat_vars:
    vals = Xs_rep[:, cidx]
    binned = np.array([np.nanmean(vals[(pos_rep >= pos_bins[i]) & (pos_rep < pos_bins[i+1])])
                       for i in range(n_bins)])
    lbl = FEATURE_NAMES_SHORT.get(gname, gname)
    binned_vars.append((lbl, binned))

print(f"Panel B variables: {[(g, c) for g, c, _ in cat_vars]}")
print(f"Representative session S{best_sess+1:02d}: "
      f"speed-pos ρ={rho_spd_pos:.2f}, head_angle-pos ρ={rho_ha_pos:.2f}")

# ── Figure ────────────────────────────────────────────────────────────────────
# Non-position feature mask (exclude group 6 = frame_position from both panels)
non_pos = [i for i in range(n_groups) if group_names[i] != 'frame_position']
short_np  = [short[i] for i in non_pos]
gpv_np    = mean_gpv[non_pos]
fpos_np   = mean_fpos[non_pos]

# Color points by category: continuous (0-5) = blue, categorical (7-10) = orange
CONT_CLR = '#2166ac'
CAT_CLR  = '#d6604d'
point_colors = [CAT_CLR if group_names[i] in
                ('cue_visible','upcoming_choice','reward_window','lick_detected')
                else CONT_CLR
                for i in non_pos]

fig, axes = plt.subplots(1, 2, figsize=(FIG.FULL[0], 4.2),
                          gridspec_kw={'width_ratios': [1.3, 1], 'wspace': 0.42})
apply_style(fig, axes)
fig.subplots_adjust(bottom=0.18)

# Panel A — scatter: attribution vs position-correlation per feature
ax = axes[0]
for xi, yi, ci, lbl in zip(fpos_np, gpv_np, point_colors, short_np):
    ax.scatter(xi, yi, s=60, color=ci, zorder=3, edgecolors='none')
    ax.annotate(lbl, (xi, yi), fontsize=FONT.TICK - 1,
                xytext=(4, 3), textcoords='offset points', color=ci)
ax.set_xlabel('Mean |ρ| with Position', fontsize=FONT.LABEL)
ax.set_ylabel('Mean GPV (ΔR²)', fontsize=FONT.LABEL)
ax.set_title('Feature attribution vs. feature–position correlation\n'
             '(position-sensitive pairs; position feature excluded)',
             fontsize=FONT.LABEL - 0.5, pad=6)
ax.text(0.03, 0.97, 'A', transform=ax.transAxes,
        fontsize=FONT.PANEL, fontweight='bold', va='top', ha='left')
# legend
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=CONT_CLR,
                  markersize=7, label='Kinematic'),
           Line2D([0],[0], marker='o', color='w', markerfacecolor=CAT_CLR,
                  markersize=7, label='Task event')]
ax.legend(handles=handles, fontsize=FONT.LEGEND, frameon=False)

# Panel B — position-binned task event occupancy for representative session
markers = ['o', 's', '^', 'D']
panel_colors = ['#9467bd', '#8c564b', '#2ca02c', '#1f77b4']
ax = axes[1]
for (lbl, binned), mk, pc in zip(binned_vars, markers, panel_colors):
    ax.plot(bin_centers, binned, color=pc, lw=2, label=lbl, marker=mk, ms=3)
ax.set_xlabel('Track Position (z-scored)', fontsize=FONT.LABEL)
ax.set_ylabel('Mean one-hot value', fontsize=FONT.LABEL)
ax.set_title(f'Task events vs. position\n(S{best_sess+1:02d}, binned)',
             fontsize=FONT.LABEL, pad=6)
ax.legend(fontsize=FONT.LEGEND, frameon=False)
ax.text(0.03, 0.97, 'B', transform=ax.transAxes,
        fontsize=FONT.PANEL, fontweight='bold', va='top', ha='left')

add_footnote(fig,
    f"{pos_sensitive.sum()} position-sensitive pairs (R²≥0.01, |ρ_pos-neural|≥{POS_RHO_THRESH}); "
    f"{has_pair.sum()} sessions; panel B: S{best_sess+1:02d}")

savefig_manifest(fig, 'position_collinearity.png', OUT_DIRS)
print("Generated position_collinearity.png")
