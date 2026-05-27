#!/usr/bin/env python3
"""
eval_head_angle_stability.py

Phase 1b: Per-session head angle tuning curve stability for a single ensemble.

Tests E08 (idx 7) and E18 (idx 17) and picks the one with higher cross-session
stability (lower std of preferred head angle across sessions).

Output: head_angle_stability.png  (FIG.FULL = 9.5 × 4.2")
        head_angle_tuning_6panel.png  (FIG.FULL = 9.5 × 4.2") — 6 varied shapes
"""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.figure_style import (
    FIG, DPI, FONT, FEATURE_NAMES, AXIS_LABELS,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
N_BINS      = 10      # decile bins for head angle
MIN_BIN_PTS = 5       # skip sessions with fewer than this per bin
HA_IDX      = 5       # head_angle column index in session dataset
CANDIDATE_ENSEMBLES = [7, 17]   # E08 (idx 7), E18 (idx 17)

base   = os.path.dirname(os.path.abspath(__file__))
root   = os.path.join(base, "..")
mdir   = os.path.join(root, "outputs", "mlps", "ensembles_multiseed")
OUT_DIRS = [mdir, '/mnt/c/Users/amits/Desktop']

# ─── LOAD ─────────────────────────────────────────────────────────────────────
with open(os.path.join(root, "outputs", "session_dataset_ensembles.pkl"), "rb") as f:
    ds = pickle.load(f)
session_ids = list(ds.keys())

all_r2  = np.load(os.path.join(mdir, "all_r2.npy"))    # (5, 29, 23)
mean_r2 = np.nanmean(all_r2, axis=0)                   # (29, 23)

# ─── TUNING CURVE COMPUTATION ─────────────────────────────────────────────────
def compute_tuning_curve(s_idx, n_idx, n_bins=N_BINS):
    """
    Returns (bin_centers, bin_means, bin_sems) for head_angle tuning.
    Returns None if too few data points.
    """
    sess_id = session_ids[s_idx]
    sd      = ds[sess_id]
    all_t   = list(sd['data'].keys())
    Xs = np.concatenate([sd['data'][t]   for t in all_t]).astype(np.float32)
    Ys = np.concatenate([sd['labels'][t] for t in all_t]).astype(np.float32)

    ha  = Xs[:, HA_IDX].astype(float)
    act = Ys[:, n_idx].astype(float)

    # Decile bin edges from actual data
    q = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(ha, q)
    edges[-1] += 1e-6

    bin_centers, bin_means, bin_sems = [], [], []
    for b in range(n_bins):
        mask = (ha >= edges[b]) & (ha < edges[b + 1])
        pts  = act[mask]
        if len(pts) < MIN_BIN_PTS:
            return None
        bin_centers.append(0.5 * (edges[b] + edges[b + 1]))
        bin_means.append(np.mean(pts))
        bin_sems.append(np.std(pts) / np.sqrt(len(pts)))

    return np.array(bin_centers), np.array(bin_means), np.array(bin_sems)


# ─── EVALUATE CANDIDATES ──────────────────────────────────────────────────────
best_ensemble = None
best_stability = np.inf
candidate_data = {}

for n_idx in CANDIDATE_ENSEMBLES:
    curves = {}
    pref_angles = []
    for s_idx in range(mean_r2.shape[0]):
        if not np.isfinite(mean_r2[s_idx, n_idx]) or mean_r2[s_idx, n_idx] < 0.01:
            continue
        result = compute_tuning_curve(s_idx, n_idx)
        if result is None:
            continue
        centers, means, sems = result
        curves[s_idx] = (centers, means, sems)
        # Preferred angle = center of bin with max mean activation
        pref_angles.append(centers[np.argmax(means)])

    if len(pref_angles) < 3:
        print(f"E{n_idx+1:02d}: only {len(pref_angles)} sessions — too few")
        continue

    stability = float(np.std(pref_angles))
    print(f"E{n_idx+1:02d}: {len(curves)} sessions  "
          f"pref_angle_std={stability:.3f}  "
          f"mean_pref={np.mean(pref_angles):.3f}")

    candidate_data[n_idx] = {'curves': curves, 'pref_angles': pref_angles}

    if stability < best_stability:
        best_stability = stability
        best_ensemble  = n_idx

if best_ensemble is None:
    print("ERROR: no ensemble had enough valid sessions")
    raise SystemExit(1)

print(f"\nPicked E{best_ensemble+1:02d} (std of preferred angle = {best_stability:.3f})")

# ─── STABILITY FIGURE ─────────────────────────────────────────────────────────
curves      = candidate_data[best_ensemble]['curves']
pref_angles = candidate_data[best_ensemble]['pref_angles']
n_curves    = len(curves)

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG.FULL, gridspec_kw={'width_ratios': [2, 1]})
apply_style(fig, [ax_l, ax_r])

# Left: overlaid curves
all_means = np.stack([curves[s][1] for s in curves])
mean_curve = np.mean(all_means, axis=0)
example_centers = curves[list(curves.keys())[0]][0]

for s_idx, (centers, means, sems) in curves.items():
    ax_l.plot(centers, means, alpha=0.25, linewidth=0.9, color='#4a90d9')

ax_l.plot(example_centers, mean_curve, color='#1a5fa8', linewidth=2.5,
          label=f'Mean (n={n_curves} sessions)')
ax_l.axhline(0, color='#888', lw=0.7, linestyle='--')
ax_l.set_xlabel(FEATURE_NAMES['head_angle'], fontsize=FONT.LABEL)
ax_l.set_ylabel(AXIS_LABELS['activity'], fontsize=FONT.LABEL)
ax_l.legend(fontsize=FONT.LEGEND, frameon=False)
add_panel_label(ax_l, 'A')

# Right: preferred angle bar per session
session_indices = list(curves.keys())
x = np.arange(len(session_indices))
ax_r.bar(x, pref_angles, color='#4a90d9', alpha=0.85, width=0.65)
ax_r.axhline(np.mean(pref_angles), color='#1a5fa8', lw=1.5, linestyle='--',
             label=f'mean = {np.mean(pref_angles):.2f}')
ax_r.set_xlabel(AXIS_LABELS['session'], fontsize=FONT.LABEL)
ax_r.set_ylabel(FEATURE_NAMES['head_angle'], fontsize=FONT.LABEL)
ax_r.set_xticks(x[::2])
ax_r.set_xticklabels([f"S{s+1}" for s in session_indices[::2]],
                     rotation=45, ha='right', fontsize=FONT.TICK - 2)
ax_r.legend(fontsize=FONT.LEGEND, frameon=False)
add_panel_label(ax_r, 'B')

add_footnote(fig,
    f"E{best_ensemble+1:02d}; {n_curves} sessions; "
    f"preferred head angle std = {best_stability:.3f}; "
    f"decile bins, mean ± SEM per bin")

savefig_manifest(fig, "head_angle_stability.png", OUT_DIRS)
print("Generated head_angle_stability.png")

# Update pending_questions: record which ensemble was chosen
chosen = f"E{best_ensemble+1:02d}"
other  = [f"E{i+1:02d}" for i in CANDIDATE_ENSEMBLES if i != best_ensemble]
print(f"\nResolved pending question: {chosen} wins (pref_angle_std={best_stability:.3f})")
print(f"Other candidate(s): {other}")


# ─── 6-PANEL TUNING CURVES FIGURE ────────────────────────────────────────────
# Pick 6 pairs from the best ensemble showing varied tuning shapes
# (mix of sessions with distinct tuning curve profiles)
all_s = list(curves.keys())
n_show = min(6, len(all_s))
# Sort by shape diversity: use variance of tuning curve as proxy
curve_var = [(s, np.var(curves[s][1])) for s in all_s]
curve_var_sorted = sorted(curve_var, key=lambda x: x[1], reverse=True)
show_sessions = [s for s, _ in curve_var_sorted[:n_show]]

fig6, axes6 = plt.subplots(2, 3, figsize=FIG.FULL)
apply_style(fig6, axes6.ravel())

panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']
for ax, s_idx, pl in zip(axes6.ravel(), show_sessions, panel_labels):
    centers, means, sems = curves[s_idx]
    ax.plot(centers, means, color='#2ca02c', linewidth=1.8)
    ax.fill_between(centers, means - sems, means + sems,
                    color='#2ca02c', alpha=0.25)
    ax.axhline(0, color='#888', lw=0.6, linestyle='--')
    ax.set_xlabel(FEATURE_NAMES['head_angle'], fontsize=FONT.LABEL - 2)
    if pl in ['A', 'D']:
        ax.set_ylabel(AXIS_LABELS['activity'], fontsize=FONT.LABEL - 2)
    ax.tick_params(labelsize=FONT.TICK - 2)
    add_panel_label(ax, pl)
    ax.text(0.98, 0.97, f"S{s_idx+1:02d}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FONT.ANNOTATION - 1, color='dimgray')

add_footnote(fig6,
    f"E{best_ensemble+1:02d}; 6 sessions selected for shape diversity; "
    f"mean ± SEM across timepoints in each decile bin")

savefig_manifest(fig6, "head_angle_tuning_6panel.png", OUT_DIRS)
print("Generated head_angle_tuning_6panel.png")
