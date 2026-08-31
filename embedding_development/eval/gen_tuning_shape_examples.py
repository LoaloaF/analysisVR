#!/usr/bin/env python3
"""
gen_tuning_shape_examples.py

Classify head-angle tuning curves by shape, then show the 2 highest-IG
examples (R² >= 0.05) for each category:

  Monotone | Inverted-U | U-shape | W-shape | M-shape | Complex

Shape detection: 3-point smoothed bin means → count interior peaks/troughs
with prominence >= 18 % of the curve range.

Output: outputs/mlps/ensembles_multiseed/tuning_shape_examples.png
"""
import os, sys, pickle
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import DPI, FONT, apply_style, savefig_manifest

root     = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
mdir     = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [mdir, '/mnt/c/Users/amits/Desktop']

R2_THR   = 0.05
N_BINS   = 10
MIN_PTS  = 5
HA_F_IDX = 5   # head_angle column in feature matrix

# ── Load ──────────────────────────────────────────────────────────────────────
ig_mat = np.load(os.path.join(mdir, 'importance_ig_semantic.npy'))  # (n_sess, n_ens, n_grp)
r2_all = np.load(os.path.join(mdir, 'all_r2.npy'))                  # (n_seeds, n_sess, n_ens)

with open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb') as f:
    sg = pickle.load(f)
group_names = [g[0] for g in sg]
ha_g_idx    = group_names.index('head_angle')

mean_r2    = np.nanmean(r2_all, axis=0)                             # (n_sess, n_ens)
valid_mask = (mean_r2 >= R2_THR) & np.isfinite(mean_r2)

with open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb') as f:
    ds = pickle.load(f)
session_ids = list(ds.keys())

# ── Compute tuning curves ─────────────────────────────────────────────────────
print('Computing tuning curves...')
tuning_curves = {}   # (s_idx, e_idx) → (centers, means, sems)

for s_idx, sess_id in enumerate(session_ids):
    sd         = ds[sess_id]
    trial_keys = sorted(sd['data'].keys())
    X = np.concatenate([sd['data'][t]   for t in trial_keys], axis=0).astype(float)
    Y = np.concatenate([sd['labels'][t] for t in trial_keys], axis=0).astype(float)
    ha = X[:, HA_F_IDX]

    bin_edges      = np.percentile(ha, np.linspace(0, 100, N_BINS + 1))
    bin_edges[-1] += 1e-9

    for e_idx in range(Y.shape[1]):
        if not valid_mask[s_idx, e_idx]:
            continue
        y = Y[:, e_idx]
        centers, means, sems = [], [], []
        for b in range(N_BINS):
            in_b = (ha >= bin_edges[b]) & (ha < bin_edges[b + 1])
            if in_b.sum() >= MIN_PTS:
                vals = y[in_b]
                centers.append(float((bin_edges[b] + bin_edges[b + 1]) / 2))
                means.append(float(vals.mean()))
                sems.append(float(vals.std() / np.sqrt(len(vals))))
        if len(means) >= 6:
            tuning_curves[(s_idx, e_idx)] = (
                np.array(centers), np.array(means), np.array(sems))

print(f'  {len(tuning_curves)} valid tuning curves')

# ── Shape classifier ──────────────────────────────────────────────────────────
def classify_shape(means, prom=0.08):
    m  = np.array(means, dtype=float)
    n  = len(m)
    rng = m.max() - m.min()
    if rng < 1e-8:
        return 'flat'

    mn = (m - m.min()) / rng          # normalise to [0, 1]

    # 3-point moving average, edge-corrected
    sm      = np.convolve(mn, [1/3, 1/3, 1/3], mode='same')
    sm[0]   = (mn[0]  + mn[1])  / 2
    sm[-1]  = (mn[-2] + mn[-1]) / 2

    peaks   = [i for i in range(1, n - 1)
               if sm[i] > sm[i-1] and sm[i] > sm[i+1]
               and sm[i] - min(sm[i-1], sm[i+1]) >= prom]
    troughs = [i for i in range(1, n - 1)
               if sm[i] < sm[i-1] and sm[i] < sm[i+1]
               and max(sm[i-1], sm[i+1]) - sm[i] >= prom]

    np_, nt = len(peaks), len(troughs)

    if np_ == 0 and nt == 0:
        slope = float(np.polyfit(range(n), mn, 1)[0])
        return 'monotone_inc' if slope >= 0 else 'monotone_dec'
    if np_ == 1 and nt == 0:
        return 'inverted_u'
    if np_ == 0 and nt == 1:
        return 'u_shape'
    if np_ >= 2 and nt >= 1:
        return 'w_shape'
    if np_ >= 1 and nt >= 2:
        return 'm_shape'
    return 'complex'

shapes = {k: classify_shape(v[1]) for k, v in tuning_curves.items()}

print('\nShape distribution:')
for shape, count in sorted(Counter(shapes.values()).items()):
    print(f'  {shape}: {count}')

# ── Category definitions ──────────────────────────────────────────────────────
CATS = [
    ('monotone',   'Monotone',   '#1F77B4'),
    ('inverted_u', 'Inverted-U', '#2CA02C'),
    ('u_shape',    'U-shape',    '#FF7F0E'),
    ('complex',    'Complex',    '#8C564B'),
]

def canonical(s):
    if s in ('monotone_inc', 'monotone_dec'):
        return 'monotone'
    if s in ('w_shape', 'm_shape', 'flat'):
        return 'complex'
    return s

# ── Per-shape counts across all valid tuning-curve pairs ─────────────────────
total = len(shapes)
shape_counts = {}
for cat_key, _, _ in CATS:
    n = sum(1 for sh in shapes.values() if canonical(sh) == cat_key)
    shape_counts[cat_key] = n
print(f'\nShape counts (of {total} valid tuning-curve pairs):')
for cat_key, cat_label, _ in CATS:
    print(f'  {cat_label}: {shape_counts[cat_key]}  ({100*shape_counts[cat_key]/total:.1f}%)')

# ── Pick top-2 per category by IG ─────────────────────────────────────────────
N_EX = 2
selected = {}
print()
for cat_key, cat_label, _ in CATS:
    candidates = sorted(
        [(s, e, float(ig_mat[s, e, ha_g_idx]))
         for (s, e), sh in shapes.items()
         if canonical(sh) == cat_key],
        key=lambda x: x[2], reverse=True
    )
    selected[cat_key] = candidates[:N_EX]
    print(f'{cat_label}: {len(candidates)} candidates  '
          + '  '.join(f'S{s+1:02d}E{e+1:02d} IG={ig:.3f}'
                      for s, e, ig in selected[cat_key]))

# ── Figure: 2×2, one example per category ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.5),
                          gridspec_kw={'hspace': 0.55, 'wspace': 0.32})
apply_style(fig, axes.flatten())
fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.12)

for idx, (cat_key, cat_label, color) in enumerate(CATS):
    ax = axes[idx // 2, idx % 2]
    examples = selected[cat_key]
    if not examples:
        ax.set_visible(False)
        continue

    s, e, ig_val = examples[0]
    centers, means, sems = tuning_curves[(s, e)]
    ax.errorbar(centers, means, yerr=sems, fmt='-o', color=color,
                markersize=4, lw=1.6, elinewidth=1.0,
                capsize=2.5, capthick=1.0)
    ax.axhline(0, color='#aaa', lw=0.6, ls='--')
    pct = 100 * shape_counts[cat_key] / total
    ax.set_title(f'{cat_label}  ({pct:.0f}% of pairs)\n|IG|={ig_val:.3f}   R²={mean_r2[s,e]:.2f}',
                 fontsize=FONT.TICK, fontweight='bold', color=color)
    ax.set_xlabel('Head Angle (z)', fontsize=FONT.TICK - 1)
    ax.set_ylabel('Activity (z)', fontsize=FONT.TICK - 1)

savefig_manifest(fig, 'tuning_shape_examples.png', OUT_DIRS, skip_tight_layout=True)
print('\nSaved tuning_shape_examples.png')
