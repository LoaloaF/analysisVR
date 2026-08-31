#!/usr/bin/env python3
"""
eval_evolution_lineplots.py  —  S47 (or wherever inserted)

Redraws the attribution evolution lineplots from pre-computed saved arrays —
no recomputation of GPV / IG required.

Outputs:
  outputs/mlps/ensembles_multiseed/evolution_lineplot_ig.png
  outputs/mlps/ensembles_multiseed/evolution_lineplot_gpv.png
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import (
    FONT, LINE, PALETTE, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, savefig_manifest,
)

root   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ddir   = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [ddir, '/mnt/c/Users/amits/Desktop']

SEEDS       = [42, 43, 44, 45, 46]
R2_THRESH   = 0.01
PREFIX      = 'E'
N_SESSIONS  = 29
N_ENSEMBLES = 23
N_GROUPS    = 11

GROUP_NAMES_RAW = [
    'frame_raw_500msMedian',
    'frame_raw_abs_acc_500msMedian',
    'frame_YawPitch_abs_vel_sum_500msMedian',
    'frame_YawPitch_abs_acc_sum_500msMedian',
    'head_angle_vel',
    'head_angle',
    'frame_position',
    'cue_visible',
    'upcoming_choice',
    'reward_window',
    'lick_detected',
]
GROUP_LABELS = [FEATURE_NAMES_SHORT.get(g, g) for g in GROUP_NAMES_RAW]

# 11 distinct colors — PALETTE (10) + one extra
_COLORS = PALETTE + ['#B8860B']

# ── Load data ────────────────────────────────────────────────────────────────
all_r2  = np.load(os.path.join(ddir, 'all_r2.npy'))          # (5, 29, 23)
gpv_med = np.load(os.path.join(ddir, 'importance_global_pv_semantic.npy'))  # (29,23,11)

gpv_seeds = []
for s in SEEDS:
    p = os.path.join(ddir, f'importance_global_pv_semantic_seed{s}.npy')
    if os.path.exists(p):
        gpv_seeds.append(np.load(p))

ig_seeds = []
for s in SEEDS:
    p = os.path.join(ddir, f'ig_checkpoint_seed{s}.npz')
    if os.path.exists(p):
        ig_seeds.append(np.load(p)['sem'])

ig_med = np.nanmedian(np.stack(ig_seeds, axis=0), axis=0) if ig_seeds else gpv_med.copy()

# ── Masks ────────────────────────────────────────────────────────────────────
mask_3d = np.isnan(all_r2)                     # (5, 29, 23)
mask    = np.any(mask_3d, axis=0)              # (29, 23) — any seed missing
mean_r2 = np.nanmean(np.where(mask_3d, np.nan, all_r2.astype(float)), axis=0)
low_r2  = mean_r2 < R2_THRESH                  # (29, 23)
full_mask = mask | low_r2                      # (29, 23)


def _apply_mask(arr):
    """NaN out (session, ensemble) pairs below R² threshold."""
    return np.where(full_mask[:, :, np.newaxis], np.nan, arr)


gpv_m = _apply_mask(gpv_med)
ig_m  = _apply_mask(ig_med)

gpv_seed_stack = [_apply_mask(a) for a in gpv_seeds]
ig_seed_stack  = [_apply_mask(a) for a in ig_seeds]

sess_x = np.arange(1, N_SESSIONS + 1)


# ── Pair selection ───────────────────────────────────────────────────────────
def _diverse_pairs(imp, max_pairs=9, max_per_neuron=2, max_per_group=2,
                   n_min_above=2, above_thr=0.05):
    mean_ng = np.nanmean(imp, axis=0)   # (ensembles, groups)
    std_ng  = np.nanstd(imp,  axis=0)
    n_above = np.array([
        [(~np.isnan(imp[:, ni, gi]) & (imp[:, ni, gi] >= above_thr)).sum()
         for gi in range(N_GROUPS)]
        for ni in range(N_ENSEMBLES)
    ])
    qualifies    = n_above >= n_min_above
    cv           = np.where((mean_ng > 1e-10) & qualifies, std_ng / (mean_ng + 1e-12), 0.0)
    mean_floored = np.where(qualifies, mean_ng, 0.0)
    cv           = np.nan_to_num(cv)
    mean_floored = np.nan_to_num(mean_floored)

    n_anchor = max_pairs // 2
    n_dyn    = max_pairs - n_anchor
    cnt_n, cnt_g = {}, {}

    def _pick(score, budget, exclude=None):
        order = np.argsort(score.ravel())[::-1]
        picked = []
        for fi in order:
            if len(picked) >= budget:
                break
            ni, gi = fi // N_GROUPS, fi % N_GROUPS
            if score[ni, gi] <= 0:
                break
            if exclude and (ni, gi) in exclude:
                continue
            if cnt_n.get(ni, 0) >= max_per_neuron:
                continue
            if cnt_g.get(gi, 0) >= max_per_group:
                continue
            picked.append((ni, gi))
            cnt_n[ni] = cnt_n.get(ni, 0) + 1
            cnt_g[gi]  = cnt_g.get(gi,  0) + 1
        return picked

    anchors  = _pick(mean_floored, n_anchor)
    dynamics = _pick(cv, n_dyn, exclude=set(anchors))
    return anchors + dynamics


# ── Plot ─────────────────────────────────────────────────────────────────────
def _draw_lineplots(imp, seed_stack, pairs, ylabel, fname):
    from collections import OrderedDict
    by_ensemble = OrderedDict()
    for ni, gi in pairs:
        by_ensemble.setdefault(ni, []).append(gi)
    neurons = list(by_ensemble.keys())

    ncols = min(3, len(neurons))
    nrows = int(np.ceil(len(neurons) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.17, nrows * 2.0),
                             sharey=True, squeeze=False)
    apply_style(fig, axes.flatten())

    mean_ng = np.nanmean(imp, axis=0)

    for pi, ni in enumerate(neurons):
        ax = axes[pi // ncols][pi % ncols]
        used_colors = []
        title_parts = []
        for li, gi in enumerate(by_ensemble[ni]):
            col = _COLORS[gi % len(_COLORS)]
            if col in used_colors:
                col = _COLORS[(gi + 7) % len(_COLORS)]
            used_colors.append(col)

            series = imp[:, ni, gi]
            if seed_stack:
                ss   = np.stack([sv[:, ni, gi] for sv in seed_stack], axis=0)
                lo   = np.nanpercentile(ss, 25, axis=0)
                hi   = np.nanpercentile(ss, 75, axis=0)
                ok   = ~(np.isnan(lo) | np.isnan(hi))
                if ok.any():
                    ax.fill_between(sess_x[ok], lo[ok], hi[ok], alpha=0.18, color=col)

            mu = mean_ng[ni, gi]
            ax.plot(sess_x, series, '-o', color=col,
                    markersize=3, lw=LINE.DATA)
            title_parts.append(f'{GROUP_LABELS[gi]} ({mu:.2f})')

        ax.axhline(0, color='#888', lw=LINE.REF, ls='--')
        title_str = f'{PREFIX}{ni+1:02d}: ' + '  |  '.join(title_parts)
        ax.set_title(title_str, fontsize=FONT.TICK, fontweight='bold')
        ax.set_xticks(sess_x[::5])
        ax.set_xticklabels([f'S{s}' for s in sess_x[::5]],
                           rotation=45, ha='right', fontsize=FONT.TICK - 1)
        if pi % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=FONT.TICK)

    for pi in range(len(neurons), nrows * ncols):
        axes[pi // ncols][pi % ncols].set_visible(False)

    plt.tight_layout()

    n_valid = int((~full_mask).sum())
    add_footnote(fig,
        f'{len(neurons)} ensembles shown (diverse selection: top mean + top temporal CV).  '
        f'IQR band = across {len(seed_stack)} seeds.  '
        f'{n_valid} valid (session, ensemble) pairs (R²≥{R2_THRESH}).')

    savefig_manifest(fig, fname, OUT_DIRS)
    print(f'Saved {fname}')


pairs_ig  = _diverse_pairs(ig_m,  max_pairs=6, max_per_neuron=1)
pairs_gpv = _diverse_pairs(gpv_m, max_pairs=6, max_per_neuron=1)

_draw_lineplots(
    ig_m, ig_seed_stack, pairs_ig,
    ylabel='Mean |IG|',
    fname='evolution_lineplot_ig.png',
)

_draw_lineplots(
    gpv_m, gpv_seed_stack, pairs_gpv,
    ylabel='GPV (ΔR²)',
    fname='evolution_lineplot_gpv.png',
)

print('Done.')
