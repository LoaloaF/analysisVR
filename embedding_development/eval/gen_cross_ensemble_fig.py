#!/usr/bin/env python3
"""
gen_cross_ensemble_fig.py

Re-filters cross_ensemble_prediction.npy with strict dual validity:
  - source ensemble must have mean R² ≥ 0.1 (model actually predicts it)
  - target ensemble must also have mean R² ≥ 0.1 (target is decodable,
    so failure to cross-predict reflects encoding specificity, not noise)

Then plots self-target vs cross-target R² as a violin+jitter figure per model.

Output: outputs/mlps/ensembles_multiseed/cross_ensemble_prediction.png (9.5"×4.2")
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import FONT, FIG, apply_style, add_footnote, savefig_manifest

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, '..')
mdir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')

R2_THR = 0.1

# ── Load precomputed records ───────────────────────────────────────────────────
recs = np.load(os.path.join(mdir, 'cross_ensemble_prediction.npy'), allow_pickle=True)

mlp_r2 = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)
cc_r2  = np.nanmean(np.load(os.path.join(root, 'outputs', 'cebra_64d_eval',
                                           'ensembles', 'all_r2.npy')), axis=0)
cp_r2  = np.nanmean(np.load(os.path.join(root, 'outputs', 'cebra_pred_64d_eval',
                                           'ensembles', 'all_r2.npy')), axis=0)
R2_MATS = {'mlp': mlp_r2, 'cc': cc_r2, 'cp': cp_r2}

MODEL_LABELS = {'mlp': 'MLP', 'cc': 'TC-Cont', 'cp': 'TC-Pred'}
COLORS       = {'mlp': '#2CA02C', 'cc': '#1F77B4', 'cp': '#FF7F0E'}

# ── Filter: both source AND target valid ──────────────────────────────────────
def is_valid(rec):
    r2_mat = R2_MATS.get(rec['model'])
    if r2_mat is None: return False
    s = rec['s_idx']
    return (r2_mat[s, rec['src_e']] >= R2_THR and
            r2_mat[s, rec['tgt_e']] >= R2_THR)

filtered = [r for r in recs if is_valid(r)]
print(f'Records after dual-validity filter: {len(filtered)} / {len(recs)}')

models = ['mlp', 'cc', 'cp']

self_data  = {m: [] for m in models}
cross_data = {m: [] for m in models}

for r in filtered:
    m = r['model']
    if r['is_self'] or r['src_e'] == r['tgt_e']:
        self_data[m].append(r['r2'])
    else:
        cross_data[m].append(r['r2'])

for m in models:
    n_s = len(self_data[m]); n_c = len(cross_data[m])
    med_s = np.median(self_data[m]) if n_s else np.nan
    med_c = np.median(cross_data[m]) if n_c else np.nan
    print(f'{MODEL_LABELS[m]:12s}  self n={n_s} med={med_s:.3f}   cross n={n_c} med={med_c:.3f}')

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=FIG.FULL, sharey=True)
apply_style(fig, list(axes))
rng = np.random.default_rng(0)

for ax, m in zip(axes, models):
    s_data = np.array(self_data[m])
    c_data = np.array(cross_data[m])
    color  = COLORS[m]

    vp = ax.violinplot([s_data, c_data], positions=[0, 1],
                       showmedians=True, showextrema=True, widths=0.55)
    for pc in vp['bodies']:
        pc.set_facecolor(color); pc.set_alpha(0.45)
    for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
        vp[part].set_color('#333'); vp[part].set_linewidth(0.9)

    for pos, data in [(0, s_data), (1, c_data)]:
        jitter = rng.uniform(-0.07, 0.07, len(data))
        ax.scatter(pos + jitter, data, s=4, color=color, alpha=0.35,
                   linewidths=0, zorder=3)
        ax.text(pos, np.median(data) + 0.01,
                f'{np.median(data):.3f}',
                ha='center', va='bottom',
                fontsize=FONT.ANNOTATION - 1, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Self-target', 'Cross-target'], fontsize=FONT.TICK - 1)
    ax.set_title(MODEL_LABELS[m], fontsize=FONT.LABEL - 1)
    ax.axhline(0, color='#888', lw=0.6, ls='--')
    ax.axhline(R2_THR, color='#E53935', lw=0.8, ls=':', alpha=0.7)

axes[0].set_ylabel('Cross-ensemble ridge R²', fontsize=FONT.LABEL - 1)

n_filt = len(filtered)
add_footnote(fig,
    f'Source and target both R²≥{R2_THR} (n={n_filt} filtered records); '
    'red dotted line = R²=0.1 threshold; '
    'cross-target: source embedding predicts a different ensemble\'s activity.')

savefig_manifest(fig, 'cross_ensemble_prediction.png',
                 [mdir, '/mnt/c/Users/amits/Desktop'])
print('Saved cross_ensemble_prediction.png')
