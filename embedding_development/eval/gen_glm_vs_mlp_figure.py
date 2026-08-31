"""
GLM vs MLP figure (single-unit spikes):
  A: paired R2 scatter on signal-bearing units, colored by tuning monotonicity
  B: mean MLP-GLM gap for monotone vs non-monotone units

Depends on outputs/glm_results/glm_vs_mlp_spikes.npz (from gen_glm_vs_mlp.py).
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

os.chdir("/home/amitsant2000/ethz/VirtualReality/analysisVR/embedding_development")
sys.path.insert(0, os.path.abspath("."))
from utils.figure_style import FIG, apply_style, savefig_manifest

B = "outputs/glm_input_data/"
vals = np.load(B + "behavior_glm_input.npy", allow_pickle=True)
cols = list(np.load(B + "behavior_glm_input_columns.npy", allow_pickle=True))
idx  = np.load(B + "behavior_glm_input_index.npy", allow_pickle=True)
fr   = np.load(B + "fr_full.npy")
d = np.load("outputs/glm_results/glm_vs_mlp_spikes.npz", allow_pickle=True)
glm, mlp, sess_order = d['glm_r2'], d['mlp_r2'], d['sessions']

CONT = ['frame_raw_500msMedian','frame_raw_abs_acc_500msMedian','frame_YawPitch_abs_vel_sum_500msMedian',
        'frame_YawPitch_abs_acc_sum_500msMedian','head_angle_vel','head_angle','frame_position']
CAT  = ['cue_visible','upcoming_choice','reward_window','lick_detected']
sess_name = np.array([t[0] for t in idx])
ha = pd.to_numeric(pd.Series(vals[:, cols.index('head_angle')]), errors='coerce').values
Xc = np.column_stack([pd.to_numeric(pd.Series(vals[:, cols.index(c)]), errors='coerce').values for c in CONT])
nanm = np.isnan(Xc).any(1) | np.isnan(ha)
for c in CAT:
    nanm |= np.isnan(pd.to_numeric(pd.Series(vals[:, cols.index(c)]), errors='coerce').values)
valid = ~nanm

def eta2_rho2(x, y, nb=9):
    if len(np.unique(x)) < nb or y.std() < 1e-9: return np.nan, np.nan
    rho = spearmanr(x, y)[0]; rho = 0 if np.isnan(rho) else rho
    q = np.quantile(x, np.linspace(0, 1, nb + 1)); q[-1] += 1e-9
    b = np.clip(np.digitize(x, q[1:-1]), 0, nb - 1)
    ybar = y.mean(); sst = ((y - ybar) ** 2).sum()
    ssb = sum(((y[b == k].mean() - ybar) ** 2) * (b == k).sum() for k in range(nb) if (b == k).sum() > 0)
    return (ssb / sst if sst > 0 else np.nan), rho ** 2

G, M, NB = [], [], []
for si, s in enumerate(sess_order):
    r = np.where(valid & (sess_name == s))[0]
    x = ha[r]; Y = fr[r]
    for u in range(fr.shape[1]):
        g, m = glm[si, u], mlp[si, u]
        if not (np.isfinite(g) and np.isfinite(m)) or max(g, m) < 0.02:
            continue
        e2, r2 = eta2_rho2(x, Y[:, u].astype(float))
        if np.isnan(e2): continue
        G.append(g); M.append(m); NB.append(max(0, e2 - r2))
G, M, NB = np.array(G), np.array(M), np.array(NB)
nonmono = NB >= 0.01
gap = M - G
winrate = (M > G).mean()
print(f"n={len(G)}  MLP>GLM={winrate:.0%}  mono gap={gap[~nonmono].mean():+.3f}({(gap[~nonmono]>0).mean():.0%})"
      f"  nonmono gap={gap[nonmono].mean():+.3f}({(gap[nonmono]>0).mean():.0%})")

C_MONO, C_NON = '#1565C0', '#C62828'
fig, (axA, axB) = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig, [axA, axB])

# Panel A: paired scatter
lim = max(G.max(), M.max()) * 1.06
axA.plot([-0.05, lim], [-0.05, lim], 'k--', lw=1.0, alpha=0.6, zorder=2)
axA.scatter(G[~nonmono], M[~nonmono], s=14, alpha=0.55, c=C_MONO, linewidths=0, label='Monotone tuning', zorder=3)
axA.scatter(G[nonmono],  M[nonmono],  s=16, alpha=0.6,  c=C_NON,  linewidths=0, label='Non-monotone tuning', zorder=4)
axA.set_xlim(-0.05, lim); axA.set_ylim(-0.05, lim); axA.set_aspect('equal')
axA.set_xlabel('GLM  $R^2$'); axA.set_ylabel('MLP  $R^2$')
axA.legend(fontsize=7, frameon=False, loc='lower right')
axA.text(0.04, 0.96, f'MLP > GLM: {winrate:.0%}', transform=axA.transAxes, va='top', fontsize=9, color='dimgray')

# Panel B: mean gap by tuning type
means = [gap[~nonmono].mean(), gap[nonmono].mean()]
wins  = [(gap[~nonmono] > 0).mean(), (gap[nonmono] > 0).mean()]
ns    = [(~nonmono).sum(), nonmono.sum()]
bars = axB.bar([0, 1], means, color=[C_MONO, C_NON], width=0.6, alpha=0.85)
axB.axhline(0, color='k', lw=0.7)
for i, (mn, w, n) in enumerate(zip(means, wins, ns)):
    axB.text(i, mn + 0.002, f'{w:.0%} win\n(n={n})', ha='center', va='bottom', fontsize=8)
axB.set_xticks([0, 1]); axB.set_xticklabels(['Monotone', 'Non-monotone'])
axB.set_ylabel('mean  MLP $-$ GLM  $R^2$')
axB.set_ylim(0, max(means) * 1.35)

savefig_manifest(fig, 'glm_vs_mlp.png', ['outputs/glm_results',
    '/home/amitsant2000/ethz/Ensemble-Based-Action-Embeddings-using-Neural-Network-Based-Encoding-Models/images'])
print("saved glm_vs_mlp.png")
