#!/usr/bin/env python3
"""
gen_attribution_consistency.py

Two-panel violin figure showing attribution profile consistency:
  Left:  Cross-architecture — Spearman ρ between GPV profiles of different model pairs
         (MLP×TC-Cont, MLP×TC-Pred, TC-Cont×TC-Pred) for pairs where both R²≥0.1
  Right: Cross-seed — Spearman ρ between MLP GPV profiles across random seeds
         for the same (session, ensemble) pair

Output: outputs/mlps/ensembles_multiseed/attribution_consistency_merged.png (9.5"×4.2")
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.figure_style import FONT, apply_style, add_footnote, savefig_manifest

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, '..')
mdir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
cdir = os.path.join(root, 'outputs', 'cebra_comparison')

SEEDS   = [42, 43, 44, 45, 46]
R2_THR  = 0.1

# ── Load ──────────────────────────────────────────────────────────────────────
mlp_gpv = np.load(os.path.join(mdir, 'importance_global_pv_semantic_seed42.npy'),
                  allow_pickle=True)                             # (29,23,11) seed42 only
mlp_r2  = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)  # (29,23)
cc_r2   = np.nanmean(np.load(os.path.join(root, 'outputs', 'cebra_eval',
                                            'ensembles', 'all_r2.npy')), axis=0)
cp_r2   = np.nanmean(np.load(os.path.join(root, 'outputs', 'cebra_pred_64d_eval',
                                            'ensembles', 'all_r2.npy')), axis=0)

def _load_gpv(subdir):
    """Load mean GPV: try per-seed files first, fall back to aggregate."""
    per_seed = []
    for s in SEEDS:
        p = os.path.join(root, 'outputs', subdir, 'ensembles',
                         f'importance_global_pv_semantic_seed{s}.npy')
        if os.path.exists(p):
            per_seed.append(np.load(p, allow_pickle=True))
    if per_seed:
        return np.nanmean(per_seed, axis=0)
    agg = os.path.join(root, 'outputs', subdir, 'ensembles',
                       'importance_global_pv_semantic.npy')
    return np.load(agg) if os.path.exists(agg) else None

cc_gpv = _load_gpv('cebra_eval')
cp_gpv = _load_gpv('cebra_pred_64d_eval')

# ── Cross-architecture Spearman ρ ─────────────────────────────────────────────
n_sess, n_ens = mlp_r2.shape

def cross_rhos(gpv_a, r2_a, gpv_b, r2_b):
    rhos = []
    for s in range(n_sess):
        for e in range(n_ens):
            if r2_a[s, e] < R2_THR or r2_b[s, e] < R2_THR: continue
            va, vb = gpv_a[s, e], gpv_b[s, e]
            ok = np.isfinite(va) & np.isfinite(vb)
            if ok.sum() < 4: continue
            r, _ = spearmanr(va[ok], vb[ok])
            rhos.append(r)
    return np.array(rhos)

arch_pairs = [
    ('MLP\nvs TC-Cont', cross_rhos(mlp_gpv, mlp_r2, cc_gpv, cc_r2), '#4CAF50'),
    ('MLP\nvs TC-Pred', cross_rhos(mlp_gpv, mlp_r2, cp_gpv, cp_r2), '#FF9800'),
    ('TC-Cont\nvs TC-Pred', cross_rhos(cc_gpv, cc_r2, cp_gpv, cp_r2), '#2196F3'),
]

# ── Cross-seed Spearman ρ (MLP + TC-Cont + TC-Pred) ──────────────────────────
def cross_seed_rho_list(seed_dir, r2_mat, seeds=SEEDS):
    """Compute all pairwise cross-seed Spearman ρ values for a model directory."""
    per_seed = []
    for s in seeds:
        p = os.path.join(root, 'outputs', seed_dir, 'ensembles',
                         f'importance_global_pv_semantic_seed{s}.npy') \
            if seed_dir != 'mlps/ensembles_multiseed' else \
            os.path.join(mdir, f'importance_global_pv_semantic_seed{s}.npy')
        if os.path.exists(p):
            per_seed.append(np.load(p, allow_pickle=True))
    if len(per_seed) < 2:
        return np.array([])
    rhos = []
    for s_idx in range(n_sess):
        for e_idx in range(n_ens):
            if r2_mat[s_idx, e_idx] < R2_THR: continue
            profiles = [g[s_idx, e_idx] for g in per_seed]
            for i in range(len(profiles)):
                for j in range(i + 1, len(profiles)):
                    va, vb = profiles[i], profiles[j]
                    ok = np.isfinite(va) & np.isfinite(vb)
                    if ok.sum() < 4: continue
                    r, _ = spearmanr(va[ok], vb[ok])
                    rhos.append(r)
    return np.array(rhos)

cross_seed_mlp  = cross_seed_rho_list('mlps/ensembles_multiseed', mlp_r2)
cross_seed_cc   = cross_seed_rho_list('cebra_eval',               cc_r2)
cross_seed_cp   = cross_seed_rho_list('cebra_pred_64d_eval',      cp_r2)

cross_seed_models = [
    ('MLP',      cross_seed_mlp,  '#2CA02C'),
    ('TC-Cont',  cross_seed_cc,   '#2196F3'),
    ('TC-Pred',  cross_seed_cp,   '#FF9800'),
]
for name, rhos, _ in cross_seed_models:
    print(f'  Cross-seed {name}: n={len(rhos)}  median={np.median(rhos):.3f}')

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.5, 4.2))
apply_style(fig, ax_l)
apply_style(fig, ax_r)
rng = np.random.default_rng(0)

# Left: cross-architecture
data_l = [p[1] for p in arch_pairs]
vp = ax_l.violinplot(data_l, positions=[0, 1, 2],
                     showmedians=True, showextrema=True, widths=0.55)
for pc, (_, _, c) in zip(vp['bodies'], arch_pairs):
    pc.set_facecolor(c); pc.set_alpha(0.45)
for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
    vp[part].set_color('#333'); vp[part].set_linewidth(0.9)
for i, (_, rhos, c) in enumerate(arch_pairs):
    jitter = rng.uniform(-0.08, 0.08, len(rhos))
    ax_l.scatter(i + jitter, rhos, s=5, color=c, alpha=0.35, linewidths=0, zorder=3)
    ax_l.text(i, np.median(rhos) + 0.025, f'ρ={np.median(rhos):.2f}',
              ha='center', va='bottom', fontsize=FONT.ANNOTATION - 1, fontweight='bold')
ax_l.set_xticks([0, 1, 2])
ax_l.set_xticklabels([p[0] for p in arch_pairs], fontsize=FONT.TICK - 1)
ax_l.set_ylabel('Spearman ρ of GPV profiles', fontsize=FONT.LABEL - 1)
ax_l.set_ylim(-0.15, 1.18)
ax_l.axhline(0.9, color='#888', ls=':', lw=0.8)
ax_l.set_title('Cross-architecture GPV agreement', fontsize=FONT.LABEL - 1)

# Right: cross-seed — all three models
valid_models = [(name, rhos, c) for name, rhos, c in cross_seed_models if len(rhos) >= 2]
positions_r  = list(range(len(valid_models)))

data_r = [rhos for _, rhos, _ in valid_models]
vp2 = ax_r.violinplot(data_r, positions=positions_r,
                      showmedians=True, showextrema=True, widths=0.55)
for pc, (_, _, c) in zip(vp2['bodies'], valid_models):
    pc.set_facecolor(c); pc.set_alpha(0.45)
for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
    vp2[part].set_color('#333'); vp2[part].set_linewidth(0.9)
for i, (name, rhos, c) in enumerate(valid_models):
    jitter = rng.uniform(-0.08, 0.08, len(rhos))
    ax_r.scatter(np.array(positions_r[i]) + jitter, rhos,
                 s=5, color=c, alpha=0.35, linewidths=0, zorder=3)
    med = np.median(rhos)
    ax_r.text(i, med + 0.025, f'ρ={med:.2f}',
              ha='center', va='bottom', fontsize=FONT.ANNOTATION - 1, fontweight='bold')

ax_r.set_xticks(positions_r)
ax_r.set_xticklabels([f'{name}\n(5 seeds)' for name, _, _ in valid_models],
                     fontsize=FONT.TICK - 1)
ax_r.set_ylabel('Spearman ρ of GPV profiles', fontsize=FONT.LABEL - 1)
ax_r.set_ylim(-0.15, 1.18)
ax_r.axhline(0.9, color='#888', ls=':', lw=0.8)
ax_r.set_title('Cross-seed GPV agreement', fontsize=FONT.LABEL - 1)

n_arch = sum(len(p[1]) for p in arch_pairs)
n_seed_total = sum(len(rhos) for _, rhos, _ in valid_models)
add_footnote(fig,
    f'GPV Spearman ρ; R²≥{R2_THR} for both models; '
    f'cross-arch n={n_arch}; cross-seed n={n_seed_total} pairs; '
    f'TC GPV via linear readout')

OUT_DIRS = [mdir, cdir, '/mnt/c/Users/amits/Desktop']
savefig_manifest(fig, 'attribution_consistency_merged.png', OUT_DIRS)
print('Saved attribution_consistency_merged.png')
