#!/usr/bin/env python3
"""
eval_e07_joint_gpv.py

Joint GPV for E07 × {cue_visible, frame_position}.

For each valid E07 session, computes:
  GPV(cue)      = R²_base − R²_{cue permuted}
  GPV(position) = R²_base − R²_{position permuted}
  GPV(joint)    = R²_base − R²_{cue + position permuted together}
  Interaction   = GPV(joint) − GPV(cue) − GPV(position)

If GPV(joint) >> GPV(cue) + GPV(position), E07 encodes cue and position
jointly (synergistic interaction — the yellow peak in the heatmap).

Uses saved MLP checkpoints — no retraining needed.
"""
import os, sys, pickle
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.models import MLP
from utils.figure_style import (
    FIG, DPI, FONT, MODEL_COLORS,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

root   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
mdir   = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [
    os.path.join(root, 'outputs', 'ablation_vs_attribution'),
    '/mnt/c/Users/amits/Desktop',
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

SEED      = 42
N_PERMS   = 10    # permutation repeats for stable estimate
E_IDX     = 6     # ensemble 07
R2_THRESH = 0.01
device    = torch.device('cpu')

# ── Load ──────────────────────────────────────────────────────────────────────
ds       = pickle.load(open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb'))
sessions = list(ds.keys())
sg       = pickle.load(open(os.path.join(mdir, 'semantic_groups.pkl'), 'rb'))
group_names = [g[0] for g in sg]
feat_idx    = {g: cols for g, cols in sg}
r2_all      = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)
tidx_map    = np.load(os.path.join(root, 'splits', f'split_seed{SEED}.npy'),
                      allow_pickle=True).item()

CUE_COLS = feat_idx['cue_visible']       # one-hot [7,8,9]
POS_COLS = feat_idx['frame_position']    # [6]
JOINT_COLS = CUE_COLS + POS_COLS

valid_sessions = [s for s in range(len(sessions)) if r2_all[s, E_IDX] >= R2_THRESH]
print(f'E07: {len(valid_sessions)} valid sessions (R²≥{R2_THRESH})')

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_mlp(s):
    path = os.path.join(root, 'models', 'mlps', 'ensembles',
                        f'seed{SEED}', f'session_{s:02d}_neuron_{E_IDX:02d}.pt')
    if not os.path.exists(path):
        return None
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        w = sd['model_state_dict']['fc.0.weight']
    else:
        w = sd['fc.0.weight']
    h, nin = w.shape
    m = MLP(nin, h, 2, 1).to(device)
    m.load_state_dict(sd['model_state_dict'] if isinstance(sd, dict) and 'model_state_dict' in sd else sd)
    m.eval()
    return m


def predict(mlp, X):
    with torch.no_grad():
        out = mlp(torch.tensor(X, dtype=torch.float32))
        # MLP returns (embedding, prediction) — take the prediction (index 1)
        pred = out[1] if isinstance(out, tuple) else out
        return pred.cpu().numpy().ravel()


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0


def gpv_permuted(mlp, X, y, cols, rng, n_perms):
    """Mean R² over n_perms with cols jointly shuffled (same row permutation)."""
    r2s = []
    for _ in range(n_perms):
        X_p = X.copy()
        idx = rng.permutation(len(X_p))
        X_p[:, cols] = X_p[idx][:, cols]
        r2s.append(r2(y, predict(mlp, X_p)))
    return float(np.mean(r2s))


# ── Per-session computation ────────────────────────────────────────────────────
rng = np.random.default_rng(0)
rows = []

for s_idx in valid_sessions:
    sess_id  = sessions[s_idx]
    sd       = ds[sess_id]
    all_t    = list(sd['data'].keys())
    test_t   = [t for t in tidx_map.get(sess_id, []) if t in sd['data']]
    if not test_t:
        print(f'  S{s_idx+1:02d}: no test trials, skipping')
        continue

    Xte = np.concatenate([sd['data'][t]              for t in test_t]).astype(float)
    yte = np.concatenate([sd['labels'][t][:, E_IDX]  for t in test_t]).astype(float)

    mlp = load_mlp(s_idx)
    if mlp is None:
        print(f'  S{s_idx+1:02d}: no checkpoint')
        continue

    r2_base  = r2(yte, predict(mlp, Xte))
    r2_cue   = gpv_permuted(mlp, Xte, yte, CUE_COLS,   rng, N_PERMS)
    r2_pos   = gpv_permuted(mlp, Xte, yte, POS_COLS,   rng, N_PERMS)
    r2_joint = gpv_permuted(mlp, Xte, yte, JOINT_COLS, rng, N_PERMS)

    gpv_cue   = r2_base - r2_cue
    gpv_pos   = r2_base - r2_pos
    gpv_j     = r2_base - r2_joint
    additive  = gpv_cue + gpv_pos
    interaction = gpv_j - additive

    rows.append(dict(
        s=s_idx, r2_base=r2_base,
        gpv_cue=gpv_cue, gpv_pos=gpv_pos,
        gpv_joint=gpv_j, additive=additive,
        interaction=interaction,
    ))
    print(f'  S{s_idx+1:02d}  R²={r2_base:.3f}  '
          f'GPV(cue)={gpv_cue:.4f}  GPV(pos)={gpv_pos:.4f}  '
          f'GPV(joint)={gpv_j:.4f}  additive={additive:.4f}  '
          f'interaction={interaction:+.4f}')

if not rows:
    print('No results'); raise SystemExit(1)

# ── Aggregate ─────────────────────────────────────────────────────────────────
mean_cue   = np.mean([r['gpv_cue']     for r in rows])
mean_pos   = np.mean([r['gpv_pos']     for r in rows])
mean_joint = np.mean([r['gpv_joint']   for r in rows])
mean_add   = np.mean([r['additive']    for r in rows])
mean_int   = np.mean([r['interaction'] for r in rows])

print(f'\nAggregated ({len(rows)} sessions):')
print(f'  GPV(cue)          = {mean_cue:.4f}')
print(f'  GPV(position)     = {mean_pos:.4f}')
print(f'  Additive baseline = {mean_add:.4f}')
print(f'  GPV(joint)        = {mean_joint:.4f}')
print(f'  Interaction       = {mean_int:+.4f}  '
      f'({"synergistic ↑" if mean_int > 0 else "sub-additive ↓"})')

# ── Figure: 2 panels ──────────────────────────────────────────────────────────
C_CUE   = '#FF7F0E'
C_POS   = '#8C564B'
C_JOINT = '#9467BD'
C_ADD   = '#AAAAAA'
C_INT   = '#D62728'

fig, (ax_agg, ax_sess) = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig, [ax_agg, ax_sess])

# Panel A — aggregate bar chart
labels_agg = ['GPV\n(cue)', 'GPV\n(pos)', 'Additive\nbaseline', 'GPV\n(joint)', 'Interaction\n(joint−add.)']
vals_agg   = [mean_cue, mean_pos, mean_add, mean_joint, mean_int]
colors_agg = [C_CUE, C_POS, C_ADD, C_JOINT, C_INT]
x = np.arange(len(labels_agg))
bars = ax_agg.bar(x, vals_agg, color=colors_agg, alpha=0.85, width=0.6)
ax_agg.axhline(0, color='#555', lw=0.8, ls='--')
ax_agg.set_xticks(x)
ax_agg.set_xticklabels(labels_agg, fontsize=FONT.TICK - 1)
ax_agg.set_ylabel('Mean GPV (ΔR²)', fontsize=FONT.LABEL - 1)
ax_agg.set_title(
    f'E07 — joint attribution (n={len(rows)} sessions)\n'
    f'Interaction {"> 0 → synergistic" if mean_int > 0 else "≈ 0 → additive"}',
    fontsize=FONT.LABEL - 1, pad=3)
add_panel_label(ax_agg, 'A')

# Panel B — per-session scatter: GPV(joint) vs additive baseline
add_vals  = np.array([r['additive']  for r in rows])
joint_vals = np.array([r['gpv_joint'] for r in rows])
s_idxs    = [r['s']                 for r in rows]
lim_max   = max(add_vals.max(), joint_vals.max()) * 1.15
lim_max   = max(lim_max, 0.01)

ax_sess.scatter(add_vals, joint_vals, c=C_JOINT, s=65, alpha=0.85, zorder=3,
                edgecolors='white', linewidths=0.4)
for ai, ji, si in zip(add_vals, joint_vals, s_idxs):
    ax_sess.annotate(f'S{si+1:02d}', (ai, ji), fontsize=5.5,
                     xytext=(3, 3), textcoords='offset points', color='#444')

# Diagonal = additive prediction
diag = np.linspace(0, lim_max, 50)
ax_sess.plot(diag, diag, '--', color='#999', lw=1.0, alpha=0.7,
             label='Additive prediction')
ax_sess.set_xlim(0, lim_max); ax_sess.set_ylim(0, lim_max)
ax_sess.set_xlabel('GPV(cue) + GPV(position)  [additive]', fontsize=FONT.LABEL - 1)
ax_sess.set_ylabel('GPV(cue, position jointly)', fontsize=FONT.LABEL - 1)
ax_sess.set_title('Points above diagonal → synergistic interaction',
                  fontsize=FONT.LABEL - 1, pad=3)
ax_sess.legend(fontsize=FONT.LEGEND - 1, frameon=False)
add_panel_label(ax_sess, 'B')

add_footnote(fig,
    f'E07 (ensemble 6), {len(rows)} sessions, seed {SEED}.  '
    f'Joint permutation: cue_visible + frame_position shuffled simultaneously with same row index.  '
    f'GPV = R²_base − R²_permuted (mean over {N_PERMS} permutations).  '
    f'Interaction = GPV(joint) − GPV(cue) − GPV(position).')

savefig_manifest(fig, 'e07_joint_gpv.png', OUT_DIRS)
print('\nSaved e07_joint_gpv.png')
