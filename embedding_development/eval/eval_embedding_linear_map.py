#!/usr/bin/env python3
"""
eval_embedding_linear_map.py

Tests geometric consistency of embeddings via linear map R².

Four comparison axes (one variable changed at a time):
  cross-seed    : same model type, session, ensemble — different random seed
  cross-session : same model type, ensemble, seed=42 — different training session
  cross-ensemble: same model type, session, seed=42 — different target ensemble
  cross-model   : same session, ensemble, seed=42 — different architecture

For cross-session, model_sj is applied to session_i's behavioral test data X_i.
Both model_si(X_i) and model_sj(X_i) are compared on the same input — the only
variable that changes is which neural population the model was trained on.

Speed: each model is loaded ONCE and applied to all needed X arrays before unloading.
"""
import os, sys, pickle, warnings, time
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.models import MLP
from utils.load_encoder import build_windows, embed_windows, load_encoder
from utils.figure_style import FIG, DPI, FONT, apply_style, add_footnote, savefig_manifest

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, '..')
mdir = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
cdir = os.path.join(root, 'outputs', 'cebra_comparison')

SEEDS       = [42, 43, 44, 45, 46]
REF_SEED    = 42
R2_THR      = 0.1
RIDGE_ALPHA = 1.0
CV_FOLDS    = 5
N_UNCUED    = 7   # sessions 0-6: uncued; sessions 7-28: cued

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t0 = time.time()
def ts():
    return f'[+{time.time()-t0:5.0f}s]'

print(f'Device: {device}', flush=True)

# ── Load metadata ──────────────────────────────────────────────────────────────
with open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb') as f:
    ds = pickle.load(f)
session_ids = list(ds.keys())
n_sess      = len(session_ids)

splits = {
    s: np.load(os.path.join(root, 'splits', f'split_seed{s}.npy'),
               allow_pickle=True).item()
    for s in SEEDS
}

mlp_r2 = np.nanmean(np.load(os.path.join(mdir, 'all_r2.npy')), axis=0)
cc_r2  = np.nanmean(np.load(os.path.join(root, 'outputs', 'cebra_64d_eval',
                                          'ensembles', 'all_r2.npy')), axis=0)
cp_r2  = np.nanmean(np.load(os.path.join(root, 'outputs', 'cebra_pred_64d_eval',
                                          'ensembles', 'all_r2.npy')), axis=0)
n_ens = mlp_r2.shape[1]

def sess_cond(s_idx):
    return 'uncued' if s_idx < N_UNCUED else 'cued'

def get_test_X(sess_id, seed):
    sd     = ds[sess_id]
    trials = sorted(t for t in splits[seed].get(sess_id, []) if t in sd['data'])
    if not trials:
        return None
    return np.concatenate([sd['data'][t] for t in trials], axis=0).astype(np.float32)


# ── Embed helpers (batch: load model once, embed multiple X arrays) ────────────

def mlp_embed_batch(seed, s_idx, e_idx, Xs):
    """Load MLP once; return list of embeddings (None where X is None)."""
    path = os.path.join(root, 'models', 'mlps', 'ensembles',
                        f'seed{seed}', f'session_{s_idx:02d}_neuron_{e_idx:02d}.pt')
    if not os.path.exists(path):
        return [None] * len(Xs)
    model = MLP(17, 64, 2, 1).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    out = []
    for X in Xs:
        if X is None:
            out.append(None)
        else:
            with torch.no_grad():
                H = model.embed(torch.tensor(X, device=device,
                                             dtype=torch.float32)).cpu().numpy()
            out.append(H)
    del model; torch.cuda.empty_cache()
    return out


def tc_embed_batch(arm, seed, s_idx, e_idx, Xs):
    """Load TC encoder once; return list of embeddings (None where X is None or NaN)."""
    path = os.path.join(root, 'models', f'{arm}_64d', 'ensembles',
                        f'seed{seed}', f'session_{s_idx:02d}_neuron_{e_idx:02d}.pt')
    if not os.path.exists(path):
        return [None] * len(Xs)
    encoder, _, _ = load_encoder(path, device=str(device))
    out = []
    for X in Xs:
        if X is None:
            out.append(None)
        else:
            wins = build_windows(X)
            H    = embed_windows(encoder, wins, device=str(device))
            out.append(None if np.isnan(H).any() else H)
    del encoder; torch.cuda.empty_cache()
    return out


def embed_batch(model_key, seed, s_idx, e_idx, Xs):
    if model_key == 'mlp':
        return mlp_embed_batch(seed, s_idx, e_idx, Xs)
    elif model_key == 'cc':
        return tc_embed_batch('cebra', seed, s_idx, e_idx, Xs)
    else:
        return tc_embed_batch('cebra_pred', seed, s_idx, e_idx, Xs)


R2_MATS = {'mlp': mlp_r2, 'cc': cc_r2, 'cp': cp_r2}
MODEL_KEYS = ['mlp', 'cc', 'cp']


def linear_map_r2(H_A, H_B):
    if H_A is None or H_B is None:
        return np.nan
    n = min(len(H_A), len(H_B))
    if n < CV_FOLDS * 2:
        return np.nan
    H_A, H_B = H_A[:n], H_B[:n]
    keep = np.std(H_B, axis=0) >= 1e-8          # drop constant target dims
    if not keep.any():
        return np.nan
    Y = H_B[:, keep]
    # Vectorized equivalent of a per-dimension 5-fold ridge R2: one multi-output
    # ridge per fold, per-dimension R2 clipped to [0,1], averaged over folds
    # then over dimensions. Mathematically identical to the per-dim loop but
    # ~D times fewer solves.
    per_dim = np.empty((CV_FOLDS, Y.shape[1]))
    for f, (tr, te) in enumerate(KFold(n_splits=CV_FOLDS).split(H_A)):
        model = Ridge(RIDGE_ALPHA).fit(H_A[tr], Y[tr])
        per_dim[f] = r2_score(Y[te], model.predict(H_A[te]),
                              multioutput='raw_values')
    return float(np.clip(per_dim, 0, 1).mean(axis=0).mean())


def linear_map_r2_sym(H_A, H_B):
    """Symmetric (bidirectional) linear-map R2: mean of the A->B and B->A ridge fits.
    Ridge R2 is directional, so we average both directions to get a
    direction-invariant consistency score."""
    r_ab = linear_map_r2(H_A, H_B)
    r_ba = linear_map_r2(H_B, H_A)
    vals = [r for r in (r_ab, r_ba) if np.isfinite(r)]
    return float(np.mean(vals)) if vals else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# Results containers — each entry is a dict with r2 + full metadata so results
# can be filtered post-hoc (e.g. "session 5, MLP cross-seed").
# ══════════════════════════════════════════════════════════════════════════════
def rec(axis, model, r2, **meta):
    return dict(axis=axis, model=model, r2=r2, **meta)

CACHE_FILE = os.path.join(mdir, 'embedding_linear_map_results.npy')

if os.path.exists(CACHE_FILE):
    print(f'\nLoading cached results from {CACHE_FILE}', flush=True)
    records = list(np.load(CACHE_FILE, allow_pickle=True))
    print(f'Loaded {len(records)} records', flush=True)
else:
    # ══════════════════════════════════════════════════════════════════════════
    # Precompute seed=42 test data for each session
    # ══════════════════════════════════════════════════════════════════════════
    X42 = [get_test_X(session_ids[s], REF_SEED) for s in range(n_sess)]
    valid_X = [X is not None and len(X) >= CV_FOLDS * 2 for X in X42]

    records = []

    # Phase A: seed=42 embeddings + cross-session
    print(f'\n{ts()} Phase A: precomputing seed=42 embeddings + cross-session...', flush=True)
    E = {mk: [{} for _ in range(n_sess)] for mk in MODEL_KEYS}

    for s_j in range(n_sess):
        print(f'  {ts()} session {s_j+1}/{n_sess}', flush=True)
        if not valid_X[s_j]:
            continue
        cond_j = sess_cond(s_j)
        for e_idx in range(n_ens):
            for mk in MODEL_KEYS:
                r2_mat = R2_MATS[mk]
                if r2_mat[s_j, e_idx] < R2_THR:
                    continue
                prior_si = [s_i for s_i in range(s_j)
                            if valid_X[s_i] and r2_mat[s_i, e_idx] >= R2_THR
                            and E[mk][s_i].get(e_idx) is not None]
                Xs = [X42[s_j]] + [X42[s_i] for s_i in prior_si]
                Hs = embed_batch(mk, REF_SEED, s_j, e_idx, Xs)
                E[mk][s_j][e_idx] = Hs[0]
                for k, s_i in enumerate(prior_si):
                    H_si_on_Xi = E[mk][s_i][e_idx]
                    H_sj_on_Xi = Hs[k + 1]
                    r = linear_map_r2_sym(H_si_on_Xi, H_sj_on_Xi)
                    if not np.isfinite(r):
                        continue
                    cond_i = sess_cond(s_i)
                    pair_tag = ('uu' if cond_i == 'uncued' and cond_j == 'uncued' else
                                'cc' if cond_i == 'cued'   and cond_j == 'cued'   else 'uc')
                    records.append(rec('cross_session', mk, r,
                                       s_i=s_i, s_j=s_j, e_idx=e_idx,
                                       cond_i=cond_i, cond_j=cond_j, cond_pair=pair_tag))

    # Phase B: cross-ensemble
    print(f'\n{ts()} Phase B: cross-ensemble...', flush=True)
    for s_idx in range(n_sess):
        if not valid_X[s_idx]:
            continue
        for e_i in range(n_ens):
            for e_j in range(e_i + 1, n_ens):
                for mk in MODEL_KEYS:
                    H_i = E[mk][s_idx].get(e_i)
                    H_j = E[mk][s_idx].get(e_j)
                    r = linear_map_r2_sym(H_i, H_j)
                    if np.isfinite(r):
                        records.append(rec('cross_ensemble', mk, r,
                                           s_idx=s_idx, e_i=e_i, e_j=e_j))

    # Phase C: cross-model
    print(f'\n{ts()} Phase C: cross-model...', flush=True)
    for s_idx in range(n_sess):
        if not valid_X[s_idx]:
            continue
        for e_idx in range(n_ens):
            H_mlp = E['mlp'][s_idx].get(e_idx)
            H_cc  = E['cc'][s_idx].get(e_idx)
            H_cp  = E['cp'][s_idx].get(e_idx)
            for pair, H_A, H_B in [('mlp_cc', H_mlp, H_cc),
                                   ('mlp_cp', H_mlp, H_cp),
                                   ('cc_cp',  H_cc,  H_cp)]:
                r = linear_map_r2_sym(H_A, H_B)
                if np.isfinite(r):
                    records.append(rec('cross_model', pair, r,
                                       s_idx=s_idx, e_idx=e_idx))

    # Phase D: cross-seed
    print(f'\n{ts()} Phase D: cross-seed...', flush=True)
    for s_idx in range(n_sess):
        print(f'  {ts()} session {s_idx+1}/{n_sess}', flush=True)
        if not valid_X[s_idx]:
            continue
        X = X42[s_idx]
        for e_idx in range(n_ens):
            for mk in MODEL_KEYS:
                if R2_MATS[mk][s_idx, e_idx] < R2_THR:
                    continue
                Hs = {REF_SEED: E[mk][s_idx].get(e_idx)}
                for seed in SEEDS:
                    if seed == REF_SEED:
                        continue
                    Hs[seed] = embed_batch(mk, seed, s_idx, e_idx, [X])[0]
                for si_idx, seed_i in enumerate(SEEDS):
                    for seed_j in SEEDS[si_idx + 1:]:
                        r = linear_map_r2_sym(Hs[seed_i], Hs[seed_j])
                        if np.isfinite(r):
                            records.append(rec('cross_seed', mk, r,
                                               s_idx=s_idx, e_idx=e_idx,
                                               seed_i=seed_i, seed_j=seed_j))

    np.save(CACHE_FILE, records, allow_pickle=True)

# ══════════════════════════════════════════════════════════════════════════════
# Summary — build from records
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd
df = pd.DataFrame(records)

print(f'\n{ts()} ── Linear map R² summary ──')
summary_groups = [
    ('Cross-seed MLP',          df[(df.axis=='cross_seed')    & (df.model=='mlp')]),
    ('Cross-seed TC-Cont',      df[(df.axis=='cross_seed')    & (df.model=='cc')]),
    ('Cross-seed TC-Pred',      df[(df.axis=='cross_seed')    & (df.model=='cp')]),
    ('Cross-session MLP (all)', df[(df.axis=='cross_session') & (df.model=='mlp')]),
    ('Cross-session TC-Cont',   df[(df.axis=='cross_session') & (df.model=='cc')]),
    ('Cross-session TC-Pred',   df[(df.axis=='cross_session') & (df.model=='cp')]),
    ('MLP uncued×uncued',       df[(df.axis=='cross_session') & (df.model=='mlp') & (df.cond_pair=='uu')]),
    ('MLP cued×cued',           df[(df.axis=='cross_session') & (df.model=='mlp') & (df.cond_pair=='cc')]),
    ('MLP uncued×cued',         df[(df.axis=='cross_session') & (df.model=='mlp') & (df.cond_pair=='uc')]),
    ('TC-Pred uncued×uncued',   df[(df.axis=='cross_session') & (df.model=='cp')  & (df.cond_pair=='uu')]),
    ('TC-Pred cued×cued',       df[(df.axis=='cross_session') & (df.model=='cp')  & (df.cond_pair=='cc')]),
    ('TC-Pred uncued×cued',     df[(df.axis=='cross_session') & (df.model=='cp')  & (df.cond_pair=='uc')]),
    ('Cross-ensemble MLP',      df[(df.axis=='cross_ensemble') & (df.model=='mlp')]),
    ('Cross-ensemble TC-Cont',  df[(df.axis=='cross_ensemble') & (df.model=='cc')]),
    ('Cross-ensemble TC-Pred',  df[(df.axis=='cross_ensemble') & (df.model=='cp')]),
    ('MLP ↔ TC-Cont',           df[(df.axis=='cross_model')   & (df.model=='mlp_cc')]),
    ('MLP ↔ TC-Pred',           df[(df.axis=='cross_model')   & (df.model=='mlp_cp')]),
    ('TC-Cont ↔ TC-Pred',       df[(df.axis=='cross_model')   & (df.model=='cc_cp')]),
]
for label, sub in summary_groups:
    if len(sub) == 0:
        print(f'  {label:30s}: n=   0')
    else:
        print(f'  {label:30s}: n={len(sub):5d}  '
              f'median={sub.r2.median():.3f}  mean={sub.r2.mean():.3f}')

# ══════════════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════════════
groups = [
    ('Cross-seed', [
        ('cross_seed_mlp',       'MLP'),
        ('cross_seed_cc',        'TC-Cont'),
        ('cross_seed_cp',        'TC-Pred'),
    ]),
    ('Cross-session', [
        ('cross_session_mlp',    'MLP'),
        ('cross_session_cc',     'TC-Cont'),
        ('cross_session_cp',     'TC-Pred'),
    ]),
    ('Cross-ensemble', [
        ('cross_ensemble_mlp',   'MLP'),
        ('cross_ensemble_cc',    'TC-Cont'),
        ('cross_ensemble_cp',    'TC-Pred'),
    ]),
    ('Cross-architecture', [
        ('cross_model_mlp_cc',   'MLP-CC'),
        ('cross_model_mlp_cp',   'MLP-CP'),
        ('cross_model_cc_cp',    'CC-CP'),
    ]),
]

MODEL_COLORS = {
    'mlp': '#4CAF50', 'cc': '#2196F3', 'cp': '#FF9800',
    'uu':  '#9C27B0', 'ccond': '#E91E63', 'uc': '#795548',
}

def _color(key):
    for suf, c in [('mlp', '#4CAF50'), ('cc', '#2196F3'), ('cp', '#FF9800')]:
        if key.endswith(suf) or f'_{suf}_' in key:
            return c
    if '_uu' in key: return '#9C27B0'
    if '_cc' in key: return '#E91E63'
    if '_uc' in key: return '#795548'
    return '#888'

def _query(axis, model, cond_pair=None):
    mask = (df.axis == axis) & (df.model == model)
    if cond_pair is not None:
        mask &= (df.cond_pair == cond_pair)
    return df[mask].r2.values

flat_keys   = [k for _, conds in groups for k, _ in conds]
flat_labels = [l for _, conds in groups for _, l in conds]
flat_colors = [_color(k) for k in flat_keys]

# Map group keys → DataFrame queries
def _key_to_data(k):
    if k.startswith('cross_seed_'):
        return _query('cross_seed', k[len('cross_seed_'):])
    if k.startswith('cross_ensemble_'):
        return _query('cross_ensemble', k[len('cross_ensemble_'):])
    if k.startswith('cross_model_'):
        return _query('cross_model', k[len('cross_model_'):])
    if k.startswith('cross_session_'):
        rest = k[len('cross_session_'):]
        for mk in ('mlp', 'cc', 'cp'):
            if rest == mk:
                return _query('cross_session', mk)
            for tag in ('uu', 'cc', 'uc'):
                if rest == f'{mk}_{tag}':
                    return _query('cross_session', mk, cond_pair=tag)
    return np.array([])

flat_data = [_key_to_data(k) for k in flat_keys]

fig, ax = plt.subplots(figsize=(9.5, 4.2))
apply_style(fig, ax)

rng = np.random.default_rng(0)
gap = 0.7
pos = []
x = 0.0
for _, conds in groups:
    for _ in conds:
        pos.append(x); x += 1.0
    x += gap

nonempty = [(i, d) for i, d in enumerate(flat_data) if len(d) > 0]
if nonempty:
    vp_idx, vp_data = zip(*nonempty)
    vp = ax.violinplot([flat_data[i] for i in vp_idx],
                       positions=[pos[i] for i in vp_idx],
                       showmedians=True, showextrema=True, widths=0.65)
    for pc, i in zip(vp['bodies'], vp_idx):
        pc.set_facecolor(flat_colors[i]); pc.set_alpha(0.4)
    for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
        vp[part].set_color('#333'); vp[part].set_linewidth(0.9)

for i, (vals, c, p) in enumerate(zip(flat_data, flat_colors, pos)):
    if len(vals) == 0:
        continue
    jitter = rng.uniform(-0.1, 0.1, size=len(vals))
    ax.scatter(p + jitter, vals, s=4, color=c, alpha=0.3, linewidths=0, zorder=3)
    med = float(np.nanmedian(vals))
    ax.text(p, med + 0.03, f'{med:.2f}', ha='center', va='bottom',
            fontsize=FONT.ANNOTATION - 1, fontweight='bold')

ax.set_xticks(pos)
ax.set_xticklabels(flat_labels, fontsize=FONT.TICK - 2, rotation=45, ha='right')
ax.set_ylabel('Linear map R²', fontsize=FONT.LABEL)
ax.axhline(0,   color='#bbb', lw=0.7, linestyle=':')
ax.axhline(0.9, color='#555', lw=0.9, linestyle='--', alpha=0.6)
ax.set_ylim(-0.2, 1.15)

x_cursor = 0.0
for _, (g_label, conds) in enumerate(groups):
    g_start = x_cursor
    g_end   = x_cursor + len(conds) - 1.0
    mid     = (g_start + g_end) / 2.0
    ax.annotate('', xy=(g_end + 0.35, 1.05), xytext=(g_start - 0.35, 1.05),
                xycoords=('data', 'axes fraction'),
                textcoords=('data', 'axes fraction'),
                arrowprops=dict(arrowstyle='-', color='#666', lw=1.0))
    ax.text(mid, 1.07, g_label, ha='center', va='bottom',
            fontsize=FONT.TICK, transform=ax.get_xaxis_transform(),
            fontweight='bold', color='#444')
    x_cursor += len(conds) + gap

add_footnote(fig,
    f'Bidirectional ridge regression (5-fold CV, mean of both directions); R²≥{R2_THR}; '
    f'cross-session/ensemble: seed 42; cross-seed: common test input')

OUT_DIRS = [mdir, cdir, '/mnt/c/Users/amits/Desktop']
savefig_manifest(fig, 'embedding_linear_map.png', OUT_DIRS)
print('\nDone. Generated embedding_linear_map.png')
