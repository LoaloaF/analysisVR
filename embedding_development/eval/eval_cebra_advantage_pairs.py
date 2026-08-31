#!/usr/bin/env python3
"""
eval_cebra_advantage_pairs.py

Why does TempConv outperform MLP on certain (session, ensemble) pairs?

Strategy:
  1. Select top pairs by R² advantage (TempConv-Cont - MLP), from shared valid pairs.
  2. For those pairs, compute the full TempConv temporal IG profile (17, 10) —
     which window position (t-5..t+4) drives the attribution for each feature.
  3. Extract per-pair lagged ρ and η² from the pre-cached arrays.
  4. For each feature group, compute the correlation between the 10-step IG
     window profile and the 10-step lagged ρ / η² profile — per pair.
  5. Compare these correlations to the random background (all valid pairs).

Key question: in pairs where TempConv wins, does its temporal IG actually track
the raw-data temporal structure better than it does on average?

Figures:
  fig_adv_A_ig_profiles.png   — IG window profiles for top pairs, per feature group
  fig_adv_B_correlation.png   — per-pair shape ρ(IG, η²) for advantage vs background
  fig_adv_C_summary.png       — which feature groups drive TempConv advantage?
"""

import os, sys, argparse, pickle
import numpy as np
import torch
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr, mannwhitneyu
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from figure_style import (
    FIG, DPI, FONT, LINE, MARKER,
    MODEL_COLORS, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from load_encoder import build_windows, load_encoder

# ─── config ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--top_n',    type=int, default=20)
parser.add_argument('--seed',     type=int, default=42)
parser.add_argument('--ig_steps', type=int, default=50)
parser.add_argument('--adv_thresh', type=float, default=0.01,
                    help='Min R² advantage to be in the "advantage" group')
args = parser.parse_args()

SEED     = args.seed
IG_STEPS = args.ig_steps
TOP_N    = args.top_n
ADV_THR  = args.adv_thresh
R2_THRESH = 0.01
WIN_OFFSETS = list(range(-5, 5))
LAGS        = list(range(-5, 5))

out_dir  = 'outputs/temporal_advantage'
desk_dir = '/mnt/c/Users/amits/Desktop/temporal_advantage'
os.makedirs(out_dir,  exist_ok=True)
os.makedirs(desk_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}  seed={SEED}  top_n={TOP_N}')

# ─── load shared data ─────────────────────────────────────────────────────────
with open('outputs/session_dataset_ensembles.pkl', 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())

with open('outputs/mlps/ensembles_multiseed/semantic_groups.pkl', 'rb') as f:
    sg = pickle.load(f)
group_names = [g[0] for g in sg]
SHORT = {
    'frame_raw_500msMedian':                   'Speed',
    'frame_raw_abs_acc_500msMedian':            'Speed acc.',
    'frame_YawPitch_abs_vel_sum_500msMedian':   'Rot. vel.',
    'frame_YawPitch_abs_acc_sum_500msMedian':   'Rot. acc.',
    'head_angle_vel':                           'Head vel.',
    'head_angle':                               'Head angle',
    'frame_position':                           'Position',
    'lick_detected':                            'Lick det.',
}
glabels = [SHORT.get(g, g) for g in group_names]
continuous_groups = [i for i, (_, cols) in enumerate(sg) if len(cols) == 1]
cont_feat_cols    = [sg[i][1][0] for i in continuous_groups]
n_cont = len(continuous_groups)

mlp_r2_all = np.load('outputs/mlps/ensembles_multiseed/all_r2.npy')
ceb_r2_all = np.load('outputs/cebra_eval/ensembles/all_r2.npy')
mlp_valid  = (~np.all(np.isnan(mlp_r2_all), axis=0)) & (np.nanmean(mlp_r2_all, axis=0) >= R2_THRESH)
ceb_valid  = (~np.all(np.isnan(ceb_r2_all), axis=0)) & (np.nanmean(ceb_r2_all, axis=0) >= R2_THRESH)
shared_valid = mlp_valid & ceb_valid

mlp_mean_r2 = np.nanmean(mlp_r2_all, axis=0)
ceb_mean_r2 = np.nanmean(ceb_r2_all, axis=0)
r2_adv      = np.where(shared_valid, ceb_mean_r2 - mlp_mean_r2, np.nan)

# ─── select pairs ─────────────────────────────────────────────────────────────
flat_adv = r2_adv.flatten()
top_flat  = np.argsort(-np.where(np.isnan(flat_adv), -np.inf, flat_adv))
adv_pairs = []
for fi in top_flat:
    s, n = divmod(int(fi), 23)
    if not np.isnan(r2_adv[s, n]):
        adv_pairs.append((s, n, float(r2_adv[s, n])))
    if len(adv_pairs) >= TOP_N:
        break

# background = shared valid pairs NOT in top advantage
adv_set = {(s, n) for s, n, _ in adv_pairs}
bg_pairs = [(s, n) for s in range(len(sessions)) for n in range(23)
            if shared_valid[s, n] and (s, n) not in adv_set
            and r2_adv[s, n] < ADV_THR]

print(f'Advantage pairs: {len(adv_pairs)}  Background pairs: {len(bg_pairs)}')
for s, n, a in adv_pairs[:5]:
    print(f'  S{s:02d} E{n:02d}  MLP={mlp_mean_r2[s,n]:.3f}  CEB={ceb_mean_r2[s,n]:.3f}  adv={a:+.3f}')

split_map = np.load(f'splits/split_seed{SEED}.npy', allow_pickle=True).item()

# ─── load cached lagged arrays ────────────────────────────────────────────────
lagged_rho  = np.load(os.path.join(out_dir, 'lagged_rho.npy'))   # (29,23,8,10)
lagged_eta2 = np.load(os.path.join(out_dir, 'lagged_eta2.npy'))  # (29,23,8,10)
print('Loaded lagged_rho and lagged_eta2')


# ─── temporal IG helper ───────────────────────────────────────────────────────
class EncoderRidge(torch.nn.Module):
    def __init__(self, encoder, coef, bias):
        super().__init__()
        self.encoder = encoder
        self.register_buffer('coef', torch.tensor(coef, dtype=torch.float32))
        self.register_buffer('bias', torch.tensor(float(bias), dtype=torch.float32))
    def forward(self, x):
        z = self.encoder(x)
        if z.dim() == 3:
            z = z.squeeze(-1)
        return z @ self.coef + self.bias


def compute_temporal_ig(s_idx, n_idx, arm='cebra'):
    """Returns (17, 10) mean |IG| per feature per window position, or None."""
    sess = sessions[s_idx]
    test_trials  = [int(i) for i in split_map[sess]]
    all_trials   = list(ds[sess]['data'].keys())
    train_trials = [t for t in all_trials if t not in test_trials]

    def get(trials):
        vt = [t for t in trials if t in ds[sess]['data']]
        if not vt:
            return np.zeros((0, 17), np.float32), np.zeros((0, 23), np.float32)
        X = np.concatenate([ds[sess]['data'][t]   for t in vt]).astype(np.float32)
        Y = np.concatenate([ds[sess]['labels'][t] for t in vt]).astype(np.float32)
        return X, Y

    Xtr, Ytr = get(train_trials)
    Xte, _   = get(test_trials)
    if len(Xtr) == 0 or len(Xte) == 0:
        return None

    mpath = f'models/{arm}/ensembles/seed{SEED}/session_{s_idx:02d}_neuron_{n_idx:02d}.pt'
    if not os.path.exists(mpath):
        return None

    encoder, _, _ = load_encoder(mpath, device=device)
    encoder.eval()
    wins_tr = build_windows(Xtr).astype(np.float32)
    wins_te = build_windows(Xte).astype(np.float32)

    with torch.no_grad():
        wt = torch.tensor(wins_tr, dtype=torch.float32, device=device)
        emb_chunks = []
        for b in range(0, len(wt), 512):
            z = encoder(wt[b:b+512])
            if z.dim() == 3: z = z.squeeze(-1)
            emb_chunks.append(z.cpu().numpy())
        emb_tr = np.concatenate(emb_chunks)

    ridge = Ridge(alpha=1.0)
    ridge.fit(emb_tr, Ytr[:, n_idx])

    model_ig = EncoderRidge(encoder, ridge.coef_, ridge.intercept_).to(device).eval()
    test_t   = torch.tensor(wins_te, dtype=torch.float32, device=device)
    alphas   = np.linspace(0.0, 1.0, IG_STEPS + 1)[1:]
    grads_sum = np.zeros_like(wins_te, dtype=np.float64)

    for alpha in alphas:
        x_int = (float(alpha) * test_t).detach().requires_grad_(True)
        with torch.enable_grad():
            model_ig(x_int).sum().backward()
        grads_sum += x_int.grad.detach().cpu().numpy()
        del x_int

    ig_full = wins_te * (grads_sum / IG_STEPS)  # (T, 17, 10)
    profile = np.mean(np.abs(ig_full), axis=0)   # (17, 10)

    del encoder, model_ig, emb_tr
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return profile


# ─── compute / load temporal IG for advantage pairs ──────────────────────────
ig_cache_path = os.path.join(out_dir, 'adv_ig_profiles.npy')
ig_cache = np.load(ig_cache_path, allow_pickle=True).item() if os.path.exists(ig_cache_path) else {}

for s_idx, n_idx, adv in adv_pairs:
    key = (s_idx, n_idx)
    if key not in ig_cache:
        print(f'  Computing IG: S{s_idx:02d} E{n_idx:02d}  adv={adv:+.3f}')
        prof = compute_temporal_ig(s_idx, n_idx)
        if prof is not None:
            ig_cache[key] = prof
    else:
        print(f'  Cached:      S{s_idx:02d} E{n_idx:02d}')

np.save(ig_cache_path, ig_cache)
print(f'Computed/loaded IG for {len(ig_cache)} advantage pairs')

# ─── also compute temporal IG for a sample of background pairs ───────────────
bg_cache_path = os.path.join(out_dir, 'bg_ig_profiles.npy')
bg_cache = np.load(bg_cache_path, allow_pickle=True).item() if os.path.exists(bg_cache_path) else {}

# sample up to TOP_N background pairs
rng = np.random.default_rng(0)
bg_sample = [bg_pairs[i] for i in rng.choice(len(bg_pairs), min(TOP_N, len(bg_pairs)), replace=False)]
for s_idx, n_idx in bg_sample:
    key = (s_idx, n_idx)
    if key not in bg_cache:
        print(f'  BG IG: S{s_idx:02d} E{n_idx:02d}')
        prof = compute_temporal_ig(s_idx, n_idx)
        if prof is not None:
            bg_cache[key] = prof

np.save(bg_cache_path, bg_cache)
print(f'Background IG computed/loaded for {len(bg_cache)} pairs')


# ══════════════════════════════════════════════════════════════════════════════
# Figure A — Mean TempConv IG window profiles for advantage vs background pairs
# Per continuous feature group
# ══════════════════════════════════════════════════════════════════════════════
ncols = 4
nrows = (n_cont + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows), squeeze=False)
apply_style(fig, axes.flatten())
win_x = np.array(WIN_OFFSETS)

for fi, g_idx in enumerate(continuous_groups):
    ax       = axes[fi // ncols][fi % ncols]
    feat_col = sg[g_idx][1][0]
    lab      = glabels[g_idx]

    # advantage pairs
    adv_profs = [ig_cache[(s, n)][feat_col, :] for s, n, _ in adv_pairs if (s, n) in ig_cache]
    # background pairs
    bg_profs  = [bg_cache[(s, n)][feat_col, :] for s, n in bg_sample if (s, n) in bg_cache]

    for profs, color, label in [
        (adv_profs, '#E65100', f'Adv. (n={len(adv_profs)})'),
        (bg_profs,  '#1565C0', f'BG   (n={len(bg_profs)})'),
    ]:
        if not profs:
            continue
        arr = np.array(profs)
        mn  = arr.mean(axis=0)
        sem = arr.std(axis=0) / np.sqrt(len(arr))
        mn_n  = mn  / (mn.max() + 1e-12)
        sem_n = sem / (mn.max() + 1e-12)
        ax.plot(win_x, mn_n, color=color, linewidth=2.5, label=label)
        ax.fill_between(win_x, mn_n - sem_n, mn_n + sem_n, color=color, alpha=0.2)

    ax.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.set_title(lab, fontsize=10)
    ax.set_xlabel('Window offset', fontsize=8)
    ax.set_ylabel('Norm. |IG|', fontsize=8)
    ax.legend(fontsize=7)
    ax.set_xticks(win_x)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, 1.2)

for fi in range(n_cont, nrows * ncols):
    axes[fi // ncols][fi % ncols].set_visible(False)

plt.suptitle('TempConv temporal IG window profiles\n'
             'Orange = top advantage pairs  |  Blue = background pairs  |  Normalised',
             fontsize=11)
savefig_manifest(fig, 'fig_adv_A_ig_profiles.png', [out_dir, desk_dir])
print('Saved fig_adv_A_ig_profiles.png')


# ══════════════════════════════════════════════════════════════════════════════
# Figure B — Per-pair shape ρ(IG window, lagged η²) for advantage vs background
# Each dot = one pair × one feature group
# ══════════════════════════════════════════════════════════════════════════════
def per_pair_shape_rho(pairs, cache, use_eta=True):
    """Returns array (n_pairs, n_cont) of shape-ρ between IG window and lagged curve."""
    arr_lag = lagged_eta2 if use_eta else np.abs(lagged_rho)
    out = np.full((len(pairs), n_cont), np.nan)
    for pi, pair_info in enumerate(pairs):
        s_idx, n_idx = pair_info[0], pair_info[1]
        if (s_idx, n_idx) not in cache:
            continue
        prof = cache[(s_idx, n_idx)]   # (17, 10)
        for fi, g_idx in enumerate(continuous_groups):
            feat_col = sg[g_idx][1][0]
            ig_win   = prof[feat_col, :]            # (10,) IG over window positions
            lag_prof = arr_lag[s_idx, n_idx, fi, :] # (10,) lagged stat over LAGS
            ok = ~np.isnan(lag_prof)
            if ok.sum() < 5:
                continue
            rho, _ = spearmanr(ig_win, lag_prof)
            out[pi, fi] = float(rho)
    return out

adv_shape = per_pair_shape_rho(adv_pairs, ig_cache, use_eta=True)   # (n_adv, n_cont)
bg_shape  = per_pair_shape_rho([(s, n, 0) for s, n in bg_sample], bg_cache, use_eta=True)

fig, axes = plt.subplots(2, n_cont // 2 + n_cont % 2,
                         figsize=(16, 6), squeeze=False)
apply_style(fig, axes.flatten())
axes = axes.flatten()

print('\nPer-feature shape ρ(IG, η²): advantage vs background (Mann-Whitney U):')
for fi, g_idx in enumerate(continuous_groups):
    ax  = axes[fi]
    lab = glabels[g_idx]
    adv_vals = adv_shape[:, fi][~np.isnan(adv_shape[:, fi])]
    bg_vals  = bg_shape[:,  fi][~np.isnan(bg_shape[:,  fi])]

    # violin / strip
    parts = ax.violinplot([adv_vals, bg_vals], positions=[0, 1],
                          showmedians=True, showextrema=False)
    parts['bodies'][0].set_facecolor('#E65100')
    parts['bodies'][1].set_facecolor('#1565C0')
    for b in parts['bodies']:
        b.set_alpha(0.6)
    ax.scatter(np.zeros(len(adv_vals)) + np.random.default_rng(fi).uniform(-0.07,0.07,len(adv_vals)),
               adv_vals, s=20, color='#E65100', alpha=0.7, zorder=3)
    ax.scatter(np.ones(len(bg_vals))  + np.random.default_rng(fi+100).uniform(-0.07,0.07,len(bg_vals)),
               bg_vals,  s=20, color='#1565C0', alpha=0.7, zorder=3)

    if len(adv_vals) > 2 and len(bg_vals) > 2:
        _, p = mannwhitneyu(adv_vals, bg_vals, alternative='greater')
        p_str = f'p={p:.3f}'
        ax.set_title(f'{lab}\n{p_str}', fontsize=9,
                     color='green' if p < 0.05 else 'black')
        print(f'  {lab:<14}: adv med={np.median(adv_vals):+.3f}  bg med={np.median(bg_vals):+.3f}  {p_str}')
    else:
        ax.set_title(lab, fontsize=9)

    ax.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Adv.', 'BG'], fontsize=8)
    ax.set_ylabel('shape ρ(IG, η²)', fontsize=7)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(alpha=0.3)

for fi in range(n_cont, len(axes)):
    axes[fi].set_visible(False)

plt.suptitle('Per-pair shape ρ(TempConv IG window, lagged η²)\n'
             'Orange = TempConv advantage pairs  |  Blue = background pairs\n'
             'Green title = advantage pairs have significantly higher correlation (p<0.05)',
             fontsize=10)
savefig_manifest(fig, 'fig_adv_B_correlation.png', [out_dir, desk_dir])
print('Saved fig_adv_B_correlation.png')


# ══════════════════════════════════════════════════════════════════════════════
# Figure C — What actually differs between advantage and background pairs?
# For each feature group: compare mean lagged η² at each lag offset
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows), squeeze=False)
apply_style(fig, axes.flatten())
lag_x = np.array(LAGS)

print('\nLagged η² profiles — advantage vs background pairs:')
for fi, g_idx in enumerate(continuous_groups):
    ax  = axes[fi // ncols][fi % ncols]
    lab = glabels[g_idx]

    adv_eta = np.array([lagged_eta2[s, n, fi, :] for s, n, _ in adv_pairs
                        if not np.all(np.isnan(lagged_eta2[s, n, fi, :]))])
    bg_eta  = np.array([lagged_eta2[s, n, fi, :] for s, n in bg_sample
                        if not np.all(np.isnan(lagged_eta2[s, n, fi, :]))])

    for arr, color, label in [(adv_eta, '#E65100', 'Adv.'), (bg_eta, '#1565C0', 'BG')]:
        if len(arr) == 0:
            continue
        mn  = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0) / np.sqrt(len(arr))
        ax.plot(lag_x, mn, color=color, linewidth=2.5, label=f'{label} (n={len(arr)})')
        ax.fill_between(lag_x, mn - sem, mn + sem, color=color, alpha=0.2)

    # Test at each lag: are adv pairs higher?
    sig_lags = []
    for li, lag in enumerate(LAGS):
        a = adv_eta[:, li][~np.isnan(adv_eta[:, li])] if len(adv_eta) > 0 else []
        b = bg_eta[:,  li][~np.isnan(bg_eta[:,  li])]  if len(bg_eta)  > 0 else []
        if len(a) > 2 and len(b) > 2:
            _, p = mannwhitneyu(a, b, alternative='greater')
            if p < 0.05:
                sig_lags.append(lag)
                ax.axvline(lag, color='#E65100', linewidth=1, linestyle=':', alpha=0.6)

    ax.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.4)
    sig_str = f'sig lags: {sig_lags}' if sig_lags else ''
    ax.set_title(f'{lab}\n{sig_str}', fontsize=8, color='green' if sig_lags else 'black')
    ax.set_xlabel('Lag (feature offset)', fontsize=8)
    ax.set_ylabel('Mean η²', fontsize=8)
    ax.legend(fontsize=7)
    ax.set_xticks(lag_x)
    ax.grid(alpha=0.3)

for fi in range(n_cont, nrows * ncols):
    axes[fi // ncols][fi % ncols].set_visible(False)

plt.suptitle('Lagged η² profiles — do advantage pairs have stronger temporal signal?\n'
             'Orange dotted line = lag where advantage pairs significantly higher (p<0.05)',
             fontsize=11)
savefig_manifest(fig, 'fig_adv_C_lag_eta2.png', [out_dir, desk_dir])
print('Saved fig_adv_C_lag_eta2.png')


# ─── text summary ─────────────────────────────────────────────────────────────
print('\n=== Summary: What drives TempConv advantage? ===')
print(f'Top {TOP_N} pairs by R² advantage (TempConv-Cont - MLP):')
for s, n, a in adv_pairs:
    print(f'  S{s:02d} E{n:02d}  MLP={mlp_mean_r2[s,n]:.3f}  CEB={ceb_mean_r2[s,n]:.3f}  '
          f'adv={a:+.3f}  IG_computed={"yes" if (s,n) in ig_cache else "no"}')

print('\nMean IG window peak offset per feature group (advantage pairs):')
for fi, g_idx in enumerate(continuous_groups):
    feat_col = sg[g_idx][1][0]
    profs = [ig_cache[(s,n)][feat_col,:] for s,n,_ in adv_pairs if (s,n) in ig_cache]
    if profs:
        mn = np.array(profs).mean(axis=0)
        pk = WIN_OFFSETS[int(np.argmax(mn))]
        # is the profile meaningfully non-flat?
        spread = mn.max() - mn.min()
        print(f'  {glabels[g_idx]:<14}: peak t{pk:+d}  spread={spread:.4f}')
print('\nDone.')
