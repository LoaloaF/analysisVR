#!/usr/bin/env python3
"""
eval_temporal_advantage.py

Story: TempConv's 10-step temporal window lets it capture lagged behavioral → neural
dependencies that MLP (single timestep) structurally cannot access.

Two independent analyses:
  1. Lagged Spearman ρ (no ML): ρ(feature_{t+offset}, neural_t) for offsets -5..+4.
     Computed within-trial to avoid cross-trial contamination.
     This is the ground-truth verification that temporal deps exist in the raw data.

  2. TempConv temporal IG: for the top pairs where TempConv-Cont IG >> MLP GPV, recompute
     the full (T, 17, 10) IG and extract the window-position profile.
     Shows TempConv's encoder specifically exploits the lags verified by analysis 1.

Key expected finding: for rotational acceleration (group 3) and head velocity (group 4),
lagged ρ peaks at t-1 or t-2 — TempConv can use this; MLP cannot.

Figures:
  fig_A_lagged_rho.png     — lagged ρ profiles per feature group (mean ± SEM over pairs)
  fig_B_tempconv_ig_window.png — TempConv temporal IG window profiles for top pairs
  fig_C_r2_vs_lag.png      — R² advantage (TempConv-Cont - MLP) vs lag-effect strength
  fig_D_feature_comparison.png — combined summary figure for supervisor

Usage:
    cd embedding_development
    python eval/eval_temporal_advantage.py [--n_pairs 20] [--seed 42]
"""

import os, sys, argparse, pickle
import numpy as np
import torch
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr, friedmanchisquare, wilcoxon
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from load_encoder import build_windows, load_encoder
from figure_style import (
    FIG, DPI, FONT, LINE, MARKER,
    MODEL_COLORS, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

# ─── args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--n_pairs',  type=int, default=15,
                    help='Max pairs per continuous group for temporal IG computation')
parser.add_argument('--seed',     type=int, default=42)
parser.add_argument('--ig_steps', type=int, default=50)
args = parser.parse_args()

SEED     = args.seed
IG_STEPS = args.ig_steps
N_PAIRS  = args.n_pairs
R2_THRESH = 0.01
WIN_LEN   = 10
WIN_OFFSETS = list(range(-5, 5))   # k=0 → t-5, k=5 → t, k=9 → t+4

out_dir  = 'outputs/temporal_advantage'
desk_dir = '/mnt/c/Users/amits/Desktop/temporal_advantage'
os.makedirs(out_dir,  exist_ok=True)
os.makedirs(desk_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}  seed={SEED}  n_pairs={N_PAIRS}  ig_steps={IG_STEPS}')

# ─── load shared data ─────────────────────────────────────────────────────────
with open('outputs/session_dataset_ensembles.pkl', 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())

with open('outputs/mlps/ensembles_multiseed/semantic_groups.pkl', 'rb') as f:
    sg = pickle.load(f)
group_names = [g[0] for g in sg]
n_groups    = len(group_names)

SHORT = {
    'frame_raw_500msMedian':                   'Speed',
    'frame_raw_abs_acc_500msMedian':            'Speed acc.',
    'frame_YawPitch_abs_vel_sum_500msMedian':   'Rot. vel.',
    'frame_YawPitch_abs_acc_sum_500msMedian':   'Rot. acc.',
    'head_angle_vel':                           'Head vel.',
    'head_angle':                               'Head angle',
    'frame_position':                           'Position',
    'cue_visible':                              'Cue vis.',
    'upcoming_choice':                          'Up. choice',
    'reward_window':                            'Reward win.',
    'lick_detected':                            'Lick det.',
}
glabels = [SHORT.get(g, g) for g in group_names]

# continuous group indices (single-column features, no one-hot)
continuous_groups = [i for i, (_, cols) in enumerate(sg) if len(cols) == 1]
cont_feat_cols    = [sg[i][1][0] for i in continuous_groups]  # feature column indices

mlp_r2_all  = np.load('outputs/mlps/ensembles_multiseed/all_r2.npy')
ceb_r2_all  = np.load('outputs/cebra_eval/ensembles/all_r2.npy')
mlp_valid   = (~np.all(np.isnan(mlp_r2_all), axis=0)) & (np.nanmean(mlp_r2_all, axis=0) >= R2_THRESH)
ceb_valid   = (~np.all(np.isnan(ceb_r2_all), axis=0)) & (np.nanmean(ceb_r2_all, axis=0) >= R2_THRESH)
shared_valid = mlp_valid & ceb_valid   # (29, 23)

ceb_ig_arr  = np.load('outputs/cebra_eval/ensembles/importance_ig_semantic.npy')   # (29, 23, 11)
mlp_gpv_arr = np.load('outputs/mlps/ensembles_multiseed/importance_global_pv_semantic.npy')

mlp_mean_r2 = np.nanmean(mlp_r2_all, axis=0)   # (29, 23)
ceb_mean_r2 = np.nanmean(ceb_r2_all, axis=0)
r2_advantage = ceb_mean_r2 - mlp_mean_r2        # positive = TempConv better

split_map = np.load(f'splits/split_seed{SEED}.npy', allow_pickle=True).item()


def get_arrays(sess, trial_ids):
    sd = ds[sess]
    vt = [t for t in trial_ids if t in sd['data']]
    if not vt:
        return np.zeros((0, 17), np.float32), np.zeros((0, 23), np.float32)
    X = np.concatenate([sd['data'][t]   for t in vt], axis=0).astype(np.float32)
    Y = np.concatenate([sd['labels'][t] for t in vt], axis=0).astype(np.float32)
    return X, Y


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — Lagged Spearman ρ AND η² from raw data
# η² = correlation ratio = SS_between / SS_total from decile binning of feature
# captures nonlinear tuning that ρ misses (U-shaped, threshold, etc.)
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Analysis 1: Lagged Spearman ρ and η² ===')

LAGS   = list(range(-5, 5))
N_BINS = 10   # decile bins for η²


def _slice(x, y, lag):
    """Return aligned (x_lag, y) slices for ρ(feature_{t+lag}, neural_t)."""
    if lag > 0:
        return x[lag:], y[:-lag]
    elif lag < 0:
        return x[:lag], y[-lag:]
    return x, y


def lagged_rho_within_trial(x_trial, y_trial, lag):
    x_sl, y_sl = _slice(x_trial, y_trial, lag)
    if len(x_sl) < 5:
        return np.nan
    rho, _ = spearmanr(x_sl, y_sl)
    return float(rho)


def lagged_eta2_within_trial(x_trial, y_trial, lag):
    """η²(feature_{t+lag}, neural_t) via decile binning within a single trial."""
    x_sl, y_sl = _slice(x_trial, y_trial, lag)
    if len(x_sl) < N_BINS * 3:
        return np.nan
    try:
        # decile bin edges on x
        edges = np.unique(np.percentile(x_sl, np.linspace(0, 100, N_BINS + 1)))
        if len(edges) < 3:
            return np.nan
        bins = np.digitize(x_sl, edges[1:-1])   # integer 0..n_bins-1
        y_mean = y_sl.mean()
        ss_tot = float(np.sum((y_sl - y_mean) ** 2))
        if ss_tot < 1e-12:
            return np.nan
        ss_within = sum(
            float(np.sum((y_sl[bins == b] - y_sl[bins == b].mean()) ** 2))
            for b in np.unique(bins) if (bins == b).sum() > 1
        )
        return float(np.clip(1.0 - ss_within / ss_tot, 0.0, 1.0))
    except Exception:
        return np.nan


# lagged_rho[s_idx, n_idx, feat_col, lag_idx]
lagged_rho_path = os.path.join(out_dir, 'lagged_rho.npy')

if os.path.exists(lagged_rho_path):
    lagged_rho = np.load(lagged_rho_path)
    print(f'  Loaded cached lagged_rho.npy  shape={lagged_rho.shape}')
else:
    lagged_rho = np.full((len(sessions), 23, len(cont_feat_cols), len(LAGS)), np.nan)

    for s_idx, sess in enumerate(sessions):
        sd = ds[sess]
        all_trials = list(sd['data'].keys())
        for n_idx in range(23):
            if not mlp_valid[s_idx, n_idx]:
                continue
            for fi, feat_col in enumerate(cont_feat_cols):
                rhos_by_lag = [[] for _ in range(len(LAGS))]
                for t in all_trials:
                    x_t = sd['data'][t][:, feat_col].astype(np.float32)
                    y_t = sd['labels'][t][:, n_idx].astype(np.float32)
                    for li, lag in enumerate(LAGS):
                        r = lagged_rho_within_trial(x_t, y_t, lag)
                        if not np.isnan(r):
                            rhos_by_lag[li].append(r)
                for li in range(len(LAGS)):
                    if rhos_by_lag[li]:
                        lagged_rho[s_idx, n_idx, fi, li] = np.mean(rhos_by_lag[li])

        print(f'  S{s_idx:02d} done')

    np.save(lagged_rho_path, lagged_rho)
    print(f'  Saved lagged_rho.npy  shape={lagged_rho.shape}')

# ── η² (same structure as lagged_rho) ────────────────────────────────────────
lagged_eta2_path = os.path.join(out_dir, 'lagged_eta2.npy')

if os.path.exists(lagged_eta2_path):
    lagged_eta2 = np.load(lagged_eta2_path)
    print(f'  Loaded cached lagged_eta2.npy  shape={lagged_eta2.shape}')
else:
    lagged_eta2 = np.full((len(sessions), 23, len(cont_feat_cols), len(LAGS)), np.nan)
    for s_idx, sess in enumerate(sessions):
        sd = ds[sess]
        all_trials = list(sd['data'].keys())
        for n_idx in range(23):
            if not mlp_valid[s_idx, n_idx]:
                continue
            for fi, feat_col in enumerate(cont_feat_cols):
                eta_by_lag = [[] for _ in range(len(LAGS))]
                for t in all_trials:
                    x_t = sd['data'][t][:, feat_col].astype(np.float32)
                    y_t = sd['labels'][t][:, n_idx].astype(np.float32)
                    for li, lag in enumerate(LAGS):
                        e = lagged_eta2_within_trial(x_t, y_t, lag)
                        if not np.isnan(e):
                            eta_by_lag[li].append(e)
                for li in range(len(LAGS)):
                    if eta_by_lag[li]:
                        lagged_eta2[s_idx, n_idx, fi, li] = np.mean(eta_by_lag[li])
        print(f'  η² S{s_idx:02d} done')
    np.save(lagged_eta2_path, lagged_eta2)
    print(f'  Saved lagged_eta2.npy  shape={lagged_eta2.shape}')

# Peak lag for each (session, ensemble, continuous feature group)
abs_lagged = np.abs(lagged_rho)  # (29, 23, n_cont_feats, n_lags)

# Safe nanargmax — returns 0 (lag=LAGS[0]) where all NaN
_no_nan = np.where(np.isnan(abs_lagged), -np.inf, abs_lagged)
lag_argmax = np.argmax(_no_nan, axis=3)  # (29, 23, n_cont_feats)
# mask out all-NaN slices
lag_argmax = np.where(np.all(np.isnan(abs_lagged), axis=3), -1, lag_argmax)
peak_lag   = np.where(lag_argmax >= 0,
                      np.array(LAGS)[np.clip(lag_argmax, 0, len(LAGS)-1)],
                      np.nan).astype(float)

# Mean |ρ| profile over valid (session, ensemble) pairs, per feature group
print('\n  Mean |ρ| profiles per continuous feature group:')
for fi, (g_idx, feat_col) in enumerate(zip(continuous_groups, cont_feat_cols)):
    # valid pairs
    valid_mask_2d = mlp_valid  # (29, 23)
    vals = abs_lagged[:, :, fi, :]  # (29, 23, n_lags)
    vals_valid = vals[valid_mask_2d]  # (n_valid, n_lags)
    means = np.nanmean(vals_valid, axis=0)
    pk_off = LAGS[int(np.nanargmax(means))]
    print(f'  {glabels[g_idx]:15s}: peak lag t{pk_off:+d}  '
          f'profile={np.array2string(means, precision=3, floatmode="fixed")}')


# ── Figure A: lagged ρ profiles ───────────────────────────────────────────────
n_cont = len(continuous_groups)
ncols  = 4
nrows  = (n_cont + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows), squeeze=False)
apply_style(fig, axes.flatten())

lag_x = np.array(LAGS)
colors_lag = plt.cm.plasma(np.linspace(0.1, 0.85, n_cont))

for fi, (g_idx, feat_col) in enumerate(zip(continuous_groups, cont_feat_cols)):
    ax = axes[fi // ncols][fi % ncols]
    vals = abs_lagged[:, :, fi, :]   # (29, 23, n_lags)
    vals_v = vals[mlp_valid]
    mn  = np.nanmean(vals_v, axis=0)
    sem = np.nanstd(vals_v, axis=0) / np.sqrt(np.sum(~np.isnan(vals_v[:, 0])))
    ax.plot(lag_x, mn, color=colors_lag[fi], linewidth=2)
    ax.fill_between(lag_x, mn - sem, mn + sem, color=colors_lag[fi], alpha=0.25)
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5, label='t=0 (now)')
    pk_off = LAGS[int(np.nanargmax(mn))]
    ax.axvline(pk_off, color=colors_lag[fi], linewidth=1.2, linestyle=':', alpha=0.8,
               label=f'peak t{pk_off:+d}')
    ax.set_title(glabels[g_idx], fontsize=10)
    ax.set_xlabel('Temporal offset (steps)', fontsize=8)
    ax.set_ylabel('Mean |Spearman ρ|', fontsize=8)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xticks(lag_x)
    ax.grid(alpha=0.3)

# hide unused
for fi in range(n_cont, nrows * ncols):
    axes[fi // ncols][fi % ncols].set_visible(False)

plt.suptitle('Lagged Spearman |ρ| — feature vs neural activity\n'
             'Verification of temporal dependencies without ML',
             fontsize=12)
savefig_manifest(fig, 'fig_A_lagged_rho.png', [out_dir, desk_dir])
print('Saved fig_A_lagged_rho.png')


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — TempConv temporal IG window profiles for top pairs
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Analysis 2: TempConv temporal IG window profiles ===')

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


def compute_temporal_ig(s_idx, n_idx, arm='cebra', seed=SEED):
    """Returns ig_profile: (17, 10) mean |IG| per feature per window position."""
    sess = sessions[s_idx]
    test_trials  = [int(i) for i in split_map[sess]]
    all_trials   = list(ds[sess]['data'].keys())
    train_trials = [t for t in all_trials if t not in test_trials]

    Xtr, Ytr = get_arrays(sess, train_trials)
    Xte, _   = get_arrays(sess, test_trials)
    if len(Xtr) == 0 or len(Xte) == 0:
        return None

    wins_tr = build_windows(Xtr).astype(np.float32)
    wins_te = build_windows(Xte).astype(np.float32)

    mpath = f'models/{arm}/ensembles/seed{seed}/session_{s_idx:02d}_neuron_{n_idx:02d}.pt'
    if not os.path.exists(mpath):
        return None

    encoder, _, _ = load_encoder(mpath, device=device)
    encoder.eval()

    with torch.no_grad():
        wt = torch.tensor(wins_tr, dtype=torch.float32, device=device)
        emb_chunks = []
        for b in range(0, len(wt), 512):
            z = encoder(wt[b:b + 512])
            if z.dim() == 3:
                z = z.squeeze(-1)
            emb_chunks.append(z.cpu().numpy())
        emb_tr = np.concatenate(emb_chunks, axis=0)

    ridge = Ridge(alpha=1.0)
    ridge.fit(emb_tr, Ytr[:, n_idx])

    model_ig = EncoderRidge(encoder, ridge.coef_, ridge.intercept_).to(device).eval()

    test_t = torch.tensor(wins_te, dtype=torch.float32, device=device)
    alphas = np.linspace(0.0, 1.0, IG_STEPS + 1)[1:]
    grads_sum = np.zeros_like(wins_te, dtype=np.float64)  # (T, 17, 10)

    for alpha in alphas:
        x_int = (float(alpha) * test_t).detach().requires_grad_(True)
        with torch.enable_grad():
            model_ig(x_int).sum().backward()
        grads_sum += x_int.grad.detach().cpu().numpy()
        del x_int

    ig_full  = wins_te * (grads_sum / IG_STEPS)  # (T, 17, 10)
    ig_abs   = np.abs(ig_full)                    # (T, 17, 10)
    ig_profile = np.mean(ig_abs, axis=0)          # (17, 10) mean per feat per window pos

    del encoder, model_ig, emb_tr
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return ig_profile


# Select top pairs for each continuous feature group by CEB IG - MLP GPV
IG_CACHE = os.path.join(out_dir, 'temporal_ig_profiles.npy')
IG_META  = os.path.join(out_dir, 'temporal_ig_meta.npy')

if os.path.exists(IG_CACHE):
    ig_profiles = np.load(IG_CACHE, allow_pickle=True).item()
    meta_list   = np.load(IG_META,  allow_pickle=True).tolist()
    print(f'  Loaded {len(ig_profiles)} cached temporal IG profiles')
else:
    ig_profiles = {}   # key: (s_idx, n_idx) → (17, 10) profile
    meta_list   = []   # list of (s_idx, n_idx, g_idx)

    for g_idx in continuous_groups:
        # score = TempConv IG for this group, among shared valid pairs
        score = np.where(shared_valid, ceb_ig_arr[:, :, g_idx], np.nan)
        # rank all valid pairs
        flat_scores = score.flatten()
        top_flat = np.argsort(-np.where(np.isnan(flat_scores), -np.inf, flat_scores))
        count = 0
        for flat_idx in top_flat:
            if count >= N_PAIRS:
                break
            s_idx, n_idx = divmod(int(flat_idx), 23)
            if np.isnan(score[s_idx, n_idx]):
                continue
            key = (s_idx, n_idx)
            if key not in ig_profiles:
                print(f'    Computing temporal IG: S{s_idx:02d} E{n_idx:02d}  g={glabels[g_idx]}')
                prof = compute_temporal_ig(s_idx, n_idx)
                if prof is not None:
                    ig_profiles[key] = prof
                    meta_list.append((s_idx, n_idx, g_idx))
            elif (s_idx, n_idx, g_idx) not in [(m[0], m[1], m[2]) for m in meta_list]:
                meta_list.append((s_idx, n_idx, g_idx))
            count += 1

    np.save(IG_CACHE, ig_profiles)
    np.save(IG_META,  meta_list)
    print(f'  Computed {len(ig_profiles)} temporal IG profiles')

# ── Figure B: TempConv temporal IG window profiles ───────────────────────────────
# For each continuous group, show the mean window profile
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows), squeeze=False)
apply_style(fig, axes.flatten())
win_x = np.array(WIN_OFFSETS)

colors_cg = plt.cm.viridis(np.linspace(0.1, 0.9, n_cont))

for fi, g_idx in enumerate(continuous_groups):
    ax = axes[fi // ncols][fi % ncols]
    feat_col = sg[g_idx][1][0]

    # gather profiles for pairs tagged with this group
    profs_this_g = [ig_profiles[(s, n)][feat_col, :] for (s, n, gi) in meta_list if gi == g_idx and (s, n) in ig_profiles]

    if not profs_this_g:
        ax.set_visible(False)
        continue

    profs = np.array(profs_this_g)    # (n_pairs, 10)
    mn  = profs.mean(axis=0)
    sem = profs.std(axis=0) / np.sqrt(len(profs))

    # normalise for display
    mn_norm  = mn  / (mn.sum() + 1e-12)
    sem_norm = sem / (mn.sum() + 1e-12)

    ax.plot(win_x, mn_norm, color=colors_cg[fi], linewidth=2.5)
    ax.fill_between(win_x, mn_norm - sem_norm, mn_norm + sem_norm,
                    color=colors_cg[fi], alpha=0.25)
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    pk_off = WIN_OFFSETS[int(np.argmax(mn_norm))]
    ax.axvline(pk_off, color=colors_cg[fi], linewidth=1.5, linestyle=':',
               label=f'peak t{pk_off:+d}')
    ax.set_title(f'{glabels[g_idx]}  (n={len(profs)} pairs)', fontsize=10)
    ax.set_xlabel('Window offset (steps)', fontsize=8)
    ax.set_ylabel('Norm. mean |IG|', fontsize=8)
    ax.legend(fontsize=8)
    ax.set_xticks(win_x)
    ax.grid(alpha=0.3)

for fi in range(n_cont, nrows * ncols):
    axes[fi // ncols][fi % ncols].set_visible(False)

plt.suptitle('TempConv temporal IG window profile per feature group\n'
             f'(seed {SEED}, top pairs by TempConv attribution)',
             fontsize=12)
savefig_manifest(fig, 'fig_B_tempconv_ig_window.png', [out_dir, desk_dir])
print('Saved fig_B_tempconv_ig_window.png')


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — R² advantage vs temporal lag strength
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Analysis 3: R² advantage vs lag-effect strength ===')

# lag_effect_strength[s, n, fi] = (max lagged |ρ| - |ρ at lag=0|) for each pair
lag0_idx = LAGS.index(0)
lag_zero = abs_lagged[:, :, :, lag0_idx]        # (29, 23, n_cont)
lag_max  = np.nanmax(abs_lagged, axis=3)         # (29, 23, n_cont)
lag_gain = lag_max - lag_zero                    # positive = temporal lag helps

# For each valid pair, compute overall lag strength = mean over continuous groups
lag_strength_per_pair = np.nanmean(lag_gain, axis=2)  # (29, 23)

# Compare with R² advantage
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
apply_style(fig, axes)

# Main scatter: lag strength vs R² advantage (shared valid pairs)
ax = axes[0]
mask = shared_valid
xs = lag_strength_per_pair[mask]
ys = r2_advantage[mask]
ok = ~(np.isnan(xs) | np.isnan(ys))
ax.scatter(xs[ok], ys[ok], s=30, alpha=0.6, color='#6A1B9A')
ax.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
rho, p = spearmanr(xs[ok], ys[ok])
ax.set_xlabel('Mean lag gain |ρ_max| - |ρ_0|\n(raw data, no ML)', fontsize=10)
ax.set_ylabel('R² advantage (TempConv-Cont - MLP)', fontsize=10)
ax.set_title(f'rho={rho:+.3f}  p={p:.3f}  n={ok.sum()}', fontsize=10)
ax.grid(alpha=0.3)

# Per-group scatter: for each continuous group separately
ax = axes[1]
gcolors = plt.cm.tab10(np.linspace(0, 1, n_cont))
for fi, g_idx in enumerate(continuous_groups):
    xs_g = lag_gain[:, :, fi][shared_valid]
    ys_g = r2_advantage[shared_valid]
    ok_g = ~(np.isnan(xs_g) | np.isnan(ys_g))
    rho_g, p_g = spearmanr(xs_g[ok_g], ys_g[ok_g])
    ax.scatter(xs_g[ok_g], ys_g[ok_g], s=15, alpha=0.5,
               color=gcolors[fi], label=f'{glabels[g_idx]} ρ={rho_g:+.2f}')

ax.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xlabel('Lag gain (max |ρ| - |ρ_0|) per feature', fontsize=10)
ax.set_ylabel('R² advantage (TempConv-Cont - MLP)', fontsize=10)
ax.set_title('Per-feature-group', fontsize=10)
ax.legend(fontsize=6, ncol=2)
ax.grid(alpha=0.3)

plt.suptitle('Does temporal lag strength predict where TempConv outperforms MLP?', fontsize=11)
savefig_manifest(fig, 'fig_C_r2_vs_lag.png', [out_dir, desk_dir])
print('Saved fig_C_r2_vs_lag.png')


# ══════════════════════════════════════════════════════════════════════════════
# Figure D — Combined summary for supervisor
# Shows: lagged ρ profile (best feature) + TempConv IG window profile + R² vs lag
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Figure D: Supervisor summary ===')

# Find the feature group with the strongest temporal effect (mean lag_gain over valid pairs)
mean_lag_gain_per_group = np.array([
    np.nanmean(lag_gain[:, :, fi][mlp_valid])
    for fi in range(n_cont)
])
best_fi  = int(np.argmax(mean_lag_gain_per_group))
best_g   = continuous_groups[best_fi]
best_lab = glabels[best_g]
best_col = cont_feat_cols[best_fi]
print(f'  Feature with strongest temporal lag: {best_lab} (group {best_g}, fi={best_fi})')

fig = plt.figure(figsize=(15, 5))
apply_style(fig)
gs  = fig.add_gridspec(1, 3, wspace=0.35)
ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

# Panel A: lagged ρ for best feature
vals_g  = abs_lagged[:, :, best_fi, :]
vals_v  = vals_g[mlp_valid]
mn_r    = np.nanmean(vals_v, axis=0)
sem_r   = np.nanstd(vals_v, axis=0) / np.sqrt(np.sum(~np.isnan(vals_v[:, 0])))
pk_lag  = LAGS[int(np.argmax(mn_r))]
ax1.plot(lag_x, mn_r, color='#1565C0', linewidth=2.5)
ax1.fill_between(lag_x, mn_r - sem_r, mn_r + sem_r, color='#1565C0', alpha=0.25)
ax1.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
ax1.axvline(pk_lag, color='red', linewidth=1.5, linestyle=':', label=f'peak t{pk_lag:+d}')
ax1.set_xlabel('Temporal offset (steps)', fontsize=10)
ax1.set_ylabel('Mean |Spearman ρ|', fontsize=10)
ax1.set_title(f'(A) Lagged correlation: {best_lab}\n'
              f'MLP uses t=0 only; TempConv can use t{pk_lag:+d}', fontsize=9)
ax1.legend(fontsize=9)
ax1.set_xticks(lag_x)
ax1.grid(alpha=0.3)

# Panel B: TempConv temporal IG for best feature
profs_best = [ig_profiles[(s, n)][best_col, :] for (s, n, gi) in meta_list
              if gi == best_g and (s, n) in ig_profiles]
if profs_best:
    pb = np.array(profs_best)
    mn_b  = pb.mean(axis=0)
    sem_b = pb.std(axis=0) / np.sqrt(len(pb))
    mn_b_n  = mn_b  / (mn_b.sum() + 1e-12)
    sem_b_n = sem_b / (mn_b.sum() + 1e-12)
    pk_ig = WIN_OFFSETS[int(np.argmax(mn_b_n))]
    ax2.plot(win_x, mn_b_n, color='#6A1B9A', linewidth=2.5)
    ax2.fill_between(win_x, mn_b_n - sem_b_n, mn_b_n + sem_b_n, color='#6A1B9A', alpha=0.25)
    ax2.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.axvline(pk_ig, color='red', linewidth=1.5, linestyle=':', label=f'peak t{pk_ig:+d}')
    ax2.set_xlabel('Window offset (steps)', fontsize=10)
    ax2.set_ylabel('Normalised mean |IG|', fontsize=10)
    ax2.set_title(f'(B) TempConv IG window profile: {best_lab}\n'
                  f'Encoder uses window position t{pk_ig:+d} most', fontsize=9)
    ax2.legend(fontsize=9)
    ax2.set_xticks(win_x)
    ax2.grid(alpha=0.3)

# Panel C: R² advantage vs lag gain for best feature
xs_c = lag_gain[:, :, best_fi][shared_valid]
ys_c = r2_advantage[shared_valid]
ok_c = ~(np.isnan(xs_c) | np.isnan(ys_c))
rho_c, p_c = spearmanr(xs_c[ok_c], ys_c[ok_c])
ax3.scatter(xs_c[ok_c], ys_c[ok_c], s=35, alpha=0.7, color='#E65100')
ax3.axhline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
ax3.set_xlabel(f'Lag gain for {best_lab}\n(raw data)', fontsize=10)
ax3.set_ylabel('R² advantage (TempConv-Cont − MLP)', fontsize=10)
ax3.set_title(f'(C) Temporal lag predicts TempConv advantage\n'
              f'rho={rho_c:+.3f}  p={p_c:.3f}  n={ok_c.sum()}', fontsize=9)
ax3.grid(alpha=0.3)

plt.suptitle(f'Temporal advantage: TempConv discovers past-{abs(pk_lag)}-step dependency\n'
             f'Verified by lagged Spearman ρ (no ML required)',
             fontsize=12)
savefig_manifest(fig, 'fig_D_supervisor_temporal.png', [out_dir, desk_dir])
print('Saved fig_D_supervisor_temporal.png')

# ══════════════════════════════════════════════════════════════════════════════
# Figure E — Significant temporal effects only: IG vs lagged ρ overlay
#
# Strategy:
#   1. Friedman test across lags per feature (using per-pair ρ values as subjects)
#      → only keep features where temporal profile is significantly non-flat
#   2. For those features, overlay mean TempConv IG window profile
#   3. Report profile shape correlation (Spearman ρ between the two mean curves)
#      rather than noisy peak-position matching
# ══════════════════════════════════════════════════════════════════════════════
print('\n=== Figure E: IG vs lagged ρ and η² (significant temporal effects only) ===')

FRIEDMAN_ALPHA = 0.05
WILCOXON_ALPHA = 0.05

# ── per-feature stats table ───────────────────────────────────────────────────
# Each entry: (lab, p_friedman, p_wilcoxon_rho, p_wilcoxon_eta, pk_rho, pk_eta,
#              ig_pk, shape_rho_rho, shape_rho_eta, flag)
results_table = []
sig_fi_list   = []

for fi, g_idx in enumerate(continuous_groups):
    lab      = glabels[g_idx]
    feat_col = sg[g_idx][1][0]

    # ── lagged |ρ| ──
    vals_rho   = abs_lagged[:, :, fi, :][mlp_valid]         # (n_valid, n_lags)
    ok_rho     = ~np.any(np.isnan(vals_rho), axis=1)
    clean_rho  = vals_rho[ok_rho]
    mn_rho     = np.nanmean(vals_rho, axis=0)
    pk_rho     = LAGS[int(np.argmax(mn_rho))] if not np.all(np.isnan(mn_rho)) else 0

    try:
        _, p_friedman = friedmanchisquare(*[clean_rho[:, li] for li in range(len(LAGS))])
    except Exception:
        p_friedman = 1.0

    if pk_rho != 0 and clean_rho.shape[0] > 4:
        try:
            _, p_wil_rho = wilcoxon(clean_rho[:, LAGS.index(pk_rho)],
                                    clean_rho[:, LAGS.index(0)], alternative='greater')
        except Exception:
            p_wil_rho = 1.0
    else:
        p_wil_rho = 1.0

    # ── lagged η² ──
    vals_eta   = lagged_eta2[:, :, fi, :][mlp_valid]
    ok_eta     = ~np.any(np.isnan(vals_eta), axis=1)
    clean_eta  = vals_eta[ok_eta]
    mn_eta     = np.nanmean(vals_eta, axis=0)
    pk_eta     = LAGS[int(np.argmax(mn_eta))] if not np.all(np.isnan(mn_eta)) else 0

    if pk_eta != 0 and clean_eta.shape[0] > 4:
        try:
            _, p_wil_eta = wilcoxon(clean_eta[:, LAGS.index(pk_eta)],
                                    clean_eta[:, LAGS.index(0)], alternative='greater')
        except Exception:
            p_wil_eta = 1.0
    else:
        p_wil_eta = 1.0

    # ── TempConv IG ──
    profs_g = [ig_profiles[(s, n)][feat_col, :]
               for (s, n, gi) in meta_list if gi == g_idx and (s, n) in ig_profiles]
    if profs_g:
        mn_ig = np.array(profs_g).mean(axis=0)
        ig_pk = WIN_OFFSETS[int(np.argmax(mn_ig))]
        shape_rr, _ = spearmanr(mn_rho, mn_ig)
        shape_er, _ = spearmanr(mn_eta, mn_ig)
    else:
        mn_ig = None
        ig_pk = None
        shape_rr = shape_er = np.nan

    # significant = Friedman AND (Wilcoxon ρ OR Wilcoxon η²)
    sig = (p_friedman < FRIEDMAN_ALPHA) and (
        p_wil_rho < WILCOXON_ALPHA or p_wil_eta < WILCOXON_ALPHA
    )
    if sig:
        sig_fi_list.append(fi)

    results_table.append(dict(lab=lab, pF=p_friedman,
                              pW_rho=p_wil_rho, pW_eta=p_wil_eta,
                              pk_rho=pk_rho, pk_eta=pk_eta, ig_pk=ig_pk,
                              sr_rho=shape_rr, sr_eta=shape_er,
                              mn_rho=mn_rho, mn_eta=mn_eta, mn_ig=mn_ig,
                              flag='SIG' if sig else 'ns'))

# Print table
print(f'\n  {"Feature":<14} {"Fried.p":>8} {"Wil.p(ρ)":>9} {"Wil.p(η²)":>10} '
      f'{"ρ peak":>7} {"η² peak":>8} {"IG peak":>8} '
      f'{"sh.ρ(ρ)":>9} {"sh.ρ(η²)":>10}  sig?')
print(f'  {"-"*100}')
for r in results_table:
    def _f(v, fmt='.3f'): return format(v, fmt) if v is not None and not np.isnan(v) else ' N/A'
    def _p(v): return f't{v:+d}' if v is not None else ' N/A'
    marker = '  <--' if r['flag'] == 'SIG' else ''
    print(f'  {r["lab"]:<14} {_f(r["pF"]):>8} {_f(r["pW_rho"]):>9} {_f(r["pW_eta"]):>10} '
          f'{_p(r["pk_rho"]):>7} {_p(r["pk_eta"]):>8} {_p(r["ig_pk"]):>8} '
          f'{_f(r["sr_rho"],"+.3f"):>9} {_f(r["sr_eta"],"+.3f"):>10}  {r["flag"]}{marker}')

if not sig_fi_list:
    print('\n  No features survived filter — showing all.')
    sig_fi_list = list(range(n_cont))

# ── Plot: 3 curves per significant feature ───────────────────────────────────
n_show  = len(sig_fi_list)
ncols_e = min(n_show, 4)
nrows_e = max(1, (n_show + ncols_e - 1) // ncols_e)
fig, axes = plt.subplots(nrows_e, ncols_e,
                         figsize=(5.5 * ncols_e, 4.5 * nrows_e), squeeze=False)
apply_style(fig, axes.flatten())

for plot_i, fi in enumerate(sig_fi_list):
    ax   = axes[plot_i // ncols_e][plot_i % ncols_e]
    r    = next(x for x in results_table if x['lab'] == glabels[continuous_groups[fi]])

    # normalize each curve to [0,1] so shapes are directly comparable
    def _norm(v):
        mn, mx = np.nanmin(v), np.nanmax(v)
        if mx - mn < 1e-12:
            return v - mn
        return (v - mn) / (mx - mn)

    mn_rho_n = _norm(r['mn_rho'])
    mn_eta_n = _norm(r['mn_eta'])

    ax.plot(lag_x, mn_rho_n, color='#1565C0', linewidth=2.5,
            label='Lagged |ρ|', zorder=3)
    ax.plot(lag_x, mn_eta_n, color='#2E7D32', linewidth=2.5, linestyle='-.',
            label='Lagged η²', zorder=3)

    if r['mn_ig'] is not None:
        mn_ig_n = _norm(r['mn_ig'])
        ax.plot(win_x, mn_ig_n, color='#6A1B9A', linewidth=2.5, linestyle='--',
                label=f'TempConv IG (n={sum(1 for x in meta_list if x[2]==continuous_groups[fi])})',
                zorder=3)
        title_ann = (f'sh.ρ(IG,ρ)={r["sr_rho"]:+.2f}  '
                     f'sh.ρ(IG,η²)={r["sr_eta"]:+.2f}')
    else:
        title_ann = 'no IG profiles'

    ax.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.4)
    sig_str = (f'Wil.ρ p={r["pW_rho"]:.3f}' if r['pW_rho'] < WILCOXON_ALPHA
               else f'Wil.η² p={r["pW_eta"]:.3f}')
    ax.set_title(f'{r["lab"]}  [{sig_str}]\n{title_ann}', fontsize=8)
    ax.set_xlabel('Time offset (steps)', fontsize=8)
    ax.set_ylabel('Norm. amplitude [0–1]', fontsize=8)
    ax.legend(fontsize=7, loc='lower center')
    ax.set_xticks(lag_x)
    ax.set_ylim(-0.1, 1.15)
    ax.grid(alpha=0.3)

for i in range(n_show, nrows_e * ncols_e):
    axes[i // ncols_e][i % ncols_e].set_visible(False)

plt.suptitle('Temporal tuning: TempConv IG vs lagged |ρ| vs lagged η²\n'
             'Only features with significant off-zero peak shown  |  '
             'All curves min-max normalised  |  sh.ρ = profile shape correlation',
             fontsize=10)
savefig_manifest(fig, 'fig_E_ig_vs_lagged_rho.png', [out_dir, desk_dir])
print('Saved fig_E_ig_vs_lagged_rho.png')

# ─── summary ─────────────────────────────────────────────────────────────────
print('\n=== Summary ===')
print(f'Features significant: {[glabels[continuous_groups[fi]] for fi in sig_fi_list]}')
print(f'R² advantage vs lag gain (best feature): rho={rho_c:+.3f}  p={p_c:.3f}')
print('\nDone.')
