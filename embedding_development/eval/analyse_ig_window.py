#!/usr/bin/env python3
"""
Analyse which window positions and input value ranges drive TempConv IG
for rotational-acceleration, head-angle-velocity, and head-angle (baseline).
"""
import sys, pickle
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, '../utils')
from load_encoder import build_windows, load_encoder

# ── data setup ────────────────────────────────────────────────────────────────
with open('../outputs/mlps/ensembles_multiseed/semantic_groups.pkl', 'rb') as f:
    sg = pickle.load(f)
group_names = [g[0] for g in sg]
feat_idx_by_group = {g: cols for g, cols in sg}

with open('../outputs/session_dataset_ensembles.pkl', 'rb') as f:
    ds = pickle.load(f)
sessions = list(ds.keys())

mlp_r2_all = np.load('../outputs/mlps/ensembles_multiseed/all_r2.npy')
ceb_r2_all = np.load('../outputs/cebra_eval/ensembles/all_r2.npy')
R2_THRESH  = 0.01
shared = (
    (~np.all(np.isnan(mlp_r2_all), axis=0)) & (np.nanmean(mlp_r2_all, axis=0) >= R2_THRESH) &
    (~np.all(np.isnan(ceb_r2_all), axis=0)) & (np.nanmean(ceb_r2_all, axis=0) >= R2_THRESH)
)

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IG_STEPS  = 50
SEED      = 42
WIN_LEN   = 10

# Window offset: position k in the window corresponds to timestep t + (k - 5)
# k=0 => t-5 (past), k=5 => t (current), k=9 => t+4 (future)
WIN_OFFSETS = list(range(-5, 5))   # -5, -4, ..., +4


class EncoderRidge(torch.nn.Module):
    def __init__(self, enc, coef, bias):
        super().__init__()
        self.encoder = enc
        self.register_buffer('coef', torch.tensor(coef, dtype=torch.float32))
        self.register_buffer('bias', torch.tensor(float(bias), dtype=torch.float32))

    def forward(self, x):
        z = self.encoder(x)
        if z.dim() == 3:
            z = z.squeeze(-1)
        return z @ self.coef + self.bias


def compute_ig_full(s_idx, n_idx, arm='cebra', seed=SEED):
    sess = sessions[s_idx]
    split_map = np.load(f'../splits/split_seed{seed}.npy', allow_pickle=True).item()
    test_trials  = [int(i) for i in split_map[sess]]
    all_trials   = list(ds[sess]['data'].keys())
    train_trials = [t for t in all_trials if t not in test_trials]

    def get(trial_ids):
        vt = [t for t in trial_ids if t in ds[sess]['data']]
        X  = np.concatenate([ds[sess]['data'][t]   for t in vt], axis=0).astype(np.float32)
        Y  = np.concatenate([ds[sess]['labels'][t] for t in vt], axis=0).astype(np.float32)
        return X, Y

    Xtr, Ytr = get(train_trials)
    Xte, _   = get(test_trials)
    wins_tr  = build_windows(Xtr).astype(np.float32)
    wins_te  = build_windows(Xte).astype(np.float32)

    mpath = f'../models/{arm}/ensembles/seed{seed}/session_{s_idx:02d}_neuron_{n_idx:02d}.pt'
    encoder, _, _ = load_encoder(mpath, device=device)
    encoder.eval()

    with torch.no_grad():
        wt = torch.tensor(wins_tr, device=device)
        emb = [encoder(wt[b:b+256]).cpu().numpy() for b in range(0, len(wt), 256)]
        emb_tr = np.concatenate(emb, axis=0)

    ridge = Ridge(alpha=1.0)
    ridge.fit(emb_tr, Ytr[:, n_idx])

    model  = EncoderRidge(encoder, ridge.coef_, ridge.intercept_).to(device).eval()
    test_t = torch.tensor(wins_te, device=device)
    alphas = np.linspace(0, 1, IG_STEPS + 1)[1:]

    grads_sum = np.zeros_like(wins_te, dtype=np.float64)
    for alpha in alphas:
        x_int = (float(alpha) * test_t).detach().requires_grad_(True)
        with torch.enable_grad():
            model(x_int).sum().backward()
        grads_sum += x_int.grad.detach().cpu().numpy()
        del x_int

    ig_full = wins_te * (grads_sum / IG_STEPS)   # (T, 17, 10)
    return ig_full, wins_te


# ── pairs ─────────────────────────────────────────────────────────────────────
rota_col = feat_idx_by_group['frame_YawPitch_abs_acc_sum_500msMedian'][0]
hvel_col = feat_idx_by_group['head_angle_vel'][0]
hang_col = feat_idx_by_group['head_angle'][0]

PAIRS = [
    ('rot_acc  S01E05', 1,  5,  rota_col, 'rota'),
    ('rot_acc  S11E07', 11, 7,  rota_col, 'rota'),
    ('rot_acc  S28E06', 28, 6,  rota_col, 'rota'),
    ('head_vel S12E13', 12, 13, hvel_col, 'hvel'),
    ('head_vel S10E09', 10, 9,  hvel_col, 'hvel'),
    ('head_vel S09E09', 9,  9,  hvel_col, 'hvel'),
    ('head_ang S00E02', 0,  2,  hang_col, 'base'),
    ('head_ang S00E01', 0,  1,  hang_col, 'base'),
]

print('Window layout: position k => time offset k-5')
print('  k=0: t-5   k=5: t(now)   k=9: t+4')
print()

hdr = 'label                   '
hdr += '  '.join(f't{o:+d}' for o in WIN_OFFSETS)
hdr += '   peak_offset   rho(|x|,|ig|)   rho(x,ig_signed)'
print(hdr)
print('-' * len(hdr))

results = {}
for label, s_idx, n_idx, feat_col, cat in PAIRS:
    ig_full, wins_te = compute_ig_full(s_idx, n_idx)

    feat_ig_abs  = np.abs(ig_full[:, feat_col, :])   # (T, 10)
    feat_ig_sgn  = ig_full[:, feat_col, :]            # (T, 10) signed

    win_profile = feat_ig_abs.mean(axis=0)            # (10,) mean |IG| per window pos
    peak_k      = int(np.argmax(win_profile))
    peak_offset = WIN_OFFSETS[peak_k]

    # Magnitude correlation: does |IG| rise when input is large?
    input_at_peak = wins_te[:, feat_col, peak_k]
    ig_at_peak    = feat_ig_abs[:, peak_k]
    ok = ~(np.isnan(input_at_peak) | np.isnan(ig_at_peak))
    rho_mag,  _ = spearmanr(np.abs(input_at_peak[ok]), ig_at_peak[ok])

    # Signed correlation: does positive input get positive IG? (sign consistency)
    ig_sgn_at_peak = feat_ig_sgn[:, peak_k]
    rho_sgn, _ = spearmanr(input_at_peak[ok], ig_sgn_at_peak[ok])

    profile_str = '  '.join(f'{v:.4f}' for v in win_profile)
    print(f'{label:25s} [{profile_str}]   t{peak_offset:+d}   '
          f'rho_mag={rho_mag:+.3f}   rho_sgn={rho_sgn:+.3f}')

    results[label] = dict(cat=cat, win_profile=win_profile,
                          peak_k=peak_k, peak_offset=peak_offset,
                          rho_mag=rho_mag, rho_sgn=rho_sgn,
                          ig_full=ig_full, wins_te=wins_te, feat_col=feat_col)

# ── mean profile per category ─────────────────────────────────────────────────
print()
for cat_label, cat_id in [('rot_acc  (CEBRA-excl)', 'rota'),
                           ('head_vel (CEBRA-excl)', 'hvel'),
                           ('head_ang (baseline)',   'base')]:
    profs = [v['win_profile'] for v in results.values() if v['cat'] == cat_id]
    if not profs:
        continue
    mp   = np.mean(profs, axis=0)
    norm = mp / mp.sum()
    print(f'{cat_label}:')
    print('  mean |IG| profile (normalised): ' +
          '  '.join(f't{o:+d}={v:.3f}' for o, v in zip(WIN_OFFSETS, norm)))
    print(f'  peak at t{WIN_OFFSETS[int(np.argmax(mp))]:+d}')
    rho_mags = [v['rho_mag'] for v in results.values() if v['cat'] == cat_id]
    rho_sgns = [v['rho_sgn'] for v in results.values() if v['cat'] == cat_id]
    print(f'  mean rho(|x|,|ig|) = {np.mean(rho_mags):.3f}   '
          f'mean rho(x,ig_sgn) = {np.mean(rho_sgns):.3f}')
    print()

# ── extra: look at the INPUT value distribution at high vs low |IG| timesteps ─
print('=== Input value distribution: high |IG| vs low |IG| timesteps ===')
for cat_id, feat_name in [('rota', 'rot_acc'), ('hvel', 'head_vel'), ('base', 'head_ang')]:
    pair_label = next(l for l, v in results.items() if v['cat'] == cat_id)
    r = results[pair_label]
    k = r['peak_k']
    ig_at_k  = np.abs(r['ig_full'][:, r['feat_col'], k])
    x_at_k   = r['wins_te'][:, r['feat_col'], k]
    thr      = np.percentile(ig_at_k, 75)
    hi_mask  = ig_at_k >= thr
    lo_mask  = ig_at_k <  thr
    poff = r['peak_offset']
    print(f'{pair_label} ({feat_name}) at peak window position k={k} (t{poff:+d}):')
    print(f'  |x| mean: high-|IG|={np.abs(x_at_k[hi_mask]).mean():.4f}  '
          f'low-|IG|={np.abs(x_at_k[lo_mask]).mean():.4f}  '
          f'ratio={np.abs(x_at_k[hi_mask]).mean()/max(np.abs(x_at_k[lo_mask]).mean(),1e-9):.2f}x')
    # percentile breakdown of x values in high-IG timesteps
    pct = np.percentile(np.abs(x_at_k[hi_mask]), [25, 50, 75, 90, 99])
    print(f'  |x| percentiles in high-|IG| subset: '
          f'p25={pct[0]:.3f} p50={pct[1]:.3f} p75={pct[2]:.3f} p90={pct[3]:.3f} p99={pct[4]:.3f}')
    print()
