#!/usr/bin/env python3
"""
eval_freq_trace_example.py

Finds the (session, ensemble) pair where TempConv-Pred has the largest
R²_noise advantage over MLP, then plots a multi-trial trace showing:
  - Actual neural activity (black)
  - MLP prediction (green)
  - TempConv-Pred prediction (orange)

The fast component (high-frequency residual after 500 ms trend removal)
is shown separately below to make the gap visually obvious.

Output: outputs/mlps/ensembles_multiseed/freq_trace_example.png
"""
import os, sys, pickle
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.models import MLP
from utils.load_encoder import build_windows, load_encoder
from utils.figure_style import (
    FIG, DPI, FONT, MODEL_COLORS,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

root   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
mdir   = os.path.join(root, 'outputs', 'mlps', 'ensembles_multiseed')
OUT_DIRS = [mdir, '/mnt/c/Users/amits/Desktop']

SEED         = 42
TREND_WINDOW = 12   # 500 ms at 40 ms/bin
RIDGE_ALPHA  = 1.0
N_TOP        = 8    # candidate pairs to evaluate
SHOW_BINS    = 200  # consecutive bins to display in trace (~8 seconds)
device       = torch.device('cpu')

# ── Load ──────────────────────────────────────────────────────────────────────
all_r2  = np.load(os.path.join(mdir, 'all_r2.npy'))
mean_r2 = np.nanmean(all_r2, axis=0)
n_s, n_e = mean_r2.shape

ds       = pickle.load(open(os.path.join(root, 'outputs', 'session_dataset_ensembles.pkl'), 'rb'))
sessions = list(ds.keys())
tidx_map = np.load(os.path.join(root, 'splits', f'split_seed{SEED}.npy'),
                   allow_pickle=True).item()

flat_order = np.argsort(mean_r2.ravel())[::-1]
top_pairs  = []
for fi in flat_order:
    s, e = fi // n_e, fi % n_e
    if np.isfinite(mean_r2[s, e]) and mean_r2[s, e] > 0:
        top_pairs.append((s, e))
    if len(top_pairs) == N_TOP:
        break


# ── Model helpers ─────────────────────────────────────────────────────────────
def load_mlp(s, e):
    path = os.path.join(root, 'models', 'mlps', 'ensembles',
                        f'seed{SEED}', f'session_{s:02d}_neuron_{e:02d}.pt')
    if not os.path.exists(path):
        return None
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        h = sd['model_state_dict']['fc.0.weight'].shape[0]
        nin = sd['model_state_dict']['fc.0.weight'].shape[1]
        m = MLP(nin, h, 2, 1).to(device)
        m.load_state_dict(sd['model_state_dict'])
    else:
        h = sd['fc.0.weight'].shape[0]
        nin = sd['fc.0.weight'].shape[1]
        m = MLP(nin, h, 2, 1).to(device)
        m.load_state_dict(sd)
    m.eval()
    return m


def embed(encoder, X_np):
    wins = build_windows(X_np)
    with torch.no_grad():
        z = encoder(torch.tensor(wins, dtype=torch.float32, device=device))
        if z.dim() == 3:
            z = z.squeeze(-1)
    return z.cpu().numpy()


def cebra_predict(arm, s, e, Xtr, ytr, Xte):
    path = os.path.join(root, 'models', arm, 'ensembles',
                        f'seed{SEED}', f'session_{s:02d}_neuron_{e:02d}.pt')
    if not os.path.exists(path):
        return None
    enc, _, _ = load_encoder(path, device=str(device))
    enc.eval()
    Ztr = embed(enc, Xtr)
    Zte = embed(enc, Xte)
    del enc; torch.cuda.empty_cache()
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(Ztr, ytr)
    return ridge.predict(Zte)


def noise_r2(y_true, y_pred):
    n = min(len(y_true), len(y_pred))
    yt, yp = y_true[:n], y_pred[:n]
    ft = yt - uniform_filter1d(yt.astype(float), TREND_WINDOW, mode='nearest')
    fp = yp - uniform_filter1d(yp.astype(float), TREND_WINDOW, mode='nearest')
    if np.var(ft) < 1e-8:
        return np.nan
    return float(r2_score(ft, fp))


# ── Find best pair ─────────────────────────────────────────────────────────────
best = dict(gap=-np.inf)

for s, e in top_pairs:
    sess_id = sessions[s]
    sd      = ds[sess_id]
    all_t   = list(sd['data'].keys())
    test_t  = [t for t in tidx_map.get(sess_id, []) if t in sd['data']]
    train_t = [t for t in all_t if t not in test_t]
    if not test_t or not train_t:
        continue

    Xte = np.concatenate([sd['data'][t] for t in test_t]).astype(float)
    yte = np.concatenate([sd['labels'][t][:, e] for t in test_t]).astype(float)
    Xtr = np.concatenate([sd['data'][t] for t in train_t]).astype(float)
    ytr = np.concatenate([sd['labels'][t][:, e] for t in train_t]).astype(float)

    mlp = load_mlp(s, e)
    if mlp is None:
        continue
    with torch.no_grad():
        out = mlp(torch.tensor(Xte, dtype=torch.float32))
        y_mlp = (out[0] if isinstance(out, tuple) else out).cpu().numpy().ravel()

    y_tcp = cebra_predict('cebra_pred', s, e, Xtr, ytr, Xte)
    if y_tcp is None:
        continue

    n = min(len(yte), len(y_mlp), len(y_tcp))
    r2n_mlp = noise_r2(yte[:n], y_mlp[:n])
    r2n_tcp = noise_r2(yte[:n], y_tcp[:n])

    if np.isnan(r2n_mlp) or np.isnan(r2n_tcp):
        continue

    gap = r2n_tcp - r2n_mlp
    print(f'  S{s+1:02d} E{e+1:02d}  R²={mean_r2[s,e]:.3f}  '
          f'noise_MLP={r2n_mlp:.3f}  noise_TC={r2n_tcp:.3f}  gap={gap:+.3f}')

    if gap > best['gap']:
        best = dict(s=s, e=e, gap=gap, r2=mean_r2[s, e],
                    yte=yte[:n], y_mlp=y_mlp[:n], y_tcp=y_tcp[:n],
                    r2n_mlp=r2n_mlp, r2n_tcp=r2n_tcp)

if 's' not in best:
    print('No valid pair found'); raise SystemExit(1)

print(f'\nBest: S{best["s"]+1:02d} E{best["e"]+1:02d}  '
      f'gap={best["gap"]:+.3f}  R²={best["r2"]:.3f}')

# ── Pick the most informative window ──────────────────────────────────────────
yte   = best['yte']
y_mlp = best['y_mlp']
y_tcp = best['y_tcp']
n     = len(yte)

# Find SHOW_BINS-long window with highest variance in y_tcp fast component
fast_tcp = y_tcp - uniform_filter1d(y_tcp.astype(float), TREND_WINDOW, mode='nearest')
best_start = 0
best_var   = -1
for start in range(0, n - SHOW_BINS, 10):
    v = np.var(fast_tcp[start:start + SHOW_BINS])
    if v > best_var:
        best_var   = v
        best_start = start
sl = slice(best_start, best_start + SHOW_BINS)
t  = np.arange(SHOW_BINS) * 0.040   # seconds

yte_w   = yte[sl]
y_mlp_w = y_mlp[sl]
y_tcp_w = y_tcp[sl]
trend_t = uniform_filter1d(yte_w.astype(float),   TREND_WINDOW, mode='nearest')
fast_t  = yte_w   - trend_t
fast_mlp= y_mlp_w - uniform_filter1d(y_mlp_w.astype(float), TREND_WINDOW, mode='nearest')
fast_tcp_w= y_tcp_w - uniform_filter1d(y_tcp_w.astype(float), TREND_WINDOW, mode='nearest')

# ── Figure: 2 rows × 1 column ─────────────────────────────────────────────────
C_ACTUAL = '#333333'
C_MLP    = MODEL_COLORS.get('MLP',    '#2CA02C')
C_TCP    = MODEL_COLORS.get('TC-Pred','#FF7F0E')

fig, (ax_full, ax_fast) = plt.subplots(2, 1, figsize=FIG.FULL,
                                        gridspec_kw={'height_ratios': [1.6, 1],
                                                     'hspace': 0.45})
apply_style(fig, [ax_full, ax_fast])
fig.subplots_adjust(bottom=0.14)

# Row 1 — raw signal + predictions
ax_full.plot(t, yte_w,   color=C_ACTUAL, lw=1.2, alpha=0.90, label='Actual', zorder=3)
ax_full.plot(t, y_mlp_w, color=C_MLP,    lw=1.0, alpha=0.80, label=f'MLP  (R²_noise={best["r2n_mlp"]:.2f})', ls='--')
ax_full.plot(t, y_tcp_w, color=C_TCP,    lw=1.0, alpha=0.80, label=f'TC-Pred (R²_noise={best["r2n_tcp"]:.2f})')
ax_full.set_ylabel('z-scored activity', fontsize=FONT.LABEL - 1)
ax_full.legend(fontsize=FONT.LEGEND - 1, frameon=True, facecolor='white',
               framealpha=0.85, edgecolor='none', loc='upper right')
add_panel_label(ax_full, 'A')

# Row 2 — fast (residual) component only
ax_fast.plot(t, fast_t,     color=C_ACTUAL, lw=1.2, alpha=0.90, label='Actual (fast)', zorder=3)
ax_fast.plot(t, fast_mlp,   color=C_MLP,    lw=1.0, alpha=0.80, label='MLP', ls='--')
ax_fast.plot(t, fast_tcp_w, color=C_TCP,    lw=1.0, alpha=0.80, label='TC-Pred')
ax_fast.axhline(0, color='#888', lw=0.5, ls=':')
ax_fast.set_xlabel('Time (s)', fontsize=FONT.LABEL - 1)
ax_fast.set_ylabel('Residual', fontsize=FONT.LABEL - 1)
ax_fast.legend(fontsize=FONT.LEGEND - 1, frameon=True, facecolor='white',
               framealpha=0.85, edgecolor='none', loc='upper right')
add_panel_label(ax_fast, 'B')

add_footnote(fig,
    f'S{best["s"]+1:02d} E{best["e"]+1:02d}: '
    f'R²_noise: MLP={best["r2n_mlp"]:.3f}, TC-Pred={best["r2n_tcp"]:.3f} '
    f'(Δ={best["gap"]:+.3f}).  '
    f'Fast component = signal minus 500 ms moving average.  '
    f'Window shown: {best_start*0.04:.1f}–{(best_start+SHOW_BINS)*0.04:.1f} s of test set.')

savefig_manifest(fig, 'freq_trace_example.png', OUT_DIRS)
print('Saved freq_trace_example.png')
