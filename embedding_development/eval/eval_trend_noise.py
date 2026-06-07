#!/usr/bin/env python3
"""
eval_trend_noise.py

Trend vs Noise R² decomposition for the top-5 ensemble-session pairs,
comparing MLP, TempConv-Cont, and TempConv-Pred on the same pairs.

Data is at 40 ms/frame (25 Hz).
Slow component = 500 ms moving average → TREND_WINDOW = 12 bins.
Fast component = residual (y_true − slow).

Outputs: trend_noise_comparison.png
"""
import os, sys, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.models import MLP
from utils.load_encoder import build_windows, load_encoder
from utils.figure_style import (
    FIG, DPI, FONT, MODEL_COLORS,
    apply_style, add_footnote, savefig_manifest,
)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
SEED         = 42
N_EXAMPLE    = 5
TREND_WINDOW = 12     # int(0.5 / 0.040) = 12 bins @ 40 ms/frame = 500 ms
RIDGE_ALPHA  = 1.0

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, "..")
mdir = os.path.join(root, "outputs", "mlps", "ensembles_multiseed")

OUT_DIRS = [
    mdir,
    '/mnt/c/Users/amits/Desktop',
]

all_r2          = np.load(os.path.join(mdir, "all_r2.npy"))   # (seeds, sessions, ensembles)
n_seeds, n_sessions, n_ensembles = all_r2.shape
mean_r2         = np.nanmean(all_r2, axis=0)

flat_order = np.argsort(mean_r2.ravel())[::-1]
top_pairs  = []
for fi in flat_order:
    s_idx, n_idx = fi // n_ensembles, fi % n_ensembles
    if np.isfinite(mean_r2[s_idx, n_idx]) and mean_r2[s_idx, n_idx] > 0:
        top_pairs.append((s_idx, n_idx))
    if len(top_pairs) == N_EXAMPLE:
        break

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
cache = os.path.join(root, "outputs", "session_dataset_ensembles.pkl")
with open(cache, "rb") as f:
    ds = pickle.load(f)
session_ids = list(ds.keys())

tidx_map = np.load(os.path.join(root, "splits", f"split_seed{SEED}.npy"),
                   allow_pickle=True).item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")


def _load_mlp(s_idx, n_idx):
    path = os.path.join(root, "models", "mlps", "ensembles",
                        f"seed{SEED}", f"session_{s_idx:02d}_neuron_{n_idx:02d}.pt")
    if not os.path.exists(path):
        return None
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        h   = sd['model_state_dict']['fc.0.weight'].shape[0]
        nin = sd['model_state_dict']['fc.0.weight'].shape[1]
        m   = MLP(nin, h, 2, 1).to(device)
        m.load_state_dict(sd['model_state_dict'])
    else:
        h   = sd['fc.0.weight'].shape[0]
        nin = sd['fc.0.weight'].shape[1]
        m   = MLP(nin, h, 2, 1).to(device)
        m.load_state_dict(sd)
    m.eval()
    return m


def _embed(encoder, X_np):
    wins = build_windows(X_np)
    with torch.no_grad():
        z = encoder(torch.tensor(wins, dtype=torch.float32, device=device))
        if z.dim() == 3:
            z = z.squeeze(-1)
    return z.cpu().numpy()


def _cebra_predict(arm, s_idx, n_idx, Xtr, Ytr_col, Xte):
    mpath = os.path.join(root, "models", arm, "ensembles",
                         f"seed{SEED}", f"session_{s_idx:02d}_neuron_{n_idx:02d}.pt")
    if not os.path.exists(mpath):
        return None
    encoder, _, _ = load_encoder(mpath, device=str(device))
    encoder.eval()
    Z_tr  = _embed(encoder, Xtr)
    Z_te  = _embed(encoder, Xte)
    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(Z_tr, Ytr_col)
    pred  = ridge.predict(Z_te)
    del encoder; torch.cuda.empty_cache()
    return pred


def _decompose(y_true, y_pred):
    """
    Returns (r2_trend, r2_noise).
    Slow component of true and pred: 500 ms moving average.
    r2_trend = R²(slow_true, slow_pred); r2_noise = R²(fast_true, fast_pred).
    """
    slow_true = uniform_filter1d(y_true, size=TREND_WINDOW, mode='nearest')
    fast_true = y_true - slow_true
    slow_pred = uniform_filter1d(y_pred, size=TREND_WINDOW, mode='nearest')
    fast_pred = y_pred - slow_pred

    r2_t = r2_score(slow_true, slow_pred)
    r2_n = (r2_score(fast_true, fast_pred)
            if np.var(fast_true) > 1e-8 else np.nan)
    return r2_t, r2_n


# ─── COMPUTE DECOMPOSITION FOR ALL 3 MODELS ───────────────────────────────────
ARMS = [
    ("MLP",           None),
    ("TempConv-Cont", "cebra"),
    ("TempConv-Pred", "cebra_pred"),
]

pair_labels = []
results     = {name: {"trend": [], "noise": []} for name, _ in ARMS}

for s_idx, n_idx in top_pairs:
    sess_id     = session_ids[s_idx]
    test_trials = tidx_map.get(sess_id, [])
    all_trials  = list(ds[sess_id]["data"].keys())
    train_trials = [t for t in all_trials if t not in test_trials]
    valid_t      = [t for t in test_trials if t in ds[sess_id]["data"]]

    short_sess = sess_id[:10]
    pair_labels.append(f"{short_sess}\nE{n_idx+1:02d}")

    if not valid_t or not train_trials:
        for name, _ in ARMS:
            results[name]["trend"].append(np.nan)
            results[name]["noise"].append(np.nan)
        continue

    Xte = np.concatenate([ds[sess_id]["data"][t]   for t in valid_t], 0).astype(np.float32)
    Yte = np.concatenate([ds[sess_id]["labels"][t] for t in valid_t], 0).astype(np.float32)
    Xtr = np.concatenate([ds[sess_id]["data"][t]   for t in train_trials
                          if t in ds[sess_id]["data"]], 0).astype(np.float32)
    Ytr = np.concatenate([ds[sess_id]["labels"][t] for t in train_trials
                          if t in ds[sess_id]["data"]], 0).astype(np.float32)
    y_te = Yte[:, n_idx].astype(float)

    for name, arm in ARMS:
        if arm is None:
            model = _load_mlp(s_idx, n_idx)
            if model is None:
                results[name]["trend"].append(np.nan)
                results[name]["noise"].append(np.nan)
                continue
            with torch.no_grad():
                _, yb = model(torch.tensor(Xte, device=device))
            pred = yb.squeeze(-1).cpu().numpy().astype(float)
            del model; torch.cuda.empty_cache()
        else:
            pred = _cebra_predict(arm, s_idx, n_idx,
                                  Xtr, Ytr[:, n_idx], Xte)
            if pred is None:
                results[name]["trend"].append(np.nan)
                results[name]["noise"].append(np.nan)
                continue
            pred = pred.astype(float)

        r2_t, r2_n = _decompose(y_te, pred)
        results[name]["trend"].append(r2_t)
        results[name]["noise"].append(r2_n)
        r2_n_str = f"{r2_n:.3f}" if np.isfinite(r2_n) else "nan"
        print(f"  {name:20s} S{s_idx+1:02d} E{n_idx+1:02d}: "
              f"R2_trend={r2_t:.3f}  R2_noise={r2_n_str}")

# ─── PLOT ─────────────────────────────────────────────────────────────────────
n_pairs = len(pair_labels)
x       = np.arange(n_pairs)

# Two bars per model (trend=solid, noise=pale); n_models=3 → 6 bars per x-tick
def _pale(hex_color, alpha=0.45):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hex_color)
    return (r + (1 - r) * alpha, g + (1 - g) * alpha, b + (1 - b) * alpha)

bar_w   = 0.12
n_bars  = 2 * len(ARMS)
offsets = np.linspace(-(n_bars - 1) / 2 * bar_w,
                       (n_bars - 1) / 2 * bar_w, n_bars)

fig, ax = plt.subplots(figsize=FIG.FULL)
apply_style(fig, ax)

bar_idx  = 0
handles  = []
for name, _ in ARMS:
    c_solid = MODEL_COLORS[name]
    c_pale  = _pale(c_solid)
    trend_vals = [v if np.isfinite(v) else 0 for v in results[name]["trend"]]
    noise_vals = [v if np.isfinite(v) else 0 for v in results[name]["noise"]]

    b1 = ax.bar(x + offsets[bar_idx],     trend_vals, bar_w,
                color=c_solid, label=f'{name} (slow)')
    b2 = ax.bar(x + offsets[bar_idx + 1], noise_vals, bar_w,
                color=c_pale,  label=f'{name} (fast)')
    handles += [b1, b2]
    bar_idx += 2

ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(pair_labels, fontsize=FONT.TICK)
ax.set_ylabel('R²', fontsize=FONT.LABEL)
ax.set_ylim(-0.6, 1.05)
ax.legend(handles=handles, fontsize=FONT.LEGEND, frameon=False,
          ncol=3, loc='upper right')

add_footnote(fig,
    f"Top-{N_EXAMPLE} ensemble-session pairs by mean MLP R²; seed={SEED}; "
    f"slow = {TREND_WINDOW}-bin ({TREND_WINDOW * 40} ms) moving average; fast = residual")

savefig_manifest(fig, "trend_noise_comparison.png", OUT_DIRS)
print("Done.")
