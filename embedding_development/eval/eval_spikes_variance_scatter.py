#!/usr/bin/env python3
"""
eval_spikes_variance_scatter.py

S10 scatter plots using z-scored firing rates (as trained):
  img[0]: test-set variance vs MSE   (reference lines: MSE=1, MSE=variance)
  img[1]: test-set variance vs R²    (reference line: R²=1)

Each data point is one (session, neuron) pair.  Variance = mean over test
trials of within-trial temporal variance of z-scored labels.
"""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from figure_style import (
    FIG, DPI, FONT, LINE, MARKER,
    MODEL_COLORS, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

SEED = 42

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base, "..")
spk  = os.path.join(root, "outputs", "mlps", "spikes_multiseed")

all_r2 = np.load(os.path.join(spk, "all_r2.npy"))   # (seeds, sessions, neurons)
n_seeds, n_sessions, n_neurons = all_r2.shape

cache = os.path.join(root, "outputs", "session_dataset_spikes.pkl")
with open(cache, "rb") as f:
    ds_spk = pickle.load(f)

split = np.load(os.path.join(root, "splits", f"split_seed{SEED}.npy"),
                allow_pickle=True).item()

# ── Per-(session, neuron) arrays ──────────────────────────────────────────────
variance_all, mse_all, r2_all = [], [], []

for s_idx, (sess_id, sess_data) in enumerate(ds_spk.items()):
    test_trials = [t for t in split.get(sess_id, [])
                   if t in sess_data["labels"]]
    if not test_trials:
        continue

    # Variance = mean over test trials of within-trial variance (z-scored labels)
    test_labels = {t: sess_data["labels"][t] for t in test_trials}

    r2_s = np.nanmedian(all_r2[:, s_idx, :], axis=0)   # (neurons,)

    for n_idx in range(n_neurons):
        if not np.isfinite(r2_s[n_idx]):
            continue
        var_n = float(np.mean([np.var(test_labels[t][:, n_idx])
                                for t in test_trials]))
        mse_n = var_n * max(0.0, 1.0 - r2_s[n_idx])
        variance_all.append(var_n)
        mse_all.append(mse_n)
        r2_all.append(float(r2_s[n_idx]))

variance_all = np.array(variance_all)
mse_all      = np.array(mse_all)
r2_all       = np.clip(np.array(r2_all), 0, 1)

r_mse = float(np.corrcoef(variance_all, mse_all)[0, 1])
r_r2  = float(np.corrcoef(variance_all, r2_all)[0, 1])

# ── Plot 1: variance vs MSE ───────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(3.52, 2.74))
apply_style(fig1, ax1)
ax1.scatter(variance_all, mse_all, s=8, alpha=0.15, color='steelblue', linewidths=0)
m, b = np.polyfit(variance_all, mse_all, 1)
xl = np.linspace(variance_all.min(), variance_all.max(), 200)
ax1.plot(xl, m * xl + b, color='firebrick', lw=1.2, label=f'r = {r_mse:.2f}')
ax1.axhline(1.0, color='gray', linestyle='--', lw=0.8, label='Baseline MSE = 1')
ax1.plot(xl, xl, color='darkorange', linestyle=':', lw=0.8, label='MSE = Variance')
ax1.set_xlabel('Variance (z-scored FR, test set)', fontsize=8)
ax1.set_ylabel('MSE (z-scored FR, test set)', fontsize=8)
ax1.set_title('Firing Rate Variance vs. Prediction MSE', fontsize=8)
ax1.legend(fontsize=6, frameon=False)
ax1.spines[['top', 'right']].set_visible(False)
savefig_manifest(fig1, "variance_vs_mse.png", [spk])
print(f"Saved {os.path.join(spk, 'variance_vs_mse.png')}  (r={r_mse:.2f})")

# ── Plot 2: variance vs R² ────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(3.39, 2.73))
apply_style(fig2, ax2)
ax2.scatter(variance_all, r2_all, s=8, alpha=0.15, color='darkorange', linewidths=0)
m2, b2 = np.polyfit(variance_all, r2_all, 1)
xl2 = np.linspace(variance_all.min(), variance_all.max(), 200)
ax2.plot(xl2, m2 * xl2 + b2, color='firebrick', lw=1.2, label=f'r = {r_r2:.2f}')
ax2.axhline(1.0, color='gray', linestyle='--', lw=0.8, label='R² = 1')
ax2.set_xlabel('Variance (z-scored FR, test set)', fontsize=8)
ax2.set_ylabel('R² (z-scored FR, test set)', fontsize=8)
ax2.set_title('Firing Rate Variance vs. Prediction R²', fontsize=8)
ax2.legend(fontsize=6, frameon=False)
ax2.spines[['top', 'right']].set_visible(False)
savefig_manifest(fig2, "variance_vs_r2.png", [spk])
print(f"Saved {os.path.join(spk, 'variance_vs_r2.png')}  (r={r_r2:.2f})")
