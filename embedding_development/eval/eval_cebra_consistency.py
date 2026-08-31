#!/usr/bin/env python3
"""
eval_cebra_consistency.py

Cross-seed embedding consistency for MLP, TempConv-Cont, and TempConv-Pred.

Metric: mean pairwise Pearson r of predictions on overlapping test trials
        (trials present in ≥2 seeds' per-seed test sets), same as
        eval_embedding_consistency.py for MLP.

For CEBRA: per-seed encoder + Ridge(alpha=1.0) probe, trained on that seed's
training trials, predicts on overlapping test trials.

Reads consistency_matrix_mlp.npy (precomputed by eval_embedding_consistency.py)
and computes TempConv-Cont / TempConv-Pred consistency matrices, then plots a violin.

Outputs
-------
outputs/cebra_comparison/consistency_matrix_cebra_contrast.npy
outputs/cebra_comparison/consistency_matrix_cebra_pred.npy
outputs/cebra_comparison/embedding_consistency_violin.png
"""
import os, sys, itertools, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from load_encoder import build_windows, load_encoder
from figure_style import (
    FIG, DPI, FONT, LINE, MARKER,
    MODEL_COLORS, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
SEEDS          = [42, 43, 44, 45, 46]
RIDGE_ALPHA    = 1.0
R2_THRESHOLD   = 0.1
MIN_TIMEPOINTS = 5

base        = os.path.dirname(os.path.abspath(__file__))
root        = os.path.join(base, "..")
splits_dir  = os.path.join(root, "splits")
out_dir     = os.path.join(root, "outputs", "cebra_comparison")
mlp_dir     = os.path.join(root, "outputs", "mlps", "ensembles_multiseed")
cc_dir      = os.path.join(root, "outputs", "cebra_64d_eval",      "ensembles")
cp_dir      = os.path.join(root, "outputs", "cebra_pred_64d_eval", "ensembles")
os.makedirs(out_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

# ─── LOAD SESSION DATASET ─────────────────────────────────────────────────────
cache_path = os.path.join(root, "outputs", "session_dataset_ensembles.pkl")
with open(cache_path, "rb") as f:
    ds = pickle.load(f)
session_ids = list(ds.keys())
n_sessions  = len(session_ids)

# ─── PER-SEED SPLITS ──────────────────────────────────────────────────────────
splits = {
    seed: np.load(os.path.join(splits_dir, f"split_seed{seed}.npy"),
                  allow_pickle=True).item()
    for seed in SEEDS
}

# ─── OVERLAP MAP: trials in ≥2 seeds' test sets ───────────────────────────────
overlap_map = {}
for s_idx, sess in enumerate(session_ids):
    trial_seeds = {}
    for seed in SEEDS:
        for trial in splits[seed].get(sess, []):
            trial_seeds.setdefault(trial, []).append(seed)
    overlap_map[sess] = trial_seeds

# ─── MEAN R² MASKS (threshold out silent ensembles) ───────────────────────────
mlp_r2  = np.load(os.path.join(mlp_dir, "all_r2.npy"))   # (5,29,23)
cc_r2   = np.load(os.path.join(cc_dir,  "all_r2.npy"))
cp_r2   = np.load(os.path.join(cp_dir,  "all_r2.npy"))

n_seeds, n_sess_chk, n_ensembles = mlp_r2.shape
assert n_sess_chk == n_sessions

mean_r2_mlp = np.nanmean(mlp_r2, axis=0)   # (29,23)
mean_r2_cc  = np.nanmean(cc_r2,  axis=0)
mean_r2_cp  = np.nanmean(cp_r2,  axis=0)

# ─── HELPER: TempConv CONSISTENCY ────────────────────────────────────────────────
def compute_cebra_consistency(arm, mean_r2_mat):
    """
    Returns consistency_matrix (n_sessions, n_ensembles) for a TempConv arm.
    arm: 'cebra' or 'cebra_pred'
    """
    models_root = os.path.join(root, "models", arm + "_64d", "ensembles")
    cons_mat    = np.full((n_sessions, n_ensembles), np.nan)

    for s_idx, sess in enumerate(session_ids):
        sd = ds[sess]
        all_trials = list(sd["data"].keys())

        multi_seed_trials = {
            t: sl for t, sl in overlap_map[sess].items()
            if len(sl) >= 2 and t in sd["data"]
        }
        if not multi_seed_trials:
            print(f"  S{s_idx+1}: no overlapping trials")
            continue

        # Collect per-seed per-ensemble predictions on overlapping test trials
        # seed_preds[seed][e_idx][trial] = 1-D np array of predictions
        seed_preds = {seed: {} for seed in SEEDS}

        for seed in SEEDS:
            trials_needed = [t for t, sl in multi_seed_trials.items() if seed in sl]
            if not trials_needed:
                continue

            test_trials_this_seed = set(splits[seed].get(sess, []))
            train_trials          = [t for t in all_trials if t not in test_trials_this_seed]

            if not train_trials:
                continue

            X_train = np.concatenate(
                [sd["data"][t] for t in train_trials if t in sd["data"]], axis=0
            ).astype(np.float32)
            Y_train = np.concatenate(
                [sd["labels"][t] for t in train_trials if t in sd["data"]], axis=0
            ).astype(np.float32)

            if len(X_train) == 0:
                continue

            # Embed training data once per (seed, session, ensemble)
            for e_idx in range(n_ensembles):
                if mean_r2_mat[s_idx, e_idx] < R2_THRESHOLD:
                    continue

                mpath = os.path.join(models_root, f"seed{seed}",
                                     f"session_{s_idx:02d}_neuron_{e_idx:02d}.pt")
                if not os.path.exists(mpath):
                    continue

                encoder, _, _ = load_encoder(mpath, device=str(device))
                encoder.eval()

                # Embed training set and fit Ridge
                Z_train = _embed(encoder, X_train)
                if np.isnan(Z_train).any():
                    del encoder; torch.cuda.empty_cache(); continue
                ridge   = Ridge(alpha=RIDGE_ALPHA)
                ridge.fit(Z_train, Y_train[:, e_idx])

                # Predict each overlapping test trial
                seed_preds[seed][e_idx] = {}
                for trial in trials_needed:
                    x = sd["data"][trial]
                    if len(x) < MIN_TIMEPOINTS:
                        continue
                    Z_te = _embed(encoder, x.astype(np.float32))
                    seed_preds[seed][e_idx][trial] = ridge.predict(Z_te)

                del encoder
                torch.cuda.empty_cache()

        # Compute pairwise Pearson r per ensemble
        for e_idx in range(n_ensembles):
            if mean_r2_mat[s_idx, e_idx] < R2_THRESHOLD:
                continue

            pairwise_rs = []
            for trial, seed_list in multi_seed_trials.items():
                trial_preds = []
                for seed in seed_list:
                    p = seed_preds[seed].get(e_idx, {}).get(trial)
                    if p is not None and len(p) >= MIN_TIMEPOINTS:
                        trial_preds.append(p)

                if len(trial_preds) < 2:
                    continue

                for pi, pj in itertools.combinations(trial_preds, 2):
                    n = min(len(pi), len(pj))
                    pi, pj = pi[:n], pj[:n]
                    if np.std(pi) < 1e-8 or np.std(pj) < 1e-8:
                        continue
                    r, _ = pearsonr(pi, pj)
                    pairwise_rs.append(r)

            if pairwise_rs:
                cons_mat[s_idx, e_idx] = float(np.mean(pairwise_rs))

        n_valid = sum(1 for sl in multi_seed_trials.values() if len(sl) >= 2)
        print(f"  S{s_idx+1} ({sess}): {n_valid} overlapping trials", flush=True)

    return cons_mat


def _embed(encoder, X_np):
    wins = build_windows(X_np)
    with torch.no_grad():
        z = encoder(torch.tensor(wins, dtype=torch.float32, device=device))
        if z.dim() == 3:
            z = z.squeeze(-1)
    return z.cpu().numpy()


# ─── LOAD OR COMPUTE MLP CONSISTENCY ─────────────────────────────────────────
mlp_cons_path = os.path.join(mlp_dir, "consistency_matrix_mlp.npy")
if os.path.exists(mlp_cons_path):
    cons_mlp = np.load(mlp_cons_path)
    print(f"Loaded MLP consistency from {mlp_cons_path}")
else:
    raise FileNotFoundError(
        f"MLP consistency not found at {mlp_cons_path}. "
        "Run eval_embedding_consistency.py first."
    )

# ─── COMPUTE TempConv CONSISTENCY ───────────────────────────────────────────────
cc_cons_path = os.path.join(out_dir, "consistency_matrix_cebra_contrast.npy")
cp_cons_path = os.path.join(out_dir, "consistency_matrix_cebra_pred.npy")

if os.path.exists(cc_cons_path):
    cons_cc = np.load(cc_cons_path)
    print(f"Loaded TempConv-Cont consistency from {cc_cons_path}")
else:
    print("\n=== Computing TempConv-Cont consistency ===")
    cons_cc = compute_cebra_consistency("cebra", mean_r2_cc)
    np.save(cc_cons_path, cons_cc)
    print(f"Saved {cc_cons_path}")

if os.path.exists(cp_cons_path):
    cons_cp = np.load(cp_cons_path)
    print(f"Loaded TempConv-Pred consistency from {cp_cons_path}")
else:
    print("\n=== Computing TempConv-Pred consistency ===")
    cons_cp = compute_cebra_consistency("cebra_pred", mean_r2_cp)
    np.save(cp_cons_path, cons_cp)
    print(f"Saved {cp_cons_path}")

# ─── SUMMARY STATS ────────────────────────────────────────────────────────────
models_list = [
    ("MLP",             cons_mlp, 'steelblue'),
    ("TempConv-Cont",  cons_cc,  'darkorange'),
    ("TempConv-Pred",      cons_cp,  'forestgreen'),
]

print("\n── Consistency summary (Pearson r, test-only, ≥2-seed overlap) ──")
flat_data, flat_labels, flat_colors = [], [], []
for label, cons, color in models_list:
    vals = cons.ravel()
    vals = vals[np.isfinite(vals)]
    frac_high = (vals >= 0.9).mean() if len(vals) else float('nan')
    print(f"  {label:20s}: n={len(vals):4d}  "
          f"median={np.median(vals) if len(vals) else float('nan'):.3f}  "
          f"mean={np.mean(vals) if len(vals) else float('nan'):.3f}  "
          f"≥0.9: {frac_high:.1%}")
    flat_data.append(vals)
    flat_labels.append(label)
    flat_colors.append(color)

# ─── VIOLIN + JITTERED POINTS PLOT ───────────────────────────────────────────
rng = np.random.default_rng(0)

fig, ax = plt.subplots(figsize=(9.5, 4.5))
apply_style(fig, ax)

parts = ax.violinplot(flat_data, positions=[0, 1, 2],
                      showmedians=True, showextrema=True, widths=0.6)
for pc, color in zip(parts['bodies'], flat_colors):
    pc.set_facecolor(color)
    pc.set_alpha(0.40)
for part in ['cmedians', 'cmins', 'cmaxes', 'cbars']:
    parts[part].set_color('k')
    parts[part].set_linewidth(0.8)

for i, (vals, label, color) in enumerate(zip(flat_data, flat_labels, flat_colors)):
    if not len(vals):
        continue
    jitter = rng.uniform(-0.08, 0.08, size=len(vals))
    ax.scatter(i + jitter, vals, s=14, color=color, alpha=0.7,
               linewidths=0, zorder=3)
    med = float(np.median(vals))
    ax.text(i, med + 0.025, f'{med:.2f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
    ax.text(i, -0.07, f'n={len(vals)}', ha='center', va='top', fontsize=9,
            color='gray', transform=ax.get_xaxis_transform())

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(flat_labels, fontsize=11)
ax.set_ylabel('Consistency (mean pairwise Pearson r, test-only)', fontsize=10)
ax.set_ylim(-0.05, 1.15)
ax.axhline(0.9, color='k', linestyle=':', lw=0.8, label='0.9 threshold')
ax.legend(fontsize=9, frameon=False, loc='upper right')
ax.set_title('Cross-seed consistency — overlapping test trials only\n'
             '(dots = individual (session, ensemble) pairs; median bold)',
             fontsize=11)
ax.spines[['top', 'right']].set_visible(False)

savefig_manifest(fig, "embedding_consistency_violin.png", [out_dir])
print(f"\nSaved {os.path.join(out_dir, 'embedding_consistency_violin.png')}")
