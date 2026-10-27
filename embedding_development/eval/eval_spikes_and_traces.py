#!/usr/bin/env python3
"""
eval_spikes_and_traces.py

Two jobs:
  A. Single-unit (spikes) R² evaluation for MLP and Linear — r2_bar + r2_heatmap
  B. Cross-model trial trace comparison: actual vs predicted on the same
     session × ensemble × trial for Linear, MLP, TempConv-Cont, TempConv-Pred
"""
import os, sys, pickle, random
import numpy as np
import numpy.ma as ma
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from models import MLP, LinearModel
from load_encoder import build_windows, load_encoder
from figure_style import (
    FIG, DPI, FONT, LINE, MARKER,
    MODEL_COLORS, FEATURE_NAMES_SHORT,
    apply_style, add_footnote, add_panel_label, savefig_manifest,
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SEEDS         = [42, 43, 44, 45, 46]
RIDGE_ALPHA   = 1.0
N_TRACE_PAIRS = 3       # how many session×ensemble panels to plot
R2_MIN_TRACE  = 0.05    # all 4 models must clear this to be a "canonical" pair

TOP_BAR   = 40   # top N neurons shown in bar plot
TOP_LABEL = 8    # how many bars get bold labels above them

base_dir    = os.path.dirname(os.path.abspath(__file__))
root        = os.path.join(base_dir, "..")
data_dir    = os.path.join(root, "outputs", "glm_input_data")
splits_dir  = os.path.join(root, "splits")
out_spk_mlp = os.path.join(root, "outputs", "mlps",   "spikes_multiseed")
out_spk_lin = os.path.join(root, "outputs", "linear", "spikes_multiseed")
out_traces  = os.path.join(root, "outputs", "trial_traces_comparison")
for d in [out_spk_mlp, out_spk_lin, out_traces]:
    os.makedirs(d, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

# ─── LOAD RAW DATA ────────────────────────────────────────────────────────────
beh_vals = np.load(os.path.join(data_dir, "behavior_glm_input.npy"),         allow_pickle=True)
beh_idx  = np.load(os.path.join(data_dir, "behavior_glm_input_index.npy"),   allow_pickle=True)
beh_cols = np.load(os.path.join(data_dir, "behavior_glm_input_columns.npy"), allow_pickle=True)
spk_vals = np.load(os.path.join(data_dir, "fr_full.npy"))
spk_idx  = np.load(os.path.join(data_dir, "fr_full_index.npy"),              allow_pickle=True)
ens_vals = np.load(os.path.join(data_dir, "ensembles.npy"))

behavior_glm_loaded = pd.DataFrame(beh_vals, index=pd.Index(beh_idx), columns=beh_cols)
spikes_loaded       = pd.DataFrame(spk_vals, index=pd.Index(spk_idx))

spikes_loaded.index = pd.MultiIndex.from_tuples(
    spikes_loaded.index.map(lambda t: (t[0], t[1] // 40000 - 1))
)
spikes_unique_trials = set(idx[0] for idx in spikes_loaded.index)
behavior_glm_loaded  = behavior_glm_loaded[
    behavior_glm_loaded.index.map(lambda t: t[0] in spikes_unique_trials)
]
non_nan_rows        = behavior_glm_loaded.index[~behavior_glm_loaded.isna().any(axis=1)]
behavior_glm_loaded = behavior_glm_loaded.loc[non_nan_rows]
spikes_loaded       = spikes_loaded.loc[non_nan_rows]
session_ids         = behavior_glm_loaded.index.map(lambda x: x[0]).unique()
print(f"{len(session_ids)} sessions")

# ─── FEATURES ─────────────────────────────────────────────────────────────────
non_categorical_cols = [
    'frame_raw_500msMedian',
    'frame_raw_abs_acc_500msMedian',
    'frame_YawPitch_abs_vel_sum_500msMedian',
    'frame_YawPitch_abs_acc_sum_500msMedian',
    'head_angle_vel',
    'head_angle',
    'frame_position',
]
categorical_variables = ['cue_visible', 'upcoming_choice', 'reward_window', 'lick_detected']

zone_onehot_cols = []
for col in categorical_variables:
    col_vals = sorted(behavior_glm_loaded[col].dropna().unique().astype(int))
    for v in col_vals:
        behavior_glm_loaded[f'{col}_{v}'] = (behavior_glm_loaded[col] == v).astype(float)
    if len(col_vals) == 2:
        zone_onehot_cols.append(f'{col}_{col_vals[-1]}')
    else:
        zone_onehot_cols.extend([f'{col}_{v}' for v in col_vals])

all_feat_cols = non_categorical_cols + zone_onehot_cols
n_feats       = len(all_feat_cols)
print(f"{n_feats} features")

# ─── BUILD SESSION DATASETS ───────────────────────────────────────────────────
cache_ens = os.path.join(root, "outputs", "session_dataset_ensembles.pkl")
cache_spk = os.path.join(root, "outputs", "session_dataset_spikes.pkl")

def build_session_dataset(use_ensembles, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            ds = pickle.load(f)
        print(f"Loaded cache: {cache_path}")
        return ds

    ds = {}
    for session_id in session_ids:
        mask    = behavior_glm_loaded.index.map(lambda x: x[0]) == session_id
        beh     = behavior_glm_loaded[mask].copy()
        spk_ses = spikes_loaded[mask].astype(float)

        for col in non_categorical_cols:
            if col in beh.columns:
                mu, sd = beh[col].mean(), beh[col].std() + 1e-8
                beh[col] = (beh[col] - mu) / sd

        if use_ensembles:
            raw = spk_ses.values
            new = pd.DataFrame(index=spk_ses.index)
            for i in range(ens_vals.shape[1]):
                new[f'ensemble_{i}'] = raw @ ens_vals[:, i]
            spk_ses = new

        label_stds = []
        for col in spk_ses.columns:
            mu, sd = spk_ses[col].mean(), spk_ses[col].std() + 1e-8
            label_stds.append(sd)
            spk_ses[col] = (spk_ses[col] - mu) / sd

        data_by_trial, labels_by_trial = {}, {}
        for trial_id in beh["trial_id"].unique():
            tmask = beh["trial_id"] == trial_id
            data_by_trial[trial_id]   = beh.loc[tmask, all_feat_cols].values.astype(np.float32)
            labels_by_trial[trial_id] = spk_ses.loc[tmask].values.astype(np.float32)

        ds[session_id] = {"data": data_by_trial, "labels": labels_by_trial, "label_stds": label_stds}

    with open(cache_path, 'wb') as f:
        pickle.dump(ds, f)
    print(f"Built and cached: {cache_path}")
    return ds

ds_ens = build_session_dataset(use_ensembles=True,  cache_path=cache_ens)
ds_spk = build_session_dataset(use_ensembles=False, cache_path=cache_spk)
session_ids = pd.Index(list(ds_ens.keys()))
num_sessions = len(session_ids)
num_ensembles = len(next(iter(ds_ens.values()))['label_stds'])
num_neurons   = len(next(iter(ds_spk.values()))['label_stds'])
print(f"{num_sessions} sessions, {num_ensembles} ensembles, {num_neurons} neurons")

# ─── HELPER: evaluate one model family ────────────────────────────────────────
def _load_mlp(path):
    sd = torch.load(path, map_location=device)
    # infer hidden_size from first layer weight
    h = list(sd.values())[0].shape[0]
    m = MLP(n_feats, h, 2, 1).to(device)
    m.load_state_dict(sd)
    m.eval()
    return m

def _load_linear(path):
    sd = torch.load(path, map_location=device)
    m  = LinearModel(n_feats, 1).to(device)
    m.load_state_dict(sd)
    m.eval()
    return m

def eval_multiseed(ds, models_root, load_fn, model_prefix, num_units, out_dir):
    """Returns all_r2 array shape (n_seeds, n_sessions, n_units)."""
    npy_path = os.path.join(out_dir, "all_r2.npy")
    if os.path.exists(npy_path):
        print(f"Loading cached R²: {npy_path}")
        return np.load(npy_path)

    all_r2 = np.full((len(SEEDS), num_sessions, num_units), np.nan)
    for si, seed in enumerate(SEEDS):
        split = np.load(os.path.join(splits_dir, f"split_seed{seed}.npy"), allow_pickle=True).item()
        mdir  = os.path.join(models_root, f"seed{seed}")
        for ti, sess in enumerate(session_ids):
            test_idx = split[sess]
            Xte = torch.tensor(
                np.concatenate([ds[sess]["data"][i]   for i in test_idx], 0),
                dtype=torch.float32, device=device)
            Yte = np.concatenate([ds[sess]["labels"][i] for i in test_idx], 0)
            for ni in range(num_units):
                mpath = os.path.join(mdir, f"session_{ti:02d}_neuron_{ni:02d}.pt")
                if not os.path.exists(mpath):
                    continue
                model = load_fn(mpath)
                with torch.no_grad():
                    preds = model(Xte)[1].squeeze().cpu().numpy() if hasattr(model, 'fc') else \
                            model(Xte)[1].squeeze().cpu().numpy()
                actual = Yte[:, ni]
                if np.var(actual) > 1e-8:
                    all_r2[si, ti, ni] = r2_score(actual, preds)
                del model
            del Xte, Yte
            torch.cuda.empty_cache()
        print(f"  seed {seed} done.")
    np.save(npy_path, all_r2)
    np.save(os.path.join(out_dir, "all_mse.npy"), np.zeros_like(all_r2))  # placeholder
    print(f"Saved {npy_path}")
    return all_r2

# ─── PART A: SPIKES R² FOR MLP AND LINEAR ────────────────────────────────────
def plot_r2_summary(all_r2, out_dir, label, prefix, ylim_bar=None):
    median_r2   = np.nanmedian(all_r2, axis=0)   # (sessions, units)
    neuron_mean = np.nanmean(median_r2, axis=0)
    num_units   = all_r2.shape[2]
    order_full  = np.argsort(neuron_mean)          # ascending, all units

    # bar — show only top TOP_BAR neurons, mako_r colormap, bold top labels
    order_top  = order_full[-TOP_BAR:]             # best TOP_BAR (ascending)
    means_top  = np.clip(neuron_mean[order_top], 0, None)
    labels_top = [f"{prefix}{i+1:02d}" for i in order_top]
    pal = sns.color_palette("mako_r", as_cmap=True)(np.linspace(0.15, 0.85, TOP_BAR))

    fig, ax = plt.subplots(figsize=(16, 5))
    apply_style(fig, ax)
    ax.bar(range(TOP_BAR), means_top, width=0.7, color=pal, zorder=3, linewidth=0)
    ax.axhline(0, color='firebrick', ls='--', lw=1)
    for pos in range(TOP_BAR - TOP_LABEL, TOP_BAR):
        ax.text(pos, means_top[pos] + 0.002, labels_top[pos],
                ha='center', va='bottom', fontsize=9, fontweight='bold', rotation=90)
    ax.set_xticks(range(0, TOP_BAR, 3))
    ax.set_xticklabels(labels_top[::3], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel("Neuron (sorted by mean R²)")
    ax.set_ylabel("Mean R² (z-scored FR)")
    ax.set_title(f"{label} — Single-Unit R² Across {len(SEEDS)} Seeds × {num_sessions} Sessions")
    ax.spines[['top', 'right']].set_visible(False)
    if ylim_bar is not None:
        ax.set_ylim(0, ylim_bar)
    savefig_manifest(fig, "r2_bar.png", [out_dir])

    # heatmap (top 30 neurons)
    NO_DATA_COLOR = '#bbbbbb'
    top30    = np.argsort(neuron_mean)[::-1][:30]
    nan_mask = np.isnan(median_r2[:, top30]).T   # (30, sessions)
    r2_clip  = np.clip(median_r2, 0, 1)
    r2_show  = r2_clip[:, top30].T.astype(float)
    r2_show[nan_mask] = np.nan
    cmap_r2 = plt.cm.Blues.copy(); cmap_r2.set_bad(color=NO_DATA_COLOR)
    cell_h, cell_w = 0.25, 0.40
    fig, ax = plt.subplots(figsize=(max(10, num_sessions * cell_w), max(4, 30 * cell_h)))
    apply_style(fig, ax)
    sns.heatmap(r2_show, cmap=cmap_r2, vmin=0, vmax=1, ax=ax,
                xticklabels=[f"S{i+1}" for i in range(num_sessions)],
                yticklabels=[f"{prefix}{n+1:02d}" for n in top30])
    for (row, col) in zip(*np.where(nan_mask)):
        ax.add_patch(plt.Rectangle([col, row], 1, 1, fill=True, facecolor=NO_DATA_COLOR,
                                   hatch='////', edgecolor='#888888', lw=0.5, zorder=2))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_title(f"{label} — Median R² per Session×Neuron (Top 30 Neurons)", fontsize=9, pad=8)
    savefig_manifest(fig, "r2_heatmap.png", [out_dir])
    print(f"{label} grand mean R² = {np.nanmean(neuron_mean):.4f}")

print("\n=== MLP SPIKES ===")
mlp_spk_r2 = eval_multiseed(ds_spk,
    models_root=os.path.join(root, "models", "mlps", "spikes"),
    load_fn=_load_mlp, model_prefix="U",
    num_units=num_neurons, out_dir=out_spk_mlp)

print("\n=== LINEAR SPIKES ===")
lin_spk_r2 = eval_multiseed(ds_spk,
    models_root=os.path.join(root, "models", "linear", "spikes"),
    load_fn=_load_linear, model_prefix="U",
    num_units=num_neurons, out_dir=out_spk_lin)

# Shared y-axis scale so MLP and Linear bars are directly comparable
_mlp_unit_means = np.nanmean(np.nanmedian(mlp_spk_r2, axis=0), axis=0)
_lin_unit_means = np.nanmean(np.nanmedian(lin_spk_r2, axis=0), axis=0)
plot_r2_summary(mlp_spk_r2, out_spk_mlp, "MLP — Single Units", "U", ylim_bar=0.2)
plot_r2_summary(lin_spk_r2, out_spk_lin, "Linear — Single Units", "U", ylim_bar=0.2)

# ─── PART B: CROSS-MODEL TRIAL TRACES ─────────────────────────────────────────
print("\n=== CROSS-MODEL TRIAL TRACES ===")

# Load ensemble R² medians for all 4 models
mlp_r2  = np.nanmedian(np.load(os.path.join(root,"outputs","mlps","ensembles_multiseed","all_r2.npy")),  axis=0)
lin_r2  = np.nanmedian(np.load(os.path.join(root,"outputs","linear","ensembles_multiseed","all_r2.npy")), axis=0)
ccon_r2 = np.nanmedian(np.load(os.path.join(root,"outputs","cebra_eval","ensembles","all_r2.npy")),      axis=0)
cprd_r2 = np.nanmedian(np.load(os.path.join(root,"outputs","cebra_pred_eval","ensembles","all_r2.npy")), axis=0)

# Find pairs where ALL 4 models exceed threshold
min_r2 = np.minimum.reduce([mlp_r2, lin_r2, ccon_r2, cprd_r2])
mean_r2_all4 = (mlp_r2 + lin_r2 + ccon_r2 + cprd_r2) / 4.0
candidates = np.argwhere(min_r2 >= R2_MIN_TRACE)
if len(candidates) == 0:
    print(f"No pairs with all 4 models R²≥{R2_MIN_TRACE} — lowering threshold to 0.02")
    candidates = np.argwhere(min_r2 >= 0.02)

# Sort by mean R² across all 4, pick top N
scores  = mean_r2_all4[candidates[:, 0], candidates[:, 1]]
top_idx = candidates[np.argsort(scores)[::-1][:N_TRACE_PAIRS]]
print(f"Top {len(top_idx)} canonical pairs:")
for s_idx, n_idx in top_idx:
    print(f"  S{s_idx+1:02d} E{n_idx:02d}: lin={lin_r2[s_idx,n_idx]:.3f} mlp={mlp_r2[s_idx,n_idx]:.3f} "
          f"ccon={ccon_r2[s_idx,n_idx]:.3f} cprd={cprd_r2[s_idx,n_idx]:.3f}")

# Prediction helpers
def predict_mlp(sess_idx, sess_id, n_idx, trial_id, split):
    mpath = os.path.join(root, "models", "mlps", "ensembles", "seed42",
                         f"session_{sess_idx:02d}_neuron_{n_idx:02d}.pt")
    if not os.path.exists(mpath):
        return None
    model = _load_mlp(mpath)
    X = torch.tensor(ds_ens[sess_id]["data"][trial_id], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(X)[1].squeeze().cpu().numpy()
    del model; torch.cuda.empty_cache()
    return pred

def predict_linear(sess_idx, sess_id, n_idx, trial_id, split):
    mpath = os.path.join(root, "models", "linear", "ensembles", "seed42",
                         f"session_{sess_idx:02d}_neuron_{n_idx:02d}.pt")
    if not os.path.exists(mpath):
        return None
    model = _load_linear(mpath)
    X = torch.tensor(ds_ens[sess_id]["data"][trial_id], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(X)[1].squeeze().cpu().numpy()
    del model; torch.cuda.empty_cache()
    return pred

def _embed_cebra(encoder, X_np):
    wins = build_windows(X_np)
    with torch.no_grad():
        z = encoder(torch.tensor(wins, dtype=torch.float32, device=device))
        if z.dim() == 3: z = z.squeeze(-1)
    return z.cpu().numpy()

def predict_cebra(arm, sess_idx, sess_id, n_idx, trial_id, split):
    mpath = os.path.join(root, "models", arm, "ensembles", "seed42",
                         f"session_{sess_idx:02d}_neuron_{n_idx:02d}.pt")
    if not os.path.exists(mpath):
        return None
    encoder, _, _ = load_encoder(mpath, device=device)
    encoder.eval()

    test_trials  = split[sess_id]
    train_trials = [t for t in ds_ens[sess_id]["data"] if t not in test_trials]
    if len(train_trials) == 0:
        return None

    Xtr = np.concatenate([ds_ens[sess_id]["data"][t]   for t in train_trials], 0).astype(np.float32)
    Ytr = np.concatenate([ds_ens[sess_id]["labels"][t] for t in train_trials], 0)[:, n_idx]
    emb_tr    = _embed_cebra(encoder, Xtr)
    emb_trial = _embed_cebra(encoder, ds_ens[sess_id]["data"][trial_id].astype(np.float32))

    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(emb_tr, Ytr)
    pred = ridge.predict(emb_trial)
    del encoder; torch.cuda.empty_cache()
    return pred

# Plot
split42 = np.load(os.path.join(splits_dir, "split_seed42.npy"), allow_pickle=True).item()
COLORS = {'Linear': '#8B6552', 'MLP': '#2CA02C', 'TempConv-Cont': '#1F77B4', 'TempConv-Pred': '#FF7F0E'}

for pair_num, (s_idx, n_idx) in enumerate(top_idx):
    sess_id  = session_ids[s_idx]
    test_trials = split42[sess_id]
    valid_trials = [t for t in test_trials if t in ds_ens[sess_id]["data"]]
    if not valid_trials:
        continue
    # pick the most dynamic test trial (highest variance in actual signal)
    trial_id = max(valid_trials,
                   key=lambda t: ds_ens[sess_id]["labels"][t][:, n_idx].var())

    actual = ds_ens[sess_id]["labels"][trial_id][:, n_idx]
    t_axis = np.arange(len(actual)) * 40  # ms (40ms bins)

    preds = {
        'Linear':     predict_linear(s_idx, sess_id, n_idx, trial_id, split42),
        'MLP':        predict_mlp(   s_idx, sess_id, n_idx, trial_id, split42),
        'TempConv-Cont':  predict_cebra('cebra',      s_idx, sess_id, n_idx, trial_id, split42),
        'TempConv-Pred': predict_cebra('cebra_pred', s_idx, sess_id, n_idx, trial_id, split42),
    }

    r2s = {'Linear': lin_r2[s_idx,n_idx], 'MLP': mlp_r2[s_idx,n_idx],
           'TempConv-Cont': ccon_r2[s_idx,n_idx], 'TempConv-Pred': cprd_r2[s_idx,n_idx]}

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    apply_style(fig, axes)
    for ax, (name, pred) in zip(axes, preds.items()):
        ax.plot(t_axis, actual, color='#555555', lw=1.5, alpha=0.8, label='Actual')
        if pred is not None:
            ax.plot(t_axis, pred, color=COLORS[name], lw=1.5, alpha=0.9,
                    label=f'Predicted  R²={r2s[name]:.3f}')
        ax.set_ylabel('z-scored FR', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold', color=COLORS[name])
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.25)
        ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xlabel('Time (ms)', fontsize=11)
    fig.suptitle(f'Session {sess_id} — Ensemble E{n_idx+1:02d} — Trial {trial_id}\n'
                 f'Actual vs Predicted (z-scored firing rate)',
                 fontsize=12, fontweight='bold')
    fname = f"traces_S{s_idx+1:02d}_E{n_idx+1:02d}.png"
    savefig_manifest(fig, fname, [out_traces])
    print(f"  Saved {fname}")

# Also save a 2-column overlay version (actual + all 4 preds on one axes per pair)
for pair_num, (s_idx, n_idx) in enumerate(top_idx):
    sess_id     = session_ids[s_idx]
    test_trials = split42[sess_id]
    valid_trials = [t for t in test_trials if t in ds_ens[sess_id]["data"]]
    if not valid_trials:
        continue
    trial_id = max(valid_trials,
                   key=lambda t: ds_ens[sess_id]["labels"][t][:, n_idx].var())
    actual   = ds_ens[sess_id]["labels"][trial_id][:, n_idx]
    t_axis   = np.arange(len(actual)) * 40

    preds = {
        'Linear':     predict_linear(s_idx, sess_id, n_idx, trial_id, split42),
        'MLP':        predict_mlp(   s_idx, sess_id, n_idx, trial_id, split42),
        'TempConv-Cont':  predict_cebra('cebra',      s_idx, sess_id, n_idx, trial_id, split42),
        'TempConv-Pred': predict_cebra('cebra_pred', s_idx, sess_id, n_idx, trial_id, split42),
    }

    fig, ax = plt.subplots(figsize=(14, 4))
    apply_style(fig, ax)
    ax.plot(t_axis, actual, color='#222222', lw=2, alpha=0.85, label='Actual', zorder=5)
    for name, pred in preds.items():
        if pred is not None:
            r2 = {'Linear': lin_r2, 'MLP': mlp_r2,
                  'TempConv-Cont': ccon_r2, 'TempConv-Pred': cprd_r2}[name][s_idx, n_idx]
            ax.plot(t_axis, pred, color=COLORS[name], lw=1.3, alpha=0.8,
                    label=f'{name} (R²={r2:.3f})')
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('z-scored FR', fontsize=11)
    ax.set_title(f'Session {sess_id} — Ensemble E{n_idx+1:02d} — Trial {trial_id} — All models overlay',
                 fontsize=12)
    ax.legend(fontsize=9, ncol=5)
    ax.grid(True, alpha=0.25)
    ax.spines[['top','right']].set_visible(False)
    fname = f"overlay_S{s_idx+1:02d}_E{n_idx+1:02d}.png"
    savefig_manifest(fig, fname, [out_traces])
    print(f"  Saved {fname}")

print("\nDone.")
