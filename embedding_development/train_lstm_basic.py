"""
train_lstm.py
Per-session LSTM with a past-context window.
Predicts z-scored firing rate of each neuron from a sliding window
of behavioural features ending at the current timestep (causal).

Outputs (all under ./outputs/lstm/):
  models/session_XX_neuron_YY.pt   — saved state dicts
  mse_matrix.npy                   — (num_sessions, num_neurons)
  r2_matrix.npy                    — (num_sessions, num_neurons)
  pearson_r2_matrix.npy            — (num_sessions, num_neurons)  r² = corr²
  predictions/session_XX_neuron_YY_test.npz
      arrays: actual, predicted, trial_ids
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm

from models import NeuronLSTM

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--window",        type=int,   default=10,
                    help="Past context window size in timesteps (default 10)")
parser.add_argument("--hidden_size",   type=int,   default=64)
parser.add_argument("--num_layers",    type=int,   default=1)
parser.add_argument("--num_epochs",    type=int,   default=100)
parser.add_argument("--lr",            type=float, default=1e-3)
parser.add_argument("--num_neurons",   type=int,   default=77)
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--train_split",   type=float, default=0.8)
parser.add_argument("--base",          type=str,   default="./outputs/glm_input_data")
parser.add_argument("--out_dir",       type=str,   default="./outputs/lstm")
parser.add_argument("--batch_size",    type=int,   default=512,
                    help="Mini-batch size for training (default 512)")
parser.add_argument("--split_path",    type=str,   default="test_indices_by_session.npy",
                    help="Reuse existing train/test split if found")
args = parser.parse_args()

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Output dirs ───────────────────────────────────────────────────────────────
models_dir = os.path.join(args.out_dir, "models")
preds_dir  = os.path.join(args.out_dir, "predictions")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(preds_dir,  exist_ok=True)

# ── Feature columns ───────────────────────────────────────────────────────────
action_enc_cols = [
    "frame_raw_500msMedian",
    "frame_YawPitch_abs_vel_sum_500msMedian",
    "upcoming_choice",
    "reward_window",
    "frame_raw_abs_acc_500msMedian",
    "frame_YawPitch_abs_acc_sum_500msMedian",
    "forward_vs_rotation_corr",
    "head_angle",
    "head_angle_vel",
    "movement_energy_smooth5",
    "lick_detected",
]
input_size = len(action_enc_cols)

# ── Data loading ──────────────────────────────────────────────────────────────
print("Loading data …")
base = args.base

# ── Load data ──────────────────────────────────────────────────────────────
base = "./outputs/glm_input_data/"

beh_vals = np.load(os.path.join(base, "behavior_glm_input.npy"), allow_pickle=True)
beh_idx  = np.load(os.path.join(base, "behavior_glm_input_index.npy"), allow_pickle=True)
beh_cols = np.load(os.path.join(base, "behavior_glm_input_columns.npy"), allow_pickle=True)

spk_vals = np.load(os.path.join(base, "fr_full.npy"))
spk_idx  = np.load(os.path.join(base, "fr_full_index.npy"), allow_pickle=True)
spk_cols = np.load(os.path.join(base, "fr_full_columns.npy"), allow_pickle=True)

beh_index = pd.Index(beh_idx)

spk_index = pd.Index(spk_idx)

behavior_glm_loaded = pd.DataFrame(beh_vals, index=beh_index, columns=beh_cols)
spikes_loaded       = pd.DataFrame(spk_vals, index=spk_index, columns=spk_cols)

print("behavior_glm_loaded shape", behavior_glm_loaded.shape)
print("spikes_loaded shape",       spikes_loaded.shape)

# ── Preprocessing (identical to MLP notebook) ──────────────────────────────
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

behavior_glm_loaded = behavior_glm_loaded.drop("track_zone", axis=1)

session_ids = behavior_glm_loaded.index.map(lambda x: x[0]).unique()
print(f"{len(session_ids)} sessions")

# ── Build per-session dataset ─────────────────────────────────────────────────
print("Building session datasets …")
session_dataset_singles = {}

for session_id in session_ids:
    session_mask   = behavior_glm_loaded.index.map(lambda x: x[0]) == session_id
    beh            = behavior_glm_loaded[session_mask].copy()
    spikes_session = spikes_loaded[session_mask].copy()

    for col in action_enc_cols:
        if col in beh.columns:
            beh[col]  = beh[col].astype(float)
            mean_val  = beh[col].mean()
            std_val   = beh[col].std() + 1e-8
            beh[col]  = (beh[col] - mean_val) / std_val

    spikes_session = spikes_session.astype(float)
    label_stds     = []
    for col in spikes_session.columns:
        mean_val = spikes_session[col].mean()
        std_val  = spikes_session[col].std() + 1e-8
        label_stds.append(std_val)
        spikes_session[col] = (spikes_session[col] - mean_val) / std_val

    data_by_trial, labels_by_trial = {}, {}
    for trial_id in beh["trial_id"].unique():
        trial_mask              = beh["trial_id"] == trial_id
        data_by_trial[trial_id]   = beh.loc[trial_mask, action_enc_cols].values.astype(np.float32)
        labels_by_trial[trial_id] = spikes_session.loc[trial_mask].values.astype(np.float32)

    session_dataset_singles[session_id] = {
        "data":       data_by_trial,
        "labels":     labels_by_trial,
        "label_stds": label_stds,
    }

# ── Train/test split ──────────────────────────────────────────────────────────
if os.path.exists(args.split_path):
    test_indices_by_session = np.load(
        args.split_path, allow_pickle=True).item()
    print(f"Reusing existing split from {args.split_path}")
else:
    test_indices_by_session = {}
    for session_id in session_ids:
        indices   = list(session_dataset_singles[session_id]["data"].keys())
        np.random.shuffle(indices)
        split_idx = int(len(indices) * args.train_split)
        test_indices_by_session[session_id] = indices[split_idx:]
    np.save(args.split_path, test_indices_by_session, allow_pickle=True)
    print(f"Saved new split to {args.split_path}")

# ── Window helper ─────────────────────────────────────────────────────────────
def make_windows(data_dict, labels_dict, trial_ids, window):
    """
    For each trial, slide a window of size `window` over timesteps.
    Pads the beginning of each trial with zeros so every timestep
    has a prediction (causal — no future leakage).

    Returns:
        X : (N, window, input_size)  float32
        y : (N,)                     float32  — label at final timestep
    """
    X_list, y_list = [], []
    for tid in trial_ids:
        if tid not in data_dict:
            continue
        data   = data_dict[tid]    # (T, input_size)
        labels = labels_dict[tid]  # (T, num_neurons)
        T      = len(data)
        padded = np.zeros((window - 1 + T, data.shape[1]), dtype=np.float32)
        padded[window - 1:] = data
        for t in range(T):
            X_list.append(padded[t: t + window])   # (window, input_size)
            y_list.append(labels[t])                # (num_neurons,)
    if not X_list:
        return None, None
    return np.stack(X_list), np.stack(y_list)      # (N, window, F), (N, num_neurons)


# ── Mask for silent neurons ───────────────────────────────────────────────────
num_sessions = len(session_ids)
num_neurons  = args.num_neurons
mask         = np.zeros((num_sessions, num_neurons), dtype=bool)
for i, session_id in enumerate(session_ids):
    for j, std in enumerate(session_dataset_singles[session_id]["label_stds"]):
        if std == 1e-8:
            mask[i, j] = True

# ── Result matrices ───────────────────────────────────────────────────────────
mse_matrix       = np.full((num_sessions, num_neurons), np.nan)
r2_matrix        = np.full((num_sessions, num_neurons), np.nan)
pearson_r2_matrix = np.full((num_sessions, num_neurons), np.nan)

# ═════════════════════════════════════════════════════════════════════════════
# LOOP 1 — TRAIN AND SAVE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"TRAINING  window={args.window}  hidden={args.hidden_size}"
      f"  layers={args.num_layers}  epochs={args.num_epochs}")
print("=" * 60)

for test_idx, test_session in enumerate(session_ids):
    sd          = session_dataset_singles[test_session]
    test_idx_   = test_indices_by_session[test_session]
    all_indices = list(sd["data"].keys())
    train_idx   = [i for i in all_indices if i not in set(test_idx_)]

    X_train, y_train = make_windows(
        sd["data"], sd["labels"], train_idx, args.window)
    if X_train is None:
        print(f"  Session {test_idx+1}: no train data, skipping")
        continue

    # Keep data on CPU; move batches to device inside the loop
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    print(f"Session {test_idx+1}/{num_sessions} — "
          f"train windows: {X_train_t.shape}")

    for neuron_idx in range(num_neurons):
        if mask[test_idx, neuron_idx]:
            continue

        dataset   = TensorDataset(X_train_t, y_train_t[:, neuron_idx])
        loader    = DataLoader(dataset, batch_size=args.batch_size,
                               shuffle=True, pin_memory=(device.type == "cuda"))

        model     = NeuronLSTM(input_size, args.hidden_size,
                               args.num_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()

        epoch_bar = tqdm(range(args.num_epochs),
                         desc=f"  S{test_idx+1:02d} N{neuron_idx+1:02d}",
                         leave=False, unit="ep")
        for epoch in epoch_bar:
            model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_bar.set_postfix(loss=f"{epoch_loss / len(loader):.4f}")

        model_path = os.path.join(
            models_dir,
            f"session_{test_idx:02d}_neuron_{neuron_idx:02d}.pt")
        torch.save(model.state_dict(), model_path)

        del model, optimizer, criterion, dataset, loader
        torch.cuda.empty_cache()

    del X_train_t, y_train_t
    torch.cuda.empty_cache()
    print(f"  Session {test_idx+1} saved.")

# ═════════════════════════════════════════════════════════════════════════════
# LOOP 2 — EVALUATE AND SAVE PREDICTIONS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EVALUATING")
print("=" * 60)

for test_idx, test_session in enumerate(session_ids):
    sd        = session_dataset_singles[test_session]
    test_idx_ = test_indices_by_session[test_session]

    X_test, y_test = make_windows(
        sd["data"], sd["labels"], test_idx_, args.window)
    if X_test is None:
        print(f"  Session {test_idx+1}: no test data, skipping")
        continue

    X_test_t = torch.tensor(X_test, dtype=torch.float32)  # keep on CPU
    print(f"Session {test_idx+1}/{num_sessions} — "
          f"test windows: {X_test_t.shape}")

    for neuron_idx in range(num_neurons):
        if mask[test_idx, neuron_idx]:
            continue

        model_path = os.path.join(
            models_dir,
            f"session_{test_idx:02d}_neuron_{neuron_idx:02d}.pt")
        if not os.path.exists(model_path):
            continue

        model = NeuronLSTM(input_size, args.hidden_size,
                           args.num_layers).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        eval_loader = DataLoader(TensorDataset(X_test_t),
                                 batch_size=args.batch_size * 4,
                                 pin_memory=(device.type == "cuda"))
        preds_np_list = []
        with torch.no_grad():
            for (xb,) in eval_loader:
                preds_np_list.append(model(xb.to(device)).cpu().numpy())
        preds_np = np.concatenate(preds_np_list)

        actual = y_test[:, neuron_idx]

        mse = mean_squared_error(actual, preds_np)
        r2  = r2_score(actual, preds_np)
        r, _ = pearsonr(actual, preds_np)
        pr2  = r ** 2

        mse_matrix[test_idx, neuron_idx]        = mse
        r2_matrix[test_idx, neuron_idx]         = r2
        pearson_r2_matrix[test_idx, neuron_idx] = pr2

        # Save raw predictions for later inspection
        pred_path = os.path.join(
            preds_dir,
            f"session_{test_idx:02d}_neuron_{neuron_idx:02d}_test.npz")
        np.savez_compressed(pred_path,
                            actual=actual,
                            predicted=preds_np,
                            trial_ids=np.array(list(test_idx_)))

        print(f"  S{test_idx+1:02d} N{neuron_idx+1:02d}  "
              f"MSE={mse:.4f}  R²={r2:.4f}  PearsonR²={pr2:.4f}  "
              f"Std={sd['label_stds'][neuron_idx]:.4f}")

        del model, preds_np, eval_loader
        torch.cuda.empty_cache()

    del X_test_t
    torch.cuda.empty_cache()

# ── Save matrices ─────────────────────────────────────────────────────────────
np.save(os.path.join(args.out_dir, "mse_matrix.npy"),        mse_matrix)
np.save(os.path.join(args.out_dir, "r2_matrix.npy"),         r2_matrix)
np.save(os.path.join(args.out_dir, "pearson_r2_matrix.npy"), pearson_r2_matrix)
np.save(os.path.join(args.out_dir, "mask.npy"),              mask)

# ── Summary ───────────────────────────────────────────────────────────────────
valid       = ~mask
valid_pr2   = pearson_r2_matrix[valid]
valid_r2    = r2_matrix[valid]

print("\n" + "=" * 60)
print("SUMMARY")
print(f"  Window size          : {args.window}")
print(f"  Valid neuron-sessions: {valid.sum()}")
print(f"  Sklearn R²  — mean   : {np.nanmean(valid_r2):.4f}  "
      f"median: {np.nanmedian(valid_r2):.4f}")
print(f"  Pearson R²  — mean   : {np.nanmean(valid_pr2):.4f}  "
      f"median: {np.nanmedian(valid_pr2):.4f}")
print(f"  Fraction Pearson R²>0: "
      f"{np.mean(valid_pr2 > 0)*100:.1f}%")
print(f"\nOutputs saved to: {args.out_dir}")
print("=" * 60)