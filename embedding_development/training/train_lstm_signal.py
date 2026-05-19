"""
train_lstm_signal.py
Per-session multi-output LSTM trained on neurons with behavioral signal.

Reads an r² matrix (sessions x neurons) and an r² threshold to select which
neurons to predict per session.  Trains ONE multi-output LSTM per session.
The action embedding is the LSTM output at the final timestep (pre-FC),
extractable via model.embed(x) — shape (batch, hidden_size).

The output neuron×session metric matrix will be incomplete by design — cells
are NaN wherever a neuron was not selected for that session.

Outputs (all under --out_dir):
  models/session_XX.pt               — saved state dicts (one per session)
  r2_matrix.npy                      — (num_sessions, num_neurons)  NaN = not selected
  mse_matrix.npy                     — same shape
  pearson_r2_matrix.npy              — same shape
  mask.npy                           — bool (num_sessions, num_neurons)  True = silent
  active_neurons_by_session.npy      — dict {session_id: [neuron_indices]}
  predictions/session_XX_test.npz    — arrays: actual (N, k), predicted (N, k),
                                       neuron_indices, trial_ids
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

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from models import NeuronLSTM

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--r2_path",       type=str,   required=True,
                    help="Path to an existing r² .npy file (num_sessions x num_neurons)")
parser.add_argument("--r2_threshold",  type=float, default=0.05,
                    help="Min r² for a neuron to be included per session (default 0.05)")
parser.add_argument("--window",        type=int,   default=10)
parser.add_argument("--hidden_size_base",   type=int,   default=64)
parser.add_argument("--num_layers",    type=int,   default=2)
parser.add_argument("--num_epochs",    type=int,   default=100)
parser.add_argument("--lr",            type=float, default=1e-3)
parser.add_argument("--num_neurons",   type=int,   default=77)
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--train_split",   type=float, default=0.8)
parser.add_argument("--base",          type=str,   default="./outputs/glm_input_data")
parser.add_argument("--out_dir",       type=str,   default="./outputs/lstm_signal")
parser.add_argument("--batch_size",    type=int,   default=512)
parser.add_argument("--split_path",    type=str,   default="test_indices_by_session.npy")
parser.add_argument("--dropout",       type=float, default=0.0)
parser.add_argument("--weight_decay",  type=float, default=0.0)
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
    'frame_raw_500msMedian',                  # forward_velocity
    'frame_raw_abs_acc_500msMedian',          # forward_acceleration
    'frame_YawPitch_abs_vel_sum_500msMedian', # off_rotation_velocity
    'frame_YawPitch_abs_acc_sum_500msMedian', # off_rotation_acceleration
    'head_angle_vel',
    'head_angle',
    'movement_energy_smooth5',                # movement_energy
    'upcoming_choice',
    'reward_window',
    'lick_detected',
]
state_enc_cols = [
    'cue_visible',
    'track_zone_int',
]
input_size = len(action_enc_cols) + len(state_enc_cols)

# ── Load r² reference matrix ──────────────────────────────────────────────────
print(f"Loading r² matrix from {args.r2_path} …")
r2_ref = np.load(args.r2_path)  # (num_sessions, num_neurons)
assert r2_ref.ndim == 2, "r² file must be 2-D (sessions × neurons)"
print(f"  r² matrix shape: {r2_ref.shape}")

# ── Data loading ──────────────────────────────────────────────────────────────
print("Loading data …")
base = args.base

beh_vals = np.load(os.path.join(base, "behavior_glm_input.npy"), allow_pickle=True)
beh_idx  = np.load(os.path.join(base, "behavior_glm_input_index.npy"), allow_pickle=True)
beh_cols = np.load(os.path.join(base, "behavior_glm_input_columns.npy"), allow_pickle=True)

spk_vals = np.load(os.path.join(base, "fr_full.npy"))
spk_idx  = np.load(os.path.join(base, "fr_full_index.npy"), allow_pickle=True)
spk_cols = np.load(os.path.join(base, "fr_full_columns.npy"), allow_pickle=True)

behavior_glm_loaded = pd.DataFrame(beh_vals, index=pd.Index(beh_idx), columns=beh_cols)
spikes_loaded       = pd.DataFrame(spk_vals, index=pd.Index(spk_idx), columns=spk_cols)

print("behavior_glm_loaded shape", behavior_glm_loaded.shape)
print("spikes_loaded shape",       spikes_loaded.shape)

# ── Preprocessing ─────────────────────────────────────────────────────────────
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
num_sessions = len(session_ids)
num_neurons  = args.num_neurons
print(f"{num_sessions} sessions, {num_neurons} neurons")

# ── Build per-session dataset ─────────────────────────────────────────────────
print("Building session datasets …")
session_dataset_singles = {}

for session_id in session_ids:
    session_mask   = behavior_glm_loaded.index.map(lambda x: x[0]) == session_id
    beh            = behavior_glm_loaded[session_mask].copy()
    spikes_session = spikes_loaded[session_mask].copy()

    for col in action_enc_cols + state_enc_cols:
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
        trial_mask                = beh["trial_id"] == trial_id
        data_by_trial[trial_id]   = beh.loc[trial_mask, action_enc_cols + state_enc_cols].values.astype(np.float32)
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

# ── Silence mask (neuron never fires in this session) ─────────────────────────
silence_mask = np.zeros((num_sessions, num_neurons), dtype=bool)
for i, session_id in enumerate(session_ids):
    for j, std in enumerate(session_dataset_singles[session_id]["label_stds"]):
        if std <= 1e-8:
            silence_mask[i, j] = True

# ── Active neurons per session (r² > threshold AND not silent) ────────────────
# r2_ref rows must align with session_ids order — assumed same ordering as data
active_neurons_by_session = {}
for i, session_id in enumerate(session_ids):
    r2_row    = r2_ref[i] if i < r2_ref.shape[0] else np.zeros(num_neurons)
    r2_row    = np.nan_to_num(r2_row, nan=0.0)
    active    = np.where((r2_row >= args.r2_threshold) & (~silence_mask[i]))[0]
    active_neurons_by_session[session_id] = active.tolist()
    print(f"  Session {i+1:02d} ({session_id}): {len(active)} neurons with r²≥{args.r2_threshold}")

np.save(os.path.join(args.out_dir, "active_neurons_by_session.npy"),
        active_neurons_by_session, allow_pickle=True)

# ── Window helper ─────────────────────────────────────────────────────────────
def make_windows(data_dict, labels_dict, trial_ids, window, neuron_indices):
    """
    Returns:
        X : (N, window, input_size)  float32
        y : (N, k)                   float32  — k = len(neuron_indices)
    """
    X_list, y_list = [], []
    for tid in trial_ids:
        if tid not in data_dict:
            continue
        data   = data_dict[tid]                         # (T, input_size)
        labels = labels_dict[tid][:, neuron_indices]    # (T, k)
        T      = len(data)
        padded = np.zeros((window - 1 + T, data.shape[1]), dtype=np.float32)
        padded[window - 1:] = data
        for t in range(T):
            X_list.append(padded[t: t + window])
            y_list.append(labels[t])
    if not X_list:
        return None, None
    return np.stack(X_list), np.stack(y_list)  # (N, window, F), (N, k)


# ── Result matrices ───────────────────────────────────────────────────────────
mse_path  = os.path.join(args.out_dir, "mse_matrix.npy")
r2_path   = os.path.join(args.out_dir, "r2_matrix.npy")
pr2_path  = os.path.join(args.out_dir, "pearson_r2_matrix.npy")

if os.path.exists(mse_path) and os.path.exists(r2_path) and os.path.exists(pr2_path):
    mse_matrix        = np.load(mse_path)
    r2_matrix_out     = np.load(r2_path)
    pearson_r2_matrix = np.load(pr2_path)
    print("Loaded existing result matrices — filling missing entries only.")
else:
    mse_matrix        = np.full((num_sessions, num_neurons), np.nan)
    r2_matrix_out     = np.full((num_sessions, num_neurons), np.nan)
    pearson_r2_matrix = np.full((num_sessions, num_neurons), np.nan)

# ═════════════════════════════════════════════════════════════════════════════
# LOOP 1 — TRAIN
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"TRAINING  window={args.window}  hidden base={args.hidden_size_base}"
      f"  layers={args.num_layers}  epochs={args.num_epochs}"
      f"  r2_thresh={args.r2_threshold}")
print("=" * 60)

for sess_idx, session_id in enumerate(session_ids):
    active = active_neurons_by_session[session_id]
    if len(active) == 0:
        print(f"  Session {sess_idx+1}: no active neurons, skipping")
        continue

    model_path = os.path.join(models_dir, f"session_{sess_idx:02d}.pt")
    if os.path.exists(model_path):
        print(f"  Session {sess_idx+1}: model exists, skipping training")
        continue

    sd        = session_dataset_singles[session_id]
    test_idx_ = test_indices_by_session[session_id]
    all_ids   = list(sd["data"].keys())
    train_ids = [i for i in all_ids if i not in set(test_idx_)]

    X_train, y_train = make_windows(
        sd["data"], sd["labels"], train_ids, args.window, np.array(active))
    if X_train is None:
        print(f"  Session {sess_idx+1}: no training windows, skipping")
        continue

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)  # (N, k)
    print(f"Session {sess_idx+1}/{num_sessions} ({session_id})"
          f" — neurons: {len(active)}  train windows: {X_t.shape[0]}")

    output_size = len(active)
    hidden_size = round(args.hidden_size_base * (output_size ** 0.5))
    model     = NeuronLSTM(input_size, hidden_size,
                           args.num_layers, output_size=output_size,
                           dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=  args.weight_decay)
    criterion = nn.MSELoss()

    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, pin_memory=(device.type == "cuda"))

    epoch_bar = tqdm(range(args.num_epochs),
                     desc=f"  S{sess_idx+1:02d}",
                     leave=True, unit="ep")
    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)           # (batch, k)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_bar.set_postfix(loss=f"{epoch_loss / len(loader):.4f}")

    torch.save(model.state_dict(), model_path)
    del model, optimizer, criterion, dataset, loader, X_t, y_t
    torch.cuda.empty_cache()
    print(f"  Session {sess_idx+1} saved → {model_path}")

# ═════════════════════════════════════════════════════════════════════════════
# LOOP 2 — EVALUATE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EVALUATING")
print("=" * 60)

for sess_idx, session_id in enumerate(session_ids):
    active = active_neurons_by_session[session_id]
    if len(active) == 0:
        continue

    model_path = os.path.join(models_dir, f"session_{sess_idx:02d}.pt")
    pred_path  = os.path.join(preds_dir,  f"session_{sess_idx:02d}_test.npz")

    if not os.path.exists(model_path):
        continue
    if os.path.exists(pred_path):
        # Re-fill metric matrices from saved predictions
        saved = np.load(pred_path)
        actual_mat = saved["actual"]      # (N, k)
        pred_mat   = saved["predicted"]   # (N, k)
        for ki, nidx in enumerate(active):
            if np.all(actual_mat[:, ki] == actual_mat[0, ki]):
                continue  # constant — skip
            mse_matrix[sess_idx, nidx]        = mean_squared_error(actual_mat[:, ki], pred_mat[:, ki])
            r2_matrix_out[sess_idx, nidx]     = r2_score(actual_mat[:, ki], pred_mat[:, ki])
            r, _                              = pearsonr(actual_mat[:, ki], pred_mat[:, ki])
            pearson_r2_matrix[sess_idx, nidx] = r ** 2
        print(f"  Session {sess_idx+1}: reloaded metrics from {pred_path}")
        continue

    sd        = session_dataset_singles[session_id]
    test_idx_ = test_indices_by_session[session_id]

    X_test, y_test = make_windows(
        sd["data"], sd["labels"], test_idx_, args.window, np.array(active))
    if X_test is None:
        print(f"  Session {sess_idx+1}: no test windows, skipping")
        continue

    output_size = len(active)
    hidden_size = round(args.hidden_size_base * (output_size ** 0.5))
    model = NeuronLSTM(input_size, hidden_size,
                       args.num_layers, output_size=output_size,
                       dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    X_test_t   = torch.tensor(X_test, dtype=torch.float32)
    eval_loader = DataLoader(TensorDataset(X_test_t),
                             batch_size=args.batch_size * 4,
                             pin_memory=(device.type == "cuda"))
    preds_list = []
    with torch.no_grad():
        for (xb,) in eval_loader:
            out = model(xb.to(device))
            # ensure 2-D even when output_size == 1
            if out.dim() == 1:
                out = out.unsqueeze(1)
            preds_list.append(out.cpu().numpy())
    preds_mat = np.concatenate(preds_list, axis=0)   # (N, k)

    # y_test shape: (N, k) always (make_windows stacks neuron_indices slice)
    actual_mat = y_test  # (N, k)

    for ki, nidx in enumerate(active):
        actual = actual_mat[:, ki]
        pred   = preds_mat[:, ki]
        if np.all(actual == actual[0]):
            continue
        mse = mean_squared_error(actual, pred)
        r2  = r2_score(actual, pred)
        r, _ = pearsonr(actual, pred)
        mse_matrix[sess_idx, nidx]        = mse
        r2_matrix_out[sess_idx, nidx]     = r2
        pearson_r2_matrix[sess_idx, nidx] = r ** 2
        print(f"  S{sess_idx+1:02d} N{nidx+1:02d}  "
              f"MSE={mse:.4f}  R²={r2:.4f}  PearsonR²={r**2:.4f}")

    np.savez_compressed(pred_path,
                        actual=actual_mat,
                        predicted=preds_mat,
                        neuron_indices=np.array(active),
                        trial_ids=np.array(list(test_idx_)))

    del model, X_test_t, eval_loader, preds_mat
    torch.cuda.empty_cache()

# ── Save matrices ─────────────────────────────────────────────────────────────
np.save(os.path.join(args.out_dir, "mse_matrix.npy"),        mse_matrix)
np.save(os.path.join(args.out_dir, "r2_matrix.npy"),         r2_matrix_out)
np.save(os.path.join(args.out_dir, "pearson_r2_matrix.npy"), pearson_r2_matrix)
np.save(os.path.join(args.out_dir, "mask.npy"),              silence_mask)

# ── Summary ───────────────────────────────────────────────────────────────────
selected_mask = ~np.isnan(r2_matrix_out)
valid_r2      = r2_matrix_out[selected_mask]
valid_pr2     = pearson_r2_matrix[selected_mask]

print("\n" + "=" * 60)
print("SUMMARY")
print(f"  r² threshold             : {args.r2_threshold}")
print(f"  Window size              : {args.window}")
print(f"  Selected neuron-sessions : {selected_mask.sum()}")
print(f"  Sklearn R²  — mean       : {np.nanmean(valid_r2):.4f}  "
      f"median: {np.nanmedian(valid_r2):.4f}")
print(f"  Pearson R²  — mean       : {np.nanmean(valid_pr2):.4f}  "
      f"median: {np.nanmedian(valid_pr2):.4f}")
print(f"  Fraction Pearson R²>0    : {np.mean(valid_pr2 > 0)*100:.1f}%")
print(f"\nOutputs saved to: {args.out_dir}")
print("=" * 60)
