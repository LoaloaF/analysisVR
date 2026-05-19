import os
import argparse
import numpy as np
import pandas as pd
import torch
import random

import torch.nn as nn

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from models import MLP

parser = argparse.ArgumentParser()
parser.add_argument("--split_path",    type=str,  default="test_indices_by_session.npy",
                    help="Path to train/test split file (generate with generate_split.py)")
parser.add_argument("--seed",          type=int,  default=42)
parser.add_argument("--use_ensembles", action="store_true", default=False,
                    help="Predict ensemble activity instead of single-unit spikes")
parser.add_argument("--models_dir",       type=str,  default=None,
                    help="Directory to save model checkpoints (default: auto from --use_ensembles)")
parser.add_argument("--n_epochs",         type=int,  default=100,
                    help="Training epochs per neuron")
parser.add_argument("--learning_rate",    type=float, default=1e-3)
parser.add_argument("--hidden_size",      type=int,  default=64)
parser.add_argument("--num_hidden_layers", type=int, default=2)
args = parser.parse_args()

USE_ENSEMBLES = args.use_ensembles

# ── Load data ──────────────────────────────────────────────────────────────
base = "./outputs/glm_input_data/"

beh_vals = np.load(os.path.join(base, "behavior_glm_input.npy"), allow_pickle=True)
beh_idx  = np.load(os.path.join(base, "behavior_glm_input_index.npy"), allow_pickle=True)
beh_cols = np.load(os.path.join(base, "behavior_glm_input_columns.npy"), allow_pickle=True)

spk_vals = np.load(os.path.join(base, "fr_full.npy"))
spk_idx  = np.load(os.path.join(base, "fr_full_index.npy"), allow_pickle=True)
spk_cols = np.load(os.path.join(base, "fr_full_columns.npy"), allow_pickle=True)

ensembles_values = np.load(os.path.join(base, "ensembles.npy"))

try:
    beh_index = pd.MultiIndex.from_tuples(beh_idx, names=behavior_glm_input.index.names)
except Exception:
    beh_index = pd.Index(beh_idx)

try:
    spk_index = pd.MultiIndex.from_tuples(spk_idx, names=fr_first7.index.names)
except Exception:
    spk_index = pd.Index(spk_idx)

behavior_glm_loaded = pd.DataFrame(beh_vals, index=beh_index, columns=beh_cols)
spikes_loaded       = pd.DataFrame(spk_vals, index=spk_index, columns=spk_cols)

print("behavior_glm_loaded shape", behavior_glm_loaded.shape)
print("spikes_loaded shape",       spikes_loaded.shape)

# ── Preprocessing ──────────────────────────────────────────────────────────
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

session_ids = behavior_glm_loaded.index.map(lambda x: x[0]).unique()
print(f"{len(session_ids)} sessions")

# ── Feature columns ────────────────────────────────────────────────────────
non_categorical_cols = [
    'frame_raw_500msMedian',                  # forward_velocity
    'frame_raw_abs_acc_500msMedian',          # forward_acceleration
    'frame_YawPitch_abs_vel_sum_500msMedian', # off_rotation_velocity
    'frame_YawPitch_abs_acc_sum_500msMedian', # off_rotation_acceleration
    'head_angle_vel',
    'head_angle',
    'frame_position',
]

categorical_variables = [
    'cue_visible',
    'upcoming_choice',
    'reward_window',
    'lick_detected',
]

zone_onehot_cols = []
for col in categorical_variables:
    col_vals = sorted(behavior_glm_loaded[col].dropna().unique().astype(int))
    for v in col_vals:
        behavior_glm_loaded[f'{col}_{v}'] = (behavior_glm_loaded[col] == v).astype(float)
    if len(col_vals) == 2:
        zone_onehot_cols.append(f'{col}_{col_vals[-1]}')
    else:
        zone_onehot_cols.extend([f'{col}_{v}' for v in col_vals])

# ── Build session dataset ──────────────────────────────────────────────────
session_dataset_singles = {}

for session_id in session_ids:
    session_mask   = behavior_glm_loaded.index.map(lambda x: x[0]) == session_id
    beh            = behavior_glm_loaded[session_mask]
    spikes_session = spikes_loaded[session_mask]

    for col in non_categorical_cols:
        if col in beh.columns:
            beh.loc[:, col] = beh[col].astype(float)
            mean_val = beh[col].mean()
            std_val  = beh[col].std() + 1e-8
            if std_val != 0:
                beh.loc[:, col] = (beh[col] - mean_val) / std_val

    spikes_session = spikes_session.astype(float)
    if USE_ENSEMBLES:
        spikes_values = spikes_session.values
        new_spikes_session = pd.DataFrame(index=spikes_session.index)
        for i in range(ensembles_values.shape[1]):
            ensemble_col = spikes_values @ ensembles_values[:, i]
            new_spikes_session[f'ensemble_{i}'] = ensemble_col
        spikes_session = new_spikes_session

    label_stds = []
    for col in spikes_session.columns:
        mean_val = spikes_session[col].mean()
        std_val  = spikes_session[col].std() + 1e-8
        label_stds.append(std_val)
        if std_val != 0:
            spikes_session[col] = (spikes_session[col] - mean_val) / std_val

    data_by_trial, labels_by_trial = {}, {}
    for trial_id in beh["trial_id"].unique():
        trial_mask                = beh["trial_id"] == trial_id
        data_by_trial[trial_id]   = beh.loc[trial_mask, non_categorical_cols + zone_onehot_cols].values.astype(np.float16)
        labels_by_trial[trial_id] = spikes_session.loc[trial_mask].values.astype(np.float16)

    session_dataset_singles[session_id] = {
        "data":       data_by_trial,
        "labels":     labels_by_trial,
        "label_stds": label_stds,
    }

print("Dataset built.")

# ── Train/test split ────────────────────────────────────────────────────────
if os.path.exists(args.split_path):
    test_indices_by_session = np.load(args.split_path, allow_pickle=True).item()
    print(f"Loaded existing split from {args.split_path}")
else:
    raise FileNotFoundError(
        f"Split file not found: {args.split_path}\n"
        "Generate one first with: python generate_split.py --out <path>"
    )

# ── Config ──────────────────────────────────────────────────────────────────
_default_models_dir = "./models/mlps/ensembles" if USE_ENSEMBLES else "./models/mlps/spikes"
models_dir = args.models_dir if args.models_dir is not None else _default_models_dir
os.makedirs(models_dir, exist_ok=True)

input_size   = len(non_categorical_cols) + len(zone_onehot_cols)
output_size  = 1
num_sessions = len(session_ids)
num_neurons  = ensembles_values.shape[1] if USE_ENSEMBLES else spikes_loaded.shape[1]

print(f"input_size={input_size}, num_sessions={num_sessions}, num_neurons={num_neurons}")
print(f"hidden_size={args.hidden_size}, num_hidden_layers={args.num_hidden_layers}, n_epochs={args.n_epochs}, learning_rate={args.learning_rate}")

# ── Train and save ──────────────────────────────────────────────────────────
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)

for test_idx, test_session in enumerate(session_ids):
    session_data   = session_dataset_singles[test_session]["data"]
    session_labels = session_dataset_singles[test_session]["labels"]

    test_idx_   = test_indices_by_session[test_session]
    all_indices = list(session_data.keys())
    train_idx   = np.setdiff1d(all_indices, test_idx_)

    train_data      = torch.tensor(
        np.concatenate([session_data[idx]   for idx in train_idx], axis=0),
        dtype=torch.float32).to(device)
    train_labels_np = np.concatenate(
        [session_labels[idx] for idx in train_idx], axis=0)

    print(f"Session {test_idx+1}/{num_sessions} — train shape: {train_data.shape}")

    for neuron_idx in range(num_neurons):
        model_path = os.path.join(models_dir, f"session_{test_idx:02d}_neuron_{neuron_idx:02d}.pt")
        if os.path.exists(model_path):
            print(f"  Skip:  session {test_idx+1}, neuron {neuron_idx+1}/{num_neurons} (already saved)")
            continue

        train_label = torch.tensor(
            train_labels_np[:, neuron_idx], dtype=torch.float32).to(device)

        model_seed = args.seed * 10_000_000 + test_idx * 10_000 + neuron_idx
        torch.manual_seed(model_seed)
        np.random.seed(model_seed % (2**32))
        model     = MLP(input_size, args.hidden_size, args.num_hidden_layers, output_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(args.n_epochs):
            model.train()
            optimizer.zero_grad()
            _, outputs = model(train_data)
            loss = criterion(outputs.squeeze(), train_label)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), model_path)
        print(f"  Saved: session {test_idx+1}, neuron {neuron_idx+1}/{num_neurons}")

        del model, optimizer, criterion, train_label
        torch.cuda.empty_cache()

    del train_data, train_labels_np
    torch.cuda.empty_cache()

print("Training complete.")
