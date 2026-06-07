

# ============================================================
# LSTM with Session-Adaptive Head  —  drop-in replacement
# ============================================================
# Architecture:
#   - Shared LSTM trunk: trained on ALL sessions except the test session
#     (learns session-invariant action → neural mapping)
#   - Per-session affine head: scale + bias per neuron, per session
#     (absorbs firing-rate drift without rotating the representation)
#
# Protocol per held-out session:
#   1. Train trunk + all train-session heads on temporal splits
#      (first 80% of timepoints per session — no window shuffle)
#   2. Freeze trunk; fit test-session head on first `adapt_frac`
#      of the test session's timepoints  (default 20%)
#   3. Evaluate on the remaining held-out timepoints
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import random

import torch.nn as nn
from sklearn.metrics import mean_squared_error


from tqdm import tqdm



# reload the data we dumped in cell 31

base = "./outputs/glm_input_data/"

beh_vals = np.load(os.path.join(base, "behavior_glm_input.npy"), allow_pickle=True)
beh_idx = np.load(os.path.join(base, "behavior_glm_input_index.npy"),
                  allow_pickle=True)
beh_cols = np.load(os.path.join(base, "behavior_glm_input_columns.npy"),
                   allow_pickle=True)

spk_vals = np.load(os.path.join(base, "fr_full.npy"))
spk_idx = np.load(os.path.join(base, "fr_full_index.npy"),
                  allow_pickle=True)
spk_cols = np.load(os.path.join(base, "fr_full_columns.npy"),
                   allow_pickle=True)

# fr_vals = np.load(os.path.join(base, "fr_full.npy"))
# fr_idx = np.load(os.path.join(base, "fr_full_index.npy"),
#                  allow_pickle=True)
# fr_cols = np.load(os.path.join(base, "fr_full_columns.npy"),
#                   allow_pickle=True)

# reconstruct indexes (they were saved as arrays of tuples)
try:
    beh_index = pd.MultiIndex.from_tuples(beh_idx,
                                          names=behavior_glm_input.index.names)
except Exception:
    beh_index = pd.Index(beh_idx)

try:
    spk_index = pd.MultiIndex.from_tuples(spk_idx,
                                          names=fr_first7.index.names)
except Exception:
    spk_index = pd.Index(spk_idx)

# try:
#     fr_index = pd.MultiIndex.from_tuples(fr_idx, names=fr.index.names)
# except Exception:
#     fr_index = pd.Index(fr_idx)

# build dataframes
behavior_glm_loaded = pd.DataFrame(beh_vals,
                                   index=beh_index,
                                   columns=beh_cols)
spikes_loaded = pd.DataFrame(spk_vals,
                             index=spk_index,
                             columns=spk_cols)
# fr_full_loaded = pd.DataFrame(fr_vals,
#                               index=fr_index,
#                               columns=fr_cols)

print("behavior_glm_loaded shape", behavior_glm_loaded.shape)
print("spikes_loaded shape", spikes_loaded.shape)

spikes_loaded.index = pd.MultiIndex.from_tuples(
    spikes_loaded.index.map(lambda t: (t[0], t[1] // 40000 - 1))
)


spikes_unique_trials = set()
for idx in spikes_loaded.index:
    spikes_unique_trials.add(idx[0])

behavior_glm_loaded = behavior_glm_loaded[behavior_glm_loaded.index.map(lambda t: t[0] in spikes_unique_trials)]

non_nan_rows = behavior_glm_loaded.index[~behavior_glm_loaded.isna().any(axis=1)]
behavior_glm_loaded = behavior_glm_loaded.loc[non_nan_rows]
spikes_loaded = spikes_loaded.loc[non_nan_rows]

unique_trials = set()

for idx in behavior_glm_loaded.index:
    unique_trials.add(idx[0])

# Select only numeric columns
behavior_glm_loaded = behavior_glm_loaded.drop("track_zone", axis=1)

session_ids = behavior_glm_loaded.index.map(lambda x: x[0]).unique()
session_ids

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
    'track_zone',
]

session_dataset_singles = {}

for session_id in session_ids:
    session_mask = behavior_glm_loaded.index.map(lambda x: x[0]) == session_id
    beh = behavior_glm_loaded[
        session_mask
    ]

    spikes_session = spikes_loaded[
        session_mask
    ]

    # Convert specified columns in beh to z-scores
    for col in action_enc_cols + state_enc_cols:
        if col in beh.columns:
            beh.loc[:, col] = beh[col].astype(float)
            mean_val = beh[col].mean()
            std_val = beh[col].std() + 1e-8
            if std_val != 0:
                beh.loc[:, col] = (beh[col] - mean_val) / std_val

    spikes_session = spikes_session.astype(float)
    label_stds = []
    for col in spikes_session.columns:
        mean_val = spikes_session[col].mean()
        std_val = spikes_session[col].std() + 1e-8
        label_stds.append(std_val)
        if std_val != 0:
            spikes_session[col] = (spikes_session[col] - mean_val) / std_val

    session_dataset_singles[session_id] = {"data": beh[action_enc_cols + state_enc_cols].values.astype(np.float16), "labels": spikes_session.values.astype(np.float16), "label_stds": label_stds}






torch.cuda.empty_cache()

# ── Hyperparameters ──────────────────────────────────────────
window_size   = 50
adapt_frac    = 1      # fraction of test session used to fit the affine head
hidden_size   = 64
num_epochs    = 10
adapt_epochs  = 30        # epochs to fit the session head (trunk frozen)
learning_rate = 0.001
adapt_lr      = 0.005     # can be higher — only 2*num_neurons params
batch_size    = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

np.random.seed(42); torch.manual_seed(42); random.seed(42)

num_neurons  = session_dataset_singles[session_ids[0]]["labels"].shape[1]
num_sessions = len(session_ids)
input_size   = len(action_enc_cols) + len(state_enc_cols)

# ── Model definitions ────────────────────────────────────────

class LSTMTrunk(nn.Module):
    """Shared trunk — identical to the original LSTM but stops before the fc head."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        added_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0); added_batch = True
        h0 = x.new_zeros(1, x.size(0), self.hidden_size)
        c0 = x.new_zeros(1, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))          # (batch, seq, hidden)
        if added_batch:
            out = out.squeeze(0)
        return out

class SessionHead(nn.Module):
    """
    Small MLP decoder + per-neuron affine correction.
    The affine (scale, bias) absorbs session-level gain/rate drift.
    """
    def __init__(self, hidden_size, num_neurons):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_neurons),
        )
        # Per-neuron affine — initialised to identity (scale=1, bias=0)
        self.scale = nn.Parameter(torch.ones(num_neurons))
        self.bias  = nn.Parameter(torch.zeros(num_neurons))

    def forward(self, hidden):
        out = self.fc(hidden)                      # (batch, seq, neurons)
        return out * self.scale + self.bias        # broadcast over batch & seq

# ── Helper: lazy windowed dataset ────────────────────────────

class WindowDataset(torch.utils.data.Dataset):
    """Returns windows on-the-fly — no full materialisation in RAM."""
    def __init__(self, data, labels, window_size, indices):
        self.data        = data          # float16 numpy, stays on CPU
        self.labels      = labels
        self.window_size = window_size
        self.indices     = indices       # which start-positions to expose

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        x = torch.tensor(self.data  [j:j + self.window_size], dtype=torch.float32)
        y = torch.tensor(self.labels[j:j + self.window_size], dtype=torch.float32)
        return x, y

def temporal_split(n_windows, train_frac):
    split = int(n_windows * train_frac)
    return np.arange(split), np.arange(split, n_windows)

# ── Main loop ────────────────────────────────────────────────

mse_matrix_lstm = np.zeros((num_sessions, num_neurons))

for test_idx, test_session in enumerate(session_ids):

    train_sessions = [s for s in session_ids if s != test_session]

    # ── Phase 1: build training data from all non-test sessions ──
    trunk = LSTMTrunk(input_size, hidden_size).to(device)

    # One head per training session (absorbs each session's gain)
    train_heads = nn.ModuleList([
        SessionHead(hidden_size, num_neurons).to(device)
        for _ in train_sessions
    ])

    optimizer = torch.optim.Adam(
        list(trunk.parameters()) + list(train_heads.parameters()),
        lr=learning_rate
    )
    criterion = nn.MSELoss()

    # Build lazy DataLoaders — no full window materialisation
    train_loaders = []
    for s in train_sessions:
        data   = session_dataset_singles[s]["data"]
        labels = session_dataset_singles[s]["labels"]
        n_windows = len(data) - window_size + 1
        tr_idx, _ = temporal_split(n_windows, 0.8)
        ds = WindowDataset(data, labels, window_size, tr_idx)
        train_loaders.append(
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda'))
        )

    # Training loop over sessions jointly
    for epoch in range(num_epochs):
        trunk.train()
        for head in train_heads: head.train()

        epoch_loss = 0.0
        total_batches = 0
        for sess_i, loader in tqdm(enumerate(train_loaders)):
            for bx, by in loader:
                bx = bx.to(device)
                by = by.to(device)

                optimizer.zero_grad()
                hidden = trunk(bx)
                preds  = train_heads[sess_i](hidden)
                loss   = criterion(preds, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                total_batches += 1

        if (epoch + 1) % 1 == 0:
            print(f"  [Session {test_idx+1}/{num_sessions}] "
                  f"Epoch {epoch+1}/{num_epochs}  loss={(epoch_loss / total_batches):.4f}")

    # ── Phase 2: freeze trunk, fit test-session affine head ──────
    for p in trunk.parameters():
        p.requires_grad_(False)

    test_head = SessionHead(hidden_size, num_neurons).to(device)
    adapt_opt = torch.optim.Adam(test_head.parameters(), lr=adapt_lr)

    test_data_raw   = session_dataset_singles[test_session]["data"]
    test_labels_raw = session_dataset_singles[test_session]["labels"]
    n_windows = len(test_data_raw) - window_size + 1
    adapt_end = int(n_windows * adapt_frac)

    adapt_ds = WindowDataset(test_data_raw, test_labels_raw, window_size, np.arange(adapt_end))
    adapt_loader = torch.utils.data.DataLoader(
        adapt_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda')
    )

    trunk.eval()
    test_head.train()
    for epoch in range(adapt_epochs):
        for bx, by in adapt_loader:
            bx = bx.to(device)
            by = by.to(device)
            adapt_opt.zero_grad()
            with torch.no_grad():
                hidden = trunk(bx)
            preds = test_head(hidden)
            loss  = criterion(preds, by)
            loss.backward()
            adapt_opt.step()

    # ── Phase 3: evaluate on held-out portion of test session ────
    eval_ds = WindowDataset(test_data_raw, test_labels_raw, window_size, np.arange(adapt_end))
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == 'cuda')
    )

    trunk.eval(); test_head.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in eval_loader:
            bx = bx.to(device)
            hidden = trunk(bx)
            all_preds.append(test_head(hidden).cpu().numpy())
            all_labels.append(by.numpy())
    predictions = np.concatenate(all_preds,  axis=0)   # (N, seq, neurons)
    eval_labels = np.concatenate(all_labels, axis=0)

    session_label_stds = session_dataset_singles[test_session]["label_stds"]
    for neuron_idx in range(num_neurons):
        mse = mean_squared_error(
            eval_labels[:, :, neuron_idx],
            predictions[:, :, neuron_idx]
        )
        mse_matrix_lstm[test_idx, neuron_idx] = mse
        print(f"Session {test_idx+1}/{num_sessions}, "
              f"Neuron {neuron_idx+1}/{num_neurons}, "
              f"MSE: {mse:.4f}  STD: {session_label_stds[neuron_idx]:.4f}")

    print(f"Session {test_idx+1}/{num_sessions} — done.")

    # ── Cleanup ───────────────────────────────────────────────────
    del trunk, train_heads, test_head, optimizer, adapt_opt
    del train_loaders, adapt_loader, eval_loader, predictions, eval_labels
    torch.cuda.empty_cache()

np.save("mse_matrix_lstm_trunk_ver.npy", mse_matrix_lstm)
