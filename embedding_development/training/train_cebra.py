import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random

import cebra
import cebra.models
from torch.func import vmap, functional_call

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from load_encoder import build_windows


def make_label_nn_table(label_1d: np.ndarray, k: int = 20) -> np.ndarray:
    """Precompute k nearest neighbors in label space. O(T log T), then O(1) lookup."""
    T = len(label_1d)
    sorted_idx = np.argsort(label_1d)
    rank       = np.argsort(sorted_idx)
    half = k // 2
    cols = []
    for delta in range(-half, 0):
        cols.append(sorted_idx[np.clip(rank + delta, 0, T - 1)])
    for delta in range(1, k - half + 1):
        cols.append(sorted_idx[np.clip(rank + delta, 0, T - 1)])
    return np.stack(cols, axis=1).astype(np.int64)  # (T, k)


def batched_infoNCE(ref_z: torch.Tensor, pos_z: torch.Tensor,
                    neg_z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Vectorized InfoNCE matching CEBRA's FixedEuclideanInfoNCE.

    For each anchor i, positive is pos_z[i] and ALL neg_z[j] (j=0..B-1) are
    negatives — same as CEBRA's batch-level contrastive loss.
    Random baseline: log(B+1) ≈ log(513) ≈ 6.24 for B=512.
    Uses bmm for O(N·B²·D) GEMM instead of materializing the (N,B,B,D) tensor.
    """
    # ref_z, pos_z, neg_z: (N, B, D)
    pos_d = ((ref_z - pos_z) ** 2).sum(-1)                               # (N, B)
    # Efficient pairwise squared-Euclidean via: ||a-b||² = ||a||² + ||b||² - 2a·b
    ref_sq  = (ref_z ** 2).sum(-1, keepdim=True)                          # (N, B, 1)
    neg_sq  = (neg_z ** 2).sum(-1, keepdim=True).transpose(-1, -2)       # (N, 1, B)
    dot     = torch.bmm(ref_z, neg_z.transpose(-1, -2))                  # (N, B, B)
    neg_d_all = ref_sq + neg_sq - 2.0 * dot                              # (N, B, B)
    pos_s = torch.exp(-pos_d / temperature)                               # (N, B)
    neg_s = torch.exp(-neg_d_all / temperature).sum(-1)                  # (N, B)
    return (-torch.log(pos_s / (pos_s + neg_s + 1e-8))).mean()

parser = argparse.ArgumentParser()
parser.add_argument("--split_path",    type=str,  default="test_indices_by_session.npy",
                    help="Path to train/test split file (generate with generate_split.py)")
parser.add_argument("--seed",          type=int,  default=42)
parser.add_argument("--use_ensembles", action="store_true", default=False,
                    help="Use ensemble activity as the contrastive label instead of single-unit spikes")
parser.add_argument("--models_dir",    type=str,  default=None,
                    help="Directory to save model checkpoints (default: auto from --use_ensembles)")
parser.add_argument("--n_epochs",      type=int,  default=20,
                    help="Training epochs per session (iterations = n_epochs × T // batch_size)")
parser.add_argument("--batch_size",    type=int,  default=128)
parser.add_argument("--embed_dim",     type=int,  default=8)
parser.add_argument("--num_units",     type=int,  default=32)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--time_offset",  type=int,  default=10,
                    help="Time offset for trial-boundary masking")
parser.add_argument("--nn_k",         type=int,  default=20,
                    help="Number of nearest neighbors in label space for positive pair sampling")
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
    'frame_raw_500msMedian',
    'frame_raw_abs_acc_500msMedian',
    'frame_YawPitch_abs_vel_sum_500msMedian',
    'frame_YawPitch_abs_acc_sum_500msMedian',
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

    for col in spikes_session.columns:
        mean_val = spikes_session[col].mean()
        std_val  = spikes_session[col].std() + 1e-8
        if std_val != 0:
            spikes_session[col] = (spikes_session[col] - mean_val) / std_val

    data_by_trial, labels_by_trial = {}, {}
    for trial_id in beh["trial_id"].unique():
        trial_mask                = beh["trial_id"] == trial_id
        data_by_trial[trial_id]   = beh.loc[trial_mask, non_categorical_cols + zone_onehot_cols].values.astype(np.float32)
        labels_by_trial[trial_id] = spikes_session.loc[trial_mask].values.astype(np.float32)

    session_dataset_singles[session_id] = {
        "data":   data_by_trial,
        "labels": labels_by_trial,
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
_default_models_dir = "./models/cebra/ensembles" if USE_ENSEMBLES else "./models/cebra/spikes"
models_dir = args.models_dir if args.models_dir is not None else _default_models_dir
os.makedirs(models_dir, exist_ok=True)

input_size  = len(non_categorical_cols) + len(zone_onehot_cols)
num_sessions = len(session_ids)
num_neurons  = ensembles_values.shape[1] if USE_ENSEMBLES else spikes_loaded.shape[1]

print(f"input_size={input_size}, num_sessions={num_sessions}, num_neurons={num_neurons}")
print(f"embed_dim={args.embed_dim}, num_units={args.num_units}, n_epochs={args.n_epochs}, batch_size={args.batch_size}")

# ── Train ───────────────────────────────────────────────────────────────────
torch.cuda.empty_cache()
device = torch.device('cuda')
print(f"Using device: {device}")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)

for session_idx, session_id in enumerate(session_ids):
    session_data   = session_dataset_singles[session_id]["data"]
    session_labels = session_dataset_singles[session_id]["labels"]

    test_idx_   = test_indices_by_session[session_id]
    all_indices = list(session_data.keys())
    train_idx   = np.setdiff1d(all_indices, test_idx_)

    # Concatenate training trials
    train_actions_list  = [session_data[idx]   for idx in train_idx]
    train_actions       = np.concatenate(train_actions_list,  axis=0).astype(np.float32)
    train_labels_np     = np.concatenate([session_labels[idx] for idx in train_idx], axis=0).astype(np.float32)

    print(f"Session {session_idx+1}/{num_sessions} — train: {len(train_actions)} timesteps")

    # Sliding windows over concatenated training actions
    train_windows   = build_windows(train_actions)                      # (T_train, n_feat, 10)
    train_windows_t = torch.tensor(train_windows, dtype=torch.float32, device=device)

    # Valid reference indices: exclude the last time_offset frames before each
    # trial boundary so positive pairs don't jump across trials.
    trial_lengths = [len(x) for x in train_actions_list]
    valid_mask    = np.ones(len(train_actions), dtype=bool)
    cumlen        = 0
    for tlen in trial_lengths[:-1]:       # skip the last trial — nothing follows it
        cumlen += tlen
        valid_mask[max(0, cumlen - args.time_offset):cumlen] = False
    valid_indices  = torch.tensor(np.where(valid_mask)[0], dtype=torch.long, device=device)
    n_valid        = len(valid_indices)
    max_iterations = max(1, args.n_epochs * (n_valid // args.batch_size))

    # Skip session if all neuron checkpoints already exist
    session_paths = [
        os.path.join(models_dir, f"session_{session_idx:02d}_neuron_{neuron_idx:02d}.pt")
        for neuron_idx in range(num_neurons)
    ]
    if all(os.path.exists(p) for p in session_paths):
        print(f"  Skip:  session {session_idx+1} — all {num_neurons} neurons already saved")
        del train_windows_t, valid_indices
        torch.cuda.empty_cache()
        continue

    resume_path = os.path.join(models_dir, f".resume_session_{session_idx:02d}.pt")

    # ── Precompute NN tables for ALL neurons at once ─────────────────────────
    session_seed = args.seed * 10_000_000 + session_idx * 10_000
    torch.manual_seed(session_seed)
    np.random.seed(session_seed % (2**32))

    all_nn = np.stack([
        make_label_nn_table(train_labels_np[:, n], k=args.nn_k)
        for n in range(num_neurons)
    ])                                              # (N, T, k)
    all_nn_t = torch.tensor(all_nn, dtype=torch.long, device=device)
    n_idx    = torch.arange(num_neurons, device=device)

    # ── Initialize one parameter copy per neuron via ParameterDict ───────────
    base_encoder = cebra.models.init(
        'offset10-model-mse',
        num_neurons=input_size,
        num_units=args.num_units,
        num_output=args.embed_dim,
    )
    param_store = nn.ParameterDict({
        k.replace('.', '__'): nn.Parameter(
            torch.stack([v.clone() for _ in range(num_neurons)]).to(device)
        )
        for k, v in base_encoder.named_parameters()
    })

    def fwd_one(params, x):
        return functional_call(base_encoder, params, (x,))
    batched_fwd = vmap(fwd_one, in_dims=(0, 0))

    optimizer = torch.optim.Adam(param_store.parameters(), lr=args.learning_rate)
    start_iter = 0
    loss_history = []

    # Resume from mid-session checkpoint if one exists
    if os.path.exists(resume_path):
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        for k in param_store:
            param_store[k].data.copy_(resume['param_store'][k])
        optimizer.load_state_dict(resume['optimizer'])
        start_iter   = resume['iteration'] + 1
        loss_history = resume['loss_history']
        print(f"  Resumed session {session_idx+1} from iter {start_iter}/{max_iterations}")

    # Build bp dict once; it stays valid across optimizer steps (in-place updates)
    bp = {k.replace('__', '.'): param_store[k] for k in param_store}

    print(f"  Session {session_idx+1}: {n_valid} valid timesteps → {max_iterations} iters ({args.n_epochs} epochs)")
    checkpoint_interval = max(1, max_iterations // 10)   # save ~every 10% of training
    for iteration in range(start_iter, max_iterations):
        ri      = torch.randint(0, n_valid, (args.batch_size,), device=device)
        ref_idx = valid_indices[ri]
        ni      = torch.randint(0, n_valid, (args.batch_size,), device=device)
        neg_idx = valid_indices[ni]
        ki      = torch.randint(0, args.nn_k, (num_neurons, args.batch_size), device=device)

        # pos_idx[n, b] = label-NN neighbor for neuron n, batch element b
        pos_idx = all_nn_t[n_idx.unsqueeze(1), ref_idx.unsqueeze(0), ki]  # (N, B)

        pos_win = train_windows_t[pos_idx.reshape(-1)].reshape(
            num_neurons, args.batch_size, input_size, 10)
        ref_win = train_windows_t[ref_idx].unsqueeze(0).expand(
            num_neurons, -1, -1, -1).contiguous()
        neg_win = train_windows_t[neg_idx].unsqueeze(0).expand(
            num_neurons, -1, -1, -1).contiguous()

        # Batched forward over all neurons simultaneously: (N, B, embed_dim)
        ref_z = batched_fwd(bp, ref_win).squeeze(-1)
        pos_z = batched_fwd(bp, pos_win).squeeze(-1)
        neg_z = batched_fwd(bp, neg_win).squeeze(-1)

        loss = batched_infoNCE(ref_z, pos_z, neg_z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if (iteration + 1) % checkpoint_interval == 0:
            torch.save({
                'param_store':  {k: param_store[k].detach().cpu() for k in param_store},
                'optimizer':    optimizer.state_dict(),
                'iteration':    iteration,
                'loss_history': loss_history,
            }, resume_path)

    # ── Save each neuron's checkpoint separately ─────────────────────────────
    for neuron_idx in range(num_neurons):
        model_path = session_paths[neuron_idx]
        neuron_state = {
            k.replace('__', '.'): param_store[k][neuron_idx].detach().cpu()
            for k in param_store
        }
        torch.save({
            'encoder_state_dict': neuron_state,
            'encoder_config': {
                'model_architecture': 'offset10-model-mse',
                'num_neurons':        input_size,
                'num_units':          args.num_units,
                'num_output':         args.embed_dim,
            },
            'training_config': {
                'arm':            'cebra_contrastive',
                'n_epochs':       args.n_epochs,
                'max_iterations': max_iterations,
                'batch_size':     args.batch_size,
                'learning_rate':  args.learning_rate,
                'time_offset':    args.time_offset,
                'temperature':    1.0,
                'seed':           args.seed,
                'session_id':     str(session_id),
                'neuron_idx':     neuron_idx,
                'use_ensembles':  USE_ENSEMBLES,
            },
            'loss_history': np.array(loss_history, dtype=np.float32),
            'final_loss':   float(loss_history[-1]),
            'cebra_version': str(cebra.__version__),
            'torch_version': str(torch.__version__),
        }, model_path)

    print(f"  Saved: session {session_idx+1}/{num_sessions}, {num_neurons} neurons"
          f" — final_loss={loss_history[-1]:.4f}")

    if os.path.exists(resume_path):
        os.remove(resume_path)

    del param_store, optimizer, all_nn_t, all_nn, train_windows_t, valid_indices
    torch.cuda.empty_cache()

print("Training complete.")
