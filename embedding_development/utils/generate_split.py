"""
Generates a reproducible train/test trial split and saves it to disk.

The output is a dict  { session_id: [test_trial_id, ...] }  containing the 20%
of trials held out per session.  Pass the file to any training script via
--split_path so that all models evaluate on the same held-out trials.

Usage
-----
    python generate_split.py
    python generate_split.py --out my_split.npy --train_frac 0.8 --seed 42
"""

import os
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Generate a train/test trial split.")
parser.add_argument("--base",       type=str,   default="./outputs/glm_input_data",
                    help="Directory containing behavior_glm_input.npy etc.")
parser.add_argument("--out",        type=str,   default="test_indices_by_session.npy",
                    help="Output path for the split file")
parser.add_argument("--train_frac", type=float, default=0.8,
                    help="Fraction of trials used for training (default 0.8)")
parser.add_argument("--seed",       type=int,   default=42)
args = parser.parse_args()

np.random.seed(args.seed)

# ── Load & preprocess (mirrors train_mlp.py) ────────────────────────────────
base = args.base

beh_vals = np.load(os.path.join(base, "behavior_glm_input.npy"), allow_pickle=True)
beh_idx  = np.load(os.path.join(base, "behavior_glm_input_index.npy"), allow_pickle=True)
beh_cols = np.load(os.path.join(base, "behavior_glm_input_columns.npy"), allow_pickle=True)

spk_vals = np.load(os.path.join(base, "fr_full.npy"))
spk_idx  = np.load(os.path.join(base, "fr_full_index.npy"), allow_pickle=True)

try:
    beh_index = pd.MultiIndex.from_tuples(beh_idx, names=behavior_glm_input.index.names)
except Exception:
    beh_index = pd.Index(beh_idx)

try:
    spk_index = pd.MultiIndex.from_tuples(spk_idx, names=fr_first7.index.names)
except Exception:
    spk_index = pd.Index(spk_idx)

behavior_glm_loaded = pd.DataFrame(beh_vals, index=beh_index, columns=beh_cols)
spikes_loaded       = pd.DataFrame(spk_vals, index=spk_index)

spikes_loaded.index = pd.MultiIndex.from_tuples(
    spikes_loaded.index.map(lambda t: (t[0], t[1] // 40000 - 1))
)

spikes_unique_trials = set(idx[0] for idx in spikes_loaded.index)
behavior_glm_loaded  = behavior_glm_loaded[
    behavior_glm_loaded.index.map(lambda t: t[0] in spikes_unique_trials)
]

non_nan_rows        = behavior_glm_loaded.index[~behavior_glm_loaded.isna().any(axis=1)]
behavior_glm_loaded = behavior_glm_loaded.loc[non_nan_rows]

behavior_glm_loaded = behavior_glm_loaded.drop("track_zone", axis=1)

session_ids = behavior_glm_loaded.index.map(lambda x: x[0]).unique()
print(f"{len(session_ids)} sessions found")

# ── Generate split ───────────────────────────────────────────────────────────
test_indices_by_session = {}
for session_id in session_ids:
    session_mask = behavior_glm_loaded.index.map(lambda x: x[0]) == session_id
    trial_ids    = list(behavior_glm_loaded[session_mask]["trial_id"].unique())
    np.random.shuffle(trial_ids)
    split_idx    = int(len(trial_ids) * args.train_frac)
    test_indices_by_session[session_id] = trial_ids[split_idx:]
    n_test = len(trial_ids) - split_idx
    print(f"  {session_id}: {len(trial_ids)} trials  →  train={split_idx}, test={n_test}")

np.save(args.out, test_indices_by_session, allow_pickle=True)
print(f"\nSaved split ({len(session_ids)} sessions) to {args.out}")
