# import statsmodels.api as sm

import sys
import argparse
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
import cebra
from cebra import CEBRA
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
from models import AutoEncoder, LinearAutoEncoder

EMBEDDING_DIM = 3
BETWEEN_SUBJECTS_CONSISTENCY = False
BETWEEN_RUNS_CONSISTENCY = False
PLOT_EMBEDDINGS = True
COLOR_CODING_COLUMN = "head_angle"

def set_all_seeds(seed):
    """
    Set seeds for reproducibility across torch, numpy, and standard random modules.
    """
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


batch_size = 4096
contrast_weight = 1
hidden_size = 20


data = pd.read_csv("behavior_glm_input_flat.csv")

columns_to_drop = ["head_angle_left","head_angle_right","head_angle_velocity","head_angle_velocity_left","head_angle_velocity_right"]
data = data.drop(columns=columns_to_drop)
data = data.dropna(axis=0)

data = data.sort_values(by="from_ephys_timestamp")

groups = [list(range(1, 8)), list(range(8, 20)), list(range(20, 27)), list(range(27, 34))]

groups_xs = []
groups_ys_train = []  # Original timestamps for training
groups_ys = []  # Normalized timestamps for consistency scoring
x_columns = list(data.columns[3:])
for group in groups:
    data_session = data[data["session_id"].isin(group)]

    # Defensive: Check if data_session is empty
    if data_session.empty:
        continue

    # Use explicit column selection by name for clarity and robustness
    x = torch.from_numpy(data_session[x_columns].values).float()
    y = torch.from_numpy(data_session["from_ephys_timestamp"].values).float()
    
    # Normalize timestamps within each group to [0, 1] range for consistency scoring
    # This ensures all groups have compatible label ranges
    y_min = y.min()
    y_max = y.max()
    if y_max > y_min:
        y_normalized = (y - y_min) / (y_max - y_min)
    else:
        y_normalized = y
    
    groups_xs.append(x)
    groups_ys_train.append(y)  # Original timestamps for training
    groups_ys.append(y_normalized.numpy())  # Normalized numpy array for consistency_score

# Defensive: Check if enough groups for consistency score
if len(groups_xs) < 2 or len(groups_ys) < 2:
    raise ValueError("Not enough non-empty groups found for consistency scoring.")

if BETWEEN_SUBJECTS_CONSISTENCY:
    set_all_seeds(42)
    multi_cebra_model = cebra.CEBRA(batch_size=512,
                                    output_dimension=EMBEDDING_DIM,
                                    max_iterations=10,
                                    max_adapt_iterations=10)
    multi_cebra_model.fit(groups_xs, groups_ys)

    embeddings = []
    for i in range(len(groups_xs)):
        emb = multi_cebra_model.transform(groups_xs[i], session_id=i)
        embeddings.append(emb)

    # Use normalized labels for consistency scoring
    (scores_datasets,
        pairs_datasets,
        ids_datasets) = cebra.sklearn.metrics.consistency_score(
            embeddings=embeddings,
            labels=groups_ys,
            dataset_ids=list(range(len(embeddings))),
            between="datasets"
        )


    cebra.plot_consistency(scores_datasets, pairs_datasets, ids_datasets, vmin=0, vmax=100, title="Between-subjects consistencies")

    plt.show()

if BETWEEN_RUNS_CONSISTENCY:
    set_all_seeds(42)
    multi_cebra_model = cebra.CEBRA(batch_size=512,
                                    output_dimension=EMBEDDING_DIM,
                                    max_iterations=10,
                                    max_adapt_iterations=10)
    #Between runs consistency
    runs = 10
    embbeddings = {i: [] for i in range(len(groups_xs))}
    for run in range(runs):
        multi_cebra_model.fit(groups_xs, groups_ys)
        for i in range(len(groups_xs)):
            emb = multi_cebra_model.transform(groups_xs[i], session_id=i)
            embbeddings[i].append(emb)

    fig, axes = plt.subplots(2, round(len(groups_xs) / 2), figsize=(6 * round(len(groups_xs) / 2), 12), squeeze=False)

    even = True
    for i in range(len(groups_xs)):
        (
            scores_datasets,
            pairs_datasets,
            ids_datasets
        ) = cebra.sklearn.metrics.consistency_score(
            embeddings=embbeddings[i],
            between="runs"
        )
        if even:
            ax = axes[0, i // 2]
            even = False
        else:
            ax = axes[1, i // 2]
            even = True
        cebra.plot_consistency(
            scores_datasets, pairs_datasets, ids_datasets, 
            vmin=0, vmax=100, 
            title=f"Between-runs consistencies for session {i}",
            ax=ax
        )

    plt.tight_layout()
    plt.show()

if PLOT_EMBEDDINGS:
    set_all_seeds(42)
    multi_cebra_model = cebra.CEBRA(batch_size=512,
                                    output_dimension=EMBEDDING_DIM,
                                    max_iterations=100,
                                    max_adapt_iterations=100)
    multi_cebra_model.fit(groups_xs, groups_ys)
    embeddings = []
    for i in range(len(groups_xs)):
        emb = multi_cebra_model.transform(groups_xs[i], session_id=i)
        embeddings.append(emb)

    fig = plt.figure(figsize=(6 * round(len(groups_xs) / 2), 12))
    axes = []
    for i in range(2):
        row_axes = []
        for j in range(round(len(groups_xs) / 2)):
            ax = fig.add_subplot(2, round(len(groups_xs) / 2), i * round(len(groups_xs) / 2) + j + 1, projection='3d')
            row_axes.append(ax)
        axes.append(row_axes)
    axes = np.array(axes)
    even = True
    for group_idx in range(len(groups_xs)):
        embedding = embeddings[group_idx]
        if even:
            ax = axes[0, group_idx // 2]
            even = False
        else:
            ax = axes[1, group_idx // 2]
            even = True
        cebra.plot_embedding(embedding, ax=ax, embedding_labels=groups_xs[group_idx][:, x_columns.index(COLOR_CODING_COLUMN)].numpy())
        ax.set_title(f"Session Group {group_idx}")

    

    plt.tight_layout()
    plt.savefig(f"images/embeddings_{COLOR_CODING_COLUMN}.png")

    set_all_seeds(42)

    embeddings = []
    for x in groups_xs:
        # Perform PCA separately for each group
        pca = PCA(n_components=EMBEDDING_DIM)
        emb = pca.fit_transform(x.numpy())
        embeddings.append(emb)
    fig = plt.figure(figsize=(6 * round(len(groups_xs) / 2), 12))
    axes = []
    for i in range(2):
        row_axes = []
        for j in range(round(len(groups_xs) / 2)):
            ax = fig.add_subplot(2, round(len(groups_xs) / 2), i * round(len(groups_xs) / 2) + j + 1, projection='3d')
            row_axes.append(ax)
        axes.append(row_axes)
    axes = np.array(axes)
    even = True
    for group_idx in range(len(groups_xs)):
        embedding = embeddings[group_idx]
        if even:
            ax = axes[0, group_idx // 2]
            even = False
        else:
            ax = axes[1, group_idx // 2]
            even = True
        cebra.plot_embedding(embedding, ax=ax, embedding_labels=groups_ys[group_idx])
        ax.set_title(f"Session Group {group_idx}")

    plt.tight_layout()
    plt.savefig(f"images/embeddings_pca_{COLOR_CODING_COLUMN}.png")