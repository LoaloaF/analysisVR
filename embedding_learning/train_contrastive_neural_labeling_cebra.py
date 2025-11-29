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

import cebra
from cebra import CEBRA

from models import AutoEncoder, LinearAutoEncoder

USE_LINEAR_AUTOENCODER = False
USE_OLD_SHUFFLED_INDICES = True
EMBEDDING_DIM = 3

batch_size = 4096
contrast_weight = 0.1
decorrelation_weight = 0.1  # Weight for decorrelation loss to prevent dimension collapse
hidden_size = 50


data = pd.read_csv("joined.csv")

data['from_ephys_timestamp'] = data['t_interval'].apply(lambda x: int(x.split(",")[0][1:]))

data.set_index('from_ephys_timestamp', inplace=True)

data.drop(columns=['t_interval'], inplace=True)

data = data.sort_values(by="from_ephys_timestamp")


state_xs = [data[data["session_id"] == i].iloc[:, -5:].values for i in sorted(data["session_id"].unique())]
neural_xs = [data[data["session_id"] == i].iloc[:, 2:-5].values for i in sorted(data["session_id"].unique())]

neural_x = [torch.from_numpy(neural_x).float() for neural_x in neural_xs]
state_x = [torch.from_numpy(state_x).float() for state_x in state_xs]

failed_sessions = []
for session_id, nx, sx in zip(sorted(data["session_id"].unique()), neural_x, state_x):
    if torch.count_nonzero(nx.isnan()) > 0:
        print(torch.count_nonzero(nx.isnan()))
        print("Neural data has NaN values")
        failed_sessions.append(session_id)
    if torch.count_nonzero(sx.isnan()) > 0:
        print(torch.count_nonzero(sx.isnan()))
        print("State data has NaN values")
        failed_sessions.append(session_id)
if failed_sessions:
    print(f"Failed sessions: {failed_sessions}")
    exit()

model = CEBRA(
    model_architecture = "offset10-model",
    batch_size = 1024,
    learning_rate = 0.001,
    max_iterations = 10,
    time_offsets = 10,
    output_dimension = 2,
    device = "cuda_if_available",
    verbose = False
)

embeddings_runs = []
for i in range(10):
    embeddings_runs.append(model.fit_transform(state_x, neural_x))

scores_runs, pairs_runs, ids_runs = cebra.sklearn.metrics.consistency_score(embeddings=embeddings_runs,
                                                                            between="runs")

cebra.plot_consistency(scores_runs, pairs_runs, ids_runs, vmin=0, vmax=100, title="Between-runs consistencies")
plt.show()

