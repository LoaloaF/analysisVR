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


data = np.load("fr_behavior_glm_input.npy", allow_pickle=True)


neural_xs = []
state_xs = []
for i in range(1, 33):
    data_i = data[data[:, 0].astype(int) == i]
    print(data[:, 0])
    mask = ~pd.isna(data_i[:, 1:]).any(axis=1)
    neural_x = data_i[mask, 1:-5].astype(np.float32)
    state_x = data_i[mask, -5:].astype(np.float32)
    neural_xs.append(neural_x)
    state_xs.append(state_x)


neural_x = [torch.from_numpy(neural_x).float() for neural_x in neural_xs]
state_x = [torch.from_numpy(state_x).float() for state_x in state_xs]

# for i in range(len(neural_x)):
#     print(neural_x[i].shape)
#     print(state_x[i].shape)

failed_sessions = []
for session_id, nx, sx in zip(range(1, 33), neural_x, state_x):
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
else:
    print("No failed sessions")



multi_cebra_model_discrete = cebra.CEBRA(batch_size=512,
                                output_dimension=3,
                                max_iterations=10,
                                max_adapt_iterations=10)



embeddings_runs = []
for i in range(10):
    multi_cebra_model_discrete.fit(neural_x, state_x)
    embeddings_runs.append(multi_cebra_model_discrete.embeddings)

scores_runs, pairs_runs, ids_runs = cebra.sklearn.metrics.consistency_score(embeddings=embeddings_runs,
                                                                            between="runs")

cebra.plot_consistency(scores_runs, pairs_runs, ids_runs, vmin=0, vmax=100, title="Between-runs consistencies")
plt.show()

