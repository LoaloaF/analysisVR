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
for i in range(1, 34):
    data_i = data[data[:, 0] == i]
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
for session_id, nx, sx in zip(range(1, 34), neural_x, state_x):
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

session_ids = [i for i in range(len(neural_x)) if neural_x[i].shape[0] > 0 and state_x[i].shape[0] > 0]
neural_x = [nx for nx in neural_x if nx.shape[0] > 0]
state_x = [sx for sx in state_x if sx.shape[0] > 0]


multi_cebra_model_discrete = cebra.CEBRA(batch_size=512,
                                output_dimension=3,
                                max_iterations=10,
                                max_adapt_iterations=10)



multi_cebra_model_discrete.fit(state_x, neural_x)
curr_embeddings = []
labels = []
for session_id in range(len(neural_x)):
    curr_embeddings.append(multi_cebra_model_discrete.transform(state_x[session_id], session_id = session_id))
    labels.append(neural_x[session_id][:, 0])
# import pdb; pdb.set_trace()

scores_runs, pairs_runs, ids_runs = cebra.sklearn.metrics.consistency_score(embeddings=curr_embeddings,
                                                                            between="datasets",
                                                                            dataset_ids= range(len(neural_x)),
                                                                            labels=labels,
                                                                            )

cebra.plot_consistency(scores_runs, pairs_runs, ids_runs, vmin=0, vmax=100, title="Between-datasets consistencies")
plt.show()

