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

batch_size = 4096
contrast_weight = 1
hidden_size = 20


data = pd.read_csv("o.csv")

data = data.sort_values(by="bin_start")

x = data.iloc[:, 1:-2].values
y = data.iloc[:, -2].values

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

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
    embeddings_runs.append(model.fit_transform(x))

scores_runs, pairs_runs, ids_runs = cebra.sklearn.metrics.consistency_score(embeddings=embeddings_runs,
                                                                            between="runs")

cebra.plot_consistency(scores_runs, pairs_runs, ids_runs, vmin=0, vmax=100, title="Between-runs consistencies")
plt.show()