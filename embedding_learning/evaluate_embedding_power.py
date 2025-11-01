from models import AutoEncoder
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

batch_size = 4096
hidden_size = 20
data = pd.read_csv("o.csv")

x = data.iloc[:, 1:-2].values
y = data.iloc[:, -2].values

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

model = AutoEncoder(input_size=x.shape[1], hidden_size=hidden_size, output_size=2)

model.load_state_dict(torch.load("contrastive_model.pth"))
shuffled_indices = torch.load("shuffled_indices.pt")
shuffled_x = x[shuffled_indices]
shuffled_y = y[shuffled_indices]
train_x = shuffled_x[:int(0.8*len(shuffled_x))]
train_y = shuffled_y[:int(0.8*len(shuffled_y))]
test_x = shuffled_x[int(0.8*len(shuffled_x)):]
test_y = shuffled_y[int(0.8*len(shuffled_y)):]
mean = train_x.mean(dim=0, keepdim=True)
std = train_x.std(dim=0, keepdim=True) + 1e-8
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std
test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=True)
train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)


# Take a sequential sequence of x values from the test set, apply PCA, and plot their projections

from sklearn.feature_selection import mutual_info_regression

# Compute embeddings and PCA projections for the whole test set
with torch.no_grad():
    all_test_x = []
    all_test_y = []
    for batch_x, batch_y in test_loader:
        all_test_x.append(batch_x)
        all_test_y.append(batch_y)
    all_test_x = torch.cat(all_test_x, dim=0)
    all_test_y = torch.cat(all_test_y, dim=0)
    all_embeddings = model.encoder(all_test_x).cpu().numpy()
all_test_x_np = all_test_x.cpu().numpy()

# Flatten the input for mutual info computation (samples, features)
num_samples = all_test_x_np.shape[0]
flat_test_x = all_test_x_np.reshape(num_samples, -1)

# Fit PCA on test set and project
test_pca = PCA(n_components=all_embeddings.shape[1] if all_embeddings.shape[1] < flat_test_x.shape[1] else 10)
pca_z = test_pca.fit_transform(flat_test_x)

# Compute mutual information between original data and embeddings
# We'll compute mean MI across dimensions

def mean_mutual_info(X, Y):
    # X: [samples, features], Y: [samples, features_2]
    # Returns: average mutual information across features in Y
    mi_list = []
    for i in range(Y.shape[1]):
        mi = mutual_info_regression(X, Y[:, i])
        mi_list.append(np.mean(mi))
    return np.mean(mi_list)

embedding_mi = mean_mutual_info(flat_test_x, all_embeddings)
pca_mi = mean_mutual_info(flat_test_x, pca_z)

print(f"Average Mutual Information (Original <-> Embedding): {embedding_mi:.4f}")
print(f"Average Mutual Information (Original <-> PCA): {pca_mi:.4f}")
