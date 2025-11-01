from models import AutoEncoder
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import cm


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


# Visualize how embeddings evolve through time (assumes batch_x is ordered in time)

batch_x = (x[:200] - mean) / std

latent_2d = model.encoder(batch_x)
latent_2d = latent_2d.detach().numpy()

if hasattr(batch_x, "shape") and batch_x.shape[0] > 1:
    # get time index for each sample (assume sequential order)
    time_indices = np.arange(len(batch_x))
    norm = plt.Normalize(time_indices.min(), time_indices.max())
    cmap = cm.get_cmap('viridis')

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(time_indices)):
        color = cmap(norm(time_indices[i]))
        ax.scatter(latent_2d[i, 0], latent_2d[i, 1], color=color, s=30)
        if i > 0:
            # draw line from previous embedding to current
            ax.plot(
                [latent_2d[i-1, 0], latent_2d[i, 0]],
                [latent_2d[i-1, 1], latent_2d[i, 1]],
                color=color,
                linewidth=1,
                alpha=0.8
            )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Time Index')
    ax.set_title("Trajectory of Embeddings Through Time in Latent Space")
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    plt.tight_layout()
    plt.show()
else:
    print("Batch size too small to visualize time evolution.")

# For example, take the first N samples
N_seq = 200
# If test_loader.dataset supports slicing, otherwise collect from batches:
sequential_x = (x[:N_seq] - mean) / std
sequential_x_flat = sequential_x.reshape(N_seq, -1)
pca_seq = PCA(n_components=2)
proj_seq = pca_seq.fit_transform(sequential_x_flat)

fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(proj_seq[:, 0], proj_seq[:, 1], c=np.arange(N_seq), cmap='viridis', marker='o')
for i in range(N_seq - 1):
    ax.plot([proj_seq[i, 0], proj_seq[i+1, 0]], [proj_seq[i, 1], proj_seq[i+1, 1]], color='gray', alpha=0.6)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Sequence Index')
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_title(f"PCA Projection of {N_seq} Sequential X Samples")
plt.tight_layout()
plt.show()

