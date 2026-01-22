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

# Get a batch from test_loader
batch_x, batch_y = next(iter(test_loader))
with torch.no_grad():
    embeddings = model.encoder(batch_x)

# Select a few pairs of test points that are close to each other and some far apart
# We'll use the labels in batch_y as "closeness"
pair_indices = []
for i in range(len(batch_y)):
    distances = torch.abs(batch_y - batch_y[i])
    # Find a close one (excluding self)
    close_idx = (distances > 0).nonzero(as_tuple=True)[0][(distances[distances > 0]).argmin()]
    # Find a far one
    far_idx = distances.argmax()
    pair_indices.append((i, close_idx.item(), far_idx.item()))
    # Only get a few examples
    if len(pair_indices) >= 5:
        break

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# Reduce embeddings to 2D for visualization using PCA if necessary
latent_np = embeddings.cpu().detach().numpy()
if latent_np.shape[1] > 2:
    pca_latent = PCA(n_components=2)
    latent_2d = pca_latent.fit_transform(latent_np)
else:
    latent_2d = latent_np

fig, ax = plt.subplots(figsize=(8, 6))

for idx, (i, close_idx, far_idx) in enumerate(pair_indices):
    c = colors[idx % len(colors)]
    # Plot the latent points
    ax.scatter(latent_2d[i, 0], latent_2d[i, 1], marker='o', color=c, label=f'Point {i}')
    ax.scatter(latent_2d[close_idx, 0], latent_2d[close_idx, 1], marker='o', color=c, edgecolors='k', alpha=0.5, label=f'Close {close_idx}')
    ax.scatter(latent_2d[far_idx, 0], latent_2d[far_idx, 1], marker='o', color=c, edgecolors='w', alpha=0.5, label=f'Far {far_idx}')
    # Draw lines between point and their close/far counterparts
    ax.plot([latent_2d[i,0], latent_2d[close_idx,0]], [latent_2d[i,1], latent_2d[close_idx,1]], color=c, linestyle='-', linewidth=1, alpha=0.7)
    ax.plot([latent_2d[i,0], latent_2d[far_idx,0]], [latent_2d[i,1], latent_2d[far_idx,1]], color=c, linestyle='--', linewidth=1, alpha=0.7)

ax.set_title('Embedding Space: Close & Far Test Points in Latent (Contrastive) Space')
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), loc='best', fontsize="small")
plt.tight_layout()
plt.show()

batch_x, batch_y = next(iter(test_loader))
num_examples = min(8, len(batch_x))
with torch.no_grad():
    encoded = model.encoder(batch_x[:num_examples])
if isinstance(encoded, tuple):
    z = encoded[0]
else:
    z = encoded
with torch.no_grad():
    recon = model.decoder(z)

original = batch_x[:num_examples]
# Convert to numpy if needed
original = original.detach().cpu().numpy() if hasattr(original, "detach") else original
recon = recon.detach().cpu().numpy() if hasattr(recon, "detach") else recon

# Flatten if needed (ensure shape [num_examples, n_features])
original_flat = original.reshape(num_examples, -1)
recon_flat = recon.reshape(num_examples, -1)

# Apply PCA to both original and reconstructed (joint fit)
pca = PCA(n_components=2)
pca.fit(np.concatenate([original_flat, recon_flat], axis=0))
original_pca = pca.transform(original_flat)
recon_pca = pca.transform(recon_flat)

fig, ax = plt.subplots(figsize=(6, 6))
for i in range(num_examples):
    # Original point
    ax.scatter(original_pca[i, 0], original_pca[i, 1], color='blue', label='Original' if i == 0 else "", marker='o')
    # Reconstructed point
    ax.scatter(recon_pca[i, 0], recon_pca[i, 1], color='orange', label='Reconstruction' if i == 0 else "", marker='x')
    # Line connecting them
    ax.plot([original_pca[i, 0], recon_pca[i, 0]], [original_pca[i, 1], recon_pca[i, 1]], color='gray', linestyle='--', alpha=0.7)

ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_title("Original vs Reconstruction in PCA Space")
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys())
plt.tight_layout()
plt.show()