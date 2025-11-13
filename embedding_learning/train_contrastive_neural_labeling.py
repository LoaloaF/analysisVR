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

from models import AutoEncoder, LinearAutoEncoder

USE_LINEAR_AUTOENCODER = False
USE_OLD_SHUFFLED_INDICES = True
EMBEDDING_DIM = 3

batch_size = 4096
contrast_weight = 0.1
decorrelation_weight = 0.1  # Weight for decorrelation loss to prevent dimension collapse
hidden_size = 50


data = pd.read_csv("merged.csv")

neural_x = data.iloc[:, 12:].values
state_x = data.iloc[:, 1:10].values
y = data.iloc[:, 10].values

neural_x = torch.from_numpy(neural_x).float()
state_x = torch.from_numpy(state_x).float()
y = torch.from_numpy(y).float()

if USE_LINEAR_AUTOENCODER:
    model = LinearAutoEncoder(input_size=state_x.shape[1], output_size=EMBEDDING_DIM)
else:
    model = AutoEncoder(input_size=state_x.shape[1], hidden_size=hidden_size, output_size=EMBEDDING_DIM)

if USE_OLD_SHUFFLED_INDICES:
    shuffled_indices = torch.load("shuffled_indices.pt")
else:
    shuffled_indices = torch.randperm(len(neural_x))
shuffled_neural_x = neural_x[shuffled_indices]
shuffled_state_x = state_x[shuffled_indices]
shuffled_y = y[shuffled_indices]

train_neural_x = shuffled_neural_x[:int(0.8*len(shuffled_neural_x))]
train_state_x = shuffled_state_x[:int(0.8*len(shuffled_state_x))]
train_y = shuffled_y[:int(0.8*len(shuffled_y))]
test_neural_x = shuffled_neural_x[int(0.8*len(shuffled_neural_x)):]
test_state_x = shuffled_state_x[int(0.8*len(shuffled_state_x)):]
test_y = shuffled_y[int(0.8*len(shuffled_y)):]

mean = train_neural_x.mean(dim=0, keepdim=True)
std_neural = train_neural_x.std(dim=0, keepdim=True) + 1e-8
train_neural_x = (train_neural_x - mean) / std_neural
test_neural_x = (test_neural_x - mean) / std_neural

mean = train_state_x.mean(dim=0, keepdim=True)
std_state = train_state_x.std(dim=0, keepdim=True) + 1e-8
train_state_x = (train_state_x - mean) / std_state
test_state_x = (test_state_x - mean) / std_state

train_dataset = TensorDataset(train_neural_x, train_state_x, train_y)
test_dataset = TensorDataset(test_neural_x, test_state_x, test_y)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_loss = []
test_loss = []
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-4, factor=0.1, patience=5)
criterion = nn.MSELoss()

train_dataset = TensorDataset(train_state_x, train_neural_x)
test_dataset = TensorDataset(test_state_x, test_neural_x)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_loss = []
test_loss = []
test_recon_loss = []
test_contr_loss = []
test_decorr_loss = []
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
criterion = nn.MSELoss()

def contrastive_loss(embeddings, neural_vector, margin=1.0):
    # Compute all-pairs distance between embeddings, shape (batch, batch)
    diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)  # [batch, batch, embdim]
    dists = torch.norm(diff, dim=2)  # [batch, batch]

    # Create label difference matrix (absolute difference of y-value, broadcasted)
    label_diff = torch.norm(neural_vector.unsqueeze(1) - neural_vector.unsqueeze(0), dim=2)  # [batch, batch]
    # Normalize label_diff to [0, 1]
    # label_diff = label_diff / label_diff.max().clamp(min=1e-8)

    # Contrastive loss: push apart pairs with large y-difference
    positive_mask = (label_diff < 0.5 * neural_vector.shape[1])  # "close" y
    negative_mask = (label_diff > 1.5 * neural_vector.shape[1]) # "far" y

    positive_loss = (dists * positive_mask.float()).sum() / (positive_mask.float().sum() + 1e-8)
    negative_loss = ((margin - dists).clamp(min=0) * negative_mask.float()).sum() / (negative_mask.float().sum() + 1e-8)
    total_contrastive = positive_loss + negative_loss

    return total_contrastive

def decorrelation_loss(embeddings):
    """
    Penalize correlation between embedding dimensions to prevent collapse to 1D.
    This encourages the model to use both dimensions independently.
    """
    # Center the embeddings
    emb_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    
    # Compute covariance matrix
    cov = torch.mm(emb_centered.t(), emb_centered) / (embeddings.shape[0] - 1)
    
    # Extract off-diagonal elements (correlations between different dimensions)
    # We want these to be close to zero
    n_dims = embeddings.shape[1]
    off_diag = []
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            # Normalize by standard deviations to get correlation
            std_i = torch.sqrt(cov[i, i] + 1e-8)
            std_j = torch.sqrt(cov[j, j] + 1e-8)
            corr = cov[i, j] / (std_i * std_j + 1e-8)
            off_diag.append(corr ** 2)
    
    if len(off_diag) == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    return torch.stack(off_diag).mean()

def calculate_loss(batch_state_x, batch_neural_x):
    embedding = model.encoder(batch_state_x)
    reconstruction = model.decoder(embedding)
    recon_loss = criterion(reconstruction, batch_state_x)
    contr_loss = contrastive_loss(embedding, batch_neural_x)
    decorr_loss = decorrelation_loss(embedding)
    loss = recon_loss + contrast_weight * contr_loss + decorrelation_weight * decorr_loss
    return loss, recon_loss, contr_loss, decorr_loss

with torch.no_grad():
    running_test_loss = []
    running_test_recon_loss = []
    running_test_contr_loss = []
    running_test_decorr_loss = []
    for batch_state_x, batch_neural_x in tqdm(test_loader):
        loss, recon_loss, contr_loss, decorr_loss = calculate_loss(batch_state_x, batch_neural_x)
        running_test_loss.append(loss.item() * len(batch_neural_x))
        running_test_recon_loss.append(recon_loss.item() * len(batch_neural_x))
        running_test_contr_loss.append(contr_loss.item() * len(batch_neural_x))
        running_test_decorr_loss.append(decorr_loss.item() * len(batch_neural_x))
    print(f"Test Loss: {np.sum(running_test_loss) / len(test_dataset)}")
    print(f"Test Recon Loss: {np.sum(running_test_recon_loss) / len(test_dataset)}")
    print(f"Test Contrastive Loss: {np.sum(running_test_contr_loss) / len(test_dataset)}")
    print(f"Test Decorrelation Loss: {np.sum(running_test_decorr_loss) / len(test_dataset)}")

for epoch in range(250):
    running_test_loss = []
    running_test_recon_loss = []
    running_test_contr_loss = []
    running_test_decorr_loss = []
    running_train_loss = []
    for batch_state_x, batch_neural_x in tqdm(train_loader):
        optimizer.zero_grad()
        # Forward through encoder and decoder
        loss, _, __, ___ = calculate_loss(batch_state_x, batch_neural_x)
        loss.backward()
        optimizer.step()
        running_train_loss.append(loss.item() * len(batch_state_x))
    scheduler.step(np.sum(running_train_loss) / len(train_dataset))
    
    with torch.no_grad():
        for batch_state_x, batch_neural_x in tqdm(test_loader):
            loss, recon_loss, contr_loss, decorr_loss = calculate_loss(batch_state_x, batch_neural_x)
            running_test_loss.append(loss.item() * len(batch_neural_x))
            running_test_recon_loss.append(recon_loss.item() * len(batch_neural_x))
            running_test_contr_loss.append(contr_loss.item() * len(batch_neural_x))
            running_test_decorr_loss.append(decorr_loss.item() * len(batch_neural_x))
    print(f"Epoch {epoch}, Train Loss: {np.sum(running_train_loss) / len(train_dataset)}, Test Loss: {np.sum(running_test_loss) / len(test_dataset)}, Test Recon Loss: {np.sum(running_test_recon_loss) / len(test_dataset)}, Test Contrastive Loss: {np.sum(running_test_contr_loss) / len(test_dataset)}, Test Decorrelation Loss: {np.sum(running_test_decorr_loss) / len(test_dataset)}")
    train_loss.append(np.sum(running_train_loss) / len(train_dataset))
    test_loss.append(np.sum(running_test_loss) / len(test_dataset))
    test_recon_loss.append(np.sum(running_test_recon_loss) / len(test_dataset))
    test_contr_loss.append(np.sum(running_test_contr_loss) / len(test_dataset))
    test_decorr_loss.append(np.sum(running_test_decorr_loss) / len(test_dataset))

if USE_LINEAR_AUTOENCODER:
    torch.save(model.state_dict(), "contrastive_neural_labeling_linear_model.pth")
else:
    torch.save(model.state_dict(), "contrastive_neural_labeling_model.pth")

torch.save(shuffled_indices, "shuffled_indices.pt")

plt.plot(train_loss)
plt.plot(test_loss)
plt.show()