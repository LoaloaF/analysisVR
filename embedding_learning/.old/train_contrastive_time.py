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

batch_size = 4096
contrast_weight = 1
hidden_size = 20


data = pd.read_csv("o.csv")

x = data.iloc[:, 1:-2].values
y = data.iloc[:, -2].values

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

if USE_LINEAR_AUTOENCODER:
    model = LinearAutoEncoder(input_size=x.shape[1], output_size=2)
else:
    model = AutoEncoder(input_size=x.shape[1], hidden_size=hidden_size, output_size=2)

if USE_OLD_SHUFFLED_INDICES:
    shuffled_indices = torch.load("shuffled_indices.pt")
else:
    shuffled_indices = torch.randperm(len(x))
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


train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_loss = []
test_loss = []
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
criterion = nn.MSELoss()

def contrastive_loss(embeddings, labels, margin=1.0):
    TIME_UNIT = 40000
    batch_size = embeddings.size(0)
    # Compute all-pairs distance between embeddings, shape (batch, batch)
    diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)  # [batch, batch, embdim]
    dists = torch.norm(diff, dim=2)  # [batch, batch]

    # Create label difference matrix (absolute difference of y-value, broadcasted)
    label_diff = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))  # [batch, batch]
    # Normalize label_diff to [0, 1]
    # label_diff = label_diff / label_diff.max().clamp(min=1e-8)

    # Contrastive loss: push apart pairs with large y-difference
    positive_mask = (label_diff < TIME_UNIT * 3 + 1)  # "close" y
    negative_mask = (label_diff > TIME_UNIT * 25 - 1) # "far" y

    positive_loss = (dists * positive_mask.float()).sum() / (positive_mask.float().sum() + 1e-8)
    negative_loss = ((margin - dists).clamp(min=0) * negative_mask.float()).sum() / (negative_mask.float().sum() + 1e-8)
    total_contrastive = positive_loss + negative_loss

    return total_contrastive

def calculate_loss(batch_x, batch_y):
    embedding = model.encoder(batch_x)
    reconstruction = model.decoder(embedding)
    recon_loss = criterion(reconstruction, batch_x)
    contr_loss = contrastive_loss(embedding, batch_y)
    loss = recon_loss + contrast_weight * contr_loss
    return loss

with torch.no_grad():
    for batch_x, batch_y in tqdm(test_loader):
        loss = calculate_loss(batch_x, batch_y)
        test_loss.append(loss.item() * len(batch_x))
    print(f"Test Loss: {np.sum(test_loss) / len(test_dataset)}")

running_train_loss = []
for epoch in range(250):
    for batch_x, batch_y in tqdm(train_loader):
        optimizer.zero_grad()
        # Forward through encoder and decoder
        loss = calculate_loss(batch_x, batch_y)
        loss.backward()
        optimizer.step()
        running_train_loss.append(loss.item() * len(batch_x))
    scheduler.step(np.sum(running_train_loss) / len(train_dataset))
    with torch.no_grad():
        running_test_loss = []
        for batch_x, batch_y in tqdm(test_loader):
            loss = calculate_loss(batch_x, batch_y)
            running_test_loss.append(loss.item() * len(batch_x))
    train_loss.append(np.sum(running_train_loss) / len(train_dataset))
    print(f"Epoch {epoch}, Train Loss: {np.sum(running_train_loss) / len(train_dataset)}, Test Loss: {np.sum(running_test_loss) / len(test_dataset)}")
    running_train_loss = []
    test_loss.append(np.sum(running_test_loss) / len(test_dataset))

if USE_LINEAR_AUTOENCODER:
    torch.save(model.state_dict(), "contrastive_linear_model.pth")
else:
    torch.save(model.state_dict(), "contrastive_model.pth")

torch.save(shuffled_indices, "shuffled_indices.pt")

plt.plot(train_loss)
plt.plot(test_loss)
plt.show()