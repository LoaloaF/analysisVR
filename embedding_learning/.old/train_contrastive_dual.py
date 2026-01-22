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
EMBEDDING_DIM_STATE = 3
EMBEDDING_DIM_NEURAL = 3

batch_size = 4096
contrast_weight = 1
hidden_size = 20


data = pd.read_csv("merged.csv")

state_x = data.iloc[:, 1:10].values
neural_x = data.iloc[:, 12:].values
y = data.iloc[:, 10].values

state_x = torch.from_numpy(state_x).float()
neural_x = torch.from_numpy(neural_x).float()
y = torch.from_numpy(y).float()

if USE_LINEAR_AUTOENCODER:
    neural_model = LinearAutoEncoder(input_size=neural_x.shape[1], output_size=EMBEDDING_DIM_NEURAL)
    state_model = LinearAutoEncoder(input_size=state_x.shape[1], output_size=EMBEDDING_DIM_STATE)
else:
    neural_model = AutoEncoder(input_size=neural_x.shape[1], hidden_size=hidden_size, output_size=EMBEDDING_DIM_NEURAL)
    state_model = AutoEncoder(input_size=state_x.shape[1], hidden_size=hidden_size, output_size=EMBEDDING_DIM_STATE)

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
train_neural_recon_loss = []
train_state_recon_loss = []
train_contr_loss = []
test_loss = []
test_neural_recon_loss = []
test_state_recon_loss = []
test_contr_loss = []
optimizer = torch.optim.Adam(list(neural_model.parameters()) + list(state_model.parameters()), lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-4, factor=0.1, patience=5)
criterion = nn.MSELoss()

def contrastive_loss(neural_embeddings, state_embeddings, batch_y, margin=1.0):
    TIME_UNIT = 40000
    # Compute all-pairs distance between embeddings, shape (batch, batch)
    diff = neural_embeddings.unsqueeze(1) - state_embeddings.unsqueeze(0)  # [batch, batch, embdim]
    dists = torch.norm(diff, dim=2)  # [batch, batch]

    # Create label difference matrix (absolute difference of y-value, broadcasted)
    label_diff = torch.abs(batch_y.unsqueeze(1) - batch_y.unsqueeze(0))  # [batch, batch]
    # Normalize label_diff to [0, 1]
    # label_diff = label_diff / label_diff.max().clamp(min=1e-8)

    # Contrastive loss: push apart pairs with large y-difference
    positive_mask = (label_diff < TIME_UNIT * 3 + 1)  # "close" y
    negative_mask = (label_diff > TIME_UNIT * 25 - 1) # "far" y

    positive_loss = (dists * positive_mask.float()).sum() / (positive_mask.float().sum() + 1e-8)
    negative_loss = ((margin - dists).clamp(min=0) * negative_mask.float()).sum() / (negative_mask.float().sum() + 1e-8)
    total_contrastive = positive_loss + negative_loss


    return total_contrastive

def calculate_loss(batch_neural_x, batch_state_x, batch_y):
    neural_embedding = neural_model.encoder(batch_neural_x)
    state_embedding = state_model.encoder(batch_state_x)
    state_reconstruction = state_model.decoder(state_embedding)
    neural_reconstruction = neural_model.decoder(neural_embedding)
    neural_recon_loss = criterion(neural_reconstruction, batch_neural_x)
    state_recon_loss = criterion(state_reconstruction, batch_state_x)
    contr_loss = contrastive_loss(neural_embedding, state_embedding, batch_y)
    loss = neural_recon_loss + state_recon_loss + contrast_weight * contr_loss
    return loss, neural_recon_loss, state_recon_loss, contr_loss

with torch.no_grad():
    running_test_loss = []
    running_test_neural_recon_loss = []
    running_test_state_recon_loss = []
    running_test_contr_loss = []
    for batch_neural_x, batch_state_x, batch_y in tqdm(test_loader):
        loss, neural_recon_loss, state_recon_loss, contr_loss = calculate_loss(batch_neural_x, batch_state_x, batch_y)
        running_test_loss.append(loss.item() * len(batch_neural_x))
        running_test_neural_recon_loss.append(neural_recon_loss.item() * len(batch_neural_x))
        running_test_state_recon_loss.append(state_recon_loss.item() * len(batch_state_x))
        running_test_contr_loss.append(contr_loss.item() * len(batch_neural_x))
    print(f"Test Loss: {np.sum(running_test_loss) / len(test_dataset)}")
    print(f"Test Neural Recon Loss: {np.sum(running_test_neural_recon_loss) / len(test_dataset)}")
    print(f"Test State Recon Loss: {np.sum(running_test_state_recon_loss) / len(test_dataset)}")
    print(f"Test Contrast Loss: {np.sum(running_test_contr_loss) / len(test_dataset)}")

for epoch in range(250):
    print("Last LR: ", scheduler.get_last_lr())
    running_train_loss = []
    running_train_neural_recon_loss = []
    running_train_state_recon_loss = []
    running_train_contr_loss = []
    running_test_loss = []
    running_test_neural_recon_loss = []
    running_test_state_recon_loss = []
    running_test_contr_loss = []
    for batch_neural_x, batch_state_x, batch_y in tqdm(train_loader):
        optimizer.zero_grad()
        # Forward through encoder and decoder
        loss, neural_recon_loss, state_recon_loss, contr_loss = calculate_loss(batch_neural_x, batch_state_x, batch_y)
        loss.backward()
        optimizer.step()
        running_train_loss.append(loss.item() * len(batch_neural_x))
        running_train_neural_recon_loss.append(neural_recon_loss.item() * len(batch_neural_x))
        running_train_state_recon_loss.append(state_recon_loss.item() * len(batch_state_x))
        running_train_contr_loss.append(contr_loss.item() * len(batch_neural_x))
    scheduler.step(np.sum(running_train_loss) / len(train_dataset))

    with torch.no_grad():
        for batch_neural_x, batch_state_x, batch_y in tqdm(test_loader):
            loss, neural_recon_loss, state_recon_loss, contr_loss = calculate_loss(batch_neural_x, batch_state_x, batch_y)
            running_test_loss.append(loss.item() * len(batch_neural_x))
            running_test_neural_recon_loss.append(neural_recon_loss.item() * len(batch_neural_x))
            running_test_state_recon_loss.append(state_recon_loss.item() * len(batch_state_x))
            running_test_contr_loss.append(contr_loss.item() * len(batch_neural_x))
    print(f"Epoch {epoch}, Train Loss: {np.sum(running_train_loss) / len(train_dataset)}, Test Loss: {np.sum(running_test_loss) / len(test_dataset)}")
    print(f"Train Neural Recon Loss: {np.sum(running_train_neural_recon_loss) / len(train_dataset)}, Test Neural Recon Loss: {np.sum(running_test_neural_recon_loss) / len(test_dataset)}")
    print(f"Train State Recon Loss: {np.sum(running_train_state_recon_loss) / len(train_dataset)}, Test State Recon Loss: {np.sum(running_test_state_recon_loss) / len(test_dataset)}")
    print(f"Train Contrast Loss: {np.sum(running_train_contr_loss) / len(train_dataset)}, Test Contrast Loss: {np.sum(running_test_contr_loss) / len(test_dataset)}")

    train_loss.append(np.sum(running_train_loss) / len(train_dataset))
    train_neural_recon_loss.append(np.sum(running_train_neural_recon_loss) / len(train_dataset))
    train_state_recon_loss.append(np.sum(running_train_state_recon_loss) / len(train_dataset))
    train_contr_loss.append(np.sum(running_train_contr_loss) / len(train_dataset))
    test_loss.append(np.sum(running_test_loss) / len(test_dataset))
    test_neural_recon_loss.append(np.sum(running_test_neural_recon_loss) / len(test_dataset))
    test_state_recon_loss.append(np.sum(running_test_state_recon_loss) / len(test_dataset))
    test_contr_loss.append(np.sum(running_test_contr_loss) / len(test_dataset))

if USE_LINEAR_AUTOENCODER:
    torch.save(neural_model.state_dict(), "contrastive_neural_linear_model.pth")
    torch.save(state_model.state_dict(), "contrastive_state_linear_model.pth")
else:
    torch.save(neural_model.state_dict(), "contrastive_neural_model.pth")
    torch.save(state_model.state_dict(), "contrastive_state_model.pth")

torch.save(shuffled_indices, "shuffled_indices.pt")

plt.plot(train_loss)
plt.plot(test_loss)
plt.show()