from models import AutoEncoder, LinearAutoEncoder
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

model_nonlinear = AutoEncoder(input_size=x.shape[1], hidden_size=hidden_size, output_size=2)

model_nonlinear.load_state_dict(torch.load("contrastive_model.pth"))
model_linear = LinearAutoEncoder(input_size=x.shape[1], output_size=2)
model_linear.load_state_dict(torch.load("contrastive_linear_model.pth"))
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


# Compute embeddings and PCA projections for the whole test set
with torch.no_grad():
    all_test_x = []
    all_test_y = []
    for batch_x, batch_y in test_loader:
        all_test_x.append(batch_x)
        all_test_y.append(batch_y)
    all_test_x = torch.cat(all_test_x, dim=0)
    all_test_y = torch.cat(all_test_y, dim=0)
    all_reconstruction_nonlinear = model_nonlinear.decoder(model_nonlinear.encoder(all_test_x)).cpu().numpy()
    all_reconstruction_linear = model_linear.decoder(model_linear.encoder(all_test_x)).cpu().numpy()

# Flatten the input for mutual info computation (samples, features)
all_test_x_np = all_test_x.cpu().numpy()
num_samples = all_test_x_np.shape[0]
flat_test_x = all_test_x_np.reshape(num_samples, -1)

# Perform PCA (2 components), then reconstruct the test_x from the compressed representation
test_pca = PCA(n_components=2)
pca_2_z = test_pca.fit_transform(flat_test_x)
pca_2_reconstruction = test_pca.inverse_transform(pca_2_z)

# Perform PCA (2 components), then reconstruct the test_x from the compressed representation
test_pca = PCA(n_components=9)
pca_9_z = test_pca.fit_transform(flat_test_x)
pca_9_reconstruction = test_pca.inverse_transform(pca_9_z)

# Calculate the featurewise loss between the original test_x and the reconstructed test_x
featurewise_loss_nonlinear = np.mean(np.abs(all_test_x_np - all_reconstruction_nonlinear), axis=0)
featurewise_loss_linear = np.mean(np.abs(all_test_x_np - all_reconstruction_linear), axis=0)
featurewise_loss_pca_2 = np.mean(np.abs(all_test_x_np - pca_2_reconstruction), axis=0)
featurewise_loss_pca_9 = np.mean(np.abs(all_test_x_np - pca_9_reconstruction), axis=0)

# Plot the featurewise loss
plt.figure(figsize=(10, 5))
plt.plot(featurewise_loss_nonlinear, label='Nonlinear')
plt.plot(featurewise_loss_linear, label='Linear')
plt.plot(featurewise_loss_pca_2, label='PCA 2')
plt.plot(featurewise_loss_pca_9, label='PCA 9')
plt.legend()
plt.xlabel('Feature')
plt.ylabel('Featurewise Loss')
plt.title('Featurewise Loss of Reconstructed Test Data')
plt.savefig('featurewise_loss.png')
plt.show()