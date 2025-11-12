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
    all_embeddings_nonlinear = model_nonlinear.encoder(all_test_x).cpu().numpy()
    all_embeddings_linear = model_linear.encoder(all_test_x).cpu().numpy()

# Flatten the input for mutual info computation (samples, features)
all_test_x_np = all_test_x.cpu().numpy()
num_samples = all_test_x_np.shape[0]
flat_test_x = all_test_x_np.reshape(num_samples, -1)

# Fit PCA on test set and project
test_pca = PCA(n_components=2)
pca_2_z = test_pca.fit_transform(flat_test_x)

# Fit PCA on test set and project
test_pca = PCA(n_components=9)
pca_9_z = test_pca.fit_transform(flat_test_x)

# Compute mutual information between original data and embeddings
# We'll compute mean MI across dimensions

def evaluate_embedding(X_orig, X_emb, y_labels=None):
    metrics = {}
    # A. Pairwise correlation
    from scipy.spatial.distance import pdist
    D_orig = pdist(X_orig, metric='euclidean')
    D_emb  = pdist(X_emb,  metric='euclidean')
    metrics['distance_corr'] = np.corrcoef(D_orig, D_emb)[0,1]

    # B. Trustworthiness
    from sklearn.manifold import trustworthiness
    metrics['trustworthiness'] = trustworthiness(X_orig, X_emb, n_neighbors=10)

    from npeet import entropy_estimators as ee
    I = ee.mi(X_orig.tolist(), X_emb.tolist())
    metrics['mutual_information'] = I


    return metrics

embedding_metrics_nonlinear = evaluate_embedding(flat_test_x, all_embeddings_nonlinear)
embedding_metrics_linear = evaluate_embedding(flat_test_x, all_embeddings_linear)
pca_metrics = evaluate_embedding(flat_test_x, pca_2_z)
pca_9_metrics = evaluate_embedding(flat_test_x, pca_9_z)

import matplotlib.pyplot as plt

metric_names = ["distance_corr", "trustworthiness", "mutual_information"]
methods = ["Embedding NL", "Embedding L", "PCA 2D", "PCA 9D"]
all_metrics = [embedding_metrics_nonlinear, embedding_metrics_linear, pca_metrics, pca_9_metrics]

# Prepare data for bar plots
metric_values = {metric: [m[metric] for m in all_metrics] for metric in metric_names}

fig, axs = plt.subplots(1, 3, figsize=(15,5))
for i, metric in enumerate(metric_names):
    axs[i].bar(methods, metric_values[metric], color=['#4C72B0', '#55A868', '#C44E52', '#C44E52'])
    axs[i].set_title(metric.replace("_", " ").title())
    axs[i].set_ylabel(metric.replace("_", " ").title())
    axs[i].set_ylim([0, 1.05 * max(metric_values[metric])])  # give space above bars for clarity

print("Embedding NL: ", embedding_metrics_nonlinear)
print("Embedding L: ", embedding_metrics_linear)
print("PCA 2D: ", pca_metrics)
print("PCA 9D: ", pca_9_metrics)

plt.tight_layout()
plt.show()

