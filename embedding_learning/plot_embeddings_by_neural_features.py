import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import os

from models import AutoEncoder, LinearAutoEncoder

# Configuration - should match training script
USE_LINEAR_AUTOENCODER = False
USE_OLD_SHUFFLED_INDICES = True
EMBEDDING_DIM = 3
hidden_size = 50

# Load data
data = pd.read_csv("merged.csv")

neural_x = data.iloc[:, 12:].values
state_x = data.iloc[:, 1:10].values
y = data.iloc[:, 10].values

# Get neural feature names (column names starting from column 12)
neural_feature_names = data.columns[12:].tolist()

neural_x = torch.from_numpy(neural_x).float()
state_x = torch.from_numpy(state_x).float()
y = torch.from_numpy(y).float()

# Load shuffled indices if available
if USE_OLD_SHUFFLED_INDICES and os.path.exists("shuffled_indices.pt"):
    shuffled_indices = torch.load("shuffled_indices.pt")
else:
    shuffled_indices = torch.randperm(len(neural_x))

shuffled_neural_x = neural_x[shuffled_indices]
shuffled_state_x = state_x[shuffled_indices]
shuffled_y = y[shuffled_indices]

# Split into train and test (same as training script)
train_neural_x = shuffled_neural_x[:int(0.8*len(shuffled_neural_x))]
train_state_x = shuffled_state_x[:int(0.8*len(shuffled_state_x))]
train_y = shuffled_y[:int(0.8*len(shuffled_y))]
test_neural_x = shuffled_neural_x[int(0.8*len(shuffled_neural_x)):]
test_state_x = shuffled_state_x[int(0.8*len(shuffled_state_x)):]
test_y = shuffled_y[int(0.8*len(shuffled_y)):]

# Save original unnormalized neural features for visualization
test_neural_x_original = test_neural_x.clone().numpy()

# Normalize using training statistics (same as training script)
mean = train_neural_x.mean(dim=0, keepdim=True)
std_neural = train_neural_x.std(dim=0, keepdim=True) + 1e-8
train_neural_x = (train_neural_x - mean) / std_neural
test_neural_x = (test_neural_x - mean) / std_neural

mean = train_state_x.mean(dim=0, keepdim=True)
std_state = train_state_x.std(dim=0, keepdim=True) + 1e-8
train_state_x = (train_state_x - mean) / std_state
test_state_x = (test_state_x - mean) / std_state

# Initialize model (same architecture as training)
if USE_LINEAR_AUTOENCODER:
    model = LinearAutoEncoder(input_size=state_x.shape[1], output_size=EMBEDDING_DIM)
    model_path = "contrastive_neural_labeling_linear_model.pth"
else:
    model = AutoEncoder(input_size=state_x.shape[1], hidden_size=hidden_size, output_size=EMBEDDING_DIM)
    model_path = "contrastive_neural_labeling_model.pth"

# Load trained model weights
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")
else:
    raise FileNotFoundError(f"Model file not found: {model_path}")

model.eval()

# Generate embeddings for test set
print("Generating embeddings for test set...")
with torch.no_grad():
    test_embeddings = model.encoder(test_state_x).numpy()

print(f"Embeddings shape: {test_embeddings.shape}")
print(f"Original neural features shape: {test_neural_x_original.shape}")
print(f"Number of neural features: {len(neural_feature_names)}")

# Create output directory for plots
os.makedirs("embedding_plots", exist_ok=True)

# Plot embeddings color-coded by each neural feature
print(f"\nGenerating plots for {len(neural_feature_names)} neural features...")
for i, feature_name in enumerate(tqdm(neural_feature_names)):
    # Use original unnormalized values for better interpretability
    feature_values = test_neural_x_original[:, i]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(test_embeddings[:, 0], test_embeddings[:, 1], test_embeddings[:, 2],
                         c=feature_values, cmap='viridis', s=1, alpha=0.6)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(f'{feature_name}', fontsize=12)
    ax.set_xlabel('Embedding Dimension 1', fontsize=12)
    ax.set_ylabel('Embedding Dimension 2', fontsize=12)
    ax.set_zlabel('Embedding Dimension 3', fontsize=12)
    ax.set_title(f'Latent Space Embeddings Colored by {feature_name}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Save plot
    safe_feature_name = feature_name.replace('/', '_').replace('\\', '_')
    plt.savefig(f"embedding_plots/embedding_{safe_feature_name}.png", dpi=150, bbox_inches='tight')
    plt.close()

print(f"\nAll plots saved to 'embedding_plots' directory")
print(f"Generated {len(neural_feature_names)} plots")

