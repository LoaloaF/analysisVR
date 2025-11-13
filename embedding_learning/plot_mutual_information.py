import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from models import AutoEncoder, LinearAutoEncoder
from npeet import entropy_estimators as ee

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

# Save original unnormalized neural features and state vectors for MI computation
test_neural_x_original = test_neural_x.clone().numpy()
test_state_x_original = test_state_x.clone().numpy()

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

# Calculate mutual information between each neural feature and embedding vectors
print(f"\nCalculating mutual information for {len(neural_feature_names)} neural features...")
print("Computing MI with embeddings...")
print("Note: Negative MI values (estimation errors) will be clipped to 0")

# Store MI values: one for each neural feature with the full embedding vector
mi_values_embedding = []

# Also calculate MI for each embedding dimension separately
mi_by_dimension_embedding = [[] for _ in range(EMBEDDING_DIM)]

# Track negative values (estimation errors)
negative_count_embedding = 0

for i, feature_name in enumerate(tqdm(neural_feature_names)):
    # Get neural feature values (use original unnormalized values)
    neural_feature = test_neural_x_original[:, i].reshape(-1, 1)
    
    # Calculate MI between neural feature and full embedding vector
    # Convert to list format required by npeet
    neural_feature_list = neural_feature.tolist()
    embedding_list = test_embeddings.tolist()
    
    mi_full = ee.mi(neural_feature_list, embedding_list)
    # Clip negative values to 0 (MI should be non-negative, negatives are estimation errors)
    if mi_full < 0:
        negative_count_embedding += 1
    mi_full = max(0.0, mi_full)
    mi_values_embedding.append(mi_full)
    
    # Calculate MI for each embedding dimension separately
    for dim in range(EMBEDDING_DIM):
        embedding_dim = test_embeddings[:, dim].reshape(-1, 1)
        embedding_dim_list = embedding_dim.tolist()
        mi_dim = ee.mi(neural_feature_list, embedding_dim_list)
        # Clip negative values to 0
        if mi_dim < 0:
            negative_count_embedding += 1
        mi_dim = max(0.0, mi_dim)
        mi_by_dimension_embedding[dim].append(mi_dim)

# Convert to numpy arrays for easier plotting
mi_values_embedding = np.array(mi_values_embedding)
mi_by_dimension_embedding = np.array(mi_by_dimension_embedding)

# Calculate mutual information between each neural feature and original state vectors
print("\nComputing MI with original state vectors...")

# Store MI values: one for each neural feature with the full original state vector
mi_values_original = []

# Also calculate MI for each state dimension separately
state_dim = test_state_x_original.shape[1]
mi_by_dimension_original = [[] for _ in range(state_dim)]

# Track negative values (estimation errors)
negative_count_original = 0

for i, feature_name in enumerate(tqdm(neural_feature_names)):
    # Get neural feature values (use original unnormalized values)
    neural_feature = test_neural_x_original[:, i].reshape(-1, 1)
    
    # Calculate MI between neural feature and full original state vector
    neural_feature_list = neural_feature.tolist()
    state_list = test_state_x_original.tolist()
    
    mi_full = ee.mi(neural_feature_list, state_list)
    # Clip negative values to 0 (MI should be non-negative, negatives are estimation errors)
    if mi_full < 0:
        negative_count_original += 1
    mi_full = max(0.0, mi_full)
    mi_values_original.append(mi_full)
    
    # Calculate MI for each state dimension separately
    for dim in range(state_dim):
        state_dim_vec = test_state_x_original[:, dim].reshape(-1, 1)
        state_dim_list = state_dim_vec.tolist()
        mi_dim = ee.mi(neural_feature_list, state_dim_list)
        # Clip negative values to 0
        if mi_dim < 0:
            negative_count_original += 1
        mi_dim = max(0.0, mi_dim)
        mi_by_dimension_original[dim].append(mi_dim)

# Convert to numpy arrays for easier plotting
mi_values_original = np.array(mi_values_original)
mi_by_dimension_original = np.array(mi_by_dimension_original)

# Create output directory for plots
os.makedirs("embedding_plots", exist_ok=True)

# Plot 1: Comparison of MI between each neural feature and full vectors (original vs embedding)
plt.figure(figsize=(14, 8))
# Sort by embedding MI value (descending)
sorted_indices = np.argsort(mi_values_embedding)[::-1]
sorted_mi_embedding = mi_values_embedding[sorted_indices]
sorted_mi_original = mi_values_original[sorted_indices]
sorted_names = [neural_feature_names[i] for i in sorted_indices]

x_pos = np.arange(len(sorted_names))
width = 0.35
plt.barh(x_pos - width/2, sorted_mi_original, width, label='Original State Vectors', color='steelblue', alpha=0.8)
plt.barh(x_pos + width/2, sorted_mi_embedding, width, label='Embedding Vectors', color='coral', alpha=0.8)
plt.yticks(x_pos, sorted_names, fontsize=8)
plt.xlabel('Mutual Information', fontsize=12)
plt.ylabel('Neural Feature', fontsize=12)
plt.title('Mutual Information: Neural Features vs Original State Vectors vs Embedding Vectors', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("embedding_plots/mutual_information_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 1b: MI between each neural feature and full embedding vector (original plot)
plt.figure(figsize=(14, 8))
sorted_indices = np.argsort(mi_values_embedding)[::-1]
sorted_mi = mi_values_embedding[sorted_indices]
sorted_names = [neural_feature_names[i] for i in sorted_indices]

plt.barh(range(len(sorted_names)), sorted_mi, color='steelblue')
plt.yticks(range(len(sorted_names)), sorted_names, fontsize=8)
plt.xlabel('Mutual Information', fontsize=12)
plt.ylabel('Neural Feature', fontsize=12)
plt.title('Mutual Information between Neural Features and Embedding Vectors', fontsize=14)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("embedding_plots/mutual_information_full_embedding.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 1c: MI between each neural feature and original state vectors
plt.figure(figsize=(14, 8))
sorted_indices = np.argsort(mi_values_original)[::-1]
sorted_mi = mi_values_original[sorted_indices]
sorted_names = [neural_feature_names[i] for i in sorted_indices]

plt.barh(range(len(sorted_names)), sorted_mi, color='steelblue')
plt.yticks(range(len(sorted_names)), sorted_names, fontsize=8)
plt.xlabel('Mutual Information', fontsize=12)
plt.ylabel('Neural Feature', fontsize=12)
plt.title('Mutual Information between Neural Features and Original State Vectors', fontsize=14)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("embedding_plots/mutual_information_full_original.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: MI for each embedding dimension separately
fig, axes = plt.subplots(1, EMBEDDING_DIM, figsize=(5*EMBEDDING_DIM, 8))
if EMBEDDING_DIM == 1:
    axes = [axes]

for dim in range(EMBEDDING_DIM):
    sorted_indices_dim = np.argsort(mi_by_dimension_embedding[dim])[::-1]
    sorted_mi_dim = mi_by_dimension_embedding[dim][sorted_indices_dim]
    sorted_names_dim = [neural_feature_names[i] for i in sorted_indices_dim]
    
    axes[dim].barh(range(len(sorted_names_dim)), sorted_mi_dim, color='steelblue')
    axes[dim].set_yticks(range(len(sorted_names_dim)))
    axes[dim].set_yticklabels(sorted_names_dim, fontsize=8)
    axes[dim].set_xlabel('Mutual Information', fontsize=12)
    axes[dim].set_ylabel('Neural Feature', fontsize=12)
    axes[dim].set_title(f'MI with Embedding Dimension {dim+1}', fontsize=12)
    axes[dim].grid(True, alpha=0.3, axis='x')

plt.suptitle('Mutual Information between Neural Features and Each Embedding Dimension', fontsize=14)
plt.tight_layout()
plt.savefig("embedding_plots/mutual_information_by_dimension.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Heatmap showing MI for each feature-dimension pair (embeddings)
plt.figure(figsize=(12, max(8, len(neural_feature_names) * 0.3)))
plt.imshow(mi_by_dimension_embedding, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(label='Mutual Information')
plt.xlabel('Embedding Dimension', fontsize=12)
plt.ylabel('Neural Feature', fontsize=12)
plt.title('Mutual Information Heatmap: Neural Features vs Embedding Dimensions', fontsize=14)
plt.yticks(range(len(neural_feature_names)), neural_feature_names, fontsize=8)
plt.xticks(range(EMBEDDING_DIM), [f'Dim {i+1}' for i in range(EMBEDDING_DIM)], fontsize=10)
plt.tight_layout()
plt.savefig("embedding_plots/mutual_information_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 4: Heatmap showing MI for each feature-dimension pair (original state vectors)
plt.figure(figsize=(12, max(8, len(neural_feature_names) * 0.3)))
plt.imshow(mi_by_dimension_original, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(label='Mutual Information')
plt.xlabel('State Dimension', fontsize=12)
plt.ylabel('Neural Feature', fontsize=12)
plt.title('Mutual Information Heatmap: Neural Features vs Original State Dimensions', fontsize=14)
plt.yticks(range(len(neural_feature_names)), neural_feature_names, fontsize=8)
plt.xticks(range(state_dim), [f'State {i+1}' for i in range(state_dim)], fontsize=10)
plt.tight_layout()
plt.savefig("embedding_plots/mutual_information_heatmap_original.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 5: Scatter plot comparing MI values (original vs embedding)
plt.figure(figsize=(10, 8))
plt.scatter(mi_values_original, mi_values_embedding, alpha=0.6, s=50)
# Add diagonal line
max_val = max(np.max(mi_values_original), np.max(mi_values_embedding))
min_val = min(np.min(mi_values_original), np.min(mi_values_embedding))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
plt.xlabel('MI with Original State Vectors', fontsize=12)
plt.ylabel('MI with Embedding Vectors', fontsize=12)
plt.title('Mutual Information: Original State Vectors vs Embedding Vectors', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("embedding_plots/mutual_information_scatter_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# Print summary statistics
print(f"\nMutual Information Summary:")
print(f"\n--- Embedding Vectors ---")
print(f"Mean MI (full embedding): {np.mean(mi_values_embedding):.4f}")
print(f"Std MI (full embedding): {np.std(mi_values_embedding):.4f}")
print(f"Max MI (full embedding): {np.max(mi_values_embedding):.4f}")
print(f"Min MI (full embedding): {np.min(mi_values_embedding):.4f}")
total_embedding_calculations = len(neural_feature_names) * (1 + EMBEDDING_DIM)
print(f"Negative MI values clipped (estimation errors): {negative_count_embedding}/{total_embedding_calculations} ({100*negative_count_embedding/total_embedding_calculations:.1f}%)")

print(f"\n--- Original State Vectors ---")
print(f"Mean MI (full original): {np.mean(mi_values_original):.4f}")
print(f"Std MI (full original): {np.std(mi_values_original):.4f}")
print(f"Max MI (full original): {np.max(mi_values_original):.4f}")
print(f"Min MI (full original): {np.min(mi_values_original):.4f}")
total_original_calculations = len(neural_feature_names) * (1 + state_dim)
print(f"Negative MI values clipped (estimation errors): {negative_count_original}/{total_original_calculations} ({100*negative_count_original/total_original_calculations:.1f}%)")

print(f"\n--- Comparison ---")
print(f"Mean MI ratio (embedding/original): {np.mean(mi_values_embedding / (mi_values_original + 1e-10)):.4f}")
print(f"Correlation between original and embedding MI: {np.corrcoef(mi_values_original, mi_values_embedding)[0,1]:.4f}")

print(f"\nTop 5 neural features by MI (Embedding):")
top5_indices = np.argsort(mi_values_embedding)[::-1][:5]
for idx in top5_indices:
    print(f"  {neural_feature_names[idx]}: Embedding={mi_values_embedding[idx]:.4f}, Original={mi_values_original[idx]:.4f}")

print(f"\nTop 5 neural features by MI (Original):")
top5_indices_orig = np.argsort(mi_values_original)[::-1][:5]
for idx in top5_indices_orig:
    print(f"  {neural_feature_names[idx]}: Embedding={mi_values_embedding[idx]:.4f}, Original={mi_values_original[idx]:.4f}")

print(f"\nAll plots saved to 'embedding_plots' directory")

