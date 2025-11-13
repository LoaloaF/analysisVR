import torch
import numpy as np
import pandas as pd
from models import AutoEncoder

# Load model
model = AutoEncoder(input_size=9, hidden_size=20, output_size=2)
model.load_state_dict(torch.load('contrastive_neural_labeling_model.pth'))
model.eval()

# Load and process data
data = pd.read_csv('merged.csv')
state_x = torch.from_numpy(data.iloc[:, 1:10].values).float()
shuffled_indices = torch.load('shuffled_indices.pt')
shuffled_state_x = state_x[shuffled_indices]

# Split and normalize
train_state_x = shuffled_state_x[:int(0.8*len(shuffled_state_x))]
test_state_x = shuffled_state_x[int(0.8*len(shuffled_state_x)):]

mean = train_state_x.mean(dim=0, keepdim=True)
std = train_state_x.std(dim=0, keepdim=True) + 1e-8
test_state_x = (test_state_x - mean) / std

# Get embeddings
with torch.no_grad():
    emb = model.encoder(test_state_x).numpy()

print('Embedding Statistics:')
print(f'  Dim 0: mean={emb[:,0].mean():.4f}, std={emb[:,0].std():.4f}, min={emb[:,0].min():.4f}, max={emb[:,0].max():.4f}')
print(f'  Dim 1: mean={emb[:,1].mean():.4f}, std={emb[:,1].std():.4f}, min={emb[:,1].min():.4f}, max={emb[:,1].max():.4f}')
print(f'  Correlation between dims: {np.corrcoef(emb[:,0], emb[:,1])[0,1]:.4f}')
print(f'  Variance ratio (dim1/dim0): {(emb[:,1].std()**2)/(emb[:,0].std()**2 + 1e-8):.4f}')

# Check if one dimension has very low variance
if emb[:,1].std() < 0.01 * emb[:,0].std():
    print('\n⚠️  WARNING: Dimension 1 has very low variance compared to dimension 0!')
    print('   This suggests the embedding has collapsed to 1D.')

# Check correlation
if abs(np.corrcoef(emb[:,0], emb[:,1])[0,1]) > 0.9:
    print('\n⚠️  WARNING: High correlation between dimensions!')
    print('   This suggests the embedding is effectively 1D.')



