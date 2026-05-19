"""
Train an MLP on a high-performing session-neuron pair and produce two
complementary visualizations at each checkpoint epoch:

  scatter — 3D PCA of the *embedding* space (last hidden layer), two viewing
            angles, color-coded by true neuron activity.  Shows how the
            geometry of the learned representation evolves during training.

  heatmap — 2D PCA of the *raw inputs* (fixed axes), side-by-side maps of
            true vs predicted activity binned across the input manifold.
            Shows how well the model's prediction surface covers the input
            space at each epoch.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import pickle
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

import torch.nn as nn
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from models import MLP

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("WARNING: imageio not installed. GIF creation will be skipped.")
    print("Install with: pip install imageio")

# ============================================================================
# CONFIGURATION
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--split_path", type=str, default="test_indices_by_session.npy",
                    help="Path to train/test split file")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
parser.add_argument("--use_ensembles", action="store_true", default=False,
                    help="Predict ensemble activity instead of single-unit spikes")
parser.add_argument("--models_dir", type=str, default=None,
                    help="Directory with pre-trained models for auto-selection")
parser.add_argument("--session_idx", type=int, default=None,
                    help="Specific session index (optional, for manual selection)")
parser.add_argument("--neuron_idx", type=int, default=None,
                    help="Specific neuron index (optional, for manual selection)")
parser.add_argument("--output_dir", type=str, default="./outputs/pca_visualizations",
                    help="Directory to save PCA visualizations")
parser.add_argument("--num_epochs", type=int, default=100,
                    help="Number of training epochs")
parser.add_argument("--save_every", type=int, default=5,
                    help="Save visualization every N epochs")
parser.add_argument("--make_gif", action="store_true", default=False,
                    help="Create animated GIF from visualization frames")
parser.add_argument("--gif_fps", type=int, default=3,
                    help="Frames per second for GIF animation (default: 3)")
args = parser.parse_args()

USE_ENSEMBLES = args.use_ensembles
os.makedirs(args.output_dir, exist_ok=True)

# ============================================================================
# LOAD DATA (same as train_mlp.py)
# ============================================================================

print("Loading data...")
base = "./outputs/glm_input_data/"


beh_vals = np.load(os.path.join(base, "behavior_glm_input.npy"), allow_pickle=True)
beh_idx  = np.load(os.path.join(base, "behavior_glm_input_index.npy"), allow_pickle=True)
beh_cols = np.load(os.path.join(base, "behavior_glm_input_columns.npy"), allow_pickle=True)

spk_vals = np.load(os.path.join(base, "fr_full.npy"))
spk_idx  = np.load(os.path.join(base, "fr_full_index.npy"), allow_pickle=True)
spk_cols = np.load(os.path.join(base, "fr_full_columns.npy"), allow_pickle=True)

ensembles_values = np.load(os.path.join(base, "ensembles.npy"))

try:
    beh_index = pd.MultiIndex.from_tuples(beh_idx, names=behavior_glm_input.index.names)
except Exception:
    beh_index = pd.Index(beh_idx)

try:
    spk_index = pd.MultiIndex.from_tuples(spk_idx, names=fr_first7.index.names)
except Exception:
    spk_index = pd.Index(spk_idx)

behavior_glm_loaded = pd.DataFrame(beh_vals, index=beh_index, columns=beh_cols)
spikes_loaded       = pd.DataFrame(spk_vals, index=spk_index, columns=spk_cols)

print("behavior_glm_loaded shape", behavior_glm_loaded.shape)
print("spikes_loaded shape",       spikes_loaded.shape)

# ── Preprocessing ──────────────────────────────────────────────────────────
spikes_loaded.index = pd.MultiIndex.from_tuples(
    spikes_loaded.index.map(lambda t: (t[0], t[1] // 40000 - 1))
)

spikes_unique_trials = set(idx[0] for idx in spikes_loaded.index)
behavior_glm_loaded  = behavior_glm_loaded[
    behavior_glm_loaded.index.map(lambda t: t[0] in spikes_unique_trials)
]

non_nan_rows        = behavior_glm_loaded.index[~behavior_glm_loaded.isna().any(axis=1)]
behavior_glm_loaded = behavior_glm_loaded.loc[non_nan_rows]
spikes_loaded       = spikes_loaded.loc[non_nan_rows]

behavior_glm_loaded = behavior_glm_loaded.drop("track_zone", axis=1)

session_ids = behavior_glm_loaded.index.map(lambda x: x[0]).unique()
print(f"{len(session_ids)} sessions")

# ── Feature columns ────────────────────────────────────────────────────────
action_enc_cols = [
    'frame_raw_500msMedian',                  # forward_velocity
    'frame_raw_abs_acc_500msMedian',          # forward_acceleration
    'frame_YawPitch_abs_vel_sum_500msMedian', # off_rotation_velocity
    'frame_YawPitch_abs_acc_sum_500msMedian', # off_rotation_acceleration
    'head_angle_vel',
    'head_angle',
    'movement_energy_smooth5',                # movement_energy
    'upcoming_choice',
    'reward_window',
    'lick_detected',
]
state_enc_cols = [
    'cue_visible',
]

zone_values      = sorted(behavior_glm_loaded['track_zone_int'].dropna().unique().astype(int))
zone_onehot_cols = [f'track_zone_{z}' for z in zone_values]
for z in zone_values:
    behavior_glm_loaded[f'track_zone_{z}'] = (behavior_glm_loaded['track_zone_int'] == z).astype(float)

# ── Build session dataset ──────────────────────────────────────────────────
session_dataset_singles = {}

for session_id in session_ids:
    session_mask   = behavior_glm_loaded.index.map(lambda x: x[0]) == session_id
    beh            = behavior_glm_loaded[session_mask]
    spikes_session = spikes_loaded[session_mask]

    for col in action_enc_cols + state_enc_cols:
        if col in beh.columns:
            beh.loc[:, col] = beh[col].astype(float)
            mean_val = beh[col].mean()
            std_val  = beh[col].std() + 1e-8
            if std_val != 0:
                beh.loc[:, col] = (beh[col] - mean_val) / std_val

    spikes_session = spikes_session.astype(float)
    if USE_ENSEMBLES:
        spikes_values = spikes_session.values
        new_spikes_session = pd.DataFrame(index=spikes_session.index)
        for i in range(ensembles_values.shape[1]):
            ensemble_col = spikes_values @ ensembles_values[:, i]
            new_spikes_session[f'ensemble_{i}'] = ensemble_col
        spikes_session = new_spikes_session

    label_stds = []
    for col in spikes_session.columns:
        mean_val = spikes_session[col].mean()
        std_val  = spikes_session[col].std() + 1e-8
        label_stds.append(std_val)
        if std_val != 0:
            spikes_session[col] = (spikes_session[col] - mean_val) / std_val

    data_by_trial, labels_by_trial = {}, {}
    for trial_id in beh["trial_id"].unique():
        trial_mask                = beh["trial_id"] == trial_id
        data_by_trial[trial_id]   = beh.loc[trial_mask, action_enc_cols + state_enc_cols + zone_onehot_cols].values.astype(np.float16)
        labels_by_trial[trial_id] = spikes_session.loc[trial_mask].values.astype(np.float16)

    session_dataset_singles[session_id] = {
        "data":       data_by_trial,
        "labels":     labels_by_trial,
        "label_stds": label_stds,
    }

print("Dataset built.")

# ── Train/test split ────────────────────────────────────────────────────────
if os.path.exists(args.split_path):
    test_indices_by_session = np.load(args.split_path, allow_pickle=True).item()
    print(f"Loaded existing split from {args.split_path}")
else:
    raise FileNotFoundError(f"Split file not found: {args.split_path}")

# ============================================================================
# FIND HIGH-PERFORMING SESSION-NEURON PAIR
# ============================================================================

print("\nSelecting session-neuron pair...")

mode_str = "ensembles" if USE_ENSEMBLES else "spikes"

# If session and neuron are manually specified, use those
if args.session_idx is not None and args.neuron_idx is not None:
    best_session_idx = args.session_idx
    best_neuron_idx = args.neuron_idx
    best_session_id = list(session_ids)[best_session_idx]
    print(f"Using manually specified pair:")
    print(f"  Session index: {best_session_idx}")
    print(f"  Neuron index: {best_neuron_idx}")
else:
    # Try to auto-detect high-performing pair from eval results
    eval_output_dir = f"./outputs/mlps/{mode_str}_multiseed"
    eval_results_path = os.path.join(eval_output_dir, "all_r2.npy")

    if os.path.exists(eval_results_path):
        print(f"Loading evaluation results from {eval_results_path}...")
        all_r2 = np.load(eval_results_path)   # (n_seeds, n_sessions, n_neurons)
        mean_r2 = all_r2.mean(axis=0)         # (n_sessions, n_neurons)
        
        # Find best overall session-neuron pair
        # Mask out pairs with very low R²
        mask = mean_r2 < 0.01
        masked_r2 = np.ma.array(mean_r2, mask=mask)
        neuron_mean_r2 = masked_r2.mean(axis=0).filled(np.nan)

        # Find best neuron (highest mean R² across sessions)
        best_neuron_idx = np.nanargmax(neuron_mean_r2)
        best_neurons = np.argsort(neuron_mean_r2)[-5:][::-1]  # Top 5

        print(f"\nTop 5 neurons by mean R²:")
        for i, neuron_idx in enumerate(best_neurons):
            print(f"  {i+1}. Neuron {neuron_idx}: mean R² = {neuron_mean_r2[neuron_idx]:.4f}")

        # Find best session for best neuron
        best_session_idx = np.nanargmax(mean_r2[:, best_neuron_idx])
        best_session_id = list(session_ids)[best_session_idx]
    else:
        raise FileNotFoundError(
            f"No eval results found at {eval_results_path}.\n"
            "Either run the eval script first, or specify a pair manually:\n"
            "  --session_idx <int> --neuron_idx <int>"
        )
        best_session_id = list(session_ids)[0]

print(f"\nSelected session-neuron pair:")
print(f"  Session ID: {best_session_id} (index {best_session_idx})")
print(f"  Neuron: {best_neuron_idx}")

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

test_session = best_session_id
neuron_idx = best_neuron_idx
seed = args.seed

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

session_data   = session_dataset_singles[test_session]["data"]
session_labels = session_dataset_singles[test_session]["labels"]

test_idx_   = test_indices_by_session[test_session]
all_indices = list(session_data.keys())
train_idx   = np.setdiff1d(all_indices, test_idx_)

train_data      = torch.tensor(
    np.concatenate([session_data[idx]   for idx in train_idx], axis=0),
    dtype=torch.float32).to(device)
train_labels_np = np.concatenate(
    [session_labels[idx] for idx in train_idx], axis=0)

print(f"\nTraining on session {test_session}, neuron {neuron_idx}")
print(f"  Train data shape: {train_data.shape}")
print(f"  Train labels shape: {train_labels_np.shape}")

train_label = torch.tensor(
    train_labels_np[:, neuron_idx], dtype=torch.float32).to(device)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

input_size        = len(action_enc_cols) + len(state_enc_cols) + len(zone_onehot_cols)
hidden_size       = 64
num_hidden_layers = 2
output_size       = 1
learning_rate     = 0.001

model     = MLP(input_size, hidden_size, num_hidden_layers, output_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# ============================================================================
# VISUALIZATION HELPER FUNCTIONS
# ============================================================================

def _create_scatter_viz(args, epoch, pca_embed, embeddings, train_labels_np,
                        neuron_idx, session_idx, vmin, vmax):
    """3D PCA scatter of the embedding space, colored by true neuron activity.

    Two panels show two viewing angles of the same 3D point cloud so the
    depth structure is visible.  Points are colored by the (fixed) true
    label — what changes across epochs is the *geometry* of the cloud as
    the network learns.
    """
    pca_coords_3d = pca_embed.transform(embeddings)
    targets = train_labels_np[:, neuron_idx]
    ev = pca_embed.explained_variance_ratio_

    fig = plt.figure(figsize=(16, 7))

    for i, (elev, azim) in enumerate([(20, 45), (20, 135)]):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        sc = ax.scatter(
            pca_coords_3d[:, 0], pca_coords_3d[:, 1], pca_coords_3d[:, 2],
            c=targets, cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
            s=8, alpha=0.5, edgecolors='none'
        )
        ax.set_xlabel(f'PC1 ({ev[0]:.1%})', fontsize=9, fontweight='bold')
        ax.set_ylabel(f'PC2 ({ev[1]:.1%})', fontsize=9, fontweight='bold')
        ax.set_zlabel(f'PC3 ({ev[2]:.1%})', fontsize=9, fontweight='bold')
        ax.set_title(f'View {i + 1}  (az={azim}°)', fontsize=10)
        ax.view_init(elev=elev, azim=azim)

    cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.5, aspect=20, pad=0.1)
    cbar.set_label('True Activity (z-scored)', fontweight='bold')
    fig.suptitle(
        f'3D Embedding PCA | Session {session_idx} | Neuron {neuron_idx} | Epoch {epoch + 1}',
        fontsize=13, fontweight='bold'
    )

    fig_path = os.path.join(args.output_dir,
                            f"scatter_epoch_{epoch+1:03d}_s{session_idx:02d}_n{neuron_idx:02d}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    if args.make_gif and IMAGEIO_AVAILABLE:
        gif_frames_scatter.append(fig_path)


def _create_heatmap_viz(args, epoch, pca_input, input_pca_coords, predictions,
                        train_labels_np, neuron_idx, session_idx, vmin, vmax):
    """2D PCA of the raw input space: side-by-side heatmaps of true vs predicted activity.

    The axes are fixed (input PCA doesn't change), so across epochs you see the
    model's prediction surface evolving to match the true activity map.
    Finer bins (40×40) give more spatial granularity than the scatter view.
    """
    targets = train_labels_np[:, neuron_idx]
    bins = 40
    ev = pca_input.explained_variance_ratio_

    count, xedges, yedges = np.histogram2d(
        input_pca_coords[:, 0], input_pca_coords[:, 1], bins=bins
    )
    h_target, _, _ = np.histogram2d(
        input_pca_coords[:, 0], input_pca_coords[:, 1],
        bins=bins, weights=targets
    )
    h_pred, _, _ = np.histogram2d(
        input_pca_coords[:, 0], input_pca_coords[:, 1],
        bins=bins, weights=predictions
    )
    h_target = np.divide(h_target, count, where=count > 0, out=np.zeros_like(h_target))
    h_pred   = np.divide(h_pred,   count, where=count > 0, out=np.zeros_like(h_pred))

    # Smooth before plotting; sigma=1.5 blurs across ~1-2 bins without destroying structure
    h_target = gaussian_filter(h_target, sigma=1.5)
    h_pred   = gaussian_filter(h_pred,   sigma=1.5)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, h, title, label in [
        (axes[0], h_target, 'True Activity',     'Mean True Activity (z-scored)'),
        (axes[1], h_pred,   'MLP Predictions',   'Mean Prediction (z-scored)'),
    ]:
        im = ax.imshow(h.T, extent=extent, origin='lower', aspect='auto',
                       cmap='RdYlBu_r', interpolation='bilinear', vmin=vmin, vmax=vmax)
        ax.set_xlabel(f'PC1 ({ev[0]:.1%} var. explained)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({ev[1]:.1%} var. explained)', fontsize=12, fontweight='bold')
        ax.set_title(f'{title} | Input PCA\nSession {session_idx} | Neuron {neuron_idx} | Epoch {epoch + 1}',
                     fontsize=13, fontweight='bold', pad=12)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(label, fontweight='bold')

    plt.tight_layout()

    fig_path = os.path.join(args.output_dir,
                            f"heatmap_epoch_{epoch+1:03d}_s{session_idx:02d}_n{neuron_idx:02d}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    if args.make_gif and IMAGEIO_AVAILABLE:
        gif_frames_heatmap.append(fig_path)

# ============================================================================
# TRAINING WITH PCA VISUALIZATION
# ============================================================================

print(f"\nTraining MLP for {args.num_epochs} epochs...")
print(f"Saving visualizations every {args.save_every} epochs to {args.output_dir}")

# 3-component PCA of the embedding space — for the 3D scatter animation
model.eval()
with torch.no_grad():
    initial_embeddings = model.embed(train_data).cpu().numpy()

pca_embed = PCA(n_components=3)
pca_embed.fit(initial_embeddings)
print(f"Embedding PCA explained variance: {pca_embed.explained_variance_ratio_}")

# 2-component PCA of the raw inputs — fixed axes for the heatmap
train_data_np = train_data.cpu().numpy()
pca_input = PCA(n_components=2)
pca_input.fit(train_data_np)
input_pca_coords = pca_input.transform(train_data_np)
print(f"Input PCA explained variance: {pca_input.explained_variance_ratio_}")

# Fixed color scale from the true labels so all frames are comparable
color_vmin = float(train_labels_np[:, neuron_idx].min())
color_vmax = float(train_labels_np[:, neuron_idx].max())

# Store training history
train_losses = []
epoch_visualizations = []
gif_frames_scatter = []  # For scatter plot GIF
gif_frames_heatmap = []   # For heatmap GIF

for epoch in range(args.num_epochs):
    model.train()
    optimizer.zero_grad()
    _, outputs = model(train_data)
    loss = criterion(outputs.squeeze(), train_label)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    # Create visualization every N epochs
    if (epoch + 1) % args.save_every == 0:
        print(f"  Epoch {epoch+1:3d} | Loss: {loss.item():.4f}")
        
        model.eval()
        with torch.no_grad():
            embeddings  = model.embed(train_data).cpu().numpy()
            predictions = model(train_data)[1].squeeze().cpu().numpy()

        _create_scatter_viz(args, epoch, pca_embed, embeddings,
                            train_labels_np, neuron_idx, best_session_idx,
                            color_vmin, color_vmax)

        _create_heatmap_viz(args, epoch, pca_input, input_pca_coords, predictions,
                            train_labels_np, neuron_idx, best_session_idx,
                            color_vmin, color_vmax)
        
        epoch_visualizations.append({
            'epoch': epoch + 1,
            'loss': loss.item(),
            'target': train_labels_np[:, neuron_idx],
            'predictions': predictions,
        })

print("\nTraining complete!")

# ============================================================================
# SAVE RESULTS AND METADATA
# ============================================================================

# Save training metadata
metadata = {
    'session_id': test_session,
    'session_idx': best_session_idx,
    'neuron_idx': neuron_idx,
    'seed': seed,
    'input_size': input_size,
    'hidden_size': hidden_size,
    'num_hidden_layers': num_hidden_layers,
    'learning_rate': learning_rate,
    'num_epochs': args.num_epochs,
    'losses': train_losses,
    'pca_embed_components': pca_embed.components_,
    'pca_embed_explained_variance': pca_embed.explained_variance_ratio_,
    'pca_input_components': pca_input.components_,
    'pca_input_explained_variance': pca_input.explained_variance_ratio_,
}

metadata_path = os.path.join(args.output_dir, 
                            f"metadata_s{best_session_idx:02d}_n{neuron_idx:02d}.pkl")
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

print(f"Metadata saved to {metadata_path}")

# Save model
model_path = os.path.join(args.output_dir, 
                         f"model_s{best_session_idx:02d}_n{neuron_idx:02d}.pt")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Create summary plot of loss over epochs
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_losses, linewidth=2.5, color='steelblue', label='Training MSE Loss')
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
ax.set_title(f'MLP Training Convergence: MSE Loss Trajectory\nSession {best_session_id} (Index {best_session_idx}) | Neuron {neuron_idx} | Seed {seed}', 
             fontsize=13, fontweight='bold', pad=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
plt.tight_layout()

summary_path = os.path.join(args.output_dir, 
                           f"loss_curve_s{best_session_idx:02d}_n{neuron_idx:02d}.png")
plt.savefig(summary_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"Summary plot saved to {summary_path}")

# ============================================================================
# CREATE ANIMATED GIFS
# ============================================================================

if args.make_gif and IMAGEIO_AVAILABLE:
    print("\nCreating animated GIFs...")

    if gif_frames_scatter:
        try:
            scatter_gif_path = os.path.join(args.output_dir,
                                            f"scatter_animation_s{best_session_idx:02d}_n{neuron_idx:02d}.gif")
            frames_scatter = [imageio.imread(f) for f in gif_frames_scatter]
            imageio.mimsave(scatter_gif_path, frames_scatter, fps=args.gif_fps, loop=0)
            print(f"✓ Scatter GIF saved: {scatter_gif_path}")
            print(f"  ({len(gif_frames_scatter)} frames, {args.gif_fps} fps)")
        except Exception as e:
            print(f"✗ Failed to create scatter GIF: {e}")

    if gif_frames_heatmap:
        try:
            heatmap_gif_path = os.path.join(args.output_dir,
                                            f"heatmap_animation_s{best_session_idx:02d}_n{neuron_idx:02d}.gif")
            frames_heatmap = [imageio.imread(f) for f in gif_frames_heatmap]
            imageio.mimsave(heatmap_gif_path, frames_heatmap, fps=args.gif_fps, loop=0)
            print(f"✓ Heatmap GIF saved: {heatmap_gif_path}")
            print(f"  ({len(gif_frames_heatmap)} frames, {args.gif_fps} fps)")
        except Exception as e:
            print(f"✗ Failed to create heatmap GIF: {e}")
elif args.make_gif and not IMAGEIO_AVAILABLE:
    print("\nWARNING: GIF creation requested but imageio not installed.")
    print("Install with: pip install imageio")

print(f"\nAll visualizations saved to: {args.output_dir}/")
