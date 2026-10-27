import numpy as np
import torch
import cebra.models


def build_windows(arr, window_len=10):
    """Sliding windows of length window_len over the time axis.

    Returns shape (n, num_features, window_len). The window at position i
    is centered at timestep i using offset (5, 5), with zero-padding at edges.
    Matches the offset10-model receptive field exactly — no CEBRA replication padding.
    """
    half = window_len // 2
    n, f = arr.shape
    padded = np.concatenate([
        np.zeros((half, f), dtype=np.float32),
        arr.astype(np.float32),
        np.zeros((half, f), dtype=np.float32),
    ], axis=0)
    windows = np.stack([padded[i:i + window_len] for i in range(n)], axis=0)
    return windows.transpose(0, 2, 1)  # (n, features, 10)


def load_encoder(checkpoint_path, device='cuda'):
    """Load a trained encoder from an encoder.pt checkpoint.

    Returns the encoder module in eval mode on the specified device,
    plus the config dicts. Works for checkpoints from either training arm.
    """
    # weights_only=False: checkpoint contains plain Python dicts/strings, not arbitrary code.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ec = ckpt['encoder_config']
    # cebra.models.init takes 'name' as the first positional arg; checkpoint stores it as 'model_architecture'
    encoder = cebra.models.init(
        ec['model_architecture'],
        num_neurons=ec['num_neurons'],
        num_units=ec['num_units'],
        num_output=ec['num_output'],
    )
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder.to(device).eval()
    return encoder, ckpt['encoder_config'], ckpt['training_config']


def embed_windows(encoder, windows, device='cuda', batch_size=1024):
    """Run a trained encoder over sliding windows.

    windows: numpy array or tensor of shape (n_windows, num_features, 10)
    Returns: numpy array of shape (n_windows, embed_dim) on CPU.
    """
    encoder.eval()
    embeds = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.as_tensor(windows[i:i+batch_size], dtype=torch.float32, device=device)
            z = encoder(batch)
            if z.dim() == 3:
                z = z.squeeze(-1)
            embeds.append(z.cpu().numpy())
    return np.concatenate(embeds, axis=0)
