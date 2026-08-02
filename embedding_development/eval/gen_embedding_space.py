"""
Embedding-space geometry: TempConv-Pred vs TempConv-Cont.

For the best-predicted (session, ensemble) pair, embed the held-out behavioral
windows through the predictive and contrastive encoders, project each 64-d
embedding to 2D with PCA, and color points by the true ensemble activity.
Shows that the two objectives organize the embedding space differently.

Output: outputs/mlps/ensembles_multiseed/embedding_space_pred_vs_cont.png
"""
import os, sys, pickle
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir("/home/amitsant2000/ethz/VirtualReality/analysisVR/embedding_development")
sys.path.insert(0, os.path.abspath("."))
from utils.load_encoder import build_windows, embed_windows, load_encoder
from utils.figure_style import FIG, apply_style, savefig_manifest

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds = pickle.load(open("outputs/session_dataset_ensembles.pkl", "rb"))
session_ids = list(ds.keys())
split = np.load("splits/split_seed42.npy", allow_pickle=True).item()

# pick best-predicted pair (both contrastive and predictive)
cc = np.nanmean(np.load("outputs/cebra_64d_eval/ensembles/all_r2.npy"), axis=0)
cp = np.nanmean(np.load("outputs/cebra_pred_64d_eval/ensembles/all_r2.npy"), axis=0)
comb = np.where(np.isfinite(cc) & np.isfinite(cp), cc + cp, -np.inf)
s_idx, e_idx = map(int, np.unravel_index(np.argmax(comb), comb.shape))
sess = session_ids[s_idx]
print(f"session {s_idx} ({sess})  ensemble E{e_idx+1:02d}  cc R2={cc[s_idx,e_idx]:.3f}  cp R2={cp[s_idx,e_idx]:.3f}")

trials = sorted(t for t in split.get(sess, []) if t in ds[sess]['data'])
X = np.concatenate([ds[sess]['data'][t] for t in trials], axis=0).astype(np.float32)
y = np.concatenate([ds[sess]['labels'][t][:, e_idx] for t in trials]).astype(np.float32)

def tc_embed(arm):
    path = f"models/{arm}_64d/ensembles/seed42/session_{s_idx:02d}_neuron_{e_idx:02d}.pt"
    enc, _, _ = load_encoder(path, device=str(device))
    H = embed_windows(enc, build_windows(X), device=str(device))
    del enc; torch.cuda.empty_cache()
    return H

H_pred, H_cont = tc_embed('cebra_pred'), tc_embed('cebra')
n = min(len(H_pred), len(H_cont), len(y))
H_pred, H_cont, y = H_pred[:n], H_cont[:n], y[:n]
print(f"embeddings: pred {H_pred.shape}, cont {H_cont.shape}")

rng = np.random.RandomState(0)
idx = rng.choice(n, size=min(4000, n), replace=False)
vmin, vmax = np.percentile(y[idx], [2, 98])

fig, axes = plt.subplots(1, 2, figsize=FIG.FULL)
apply_style(fig, list(axes))
sc = None
for ax, H, title in [(axes[0], H_pred, "TempConv-Pred embedding"),
                     (axes[1], H_cont, "TempConv-Cont embedding")]:
    p = PCA(n_components=2).fit(H)
    Z = p.transform(H)
    sc = ax.scatter(Z[idx, 0], Z[idx, 1], c=y[idx], cmap='coolwarm', s=7, alpha=0.6,
                    linewidths=0, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"PC1 ({p.explained_variance_ratio_[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({p.explained_variance_ratio_[1]*100:.0f}%)")
    ax.set_xticks([]); ax.set_yticks([])

cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85)
cbar.set_label("ensemble activity (z-scored)")
savefig_manifest(fig, 'embedding_space_pred_vs_cont.png',
                 ['outputs/mlps/ensembles_multiseed'], skip_tight_layout=True)
print("saved embedding_space_pred_vs_cont.png")
