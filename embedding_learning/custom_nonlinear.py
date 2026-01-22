from models import AutoEncoder
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

EPOCHS = 10

def info_nce_thresholded(
    output: torch.Tensor,
    y: torch.Tensor,
    *,
    pos_thresh: float,
    neg_thresh: float,
    tau: float = 0.1,
    normalize: bool = True,
    exclude_self: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    InfoNCE with thresholded positives/negatives based on |y_i - y_j|.

    - Positives: pairs with |y_i - y_j| <= pos_thresh
    - Negatives: pairs with |y_i - y_j| >= neg_thresh
    - Pairs in between are ignored.

    For each anchor i, we compute:
        L_i = - log ( sum_{j in P(i)} exp(sim(i,j)/tau) / sum_{k in P(i)∪N(i)} exp(sim(i,k)/tau) )
    and average over anchors that have at least one positive and one negative (or at least one positive).

    Args:
        output: (B, D) embeddings
        y: (B,) labels (e.g., time)
        pos_thresh: positive threshold on |y_i - y_j|
        neg_thresh: negative threshold on |y_i - y_j|
        tau: temperature
        normalize: if True, L2-normalize embeddings -> cosine similarity
        exclude_self: if True, exclude i==j pairs
        eps: numerical stability

    Returns:
        Scalar loss tensor.
    """
    B, D = output.shape
    device = output.device

    z = F.normalize(output, dim=1) if normalize else output
    sim = z @ z.T  # (B, B) dot-product (cosine if normalized)

    # Pairwise label diffs
    y = y.view(-1)
    label_diff = (y[:, None] - y[None, :]).abs()  # (B, B)

    pos_mask = label_diff <= pos_thresh
    neg_mask = label_diff >= neg_thresh

    if exclude_self:
        eye = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = pos_mask & ~eye
        neg_mask = neg_mask & ~eye

    # Only consider pairs that are either positive or negative
    valid_mask = pos_mask | neg_mask

    # Mask out invalid pairs by setting logits to -inf so exp -> 0
    logits = sim / tau
    logits = logits.masked_fill(~valid_mask, float("-inf"))

    # Numerically stable logsumexp
    denom = torch.logsumexp(logits, dim=1)  # (B,)

    # Positive numerator: log sum exp over positives
    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
    numer = torch.logsumexp(pos_logits, dim=1)  # (B,)

    # Anchors with at least one positive contribute; others ignored
    has_pos = pos_mask.any(dim=1)
    # (Optional) require at least one negative too:
    # has_neg = neg_mask.any(dim=1)
    # keep = has_pos & has_neg
    keep = has_pos

    if keep.sum() == 0:
        # No valid anchors; return 0 with gradient
        return output.sum() * 0.0

    loss_per_anchor = -(numer - denom)  # (B,)
    loss = loss_per_anchor[keep].mean()
    return loss


# Example usage matching your thresholds:
def contrastive_info_nce_loss(output, y):
    TIME_UNIT = 40000
    pos_thresh = TIME_UNIT * 3 + 1
    neg_thresh = TIME_UNIT * 25 - 1
    return info_nce_thresholded(output, y, pos_thresh=pos_thresh, neg_thresh=neg_thresh, tau=0.1)


def generate_custom_nonlinear_embeddings(groups_xs, groups_ys, EMBEDDING_DIM):
    model_nonlinear = AutoEncoder(input_size=groups_xs[0].shape[1] + 1, hidden_size=20, output_size=EMBEDDING_DIM)
    optimizer = torch.optim.Adam(model_nonlinear.parameters(), lr=0.001)
    groups_ys = [torch.from_numpy(groups_ys[i]).float() for i in range(len(groups_xs))]
    loaders = [DataLoader(TensorDataset(groups_xs[i], groups_ys[i]), batch_size=1024, shuffle=True) for i in range(len(groups_xs))]
    for epoch in tqdm(range(EPOCHS)):
        for i, loader in enumerate(loaders):
            for x, y in loader:
                optimizer.zero_grad()
                output = model_nonlinear.encoder(torch.cat([x, i*torch.ones(x.shape[0], 1)], dim=1))
                loss = contrastive_info_nce_loss(output, y)
                loss.backward()
                optimizer.step()

    embeddings = []
    unshuffled_loaders = [DataLoader(TensorDataset(groups_xs[i], groups_ys[i]), batch_size=1024, shuffle=False) for i in range(len(groups_xs))]
    for i, loader in enumerate(unshuffled_loaders):
        curr_embeddings = []
        for x, y in loader:
            emb = model_nonlinear.encoder(torch.cat([x, i*torch.ones(x.shape[0], 1)], dim=1)).detach().cpu().numpy()
            curr_embeddings.append(emb)
        embeddings.append(np.concatenate(curr_embeddings, axis=0))
    return embeddings