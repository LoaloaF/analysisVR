"""PCA and subspace geometry shared by the SVM/W and population-axes analyses.

Both `w_plotting_lib` (how much variance does a fitted decoder direction explain?) and
`dim_plotting_lib` (what are the population's own dominant directions, and how do they
drift across sessions?) need the same eigendecomposition, the same canonical-angle
comparison and the same eigenvalue-spectrum panel. They used to have separate copies -
`pca_eigen`/`canonical_angles_deg` inline in svm_analysis_2026.py, and
`_pca_basis_for_n_pcs`/`scipy.linalg.subspace_angles` in dim_plotting_lib.py - which is
also why `dim_plotting_lib` importing from the W library would have been the wrong
dependency direction.

Note `pca_eigen` (covariance eigendecomposition, full spectrum + directions) and
`dim_plotting_lib._pca_basis_for_n_pcs` (SVD, fixed-k basis + per-neuron contributions)
are deliberately still separate: they answer different questions and the latter is tied
to the per-timebin trial-matrix convention. Only what was genuinely duplicated moved here.
"""

import textwrap

import numpy as np


def pca_eigen(data):
    """Eigen-decomposition of `data`'s (n_samples, n_features) covariance matrix. `data`
    is centered internally (per-feature mean subtracted) before computing the
    covariance - necessary even for already globally z-scored FiringRate40msZ data,
    since a small window's/session's own local mean generally isn't exactly zero.
    Negative eigenvalues from floating-point noise (a valid covariance matrix is PSD, so
    these should only ever be ~0) are clipped to 0 before normalizing. Returns
    (eigenvalues_pct, eigenvectors, total_var): eigenvalues_pct is % of total variance
    (sum of eigenvalues == trace(cov)), sorted descending; eigenvectors is
    (n_features, n_components), columns in the same descending order (orthonormal,
    straight from eigh) - feeds plot_session_pca_similarity's canonical-angle subspace
    comparison, which needs the actual PC directions, not just their variances;
    total_var is kept for callers that need the same denominator for a non-PCA
    direction's own variance (see plot_w_variance_explained)."""
    centered = data - data.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]  # descending, columns stay matched to eigvals
    eigvals = np.clip(eigvals, 0, None)
    total_var = eigvals.sum()
    return eigvals / total_var * 100, eigvecs, total_var


def canonical_angles_deg(subspace_a, subspace_b):
    """Canonical (principal) angles, in degrees, between the subspaces spanned by
    `subspace_a`/`subspace_b`'s own columns - each an orthonormal basis, e.g. a
    contiguous slice of pca_eigen's eigenvectors, which are orthonormal by
    construction. Same method already used elsewhere in this codebase for comparing PCA
    subspaces across sessions (get_SessionPCs40msCAs in
    ephys_preprocessing/postproc_mea1k_ephys.py; plot_EvolvingPCSubspace.render_plot in
    dashsrc/plot_components/plots/), reused here rather than reinvented: SVD of the
    cross-subspace Gram matrix gives singular values that are exactly the cosines of the
    principal angles (Bjorck-Golub). Returns one angle per min(k_a, k_b) - 0 degrees
    means that pair of directions is identical, 90 means orthogonal."""
    M = subspace_a.T @ subspace_b
    singular_values = np.linalg.svd(M, compute_uv=False)
    return np.rad2deg(np.arccos(np.clip(singular_values, -1, 1)))


def plot_pca_eigenvalues(ax, eigenvalues_pct, w_var_pct=None, w_label=None, max_pcs=30, title=None):
    """Horizontal bar plot of PCA eigenvalues (already a % of total variance - see
    pca_eigen), ordered by magnitude with PC1 at the top. `max_pcs`, if given
    (default 30), only shows the top N components - a session/interval can have dozens
    of neurons, and past the first few tens the tail is uninformative and just wastes
    vertical space. `w_var_pct`, if given, draws a vertical dashed purple line at that
    x-value across every bar - the % of variance a specific W direction explains in this
    same data (see plot_w_variance_explained; unrelated to pca_eigen, which this value
    isn't derived from) - for a direct visual comparison against
    the ranked PC spectrum, in particular PC1 (the max direction): if the line reaches
    close to/past PC1's bar, that decoder direction captures about as much variance as
    the single largest source of variability in the data; if it's far short of even the
    smaller PCs shown, it's a minor contributor. `w_label` (e.g. a highlight_selection's
    own `name`) annotates that line; defaults to plain 'W'."""
    if max_pcs is not None:
        eigenvalues_pct = eigenvalues_pct[:max_pcs]
    pc_idx = np.arange(len(eigenvalues_pct))
    ax.barh(pc_idx, eigenvalues_pct, color='#555555', height=0.8, zorder=5)
    ax.invert_yaxis()  # PC1 at top
    ax.set_yticks(pc_idx)
    ax.set_yticklabels([f'PC{i + 1}' for i in pc_idx], fontsize=6)
    ax.set_xlabel('% of total variance', fontsize=8)
    ax.tick_params(labelsize=7)
    if w_var_pct is not None:
        ax.axvline(w_var_pct, color='purple', linestyle='--', linewidth=1.5, zorder=10)
        # Anchored in axes-fraction space (top-right corner), not at the line's own data
        # x-position - keeps the label off the title above regardless of where the line
        # falls (near 0% or near 100%) and regardless of how long w_label is (wrapped +
        # given a translucent background so it stays legible over a long PC1 bar).
        label_text = textwrap.fill(f'{w_label or "W"}: {w_var_pct:.1f}%', width=20)
        ax.text(0.98, 0.98, label_text, transform=ax.transAxes, color='purple', fontsize=7,
                ha='right', va='top', zorder=15,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.75))
    if title:
        ax.set_title(title, fontsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
