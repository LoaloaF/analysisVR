"""How much of a sample set's variance lies along one direction, and what to compare
that against.

Companion to `pca.py`: that module answers "what are the data's own dominant
directions"; this one answers "how much variance does *this* externally supplied
direction carry, and is that more than nothing". The externally-supplied part is what
makes the two references here necessary - see `cv_pc_var_explained` (in-sample
eigenvalues are biased upward, a fitted direction is not, so the naive comparison is
rigged) and `random_direction_null` (a point estimate at 1/n_features understates what
chance produces).

Nothing here knows about SVMs, timebins or neurons - `var_plotting_lib` supplies that.
"""

import numpy as np


def total_variance(X):
    """Sum of per-feature variances of `X` (n_samples, n_features) - i.e. trace of its
    covariance matrix, the denominator every fraction in this module is measured
    against. Uses ddof=1 throughout so it stays consistent with `np.var(..., ddof=1)`
    in the numerators. Returns nan for fewer than 2 samples."""
    X = np.asarray(X, dtype=float)
    if X.shape[0] < 2:
        return np.nan
    return float(np.var(X, axis=0, ddof=1).sum())


def prepare_var_basis(X):
    """(centered, total) for `X`, so many directions can be tested against the same sample
    set without re-centering it each time.

    Exists for the whole-session comparison, where one fixed sample matrix (every 40ms bin
    of a recording, ~10^5 rows) is projected onto a different W for every timebin. Calling
    `var_explained` in that loop would copy and re-reduce the whole matrix per call; this
    does it once. Returns nan `total` if `X` has fewer than 2 rows or no variance."""
    X = np.asarray(X, dtype=float)
    if X.shape[0] < 2:
        return X, np.nan
    centered = X - X.mean(axis=0, keepdims=True)
    total = np.var(centered, axis=0, ddof=1).sum()
    if not np.isfinite(total) or total <= 0:
        return centered, np.nan
    return centered, float(total)


def var_explained_from_basis(centered, total, direction):
    """`var_explained` against a basis from `prepare_var_basis` - same value, no recompute
    of the centering or the denominator."""
    d = np.asarray(direction, dtype=float).ravel()
    norm = np.linalg.norm(d)
    if norm == 0 or not np.isfinite(total):
        return np.nan
    return float(np.var(centered @ (d / norm), ddof=1) / total)


def var_explained(X, direction):
    """Fraction of `X`'s total variance that lies along `direction`.

    `X` is (n_samples, n_features); `direction` is any non-zero vector of length
    n_features - it is unit-normalized internally, so an unnormalized fitted `w` (e.g.
    the bootstrap-mean coef_ stored on an SVM row, whose norm carries no directional
    meaning) can be passed straight in. `X` is centered internally.

    Unlike a PCA eigenvalue, this estimate is unbiased at any sample size *provided the
    direction was not fitted on `X`* - both the numerator (a one-dimensional variance
    along a fixed axis) and the denominator (a sum of per-feature variances) are
    ordinary unbiased estimates. That is the whole reason `cv_pc_var_explained` exists:
    it puts the PC reference on the same footing. Returns nan if `X` has no variance.
    """
    X = np.asarray(X, dtype=float)
    d = np.asarray(direction, dtype=float).ravel()
    norm = np.linalg.norm(d)
    if norm == 0 or X.shape[0] < 2:
        return np.nan
    d = d / norm
    centered = X - X.mean(axis=0, keepdims=True)
    total = np.var(centered, axis=0, ddof=1).sum()
    if not np.isfinite(total) or total <= 0:
        return np.nan
    return float(np.var(centered @ d, ddof=1) / total)


def cv_pc_var_explained(X, n_pcs=3, n_splits=5, n_repeats=5, rng=None):
    """Cross-validated variance explained by each of the top `n_pcs` principal
    directions, as a fraction, one entry per PC.

    Why not just use `pca_eigen`'s eigenvalues: in-sample eigenvalues are biased upward
    whenever n_samples is not much larger than n_features (~n_trials against ~54 units
    here), because the eigenvectors are fitted to that sample's own noise. A decoder
    direction fitted elsewhere carries no such bias, so plotting it against raw
    eigenvalues systematically flatters the PCs and understates the decoder. Fitting the
    eigenvectors on a train split and measuring their variance on a held-out split puts
    both on the same (unbiased) footing, which is the only way the comparison means
    anything.

    Splits are contiguous K-fold over the sample axis after a shuffle (`rng`, default
    seeded so a figure redraws identically). Per fold, eigenvectors come from the train
    part and the fraction is measured against the *test* part's own total variance.
    Folds are averaged. A fold too small to estimate anything contributes nan and is
    skipped; if every fold fails, returns all-nan.

    **`n_repeats` matters more than it looks.** A cross-validated spectrum is NOT
    guaranteed to decrease with PC index: the eigenvector ordering is fixed by the train
    split but the variance is scored on held-out samples, so at modest n the k-th
    direction routinely beats the (k-1)-th on a given fold. Drawn as a reference curve
    per timebin, a single K-fold pass therefore produces PC1/PC2/PC3 lines that visibly
    cross and jitter bin to bin, which makes them useless to read a decoder direction
    against. Repeating the whole split `n_repeats` times with different shuffles and
    averaging cuts that fold-to-fold noise (crossings become rare rather than routine)
    for a handful more 54x54 eigendecompositions. It does not remove the effect - genuine
    near-ties between adjacent PCs can still cross, and that is real information about a
    flat spectrum, not noise to be smoothed away.
    """
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    n_pcs = min(n_pcs, n_features)
    rng = np.random.default_rng(0) if rng is None else rng
    if n_samples < 2 * n_splits or n_features < 1:
        return np.full(n_pcs, np.nan)

    per_fold = []
    for _ in range(max(1, n_repeats)):
        order = rng.permutation(n_samples)
        for test_idx in np.array_split(order, n_splits):
            train_idx = np.setdiff1d(order, test_idx, assume_unique=False)
            if len(train_idx) < 2 or len(test_idx) < 2:
                continue
            train, test = X[train_idx], X[test_idx]
            train_c = train - train.mean(axis=0, keepdims=True)
            cov = np.cov(train_c, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(np.atleast_2d(cov))
            eigvecs = eigvecs[:, ::-1][:, :n_pcs]  # descending eigenvalue order
            test_c = test - test.mean(axis=0, keepdims=True)
            total = np.var(test_c, axis=0, ddof=1).sum()
            if not np.isfinite(total) or total <= 0:
                continue
            per_fold.append(np.var(test_c @ eigvecs, axis=0, ddof=1) / total)

    if not per_fold:
        return np.full(n_pcs, np.nan)
    return np.mean(np.vstack(per_fold), axis=0)


def insample_pc_var_explained(X, n_pcs=3):
    """Plain in-sample variance explained by the top `n_pcs` PCs, as fractions.

    The optimistic half of the PC1 bracket (see `cv_pc_var_explained` for the pessimistic
    half). Eigenvectors fitted to a sample and scored on that same sample overstate the
    truth whenever n_samples is not much larger than n_features - measured on synthetic
    data with a near-flat spectrum at n=83/p=54, in-sample PC1 read 6.05% against a true
    2.75%. Not a number to quote on its own; useful precisely because its bias runs
    opposite to the cross-validated one, so the pair brackets the truth.
    """
    X = np.asarray(X, dtype=float)
    n_pcs = min(n_pcs, X.shape[1])
    if X.shape[0] < 2:
        return np.full(n_pcs, np.nan)
    centered = X - X.mean(axis=0, keepdims=True)
    total = np.var(centered, axis=0, ddof=1).sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(n_pcs, np.nan)
    eigvals = np.clip(np.linalg.eigvalsh(np.atleast_2d(np.cov(centered, rowvar=False))), 0, None)
    return np.sort(eigvals)[::-1][:n_pcs] / total


def random_direction_null(X, percentiles=(5, 95), n_draws=2000, rng=None):
    """Percentile band of the variance explained by a *uniformly random* unit direction
    in `X`'s feature space - the chance level any direction must beat to be interesting.

    A single line at 1/n_features is the exact *mean* of this distribution but not a
    usable threshold: the spread around it is wide (it is a random convex combination of
    the covariance eigenvalues, and those are far from equal), so a direction can land
    well above 1/n_features by luck alone. Hence a band.

    Computed exactly rather than by sampling directions and re-projecting: for a random
    unit u, the variance along u is sum_i lambda_i * c_i where lambda are the covariance
    eigenvalues and c is u expressed in the eigenbasis and squared - and c is itself
    Dirichlet-distributed (a normalized vector of squared standard normals), independent
    of the eigenbasis. So drawing `n_draws` weight vectors and dotting them against the
    eigenvalues gives the same distribution at a fraction of the cost, with no matrix
    products at all. Returns one fraction per entry of `percentiles`.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(0) if rng is None else rng
    if X.shape[0] < 2:
        return np.full(len(percentiles), np.nan)
    centered = X - X.mean(axis=0, keepdims=True)
    cov = np.atleast_2d(np.cov(centered, rowvar=False))
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0, None)
    total = eigvals.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(percentiles), np.nan)
    weights = rng.standard_normal((n_draws, len(eigvals))) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    draws = (weights @ eigvals) / total
    return np.percentile(draws, percentiles)


def _ve_for_masks(X, direction, masks):
    """`var_explained` for many row-subsets of `X` at once - (n_masks,) fractions.

    Same quantity as `[var_explained(X[m], direction) for m in masks]` (verified to 7e-18)
    but as three matrix products instead of a python loop, which is what makes a
    permutation band affordable per timebin. Both the numerator (variance of the projection
    within the subset) and the denominator (that subset's own total variance) are rebuilt
    from per-subset sums:

        Var_m(Xu) = (Σ p² - (Σ p)²/n_m) / (n_m - 1),  p = Xu
        T_m       = (Σ_i ‖x_i‖² - ‖Σ_i x_i‖²/n_m) / (n_m - 1)
    """
    X = np.asarray(X, dtype=float)
    d = np.asarray(direction, dtype=float).ravel()
    norm = np.linalg.norm(d)
    M = np.asarray(masks, dtype=float)
    if norm == 0 or not np.isfinite(norm):
        return np.full(M.shape[0], np.nan)
    u = d / norm
    n_m = M.sum(1)
    proj = X @ u
    s1, s2 = M @ proj, M @ (proj ** 2)
    proj_var = (s2 - s1 ** 2 / n_m) / (n_m - 1)
    col_sum = M @ X
    sq_sum = M @ (X ** 2).sum(1)
    total = (sq_sum - (col_sum ** 2).sum(1) / n_m) / (n_m - 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = proj_var / total
    return np.where(np.isfinite(total) & (total > 0), out, np.nan)


def var_explained_subset_null(X, direction, n_subset, reps=200, percentiles=(5, 95), rng=None):
    """Percentile band of `var_explained` over RANDOM subsets of `n_subset` rows of `X`.

    The null a per-condition variance curve has to be read against, and it is not the same
    as `random_direction_null`: that one asks whether the *direction* beats a random
    direction, on every sample. This one holds the direction fixed and asks whether *this
    split of the trials* beats a random split of the same size - which is the question two
    condition curves pose, and the only one that accounts for a condition having fewer
    trials than its partner (variance along a fixed axis has relative sd ~sqrt(2/(n-1)), so
    a 34-trial condition wobbles half again as much as a 53-trial one for no reason at all).

    Choosing a random subset of size `n_subset` *is* the label permutation - permuting a
    two-level label is exactly drawing which trials go in group 0 - so the band is the
    permutation null for that group.

    **Calibration, measured** (S07, 70 usable bins, 2000 draws/bin): a random label of the
    real group sizes lands outside the 5-95% band in 4.3% of bins, against the 5% the test
    claims - it is exact. Real labels on the same session: cue 11.4%, choice_R1 11.4%,
    session-half 14.3%. Two traps found while establishing that, both worth not repeating:
    a bin whose stored `w` is **all zeros** (L1 shrank everything; 9 of 79 bins here, median
    `acc_aggr` exactly 0.500) yields NaN once normalized and will be silently counted as
    significant by any `null >= obs` test that does not exclude it; and neighbouring bins
    are not independent (overlapping 3-bin fit windows plus `smooth`), so a run of
    significant bins is one event, not several.

    Returns one fraction per entry of `percentiles`, or all-nan if `X` cannot support a
    subset of that size.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(0) if rng is None else rng
    n = X.shape[0]
    if not (2 <= n_subset <= n - 1) or reps <= 0:
        return np.full(len(percentiles), np.nan)
    # argsort of uniform noise = a fresh random permutation per rep, vectorized
    masks = np.argsort(rng.random((reps, n)), axis=1) < n_subset
    draws = _ve_for_masks(X, direction, masks)
    if not np.isfinite(draws).any():
        return np.full(len(percentiles), np.nan)
    return np.nanpercentile(draws, percentiles)
