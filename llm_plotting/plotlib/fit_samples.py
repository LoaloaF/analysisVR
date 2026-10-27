"""How a firing-rate frame becomes the sample matrix the SVM was actually fitted on.

One home for the preprocessing every W-consuming figure has to replicate, because a
number computed on differently-preprocessed samples is not the number the decoder saw.
Two steps, in this order (see `prepare`):

  1. a centered `TIMEBINS_PER_FIT`-bin mean along the bin axis  (`window_mean`)
  2. per-sample L2 normalization                                (`l2_rows`)

Both come from the fit side, not from taste: the fit builds one sample per trial as
`np.stack([X_data[m].mean() for m in X_data_masks])` over a 3-native-bin sliding window
(`get_SVMCueOutcomeChoicePred`/`timeline_decoding` in
ephys_preprocessing/postproc_mea1k_ephys.py), and passes that through
`Normalizer(norm='l2')` as the first step of its Pipeline - when that step is enabled,
which is the whole reason `mode` below is data-driven rather than a constant.

WHY THIS MODULE EXISTS (2026-08-20): the same logic had been written three times, and
the three disagreed. `var_plotting_lib.timebin_samples` had it right (3-bin mean then
L2); `w_plotting_lib.get_trial_w_projection` normalized *single* native bins (and
`normalize(mean(bins)) != mean(normalize(bins))`, so its trace was not the classifier's
own decision value even up to smoothing); `w_plotting_lib.get_interval_window_activity`
did neither, which meant the '% var' number annotated onto the projection row was
computed on different samples than the row it was drawn on.

THE L2 STEP IS NOT ALWAYS ON. It is commented out in the current working tree of
postproc_mea1k_ephys.py (`# ('norm', Normalizer(norm='l2')),` in both `_one_bootstrap`
and `_fit_score_fold`) - the note there is that per-sample row norms vary a lot, so the
step has a real effect - while `ephys_preprocessing/svm_task_subspaces.py` keeps it and,
better, records what it did on every row (`normalizer='l2_sample'`). So the display-side
preprocessing follows the W's own provenance: `mode_for_w_row` reads that column, and
only falls back to `LEGACY_POSTPROC_MODE` (with a one-time warning) for the older cached
tables that don't carry it.

Nothing here knows about intervals, plots or PCA. `w_plotting_lib` and
`var_plotting_lib` supply that.
"""

import numpy as np


# Native 40ms bins per SVM fit window - the sliding window's width, mirroring
# `timebins_per_fit` in get_SVMCueOutcomeChoicePred. Not something the display side is
# free to choose: a different value here evaluates a W on samples it was never fitted
# on. There is still no shared source of truth with the fit side (it is a local in that
# function), so a change there must be mirrored here by hand.
TIMEBINS_PER_FIT = 3

# --- preprocessing modes ---------------------------------------------------------
# FIT_L2/FIT_RAW are the two fit-parity modes (which one is right depends on whether the
# fit's Normalizer step was enabled); L2_BIN is what w_plotting_lib did before 2026-08-20
# and is kept only for A/B-ing that change; RAW is the bare dot product.
FIT_L2 = 'fit_l2'    # k-bin mean, then per-sample L2   <- fit parity, Normalizer on
FIT_RAW = 'fit_raw'  # k-bin mean only                  <- fit parity, Normalizer off
L2_BIN = 'l2_bin'    # single native bin, per-sample L2  (legacy display behaviour)
RAW = 'raw'          # single native bin, untouched
AUTO = 'auto'        # resolve from the fitted row's own `normalizer` column
MODES = (FIT_L2, FIT_RAW, L2_BIN, RAW)

# What to assume for a fitted table that carries no `normalizer` column at all - i.e.
# every SVM analytic cached before svm_task_subspaces.py started recording it. FIT_L2
# because the *committed* postproc pipeline had `Normalizer(norm='l2')` active, which is
# what those cached rows were fitted with. If you re-cache with the current working tree
# (Normalizer commented out) and the table still has no `normalizer` column, this is
# wrong and should be flipped to FIT_RAW - the durable fix is to store the column.
LEGACY_POSTPROC_MODE = FIT_L2

_WARNED = set()


def _warn_once(key, message):
    """Print `message` the first time `key` comes up in this process.

    A plain print, not `warnings.warn`: these are notebook-facing notes about which
    preprocessing a figure ended up using, and they have to survive the notebook's usual
    warning filters. Once per process, since the alternative is one line per session in
    every cross-session loop, which is how a real notice gets scrolled past.
    """
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f'[fit_samples] {message}', flush=True)


def l2_rows(a):
    """`a` with every row (last axis) scaled to unit L2 norm - the fit's
    `Normalizer(norm='l2')`, which is per-sample, not per-feature.

    Zero rows are left alone rather than producing 0/0. NaN rows stay NaN, which is what
    keeps `window_mean`'s edge padding propagating instead of turning into zeros.
    Works on both (n_samples, n_features) and (n_trials, n_bins, n_features).
    """
    a = np.asarray(a, dtype=float)
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.where(norms == 0, 1.0, norms)


def window_mean(a, timebins_per_fit=TIMEBINS_PER_FIT, axis=-2):
    """Centered rolling mean of width `timebins_per_fit` along `axis`, NaN-padded to
    `a`'s original length.

    `axis=-2` is the bin axis for both shapes this is used with: (n_bins, n_features)
    for a whole session and (n_trials, n_bins, n_features) for a trial stack.

    Keeping the original length (rather than returning the n-k+1 valid windows) is
    deliberate: every consumer indexes these arrays by native-bin index - the x-axis
    offsets, `highlight_range`, `_window_trial_bin_values` - and with k=3 the first
    valid window is centered on index 1, which is exactly the 1-based `timebin` the fit
    stores ('center of fit interval'). So index == stored timebin holds, and the two
    bins that have no full window read as NaN, i.e. as the gap they are, the same way
    the fit simply has no timebin there.
    """
    a = np.asarray(a, dtype=float)
    if not timebins_per_fit or timebins_per_fit <= 1:
        return a
    out = np.full(a.shape, np.nan)
    if a.shape[axis] < timebins_per_fit:
        return out
    # sliding_window_view appends the window as a new last axis; mean over it gives an
    # array shaped like `a` but shortened to n-k+1 along `axis`.
    means = np.lib.stride_tricks.sliding_window_view(a, timebins_per_fit, axis=axis).mean(axis=-1)
    half = timebins_per_fit // 2
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(half, half + means.shape[axis])
    out[tuple(sl)] = means
    return out


def resolve_mode(mode):
    """A caller's `normalize=` argument -> one of `MODES`.

    Accepts a mode string, or a bool for the pre-2026-08-20 keyword: True means "match
    the classifier" and now resolves to FIT_L2 (it used to mean L2_BIN - that is the
    behaviour change, pass 'l2_bin' explicitly to get the old trace back), False to RAW.
    `AUTO` is *not* resolved here - it needs the fitted row, see `mode_for_w_row`.
    """
    if mode is True:
        return FIT_L2
    if mode is False or mode is None:
        return RAW
    if mode in MODES or mode == AUTO:
        return mode
    raise ValueError(f'unknown preprocessing mode {mode!r} - expected a bool or one of '
                     f'{MODES + (AUTO,)}')


def mode_for_normalizer(normalizer):
    """The fit-parity mode for a `normalizer` value as the fitted table records it.

    'l2_sample' (what svm_task_subspaces.py writes when its `Normalizer(norm='l2')` step
    ran) -> FIT_L2; anything else that is an actual recorded value -> FIT_RAW. A missing
    value is not this function's business - see `mode_for_w_row`.
    """
    return FIT_L2 if normalizer == 'l2_sample' else FIT_RAW


def mode_for_w_row(w_row, mode=AUTO):
    """Resolve `mode` against one fitted row, so a figure's preprocessing follows the W
    it is drawing rather than a default that can silently disagree with the fit.

    Only AUTO consults the row; an explicit mode (or bool) is passed straight through, so
    a caller can always pin the preprocessing by hand. AUTO reads the row's `normalizer`
    column (`mode_for_normalizer`); a row without one falls back to
    `LEGACY_POSTPROC_MODE` and says so once.
    """
    resolved = resolve_mode(mode)
    if resolved != AUTO:
        return resolved
    normalizer = w_row.get('normalizer') if hasattr(w_row, 'get') else None
    if normalizer is None or (isinstance(normalizer, float) and np.isnan(normalizer)):
        _warn_once('missing-normalizer',
                   f"fitted row carries no 'normalizer' column (cached analytic predating it) - "
                   f"assuming LEGACY_POSTPROC_MODE={LEGACY_POSTPROC_MODE!r}. Pass normalize='fit_raw' "
                   f"if this table was re-cached with postproc's Normalizer step commented out.")
        return LEGACY_POSTPROC_MODE
    return mode_for_normalizer(normalizer)


def mode_for_frame(svm_res, mode=AUTO):
    """`mode_for_w_row` for a whole fitted frame - for callers that consume every row of
    a table at once (var_plotting_lib's per-timebin curve) rather than one hand-picked W.

    Resolves off the frame's `normalizer` column, which is per-row in principle but is a
    property of the fitting run in practice; a frame that somehow mixes values resolves
    to FIT_L2 only if *all* its recorded values are 'l2_sample', since silently scoring
    half a curve against the wrong preprocessing is worse than being conservative about
    it.
    """
    resolved = resolve_mode(mode)
    if resolved != AUTO:
        return resolved
    if 'normalizer' not in getattr(svm_res, 'columns', ()):
        _warn_once('missing-normalizer',
                   f"fitted table carries no 'normalizer' column (cached analytic predating it) - "
                   f"assuming LEGACY_POSTPROC_MODE={LEGACY_POSTPROC_MODE!r}.")
        return LEGACY_POSTPROC_MODE
    recorded = svm_res['normalizer'].dropna().unique()
    if len(recorded) == 0:
        return LEGACY_POSTPROC_MODE
    return FIT_L2 if all(mode_for_normalizer(v) == FIT_L2 for v in recorded) else FIT_RAW


def fit_mode(normalize):
    """`normalize` -> FIT_L2 or FIT_RAW, for the sample matrices where the fit window's
    mean is intrinsic and only the L2 step is in question (var_plotting_lib's
    `timebin_samples`/`session_samples`).

    This is why the bool cannot just go through `resolve_mode`: there, False means "no
    preprocessing at all" (RAW), while here it can only mean "the fit window's mean,
    without the L2" - dropping the mean would stop the samples being the classifier's
    samples in the first place.
    """
    if normalize is True or normalize is None:
        return FIT_L2
    if normalize is False:
        return FIT_RAW
    if normalize in (FIT_L2, FIT_RAW):
        return normalize
    raise ValueError(f'expected a bool or one of {(FIT_L2, FIT_RAW)}, got {normalize!r} - '
                     f'{(L2_BIN, RAW)} do not apply to a per-timebin sample matrix')


def bias_for_w_row(w_row):
    """The fitted decision-function intercept `b` for one row, or 0.0 if the table has
    none (with a one-time notice).

    Projections are drawn as `X @ w + b` so that y=0 is the decision boundary rather
    than an arbitrary offset. Older analytics have no `b`: `_bootstrap_and_aggregate`
    computed and collected the per-bootstrap intercepts and then dropped them from its
    result dict, so nothing downstream can recover them - those tables fall back to 0.0,
    which leaves the trace exactly where it was before, just without a meaningful zero.
    """
    b = w_row.get('b') if hasattr(w_row, 'get') else None
    if b is None or (np.ndim(b) == 0 and not np.isfinite(np.asarray(b, dtype=float))):
        _warn_once('missing-bias',
                   "fitted row carries no 'b' (intercept) - falling back to b=0.0, so the "
                   "projection is an offset direction score and its zero line is NOT the "
                   "decision boundary. Re-cache the SVM analytic to get the stored intercept.")
        return 0.0
    b = np.asarray(b, dtype=float).ravel()
    if b.size == 0 or not np.isfinite(b).any():
        _warn_once('missing-bias', "fitted row's 'b' is empty/non-finite - falling back to b=0.0.")
        return 0.0
    return float(np.nanmean(b))


def prepare(values, mode=FIT_L2, timebins_per_fit=TIMEBINS_PER_FIT, axis=-2):
    """`values` preprocessed the way the classifier's own samples were - the one function
    every W-consuming figure should go through.

    `values`: (n_bins, n_features) or (n_trials, n_bins, n_features), with the bin axis at
    `axis`. `mode`: one of `MODES` (resolve AUTO first, via `mode_for_w_row`). Returns an
    array of the same shape; the fit-parity modes NaN-pad the bins that have no full
    window (see `window_mean`), so consumers must be NaN-aware (nanmean/nanstd, or
    `drop_nan_rows` before a PCA).
    """
    mode = resolve_mode(mode)
    if mode == AUTO:
        raise ValueError("prepare() needs a concrete mode - resolve AUTO via mode_for_w_row first")
    values = np.asarray(values, dtype=float)
    if mode in (FIT_L2, FIT_RAW):
        values = window_mean(values, timebins_per_fit, axis=axis)
    if mode in (FIT_L2, L2_BIN):
        values = l2_rows(values)
    return values


def drop_nan_rows(a):
    """`a` (n_samples, n_features) without the rows that carry any NaN - for consumers
    that cannot take `window_mean`'s edge padding (PCA, covariance, variance fractions).
    Rows already NaN in the raw data (a unit with missing bins) drop here too, which is
    the same thing those computations would need anyway."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f'expected (n_samples, n_features), got shape {a.shape}')
    return a[np.isfinite(a).all(axis=1)]
