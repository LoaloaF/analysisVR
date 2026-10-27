"""Plotting library for dim_analysis.py.

Mirrors the split used for svm_plotting_lib.py: dim_analysis.py stays the live,
interactively-edited notebook (data loading, session selection, cell-by-cell exploration);
this file holds pure plotting functions that take their data as explicit arguments, so
nothing here does I/O or has side effects just from being imported.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
from scipy.linalg import subspace_angles
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import pdist

# Re-exported for callers that did `from dim_plotting_lib import build_canvas` - the
# definitions now live once in plotlib/export.py.
from plotlib.export import build_canvas, build_canvas_svg, save_figure  # noqa: F401
from plotlib.pca import pca_eigen, canonical_angles_deg, plot_pca_eigenvalues
from plotlib.session_data import session_fr_for_modality
# The canonical per-label colors, shared with the SVM figures (which is where they were
# defined until dim_analysis' trial grouping needed the same ones) - so a cue-1 trial is
# the same color in both, see plotlib/colors.py.
from plotlib.colors import group_colors
# The trial-extraction/layout layer this module originally defined now lives in plotlib,
# so the SVM side can share it instead of keeping its own older generation. Aliased back
# to the private names this module already uses throughout.
from plotlib.trial_traces import (
    BIN_US, SEC_BINS, get_trial_metric_traces, layout_intervals as _layout_intervals,
    interval_short_label as _interval_short_label, smooth as _smooth,
)
from analytics_processing import analytics  # only for the whole-session PCA block at the end


# _interval_short_label: moved to plotlib/trial_traces.py (imported at the top of this module)


# BIN_US: moved to plotlib/trial_traces.py (imported at the top of this module)


# get_trial_metric_traces: moved to plotlib/trial_traces.py (imported at the top of this module)


def get_trial_position_traces(t0_events, behavior, s_id, which_interval):
    """Every trial's frame_position trace for one (session, t0 interval) - see
    get_trial_metric_traces, of which this is the position-specific case."""
    return get_trial_metric_traces(t0_events, behavior, s_id, which_interval,
                                    value_fn=lambda dat: dat.frame_position.values)


def population_rms(dat, unit_cols):
    """Per-bin population-activity magnitude, root-mean-square (not raw L2 norm) across
    `unit_cols`: RMS = norm/sqrt(n_units) is comparable across sessions/regions with
    different unit counts, whereas a raw norm grows with n_units even for the same
    underlying per-unit activity level (each squared term adds positively to the sum,
    regardless of unit count). NaNs (get_FiringRate40msZ leaves all-zero/no-variance units
    as NaN) are excluded from the mean rather than propagating and NaN-ing the whole bin.
    """
    vals = dat[unit_cols].to_numpy(dtype=float)
    return np.sqrt(np.nanmean(vals ** 2, axis=1))


def population_mean(dat, unit_cols):
    """Per-bin population activity as a plain average (not RMS - see population_rms)
    across `unit_cols`: unlike RMS (always non-negative, dominated by whichever units
    deviate most from baseline in either direction), a plain mean can come out negative
    or near-zero whenever units' z-scores partially cancel (a mix of above- and
    below-baseline units) - it reads as "is the population net-elevated or
    net-suppressed", not "how much did it move overall, in either direction". NaNs (see
    population_rms) are excluded from the mean rather than propagating.
    """
    vals = dat[unit_cols].to_numpy(dtype=float)
    return np.nanmean(vals, axis=1)


def get_trial_neural_magnitude_traces(t0_events, fr_z, s_id, which_interval, unit_cols):
    """Every trial's population-RMS trace (see population_rms) for one (session, t0
    interval) - the firing-rate-specific case of get_trial_metric_traces."""
    return get_trial_metric_traces(t0_events, fr_z, s_id, which_interval,
                                    value_fn=lambda dat: population_rms(dat, unit_cols))


# SEC_BINS: moved to plotlib/trial_traces.py (imported at the top of this module)


# _layout_intervals: moved to plotlib/trial_traces.py (imported at the top of this module)


def _draw_scaffolding(ax, layout, show_bin_lines=True):
    """Per-bin gridlines + t0 dashed markers for a layout from _layout_intervals - the
    x-axis "scaffolding" shared by every row of a multi-row, sharex=True figure (see
    plot_session_kinematics_and_neural), independent of whatever each row actually plots.
    Per-bin gridlines are solid, not dashed: with ~30 densely-packed lines per interval, a
    dashed on/off pattern lines up across all of them and produces a Moire effect that
    reads as fake banding through the traces (see feedback_matplotlib_scratchpad_verify
    memory).
    """
    for _, _, bins, offset in layout:
        if show_bin_lines:
            for b in bins:
                ax.axvline(x=b + offset, color='k', linestyle='-', alpha=0.04, zorder=1)
        ax.axvline(x=offset, color='k', linestyle='--', alpha=0.4, zorder=3)  # t0 sits at bin 0


def _apply_xticks(ax, layout):
    """Labeled t0 ticks plus unlabeled minor ticks every 1s (SEC_BINS bins) within each
    interval, for scale without cluttering the axis with numbers. Meant to be called once,
    on whichever row of a multi-row figure should carry the (shared) x-axis labels."""
    xticks, xtick_labels, minor_ticks = [], [], []
    for which_interval, _, bins, offset in layout:
        xticks.append(offset)
        xtick_labels.append(f'$t_{{{_interval_short_label(which_interval)}}}$')
        k_lo = int(np.ceil(bins.min() / SEC_BINS))
        k_hi = int(np.floor(bins.max() / SEC_BINS))
        minor_ticks.extend(offset + k * SEC_BINS for k in range(k_lo, k_hi + 1) if k != 0)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.set_xticks(minor_ticks, minor=True)
    ax.tick_params(axis='x', which='minor', length=4)


# _smooth: moved to plotlib/trial_traces.py (imported at the top of this module)


def _draw_trial_lines(ax, layout, alpha=0.05, color='k', smooth=1):
    """Row 0 style: every trial's own line, drawn individually, optionally smoothed (see
    _smooth) - off by default (smooth=1). Adapted from row 0 of
    svm_plotting_lib.create_subplot, stripped of everything decoding-specific (no SVM
    timebin resampling, no score-colored scatter, no cue-zone illustration)."""
    for _, traces, bins, offset in layout:
        for vals in traces:
            ax.plot(bins + offset, _smooth(vals, smooth), color=color, alpha=alpha, linewidth=1, zorder=2)


def _draw_mean_band(ax, layout, color='k', band='sem', smooth=1,
                     scatter_by_x=None, scatter_cmap='coolwarm', scatter_norm=None, scatter_size=25,
                     band_by_interval=None, band_label=None):
    """One row: mean-across-trials line + a shaded band per bin, instead of one line per
    trial - meant for a metric whose per-trial lines would just be noise (e.g. population
    firing-rate magnitude) rather than something individually meaningful the way a
    trial's own position trace is. `band`: 'sem' (default - standard error of the
    trial mean, i.e. how precisely the mean itself is known) or 'sd' (raw across-trial
    spread, i.e. how much individual trials vary). NB the default really is sem, not one sd -
    they differ by sqrt(n), so on ~100 trials the sem band is ~10x tighter than the trial
    spread and is a statement about the mean, not about trials.

    `band_by_interval` (default None) replaces that computed band entirely with a
    caller-supplied one: a dict `{interval_name: (lo, hi)}` of arrays matching that interval's
    own bins - see _group_shuffle_band, whose permutation band is drawn this way. An interval
    missing from the dict gets no band at all rather than silently falling back to sem, since
    mixing the two in one row would put two different meanings behind the same shading.
    `band_label` goes on the *first* band drawn, for the legend.

    `smooth` (see _smooth, off by default)
    is applied to each trial's own trace *before* averaging, not to the resulting
    mean/band - the mean line comes out identical either way (both smoothing and
    averaging-across-trials are linear, so they commute), but the band doesn't: smoothing
    per-trial first gives the spread of genuinely smoothed trials, not just a cosmetically
    smoothed line drawn around the raw (noisier) spread.

    `scatter_by_x` (default None - no scatter, just the plain line): a dict mapping every
    x position in `layout` to a color value (e.g. per-timebin PCA variance explained - see
    plot_session_kinematics_neural_and_pca's `var_explained_display='scatter'` option). If given,
    a scatter marker is drawn at each (x, mean) point, colored via `scatter_cmap`/
    `scatter_norm`, *before* the mean line - the line is drawn on top at its normal
    thinner width, so the larger colored marker shows through as a halo around it rather
    than being fully occluded. x positions missing from `scatter_by_x` (gap bins) get NaN,
    which matplotlib's colormapping simply skips. Returns the last scatter PathCollection
    drawn (for a caller that wants its own colorbar - see plot_session_kinematics_neural_
    and_pca's `draw_colorbar`), or None if `scatter_by_x` wasn't given.
    """
    mappable = None
    drawn_band = False
    for interval, traces, bins, offset in layout:
        stacked = np.asarray([_smooth(vals, smooth) for vals in traces], dtype=float)  # (n_trials, n_bins)
        mean = np.nanmean(stacked, axis=0)
        x = bins + offset
        if band_by_interval is not None:
            lo_hi = band_by_interval.get(interval)
            lo, hi = lo_hi if lo_hi is not None else (None, None)
        else:
            spread = np.nanstd(stacked, axis=0)
            if band == 'sem':
                n = np.sum(~np.isnan(stacked), axis=0)
                spread = spread / np.sqrt(np.maximum(n, 1))
            lo, hi = mean - spread, mean + spread
        if lo is not None:
            ax.fill_between(x, lo, hi, color=color, alpha=0.2, zorder=2, edgecolor=None,
                            label=None if drawn_band else band_label)
            drawn_band = True
        if scatter_by_x is not None:
            c = np.array([scatter_by_x.get(xi, np.nan) for xi in x])
            mappable = ax.scatter(x, mean, c=c, cmap=scatter_cmap, norm=scatter_norm,
                                   s=scatter_size, zorder=2.5, edgecolor='none')
        ax.plot(x, mean, color=color, linewidth=1.2, zorder=3)
    return mappable


def _draw_mean_line(ax, layout, color='k', linestyle=':', linewidth=1.2, smooth=1, label=None):
    """One row: just the mean-across-trials line, no band - a lighter-weight variant of
    _draw_mean_band for a secondary annotation line (e.g. plot_session_kinematics_neural_
    and_pca's `mean_z_range` twin-axis overlay) where a shaded band would add visual
    clutter without much value, since the line itself is already secondary to whatever
    shares its axes. `smooth` (see _smooth, off by default) is applied per-trial before
    averaging, same as _draw_mean_band. `label` (default None) goes on the *first* interval's
    segment only - the line is one plot call per interval, so labelling every one of them
    would repeat the same entry once per interval in the legend."""
    for i, (_, traces, bins, offset) in enumerate(layout):
        stacked = np.asarray([_smooth(vals, smooth) for vals in traces], dtype=float)
        mean = np.nanmean(stacked, axis=0)
        x = bins + offset
        ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=linewidth, zorder=3,
                label=label if i == 0 else None)


def _finalize_axes(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)


def _annotate_trial_counts(ax, layout, fontsize=6):
    """Small 'n=<trials>' label near each interval's own t0 tick, in the axes' top
    margin - trial counts can differ slightly across intervals (see
    get_trial_metric_traces' modal-shape filtering), so this is per-interval, not one
    overall count. Uses the x-axis transform (data x, axes-fraction y) so it sits at a
    fixed height regardless of that row's own y-range."""
    for _, traces, _, offset in layout:
        ax.text(offset+5, 0.9, f'n={len(traces)}', transform=ax.get_xaxis_transform(),
                fontsize=fontsize, ha='center', va='top', color='gray')


def plot_trial_position_kinematics(ax, t0_events, behavior, s_id, which_intervals,
                                    alpha=0.05, color='k', show_bin_lines=True, gap_bins=2, smooth=1):
    """One panel: every trial's position trace vs. time, concatenated left-to-right
    across `which_intervals` with a `gap_bins`-bin gap between them - see _layout_intervals
    for how intervals are placed. `smooth` (see _smooth, off by default) is a centered
    rolling-mean window (in timebins) applied to each trial's own line."""
    layout, _ = _layout_intervals(t0_events, behavior, s_id, which_intervals,
                                   value_fn=lambda dat: dat.frame_position.values, gap_bins=gap_bins)
    _draw_scaffolding(ax, layout, show_bin_lines=show_bin_lines)
    _draw_trial_lines(ax, layout, alpha=alpha, color=color, smooth=smooth)
    _annotate_trial_counts(ax, layout)
    _apply_xticks(ax, layout)
    _finalize_axes(ax, 'Track position (cm)')
    return ax


def plot_session_trial_kinematics(t0_events, behavior, s_id, which_intervals,
                                   height=3, width_per_bin=0.055, min_width=3, gap_bins=2,
                                   alpha=0.05, color='k', show_bin_lines=True, smooth=1):
    """Convenience wrapper: sizes the figure width to the total number of concatenated x
    bins (`total_bins * width_per_bin`, floored at `min_width`) rather than a fixed
    figsize, so a `which_intervals` selection spanning many more bins (e.g. adding
    ITI_interval's 50-bin window) isn't squeezed into the same width as a short one -
    then draws via plot_trial_position_kinematics. `smooth` (see _smooth, off by default)
    is a centered rolling-mean window (in timebins) applied to each trial's own line."""
    layout, total_bins = _layout_intervals(t0_events, behavior, s_id, which_intervals,
                                            value_fn=lambda dat: dat.frame_position.values, gap_bins=gap_bins)
    width = max(min_width, total_bins * width_per_bin)
    fig, ax = plt.subplots(figsize=(width, height))
    _draw_scaffolding(ax, layout, show_bin_lines=show_bin_lines)
    _draw_trial_lines(ax, layout, alpha=alpha, color=color, smooth=smooth)
    _annotate_trial_counts(ax, layout)
    _apply_xticks(ax, layout)
    _finalize_axes(ax, 'Track position (cm)')
    fig.tight_layout()
    return fig, ax


def plot_session_kinematics_and_neural(t0_events, behavior, fr_z, s_id, which_intervals,
                                        unit_cols=None, height_ratios=(2, 1.6), width_per_bin=0.055,
                                        min_width=3, gap_bins=2, smooth=1, position_kwargs=None,
                                        neural_kwargs=None):
    """Two stacked panels sharing one x-axis: row 0 position kinematics (every trial's own
    line - see plot_trial_position_kinematics), row 1 population neural-activation
    magnitude (per-trial population RMS firing rate, see population_rms/
    get_trial_neural_magnitude_traces), averaged across trials per bin with a shaded band
    (mean +/- SEM by default - see _draw_mean_band, pass neural_kwargs={'band': 'sd'} for
    raw across-trial spread instead). x-axis ticks/labels only go on row 1 (bottom); row 0
    still gets the same t0/bin scaffolding drawn (see _draw_scaffolding) so both rows'
    interval boundaries visually line up.

    `smooth` (see _smooth, off by default) is a centered rolling-mean window (in
    timebins) applied per-trial before drawing, in both rows - pass
    position_kwargs={'smooth': ...} / neural_kwargs={'smooth': ...} to override it for
    just one row.

    `fr_z` must already carry `behavior`'s own 'trial_id' column and IntervalIndex
    construction (see dim_analysis.py's ephys cells and the get_trial_metric_traces
    docstring for why that copy is safe). `unit_cols`, if None, auto-detects every column
    of `fr_z` starting with 'Unit'.
    """
    position_kwargs = position_kwargs or {}
    neural_kwargs = neural_kwargs or {}
    if unit_cols is None:
        unit_cols = [c for c in fr_z.columns if c.startswith('Unit')]

    pos_layout, total_bins = _layout_intervals(
        t0_events, behavior, s_id, which_intervals,
        value_fn=lambda dat: dat.frame_position.values, gap_bins=gap_bins)
    neural_layout, _ = _layout_intervals(
        t0_events, fr_z, s_id, which_intervals,
        value_fn=lambda dat: population_rms(dat, unit_cols), gap_bins=gap_bins)

    width = max(min_width, total_bins * width_per_bin)
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(width, sum(height_ratios)),
                                    gridspec_kw={'height_ratios': height_ratios})

    # sharex axes share the same tick locator/formatter by default, not just the view
    # range - so _apply_xticks' set_xticklabels on ax1 below would otherwise also
    # propagate those same labels onto ax0. Confirmed this was happening (every row
    # showing the same labels, not just the bottom one) despite this function's own
    # docstring claiming otherwise - hide ax0's explicitly.
    ax0.tick_params(axis='x', labelbottom=False)

    _draw_scaffolding(ax0, pos_layout, show_bin_lines=position_kwargs.get('show_bin_lines', True))
    _draw_trial_lines(ax0, pos_layout, alpha=position_kwargs.get('alpha', 0.05),
                       color=position_kwargs.get('color', 'k'), smooth=position_kwargs.get('smooth', smooth))
    _annotate_trial_counts(ax0, pos_layout)
    _finalize_axes(ax0, 'Track position (cm)')

    _draw_scaffolding(ax1, neural_layout, show_bin_lines=neural_kwargs.get('show_bin_lines', True))
    _draw_mean_band(ax1, neural_layout, color=neural_kwargs.get('color', 'k'),
                     band=neural_kwargs.get('band', 'sem'), smooth=neural_kwargs.get('smooth', smooth))
    _apply_xticks(ax1, neural_layout)
    _finalize_axes(ax1, 'Population RMS\n(z-scored Hz)')

    fig.tight_layout()
    return fig, (ax0, ax1)


def get_trial_neural_population_traces(t0_events, fr_z, s_id, which_interval, unit_cols):
    """Every trial's full per-neuron population trace (n_bins x n_units, NOT reduced to a
    scalar - see get_trial_neural_magnitude_traces for that) - what
    _pc_block_with_separators' per-timebin PCA needs, since it requires each trial's whole
    (unreduced) neuron vector at every timebin, not a magnitude.
    get_trial_metric_traces' modal-shape filtering already works unchanged on a 2D
    value_fn result - `len(vals)` on a (n_bins, n_units) array still gives its bin count
    (the first axis), which is all the shape check needs."""
    return get_trial_metric_traces(t0_events, fr_z, s_id, which_interval,
                                    value_fn=lambda dat: dat[unit_cols].to_numpy(dtype=float))


def _smooth_2d(arr2d, window):
    """_smooth applied along axis 0 (time) independently for every column (unit) of a
    (n_bins, n_units) array - see _pc_block_with_separators.

    One `DataFrame.rolling` covering every column at once, NOT a python loop of
    `plotlib.trial_traces.smooth` per column. Same pandas kernel applied column-wise, so the
    result is identical (verified elementwise on real session data), but it constructs one
    DataFrame per trial instead of one Series per (trial, unit): the loop version issued
    48,147 pandas calls for a single 4-interval figure and was, at 20 of its 31 seconds, by
    far the most expensive thing in it - more than the PCA and the shuffle null together.
    The `.mask(isna())` is what keeps a NaN input NaN in the output rather than letting
    `min_periods=1` fill it from its neighbours (see that function for why that matters)."""
    if not window or window <= 1:
        return arr2d
    df = pd.DataFrame(arr2d, dtype=float)
    return df.rolling(window, center=True, min_periods=1).mean().mask(df.isna()).to_numpy()


def _fill_nan_columns(M):
    """Fill NaN entries of an (n_trials, n_units) matrix with that column's own
    across-trial mean (a column that's NaN in *every* trial - an always-silent unit, per
    get_FiringRate40msZ - is filled with 0 instead, since its own mean is itself NaN) -
    so a silent/no-variance unit contributes zero variance to a PCA rather than
    propagating NaN into the whole decomposition."""
    all_nan = np.all(np.isnan(M), axis=0)
    col_mean = np.zeros((1, M.shape[1]))
    if not all_nan.all():
        col_mean[:, ~all_nan] = np.nanmean(M[:, ~all_nan], axis=0, keepdims=True)
    return np.where(np.isnan(M), col_mean, M)


_PCA_SCALINGS = ('correlation', 'covariance')


def _prep_for_svd(M, scaling):
    """The matrix `_pca_basis_for_n_pcs` and `_var_explained_null` both decompose, and which
    of its columns survived: NaNs filled (see _fill_nan_columns), centered across trials, and
    - under `'correlation'` - divided by each column's own across-trial std, dropping the
    columns that have none. Returns (prepared, keep_mask), or (None, keep_mask) when
    `'correlation'` leaves fewer than two columns to correlate.

    Factored out so the basis path and the null path cannot drift apart in what they consider
    the same decomposition; the null would otherwise silently compare against a slightly
    different preprocessing than the curve it is the reference for."""
    if scaling not in _PCA_SCALINGS:
        raise ValueError(f"scaling must be one of {_PCA_SCALINGS}, got {scaling!r}")
    M = _fill_nan_columns(M)
    centered = M - M.mean(axis=0, keepdims=True)
    keep = np.ones(M.shape[1], dtype=bool)
    if scaling == 'correlation':
        col_sd = centered.std(axis=0, ddof=0)
        keep = col_sd > 1e-12
        if keep.sum() < 2:
            return None, keep
        centered = centered[:, keep] / col_sd[keep]
    return centered, keep


def _var_explained_svals(M, n_pcs, scaling):
    """`_pca_basis_for_n_pcs`' `var_explained` alone, via `compute_uv=False`.

    The null needs only the eigenvalue spectrum, never the eigenvectors, and skipping them is
    ~4x faster per SVD (measured 0.003s vs 0.012s for 25 draws of a 101x57 matrix) - which is
    the whole cost of the chance band. Conventions match _pca_basis_for_n_pcs exactly
    (degenerate -> 1.0), since the two are read against each other."""
    prepped, _ = _prep_for_svd(M, scaling)
    if prepped is None:
        return 1.0
    S = np.linalg.svd(prepped, compute_uv=False)
    total_var = np.sum(S ** 2)
    if total_var == 0:
        return 1.0
    return float(np.sum(S[:min(n_pcs, S.size)] ** 2) / total_var)


def _pca_basis_for_n_pcs(M, n_pcs, scaling='correlation'):
    """Top `n_pcs` PCA components (via SVD, trials as observations/repetitions, neurons as
    the dimension being reduced - see plot_subspace_angle_heatmap) of an (n_trials,
    n_units) matrix - a fixed component count at every timebin (unlike an earlier,
    since-replaced version of this function that instead grew/shrank k to hit a target
    cumulative-variance threshold), so canonical-angle comparisons across timebins (see
    plot_subspace_angle_heatmap) are always between equal-dimension subspaces. Capped at
    `Vt.shape[0]` (== min(n_trials, n_units) - 1 after centering) whenever that's smaller
    than `n_pcs`.

    `scaling` decides WHAT is decomposed, and it changes what `var_explained` below can even
    mean. This matters more than any other parameter in this file, so it is spelled out:

    - `'correlation'` (default): each unit's centered trace is divided by its own
      across-trial std at that timebin, i.e. the correlation matrix is decomposed. Every unit
      then contributes exactly the same total variance, so `var_explained` can only be raised
      by units *co-fluctuating*, which is the thing a dimensionality measure is supposed to
      report.
    - `'covariance'`: the raw centered matrix, i.e. what this function used to do
      unconditionally. Kept because it is the natural measure when the units' scales are
      themselves the signal - but be aware of its failure mode, which is severe on this data:
      a top-k eigenvalue share of a covariance matrix is maximal when ONE diagonal entry
      dominates, and one loud neuron is trivially one direction. Measured on real sessions
      (animal 6, mPFC): 8.6% of timebins showed a jump of +0.1-0.2 in var_explained that was
      entirely one unit in one trial - dropping that single unit, or that single trial,
      removed 96% of the excess, while a per-unit trial shuffle (which destroys every
      cross-neuron correlation but preserves each unit's own variance) left it standing. The
      culprits were near-silent units (median 0.009 Hz vs 0.230 Hz elsewhere) whose single
      spike becomes z~80 because get_FiringRate40msZ divides by a std estimated over a
      session in which they never fire. `'correlation'` removes that failure mode by
      construction; a firing-rate floor (see plot_session_kinematics_neural_and_pca's
      `min_unit_rate_hz`) removes its cause and is worth applying on top, since
      standardizing a unit that is zero in all trials but one turns it into a pure leverage
      point rather than a scale problem.

    Units with no across-trial variance at all are dropped from a `'correlation'`
    decomposition (they have no correlation with anything) and re-embedded afterwards as
    zero columns of `basis` / zero entries of `neuron_contrib`, so every return value keeps
    full `n_units` indexing and equal-dimension subspaces across timebins either way.

    Returns (basis, var_explained, neuron_contrib):

    - `basis` a (k, n_units) array (top-k rows of Vt).
    - `var_explained` those k components' cumulative fraction of this timebin's own
      across-trial variance - degenerate (no variance at all) timebins report 1.0 (nothing
      left unexplained) since dividing by a zero total_var is undefined.
    - `neuron_contrib` a (n_units,) array: each neuron's *eigenvalue-weighted* share of
      that same variance, `sum_k (S_k^2 * Vt[k, i]^2) / total_var` - i.e. how much of this
      timebin's total across-trial variance the top-k subspace captures *through neuron i*.
      Weighting each component's squared loading by its own eigenvalue (rather than reading
      raw loadings) is what makes this comparable across components: a neuron dominating a
      near-zero-eigenvalue PC contributes almost nothing to the subspace, however large its
      loading there is. Since every basis row is unit-norm, `neuron_contrib.sum() ==
      var_explained` exactly - so this decomposes the variance-explained value plotted
      elsewhere (see _draw_neuron_contribution_summary's ceiling line) into per-neuron parts,
      and a whole column of _neuron_contribution_matrix is on the same scale/units as it.
      All-NaN at a degenerate timebin: an apportionment of zero variance has no meaningful
      per-neuron split (and that SVD's basis is an arbitrary rotation), unlike
      `var_explained`'s 1.0 convention above.
    """
    n_units = M.shape[1]
    prepped, keep = _prep_for_svd(M, scaling)
    if prepped is None:   # nothing left to correlate - degenerate, same convention as below
        return np.zeros((max(min(n_pcs, 1), 1), n_units)), 1.0, np.full(n_units, np.nan)
    centered = prepped

    def _embed(rows):
        """(k, n_kept) basis rows back into full n_units columns (a no-op when nothing was
        dropped) - so the caller always gets one consistent neuron indexing, and
        plot_subspace_angle_heatmap keeps comparing subspaces that live in the same space."""
        if keep.all():
            return rows
        full = np.zeros((rows.shape[0], n_units))
        full[:, keep] = rows
        return full

    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    total_var = np.sum(S ** 2)
    k = min(n_pcs, Vt.shape[0])
    if total_var == 0:
        return _embed(Vt[:max(k, 1)]), 1.0, np.full(n_units, np.nan)
    var_explained = np.sum(S[:k] ** 2) / total_var
    contrib = (S[:k] ** 2) @ (Vt[:k] ** 2) / total_var
    if not keep.all():
        full_contrib = np.zeros(n_units)
        full_contrib[keep] = contrib
        contrib = full_contrib
    return _embed(Vt[:k]), var_explained, contrib


def _var_explained_null(M, n_pcs, scaling, reps, rng):
    """`reps` draws of var_explained from the null in which each unit's values are permuted
    INDEPENDENTLY across trials. That keeps every unit's own across-trial variance - hence the
    whole marginal variance profile, which is most of what a top-k eigenvalue share reflects -
    and destroys only the cross-neuron correlation, so `real - null` is exactly the part of the
    curve that is about population structure rather than about a few units happening to be more
    variable than the others at that timebin.

    Worth knowing before reading any variance-explained curve here: measured against this null
    on real sessions (animal 6, mPFC, 440 timebins), the real curve sits above the null's 95th
    percentile in ~89% of timebins but by a median of only 2.5 percentage points (0.326 vs
    0.308 under `'correlation'`; 0.389 vs 0.364 under `'covariance'`). The correlation
    structure is real and reliable, but it accounts for ~2.5pp of a ~33% curve - the curve's
    absolute level, and most of its shape, is the marginal profile. Drawing the null as a band
    under the curve (see _draw_neuron_contribution_summary) is what keeps that visible.

    All `reps` permutations are built at once via argsort of a random matrix rather than a
    per-column python loop over rng.permutation (~n_units x reps calls per timebin, which
    dominated the runtime of the first version)."""
    Mf = _fill_nan_columns(M)
    n_tr = Mf.shape[0]
    if n_tr < 3 or reps <= 0:
        return np.array([])
    # independent permutation per (rep, column): argsort along the trial axis of uniform noise
    idx = rng.random((reps, n_tr, Mf.shape[1])).argsort(axis=1)
    draws = np.take_along_axis(Mf[None, :, :], idx, axis=1)
    return np.array([_var_explained_svals(draws[r], n_pcs, scaling) for r in range(reps)],
                    dtype=float)


def _subspace_block_with_separators(t0_events, fr_z, s_id, which_intervals, unit_cols, gap_bins=2,
                                     smooth=5, n_pcs=8, scaling='correlation', null_reps=0,
                                     null_seed=0):
    """One PCA subspace basis per integer x-position across the full concatenated
    `which_intervals` range (see _pca_basis_for_n_pcs) - the exact same offsets/gaps
    _layout_intervals resolves elsewhere (see plot_session_kinematics_and_neural), not a
    locally-renumbered 0..N index, so plot_subspace_angle_heatmap's own ticks/gridlines
    can share the same x-axis as the position/neural rows above it. Gap positions (no
    trial data there) get an empty (0, n_units) basis - zero-dimensional, so it has no
    canonical angles against anything, rendering as a blank/white separator in the
    heatmap (same visual role as the zero-row trick in
    svm_plotting_lib._w_block_with_interval_separators, adapted for a variable-dimension
    subspace instead of a single fixed-length vector).

    `smooth` (default 5, *not* 1 like the display-only smoothing elsewhere in this file -
    see plot_trial_position_kinematics) is a centered rolling-mean window (see _smooth)
    applied to each trial's own per-neuron trace *before* PCA: PCA on raw single-40ms-bin
    firing rates is dominated by spike-count noise at that resolution, so smoothing the
    input is the default behavior here, not merely an optional extra.

    `n_pcs` (default 8) is the fixed component count every timebin's basis is truncated
    to (see _pca_basis_for_n_pcs) - capped, not padded, at timebins with fewer than n_pcs
    trials/units available. `scaling` ('correlation' by default) passes straight through to
    _pca_basis_for_n_pcs - read its docstring before changing it, the choice decides whether
    `var_explained` measures population dimensionality or single-unit variance heterogeneity.

    `null_reps` (default 0 = don't compute it) adds the per-timebin trial-shuffle null
    (see _var_explained_null) that says how much of `var_explained` is reachable without any
    cross-neuron correlation at all. It costs `null_reps` singular-value-only SVDs per timebin
    - **measured** 1.8s of a 4.6s real 113-bin figure at 25 reps, i.e. noticeable but not the
    thing to blame a slow notebook on (that was `_smooth_2d`, see its docstring). A fixed
    `null_seed` keeps a re-render numerically identical.

    Returns (x_coords, bases, var_explained, layout, contribs, null) - SIX elements, was five
    before `null` was added (a positional unpack in dim_analysis.py's throwaway cross-session
    cell needed updating; nothing else unpacks it). `null` is None when `null_reps` is 0, else
    a dict of (len(x_coords),) arrays 'mean'/'lo'/'hi' (the null's mean and 5th/95th
    percentiles, NaN at gap positions) plus the scalar 'reps'. `bases[i]` is the
    (k_i, n_units) basis for `x_coords[i]` (k_i == n_pcs except where capped - see
    _pca_basis_for_n_pcs); `var_explained[i]` is that basis's own cumulative fraction of
    `x_coords[i]`'s across-trial variance (NaN at gap positions, where there's no basis to
    report on); `layout` is _layout_intervals' own return (reused here so
    plot_subspace_angle_heatmap doesn't need to re-fetch it separately just to draw t0
    reference lines); `contribs[i]` is that basis's per-neuron, eigenvalue-weighted split
    of `var_explained[i]` (a (n_units,) array summing to it - see _pca_basis_for_n_pcs;
    all-NaN at gap positions, same as `var_explained` there).
    """
    layout, _ = _layout_intervals(t0_events, fr_z, s_id, which_intervals,
                                   value_fn=lambda dat: dat[unit_cols].to_numpy(dtype=float),
                                   gap_bins=gap_bins)
    if not layout:
        return np.array([]), [], [], layout, [], None

    left_edge = layout[0][3] + layout[0][2].min()
    right_edge = layout[-1][3] + layout[-1][2].max()
    x_coords = np.arange(left_edge, right_edge + 1)
    bases = [np.zeros((0, len(unit_cols))) for _ in x_coords]
    var_explained = [np.nan for _ in x_coords]
    contribs = [np.full(len(unit_cols), np.nan) for _ in x_coords]
    null = None
    if null_reps > 0:
        null = {'mean': np.full(len(x_coords), np.nan), 'lo': np.full(len(x_coords), np.nan),
                'hi': np.full(len(x_coords), np.nan), 'reps': int(null_reps)}
    rng = np.random.default_rng(null_seed)

    for _, traces, bins, offset in layout:
        stacked = np.stack([_smooth_2d(trial, smooth) for trial in traces], axis=0)  # (n_trials, n_bins, n_units)
        for local_t, b in enumerate(bins):
            idx = b + offset - left_edge
            bases[idx], var_explained[idx], contribs[idx] = _pca_basis_for_n_pcs(
                stacked[:, local_t, :], n_pcs, scaling=scaling)
            if null is not None:
                draws = _var_explained_null(stacked[:, local_t, :], n_pcs, scaling, null_reps, rng)
                if draws.size and np.isfinite(draws).any():
                    null['mean'][idx] = np.nanmean(draws)
                    null['lo'][idx] = np.nanpercentile(draws, 5)
                    null['hi'][idx] = np.nanpercentile(draws, 95)

    return x_coords, bases, var_explained, layout, contribs, null


def _layout_geometry(layout):
    """A layout's x-geometry alone - (interval, left bin, right bin, offset) per interval -
    which is what two blocks must agree on for their curves to be comparable at the same x."""
    return [(iv, int(bins.min()), int(bins.max()), int(offset)) for iv, _, bins, offset in layout]


def _warn_if_layout_differs(reference, other, name):
    """Layout offsets come from each interval's own modal (length, t0_offset) shape across
    the trials it was given (see get_trial_metric_traces), so a *subset* of trials can in
    principle resolve to a different shape and land its curve at shifted x positions. It
    doesn't happen in practice (interval geometry is a property of the interval, and the
    modal shape survives dropping trials), but it would be invisible in the figure if it
    ever did - so say so rather than plotting a silently misaligned curve."""
    ref_geom, other_geom = _layout_geometry(reference), _layout_geometry(other)
    if ref_geom != other_geom:
        print(f"  WARNING: {name} resolved a different interval geometry than the figure's "
              f"own ({other_geom} vs {ref_geom}) - its curve's x positions may be shifted")


def plot_subspace_angle_heatmap(ax, t0_events, fr_z, s_id, which_intervals, unit_cols=None,
                                 gap_bins=2, smooth=5, n_pcs=8, summary='mean', angle_range=(0, 90),
                                 scaling='correlation', block=None):
    """Heatmap of the canonical (principal) angle between the per-timebin PCA subspace
    (see _pca_basis_for_n_pcs/_subspace_block_with_separators) at every pair of
    timebins across `which_intervals`, sharing the exact same x-axis (bin positions/
    offsets) as plot_session_kinematics_and_neural's other rows. PCA runs independently
    per timebin: at each bin, that bin's own (n_trials, n_units) matrix is decomposed
    with trials as observations/repetitions and neurons as the dimension being reduced,
    keeping the top `n_pcs` (default 8) components - a fixed count at every timebin, so
    every pairwise comparison below is between equal-dimension subspaces (how much of
    each timebin's own variance those n_pcs components actually capture is instead
    plotted separately - see _draw_neuron_contribution_summary).

    Two subspaces of dimension k_i/k_j have min(k_i, k_j) canonical angles (via
    scipy.linalg.subspace_angles), each in [0, 90] degrees - unlike a plain
    vector-to-vector angle, a subspace has no notion of "opposite direction", so there's
    no [90, 180] half to this range, unlike an earlier, since-replaced version of this
    heatmap that compared single PC1 vectors directly. `summary` picks how those
    min(k_i, k_j) angles get reduced to the single scalar a heatmap cell needs: 'mean'
    (default - overall subspace overlap), 'max' (the worst-aligned pair of directions -
    the standard subspace-distance/Grassmannian convention) or 'min' (the best-aligned
    pair).

    No significance/chance-band masking is applied here (unlike
    svm_plotting_lib.plot_w_comparison's angle_2std_thr masking, for the previous
    single-vector version of this heatmap) - every comparable cell is shown at full
    opacity. Interval-gap separators are still shown (see
    _subspace_block_with_separators) - only the *masking*, not the gaps, was dropped.

    `block` (default None - fetch and decompose it here) lets a caller that has ALREADY run
    _subspace_block_with_separators for the same (s_id, which_intervals, gap_bins, smooth,
    n_pcs, scaling) hand over its `(x_coords, bases, layout)` instead. That is not an
    optimization worth ignoring: this function's own fetch+PCA pass cost as much as the entire
    rest of plot_session_kinematics_neural_and_pca, which was computing the identical block
    already and then paying for it a second time here (10s of a 31s figure). Nothing is
    checked about a passed-in block - it is the caller's job to pass one built with matching
    arguments, since a mismatched one silently mislabels the axes rather than failing.

    Renders with aspect='equal' (square cells) - sized correctly (no blank margins) by
    plot_session_kinematics_neural_and_pca giving this row's axes box a physical height
    equal to its width; called standalone with a mismatched axes box, the square
    constraint will instead shrink to fit and leave blank margins on the long side.

    Colored with viridis (sequential), not a diverging colormap: angles are inherently
    non-negative with no symmetric "opposite direction" structure the way a plain vector
    angle would have at 180 - a diverging colormap's implied meaningful midpoint doesn't
    apply here. `angle_range` (default (0, 90) - the true range a canonical angle can
    reach) sets the color scale's (vmin, vmax); narrowing it (e.g. to (20, 70)) adds
    contrast within the range that actually varies in the data, at the cost of clipping
    anything outside it to the nearest end color. Angles are always computed over the
    full [0, 90] range regardless - only their display is affected. Returns (ax, mesh): `mesh` is the
    pcolormesh mappable, for a caller that wants its own colorbar (see
    plot_session_kinematics_neural_and_pca's `draw_colorbar` option for the layout-safe
    way to add one to the combined figure).
    """
    if unit_cols is None:
        unit_cols = [c for c in fr_z.columns if c.startswith('Unit')]
    if block is not None:
        x_coords, bases, layout = block
    else:
        x_coords, bases, _, layout, _, _ = _subspace_block_with_separators(
            t0_events, fr_z, s_id, which_intervals, unit_cols, gap_bins=gap_bins,
            smooth=smooth, n_pcs=n_pcs, scaling=scaling)
    if len(x_coords) == 0:
        return ax, None

    reducer = {'mean': np.mean, 'max': np.max, 'min': np.min}[summary]
    n = len(x_coords)
    angles = np.full((n, n), np.nan)
    for i in range(n):
        if bases[i].shape[0] == 0:
            continue
        for j in range(i, n):
            if bases[j].shape[0] == 0:
                continue
            theta = subspace_angles(bases[i].T, bases[j].T)
            deg = reducer(np.degrees(theta))
            angles[i, j] = angles[j, i] = deg

    cmap = plt.cm.viridis.with_extremes(bad='white')
    X, Y = np.meshgrid(x_coords, x_coords)
    mesh = ax.pcolormesh(X, Y, np.ma.masked_invalid(angles), cmap=cmap,
                          vmin=angle_range[0], vmax=angle_range[1], shading='nearest')

    _draw_scaffolding(ax, layout, show_bin_lines=False)
    for _, _, _, offset in layout:
        ax.axhline(y=offset, color='k', linestyle='--', alpha=0.4, zorder=3)

    # adjustable='datalim' (not the 'box' default): 'box' has matplotlib resize/reposition
    # this axes' own box on every draw to enforce the square ratio, which silently undoes
    # plot_session_kinematics_neural_and_pca's own manual position override (needed to
    # correct a tight_layout/aspect interaction that otherwise misaligns this row's x-axis
    # against the sharex rows above it - confirmed on real data). 'datalim' instead adjusts
    # the data limits, never touching the box - a no-op here in practice, since the box is
    # already square (that override) and the data domain (x_coords vs itself) already is too.
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_ylabel('Time (bin)')
    return ax, mesh


# Fixed geometry/aesthetics for plot_session_kinematics_neural_and_pca - not exposed as
# parameters since nothing in this codebase has ever overridden them; edit here directly
# if a real need for a different value comes up.
_GAP_BINS = 2
_ROW_HEIGHTS_IN = (2, 1.6)  # row0 (position), row1 (neural metric), in inches
_WIDTH_PER_BIN_IN = 0.055
_MIN_WIDTH_IN = 3
_COLORBAR_HEIGHT_IN = 0.3
_VAR_EXPLAINED_CMAP = 'coolwarm'
_ANGLE_SUMMARY = 'mean'
_NEURON_CONTRIB_ROW_IN = 0.075  # rendered height per neuron row - see _neuron_contrib_height_in
_NEURON_CONTRIB_MIN_HEIGHT_IN = 1.0
_NEURON_CONTRIB_SUMMARY_HEIGHT_IN = 1.0
# Sequential, light-at-the-low-end (not viridis/plasma/magma, whose low end is near-black):
# most cells of that panel are neurons that only barely qualified at that timebin, and with a
# dark-low colormap every one of them carries as much visual weight as a genuine top
# contributor, which buries the block structure the row exists to show (compared against
# plasma/viridis/magma_r on realistic synthetic data - YlOrRd was the only one where the weak
# cells recede and the strong ones read as bands).
_NEURON_CONTRIB_CMAP = 'YlOrRd'
_NEURON_CONTRIB_VMAX_PCTL = 98  # autoscale ceiling percentile - see _draw_neuron_contribution_heatmap
# tight_layout hands a row roughly this fraction of its nominal height_ratios height once
# margins/labels are taken out (measured ~0.8 on realistic multi-interval figures) - used to
# turn a target *rendered* per-neuron row height into the nominal one to ask for, so a rough
# factor is enough.
_TIGHT_LAYOUT_ROW_SHRINK = 0.8


def _neuron_contrib_height_in(n_rows):
    """Nominal (pre-tight_layout) height for the neuron-contribution panel: it grows with the
    row count instead of being a fixed box, so every neuron keeps a fixed physical slice of
    ~_NEURON_CONTRIB_ROW_IN and its ID label stays drawable however many rows there are (the
    earlier fixed-height version silently dropped the labels past ~32 rows, which defeated
    the point of the panel - identifying individual neurons). Only the figure gets taller;
    nothing else about the layout changes. Sessions whose kept-row count differs will differ
    in total figure height, which matters for build_canvas' side-by-side assembly - in
    practice most sessions saturate `max_rows` and so come out identical anyway."""
    return max(_NEURON_CONTRIB_MIN_HEIGHT_IN, n_rows * _NEURON_CONTRIB_ROW_IN / _TIGHT_LAYOUT_ROW_SHRINK)


def _neuron_contribution_matrix(bases, contribs, n_units, select='per_pc', top_per_pc=3, top_n=12,
                                 max_rows=25, fill='dense', cluster=True):
    """Which neurons carry each timebin's PCA subspace, as a (n_kept_neurons, n_timebins)
    matrix - the data behind plot_neuron_contribution_heatmap. Takes
    _subspace_block_with_separators' own `bases`/`contribs` (rather than re-running the
    per-timebin PCA itself) so a caller that already has them doesn't pay for a second
    decomposition pass.

    Every cell holds that neuron's own eigenvalue-weighted contribution for the *whole*
    subspace at that timebin (see _pca_basis_for_n_pcs' `neuron_contrib`), never a per-PC
    value - so cell values never depend on `select`, which only decides which neurons get a
    row. `fill` decides what the kept rows show:

    - `'dense'` (default): every kept row is filled at every timebin - the selection is used
      *only* to rank/pick the rows (aggregated over the whole x-axis), then each of those
      neurons' actual contribution is shown everywhere. This is the honest version: a value
      exists at every (neuron, timebin), so leaving cells blank would hide known data, and it
      also fixes the clustering, which otherwise treats "wasn't nominated here" as a zero and
      so partly clusters rows by their *selection* pattern instead of by their contributions.
    - `'selected'`: only cells the per-timebin selection picked out are drawn, the rest blank.
      Shows the turnover of which neurons lead the subspace at each timebin as an explicit
      on/off pattern, at the cost of hiding values that are perfectly well defined and of
      making a column's apparent brightness partly a selection artifact.

    `select` (how rows get ranked, and what `fill='selected'` draws):

    - `'per_pc'`: each of that timebin's basis rows nominates its `top_per_pc`
      largest-|loading| neurons, so a timebin draws the union of up to `n_pcs * top_per_pc`
      of them (fewer when one neuron leads several PCs at once - a union, not a count).
      Reads as "who leads each individual eigenvector", and lets a neuron that dominates one
      low-eigenvalue PC show up next to the PC1 leaders. Its drawback is that the drawn set
      is *not* the largest-contributing neurons: a neuron ranking just below `top_per_pc` in
      several PCs at once can out-contribute a drawn one and still be blank, so a column sum
      is not "the top cells' share" of anything in particular.
    - `'top_contrib'`: the `top_n` largest-contribution neurons of that timebin, full stop -
      the eigenvectors only decide the *values*, not the selection. A column is then exactly
      the largest `top_n` terms of that timebin's variance split, so its sum is the biggest
      share `top_n` neurons can account for (and, unlike 'per_pc', is monotone in `top_n` and
      directly comparable across timebins). The tradeoff is the opposite one: a neuron that
      owns a single low-eigenvalue PC outright is dropped in favour of a mid-ranking
      contributor to PC1/PC2, so the panel says less about the structure *within* the
      subspace.

    Rows are capped at `max_rows`, keeping whichever neurons have the largest total
    contribution summed across the timebins where they were selected ("contributes most to
    the subspace in general", the reason for looking at this at all) - selecting more than a
    couple of neurons per timebin over a long multi-interval x-axis unions into far more
    distinct neurons than a fixed-height panel can show as distinguishable rows. The cap is
    reported by the caller, never silent.

    Summing only over *selected* timebins is what keeps `select` meaningful under
    `fill='dense'`, and it deliberately favours transient specialists: a neuron that dominates
    a PC during one interval and is quiet elsewhere outranks one with a mediocre contribution
    everywhere, which a plain sum over all timebins would reverse. For that plain "largest
    overall carriers" ranking instead, pass `select='top_contrib'` with `top_n` >= n_units, so
    every neuron counts at every timebin and the ranking becomes the unweighted total.
    Under `fill='selected'` the cap has a second, cosmetic consequence: a column can look pale
    for a reason that has nothing to do with that timebin, if its own leaders lost the
    across-the-whole-axis ranking (measured on real data: down to 2 of 13 selected cells
    surviving in the worst column, at max_rows=25). `fill='dense'` doesn't have that problem -
    every kept row is drawn everywhere.

    `cluster` (default True) orders the kept rows by hierarchical clustering (Ward, with
    scipy's optimal leaf ordering) of the very values that get drawn, so neurons that dominate
    the subspace during the same epochs end up as adjacent bands instead of being scattered
    over the panel. Each row is normalized by its own peak first (values are still drawn raw):
    without that, rows cluster by overall magnitude - strong contributors with unrelated timing
    grouping together - rather than by *when* they contribute, which is what makes the panel
    readable.

    Returns (rows, row_unit_idx, info): `rows` the (n_kept, n_timebins) value matrix in
    plot order, `row_unit_idx` the matching neuron (unit_cols) indices in that same order,
    `info` a dict for reporting and for the summary row (see
    _draw_neuron_contribution_summary): n_present/n_kept/n_dropped/kept_mass_frac overall,
    plus the per-timebin arrays col_sum_selected (variance fraction captured by the cells the
    per-timebin selection picked, before the row cap) and col_sum_drawn (captured by what is
    actually drawn - the kept rows' selected cells under `fill='selected'`, or those rows at
    every timebin under `'dense'`; NaN at gap positions), and their cell counts
    n_selected_per_col/n_drawn_per_col. The selected-vs-drawn pair is what makes the row cap
    visible instead of silent; the selected count is what makes the difference between the two
    `select` rules visible (variable under 'per_pc', flat at `top_n` under 'top_contrib').
    """
    if select not in ('per_pc', 'top_contrib'):
        raise ValueError(f"select must be 'per_pc' or 'top_contrib', got {select!r}")
    if fill not in ('dense', 'selected'):
        raise ValueError(f"fill must be 'dense' or 'selected', got {fill!r}")

    n_cols = len(bases)
    M = np.full((n_units, n_cols), np.nan)      # selected cells only
    D = np.full((n_units, n_cols), np.nan)      # every neuron, every timebin
    for i, (basis, contrib) in enumerate(zip(bases, contribs)):
        if basis.shape[0] == 0 or not np.isfinite(contrib).any():
            continue  # gap position, or a degenerate (zero-variance) timebin
        D[:, i] = contrib
        if select == 'top_contrib':
            k = min(top_n, n_units)
            # nan_to_num, not the raw array: an all-NaN-in-every-trial unit's contribution can
            # come out NaN, and argpartition would otherwise sort it as if it were large.
            top = np.argpartition(-np.nan_to_num(contrib, nan=-np.inf), k - 1)[:k]
            M[top, i] = contrib[top]
            continue
        for w in basis:
            k = min(top_per_pc, w.size)
            top = np.argpartition(-np.abs(w), k - 1)[:k]
            M[top, i] = contrib[top]

    finite = np.isfinite(M)
    has_cells = finite.any(axis=0)  # False at gap/degenerate timebins - kept NaN, not 0, below
    present = np.flatnonzero(finite.any(axis=1))
    nan_cols = np.full(n_cols, np.nan)
    info = {'n_present': int(present.size), 'n_kept': 0, 'n_dropped': 0, 'kept_mass_frac': np.nan,
            'col_sum_selected': nan_cols.copy(), 'col_sum_drawn': nan_cols.copy(),
            'n_selected_per_col': nan_cols.copy(), 'n_drawn_per_col': nan_cols.copy()}
    if present.size == 0:
        return np.zeros((0, n_cols)), present, info

    total_mass = np.nansum(M[present])
    ranked = present[np.argsort(-np.nansum(M[present], axis=1))]
    kept = ranked[:max_rows]
    info['n_kept'] = int(kept.size)
    info['n_dropped'] = int(ranked.size - kept.size)
    info['kept_mass_frac'] = float(np.nansum(M[kept]) / total_mass) if total_mass > 0 else np.nan
    drawn = D if fill == 'dense' else M
    drawn_finite = np.isfinite(drawn)
    info['col_sum_selected'][has_cells] = np.nansum(M, axis=0)[has_cells]
    info['col_sum_drawn'][has_cells] = np.nansum(drawn[kept], axis=0)[has_cells]
    info['n_selected_per_col'][has_cells] = finite.sum(axis=0)[has_cells]
    info['n_drawn_per_col'][has_cells] = drawn_finite[kept].sum(axis=0)[has_cells]

    if cluster and kept.size > 2:
        pattern = np.nan_to_num(drawn[kept], nan=0.0)
        peak = np.abs(pattern).max(axis=1, keepdims=True)
        pattern = pattern / np.where(peak == 0, 1.0, peak)
        Z = optimal_leaf_ordering(linkage(pattern, method='ward'), pdist(pattern))
        kept = kept[leaves_list(Z)]

    return drawn[kept], kept, info


def _draw_neuron_contribution_heatmap(ax, x_coords, rows, row_unit_idx, unit_cols, layout,
                                       value_range=None, panel_height_in=None):
    """Draw a _neuron_contribution_matrix result (see there for what a cell means) as one
    heatmap row of a figure whose x-axis is shared with everything else - see
    plot_neuron_contribution_heatmap for the standalone, fetches-its-own-data version.
    Values are drawn as percentages of the timebin's total across-trial variance.

    `value_range` (default None - autoscaled, see below) is a (min, max) pair *in
    fractions*, e.g. (0, 0.1) for 0-10%; fix it to make cell colors comparable across
    sessions the way `angle_range`/`var_explained_range` do elsewhere in this file, at the
    cost of clipping anything outside it to the end colors.

    Passing None autoscales instead, to (0, `_NEURON_CONTRIB_VMAX_PCTL`th percentile of the
    drawn cells) rather than (0, their max): real sessions' contributions are heavy-tailed
    (confirmed on real data - a single neuron in a single timebin reached several times the
    next-largest cell), and scaling to that one outlier flattened the entire rest of the panel
    to the palest end of the colormap, hiding exactly the structure this row is for. A fixed
    range is still the better default, since that percentile turned out to be stable anyway
    (7.9-11.0% across 6 real sessions, both `select` rules) - so a fixed 0-10% gets the same
    contrast *and* the same color meaning from session to session, which autoscaling can't.

    Row (neuron-ID) tick labels are always drawn, shrinking with the rows if need be (down to
    2pt) rather than being dropped: the panel exists to identify individual neurons, so a
    label too small to read comfortably is still worth more than no label. `panel_height_in`
    (default None - assume the height grows with the row count, per _neuron_contrib_height_in,
    which keeps the labels at a constant ~4.6pt) is only used to pick that font size, so a
    caller drawing into a fixed-size axes should pass its real height to avoid overlap.
    Returns the pcolormesh mappable (None if there was nothing to draw), for a caller's own
    colorbar.
    """
    _draw_scaffolding(ax, layout, show_bin_lines=False)
    ax.set_ylabel('Neuron\n(clustered)')
    if rows.shape[0] == 0:
        return None

    vals = rows * 100
    if value_range is None:
        vmax = np.nanpercentile(vals, _NEURON_CONTRIB_VMAX_PCTL)
        vmin, vmax = 0.0, (vmax if vmax > 0 else np.nanmax(vals))
    else:
        vmin, vmax = value_range[0] * 100, value_range[1] * 100
    cmap = plt.get_cmap(_NEURON_CONTRIB_CMAP).with_extremes(bad='white')
    X, Y = np.meshgrid(x_coords, np.arange(rows.shape[0]))
    mesh = ax.pcolormesh(X, Y, np.ma.masked_invalid(vals), cmap=cmap, vmin=vmin, vmax=vmax,
                         shading='nearest')
    ax.set_ylim(-0.5, rows.shape[0] - 0.5)

    if panel_height_in is None:
        panel_height_in = _neuron_contrib_height_in(rows.shape[0])
    row_pt = _TIGHT_LAYOUT_ROW_SHRINK * panel_height_in * 72 / rows.shape[0]
    ax.set_yticks(np.arange(rows.shape[0]))
    ax.set_yticklabels([unit_cols[i].replace('Unit', '') for i in row_unit_idx],
                       fontsize=min(6, max(2, row_pt * 0.85)))
    ax.tick_params(axis='y', length=2, pad=1)
    return mesh


def _varies(values):
    """True if `values` has at least two distinct finite entries - i.e. plotting it as a curve
    would show something. Used to drop constant-by-construction lines (see
    _draw_neuron_contribution_summary) rather than drawing a flat line that reads as a
    threshold."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return finite.size > 0 and finite.min() != finite.max()


def _barely_varies(values, ceiling, min_span_frac=0.4):
    """True if `values` never moves by more than `min_span_frac` of its own structural
    `ceiling` - it *technically* varies (so `_varies` keeps it) but not enough to be worth its
    own axis, which is the stronger thing to ask of a twin-axis line.

    The selected-cell count under `select='per_pc'` is the case this exists for. Its ceiling is
    `n_pcs * top_per_pc` - the union of every eigenvector's leaders, when no two eigenvectors
    share one - and on a real session it ran 14-18 of a possible 18: a 4-cell wiggle, drawn on
    an axis scaled to ~29, which reads as a flat line pinned at the top while costing a whole
    right-hand axis and label. Keyed on the span relative to the *structural* ceiling rather
    than on any particular count, so it stays correct when `n_pcs`/`top_per_pc` change, and it
    subsumes the exactly-constant case.

    `min_span_frac=0.4` is deliberately generous rather than tight. Measured spans were 3, 4, 5
    and 3 of 18 across four real sessions (and 1 of 6 at `top_per_pc=1`), so a 0.25 threshold
    put one session on the wrong side of the line - making the twin axis appear on some panels
    and not others, which breaks the side-by-side canvas assembly these figures exist for (see
    build_canvas). 0.4 clears every measured span with margin, so the axis is consistently
    absent unless the count genuinely moves over a large part of its range."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0 or not np.isfinite(ceiling) or ceiling <= 0:
        return False
    return (v.max() - v.min()) < min_span_frac * ceiling


_SESSION_TIME_NAMES = {2: ('early', 'late'), 3: ('early', 'mid', 'late')}


def _session_time_names(n_bins):
    """Ordered names for a `groupby='session_time'` split - spelled out for the 2/3-bin
    cases anyone actually asks for, numbered (q1..qn) beyond that."""
    return _SESSION_TIME_NAMES.get(n_bins, tuple(f'q{i + 1}' for i in range(n_bins)))


def trial_group_labels(t0_events, behavior, groupby, n_bins=3, s_id=None):
    """One group label per *row* of `t0_events` (i.e. per (trial, zone) row, not per trial),
    plus the group values in display order: `(labels, order)`.

    Rows on a label-less trial are NaN and belong to no group. `s_id` restricts the work
    (and the labels) to one session - which is what the per-session figure wants, and keeps
    this from re-grouping the whole multi-session behavior table once per figure.

    `groupby` is either a per-trial column of `behavior` ('cue', 'choice_R1', 'choice_R2',
    'trial_outcome', ...), collapsed to one value per trial with .first() since behavior is
    per 40ms bin and its leading bins are NaN - or the special 'session_time', which splits
    each session's own trials into `n_bins` equal-count groups *in trial order*
    (early/mid/late for the default 3). session_time is ordered, so `order` is early->late
    rather than alphabetical; a behavior column's `order` is its sorted unique values.
    """
    idx = t0_events.index
    sessions = np.asarray(idx.get_level_values('session_id'))
    trials = np.asarray(idx.get_level_values('trial_id'))
    mask = np.ones(len(idx), dtype=bool) if s_id is None else (sessions == s_id)
    labels = pd.Series(np.nan, index=idx, dtype=object)

    if groupby == 'session_time':
        names = _session_time_names(n_bins)
        for sess in pd.unique(sessions[mask]):
            m = mask & (sessions == sess)
            sess_trials = np.unique(trials[m])
            # equal-count (not equal-duration) bins: n_bins slices of the session's own
            # trial order, so each group carries the same number of trials by construction -
            # which is exactly what makes their variance-explained curves comparable (see
            # `groupby_match_n` in plot_session_kinematics_neural_and_pca for why that matters).
            edges = np.linspace(0, len(sess_trials), n_bins + 1).round().astype(int)
            per_trial = {t: names[k] for k in range(n_bins)
                         for t in sess_trials[edges[k]:edges[k + 1]]}
            labels.iloc[np.flatnonzero(m)] = np.array([per_trial[t] for t in trials[m]],
                                                       dtype=object)
        present = [nm for nm in names if (labels == nm).any()]
        return labels, present

    if groupby not in behavior.columns:
        raise ValueError(f"groupby={groupby!r} is neither 'session_time' nor a column of "
                         f"behavior ({sorted(behavior.columns)})")
    beh = behavior if s_id is None else behavior.xs(s_id, level='session_id', drop_level=False)
    per_trial = beh.reset_index().groupby(['session_id', 'trial_id'])[groupby].first()
    values = per_trial.reindex(idx).to_numpy()
    labels.iloc[np.flatnonzero(mask)] = values[mask]
    order = sorted(pd.unique(labels.dropna()))
    return labels, order


def _group_shuffle_band(t0_events, data, s_id, which_intervals, value_fn, neural_layout,
                        group_labels, group_values, reps=200, seed=0, smooth=1,
                        pctl=(5, 95)):
    """The band row 1 draws when `draw_shuffle=True`: what a group's own mean trace looks like
    when the trial split is RANDOM.

    Per timebin, `reps` times, the real group labels are permuted across trials (a permutation,
    so every group keeps its exact trial count) and each shuffled group's mean is recomputed;
    the band is the `pctl` (default 5th-95th) percentile of that pooled distribution. A real
    group line leaving the band at some timebin is a split effect that a random split of these
    same trials would rarely produce. That is a genuine test of the comparison this row is
    for, which is why it lives here rather than around the variance-explained curve - the null
    there answers a question about dimensionality, not about conditions.

    It replaces the sem band rather than adding to it: two differently-meaning shaded regions
    in one row cannot be told apart. Note the band is *wider* than the sem band it replaces -
    5-95% of a null is +/-1.64 sd of it, where sem is +/-1 - so the row will look noisier at
    first glance without anything having changed about the data.

    Group sizes matter: a smaller group has a wider null, and the band drawn is the union over
    the groups actually present, i.e. it is the *largest* group's band that gets hidden inside
    the smallest one's. With `groupby_match_n=True` (the default) every group has the same
    count and the union is exact; this prints a warning when they differ so the union isn't
    read as one common level.

    Built from its own `get_trial_metric_traces` pass (with `return_trial_ids=True`, the only
    way to join a trace back to its trial's label after modal-shape filtering), reusing
    `neural_layout`'s intervals so it cannot disagree with what row 1 drew. One extra fetch of
    the same traces, which is why this is opt-in.

    `group_labels` must be aligned row-for-row with the `t0_events` passed in - if the caller
    filtered that table, it has to filter the labels by the same mask."""
    label_of = {}
    sess = np.asarray(t0_events.index.get_level_values('session_id')) == s_id
    tids = np.asarray(t0_events.index.get_level_values('trial_id'))[sess]
    labs = np.asarray(group_labels)[sess]
    for tid, lab in zip(tids, labs):
        # t0_events has one row per (trial, zone), so a trial_id repeats with the same label;
        # NaN is a trial the grouping (or groupby_match_n's subsample) excluded.
        if pd.notna(lab):
            label_of.setdefault(tid, lab)

    rng = np.random.default_rng(seed)
    bands, sizes = {}, {}
    for interval, _, _, _ in neural_layout:
        traces, bins, trial_ids = get_trial_metric_traces(
            t0_events, data, s_id, interval, value_fn=value_fn, return_trial_ids=True)
        if bins is None:
            continue
        labels = np.array([label_of.get(t, None) for t in trial_ids], dtype=object)
        keep = np.array([lab in group_values for lab in labels])
        if keep.sum() < len(group_values) + 1:
            print(f"  shuffle band: {interval} has only {int(keep.sum())} labelled trials, skipped")
            continue
        X = np.asarray([_smooth(vals, smooth) for vals in traces], dtype=float)[keep]
        labels = labels[keep]
        counts = [int(np.sum(labels == g)) for g in group_values]
        sizes[interval] = counts
        draws = np.empty((reps * len(group_values), X.shape[1]), dtype=float)
        r = 0
        for _ in range(reps):
            perm = rng.permutation(len(labels))
            for g in group_values:
                # index by the permutation restricted to this group's own (unchanged) size:
                # equivalent to shuffling the labels, and it keeps every group's n exact
                draws[r] = np.nanmean(X[perm[labels == g]], axis=0)
                r += 1
        with np.errstate(invalid='ignore'):
            bands[interval] = (np.nanpercentile(draws, pctl[0], axis=0),
                               np.nanpercentile(draws, pctl[1], axis=0))
    # A one-trial difference is normal even under groupby_match_n: the counts are matched on
    # t0_events rows, while these are counted after each interval's own modal-shape filtering,
    # which can drop a trial from one group and not the other. Only a materially uneven split
    # makes the union band misleading, so only warn about that.
    uneven = {iv: c for iv, c in sizes.items() if max(c) > 1.1 * max(min(c), 1)}
    if uneven:
        print(f"  shuffle band: group trial counts differ materially ({uneven}) - the band is "
              f"the union over groups, so it is too wide for the largest group. Set "
              f"groupby_match_n=True.")
    return bands


def _match_group_trial_counts(t0_events, labels, order, s_id, seed=0):
    """Drop trials (to NaN in `labels`) until every group of session `s_id` has the same
    number of them, and return `(labels, n_per_group)`.

    Not cosmetic: a PCA subspace's variance explained is biased upward when it is fitted on
    fewer trials, so a 53-trial group and a 34-trial group produce curves that differ partly
    because of their trial counts and not because of the grouping - which is the one
    comparison the group curves exist to make. Which excess trials go is a fixed-seed random
    draw (np.random.default_rng(`seed`)), not the first/last n, so dropping doesn't itself
    confound the split with time-in-session.
    """
    sess = np.asarray(t0_events.index.get_level_values('session_id')) == s_id
    trials = np.asarray(t0_events.index.get_level_values('trial_id'))
    per_group = {g: np.unique(trials[sess & (labels == g).to_numpy()]) for g in order}
    n_min = min((len(t) for t in per_group.values()), default=0)
    if n_min == 0 or all(len(t) == n_min for t in per_group.values()):
        return labels, n_min
    rng = np.random.default_rng(seed)
    keep = np.concatenate([rng.choice(t, n_min, replace=False) for t in per_group.values()])
    labels = labels.copy()
    labels[sess & ~np.isin(trials, keep)] = np.nan
    dropped = {g: len(t) - n_min for g, t in per_group.items() if len(t) > n_min}
    print(f"  group trial counts matched to n={n_min} per group (dropped {dropped})")
    return labels, n_min


_NEURON_CONTRIB_SEL_COLOR = '#e08214'   # selected cells (pre row cap)
_NEURON_CONTRIB_DRAWN_COLOR = '#b2182b'  # cells actually drawn in the heatmap
_NEURON_CONTRIB_COUNT_COLOR = '#4393c3'  # cells per timebin, on the twin axis


def _draw_neuron_contribution_summary(ax, x_coords, info, var_explained, layout, select,
                                       top_per_pc, top_n, n_pcs, fill='dense', null=None,
                                       mass_line='drawn'):
    """Thin row under the neuron-contribution heatmap that says, in the one unit that's
    intuitive here (% of that timebin's total across-trial variance), how much of the picture
    above is actually accounted for - and what the selection settings cost:

    `mass_line` decides which mass curve is drawn, and the default is deliberately ONE of
    them: `'drawn'` (default) - what the neurons visible in the heatmap above sum to, i.e.
    "how much of this timebin's variance can I actually see in that column"; `'selected'` -
    what the per-timebin selection rule picked, before the row cap; `'both'` - the old
    two-curve version. The two answer nearly the same question at this row's height and
    drawing both mostly obscured the one being read; `'selected'`'s own job (making the
    `select` rule legible) is better served by the count line below when that line varies.

    - dotted gray, `<n_pcs> PCs explained var`: the full variance explained by the PCA subspace,
      the one curve in this figure that is about the whole population. This is the ceiling: it's what the
      *whole* population's contributions sum to, so no other line here can exceed it. Kept even
      under a single mass curve: without it the mass curve is a bare percentage with nothing to
      read it against, and the per-interval ratio annotation has no denominator.
    - shaded gray band, `chance (trial-shuffled)`, when `null` is passed: the 5th-95th
      percentile of that same ceiling under a null that permutes each unit independently
      across trials (see _var_explained_null), i.e. the level the ceiling reaches with every
      cross-neuron correlation destroyed but each unit's own variance intact. Read the ceiling
      *against this band*, never against zero: on real sessions the ceiling sits only ~2.5
      percentage points above it, so most of its height - and most of its shape - is the
      marginal variance profile rather than population structure. The per-interval `+N.Npp`
      annotation is that gap's own median, which is the number worth quoting.
    - orange, `selected`: what the cells `select` picked out at each timebin sum to. The gap
      between this and the ceiling is the variance living in neurons the rule didn't pick -
      i.e. how spread-out (vs. concentrated on a few units) the subspace is at that timebin.
    - dark red, `shown neurons`: what the cells *actually drawn above* sum to. Under
      `fill='dense'` that's the kept rows' whole contribution at every timebin, so this is the
      one line to read as "the neurons you can see carry this much", and it can exceed the
      orange line wherever a shown neuron wasn't selected at that timebin. Under
      `fill='selected'` it's the selected cells that survived the `max_rows` cap, and the gap
      below orange is purely that cap - a column can look pale just because its leaders lost
      the across-the-whole-axis ranking, which is visible here rather than mistakable for
      structure.
    - blue, right axis, in *cells* rather than %: how many cells each timebin selected (dashed)
      and how many of those survived the row cap (solid step). Either is drawn only if it
      actually varies, so this axis appears at all only when there's something to see - see the
      comment in the code. The selected count is the clearest read on what `select` means:
      `'per_pc'` lets it vary (each eigenvector nominates its own leaders, and they overlap by
      different amounts bin to bin, so a low count *is* a concentrated subspace), while
      `'top_contrib'` pins it at `top_n` by construction (nothing to plot - the information is
      in the orange line's height instead).

    Per interval, the median drawn/ceiling ratio is annotated in the drawn line's own color -
    one number per interval for "what fraction of the subspace's variance the neurons you can
    see here carry", which is the summary worth quoting.
    """
    if mass_line not in ('drawn', 'selected', 'both'):
        raise ValueError(f"mass_line must be 'drawn', 'selected' or 'both', got {mass_line!r}")
    ve = np.asarray(var_explained, dtype=float) * 100
    sel = info['col_sum_selected'] * 100
    drawn = info['col_sum_drawn'] * 100
    rule = f'top-{top_per_pc}/PC' if select == 'per_pc' else f'top-{top_n} contrib'
    null_mean = np.asarray(null['mean'], dtype=float) * 100 if null is not None else None

    _draw_scaffolding(ax, layout, show_bin_lines=False)
    # Behind every curve (zorder 1): it is a reference level, not a quantity of its own.
    if null is not None:
        ax.fill_between(x_coords, np.asarray(null['lo']) * 100, np.asarray(null['hi']) * 100,
                        color='gray', alpha=0.30, linewidth=0, zorder=1,
                        label='chance (trial-shuffled)')
    ax.plot(x_coords, ve, color='gray', linestyle=':', linewidth=1,
            label=f'{n_pcs} PCs explained var (all units)', zorder=2)
    if mass_line in ('selected', 'both'):
        ax.plot(x_coords, sel, color=_NEURON_CONTRIB_SEL_COLOR, linewidth=1,
                label=f'selected ({rule})', zorder=2)
    if mass_line in ('drawn', 'both'):
        ax.plot(x_coords, drawn, color=_NEURON_CONTRIB_DRAWN_COLOR, linewidth=1.2,
                label='shown neurons' if fill == 'dense' else 'drawn (after row cap)', zorder=3)
    highest = [ve] + ([np.asarray(null['hi']) * 100] if null is not None else [])
    highest = np.concatenate(highest)
    top = min(100, (np.nanmax(highest) if np.isfinite(highest).any() else 100) * 1.35)
    # Headroom above for the legend, and a band *below* zero reserved for the per-interval
    # annotations: at this row's physical height a 6pt label spans >10% of the axes, so text
    # placed just above y=0 sat right on top of the curves wherever they dipped low
    # (confirmed on real data before reserving the band).
    ax.set_ylim(-top * 0.18, top)
    ax.set_yticks([t for t in (0, 25, 50, 75, 100) if t <= top])
    ax.set_ylabel('Variance captured\n(% of total)')

    # Cell counts live on a twin axis in different units (cells, not %), so a *constant* one is
    # worse than useless here: it draws a horizontal line straight across the variance curves
    # at whatever height its own scale happens to put it, reading as a threshold on the left
    # axis (real confusion, reported on a 'top_contrib'/dense figure where the flat 12-cells
    # line looked like it sat above the "all units" variance curve at 60%). Both counts can be
    # constant by construction - the selected count under 'top_contrib' (always `top_n`, which
    # the orange line's own label already states), the drawn count under dense fill (always the
    # kept-row count) - so only draw the ones that actually vary, and skip the twin axis
    # entirely when neither does.
    # ...and a line that only wiggles within a small fraction of its own structural ceiling is
    # just as uninformative as a constant one, without being caught by _varies (see
    # _barely_varies): under 'per_pc' the selected count ran 14-18 of a possible 18 on a real
    # session, which reads as pinned at the top while costing a whole right-hand axis.
    sel_ceiling = n_pcs * top_per_pc if select == 'per_pc' else min(top_n, info['n_present'])
    count_lines = [(info['n_selected_per_col'], 'cells selected', '--', None, sel_ceiling),
                   (info['n_drawn_per_col'], 'cells drawn', '-', 'steps-mid', info['n_kept'])]
    count_lines = [c for c in count_lines
                   if _varies(c[0]) and not _barely_varies(c[0], c[4])]
    twin = None
    if count_lines:
        twin = ax.twinx()
        # twinx() re-shows its own x tick labels over the (hidden) shared ones - see the same
        # fix for the mean_z twin axis in plot_session_kinematics_neural_and_pca.
        twin.tick_params(axis='x', labelbottom=False, bottom=False)
        for values, label, style, drawstyle, _ in count_lines:
            twin.plot(x_coords, values, color=_NEURON_CONTRIB_COUNT_COLOR, linewidth=0.8,
                      linestyle=style, drawstyle=drawstyle or 'default', label=label, zorder=2)
        cap = max(np.nanmax(v) for v, *_rest in count_lines) * 1.6
        # Same 0.18 padding fraction as the left axis, so both axes' zero lands at the same
        # height and the count lines stay out of the annotation band too (they otherwise ran
        # straight through the numbers wherever a timebin drew few cells).
        twin.set_ylim(-cap * 0.18, cap)
        twin.set_yticks([t for t in twin.get_yticks() if t >= 0])
        twin.set_ylabel('Cells\nper bin', color=_NEURON_CONTRIB_COUNT_COLOR)
        twin.tick_params(axis='y', colors=_NEURON_CONTRIB_COUNT_COLOR)
        twin.spines[['top']].set_visible(False)

    handles = ax.get_legend_handles_labels()[0]
    if twin is not None:
        handles += twin.get_legend_handles_labels()[0]
    ax.legend(handles=handles, fontsize=5, ncol=5, frameon=False, loc='upper left',
              handlelength=1.6, columnspacing=1.0, borderpad=0.1)

    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = (sel if mass_line == 'selected' else drawn) / ve
    excess = ve - null_mean if null_mean is not None else None
    for _, _, bins, offset in layout:
        idx = bins + offset - x_coords[0]
        idx = idx[(idx >= 0) & (idx < len(ratio))]
        vals = ratio[idx]
        if not np.isfinite(vals).any():
            continue
        # Both numbers are placed at a fraction of the interval's OWN bin span, not at a fixed
        # `offset + N`: intervals differ in how much of their window sits before t0
        # (measured: track_start -5..19, cue_entry -3..21, cue_exit -20..5, R1_entry -4..29),
        # so the old fixed offset+5 landed on cue_exit's very right edge - outside its own
        # panel, next to the following interval's numbers. Fractions keep every interval's
        # pair inside it and in the same left-to-right order. 0.35/0.68 rather than something
        # more centred because t0 itself sits at a different relative position in each
        # interval (0.21 of the span in track_start, 0.80 in cue_exit) and these two keep both
        # labels clear of that dashed line in every interval this figure draws.
        x_lo, span = offset + bins.min(), offset + bins.max() - (offset + bins.min())
        ax.text(x_lo + 0.35 * span, 0.01, f'{np.nanmedian(vals):.0%}',
                transform=ax.get_xaxis_transform(), fontsize=6, ha='center', va='bottom',
                color=(_NEURON_CONTRIB_SEL_COLOR if mass_line == 'selected'
                       else _NEURON_CONTRIB_DRAWN_COLOR))
        if excess is None:
            continue
        e = excess[idx]   # the over-chance gap, the number worth quoting from this row
        if not np.isfinite(e).any():
            continue
        ax.text(x_lo + 0.68 * span, 0.01, f'+{np.nanmedian(e):.1f}pp',
                transform=ax.get_xaxis_transform(), fontsize=6, ha='center', va='bottom',
                color='gray')
    return twin


def plot_neuron_contribution_heatmap(ax, t0_events, fr_z, s_id, which_intervals, unit_cols=None,
                                      gap_bins=2, smooth=5, n_pcs=8, select='per_pc', top_per_pc=3,
                                      top_n=12, max_rows=25, fill='dense', value_range=None,
                                      verbose=True, scaling='correlation'):
    """Standalone (fetches its own per-timebin PCA) version of the neuron-contribution
    heatmap: rows are neuron IDs, columns are timebins on the same concatenated
    `which_intervals` x-axis as every other row in this file, and a cell shows how much of
    that timebin's across-trial variance the top-`n_pcs` subspace captures through that
    neuron - drawn for whichever neurons `select` picks out (see _neuron_contribution_matrix
    for the selection/capping/clustering rules, and _pca_basis_for_n_pcs for the value
    itself). plot_session_kinematics_neural_and_pca draws the same panel from an
    already-computed subspace block instead of calling this. Returns (ax, mesh)."""
    if unit_cols is None:
        unit_cols = [c for c in fr_z.columns if c.startswith('Unit')]
    x_coords, bases, _, layout, contribs, _ = _subspace_block_with_separators(
        t0_events, fr_z, s_id, which_intervals, unit_cols, gap_bins=gap_bins,
        smooth=smooth, n_pcs=n_pcs, scaling=scaling)
    if len(x_coords) == 0:
        return ax, None
    rows, row_unit_idx, info = _neuron_contribution_matrix(
        bases, contribs, len(unit_cols), select=select, top_per_pc=top_per_pc, top_n=top_n,
        max_rows=max_rows, fill=fill)
    if verbose:
        _report_neuron_contribution(info, select, top_per_pc, top_n, max_rows, fill=fill)
    mesh = _draw_neuron_contribution_heatmap(ax, x_coords, rows, row_unit_idx, unit_cols, layout,
                                              value_range=value_range)
    return ax, mesh


def _report_neuron_contribution(info, select, top_per_pc, top_n, max_rows, fill='dense'):
    """Print which selection rule ran and what the row cap dropped - the panel itself can't
    show either, and silently truncating to max_rows would read as "these are all the neurons
    that ever mattered"."""
    rule = (f"top-{top_per_pc} per PC per timebin" if select == 'per_pc'
            else f"top-{top_n} contributors per timebin")
    msg = (f"  neuron contribution: {info['n_kept']}/{info['n_present']} neurons shown "
           f"({rule}, ranked over selected bins, {fill} fill)")
    if info['n_dropped']:
        msg += (f" - {info['n_dropped']} dropped by max_rows={max_rows}, kept rows still "
                f"carry {info['kept_mass_frac']:.0%} of the total selected contribution")
    print(msg)


def _apply_unit_rate_floor(unit_cols, s_id, unit_rate_hz, min_hz):
    """`unit_cols` minus the units whose own mean firing rate in this session is below
    `min_hz`, reported rather than silently applied.

    Why this exists: a unit that fires at ~0.005 Hz (one spike every few minutes) is z-scored
    by get_FiringRate40msZ against a std estimated over a session in which it is almost always
    zero, so its one spike arrives as z~80 and can single-handedly own a timebin's covariance
    spectrum - see _pca_basis_for_n_pcs' `scaling` note for the measurements. `'correlation'`
    scaling stops such a unit dominating by scale, but standardizing a column that is zero in
    every trial but one turns it into a leverage point instead, so the floor is worth applying
    on top of it, not as an alternative.

    Applied once, to the unit list every row of the figure uses - the population RMS/mean
    lines, the PCA, the neuron rows and the angle heatmap - so the whole figure describes one
    population. Rates come from the caller (this module does no I/O): pass
    plotlib.session_data.unit_mean_rates' DataFrame, or any Series/DataFrame indexed by unit
    column name (a DataFrame is indexed by session_id first). Passing `min_hz` without rates
    raises rather than quietly skipping the filter."""
    if min_hz is None:
        return unit_cols
    if unit_rate_hz is None:
        raise ValueError("min_unit_rate_hz was given but unit_rate_hz is None - the plotting "
                         "library does no I/O, so pass the rates in (see "
                         "plotlib.session_data.unit_mean_rates)")
    rates = unit_rate_hz
    if isinstance(rates, pd.DataFrame):
        if s_id not in rates.index:
            raise KeyError(f"unit_rate_hz has no row for session {s_id!r} "
                           f"(has {list(rates.index)[:3]}...)")
        rates = rates.loc[s_id]
    missing = [c for c in unit_cols if c not in rates.index]
    if missing:
        raise KeyError(f"unit_rate_hz is missing {len(missing)} of this figure's unit columns "
                       f"(e.g. {missing[:3]}) - it must cover every column being plotted")
    kept = [c for c in unit_cols if float(rates[c]) >= min_hz]
    dropped = [c for c in unit_cols if c not in set(kept)]
    if len(kept) < 3:
        raise ValueError(f"min_unit_rate_hz={min_hz} leaves only {len(kept)} of {len(unit_cols)} "
                         f"units in session {s_id} - nothing to decompose")
    if dropped:
        worst = sorted(dropped, key=lambda c: float(rates[c]))[:5]
        print(f"  unit rate floor {min_hz} Hz: kept {len(kept)}/{len(unit_cols)} units, dropped "
              f"{len(dropped)} (lowest: "
              + ", ".join(f"{c.replace('Unit', '')}@{float(rates[c]):.3f}Hz" for c in worst)
              + (", ..." if len(dropped) > 5 else "") + ")")
    return kept


def _neuron_contrib_range_default(scaling, n_units):
    """The default cell color range for the neuron-contribution heatmap, which cannot be one
    fixed pair because `scaling` changes the range the values can even occupy.

    Under `'covariance'`: (0, 0.10), the measured-stable choice (the 98th percentile of drawn
    cells sat between 7.9% and 11.0% across 6 real sessions).

    Under `'correlation'`: (0, 0.8/n_units). Standardizing gives every unit exactly
    `1/n_units` of the timebin's total variance, so `1/n_units` is the arithmetic ceiling for a
    single cell (a cell at it means "this neuron lies entirely inside the subspace") and the
    scale is a fraction *of that*. Reusing the covariance range here instead would render the
    whole panel in the palest 17% of the colormap (1/57 = 1.75% of a 0-10% scale), i.e. blank.
    The absolute number differs between sessions with different unit counts, but its *meaning*
    (fraction of that neuron's own variance captured) does not, so colors stay comparable
    across sessions - which is the property a fixed range exists for.

    0.8 rather than 1.0 comes from the same measurement that picked 0.10 for the covariance
    case: the 98th percentile of drawn cells, in units of the 1/n_units ceiling, was 0.65-0.87
    (median 0.745) across 4 real sessions at max_rows=12/dense/per_pc, with the single largest
    cell reaching 0.83-1.01. Worth knowing when reading that panel: under `'correlation'` the
    drawn cells are genuinely much more uniform than under `'covariance'` (measured p50/p90/p98
    of 0.41/0.62/0.75 of the ceiling, versus 0.41/1.36/2.35 - i.e. a 1.8x spread instead of a
    5.7x one). The flatter-looking panel is the finding, not a scaling mistake: once no unit
    can dominate by scale, there is no small cast carrying the subspace."""
    if scaling == 'covariance':
        return (0.0, 0.10)
    return (0.0, 0.8 / max(n_units, 1))


def plot_session_kinematics_neural_and_pca(t0_events, behavior, fr_z, s_id, which_intervals,
                                            unit_cols=None, smooth=3, pca_smooth=3, pca_n_pcs=8,
                                            pca_scaling='correlation', pca_null_reps=0,
                                            unit_rate_hz=None, min_unit_rate_hz=None,
                                            draw_shuffle=False, shuffle_reps=200, shuffle_seed=0,
                                            angle_range=(0, 90), metric_range=None, mean_z_range=None,
                                            draw_colorbar=True, var_explained_display='scatter', var_explained_center=0.4,
                                            var_explained_range=(0.1, 0.7), show_neuron_contrib=False,
                                            neuron_contrib_select='per_pc', neuron_contrib_top_per_pc=3,
                                            neuron_contrib_top_n=12, neuron_contrib_max_rows=25,
                                            neuron_contrib_fill='dense', neuron_contrib_range='default',
                                            show_neuron_contrib_summary=True,
                                            neuron_contrib_summary_line='drawn', groupby=None,
                                            highlight='all', groupby_n_bins=3,
                                            groupby_match_n=True):
    """Stacked panels sharing one x-axis: rows 0-1 as in plot_session_kinematics_and_neural
    (position kinematics, population-RMS mean+band, optionally overlaid with a second,
    dotted population-mean line on its own twin y-axis - see `mean_z_range` below), the
    fixed-`pca_n_pcs`-component PCA basis's own variance explained at each timebin (see
    `var_explained_display` below), optionally which individual neurons carry that basis
    (see `show_neuron_contrib` below), then the per-timebin-PCA-subspace canonical-angle
    heatmap (see plot_subspace_angle_heatmap) - all on the same t0/bin x-axis, tick labels
    only on the bottom-most (heatmap) row.

    `smooth` (see _smooth, off by default) applies to rows 0-1's display only.
    `pca_smooth` (see _subspace_block_with_separators) is the separate, on-by-default
    smoothing applied to each trial's trace before PCA, feeding both the heatmap and the
    variance-explained display - kept independent of `smooth` since that PCA input
    genuinely needs smoothing to not be dominated by single-bin spike noise, unlike rows
    0-1's purely cosmetic smoothing. `pca_n_pcs` (default 8, see _pca_basis_for_n_pcs) is
    the fixed number of top components kept at every timebin. `angle_range` (default
    (0, 90), the true range) passes through to plot_subspace_angle_heatmap's own
    `angle_range` - narrow it (e.g. (20, 70)) to add color contrast within whatever
    sub-range the data actually varies in.

    THE THREE PARAMETERS THAT DECIDE WHETHER THIS FIGURE MEANS ANYTHING, all added after the
    per-timebin variance-explained curve turned out to be reporting mostly artifact (the
    measurements are in _pca_basis_for_n_pcs' `scaling` note and _var_explained_null's):

    - `pca_scaling` (default `'correlation'`) - what gets decomposed at each timebin. The
      default standardizes every unit first, so no single unit can raise variance-explained by
      being loud; `'covariance'` is the old, unconditional behavior, on which 8.6% of timebins
      carried a spurious +0.1-0.2 jump produced by one near-silent unit firing once in one
      trial. Read `_pca_basis_for_n_pcs` before switching back. This feeds the
      variance-explained display, the neuron rows, the summary row AND the angle heatmap - one
      choice for the whole figure, never a per-row one.
    - `pca_null_reps` (default 0 = off) - a chance band on the summary row's
      variance-explained ceiling: per timebin, that many draws with each unit's trials
      permuted independently, keeping every unit's own variance and destroying all
      cross-neuron correlation (~1.8s per figure at 25 reps; only computed when
      `show_neuron_contrib_summary` is on). **Off by default because of what it does and does
      not answer.** It is a chance level for that one curve's own value - "is this population
      lower-dimensional than these units' individual variances alone would make it look" - and
      the answer on this data is a dull, near-constant "yes, by ~2.5 percentage points". It
      says nothing about a difference between conditions. Turn it on to re-check that
      dimensionality claim on new data; for a figure about a trial split use `draw_shuffle`.

    `draw_shuffle` (default False) is the shuffle that IS about the trial split, and it lives
    on row 1 where the split is actually drawn. It replaces that row's sem band with the
    permutation band from _group_shuffle_band: `shuffle_reps` (default 200) random
    re-assignments of `groupby`'s labels across trials, and the 5th-95th percentile of the
    resulting group-mean traces. A colored group line outside the band is a split effect a
    random split of the same trials would rarely produce. `shuffle_seed` (default 0) keeps a
    re-render identical. Requires `groupby`, and `highlight='all'` (with one group's trials
    alone there is nothing to permute); both raise rather than degrade quietly. Two things to
    know: the band is ~1.6x wider than the sem band it replaces (5-95% of a null is +/-1.64
    sd, sem is +/-1), and it applies to the population-RMS band only - the `mean_z_range` twin
    line has no band to replace. It also makes row 1's own pooled black mean the mean of the
    *labelled* trials rather than of every trial, so that the band is centred on the line it
    surrounds (see the comment at the call site); no other row's trial selection changes.
    - `min_unit_rate_hz` (default None = keep every unit) with `unit_rate_hz` - a firing-rate
      floor on the units, applied once to the unit list every row of the figure uses (see
      _apply_unit_rate_floor). 0.1 Hz drops the units whose z-scores are single spikes divided
      by a near-zero std; 0.5 Hz is defensible for mPFC and drops ~1/3 of them. This module
      does no I/O, so the rates come from the caller -
      `plotlib.session_data.unit_mean_rates(session_names)`, whose DataFrame can be passed
      straight in. Independent of `pca_scaling`: the scaling stops such a unit dominating by
      scale, the floor removes it before it can act as a one-trial leverage point instead.

    `metric_range` (default None - autoscaled) fixes row 1's own y-axis limits (its
    population-RMS mean+band - see population_rms) as a (min, max) pair, e.g. so the
    scale is comparable across sessions rather than each one autoscaling to its own
    min/max.

    `mean_z_range` (default None - no line at all) optionally overlays row 1 with a
    second, dotted line: the plain per-bin population mean (see population_mean, can be
    negative, unlike RMS - reads as "is the population net-elevated or net-suppressed"
    rather than "how much did it move overall"), on its own twin y-axis (ax1.twinx()) so
    it doesn't have to share RMS's scale. Passing None skips computing/drawing this line
    entirely; passing a (min, max) pair draws it and fixes that twin axis's limits to it
    (always explicit, never autoscaled - a dotted secondary line reads ambiguously without
    a known scale to anchor it against).

    The fixed-`pca_n_pcs`-component basis's own cumulative fraction of each timebin's
    across-trial variance - directly explains otherwise puzzling structure in the heatmap
    below: a timebin whose basis explains much less variance than its neighbors is a
    noisier subspace to compare against, which can show up as a whole anomalous
    row/column there - is shown per `var_explained_display`. (A third option, `'row'`, used to
    draw it as its own dedicated line row; it was removed as redundant once
    `show_neuron_contrib_summary`'s own ceiling line started plotting the same curve, on the
    same axis as the quantities it bounds. Passing it now raises.)
    - `'scatter'` (default): row 1's population-RMS mean line (see
      _draw_mean_band) gets a scatter marker at every (x, mean) point, color-mapped by
      that timebin's variance explained and drawn *behind* the (thinner, on-top) mean
      line, so it reads as a colored halo running along the line. Colored via a fixed
      diverging colormap with a fixed (not data-autoscaled) color range
      `var_explained_range` (default (0.1, 0.7)) diverging from `var_explained_center`
      (default 0.4, i.e. 40%) - fixing the range/center (rather than autoscaling to each
      session's own min/max) keeps the color meaning ("well above/below 40% explained")
      comparable across sessions.
    - `'off'`: neither - the heatmap loses its otherwise-puzzling-structure explanation,
      but nothing else changes.

    `show_neuron_contrib` (default False) adds one more heatmap row, directly above the
    canonical-angle heatmap: rows are individual neuron IDs, columns the same timebins, and
    a cell shows how much of that timebin's across-trial variance the top-`pca_n_pcs`
    subspace captures *through that neuron* (each PC's squared loading weighted by its own
    eigenvalue - see _pca_basis_for_n_pcs' `neuron_contrib`). It answers "who is carrying this subspace,
    and does that cast change over time" - which the variance-explained display above it
    can't, being a single number per timebin. The two are consistent by construction: a
    timebin's *full* per-neuron split sums exactly to its variance-explained value, and this
    panel draws the row-capped largest terms of that same split, in the same units - so a
    column here is a partial decomposition of the value plotted above it, never a
    separately-normalized quantity.

    `neuron_contrib_fill` (default `'dense'`) decides what the rows show: `'dense'` fills every
    kept row at every timebin, using the selection only to pick which neurons get a row at all;
    `'selected'` draws only the per-timebin selected cells and leaves the rest blank, showing
    the turnover of who leads the subspace at the cost of hiding well-defined values (and of
    letting the row cap make a column look artificially pale). `neuron_contrib_select` (default
    `'per_pc'`) is that selection: `'per_pc'` takes the union of each eigenvector's
    `neuron_contrib_top_per_pc` (default 3) largest-|loading| neurons; `'top_contrib'` takes
    that timebin's `neuron_contrib_top_n` (default 12) largest-contribution neurons outright.
    Under `'dense'` fill this only reweights the row ranking (per_pc favours transient
    specialists, top_contrib favours steady carriers); under `'selected'` it also decides every
    drawn cell - see _neuron_contribution_matrix for the full tradeoff.
    `neuron_contrib_max_rows` (default 25)
    caps the total row count (kept by largest summed contribution, reported when it drops
    anything - raising it now costs figure height rather than the row labels, see below), and
    `neuron_contrib_range` (default `'default'`, i.e. resolved from `pca_scaling` by
    _neuron_contrib_range_default: 0-10% of total variance per cell under `'covariance'`,
    0-(1/n_units) under `'correlation'`, where 1/n_units is the arithmetic per-cell ceiling)
    is the color scale, as a (min, max) pair of *fractions*
    of that timebin's total across-trial variance - the same quantity, and the same units, as
    the variance-explained curve, so a cell's color is directly readable against it. Fixed
    rather than autoscaled so the colors mean the same thing in every session (the measured
    98th percentile only moved between 7.9% and 11.0% across 6 sessions, so little contrast is
    lost); pass None to autoscale to this session's own distribution instead - see
    _neuron_contribution_matrix/_draw_neuron_contribution_heatmap. This
    row is independent of `var_explained_display`: any combination of the two works (they
    show the same underlying quantity at different granularity, not alternative
    presentations of one thing), and the per-timebin PCA feeding both is computed once. Its
    axes is deliberately *not* added to the returned tuple (still (ax0, ax1, ax2), so
    existing callers' unpacking keeps working) - reach it via `fig.axes` if it ever needs
    annotating.

    That row's height is not fixed: it grows with however many neuron rows survive
    `neuron_contrib_max_rows`, so every neuron keeps a readable physical slice and its ID label
    is always drawn (see _neuron_contrib_height_in). Only the figure gets taller.

    `show_neuron_contrib_summary` (default True) adds a thin line row underneath it that says,
    in % of total across-trial variance, how much of each timebin the heatmap column above it
    actually accounts for: the subspace's full variance explained as a dotted ceiling, and ONE
    mass curve under it, picked by `neuron_contrib_summary_line` - `'drawn'` (default, dark red:
    what the neurons you can see sum to), `'selected'` (orange: what the selection rule picked,
    before the row cap) or `'both'`. One curve by default because the two answer nearly the same
    question at this row's height, and drawing both obscured whichever was being read. A
    cell-count line appears on a twin axis only when it carries information - not when it is
    constant by construction, and not when its whole range is a small fraction of its own
    structural ceiling (`n_pcs * neuron_contrib_top_per_pc` under `'per_pc'`, where it measured
    14-18 of 18 on a real session) - so that axis is usually absent entirely now. See _draw_neuron_contribution_summary for how
    to read the row. It's an independent switch (it can be shown with `show_neuron_contrib=False`,
    as a compact stand-in for the whole heatmap, or suppressed while keeping the heatmap), and
    it reuses the same computed matrix, so it costs nothing extra.

    `groupby` (default None - one undivided figure, as before) splits this session's trials
    into groups and draws each group's own row-1 lines over the pooled ones: it takes either a
    per-trial column of `behavior` ('cue', 'choice_R1', 'choice_R2', 'trial_outcome', ...) or
    the special 'session_time' (each session's own trials in order, cut into `groupby_n_bins`
    (default 3) equal-count groups: early/mid/late), and adds one colored population-RMS line
    per group value - plus, when `mean_z_range` draws it, that group's dotted population-mean
    line on the twin axis - each computed from that group's trials alone. Row 1 carries the
    comparison because it is the tallest, least crowded row and its two lines are the ones
    worth comparing between groups; everything is one figure rather than one figure per group
    precisely because two separately-rendered figures are what makes such a comparison hard to
    read. Grouping is resolved by `trial_group_labels`; the colors come from
    plotlib.colors.group_colors, i.e. the same per-label colors the SVM figures use (cue 1
    orange / cue 2 purple, ...), with an ordered grouping like 'session_time' sampled
    green->yellow off viridis instead.

    `highlight` (default 'all') is the separate decision of what every row *including* row 1's
    own pooled black mean+band uses - 'all' means all of this session's trials (so the group
    lines are read against a pooled reference), while a group value (e.g. 1, or 'early')
    restricts the position/RMS/neuron-contribution/angle rows to that group, making the whole
    figure that group's - with the other groups' row-1 lines still drawn for reference. It is
    independent of `groupby`'s own value list beyond having to be one of them, so the same
    split can be rendered once pooled and once per group without changing anything else.

    `groupby_match_n` (default True) equalizes the groups' trial counts - a fixed-seed random
    subsample of each group down to the smallest one, reported when it drops anything. It
    matters most for the *band-free* comparison this row draws: a group with half the trials
    has a visibly noisier mean line at the same nominal smoothing, which reads as a difference
    between groups rather than between trial counts (see _match_group_trial_counts). It
    applies to every per-group subset, including a highlighted one, but never to
    `highlight='all'`.

    `draw_colorbar` (default True) controls every colorbar this figure can have, each in
    its own thin row *below* everything else (rather than stealing space from its own
    axes, which would break the square-heatmap sizing below, or placing it to the side,
    which would need its own column, shifting the other rows out of alignment) - these
    dedicated rows aren't part of the sharex group, since a colorbar's own x-axis is a
    color scale, not time. Always includes the heatmap's canonical-angle colorbar when
    there's a valid heatmap; additionally includes one for the
    `var_explained_display='scatter'` marker color and one for the `show_neuron_contrib`
    row, stacked below the first one, whenever those are active.

    The canonical-angle heatmap row is always sized square: since
    plot_subspace_angle_heatmap draws with aspect='equal', its axes box needs a physical
    height equal to its physical width for the square cells to actually fill the row -
    enforced below via a post-tight_layout position correction, not exposed as a parameter
    since no caller has ever needed a non-square heatmap here. The `show_neuron_contrib`
    row deliberately does *not* get that treatment: its row count is data-driven (how many
    neurons ever led a PC) while its height is a fixed _NEURON_CONTRIB_ROW_HEIGHT_IN, so
    its cells are simply stretched/squashed vertically - square cells there would make
    every session's figure a different height, which the side-by-side canvas assembly
    (see build_canvas) reads poorly.
    """
    if unit_cols is None:
        unit_cols = [c for c in fr_z.columns if c.startswith('Unit')]
    # One unit list for the whole figure, before anything is computed from it.
    unit_cols = _apply_unit_rate_floor(unit_cols, s_id, unit_rate_hz, min_unit_rate_hz)
    if pca_scaling not in _PCA_SCALINGS:
        raise ValueError(f"pca_scaling must be one of {_PCA_SCALINGS}, got {pca_scaling!r}")
    if draw_shuffle:
        if groupby is None:
            raise ValueError("draw_shuffle=True needs a trial split to shuffle - pass groupby")
        if highlight not in ('all', None):
            raise ValueError(f"draw_shuffle=True needs highlight='all' (got {highlight!r}): the "
                             f"band is the null for splitting ALL of this session's trials, so "
                             f"with one group's trials alone there is nothing to permute")
    if neuron_contrib_range == 'default':
        neuron_contrib_range = _neuron_contrib_range_default(pca_scaling, len(unit_cols))

    # Trial grouping is entirely a question of which t0_events rows each block of the figure
    # sees (get_trial_metric_traces iterates them) - so it resolves to exactly two things: a
    # `t0_view` that every existing row below uses, and one filtered view per group for the
    # summary row's own curves. Nothing else in this function needs to know about it.
    group_labels, group_values, n_matched = None, [], None
    if groupby is not None:
        group_labels, group_values = trial_group_labels(t0_events, behavior, groupby,
                                                         n_bins=groupby_n_bins, s_id=s_id)
        if not group_values:
            raise ValueError(f"groupby={groupby!r} has no labelled trials in session {s_id}")
        if highlight not in ('all', None) and highlight not in group_values:
            raise ValueError(f"highlight={highlight!r} is not one of {groupby}'s values in "
                             f"session {s_id}: {group_values} (or 'all' for every trial)")
        if groupby_match_n:
            group_labels, n_matched = _match_group_trial_counts(
                t0_events, group_labels, group_values, s_id)

    def _t0_for(value):
        """t0_events restricted to `value`'s trials - the whole table for 'all'/None."""
        if group_labels is None or value in ('all', None):
            return t0_events
        return t0_events[(group_labels == value).to_numpy()]

    t0_view = _t0_for(highlight)

    pos_layout, total_bins = _layout_intervals(
        t0_view, behavior, s_id, which_intervals,
        value_fn=lambda dat: dat.frame_position.values, gap_bins=_GAP_BINS)
    neural_layout, _ = _layout_intervals(
        t0_view, fr_z, s_id, which_intervals,
        value_fn=lambda dat: population_rms(dat, unit_cols), gap_bins=_GAP_BINS)

    if var_explained_display not in ('scatter', 'off'):
        raise ValueError(f"var_explained_display must be 'scatter' or 'off', got "
                         f"{var_explained_display!r} ('row' was removed - the same curve is now "
                         f"the ceiling line of the show_neuron_contrib_summary row)")
    show_var_explained_scatter = var_explained_display == 'scatter'
    # One per-timebin PCA pass feeds the variance-explained display, the neuron-contribution
    # row and its summary row - each independently optional, so fetch if any wants it.
    need_neuron_contrib = show_neuron_contrib or show_neuron_contrib_summary
    # The null is only ever drawn by the summary row, so don't pay for it otherwise.
    null_reps = pca_null_reps if (show_neuron_contrib_summary and pca_null_reps) else 0
    if var_explained_display != 'off' or need_neuron_contrib:
        x_coords_pc, bases_pc, var_explained_pc, layout_pc, contribs_pc, null_pc = \
            _subspace_block_with_separators(
                t0_view, fr_z, s_id, which_intervals, unit_cols, gap_bins=_GAP_BINS,
                smooth=pca_smooth, n_pcs=pca_n_pcs, scaling=pca_scaling, null_reps=null_reps)
    else:
        x_coords_pc, bases_pc, var_explained_pc, layout_pc, contribs_pc = np.array([]), [], [], [], []
        null_pc = None

    nc_rows, nc_row_unit_idx, nc_info = np.zeros((0, 0)), np.array([], dtype=int), None
    if need_neuron_contrib and len(x_coords_pc):
        nc_rows, nc_row_unit_idx, nc_info = _neuron_contribution_matrix(
            bases_pc, contribs_pc, len(unit_cols), select=neuron_contrib_select,
            top_per_pc=neuron_contrib_top_per_pc, top_n=neuron_contrib_top_n,
            max_rows=neuron_contrib_max_rows, fill=neuron_contrib_fill)
        _report_neuron_contribution(nc_info, neuron_contrib_select, neuron_contrib_top_per_pc,
                                    neuron_contrib_top_n, neuron_contrib_max_rows,
                                    fill=neuron_contrib_fill)

    # One further pass of row 1's own layout per group - the same population-RMS (and, if
    # drawn, population-mean) traces, on that group's trials alone - collected as
    # (label, rms_layout, mean_layout, color) for row 1 to draw over its pooled line.
    group_lines = []
    if groupby is not None:
        sess_rows = np.asarray(t0_events.index.get_level_values('session_id')) == s_id
        all_trials = np.asarray(t0_events.index.get_level_values('trial_id'))
        colors = group_colors(groupby, group_values)
        for gi, g in enumerate(group_values):
            t0_g = _t0_for(g)
            rms_layout_g, _ = _layout_intervals(
                t0_g, fr_z, s_id, which_intervals,
                value_fn=lambda dat: population_rms(dat, unit_cols), gap_bins=_GAP_BINS)
            if not rms_layout_g:
                print(f"  group {groupby}={g}: no trial had data, no line drawn")
                continue
            _warn_if_layout_differs(neural_layout, rms_layout_g, f'group {groupby}={g}')
            mean_layout_g = None
            if mean_z_range is not None:
                mean_layout_g, _ = _layout_intervals(
                    t0_g, fr_z, s_id, which_intervals,
                    value_fn=lambda dat: population_mean(dat, unit_cols), gap_bins=_GAP_BINS)
            n_g = len(np.unique(all_trials[sess_rows & (group_labels == g).to_numpy()]))
            # `groupby` itself is the legend's title, so the entries only carry the value -
            # spelling it out per entry ran the 3-group legend off the right edge of the axes
            # (checked on real data).
            group_lines.append((f'{g} (n={n_g})', rms_layout_g, mean_layout_g, colors[gi]))

    # The permutation band for row 1 - needs the (possibly count-matched) labels above, and
    # neural_layout for which intervals row 1 actually drew.
    shuffle_bands, shuffle_label = None, None
    rms_layout = neural_layout
    if draw_shuffle and neural_layout:
        # Row 1's pooled mean is re-derived from the LABELLED trials only, and the band from
        # the same table. Otherwise the two disagree: the null's own centre is by construction
        # the mean of the trials that carry a group label (and survived groupby_match_n's
        # subsample), which on real data sat 0.010-0.012 RMS units off the all-trials mean the
        # black line was drawn from - a band visibly not centred on the line it surrounds.
        # Only row 1 changes; the PCA/heatmap rows keep using `t0_view`, so `draw_shuffle`
        # stays a row-1 display choice and does not quietly re-select trials for everything.
        labelled = pd.notna(group_labels).to_numpy()
        t0_labelled = t0_events[labelled]
        rms_layout, _ = _layout_intervals(
            t0_labelled, fr_z, s_id, which_intervals,
            value_fn=lambda dat: population_rms(dat, unit_cols), gap_bins=_GAP_BINS)
        _warn_if_layout_differs(neural_layout, rms_layout, 'shuffle-band trial subset')
        shuffle_bands = _group_shuffle_band(
            t0_labelled, fr_z, s_id, which_intervals,
            lambda dat: population_rms(dat, unit_cols), rms_layout,
            group_labels[labelled], group_values, reps=shuffle_reps, seed=shuffle_seed,
            smooth=smooth)
        shuffle_label = f'shuffled split (5-95%, {shuffle_reps}x)'

    width = max(_MIN_WIDTH_IN, total_bins * _WIDTH_PER_BIN_IN)
    row2_height = width  # square heatmap row - see the aspect-ratio correction below
    ratios = list(_ROW_HEIGHTS_IN)
    if show_neuron_contrib:
        ratios.append(_neuron_contrib_height_in(nc_rows.shape[0]))
    if show_neuron_contrib_summary:
        ratios.append(_NEURON_CONTRIB_SUMMARY_HEIGHT_IN)
    ratios.append(row2_height)
    if draw_colorbar:
        ratios.append(_COLORBAR_HEIGHT_IN)
        if show_var_explained_scatter:
            ratios.append(_COLORBAR_HEIGHT_IN)
        if show_neuron_contrib:
            ratios.append(_COLORBAR_HEIGHT_IN)
    fig = plt.figure(figsize=(width, sum(ratios)))
    gs = GridSpec(nrows=len(ratios), ncols=1, height_ratios=ratios, figure=fig)
    row_idx = 0
    ax0 = fig.add_subplot(gs[row_idx]); row_idx += 1
    ax1 = fig.add_subplot(gs[row_idx], sharex=ax0); row_idx += 1
    ax_nc = fig.add_subplot(gs[row_idx], sharex=ax0) if show_neuron_contrib else None
    if show_neuron_contrib:
        row_idx += 1
    ax_ncs = fig.add_subplot(gs[row_idx], sharex=ax0) if show_neuron_contrib_summary else None
    if show_neuron_contrib_summary:
        row_idx += 1
    ax2 = fig.add_subplot(gs[row_idx], sharex=ax0); row_idx += 1
    cax = fig.add_subplot(gs[row_idx]) if draw_colorbar else None
    if draw_colorbar:
        row_idx += 1
    cax_ve = fig.add_subplot(gs[row_idx]) if show_var_explained_scatter and draw_colorbar else None
    if cax_ve is not None:
        row_idx += 1
    cax_nc = fig.add_subplot(gs[row_idx]) if show_neuron_contrib and draw_colorbar else None

    # sharex axes default to sharing the same tick locator/formatter (not just the view
    # range) - so _apply_xticks' set_xticklabels on ax2 below would otherwise also
    # propagate those same labels onto every row above it. Explicitly hide labelbottom on
    # everything but the last (heatmap) row, which is the one that actually calls
    # _apply_xticks, to get the intended "labels only on the bottom row" (confirmed this
    # was NOT happening by default - every row was showing them - before adding this).
    ax0.tick_params(axis='x', labelbottom=False)
    ax1.tick_params(axis='x', labelbottom=False)

    _draw_scaffolding(ax0, pos_layout)
    _draw_trial_lines(ax0, pos_layout, smooth=smooth)
    _annotate_trial_counts(ax0, pos_layout)
    _finalize_axes(ax0, 'Track position (cm)')

    _draw_scaffolding(ax1, neural_layout)
    scatter_mappable = None
    if show_var_explained_scatter:
        scatter_by_x = dict(zip(x_coords_pc, var_explained_pc))
        scatter_norm = TwoSlopeNorm(vmin=var_explained_range[0], vcenter=var_explained_center,
                                     vmax=var_explained_range[1])
        scatter_mappable = _draw_mean_band(
            ax1, rms_layout, smooth=smooth, scatter_by_x=scatter_by_x,
            scatter_cmap=_VAR_EXPLAINED_CMAP, scatter_norm=scatter_norm,
            band_by_interval=shuffle_bands, band_label=shuffle_label)
    else:
        _draw_mean_band(ax1, rms_layout, smooth=smooth,
                        band_by_interval=shuffle_bands, band_label=shuffle_label)
    # Group lines go *over* the pooled mean+band, in the same units on the same axis: solid
    # for RMS (left axis), and dotted for the population mean on the twin below - i.e. the
    # same solid/dotted convention the pooled black lines already use, with color carrying
    # the group. No per-group band: three sem bands on top of each other read as mud, and
    # the pooled band is still there for the scale of "how precisely is a mean known here".
    for label, rms_layout_g, _, color in group_lines:
        _draw_mean_line(ax1, rms_layout_g, color=color, linestyle='-', linewidth=1.3,
                        smooth=smooth, label=label)
    _finalize_axes(ax1, 'Population RMS\n(z-scored Hz)')
    if metric_range is not None:
        ax1.set_ylim(*metric_range)
    if group_lines:
        # Above the axes rather than inside it: `metric_range` is typically set tight around
        # the traces, so an in-axes legend sits on top of the very lines it labels.
        legend = ax1.legend(title=groupby, fontsize=6,
                            ncol=len(group_lines) + (1 if shuffle_label else 0), frameon=False,
                            loc='lower left', bbox_to_anchor=(0, 1.0), handlelength=1.6,
                            columnspacing=1.2, borderpad=0.1)
        legend.get_title().set_fontsize(6)

    if mean_z_range is not None:
        mean_z_layout, _ = _layout_intervals(
            t0_view, fr_z, s_id, which_intervals,
            value_fn=lambda dat: population_mean(dat, unit_cols), gap_bins=_GAP_BINS)
        ax1_twin = ax1.twinx()
        # twinx() shares ax1's x-axis but defaults to showing its own x tick labels too -
        # would duplicate row 1's (already-hidden, see ax1.tick_params above) bottom
        # labels right on top of it.
        ax1_twin.tick_params(axis='x', labelbottom=False, bottom=False)
        _draw_mean_line(ax1_twin, mean_z_layout, smooth=smooth)
        for _, _, mean_layout_g, color in group_lines:
            if mean_layout_g:
                _draw_mean_line(ax1_twin, mean_layout_g, color=color, smooth=smooth)
        ax1_twin.set_ylim(*mean_z_range)
        ax1_twin.set_ylabel('Population mean\n(z-scored Hz)')
        ax1_twin.spines[['top']].set_visible(False)

    nc_mesh = None
    if show_neuron_contrib:
        ax_nc.tick_params(axis='x', labelbottom=False)
        nc_mesh = _draw_neuron_contribution_heatmap(
            ax_nc, x_coords_pc, nc_rows, nc_row_unit_idx, unit_cols, layout_pc,
            value_range=neuron_contrib_range)
        ax_nc.spines[['top', 'right']].set_visible(False)

    if show_neuron_contrib_summary:
        ax_ncs.tick_params(axis='x', labelbottom=False)
        ax_ncs.spines[['top']].set_visible(False)
        if nc_info is not None:
            _draw_neuron_contribution_summary(
                ax_ncs, x_coords_pc, nc_info, var_explained_pc, layout_pc,
                neuron_contrib_select, neuron_contrib_top_per_pc, neuron_contrib_top_n,
                pca_n_pcs, fill=neuron_contrib_fill, null=null_pc,
                mass_line=neuron_contrib_summary_line)

    _, mesh = plot_subspace_angle_heatmap(ax2, t0_view, fr_z, s_id, which_intervals, unit_cols=unit_cols,
                                           gap_bins=_GAP_BINS, smooth=pca_smooth, n_pcs=pca_n_pcs,
                                           summary=_ANGLE_SUMMARY, angle_range=angle_range,
                                           scaling=pca_scaling,
                                           # the identical block, already computed above
                                           block=((x_coords_pc, bases_pc, layout_pc)
                                                  if len(x_coords_pc) else None))
    _apply_xticks(ax2, neural_layout)
    ax2.spines[['top', 'right']].set_visible(False)

    if draw_colorbar and mesh is not None:
        fig.colorbar(mesh, cax=cax, orientation='horizontal', label='Canonical angle (deg)')
    if draw_colorbar and scatter_mappable is not None:
        fig.colorbar(scatter_mappable, cax=cax_ve, orientation='horizontal',
                     label=f'Variance explained (top {pca_n_pcs} PCs, {pca_scaling})')
    if draw_colorbar and nc_mesh is not None:
        fig.colorbar(nc_mesh, cax=cax_nc, orientation='horizontal',
                     label=f'Neuron variance share (top {pca_n_pcs} PCs, {pca_scaling}, %)')

    fig.tight_layout()
    # tight_layout gives ax2 a different left/right margin than ax0/ax1 - aspect='equal'
    # (see plot_subspace_angle_heatmap) makes it reserve extra inset padding to keep cells
    # square, which tight_layout's per-row margin computation doesn't account for. All
    # three report the same sharex xlim, but that different margin means the same x data
    # value lands at a different pixel column in ax2 than in ax0/ax1 - confirmed on real
    # data (ax0/ax1 x0=0.0719/x1=0.9792 vs ax2 x0=0.0915/x1=0.9596).
    #
    # Force ax2 to ax0's exact horizontal extent, and re-derive ax2's height from ax0's
    # *actual* rendered width (in inches) rather than trusting the row2_height guess above
    # (that guess is nominal-figure-width-based, computed before tight_layout trims
    # ax0/ax1's own margins for their tick labels - close, but not exactly what ax0's
    # plot area comes out to physically, which is what actually needs matching for square
    # cells). ax2's top edge (adjacent to ax1) is kept fixed; the colorbar is shifted by
    # however much ax2's bottom edge moved, so their own gap stays what tight_layout chose.
    fig_w, fig_h = fig.get_size_inches()
    pos0 = ax0.get_position()
    pos2 = ax2.get_position()
    side_frac_w = pos0.width
    side_frac_h = (side_frac_w * fig_w) / fig_h
    top2 = pos2.y0 + pos2.height
    new_y0_2 = top2 - side_frac_h
    ax2.set_position([pos0.x0, new_y0_2, side_frac_w, side_frac_h])
    # Every colorbar row rides along below ax2's new bottom edge, each keeping whatever gap
    # tight_layout already put between it and the row above it - so the whole stack (angle,
    # var-explained scatter, neuron contribution: however many are present) shifts as one
    # block, with its internal spacing untouched.
    bottom_axes = [a for a in (cax, cax_ve, cax_nc) if a is not None]
    prev_orig_y0, prev_new_y0 = pos2.y0, new_y0_2
    for c in bottom_axes:
        pos_c = c.get_position()
        gap = prev_orig_y0 - (pos_c.y0 + pos_c.height)
        new_c_y0 = prev_new_y0 - gap - pos_c.height
        c.set_position([pos0.x0, new_c_y0, side_frac_w, pos_c.height])
        prev_orig_y0, prev_new_y0 = pos_c.y0, new_c_y0

    # The square-heatmap correction above keeps ax2's *top* fixed and derives its bottom
    # from the height a true square actually needs - which can come out *taller* than
    # tight_layout originally gave ax2 (confirmed on real multi-interval session data: a
    # wide, many-interval x-axis needs a taller square heatmap than tight_layout guessed).
    # Since the colorbar rows are the only thing below ax2, they absorb that entire deficit
    # and can get pushed to a negative y0 - off the bottom of the canvas, rendering invisibly
    # rather than erroring. Rather than let that happen silently, grow the whole figure at
    # the bottom and shift every axes up by the same number of inches, preserving each
    # axes' own absolute (inch) position/size - this only ever grows the canvas, never
    # touches anything if there was already enough room (the common case).
    #
    # What actually needs to stay on-canvas is each colorbar's full *rendered* extent -
    # its tick labels and axis label render below its own get_position() box, not inside
    # it, so checking the box's own y0 alone isn't enough (confirmed on real data: a
    # box with a small positive y0 still had its label text clipped off the bottom, since
    # that text needs another ~0.3-0.5in below the box that a box-only check never sees).
    # get_tightbbox needs an actual draw first so text extents are known; it returns
    # device (pixel) coordinates relative to the whole figure, so dividing by fig.dpi
    # converts directly to inches from the canvas's own bottom edge (y=0).
    if bottom_axes:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        buffer_in = 0.05
        min_bbox_y0_in = min(a.get_tightbbox(renderer).y0 / fig.dpi for a in bottom_axes)
        if min_bbox_y0_in < buffer_in:
            deficit_in = buffer_in - min_bbox_y0_in
            new_fig_h = fig_h + deficit_in
            for a in fig.axes:
                p = a.get_position()
                y0_in = p.y0 * fig_h + deficit_in
                height_in = p.height * fig_h
                a.set_position([p.x0, y0_in / new_fig_h, p.width, height_in / new_fig_h])
            fig.set_size_inches(fig_w, new_fig_h)
    return fig, (ax0, ax1, ax2)



# ---------------------------------------------------------------------------------
# Levels, not dynamics: the same population metric as row 1 of
# plot_session_kinematics_neural_and_pca, one session per row, x squeezed to a strip.
# ---------------------------------------------------------------------------------

# Which per-bin population metric a cross-session figure plots - the same two quantities
# row 1 of plot_session_kinematics_neural_and_pca draws (solid RMS / dotted population
# mean), here selectable instead of one being the row and the other its twin-axis overlay.
_SESSION_METRICS = {
    'rms': (population_rms, 'Population RMS\n(z-scored Hz)'),
    'mean_z': (population_mean, 'Population mean\n(z-scored Hz)'),
}
_POINT_LEVELS = ('bin', 'trial', 'trial_bin')


def _points_for_stack(stack, bins, point_level):
    """(x, y) scatter points for one group's (n_trials, n_bins) `stack`, per `point_level` -
    see plot_metric_across_sessions for what each level means. x always spans `bins`, so
    every level fills the same strip and the levels are interchangeable in one figure."""
    if point_level == 'bin':
        return bins.astype(float), np.nanmean(stack, axis=0)
    if point_level == 'trial_bin':
        return np.tile(bins.astype(float), stack.shape[0]), stack.ravel()
    # 'trial': one point per trial (its own mean over the interval's bins), spread evenly
    # across the strip *in trial order* - x carries no time-within-interval meaning here,
    # unlike the other two levels, but the ordering makes a within-session drift visible.
    per_trial = np.nanmean(stack, axis=1)
    x = (np.linspace(bins.min(), bins.max(), len(per_trial)) if len(per_trial) > 1
         else np.array([np.mean([bins.min(), bins.max()])]))
    return x, per_trial


def plot_metric_across_sessions(t0_events, behavior, fr_z, session_ids, which_intervals,
                                 unit_cols=None, metric='rms', groupby='cue',
                                 groupby_n_bins=3, groupby_match_n=True,
                                 point_level='bin', smooth=1,
                                 unit_rate_hz=None, min_unit_rate_hz=None,
                                 metric_range=None, slot_width_in=0.22, slot_gap_in=0.04,
                                 interval_gap_in=0.18, row_height_in=0.6, label_width_in=0.75,
                                 point_size=4, point_alpha=0.35, mean_linewidth=2.5,
                                 show_pooled=False, session_labels=None, title=True,
                                 show_t0=True, xtick_rotation=45, fontsize=6):
    """One session per row, one narrow slot per (interval, trial group): where does this
    population metric sit, and does the difference between groups hold across sessions?

    Box-plot logic with a scatter instead of a box. Every group of every interval gets its
    own `slot_width_in`-wide column - the groups sit *beside* each other, never overlaid -
    and inside a column the interval's timebins are squeezed onto that width, so the time
    course survives only as a faint left-to-right smear, not as something to read. What
    should jump out of the figure is the thick short horizontal line: that group's mean over
    the whole interval, one per column. Columns are grouped into one block per entry of
    `which_intervals` (in the given order, labelled on the bottom row's x axis), and every
    row shares one y-axis (`metric_range`, else autoscaled over all rows - never per row).

    This is the level comparison plot_session_kinematics_neural_and_pca cannot make: that
    figure is one session, spending its whole width on one interval's time course; this one
    drops the time course to fit every interval, both groups and every session into one
    picture.

    `metric` picks the quantity (see _SESSION_METRICS): `'rms'` (default, population_rms -
    "how much did the population move, in either direction") or `'mean_z'` (population_mean -
    "is it net-elevated or net-suppressed", can be negative, and gets a faint zero line since
    that is the reference its sign is read against).

    `point_level` is what one dot is:
    - `'bin'` (default): that timebin's mean across the group's trials - i.e. the points row 1
      of the timeline figure draws its line through, scattered into the column instead.
    - `'trial'`: one dot per trial, its own mean over the interval, spread across the column in
      trial order. The across-trial spread the group mean is actually estimated from, which
      `'bin'` hides (a per-bin mean over 50 trials is far tighter than the trials themselves).
    - `'trial_bin'`: every (trial, bin) value - the full cloud, the only level whose drawn
      spread is the raw data's.
    The mean line is always the mean of that column's own drawn dots, so it means the same
    thing at every level (they differ by <1e-12 on complete data; with NaNs, by how the
    missing bins are weighted).

    `groupby` (default `'cue'`) is resolved exactly as in
    plot_session_kinematics_neural_and_pca - a per-trial `behavior` column, or `'session_time'`
    (each session's trials in order, cut into `groupby_n_bins` equal-count groups) - but *once
    over all sessions* rather than per session, so a group keeps the same slot, the same
    color (plotlib.colors.group_colors: cue 1 orange / cue 2 purple, as in the SVM figures)
    and the same legend entry in every row. `groupby=None` draws a single black column per
    interval. `groupby_match_n` (default True) equalizes the groups' trial counts per session
    (see _match_group_trial_counts): nothing here shows a confidence interval, so a group with
    half the trials would read as a level difference rather than as a noisier estimate.

    A group with no trials in an interval keeps its (empty) slot rather than letting the
    others slide over - the x layout is fixed by `which_intervals` x `groupby`, never by what
    happened to be present, so the same column means the same thing in every row.

    `show_pooled` (default False) adds an all-trials mean as a dashed black line across each
    interval's whole block, for a reference the group lines can be read against. `smooth`
    (default 1 = off, unlike the timeline figure's display smoothing) smooths each trial's own
    trace before anything is averaged - leave it off for `'bin'`/`'trial_bin'`, where it makes
    neighbouring dots correlated and the cloud tighter than the data is. `show_t0` (default
    True) draws each column's own t0 as a hairline, the one x position inside a column that
    still means something; it is deliberately faint, since the whole point is that x is no
    longer a readable axis.

    Geometry is in inches and additive, and the x axis is *in those same inches* (xlim is the
    panel's own width), so `slot_width_in` (default 0.22, one group of one interval),
    `slot_gap_in` (0.04, between groups of the same interval) and `interval_gap_in` (0.18,
    between intervals) place the columns exactly - no figsize guessing. `row_height_in`
    (default 0.6) is one session row and `label_width_in` (0.75) the left margin the row
    labels and y ticks live in, beside the panel rather than eating into it. `fontsize`
    (default 6, the legend one point below it) scales every text in the figure *and* the
    margins reserved for it, so a bigger canvas doesn't come out with the same hairline
    labels - it is a size in points, not a factor. Anything that still renders past an edge
    (long labels, a many-group legend, rotated interval labels) grows the canvas instead of
    being clipped.

    `session_labels` optionally maps a session id to its row label (e.g.
    `{s_id: f'{i:02d} {s_id}'}`); missing ids fall back to the id itself. A session with no
    trials in any interval gets no row rather than an empty one. `title` (default True) puts
    the metric and point level on the suptitle; pass False to set your own.
    `xtick_rotation` (default 45) rotates the interval labels - a block is under half an inch
    wide, so horizontal labels only fit for short interval names.

    Returns `(fig, axes)`, one axes per drawn row, top to bottom, and prints one table per
    interval of the group means per session (with their difference for a two-group split) -
    the numbers behind the figure, which columns this narrow are not meant to be read off.
    """
    if metric not in _SESSION_METRICS:
        raise ValueError(f"metric must be one of {tuple(_SESSION_METRICS)}, got {metric!r}")
    if point_level not in _POINT_LEVELS:
        raise ValueError(f"point_level must be one of {_POINT_LEVELS}, got {point_level!r}")
    if isinstance(which_intervals, str):
        raise ValueError(f"which_intervals must be a list of interval names, got the single "
                         f"string {which_intervals!r} - pass [{which_intervals!r}]")
    which_intervals = list(which_intervals)
    metric_fn, ylabel = _SESSION_METRICS[metric]
    if unit_cols is None:
        unit_cols = [c for c in fr_z.columns if c.startswith('Unit')]

    # Grouping resolved once over every session, so group->slot, group->color and the legend
    # are the same in every row (a per-session call returns only that session's own present
    # values, which would shift both the slots and the fallback colors of a session missing
    # a group).
    if groupby is None:
        group_values, colors, group_labels = ['all'], ['k'], None
    else:
        group_labels, group_values = trial_group_labels(t0_events, behavior, groupby,
                                                         n_bins=groupby_n_bins)
        if not group_values:
            raise ValueError(f"groupby={groupby!r} has no labelled trials in any session")
        colors = group_colors(groupby, group_values)

    # x is in inches, so the column layout is exact and fixed by (interval, group) alone -
    # computed before any data is touched, which is what keeps a column meaning the same
    # thing in every row even when a session is missing that group entirely.
    n_groups = len(group_values)
    block_in = n_groups * slot_width_in + (n_groups - 1) * slot_gap_in
    block_x0 = [ii * (block_in + interval_gap_in) for ii in range(len(which_intervals))]
    panel_width_in = (len(which_intervals) * block_in
                       + max(len(which_intervals) - 1, 0) * interval_gap_in)
    slot_inset_in = 0.08 * slot_width_in   # keeps dots off their own column's edges

    def _slot_x(x_bins, bins, ii, gi):
        """Bin positions -> x (inches) inside interval `ii`'s group-`gi` column."""
        x0 = block_x0[ii] + gi * (slot_width_in + slot_gap_in)
        lo, hi = float(bins.min()), float(bins.max())
        frac = 0.5 * np.ones_like(np.asarray(x_bins, dtype=float)) if hi == lo else \
            (np.asarray(x_bins, dtype=float) - lo) / (hi - lo)
        return x0 + slot_inset_in + frac * (slot_width_in - 2 * slot_inset_in)

    # Everything computed before the figure exists: the row count is data-driven (a session
    # with nothing in any interval contributes none), so it isn't known until this is done.
    rows, records = [], []
    for s_id in dict.fromkeys(session_ids):   # one row per session, first-seen order
        if s_id not in t0_events.index.unique('session_id') or \
                s_id not in fr_z.index.unique('session_id'):
            print(f"Session {s_id}: not in t0_events/fr_z, skipping")
            continue
        unit_cols_s = _apply_unit_rate_floor(unit_cols, s_id, unit_rate_hz, min_unit_rate_hz)
        value_fn = lambda dat: metric_fn(dat, unit_cols_s)  # noqa: B023 - used in this iteration
        labels_s = group_labels
        if group_labels is not None and groupby_match_n:
            labels_s, _ = _match_group_trial_counts(t0_events, group_labels, group_values, s_id)

        columns, pooled = [], []
        for ii, which_interval in enumerate(which_intervals):
            for gi, g in enumerate(group_values):
                t0_g = t0_events if labels_s is None else t0_events[(labels_s == g).to_numpy()]
                traces, bins = get_trial_metric_traces(t0_g, fr_z, s_id, which_interval, value_fn)
                if bins is None:
                    print(f"  session {s_id}, {which_interval}, {groupby}={g}: no trial had "
                          f"data, column left empty")
                    continue
                stack = np.asarray([_smooth(vals, smooth) for vals in traces], dtype=float)
                x_bins, y = _points_for_stack(stack, bins, point_level)
                mean = np.nanmean(y) if np.any(np.isfinite(y)) else np.nan
                t0_x = (_slot_x([0], bins, ii, gi)[0]
                        if show_t0 and bins.min() <= 0 <= bins.max() else None)
                # The legend entry is the bare group value: trial counts differ per
                # interval (get_trial_metric_traces' modal-shape drops) and per session, so a
                # 'cue 2 (n=47)' entry would either be wrong for most columns or, worse, get
                # a second legend row per distinct n - it does belong in the printed table
                # below instead.
                columns.append((ii, gi, _slot_x(x_bins, bins, ii, gi), y, mean, t0_x,
                                colors[gi], f'{g}'))
                records.append({'session_id': s_id, 'interval': which_interval, 'group': g,
                                'n_trials': len(traces), 'mean': mean})
            if show_pooled:
                traces, bins = get_trial_metric_traces(t0_events, fr_z, s_id, which_interval,
                                                        value_fn)
                if bins is not None:
                    stack = np.asarray([_smooth(vals, smooth) for vals in traces], dtype=float)
                    pooled.append((ii, np.nanmean(_points_for_stack(stack, bins, point_level)[1])))
        if not columns:
            print(f"Session {s_id}: no interval/group had data, no row drawn")
            continue
        rows.append((s_id, (session_labels or {}).get(s_id, s_id), columns, pooled))

    if not rows:
        raise ValueError(f"no session had trials in any of {which_intervals} - nothing to draw")

    # Explicit inch margins, not tight_layout: the row labels (a 16-char session id at
    # fontsize 6 is ~0.6in) sit next to a panel that is deliberately only ~2in wide, which
    # tight_layout cannot fit at all ("left and right margins cannot be made large enough",
    # confirmed) - it needs the *figure* to be wider than the panel, which is what this
    # reserves. The legend is one row per group (a horizontal one would run off a panel this
    # narrow), so its height is group-count-driven; anything still overflowing is absorbed by
    # the grow pass below, so these are starting points rather than limits.
    text_scale = fontsize / 6.0
    legend_in = (0.12 + 0.10 * n_groups + (0.10 if show_pooled else 0)) * text_scale
    top_in = 0.10 + legend_in + (0.34 * text_scale if title else 0)  # legend, 2-line suptitle
    bottom_in = (0.30 + (0.25 if xtick_rotation else 0)) * text_scale  # interval labels
    fig_w = label_width_in + panel_width_in + 0.10
    fig_h = bottom_in + row_height_in * len(rows) + top_in
    fig, axes = plt.subplots(nrows=len(rows), ncols=1, sharex=True, sharey=True,
                             figsize=(fig_w, fig_h), squeeze=False)
    axes = list(axes[:, 0])
    fig.subplots_adjust(left=label_width_in / fig_w, right=1 - 0.10 / fig_w,
                        bottom=bottom_in / fig_h, top=1 - top_in / fig_h, hspace=0.15)
    seen_labels = set()
    for ax, (s_id, label, columns, pooled) in zip(axes, rows):
        if metric == 'mean_z':
            ax.axhline(0, color='k', linewidth=0.6, alpha=0.25, zorder=1)
        for ii, gi, x, y, mean, t0_x, color, glabel in columns:
            if t0_x is not None:
                ax.axvline(t0_x, color='k', linewidth=0.5, alpha=0.12, zorder=1.5)
            # The dots are the faint indicator, the mean line is the message: a thick,
            # column-wide horizontal bar drawn on top of them.
            ax.scatter(x, y, s=point_size, color=color, alpha=point_alpha, edgecolor='none',
                       zorder=2, label=glabel if glabel not in seen_labels else None)
            seen_labels.add(glabel)
            x0 = block_x0[ii] + gi * (slot_width_in + slot_gap_in)
            ax.hlines(mean, x0, x0 + slot_width_in, color=color, linewidth=mean_linewidth,
                      zorder=4, capstyle='butt')
        for ii, pooled_mean in pooled:
            if np.isfinite(pooled_mean):
                ax.hlines(pooled_mean, block_x0[ii], block_x0[ii] + block_in, color='k',
                          linestyle='--', linewidth=1.0, zorder=3.5,
                          label='all trials' if 'all trials' not in seen_labels else None)
                seen_labels.add('all trials')
        ax.set_xlim(-interval_gap_in / 2, panel_width_in + interval_gap_in / 2)
        ax.set_ylabel(label, fontsize=fontsize, rotation=0, ha='right', va='center')
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.tick_params(axis='x', length=2)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.spines[['top', 'right']].set_visible(False)
    if metric_range is not None:
        axes[0].set_ylim(*metric_range)   # sharey - one row's limits are every row's
    # One labelled tick per interval block (not per column): the columns within a block are
    # told apart by color, via the legend, not by a tick of their own.
    axes[-1].set_xticks([x0 + block_in / 2 for x0 in block_x0])
    axes[-1].set_xticklabels([_interval_short_label(iv) for iv in which_intervals],
                              fontsize=fontsize, rotation=xtick_rotation,
                              ha='right' if xtick_rotation else 'center')

    legend = axes[0].legend(title=groupby, fontsize=fontsize - 1, frameon=False,
                            loc='lower left', bbox_to_anchor=(0, 1.0), handlelength=1.0,
                            borderpad=0.1, labelspacing=0.2, handletextpad=0.4)
    legend.get_title().set_fontsize(fontsize - 1)
    if title:
        fig.suptitle(f'{ylabel.replace(chr(10), " ")}\nmean per {point_level}',
                     fontsize=fontsize)

    # Grow the canvas at whichever edge something actually renders past, keeping every axes'
    # own inch position/size - same idea as the colorbar rescue in
    # plot_session_kinematics_neural_and_pca, here for the row labels (left), the rotated
    # interval labels (bottom) and the legend (top, since it is an artist of axes[0] and so
    # shows up in that axes' tight bbox). Needs a draw first: text extents are unknown until
    # then, and get_tightbbox returns device pixels, so /fig.dpi converts to inches from the
    # canvas' own edges. The suptitle is deliberately not measured - it is positioned in
    # *figure* fractions, so growing the canvas moves it along with the edge instead of
    # freeing room under it; growing for the legend pushes it further from the title.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    buffer_in = 0.03
    boxes = [a.get_tightbbox(renderer) for a in fig.axes]
    over_left = max(0.0, buffer_in - min(b.x0 for b in boxes) / fig.dpi)
    over_bottom = max(0.0, buffer_in - min(b.y0 for b in boxes) / fig.dpi)
    over_right = max(0.0, max(b.x1 for b in boxes) / fig.dpi - (fig_w - buffer_in))
    over_top = max(0.0, max(b.y1 for b in boxes) / fig.dpi - (fig_h - buffer_in))
    if over_left or over_bottom or over_right or over_top:
        new_w = fig_w + over_left + over_right
        new_h = fig_h + over_bottom + over_top
        for a in fig.axes:
            p = a.get_position()
            a.set_position([(p.x0 * fig_w + over_left) / new_w, (p.y0 * fig_h + over_bottom) / new_h,
                            p.width * fig_w / new_w, p.height * fig_h / new_h])
        fig.set_size_inches(new_w, new_h)

    summary = pd.DataFrame(records)
    if not summary.empty:
        for which_interval in which_intervals:
            part = summary[summary.interval == which_interval]
            if part.empty:
                continue
            wide = part.pivot(index='session_id', columns='group', values='mean')
            # pivot sorts its columns alphabetically - restore the grouping's own display
            # order, which is what makes an ordered one (session_time: early->late) readable.
            present = [g for g in group_values if g in wide.columns]
            wide = wide.reindex(columns=present)
            if wide.shape[1] == 2:
                wide['diff'] = wide.iloc[:, 1] - wide.iloc[:, 0]
            # n per group belongs here rather than in the legend, since it varies by interval
            # and session (see the columns.append comment above).
            n_wide = part.pivot(index='session_id', columns='group', values='n_trials')
            wide = wide.join(n_wide.reindex(columns=present).rename(
                columns=lambda g: f'n({g})'))
            print(f"\n{metric} means per session in {which_interval} "
                  f"(point_level={point_level}):")
            print(wide.round(4).to_string())
    return fig, axes


# Canvas assembly - ported verbatim from svm_plotting_lib.py/svm_analysis_2026.py (same
# helper there, kept manually in sync between the two - see that file's own docstring),
# since dim_analysis.py wants the same "one PNG/SVG per session -> one combined poster"
# workflow for its own per-session figures.


# ---------------------------------------------------------------------------------
# Whole-session PCA and how its subspace drifts across sessions.
#
# Moved here 2026-08-11 from svm_analysis_2026.py: this is population geometry, not
# anything to do with an SVM fit, so it belongs beside the per-timebin subspace work
# above rather than in the decoding libraries. Its shared pieces (the eigendecomposition,
# the canonical-angle comparison, the spectrum panel) went to plotlib/pca.py, which is
# what keeps this module from having to import the W library.
#
# KNOWN DEBT, as in w_plotting_lib: plot_session_pca/plot_pca_reference_drift still fetch
# FiringRate40msZ themselves instead of taking it as an argument.
# ---------------------------------------------------------------------------------

def plot_session_pca(s_id, session_names, x_modality, max_pcs=50, save_dir=None,
                      ith_to_dt_session=None):
    """Standalone baseline for the W-variance-explained analysis (part 1 - see
    plot_w_variance_explained for part 2/3, which compares a specific decoder direction
    against this same kind of spectrum but restricted to one interval/timebin-range and
    pooled across trials): PCA across session `s_id`'s *entire* continuous recording -
    every native 40ms bin of FiringRate40msZ, not just trial windows - for one
    X_modality, not tied to any specific decoder. Plots the eigenvalue spectrum (%
    of total variance - see _pca_eigen/plot_pca_eigenvalues) as a horizontal
    bar chart ordered by magnitude. Saves to save_dir/session_{s_id}_S{s_num}_pca.png if
    save_dir is given. Returns (save_path, eigenvalues_pct, eigenvectors) - eigenvectors
    (see _pca_eigen) feeds plot_session_pca_similarity's cross-session canonical-angle
    comparison, called once per session the same way this function is."""
    session_name = next(sn for sn in session_names if sn.startswith(s_id))
    fr_z_raw = analytics.get_analytics('FiringRate40msZ', session_names=[session_name]).xs(s_id, level='session_id')
    fr_x = session_fr_for_modality(fr_z_raw, x_modality)
    eigenvalues_pct, eigenvectors, _ = pca_eigen(fr_x.values)

    s_num = (ith_to_dt_session or {}).get(s_id, "Unknown")
    s_label = f'S{s_num:02d}' if isinstance(s_num, int) else f'S{s_num}'
    n_shown = min(max_pcs or len(eigenvalues_pct), len(eigenvalues_pct))
    fig, ax = plt.subplots(figsize=(4, max(3, 0.18 * n_shown)), dpi=150)
    plot_pca_eigenvalues(ax, eigenvalues_pct, max_pcs=max_pcs,
                          title=f'{s_label}: whole-session PCA ({x_modality})')
    plt.tight_layout()

    save_path = None
    if save_dir is not None:
        save_path = os.path.join(save_dir, f'session_{s_id}_S{s_num}_pca.png')
        fig.savefig(save_path, dpi=150)
    plt.show()
    return save_path, eigenvalues_pct, eigenvectors


def plot_session_pca_similarity(session_eigvecs, session_eigvals_pct, session_labels, var_thresh=0.8,
                                 agg='mean', vmax=90, save_path=None):
    """Cross-session similarity of the whole-session PCA subspaces from plot_session_pca
    (looped over sessions), via canonical/principal angles - see canonical_angles_deg
    for the method, already established elsewhere in this codebase
    (get_SessionPCs40msCAs/plot_EvolvingPCSubspace) for exactly this kind of
    cross-session PCA-subspace comparison, ported here as a static matplotlib heatmap
    for this notebook's own whole-session PCA. For each session, only the top-k
    eigenvectors needed to reach `var_thresh` cumulative variance (default 0.8,
    matching that same established convention) are used as its subspace - a
    low-variance session contributes fewer dimensions than a high-variance one, same as
    get_SessionPCs40msCAs (guarded to at least 1 component, in case a single PC alone
    already exceeds var_thresh). Each pair of sessions gets min(k_i, k_j) canonical
    angles; `agg` ('mean' by default, or 'median') collapses that set to one scalar per
    pair for the heatmap - same aggregation choice plot_EvolvingPCSubspace's
    render_plot offers. Diagonal entries are always 0 (a subspace compared to itself).
    Lower = more similar (aligned) subspaces. Saves to save_path if given. Returns the
    (n_sessions, n_sessions) similarity matrix."""
    n = len(session_labels)
    agg_fn = np.mean if agg == 'mean' else np.median
    thresh_pct = var_thresh * 100
    n_components = [max(1, int((eigvals_pct.cumsum() < thresh_pct).sum())) for eigvals_pct in session_eigvals_pct]

    sim = np.zeros((n, n))
    for i in range(n):
        subspace_i = session_eigvecs[i][:, :n_components[i]]
        for j in range(n):
            subspace_j = session_eigvecs[j][:, :n_components[j]]
            sim[i, j] = agg_fn(canonical_angles_deg(subspace_i, subspace_j))

    fig, ax = plt.subplots(figsize=(max(4, 0.45 * n), max(3.5, 0.45 * n)), dpi=150)
    im = ax.imshow(sim, cmap='viridis', vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(session_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(session_labels, fontsize=8)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f'{agg.capitalize()} canonical angle (degrees)', fontsize=8)
    ax.set_title(f'Whole-session PCA subspace similarity\n(top-k @ {int(var_thresh * 100)}% variance per session)',
                 fontsize=9)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()
    return sim


def plot_pca_reference_drift(ref_s_id, session_ids, session_names, x_modality, var_thresh=0.8,
                              save_path=None, ith_to_dt_session=None):
    """More intuitive alternative to plot_session_pca_similarity's canonical-angle
    comparison (canonical angles in degrees aren't a very directly readable quantity) -
    fixes one reference session's whole-session PCA subspace (its top-k eigenvectors, k
    chosen the same var_thresh-cumulative-variance way as plot_session_pca_similarity)
    and asks one directly interpretable question for every session in `session_ids`
    (pass these in chronological order, typically every session up to and including
    ref_s_id - e.g. plot_session_ids[:-1] with ref_s_id = plot_session_ids[-1], the
    "does the most recent session's PCA basis still explain earlier sessions'
    variance" framing): "if I keep using ref_s_id's own principal directions, how much
    of *this* session's own variance do they still capture?" - i.e. project each
    session's own (centered) activity onto the reference's fixed top-k subspace and
    divide by that session's own total variance (trace of its own covariance) - no
    per-session PCA needed at all beyond the reference's own, unlike the canonical-angle
    comparison, which needs every session's own eigenvectors too. Plotted as one bar per
    session (in the order given), with a dashed reference line at var_thresh*100% - the
    value the reference session would trivially hit against itself (k was chosen to
    reach exactly that threshold there) - so any bar noticeably below the line directly
    shows how much that session's activity has drifted out of the reference's subspace,
    in the same %-of-variance units used throughout this section. Saves to save_path if
    given. Returns (labels, pct_explained)."""
    ref_session_name = next(sn for sn in session_names if sn.startswith(ref_s_id))
    ref_fr_z_raw = analytics.get_analytics('FiringRate40msZ',
                                            session_names=[ref_session_name]).xs(ref_s_id, level='session_id')
    ref_fr_x = session_fr_for_modality(ref_fr_z_raw, x_modality)
    ref_eigvals_pct, ref_eigvecs, _ = pca_eigen(ref_fr_x.values)
    k = max(1, int((ref_eigvals_pct.cumsum() < var_thresh * 100).sum()))
    ref_subspace = ref_eigvecs[:, :k]

    labels, pct_explained = [], []
    for s_id in session_ids:
        session_name = next(sn for sn in session_names if sn.startswith(s_id))
        fr_z_raw = analytics.get_analytics('FiringRate40msZ', session_names=[session_name]).xs(s_id, level='session_id')
        data = session_fr_for_modality(fr_z_raw, x_modality).values
        centered = data - data.mean(axis=0, keepdims=True)
        total_var = np.var(centered, axis=0, ddof=1).sum()  # trace(cov) - basis-invariant, no eigendecomposition needed
        projected_var = np.sum((centered @ ref_subspace) ** 2) / (len(centered) - 1)  # trace(P^T cov P), P orthonormal
        pct_explained.append(projected_var / total_var * 100)

        s_num = (ith_to_dt_session or {}).get(s_id, "Unknown")
        labels.append(f'S{s_num:02d}' if isinstance(s_num, int) else f'S{s_num}')

    ref_num = (ith_to_dt_session or {}).get(ref_s_id, "Unknown")
    ref_label = f'S{ref_num:02d}' if isinstance(ref_num, int) else f'S{ref_num}'

    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(session_ids)), 3.5), dpi=150)
    ax.bar(np.arange(len(session_ids)), pct_explained, color='#555555')
    ax.axhline(var_thresh * 100, color='purple', linestyle='--', linewidth=1.2,
               label=f"{ref_label}'s own {int(var_thresh * 100)}% (k={k} PCs)")
    ax.set_xticks(np.arange(len(session_ids)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("% of session's own variance\nexplained by reference PCs", fontsize=8)
    ax.set_title(f"Variance explained by {ref_label}'s PCA subspace, across sessions ({x_modality})", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    for sp in ax.spines.values():
        sp.set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()
    return labels, pct_explained
