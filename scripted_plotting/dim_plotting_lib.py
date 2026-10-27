"""Plotting library for dim_analysis.py.

Mirrors the split used for svm_plotting_lib.py: dim_analysis.py stays the live,
interactively-edited notebook (data loading, session selection, cell-by-cell exploration);
this file holds pure plotting functions that take their data as explicit arguments, so
nothing here does I/O or has side effects just from being imported.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
from scipy.linalg import subspace_angles
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import pdist


def _interval_short_label(which_interval):
    """CamelCase label for a t0 x-tick, e.g. 'cue_entry_interval' -> 'CueEntry' - parts
    that are already all-caps/alphanumeric (like 'R1') are left as-is rather than
    re-capitalized."""
    parts = which_interval.replace('_interval', '').split('_')
    return ''.join(p if p.isupper() else p.capitalize() for p in parts)


BIN_US = 40_000  # native Behavior40msAligned/FiringRate40ms* bin width


def get_trial_metric_traces(t0_events, data, s_id, which_interval, value_fn):
    """Every trial's per-bin metric for one (session, t0 interval), generalized over which
    40ms-binned table `data` is (Behavior40msAligned, FiringRate40msZ, ...) and what
    `value_fn(dat) -> 1D array` (one scalar per row of the trial's sliced window) actually
    extracts from it - e.g. `lambda dat: dat.frame_position.values` for position,
    or a population-magnitude computation for firing rate (see population_rms). `data`
    must carry the same 'trial_id' column and IntervalIndex (session_id, then a 40ms-bin
    IntervalIndex) that `behavior` does in dim_analysis.py - for FiringRate40msZ/Hz this
    means copying `behavior`'s own trial_id/index onto it first, since that table has
    neither natively (see dim_analysis.py's ephys cells: confirmed row-for-row identical
    to behavior's own 40ms grid on real data, since both derive from the exact same
    FiringRate40msHz timestamp columns - see get_Behavior40msAligned).

    t0_events has one row per (trial, detected zone) - not one row per trial - so
    `which_interval` is only non-null on the single row whose own zone matches that
    interval; select on that column directly rather than re-indexing by trial_id
    afterwards (ambiguous - resolves to *all* of a trial's zone rows, not the one that was
    just matched).

    Bin 0 is computed as `round((t0 - dat's own first row's left edge) / BIN_US)` - i.e.
    t0's offset (in bins) from wherever this trial's sliced data actually starts. Three
    other approaches were tried and rejected on real data before this one:
    - `{which_interval}_bins` (currently broken: get_TrialWiseT0Events40ms wraps its
      bin-offset array in a bare pd.Series before a `.loc[t0s.index, ...]` assignment,
      which aligns by index label rather than position - only rows whose label happens to
      collide with the Series' own default 0..N range get a real value, the rest None).
    - dividing `(time - t0)` by BIN_US using bin *centers* (`.mid`): t0 generally lands
      exactly on a bin *edge*, not a bin center, so that division produces duplicate
      half-integer-rounded bins (e.g. -2,-2,0,0,2,2,...).
    - locating whichever row's own interval *contains* t0 (`get_indexer`): fails whenever
      t0 sits exactly on the window's own left edge - true whenever an interval's
      n_bins_left is 0 (e.g. ITI_interval, aligned to track_end) - since a closed='right'
      interval's left edge belongs to the *previous*, excluded bin, get_indexer returns -1
      for every trial and the interval silently disappears from the plot.
    The edge-relative version above has none of these failure modes: it's exact for any
    n_bins_left (including 0), and self-corrects if a trial's window got truncated at the
    start (session edge) since it's relative to dat's actual first row, not the nominal
    window bounds.

    A trial is dropped if its actual (length, t0 offset) doesn't match the modal shape
    across trials of this interval - everything downstream stacks trials into one
    rectangular (n_trials, n_bins) array, so a single differently-shaped trial would
    break that. The mismatch is usually interval spilling past the start/end of the
    recording, at a session's edges - but that's not the only cause: an interval whose
    own end/start is anchored to a second, variable-latency event (rather than a fixed
    bin offset from t0) can legitimately have a different real-world duration - and thus
    bin count - from one trial to the next even nowhere near a session edge; separately,
    a trial can also mismatch on t0_offset alone (identical length, different offset) if
    its underlying event timestamps land on a different bin of the 40ms grid than most
    trials' due to sub-bin timing jitter. Dropped trial_ids and their own (length, offset)
    vs. the modal one are printed (see below) precisely so a mismatch can be told apart
    from the others by going and looking at that specific trial. Returns (traces, bins):
    `traces` a list of equal-length arrays (one per kept trial), `bins` the shared offset
    array (None if no trial survived).
    """
    sess_t0 = t0_events.xs(s_id, level='session_id')
    interval_rows = sess_t0[sess_t0[which_interval].notna()]
    data_s = data.xs(s_id, level='session_id')

    raw = []
    no_overlap_trial_ids = []
    for trial_id, row in interval_rows.iterrows():
        trial_data = data_s[data_s['trial_id'] == trial_id]
        dat = trial_data.loc[trial_data.index.overlaps(row[which_interval])]
        if len(dat) == 0:
            no_overlap_trial_ids.append(trial_id)
            continue
        t0_offset = int(round((row['t0'] - dat.index[0].left) / BIN_US))
        raw.append((trial_id, value_fn(dat), np.arange(len(dat)) - t0_offset))

    if no_overlap_trial_ids:
        print(f"  {which_interval}: trial_id(s) {no_overlap_trial_ids} had no overlapping "
              f"data at all (out of {len(interval_rows)} trials with this event)")

    if not raw:
        print(f"  {which_interval}: no trial had any overlapping data")
        return [], None
    shapes = pd.Series([(len(vals), bins[0]) for _, vals, bins in raw])
    modal_shape = shapes.value_counts().index[0]
    kept = [(vals, bins) for (_, vals, bins), shape in zip(raw, shapes) if shape == modal_shape]
    dropped = [(trial_id, shape) for (trial_id, _, _), shape in zip(raw, shapes) if shape != modal_shape]
    if dropped:
        print(f"  {which_interval}: kept {len(kept)}/{len(raw)} trials at the modal "
              f"(length, t0_offset) shape {modal_shape} - dropped {dropped}")
    return [vals for vals, _ in kept], kept[0][1]


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


SEC_BINS = 25  # 1000ms / 40ms-per-bin


def _layout_intervals(t0_events, data, s_id, which_intervals, value_fn, gap_bins=2):
    """Fetch traces/bins for every interval once (via get_trial_metric_traces) and resolve
    each one's x-offset in a single pass, so a caller can size a figure to the total bin
    span before drawing anything into it (see plot_session_trial_kinematics) without
    fetching/computing the same layout twice. Returns (layout, total_bins): `layout` a
    list of (which_interval, traces, bins, offset) tuples (intervals with no surviving
    trial omitted), `total_bins` the full x-axis span the concatenated intervals occupy.

    Placing interval i+1 must account for both edges: its own left edge sits at
    `bins.min()` (negative - R1/R2-entry intervals start well before their own t0), not
    at 0, so advancing the offset by just the *previous* interval's `bins.max()` alone
    made intervals overlap in x (confirmed on real data - R1/R2-entry's -9 left edge sat
    9 bins inside the prior interval's right edge, reading as a missing "pre-t0" tail
    rather than as overlap). `offset` is instead solved so this interval's own left edge
    lands exactly `gap_bins` past the previous interval's right edge.

    A second call with a different `data`/`value_fn` (e.g. firing rate instead of
    position) but the same `which_intervals`/`gap_bins` produces numerically identical
    offsets, since offsets only depend on each interval's bin range, which is itself
    governed by the shared 40ms timestamp grid, not by which table got sliced - confirmed
    on real data - so two such layouts drawn on sharex=True axes line up without needing
    to force one row's offsets onto the other.
    """
    layout = []
    prev_right_edge = None
    for which_interval in which_intervals:
        traces, bins = get_trial_metric_traces(t0_events, data, s_id, which_interval, value_fn)
        if bins is None:
            continue
        offset = 0 if prev_right_edge is None else prev_right_edge + gap_bins - bins.min()
        prev_right_edge = bins.max() + offset
        layout.append((which_interval, traces, bins, offset))

    if not layout:
        return layout, 0
    left_edge = layout[0][3] + layout[0][2].min()
    total_bins = prev_right_edge - left_edge
    return layout, total_bins


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


def _smooth(values, window):
    """Centered rolling mean over `window` timebins - a no-op if window is falsy/<=1.
    min_periods=1 keeps the array's original length (no NaN edges), so this is safe to
    apply per-interval right before plotting without shrinking/misaligning the x-axis."""
    if not window or window <= 1:
        return values
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().values


def _draw_trial_lines(ax, layout, alpha=0.05, color='k', smooth=1):
    """Row 0 style: every trial's own line, drawn individually, optionally smoothed (see
    _smooth) - off by default (smooth=1). Adapted from row 0 of
    svm_plotting_lib.create_subplot, stripped of everything decoding-specific (no SVM
    timebin resampling, no score-colored scatter, no cue-zone illustration)."""
    for _, traces, bins, offset in layout:
        for vals in traces:
            ax.plot(bins + offset, _smooth(vals, smooth), color=color, alpha=alpha, linewidth=1, zorder=2)


def _draw_mean_band(ax, layout, color='k', band='sem', smooth=1,
                     scatter_by_x=None, scatter_cmap='coolwarm', scatter_norm=None, scatter_size=25):
    """One row: mean-across-trials line + a shaded band per bin, instead of one line per
    trial - meant for a metric whose per-trial lines would just be noise (e.g. population
    firing-rate magnitude) rather than something individually meaningful the way a
    trial's own position trace is. `band`: 'sem' (default - standard error of the
    trial mean, i.e. how precisely the mean itself is known) or 'sd' (raw across-trial
    spread, i.e. how much individual trials vary). `smooth` (see _smooth, off by default)
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
    for _, traces, bins, offset in layout:
        stacked = np.asarray([_smooth(vals, smooth) for vals in traces], dtype=float)  # (n_trials, n_bins)
        mean = np.nanmean(stacked, axis=0)
        spread = np.nanstd(stacked, axis=0)
        if band == 'sem':
            n = np.sum(~np.isnan(stacked), axis=0)
            spread = spread / np.sqrt(np.maximum(n, 1))
        x = bins + offset
        ax.fill_between(x, mean - spread, mean + spread, color=color, alpha=0.2, zorder=2, edgecolor=None)
        if scatter_by_x is not None:
            c = np.array([scatter_by_x.get(xi, np.nan) for xi in x])
            mappable = ax.scatter(x, mean, c=c, cmap=scatter_cmap, norm=scatter_norm,
                                   s=scatter_size, zorder=2.5, edgecolor='none')
        ax.plot(x, mean, color=color, linewidth=1.2, zorder=3)
    return mappable


def _draw_mean_line(ax, layout, color='k', linestyle=':', linewidth=1.2, smooth=1):
    """One row: just the mean-across-trials line, no band - a lighter-weight variant of
    _draw_mean_band for a secondary annotation line (e.g. plot_session_kinematics_neural_
    and_pca's `mean_z_range` twin-axis overlay) where a shaded band would add visual
    clutter without much value, since the line itself is already secondary to whatever
    shares its axes. `smooth` (see _smooth, off by default) is applied per-trial before
    averaging, same as _draw_mean_band."""
    for _, traces, bins, offset in layout:
        stacked = np.asarray([_smooth(vals, smooth) for vals in traces], dtype=float)
        mean = np.nanmean(stacked, axis=0)
        x = bins + offset
        ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=linewidth, zorder=3)


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
    (n_bins, n_units) array - see _pc_block_with_separators."""
    if not window or window <= 1:
        return arr2d
    return np.column_stack([_smooth(arr2d[:, j], window) for j in range(arr2d.shape[1])])


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


def _pca_basis_for_n_pcs(M, n_pcs):
    """Top `n_pcs` PCA components (via SVD, trials as observations/repetitions, neurons as
    the dimension being reduced - see plot_subspace_angle_heatmap) of an (n_trials,
    n_units) matrix - a fixed component count at every timebin (unlike an earlier,
    since-replaced version of this function that instead grew/shrank k to hit a target
    cumulative-variance threshold), so canonical-angle comparisons across timebins (see
    plot_subspace_angle_heatmap) are always between equal-dimension subspaces. Capped at
    `Vt.shape[0]` (== min(n_trials, n_units) - 1 after centering) whenever that's smaller
    than `n_pcs`. Returns (basis, var_explained, neuron_contrib):

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
    M = _fill_nan_columns(M)
    centered = M - M.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    total_var = np.sum(S ** 2)
    k = min(n_pcs, Vt.shape[0])
    if total_var == 0:
        return Vt[:max(k, 1)], 1.0, np.full(M.shape[1], np.nan)
    var_explained = np.sum(S[:k] ** 2) / total_var
    neuron_contrib = (S[:k] ** 2) @ (Vt[:k] ** 2) / total_var
    return Vt[:k], var_explained, neuron_contrib


def _subspace_block_with_separators(t0_events, fr_z, s_id, which_intervals, unit_cols, gap_bins=2,
                                     smooth=5, n_pcs=8):
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
    trials/units available.

    Returns (x_coords, bases, var_explained, layout, contribs): `bases[i]` is the
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
        return np.array([]), [], [], layout, []

    left_edge = layout[0][3] + layout[0][2].min()
    right_edge = layout[-1][3] + layout[-1][2].max()
    x_coords = np.arange(left_edge, right_edge + 1)
    bases = [np.zeros((0, len(unit_cols))) for _ in x_coords]
    var_explained = [np.nan for _ in x_coords]
    contribs = [np.full(len(unit_cols), np.nan) for _ in x_coords]

    for _, traces, bins, offset in layout:
        stacked = np.stack([_smooth_2d(trial, smooth) for trial in traces], axis=0)  # (n_trials, n_bins, n_units)
        for local_t, b in enumerate(bins):
            idx = b + offset - left_edge
            bases[idx], var_explained[idx], contribs[idx] = _pca_basis_for_n_pcs(
                stacked[:, local_t, :], n_pcs)

    return x_coords, bases, var_explained, layout, contribs


def plot_subspace_angle_heatmap(ax, t0_events, fr_z, s_id, which_intervals, unit_cols=None,
                                 gap_bins=2, smooth=5, n_pcs=8, summary='mean', angle_range=(0, 90)):
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
    x_coords, bases, _, layout, _ = _subspace_block_with_separators(
        t0_events, fr_z, s_id, which_intervals, unit_cols, gap_bins=gap_bins,
        smooth=smooth, n_pcs=n_pcs)
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


_NEURON_CONTRIB_SEL_COLOR = '#e08214'   # selected cells (pre row cap)
_NEURON_CONTRIB_DRAWN_COLOR = '#b2182b'  # cells actually drawn in the heatmap
_NEURON_CONTRIB_COUNT_COLOR = '#4393c3'  # cells per timebin, on the twin axis


def _draw_neuron_contribution_summary(ax, x_coords, info, var_explained, layout, select,
                                       top_per_pc, top_n, n_pcs, fill='dense'):
    """Thin row under the neuron-contribution heatmap that says, in the one unit that's
    intuitive here (% of that timebin's total across-trial variance), how much of the picture
    above is actually accounted for - and what the selection settings cost:

    - dotted gray, `<n_pcs> PCs explained var`: the full variance explained by the PCA subspace,
      the one curve in this figure that is about the whole population. This is the ceiling: it's what the
      *whole* population's contributions sum to, so no other line here can exceed it.
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
    ve = np.asarray(var_explained, dtype=float) * 100
    sel = info['col_sum_selected'] * 100
    drawn = info['col_sum_drawn'] * 100
    rule = f'top-{top_per_pc}/PC' if select == 'per_pc' else f'top-{top_n} contrib'

    _draw_scaffolding(ax, layout, show_bin_lines=False)
    ax.plot(x_coords, ve, color='gray', linestyle=':', linewidth=1,
            label=f'{n_pcs} PCs explained var (all units)', zorder=2)
    ax.plot(x_coords, sel, color=_NEURON_CONTRIB_SEL_COLOR, linewidth=1,
            label=f'selected ({rule})', zorder=2)
    ax.plot(x_coords, drawn, color=_NEURON_CONTRIB_DRAWN_COLOR, linewidth=1.2,
            label='shown neurons' if fill == 'dense' else 'drawn (after row cap)', zorder=3)
    top = min(100, (np.nanmax(ve) if np.isfinite(ve).any() else 100) * 1.35)
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
    count_lines = [(info['n_selected_per_col'], 'cells selected', '--', None),
                   (info['n_drawn_per_col'], 'cells drawn', '-', 'steps-mid')]
    count_lines = [c for c in count_lines if _varies(c[0])]
    twin = None
    if count_lines:
        twin = ax.twinx()
        # twinx() re-shows its own x tick labels over the (hidden) shared ones - see the same
        # fix for the mean_z twin axis in plot_session_kinematics_neural_and_pca.
        twin.tick_params(axis='x', labelbottom=False, bottom=False)
        for values, label, style, drawstyle in count_lines:
            twin.plot(x_coords, values, color=_NEURON_CONTRIB_COUNT_COLOR, linewidth=0.8,
                      linestyle=style, drawstyle=drawstyle or 'default', label=label, zorder=2)
        cap = max(np.nanmax(v) for v, *_ in count_lines) * 1.6
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
        ratio = drawn / ve
    for _, _, bins, offset in layout:
        idx = bins + offset - x_coords[0]
        idx = idx[(idx >= 0) & (idx < len(ratio))]
        vals = ratio[idx]
        if not np.isfinite(vals).any():
            continue
        ax.text(offset + 5, 0.01, f'{np.nanmedian(vals):.0%}', transform=ax.get_xaxis_transform(),
                fontsize=6, ha='center', va='bottom', color=_NEURON_CONTRIB_DRAWN_COLOR)
    return twin


def plot_neuron_contribution_heatmap(ax, t0_events, fr_z, s_id, which_intervals, unit_cols=None,
                                      gap_bins=2, smooth=5, n_pcs=8, select='per_pc', top_per_pc=3,
                                      top_n=12, max_rows=25, fill='dense', value_range=None,
                                      verbose=True):
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
    x_coords, bases, _, layout, contribs = _subspace_block_with_separators(
        t0_events, fr_z, s_id, which_intervals, unit_cols, gap_bins=gap_bins,
        smooth=smooth, n_pcs=n_pcs)
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


def plot_session_kinematics_neural_and_pca(t0_events, behavior, fr_z, s_id, which_intervals,
                                            unit_cols=None, smooth=3, pca_smooth=3, pca_n_pcs=8,
                                            angle_range=(0, 90), metric_range=None, mean_z_range=None,
                                            draw_colorbar=True, var_explained_display='scatter', var_explained_center=0.4,
                                            var_explained_range=(0.1, 0.7), show_neuron_contrib=False,
                                            neuron_contrib_select='per_pc', neuron_contrib_top_per_pc=3,
                                            neuron_contrib_top_n=12, neuron_contrib_max_rows=25,
                                            neuron_contrib_fill='dense', neuron_contrib_range=(0, 0.10),
                                            show_neuron_contrib_summary=True):
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
    `neuron_contrib_range` (default (0, 0.10), i.e.
    0-10% of total variance per cell) is the color scale, as a (min, max) pair of *fractions*
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

    `show_neuron_contrib_summary` (default True) adds a thin line row underneath it that
    quantifies what the selection settings are actually showing you, in % of total across-trial
    variance: the subspace's full variance explained as a ceiling, what the selected cells sum
    to, what the drawn cells sum to after the row cap, and how many cells each timebin drew -
    see _draw_neuron_contribution_summary for how to read it and what each gap between the
    curves means. It's an independent switch (it can be shown with `show_neuron_contrib=False`,
    as a compact stand-in for the whole heatmap, or suppressed while keeping the heatmap), and
    it reuses the same computed matrix, so it costs nothing extra.

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

    pos_layout, total_bins = _layout_intervals(
        t0_events, behavior, s_id, which_intervals,
        value_fn=lambda dat: dat.frame_position.values, gap_bins=_GAP_BINS)
    neural_layout, _ = _layout_intervals(
        t0_events, fr_z, s_id, which_intervals,
        value_fn=lambda dat: population_rms(dat, unit_cols), gap_bins=_GAP_BINS)

    if var_explained_display not in ('scatter', 'off'):
        raise ValueError(f"var_explained_display must be 'scatter' or 'off', got "
                         f"{var_explained_display!r} ('row' was removed - the same curve is now "
                         f"the ceiling line of the show_neuron_contrib_summary row)")
    show_var_explained_scatter = var_explained_display == 'scatter'
    # One per-timebin PCA pass feeds the variance-explained display, the neuron-contribution
    # row and its summary row - each independently optional, so fetch if any wants it.
    need_neuron_contrib = show_neuron_contrib or show_neuron_contrib_summary
    if var_explained_display != 'off' or need_neuron_contrib:
        x_coords_pc, bases_pc, var_explained_pc, layout_pc, contribs_pc = \
            _subspace_block_with_separators(
                t0_events, fr_z, s_id, which_intervals, unit_cols, gap_bins=_GAP_BINS,
                smooth=pca_smooth, n_pcs=pca_n_pcs)
    else:
        x_coords_pc, bases_pc, var_explained_pc, layout_pc, contribs_pc = np.array([]), [], [], [], []

    nc_rows, nc_row_unit_idx, nc_info = np.zeros((0, 0)), np.array([], dtype=int), None
    if need_neuron_contrib and len(x_coords_pc):
        nc_rows, nc_row_unit_idx, nc_info = _neuron_contribution_matrix(
            bases_pc, contribs_pc, len(unit_cols), select=neuron_contrib_select,
            top_per_pc=neuron_contrib_top_per_pc, top_n=neuron_contrib_top_n,
            max_rows=neuron_contrib_max_rows, fill=neuron_contrib_fill)
        _report_neuron_contribution(nc_info, neuron_contrib_select, neuron_contrib_top_per_pc,
                                    neuron_contrib_top_n, neuron_contrib_max_rows,
                                    fill=neuron_contrib_fill)

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
            ax1, neural_layout, smooth=smooth, scatter_by_x=scatter_by_x,
            scatter_cmap=_VAR_EXPLAINED_CMAP, scatter_norm=scatter_norm)
    else:
        _draw_mean_band(ax1, neural_layout, smooth=smooth)
    _finalize_axes(ax1, 'Population RMS\n(z-scored Hz)')
    if metric_range is not None:
        ax1.set_ylim(*metric_range)

    if mean_z_range is not None:
        mean_z_layout, _ = _layout_intervals(
            t0_events, fr_z, s_id, which_intervals,
            value_fn=lambda dat: population_mean(dat, unit_cols), gap_bins=_GAP_BINS)
        ax1_twin = ax1.twinx()
        # twinx() shares ax1's x-axis but defaults to showing its own x tick labels too -
        # would duplicate row 1's (already-hidden, see ax1.tick_params above) bottom
        # labels right on top of it.
        ax1_twin.tick_params(axis='x', labelbottom=False, bottom=False)
        _draw_mean_line(ax1_twin, mean_z_layout, smooth=smooth)
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
                pca_n_pcs, fill=neuron_contrib_fill)

    _, mesh = plot_subspace_angle_heatmap(ax2, t0_events, fr_z, s_id, which_intervals, unit_cols=unit_cols,
                                           gap_bins=_GAP_BINS, smooth=pca_smooth, n_pcs=pca_n_pcs,
                                           summary=_ANGLE_SUMMARY, angle_range=angle_range)
    _apply_xticks(ax2, neural_layout)
    ax2.spines[['top', 'right']].set_visible(False)

    if draw_colorbar and mesh is not None:
        fig.colorbar(mesh, cax=cax, orientation='horizontal', label='Canonical angle (deg)')
    if draw_colorbar and scatter_mappable is not None:
        fig.colorbar(scatter_mappable, cax=cax_ve, orientation='horizontal',
                     label=f'Variance explained (top {pca_n_pcs} PCs)')
    if draw_colorbar and nc_mesh is not None:
        fig.colorbar(nc_mesh, cax=cax_nc, orientation='horizontal',
                     label=f'Neuron variance share (top {pca_n_pcs} PCs, %)')

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


# Canvas assembly - ported verbatim from svm_plotting_lib.py/svm_analysis_2026.py (same
# helper there, kept manually in sync between the two - see that file's own docstring),
# since dim_analysis.py wants the same "one PNG/SVG per session -> one combined poster"
# workflow for its own per-session figures.

def _stack_vertically(image_paths):
    """Stack a session's own plots top to bottom, left-aligned."""
    images = [Image.open(p) for p in image_paths]
    width = max(img.width for img in images)
    height = sum(img.height for img in images)
    stacked = Image.new('RGB', (width, height), color='white')
    y = 0
    for img in images:
        stacked.paste(img, (0, y))
        y += img.height
    return stacked


def build_canvas(session_image_paths, canvas_path):
    """session_image_paths: one list of PNG paths per session (in the order they should
    stack vertically, e.g. main plot first then any secondary plot - pass a single-element
    list per session, e.g. `[[p] for p in paths]`, for one image per session with no
    vertical stacking). Each session's stack is then placed side by side with every other
    session's, left-aligned, on a white canvas sized to fit all of them (no scaling -
    images are pasted at their native pixel size), then saved to `canvas_path`."""
    session_composites = [_stack_vertically(paths) for paths in session_image_paths]
    total_width = sum(img.width for img in session_composites)
    max_height = max(img.height for img in session_composites)
    canvas = Image.new('RGB', (total_width, max_height), color='white')
    x = 0
    for img in session_composites:
        canvas.paste(img, (x, 0))
        x += img.width
    canvas.save(canvas_path)
    return canvas


_SVG_NS = 'http://www.w3.org/2000/svg'
_XLINK_NS = 'http://www.w3.org/1999/xlink'
ET.register_namespace('', _SVG_NS)
ET.register_namespace('xlink', _XLINK_NS)


def _svg_root_and_size(svg_path):
    """Parse an SVG file's root element and its (width, height) in the same numeric units
    as its own viewBox (matplotlib writes both width/height and viewBox in points, with
    matching numbers, for every SVG it saves - see fig.savefig(*.svg))."""
    root = ET.parse(svg_path).getroot()
    view_box = root.get('viewBox')
    if view_box is not None:
        _, _, width, height = (float(v) for v in view_box.split())
    else:
        width = float(root.get('width').rstrip('pt'))
        height = float(root.get('height').rstrip('pt'))
    return root, width, height


def build_canvas_svg(session_svg_paths, canvas_path):
    """SVG analog of build_canvas, for the vector (.svg) exports saved alongside the
    PNGs - same one-list-of-paths-per-session layout (stacked vertically per session,
    sessions placed side by side). Rather than rasterizing (Image.paste, like
    build_canvas), each source SVG's own content is moved into a nested
    <svg x=.. y=.. width=.. height=..> element inside one master <svg> - nested <svg>
    elements support exactly this kind of positioning natively, so the combined file
    stays fully vector/editable (fonts, paths, etc. all stay intact and movable in
    Inkscape/Illustrator, unlike flattening to a raster canvas would)."""
    session_layouts = []
    for paths in session_svg_paths:
        parsed = [_svg_root_and_size(p) for p in paths]
        col_width = max(w for _, w, _ in parsed)
        col_height = sum(h for _, _, h in parsed)
        session_layouts.append((parsed, col_width, col_height))

    total_width = sum(col_width for _, col_width, _ in session_layouts)
    max_height = max(col_height for _, _, col_height in session_layouts)

    master = ET.Element(f'{{{_SVG_NS}}}svg', {
        'width': f'{total_width}pt', 'height': f'{max_height}pt',
        'viewBox': f'0 0 {total_width} {max_height}', 'version': '1.1',
    })

    x_cursor = 0.0
    for parsed, col_width, _ in session_layouts:
        y_cursor = 0.0
        for root, width, height in parsed:
            nested = ET.SubElement(master, f'{{{_SVG_NS}}}svg', {
                'x': str(x_cursor), 'y': str(y_cursor), 'width': str(width), 'height': str(height),
                'viewBox': f'0 0 {width} {height}',
            })
            for child in list(root):
                nested.append(child)
            y_cursor += height
        x_cursor += col_width

    ET.ElementTree(master).write(canvas_path, xml_declaration=True, encoding='utf-8')
    return canvas_path
