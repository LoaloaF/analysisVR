"""Turning t0-aligned trials into (n_trials, n_bins) arrays, and laying intervals out on x.

The layer every figure in this tree stands on, and the one that had drifted into two
generations. The older one (svm_analysis_2026 / svm_plotting_lib) assumed each interval's
t0 sat at a fixed, hardcoded bin offset; this one - written later for dim_analysis and
kept as the survivor - measures each trial's own t0 offset from the data and drops trials
that disagree with the modal shape, reporting which. See `get_trial_metric_traces` for the
three approaches that were tried and rejected against real data before this one.

Two different x-axis conventions meet here, and they are not interchangeable:

  * `layout_intervals` (this module, used by dim_plotting_lib) puts **t0 at bin 0** and
    lets bin indices go negative - the interval's own geometry is then self-describing.
  * The SVM libraries index by **timebin 0..n_timebins-1**, where t0 sits at
    `n_bins_left`, because that is how the fits themselves are indexed
    (`timeline_decoding` slides a 3-bin window from the interval's left edge).

`interval_n_bins_left` is the bridge: it measures n_bins_left from `t0_events` so the SVM
side can place its t0 markers without a hardcoded constant, and without reading the
`n_bins_left` column stamped on the fit rows (which is being removed - t0_events is the
authority on interval geometry).
"""

import numpy as np
import pandas as pd


BIN_US = 40_000   # native Behavior40msAligned / FiringRate40ms* bin width
SEC_BINS = 25     # 1000ms / 40ms-per-bin


def interval_short_label(which_interval):
    """CamelCase label for a t0 x-tick, e.g. 'cue_entry_interval' -> 'CueEntry' - parts
    that are already all-caps/alphanumeric (like 'R1') are left as-is rather than
    re-capitalized."""
    parts = which_interval.replace('_interval', '').split('_')
    return ''.join(p if p.isupper() else p.capitalize() for p in parts)


def smooth(values, window):
    """Centered rolling mean over `window` timebins - a no-op if window is falsy/<=1.
    min_periods=1 keeps the array's original length (no NaN edges), so this is safe to
    apply per-interval right before plotting without shrinking/misaligning the x-axis.

    A NaN input stays NaN in the output. min_periods=1 would otherwise fill it from the
    neighbouring bins, which is exactly wrong for the holes the SVM_TaskSubspaces tables
    carry: a NaN there is a collapsed fit (no model at all for that timebin), and it is
    meant to read as a gap in the line rather than be interpolated over."""
    if not window or window <= 1:
        return values
    series = pd.Series(values, dtype=float)
    return series.rolling(window, center=True, min_periods=1).mean().mask(series.isna()).values


def get_trial_metric_traces(t0_events, data, s_id, which_interval, value_fn,
                            return_trial_ids=False):
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

    `return_trial_ids=True` returns (traces, bins, trial_ids) instead - the trial_id of each
    kept trace, in the same order. Needed by anything that has to join these traces back to
    per-trial information the caller holds separately, e.g. a group label for a permutation
    test (see dim_plotting_lib._group_shuffle_band): the returned list is already
    modal-shape-filtered, so its length and order cannot be reconstructed from t0_events
    afterwards.
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
        return ([], None, []) if return_trial_ids else ([], None)
    shapes = pd.Series([(len(vals), bins[0]) for _, vals, bins in raw])
    modal_shape = shapes.value_counts().index[0]
    kept = [(tid, vals, bins) for (tid, vals, bins), shape in zip(raw, shapes) if shape == modal_shape]
    dropped = [(trial_id, shape) for (trial_id, _, _), shape in zip(raw, shapes) if shape != modal_shape]
    if dropped:
        print(f"  {which_interval}: kept {len(kept)}/{len(raw)} trials at the modal "
              f"(length, t0_offset) shape {modal_shape} - dropped {dropped}")
    traces, bins = [vals for _, vals, _ in kept], kept[0][2]
    return (traces, bins, [tid for tid, _, _ in kept]) if return_trial_ids else (traces, bins)


def layout_intervals(t0_events, data, s_id, which_intervals, value_fn, gap_bins=2,
                     return_trial_ids=False):
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

    `return_trial_ids=True` makes every layout entry a 5-tuple
    `(which_interval, traces, bins, offset, trial_ids)` instead of a 4-tuple - the same
    extension `get_trial_metric_traces` already offers, threaded through. Needed by
    anything that has to split this interval's traces by a per-trial label the caller holds
    separately (neuro_plotting_lib's condition heatmap): the traces are already
    modal-shape-filtered, so which trials survived cannot be reconstructed from t0_events
    afterwards.
    """
    layout = []
    prev_right_edge = None
    for which_interval in which_intervals:
        traces, bins, trial_ids = get_trial_metric_traces(
            t0_events, data, s_id, which_interval, value_fn, return_trial_ids=True)
        if bins is None:
            continue
        offset = 0 if prev_right_edge is None else prev_right_edge + gap_bins - bins.min()
        prev_right_edge = bins.max() + offset
        entry = (which_interval, traces, bins, offset)
        layout.append(entry + (trial_ids,) if return_trial_ids else entry)

    if not layout:
        return layout, 0
    left_edge = layout[0][3] + layout[0][2].min()
    total_bins = prev_right_edge - left_edge
    return layout, total_bins


def interval_n_bins_left(t0_events, interval_name, s_id=None, warn=True):
    """How many 40ms bins of `interval_name`'s window sit before t0, measured from the
    stored pd.Interval - i.e. where the t0 marker belongs on an axis indexed 0..n_bins.

    This is the authority the SVM plotting libraries should use, replacing both the
    `n_bins_left` column stamped on the fit rows (being removed) and the hardcoded
    "3 for cue_entry, else 10" fallback (which silently disagreed with the data after
    R1_entry_interval was retuned from 10 to 4).

    With `s_id`, that session's own value; without, the modal value pooled over sessions -
    which is what a figure sharing one x-axis across sessions needs. Sessions do disagree
    in practice (the stale t0 table has R1_entry_interval at 10 bins for 26 sessions and
    15 for 3), so the pooled case warns rather than silently picking one."""
    rows = t0_events[t0_events[interval_name].notna()]
    if s_id is not None:
        rows = rows[rows.index.get_level_values('session_id') == s_id]
    if len(rows) == 0:
        raise ValueError(f"no {interval_name} rows in t0_events"
                         + (f" for session {s_id}" if s_id else ""))
    left = ((rows['t0'] - rows[interval_name].array.left) / BIN_US).round().astype(int)
    counts = left.value_counts()
    if warn and len(counts) > 1 and s_id is None:
        print(f"WARNING: {interval_name} has differing n_bins_left across sessions "
              f"({dict(counts)}); using the modal value {counts.index[0]} for the shared x-axis")
    return int(counts.index[0])


def interval_x0_map(t0_events, which_intervals, s_id=None, warn=True):
    """{interval_name: n_bins_left} for several intervals at once - what the SVM figure
    composers pass down to the row-level drawing helpers, so those never need `t0_events`
    themselves (see interval_n_bins_left)."""
    return {iv: interval_n_bins_left(t0_events, iv, s_id=s_id, warn=warn)
            for iv in which_intervals if iv in t0_events.columns}


# How a t0 marker is drawn, wherever it is drawn - one definition so the per-session
# figure's rows can't drift apart in style the way they had drifted in geometry (see
# block_x). Dotted rather than dashed, and thin: the row-2 heatmap draws these as one
# segment per interval block, and at a dash's length the one-cell break at a block
# separator read as just another gap in the dash pattern. A dotted pattern also scales
# with linewidth, so the thin line is what puts the dots close together; the alpha is up
# from the old dashed 0.4 to keep it as visible as before at a third of the width.
T0_LINE_KW = dict(color='k', linestyle=':', alpha=0.7, linewidth=0.8)


def block_x(timebins, offset):
    """x positions for one interval's fit rows on the shared, gapped SVM x-axis.

    That x-axis is the *W-block row index* (svm_plotting_lib's
    _w_block_with_interval_separators): the row-2 heatmap draws block row r with
    pcolormesh's default 'nearest' shading, so row r's cell is centred exactly on x=r and
    the whole figure is xlim(-0.5, n_block_rows-0.5). The stored `timebin` is 1-based
    (`position + 1`, see svm_task_subspaces._sweep_positions), so timebin t of the
    interval whose block starts at `offset` belongs at `offset + t - 1` - i.e. the
    interval's first row sits *on* its block start, not one bin past it.

    Every row of the per-session figure must use this (and `next_block_offset` to advance
    between intervals) or it silently desynchronizes from the heatmap: plotting at
    `timebin + offset` and advancing by `timebin.max() + 1 (+1)` - what the score,
    variance and covariate rows did - is this scheme shifted by one bin per interval
    drawn, so with three intervals the line rows' t0 markers and interval gaps sat 1, 2
    and 3 bins right of the heatmap cells they were supposed to label.
    """
    return np.asarray(timebins) - 1 + offset


def next_block_offset(offset, n_bins, is_last=False):
    """Block start of the next interval: this interval's `n_bins` rows, plus the 1-row
    separator _w_block_with_interval_separators inserts between consecutive intervals
    (none after the last one, so the returned final offset is the block's total row count
    - the same `n` that xlim(-0.5, n-0.5) is built from). `n_bins` is the interval's fit-
    row count, matching the block, NOT `timebin.max()`."""
    return offset + n_bins + (0 if is_last else 1)
