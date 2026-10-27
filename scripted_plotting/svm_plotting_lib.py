"""Pure plotting library for the SVM cue/choice decoding analysis.

Extracted from `svm_analysis_2026.py` (2026-08-03) so `plot_session_analysis` (the
per-session poster figure - position traces, F1/accuracy-vs-time, W cosine-similarity
heatmap, covariate/correctness split panels, optional W-comparison figure) and its full
dependency closure can be imported from another notebook without re-running that
notebook's own top-level cells (which fetch real session data from the NAS and produce
plots as side effects just by being imported - see the "importing a notebook" discussion
that led to this file). Nothing in here does I/O or touches the NAS on import; every
function takes its data (t0_events/behavior/svm_results/etc.) as an explicit argument.

The cross-session heatmap view of the same decoding metric (one row per session, all
sessions in one figure) lives in its own module, `svm_heatmap_plotting_lib.py` - it imports
this file's conventions (METRIC_SPECS, the CCGP election, _smooth, the t0/'1s' x-ticks)
rather than duplicating them.

`svm_analysis_2026.py` remains the live, actively-edited research notebook - keep this
file in sync by re-copying the affected function(s) here if their signatures/behavior
change there (see that file's own `plot_session_analysis` docstring and the functions
listed below for the exact dependency closure this covers).
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
from PIL import Image
from sklearn.metrics import f1_score, balanced_accuracy_score


modality_colors = {
    'cue': {0: 'orange', 1: 'purple'},
    'choice_R1': {0: "#3094db", 1: "#0044ff"},
    'choice_R2': {0: "#7effda", 1: "#3aff5b"},
    'trial_outcome': {0: "#ff8585", 1: "#9dff9d"},
    'correct_R1': {0: "#690000", 1: "#004c00"},
    'correct_R2': {0: "#690000", 1: "#004c00"},
}
ALL_Y_PREDICTED_NAMES = ['cue', 'choice_R1', 'choice_R2', 'trial_outcome']

# hardcoded 2 std boundaries of 2 random angles within a high dim space, matching dims of our data:
# so HP eg is 20 dims and is therfore more spread out around 90, versus mPFC with eg 54 dims
mPFC_upper = 110.4
mPFC_lower = 68.5
HP_upper = 145.0
HP_lower = 19.2
HP_mPFC_upper = 108.75
HP_mPFC_lower = 70.27
angle_2std_thr = {
    'mPFC': (mPFC_lower, mPFC_upper),
    'HP': (HP_lower, HP_upper),
    'HP+mPFC': (HP_mPFC_lower, HP_mPFC_upper)
}

# Which pre-fit/on-the-fly metric to draw throughout plot_session_analysis and its
# helpers - pass metric='balanced_accuracy' anywhere a `metric` kwarg is accepted to
# switch every panel (row-0 scatter color, row-1 F1-vs-time, covariate/correctness
# panels, markdown export) from F1 to balanced accuracy at once. aggr/mean/ci_* map to
# the matching precomputed svm_results columns; shuffle_mean/shuffle_p to the optional
# shuffle-control columns (see n_shuffles in get_SVMCueOutcomeChoicePred); score_fn is
# used wherever a metric is recomputed on the fly from y_true/predictions (e.g. after
# splitting trials by a covariate).
METRIC_SPECS = {
    'f1': dict(
        aggr_col='f1_aggr', mean_col='f1_mean', ci_lower_col='f1_ci_lower', ci_upper_col='f1_ci_upper',
        shuffle_mean_col='shuffle_f1_mean', shuffle_p_col='shuffle_f1_p', label='F1 Score',
        score_fn=lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
    ),
    'balanced_accuracy': dict(
        aggr_col='acc_aggr', mean_col='acc_mean', ci_lower_col='acc_ci_lower', ci_upper_col='acc_ci_upper',
        shuffle_mean_col='shuffle_acc_mean', shuffle_p_col='shuffle_acc_p', label='Balanced Accuracy',
        score_fn=balanced_accuracy_score,
    ),
}


def _covariate_color(y_name, level):
    return modality_colors.get(y_name, {}).get(level, 'gray')


LEVEL_LABELS = {
    'cue': {0: 'C1', 1: 'C2'},
    'choice_R1': {0: 'skip', 1: 'stop'},
    'choice_R2': {0: 'skip', 1: 'stop'},
    'trial_outcome': {0: 'noR', 1: 'R'},
    'correct_R1': {0: 'wrong', 1: 'correct'},
    'correct_R2': {0: 'wrong', 1: 'correct'},
    # analysis='baseline_vs_t' rows: not a trial label - the two classes are the fixed
    # pre-event baseline window vs the sliding window at timebin t, same trials in both
    'baseline_vs_t': {0: 'pre', 1: 'at t'},
}

# Whether the R1/R2 choice was correct given that trial's cue. Cue1 (0) should stop
# at R1 and skip R2; cue2 (1) should skip R1 and stop at R2 (stop=1, skip=0 in
# choice_R1/choice_R2's own 0/1 encoding - see LEVEL_LABELS above).
CORRECTNESS_SPECS = [
    {'name': 'correct_R1', 'choice_name': 'choice_R1', 'is_correct': lambda cue, choice: cue != choice},
    {'name': 'correct_R2', 'choice_name': 'choice_R2', 'is_correct': lambda cue, choice: cue == choice},
]


def _level_label(y_name, level):
    return LEVEL_LABELS.get(y_name, {}).get(level, f'y={level}')


def _interval_x0(rows, which_interval):
    """Where an interval's t0 marker belongs within its block, in native bins.

    Read off the fit rows themselves: `timeline_decoding` stamps `n_bins_left` onto every
    row from the t0_events table's `<interval>_n_bins_left` column, so this can't drift
    from the geometry the fit actually used the way a copy of interval_specifier's numbers
    here would. Falls back to the pre-2026-08 hardcoded pair for analytics cached before
    that column existed - note the fallback is wrong for any interval retuned since (e.g.
    R1_entry_interval moved from 10 to 4), so a stale-cache figure can put t0 in the wrong
    place; recompute the SVM analytic to get the stamped value.
    """
    if 'n_bins_left' in rows.columns and rows['n_bins_left'].notna().all():
        return int(rows['n_bins_left'].iloc[0])
    return 3 if which_interval == 'cue_entry_interval' else 10


def _interval_short_label(which_interval):
    """CamelCase label for a t0 x-tick, e.g. 'cue_entry_interval' -> 'CueEntry',
    'R1_entry_interval' -> 'R1Entry' - parts that are already all-caps/alphanumeric (like
    'R1') are left as-is rather than re-capitalized."""
    parts = which_interval.replace('_interval', '').split('_')
    return ''.join(p if p.isupper() else p.capitalize() for p in parts)


def _resolve_column_specs(x_modalities, y_predicted_names):
    """Build the (x_modality, y_predicted_name) pair for each output column.
    One of the two may be a single value (broadcast across the other's columns);
    if both are lists, every combination gets its own column (multiplexed)."""
    x_list = list(x_modalities) if isinstance(x_modalities, (list, tuple)) else [x_modalities]
    y_list = list(y_predicted_names) if isinstance(y_predicted_names, (list, tuple)) else [y_predicted_names]
    if len(x_list) > 1 and len(y_list) > 1:
        return [(xm, yp) for xm in x_list for yp in y_list]
    if len(x_list) == 1:
        x_list = x_list * len(y_list)
    if len(y_list) == 1:
        y_list = y_list * len(x_list)
    return list(zip(x_list, y_list))


def _stack_by_timebin(rows_df, column):
    """rows_df: one row per timebin for a single (interval, predict_y_name, X_modality).
    Returns an (n_trials, n_timebins) array, sorted by timebin."""
    rows_df = rows_df.sort_values('timebin')
    return np.stack(rows_df[column].values).T


def _stack_covariate_true(interval_svm_results, name, which_interval, timebins):
    """(n_trials, n_timebins) y_true array for predict_y_name==name, aligned to timebins.
    None if that predict_y_name has no rows in this interval."""
    rows = interval_svm_results[(interval_svm_results.interval_name == which_interval) &
                                 (interval_svm_results.predict_y_name == name)]
    if rows.empty:
        return None
    rows = rows[rows.timebin.isin(timebins)]
    return _stack_by_timebin(rows, 'y_true')


def _bootstrap_aggr_ci(y_true, predictions, metric, n_bootstrap=1000, ci_pct=(10, 90), rng=None):
    """Percentile-bootstrap CI for the aggr/majority-vote metric itself, not the stored
    ci_lower_col/ci_upper_col (those are a bootstrap CI of the *mean-across-iterations*
    metric - a different quantity computed from re-fitting the SVM many times, see
    _bootstrap_and_aggregate in postproc_mea1k_ephys.py - not applicable to the aggr line,
    see _plot_f1_lines). Since `predictions` is already one fixed vector (the majority vote
    across bootstrap iterations) and `y_true` its matching true labels, resampling *trials*
    with replacement and recomputing the score on each resample is a standard bootstrap CI
    for that fixed decision rule's score - it captures trial-sampling uncertainty, not
    additional SVM-refitting variability (a joint CI over both would need re-fitting the
    SVM itself many times, out of scope for plotting - see get_SVMCueOutcomeChoicePred).
    NaN predictions (CCGP's untested-stratum rows, see _bootstrap_and_aggregate) are
    excluded before resampling. `metric` is 'f1' or 'balanced_accuracy' (see METRIC_SPECS) -
    computed directly from confusion-matrix counts (vectorized across all n_bootstrap
    resamples at once) rather than calling sklearn's f1_score/balanced_accuracy_score in a
    per-resample Python loop, since the labels are binary and both metrics reduce to closed
    forms - confirmed to match sklearn's own output exactly (to float precision) on random
    binary data before switching to this; a per-resample sklearn-call loop was originally
    ~2-3s per column at only 200 resamples, this is ~1000x fewer wall-clock seconds even at
    5x more resamples. Returns (nan, nan) if fewer than 4 valid trials, only one true class
    among them, or over half the bootstrap resamples happened to be single-class (recall
    undefined) - the same edge cases _bootstrap_and_aggregate itself guards against at fit
    time."""
    if rng is None:
        rng = np.random.default_rng(0)
    valid = ~np.isnan(predictions)
    y, p = y_true[valid].astype(int), predictions[valid].astype(int)
    n = len(y)
    if n < 4 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    yb, pb = y[idx], p[idx]
    tp = ((yb == 1) & (pb == 1)).sum(axis=1)
    fp = ((yb == 0) & (pb == 1)).sum(axis=1)
    fn = ((yb == 1) & (pb == 0)).sum(axis=1)
    tn = ((yb == 0) & (pb == 0)).sum(axis=1)
    with np.errstate(all='ignore'):
        recall_1 = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        recall_0 = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
        if metric == 'balanced_accuracy':
            scores = 0.5 * (recall_0 + recall_1)
        else:
            precision_1 = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            precision_0 = np.where(tn + fn > 0, tn / (tn + fn), 0.0)
            r1, r0 = np.nan_to_num(recall_1, nan=0.0), np.nan_to_num(recall_0, nan=0.0)
            f1_1 = np.where(precision_1 + r1 > 0, 2 * precision_1 * r1 / (precision_1 + r1), 0.0)
            f1_0 = np.where(precision_0 + r0 > 0, 2 * precision_0 * r0 / (precision_0 + r0), 0.0)
            scores = np.where(np.isnan(recall_0 * recall_1), np.nan, 0.5 * (f1_0 + f1_1))
    scores = scores[~np.isnan(scores)]
    if len(scores) < n_bootstrap // 2:
        return np.nan, np.nan
    return tuple(np.percentile(scores, ci_pct))


def _df_to_markdown_table(df, float_fmt="{:.3f}"):
    """Minimal, dependency-free DataFrame -> GitHub-flavored markdown table (no reliance
    on the optional `tabulate` package that df.to_markdown() needs)."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = [float_fmt.format(v) if isinstance(v, float) else str(v) for v in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def _ordered_svm_res(svm_res, which_intervals):
    """Row-order svm_res to (interval_name in which_intervals order, timebin ascending) -
    the same traversal create_subplot's offset-based x-positions and t0/boundary lines
    assume. svm_res's native row order doesn't guarantee this, so anything built directly
    off its positional row index (e.g. a heatmap using np.arange) needs this first to stay
    pixel-aligned with those lines."""
    return pd.concat([svm_res[svm_res.interval_name == iv].sort_values('timebin')
                       for iv in which_intervals], ignore_index=True)


def _w_block_with_interval_separators(ordered_svm_res, which_intervals, metric_col=None, smooth=None):
    """One block's W array (rows already in _ordered_svm_res order) with a zero row
    inserted between consecutive intervals. A zero row has zero norm, so its cosine
    similarity to anything is NaN - masked blank/white in plot_w_comparison, giving a
    clean separator without touching the real similarity values. Also returns each
    interval's t0 marker position (matching create_subplot's x_0) in this block's local
    row index, for faint dashed reference lines. Returns an empty (0, 0) block (with no
    t0 positions) if `ordered_svm_res` has no rows at all - e.g. a session where this
    target was never fit for any interval - plot_w_comparison already renders a
    mismatched-dimensionality block blank, which is exactly right for "nothing here".
    If `metric_col` is given (e.g. 'acc_aggr'/'f1_aggr' - see METRIC_SPECS), also returns
    a 1D array of that column's values, row-for-row aligned with the W block (NaN at each
    separator row) - see plot_w_comparison's min_metric for what this feeds. `smooth`, if
    given, applies the same rolling mean as _plot_f1_lines (see _smooth) to metric_col's
    values before returning them, separately per interval (never blending across an
    interval boundary, and never touching the separator row's NaN) - so min_metric masks
    against the same smoothed curve row 1 actually draws, instead of the noisier
    per-timebin raw score."""
    if ordered_svm_res.empty:
        return (np.zeros((0, 0)), [], (np.zeros(0) if metric_col is not None else None))
    n_features = len(ordered_svm_res.w.iloc[0])
    parts, metric_parts, t0_positions = [], [], []
    pos = 0
    for interval in which_intervals:
        rows = ordered_svm_res[ordered_svm_res.interval_name == interval]
        n_bins = len(rows)
        if n_bins == 0:
            continue
        if parts:
            parts.append(np.zeros((1, n_features)))
            if metric_col is not None:
                metric_parts.append(np.array([np.nan]))
            pos += 1
        parts.append(np.array(rows.w.to_list()))
        if metric_col is not None:
            metric_parts.append(_smooth(rows[metric_col].to_numpy(dtype=float), smooth))
        t0_positions.append(pos + _interval_x0(rows, interval))
        pos += n_bins
    Ws = np.concatenate(parts, axis=0)
    metric_values = np.concatenate(metric_parts, axis=0) if metric_col is not None else None
    return Ws, t0_positions, metric_values


def _split_ccgp_primary_secondary(df):
    """CCGP fits store both train/test directions as two separate, fully symmetric rows
    (train_stratum 0 and 1, same columns as a naive fit - see the `ccgp` arg of
    get_SVMCueOutcomeChoicePred), so `df` can hold two rows per (interval_name, timebin,
    X_modality, predict_y_name, conditioner) group where a naive fit only ever has one.
    Every per-timebin plotting routine (create_subplot, _plot_f1_lines,
    plot_covariate_f1_single, ...) needs exactly one row per group, so this splits `df`
    into `primary` (the more_reliable direction - the one trained on the larger stratum;
    ties, i.e. equal strata, break toward train_stratum==0) and `secondary` (the other,
    discarded direction). Naive rows (train_stratum is None) have nothing to split - they
    pass straight through as `primary`, and `secondary` is empty."""
    if 'train_stratum' not in df.columns or df['train_stratum'].isna().all():
        return df, df.iloc[0:0]

    is_ccgp = df['train_stratum'].notna()
    naive_rows, ccgp_rows = df[~is_ccgp], df[is_ccgp]

    group_cols = ['interval_name', 'timebin', 'X_modality', 'predict_y_name', 'conditioner']
    ccgp_sorted = ccgp_rows.sort_values(
        group_cols + ['more_reliable', 'train_stratum'],
        ascending=[True] * len(group_cols) + [False, True],
    )
    primary_ccgp = ccgp_sorted.drop_duplicates(subset=group_cols, keep='first')
    # whatever's left per group (nothing, if a group only has one direction fit at all)
    # is the discarded direction - no need to re-derive which one that is
    secondary_ccgp = ccgp_rows.drop(primary_ccgp.index)

    primary = pd.concat([naive_rows, primary_ccgp]).sort_index()
    return primary, secondary_ccgp


def _resolve_ccgp_w_stratum(primary, secondary, y_predicted_name, ccgp_w_train_stratum):
    """Which train_stratum is "elected" for this target - both row 2's heatmap (which
    direction's W it draws) and the small star in row 1's corner annotation (see
    _ccgp_n_annotation) key off this, so the two stay in agreement. `ccgp_w_train_stratum`,
    if given, is a {y_predicted_name: train_stratum} dict (see plot_session_analysis)
    pinning a specific trained-on conditioner subset - e.g. {'cue': 1} elects the 'cue'
    decoder's direction trained on the choice_R1==1 subset, since a CCGP row's
    train_stratum IS the level of the *other* target it conditioned on (see the `ccgp`
    arg of get_SVMCueOutcomeChoicePred). A value that isn't actually a fitted stratum for
    this target (typo, or a stratum this binary conditioner never had) raises - silently
    falling back would let the requested and displayed W quietly disagree. Falls back to
    the automatic more_reliable stratum (_split_ccgp_primary_secondary's `primary`) only
    when the dict doesn't mention this target at all. Returns None for naive fits
    (train_stratum is None throughout - nothing to elect), regardless of the dict, so a
    dict entry for a target that's naive-only in a given session is harmlessly ignored."""
    if primary.empty or 'train_stratum' not in primary.columns or primary['train_stratum'].isna().all():
        return None
    wanted = (ccgp_w_train_stratum or {}).get(y_predicted_name)
    if wanted is None:
        return primary['train_stratum'].dropna().iloc[0]
    combined = pd.concat([primary, secondary]) if secondary is not None and not secondary.empty else primary
    available = sorted(combined['train_stratum'].dropna().unique().tolist())
    if wanted not in available:
        raise ValueError(f"ccgp_w_train_stratum[{y_predicted_name!r}]={wanted!r} is not a fitted "
                          f"train_stratum for this target (available: {available})")
    return wanted


def _rows_for_stratum(primary, secondary, stratum):
    """Rows for one CCGP direction (train_stratum == stratum), pooling `primary` and
    `secondary` - see _resolve_ccgp_w_stratum, which is expected to have already
    validated `stratum` against this same pool. `stratum=None` (naive fit) returns
    `primary` unchanged."""
    if stratum is None:
        return primary
    combined = pd.concat([primary, secondary]) if secondary is not None and not secondary.empty else primary
    return combined[combined['train_stratum'] == stratum]


def _apply_ccgp_election(df, ccgp_w_train_stratum=None):
    """Collapse a raw svm_results slice (which may hold two rows per CCGP group - see
    _split_ccgp_primary_secondary) down to exactly one row per (interval_name, timebin,
    X_modality, predict_y_name, conditioner) group, resolving each distinct predict_y_name
    against its OWN entry in ccgp_w_train_stratum (see _resolve_ccgp_w_stratum) rather
    than blindly keeping whichever direction happened to be automatically more_reliable.
    Needed for get_session_data's interval_svm_results, which spans every target fit in
    that interval/X_modality (not just the current column's own target) - each of those
    other targets (e.g. a 'cue' covariate panel while the column itself decodes
    choice_R1) can have its own independent election in ccgp_w_train_stratum, or none
    (falling back to automatic per target, same as _resolve_ccgp_w_stratum's own
    fallback). A target missing entirely from ccgp_w_train_stratum, or naive-only, is
    unaffected - this only changes which CCGP direction's row is kept."""
    if 'train_stratum' not in df.columns or df['train_stratum'].isna().all():
        return df
    rows = []
    for y_name, group in df.groupby('predict_y_name', dropna=False):
        primary, secondary = _split_ccgp_primary_secondary(group)
        stratum = _resolve_ccgp_w_stratum(primary, secondary, y_name, ccgp_w_train_stratum)
        rows.append(_rows_for_stratum(primary, secondary, stratum))
    return pd.concat(rows).sort_index() if rows else df


def _selection_matches(selection, **fields):
    """Whether a highlight-selection dict matches these field values - only keys actually
    present in `selection` constrain the match (an omitted key is "don't care"), so a
    selection can pick out e.g. every timebin of a given interval by leaving `timebin`
    out. Field names: session_id, y_name, conditioner, train_stratum, interval, timebin -
    see get_session_sequence_Ws/plot_session_analysis's highlight_selections.
    `train_stratum` is what actually disambiguates a CCGP target's two directions (0 vs
    1) - `conditioner` alone only names which *other* target it's conditioned on, not
    which direction, so a selection meant to pick out one specific W unambiguously (e.g.
    when both directions of a target appear as separate blocks - see
    plot_session_analysis's single_y_name) should set `train_stratum` too, not just
    `conditioner`."""
    return all(selection.get(k) == v for k, v in fields.items() if k in selection)


def _highlighted_rows(ordered_svm_res, which_intervals, selections, **constant_fields):
    """(local_row_index, matched_row) pairs for rows in `ordered_svm_res` (already
    filtered to a single session/target/conditioner/direction block) that any of
    `selections` picks out. local_row_index is in the same 0-indexed row order
    _w_block_with_interval_separators itself builds (i.e. including the +1 it adds for
    each inserted separator row), for use as plot_w_comparison's highlight_positions
    row_index; matched_row is the actual dataframe row (a Series, carrying `w` and every
    other fitted column) for the caller to return/inspect, not just draw a box around.
    `constant_fields` (e.g. session_id=, y_name=, conditioner=) are the same for every row
    in this block and checked alongside each row's own interval/timebin/train_stratum.
    `session_id`, specifically, also gets stamped onto matched_row as a column (it isn't
    one already - ordered_svm_res comes from _ordered_svm_res, which drops the original
    session_id index level) so a caller collecting matched_row's across many
    sessions/columns (see get_session_sequence_Ws/plot_session_analysis's
    highlighted_rows) doesn't lose track of which session each one came from. Likewise,
    whichever selection dict actually matched has its own optional `name` (a short
    human label, e.g. 'shared_choice_07') stamped onto matched_row too, if it set one -
    see plot_w_comparison's highlight_positions and plot_w_projection_session for where
    that name gets drawn/titled."""
    if not selections:
        return []
    out = []
    pos = 0
    for interval in which_intervals:
        rows = ordered_svm_res[ordered_svm_res.interval_name == interval]
        n_bins = len(rows)
        if n_bins == 0:
            continue
        if pos > 0:
            pos += 1  # the separator row _w_block_with_interval_separators inserts
        for offset, (_, row) in enumerate(rows.iterrows()):
            fields = {**constant_fields, 'interval': interval, 'timebin': row.timebin,
                      'train_stratum': row.get('train_stratum')}
            matched_sel = next((sel for sel in selections if _selection_matches(sel, **fields)), None)
            if matched_sel is not None:
                row = row.copy()
                if 'session_id' in constant_fields:
                    row['session_id'] = constant_fields['session_id']
                if 'name' in matched_sel:
                    row['name'] = matched_sel['name']
                out.append((pos + offset, row))
        pos += n_bins
    return out


def get_session_data(s_id, t0_events, behavior, svm_results, x_modality, y_predicted_name, which_intervals,
                      ccgp_w_train_stratum=None):
    """All data needed to plot one (session, x_modality, y_predicted_name) column. Every
    per-timebin plotting routine below (create_subplot, _plot_f1_lines, plot_covariate_f1_single,
    ...) assumes exactly one row per (interval, timebin, X_modality, predict_y_name) - naive
    fits satisfy this directly, CCGP fits need `_split_ccgp_primary_secondary` first (see
    there - CCGP stores both train/test directions as separate rows, unlike a naive fit).
    `ccgp_w_train_stratum`, if given, only affects interval_svm_results (see below) -
    target_svm_results/target_svm_results_secondary are returned as the raw
    primary/secondary split regardless, since the caller (plot_session_analysis) resolves
    the current column's own election itself and needs both directions available."""
    target_svm_results = svm_results.loc[
        (svm_results.index.get_level_values('session_id') == s_id) &
        (svm_results['X_modality'] == x_modality) &
        (svm_results['predict_y_name'] == y_predicted_name) &
        (svm_results['interval_name'].isin(which_intervals))
    ]
    target_svm_results, target_svm_results_secondary = _split_ccgp_primary_secondary(target_svm_results)

    # Same slice, but *not* filtered to y_predicted_name. For a fixed (interval_name,
    # timebin, X_modality), every predict_y_name was fit on the exact same
    # svm_setup_dat (see timeline_decoding in postproc_mea1k_ephys.py: svm_setup_dat is
    # built once per interval_col_name, before the y_name loop) - so these sibling rows'
    # y_true is trial-order-aligned with target_svm_results' pred/y_true for the same
    # timebin. This is what lets plot_covariate_f1 condition on other covariates
    # without joining back to the behavior table.
    interval_svm_results = svm_results.loc[
        (svm_results.index.get_level_values('session_id') == s_id) &
        (svm_results['X_modality'] == x_modality) &
        (svm_results['interval_name'].isin(which_intervals))
    ]
    # _apply_ccgp_election (not a blanket _split_ccgp_primary_secondary): this table
    # spans every predict_y_name fit here, e.g. the 'cue' covariate panel while the
    # column itself decodes choice_R1 - each target gets its own election from
    # ccgp_w_train_stratum (or automatic, same fallback as elsewhere), so a covariate
    # panel doesn't silently disagree with what the caller explicitly requested for it.
    interval_svm_results = _apply_ccgp_election(interval_svm_results, ccgp_w_train_stratum)

    all_pos = {}
    print(f"\nGetting position trajectories for session {s_id}")
    for which_interval in which_intervals:
        trial_poss = []
        for trial_id, full_intrvl in t0_events.loc[s_id, which_interval].dropna().items():
            trial_behavior = behavior.xs(s_id, level='session_id')
            trial_behavior = trial_behavior[trial_behavior['trial_id'] == trial_id]
            dat = trial_behavior.loc[trial_behavior.index.overlaps(full_intrvl)]
            trial_poss.append(dat.frame_position.values)

        # keep only trials with the modal (most common) trajectory length
        lengths = pd.Series([len(x) for x in trial_poss])
        correct_length = lengths.value_counts().index[0]
        trial_pos = [pos for pos in trial_poss if len(pos) == correct_length]
        all_pos[which_interval] = np.stack(trial_pos)
        print(f"  {which_interval}: {len(trial_pos)}/{len(trial_poss)} trials at length {correct_length}")

    all_times = dat.index.mid.values - dat.index.mid[0]
    return target_svm_results, target_svm_results_secondary, interval_svm_results, all_pos, all_times


def _resample_pos(pos_1d, n_bins):
    """Resample a 1D position array to n_bins using linear interpolation."""
    if len(pos_1d) == n_bins:
        return pos_1d
    return np.interp(np.linspace(0, 1, n_bins), np.linspace(0, 1, len(pos_1d)), pos_1d)


def _smooth(values, window):
    """Centered rolling mean over `window` timebins, or `values` unchanged if window is
    falsy/<=1. min_periods=1 keeps the array's original length (no NaN edges), so this is
    safe to apply per-interval right before plotting without shrinking/misaligning the
    x-axis (timebins)."""
    if not window or window <= 1:
        return values
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().values


def _ccgp_n_annotation(conditioner, y_name, primary_rows, secondary_rows, elected_stratum=None):
    """Bottom-right corner text for a CCGP fit, replacing the naive n_pos/n_neg
    annotation: for each conditioner level (the stratum a direction trained on), the
    target's own class balance *within that stratum* - i.e. exactly what that
    direction's classifier was trained on, not the target's overall balance across both
    strata. n_train_pos/n_train_neg are constant across timebins for a given direction
    (the trial-level train/test split doesn't depend on timebin - see
    fit_CCGP_with_bootstrap), so any row's value works; `.iloc[0]` just picks the first.
    `elected_stratum` (see _resolve_ccgp_w_stratum), if given, marks that direction's
    line with a small star (★) - the direction whose W row 2's heatmap actually
    shows - so the corner text and the heatmap agree on which direction is "the" one,
    including when ccgp_w_train_stratum overrides the automatic more_reliable choice.
    The reminder that the two directions are each other's train/test swap, not
    independent fits, is drawn separately (see _plot_f1_lines) - kept out of this text so
    it doesn't crowd the bottom-right corner."""
    def _one(rows):
        stratum = rows['train_stratum'].dropna().iloc[0]
        neg = int(rows['n_train_neg'].dropna().iloc[0])
        pos = int(rows['n_train_pos'].dropna().iloc[0])
        star = ' ★' if elected_stratum is not None and stratum == elected_stratum else ''
        return (f"{_level_label(conditioner, stratum)}{star}: "
                f"$n_{{{_level_label(y_name, 0)}}}$={neg} $n_{{{_level_label(y_name, 1)}}}$={pos}")

    lines = [_one(primary_rows)]
    if secondary_rows is not None and not secondary_rows.empty:
        lines.append(_one(secondary_rows))
    return "\n".join(lines)


def _plot_f1_lines(ax, svm_res_target, which_intervals, ylabel, y_name, extra_lines=None, show_ci=False,
                    show_aggr_ci=True, show_n=True, extra_alpha=0.8, target_color='black', target_alpha=0.95,
                    target_linewidth=1, metric='f1', show_shuffle=False, smooth=None, show_ccgp_secondary=False,
                    svm_res_secondary=None, ccgp_w_elected_stratum=None):
    """Shared F1-vs-time drawing routine, used by both the main F1 panel (row 1) and
    each per-covariate panel below it. For a naive fit, draws the target's aggr metric
    (f1_aggr or acc_aggr, depending on `metric` - see METRIC_SPECS) as a bold,
    high-contrast black reference line. For a CCGP fit (only when show_ccgp_secondary is
    True and `svm_res_secondary` - the discarded direction, see get_session_data/
    _split_ccgp_primary_secondary - is present; otherwise falls back to the naive-style
    single black line using just `svm_res_target`), both train/test directions are drawn
    symmetrically instead: two same-weight lines, each colored by which conditioner level
    *it* trained on (_covariate_color), with no bold/dashed distinction between them -
    color plus the bottom-right corner annotation (see _ccgp_n_annotation, which replaces
    the naive n_pos/n_neg text for CCGP fits, and gets a small star next to whichever
    direction `ccgp_w_elected_stratum` - see _resolve_ccgp_w_stratum - names) is what
    tells them apart. A small top-left "train<->test" note is also added once, as a
    reminder that the two lines are each other's train/test swap, not independent fits.
    `extra_lines(which_interval, target_rows) -> [(values, color, label, linestyle), ...]`
    supplies whatever else should be compared against the naive line for that interval
    (e.g. the mean metric, or a covariate's per-level score) - a naive-fit-only feature,
    not applied per CCGP direction. With show_shuffle, also draws the shuffle-control null
    mean (see METRIC_SPECS' shuffle_mean_col) as a dashed black line, when that column is
    present. `smooth`, if given (e.g. 3), is a rolling-mean window applied to every line
    (and the CI band) before plotting - see _smooth - separately per interval, so
    smoothing never blends values across an interval boundary. Concatenates intervals
    left-to-right with the shared offset scheme used everywhere in this notebook (offset
    advances by each interval's timebin count, plus a 1-unit gap between intervals
    matching the heatmap's NaN separator row - see _w_block_with_interval_separators).
    show_aggr_ci (naive fits only, default True) draws a bootstrap CI band around the aggr
    line itself, computed at plot time via _bootstrap_aggr_ci from that row's stored
    y_true/predictions - independent of `show_ci` (which instead bands the *mean* line
    with the stored, precomputed ci_lower_col/ci_upper_col - a different quantity, see
    _bootstrap_aggr_ci's docstring for why the two aren't interchangeable). Returns the
    final offset."""
    spec = METRIC_SPECS[metric]
    offset = 0
    n_pos = n_neg = None
    n_annotation = None
    any_ccgp = False
    has_secondary = show_ccgp_secondary and svm_res_secondary is not None and not svm_res_secondary.empty
    for i, which_interval in enumerate(which_intervals):
        target_rows = svm_res_target[svm_res_target.interval_name == which_interval].sort_values('timebin')
        if target_rows.empty:
            continue
        timebins = target_rows.timebin.values
        first = offset == 0  # first interval actually drawn - only label it once

        is_ccgp = (has_secondary and 'conditioner' in target_rows.columns
                   and target_rows['conditioner'].notna().any())
        secondary_rows = (svm_res_secondary[svm_res_secondary.interval_name == which_interval].sort_values('timebin')
                           if is_ccgp else None)

        if is_ccgp:
            conditioner = target_rows['conditioner'].dropna().iloc[0]
            for rows in (target_rows, secondary_rows):
                if rows is None or rows.empty:
                    continue
                stratum = rows['train_stratum'].dropna().iloc[0]
                color = _covariate_color(conditioner, stratum)
                rows_timebins = rows.timebin.values
                ax.plot(rows_timebins + offset, _smooth(rows[spec['aggr_col']].values, smooth), color=color,
                        linewidth=target_linewidth, alpha=target_alpha, zorder=10,
                        label=f'train {_level_label(conditioner, stratum)}' if first else None)
                if show_ci:
                    # see naive branch below for why show_ci is only meaningful with show_mean
                    ax.fill_between(rows_timebins + offset, _smooth(rows[spec['ci_lower_col']].values, smooth),
                                     _smooth(rows[spec['ci_upper_col']].values, smooth),
                                     color=color, alpha=0.07, edgecolor=None, zorder=1)
        else:
            ax.plot(timebins + offset, _smooth(target_rows[spec['aggr_col']].values, smooth), color=target_color,
                    linewidth=target_linewidth, alpha=target_alpha, zorder=10,
                    label=spec['aggr_col'] if first else None)
            if show_ci:
                # NB: ci_lower_col/ci_upper_col are bootstrap percentiles of the per-iteration
                # scores (aligned with mean_col), not of aggr_col - drawing them around the aggr
                # line (the default target line) would be showing a CI that doesn't actually
                # belong to it. Only safe to enable together with show_mean.
                ax.fill_between(timebins + offset, _smooth(target_rows[spec['ci_lower_col']].values, smooth),
                                 _smooth(target_rows[spec['ci_upper_col']].values, smooth),
                                 color='black', alpha=0.07, edgecolor=None, zorder=1)
            if show_aggr_ci:
                # a CI that actually belongs to the aggr line - see _bootstrap_aggr_ci.
                y_true_2d = _stack_by_timebin(target_rows, 'y_true')
                pred_2d = _stack_by_timebin(target_rows, 'predictions')
                rng = np.random.default_rng(0)
                aggr_ci = np.array([_bootstrap_aggr_ci(y_true_2d[:, t], pred_2d[:, t], metric, rng=rng)
                                     for t in range(y_true_2d.shape[1])])
                ax.fill_between(timebins + offset, _smooth(aggr_ci[:, 0], smooth), _smooth(aggr_ci[:, 1], smooth),
                                 color=target_color, alpha=0.15, edgecolor=None, zorder=2)

        if show_shuffle and spec['shuffle_mean_col'] in target_rows.columns:
            ax.plot(timebins + offset, _smooth(target_rows[spec['shuffle_mean_col']].values, smooth), color='black',
                    linestyle='--', linewidth=1.2, alpha=0.65, zorder=9, label='shuffle null' if first else None)

        for values, color, label, ls in (extra_lines(which_interval, target_rows) if extra_lines else []):
            ax.plot(timebins + offset, _smooth(values, smooth), color=color, linestyle=ls, alpha=extra_alpha,
                    zorder=11, label=label if first else None)

        if is_ccgp:
            any_ccgp = True
            if n_annotation is None:
                n_annotation = _ccgp_n_annotation(conditioner, y_name, target_rows, secondary_rows,
                                                   elected_stratum=ccgp_w_elected_stratum)
        elif n_pos is None:
            n_pos = target_rows.n_positives.values[0]
            n_neg = target_rows.n_negatives.values[0]

        offset += timebins.max() + 1
        if i < len(which_intervals) - 1:
            offset += 1  # gap matching the heatmap's inserted NaN separator row

    ax.set_ylabel(ylabel)
    ax.set_ylim(.1, .9)
    # Nothing drawn at .5 itself (no chance-level axhline, no grid line there) - just the
    # tick label. Manual grid at .25/.75 only, not ax.yaxis.grid(True, ...) at every tick,
    # which would put one at .5 too.
    for gridline_y in (0.25, 0.75):
        ax.axhline(gridline_y, color='k', linestyle='--', alpha=0.5, zorder=0, linewidth=0.5)
    if show_n:
        if n_annotation is not None:
            ax.text(0.95, 0.05, n_annotation, transform=ax.transAxes, fontsize=7,
                    verticalalignment='bottom', horizontalalignment='right')
        elif n_pos is not None:
            ax.text(0.95, 0.05, f"$n_{{{_level_label(y_name, 0)}}}={n_neg}$, $n_{{{_level_label(y_name, 1)}}}={n_pos}$",
                    transform=ax.transAxes, fontsize=7, verticalalignment='bottom', horizontalalignment='right')
    if any_ccgp:
        # top-left, not bottom-right - keeps this reminder clear of the (already two-line)
        # per-stratum n_annotation there, rather than crowding both into one corner.
        ax.text(0.05, 0.95, '⇅ train↔test', transform=ax.transAxes, fontsize=6, alpha=0.7,
                verticalalignment='top', horizontalalignment='left')
    return offset


def create_subplot(axes, all_pos, svm_res, title, y_predicted_name, which_intervals, metric='f1',
                    show_shuffle=False, show_scatter=False, show_mean=False, show_aggr_ci=True,
                    draw_cue_zone=True, smooth=None, show_ylabel=True, show_title=True,
                    show_ccgp_secondary=False, svm_res_secondary=None, ccgp_w_elected_stratum=None,
                    show_trial_kinematics=True):
    """Row 0: single-trial trajectories; with show_scatter, also a per-timebin mean-position
    scatter colored by decode score (metric='f1' or 'balanced_accuracy' - see METRIC_SPECS) -
    off by default, since row 1 already shows that same per-timebin score as a proper line.
    Row 1: score over time (via _plot_f1_lines) - by default just the aggr score (verified
    against mean-across-bootstraps many times; they match closely enough that drawing both
    is redundant) as a solid black line, with show_aggr_ci (default True) drawing a CI band
    around it too - a genuine bootstrap CI of the aggr score itself, computed at plot time
    (see _bootstrap_aggr_ci), not the stored ci_lower_col/ci_upper_col (those belong to the
    *mean* score, a different quantity - see _plot_f1_lines); show_mean also draws the mean
    score as a thin gray line, with its own (correctly aligned) stored CI band; show_shuffle
    also draws the shuffle-control null mean as a dashed black line, when present. Leaves a
    1-unit gap between intervals, matching the NaN separator row the row-2 heatmap gets from
    _w_block_with_interval_separators - since each
    interval is its own plot()/scatter() call, the gap is just empty space, no dummy data
    point needed. draw_cue_zone controls row 0's cue-zone illustration - see the call site
    in plot_session_analysis for why it's False up to S7 and drawn as alternating
    per-cue-color hatching from S8 on. `smooth`, if given (e.g. 3), is a rolling-mean window
    applied to every row-1 line before plotting - see _plot_f1_lines/_smooth; row 0's
    scatter (off by default) stays per-timebin/unsmoothed, since each point is its own
    color rather than part of a line. show_ylabel=False hides row 1's y-axis label and tick
    labels (row 0/2 and the covariate panels already hide theirs unconditionally) - see
    plot_session_analysis for why. show_title=False hides row 0's own
    "{y_predicted_name} decoding from {x_modality}" title. Returns (scatter, xticks,
    xtick_labels): xtick_labels
    pairs with xticks for the row-2 heatmap's t0 x-ticks, one $t_{IntervalName}$/'1s' pair
    per interval (see _interval_short_label) - applied by the caller alongside n_skip/
    n_stop-style formatting (mathtext subscript, fontsize=7) for a consistent look.
    show_ccgp_secondary passes through to _plot_f1_lines (along with svm_res_secondary,
    `svm_res`'s discarded CCGP direction - see get_session_data/_split_ccgp_primary_secondary
    - and ccgp_w_elected_stratum, see _resolve_ccgp_w_stratum, for the corner
    annotation's star) - see there for what it draws on CCGP fits (a no-op on naive
    fits). show_trial_kinematics=False hides row 0 entirely (single-trial position
    traces, bin-boundary/t0 lines, scatter, cue/R1/R2 zone illustration, fixed ylim) -
    the Axes itself is hidden (set_visible(False)), still occupying its grid row rather
    than reflowing rows 1/2 upward. offset/xticks/xtick_labels are still computed as
    usual either way, since rows 1/2's own t0 lines and the heatmap's xticks depend on
    them regardless of whether row 0 is shown."""
    spec = METRIC_SPECS[metric]
    scatter = None
    offset = 0
    xticks = []
    xtick_labels = []
    pos_min, pos_max = np.inf, -np.inf
    for i, which_interval in enumerate(which_intervals):
        this_interval_pos = all_pos[which_interval]
        # sort_values: rows must be timebin-ascending for the offset x-axis below,
        # and to align row-for-row with sibling predict_y_name rows in plot_covariate_f1
        svm_res_interval = svm_res[svm_res.interval_name == which_interval].sort_values('timebin')
        n_bins = len(svm_res_interval.timebin)

        if show_trial_kinematics:
            for pos in this_interval_pos:
                pos_inner = _resample_pos(pos[1:-1], n_bins)
                pos_min, pos_max = min(pos_min, pos_inner.min()), max(pos_max, pos_inner.max())
                axes[0].plot(svm_res_interval.timebin + offset, pos_inner, alpha=0.05, color='k', zorder=4)

            # Plot bin boundaries. Solid, not dashed: with one of these per timebin (~30 densely
            # packed, near-adjacent vertical lines per interval), a dashed style's on/off pattern
            # lines up across all of them and produces a Moire effect that reads as fake
            # horizontal white banding through the R1/R2/cue zone fills below.
            for t in svm_res_interval.timebin:
                axes[0].axvline(x=t-.5 + offset, color='k', linestyle='-', alpha=0.04)

        x_0 = _interval_x0(svm_res_interval, which_interval) + offset
        xticks.extend([x_0, x_0+25])
        xtick_labels.extend([f'$t_{{{_interval_short_label(which_interval)}}}$', '1s'])
        if show_trial_kinematics:
            axes[0].axvline(x=x_0, color='k', linestyle='--', alpha=0.4)
        axes[1].axvline(x=x_0, color='k', linestyle='--', alpha=0.4)
        axes[2].axvline(x=x_0, color='k', linestyle='--', alpha=0.4)
        axes[2].axhline(y=x_0, color='k', linestyle='--', alpha=0.4)

        if show_trial_kinematics and show_scatter:
            mean_pos = _resample_pos(this_interval_pos.mean(axis=0)[1:-1], n_bins)
            scatter = axes[0].scatter(svm_res_interval.timebin + offset, mean_pos,
                                c=svm_res_interval[spec['mean_col']].values,
                                cmap=plt.cm.viridis,
                                norm=Normalize(vmin=0.2, vmax=0.8),
                                alpha=1,
                                s=30,
                                zorder=5)

        offset += svm_res_interval.timebin.max() + 1
        if i < len(which_intervals) - 1:
            offset += 1  # gap matching the heatmap's inserted NaN separator row

    extra_lines = (lambda which_interval, rows: [(rows[spec['mean_col']].values, 'gray', spec['mean_col'], '-')]
                   ) if show_mean else None
    _plot_f1_lines(
        axes[1], svm_res, which_intervals,
        ylabel=f'Decode {y_predicted_name} {spec["label"]}' if show_ylabel else '', y_name=y_predicted_name,
        extra_lines=extra_lines,
        # show_ci tied to show_mean, not a separate flag: the stored CI is a bootstrap
        # percentile band around mean_col, not aggr_col (see _plot_f1_lines), so drawing it
        # around the default aggr-only line would show a CI that doesn't belong to it.
        # show_aggr_ci is independent - it's a CI actually computed for aggr_col itself.
        show_ci=show_mean, show_aggr_ci=show_aggr_ci, target_color='black', target_alpha=0.75, target_linewidth=2,
        metric=metric, show_shuffle=show_shuffle, smooth=smooth, show_ccgp_secondary=show_ccgp_secondary,
        svm_res_secondary=svm_res_secondary, ccgp_w_elected_stratum=ccgp_w_elected_stratum,
    )
    if not show_ylabel:
        axes[1].tick_params(labelleft=False)

    # rectangle for cue/R1/R2 zones. draw_cue_zone is False up to and including S7 (see
    # plot_session_analysis); from S8 on, either cue can appear on a given trial (decided
    # per-trial, not something this generic background illustration can know) - drawn as
    # several alternating-color horizontal sub-bands spanning the zone's full width, one
    # per cue's own color, all hatched in the *same* direction: no crossing lines, and no
    # overlap so no alpha-blending into a muddy red/brown either (earlier attempts:
    # oppositely-angled overlapping hatches had both problems; a single left/right split
    # avoided them but, varying along the x/timebin axis, read as if the color meant
    # something about time rather than being a decorative alternating texture - alternating
    # along y instead doesn't have that problem, since row 0's y-axis is position, and no
    # actual data varies systematically across these bands).
    # facecolor='none'/linewidth=0: only the hatch lines themselves are colored - see the
    # global hatch.linewidth rcParam near the top of this file for the line thickness.
    if show_trial_kinematics:
        if draw_cue_zone:
            cue_colors = modality_colors['cue']
            n_cue_stripes = 8
            stripe_height = 80 / n_cue_stripes
            for stripe_i in range(n_cue_stripes):
                stripe_y0 = -80 + stripe_i * stripe_height
                level = stripe_i % 2
                axes[0].fill_betweenx(y=[stripe_y0, stripe_y0 + stripe_height], x1=0, x2=offset, facecolor='none',
                                      edgecolor=cue_colors[level], hatch='///', alpha=0.35, linewidth=0,
                                      label=f'Cue Zone ({_level_label("cue", level)})' if stripe_i < 2 else None)
        axes[0].fill_betweenx(y=[50, 130], x1=0, x2=offset, color='darkgray', alpha=0.6, label='R1 Zone')
        axes[0].fill_betweenx(y=[170, 250], x1=0, x2=offset, color='lightgray', alpha=0.6, label='R2 Zone')

        # Fixed track-position range, always - previously autoscaled per session, which made
        # row 0 subtly different heights/scales session to session (most visible comparing a
        # draw_cue_zone=False session, with nothing drawn below y=50, against a
        # draw_cue_zone=True one reaching down to the cue zone's -80). Anchored to match what
        # a cued session already shows end to end (cue zone bottom at -80 up to R2 zone top at
        # 250), plus a small fixed margin on each side - but widened (never narrowed) to the
        # actual plotted position data's own min/max when a session's real trajectories run
        # past that baseline (e.g. continuing past the R2 zone before the trial ends), so
        # real data is never silently clipped by the fixed baseline.
        ylim_lo = min(-95, pos_min - 15) if np.isfinite(pos_min) else -95
        ylim_hi = max(265, pos_max + 15) if np.isfinite(pos_max) else 265
        axes[0].set_ylim(ylim_lo, ylim_hi)

        if show_title:
            axes[0].set_title(title, fontsize=11)
        axes[0].tick_params(labelleft=False, labelbottom=False)
    else:
        axes[0].set_visible(False)

    return scatter, xticks, xtick_labels


def plot_covariate_f1_single(ax, interval_svm_results, y_predicted_name, covariate, which_intervals, metric='f1',
                              smooth=None, show_shuffle=False):
    """One small panel: the target's aggr metric baseline (f1_aggr or acc_aggr - see
    METRIC_SPECS) vs. that metric recomputed separately for each level of `covariate`
    (a sibling predict_y_name fit on the same trials - see get_session_data for why this
    alignment is safe without a trial_id join). A big gap between the two lines means
    decoding y_predicted_name is confounded with covariate rather than generalizing across
    it - this generalizes the old hardcoded cue-R1 split to run systematically over every
    other fitted label."""
    spec = METRIC_SPECS[metric]
    svm_res_target = interval_svm_results[interval_svm_results.predict_y_name == y_predicted_name]

    def extra_lines(which_interval, target_rows):
        timebins = target_rows.timebin.values
        cov_true = _stack_covariate_true(interval_svm_results, covariate, which_interval, timebins)
        if cov_true is None:
            return []
        pred = _stack_by_timebin(target_rows, 'predictions')
        true_y = _stack_by_timebin(target_rows, 'y_true')
        if cov_true.shape[0] != true_y.shape[0]:
            # print(f"plot_covariate_f1: skipping {covariate} in {which_interval}, "
            #       f"trial count mismatch ({cov_true.shape[0]} vs {true_y.shape[0]})")
            return []

        lines = []
        for level in (0, 1):
            scores = []
            n_too_few, n_single_class = 0, 0
            for t_i in range(len(timebins)):
                # ~np.isnan(pred): CCGP rows' `predictions` column has a guaranteed NaN
                # for every trial in the direction's untested (training) stratum - see
                # _bootstrap_and_aggregate's fixed-test-set sampler in
                # get_SVMCueOutcomeChoicePred - not an occasional gap like naive fits'
                # rare never-OOB trial, so this must be excluded before scoring or
                # sklearn raises on the NaN.
                mask = (cov_true[:, t_i] == level) & ~np.isnan(pred[:, t_i])
                if mask.sum() < 2:
                    n_too_few += 1
                    scores.append(np.nan)
                    continue
                if len(np.unique(true_y[mask, t_i])) < 2:
                    n_single_class += 1
                    scores.append(np.nan)
                    continue
                scores.append(spec['score_fn'](true_y[mask, t_i], pred[mask, t_i]))
            # n_valid = sum(not np.isnan(v) for v in scores)
            # print(f"plot_covariate_f1 debug: {covariate}={level} @ {which_interval}: "
            #       f"{n_valid}/{len(timebins)} timebins valid, too_few_trials={n_too_few}, "
            #       f"target_single_class={n_single_class}, total_trials_at_level={int((cov_true == level).sum())}")
            lines.append((scores, _covariate_color(covariate, level), _level_label(covariate, level), '-'))
        return lines

    _plot_f1_lines(ax, svm_res_target, which_intervals, ylabel='', y_name=y_predicted_name,
                   extra_lines=extra_lines, extra_alpha=0.4, show_n=False, metric=metric, smooth=smooth,
                   show_shuffle=show_shuffle)

    cov_rows = interval_svm_results[interval_svm_results.predict_y_name == covariate]
    n_neg = n_pos = None
    for which_interval in which_intervals:
        rows = cov_rows[cov_rows.interval_name == which_interval]
        if not rows.empty:
            n_neg, n_pos = rows.n_negatives.values[0], rows.n_positives.values[0]
            # break
    corner_text = covariate
    if n_pos is not None:
        corner_text += f"\n$n_{{{_level_label(covariate, 0)}}}={n_neg}$, $n_{{{_level_label(covariate, 1)}}}={n_pos}$"
    ax.text(0.95, 0.05, corner_text, transform=ax.transAxes,
            fontsize=7, verticalalignment='bottom', horizontalalignment='right')

    ax.tick_params(labelleft=False)
    [sp.set_visible(False) for sp in ax.spines.values()]


def plot_correctness_f1_single(ax, interval_svm_results, y_predicted_name, spec, which_intervals, metric='f1',
                                smooth=None, show_shuffle=False):
    """One small panel: the target's aggr metric baseline (f1_aggr or acc_aggr - see
    METRIC_SPECS) vs. that metric split by whether the R1/R2 choice was correct for that
    trial's cue (see CORRECTNESS_SPECS for the rule). Colored red/green like trial_outcome,
    since it's the same right/wrong distinction."""
    metric_spec = METRIC_SPECS[metric]
    svm_res_target = interval_svm_results[interval_svm_results.predict_y_name == y_predicted_name]
    name, choice_name, is_correct_fn = spec['name'], spec['choice_name'], spec['is_correct']

    def extra_lines(which_interval, target_rows):
        timebins = target_rows.timebin.values
        cue_true = _stack_covariate_true(interval_svm_results, 'cue', which_interval, timebins)
        choice_true = _stack_covariate_true(interval_svm_results, choice_name, which_interval, timebins)
        if cue_true is None or choice_true is None:
            return []
        pred = _stack_by_timebin(target_rows, 'predictions')
        true_y = _stack_by_timebin(target_rows, 'y_true')
        if cue_true.shape[0] != true_y.shape[0] or choice_true.shape[0] != true_y.shape[0]:
            # print(f"plot_correctness_f1: skipping {name} in {which_interval}, trial count mismatch "
            #       f"(cue={cue_true.shape[0]}, {choice_name}={choice_true.shape[0]}, target={true_y.shape[0]})")
            return []
        is_correct = is_correct_fn(cue_true, choice_true)

        lines = []
        for level in (0, 1):  # 0 = wrong, 1 = correct
            scores = []
            for t_i in range(len(timebins)):
                # see the matching note in plot_covariate_f1_single - CCGP rows carry a
                # guaranteed NaN in `predictions` for their untested stratum
                mask = (is_correct[:, t_i] == bool(level)) & ~np.isnan(pred[:, t_i])
                if mask.sum() < 2 or len(np.unique(true_y[mask, t_i])) < 2:
                    scores.append(np.nan)
                    continue
                scores.append(metric_spec['score_fn'](true_y[mask, t_i], pred[mask, t_i]))
            lines.append((scores, _covariate_color(name, level), _level_label(name, level), '-'))
        return lines

    _plot_f1_lines(ax, svm_res_target, which_intervals, ylabel='', y_name=y_predicted_name,
                   extra_lines=extra_lines, extra_alpha=0.4, show_n=False, metric=metric, smooth=smooth,
                   show_shuffle=show_shuffle)

    n_wrong = n_correct = None
    for which_interval in which_intervals:
        cue_rows = interval_svm_results[(interval_svm_results.interval_name == which_interval) &
                                         (interval_svm_results.predict_y_name == 'cue')]
        if cue_rows.empty:
            continue
        timebins = cue_rows.timebin.values
        cue_true = _stack_covariate_true(interval_svm_results, 'cue', which_interval, timebins)
        choice_true = _stack_covariate_true(interval_svm_results, choice_name, which_interval, timebins)
        if cue_true is None or choice_true is None or cue_true.shape[0] != choice_true.shape[0]:
            continue
        is_correct = is_correct_fn(cue_true, choice_true)[:, 0]
        n_correct, n_wrong = int(is_correct.sum()), int((~is_correct).sum())
        break
    corner_text = name
    if n_correct is not None:
        corner_text += f"\n$n_{{{_level_label(name, 0)}}}={n_wrong}$, $n_{{{_level_label(name, 1)}}}={n_correct}$"
    ax.text(0.95, 0.05, corner_text, transform=ax.transAxes,
            fontsize=7, verticalalignment='bottom', horizontalalignment='right')

    ax.tick_params(labelleft=False)
    [sp.set_visible(False) for sp in ax.spines.values()]


def _covariate_specs_for_target(y_predicted_name, fitted_names):
    """Which sibling covariates/correctness pairs can show a real split for this target:
    every ALL_Y_PREDICTED_NAMES entry (other than the target itself) that was actually
    fit as its own predict_y_name in this run, plus whichever CORRECTNESS_SPECS have both
    halves (cue and the matching choice) fit. A covariate that was never separately fit
    (e.g. trial_outcome, if the analytic's y_names list omits it) has no y_true anywhere
    for _stack_covariate_true to find, so defaulting to the fixed ALL_Y_PREDICTED_NAMES
    list regardless of what was actually fit would just draw an empty panel."""
    covariate_names = [y for y in ALL_Y_PREDICTED_NAMES if y != y_predicted_name and y in fitted_names]
    correctness_specs = [spec for spec in CORRECTNESS_SPECS
                          if 'cue' in fitted_names and spec['choice_name'] in fitted_names]
    return covariate_names, correctness_specs


def plot_covariate_f1(axes, interval_svm_results, y_predicted_name, which_intervals, covariate_names=None,
                       correctness_specs=None, metric='f1', smooth=None, show_shuffle=False):
    """One small subplot per covariate - axes is a sequence of Axes, one per entry in
    covariate_names (defaults to every ALL_Y_PREDICTED_NAMES entry, other than
    y_predicted_name itself, that was actually fit for this session - see
    _covariate_specs_for_target), plus one trailing panel per correctness_specs entry
    (defaults to whichever of CORRECTNESS_SPECS is fittable; see plot_correctness_f1_single).
    Any axes beyond that are turned off, e.g. when nrows was sized for a different column's
    (higher) covariate count."""
    if covariate_names is None:
        fitted_names = set(interval_svm_results.predict_y_name.unique())
        covariate_names, correctness_specs = _covariate_specs_for_target(y_predicted_name, fitted_names)
    elif correctness_specs is None:
        correctness_specs = CORRECTNESS_SPECS
    for ax, covariate in zip(axes, covariate_names):
        plot_covariate_f1_single(ax, interval_svm_results, y_predicted_name, covariate, which_intervals, metric,
                                  smooth, show_shuffle)
    for ax, spec in zip(axes[len(covariate_names):], correctness_specs):
        plot_correctness_f1_single(ax, interval_svm_results, y_predicted_name, spec, which_intervals, metric, smooth,
                                    show_shuffle)
    for ax in axes[len(covariate_names) + len(correctness_specs):]:
        ax.axis('off')


def _angle_alpha_mask(angles, lower, upper, mask_chance_band=True, metrics_i=None, metrics_j=None, min_metric=None):
    """Alpha mask for an angle-comparison heatmap - shared by plot_w_comparison's
    cross-block heatmap and plot_session_analysis's own per-column self-similarity
    heatmap, which used to each hand-roll this same computation (easy to edit one and
    assume it affected both). `mask_chance_band=True` (the default) dims (alpha=0.3) any
    cell whose angle falls inside [lower, upper] - the ~95% chance band a random pair of
    vectors in this X_modality's dimensionality would produce (see angle_2std_thr) -
    leaving only angles more extreme than chance at full opacity. Pass
    mask_chance_band=False to disable this entirely and show every comparable cell at
    full opacity regardless of its angle. Independently of that toggle, when min_metric
    is given along with both metrics_i/metrics_j (1D arrays, row-for-row aligned with
    `angles`' two axes - the same array on both sides for a self-comparison), also dims
    any cell where either side's own decoder scored below min_metric. NaN cells (interval
    separators, mismatched dims, a collapsed all-zero w) are the caller's own
    responsibility (cmap.set_bad + masked_invalid) - not touched here."""
    if mask_chance_band:
        alpha = ((angles > upper) | (angles < lower)).astype(float)
        alpha[alpha == 0] = .3
    else:
        alpha = np.ones_like(angles, dtype=float)
    if min_metric is not None and metrics_i is not None and metrics_j is not None:
        low_i = metrics_i < min_metric
        low_j = metrics_j < min_metric
        alpha[low_i[:, np.newaxis] | low_j[np.newaxis, :]] = .3
    return alpha


def _write_wcomp_markdown(md_path, title, block_labels, block_modalities, block_Ws, full, starts, boundaries,
                           block_metrics=None, min_metric=None, mask_chance_band=True):
    """LLM-friendly description of a plot_w_comparison figure: not a summary (that
    collapses away exactly the outliers this plot exists to surface) but a sparse list of
    every individual angle that isn't alpha-masked in the plot - i.e., when
    mask_chance_band=True (the default, matching the plot's own default), every pair whose
    weight vectors are more aligned or anti-aligned than the session's random-vector 2-std
    band predicts by chance (angle_2std_thr); with mask_chance_band=False, every
    comparable pair regardless of angle - AND (if block_metrics/min_metric are given - see
    plot_w_comparison) whose own decoders both scored at least min_metric. Each unordered
    pair is listed once; a row against itself (always 0deg) is omitted as trivial."""
    lines = [
        f"# {title}",
        "",
        "Cosine-angle similarity (degrees) between SVM decoder weight vectors. 0deg = "
        "identical direction, 90deg = orthogonal, 180deg = opposite direction. A pair is "
        "comparable only if both blocks share the same feature dimensionality (same "
        "X_modality) - incomparable pairs are omitted. "
        + ("The list below is exactly the non-alpha-masked (full-opacity) angles from the "
           "plot: pairs whose angle falls outside the session's expected chance range for "
           "random vectors of this dimensionality, i.e. more aligned/anti-aligned than "
           "chance would predict" if mask_chance_band else
           "Chance-band masking is disabled for this plot, so the list below includes "
           "every comparable pair regardless of angle")
        + (f", restricted to timebins whose own decoder scored >= min_metric={min_metric}"
           if min_metric is not None else "") + ".",
        "",
        "## Blocks",
        "",
    ]
    for i, label in enumerate(block_labels):
        lines.append(f"- Block {i}: {label.replace(chr(10), ' / ')} "
                      f"(X_modality={block_modalities[i]}, n_rows={block_Ws[i].shape[0]})")
    header = "## Significant angles (not alpha-masked)" if mask_chance_band else "## All comparable angles"
    lines += ["", header, ""]

    rows = []
    n_blocks = len(block_Ws)
    for i in range(n_blocks):
        lower, upper = angle_2std_thr[block_modalities[i]]
        for j in range(i, n_blocks):
            if block_Ws[i].shape[1] != block_Ws[j].shape[1]:
                continue
            block = full[starts[i]:boundaries[i], starts[j]:boundaries[j]]
            if mask_chance_band:
                local_i, local_j = np.where((block > upper) | (block < lower))
            else:
                local_i, local_j = np.where(~np.isnan(block))
            for li, lj in zip(local_i, local_j):
                gi, gj = starts[i] + li, starts[j] + lj
                if gi >= gj:  # dedupe the symmetric matrix and drop the trivial self-diagonal
                    continue
                if (min_metric is not None and block_metrics is not None
                        and block_metrics[i] is not None and block_metrics[j] is not None
                        and (block_metrics[i][li] < min_metric or block_metrics[j][lj] < min_metric)):
                    continue
                rows.append({
                    'block_i': i, 'row_i': int(li),
                    'block_j': j, 'row_j': int(lj),
                    'label_i': block_labels[i].replace('\n', ' / '),
                    'label_j': block_labels[j].replace('\n', ' / '),
                    'angle_deg': float(block[li, lj]),
                })

    if rows:
        summary_df = pd.DataFrame(rows)
        summary_df['deviation_from_90'] = (summary_df['angle_deg'] - 90).abs()
        summary_df = summary_df.sort_values('deviation_from_90', ascending=False).drop(
            columns='deviation_from_90').reset_index(drop=True)
        lines.append(_df_to_markdown_table(summary_df))
    else:
        lines.append("(none)")

    with open(md_path, 'w') as f:
        f.write("\n".join(lines))


def plot_w_comparison(block_Ws, block_labels, block_modalities, title, fig_width,
                       block_t0_positions=None, save_path=None, save_path_svg=None, markdown_path=None,
                       block_metrics=None, min_metric=None, highlight_positions=None,
                       reference_block=None, as_column=False, mask_chance_band=True):
    """Generic cross-block SVM-weight comparison: cosine-similarity (as an angle in
    degrees) between every pair of SVM weight vectors across a sequence of blocks,
    concatenated into one big heatmap. A block can be anything with its own set of W's -
    a session figure's column (see plot_session_analysis) or a session in a sequence (see
    plot_session_sequence_w_comparison). Blocks with mismatched feature dimensionality
    (different X_modality) can't be compared and are left blank/white - a block's own
    interval separators (zero rows - see _w_block_with_interval_separators) render blank
    the same way. Dotted lines mark where one block's set of W's ends and the next
    begins; if block_t0_positions is given (one list of local row-indices per block -
    see _w_block_with_interval_separators), faint dashed lines mark each interval's t0
    within its block. Alpha-masks each comparable block the same way the main plot's
    per-column heatmap does (angle_2std_thr). `block_metrics`/`min_metric`, if both given
    (one 1D array per block, row-for-row aligned with that block's W's - see
    _w_block_with_interval_separators's metric_col - and a scalar threshold on that same
    metric, e.g. 0.55 for 'balanced_accuracy'), extend that alpha mask: any cell where
    *either* row's own decoder scored below min_metric is forced to low alpha regardless
    of its angle - a poorly-fit decoder's weight direction isn't a comparison worth
    trusting just because it happens to land outside the chance band. `highlight_positions`,
    if given, is a list of (block_index, row_index, name) triples - row_index is local to
    that block, the same 0-indexed row order _w_block_with_interval_separators gave it (i.e.
    including its own separator-row offsets) - each drawn as a thin dotted green box
    around that row's full-width band, so a specific decoder/timebin of interest is easy
    to spot in a large comparison. `name` (may be None/falsy - see _highlighted_rows'
    optional stamping of a selection's own `name` key) is drawn as a small purple label
    just outside the circle, angled along the diagonal so it doesn't sit on top of
    neighboring rows/columns; drawn with clip_on=False so a long name isn't truncated -
    it just extends past the axes (and, for a highlight near the edge, potentially past
    the figure bounds too - shorten the name or nudge fontsize down if that happens in
    practice). Callers don't usually build these by hand - see
    get_session_sequence_Ws's highlight_selections for resolving them from
    session_id/y_name/conditioner/interval/timebin selections. save_path/
    save_path_svg save a PNG/SVG copy respectively (either, both, or neither). If
    markdown_path is given, also writes an LLM-friendly per-block-pair summary table (see
    _write_wcomp_markdown) - skipped when `reference_block` is given (that export assumes
    the full square matrix).

    `reference_block`, if given (a 0-based index into block_Ws/block_labels), restricts
    the heatmap to just that one block's own comparisons against every block (including
    itself) - e.g. "session 7 vs every other session" - instead of the full all-to-all
    matrix. The values are identical either way (cosine similarity is symmetric); this
    only picks which single block's own timebins get an axis to themselves (labeled with
    that block's own name via set_ylabel/set_xlabel), with every block's usual
    center/label/boundary/t0 markings along the other axis. Drawn as a wide horizontal
    strip by default (reference block's timebins on y, every block concatenated along x)
    - pass `as_column=True` to transpose that into a tall vertical strip instead (same
    values, different orientation). `highlight_positions` entries for a block other than
    `reference_block` are silently skipped, since that block's own row/column isn't part
    of this restricted view. Both params are ignored when `reference_block` is None.

    `mask_chance_band=True` (default) dims the ~95%-chance-band cells to low alpha (see
    _angle_alpha_mask/angle_2std_thr) - pass False to show every comparable cell at full
    opacity regardless of its angle, independent of the (also optional) min_metric mask."""
    sizes = [Ws.shape[0] for Ws in block_Ws]
    boundaries = np.cumsum(sizes)
    starts = np.concatenate([[0], boundaries[:-1]])
    total = boundaries[-1]

    row_blocks = [reference_block] if reference_block is not None else list(range(len(block_Ws)))
    n_rows = sizes[reference_block] if reference_block is not None else total
    full = np.full((n_rows, total), np.nan)
    alpha_full = np.ones((n_rows, total))
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in row_blocks:
            Ws_i = block_Ws[i]
            row0 = 0 if reference_block is not None else starts[i]
            row1 = row0 + sizes[i]
            for j, Ws_j in enumerate(block_Ws):
                if Ws_i.shape[1] != Ws_j.shape[1]:
                    continue
                cos_sim = np.dot(Ws_i, Ws_j.T) / (np.linalg.norm(Ws_i, axis=1)[:, np.newaxis] *
                                                   np.linalg.norm(Ws_j, axis=1)[np.newaxis, :])
                angles = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * (180 / np.pi)
                full[row0:row1, starts[j]:boundaries[j]] = angles

                lower, upper = angle_2std_thr[block_modalities[i]]
                metrics_i = block_metrics[i] if block_metrics is not None else None
                metrics_j = block_metrics[j] if block_metrics is not None else None
                block_alpha = _angle_alpha_mask(angles, lower, upper, mask_chance_band=mask_chance_band,
                                                 metrics_i=metrics_i, metrics_j=metrics_j, min_metric=min_metric)
                alpha_full[row0:row1, starts[j]:boundaries[j]] = block_alpha

    # blank cells (interval separators, mismatched-dim blocks) render fully opaque white,
    # regardless of the significance dimming above - otherwise they blend into the pale
    # dimmed background instead of standing out as clean separators
    alpha_full[np.isnan(full)] = 1.0

    if markdown_path is not None and reference_block is None:
        _write_wcomp_markdown(markdown_path, title, block_labels, block_modalities, block_Ws,
                               full, starts, boundaries, block_metrics=block_metrics, min_metric=min_metric,
                               mask_chance_band=mask_chance_band)

    # block_axis_is_x: which screen axis carries every block's own centers/labels/
    # boundaries/t0-lines - always x in the full matrix (both axes do); x by default in
    # reference mode too (the horizontal-strip layout), flipped to y by as_column.
    block_axis_is_x = not (reference_block is not None and as_column)
    if reference_block is not None and as_column:
        full, alpha_full = full.T, alpha_full.T
    n_y, n_x = full.shape

    if reference_block is not None:
        short_frac = sizes[reference_block] / total
        fig_dims = ((max(2.5, fig_width * short_frac), fig_width) if as_column
                    else (fig_width, max(2.5, fig_width * short_frac)))
    else:
        fig_dims = (fig_width, fig_width)

    fig, ax = plt.subplots(figsize=fig_dims, dpi=150)
    cmap = plt.cm.bwr_r.with_extremes(bad='white')
    X, Y = np.meshgrid(np.arange(n_x), np.arange(n_y))
    im = ax.pcolormesh(X, Y, np.ma.masked_invalid(full), alpha=alpha_full, cmap=cmap, vmin=0, vmax=180)

    if block_t0_positions is not None:
        if reference_block is None:
            for start, t0_list in zip(starts, block_t0_positions):
                for t0 in t0_list:
                    ax.axvline(x=start + t0, color='black', linestyle='--', alpha=0.15, linewidth=0.8)
                    ax.axhline(y=start + t0, color='black', linestyle='--', alpha=0.15, linewidth=0.8)
        else:
            for start, t0_list in zip(starts, block_t0_positions):
                for t0 in t0_list:
                    pos = start + t0
                    if block_axis_is_x:
                        ax.axvline(x=pos, color='black', linestyle='--', alpha=0.15, linewidth=0.8)
                    else:
                        ax.axhline(y=pos, color='black', linestyle='--', alpha=0.15, linewidth=0.8)
            for t0 in block_t0_positions[reference_block]:
                if block_axis_is_x:
                    ax.axhline(y=t0, color='black', linestyle='--', alpha=0.15, linewidth=0.8)
                else:
                    ax.axvline(x=t0, color='black', linestyle='--', alpha=0.15, linewidth=0.8)

    if reference_block is None:
        for b in boundaries[:-1]:
            ax.axhline(y=b - 0.5, color='black', linestyle=':', linewidth=1.2)
            ax.axvline(x=b - 0.5, color='black', linestyle=':', linewidth=1.2)
    else:
        for b in boundaries[:-1]:
            if block_axis_is_x:
                ax.axvline(x=b - 0.5, color='black', linestyle=':', linewidth=1.2)
            else:
                ax.axhline(y=b - 0.5, color='black', linestyle=':', linewidth=1.2)

    if highlight_positions:
        # Full-width row band, not a single cell - a specific decoder/timebin's own
        # comparisons against everything else are what's of interest (the matrix is
        # symmetric, so row g and column g carry the same info, hence row-only suffices).
        # Sized to exactly one cell's own [-0.5, +0.5] extent - alpha (not an outward pad)
        # is what keeps the heatmap color underneath visible through the thick line. The
        # circle marks (g, g), the on-diagonal cross-point the row band used to form
        # together with a now-removed matching column band. In reference-block mode the
        # band still spans the block axis's full extent, but the circle sits at
        # (this block's global column position, this row's local position) instead of on
        # a true diagonal, since only one block's own rows are shown here.
        box_kwargs = dict(fill=False, edgecolor='purple', alpha=0.15, linewidth=1, zorder=20)
        circle_kwargs = dict(fill=False, edgecolor='purple', alpha=0.5, linewidth=1.5, zorder=20)
        for block_index, row_index, name in highlight_positions:
            if reference_block is not None and block_index != reference_block:
                continue
            g_global = starts[block_index] + row_index
            if reference_block is None:
                x_pos = y_pos = g_global
                rect = Rectangle((-0.5, g_global - 0.5), total, 1, **box_kwargs)
            elif block_axis_is_x:
                x_pos, y_pos = g_global, row_index
                rect = Rectangle((-0.5, row_index - 0.5), total, 1, **box_kwargs)
            else:
                x_pos, y_pos = row_index, g_global
                rect = Rectangle((row_index - 0.5, -0.5), 1, total, **box_kwargs)
            ax.add_patch(rect)
            ax.add_patch(Circle((x_pos, y_pos), radius=0.5 * np.sqrt(2), **circle_kwargs))
            if name:
                ax.text(x_pos + 1, y_pos + 1, name, fontsize=6, color='purple', alpha=0.9, rotation=25,
                        ha='left', va='bottom', clip_on=False, zorder=21)

    centers = starts + np.array(sizes) / 2
    if reference_block is None:
        ax.set_xticks(centers)
        ax.set_xticklabels(block_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(centers)
        ax.set_yticklabels(block_labels, fontsize=8)
        ax.set_xlim(-0.5, total - 0.5)
        ax.set_ylim(-0.5, total - 0.5)
    elif block_axis_is_x:
        ax.set_xticks(centers)
        ax.set_xticklabels(block_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks([])
        ax.set_ylabel(block_labels[reference_block], fontsize=8)
        ax.set_xlim(-0.5, total - 0.5)
        ax.set_ylim(-0.5, n_y - 0.5)
    else:
        ax.set_yticks(centers)
        ax.set_yticklabels(block_labels, fontsize=8)
        ax.set_xticks([])
        ax.set_xlabel(block_labels[reference_block], fontsize=8)
        ax.set_ylim(-0.5, total - 0.5)
        ax.set_xlim(-0.5, n_x - 0.5)

    # enforce square pixels (one x-unit == one y-unit on screen) regardless of the
    # approximate fig_dims sizing above - without this, a mismatch between the figure's
    # own aspect ratio and the data's row/col counts stretches cells into rectangles.
    ax.set_aspect('equal')

    ax.set_title(title)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if save_path_svg is not None:
        fig.savefig(save_path_svg)
    plt.show()


def _write_session_main_markdown(md_path, s_id, s_num, column_specs, column_svm_res, which_intervals,
                                  metric='f1', show_shuffle=False):
    """LLM-friendly description of a plot_session_analysis main figure: what each column
    shows, plus the underlying score-vs-time data (the figure's key quantitative content,
    f1_aggr/f1_mean or acc_aggr/acc_mean depending on `metric` - see METRIC_SPECS) as one
    markdown table per column. With show_shuffle, the shuffle-control null mean/p-value
    columns are included too, when present."""
    spec = METRIC_SPECS[metric]
    lines = [
        f"# SVM Decoding Analysis - Session {s_id} (S{s_num})",
        "",
        "## Overview",
        f"- Session ID: {s_id}",
        f"- Session number: S{s_num}",
        f"- Intervals analyzed, in order: {', '.join(which_intervals)}",
        f"- Metric: {spec['label']}",
        "- Columns (X_modality -> predict_y_name):",
    ]
    for i, (x_modality, y_predicted_name) in enumerate(column_specs, 1):
        lines.append(f"  {i}. {x_modality} -> {y_predicted_name}")
    lines += [
        "",
        f"Each column's figure shows: row 0, single-trial position trajectories colored by "
        f"per-timebin decode {spec['label']}; row 1, {spec['label']} over time ({spec['aggr_col']} = "
        f"reference score computed over all trials, {spec['mean_col']} = mean across bootstrap "
        "iterations - the table below has both regardless of which the figure draws" +
        (f", dashed black line = shuffle-control null mean ({spec['shuffle_mean_col']})" if show_shuffle else "") +
        "); row 2, pairwise cosine-angle heatmap (degrees) between per-timebin SVM "
        "weight vectors within this column (blank/white gaps mark interval boundaries); "
        f"further rows, the same {spec['label']}-vs-time score recomputed after splitting trials by "
        "another behavioral covariate or by R1/R2 choice correctness, to check whether decoding is "
        f"confounded with that covariate. Below is row 1's data ({spec['label']} vs time) for each column.",
        "",
    ]
    table_cols = ['interval_name', 'timebin', spec['aggr_col'], spec['mean_col'],
                  spec['ci_lower_col'], spec['ci_upper_col'], 'n_positives', 'n_negatives']
    for (x_modality, y_predicted_name), svm_res in zip(column_specs, column_svm_res):
        lines.append(f"## {y_predicted_name} decoding from {x_modality}")
        lines.append("")
        ordered = _ordered_svm_res(svm_res, which_intervals)
        cols = table_cols.copy()
        if show_shuffle and spec['shuffle_mean_col'] in ordered.columns:
            cols += [spec['shuffle_mean_col'], spec['shuffle_p_col']]
        lines.append(_df_to_markdown_table(ordered[cols]))
        lines.append("")

    with open(md_path, 'w') as f:
        f.write("\n".join(lines))


def plot_session_analysis(s_id, t0_events, behavior, svm_results, x_modalities, y_predicted_names,
                           which_intervals, show_covariate_plot=False, show_colorbar=False, covariate_names=None,
                           show_w_comparison=False, save_dir=None, save_markdown=False, metric='balanced_accuracy',
                           show_shuffle=False, show_scatter=False, show_mean=False, show_aggr_ci=True, smooth=None,
                           show_ylabel=True, next_session_delta_days=None, show_title=True,
                           show_ccgp_secondary=False, ccgp_w_train_stratum=None, min_metric=None,
                           highlight_selections=None, ith_to_dt_session=None,
                           cue_zone_cutoff=pd.to_datetime('2024-11-26_20-00', format='%Y-%m-%d_%H-%M'),
                           show_trial_kinematics=True, mask_chance_band=True):
    """Plot the full per-session analysis. One column per (x_modality, y_predicted_name)
    pair - see _resolve_column_specs for how list(s) of either map to columns. With
    show_covariate_plot, one small extra row is added per covariate (see
    plot_covariate_f1). With show_w_comparison, a second figure is drawn below comparing
    every block's SVM weights against every other block's (see plot_w_comparison) - a
    "block" is one column, except when a single target (y_predicted_name) is being
    compared across the whole figure, in which case a CCGP column contributes its *two*
    train/test directions as separate blocks instead, so that target's own two
    directions compare against each other (there's nothing else to compare against). With
    more than one target, each CCGP column only contributes its one elected direction, so
    the comparison is across targets instead (e.g. cue vs choice_R1) - mixing both would
    blur "target vs target" with "direction vs direction" in the same heatmap. Skipped
    regardless of show_w_comparison with only a single block, since it would just compare
    that block's weights against itself. `min_metric`, if given (e.g. 0.55 for 'balanced_accuracy'),
    extends both this comparison's alpha mask and row 2's own per-column heatmap: any
    cell involving a timebin whose own decoder scored below `min_metric` (on whichever
    `metric` is active) is forced to low alpha, regardless of how extreme its angle looks
    - a poorly-fit decoder's weight direction isn't a trustworthy comparison just because
    it happens to land outside the chance band (see plot_w_comparison). The score each
    cell is checked against is smoothed the same way `smooth` smooths row 1's line (see
    _w_block_with_interval_separators), so the mask agrees with what row 1 actually shows
    instead of flickering off single noisy timebins the drawn (smoothed) line doesn't
    dip on. `highlight_selections`,
    if given, is a list of dicts (session_id/y_name/conditioner/train_stratum/interval/
    timebin - see _selection_matches; only keys a dict sets are checked) marking specific
    W's with a thin dotted green box in the cross-column comparison - each match is also
    returned as actual data (see highlighted_Ws below), not just a box. An optional `name`
    key (not itself matched against anything - see _selection_matches/_highlighted_rows)
    is stamped onto that match's row and drawn as a small purple label next to its
    box/circle (see plot_w_comparison), and feeds plot_w_projection_session's/
    plot_W_list's titles too when that same row is picked up downstream. Since a CCGP
    column can contribute two blocks here (see single_y_name above), a selection meant to
    pick out one specific direction unambiguously should set `train_stratum`, not just
    `conditioner` - see _selection_matches. With save_dir, each figure produced is also
    saved as both a PNG and an SVG there; returns (saved_png_paths, saved_svg_paths,
    highlighted_Ws) (paths both empty if save_dir is None; highlighted_Ws a DataFrame,
    empty if highlight_selections wasn't given or nothing matched). With
    save_markdown (requires save_dir), an LLM-friendly .md sidecar is also written next to
    each PNG - see _write_session_main_markdown and _write_wcomp_markdown. `metric` picks
    which score drives every panel - 'f1' (default) or 'balanced_accuracy' (see
    METRIC_SPECS). By default row 1 draws just the aggr score, in black, with show_aggr_ci
    (default True) shading a bootstrap CI band around it - computed at plot time from that
    row's own y_true/predictions (see _bootstrap_aggr_ci), not the stored ci_lower_col/
    ci_upper_col (those belong to the *mean* score, a different quantity - see
    _plot_f1_lines/create_subplot); show_mean also draws the mean-across-bootstraps score
    with its own (stored) CI band (kept off by default - the mean score tracks aggr
    closely enough that drawing both is redundant);
    show_scatter also draws row 0's per-timebin mean-position scatter colored by score (off
    by default for the same reason - row 1's line already shows it). With show_shuffle,
    row 1 and every covariate panel also draw the shuffle-control null mean as a dashed
    black line, for columns/timebins where that shuffle data exists (see n_shuffles in
    get_SVMCueOutcomeChoicePred). Row 0's cue-zone illustration is skipped for sessions up
    to and including S7 (cue_zone_cutoff, see below) and drawn from S8 on - see create_subplot. `smooth`,
    if given (e.g. 3), is a rolling-mean window (see _smooth) applied to every
    row-1/covariate-panel line - main, mean, shuffle, and per-covariate-level - before
    plotting, separately within each interval; the same window is also applied to the
    min_metric score feeding row 2's heatmap mask and the cross-column comparison below
    (see _w_block_with_interval_separators), so a decoder that only dips below
    min_metric on an isolated noisy timebin isn't masked out while its (smoothed) line
    stays above threshold. The figure's width scales with both ncols
    and len(which_intervals) - a single-interval column is much narrower than a
    three-interval one, so a fixed per-column width otherwise leaves the single-interval
    case looking far too wide. show_ylabel=False hides row 1's y-axis label and tick
    labels - meant for combining several of these exported figures side by side into one
    published panel (e.g. one per session), where only the first should carry the shared
    y-axis. next_session_delta_days, if given, draws a curved arrow with a "+Nd" label in
    the top-right corner - a placeholder meant to be manually dragged into place once you
    line this session's exported figure up against the next one's, so the reader can see
    the gap between them; the caller (not this function) knows the session ordering needed
    to compute the delta to whichever session comes next. show_title=False hides each
    column's own "{y_predicted_name} decoding from {x_modality}" title (see create_subplot)
    - the S01/S02/etc suptitle is unaffected, always shown. With show_ccgp_secondary, row 1
    draws a CCGP fit's two train/test directions symmetrically - two equal-weight lines,
    each colored by the conditioner level it trained on - with a corner annotation of each
    direction's own class balance instead of the naive n_pos/n_neg text, with a star next
    to whichever direction `ccgp_w_train_stratum` elects (see _resolve_ccgp_w_stratum and
    _ccgp_n_annotation and the `ccgp` arg of get_SVMCueOutcomeChoicePred). A no-op for
    naive fits (or older cached analytics). `ccgp_w_train_stratum`, if given, is a
    {y_predicted_name: train_stratum} dict picking which direction's W feeds row 2's
    cosine-similarity heatmap (and the cross-column comparison, if show_w_comparison) for
    that target - e.g. {'cue': 1} uses the 'cue' decoder's W trained on the choice_R1==1
    subset - instead of the automatic more_reliable choice. A no-op for naive fits or
    targets missing from the dict; a stratum that was never actually fit for a target
    raises (see _resolve_ccgp_w_stratum) rather than silently falling back. `ith_to_dt_session`,
    if given, is a {short_session_id: index} dict (see the notebook's own `ith_to_dt_session`)
    used only for the "S<n>" suptitle/save-filename/markdown numbering - a session missing
    from it (or the dict itself omitted) just shows "Unknown" instead of a number.
    `cue_zone_cutoff` is the session-timestamp threshold before/after which row 0's cue-zone
    illustration is skipped/drawn (see create_subplot) - defaults to the S7/S8 boundary this
    figure was originally tuned for; pass a different cutoff (or pd.Timestamp.min to always
    draw it) for other session sets. `show_trial_kinematics=False` hides row 0 entirely for
    every column (single-trial position traces, zone illustration, scatter - see
    create_subplot) - the row still takes up its grid slot, just blank, rather than
    reflowing rows 1/2 upward. `mask_chance_band` passes through to row 2's own per-column
    heatmap here, and (via show_w_comparison) to plot_w_comparison's cross-column
    heatmap - see _angle_alpha_mask for what it does; both heatmaps share that one
    implementation, so this single flag controls both consistently."""
    spec = METRIC_SPECS[metric]
    draw_cue_zone = pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') > cue_zone_cutoff
    column_specs = _resolve_column_specs(x_modalities, y_predicted_names)
    ncols = len(column_specs)
    # Whether a CCGP column's non-elected direction also gets its own block in the
    # cross-column comparison below (see the column loop): only when a single target is
    # being compared across the whole figure, so that target's own two directions are
    # the comparison (nothing else to compare against). With multiple targets, the
    # comparison is across targets instead (e.g. cue vs choice_R1), so each CCGP column
    # contributes only its one elected direction - mixing both would blur "target vs
    # target" with "direction vs direction" in the same heatmap.
    single_y_name = len({yp for _, yp in column_specs}) == 1

    if show_covariate_plot and covariate_names is None:
        # nrows needs the true per-column count, not a fixed len(ALL_Y_PREDICTED_NAMES)-1
        # assumption - that undercounts (silently dropping the last correctness panel, via
        # zip's implicit truncation in plot_covariate_f1) whenever a target isn't itself a
        # member of ALL_Y_PREDICTED_NAMES (e.g. a joined label like 'cueANDstopR1' - nothing
        # gets excluded from its own covariate list, so it needs one more row than a plain
        # 'cue'/'choice_R1'/... column does). Sized to the max across columns; a column
        # needing fewer rows just gets its trailing axes turned off (see plot_covariate_f1).
        fitted_names = set(svm_results.loc[svm_results.index.get_level_values('session_id') == s_id,
                                            'predict_y_name'].unique())
        n_covariates = max(sum(len(specs) for specs in _covariate_specs_for_target(yp, fitted_names))
                           for _, yp in column_specs)
    else:
        n_covariates = (len(covariate_names) + len(CORRECTNESS_SPECS)) if show_covariate_plot else 0

    # width per column scales with len(which_intervals): 3.5in/column was tuned for the
    # 3-interval case (cue/R1/R2), so a single interval at that same fixed width leaves
    # ~2/3 of each column empty - width_per_interval keeps the same per-timebin density
    # regardless of how many intervals are actually requested. Bumped ~20% above the pure
    # proportional value (3.5/3) - a single-interval column still looked a bit cramped at
    # exactly that scale.
    width_per_interval = 1.4
    column_width = width_per_interval * len(which_intervals)

    nrows = 3 + n_covariates
    # row 2 (the heatmap) no longer forces a square aspect (see create_subplot's set_aspect
    # call) - forcing it required shrinking that row's own box, which broke the shared-x
    # alignment with rows 0/1 above it. Left at 'auto', a heatmap cell isn't square, so
    # instead its height_ratio is derived to make that row's own box as tall (in inches) as
    # one column is wide - each height_ratio unit is worth exactly 1.3in of figure height
    # (fig_height below is *defined* as 1.3 * sum(height_ratios), so that conversion holds
    # no matter how many total units there are), so dividing the target physical height by
    # 1.3 gives the ratio units needed. Still an approximation (ignores subplots_adjust's
    # margins and the last column's colorbar, which shrinks its own box a bit), but it
    # tracks len(which_intervals) instead of being tuned for one fixed case.
    heatmap_row_ratio = column_width / 1.3
    # row 0 still exists (see create_subplot's show_trial_kinematics) even when hidden -
    # shrink its ratio to a sliver instead of the usual 1.5 so a hidden row doesn't leave
    # a large blank gap at the top of the figure.
    row0_ratio = 1.5 if show_trial_kinematics else 0.05
    height_ratios = [row0_ratio, 1, heatmap_row_ratio] + [0.8] * n_covariates
    fig_height = 1.3 * sum(height_ratios)
    # show_ylabel needs fixed physical space for row 1's rotated axis-label text and
    # numeric tick labels, regardless of how narrow the plot content itself is - without
    # reserving it separately, tight_layout carves that space out of an already-small
    # fig_width (e.g. a single interval/column, ~1.4in total), crushing the actual axes
    # down to almost nothing instead of just being additional margin around them.
    ylabel_margin = 1.0 if show_ylabel else 0
    fig_width = column_width * ncols + ylabel_margin

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height),
                             height_ratios=height_ratios, sharex=True, dpi=150)
    if ncols == 1:
        axes = axes.reshape(nrows, 1)
    plt.subplots_adjust(hspace=0.15, wspace=0.05)

    scatter = im = None
    column_Ws, column_labels, column_modalities, column_t0_positions, column_svm_res, column_metrics = \
        [], [], [], [], [], []
    column_highlight_positions, column_highlighted_rows = [], []
    for col_i, (x_modality, y_predicted_name) in enumerate(column_specs):
        svm_res, svm_res_secondary, interval_svm_results, all_pos, all_times = get_session_data(
            s_id, t0_events, behavior, svm_results, x_modality, y_predicted_name, which_intervals,
            ccgp_w_train_stratum=ccgp_w_train_stratum)

        if svm_res.empty:
            column_svm_res.append(svm_res)
            # This target was never fit at all for this session - e.g. an early session
            # where the label had zero variance (not yet cue-trained, say) and was
            # skipped entirely by get_SVMCueOutcomeChoicePred's per-y_name skip guard, so
            # there's no row anywhere for this (x_modality, y_predicted_name). Leave the
            # column blank rather than feeding an empty frame into create_subplot/
            # _w_block_with_interval_separators, which assume >=1 fitted row. A (0, 0) Ws
            # block keeps this column's placement in the cross-column comparison
            # (plot_w_comparison already renders mismatched-dimensionality blocks blank).
            for row_i in range(3):
                axes[row_i, col_i].axis('off')
            if show_title:
                axes[0, col_i].set_title(f'{y_predicted_name} decoding from \n{x_modality}\n(no data)')
            column_Ws.append(np.zeros((0, 0)))
            column_t0_positions.append([])
            column_metrics.append(np.zeros(0))
            column_labels.append(f'{y_predicted_name}\n{x_modality}')
            column_modalities.append(x_modality)
            continue

        # Resolved once, up front, so row 1's corner-annotation star (create_subplot ->
        # _plot_f1_lines -> _ccgp_n_annotation) and row 2's heatmap below agree on which
        # direction is "the" one for this target - see _resolve_ccgp_w_stratum (raises on
        # an explicitly-requested stratum that was never actually fit, rather than
        # silently ignoring it).
        elected_stratum = _resolve_ccgp_w_stratum(svm_res, svm_res_secondary, y_predicted_name,
                                                   ccgp_w_train_stratum)

        # get_session_data's svm_res/svm_res_secondary are still the automatic
        # more_reliable/other split from _split_ccgp_primary_secondary - the election above
        # (which respects ccgp_w_train_stratum) is only reflected in `elected_stratum`
        # itself so far. Row 1's single-line fallback (show_ccgp_secondary=False - see
        # _plot_f1_lines) draws svm_res as-is, so when a requested stratum overrides the
        # automatic pick (e.g. session s14, where more_reliable disagreed with the
        # requested train_stratum), it silently drew the wrong direction while row 2's
        # heatmap/star (which do use elected_stratum) showed the right one. Re-sorting
        # svm_res/svm_res_secondary here so svm_res is always the elected direction fixes
        # row 1 for every downstream use (create_subplot, w_source, is_ccgp_column) without
        # disturbing show_ccgp_secondary=True, whose symmetric two-line draw already colors
        # each row by its own train_stratum regardless of which dataframe is "target" vs
        # "secondary".
        if elected_stratum is not None:
            other_stratum = next((s for s in pd.concat([svm_res, svm_res_secondary])['train_stratum'].dropna().unique()
                                   if s != elected_stratum), None)
            svm_res, svm_res_secondary = (
                _rows_for_stratum(svm_res, svm_res_secondary, elected_stratum),
                _rows_for_stratum(svm_res, svm_res_secondary, other_stratum) if other_stratum is not None
                else svm_res.iloc[0:0],
            )
        column_svm_res.append(svm_res)

        scatter, xticks, xtick_labels = create_subplot(axes[:3, col_i], all_pos, svm_res,
                                          f'{y_predicted_name} decoding from \n{x_modality}',
                                          y_predicted_name, which_intervals, metric=metric,
                                          show_shuffle=show_shuffle, show_scatter=show_scatter,
                                          show_mean=show_mean, show_aggr_ci=show_aggr_ci,
                                          draw_cue_zone=draw_cue_zone, smooth=smooth,
                                          show_ylabel=show_ylabel, show_title=show_title,
                                          show_ccgp_secondary=show_ccgp_secondary,
                                          svm_res_secondary=svm_res_secondary,
                                          ccgp_w_elected_stratum=elected_stratum,
                                          show_trial_kinematics=show_trial_kinematics)
        if show_covariate_plot:
            plot_covariate_f1(axes[3:, col_i], interval_svm_results, y_predicted_name,
                               which_intervals, covariate_names, metric=metric, smooth=smooth,
                               show_shuffle=show_shuffle)

        # Row-order to match create_subplot's offset scheme (t0/boundary lines already
        # drawn on this axis above), so this heatmap's pixels line up with those instead
        # of relying on svm_res's incidental native row order. This heatmap shares the
        # same gapped W block as the cross-column comparison below (plot_w_comparison) -
        # the NaN separator row it inserts between intervals is exactly the 1-unit gap
        # create_subplot/_plot_f1_lines leave in row0/row1, so every row stays aligned.
        # w_source is the elected direction's own rows - a no-op (returns svm_res
        # unchanged) for naive fits, which only ever have one direction anyway.
        w_source = _rows_for_stratum(svm_res, svm_res_secondary, elected_stratum)
        ordered_svm_res = _ordered_svm_res(w_source, which_intervals)
        # metric_col: per-row aggr score, parallel to block_Ws - feeds min_metric below
        # (both row 2's own heatmap and the cross-column comparison's alpha mask).
        block_Ws, t0_positions, block_metric = _w_block_with_interval_separators(
            ordered_svm_res, which_intervals, metric_col=spec['aggr_col'], smooth=smooth)

        # With a single target across the whole figure, a CCGP column's non-elected
        # direction also gets its own block below, so that target's two directions
        # compare against each other (see single_y_name above). With multiple targets,
        # only the elected direction is included, so the comparison is across targets
        # (e.g. cue vs choice_R1) instead. Row 2's own heatmap right below always stays
        # elected-only regardless (a self-similarity-across-timebins view, not this
        # cross-block comparison).
        is_ccgp_column = elected_stratum is not None
        conditioner = svm_res['conditioner'].dropna().iloc[0] if is_ccgp_column else None
        elected_suffix = f' (train:{_level_label(conditioner, elected_stratum)})' if is_ccgp_column else ''

        # own_highlight_rows: same local_row values as column_highlight_positions below,
        # but kept separate (and un-offset) for row 2's own per-column heatmap further
        # down - that heatmap is a single block (this column's elected direction only,
        # see the comment above), so it needs the bare local_row, not the cross-column
        # figure's (block_index, local_row) pair.
        own_highlight_rows = []
        for local_row, matched_row in _highlighted_rows(ordered_svm_res, which_intervals, highlight_selections or [],
                                                          session_id=s_id, y_name=y_predicted_name,
                                                          conditioner=conditioner):
            name = matched_row.get('name')
            column_highlight_positions.append((len(column_Ws), local_row, name))
            column_highlighted_rows.append(matched_row)
            own_highlight_rows.append((local_row, name))
        column_Ws.append(block_Ws)
        column_t0_positions.append(t0_positions)
        column_metrics.append(block_metric)
        column_labels.append(f'{y_predicted_name}{elected_suffix}\n{x_modality}')
        column_modalities.append(x_modality)

        if single_y_name and is_ccgp_column and svm_res_secondary is not None and not svm_res_secondary.empty:
            # "Other" direction = whichever fitted train_stratum isn't the elected one,
            # from the full primary+secondary pool - NOT svm_res_secondary's own stratum
            # label. svm_res_secondary is whichever direction _split_ccgp_primary_secondary
            # called non-more_reliable, decided *before* ccgp_w_train_stratum is applied -
            # when that request overrides the automatic pick, svm_res_secondary's own
            # stratum collapses to the *same* value as elected_stratum, duplicating the
            # elected block instead of showing the genuine other direction (same class of
            # bug already hit and fixed in get_session_sequence_Ws).
            combined_strata = pd.concat([svm_res, svm_res_secondary])['train_stratum'].dropna().unique()
            other_strata = [s for s in combined_strata if s != elected_stratum]
            if other_strata:
                secondary_stratum = other_strata[0]
                secondary_source = _rows_for_stratum(svm_res, svm_res_secondary, secondary_stratum)
                secondary_ordered = _ordered_svm_res(secondary_source, which_intervals)
                secondary_Ws, secondary_t0, secondary_metric = _w_block_with_interval_separators(
                    secondary_ordered, which_intervals, metric_col=spec['aggr_col'], smooth=smooth)
                for local_row, matched_row in _highlighted_rows(secondary_ordered, which_intervals,
                                                                  highlight_selections or [], session_id=s_id,
                                                                  y_name=y_predicted_name, conditioner=conditioner):
                    column_highlight_positions.append((len(column_Ws), local_row, matched_row.get('name')))
                    column_highlighted_rows.append(matched_row)
                column_Ws.append(secondary_Ws)
                column_t0_positions.append(secondary_t0)
                column_metrics.append(secondary_metric)
                column_labels.append(f'{y_predicted_name} (train:{_level_label(conditioner, secondary_stratum)})'
                                      f'\n{x_modality}')
                column_modalities.append(x_modality)

        timebins = np.arange(len(block_Ws))
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_sim_matrix = np.dot(block_Ws, block_Ws.T) / (np.linalg.norm(block_Ws, axis=1)[:, np.newaxis] *
                                                              np.linalg.norm(block_Ws, axis=1)[np.newaxis, :])
            cos_sim_matrix = np.arccos(np.clip(cos_sim_matrix, -1.0, 1.0)) * (180/np.pi)

        lower, upper = angle_2std_thr[x_modality]
        high_angles_alpha = _angle_alpha_mask(cos_sim_matrix, lower, upper, mask_chance_band=mask_chance_band,
                                               metrics_i=block_metric, metrics_j=block_metric, min_metric=min_metric)
        high_angles_alpha[np.isnan(cos_sim_matrix)] = 1.0

        # Use pcolormesh instead of imshow for better alignment
        cmap = plt.cm.bwr_r.with_extremes(bad='white')
        X, Y = np.meshgrid(timebins, timebins)
        im = axes[2,col_i].pcolormesh(X, Y, np.ma.masked_invalid(cos_sim_matrix),
                                  alpha=high_angles_alpha,
                                 cmap=cmap, vmin=0, vmax=180)

        # Same purple row-band + cross-point circle (plus name label) as
        # plot_w_comparison's highlight_positions (see there), drawn here too since a
        # single-block session (e.g. a naive fit) never gets a cross-column comparison
        # figure at all (skipped below when len(column_Ws) <= 1) - this is the only
        # heatmap such a session has.
        total = len(timebins)
        for g, name in own_highlight_rows:
            axes[2,col_i].add_patch(Rectangle((-0.5, g - 0.5), total, 1, fill=False,
                                               edgecolor='purple', alpha=0.15, linewidth=1, zorder=20))
            axes[2,col_i].add_patch(Circle((g, g), radius=0.5 * np.sqrt(2), fill=False,
                                            edgecolor='purple', alpha=0.15, linewidth=1.5, zorder=20))
            if name:
                axes[2,col_i].text(g + 1, g + 1, name, fontsize=5, color='purple', alpha=0.9, rotation=25,
                                    ha='left', va='bottom', clip_on=False, zorder=21)

        axes[2,col_i].set_xticks(xticks)
        # fontsize=7 to match the n_skip/n_stop-style corner annotations (_plot_f1_lines) -
        # the default (poster) rcParams font size is far too large for how little room
        # these labels have, especially now that column width scales down for fewer
        # intervals (see width_per_interval above).
        axes[2,col_i].set_xticklabels(xtick_labels, fontsize=7)
        axes[2,col_i].set_xlim(-0.5, len(timebins)-0.5)
        axes[2,col_i].set_ylim(-0.5, len(timebins)-0.5)
        # Deliberately NOT set_aspect('equal') here: forcing a square aspect makes matplotlib
        # shrink/recenter this Axes' own bounding box within its GridSpec cell to fit a
        # square, which breaks the shared-x alignment with rows 0/1 above it (their box stays
        # full-width, this one doesn't) - the t0 dashed lines above and below then visibly
        # drift apart. Left at the default 'auto' aspect, cells aren't perfectly square, but
        # the box - and so the alignment - stays identical to every other row; height_ratios
        # above gives this row extra height to compensate for the resulting flatter cells.
        axes[2,col_i].tick_params(labelleft=False)

        # turn off all spines
        for row_i in range(3):
            [sp.set_visible(False) for sp in axes[row_i, col_i].spines.values()]

    if show_colorbar:
        if scatter is not None:
            cbar = plt.colorbar(scatter, ax=axes[0, -1])
            cbar.set_ticks([.75, .5, .25])
            cbar.set_label(f'SVM {spec["label"]}', labelpad=15)

        cbar2 = plt.colorbar(im, ax=axes[2, -1])
        cbar2.set_ticks([0, 60, 90, 120, 180])
        cbar2.set_label('Angle between SVM planes (degrees)', labelpad=15)

    s_num = (ith_to_dt_session or {}).get(s_id, "Unknown")
    title = f'S{s_num:02d}' if isinstance(s_num, int) else f'S{s_num}'
    plt.suptitle(title, fontsize=16, y=.99)
    plt.tight_layout()

    # Placeholder for manually re-positioning once this session's exported figure is lined
    # up against the next one's (e.g. dragged into an SVG editor) - the arrow/label live in
    # figure-fraction coordinates so they move as one unit with the rest of the figure until
    # then. Drawn last so it sits on top of everything else.
    if next_session_delta_days is not None:
        # coordinates tuned (see scratchpad render) so the curve's full bulge stays inside
        # the figure bounds - clip_on=False only exempts it from axes clipping, not from
        # the canvas edge, so a curve that arcs outside [0, 1] figure-fraction gets cut off
        # in the saved file rather than just extending past the axes as intended.
        arrow = FancyArrowPatch((0.65, 0.88), (0.96, 0.88), transform=fig.transFigure,
                                 connectionstyle='arc3,rad=-0.25', arrowstyle='-|>',
                                 mutation_scale=14, linewidth=1.6, color='black', clip_on=False)
        fig.add_artist(arrow)
        fig.text(0.805, 0.905, f'+{next_session_delta_days}d', transform=fig.transFigure,
                  ha='center', va='bottom', fontsize=11, fontweight='bold')

    # saved_png_paths/saved_svg_paths feed build_canvas/build_canvas_svg respectively -
    # PNGs are opened as raster images (PIL), SVGs are merged as vector groups (ElementTree)
    # so the exported figure (including the drag-into-place arrow above) stays fully
    # editable. .md sidecars are written to save_dir as a side effect but not tracked here.
    saved_png_paths, saved_svg_paths = [], []
    if save_dir is not None:
        main_path = os.path.join(save_dir, f'session_{s_id}_S{s_num}_main.png')
        main_svg_path = os.path.join(save_dir, f'session_{s_id}_S{s_num}_main.svg')
        fig.savefig(main_path, dpi=150)
        fig.savefig(main_svg_path)
        saved_png_paths.append(main_path)
        saved_svg_paths.append(main_svg_path)
        if save_markdown:
            main_md_path = os.path.join(save_dir, f'session_{s_id}_S{s_num}_main.md')
            _write_session_main_markdown(main_md_path, s_id, s_num, column_specs, column_svm_res, which_intervals,
                                          metric=metric, show_shuffle=show_shuffle)

    plt.show()

    # Only meaningful with more than one block to compare - a single block would only
    # compare its own weights against itself, so skip it entirely rather than draw a
    # degenerate 1x1 heatmap. Guard on len(column_Ws) (block count), not ncols (column
    # count): a lone CCGP column still contributes two blocks (one per train/test
    # direction - see above), so it's comparable even at ncols==1. Blocks can differ by
    # X_modality, predict_y_name, direction, or any combination (see
    # _resolve_column_specs) - e.g. cue vs choice_R1, or cue's own two directions against
    # each other, are equally meaningful comparisons, so this doesn't check
    # column_modalities specifically, only that there's more than one block at all.
    if show_w_comparison and len(column_Ws) > 1:
        wcomp_path = os.path.join(save_dir, f'session_{s_id}_S{s_num}_wcomp.png') if save_dir is not None else None
        wcomp_svg_path = (os.path.join(save_dir, f'session_{s_id}_S{s_num}_wcomp.svg')
                           if save_dir is not None else None)
        wcomp_md_path = (os.path.join(save_dir, f'session_{s_id}_S{s_num}_wcomp.md')
                          if save_dir is not None and save_markdown else None)
        plot_w_comparison(column_Ws, column_labels, column_modalities, title, fig_width=3.5 * ncols,
                           block_t0_positions=column_t0_positions, save_path=wcomp_path,
                           save_path_svg=wcomp_svg_path, markdown_path=wcomp_md_path,
                           block_metrics=column_metrics, min_metric=min_metric,
                           highlight_positions=column_highlight_positions, mask_chance_band=mask_chance_band)
        if wcomp_path is not None:
            saved_png_paths.append(wcomp_path)
            saved_svg_paths.append(wcomp_svg_path)

    highlighted_Ws = pd.DataFrame(column_highlighted_rows) if column_highlighted_rows else pd.DataFrame()
    return saved_png_paths, saved_svg_paths, highlighted_Ws

    # fname = os.path.join(output_dir, f'session_S{s_id}_{y_predicted_name}_svm_analysis_S15_Rest.svg')
    # fig.savefig(fname, dpi=300)


def get_session_sequence_Ws(session_ids, svm_results, x_modality, y_predicted_name, which_intervals,
                             metric='f1', ccgp_w_train_stratum=None, highlight_selections=None,
                             ith_to_dt_session=None, smooth=None):
    """Per-session SVM weight blocks (with zero-row interval separators - see
    _w_block_with_interval_separators) for a single (x_modality, y_predicted_name) fit,
    across which_intervals, in the given session order - one block per session, always
    (unlike plot_session_analysis's cross-column comparison, which shows a CCGP column's
    two directions as separate blocks - a long session sequence would double in width for
    every CCGP session in it, which gets unwieldy fast). For a CCGP session,
    `ccgp_w_train_stratum` (see _resolve_ccgp_w_stratum) selects which single direction
    represents that session - default: the automatic more_reliable choice - and that
    direction's conditioner level is noted in the block's label ("S<n> (train:<level>)")
    since there's no second block here to tell it apart from. `metric` (see METRIC_SPECS)
    picks which aggr score is attached to each block as its per-row metric array, for use
    with plot_w_comparison's min_metric. `highlight_selections`, if given, is a list of
    dicts with any of session_id/y_name/conditioner/train_stratum/interval/timebin (see
    _selection_matches - only keys a dict actually sets are checked, so e.g. a dict
    without `conditioner` matches naive and CCGP fits alike; a dict without
    `train_stratum` matches whichever direction this function elected, since that's the
    only one present in any given session's block here). Each matching row is resolved
    both to a (block_index, row_index, name) triple (see plot_w_comparison's
    highlight_positions for what that draws - name is the matched selection's own `name`
    key, if it set one) and returned as actual data (its `w` and every other fitted
    column) for further inspection - that's the whole point of passing selections in, not
    just seeing a box drawn around them. `smooth`, if given (e.g. 3), passes straight
    through to _w_block_with_interval_separators - the same rolling-mean window
    plot_session_analysis applies to its own min_metric masking (see _smooth), so a
    session that only dips below min_metric on an isolated noisy timebin isn't masked out
    here either. `ith_to_dt_session`, if given, is a
    {short_session_id: index} dict used only for the "S<n>" block label (falls back to
    "?" when absent/missing a session - see plot_session_analysis's own use of this).
    Returns (block_Ws, block_labels, block_modalities, block_t0_positions, block_metrics,
    highlight_positions, highlighted_rows) - one entry per session, except
    highlight_positions/highlighted_rows, which have one entry per matched selection
    (highlighted_rows as a single DataFrame, empty if nothing matched)."""
    spec = METRIC_SPECS[metric]
    block_Ws, block_labels, block_modalities, block_t0_positions, block_metrics = [], [], [], [], []
    highlight_positions, highlighted_rows = [], []

    for s_id in session_ids:
        svm_res = svm_results.loc[
            (svm_results.index.get_level_values('session_id') == s_id) &
            (svm_results['X_modality'] == x_modality) &
            (svm_results['predict_y_name'] == y_predicted_name) &
            (svm_results['interval_name'].isin(which_intervals))
        ]
        primary, secondary = _split_ccgp_primary_secondary(svm_res)
        s_label = f'S{(ith_to_dt_session or {}).get(s_id, "?")}'

        # _resolve_ccgp_w_stratum guards primary.empty/all-naive internally, returning
        # None rather than raising on an empty session - the same "no data" case
        # plot_session_analysis handles explicitly, just without needing a separate
        # branch here.
        elected_stratum = _resolve_ccgp_w_stratum(primary, secondary, y_predicted_name, ccgp_w_train_stratum)
        label = s_label
        conditioner = None
        if elected_stratum is not None:
            conditioner = primary['conditioner'].dropna().iloc[0]
            label = f'{s_label} (train:{_level_label(conditioner, elected_stratum)})'

        rows = _rows_for_stratum(primary, secondary, elected_stratum)
        ordered = _ordered_svm_res(rows, which_intervals)

        for local_row, matched_row in _highlighted_rows(ordered, which_intervals, highlight_selections or [],
                                                         session_id=s_id, y_name=y_predicted_name,
                                                         conditioner=conditioner):
            highlight_positions.append((len(block_Ws), local_row, matched_row.get('name')))
            highlighted_rows.append(matched_row)
        Ws, t0_positions, metric_vals = _w_block_with_interval_separators(
            ordered, which_intervals, metric_col=spec['aggr_col'], smooth=smooth)

        block_Ws.append(Ws)
        block_labels.append(label)
        block_modalities.append(x_modality)
        block_t0_positions.append(t0_positions)
        block_metrics.append(metric_vals)

    highlighted_rows = pd.DataFrame(highlighted_rows) if highlighted_rows else pd.DataFrame()
    return (block_Ws, block_labels, block_modalities, block_t0_positions, block_metrics,
            highlight_positions, highlighted_rows)


def plot_session_sequence_w_comparison(session_ids, svm_results, x_modality, y_predicted_name, which_intervals,
                                        save_path=None, save_path_svg=None, metric='balanced_accuracy',
                                        ccgp_w_train_stratum=None, min_metric=None, highlight_selections=None,
                                        ith_to_dt_session=None, reference_session_id=None, as_column=False,
                                        mask_chance_band=True, smooth=None):
    """Cross-session SVM-weight similarity for a single (x_modality, y_predicted_name):
    one block per session (see plot_w_comparison for what a block means and
    get_session_sequence_Ws for how a CCGP session's single direction is chosen), in the
    given session order. Shows how stable/consistent a single decoder's weights are
    across sessions, the same way plot_w_comparison shows it across columns within one
    session. `metric`, `ccgp_w_train_stratum`, `min_metric`, `highlight_selections`,
    `ith_to_dt_session`, and `smooth` all pass straight through to get_session_sequence_Ws/
    plot_w_comparison - see there for what each does. Returns `highlighted_rows` (see
    get_session_sequence_Ws) - the actual matched data, not just the boxes drawn around
    it - empty if highlight_selections wasn't given or nothing matched.

    `reference_session_id`, if given (one of the (short) ids in `session_ids`), restricts
    the plot to just that one session's own comparisons against every session in the
    sequence - e.g. "session 7 vs every other session" - instead of the full N-by-N grid
    (see plot_w_comparison's `reference_block`, which this resolves session_id into).
    Raises ValueError if it isn't actually present in `session_ids`. `as_column` flips
    that restricted view into a vertical strip instead of the default horizontal one -
    see plot_w_comparison. Both ignored when `reference_session_id` is None.
    `mask_chance_band` passes straight through to plot_w_comparison - see there and
    _angle_alpha_mask."""
    block_Ws, block_labels, block_modalities, block_t0_positions, block_metrics, highlight_positions, \
        highlighted_rows = get_session_sequence_Ws(
            session_ids, svm_results, x_modality, y_predicted_name, which_intervals,
            metric=metric, ccgp_w_train_stratum=ccgp_w_train_stratum, highlight_selections=highlight_selections,
            ith_to_dt_session=ith_to_dt_session, smooth=smooth)

    reference_block = None
    if reference_session_id is not None:
        session_ids = list(session_ids)
        if reference_session_id not in session_ids:
            raise ValueError(f"{reference_session_id!r} not in session_ids: {session_ids}")
        reference_block = session_ids.index(reference_session_id)

    title = f'{y_predicted_name} decoding from {x_modality} across sessions'
    if reference_block is not None:
        title += f' (vs {block_labels[reference_block]})'
    fig_width = max(6, 1.2 * len(session_ids))
    plot_w_comparison(block_Ws, block_labels, block_modalities, title, fig_width,
                       block_t0_positions=block_t0_positions, save_path=save_path, save_path_svg=save_path_svg,
                       block_metrics=block_metrics, min_metric=min_metric, highlight_positions=highlight_positions,
                       reference_block=reference_block, as_column=as_column, mask_chance_band=mask_chance_band)
    return highlighted_rows


def _stack_vertically(image_paths):
    """Stack a session's own plots (main, then w-comparison) top to bottom, left-aligned."""
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
    """session_image_paths: one list of PNG paths per session (main plot first, then
    w-comparison, so they stack vertically and columns stay aligned). Each session's
    stack is then placed side by side with every other session's."""
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
    """SVG analog of build_canvas, for the vector (.svg) exports plot_session_analysis
    writes alongside its PNGs - same one-list-of-paths-per-session layout (main plot first,
    then w-comparison, stacked vertically; sessions placed side by side). Rather than
    rasterizing (Image.paste, like build_canvas), each source SVG's own content is moved
    into a nested <svg x=.. y=.. width=.. height=..> element inside one master <svg> -
    nested <svg> elements support exactly this kind of positioning natively, so the
    combined file stays fully vector/editable (fonts, paths, and - importantly - the
    drag-into-place day-delta arrows from plot_session_analysis all stay intact and
    movable in Inkscape/Illustrator, unlike flattening to a raster canvas would)."""
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
