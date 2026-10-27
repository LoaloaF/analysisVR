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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
from matplotlib.transforms import offset_copy
from sklearn.metrics import f1_score, balanced_accuracy_score

# Interval geometry and the shared trial/x-axis conventions - see plotlib/trial_traces.py.
from plotlib.trial_traces import (
    interval_x0_map, interval_short_label as _interval_short_label, smooth as _smooth,
    block_x as _block_x, next_block_offset as _next_block_offset, T0_LINE_KW)
# The optional variance row (show_w_variance). Safe to import at module level:
# var_plotting_lib depends only on plotlib and does no I/O, so this adds no cycle
# (w_plotting_lib, which imports *this* module, is the direction that would) and no
# backend dependency. `session_fr_for_modality`, which the row also needs, is imported
# lazily at its call site instead - it lives in plotlib.session_data, the one plotlib
# module that pulls in `analytics_processing`, and importing it here would make this
# module unimportable without the analytics backend, which it has never been.
from var_plotting_lib import (ACTIVATION_YLIM, w_activation_profile, plot_w_activation_row,
                              w_variance_profile, plot_w_variance_row,
                               resolve_variance_splits,
                               # the two the session summary scores its own W's with, rather
                               # than going through w_variance_profile: only the black curve
                               # is wanted there, and that function's expensive halves (the
                               # cross-validated PC bracket, the random-direction band) are
                               # neither drawn nor read across sessions
                               get_interval_trial_stack, timebin_samples)
from plotlib.variance import var_explained
from plotlib import fit_samples as fitspl

# Re-exported, not just imported: lm4_plots.py and other callers do
# `from svm_plotting_lib import plot_session_analysis, build_canvas, build_canvas_svg`,
# and those two now have a single definition in plotlib/export.py (they used to exist as
# three byte-identical copies here, in dim_plotting_lib.py and in svm_analysis_2026.py).
from plotlib.export import build_canvas, build_canvas_svg, save_figure  # noqa: F401


# Re-exported, not just imported (same as build_canvas above): svm_analysis_2026.py and
# lm4_plots.py import `modality_colors` from this module by name. Its definition moved to
# plotlib/colors.py when dim_plotting_lib's trial grouping needed the same colors.
from plotlib.colors import modality_colors  # noqa: E402,F401

ALL_Y_PREDICTED_NAMES = ['cue', 'choice_R1', 'choice_R2', 'trial_outcome']

# Styling for the highlight_selections callout (row band + cross-point circle + name
# label), shared by plot_w_comparison's cross-block figure and plot_session_analysis's
# own per-column row-2 heatmap so the two can't drift apart.
# Colour limits for every angle heatmap (plot_w_comparison's cross-block one and
# plot_session_analysis's own per-column one). The full 0-180 is the honest range but wastes
# most of the colormap: real angles between two fitted W's cluster near 90 (a random pair in
# mPFC is within 68.5-110.4 95% of the time - see angle_2std_thr), so the interesting
# variation lands in a narrow band of colour. Narrow it to bring that out - as a pair, or as
# a scalar half-width read symmetrically around 90 (30 -> (60, 120)), which is the form that
# keeps white at exactly orthogonal.
# The "S13" figure title. Bold and well above the default rcParams size: on an exported
# panel this is the only thing identifying which session the column is, and it has to stay
# legible once the canvas is scaled down into a poster column.
SUPTITLE_KW = dict(fontsize=20, fontweight='bold')
# The two small in-axes texts, sized separately because they are read differently. The
# corner annotation (see _ccgp_n_annotation) conveys the fit METHOD - train on one level,
# test on the other - rather than numbers anyone reads off precisely, so it stays small and
# out of the way. The heatmap's own time ticks label the x axis of a panel that has no
# other x labelling at all, so they get a size that survives a poster reduction.
CORNER_FONTSIZE = 7
TICK_FONTSIZE = 14
# The min_metric indicator's two stacked lines (see _draw_min_metric_indicator) - narrow
# enough to read as a marker strip rather than as another curve, and sitting on the axes'
# own bottom edge (these rows draw no spines), which is as low as the pair can go without
# running into the counts annotation in the gap below.
MIN_METRIC_INDICATOR_LW = 2.0
MIN_METRIC_INDICATOR_Y = 0.0

ANGLE_CLIM = (0, 180)


# Half-width (degrees) of the white core around 90 - see _angle_colormapping. None keeps the plain
# linear mapping. ~20 puts the fade right across the mPFC chance band (angle_2std_thr is
# 68.5-110.4, i.e. +-21), which is the point: the angles a random pair produces stay pale
# and everything beyond them saturates quickly.
ANGLE_WHITE_SIGMA = None


def _angle_colormapping(angle_clim=None, white_sigma=None, base=None, n_steps=512):
    """(cmap, norm) for the angle heatmaps: a LINEAR norm, with a gaussian white core
    blended into the colormap.

    The problem it solves: bwr_r spreads its white over a sliver of a 0-180 range, so a
    heatmap of real W-vs-W angles - which pile up near 90 - comes out uniformly pale pink,
    while narrowing the limits (angle_clim) just saturates everything at once.

    With `white_sigma` (degrees) each colour is mixed toward white by

        w(d) = exp(-d^2 / 2*sigma^2),   d = angle - 90
        rgb(d) = (1 - w) * bwr_r(d) + w * white

    so the base colormap is left ALONE away from 90 (w = 0.14 at 2*sigma, 0.011 at 3*sigma -
    a 45-degree angle keeps bwr_r's own red at sigma=25) and only the middle is whitened,
    smoothly. That is the whole intent: widen the neutral zone, do not rescale the ramp.
    An earlier version remapped which part of the ramp each angle sampled instead, which
    changed the far colours too and needed a shape exponent to compensate - a blend needs
    no such correction, and `sigma` alone says where the neutral zone ends.

    Baked into a ListedColormap sampled at `n_steps` LINEARLY spaced angles, with a plain
    Normalize: degrees stay linear on the colourbar (60 sits exactly a third of the way up
    a 0-180 bar) and what changes is which colour each degree gets.

    Sizing it: sigma ~= the chance band's half-width is the natural choice (angle_2std_thr
    is +-21 for mPFC), so the angles a random pair produces read as neutral and anything
    past ~2 sigma keeps its full colour. Note this does the same job as `mask_chance_band`'s
    alpha dimming, from the other side - running both dims the chance band twice.
    """
    vmin, vmax = _angle_clim(angle_clim)
    base = plt.cm.bwr_r if base is None else base
    norm = Normalize(vmin=vmin, vmax=vmax)
    if not white_sigma:
        return base.with_extremes(bad='white'), norm
    sigma = float(white_sigma)
    angles = np.linspace(vmin, vmax, n_steps)
    rgba = np.asarray(base(norm(angles)))
    w = np.exp(-((angles - 90.0) ** 2) / (2 * sigma ** 2))[:, np.newaxis]
    rgba[:, :3] = (1.0 - w) * rgba[:, :3] + w
    return ListedColormap(rgba).with_extremes(bad='white'), norm


def _angle_clim(angle_clim):
    """(vmin, vmax) from a pair, or from a scalar half-width around 90. Kept symmetric-
    capable on purpose: bwr_r puts white in the middle of the range, so an asymmetric pair
    moves white off 90 and the colours stop meaning "more/less aligned than orthogonal"."""
    if angle_clim is None:
        return 0, 180
    if np.isscalar(angle_clim):
        return 90 - float(angle_clim), 90 + float(angle_clim)
    return tuple(angle_clim)


HIGHLIGHT_COLOR = 'darkgreen'
HIGHLIGHT_BOX_LW = 1.5
HIGHLIGHT_CIRCLE_LW = 2.0

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
# the matching precomputed svm_results columns; null_draw_cols/null_p_cols to the
# optional null-control columns (see _null_mean_values); score_fn is used wherever a
# metric is recomputed on the fly from y_true/predictions (e.g. after splitting trials by
# a covariate).
#
# Two generations of fit table are read by these figures and they spell the same
# quantities differently (v1 = get_SVMCueOutcomeChoicePred, v2 = SVM_TaskSubspaces), so
# every schema-varying name is a *tuple of candidates*, resolved against the frame at
# read time by _first_present - first one actually present wins. Nothing renames or
# copies the frame; a v2 table is plotted as it comes off the analytic.
#
# null_draw_cols is keyed by null kind: 'perm' (labels fully permuted - v1's only null,
# and v2's `null_perm_acc`) and 'shift' (v2 only - the trial-label vector circularly
# shifted, which preserves the labels' own temporal structure and so tests against slow
# drift rather than destroying it; see svm_task_subspaces.py). Each column holds the
# per-row LIST of null draws, which is what gets averaged for the dashed line - v1's
# precomputed scalar mean is the mean of exactly that list, so one code path serves both
# (scalars are still accepted as a last candidate).
METRIC_SPECS = {
    'f1': dict(
        aggr_col='f1_aggr', mean_col='f1_mean', ci_lower_col='f1_ci_lower', ci_upper_col='f1_ci_upper',
        null_draw_cols={'perm': ('shuffle_f1_aggr', 'shuffle_f1_mean')},
        null_p_cols={'perm': ('shuffle_f1_p',)},
        label='F1 Score',
        score_fn=lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
    ),
    'balanced_accuracy': dict(
        aggr_col='acc_aggr', mean_col='acc_mean', ci_lower_col='acc_ci_lower', ci_upper_col='acc_ci_upper',
        null_draw_cols={'perm': ('shuffle_acc_aggr', 'null_perm_acc', 'shuffle_acc_mean'),
                        'shift': ('null_shift_acc',)},
        null_p_cols={'perm': ('shuffle_acc_p', 'p_point_perm'), 'shift': ('p_point_shift',)},
        label='Balanced Accuracy',
        score_fn=balanced_accuracy_score,
    ),
}

# Class-count columns, same v1/v2 candidate scheme as METRIC_SPECS (see _first_present).
N_POS_COLS = ('n_positives', 'n_pos')
N_NEG_COLS = ('n_negatives', 'n_neg')


def _first_present(df, candidates):
    """First of `candidates` that is a column of `df`, or None - how every v1/v2
    schema difference is resolved (see METRIC_SPECS)."""
    return next((c for c in candidates if c in df.columns), None)


def _null_mean_values(rows, metric, null_kind='perm'):
    """(mean null score per row, the column it came from) for `null_kind`, or (None, None)
    if this table carries no such null.

    The stored value is a list of per-draw scores (v2, and v1's own shuffle_*_aggr), so
    the line the figure draws is that list's mean; a scalar column is passed through
    unchanged for tables that only kept the mean. Empty/all-NaN draw lists give NaN, which
    leaves a hole in the dashed line rather than a fabricated 0.5."""
    col = _first_present(rows, METRIC_SPECS[metric]['null_draw_cols'].get(null_kind, ()))
    if col is None:
        return None, None

    def _mean(v):
        if v is None or np.isscalar(v):
            return float(v) if v is not None else np.nan
        arr = np.asarray(v, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else np.nan

    return rows[col].apply(_mean).to_numpy(dtype=float), col


# Re-exported, not defined here: LEVEL_LABELS/CORRECTNESS_SPECS/_covariate_color/
# _level_label moved to plotlib/trial_labels.py when var_plotting_lib's condition-split
# variance curves needed them too (var cannot import from w_plotting_lib - svm -> var and w -> svm, so
# that would close a cycle). Kept under their old private names so this module's own call
# sites, svm_analysis_2026.py's imports and _smoke_render_w.py's injection all still work.
from plotlib.trial_labels import (  # noqa: E402
    LEVEL_LABELS, CORRECTNESS_SPECS,  # noqa: F401
    covariate_color as _covariate_color, level_label as _level_label,
    trial_covariate_levels as _trial_covariate_levels)  # noqa: F401


def _interval_x0(interval_x0s, which_interval):
    """Where an interval's t0 marker belongs within its block, in native bins.

    Looked up in a map built from `t0_events` by
    `plotlib.trial_traces.interval_x0_map` - t0_events is the authority on interval
    geometry. This deliberately no longer reads the `n_bins_left` column stamped on the
    fit rows (that column is being removed), and no longer falls back to the old
    hardcoded "3 for cue_entry, else 10", which silently disagreed with the data once
    R1_entry_interval was retuned from 10 to 4 bins.

    Raises rather than guessing: a missing entry means the caller was given a t0_events
    table that does not cover this interval, which is exactly the mismatch that used to
    produce misaligned figures with no error.
    """
    if not interval_x0s or which_interval not in interval_x0s:
        raise ValueError(
            f"no t0 geometry for {which_interval!r}. Pass interval_x0s built with "
            f"plotlib.trial_traces.interval_x0_map(t0_events, which_intervals); "
            f"got {sorted(interval_x0s) if interval_x0s else None}")
    return interval_x0s[which_interval]


# _interval_short_label: now imported from plotlib.trial_traces (was a duplicate of dim_plotting_lib's)


# One output column = one *block spec*. A plain string names the target and nothing else
# (the v1 form: whatever rows that target has, with the CCGP direction resolved by
# ccgp_w_train_stratum), and that is still what every existing call site passes. A dict
# additionally pins which rows of the target are meant, which is what the v2 table needs:
# it holds the unconditioned fit AND both conditioned directions for the same target in
# ONE frame (see svm_task_subspaces.py), so "choice_R1" alone no longer identifies a
# single line. Keys:
#   y_name        - the target (required)
#   analysis      - 'naive' (unconditioned rows) or 'ccgp' (conditioned rows); inferred as
#                   'ccgp' when conditioner/train_stratum is set, otherwise "don't filter"
#   conditioner   - restrict to rows conditioned on this other target
#   train_stratum - ELECT the direction trained on this conditioner level: it gets the
#                   star, and its W is what the heatmap/variance row show. It does not
#                   drop the other direction - show_ccgp_secondary still decides whether
#                   that one's line is drawn in row 1 (they are independent: one says
#                   which direction this column is *about*, the other how much is drawn)
#   label         - what the column's title/W-block says instead of y_name
# So the naive-vs-CCGP comparison the v2 table was written for is one call:
#   y_predicted_names=['choice_R1',
#                      'cueXORchoice_R1',
#                      {'y_name': 'choice_R1', 'analysis': 'ccgp', 'train_stratum': 1}]
# and with show_w_comparison=True those three blocks' W's are compared against each other
# in the same figure (which is what plot_session_cross_analytic_w_comparison had to be
# used for while naive and CCGP lived in separate analytics).
COLUMN_SPEC_KEYS = ('y_name', 'analysis', 'conditioner', 'train_stratum', 'label')


def _as_column_spec(y_entry):
    """One y_predicted_names entry (str or dict) -> a full column-spec dict."""
    if not isinstance(y_entry, dict):
        return dict(y_name=y_entry, analysis=None, conditioner=None, train_stratum=None, label=None)
    unknown = set(y_entry) - set(COLUMN_SPEC_KEYS)
    if unknown:
        raise ValueError(f"unknown column-spec key(s) {sorted(unknown)}; known: {list(COLUMN_SPEC_KEYS)}")
    if y_entry.get('y_name') is None:
        raise ValueError(f"column spec {y_entry} needs a 'y_name'")
    spec = {k: y_entry.get(k) for k in COLUMN_SPEC_KEYS}
    if spec['analysis'] is None and (spec['conditioner'] is not None or spec['train_stratum'] is not None):
        spec['analysis'] = 'ccgp'
    if spec['analysis'] not in (None, 'naive', 'ccgp'):
        raise ValueError(f"column spec {y_entry}: analysis must be 'naive', 'ccgp' or None")
    return spec


def _column_title(col_spec):
    """What a column is called in its title, W-block label and markdown heading."""
    return col_spec['label'] or col_spec['y_name']


def _resolve_column_specs(x_modalities, y_predicted_names):
    """Build the block spec (x_modality + _as_column_spec keys) for each output column.
    One of x_modalities/y_predicted_names may be a single value (broadcast across the
    other's columns); if both are lists, every combination gets its own column
    (multiplexed)."""
    x_list = list(x_modalities) if isinstance(x_modalities, (list, tuple)) else [x_modalities]
    y_list = ([_as_column_spec(y) for y in y_predicted_names]
              if isinstance(y_predicted_names, (list, tuple)) else [_as_column_spec(y_predicted_names)])
    if len(x_list) > 1 and len(y_list) > 1:
        return [dict(spec, x_modality=xm) for xm in x_list for spec in y_list]
    if len(x_list) == 1:
        x_list = x_list * len(y_list)
    if len(y_list) == 1:
        y_list = y_list * len(x_list)
    return [dict(spec, x_modality=xm) for xm, spec in zip(x_list, y_list)]


def _stack_by_timebin(rows_df, column):
    """rows_df: one row per timebin for a single (interval, predict_y_name, X_modality).
    Returns an (n_trials, n_timebins) array, sorted by timebin.

    A collapsed fit (v2's fit_collapsed - the labels could not be separated at all, so no
    model was kept) stores an EMPTY predictions list where every other row of the group
    has one entry per trial. Padding those to NaN keeps the array rectangular and makes
    that timebin unscorable everywhere downstream (_bootstrap_aggr_ci and the covariate
    panels already drop NaN predictions), instead of np.stack raising on ragged input."""
    rows_df = rows_df.sort_values('timebin')
    values = [np.asarray(v, dtype=float).ravel() for v in rows_df[column].values]
    n = max((len(v) for v in values), default=0)
    padded = [v if len(v) == n else np.full(n, np.nan) for v in values]
    return np.stack(padded).T if padded else np.zeros((0, 0))


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


def _w_rows(rows, n_features):
    """(len(rows), n_features) float array of these rows' weight vectors, NaN for any row
    whose w isn't a full-length vector - i.e. v2's collapsed fits, which store `w=[]`
    rather than a fabricated zero/0.5 direction. NaN propagates through the cosine
    similarity and is drawn blank, so a collapsed timebin reads as a hole in the heatmap
    rather than as a comparable direction."""
    out = np.full((len(rows), n_features), np.nan)
    for i, w in enumerate(rows.w):
        w = np.asarray(w, dtype=float).ravel()
        if w.size == n_features:
            out[i] = w
    return out


def _w_block_with_interval_separators(ordered_svm_res, which_intervals, metric_col=None, smooth=None,
                                       interval_x0s=None):
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
    # max, not the first row's length: a collapsed fit (v2's fit_collapsed) stores an
    # empty w, and _w_rows turns any row that isn't full-length into a NaN row - which
    # comes out of the cosine similarity as NaN and is drawn as a blank cell, the same
    # treatment an interval separator gets. Taking len(w.iloc[0]) instead would silently
    # produce a 0-feature block whenever the first timebin happened to be the collapsed one.
    n_features = max(len(w) for w in ordered_svm_res.w)
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
        parts.append(_w_rows(rows, n_features))
        if metric_col is not None:
            metric_parts.append(_smooth(rows[metric_col].to_numpy(dtype=float), smooth))
        t0_positions.append(pos + _interval_x0(interval_x0s, interval))
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
    into `primary` (the lowest fitted train_stratum) and `secondary` (the other
    direction). Naive rows (train_stratum is None) have nothing to split - they pass
    straight through as `primary`, and `secondary` is empty.

    Which direction lands in `primary` is deliberately a *fixed* rule and not a property
    of the data: this used to elect the `more_reliable` row (the one trained on the larger
    stratum), which is a per-session flag, so it flipped between sessions whenever the two
    strata were close and a session sequence silently mixed C1-trained and C2-trained
    fits. Nothing reads more_reliable any more (v2 doesn't even store it). Pin the
    direction you mean - per column via a block spec's `train_stratum` (see
    _as_column_spec) or per target via ccgp_w_train_stratum - rather than relying on
    this ordering to pick for you."""
    if 'train_stratum' not in df.columns or df['train_stratum'].isna().all():
        return df, df.iloc[0:0]

    is_ccgp = df['train_stratum'].notna()
    naive_rows, ccgp_rows = df[~is_ccgp], df[is_ccgp]

    group_cols = ['interval_name', 'timebin', 'X_modality', 'predict_y_name', 'conditioner']
    ccgp_sorted = ccgp_rows.sort_values(group_cols + ['train_stratum'])
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
    the lowest fitted stratum (_split_ccgp_primary_secondary's `primary`, a fixed rule -
    see there) only when the dict doesn't mention this target at all. Returns None for naive fits
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
    than blindly keeping whichever direction _split_ccgp_primary_secondary put first.
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
    """(local_row_index, name) pairs for rows in `ordered_svm_res` (already filtered to a
    single session/target/conditioner/direction block) that any of `selections` picks out.
    local_row_index is in the same 0-indexed row order _w_block_with_interval_separators
    itself builds (i.e. including the +1 it adds for each inserted separator row), for use
    as plot_w_comparison's highlight_positions row_index; `name` is the matched selection's
    own optional `name` key (a short human label, e.g. 'shared_choice_07'), or None.
    `constant_fields` (e.g. session_id=, y_name=, conditioner=) are the same for every row
    in this block and checked alongside each row's own interval/timebin/train_stratum.

    Deliberately a *drawing-position* helper only: it returns where to put a box and what
    to write next to it, never the fitted row behind it. Getting the actual W's for a set
    of selections is resolve_w_selections' job, which runs off svm_results directly and
    needs no figure - see there."""
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
                out.append((pos + offset, matched_sel.get('name')))
        pos += n_bins
    return out


# Fields a selection dict may pin -> the svm_results column each one filters on.
# `session_id` is an index level, not a column, so it's handled separately below.
SELECTION_FIELD_COLUMNS = {'y_name': 'predict_y_name', 'conditioner': 'conditioner',
                           'train_stratum': 'train_stratum', 'interval': 'interval_name',
                           'timebin': 'timebin'}


def resolve_w_selections(selections, svm_results, x_modality=None, ccgp_w_train_stratum=None,
                          verbose=True):
    """The selection dicts -> the actual fitted W's. No figure involved.

    This is the only way to get W's out of a selection: the plotting functions take
    `highlight_selections` purely to draw a callout box and never hand data back (a plot
    call and a data query are different questions, and tying them together meant a W could
    only be obtained as a side effect of rendering a figure).

    `selections` is the same list of dicts the plots take (session_id/y_name/conditioner/
    train_stratum/interval/timebin, plus an optional `name`; only the keys a dict actually
    sets constrain the match - see _selection_matches). Each dict gets its `w` key filled
    in **in place**, so the selection list itself becomes the record of what was picked;
    a selection that didn't resolve to exactly one row gets `w = None` (and a printed
    reason, with verbose). Also returns the matched rows as a DataFrame - one row per
    resolved selection, carrying the full fitted row (`w`, `acc_aggr`, ...) plus
    `session_id` and `name` - which is what the downstream W figures want (see
    w_plotting_lib's `w_row` argument).

    `ccgp_w_train_stratum` must match whatever the figures were drawn with: a selection
    that doesn't pin `train_stratum` itself would otherwise match both CCGP directions and
    resolve to nothing. See _apply_ccgp_election / _resolve_ccgp_w_stratum."""
    df = svm_results
    if x_modality is not None:
        df = df[df['X_modality'] == x_modality]
    # Per session, NOT over the whole frame at once. _apply_ccgp_election ->
    # _resolve_ccgp_w_stratum decides "is this target CCGP here?" from the group it is
    # handed; pooled across sessions, one session's CCGP rows make the whole target look
    # CCGP, and _rows_for_stratum then drops every naive-only session's rows (their
    # train_stratum is NaN, so they match no stratum). That silently loses exactly the
    # early sessions that were never CCGP-fit. Session-wise election also matches how
    # plot_session_analysis/get_session_sequence_Ws elect, one session at a time.
    df = pd.concat([_apply_ccgp_election(g, ccgp_w_train_stratum)
                    for _, g in df.groupby(df.index.get_level_values('session_id'), sort=False)])

    matched = []
    for sel in selections:
        mask = pd.Series(True, index=df.index)
        if sel.get('session_id') is not None:
            mask &= df.index.get_level_values('session_id') == sel['session_id']
        for key, col in SELECTION_FIELD_COLUMNS.items():
            if sel.get(key) is not None:
                mask &= df[col] == sel[key]
        rows = df[mask]

        if len(rows) != 1:
            sel['w'] = None
            if verbose:
                why = 'no fitted row matches' if rows.empty else f'{len(rows)} rows match (ambiguous)'
                print(f"resolve_w_selections: {sel.get('name') or sel} -> {why}")
            continue

        row = rows.iloc[0].copy()
        # session_id lives in the index, not as a column - stamp it on so a frame pooled
        # across sessions doesn't lose track of where each W came from
        row['session_id'] = rows.index.get_level_values('session_id')[0]
        row['name'] = sel.get('name')
        sel['w'] = row['w']
        matched.append(row)

    if verbose:
        print(f"resolve_w_selections: {len(matched)}/{len(selections)} selections resolved")
    return pd.DataFrame(matched).reset_index(drop=True) if matched else pd.DataFrame()


def _check_one_row_per_timebin(df, what):
    """Guard the invariant every per-timebin routine below assumes: one row per
    (interval_name, timebin) for the block being drawn.

    It only ever breaks one way, and silently: a v2 fit table holds a target's
    unconditioned fit AND its two conditioned directions in one frame, so selecting rows
    by predict_y_name alone gives three overlapping lines drawn on top of each other (and
    a W block three times too tall, mis-registered against row 1). Say so, with the fix,
    rather than letting the figure come out wrong."""
    if df.empty:
        return
    dupes = df.duplicated(subset=['interval_name', 'timebin'], keep=False)
    if dupes.any():
        n = int(dupes.sum())
        raise ValueError(
            f"{what}: {n} rows share an (interval_name, timebin) - this block is more than one fit. "
            f"Pin which one with a block spec, e.g. analysis='naive' for the unconditioned fit or "
            f"analysis='ccgp' + train_stratum=<0|1> for one conditioned direction "
            f"(see _as_column_spec).")


def _filter_analysis(df, analysis):
    """Restrict `df` to unconditioned ('naive'), conditioned ('ccgp') or matched-control
    ('ccgp_control') fits; unchanged if `analysis` is None.

    On a v2 table the `analysis` column decides, because train_stratum no longer can: a
    conditioned fit's matched control deliberately carries the SAME conditioner and
    train_stratum as its twin (that is the join key between them), so a train_stratum test
    would return both and every consumer downstream would see two rows per timebin.

    The train_stratum fallback is what makes this mean the same thing on a v1 table, which
    has no `analysis` column at all: v1's naive tables have no train_stratum either (every
    row is unconditioned) and its CCGP table has one on every row. v2's paired
    'baseline_vs_t' rows are unconditioned and count as naive under both branches, which is
    right - they are a different predict_y_name, never the same line."""
    if analysis is None:
        return df
    if 'analysis' in df.columns and df['analysis'].notna().any():
        if analysis == 'naive':
            return df[~df['analysis'].isin(('ccgp', 'ccgp_control'))]
        return df[df['analysis'] == analysis]
    if 'train_stratum' not in df.columns:
        return df
    is_ccgp = df['train_stratum'].notna()
    return df[is_ccgp] if analysis == 'ccgp' else df[~is_ccgp]


def get_session_data(s_id, t0_events, behavior, svm_results, x_modality, y_predicted_name, which_intervals,
                      ccgp_w_train_stratum=None, analysis=None, conditioner=None,
                      need_positions=True):
    """All data needed to plot one (session, x_modality, y_predicted_name) column. Every
    per-timebin plotting routine below (create_subplot, _plot_f1_lines, plot_covariate_f1_single,
    ...) assumes exactly one row per (interval, timebin, X_modality, predict_y_name) - naive
    fits satisfy this directly, CCGP fits need `_split_ccgp_primary_secondary` first (see
    there - CCGP stores both train/test directions as separate rows, unlike a naive fit).
    `ccgp_w_train_stratum`, if given, only affects interval_svm_results (see below) -
    target_svm_results/target_svm_results_secondary are returned as the raw
    primary/secondary split regardless, since the caller (plot_session_analysis) resolves
    the current column's own election itself and needs both directions available.

    `analysis`/`conditioner` (from a column's block spec - see _as_column_spec) narrow the
    target rows *before* that split. They are what makes a v2 table plottable at all: it
    holds the unconditioned fit and both conditioned directions for one target in the same
    frame, so without them a "column" would be three overlapping lines. On a v1 table they
    are simply no-ops (its naive and CCGP fits live in separate analytics).

    Note what is deliberately NOT filtered here: which train_stratum a column *elects* (a
    spec's own `train_stratum`, or ccgp_w_train_stratum). Both directions must survive
    into the split, or show_ccgp_secondary would have nothing left to draw - electing is
    the caller's job (see plot_session_analysis), filtering is not.

    `need_positions=False` skips the row-0 trajectory extraction entirely, which is the
    only thing here that touches `behavior` (which may then be None)."""
    target_svm_results = svm_results.loc[
        (svm_results.index.get_level_values('session_id') == s_id) &
        (svm_results['X_modality'] == x_modality) &
        (svm_results['predict_y_name'] == y_predicted_name) &
        (svm_results['interval_name'].isin(which_intervals))
    ]
    target_svm_results = _filter_analysis(target_svm_results, analysis)
    if conditioner is not None:
        target_svm_results = target_svm_results[target_svm_results['conditioner'] == conditioner]
    target_svm_results, target_svm_results_secondary = _split_ccgp_primary_secondary(target_svm_results)
    _check_one_row_per_timebin(target_svm_results, f'{y_predicted_name} / {x_modality} / {s_id}')

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
    # Same `analysis` as the column itself: on a v2 table every sibling target also has
    # both an unconditioned and a conditioned version, and a covariate panel that mixed
    # them would be splitting the column's trials by a label vector taken from a
    # different fit.
    interval_svm_results = _filter_analysis(interval_svm_results, analysis)
    # _apply_ccgp_election (not a blanket _split_ccgp_primary_secondary): this table
    # spans every predict_y_name fit here, e.g. the 'cue' covariate panel while the
    # column itself decodes choice_R1 - each target gets its own election from
    # ccgp_w_train_stratum (or automatic, same fallback as elsewhere), so a covariate
    # panel doesn't silently disagree with what the caller explicitly requested for it.
    interval_svm_results = _apply_ccgp_election(interval_svm_results, ccgp_w_train_stratum)

    if not need_positions:
        return target_svm_results, target_svm_results_secondary, interval_svm_results, {}, None

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


# _smooth: now imported from plotlib.trial_traces (was a duplicate of dim_plotting_lib's)


def _ccgp_n_annotation(conditioner, y_name, primary_rows, secondary_rows, elected_stratum=None):
    """Bottom-right corner text for a CCGP fit, replacing the naive n_pos/n_neg
    annotation - two lines that state the fit METHOD rather than report numbers to be read
    off precisely:

        Train: stop (n_C1: 49, n_C2: 26)
        Test: skip (n_C1: 37, n_C2: 22)

    i.e. the direction shown was fit on one level of the conditioner and scored on the
    other, which is the whole claim a CCGP curve makes. Each line's n's are that stratum's
    OWN class balance for the target (n_train_neg/n_train_pos of the direction trained
    there), not the target's overall balance across both strata; the Test line's counts
    come from the opposite direction's rows, that stratum being exactly the trials this
    direction was scored on. They are constant across timebins for a given direction (the
    trial-level split doesn't depend on timebin - see fit_CCGP_with_bootstrap), so any
    row's value works and `.iloc[0]` just picks the first.

    `elected_stratum` (see _resolve_ccgp_w_stratum) decides which of the two directions is
    the Train line - the one whose W row 2's heatmap actually shows, so the corner text and
    the heatmap agree on which direction is "the" one even when ccgp_w_train_stratum
    overrides the default (lowest) stratum. That ordering replaces the star (★) this text
    used to mark the elected line with: with the two lines named Train/Test there is
    nothing left for a star to disambiguate. With only one direction stored, only the
    Train line is drawn. The reminder that the two directions are each other's train/test
    swap is drawn separately (see _plot_f1_lines)."""
    def _one(rows, role):
        stratum = rows['train_stratum'].dropna().iloc[0]
        neg = int(rows['n_train_neg'].dropna().iloc[0])
        pos = int(rows['n_train_pos'].dropna().iloc[0])
        return (f"{role}: {_level_label(conditioner, stratum)} "
                f"($n_{{{_level_label(y_name, 0)}}}$: {neg}, $n_{{{_level_label(y_name, 1)}}}$: {pos})")

    if secondary_rows is None or secondary_rows.empty:
        return _one(primary_rows, 'Train')
    train_rows, test_rows = primary_rows, secondary_rows
    # The elected direction is the one being shown, so it is the one named "Train" -
    # _split_ccgp_primary_secondary's own ordering (lowest stratum first) has nothing to do
    # with which direction ccgp_w_train_stratum picked.
    if (elected_stratum is not None
            and secondary_rows['train_stratum'].dropna().iloc[0] == elected_stratum):
        train_rows, test_rows = secondary_rows, primary_rows
    return "\n".join([_one(train_rows, 'Train'), _one(test_rows, 'Test')])


def _draw_min_metric_indicator(ax, x_bins, values, y_name, min_metric, y_frac=None):
    """A discontinuous marker along the bottom of the score axes: drawn where `values`
    (the score already smoothed exactly as the line above is) reaches `min_metric`, absent
    where it does not.

    Two narrow lines stacked rather than one, coloured with the decoded target's OWN two
    level colours (modality_colors - orange/purple for cue, the two blues for choice_R1),
    because "the decoder worked here" is a statement about separating those two classes:
    the same pair of colours the covariate panels and the condition curves use for them,
    so a reader who has learnt the colours anywhere else in the figure reads this without
    a legend. A target with no colour entry (a joined/derived label) falls back to gray
    for both, which still shows where the threshold was cleared.

    Positioned in a blended transform - x in data coords, y in axes fraction - so it stays
    pinned to the bottom of the box whatever the row's ylim is, and NaN'd rather than
    segmented by hand: a gap in a line is what makes this readable as an on/off indicator
    instead of a fifth curve. Solid butt caps so a single above-threshold timebin still
    draws a visible tick of its own rather than rounding away.

    The two are separated by exactly one linewidth in POINTS (offset_copy), not by a step
    in axes fraction: they have to touch - one two-tone strip, not two lines with a gap -
    and a fraction that does that on one panel opens a gap on the next, since these rows
    differ in height with the number of variance/activation rows above them.

    `y_frac` (axes fraction) moves the pair off the bottom edge: the score row wants it at
    the bottom (MIN_METRIC_INDICATOR_Y, the default), but a row that is nothing BUT these
    strips - plot_session_cue_choice_alignment's top row - stacks one target's pair above
    another's and so has to place each explicitly."""
    above = np.asarray(values, dtype=float) >= min_metric
    if not above.any():
        return
    x = np.asarray(x_bins, dtype=float)
    y_frac = MIN_METRIC_INDICATOR_Y if y_frac is None else y_frac
    for level, dy in ((0, 0.5 * MIN_METRIC_INDICATOR_LW), (1, -0.5 * MIN_METRIC_INDICATOR_LW)):
        transform = offset_copy(ax.get_xaxis_transform(), fig=ax.figure, y=dy, units='points')
        ax.plot(x, np.where(above, y_frac, np.nan), transform=transform,
                color=_covariate_color(y_name, level), linewidth=MIN_METRIC_INDICATOR_LW,
                solid_capstyle='butt', zorder=12, clip_on=False)


def _ccgp_sentence_segments(conditioner, y_name, elected_stratum, test_stratum):
    """The one-row form of the CCGP annotation, as (text, textprops) segments:

        Train on skip trials, Test on stop:  C1 vs C2

    The method in small plain type, the two CLASSES the decoder separates in bold and in
    their own colours (modality_colors, the same orange/purple and blues everything else in
    the figure uses for them) - so the sentence names the design and the colours name the
    contrast, without a legend and without counts. The counts form (_ccgp_n_annotation) is
    the alternative for when the exact class balance is what is being checked; see
    plot_session_analysis's `ccgp_annotation`.

    `conditioner` None (a naive fit) drops the leading clause and leaves just the coloured
    class pair, which is the whole of what a naive fit has to say here.
    """
    small = dict(fontsize=CORNER_FONTSIZE, color='black')
    bold = dict(fontsize=CORNER_FONTSIZE + 2, fontweight='bold')
    segments = []
    if conditioner is not None and elected_stratum is not None:
        clause = f'Train on {_level_label(conditioner, elected_stratum)} trials'
        if test_stratum is not None:
            clause += f', Test on {_level_label(conditioner, test_stratum)}'
        segments.append((clause + ':  ', small))
    segments += [(_level_label(y_name, 0), dict(bold, color=_covariate_color(y_name, 0))),
                 (' vs ', small),
                 (_level_label(y_name, 1), dict(bold, color=_covariate_color(y_name, 1)))]
    return segments


def _draw_text_rows(ax, rows, xy, ha='right', va='top', in_layout=True):
    """One or more lines of text whose parts differ in size, weight and colour - matplotlib
    has no rich-text Text, so the parts are packed side by side (and the lines above each
    other) as an offsetbox instead of being positioned by measuring each one's rendered
    width against a renderer (which needs a draw to have happened, and gets it wrong at
    save time when the dpi changes).

    `rows` is a list of rows, each a list of (text, textprops) segments - a second row is
    how the same annotation fits on a panel too narrow for it on one line (e.g. a
    single-interval panel, 1.4in wide).

    `xy` is in axes coordinates, `ha`/`va` say which corner of the packed block lands on
    it. align='baseline' is what keeps a 9pt bold word sitting on the same line as a 7pt
    one. `in_layout=False` hides the whole block from tight_layout: an annotation wider
    than its own axes otherwise makes tight_layout shrink the AXES to fit the text, which
    on a narrow panel crushes the plot to a sliver - overflowing the panel edge is the
    lesser evil there."""
    packed = [HPacker(children=[TextArea(text, textprops=props) for text, props in row],
                      align='baseline', pad=0, sep=0) for row in rows]
    box = packed[0] if len(packed) == 1 else VPacker(children=packed, align='left', pad=0, sep=2)
    loc = {('right', 'top'): 'upper right', ('right', 'bottom'): 'lower right',
           ('left', 'top'): 'upper left', ('left', 'bottom'): 'lower left'}[(ha, va)]
    artist = AnchoredOffsetbox(loc=loc, child=box, pad=0.0, borderpad=0.0,
                                frameon=False, bbox_to_anchor=xy,
                                bbox_transform=ax.transAxes)
    artist.set_in_layout(in_layout)
    ax.add_artist(artist)


def _draw_text_segments(ax, segments, xy, ha='right', va='top'):
    """One line of _draw_text_rows - see there."""
    _draw_text_rows(ax, [segments], xy, ha=ha, va=va)


def _plot_f1_lines(ax, svm_res_target, which_intervals, ylabel, y_name, extra_lines=None, show_ci=False,
                    show_aggr_ci=True, show_n=True, extra_alpha=0.8, target_color='black', target_alpha=0.95,
                    target_linewidth=1, metric='f1', show_shuffle=False, null_kind='perm', smooth=None,
                    show_ccgp_secondary=False, svm_res_secondary=None, ccgp_w_elected_stratum=None,
                    min_metric=None, ccgp_annotation='counts'):
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
    not applied per CCGP direction. With show_shuffle, also draws the null-control mean
    (the mean of that row's stored null draws - see _null_mean_values) as a dashed black
    line, when this table carries the requested `null_kind` ('perm' everywhere; 'shift',
    the drift-preserving null, on v2 tables only - see METRIC_SPECS). `smooth`, if given (e.g. 3), is a rolling-mean window applied to every line
    (and the CI band) before plotting - see _smooth - separately per interval, so
    smoothing never blends values across an interval boundary. Concatenates intervals
    left-to-right on the figure-wide x-axis, which is the row-2 heatmap's own block-row
    index: see plotlib.trial_traces.block_x/next_block_offset (each interval's rows, then
    one separator row - the NaN row _w_block_with_interval_separators inserts).
    show_aggr_ci (naive fits only, default True) draws a bootstrap CI band around the aggr
    line itself, computed at plot time via _bootstrap_aggr_ci from that row's stored
    y_true/predictions - independent of `show_ci` (which instead bands the *mean* line
    with the stored, precomputed ci_lower_col/ci_upper_col - a different quantity, see
    _bootstrap_aggr_ci's docstring for why the two aren't interchangeable). `min_metric`,
    if given, adds the threshold indicator along the bottom of the axes (see
    _draw_min_metric_indicator) - the same threshold that masks row 2's heatmap, so the
    reader can see which timebins the heatmap is showing without reading the score line
    against a gridline. Returns the final offset."""
    spec = METRIC_SPECS[metric]
    offset = 0
    n_pos = n_neg = None
    n_annotation = None
    # (conditioner, train_stratum, test_stratum) for the one-row sentence form - captured on
    # the first conditioned interval, same as n_annotation is
    sentence_ctx = None
    any_ccgp = False
    has_secondary = show_ccgp_secondary and svm_res_secondary is not None and not svm_res_secondary.empty
    for i, which_interval in enumerate(which_intervals):
        target_rows = svm_res_target[svm_res_target.interval_name == which_interval].sort_values('timebin')
        if target_rows.empty:
            continue
        # _block_x: x is the heatmap's own block-row index, not the raw 1-based timebin
        # (see plotlib.trial_traces.block_x) - `n_bins` is what the offset advances by.
        n_bins = len(target_rows)
        x_bins = _block_x(target_rows.timebin.values, offset)
        first = offset == 0  # first interval actually drawn - only label it once

        # is_conditioned: these rows are a CCGP direction at all. is_ccgp: *and* the other
        # direction is being drawn alongside it, which is what turns row 1 into the
        # symmetric two-line draw. A column pinned to one direction (a block spec's
        # train_stratum) is conditioned but not two-line - it keeps the single black
        # line, and only its corner annotation changes (per-stratum n's, not the target's
        # overall balance, which is not what that direction was trained on).
        is_conditioned = ('conditioner' in target_rows.columns and target_rows['conditioner'].notna().any())
        conditioner = target_rows['conditioner'].dropna().iloc[0] if is_conditioned else None
        is_ccgp = has_secondary and is_conditioned
        # The other direction's rows, resolved whenever they EXIST rather than only when
        # they are drawn: the corner annotation names both directions (Train/Test - see
        # _ccgp_n_annotation) even on a panel pinned to one of them, which is a panel where
        # show_ccgp_secondary is off and so is_ccgp is False.
        secondary_rows_all = (
            svm_res_secondary[svm_res_secondary.interval_name == which_interval].sort_values('timebin')
            if is_conditioned and svm_res_secondary is not None and not svm_res_secondary.empty else None)
        secondary_rows = secondary_rows_all if is_ccgp else None

        if is_ccgp:
            for rows in (target_rows, secondary_rows):
                if rows is None or rows.empty:
                    continue
                stratum = rows['train_stratum'].dropna().iloc[0]
                color = _covariate_color(conditioner, stratum)
                rows_x_bins = _block_x(rows.timebin.values, offset)
                ax.plot(rows_x_bins, _smooth(rows[spec['aggr_col']].values, smooth), color=color,
                        linewidth=target_linewidth, alpha=target_alpha, zorder=10,
                        label=f'train {_level_label(conditioner, stratum)}' if first else None)
                if show_ci:
                    # see naive branch below for why show_ci is only meaningful with show_mean
                    ax.fill_between(rows_x_bins, _smooth(rows[spec['ci_lower_col']].values, smooth),
                                     _smooth(rows[spec['ci_upper_col']].values, smooth),
                                     color=color, alpha=0.07, edgecolor=None, zorder=1)
        else:
            ax.plot(x_bins, _smooth(target_rows[spec['aggr_col']].values, smooth), color=target_color,
                    linewidth=target_linewidth, alpha=target_alpha, zorder=10,
                    label=spec['aggr_col'] if first else None)
            if show_ci:
                # NB: ci_lower_col/ci_upper_col are bootstrap percentiles of the per-iteration
                # scores (aligned with mean_col), not of aggr_col - drawing them around the aggr
                # line (the default target line) would be showing a CI that doesn't actually
                # belong to it. Only safe to enable together with show_mean.
                ax.fill_between(x_bins, _smooth(target_rows[spec['ci_lower_col']].values, smooth),
                                 _smooth(target_rows[spec['ci_upper_col']].values, smooth),
                                 color='black', alpha=0.07, edgecolor=None, zorder=1)
            if show_aggr_ci:
                # a CI that actually belongs to the aggr line - see _bootstrap_aggr_ci.
                y_true_2d = _stack_by_timebin(target_rows, 'y_true')
                pred_2d = _stack_by_timebin(target_rows, 'predictions')
                rng = np.random.default_rng(0)
                aggr_ci = np.array([_bootstrap_aggr_ci(y_true_2d[:, t], pred_2d[:, t], metric, rng=rng)
                                     for t in range(y_true_2d.shape[1])])
                ax.fill_between(x_bins, _smooth(aggr_ci[:, 0], smooth), _smooth(aggr_ci[:, 1], smooth),
                                 color=target_color, alpha=0.15, edgecolor=None, zorder=2)

        if show_shuffle:
            null_values, _ = _null_mean_values(target_rows, metric, null_kind)
            if null_values is not None:
                ax.plot(x_bins, _smooth(null_values, smooth), color='black',
                        linestyle='--', linewidth=1.2, alpha=0.65, zorder=9,
                        label=f'{null_kind} null' if first else None)

        for values, color, label, ls in (extra_lines(which_interval, target_rows) if extra_lines else []):
            ax.plot(x_bins, _smooth(values, smooth), color=color, linestyle=ls, alpha=extra_alpha,
                    zorder=11, label=label if first else None)

        if is_conditioned:
            any_ccgp = any_ccgp or is_ccgp
            if n_annotation is None:
                n_annotation = _ccgp_n_annotation(conditioner, y_name, target_rows, secondary_rows_all,
                                                   elected_stratum=ccgp_w_elected_stratum)
            if sentence_ctx is None:
                train_stratum = ccgp_w_elected_stratum
                if train_stratum is None:
                    train_stratum = target_rows['train_stratum'].dropna().iloc[0]
                # the level this direction was SCORED on: the other direction's own stratum
                # where that direction is present, and otherwise the conditioner's other
                # level - a CCGP fit trains on one level of a binary conditioner and tests
                # on the other by construction (see get_SVMCueOutcomeChoicePred's ccgp arg)
                if secondary_rows_all is not None and not secondary_rows_all.empty:
                    test_stratum = secondary_rows_all['train_stratum'].dropna().iloc[0]
                    if test_stratum == train_stratum:
                        test_stratum = target_rows['train_stratum'].dropna().iloc[0]
                else:
                    test_stratum = 1 - train_stratum if train_stratum in (0, 1) else None
                sentence_ctx = (conditioner, train_stratum, test_stratum)
        elif n_pos is None:
            pos_col, neg_col = _first_present(target_rows, N_POS_COLS), _first_present(target_rows, N_NEG_COLS)
            if pos_col is not None and neg_col is not None:
                n_pos, n_neg = target_rows[pos_col].values[0], target_rows[neg_col].values[0]

        # Drawn per interval, off the elected/primary direction's own scores and smoothed
        # exactly as the line above is - the threshold has to agree both with what the
        # reader sees and with the mask row 2's heatmap applies (which smooths identically,
        # see _w_block_with_interval_separators).
        if min_metric is not None:
            _draw_min_metric_indicator(ax, x_bins, _smooth(target_rows[spec['aggr_col']].values, smooth),
                                        y_name, min_metric)

        offset = _next_block_offset(offset, n_bins, is_last=i == len(which_intervals) - 1)

    ax.set_ylabel(ylabel)
    ax.set_ylim(.1, .9)
    # Nothing drawn at .5 itself (no chance-level axhline, no grid line there) - just the
    # tick label. Manual grid at .25/.75 only, not ax.yaxis.grid(True, ...) at every tick,
    # which would put one at .5 too.
    for gridline_y in (0.25, 0.75):
        ax.axhline(gridline_y, color='k', linestyle='--', alpha=0.5, zorder=0, linewidth=0.5)
    if show_n:
        # Under the axes, not in a corner inside it: on a packed panel (no y label, no tick
        # numbers) the score line uses the full box, and a two-line CCGP annotation
        # (Train/Test - see _ccgp_n_annotation) placed at the bottom right lands straight on
        # whatever the curve happens to do there. The row below is separated by hspace
        # anyway, so this is otherwise-unused space.
        if ccgp_annotation == 'sentence':
            # one row, no counts: the design as a sentence plus the two classes in their own
            # colours (see _ccgp_sentence_segments). Drawn for a naive fit too, as just the
            # class pair - there is no train/test clause to make.
            conditioner_, train_stratum, test_stratum = sentence_ctx or (None, None, None)
            # ABOVE the axes, unlike the counts form: one row of text reads as a caption to
            # the panel there, and it leaves the whole bottom edge to the min_metric
            # indicator. The counts form stays below - two lines up there would run into the
            # session's own suptitle.
            _draw_text_segments(ax, _ccgp_sentence_segments(conditioner_, y_name,
                                                             train_stratum, test_stratum),
                                 xy=(1.0, 1.02), va='bottom')
        else:
            n_text = n_annotation if n_annotation is not None else (
                f"$n_{{{_level_label(y_name, 0)}}}={n_neg}$, $n_{{{_level_label(y_name, 1)}}}={n_pos}$"
                if n_pos is not None else None)
            if n_text is not None:
                ax.text(1.0, -0.04, n_text, transform=ax.transAxes, fontsize=CORNER_FONTSIZE,
                        verticalalignment='top', horizontalalignment='right')
    if any_ccgp:
        # top-left, inside the axes - the per-stratum n_annotation now sits under the axes
        # (see above), so this only has to stay clear of the curve's own start.
        ax.text(0.05, 0.95, '⇅ train↔test', transform=ax.transAxes, fontsize=CORNER_FONTSIZE - 1,
                alpha=0.7,
                verticalalignment='top', horizontalalignment='left')
    return offset


def create_subplot(axes, all_pos, svm_res, title, y_predicted_name, which_intervals, metric='f1',
                    show_shuffle=False, null_kind='perm', show_scatter=False, show_mean=False, show_aggr_ci=True,
                    draw_cue_zone=True, smooth=None, show_ylabel=True, score_ylabel=None, show_title=True,
                    show_ccgp_secondary=False, svm_res_secondary=None, ccgp_w_elected_stratum=None,
                    show_trial_kinematics=True, interval_x0s=None, min_metric=None,
                    ccgp_annotation='counts'):
    """Row 0: single-trial trajectories; with show_scatter, also a per-timebin mean-position
    scatter colored by decode score (metric='f1' or 'balanced_accuracy' - see METRIC_SPECS) -
    off by default, since row 1 already shows that same per-timebin score as a proper line.
    `min_metric`, if given, also puts that threshold's own on/off indicator along the
    bottom of row 1 (see _draw_min_metric_indicator).
    Row 1: score over time (via _plot_f1_lines) - by default just the aggr score (verified
    against mean-across-bootstraps many times; they match closely enough that drawing both
    is redundant) as a solid black line, with show_aggr_ci (default True) drawing a CI band
    around it too - a genuine bootstrap CI of the aggr score itself, computed at plot time
    (see _bootstrap_aggr_ci), not the stored ci_lower_col/ci_upper_col (those belong to the
    *mean* score, a different quantity - see _plot_f1_lines); show_mean also draws the mean
    score as a thin gray line, with its own (correctly aligned) stored CI band; show_shuffle
    also draws the shuffle-control null mean as a dashed black line, when present. Every row is drawn on the row-2
    heatmap's own x-axis - block-row index, one separator row between intervals, see
    plotlib.trial_traces.block_x/next_block_offset - since sharex=True ties the axes'
    *limits* together but not their coordinates. Since each interval is its own
    plot()/scatter() call, the separator is just empty space, no dummy data point needed. draw_cue_zone controls row 0's cue-zone illustration - see the call site
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
    # Row 2's t0 lines are drawn after the loop, once every block's extent is known - see
    # there for why they are per-block segments rather than full-width axvline/axhline.
    heatmap_t0s, interval_bands = [], []
    pos_min, pos_max = np.inf, -np.inf
    for i, which_interval in enumerate(which_intervals):
        # .get, not [...]: with show_trial_kinematics off the caller may not have loaded
        # any behaviour at all (see get_session_data's need_positions), and rows 1/2 below
        # don't need positions.
        this_interval_pos = all_pos.get(which_interval)
        # sort_values: rows must be timebin-ascending for the offset x-axis below,
        # and to align row-for-row with sibling predict_y_name rows in plot_covariate_f1
        svm_res_interval = svm_res[svm_res.interval_name == which_interval].sort_values('timebin')
        n_bins = len(svm_res_interval.timebin)
        if n_bins == 0:
            # No fitted row for this interval: _w_block_with_interval_separators leaves it
            # out of the block entirely (no rows, and no separator either), so it must not
            # consume x here or get a t0 tick, or everything after it shifts.
            continue

        # x is the row-2 heatmap's own block-row index, not the raw 1-based timebin - see
        # plotlib.trial_traces.block_x/next_block_offset, which every row of this figure
        # shares (sharex=True only ties the *limits* together, not the coordinates).
        x_bins = _block_x(svm_res_interval.timebin.values, offset)

        if show_trial_kinematics and this_interval_pos is not None:
            for pos in this_interval_pos:
                pos_inner = _resample_pos(pos[1:-1], n_bins)
                pos_min, pos_max = min(pos_min, pos_inner.min()), max(pos_max, pos_inner.max())
                axes[0].plot(x_bins, pos_inner, alpha=0.05, color='k', zorder=4)

            # Plot bin boundaries. Solid, not dashed: with one of these per timebin (~30 densely
            # packed, near-adjacent vertical lines per interval), a dashed style's on/off pattern
            # lines up across all of them and produces a Moire effect that reads as fake
            # horizontal white banding through the R1/R2/cue zone fills below.
            for t in x_bins:
                axes[0].axvline(x=t-.5, color='k', linestyle='-', alpha=0.04)

        x_0 = _interval_x0(interval_x0s, which_interval) + offset
        xticks.extend([x_0, x_0+25])
        xtick_labels.extend([f'$t_{{{_interval_short_label(which_interval)}}}$', '1s'])
        # One style for every t0 marker in the figure (T0_LINE_KW) - rows 0/1 as a full-
        # height line, row 2 as per-block segments below.
        if show_trial_kinematics:
            axes[0].axvline(x=x_0, **T0_LINE_KW)
        axes[1].axvline(x=x_0, **T0_LINE_KW)
        heatmap_t0s.append(x_0)
        interval_bands.append((offset - 0.5, offset + n_bins - 0.5))

        if show_trial_kinematics and show_scatter and this_interval_pos is not None:
            mean_pos = _resample_pos(this_interval_pos.mean(axis=0)[1:-1], n_bins)
            scatter = axes[0].scatter(x_bins, mean_pos,
                                c=svm_res_interval[spec['mean_col']].values,
                                cmap=plt.cm.viridis,
                                norm=Normalize(vmin=0.2, vmax=0.8),
                                alpha=1,
                                s=30,
                                zorder=5)

        offset = _next_block_offset(offset, n_bins, is_last=i == len(which_intervals) - 1)

    # Row 2's t0 reference lines: one segment per interval BLOCK, not one full-width
    # axvline/axhline. A full-length line runs straight through the blank separator row/
    # column between blocks (and through the neighbouring blocks' own margins), which
    # closes the gap visually and makes the concatenated blocks read as one continuous
    # matrix. `interval_bands` holds each block's extent in block-row units, the same units
    # on both axes - the matrix is symmetric, so one list serves x and y alike.
    # zorder: a LineCollection defaults to zorder 1, i.e. *under* the pcolormesh the caller
    # draws afterwards (Line2D's axvline defaulted to 2 and so sat on top of it).
    # The dotted style itself is what makes the segmenting legible - see T0_LINE_KW.
    for t0 in heatmap_t0s:
        for band_lo, band_hi in interval_bands:
            axes[2].vlines(t0, band_lo, band_hi, zorder=3, **T0_LINE_KW)
            axes[2].hlines(t0, band_lo, band_hi, zorder=3, **T0_LINE_KW)

    extra_lines = (lambda which_interval, rows: [(rows[spec['mean_col']].values, 'gray', spec['mean_col'], '-')]
                   ) if show_mean else None
    _plot_f1_lines(
        axes[1], svm_res, which_intervals,
        # score_ylabel replaces the auto text, it does not gate it: the auto label
        # ('Decode choice_R1 Balanced Accuracy') is longer than this row is tall on a
        # narrow single-interval panel and gets clipped at the figure edge, so a poster
        # panel passes its own short one. show_ylabel still decides whether there is a
        # label (and tick labels) at all.
        ylabel=(score_ylabel if score_ylabel is not None
                else f'Decode {y_predicted_name} {spec["label"]}') if show_ylabel else '',
        y_name=y_predicted_name,
        extra_lines=extra_lines,
        # show_ci tied to show_mean, not a separate flag: the stored CI is a bootstrap
        # percentile band around mean_col, not aggr_col (see _plot_f1_lines), so drawing it
        # around the default aggr-only line would show a CI that doesn't belong to it.
        # show_aggr_ci is independent - it's a CI actually computed for aggr_col itself.
        show_ci=show_mean, show_aggr_ci=show_aggr_ci, target_color='black', target_alpha=0.75, target_linewidth=2,
        metric=metric, show_shuffle=show_shuffle, null_kind=null_kind, smooth=smooth,
        show_ccgp_secondary=show_ccgp_secondary,
        svm_res_secondary=svm_res_secondary, ccgp_w_elected_stratum=ccgp_w_elected_stratum,
        min_metric=min_metric, ccgp_annotation=ccgp_annotation,
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
                axes[0].fill_betweenx(y=[stripe_y0, stripe_y0 + stripe_height], x1=-0.5, x2=offset - 0.5,
                                      facecolor='none',
                                      edgecolor=cue_colors[level], hatch='///', alpha=0.35, linewidth=0,
                                      label=f'Cue Zone ({_level_label("cue", level)})' if stripe_i < 2 else None)
        # -0.5 .. offset-0.5, not 0 .. offset: `offset` counts heatmap block *rows*, whose
        # cells are centred on integers, so the axis these zones should span end to end is
        # xlim(-0.5, offset-0.5) - the same limits the row-2 heatmap sets.
        axes[0].fill_betweenx(y=[50, 130], x1=-0.5, x2=offset - 0.5, color='darkgray', alpha=0.6, label='R1 Zone')
        axes[0].fill_betweenx(y=[170, 250], x1=-0.5, x2=offset - 0.5, color='lightgray', alpha=0.6, label='R2 Zone')

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
                              smooth=None, show_shuffle=False, null_kind='perm'):
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
                   show_shuffle=show_shuffle, null_kind=null_kind)

    cov_rows = interval_svm_results[interval_svm_results.predict_y_name == covariate]
    pos_col, neg_col = _first_present(cov_rows, N_POS_COLS), _first_present(cov_rows, N_NEG_COLS)
    n_neg = n_pos = None
    for which_interval in which_intervals:
        rows = cov_rows[cov_rows.interval_name == which_interval]
        if not rows.empty and pos_col is not None:
            n_neg, n_pos = rows[neg_col].values[0], rows[pos_col].values[0]
            # break
    corner_text = covariate
    if n_pos is not None:
        corner_text += f"\n$n_{{{_level_label(covariate, 0)}}}={n_neg}$, $n_{{{_level_label(covariate, 1)}}}={n_pos}$"
    ax.text(0.95, 0.05, corner_text, transform=ax.transAxes,
            fontsize=7, verticalalignment='bottom', horizontalalignment='right')

    ax.tick_params(labelleft=False)
    [sp.set_visible(False) for sp in ax.spines.values()]


def plot_correctness_f1_single(ax, interval_svm_results, y_predicted_name, spec, which_intervals, metric='f1',
                                smooth=None, show_shuffle=False, null_kind='perm'):
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
                   show_shuffle=show_shuffle, null_kind=null_kind)

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
                       correctness_specs=None, metric='f1', smooth=None, show_shuffle=False, null_kind='perm'):
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
                                  smooth, show_shuffle, null_kind=null_kind)
    for ax, spec in zip(axes[len(covariate_names):], correctness_specs):
        plot_correctness_f1_single(ax, interval_svm_results, y_predicted_name, spec, which_intervals, metric, smooth,
                                    show_shuffle, null_kind=null_kind)
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
        # ~(x >= t), not (x < t): a NaN score - v2's collapsed fits, where no model
        # survived at all - must count as failing the threshold. `<` is False on NaN,
        # which would leave exactly the least trustworthy timebins at full opacity.
        low_i = ~(metrics_i >= min_metric)
        low_j = ~(metrics_j >= min_metric)
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
                       reference_block=None, as_column=False, mask_chance_band=True,
                       angle_clim=ANGLE_CLIM,
                       angle_white_sigma=ANGLE_WHITE_SIGMA):
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

    `angle_clim` sets the colour limits: a (vmin, vmax) pair, or a scalar half-width read
    symmetrically around 90 (e.g. 30 -> 60..120), which keeps white at exactly orthogonal
    since bwr_r centres white in the range. The default ANGLE_CLIM is the full 0-180;
    narrowing it spreads the colormap over the range real angles actually occupy (a random
    pair in mPFC is inside 68.5-110.4 95% of the time). Colour only - nothing is clipped
    from the data, and the alpha masking is computed from the angles themselves, so a
    saturated cell is still dimmed on its own merits.

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
    cmap, norm = _angle_colormapping(angle_clim, angle_white_sigma)
    X, Y = np.meshgrid(np.arange(n_x), np.arange(n_y))
    im = ax.pcolormesh(X, Y, np.ma.masked_invalid(full), alpha=alpha_full, cmap=cmap, norm=norm)

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
        box_kwargs = dict(fill=False, edgecolor=HIGHLIGHT_COLOR, alpha=0.15,
                          linewidth=HIGHLIGHT_BOX_LW, zorder=20)
        circle_kwargs = dict(fill=False, edgecolor=HIGHLIGHT_COLOR, alpha=0.5,
                             linewidth=HIGHLIGHT_CIRCLE_LW, zorder=20)
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
                ax.text(x_pos + 1, y_pos + 1, name, fontsize=6, color=HIGHLIGHT_COLOR, alpha=0.9,
                        rotation=25, ha='left', va='bottom', clip_on=False, zorder=21)

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
                                  metric='f1', show_shuffle=False, null_kind='perm'):
    """LLM-friendly description of a plot_session_analysis main figure: what each column
    shows, plus the underlying score-vs-time data (the figure's key quantitative content,
    f1_aggr/f1_mean or acc_aggr/acc_mean depending on `metric` - see METRIC_SPECS) as one
    markdown table per column. With show_shuffle, the null-control mean/p-value columns
    are included too, when present."""
    spec = METRIC_SPECS[metric]
    lines = [
        f"# SVM Decoding Analysis - Session {s_id} (S{s_num})",
        "",
        "## Overview",
        f"- Session ID: {s_id}",
        f"- Session number: S{s_num}",
        f"- Intervals analyzed, in order: {', '.join(which_intervals)}",
        f"- Metric: {spec['label']}",
        "- Columns (X_modality -> block):",
    ]
    for i, col_spec in enumerate(column_specs, 1):
        lines.append(f"  {i}. {col_spec['x_modality']} -> {_column_title(col_spec)}")
    lines += [
        "",
        f"Each column's figure shows: row 0, single-trial position trajectories colored by "
        f"per-timebin decode {spec['label']}; row 1, {spec['label']} over time ({spec['aggr_col']} = "
        f"reference score computed over all trials, {spec['mean_col']} = mean across bootstrap "
        "iterations - the table below has both regardless of which the figure draws" +
        (f", dashed black line = mean of the stored {null_kind} null draws" if show_shuffle else "") +
        "); row 2, pairwise cosine-angle heatmap (degrees) between per-timebin SVM "
        "weight vectors within this column (blank/white gaps mark interval boundaries); "
        f"further rows, the same {spec['label']}-vs-time score recomputed after splitting trials by "
        "another behavioral covariate or by R1/R2 choice correctness, to check whether decoding is "
        f"confounded with that covariate. Below is row 1's data ({spec['label']} vs time) for each column.",
        "",
    ]
    table_cols = ['interval_name', 'timebin', spec['aggr_col'], spec['mean_col'],
                  spec['ci_lower_col'], spec['ci_upper_col']]
    for col_spec, svm_res in zip(column_specs, column_svm_res):
        lines.append(f"## {_column_title(col_spec)} decoding from {col_spec['x_modality']}")
        lines.append("")
        ordered = _ordered_svm_res(svm_res, which_intervals)
        cols = table_cols.copy()
        cols += [c for c in (_first_present(ordered, N_POS_COLS), _first_present(ordered, N_NEG_COLS))
                 if c is not None]
        if show_shuffle:
            # the null mean is derived (the stored column is a per-row list of draws), so
            # it goes in as its own computed column rather than by name
            null_values, null_col = _null_mean_values(ordered, metric, null_kind)
            p_col = _first_present(ordered, spec['null_p_cols'].get(null_kind, ()))
            if null_values is not None:
                ordered = ordered.assign(**{f'{null_kind}_null_mean': null_values})
                cols.append(f'{null_kind}_null_mean')
            if p_col is not None:
                cols.append(p_col)
        lines.append(_df_to_markdown_table(ordered[cols]))
        lines.append("")

    with open(md_path, 'w') as f:
        f.write("\n".join(lines))


def plot_session_analysis(s_id, t0_events, behavior, svm_results, x_modalities, y_predicted_names,
                           which_intervals, show_covariate_plot=False, show_colorbar=False, covariate_names=None,
                           show_w_comparison=False, save_dir=None, save_markdown=False, metric='balanced_accuracy',
                           show_shuffle=False, null_kind='perm',
                           show_scatter=False, show_mean=False, show_aggr_ci=True, smooth=None,
                           show_ylabel=True, score_ylabel=None, next_session_delta_days=None, show_title=True,
                           show_ccgp_secondary=False, ccgp_w_train_stratum=None, min_metric=None,
                           ccgp_annotation='counts',
                           highlight_selections=None, ith_to_dt_session=None,
                           cue_zone_cutoff=pd.to_datetime('2024-11-26_20-00', format='%Y-%m-%d_%H-%M'),
                           show_trial_kinematics=True, mask_chance_band=True,
                           show_w_variance=False, fr_z_session=None, w_variance_n_pcs=3,
                           w_variance_show_session=False, w_variance_row_ratio=0.6,
                           w_variance_ylim=(0, 8), w_variance_split=None,
                           w_variance_show_pc_band=True, w_variance_split_perm_reps=200,
                           w_variance_pc_smooth=2, show_w_activation=False,
                           w_activation_split=None, w_activation_row_ratio=0.6,
                           w_activation_ylim=ACTIVATION_YLIM, w_activation_show_sem=True,
                           angle_clim=ANGLE_CLIM,
                       angle_white_sigma=ANGLE_WHITE_SIGMA):
    """Plot the full per-session analysis. One column per (x_modality, block spec) pair -
    see _resolve_column_specs for how list(s) of either map to columns, and
    _as_column_spec for the dict form of a `y_predicted_names` entry, which is how one
    figure compares a target's unconditioned fit against its conditioned directions (and
    against a derived target like cueXORchoice_R1) now that all of them live in one fit
    table. With
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
    it happens to land outside the chance band (see plot_w_comparison). It also draws the
    threshold itself along the bottom of row 1, as two narrow lines in the target's own
    level colours that are present only where the (smoothed) score clears it - see
    _draw_min_metric_indicator - so what the heatmap above is masking on is visible in the
    same panel rather than having to be read off a gridline. The score each
    cell is checked against is smoothed the same way `smooth` smooths row 1's line (see
    _w_block_with_interval_separators), so the mask agrees with what row 1 actually shows
    instead of flickering off single noisy timebins the drawn (smoothed) line doesn't
    dip on. `highlight_selections`,
    if given, is a list of dicts (session_id/y_name/conditioner/train_stratum/interval/
    timebin - see _selection_matches; only keys a dict sets are checked) marking specific
    W's with a callout box (see HIGHLIGHT_COLOR) in the cross-column comparison. This is
    **drawing only** - the matched W's are not returned; use resolve_w_selections on the
    same list to get those. An optional `name` key (not itself matched against anything -
    see _selection_matches/_highlighted_rows) is drawn as a small label next to its
    box/circle (see plot_w_comparison). Since a CCGP
    column can contribute two blocks here (see single_y_name above), a selection meant to
    pick out one specific direction unambiguously should set `train_stratum`, not just
    `conditioner` - see _selection_matches. With save_dir, each figure produced is also
    saved as both a PNG and an SVG there; returns (saved_png_paths, saved_svg_paths),
    both empty if save_dir is None. With
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
    row 1 and every covariate panel also draw the null-control mean as a dashed black
    line, for columns/timebins where null draws were stored (see _null_mean_values).
    `null_kind` picks which null that is: 'perm' (fully permuted labels - the only one a
    v1 table has) or 'shift' (v2 only - a circular shift of the trial-label vector, which
    keeps the labels' own temporal structure and so tests against slow session drift
    rather than destroying it; a curve that clears 'perm' but not 'shift' is drift).
    Row 0's cue-zone illustration is skipped for sessions up
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
    y-axis. `score_ylabel`, if given, replaces that row's auto label text ('Decode
    choice_R1 Balanced Accuracy') with your own - the auto one is longer than the row is
    tall on a narrow single-interval panel and then gets clipped at the figure edge. It
    only changes the TEXT; show_ylabel=False still hides the label (and tick labels)
    entirely. next_session_delta_days, if given, draws a curved arrow with a "+Nd" label in
    the top-right corner - a placeholder meant to be manually dragged into place once you
    line this session's exported figure up against the next one's, so the reader can see
    the gap between them; the caller (not this function) knows the session ordering needed
    to compute the delta to whichever session comes next. show_title=False hides each
    column's own "{y_predicted_name} decoding from {x_modality}" title (see create_subplot)
    - the S01/S02/etc suptitle is unaffected, always shown. With show_ccgp_secondary, row 1
    draws a CCGP fit's two train/test directions symmetrically - two equal-weight lines,
    each colored by the conditioner level it trained on - with a corner annotation of each
    direction's own class balance instead of the naive n_pos/n_neg text, the direction
    `ccgp_w_train_stratum` elects named first (see _resolve_ccgp_w_stratum and
    _ccgp_n_annotation and the `ccgp` arg of get_SVMCueOutcomeChoicePred). A no-op for
    naive fits (or older cached analytics). `ccgp_annotation` picks which form that corner
    text takes: 'counts' (default) is the two-line Train/Test listing with each stratum's
    own class balance, 'sentence' is the one-row "Train on skip trials, Test on stop:
    C1 vs C2" with the two classes bold and in their own colours and no numbers at all (see
    _ccgp_sentence_segments) - the poster form, where the design is the point and the exact
    n's are not. `ccgp_w_train_stratum`, if given, is a
    {y_predicted_name: train_stratum} dict picking which direction's W feeds row 2's
    cosine-similarity heatmap (and the cross-column comparison, if show_w_comparison) for
    that target - e.g. {'cue': 1} uses the 'cue' decoder's W trained on the choice_R1==1
    subset - instead of the default lowest fitted stratum. A no-op for naive fits or
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
    implementation, so this single flag controls both consistently. `show_w_variance`
    inserts one extra row between the score row and the heatmap, asking whether each
    timebin's W is a dominant direction in the population or a thin one - see
    var_plotting_lib.plot_w_variance_row for what it draws (a W curve read between a
    top-PCs ceiling band and a random-direction floor band). It **requires
    `fr_z_session`**: this function does no I/O
    of its own, so the caller passes that session's raw (unsliced) FiringRate40msZ frame,
    which is then sliced per column by that column's own X_modality. `w_variance_n_pcs`
    (default 3) is how many leading directions that row's ceiling band averages over - 1
    makes it PC1 alone. `w_variance_show_session` adds the dashed "same W over the whole
    recording" curve, off by default (it is also the expensive half of the computation).
    That row carries no in-panel legend at any setting - it does not fit; export the key
    once with var_plotting_lib.plot_w_variance_legend. `w_variance_row_ratio` (default
    0.6) is the row's height in height_ratios units - exposed because every unit added
    makes the whole figure taller without making it wider, which shrinks every other row
    once the result is scaled to fit. `w_variance_ylim` (default (0, 8) percent) is fixed
    rather than autoscaled so the row reads the same across session panels; None
    autoscales from 0. `w_variance_split` adds per-condition curves - the same W scored on
    one condition's trials at a time, each against that condition's own variance; see
    plot_w_variance_row for the three things that then become readable, and note that none
    of it is forced by the label coding, unlike any mean-difference readout of the same W.
    It takes a label name ('cue'), a dict pinning what is held fixed
    ({'split': 'choice_R1', 'given': 'cue'} -> the 2x2 as two panels, one per cue level),
    or a list of either - **one extra row per panel**, so this is what makes the figure
    taller. See var_plotting_lib.resolve_variance_splits. `w_variance_show_pc_band=False`
    `w_variance_split_perm_reps` (default 200, 0 = off) sets the per-condition permutation
    band - the null that actually matches a condition comparison, unlike the shared grey
    random-direction band. `w_variance_show_pc_band=False`
    drops the top-PCs ceiling band from every panel: on a conditioned panel the trial count
    is halved and that band's CV-to-in-sample bracket widens accordingly, so with several
    panels stacked it can end up covering most of them.
"""
    spec = METRIC_SPECS[metric]
    # A row's content argument must not be settable without the row: w_variance_split
    # alone reads exactly like "split the variance row by cue" and would draw nothing.
    if w_activation_split is not None and not show_w_activation:
        raise ValueError("w_activation_split set without show_w_activation=True, which would "
                         "silently draw nothing - set the row on, or drop the split.")
    if w_variance_split is not None and not show_w_variance:
        raise ValueError("w_variance_split set without show_w_variance=True, which would "
                         "silently draw nothing - it adds per-condition curves (and one row "
                         "per panel) to that row rather than being a row of its own.")
    if show_w_activation and fr_z_session is None:
        raise ValueError("show_w_activation=True needs fr_z_session - this function does no "
                         "I/O of its own (same contract as show_w_variance).")
    if show_w_variance and fr_z_session is None:
        # Fail here rather than inside the column loop: by then a half-drawn figure
        # already exists, and the AttributeError from slicing None reads as a bug in the
        # variance code instead of a missing argument.
        raise ValueError("show_w_variance=True needs fr_z_session - this function does no "
                         "I/O of its own. Pass the session's raw FiringRate40msZ frame, e.g. "
                         "plotlib.session_data.load_firing_rate([<full NAS name>])"
                         f".xs({s_id!r}, level='session_id')")
    draw_cue_zone = pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') > cue_zone_cutoff
    column_specs = _resolve_column_specs(x_modalities, y_predicted_names)
    ncols = len(column_specs)
    # Where each interval's t0 sits, measured from this session's own t0_events rows -
    # the single source of interval geometry for every row of this figure (see
    # _interval_x0). Built once here rather than re-derived per panel so row 0's dashed
    # line, row 1's x-ticks and row 2's block separators can never disagree.
    interval_x0s = interval_x0_map(t0_events, which_intervals, s_id=s_id, warn=False)
    # Whether a CCGP column's non-elected direction also gets its own block in the
    # cross-column comparison below (see the column loop): only when a single target is
    # being compared across the whole figure, so that target's own two directions are
    # the comparison (nothing else to compare against). With multiple targets, the
    # comparison is across targets instead (e.g. cue vs choice_R1), so each CCGP column
    # contributes only its one elected direction - mixing both would blur "target vs
    # target" with "direction vs direction" in the same heatmap.
    #
    # A column whose block spec pins a train_stratum is exempt either way: it asked for
    # exactly one direction, so it contributes exactly one block (see the column loop),
    # and the figure's blocks stay a 1:1 image of the specs the caller wrote.
    single_y_name = len({cs['y_name'] for cs in column_specs}) == 1

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
        n_covariates = max(sum(len(specs) for specs in _covariate_specs_for_target(cs['y_name'], fitted_names))
                           for cs in column_specs)
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

    # Row indices, resolved once: the variance row (see plot_w_variance_row) is optional
    # and sits between the score row and the heatmap, so every row below it shifts down
    # by one when it is on. Every downstream index goes through hm_row rather than a
    # literal 2/3, so turning the row on can't silently draw the heatmap into it.
    # One variance row per split panel (see var_plotting_lib.resolve_variance_splits): a
    # plain label name is one panel, and {'split': ..., 'given': <label>} expands to one
    # panel per level of the held-fixed label, i.e. the 2x2 as two panels. With no split at
    # all it is still one row, the pooled curve alone.
    var_specs = resolve_variance_splits(w_variance_split) if show_w_variance else []
    var_rows = list(range(2, 2 + max(1, len(var_specs)))) if show_w_variance else []
    # the activation row(s) sit directly below the variance row(s) - same splits, same
    # colours, same cells, so the pair reads as "how far out" above "how spread out"
    act_specs = resolve_variance_splits(w_activation_split) if show_w_activation else []
    act_start = 2 + len(var_rows)
    act_rows = (list(range(act_start, act_start + max(1, len(act_specs))))
                if show_w_activation else [])
    hm_row = act_start + len(act_rows)
    nrows = hm_row + 1 + n_covariates
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
    # w_variance_row_ratio is worth tuning rather than fixing: fig_height is defined as
    # 1.3 * sum(height_ratios) while fig_width doesn't move, so every unit added here makes
    # an already tall, narrow figure taller - and once it's scaled to fit a poster column,
    # that shrinks every other row with it. 0.6 (a bit over half the score row) is enough
    # for a line plot with 3 y-ticks; raise it if the row needs to be read closely, lower
    # it if the figure as a whole is getting squeezed.
    height_ratios = ([row0_ratio, 1] + [w_variance_row_ratio] * len(var_rows)
                     + [w_activation_row_ratio] * len(act_rows)
                     + [heatmap_row_ratio] + [0.8] * n_covariates)
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
    column_highlight_positions = []
    for col_i, col_spec in enumerate(column_specs):
        x_modality, y_predicted_name = col_spec['x_modality'], col_spec['y_name']
        column_title = _column_title(col_spec)
        svm_res, svm_res_secondary, interval_svm_results, all_pos, all_times = get_session_data(
            s_id, t0_events, behavior, svm_results, x_modality, y_predicted_name, which_intervals,
            ccgp_w_train_stratum=ccgp_w_train_stratum, analysis=col_spec['analysis'],
            conditioner=col_spec['conditioner'], need_positions=show_trial_kinematics)

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
            for row_i in range(hm_row + 1):
                axes[row_i, col_i].axis('off')
            if show_title:
                axes[0, col_i].set_title(f'{column_title} decoding from \n{x_modality}\n(no data)')
            column_Ws.append(np.zeros((0, 0)))
            column_t0_positions.append([])
            column_metrics.append(np.zeros(0))
            column_labels.append(f'{column_title}\n{x_modality}')
            column_modalities.append(x_modality)
            continue

        # Resolved once, up front, so row 1's corner-annotation star (create_subplot ->
        # _plot_f1_lines -> _ccgp_n_annotation) and row 2's heatmap below agree on which
        # direction is "the" one for this target - see _resolve_ccgp_w_stratum (raises on
        # an explicitly-requested stratum that was never actually fit, rather than
        # silently ignoring it).
        # A spec's own train_stratum is the more specific request, so it wins over the
        # figure-wide ccgp_w_train_stratum dict - expressed AS that same dict so it goes
        # through _resolve_ccgp_w_stratum's own validation (which raises on a stratum this
        # target was never fit for) rather than a second, laxer code path. It ELECTS a
        # direction; it does not filter the other one away - that is what keeps
        # show_ccgp_secondary free to draw both lines (see get_session_data).
        elected_stratum = _resolve_ccgp_w_stratum(
            svm_res, svm_res_secondary, y_predicted_name,
            {y_predicted_name: col_spec['train_stratum']} if col_spec['train_stratum'] is not None
            else ccgp_w_train_stratum)

        # get_session_data's svm_res/svm_res_secondary are still the fixed-rule
        # primary/other split from _split_ccgp_primary_secondary - the election above
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

        # create_subplot's three rows are trajectories/score/heatmap - not contiguous any
        # more once the variance row sits between the last two, hence the explicit list.
        scatter, xticks, xtick_labels = create_subplot([axes[0, col_i], axes[1, col_i],
                                                        axes[hm_row, col_i]], all_pos, svm_res,
                                          f'{column_title} decoding from \n{x_modality}',
                                          y_predicted_name, which_intervals, metric=metric,
                                          show_shuffle=show_shuffle, null_kind=null_kind,
                                          show_scatter=show_scatter,
                                          show_mean=show_mean, show_aggr_ci=show_aggr_ci,
                                          draw_cue_zone=draw_cue_zone, smooth=smooth,
                                          show_ylabel=show_ylabel, score_ylabel=score_ylabel,
                                          show_title=show_title,
                                          show_ccgp_secondary=show_ccgp_secondary,
                                          svm_res_secondary=svm_res_secondary,
                                          ccgp_w_elected_stratum=elected_stratum,
                                          show_trial_kinematics=show_trial_kinematics,
                                          interval_x0s=interval_x0s, min_metric=min_metric,
                                          ccgp_annotation=ccgp_annotation)
        if show_w_variance:
            from plotlib.session_data import session_fr_for_modality  # see the import note up top
            fr_x = session_fr_for_modality(fr_z_session, x_modality)
            # one profile per panel: a conditioned panel's samples are a different trial
            # subset, so nothing is shareable between them
            for row_i, var_spec in zip(var_rows, var_specs or [dict(split=None, given=None)]):
                profile = w_variance_profile(s_id, t0_events, fr_x, svm_res, which_intervals,
                                              n_pcs=w_variance_n_pcs,
                                              show_session=w_variance_show_session,
                                              split=var_spec['split'], given=var_spec['given'],
                                              split_perm_reps=w_variance_split_perm_reps)
                plot_w_variance_row(axes[row_i, col_i], profile, which_intervals, interval_x0s,
                                     smooth=smooth, show_ylabel=show_ylabel,
                                     show_session=w_variance_show_session,
                                     ylim=w_variance_ylim,
                                     show_pc_band=w_variance_show_pc_band,
                                     pc_smooth=w_variance_pc_smooth)

        if show_w_activation:
            from plotlib.session_data import session_fr_for_modality  # see the import note up top
            fr_x = session_fr_for_modality(fr_z_session, x_modality)
            for row_i, act_spec in zip(act_rows, act_specs or [dict(split=None, given=None)]):
                profile = w_activation_profile(s_id, t0_events, fr_x, svm_res, which_intervals,
                                                split=act_spec['split'], given=act_spec['given'])
                plot_w_activation_row(axes[row_i, col_i], profile, which_intervals, interval_x0s,
                                       smooth=smooth, show_ylabel=show_ylabel,
                                       ylim=w_activation_ylim, show_sem=w_activation_show_sem)

        if show_covariate_plot:
            plot_covariate_f1(axes[hm_row + 1:, col_i], interval_svm_results, y_predicted_name,
                               which_intervals, covariate_names, metric=metric, smooth=smooth,
                               show_shuffle=show_shuffle, null_kind=null_kind)

        # Row-order to match create_subplot's offset scheme (t0/boundary lines already
        # drawn on this axis above), so this heatmap's pixels line up with those instead
        # of relying on svm_res's incidental native row order. This heatmap shares the
        # same gapped W block as the cross-column comparison below (plot_w_comparison),
        # and that block IS the x-axis every other row is drawn on (block_x /
        # next_block_offset in plotlib.trial_traces) - one NaN separator row per interval
        # boundary, cells centred on integer x by pcolormesh's 'nearest' shading.
        # w_source is the elected direction's own rows - a no-op (returns svm_res
        # unchanged) for naive fits, which only ever have one direction anyway.
        w_source = _rows_for_stratum(svm_res, svm_res_secondary, elected_stratum)
        ordered_svm_res = _ordered_svm_res(w_source, which_intervals)
        # metric_col: per-row aggr score, parallel to block_Ws - feeds min_metric below
        # (both row 2's own heatmap and the cross-column comparison's alpha mask).
        block_Ws, t0_positions, block_metric = _w_block_with_interval_separators(
            ordered_svm_res, which_intervals, metric_col=spec['aggr_col'], smooth=smooth,
            interval_x0s=interval_x0s)

        # With a single target across the whole figure, a CCGP column's non-elected
        # direction also gets its own block below, so that target's two directions
        # compare against each other (see single_y_name above). With multiple targets,
        # only the elected direction is included, so the comparison is across targets
        # (e.g. cue vs choice_R1) instead. Row 2's own heatmap right below always stays
        # elected-only regardless (a self-similarity-across-timebins view, not this
        # cross-block comparison).
        is_ccgp_column = elected_stratum is not None
        conditioner = (svm_res['conditioner'].dropna().iloc[0]
                       if is_ccgp_column and svm_res['conditioner'].notna().any() else None)
        # A spec-supplied label replaces the whole block label, suffix included - it was
        # written to say what this block is, so appending the direction to it would just
        # repeat what the caller already decided how to phrase.
        elected_suffix = (f' (train:{_level_label(conditioner, elected_stratum)})'
                          if is_ccgp_column and col_spec['label'] is None else '')

        # own_highlight_rows: same local_row values as column_highlight_positions below,
        # but kept separate (and un-offset) for row 2's own per-column heatmap further
        # down - that heatmap is a single block (this column's elected direction only,
        # see the comment above), so it needs the bare local_row, not the cross-column
        # figure's (block_index, local_row) pair.
        own_highlight_rows = []
        for local_row, name in _highlighted_rows(ordered_svm_res, which_intervals, highlight_selections or [],
                                                   session_id=s_id, y_name=y_predicted_name,
                                                   conditioner=conditioner):
            column_highlight_positions.append((len(column_Ws), local_row, name))
            own_highlight_rows.append((local_row, name))
        column_Ws.append(block_Ws)
        column_t0_positions.append(t0_positions)
        column_metrics.append(block_metric)
        column_labels.append(f'{column_title}{elected_suffix}\n{x_modality}')
        column_modalities.append(x_modality)

        # The pinned-spec exemption is about W BLOCKS, not about row 1's lines (that is
        # show_ccgp_secondary's job, and it stays free to draw both): a spec that names a
        # direction is one block, so the figure's blocks stay a 1:1 image of the specs the
        # caller wrote. Want both directions as blocks? Write both specs.
        if (single_y_name and is_ccgp_column and col_spec['train_stratum'] is None
                and svm_res_secondary is not None and not svm_res_secondary.empty):
            # "Other" direction = whichever fitted train_stratum isn't the elected one,
            # from the full primary+secondary pool - NOT svm_res_secondary's own stratum
            # label. svm_res_secondary is whichever direction _split_ccgp_primary_secondary
            # called primary, decided *before* ccgp_w_train_stratum is applied -
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
                    secondary_ordered, which_intervals, metric_col=spec['aggr_col'], smooth=smooth,
                    interval_x0s=interval_x0s)
                for local_row, name in _highlighted_rows(secondary_ordered, which_intervals,
                                                           highlight_selections or [], session_id=s_id,
                                                           y_name=y_predicted_name, conditioner=conditioner):
                    column_highlight_positions.append((len(column_Ws), local_row, name))
                column_Ws.append(secondary_Ws)
                column_t0_positions.append(secondary_t0)
                column_metrics.append(secondary_metric)
                column_labels.append(f'{column_title} (train:{_level_label(conditioner, secondary_stratum)})'
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
        cmap, hm_norm = _angle_colormapping(angle_clim, angle_white_sigma)
        X, Y = np.meshgrid(timebins, timebins)
        im = axes[hm_row,col_i].pcolormesh(X, Y, np.ma.masked_invalid(cos_sim_matrix),
                                  alpha=high_angles_alpha, cmap=cmap, norm=hm_norm)

        # Same row-band + cross-point circle (plus name label) as plot_w_comparison's
        # highlight_positions (see there, and HIGHLIGHT_COLOR/_LW), drawn here too since a
        # single-block session (e.g. a naive fit) never gets a cross-column comparison
        # figure at all (skipped below when len(column_Ws) <= 1) - this is the only
        # heatmap such a session has.
        total = len(timebins)
        for g, name in own_highlight_rows:
            axes[hm_row,col_i].add_patch(Rectangle((-0.5, g - 0.5), total, 1, fill=False,
                                               edgecolor=HIGHLIGHT_COLOR, alpha=0.15,
                                               linewidth=HIGHLIGHT_BOX_LW, zorder=20))
            axes[hm_row,col_i].add_patch(Circle((g, g), radius=0.5 * np.sqrt(2), fill=False,
                                            edgecolor=HIGHLIGHT_COLOR, alpha=0.15,
                                            linewidth=HIGHLIGHT_CIRCLE_LW, zorder=20))
            if name:
                axes[hm_row,col_i].text(g + 1, g + 1, name, fontsize=5, color=HIGHLIGHT_COLOR, alpha=0.9,
                                    rotation=25, ha='left', va='bottom', clip_on=False, zorder=21)

        axes[hm_row,col_i].set_xticks(xticks)
        # fontsize=7 to match the n_skip/n_stop-style corner annotations (_plot_f1_lines) -
        # the default (poster) rcParams font size is far too large for how little room
        # these labels have, especially now that column width scales down for fewer
        # intervals (see width_per_interval above).
        axes[hm_row,col_i].set_xticklabels(xtick_labels, fontsize=TICK_FONTSIZE)
        axes[hm_row,col_i].set_xlim(-0.5, len(timebins)-0.5)
        axes[hm_row,col_i].set_ylim(-0.5, len(timebins)-0.5)
        # Deliberately NOT set_aspect('equal') here: forcing a square aspect makes matplotlib
        # shrink/recenter this Axes' own bounding box within its GridSpec cell to fit a
        # square, which breaks the shared-x alignment with rows 0/1 above it (their box stays
        # full-width, this one doesn't) - the t0 dashed lines above and below then visibly
        # drift apart. Left at the default 'auto' aspect, cells aren't perfectly square, but
        # the box - and so the alignment - stays identical to every other row; height_ratios
        # above gives this row extra height to compensate for the resulting flatter cells.
        axes[hm_row,col_i].tick_params(labelleft=False)

        # turn off all spines
        for row_i in range(hm_row + 1):
            [sp.set_visible(False) for sp in axes[row_i, col_i].spines.values()]

    if show_colorbar:
        if scatter is not None:
            cbar = plt.colorbar(scatter, ax=axes[0, -1])
            cbar.set_ticks([.75, .5, .25])
            cbar.set_label(f'SVM {spec["label"]}', labelpad=15)

        cbar2 = plt.colorbar(im, ax=axes[hm_row, -1])
        # ticks inside whatever the limits are, so a narrowed range is still readable
        cbar2.set_ticks([t for t in (0, 60, 90, 120, 180)
                         if _angle_clim(angle_clim)[0] <= t <= _angle_clim(angle_clim)[1]]
                        or list(_angle_clim(angle_clim)))
        cbar2.set_label('Angle between SVM planes (degrees)', labelpad=15)

    s_num = (ith_to_dt_session or {}).get(s_id, "Unknown")
    title = f'S{s_num:02d}' if isinstance(s_num, int) else f'S{s_num}'
    plt.suptitle(title, y=.99, **SUPTITLE_KW)
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
                                          metric=metric, show_shuffle=show_shuffle, null_kind=null_kind)

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
                          angle_clim=angle_clim, angle_white_sigma=angle_white_sigma,
                           block_t0_positions=column_t0_positions, save_path=wcomp_path,
                           save_path_svg=wcomp_svg_path, markdown_path=wcomp_md_path,
                           block_metrics=column_metrics, min_metric=min_metric,
                           highlight_positions=column_highlight_positions, mask_chance_band=mask_chance_band)
        if wcomp_path is not None:
            saved_png_paths.append(wcomp_path)
            saved_svg_paths.append(wcomp_svg_path)

    return saved_png_paths, saved_svg_paths

    # fname = os.path.join(output_dir, f'session_S{s_id}_{y_predicted_name}_svm_analysis_S15_Rest.svg')
    # fig.savefig(fname, dpi=300)


def _session_w_block(svm_results, s_id, x_modality, y_predicted_name, which_intervals, t0_events,
                      metric_col=None, ccgp_w_train_stratum=None, smooth=None,
                      analysis=None, conditioner_filter=None):
    """One session's W block for a single (x_modality, y_predicted_name) fit - the shared
    body of get_session_sequence_Ws' per-session loop and get_cross_analytic_Ws' per-spec
    loop, which build the same kind of block from different sources (one analytic across
    sessions vs. several analytics within one session).

    Returns (Ws, t0_positions, metric_vals, ordered, conditioner, elected_stratum).
    `ordered` comes back because the caller needs the same row order for
    _highlighted_rows; `conditioner`/`elected_stratum` because only the caller knows how to
    word its own block label. For a CCGP frame, `ccgp_w_train_stratum` picks the direction
    (see _resolve_ccgp_w_stratum, which guards the empty/all-naive cases internally and
    returns None rather than raising, so a session with no fit at all just yields an empty
    (0, 0) block - which plot_w_comparison already renders blank).

    `analysis`/`conditioner_filter` narrow the rows first, the same way a column's block
    spec does in get_session_data - needed on a v2 frame, which holds the unconditioned fit
    and both conditioned directions of a target together. Which direction the block shows
    is an election (ccgp_w_train_stratum), not a filter, exactly as it is there."""
    svm_res = svm_results.loc[
        (svm_results.index.get_level_values('session_id') == s_id) &
        (svm_results['X_modality'] == x_modality) &
        (svm_results['predict_y_name'] == y_predicted_name) &
        (svm_results['interval_name'].isin(which_intervals))
    ]
    svm_res = _filter_analysis(svm_res, analysis)
    if conditioner_filter is not None:
        svm_res = svm_res[svm_res['conditioner'] == conditioner_filter]
    primary, secondary = _split_ccgp_primary_secondary(svm_res)
    elected_stratum = _resolve_ccgp_w_stratum(primary, secondary, y_predicted_name, ccgp_w_train_stratum)
    conditioner = None
    if elected_stratum is not None and primary['conditioner'].notna().any():
        conditioner = primary['conditioner'].dropna().iloc[0]

    rows = _rows_for_stratum(primary, secondary, elected_stratum)
    _check_one_row_per_timebin(rows, f'{y_predicted_name} / {x_modality} / {s_id}')
    ordered = _ordered_svm_res(rows, which_intervals)
    Ws, t0_positions, metric_vals = _w_block_with_interval_separators(
        ordered, which_intervals, metric_col=metric_col, smooth=smooth,
        # each session's own geometry: the stale t0 table disagrees between sessions
        interval_x0s=interval_x0_map(t0_events, which_intervals, s_id=s_id, warn=False))
    return Ws, t0_positions, metric_vals, ordered, conditioner, elected_stratum


def get_session_sequence_Ws(session_ids, svm_results, t0_events, x_modality, y_predicted_name, which_intervals,
                             metric='f1', ccgp_w_train_stratum=None, highlight_selections=None,
                             ith_to_dt_session=None, smooth=None):
    """Per-session SVM weight blocks (with zero-row interval separators - see
    _w_block_with_interval_separators) for a single (x_modality, y_predicted_name) fit,
    across which_intervals, in the given session order - one block per session, always
    (unlike plot_session_analysis's cross-column comparison, which shows a CCGP column's
    two directions as separate blocks - a long session sequence would double in width for
    every CCGP session in it, which gets unwieldy fast). For a CCGP session,
    `ccgp_w_train_stratum` (see _resolve_ccgp_w_stratum) selects which single direction
    represents that session - default: the lowest fitted stratum - and that
    direction's conditioner level is noted in the block's label ("S<n> (train:<level>)")
    since there's no second block here to tell it apart from. `metric` (see METRIC_SPECS)
    picks which aggr score is attached to each block as its per-row metric array, for use
    with plot_w_comparison's min_metric. `highlight_selections`, if given, is a list of
    dicts with any of session_id/y_name/conditioner/train_stratum/interval/timebin (see
    _selection_matches - only keys a dict actually sets are checked, so e.g. a dict
    without `conditioner` matches naive and CCGP fits alike; a dict without
    `train_stratum` matches whichever direction this function elected, since that's the
    only one present in any given session's block here). Each match becomes a
    (block_index, row_index, name) triple for plot_w_comparison's highlight_positions -
    drawing only; the matched W's themselves are not returned, see resolve_w_selections
    for that. `smooth`, if given (e.g. 3), passes straight
    through to _w_block_with_interval_separators - the same rolling-mean window
    plot_session_analysis applies to its own min_metric masking (see _smooth), so a
    session that only dips below min_metric on an isolated noisy timebin isn't masked out
    here either. `ith_to_dt_session`, if given, is a
    {short_session_id: index} dict used only for the "S<n>" block label (falls back to
    "?" when absent/missing a session - see plot_session_analysis's own use of this).
    Returns (block_Ws, block_labels, block_modalities, block_t0_positions, block_metrics,
    highlight_positions) - one entry per session, except highlight_positions, which has
    one entry per matched selection."""
    spec = METRIC_SPECS[metric]
    block_Ws, block_labels, block_modalities, block_t0_positions, block_metrics = [], [], [], [], []
    highlight_positions = []
    elected_per_session = {}

    for s_id in session_ids:
        Ws, t0_positions, metric_vals, ordered, conditioner, elected_stratum = _session_w_block(
            svm_results, s_id, x_modality, y_predicted_name, which_intervals, t0_events,
            metric_col=spec['aggr_col'], ccgp_w_train_stratum=ccgp_w_train_stratum, smooth=smooth)

        s_label = f'S{(ith_to_dt_session or {}).get(s_id, "?")}'
        label = s_label
        if elected_stratum is not None:
            label = f'{s_label} (train:{_level_label(conditioner, elected_stratum)})'
            elected_per_session[s_label] = _level_label(conditioner, elected_stratum)

        for local_row, name in _highlighted_rows(ordered, which_intervals, highlight_selections or [],
                                                  session_id=s_id, y_name=y_predicted_name,
                                                  conditioner=conditioner):
            highlight_positions.append((len(block_Ws), local_row, name))

        block_Ws.append(Ws)
        block_labels.append(label)
        block_modalities.append(x_modality)
        block_t0_positions.append(t0_positions)
        block_metrics.append(metric_vals)

    # A cross-session comparison is only meaningful if every session contributes the SAME
    # CCGP direction. With ccgp_w_train_stratum left off, each session falls back to its
    # own lowest fitted stratum, so a session that only fit one direction contributes a
    # different one from its neighbours - and the odd one out then renders as a near-blank
    # block, because its weaker decoder trips min_metric everywhere. Warn rather than
    # raise, since mixing directions is a legitimate (if rare) thing to ask for
    # deliberately.
    if len(set(elected_per_session.values())) > 1:
        mixed = ', '.join(f'{s}->{lvl}' for s, lvl in elected_per_session.items())
        print(f"WARNING: get_session_sequence_Ws is mixing CCGP directions across sessions "
              f"({mixed}). Pass ccgp_w_train_stratum={{{y_predicted_name!r}: <0|1>}} to pin one.")

    return (block_Ws, block_labels, block_modalities, block_t0_positions, block_metrics,
            highlight_positions)


def plot_session_sequence_w_comparison(session_ids, svm_results, t0_events, x_modality, y_predicted_name,
                                        which_intervals,
                                        save_path=None, save_path_svg=None, metric='balanced_accuracy',
                                        ccgp_w_train_stratum=None, min_metric=None, highlight_selections=None,
                                        ith_to_dt_session=None, reference_session_id=None, as_column=False,
                                        mask_chance_band=True, smooth=None, angle_clim=ANGLE_CLIM, angle_white_sigma=ANGLE_WHITE_SIGMA):
    """Cross-session SVM-weight similarity for a single (x_modality, y_predicted_name):
    one block per session (see plot_w_comparison for what a block means and
    get_session_sequence_Ws for how a CCGP session's single direction is chosen), in the
    given session order. `t0_events` supplies each session's own interval geometry (see
    plotlib.trial_traces.interval_n_bins_left) for the t0 markers.
    Shows how stable/consistent a single decoder's weights are
    across sessions, the same way plot_w_comparison shows it across columns within one
    session. `metric`, `ccgp_w_train_stratum`, `min_metric`, `highlight_selections`,
    `ith_to_dt_session`, and `smooth` all pass straight through to get_session_sequence_Ws/
    plot_w_comparison - see there for what each does. `highlight_selections` only draws
    the callout boxes; call resolve_w_selections on the same list to get the W's.
    Returns nothing.

    `reference_session_id`, if given (one of the (short) ids in `session_ids`), restricts
    the plot to just that one session's own comparisons against every session in the
    sequence - e.g. "session 7 vs every other session" - instead of the full N-by-N grid
    (see plot_w_comparison's `reference_block`, which this resolves session_id into).
    Raises ValueError if it isn't actually present in `session_ids`. `as_column` flips
    that restricted view into a vertical strip instead of the default horizontal one -
    see plot_w_comparison. Both ignored when `reference_session_id` is None.
    `mask_chance_band` passes straight through to plot_w_comparison - see there and
    _angle_alpha_mask."""
    block_Ws, block_labels, block_modalities, block_t0_positions, block_metrics, highlight_positions = \
        get_session_sequence_Ws(
            session_ids, svm_results, t0_events, x_modality, y_predicted_name, which_intervals,
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
                      angle_clim=angle_clim, angle_white_sigma=angle_white_sigma,
                       block_t0_positions=block_t0_positions, save_path=save_path, save_path_svg=save_path_svg,
                       block_metrics=block_metrics, min_metric=min_metric, highlight_positions=highlight_positions,
                       reference_block=reference_block, as_column=as_column, mask_chance_band=mask_chance_band)


# Keys a cross-analytic block spec may set - see get_cross_analytic_Ws. Validated rather
# than ignored, because a typo'd key ('stratum' for 'train_stratum', 'target' for 'y_name')
# would otherwise silently produce the wrong block: the missing key falls back to a default
# that is itself perfectly valid, so nothing downstream can notice.
CROSS_ANALYTIC_SPEC_KEYS = {'label', 'svm_results', 'y_name', 'train_stratum', 'analysis', 'conditioner'}


def get_cross_analytic_Ws(s_id, block_specs, t0_events, x_modality, which_intervals,
                           metric='balanced_accuracy', ccgp_w_train_stratum=None,
                           highlight_selections=None, smooth=None):
    """W blocks for ONE session, one block per entry in `block_specs` - the cross-analytic
    counterpart to get_session_sequence_Ws, which instead builds one block per session from
    a single analytic.

    The reason this exists at all: the naive fits and the CCGP fits are separate analytics,
    so no single svm_results frame holds both and one frame argument cannot express the
    comparison. Each spec therefore carries its own frame:

        [dict(label='naive', svm_results=<naive frame>, y_name='choice_R1'),
         dict(label='CCGP',  svm_results=<CCGP frame>,  y_name='choice_R1', train_stratum=1)]

    Keys (see CROSS_ANALYTIC_SPEC_KEYS): `label` and `svm_results` are required; `y_name`
    defaults to the previous spec's (so the common same-target case needn't repeat it);
    `train_stratum` pins one CCGP direction for that block (and implies analysis='ccgp');
    `analysis`/`conditioner` narrow the rows the same way a plot_session_analysis block
    spec does (see _as_column_spec) - needed when one frame carries both the unconditioned
    and the conditioned fits, i.e. a v2 SVM_TaskSubspaces table, where the same
    naive-vs-CCGP comparison can be two specs over ONE frame. Deliberately a flat symmetric list rather than naive-plus-CCGP arguments - the
    two sides are the same kind of thing, and the same call shape then also covers "both
    CCGP directions", "L1 vs L2", or three-way comparisons, with no special-casing.

    The list is symmetric, but the *comparison* it feeds is not automatically apples to
    apples. Two things the caller owns, neither checkable here:
      * geometry - every frame must have been fit against the same t0-events generation as
        `t0_events`, or one block's timebin axis means a different stretch of real time
        than another's. plotlib.session_data.check_geometry tests exactly this, per frame.
      * penalty - an L1 (sparse) and an L2 (dense) W are not meaningfully comparable by
        angle: the angle then mostly reports the penalty, not the geometry. Check
        `svm_results['svm_penalty'].unique()` on both frames.

    `metric`, `highlight_selections`, `smooth` and `ccgp_w_train_stratum` behave as in
    get_session_sequence_Ws; a spec's own `train_stratum` takes precedence over
    `ccgp_w_train_stratum` for that block. Returns the same six-tuple
    get_session_sequence_Ws does (block_Ws, block_labels, block_modalities,
    block_t0_positions, block_metrics, highlight_positions), so it drops straight into
    plot_w_comparison."""
    spec = METRIC_SPECS[metric]
    block_Ws, block_labels, block_modalities, block_t0_positions, block_metrics = [], [], [], [], []
    highlight_positions = []
    y_name = None

    for block_spec in block_specs:
        unknown = set(block_spec) - CROSS_ANALYTIC_SPEC_KEYS
        if unknown:
            raise KeyError(f"unknown block_spec key(s) {sorted(unknown)}; "
                           f"known: {sorted(CROSS_ANALYTIC_SPEC_KEYS)}")
        for required in ('label', 'svm_results'):
            if block_spec.get(required) is None:
                raise KeyError(f"block_spec {block_spec.get('label', block_spec)} needs a {required!r}")
        y_name = block_spec.get('y_name') or y_name
        if y_name is None:
            raise KeyError(f"block_spec {block_spec['label']!r} needs a 'y_name' "
                           f"(only later specs may inherit it from an earlier one)")

        # a spec's own train_stratum wins over the shared dict, and is expressed AS that
        # same dict so the direction goes through _resolve_ccgp_w_stratum's own validation
        # (which raises on a stratum this target was never fit for) rather than a second,
        # laxer code path
        train_stratum = block_spec.get('train_stratum')
        stratum_arg = ({y_name: train_stratum} if train_stratum is not None
                       else ccgp_w_train_stratum)
        # 'analysis' is only meaningful for a frame that mixes conditioned and
        # unconditioned fits (v2) - inferred as 'ccgp' when a direction is pinned, so the
        # naive rows of such a frame can't leak into the block
        analysis = block_spec.get('analysis') or ('ccgp' if train_stratum is not None else None)

        Ws, t0_positions, metric_vals, ordered, conditioner, elected_stratum = _session_w_block(
            block_spec['svm_results'], s_id, x_modality, y_name, which_intervals, t0_events,
            metric_col=spec['aggr_col'], ccgp_w_train_stratum=stratum_arg, smooth=smooth,
            analysis=analysis, conditioner_filter=block_spec.get('conditioner'))

        # y_name on its own line: these labels are the heatmap's tick labels, and a
        # single-line "naive choice_R1 (train:C2)" is wide enough to overlap its neighbour
        label = f'{block_spec["label"]}\n{y_name}'
        if elected_stratum is not None:
            label += f'\n(train:{_level_label(conditioner, elected_stratum)})'

        for local_row, name in _highlighted_rows(ordered, which_intervals, highlight_selections or [],
                                                  session_id=s_id, y_name=y_name,
                                                  conditioner=conditioner):
            highlight_positions.append((len(block_Ws), local_row, name))

        block_Ws.append(Ws)
        block_labels.append(label)
        block_modalities.append(x_modality)
        block_t0_positions.append(t0_positions)
        block_metrics.append(metric_vals)

    return (block_Ws, block_labels, block_modalities, block_t0_positions, block_metrics,
            highlight_positions)


def plot_session_cross_analytic_w_comparison(s_id, block_specs, t0_events, x_modality, which_intervals,
                                             save_path=None, save_path_svg=None, markdown_path=None,
                                             metric='balanced_accuracy', ccgp_w_train_stratum=None,
                                             min_metric=None, highlight_selections=None,
                                             ith_to_dt_session=None, reference_label=None, as_column=False,
                                             mask_chance_band=True, smooth=None, angle_clim=ANGLE_CLIM, angle_white_sigma=ANGLE_WHITE_SIGMA):
    """One session's SVM-weight similarity ACROSS ANALYTICS: one block per entry in
    `block_specs` (see get_cross_analytic_Ws for the spec shape and the two comparability
    caveats), all-to-all by default. Typically two blocks - the naive fit and one CCGP
    direction of the same target - in which case the figure reads as three parts: each
    block's own diagonal square is that analytic's temporal stability (how much the decoder
    axis rotates from timebin to timebin), and the off-diagonal rectangle between them is
    the actual naive-vs-CCGP question, one cell per (naive timebin, CCGP timebin) pair.

    This is the per-session analogue of plot_session_sequence_w_comparison (same blocks,
    same alpha masking, same t0/boundary markings, via the same plot_w_comparison engine) -
    call it in a loop over session_ids and hand the returned paths to build_canvas/
    build_canvas_svg to get one strip per session, the way plot_session_analysis' own
    per-session figures are assembled.

    `reference_label` names one spec's `label` to restrict the figure to that block's own
    comparisons against every block, instead of the full square - the same thing
    plot_session_sequence_w_comparison's `reference_session_id` does, resolved by label
    rather than index (`as_column` transposes it, and both are ignored when it is None).
    Resolution by label means it must be unique among the specs: raises rather than taking
    the first match, since specs that cross analytics with targets repeat labels by design
    (a 'naive' block per target) and the wrong one would still plot.
    With two blocks and reference_label='naive' the result is the compact strip "every
    naive timebin vs. [naive | CCGP]", which is the cheapest per-session panel to assemble
    into a canvas. Everything else passes through to get_cross_analytic_Ws/
    plot_w_comparison - see there.

    Returns (save_path, save_path_svg) as actually written (either may be None if not
    requested), or (None, None) when the session has no fitted rows in at least one spec's
    frame - printed and skipped rather than raised, since a partly-computed analytic is the
    normal case in a loop over a session set (e.g. a CCGP analytic covering fewer sessions
    than the naive one). A session missing from `t0_events` is a different case and does
    raise (from interval_x0_map): that table is the interval-geometry source, so there is
    nothing to place the timebin axis or the t0 markers against."""
    block_Ws, block_labels, block_modalities, block_t0_positions, block_metrics, highlight_positions = \
        get_cross_analytic_Ws(
            s_id, block_specs, t0_events, x_modality, which_intervals, metric=metric,
            ccgp_w_train_stratum=ccgp_w_train_stratum, highlight_selections=highlight_selections,
            smooth=smooth)

    s_label = f'S{(ith_to_dt_session or {}).get(s_id, "?")}'
    empty = [label.replace('\n', ' ') for label, Ws in zip(block_labels, block_Ws) if Ws.shape[0] == 0]
    if empty:
        print(f"{s_label} ({s_id}): no fitted rows for {empty} - skipping cross-analytic W comparison")
        return None, None

    reference_block = None
    if reference_label is not None:
        labels = [block_spec['label'] for block_spec in block_specs]
        if reference_label not in labels:
            raise ValueError(f"reference_label {reference_label!r} is not one of the block labels: {labels}")
        # a repeated label is the normal case once the specs cross analytics WITH targets
        # (two 'naive' blocks, one per target); index() would then silently reference the
        # first of them, and the figure would look like a valid answer to the other one
        if labels.count(reference_label) > 1:
            raise ValueError(f"reference_label {reference_label!r} matches {labels.count(reference_label)} "
                             f"blocks ({labels}) - give the intended block a unique label")
        reference_block = labels.index(reference_label)

    title = f'{s_label}: W similarity across analytics ({x_modality})'
    if reference_block is not None:
        # a second line rather than a longer first one: reference mode's figure is a short
        # strip (fig_dims' short_frac), and an appended '- vs <label>' overruns its width
        title += f'\nvs {block_specs[reference_block]["label"]}'
    # from the row count rather than the block count: unlike a session sequence (many
    # blocks of ~30 rows each), this figure is usually 2 blocks, so block count alone would
    # size a ~70x70 matrix the same as a 2x2 one
    fig_width = max(5, 0.06 * sum(Ws.shape[0] for Ws in block_Ws))
    plot_w_comparison(block_Ws, block_labels, block_modalities, title, fig_width,
                      angle_clim=angle_clim, angle_white_sigma=angle_white_sigma,
                       block_t0_positions=block_t0_positions, save_path=save_path,
                       save_path_svg=save_path_svg, markdown_path=markdown_path,
                       block_metrics=block_metrics, min_metric=min_metric,
                       highlight_positions=highlight_positions, reference_block=reference_block,
                       as_column=as_column, mask_chance_band=mask_chance_band)
    return save_path, save_path_svg




# ===========================================================================
# The two-panel session summary - one figure for a whole sequence of sessions
# ===========================================================================
# Poster type sizes, applied around the figure rather than to rcParams globally (a
# notebook that re-runs cells in any order can't rely on a module-level rcParams write -
# see the same pattern in svm_analysis_2026's W_ANGLE_RC). Everything here is read at
# poster distance and the panels are deliberately small, so the type is what has to give
# ground last: >=13pt for anything with a number in it.
SESSION_SUMMARY_RC = {
    'axes.labelsize': 17, 'axes.titlesize': 17,
    'xtick.labelsize': 16, 'ytick.labelsize': 16,
    'legend.fontsize': 13, 'font.size': 15,
}
# 120mm wide, the width this panel is printed at - fixed here rather than derived from the
# session count so that what is checked on screen is what goes on the poster, and so the
# type sizes above mean what they say. The HEIGHT is derived instead (see
# SESSION_SUMMARY_PANEL_MM): each panel is drawn the same height as the score row of the
# per-session panels, which is what lets a reader carry an eye level from one figure to the
# other, and the figure ends up shorter than it is wide.
SESSION_SUMMARY_WIDTH_MM = 120
# One panel's own axes height, in mm: plot_session_analysis' score row is 1 height-ratio
# unit and its fig_height is 1.3in per unit (see there), i.e. 1.3in = 33mm.
SESSION_SUMMARY_PANEL_MM = 33.0
# What tight_layout eats around the panels at these type sizes - the x tick/axis-name band,
# the title, the gap between panels. Measured off a rendered figure rather than guessed; it
# only has to be close, since being a millimetre out moves the panel height by the same
# millimetre and nothing else.
SESSION_SUMMARY_MARGIN_MM = 30.8
# The two rows, top to bottom: the decoding score FIRST, the variance below it. That order
# is the reading order - whether the decoder worked at all is the precondition for the
# variance number underneath meaning anything.
# Both y labels are abbreviated on purpose: a rotated label longer than its own panel is
# tall gets clipped at the figure edge. ylabel=None falls back to METRIC_SPECS' full name.
# `yticks` is deliberately sparse - three gridless ticks are enough to read a level off, and
# a dense axis is the first thing that stops working when the panel is scaled onto a poster.
# `gridlines` are the dashed reference lines the per-session score row draws (at .25/.75,
# never at .5 - see _plot_f1_lines); the metric row here repeats that row's whole y-axis
# annotation - same limits, same ticks, same dashed lines - so a level read in one figure
# means the same thing in the other.
SESSION_SUMMARY_ROWS = (
    {'key': 'metric', 'ylabel': 'bal. accuracy', 'ylim': (0.1, 0.9),
     'yticks': (0.25, 0.5, 0.75), 'gridlines': (0.25, 0.75)},
    {'key': 'var_explained', 'ylabel': '% var. by W', 'ylim': (0, 5), 'yticks': (0, 2, 4)},
)


def _summary_series_stratum(spec, ccgp_w_train_stratum):
    """{y_name: train_stratum} for one series entry - its own `train_stratum` (a scalar,
    the shorthand a stratum-comparison caller wants) or `ccgp_w_train_stratum` (the dict
    form), falling back to the figure-wide dict. Inert on a naive table either way."""
    if spec.get('train_stratum') is not None:
        return {spec['y_name']: spec['train_stratum']}
    return spec.get('ccgp_w_train_stratum', ccgp_w_train_stratum)


def w_session_summary_values(session_ids, series, which_intervals, fits_by_analytic, t0_events,
                             fr_by_session=None, x_modality='mPFC', ith_to_dt_session=None,
                             ccgp_w_train_stratum=None, min_metric=None,
                             metric='balanced_accuracy', verbose=True):
    """One tidy row per (session, series, interval, timebin), with BOTH summary numbers on
    it: `var_explained` (% of that trial window's population variance lying along that
    timebin's own fitted W) and `metric` (that same timebin's decoding score).

    The two are computed in one pass because they are read against each other - a W is only
    a meaningful direction where its own classifier worked - and separating them would mean
    two passes over the same fits and the same firing rate.

    `series` is the comparison, one dict per line drawn:

        [dict(analytic='SVML1...CCGP_noNorm', y_name='cue', label='cue', train_stratum=1), ...]

    naming the table to read (`analytic`, a key of `fits_by_analytic`) and the target
    (`y_name`), plus an optional `label`, an optional `analysis` and either `train_stratum`
    (scalar) or `ccgp_w_train_stratum` (dict) overriding the figure-wide election. Series
    can differ in table AND target and still share one axes: the quantity is a fraction of
    that session's own population variance in that window, so the denominator is shared.
    What they do NOT share is trial count - a CCGP direction sees one stratum's trials - so
    part of any offset between two series can be an n effect rather than one axis genuinely
    being more dominant.

    The election matters and is not cosmetic: on a CCGP table a target has TWO rows per
    (interval, timebin), one per trained-on direction, and without one this would silently
    average across directions. It goes through the same _filter_analysis/_apply_ccgp_election
    the panels use, and the result is checked for one-row-per-timebin rather than assumed.

    `fr_by_session` maps a session id to that session's firing rate for `x_modality` (see
    plotlib.session_data.session_fr_for_modality). Loading it is minutes of I/O, so it is
    the caller's to hold: omit it and `var_explained` comes back all-NaN, which is exactly
    the metric-only figure (and costs no I/O at all). `min_metric`, if given, drops timebins
    whose own classifier scored below it - the same threshold the panels dim their heatmaps
    with, applied here as an exclusion, since the variance explained by a W nobody could
    decode with is a statement about a random-ish direction. How many bins each series loses
    is printed: a hard threshold can quietly empty a weak session.
    """
    metric_col = METRIC_SPECS[metric]['aggr_col']
    labels = [spec.get('label', f"{spec['y_name']} ({spec['analytic']})") for spec in series]
    dropped, records = {}, []

    for s_id in session_ids:
        s_label = f"S{(ith_to_dt_session or {}).get(s_id, '?')}"
        if isinstance((ith_to_dt_session or {}).get(s_id), int):
            s_label = f"S{ith_to_dt_session[s_id]:02d}"
        fr_x = (fr_by_session or {}).get(s_id)
        stacks = {}
        for spec, series_label in zip(series, labels):
            fits = fits_by_analytic[spec['analytic']]
            if s_id not in fits.index.unique('session_id'):
                if verbose:
                    print(f"{s_label}: not in {spec['analytic']}, skipped")
                continue
            rows = fits.xs(s_id, level='session_id')
            rows = rows[(rows.X_modality == x_modality) & (rows.predict_y_name == spec['y_name'])]
            rows = _apply_ccgp_election(_filter_analysis(rows, spec.get('analysis')),
                                        _summary_series_stratum(spec, ccgp_w_train_stratum))
            if rows.empty:
                if verbose:
                    print(f"{s_label}: no {spec['y_name']} fit in {spec['analytic']}, skipped")
                continue
            _check_one_row_per_timebin(rows, f"{spec['y_name']} in {s_id} ({spec['analytic']})")
            # one resolution per table, not per timebin: whether the fit's L2 step ran is a
            # property of the table (see plotlib.fit_samples.mode_for_frame)
            normalize = fitspl.mode_for_frame(rows)
            for interval in which_intervals:
                iv_rows = rows[rows.interval_name == interval].sort_values('timebin')
                if min_metric is not None and metric_col in iv_rows.columns:
                    # ~(x >= t), not (x < t): a NaN score (a collapsed fit) has to FAIL the
                    # threshold, and `<` is False on NaN, which would keep exactly those bins
                    below = ~(iv_rows[metric_col] >= min_metric)
                    dropped[series_label] = dropped.get(series_label, 0) + int(below.sum())
                    iv_rows = iv_rows[~below]
                if iv_rows.empty:
                    continue
                if fr_x is not None and interval not in stacks:
                    stacks[interval] = get_interval_trial_stack(s_id, t0_events, fr_x, interval)[0]
                for _, row in iv_rows.iterrows():
                    ve = np.nan
                    if fr_x is not None:
                        samples = timebin_samples(stacks[interval], int(row['timebin']),
                                                   normalize=normalize)
                        w = np.asarray(row['w'], dtype=float).ravel()
                        if samples is not None and w.size == stacks[interval].shape[2]:
                            # nan for a zero-norm w (L1 shrank everything away) - the same
                            # gaps the per-session variance row has
                            ve = 100 * var_explained(samples, w)
                    score = row[metric_col] if metric_col in iv_rows.columns else np.nan
                    records.append(dict(s_id=s_id, label=s_label, series=series_label,
                                        interval=interval, timebin=int(row['timebin']),
                                        var_explained=float(ve), metric=float(score)))
    if verbose:
        for series_label, n in dropped.items():
            print(f"{series_label}: {n} timebins below {metric} {min_metric}, excluded")
    return pd.DataFrame(records)


def plot_w_session_summary(values, series, which_intervals, stat='median',
                           metric='balanced_accuracy', rows=SESSION_SUMMARY_ROWS,
                           ylims=None, col_w=0.26, point_size=20, point_alpha=0.35,
                           point_color='0.55', stat_color='black', stat_width=0.36,
                           width_mm=SESSION_SUMMARY_WIDTH_MM,
                           panel_mm=SESSION_SUMMARY_PANEL_MM,
                           margin_mm=SESSION_SUMMARY_MARGIN_MM, title=None,
                           show_legend=True, save_dir=None, stem=None):
    """Two panels over one session sequence, sharing the x axis: the decoding score each
    fitted W reached on top, the variance that same W explains below.

    They belong on one x axis rather than in two figures because the lower panel is only
    interpretable through the upper one - a direction that explains a lot of variance in a
    session whose classifier sat at chance is not a task axis, it is a random-ish direction
    that happens to lie in a big mode - and reading that off two separately-scaled figures
    is exactly the comparison people get wrong. `values` is w_session_summary_values'
    frame; a row of `rows` whose column is all-NaN (no firing rate was loaded, say) is
    dropped, so the metric-only call is the same call with one panel.

    Each session shows the raw cloud of its own timebins per (series, interval), offset by
    `col_w` within the session, with `stat` ('median' - what the original figure used - or
    'mean'; these clouds are right-skewed, so a mean sits a few tenths above) drawn as a
    short horizontal line through it. The stats are deliberately NOT joined across
    sessions: a line between them asserts a trajectory through the gaps, which is a
    stronger claim than seven medians support, and the clouds already carry the spread that
    such a line would hide.

    Points and stat lines are neutral grey/black by default rather than coloured: with one
    series in the panel a colour encodes nothing, and the panel this sits under already
    spends the target's own colours on its two classes. A second series is told apart by
    `stat_marker` (drawn as that marker instead of a horizontal line) with
    `show_points=False`, which is how a secondary CCGP direction is added without doubling
    the ink - its median as a hollow circle, no cloud - or failing that by its own
    `linestyle`/`color`. `which_intervals` beyond the first take the next cloud marker
    shape.

    The figure is `width_mm` wide and as tall as its panels need: each is `panel_mm` of
    axes, the same height as the score row of the per-session panels (1.3in there, see
    plot_session_analysis' height_ratios), so that the two figures can be read at one eye
    level - and the metric row repeats that row's own y-axis annotation to match.

    No chance band, deliberately: chance depends on each session's own trial count, so a
    single horizontal line would be wrong for most columns. Read this for the comparison
    between sessions, and the per-session panels for "is this above chance"."""
    if values.empty:
        print('nothing to plot')
        return {'values': values, 'fig': None, 'paths': {}}

    labels = [spec.get('label', f"{spec['y_name']} ({spec['analytic']})") for spec in series]
    drawn = [(spec, lab) for spec, lab in zip(series, labels) if lab in set(values.series)]
    session_labels = sorted(values.label.unique())
    rows = [r for r in rows if values[r['key']].notna().any()]
    if not rows:
        print('nothing to plot')
        return {'values': values, 'fig': None, 'paths': {}}
    agg = {'mean': np.mean, 'median': np.median}[stat]
    markers = ['o', 's', 'D', '^']

    # Built before the figure is laid out, because the legend goes UNDER the x axis - the
    # only calls needing one at all are the dense ones (both CCGP directions, several
    # intervals), which is exactly when an in-axes legend covers the sessions it is
    # explaining - and its band is taken out of the fixed 120mm height.
    handles = []
    if show_legend and (len(drawn) > 1 or len(which_intervals) > 1):
        # a short line, drawn exactly as the stat mark is (same colour, same linestyle,
        # same weight) - with everything else in the panel neutral grey, the stat mark IS
        # the series, so a legend key that isn't one explains nothing
        handles = [Line2D([], [], color=spec.get('color', stat_color),
                          linestyle='none' if spec.get('stat_marker') else spec.get('linestyle', '-'),
                          marker=spec.get('stat_marker', ''), markerfacecolor='white',
                          markeredgewidth=1.8, markersize=7, linewidth=2.5, label=lab)
                   for spec, lab in drawn]
        if len(which_intervals) > 1:
            handles += [Line2D([], [], color=point_color, linestyle='none',
                               marker=markers[ii % len(markers)],
                               label=_interval_short_label(iv))
                        for ii, iv in enumerate(which_intervals)]
    legend_ncol = min(len(handles), 2) if handles else 1
    legend_h = 0.3 * np.ceil(len(handles) / legend_ncol) + 0.05 if handles else 0.0

    # Columns are placed around each session's own integer x, so the session tick sits in
    # the middle of its group whatever the series/interval count is.
    combos = [(k, ii) for k in range(len(drawn)) for ii in range(len(which_intervals))]
    offsets = {c: (i - (len(combos) - 1) / 2) * col_w for i, c in enumerate(combos)}
    # The stat line spans its own column and no further - it has to stay readable as "this
    # column's level", not as a segment reaching toward its neighbour. It is also why col_w
    # is as wide as it is: a dashed stat mark (how a second series is told apart) needs
    # enough length to show a dash at all.
    half_w = 0.5 * (stat_width if len(combos) == 1 else min(stat_width, 0.9 * col_w))

    with plt.rc_context(SESSION_SUMMARY_RC):
        fig, axes = plt.subplots(
            len(rows), 1, sharex=True, squeeze=False,
            figsize=(width_mm / 25.4,
                     (len(rows) * panel_mm + margin_mm) / 25.4 + legend_h))
        axes = axes[:, 0]
        for ax, row_spec in zip(axes, rows):
            for k, (spec, series_label) in enumerate(drawn):
                color = spec.get('color', stat_color)
                stat_marker = spec.get('stat_marker')
                for ii, interval in enumerate(which_intervals):
                    for si, s_label in enumerate(session_labels):
                        vals = values.loc[(values.label == s_label) & (values.interval == interval)
                                          & (values.series == series_label),
                                          row_spec['key']].to_numpy()
                        vals = vals[np.isfinite(vals)]
                        if not vals.size:
                            continue
                        x = si + offsets[(k, ii)]
                        if spec.get('show_points', True):
                            ax.scatter(np.full(vals.size, x), vals, s=point_size,
                                       marker=markers[ii % len(markers)], color=point_color,
                                       alpha=point_alpha, lw=0, zorder=3)
                        if stat_marker:
                            # a marker rather than a bar: a series drawn as its stat alone
                            # (no cloud behind it) has to look like a point estimate, which
                            # a bar of the same width as its neighbour's does not
                            ax.plot(x, agg(vals), marker=stat_marker, markerfacecolor='white',
                                    markeredgecolor=color, markeredgewidth=1.8, markersize=7,
                                    linestyle='none', zorder=5)
                        else:
                            ax.hlines(agg(vals), x - half_w, x + half_w, color=color,
                                      linestyles=spec.get('linestyle', '-'), linewidth=2.5,
                                      zorder=5)
            ylim = (ylims or {}).get(row_spec['key'], row_spec['ylim'])
            if ylim is not None:
                ax.set_ylim(*ylim)
            if row_spec.get('yticks') is not None:
                ax.set_yticks(row_spec['yticks'])
            for gridline_y in row_spec.get('gridlines', ()):
                ax.axhline(gridline_y, color='k', linestyle='--', alpha=0.5, zorder=0,
                           linewidth=0.5)
            ax.set_ylabel(row_spec['ylabel'] or METRIC_SPECS[metric]['label'])
            ax.spines[['top', 'right']].set_visible(False)
            ax.margins(x=0.05)

        # A tick per session (they carry the spacing) but a label only on the first and the
        # last: the ones in between are read off the sequence, and at this type size seven
        # of them either collide or have to be rotated.
        axes[-1].set_xticks(range(len(session_labels)))
        axes[-1].set_xticklabels([lab if i in (0, len(session_labels) - 1) else ''
                                  for i, lab in enumerate(session_labels)])
        # The axis name goes on the SAME line as those two labels, in the gap they leave,
        # rather than on its own line below them - that line is otherwise empty and the
        # panel is 120mm all-in. Placed by the tick pad in points (not set_xlabel, whose
        # labelpad measures from the axes and would drift with the tick label height).
        tick_pad = plt.rcParams['xtick.major.size'] + plt.rcParams['xtick.major.pad']
        axes[-1].annotate('SessionID', xy=(0.5, 0), xycoords='axes fraction',
                          xytext=(0, -tick_pad), textcoords='offset points',
                          ha='center', va='top', fontsize=plt.rcParams['axes.labelsize'],
                          annotation_clip=False)
        if title:
            axes[0].set_title(title)

        # rect reserves the legend band with tight_layout's own machinery rather than
        # leaving it to bbox_inches at save time - save_figure writes the figure as it is,
        # so anything not reserved here is simply drawn over the panels.
        fig.tight_layout(h_pad=0.6,
                         rect=(0, legend_h / fig.get_figheight() if handles else 0, 1, 1))
        if handles:
            fig.legend(handles=handles, frameon=False, loc='lower center', ncol=legend_ncol,
                       handlelength=1.4, labelspacing=0.25, columnspacing=1.2,
                       borderaxespad=0.2)

    paths = save_figure(fig, save_dir, stem) if save_dir and stem else {}
    return {'values': values, 'fig': fig, 'stat': stat, 'paths': paths}


# ===========================================================================
# Cue-vs-choice alignment panel
# ===========================================================================
# The question this figure asks, in one panel per session: are the two representations
# there at the same TIME, and when they are, do they point the same WAY? Row 0 answers the
# first (each target's min_metric indicator, one strip per target), row 1 the second (every
# cue timebin's W against every choice timebin's W). The box drawn around a co-active run
# in row 0 is redrawn on row 1's DIAGONAL, so the cells that answer "aligned?" for exactly
# the timebins that answered "co-active?" are the ones marked - the diagonal because both
# axes are the same timebin grid, so cell (t, t) is the two decoders compared at the same
# moment.
#
# Deliberately no score lines: this panel is not where the decoding curves are read (that
# is plot_session_analysis, which draws both), and two curves plus two strips in a row this
# short reads as four curves. The threshold crossing is the whole of what is needed here.

# Where each target's indicator pair sits in the strip row, in axes fraction - first entry
# for the first target (cue, on top), second for the second (choice). Far enough apart to
# read as two strips, close enough that one box around both is still a small box.
ALIGNMENT_STRIP_Y = (0.66, 0.30)
# The box around a co-active run: its vertical extent in the strip row (axes fraction,
# enclosing both strips with a little air), and the style shared with its twin on the
# heatmap diagonal - the same box in both rows is what ties the two rows together, so
# these must stay one definition.
ALIGNMENT_BOX_Y = (0.10, 0.86)
ALIGNMENT_BOX_KW = dict(fill=False, edgecolor='black', linewidth=0.9, linestyle=':', zorder=25)
# How tall the strip row is in height_ratios units (each unit = 1.3in of figure height,
# see plot_session_analysis) - enough for two strips and the annotation above them.
ALIGNMENT_STRIP_ROW_RATIO = 0.42
# How many CONSECUTIVE timebins both decoders must clear min_metric on before the run gets
# a box. A single co-active timebin is what two independent noisy curves produce all the
# time; a run is the claim worth marking. 3 at 40ms bins is ~120ms.
CO_ACTIVE_MIN_BINS = 3
# How the metric is named in the strip row's key. Not METRIC_SPECS' own `label`: that is an
# axis label ('Balanced Accuracy'), and spelled out in front of the two class-colour pairs
# it pushes them off the end of a two-interval panel (~2.8in wide).
ALIGNMENT_METRIC_LABEL = {'balanced_accuracy': 'bal. acc', 'f1': 'F1'}


def _contingency_flip(y_names):
    """Which of two targets' W to negate so that the task's own CORRECT pairing comes out
    ALIGNED (red) on the angle heatmap, rather than anti-aligned (blue).

    The problem this fixes is a pure labelling accident, not anything about the data. Both
    targets were fit in their own 0/1 encoding - cue1=0/cue2=1, skip=0/stop=1 - and a
    fitted W always points from its class 0 toward its class 1 (see plotlib/fit_samples.py:
    class 1 is higher by construction). But at R1 the correct behaviour is cue1 -> STOP and
    cue2 -> SKIP (CORRECTNESS_SPECS: correct_R1 is `cue != choice`), i.e. the pairing runs
    along the ANTI-diagonal of those two encodings. So a population that represents the
    contingency perfectly has its cue axis (C1->C2) pointing opposite its choice axis
    (skip->stop), and reads as blue - "anti-aligned" - purely because of how the two labels
    happen to be numbered. Negating one of the two W's re-numbers it (choice_R1 becomes
    skip=1/stop=0), which leaves every angle's MAGNITUDE untouched (an angle becomes
    180 - itself) and puts "correct" at red.

    Derived from CORRECTNESS_SPECS rather than hardcoded, so the R2 case is right for free:
    correct_R2 is `cue == choice` (cue2 -> stop at R2), already diagonal, so a
    (cue, choice_R2) panel needs no flip and gets none. Returns the y_name to negate, or
    None if this pair's correct pairing is already the aligned one (or isn't a
    cue/choice pair at all - two arbitrary targets have no contingency to line up with).
    """
    cue_name = next((y for y in y_names if y == 'cue'), None)
    if cue_name is None:
        return None
    for spec in CORRECTNESS_SPECS:
        choice_name = next((y for y in y_names if y == spec['choice_name']), None)
        if choice_name is None:
            continue
        # is_correct(1, 1): does "cue's class 1 together with the choice's class 1" mean
        # correct? If not, the correct pairing is the anti-diagonal one and one W flips.
        return None if spec['is_correct'](1, 1) else choice_name
    return None


def _co_active_runs(mask, min_bins):
    """Inclusive (start, stop) index runs of at least `min_bins` consecutive True in
    `mask`. Interval boundaries need no special handling: the separator row between two
    blocks carries a NaN score for every target (see _w_block_with_interval_separators),
    NaN >= min_metric is False, so a run can never bridge two intervals."""
    mask = np.asarray(mask, dtype=bool)
    runs, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_bins:
                runs.append((start, i - 1))
            start = None
    if start is not None and len(mask) - start >= min_bins:
        runs.append((start, len(mask) - 1))
    return runs


def _alignment_target_block(s_id, t0_events, svm_results, x_modality, y_name, which_intervals,
                             metric_col, smooth=None, ccgp_w_train_stratum=None, interval_x0s=None):
    """One target's elected W block + score, on the shared block-row x-axis - the same
    pipeline plot_session_analysis runs per column (get_session_data -> elect a CCGP
    direction -> _ordered_svm_res -> _w_block_with_interval_separators), collected into a
    dict instead of drawn, so the alignment panel compares two targets that were resolved
    exactly the way that figure would have resolved them one at a time.

    Also returns which intervals actually contributed rows (`intervals`) and how many each
    contributed (`counts`): _w_block_with_interval_separators silently leaves an interval
    with no fitted rows out of the block entirely, so the x-ticks and the heatmap's
    per-block t0 segments have to be built from what came back, not from which_intervals."""
    svm_res, svm_res_secondary, _, _, _ = get_session_data(
        s_id, t0_events, None, svm_results, x_modality, y_name, which_intervals,
        ccgp_w_train_stratum=ccgp_w_train_stratum, need_positions=False)
    if svm_res.empty:
        raise ValueError(f"no {y_name} fit for session {s_id} / {x_modality} / {which_intervals} - "
                          f"this panel compares two targets, so one missing leaves nothing to compare")
    elected = _resolve_ccgp_w_stratum(svm_res, svm_res_secondary, y_name, ccgp_w_train_stratum)
    ordered = _ordered_svm_res(_rows_for_stratum(svm_res, svm_res_secondary, elected), which_intervals)
    Ws, t0_positions, metric_values = _w_block_with_interval_separators(
        ordered, which_intervals, metric_col=metric_col, smooth=smooth, interval_x0s=interval_x0s)
    present = [iv for iv in which_intervals if (ordered['interval_name'] == iv).any()]
    conditioner = (svm_res['conditioner'].dropna().iloc[0]
                   if elected is not None and svm_res['conditioner'].notna().any() else None)
    return dict(y_name=y_name, elected=elected, conditioner=conditioner, Ws=Ws,
                t0=t0_positions, metric=metric_values, intervals=present,
                counts=[int((ordered['interval_name'] == iv).sum()) for iv in present])


def _alignment_flip_note(blocks, compact=False):
    """The key's second line when a W has been negated (see _contingency_flip): what RED
    now means, named as the two class labels whose directions the two W's point along.

        red: C2 & skip (correct)

    Drawn whenever the flip is in effect, not optionally: the heatmap's colours mean the
    opposite of what the raw fitted signs would give, and a panel that doesn't say so is a
    panel read backwards. The two class names carry their own colours, the same ones the
    strips above are drawn in.

    A W points toward its own class 1 - except a flipped one, which now points toward its
    class 0 - so that pair of classes is exactly what "aligned" (red) means here.
    `compact` drops the trailing "(correct)" - on a single-interval panel the full line is
    wider than the panel, and the class pair is the part that cannot be guessed."""
    small = dict(fontsize=CORNER_FONTSIZE, color='black')
    bold = dict(fontsize=CORNER_FONTSIZE, fontweight='bold')
    row = [('red: ', small)]
    for i, block in enumerate(blocks):
        level = 0 if block.get('flipped') else 1
        if i:
            row.append((' & ', small))
        row.append((_level_label(block['y_name'], level),
                    dict(bold, color=_covariate_color(block['y_name'], level))))
    if not compact:
        row.append((' (correct)', small))
    return row


def _alignment_annotation_rows(blocks, metric_label, min_metric, wrap=False):
    """The strip row's key, as rows of (text, textprops) segments for _draw_text_rows:

        bal. acc > 0.55:   C1/C2   skip/stop

    Each target named by the two CLASSES it separates, in the two colours its own strip is
    drawn in (modality_colors, via _draw_min_metric_indicator) and in the strips' own
    top-to-bottom order - so which strip is which is read off the colours rather than off a
    legend, the same trick _ccgp_sentence_segments uses in the score row.

    `wrap` puts the threshold clause on its own line above the two colour pairs - what a
    single-interval panel (1.4in wide) needs, where the one-line form is wider than the
    whole panel."""
    small = dict(fontsize=CORNER_FONTSIZE, color='black')
    bold = dict(fontsize=CORNER_FONTSIZE + 2, fontweight='bold')
    clause = [(f'{metric_label} > {min_metric}:' + ('' if wrap else '   '), small)]
    pairs = []
    for i, block in enumerate(blocks):
        y_name = block['y_name']
        if i:
            pairs.append(('   ', small))
        pairs += [(_level_label(y_name, 0), dict(bold, color=_covariate_color(y_name, 0))),
                  ('/', small),
                  (_level_label(y_name, 1), dict(bold, color=_covariate_color(y_name, 1)))]
    rows = [clause, pairs] if wrap else [clause + pairs]
    if any(block.get('flipped') for block in blocks):
        # Two rows either way. A third would run into the suptitle on a narrow panel (the
        # figure's height is derived from its rows, and a single-interval one has no slack
        # for a taller key), so there the note TAKES the threshold clause's row: what red
        # means and which colour is which class cannot be guessed from the panel, while the
        # threshold is one number the caller cell and the poster caption both state.
        rows = ([pairs] if wrap else rows) + [_alignment_flip_note(blocks, compact=wrap)]
    return rows


def plot_session_cue_choice_alignment(
        s_id, t0_events, svm_results, x_modality, which_intervals,
        y_names=('cue', 'choice_R1'), metric='balanced_accuracy', min_metric=0.55,
        smooth=None, ccgp_w_train_stratum=None, co_active_min_bins=CO_ACTIVE_MIN_BINS,
        flip_w='task',
        angle_clim=ANGLE_CLIM, angle_white_sigma=ANGLE_WHITE_SIGMA, mask_chance_band=False,
        show_colorbar=False, show_ylabel=True, show_annotation=True, show_boxes=True,
        ith_to_dt_session=None, save_dir=None, verbose=True):
    """One session, two rows: WHEN each of two targets is decodable, and whether the two
    decoders' directions are aligned WHERE they overlap.

    Row 0 draws nothing but each target's min_metric indicator - the same two-tone strip
    the score row of plot_session_analysis carries along its bottom (see
    _draw_min_metric_indicator; orange/purple for cue, the two blues for choice_R1), the
    first target's on top, the second's below it. Row 1 is the cross-target angle heatmap:
    cell (i, j) is the angle between the first target's W at timebin i (y axis) and the
    second target's W at timebin j (x axis), on the same colour scale as every other angle
    heatmap here (_angle_colormapping - red = aligned, blue = anti-aligned, white =
    orthogonal, i.e. two directions with nothing to do with each other), dimmed by
    _angle_alpha_mask wherever either side's own decoder scored below min_metric.
    Comparing the two is only meaningful because both W's live in the same feature space -
    the same units, in the same order, of this session's `x_modality` population - which is
    also why a dimensionality mismatch raises rather than being drawn.

    A run of at least `co_active_min_bins` consecutive timebins where BOTH targets clear
    min_metric gets a box in row 0, and the same box again on row 1's diagonal, spanning
    that run on both axes (a run of 4 -> a 4x4 box at (t0..t3, t0..t3)). That pairing is
    the point of the figure: the boxed cells are the two representations compared at the
    moments they were both present, and how red/blue/white that square is answers whether
    being co-active means being aligned. Everything outside it is the same comparison at
    moments when at least one of the two wasn't there to compare.

    `flip_w` fixes a labelling accident that would otherwise make the heatmap read
    backwards. Each target was fit in its own 0/1 encoding (cue1=0/cue2=1, skip=0/stop=1)
    and a fitted W points from class 0 toward class 1, but the correct behaviour at R1 is
    cue1 -> stop and cue2 -> skip - the anti-diagonal of those two numberings. A population
    that represents the contingency perfectly therefore has its two axes pointing opposite
    ways and comes out BLUE, for no reason but how the labels are numbered. `'task'` (the
    default) negates one of the two W's so that the task's own correct pairing is the RED
    one, deriving which one from CORRECTNESS_SPECS - so a (cue, choice_R2) panel, whose
    correct pairing is already diagonal, is correctly left alone (see _contingency_flip).
    Negating a W changes no angle's magnitude (an angle becomes 180 - itself) and nothing
    at all about the strips, which are about decodability and have no direction. `'fitted'`
    (or None) keeps the raw fitted signs; a target name (or list of them) negates exactly
    those. Whenever a flip is in effect the key gains a line saying what red now means
    ("red: C2 & skip (correct)") - the colours mean the opposite of the raw fit, so the
    panel has to say so.

    `min_metric` is required (it is what both rows key off) and is read on `metric`'s
    scale - 0.55 for 'balanced_accuracy', where chance is 0.5. `smooth` applies the same
    rolling mean per interval that plot_session_analysis's score row uses, so a strip turns
    off where the drawn curve dips rather than on isolated noisy timebins. `y_names` is a
    2-tuple, first = the top strip and the heatmap's y axis. `ccgp_w_train_stratum` elects
    each target's CCGP direction exactly as in plot_session_analysis (a
    {y_predicted_name: train_stratum} dict; a no-op for naive fits). `show_ylabel` adds the
    two axis labels naming which target is on which heatmap axis - off for panels being
    tiled into one poster figure, where that is said once beside the assembled canvas.
    `show_annotation` draws the small class-colour key above the strips
    (_alignment_annotation_rows, wrapped onto two lines on a single-interval panel);
    `show_boxes=False` drops the co-active boxes from
    both rows without changing anything else, for a look at the raw comparison.

    With `save_dir`, saves a .png and a .svg named after the session (like
    plot_session_analysis's own files, with a `_cuechoice` suffix so the two never
    collide) and returns ([png_path], [svg_path]) - the one-element lists build_canvas/
    build_canvas_svg take as a column. Returns ([], []) with no save_dir. The co-active
    runs themselves are printed (`verbose`) rather than returned, since what a caller
    tiling panels needs back is the paths.
    """
    if len(y_names) != 2:
        raise ValueError(f"y_names must be exactly two targets to compare, got {y_names!r}")
    spec = METRIC_SPECS[metric]
    # The single source of interval geometry for both rows, same as plot_session_analysis.
    interval_x0s = interval_x0_map(t0_events, which_intervals, s_id=s_id, warn=False)
    blocks = [_alignment_target_block(s_id, t0_events, svm_results, x_modality, y_name,
                                       which_intervals, spec['aggr_col'], smooth=smooth,
                                       ccgp_w_train_stratum=ccgp_w_train_stratum,
                                       interval_x0s=interval_x0s)
              for y_name in y_names]
    top, bottom = blocks
    # Both rows and both axes of the heatmap are ONE timebin grid - a box on the diagonal
    # means nothing otherwise - so the two blocks have to agree on it exactly. They do
    # whenever both targets were fit over the same intervals (the normal case, one fit
    # table); if one target is missing an interval the blocks silently differ in length and
    # every cell would be off by that interval's width.
    if top['Ws'].shape[0] != bottom['Ws'].shape[0] or top['intervals'] != bottom['intervals']:
        raise ValueError(
            f"the two targets do not share a timebin grid for session {s_id}: "
            f"{top['y_name']} has {top['Ws'].shape[0]} rows over {top['intervals']}, "
            f"{bottom['y_name']} has {bottom['Ws'].shape[0]} over {bottom['intervals']}")
    if top['Ws'].shape[1] != bottom['Ws'].shape[1]:
        raise ValueError(
            f"the two targets' W's have different dimensionality ({top['Ws'].shape[1]} vs "
            f"{bottom['Ws'].shape[1]}) - an angle between them would not be an angle in any "
            f"one space. Both must be fit on the same X_modality population.")

    # The sign convention, applied to the W's themselves rather than to the finished angle
    # matrix: the flip is a statement about one decoder's direction, and doing it here keeps
    # `flipped` on the block that owns it, which is what the key's "red means" line reads.
    if flip_w in (None, 'fitted'):
        flip_names = []
    elif flip_w == 'task':
        flip_names = [name for name in [_contingency_flip(y_names)] if name is not None]
    else:
        flip_names = [flip_w] if isinstance(flip_w, str) else list(flip_w)
    unknown = [name for name in flip_names if name not in y_names]
    if unknown:
        raise ValueError(f"flip_w names {unknown} which this panel does not plot ({list(y_names)})")
    for block in blocks:
        if block['y_name'] in flip_names:
            block['Ws'] = -block['Ws']
            block['flipped'] = True

    n_bins = top['Ws'].shape[0]
    x_bins = np.arange(n_bins)

    # Same sizing rules as plot_session_analysis, so a panel from here tiles against one
    # from there: width tracks the number of intervals, the heatmap's height_ratio is
    # whatever makes its box as tall as the panel is wide, and each ratio unit is 1.3in.
    column_width = 1.4 * len(which_intervals)
    height_ratios = [ALIGNMENT_STRIP_ROW_RATIO, column_width / 1.3]
    fig_height = 1.3 * sum(height_ratios)
    fig_width = column_width + (1.0 if show_ylabel else 0)
    fig, (strip_ax, hm_ax) = plt.subplots(nrows=2, ncols=1, figsize=(fig_width, fig_height),
                                           height_ratios=height_ratios, sharex=True, dpi=150)
    plt.subplots_adjust(hspace=0.15)

    # --- row 0: one indicator strip per target -----------------------------------------
    above = []
    for block, y_frac in zip(blocks, ALIGNMENT_STRIP_Y):
        _draw_min_metric_indicator(strip_ax, x_bins, block['metric'], block['y_name'],
                                    min_metric, y_frac=y_frac)
        above.append(np.asarray(block['metric'], dtype=float) >= min_metric)
    for t0 in top['t0']:
        strip_ax.axvline(x=t0, **T0_LINE_KW)
    strip_ax.set_ylim(0, 1)
    strip_ax.set_yticks([])
    strip_ax.tick_params(labelbottom=False, length=0)
    for sp in strip_ax.spines.values():
        sp.set_visible(False)
    if show_annotation:
        # in_layout=False: this key is wider than a narrow panel's axes, and tight_layout
        # would otherwise shrink the AXES to make room for it (a single-interval panel came
        # out as a sliver). Wrapped onto two lines below two intervals, where even
        # overflowing it would cover most of the neighbouring panel on the assembled canvas.
        _draw_text_rows(strip_ax,
                         _alignment_annotation_rows(
                             blocks, ALIGNMENT_METRIC_LABEL.get(metric, spec['label']), min_metric,
                             wrap=len(which_intervals) < 2),
                         xy=(0.0, 1.02), ha='left', va='bottom', in_layout=False)

    # --- row 1: the cross-target angle heatmap ------------------------------------------
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_sim = (np.dot(top['Ws'], bottom['Ws'].T)
                   / (np.linalg.norm(top['Ws'], axis=1)[:, np.newaxis]
                      * np.linalg.norm(bottom['Ws'], axis=1)[np.newaxis, :]))
        angles = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * (180 / np.pi)
    lower, upper = angle_2std_thr[x_modality]
    alpha = _angle_alpha_mask(angles, lower, upper, mask_chance_band=mask_chance_band,
                               metrics_i=top['metric'], metrics_j=bottom['metric'],
                               min_metric=min_metric)
    alpha[np.isnan(angles)] = 1.0
    cmap, hm_norm = _angle_colormapping(angle_clim, angle_white_sigma)
    # meshgrid + pcolormesh, not imshow, for the same cell-centred-on-integer alignment
    # every other heatmap here uses: C[i, j] lands at x=j (bottom's timebin), y=i (top's).
    X, Y = np.meshgrid(x_bins, x_bins)
    im = hm_ax.pcolormesh(X, Y, np.ma.masked_invalid(angles), alpha=alpha, cmap=cmap, norm=hm_norm)

    # t0 reference lines as per-block segments rather than full-width lines - a full line
    # runs through the blank separator between intervals and closes the gap visually,
    # making the concatenated blocks read as one matrix (see create_subplot).
    bands, pos = [], 0
    for n in top['counts']:
        if bands:
            pos += 1
        bands.append((pos - 0.5, pos + n - 0.5))
        pos += n
    for t0 in top['t0']:
        for band_lo, band_hi in bands:
            hm_ax.vlines(t0, band_lo, band_hi, zorder=3, **T0_LINE_KW)
            hm_ax.hlines(t0, band_lo, band_hi, zorder=3, **T0_LINE_KW)

    # --- the co-active runs, boxed in both rows ------------------------------------------
    runs = _co_active_runs(np.logical_and(*above), co_active_min_bins) if show_boxes else []
    for start, stop in runs:
        width = stop - start + 1
        strip_ax.add_patch(Rectangle((start - 0.5, ALIGNMENT_BOX_Y[0]), width,
                                      ALIGNMENT_BOX_Y[1] - ALIGNMENT_BOX_Y[0],
                                      transform=strip_ax.get_xaxis_transform(),
                                      clip_on=False, **ALIGNMENT_BOX_KW))
        hm_ax.add_patch(Rectangle((start - 0.5, start - 0.5), width, width, **ALIGNMENT_BOX_KW))

    xticks, xtick_labels = [], []
    for interval, t0 in zip(top['intervals'], top['t0']):
        xticks.extend([t0, t0 + 25])
        xtick_labels.extend([f'$t_{{{_interval_short_label(interval)}}}$', '1s'])
    hm_ax.set_xticks(xticks)
    hm_ax.set_xticklabels(xtick_labels, fontsize=TICK_FONTSIZE)
    hm_ax.set_xlim(-0.5, n_bins - 0.5)
    hm_ax.set_ylim(-0.5, n_bins - 0.5)
    # No y ticks at all - not just no numbers: the y axis is the same timebin grid as the x
    # axis, already labelled below it, so the default locator's tick MARKS are the only
    # thing that would be left of them, hanging off a panel that draws no spines either.
    hm_ax.set_yticks([])
    for sp in hm_ax.spines.values():
        sp.set_visible(False)
    if show_ylabel:
        hm_ax.set_ylabel(f"{top['y_name']} W")
        hm_ax.set_xlabel(f"{bottom['y_name']} W")

    if show_colorbar:
        cbar = plt.colorbar(im, ax=hm_ax)
        cbar.set_ticks([t for t in (0, 60, 90, 120, 180)
                        if _angle_clim(angle_clim)[0] <= t <= _angle_clim(angle_clim)[1]]
                       or list(_angle_clim(angle_clim)))
        cbar.set_label('Angle between SVM planes (degrees)', labelpad=15)

    s_num = (ith_to_dt_session or {}).get(s_id, "Unknown")
    title = f'S{s_num:02d}' if isinstance(s_num, int) else f'S{s_num}'
    plt.suptitle(title, y=.99, **SUPTITLE_KW)
    plt.tight_layout()

    if verbose:
        runs_txt = ', '.join(f'{a}-{b} ({b - a + 1} bins)' for a, b in runs) or '(none)'
        flipped = [b['y_name'] for b in blocks if b.get('flipped')]
        sign_txt = (f" | W sign: {', '.join(flipped)} negated, so red = the correct pairing"
                    if flipped else " | W signs as fitted")
        print(f"{title} {top['y_name']}+{bottom['y_name']} co-active runs "
              f"(>= {co_active_min_bins} bins): {runs_txt}{sign_txt}")

    saved_png_paths, saved_svg_paths = [], []
    if save_dir is not None:
        png_path = os.path.join(save_dir, f'session_{s_id}_S{s_num}_cuechoice.png')
        svg_path = os.path.join(save_dir, f'session_{s_id}_S{s_num}_cuechoice.svg')
        fig.savefig(png_path, dpi=150)
        fig.savefig(svg_path)
        saved_png_paths.append(png_path)
        saved_svg_paths.append(svg_path)
    plt.show()
    return saved_png_paths, saved_svg_paths
