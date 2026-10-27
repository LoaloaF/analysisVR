"""Cross-session summary views of the SVM decoding metric.

Companion to `svm_plotting_lib.py` (the per-session poster figure and its dependency
closure), kept as its own module because it answers a different question: instead of one
figure per session, `plot_metric_heatmap_sessions` puts *every* session in one figure as
one row per session, so the whole sequence's decoding time course is readable at a glance.
`plot_metric_distribution` collapses the time axis entirely, reducing one
(interval, target) to the distribution of its scores and of its confidence-interval widths -
the small figure to put two SVM hyperparameter settings side by side and see which fits
better and which fits more tightly.
It reuses that library's data/metric conventions (METRIC_SPECS, the CCGP election, the
per-interval smoothing, the t0/'1s' x-tick scheme) by importing them rather than
re-deriving them - so a change there (e.g. a new metric) carries over here for free.

Like svm_plotting_lib, nothing in here does I/O or touches the NAS on import; every
function takes its data (svm_results) as an explicit argument.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

from svm_plotting_lib import (
    ALL_Y_PREDICTED_NAMES, METRIC_SPECS, _apply_ccgp_election, _filter_analysis,
    _interval_short_label, _smooth,
)
from plotlib.trial_traces import interval_n_bins_left


# Display order for interval_name, mirroring `interval_specifier` in
# analytics_processing/integr_analytics.py - i.e. the order the intervals are actually
# defined/occur in a trial, which is the authoritative source; keep the two in sync if an
# interval is added there. Currently-commented-out entries there are kept here too, so
# re-enabling one sorts into its proper slot without touching this list. Only a sort key
# (see _ordered_names), not a list of intervals to plot: an interval missing from here
# still gets drawn, just sorted after these.
INTERVAL_ORDER = [
    'track_start_interval',
    'pre_cue_interval',
    'nextto_cue_interval',
    'cue_entry_interval',
    'cue_exit_interval',
    'R1_entry_interval',
    'R2_entry_interval',
    'R1_exit_interval',
    'R2_exit_interval',
    'reward1_sound_interval',
    'reward2_sound_interval',
    'ITI_interval',
]


def _ordered_names(present, canonical):
    """`present` (whatever names a dataframe column actually holds) sorted into `canonical`
    order, with any name canonical doesn't mention appended alphabetically afterwards -
    e.g. a joined target like 'cueANDstopR1' isn't in ALL_Y_PREDICTED_NAMES but still needs
    a deterministic slot. Used to resolve plot_metric_heatmap_sessions' which_intervals=
    None/y_predicted_names=None into a stable display order instead of the dataframe's own
    incidental row order."""
    present = list(present)
    known = [name for name in canonical if name in present]
    return known + sorted(name for name in present if name not in canonical)


def _resolve_intervals_targets(frames, which_intervals, y_predicted_names, x_modality, session_ids):
    """(which_intervals, y_names) with either or both resolved from what `frames` actually
    hold, when passed as None - the "show me every one there is" default both plots here
    share. Resolved against this x_modality/session slice specifically, not all of
    svm_results, so an interval or target only ever fit for another modality doesn't come
    back as an all-blank figure/panel. Whichever of the two the caller *did* pin is applied
    before the other is resolved, so e.g. naming only the track/cue/R1/R2/ITI intervals
    stops 'baseline_vs_t' (only ever fit on the reward-sound intervals) from being offered
    as a target at all. `frames` is a list so a multi-variant comparison (see
    plot_metric_distribution) resolves against the union of its variants - a target present
    in only one of them still gets a row, where every other variant simply shows nothing."""
    as_list = lambda v: (list(v) if isinstance(v, (list, tuple)) else [v]) if v is not None else None
    which_intervals, y_predicted_names = as_list(which_intervals), as_list(y_predicted_names)

    present_intervals, present_targets = set(), set()
    for frame in frames:
        rows = frame[frame['X_modality'] == x_modality]
        if session_ids is not None:
            rows = rows[rows.index.get_level_values('session_id').isin(list(session_ids))]
        if which_intervals is not None:
            rows = rows[rows['interval_name'].isin(which_intervals)]
        if y_predicted_names is not None:
            rows = rows[rows['predict_y_name'].isin(y_predicted_names)]
        present_intervals |= set(rows['interval_name'].dropna().unique())
        present_targets |= set(rows['predict_y_name'].dropna().unique())

    return (which_intervals if which_intervals is not None else _ordered_names(present_intervals, INTERVAL_ORDER),
            y_predicted_names if y_predicted_names is not None else _ordered_names(present_targets,
                                                                                   ALL_Y_PREDICTED_NAMES))


# _pooled_interval_x0 is gone: interval geometry now comes from t0_events via
# plotlib.trial_traces.interval_n_bins_left, which measures it from the stored
# intervals instead of reading the (being-removed) n_bins_left column off the fit
# rows. That also removes the need for the `interval_n_bins_left=` override kwarg,
# which only existed to patch around rows where that column was missing or wrong.


def plot_metric_heatmap_sessions(svm_results, t0_events, which_intervals=None, y_predicted_names=None,
                                  x_modality='mPFC',
                                  session_ids=None, metric='balanced_accuracy', ith_to_dt_session=None,
                                  smooth=None, vmin=0.4, vcenter=None, vmax=0.8, cmap='hot_r',
                                  ccgp_w_train_stratum=None, analysis=None, show_colorbar=True, title=None,
                                  width_per_interval=1.4, row_height=0.22,
                                  save_path=None, save_path_svg=None):
    """Sessions-by-time heatmap of the decoding metric: one figure per predict_y_name, drawn
    one after the other - each row a session, each column a timebin, cell color that
    (session, timebin) decoder's aggr score (acc_aggr or f1_aggr depending on `metric` - see
    METRIC_SPECS). The compact cross-session overview of the same quantity
    plot_session_analysis' row 1 draws as one line per session figure. Also prints the mean
    score +- spread, pooled over everything drawn and again per target (see the stats block
    below for which two spreads).

    x-axis follows the same concatenation scheme as svm_plotting_lib's panels:
    `which_intervals` left to right, a 1-column gap between consecutive intervals (left
    NaN, so it renders as a white separator - same idea as
    _w_block_with_interval_separators' zero rows), and two x-ticks per interval,
    $t_{IntervalName}$ at its t0 plus a '1s' scale tick 25
    bins later (see _interval_short_label/_interval_x0, and the tick loop below for when
    that second tick is dropped), drawn at fontsize=7 for the same reason
    plot_session_analysis' heatmap row is. Unlike a single-session figure, one shared
    x-axis has to serve every session here, so columns are aligned by each interval's own
    timebin index and its width is the widest any plotted session has - a session fit over
    fewer timebins simply leaves white cells at that interval's right edge, and t0 comes
    from _pooled_interval_x0 (see there for the differing-n_bins_left case).

    `which_intervals`/`y_predicted_names` both default to None = every interval/target
    actually fit for this x_modality and session set, in INTERVAL_ORDER/
    ALL_Y_PREDICTED_NAMES order (see _ordered_names) - so the unparameterized call is a
    "show me everything there is" overview. `session_ids` likewise defaults to every
    session in `svm_results`, chronologically; rows are labeled S{n} via
    `ith_to_dt_session` ({session_id: index}, as elsewhere) when given.
    A (session, y_name) pair that was never fit for `x_modality` is left blank rather than
    dropped, so rows stay comparable across subplots. CCGP fits are collapsed to one row
    per timebin by `ccgp_w_train_stratum` ({y_predicted_name: train_stratum}, see
    _resolve_ccgp_w_stratum/_apply_ccgp_election) exactly as in plot_session_analysis; a
    naive fit ignores it. `smooth`, if given, is the same per-interval rolling mean the
    line panels use (see _smooth), never blending across an interval boundary.

    Color: `cmap` spread linearly over vmin..vmax - by default the sequential hot_r
    (white -> yellow -> red -> black) from 0.4 to 0.8, so chance (0.5 for a balanced
    two-class score) sits pale near the bottom of the range and only genuinely informative
    timebins carry ink. Scores outside vmin..vmax clip to the end colors rather than being
    hidden. `vcenter`, if given (e.g. 0.5 with a diverging cmap like 'RdYlGn'), switches to
    a TwoSlopeNorm pinning that value to the colormap's own midpoint instead - which is what
    makes above/below-chance readable from the color when the range isn't symmetric around
    it (a plain Normalize over 0.4..0.8 puts the midpoint color at 0.6).

    `width_per_interval`/`row_height` size each figure (inches per interval along x, per
    session along y) - raise them if the cells come out too small to read.
    `t0_events` supplies the interval geometry the t0 markers are placed with, measured
    from the stored intervals themselves (see plotlib.trial_traces.interval_n_bins_left).
    It must be the t0 table `svm_results` was actually fit against - see
    plotlib.session_data.ANALYTIC_PAIRS/check_geometry. Returns (figs,
    scores): the per-target figures in y_names order, and a {y_predicted_name: (n_sessions,
    n_columns) array} dict of exactly what's drawn."""
    spec = METRIC_SPECS[metric]

    if session_ids is None:
        session_ids = sorted(svm_results.index.unique('session_id'),
                             key=lambda s_id: pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M'))
    session_ids = list(session_ids)

    which_intervals, y_names = _resolve_intervals_targets(
        [svm_results], which_intervals, y_predicted_names, x_modality, session_ids)

    plotted = svm_results.loc[
        (svm_results['X_modality'] == x_modality) &
        (svm_results.index.get_level_values('session_id').isin(session_ids)) &
        (svm_results['predict_y_name'].isin(y_names)) &
        (svm_results['interval_name'].isin(which_intervals))
    ]
    # `analysis` first: on a v2 fit table (SVM_TaskSubspaces) the unconditioned fit and
    # both conditioned directions of a target are rows of ONE frame, so without it a cell
    # here would be an arbitrary one of several fits. 'naive'/'ccgp'; None is the v1 case,
    # where the two live in separate analytics and there is nothing to separate.
    plotted = _filter_analysis(plotted, analysis)
    plotted = _apply_ccgp_election(plotted, ccgp_w_train_stratum)

    # Column layout, shared by every figure (so the per-target figures stay comparable to
    # each other): interval width = the widest any plotted session has for it, plus the
    # 1-column separator gap between consecutive intervals.
    # A requested interval with no rows at all is DROPPED here rather than given zero width:
    # it used to keep its t0 tick, its label and its dashed line, all positioned at
    # `offset + n_bins_left` with `offset` already pointing at the *next* interval's columns,
    # so an interval that drew nothing put its marker inside its neighbour's block and
    # overprinted that block's own label. It also still consumed a separator column. The
    # skip is printed: an interval vanishing silently is what made missing data read as a
    # labelling bug.
    interval_offsets, interval_x0s, offset = {}, {}, 0
    xticks, xtick_labels = [], []
    drawn_intervals, skipped_intervals = [], []
    for which_interval in which_intervals:
        interval_rows = plotted[plotted.interval_name == which_interval]
        if interval_rows.empty:
            skipped_intervals.append(which_interval)
            continue
        n_bins = int(interval_rows.timebin.max()) + 1
        if drawn_intervals:
            offset += 1  # blank separator column, before each block except the first
        interval_offsets[which_interval] = offset
        # One shared x-axis serves every session's row here, so this is the pooled modal
        # geometry across sessions (interval_n_bins_left warns if they disagree - only
        # the dashed t0 marker is affected; the rows stay aligned by their own timebin).
        n_left = interval_n_bins_left(t0_events, which_interval)
        if n_left >= n_bins:
            # t0 sits outside the columns the fits actually produced for this interval, so
            # an unclamped marker would again land in the next block. Not hypothetical:
            # cue_exit_interval puts t0 at bin 20 of a 25-column block, leaving 5 columns of
            # margin, and any session set that fit fewer timebins there overruns it.
            print(f"WARNING: {which_interval} t0 is at bin {n_left} but only {n_bins} "
                  f"columns were fit - t0 marker clamped to the block's last column, so "
                  f"its position is approximate")
            n_left = n_bins - 1
        x_0 = offset + n_left
        interval_x0s[which_interval] = x_0
        xticks.append(x_0)
        xtick_labels.append(f'$t_{{{_interval_short_label(which_interval)}}}$')
        # the '1s' scale tick is dropped when 25 bins past t0 would land outside this
        # interval's own columns - unlike a single-session figure (create_subplot, where
        # every interval is wide enough for it), interval widths here are whatever the
        # plotted sessions happen to have, and an overflowing tick would sit on top of the
        # *next* interval's t0 label rather than annotating anything
        if x_0 + 25 < offset + n_bins:
            xticks.append(x_0 + 25)
            xtick_labels.append('1s')
        offset += n_bins
        drawn_intervals.append(which_interval)
    n_columns = offset

    if skipped_intervals:
        print(f"WARNING: no {x_modality} rows for {', '.join(skipped_intervals)} in this "
              f"target/session selection - dropped from the figure. Check the "
              f"ccgp_w_train_stratum election if the analytic does have those intervals.")
    if not drawn_intervals:
        raise ValueError(f"none of {which_intervals} has any {x_modality} rows for "
                         f"{y_names} in the given sessions")
    which_intervals = drawn_intervals

    scores = {}
    for y_name in y_names:
        mat = np.full((len(session_ids), n_columns), np.nan)
        for row_i, s_id in enumerate(session_ids):
            s_rows = plotted[(plotted.index.get_level_values('session_id') == s_id) &
                             (plotted['predict_y_name'] == y_name)]
            for which_interval in which_intervals:
                interval_rows = s_rows[s_rows.interval_name == which_interval].sort_values('timebin')
                if interval_rows.empty:
                    continue
                columns = interval_rows.timebin.to_numpy(dtype=int) + interval_offsets[which_interval]
                mat[row_i, columns] = _smooth(interval_rows[spec['aggr_col']].to_numpy(dtype=float), smooth)
        scores[y_name] = mat

    # Two different spreads, both over exactly the cells drawn above (blank cells - a
    # session/target never fit, or an interval's ragged right edge - are excluded, not
    # counted as zeros): across timebins pools every cell, so it mixes real
    # session-to-session differences with within-session variation over time; across
    # sessions first averages each session's own row down to one number, so it answers "how
    # much do sessions differ from each other" without the (much larger) within-session
    # time course inflating it. A session with no cells at all contributes to neither.
    def _mean_pm_sd(mat):
        cells = mat[np.isfinite(mat)]
        if cells.size == 0:
            return "no fits"
        row_means = np.array([row[np.isfinite(row)].mean() for row in mat if np.isfinite(row).any()])
        return (f"{cells.mean():.3f} ± {cells.std():.3f} across {cells.size} timebins | "
                f"{row_means.mean():.3f} ± {row_means.std():.3f} across {len(row_means)} sessions")

    label_width = max([len('all targets')] + [len(y_name) for y_name in y_names])
    print(f"{spec['label']} ({spec['aggr_col']}) - {x_modality}, {len(session_ids)} sessions, "
          f"intervals: {', '.join(which_intervals)}")
    # axis=1 (each session stays one row, widened by every target's timebins) so the
    # across-sessions spread is still per session, not per (session, target) pair
    print(f"  {'all targets':<{label_width}} : "
          f"{_mean_pm_sd(np.concatenate([scores[y] for y in y_names], axis=1))}")
    for y_name in y_names:
        print(f"  {y_name:<{label_width}} : {_mean_pm_sd(scores[y_name])}")

    # One figure per target, drawn one after the other - NOT subplots side by side in a
    # single figure. Side by side, a realistic call (6 targets x 6 intervals) is a ~50in
    # wide by ~4in tall figure, which any notebook front-end scales down to its own column
    # width - i.e. to ~15%, illegible. Separate figures each stay a readable width, and the
    # shared column layout/color scale above is what keeps them comparable to each other.
    # width_per_interval ~ plot_session_analysis' own (same per-timebin density); the fixed
    # additions are the session labels on the left and the colorbar on the right, neither of
    # which scales with the data.
    fig_width = width_per_interval * len(which_intervals) + 1.2 + (0.8 if show_colorbar else 0)
    fig_height = max(2.0, row_height * len(session_ids) + 1.2)
    heatmap_cmap = plt.get_cmap(cmap).with_extremes(bad='white')
    norm = (TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax) if vcenter is not None
            else Normalize(vmin=vmin, vmax=vmax))
    X, Y = np.meshgrid(np.arange(n_columns), np.arange(len(session_ids)))

    figs = []
    for y_name in y_names:
        # constrained layout, not tight_layout: tight_layout does not reserve room for the
        # label of a colorbar attached to an axes, so 'SVM Balanced Accuracy' was drawn
        # past the right edge of the canvas and clipped.
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150, layout='constrained')
        figs.append(fig)
        # thin per-cell edges: at this density (one cell per 40ms timebin) they read as a
        # faint grid that keeps individual timebins/sessions countable instead of blurring
        # into a continuous field. Passed as a per-quad RGBA array (row-major, matching the
        # score matrix' own flattening) rather than one edgecolor for the whole mesh:
        # masking a cell only suppresses its *face*, so a single edgecolor would still draw
        # a full grid of empty boxes over blank regions - the separator column, an
        # interval's ragged right edge, a never-fit session/target - making "no data" look
        # like "data that happens to be white". alpha=0 per masked quad keeps those clean.
        edge_rgba = np.zeros((scores[y_name].size, 4))
        edge_rgba[:, 3] = np.where(np.isfinite(scores[y_name]).ravel(), 0.25, 0.0)
        im = ax.pcolormesh(X, Y, np.ma.masked_invalid(scores[y_name]), cmap=heatmap_cmap, norm=norm,
                           edgecolors=edge_rgba, linewidth=0.15)
        for which_interval in which_intervals:
            ax.axvline(x=interval_x0s[which_interval], color='k', linestyle='--', alpha=0.4)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, fontsize=7)
        ax.set_xlim(-0.5, n_columns - 0.5)
        # first session at the top, reading downward like the chronological session list it is
        ax.set_ylim(len(session_ids) - 0.5, -0.5)
        ax.set_yticks(np.arange(len(session_ids)))
        ax.set_yticklabels([f'S{(ith_to_dt_session or {}).get(s_id, "?")}' for s_id in session_ids],
                           fontsize=8)
        [sp.set_visible(False) for sp in ax.spines.values()]

        if show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            # explicit ticks: a TwoSlopeNorm colorbar is two differently-scaled linear
            # halves, so matplotlib's automatic ticks land at uneven-looking spacings -
            # vcenter (where the color flips sense) is the one value that has to be labeled.
            if vcenter is not None:
                cbar.set_ticks([vmin, vcenter, (vcenter + vmax) / 2, vmax])
            cbar.set_label(f'SVM {spec["label"]}', labelpad=4)

        fig.suptitle(f'{title} - {y_name}' if title is not None
                     else f'{y_name} - {spec["label"]} - {x_modality}', fontsize=13)
        # per-target filenames: '<stem>_<y_name><ext>', so the figures don't overwrite each
        # other at one path (which is what a single-figure save_path used to mean)
        for path, dpi in ((save_path, 150), (save_path_svg, None)):
            if path is None:
                continue
            stem, ext = os.path.splitext(path)
            fig.savefig(f'{stem}_{y_name}{ext}', **({'dpi': dpi} if dpi else {}))
        plt.show()
    return figs, scores


def _variant_rows(svm_res, which_interval, y_predicted_name, x_modality, session_ids,
                   ccgp_w_train_stratum, analysis=None):
    """One variant's fit rows for a single (interval, target, X_modality): the same slice +
    CCGP election plot_metric_heatmap_sessions builds a heatmap row from, minus the
    session/timebin layout - here every row is just one sample of the distribution.
    `session_ids=None` means every session in that variant's own dataframe, so two variants
    fit on different session sets are still each shown in full (their n's differ, which the
    summary table reports)."""
    rows = svm_res.loc[
        (svm_res['X_modality'] == x_modality) &
        (svm_res['predict_y_name'] == y_predicted_name) &
        (svm_res['interval_name'] == which_interval)
    ]
    if session_ids is not None:
        rows = rows[rows.index.get_level_values('session_id').isin(list(session_ids))]
    return _apply_ccgp_election(_filter_analysis(rows, analysis), ccgp_w_train_stratum)


def _draw_variant_distributions(ax, datasets, labels, colors, plot_type, reference, show_n=True):
    """One grid cell of plot_metric_distribution: `datasets` is one 1D array per variant
    (empty arrays allowed - that variant simply isn't drawn), in `labels`/`colors` order.
    Returns True if anything was drawn, so the caller can blank an entirely empty cell.
    With show_n, corner-annotates how many fits each drawn violin/histogram actually pooled
    (one number if every variant has the same count, else one per drawn variant in the same
    left-to-right order) - the spread of a distribution doesn't mean much without it, and the
    counts genuinely differ per cell: a target isn't necessarily fit on every interval, and
    two variants can span different sessions or timebin counts."""
    drawn = [(i, data) for i, data in enumerate(datasets) if data.size]
    if not drawn:
        return False
    if show_n:
        sizes = [data.size for _, data in drawn]
        n_text = f'n={sizes[0]}' if len(set(sizes)) == 1 else 'n=' + '/'.join(str(s) for s in sizes)
        # top-right with a translucent white patch behind it: with a shared y-axis across the
        # grid, no corner is reliably empty, so the text has to stay readable over a violin
        ax.text(0.98, 0.98, n_text, transform=ax.transAxes, fontsize=6, alpha=0.75,
                ha='right', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5))
    if plot_type == 'violin':
        parts = ax.violinplot([data for _, data in drawn], positions=[i for i, _ in drawn],
                               showmeans=True, showextrema=False, widths=0.8)
        for (i, _), body in zip(drawn, parts['bodies']):
            body.set_facecolor(colors[i])
            body.set_alpha(0.6)
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(1.2)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        if reference is not None:
            ax.axhline(reference, color='k', linestyle='--', alpha=0.5, linewidth=0.8)
    else:
        pooled = np.concatenate([data for _, data in drawn])
        bins = np.linspace(pooled.min(), pooled.max(), 25)
        for i, data in drawn:
            ax.hist(data, bins=bins, histtype='step', linewidth=1.2, color=colors[i], label=labels[i])
        if reference is not None:
            ax.axvline(reference, color='k', linestyle='--', alpha=0.5, linewidth=0.8)
    return True


def plot_metric_distribution(svm_results, which_intervals=None, y_predicted_names=None, x_modality='mPFC',
                              session_ids=None, metric='balanced_accuracy', ccgp_w_train_stratum=None,
                              analysis=None,
                              plot_type='violin', chance_level=0.5, title=None, show_n=True,
                              panel_width=1.1, panel_height=1.3, save_path=None, save_path_svg=None):
    """Averages and confidence-interval sizes for a whole collection of fits at once, laid
    out on the same grid as plot_metric_heatmap_sessions: `which_intervals` left to right as
    columns, one row per predict_y_name. Each cell collapses that (interval, target)'s time
    axis and sessions entirely into a distribution - one violin (or step histogram) per
    variant - so the figure answers "which setting fits better, and which fits more
    tightly", across every interval/target in one look. See
    plot_metric_heatmap_sessions when the time course itself matters.

    Two figures are produced, both on that grid and each with its own shared y-axis (so
    cells are comparable): the per-fit score (mean_col - see METRIC_SPECS), then the width
    of that fit's own confidence interval (ci_upper_col - ci_lower_col). A better setting is
    higher in the first and narrower in the second. Both quantities are read off the *mean*
    metric, not the aggr one, and that pairing is the point: the stored CI columns are
    bootstrap percentiles of the per-iteration scores, i.e. they belong to mean_col and not
    to aggr_col (see _plot_f1_lines/_bootstrap_aggr_ci in svm_plotting_lib) - so a width
    shown here is genuinely the interval around the score shown above it. If no variant has
    CI columns at all, only the score figure is drawn.

    `svm_results` is either one dataframe or a {label: dataframe} dict - one violin per
    entry within every cell, in dict order, which is how two SVM hyperparameter settings get
    compared (labels are yours to name, e.g. {'L1 C=0.1': ..., 'L2 C=1': ...}). Each is
    sliced per cell to (interval, target, x_modality), CCGP-elected via
    `ccgp_w_train_stratum` (see _apply_ccgp_election), and pooled over every
    (session, timebin) fit - one sample per fitted decoder. `which_intervals`/
    `y_predicted_names` default to None = every one available, resolved against the union of
    the variants (see _resolve_intervals_targets); a cell with no fits is left blank rather
    than dropped, keeping the grid rectangular. `session_ids`, if given, restricts every
    variant to the same sessions - what makes variants computed over different session sets
    strictly comparable.

    `plot_type='violin'` (default) puts the variants side by side within each cell; 'hist'
    overlays step histograms instead (the row's target name stays on the y-axis label there
    too, even though that axis is counts - the row identity is more useful than the word
    'fits'). `chance_level` (None to omit) draws a reference line on the score figure only.
    `show_n=False` drops the per-cell fit-count annotation (see
    _draw_variant_distributions); the session count sits in the figure title either way.
    `panel_width`/`panel_height` are inches per grid cell. Returns (figs, summary): the one
    or two figures, and a tidy per-(target, interval, variant) DataFrame - n_fits/n_sessions/
    mean/sd/median/q10/q90 plus the CI-width mean/median - which is also printed, since a
    hyperparameter choice usually turns on a few hundredths that no violin resolves."""
    spec = METRIC_SPECS[metric]
    variants = svm_results if isinstance(svm_results, dict) else {'fits': svm_results}
    labels = list(variants)
    which_intervals, y_names = _resolve_intervals_targets(
        list(variants.values()), which_intervals, y_predicted_names, x_modality, session_ids)

    # cell -> one array per variant, for each of the two quantities
    score_grid, width_grid, summary_rows = {}, {}, []
    for y_name in y_names:
        for which_interval in which_intervals:
            score_sets, width_sets = [], []
            for label, svm_res in variants.items():
                rows = _variant_rows(svm_res, which_interval, y_name, x_modality, session_ids,
                                     ccgp_w_train_stratum, analysis=analysis)
                scores = rows[spec['mean_col']].to_numpy(dtype=float) if not rows.empty else np.zeros(0)
                scores = scores[np.isfinite(scores)]
                has_ci = (not rows.empty and spec['ci_lower_col'] in rows.columns
                          and spec['ci_upper_col'] in rows.columns)
                widths = ((rows[spec['ci_upper_col']] - rows[spec['ci_lower_col']]).to_numpy(dtype=float)
                          if has_ci else np.zeros(0))
                widths = widths[np.isfinite(widths)]
                score_sets.append(scores)
                width_sets.append(widths)
                if scores.size:
                    summary_rows.append({
                        'predict_y_name': y_name, 'interval_name': which_interval, 'variant': label,
                        'n_fits': len(scores),
                        'n_sessions': rows.index.get_level_values('session_id').nunique(),
                        'mean': scores.mean(), 'sd': scores.std(), 'median': np.median(scores),
                        'q10': np.quantile(scores, 0.1), 'q90': np.quantile(scores, 0.9),
                        'ci_width_mean': widths.mean() if widths.size else np.nan,
                        'ci_width_median': np.median(widths) if widths.size else np.nan,
                    })
            score_grid[(y_name, which_interval)] = score_sets
            width_grid[(y_name, which_interval)] = width_sets

    if not summary_rows:
        print(f"nothing to plot for {x_modality}: no fits in any requested interval/target")
        return [], pd.DataFrame()
    summary = pd.DataFrame(summary_rows).set_index(['predict_y_name', 'interval_name', 'variant'])
    print(f"{spec['label']} ({spec['mean_col']}) - {x_modality}, "
          f"{summary['n_sessions'].max()} sessions, {len(labels)} variant(s)")
    print(summary.to_string(float_format=lambda v: f'{v:.3f}'))

    quantities = [(spec['label'], score_grid, chance_level, 'score')]
    if any(widths.size for sets in width_grid.values() for widths in sets):
        quantities.append(('CI width', width_grid, None, 'ciwidth'))

    colors = [plt.cm.tab10(i % 10) for i in range(len(labels))]
    figs = []
    for quantity_label, grid, reference, path_suffix in quantities:
        # sharey across the whole grid: comparing cells against each other is the point, and
        # per-cell autoscaling would silently rescale every panel to its own range
        fig, axes = plt.subplots(nrows=len(y_names), ncols=len(which_intervals), squeeze=False,
                                 sharex=True, sharey=True, dpi=150,
                                 figsize=(panel_width * len(which_intervals) + 1.8,
                                          panel_height * len(y_names) + 1.0))
        figs.append(fig)
        for row_i, y_name in enumerate(y_names):
            for col_i, which_interval in enumerate(which_intervals):
                ax = axes[row_i][col_i]
                drew = _draw_variant_distributions(ax, grid[(y_name, which_interval)], labels, colors,
                                                    plot_type, reference, show_n=show_n)
                if not drew:
                    # no fits for this (target, interval) - keep the cell, drop its frame, so
                    # the grid stays rectangular without implying an empty distribution
                    ax.set_xticks([])
                    [sp.set_visible(False) for sp in ax.spines.values()]
                if row_i == 0:
                    ax.set_title(_interval_short_label(which_interval), fontsize=8)
                if col_i == 0:
                    ax.set_ylabel(y_name, fontsize=8)
                ax.tick_params(labelsize=7)
                for spine in ('top', 'right'):
                    ax.spines[spine].set_visible(False)
        if plot_type == 'hist':
            axes[-1][0].set_xlabel(quantity_label, fontsize=8)
            if len(labels) > 1:
                axes[0][-1].legend(fontsize=6, frameon=False)
        # session count in the title, fits-per-cell in each cell's own corner (see
        # _draw_variant_distributions) - the two n's a reader needs are on different levels
        fig.suptitle(f'{title} - {quantity_label}' if title is not None
                     else f'{quantity_label} - {x_modality} - {summary["n_sessions"].max()} sessions',
                     fontsize=12)
        fig.tight_layout()
        # per-quantity filenames, same idea as the heatmap's per-target ones
        for path, dpi in ((save_path, 150), (save_path_svg, None)):
            if path is None:
                continue
            stem, ext = os.path.splitext(path)
            fig.savefig(f'{stem}_{path_suffix}{ext}', **({'dpi': dpi} if dpi else {}))
        plt.show()
    return figs, summary

