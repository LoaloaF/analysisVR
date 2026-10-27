"""What the neurons themselves do, per condition, on the same t0-aligned x-axis as everything else.

The rest of this tree plots *derived* quantities over the concatenated t0 intervals - a
decoder's balanced accuracy (`svm_plotting_lib`), the share of variance a fitted direction
explains (`var_plotting_lib`), the dimensionality of the population subspace
(`dim_plotting_lib`). None of them shows the firing rates those quantities were computed
from. This module is the missing plain view: one neuron (or the across-neuron mean), its
z-scored firing rate over time, one curve per condition.

x is laid out by `plotlib.trial_traces.layout_intervals` - the same offsets/gaps
`dim_plotting_lib` resolves, so a figure from here lines up bin-for-bin with one from there
(t0 at bin 0, `gap_bins` blank columns between intervals). Those gap columns are NaN, so
every curve simply *breaks* there: two timebins that are seconds apart are never joined by a
line segment.

One curve per cell of a condition grid: `conditions=('cue', 'choice_R1')` gives the 2x2
(C1&skip, C1&stop, C2&skip, C2&stop). A joint cell is drawn as one **bicolour dashed** line -
the same curve twice, once in each label's own colour, with complementary dash phases - so
cue1&skip reads as orange interleaved with light blue. That encoding, the colours and the
level names all come from `plotlib.trial_labels`, which is exactly what
`var_plotting_lib`'s variance/activation rows draw their 2x2 with: a cell is the same line
in all three figures. Opacity scales with the cell's trial count (`ALPHA_FLOOR`..1), because
a 5-trial mean should not read as confidently as a 40-trial one.

Two entry points, the same curves at two levels of detail:

  * `plot_condition_lines` - the working figure. Every interval the notebook is looking at,
    the full 2x2 condition grid, a legend carrying each cell's n.
  * `plot_condition_panel` - the compressed one, for the poster/a talk (`poster_26.py`).
    Two stacked rows on one x-axis: the trials' own position traces above (with the stretch
    of track the cue is rendered over marked on them), the condition curves below, two
    intervals, no per-bin scaffolding. It is the reader's grounding for every later
    t0-aligned figure, so it shows what the animal was doing in each aligned window
    alongside what the population did.

Layering: this is a `*_plotting_lib` (figure composer) and imports only from `plotlib`. It
does no NAS I/O - `fr_z` is the already-loaded, trial_id-stamped, modality-sliced frame
dim_analysis.py builds, exactly as `dim_plotting_lib`'s entry points take it.
"""

import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory

from plotlib import export
from plotlib.colors import (CUE_SPAN_HATCH, ZONE_SPAN_ALPHA, ZONE_SPAN_COLORS,
                            ZONE_SPAN_FALLBACK_COLOR, modality_colors)
from plotlib.trial_labels import (COMBO_DASH, COMBO_LINEWIDTH, covariate_color, level_label,
                                  trial_covariate_levels)
from plotlib.trial_traces import (SEC_BINS, T0_LINE_KW, interval_short_label,
                                  layout_intervals, smooth as smooth_trace)


GAP_BINS = 2          # blank columns between intervals - dim_plotting_lib's _GAP_BINS value,
                      #   so the two families of figures share an x-axis
ALPHA_FLOOR = 0.4     # opacity of the smallest condition cell, relative to the largest one in
                      #   the session (var_plotting_lib._split_alpha uses 0.3 for the same
                      #   encoding; 0.4 here is Simon's call - with four curves crossing each
                      #   other a fainter one gets lost rather than merely reading as small)
LINEWIDTH = 1.4       # a single-colour curve; a bicolour one uses trial_labels.COMBO_LINEWIDTH
SEM_ALPHA = 0.12      # the +-1 SEM band, further scaled by the curve's own alpha
MIN_TRIALS = 1        # a condition x interval cell below this many trials is left NaN (a gap
                      #   in that curve) rather than drawn as a mean of one or two trials


def resolve_units(fr_z, unit=None, unit_cols=None):
    """The unit column(s) one curve averages over, plus the label naming them.

    `unit=None` is the across-neuron mean - every column of `unit_cols` (or every 'Unit*'
    column of `fr_z` if that is None too). Otherwise `unit` is one unit or a list of them,
    each given either as the full column name ('Unit0031'), as its bare id ('0031' or 31),
    or as anything else that names a real column - so a caller can write `unit=31` without
    knowing the zero-padding width of this table's column names.

    Returns `(columns, label)`.
    """
    available = list(unit_cols) if unit_cols is not None else [c for c in fr_z.columns
                                                               if c.startswith('Unit')]
    if unit is None:
        return available, f'mean of {len(available)} units'

    wanted = [unit] if isinstance(unit, (str, int, np.integer)) else list(unit)
    columns = []
    for u in wanted:
        if isinstance(u, str) and u in available:
            columns.append(u)
            continue
        # a bare id: match on the numeric part, so 31 / '31' / '0031' all find 'Unit0031'
        try:
            target = int(str(u).replace('Unit', ''))
        except ValueError:
            raise ValueError(f"unit {u!r} is not a column of this frame and not a unit id; "
                             f"have {len(available)} units, e.g. {available[:3]}")
        matches = [c for c in available if int(c.replace('Unit', '')) == target]
        if not matches:
            raise ValueError(f"no unit {u!r} among this frame's {len(available)} units "
                             f"(e.g. {available[:3]}) - note the frame is already sliced to "
                             f"one modality, so an id from the unsliced table may be absent")
        columns.append(matches[0])

    label = columns[0] if len(columns) == 1 else f'mean of {len(columns)} units'
    return columns, label


def _value_fn(columns):
    """One scalar per 40ms row: that row's mean z across `columns`. `nanmean` rather than
    `mean` because a unit with no spikes at all has an all-NaN z column (its std is 0 - see
    get_FiringRate40msZ), and one such unit would otherwise NaN out every bin of the
    across-neuron mean."""
    if len(columns) == 1:
        col = columns[0]
        return lambda dat: dat[col].to_numpy(dtype=float)
    return lambda dat: np.nanmean(dat[columns].to_numpy(dtype=float), axis=1)


def _condition_names(conditions):
    return (conditions,) if isinstance(conditions, str) else tuple(conditions)


def _row_keys(t0_events, s_id, all_trial_ids, names):
    """The condition grid's cells, in row order: the cartesian product of each condition's
    own levels, ascending (so for ('cue', 'choice_R1'): C1&skip, C1&stop, C2&skip, C2&stop -
    level 0 before level 1, the same 0/1 encoding the fits use).

    Levels are the ones actually *present* in this session rather than a hardcoded (0, 1),
    so a session in which a condition never took one of its values contributes two rows
    instead of four empty-halved ones."""
    per_name = []
    for name in names:
        levels = trial_covariate_levels(t0_events, s_id, all_trial_ids, name)
        per_name.append(sorted(int(v) for v in np.unique(levels[np.isfinite(levels.astype(float))])))
    return list(itertools.product(*per_name)), per_name


def _levels_by_trial(t0_events, s_id, trial_ids, names):
    """(n_conditions, n_trials) level array for one interval's kept trials, in trace order."""
    return np.stack([trial_covariate_levels(t0_events, s_id, trial_ids, name) for name in names])


def condition_block(t0_events, fr_z, s_id, which_intervals, unit=None, unit_cols=None,
                    conditions=('cue', 'choice_R1'), smooth=3, gap_bins=GAP_BINS,
                    min_trials=MIN_TRIALS):
    """The curves `draw_condition_lines` draws, and everything needed to label them.

    One row per cell of the `conditions` grid (see `_row_keys`), one column per integer
    x-position across the concatenated `which_intervals` (see `layout_intervals`); the value
    is the mean, over that cell's trials, of the z-scored firing rate of the unit(s)
    `resolve_units` picks. Gap columns, and cells with fewer than `min_trials` trials in
    that interval, are NaN - which is what makes each curve break at an interval boundary
    instead of joining two timebins that are seconds apart.

    `smooth` (default 3, i.e. 120ms) is a centered rolling mean applied to *each trial's own*
    trace before the average across trials, not to the average afterwards - the two are only
    equivalent while every trial is present at every bin, which is exactly what the
    modal-shape filter does not guarantee. A single neuron's z-scored 40ms rate is mostly
    spike-count noise at the native resolution, so some smoothing is the useful default here,
    the same reasoning as `_subspace_block_with_separators`' smooth=5.

    Returned dict:
      'rows'        (n_rows, n_x) float, NaN where there is nothing to show
      'rows_sem'    (n_rows, n_x) float, the same cells' across-trial SEM
      'x_coords'    (n_x,) int - the shared x-axis positions
      'row_keys'    list of level tuples, in row order
      'row_labels'  list of str, e.g. 'C1 & stop'
      'row_n'       list of {interval: n_trials} - per interval, because a trial only has a
                    row in t0_events for the zones it actually visited, and the modal-shape
                    filter drops different trials in different intervals
      'layout'      layout_intervals' own return (4-tuples), for the x-axis scaffolding
      'names'       the condition names, in row-key order
      'unit_label'  what one cell averages over
    """
    names = _condition_names(conditions)
    columns, unit_label = resolve_units(fr_z, unit, unit_cols)
    layout = layout_intervals(t0_events, fr_z, s_id, which_intervals, _value_fn(columns),
                              gap_bins=gap_bins, return_trial_ids=True)[0]
    if not layout:
        return {'rows': np.zeros((0, 0)), 'rows_sem': np.zeros((0, 0)),
                'x_coords': np.array([]), 'row_keys': [],
                'row_labels': [], 'row_n': [], 'layout': [], 'names': names,
                'unit_label': unit_label}

    all_trial_ids = sorted({tid for *_, trial_ids in layout for tid in trial_ids})
    row_keys, _ = _row_keys(t0_events, s_id, all_trial_ids, names)

    left_edge = layout[0][3] + layout[0][2].min()
    right_edge = layout[-1][3] + layout[-1][2].max()
    x_coords = np.arange(left_edge, right_edge + 1)
    rows = np.full((len(row_keys), len(x_coords)), np.nan)
    rows_sem = np.full((len(row_keys), len(x_coords)), np.nan)
    row_n = [{} for _ in row_keys]

    for which_interval, traces, bins, offset, trial_ids in layout:
        stacked = np.stack([smooth_trace(np.asarray(t, dtype=float), smooth) for t in traces])
        levels = _levels_by_trial(t0_events, s_id, trial_ids, names)
        idx = bins + offset - left_edge
        for r, key in enumerate(row_keys):
            mask = np.all(levels == np.asarray(key)[:, None], axis=0)
            row_n[r][which_interval] = int(mask.sum())
            if mask.sum() < min_trials:
                continue
            cell = stacked[mask]
            rows[r, idx] = np.nanmean(cell, axis=0)
            # ddof=1 and the per-bin finite count, not mask.sum(): a unit can be NaN at a
            # bin (see _value_fn), and an SEM divided by the nominal n would understate it
            n_finite = np.isfinite(cell).sum(axis=0)
            with np.errstate(invalid='ignore'):
                rows_sem[r, idx] = np.where(
                    n_finite > 1, np.nanstd(cell, axis=0, ddof=1) / np.sqrt(n_finite), np.nan)

    return {'rows': rows, 'rows_sem': rows_sem, 'x_coords': x_coords, 'row_keys': row_keys,
            'row_labels': [' & '.join(level_label(n, lv) for n, lv in zip(names, key))
                           for key in row_keys],
            'row_n': row_n, 'layout': [entry[:4] for entry in layout], 'names': names,
            'unit_label': unit_label}


def _n_text(counts):
    """'n=31' when every interval kept the same trials for this condition, 'n=28-31' when
    they did not - which is the normal case rather than an anomaly (a trial has a row in
    t0_events only for the zones it visited, and the modal-shape filter drops a different
    handful per interval), so it is shown rather than silently reduced to one number."""
    values = sorted(set(counts.values()))
    if not values:
        return 'n=0'
    return f'n={values[0]}' if len(values) == 1 else f'n={values[0]}-{values[-1]}'


def _time_tick_bins(layout, time_ticks):
    """(seconds, bins, interval) for the longest of `time_ticks` that fits inside one of the
    layout's intervals, or None - see `_draw_x_axis`. Longest-first, and the first interval
    that can hold it."""
    for seconds in (time_ticks or ()):
        n_bins = int(round(seconds * SEC_BINS))
        for which_interval, _, bins, _ in layout:
            if bins.max() >= n_bins:
                return seconds, n_bins, which_interval
    return None


def _draw_x_axis(ax, layout, fontsize=8, labels=None, minor=True, time_ticks=None):
    """Labelled t0 ticks plus unlabelled 1s minor ticks inside each interval - the same
    scaffolding dim_plotting_lib's `_apply_xticks` puts on its own shared axis, so the two
    read identically.

    `labels` (default None) overrides the tick text per interval, e.g.
    {'R1_entry_interval': 'R1 entry'}: the default `$t_{R1Entry}$` names the alignment event
    the way the analysis does, which is right for a working figure and wrong for a talk.

    `minor=False` drops the unlabelled 1s ticks - a per-interval ruler a talk audience is not
    going to read. `time_ticks` (e.g. (1.0, 0.5, 0.25)) instead puts ONE labelled major tick
    at t0 + that many seconds: the whole time scale in a single tick. The candidates are tried
    longest-first and the first one that FITS inside some interval wins, because the intervals
    are not all the same length - cue entry's own window ends at +0.84s on real data, so a
    panel showing only that interval gets a '0.5s' tick where a two-interval panel gets '1s'
    under R1 entry. Nothing is drawn if none of them fits."""
    xticks, xtick_labels, minor_ticks = [], [], []
    labels = labels or {}
    time_tick_bins = _time_tick_bins(layout, time_ticks)
    for which_interval, _, bins, offset in layout:
        xticks.append(offset)
        xtick_labels.append(labels.get(which_interval,
                                       f'$t_{{{interval_short_label(which_interval)}}}$'))
        if time_tick_bins is not None:
            seconds, n_bins, tick_interval = time_tick_bins
            if which_interval == tick_interval:
                # The tick MARK goes on the axis like any other, but its label goes a line
                # BELOW the event names rather than beside them: on a half-width panel
                # 'cue entry' and '0.5s' are together wider than the axes and printed straight
                # through each other. Two lines cost a few points of height, which this panel
                # has, where it has no width to spare.
                xticks.append(offset + n_bins)
                xtick_labels.append('')
                ax.annotate(f'{seconds:g}s', xy=(offset + n_bins, 0),
                            xycoords=('data', 'axes fraction'),
                            xytext=(0, -fontsize * 1.9), textcoords='offset points',
                            ha='center', va='top', fontsize=fontsize)
        k_lo, k_hi = int(np.ceil(bins.min() / SEC_BINS)), int(np.floor(bins.max() / SEC_BINS))
        minor_ticks.extend(offset + k * SEC_BINS for k in range(k_lo, k_hi + 1) if k != 0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=fontsize)
    ax.set_xticks(minor_ticks if minor else [], minor=True)
    ax.tick_params(axis='x', which='minor', length=4)


def _row_colors(names, key):
    """One colour per condition in the cell's key - a 1-tuple for a single condition, a
    2-tuple for a 2x2 cell, which is what makes that curve bicolour (see COMBO_DASH)."""
    return tuple(covariate_color(name, level) for name, level in zip(names, key))


def _row_alpha(counts, n_max, alpha_floor=ALPHA_FLOOR):
    """Curve opacity from the cell's trial count, relative to the biggest cell in this
    session - a 5-trial curve should not read as confidently as a 40-trial one. Floored so a
    small cell stays visible rather than fading out entirely.

    Unlike `var_plotting_lib._split_alpha`, which is evaluated per interval, this takes one
    alpha for the WHOLE curve (from the cell's largest per-interval count). A curve that
    changed opacity from block to block would read as the *neuron* changing rather than the
    trial count, and the counts barely move between intervals anyway - measured 15 vs 16 on
    2024-12-10_17-20, where the difference is one trial that never reached the ITI zone."""
    n = max(counts.values()) if counts else 0
    if not n_max:
        return 1.0
    return float(np.clip(n / n_max, alpha_floor, 1.0))


def draw_condition_lines(ax, block, ylim=None, show_sem=True, show_legend=True,
                         alpha_floor=ALPHA_FLOOR, show_xticks=True, show_ylabel=True,
                         fontsize=8, legend_ncol=2, sem_z=1.0, sem_alpha=SEM_ALPHA):
    """Draw a `condition_block` result into `ax` as one curve per condition cell.

    A cell of a two-condition grid is drawn as ONE bicolour dashed line - the same curve
    plotted twice, once in each label's own colour, with the complementary dash phases of
    `trial_labels.COMBO_DASH`, so the dashes interleave into a single alternating line
    (cue1 & skip = orange interleaved with light blue). A single-condition split is a plain
    solid line in that level's colour. Identical encoding to `var_plotting_lib`'s variance
    and activation rows, deliberately: a cell is the same line wherever it is drawn.

    `ylim` (default None = autoscale to this session) **clips**. Pass a fixed pair - or a
    scalar, read as +-it - to make the figure mean the same thing in every panel of a
    canvas, which is the point of having it: a per-session autoscale silently rescales the
    y-axis under the reader between one session and the next. Note the across-neuron mean
    (`unit=None`) lives on a ~5x smaller scale than a single unit, so one fixed `ylim`
    cannot serve both.

    Opacity per curve comes from its trial count (see `_row_alpha`), and the count is also
    spelled out in the legend entry, since opacity alone is not readable as a number.

    The shaded band is `sem_z` x the across-trial SEM of that cell's mean: 1.0 (the default,
    what this figure has always drawn) is +-1 SEM, 1.96 is the 95% CI of the mean. It is a
    statement about how precisely the MEAN is known, not about how much trials vary - the
    across-trial spread is sqrt(n) wider. `sem_alpha` is how visible it is; the default is
    faint on purpose behind four crossing curves, and the compressed panel (two curves) turns
    it up. Returns the (handles, labels) it built, for a caller drawing the key elsewhere.
    """
    rows, sems, x = block['rows'], block['rows_sem'], block['x_coords']
    names = block['names']
    if show_ylabel:
        ax.set_ylabel('z-scored firing rate', fontsize=fontsize)
    handles, labels = [], []
    if rows.size == 0:
        return handles, labels

    n_max = max((max(counts.values()) for counts in block['row_n'] if counts), default=0)
    for r, key in enumerate(block['row_keys']):
        colors = _row_colors(names, key)
        alpha = _row_alpha(block['row_n'][r], n_max, alpha_floor)
        y = rows[r]
        if not np.isfinite(y).any():
            continue
        if show_sem:
            e = sems[r] * sem_z
            ax.fill_between(x, y - e, y + e, color=colors[0], alpha=sem_alpha * alpha,
                            linewidth=0, zorder=2)
        if len(colors) == 1:
            ax.plot(x, y, color=colors[0], linewidth=LINEWIDTH, alpha=alpha, zorder=3)
            handles.append(Line2D([], [], color=colors[0], linewidth=LINEWIDTH, alpha=alpha))
        else:
            for color, dashes in zip(colors, COMBO_DASH):
                ax.plot(x, y, color=color, linewidth=COMBO_LINEWIDTH, alpha=alpha,
                        zorder=3, linestyle=dashes)
            handles.append(tuple(Line2D([], [], color=c, linewidth=COMBO_LINEWIDTH,
                                        alpha=alpha, linestyle=dashes)
                                 for c, dashes in zip(colors, COMBO_DASH)))
        labels.append(f'{block["row_labels"][r]} ({_n_text(block["row_n"][r])})')

    for _, _, _, offset in block['layout']:      # t0 sits at bin 0 of its interval
        ax.axvline(offset, zorder=0, **T0_LINE_KW)
    if ylim is not None:
        lo, hi = (-ylim, ylim) if np.isscalar(ylim) else ylim
        ax.set_ylim(lo, hi)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=fontsize - 1)

    if show_xticks:
        _draw_x_axis(ax, block['layout'], fontsize=fontsize)
    else:
        ax.set_xticks([])
    if show_legend and handles:
        # ABOVE the axes, not inside it. Four curves plus their SEM bands leave no empty
        # corner to put a four-entry key in - in-axes it printed straight over the cue-entry
        # block. Anchored bottom-right of the band above the axes so a left-aligned title
        # shares that band; `legend_title_pad` is what a caller reserves for it.
        # HandlerTuple(ndivide=1), not the default ndivide=None: the default halves the
        # handle and gives two short segments instead of one alternating bicolour line.
        ax.legend(handles, labels, fontsize=fontsize - 1, frameon=False, ncol=legend_ncol,
                  loc='lower right', bbox_to_anchor=(1.0, 1.0), handlelength=2.6,
                  labelspacing=0.3, columnspacing=1.2, borderpad=0,
                  handler_map={tuple: HandlerTuple(ndivide=1)})
    return handles, labels


def legend_title_pad(n_entries, legend_ncol=2, fontsize=8):
    """Points of title padding that clear a `draw_condition_lines` legend of `n_entries`.

    The legend sits in the band above the axes (see there), so an axes title drawn at the
    default pad lands on top of it. Both are placed in that band instead - title left,
    legend right - which needs the title lifted by the legend's own height."""
    n_legend_rows = int(np.ceil(n_entries / max(legend_ncol, 1)))
    return 4 + n_legend_rows * (fontsize - 1) * 1.45


def plot_condition_lines(t0_events, fr_z, s_id, which_intervals, unit=None, unit_cols=None,
                         conditions=('cue', 'choice_R1'), smooth=3, gap_bins=GAP_BINS,
                         min_trials=MIN_TRIALS, ylim=None, show_sem=True, show_legend=True,
                         alpha_floor=ALPHA_FLOOR, legend_ncol=2, title=None,
                         width_per_bin=0.055, min_width=4.0, height=2.4, fontsize=8,
                         save_dir=None, stem=None, dpi=export.DPI):
    """One figure: `unit`'s z-scored firing rate over time, one curve per condition cell.

    The figure-level entry point - fetches (via `condition_block`), sizes a figure to the
    concatenated intervals' own bin span, draws, and optionally saves the PNG/SVG pair.
    Returns a dict `{'fig', 'ax', 'block', 'paths'}` ('paths' is `save_figure`'s
    {format: path} map, empty when `save_dir` is None).

    `unit=None` averages every unit of `unit_cols` into each curve instead - the population
    mean z, which is the row-1 dotted line of `plot_session_kinematics_neural_and_pca` split
    by condition. `ylim` is the knob to set when rendering a whole session sequence; see
    `draw_condition_lines`.
    """
    block = condition_block(t0_events, fr_z, s_id, which_intervals, unit=unit,
                            unit_cols=unit_cols, conditions=conditions, smooth=smooth,
                            gap_bins=gap_bins, min_trials=min_trials)
    width = max(min_width, 0.8 + width_per_bin * max(len(block['x_coords']), 1))
    fig, ax = plt.subplots(figsize=(width, height))
    handles, _ = draw_condition_lines(ax, block, ylim=ylim, show_sem=show_sem,
                                     show_legend=show_legend, alpha_floor=alpha_floor,
                                     fontsize=fontsize, legend_ncol=legend_ncol)
    ax.set_title(title if title is not None else f'{s_id} - {block["unit_label"]}',
                 fontsize=fontsize + 1, loc='left',
                 pad=legend_title_pad(len(handles), legend_ncol, fontsize) if show_legend else 4)
    fig.tight_layout()

    stem = stem or f'{s_id}_{block["unit_label"].replace(" ", "")}_condition_lines'
    return {'fig': fig, 'ax': ax, 'block': block,
            'paths': export.save_figure(fig, save_dir, stem, dpi=dpi)}


# ---------------------------------------------------------------------------------------
# The compressed panel: position traces above, condition curves below
# ---------------------------------------------------------------------------------------
# Everything below composes the two-row figure `plot_condition_panel` returns. It draws the
# same curves as `draw_condition_lines` above - not a second implementation of them - and
# differs only in what it leaves OUT: two intervals instead of four, no per-bin gridlines, a
# scale bar instead of a ruler under every block, and event names a reader who has never seen
# `R1_entry_interval` written down can still read.
PANEL_INTERVALS = ('cue_entry_interval', 'R1_entry_interval')
# Plain names for the two t0 markers - see _draw_x_axis's `labels`: what the alignment event
# IS, rather than the analysis' own `$t_{R1Entry}$`. 'R1 entry', not 'R1 choice' (which this
# briefly said): t0 is the moment the animal enters the zone, and the choice follows it.
PANEL_T0_LABELS = {'cue_entry_interval': 'cue entry', 'R1_entry_interval': 'R1 entry'}
PANEL_FIGSIZE = (2.4, 2.8)          # inches, per session - a canvas of N sessions is N wide.
                                    #   Sized around the TYPE, not the other way round: a
                                    #   matplotlib fontsize is a physical point size, so a
                                    #   panel that has to stay readable printed needs the room
                                    #   its labels take at that size (see behavior_plotting_
                                    #   lib's FONTSIZE=20 panels)
PANEL_HEIGHT_RATIOS = (1.0, 1.7)    # the neural row is the point; position is the grounding
PANEL_FONTSIZE = 16                 # everything but the header. A poster/talk minimum, not a
                                    #   default - at the 9pt this started on, the axis labels
                                    #   were unreadable at print size
PANEL_LABEL_FONTSIZE = 22           # the session header (S09), bold: on a canvas of panels it
                                    #   is the only thing telling the reader which column is
                                    #   which, and it is read from further away than the axes
PANEL_SMOOTH = 5                    # bins, per trial before averaging - see condition_block
PANEL_TRACE_KW = dict(color='k', alpha=0.10, linewidth=0.5)
PANEL_GAP_BINS = 3                  # blank bins between the two intervals - barely over
                                    #   GAP_BINS, and
                                    #   no more than that: the panel is space-limited and the
                                    #   gap is width spent on nothing. At the tree's usual 2
                                    #   the two intervals read as one continuous stretch of
                                    #   time, which is the one thing this panel exists to say
                                    #   they are not (they are seconds apart, and in either
                                    #   order) - but `draw_gap_mask` and `draw_split_x_spine`
                                    #   are what actually carry that, so the gap itself only
                                    #   has to be wide enough to see, not wide enough to argue.
                                    #   16 and 6 were both tried and both read as too far apart
PANEL_SEM_Z = 1.96                  # the band is the 95% CI of each curve's mean, not +-1 SEM
PANEL_SEM_ALPHA = 0.3               # and visible - two curves leave room for it
# Short, and wrapped, because a rotated y label is as tall as the string is long: 'Track
# position [cm]' set on two lines at PANEL_FONTSIZE is taller than the position row itself
# (the short row of the pair) and printed straight through the neural row's own label below.
PANEL_YLABELS = ('Position\n[cm]', 'z-scored\nfiring rate')
PANEL_YTICKS = 3                    # at most, on both rows - see MaxNLocator
# Which track_zone levels the cue is rendered across, for the `track_zone` fallback of
# cue_zone_extent. The same two behavior_plotting_lib shades on the position axis of the
# poster's kinematics figure, so the two panels mark the same stretch of track.
CUE_ZONE_TRACK_ZONES = ('visibleCue', 'nextToCue')
# The reward zones to mark on the position row, in draw order. Only R1 by default: the panel's
# intervals end at the R1 choice, so an R2 band would be a stretch of track no trace on this
# figure ever reaches.
PANEL_REWARD_ZONES = ('reward1Zone',)
# From S8 on either cue can appear on a given trial, which a background illustration cannot
# know per trial - so the cue zone is striped in BOTH cue colours: alternating horizontal
# sub-bands across the zone, all hatched in the same direction. Ported from
# svm_plotting_lib.plot_session_analysis, whose comment records what was tried first -
# oppositely-angled overlapping hatches (crossing lines, and alpha-blending into a muddy
# brown), and a left/right split (which, varying along x, read as if the colour said
# something about time). Alternating along y has neither problem: y here is position, and no
# data varies systematically across the sub-bands.
CUE_ZONE_STRIPES = 6
CUE_ZONE_STRIPE_ALPHA = 0.5
# The time scale is one labelled tick at t0 + this many seconds (see _draw_x_axis's
# `time_ticks`). It was a 1s scale bar at the bottom-left first; Simon asked for the tick
# instead - it needs no room of its own, and it says where the second is measured from, which
# a floating bar did not. Several candidates because the intervals are not all the same
# length: cue entry's own window is only 0.96s end to end, so on a cue-entry-only panel a 1s
# tick has nowhere to go and the figure would silently lose its time scale entirely.
PANEL_TIME_TICKS = (1.0, 0.5, 0.25)
# Names for the bands, OFF by default (`plot_condition_panel(zone_labels=ZONE_LABELS, ...)`
# turns them on). They were drawn on the bands themselves for a while - the alternative, a
# legend entry, would list a place on the track alongside the conditions - but Simon adds
# them by hand in Inkscape now, where he can put each one where that particular panel has
# room. Kept here so a panel read on its own can still label itself.
CUE_ZONE_LABEL = 'cue zone'
ZONE_LABELS = {'reward1Zone': 'R1 zone', 'reward2Zone': 'R2 zone'}


def track_zone_extents(behavior, s_id, zones=PANEL_REWARD_ZONES):
    """{zone: (lo, hi)} in cm for each of `zones` present in this session's own `track_zone`
    column - the same way behavior_plotting_lib's `zone_spans` reads the reward zones off the
    position bins rather than parsing SessionMetadata's track_details: every row already
    carries the zone it falls in, so this needs no second table. Empty if `behavior` was
    loaded without `track_zone` (the poster's driver asks for it explicitly)."""
    beh = behavior.xs(s_id, level='session_id')
    if 'track_zone' not in beh.columns:
        # Loudly, not silently: without this column the reward bands simply do not appear,
        # and a `behavior` frame loaded before 'track_zone' was added to the driver's column
        # list looks exactly like one where the zones legitimately have no rows.
        if zones:
            print(f"  no 'track_zone' column in behavior - no reward-zone band drawn "
                  f"({', '.join(zones)}); re-run the cell that loads Behavior40msAligned "
                  f"with 'track_zone' in its columns")
        return {}
    extents = {}
    for zone in zones:
        positions = beh.loc[beh['track_zone'] == zone, 'frame_position'].dropna()
        if not positions.empty:
            extents[zone] = (float(positions.min()), float(positions.max()))
    return extents


def cue_zone_extent(behavior, s_id, track_zones=CUE_ZONE_TRACK_ZONES):
    """(lo, hi) in cm of the stretch of track this session rendered the cue over, or None
    if it rendered none - which is the correct answer for a pre-cue session, not a missing
    feature.

    Read from `cue_visible` when `behavior` carries it, because that column is already
    gated on the session's own phase: integr_analytics builds it as the cue's value inside
    the cue zones AND `~both_R1_R2_rewarded`, so it is 0 everywhere in sessions 0-7 and this
    returns None for them. `track_zone` is the fallback and gives the same numbers on real
    data (-119.4 to 11.4cm on 2024-11-28_17-41, identical to the cue_visible span), but it
    is NOT phase-gated - the zone exists on the track whether or not anything was drawn on
    the wall - so a pre-cue session would get a band for a cue that was never shown.
    """
    beh = behavior.xs(s_id, level='session_id')
    if 'cue_visible' in beh.columns:
        mask = beh['cue_visible'].fillna(0).astype(float) > 0
    elif 'track_zone' in beh.columns:
        mask = beh['track_zone'].isin(track_zones)
    else:
        return None
    positions = beh.loc[mask.to_numpy(), 'frame_position'].dropna()
    if positions.empty:
        return None
    return float(positions.min()), float(positions.max())


def draw_position_traces(ax, layout, cue_zone=None, reward_zones=None, trace_kw=None,
                         smooth=1, cue_zone_label=None, zone_labels=None,
                         cue_zone_stripes=CUE_ZONE_STRIPES,
                         cue_zone_alpha=CUE_ZONE_STRIPE_ALPHA,
                         cue_zone_fontsize=PANEL_FONTSIZE - 4):
    """Row 0 of the panel: every trial's own position trace over the concatenated intervals,
    plus the cue-zone band.

    One line per trial rather than a mean with a band - unlike a firing rate, a single
    trial's position trace is individually meaningful (it is that trial's run down the
    track), and the spread between them is the point: it says how variable the animal's
    approach was in the window everything else is aligned to.

    `cue_zone` is an (lo, hi) pair in cm (see `cue_zone_extent`) and `reward_zones` a
    {zone: (lo, hi)} map (see `track_zone_extents`), both drawn as HORIZONTAL bands: y here
    is track position, so a stretch of track is a band across the whole row, and a trial's
    line crossing it is that trial being there. The reward zones take the poster's own greys
    (R1 dark - plotlib.colors' ZONE_SPAN_*) but not its black outline; the cue zone is
    striped in both cue colours; a band is named on itself at its left edge only if asked
    (`zone_labels`, `cue_zone_label` - both None by default, see ZONE_LABELS), and
    (see CUE_ZONE_STRIPES), because which cue a given trial showed is exactly what these
    background bands cannot know.
    """
    trace_kw = {**PANEL_TRACE_KW, **(trace_kw or {})}
    label_trans = blended_transform_factory(ax.transAxes, ax.transData)
    # Both labels hang off the LEFT edge of their band, unboxed (Simon's call, first for the
    # reward zones and then for the cue): the panel is read left to right, so that is where a
    # band is named before anything is drawn over it, and a box around grey text on a pale
    # band is more ink than the text itself.
    zone_label_kw = dict(ha='left', va='center', fontsize=cue_zone_fontsize, color='dimgrey',
                         zorder=4, transform=label_trans)
    cue_label_kw = dict(zone_label_kw)
    for zone, (lo, hi) in (reward_zones or {}).items():
        # No outline, unlike the behaviour column's kinematics figure (ZONE_SPAN_EDGE_COLOR):
        # there the zone is a narrow vertical span and the black edge is what makes it read as
        # the poster's stage diagram, but here it runs the full width of the row, where two
        # horizontal black rules read as axis furniture - Simon's call.
        ax.axhspan(lo, hi, zorder=0, linewidth=0, edgecolor='none',
                   facecolor=to_rgba(ZONE_SPAN_COLORS.get(zone, ZONE_SPAN_FALLBACK_COLOR),
                                     ZONE_SPAN_ALPHA))
        if zone_labels and zone in zone_labels:
            ax.text(0.01, (lo + hi) / 2, zone_labels[zone], **zone_label_kw)
    if cue_zone is not None:
        # facecolor='none' + linewidth=0: only the hatch lines themselves carry the colour,
        # so the stripes never fill and never alpha-blend with each other where they meet.
        cue_colors = modality_colors['cue']
        stripe_height = (cue_zone[1] - cue_zone[0]) / cue_zone_stripes
        for i in range(cue_zone_stripes):
            y0 = cue_zone[0] + i * stripe_height
            ax.axhspan(y0, y0 + stripe_height, facecolor='none', zorder=0, linewidth=0,
                       edgecolor=cue_colors[i % 2], hatch=CUE_SPAN_HATCH,
                       alpha=cue_zone_alpha)
        if cue_zone_label:
            ax.text(0.01, np.mean(cue_zone), cue_zone_label, **cue_label_kw)
    for _, traces, bins, offset in layout:
        x = bins + offset
        for vals in traces:
            ax.plot(x, smooth_trace(np.asarray(vals, dtype=float), smooth), zorder=2, **trace_kw)
    for _, _, _, offset in layout:
        ax.axvline(offset, zorder=3, **T0_LINE_KW)


def draw_gap_mask(ax, layout, zorder=3.6, color='white'):
    """Paint the blank columns between intervals out, in every row that has background bands.

    The curves already break at a gap (those columns are NaN), but the cue/R1 zone bands and
    any other axhspan run the full width of the axes and so bridge it - which says the two
    intervals are one continuous stretch of track-time, the one thing this panel exists to
    deny. A white bar over each gap breaks them at exactly the same place the data breaks,
    and reads as the discontinuity it is for the position traces above as well.

    Drawn above the bands and the traces (nothing lives in a gap column anyway) but below the
    zone labels and the scale bar."""
    for (_, _, bins_a, off_a), (_, _, bins_b, off_b) in zip(layout, layout[1:]):
        ax.axvspan(bins_a.max() + off_a, bins_b.min() + off_b, facecolor=color,
                   edgecolor='none', linewidth=0, zorder=zorder)


def draw_split_x_spine(ax, layout, linewidth=None, y=0.0):
    """The x-axis line drawn as one segment per interval instead of one continuous spine.

    The gap between two intervals is not a stretch of time that was left out - the intervals
    are seconds apart and in no fixed order - and a spine running straight through it says the
    opposite, which no amount of widening the gap fixes. So the spine is hidden and its
    segments drawn by hand, each spanning only its own interval's bins. Ticks are untouched:
    they belong to the axis, and the t0 markers still land on their own segment.

    Drawn in a blended transform (data x, axes-fraction y) at the axes' own linewidth, so it
    sits exactly where the spine did and matches the surviving left spine."""
    linewidth = plt.rcParams['axes.linewidth'] if linewidth is None else linewidth
    ax.spines['bottom'].set_visible(False)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for _, _, bins, offset in layout:
        ax.plot([bins.min() + offset, bins.max() + offset], [y, y], transform=trans,
                color=ax.spines['left'].get_edgecolor(), linewidth=linewidth,
                solid_capstyle='butt', clip_on=False, zorder=4)


def plot_condition_panel(t0_events, behavior, fr_z, s_id, which_intervals=PANEL_INTERVALS,
                         unit=None, unit_cols=None, conditions='cue', smooth=PANEL_SMOOTH,
                         gap_bins=PANEL_GAP_BINS, min_trials=MIN_TRIALS, ylim=None,
                         position_ylim=None, show_sem=True, sem_z=PANEL_SEM_Z,
                         sem_alpha=PANEL_SEM_ALPHA, show_legend=False, show_n=True,
                         show_cue_zone=True, reward_zones=PANEL_REWARD_ZONES,
                         zone_labels=None, cue_zone_label=None,
                         time_ticks=PANEL_TIME_TICKS, legend_ncol=1, show_ylabel=False,
                         show_yticklabels=True, split_x_spine=True,
                         mask_gaps=True, label=None,
                         fontsize=PANEL_FONTSIZE, label_fontsize=PANEL_LABEL_FONTSIZE,
                         figsize=PANEL_FIGSIZE,
                         height_ratios=PANEL_HEIGHT_RATIOS, t0_labels=PANEL_T0_LABELS,
                         save_dir=None, stem=None, dpi=export.DPI):
    """The compressed two-row panel: position traces above, condition curves below.

    Row 0 is every trial's own track position over the same t0-aligned, gapped x-axis the
    curves below use (see `draw_position_traces`), with the cue zone (striped in both cue
    colours) and the R1 reward zone marked on it; row 1 is `draw_condition_lines`' curves -
    the mean z-scored firing rate of `unit` (None = the across-neuron mean) per cell of
    `conditions`, each inside its 95% CI (`sem_z`). Both rows share one x-axis, so a feature
    of the curves can be read off against where the animal was when it happened, which is
    what makes this the panel that grounds every later t0-aligned figure.

    `gap_bins` defaults to PANEL_GAP_BINS, well above the rest of this tree's GAP_BINS: the
    two intervals are seconds apart and in no fixed order, and at the narrow default gap
    they read as one continuous stretch of time. `split_x_spine` and `mask_gaps` say the same
    thing again in the axis and in the background bands - see `draw_split_x_spine` and
    `draw_gap_mask`. `smooth` is per trial before averaging
    (see `condition_block`) - 5 bins = 200ms here, so these curves are smoother than the
    native 40ms rate; pass 1 for the raw thing.

    The bands are drawn unlabelled: pass `zone_labels=ZONE_LABELS` / `cue_zone_label=
    CUE_ZONE_LABEL` to name them on the figure itself, which the poster panels do not - those
    get their labels by hand in Inkscape, positioned per panel.

    `conditions` defaults to `'cue'` alone (two plain solid curves, orange/purple) rather
    than `plot_condition_lines`' 2x2 - this is the reduced form, and four bicolour dashed
    curves in a 2.9-inch panel is not something an audience resolves. Pass
    ('cue', 'choice_R1') for the full grid; the encoding is identical either way.

    `ylim` CLIPS and is what makes a sequence of these panels comparable - see
    `draw_condition_lines`, and note the across-neuron mean sits on a ~5x smaller scale than
    a single unit. `position_ylim` does the same for row 0, which otherwise autoscales to
    however far the animal happened to get in each session's own windows.

    **`show_legend` and `show_ylabel` are both OFF by default** - Simon's standing instruction
    for this panel ("never plot the ylabel, and no legends"): every column of a canvas would
    otherwise repeat the same key and the same two axis labels, and the poster names the axes
    once, beside the assembled figure. Turn them on for a panel read on its own.
    `show_yticklabels=False` blanks the numbers on the y axes (the ticks stay), for every
    column of a canvas after the first - the scale is the same in all of them. `show_n=False`
    drops the trial counts from the legend, which is what would let one key
    serve a whole canvas (the counts are per session, and are the only reason
    `plot_condition_lines`' legend cannot be shared); `legend_ncol=1` stacks it, which is what
    fits beside the header on a panel this narrow. `label` (e.g. 'S9') titles the panel, since
    a canvas column has to say which session it is.

    Returns {'fig', 'axes': (ax_pos, ax_fr), 'block', 'cue_zone', 'zones', 'paths'}.
    """
    block = condition_block(t0_events, fr_z, s_id, which_intervals, unit=unit,
                            unit_cols=unit_cols, conditions=conditions, smooth=smooth,
                            gap_bins=gap_bins, min_trials=min_trials)
    pos_layout, _ = layout_intervals(t0_events, behavior, s_id, which_intervals,
                                     value_fn=lambda dat: dat.frame_position.values,
                                     gap_bins=gap_bins)

    fig, (ax_pos, ax_fr) = plt.subplots(nrows=2, sharex=True, figsize=figsize,
                                        gridspec_kw={'height_ratios': list(height_ratios)})
    # sharex shares the tick locator/formatter, not just the view range, so _draw_x_axis on
    # ax_fr would otherwise stamp the same t0 labels onto ax_pos as well (the trap
    # dim_plotting_lib records having hit in every one of its multi-row figures).
    ax_pos.tick_params(axis='x', labelbottom=False, length=0)
    ax_pos.spines['bottom'].set_visible(False)   # the x-axis belongs to the row below; drawn
                                                 #   here too it reads as this row's own

    cue_zone = cue_zone_extent(behavior, s_id) if show_cue_zone else None
    zones = track_zone_extents(behavior, s_id, reward_zones or ())
    draw_position_traces(ax_pos, pos_layout, cue_zone=cue_zone, reward_zones=zones,
                         cue_zone_label=cue_zone_label, zone_labels=zone_labels,
                         cue_zone_fontsize=fontsize - 4)
    if mask_gaps:
        draw_gap_mask(ax_pos, pos_layout)
    if position_ylim is not None:
        ax_pos.set_ylim(*position_ylim)

    handles, legend_labels = draw_condition_lines(
        ax_fr, block, ylim=ylim, show_sem=show_sem, sem_z=sem_z, sem_alpha=sem_alpha,
        show_legend=False, show_xticks=False, show_ylabel=False, fontsize=fontsize)
    # x tick labels a step down from the body type: at PANEL_FIGSIZE's narrow width two of
    # them ('cue entry' and the time tick) are most of the axes' width at full size.
    _draw_x_axis(ax_fr, block['layout'], fontsize=fontsize - 2, labels=t0_labels, minor=False,
                 time_ticks=time_ticks)
    if mask_gaps:
        draw_gap_mask(ax_fr, block['layout'])
    if split_x_spine:
        draw_split_x_spine(ax_fr, block['layout'])

    for ax, ylabel in zip((ax_pos, ax_fr), PANEL_YLABELS):
        if show_ylabel:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=PANEL_YTICKS))
        ax.tick_params(axis='y', labelsize=fontsize - 1)
        if not show_yticklabels:
            # labelcolor='none', NOT labelleft=False: the labels stay in the layout, so
            # tight_layout reserves the same left margin as a panel that shows them and every
            # column of a canvas keeps identical axes geometry. Hiding them outright would
            # widen the plot area of every panel but the first, and the blocks would no longer
            # line up across the assembled figure.
            ax.tick_params(axis='y', labelcolor='none')
        ax.spines[['top', 'right']].set_visible(False)   # 'bottom' is left alone here: the
                                                         #   position row has none, and the
                                                         #   neural row's is drawn by hand

    if show_legend and handles:
        # Above the top row and right-aligned, sharing that band with the label - the same
        # place plot_condition_lines puts its own. Inside row 1 it would sit on the very
        # curves it labels, and above row 0 it reads as the panel's key rather than as the
        # neural row's alone.
        ax_pos.legend(handles, legend_labels if show_n else block['row_labels'],
                      fontsize=fontsize - 1, frameon=False, ncol=legend_ncol,
                      loc='lower right', bbox_to_anchor=(1.0, 1.0), handlelength=1.4,
                      handletextpad=0.5, columnspacing=1.0, borderpad=0,
                      handler_map={tuple: HandlerTuple(ndivide=1)})

    if label is not None:
        # A title rather than text inside the axes: in-axes at the top left it printed over
        # the y tick labels on a canvas column narrow enough to matter (S11 sat on its own
        # '100'), and the band above row 0 is free anyway. Drawn after the legend below so it
        # can be padded clear of it.
        # Pad the header clear of the legend when both share the band above the row - the same
        # thing legend_title_pad does for plot_condition_lines' own title.
        title_pad = (legend_title_pad(len(handles), legend_ncol, fontsize)
                     if (show_legend and handles) else 4)
        ax_pos.set_title(label, loc='left', fontsize=label_fontsize, fontweight='bold',
                         pad=title_pad)
    fig.tight_layout(h_pad=0.4)

    stem = stem or f'{s_id}_{block["unit_label"].replace(" ", "")}_condition_panel'
    return {'fig': fig, 'axes': (ax_pos, ax_fr), 'block': block, 'cue_zone': cue_zone,
            'zones': zones, 'paths': export.save_figure(fig, save_dir, stem, dpi=dpi)}
