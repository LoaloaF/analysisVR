"""What the animal did - the behaviour-only figures, before any neural data enters.

Two families, both matplotlib, both called from `poster_26.py`:

- **Kinematics over track position** (`plot_session_kinematics`), one figure per session x
  metric: the median trajectory of a movement metric against position along the track,
  split by the animal's R1 choice (stopped at the first reward zone vs skipped it), with a
  10th-90th percentile band per group. This is a matplotlib rewrite of
  `dashsrc.plot_components.plots.plot_SessionKinematics.render_plot` (plotly), which
  `lm4_plots.py` called - same columns, same grouping, same median-plus-80%-band
  statistic. Two things are deliberately dropped: the track-illustration strip that plotly
  figure draws in a slim top axis (not wanted on the poster), and the dashboard's
  `group_filter_data` pre-pass (its default filters keep every outcome, both cues, all
  thirds of the session and both choices, so all it actually did was drop rows with a
  missing label - `_labelled_trials` here does that in one line). Rewritten rather than
  reused because the poster needs .svg panels to take apart in Inkscape, and plotly's
  image export needs kaleido, which is not installed in this environment - that is the one
  cell of `lm4_plots.py` that never ran end to end.

  Colours and level names come from `plotlib.colors`/`plotlib.trial_labels` rather than
  from `dashsrc.components.dashvis_constants` (whose R1_CHOICE_CMAP holds the same two
  blues): a stop trial has to be the same blue here as in the SVM and variance figures,
  and importing dashvis_constants would pull plotly into this layer.

- **Correctness across the session sequence** (`plot_correctness_sequence`), one figure for
  the whole sequence: the fraction of trials with the correct R1 (or R2) choice, per session,
  as a SCATTER with a fitted trend line per group - the points are not joined, because ~20-60
  trials per session cannot support the wiggle a connecting line would draw. Three groups,
  each fitted separately: the pre-cue sessions pooled (grey), then cue1 and cue2 over the
  cued sessions. What counts as correct differs between the two phases - see the comment
  above CORRECTNESS_COLUMN, which is the important thing about this figure. This replaces
  `lm4_plots.py`'s `double_stop_performance.svg` cell, which showed the reward-rate
  proportions of the first 8 sessions in two stacked axes (and whose scatter-plus-fit form,
  and light/dark grey pair for the two reward locations, this follows).

Layering: a `*_plotting_lib` (figure composer). It imports only from `plotlib` and does no
NAS I/O - the frames come from `plotlib.session_data.load_trackwise`/`load_trialwise`.
"""

import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plotlib import export
from plotlib.colors import (CUE_SPAN_ALPHA, CUE_SPAN_EDGE_COLOR, CUE_SPAN_HATCH,
                            MODALITY_LEVEL_OFFSET, ZONE_SPAN_ALPHA, ZONE_SPAN_COLORS,
                            ZONE_SPAN_EDGE_COLOR, ZONE_SPAN_FALLBACK_COLOR,
                            ZONE_SPAN_LINEWIDTH, modality_colors)
from plotlib.trial_labels import level_label


# ===========================================================================
# Kinematics over track position
# ===========================================================================

# The `posbin_*` column each metric reads, plus how to label and scale it. `column` is the
# unsmoothed column; `smooth=True` appends SMOOTHED_SUFFIX to it (the analytic stores both,
# see get_BehaviorTrackwise), which is what the dashboard's `smooth_data` checkbox did.
# `ylim` replaces the dashboard's single `metric_max` number, which it turned into either
# (0, max) or (-max, max) depending on whether the metric was signed; here that is written
# out per metric instead of re-derived. `smoothable` is False for a metric with no
# `_500msMedian` variant on the NAS.
# Poster type. Every figure-level function takes `fontsize=`, but the default is the poster
# one rather than a screen one: these panels are read from a metre away, and a default that
# needs overriding at every call site is the wrong default. The figsizes below are scaled to
# match - a 20pt label in a 5.5x3in axes leaves no room for the curve.
FONTSIZE = 20
KINEMATICS_FIGSIZE = (7.0, 2.1)
# Both panels are sized to their slot on the poster - the same 65 x 135mm rectangle, each
# with the session axis along its long side.
MM = 1 / 25.4
SEQUENCE_FIGSIZE = {'x': (135 * MM, 65 * MM), 'y': (65 * MM, 135 * MM)}
# ...and the type is set for it. FONTSIZE is a physical size, not a relative one, so the
# rotated panel needs its own: it is as large as fits, bounded by the fraction axis label
# ('R2 correct [frac.]') having to stay inside 65mm. `ink_scale` carries this into the marker
# and line sizes, so the points grow with the type rather than needing their own number.
SEQUENCE_FONTSIZE = {'x': 18, 'y': 18}

SMOOTHED_SUFFIX = '_500msMedian'
KINEMATICS_METRICS = {
    'RawYawPitch Sum Velocity': {'column': 'posbin_RawYawPitch_abs_vel_sum', 'smoothable': True,
                                 'label': 'Summed Ball velocity [cm/s]', 'ylim': (0, 220),
                                 'slug': 'RYP_sum'},
    'BallForward Velocity': {'column': 'posbin_raw', 'smoothable': True,
                             'label': 'BallForward Velocity [cm/s]', 'ylim': (0, 220),
                             'slug': 'BallForward'},
    'BallSide Velocity': {'column': 'posbin_pitch', 'smoothable': True,
                          'label': 'BallSide Velocity [cm/s]', 'ylim': (-110, 110),
                          'slug': 'BallSide'},
    'BallRotation Velocity': {'column': 'posbin_yaw', 'smoothable': True,
                              'label': 'BallRotation Velocity [cm/s]', 'ylim': (-110, 110),
                              'slug': 'BallRotation'},
    'Velocity': {'column': 'posbin_velocity', 'smoothable': False,
                 'label': 'Velocity [cm/s]', 'ylim': (0, 220), 'slug': 'velocity'},
}

# Level draw order per grouping. The group drawn LAST sits on top, and for a choice that
# is the stop curve: it is the one with the feature to read (the deceleration in the
# reward zone), so it should not be buried under the skip curve's percentile band.
GROUPING_ORDER = {'choice_R1': (0, 1), 'choice_R2': (0, 1), 'cue': (1, 2)}

BAND_PERCENTILES = (10, 90)   # the dashboard's '80th percent.' variability band
BAND_ALPHA = 0.25
TRIAL_ALPHA = 0.25            # single-trial traces, when show_trials is on
MEDIAN_LINEWIDTH = 2.0
TRIAL_LINEWIDTH = 0.5
MIN_TRIALS = 3                # a choice group below this many trials is not drawn at all -
                              #   a "median trajectory" of one or two trials is a trajectory
# The zone greys (R1 dark, R2 light, both outlined - from the poster's own stage diagram)
# and their face-only alpha now live in plotlib/colors.py, imported above under these same
# names: the poster's neural panel marks the R1 zone on its position row and the two columns
# must not drift apart. See there for why the diagram wins over lm4_plots' opposite pairing.
REWARD_ZONE_SPANS = ('reward1Zone', 'reward2Zone')
# The cue is rendered across two consecutive track_zone levels ('visibleCue' then
# 'nextToCue', -120 to 10cm here), which together are exactly the mask integr_analytics
# builds its `cue_visible` feature from - so they merge into one span under this key.
# integr_analytics' zones_boundaries names cueZone_visible=-120/cueZone_entry=-80/
# cueZone_exit=25; the track_zone edges are used instead for the same reason the reward
# zones are, that the position bins carry their own zone and need no second table (the
# only disagreement is the exit, 10 vs 25, one bin's worth at the far end of the cue).
CUE_ZONE_SPANS = ('visibleCue', 'nextToCue')
CUE_SPAN_KEY = 'cue'
# How that span is drawn - hatched and unfilled, checked against a paler fill and against
# dashed boundary lines before settling on it - now lives in plotlib/colors.py (imported
# above under these same names), since the poster's neural panel draws the identical band on
# its own kinematics row and the two must not drift apart.
XTICK_STEP = 100              # cm between position ticks - a poster font does not fit the
                              #   default ~9 ticks across the track
YTICK_STEP = 100              # and the same, sparsely, up the metric axis
# The metric names are longer than a 2.1in-tall panel is: set as one line at FONTSIZE, a
# rotated 'Summed Ball velocity [cm/s]' is taller than the whole figure and silently clips
# at both ends. Wrapping keeps the words and the type size - it spends a little width, which
# this panel has - rather than shrinking the label or abbreviating the metric.
YLABEL_WRAP = 12              # characters per line of the metric label


def metric_column(metric, smooth=True):
    """The trackwise column `metric` reads, honouring `smooth` (see KINEMATICS_METRICS)."""
    spec = KINEMATICS_METRICS[metric]
    if smooth and spec['smoothable']:
        return spec['column'] + SMOOTHED_SUFFIX
    return spec['column']


def group_color(groupby, level):
    """The canonical colour of one level of a trial label, in the label's own raw encoding
    (cue as 1/2, choices as 0/1) - `modality_colors` is keyed by the fits' 0/1 level, so
    cue needs MODALITY_LEVEL_OFFSET subtracted first."""
    offset = MODALITY_LEVEL_OFFSET.get(groupby, 0)
    return modality_colors.get(groupby, {}).get(int(level) - offset, 'grey')


def group_label(groupby, level, n_trials=None):
    """'stop (n=47)' etc. - `plotlib.trial_labels`' name for the level, with its trial
    count when one is given."""
    offset = MODALITY_LEVEL_OFFSET.get(groupby, 0)
    label = level_label(groupby, int(level) - offset)
    return label if n_trials is None else f'{label} (n={n_trials})'


def _labelled_trials(trackwise_s, groupby, value_column):
    """Rows with a trial id, a group level and a value - what `group_filter_data`'s default
    filters left of a session (see the module docstring)."""
    needed = ['trial_id', groupby, value_column]
    return trackwise_s.dropna(subset=needed)


def renders_cue(trackwise_s):
    """Whether this session put a cue on the track at all.

    It did not while BOTH reward zones were rewarded - that is the same
    `& ~both_R1_R2_rewarded` term integr_analytics' cue_visible mask carries, and it is why
    the pre-cue sessions must not get a cue span drawn on them: the track_zone levels are
    still named 'visibleCue'/'nextToCue' in those sessions, but nothing was rendered there.
    Defaults to True if the column is absent, which is the post-fork case."""
    if 'both_R1_R2_rewarded' not in trackwise_s.columns:
        return True
    return not bool(trackwise_s['both_R1_R2_rewarded'].max())


def zone_spans(trackwise_s, reward_zones=REWARD_ZONE_SPANS, cue_zones=CUE_ZONE_SPANS):
    """{key: (first_bin, last_bin)} in cm - one entry per reward zone present in the
    session's own `track_zone` column, plus a merged CUE_SPAN_KEY entry spanning the stretch
    of track the cue is rendered over, when this session rendered one (`renders_cue`)."""
    if 'track_zone' not in trackwise_s.columns:
        return {}
    # observed=True: track_zone is a categorical column, and its unused categories
    # would otherwise come back as empty groups (and a FutureWarning)
    extent = {zone: (float(rows.min()), float(rows.max())) for zone, rows
              in trackwise_s.groupby('track_zone', observed=True)['from_position_bin']}

    spans = {zone: extent[zone] for zone in reward_zones if zone in extent}
    cue_parts = [extent[zone] for zone in cue_zones if zone in extent]
    if cue_parts and renders_cue(trackwise_s):
        spans[CUE_SPAN_KEY] = (min(lo for lo, _ in cue_parts), max(hi for _, hi in cue_parts))
    return spans


def kinematics_traces(trackwise_s, metric='RawYawPitch Sum Velocity', groupby='choice_R1',
                      smooth=True, min_trials=MIN_TRIALS, percentiles=BAND_PERCENTILES):
    """The curves one kinematics figure draws, as a dict - no matplotlib involved.

    `trackwise_s` is ONE session's `load_trackwise` frame. Every group's statistics are
    taken over a trials x position-bin matrix (`pivot`), so the median is a median over
    trials at each position and the band is the percentile spread across trials there -
    which is what the plotly version's pooled-rows quantiles came to, since the frame holds
    exactly one row per (trial, position bin). Position bins missing from a trial (it never
    reached them) stay NaN and are simply excluded at that bin rather than counted as zero.

    Returns {'positions', 'groups', 'metric', 'column', 'metric_label', 'ylim', 'groupby',
    'zone_spans', 'n_trials'}, where 'groups' is one dict per level in GROUPING_ORDER:
    {'level', 'label', 'color', 'median', 'lower', 'upper', 'n_trials', 'matrix'}.
    """
    spec = KINEMATICS_METRICS[metric]
    column = metric_column(metric, smooth=smooth)
    data = _labelled_trials(trackwise_s, groupby, column)
    positions = np.sort(data['from_position_bin'].unique())

    groups = []
    for level in GROUPING_ORDER.get(groupby, tuple(sorted(data[groupby].unique()))):
        group_data = data[data[groupby] == level]
        n_trials = group_data['trial_id'].nunique()
        if n_trials < min_trials:
            continue
        matrix = (group_data.pivot_table(index='trial_id', columns='from_position_bin',
                                         values=column)
                  .reindex(columns=positions))
        values = matrix.to_numpy(dtype=float)
        with np.errstate(invalid='ignore'):   # an all-NaN position column is expected
            median = np.nanmedian(values, axis=0)
            lower, upper = np.nanpercentile(values, percentiles, axis=0)
        groups.append({'level': level, 'label': group_label(groupby, level, n_trials),
                       'color': group_color(groupby, level), 'n_trials': n_trials,
                       'median': median, 'lower': lower, 'upper': upper, 'matrix': matrix})

    return {'positions': positions, 'groups': groups, 'metric': metric, 'column': column,
            'metric_label': spec['label'], 'ylim': spec['ylim'], 'groupby': groupby,
            'zone_spans': zone_spans(trackwise_s), 'n_trials': data['trial_id'].nunique()}


def axis_ticks(lo, hi, step):
    """Ticks every `step` inside [lo, hi] - the round numbers strictly within the range,
    never the endpoints, so a publication-size font gets few short labels instead of the
    default ~9 crowded ones."""
    first = int(np.ceil(lo / step) * step)
    last = int(np.floor(hi / step) * step)
    return np.arange(first, last + 1, step)


def draw_kinematics(ax, traces, ylim=None, show_band=True, show_trials=False,
                    show_zone_spans=True, show_legend=True, show_ylabel=True,
                    show_xlabel=True, show_xticklabels=False, legend_loc='best',
                    fontsize=FONTSIZE, xtick_step=XTICK_STEP, ytick_step=YTICK_STEP,
                    ylabel_wrap=YLABEL_WRAP):
    """Draw `kinematics_traces`' output into `ax`. Returns the legend handles."""
    if show_zone_spans:
        for zone, (first_bin, last_bin) in sorted(traces['zone_spans'].items()):
            if zone == CUE_SPAN_KEY:
                ax.axvspan(first_bin, last_bin, facecolor='none', zorder=0,
                           edgecolor=to_rgba(CUE_SPAN_EDGE_COLOR, CUE_SPAN_ALPHA),
                           hatch=CUE_SPAN_HATCH, linewidth=ZONE_SPAN_LINEWIDTH)
            else:
                face = ZONE_SPAN_COLORS.get(zone, ZONE_SPAN_FALLBACK_COLOR)
                ax.axvspan(first_bin, last_bin, zorder=0,
                           facecolor=to_rgba(face, ZONE_SPAN_ALPHA),
                           edgecolor=ZONE_SPAN_EDGE_COLOR, linewidth=ZONE_SPAN_LINEWIDTH)

    positions = traces['positions']
    handles = []
    for group in traces['groups']:
        if show_trials:
            for _, trial_values in group['matrix'].iterrows():
                ax.plot(positions, trial_values.to_numpy(dtype=float), color=group['color'],
                        alpha=TRIAL_ALPHA, linewidth=TRIAL_LINEWIDTH, zorder=1)
        if show_band:
            ax.fill_between(positions, group['lower'], group['upper'], color=group['color'],
                            alpha=BAND_ALPHA, linewidth=0, zorder=2)
        ax.plot(positions, group['median'], color=group['color'], linewidth=MEDIAN_LINEWIDTH,
                zorder=3)
        handles.append(Line2D([], [], color=group['color'], linewidth=MEDIAN_LINEWIDTH,
                              label=group['label']))

    ax.set_xlim(positions.min(), positions.max())
    ylim = ylim if ylim is not None else traces['ylim']
    ax.set_ylim(*ylim)
    ax.set_xticks(axis_ticks(positions.min(), positions.max(), xtick_step))
    ax.set_yticks(axis_ticks(*ylim, ytick_step))
    # The position numbers are noise on the assembled panel - the track illustration placed
    # under it in Inkscape is what says where along the track a feature sits - so the ticks
    # stay (they carry the scale) but their labels are off by default.
    if not show_xticklabels:
        ax.set_xticklabels([])
    if show_xlabel:
        # units with no numbers against them say nothing, so they go with the tick labels
        ax.set_xlabel('Track Position [cm]' if show_xticklabels else 'Track Position',
                      fontsize=fontsize)
    if show_ylabel:
        ax.set_ylabel(textwrap.fill(traces['metric_label'], ylabel_wrap), fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.grid(axis='y', color='lightgrey', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    if show_legend and handles:
        legend_handles = handles + ([Patch(facecolor='grey', alpha=BAND_ALPHA,
                                           label=percentile_label(BAND_PERCENTILES))]
                                    if show_band else [])
        # Sized to be readable, not to fit: at 7x2.1in there is no spot inside the axes that
        # does not sit on a curve, so it is left overlapping deliberately. `svg.fonttype:
        # none` keeps it as real text in the .svg, so it gets dragged to a free spot when the
        # panel is assembled in Inkscape - which is also why it is not shrunk to squeeze in.
        ax.legend(handles=legend_handles, fontsize=fontsize - 6, frameon=False,
                  loc=legend_loc, labelspacing=0.3, handlelength=1.4, handletextpad=0.4,
                  borderaxespad=0.2)
    return handles


def percentile_label(percentiles):
    lo, hi = percentiles
    return f'{lo:g}-{hi:g}th pct.'


def plot_session_kinematics(trackwise_s, s_id=None, metric='RawYawPitch Sum Velocity',
                            groupby='choice_R1', smooth=True, ylim=None, show_band=True,
                            show_trials=False, show_zone_spans=True, show_legend=True,
                            show_ylabel=True, show_xlabel=True, show_xticklabels=False,
                            legend_loc='best', min_trials=MIN_TRIALS, title=None,
                            figsize=KINEMATICS_FIGSIZE, fontsize=FONTSIZE,
                            xtick_step=XTICK_STEP, ytick_step=YTICK_STEP,
                            ylabel_wrap=YLABEL_WRAP, save_dir=None, stem=None,
                            dpi=export.DPI):
    """One session's median kinematics against track position, split by `groupby`.

    The figure-level entry point: shapes the curves (`kinematics_traces`), draws them
    (`draw_kinematics`), and optionally writes the PNG/SVG pair. `s_id` is only used for
    the default title and file stem. `title=''` suppresses the title. Returns
    {'fig', 'ax', 'traces', 'paths'}.
    """
    traces = kinematics_traces(trackwise_s, metric=metric, groupby=groupby, smooth=smooth,
                               min_trials=min_trials)
    fig, ax = plt.subplots(figsize=figsize)
    draw_kinematics(ax, traces, ylim=ylim, show_band=show_band, show_trials=show_trials,
                    show_zone_spans=show_zone_spans, show_legend=show_legend,
                    show_ylabel=show_ylabel, show_xlabel=show_xlabel,
                    show_xticklabels=show_xticklabels, legend_loc=legend_loc,
                    fontsize=fontsize, xtick_step=xtick_step, ytick_step=ytick_step,
                    ylabel_wrap=ylabel_wrap)
    if title is None:
        title = f'S{s_id} - {traces["n_trials"]} trials' if s_id is not None else ''
    if title:
        ax.set_title(title, fontsize=fontsize + 1, loc='left')
    fig.tight_layout()

    if stem is None:
        stem = (f'session{s_id}_kinematics_{groupby}_'
                f'{KINEMATICS_METRICS[metric]["slug"]}')
    return {'fig': fig, 'ax': ax, 'traces': traces,
            'paths': export.save_figure(fig, save_dir, stem, dpi=dpi)}


# ===========================================================================
# Correctness across the session sequence
# ===========================================================================

# What "correct" means is NOT the same in the two phases, and this is the thing to get right.
#
# Before the fork BOTH reward zones were rewarded and the cue predicted nothing, so there is
# no cue-dependent correct action: stopping at the zone simply IS the rewarded behaviour, and
# the measure is the fraction of trials the animal stopped there. That is what lm4_plots.py's
# double_stop_performance cell counted (off trial_outcome's two reward digits; the equivalent
# and simpler `choice_R1`/`choice_R2 == 1` is used here, and the two agree by construction -
# in this phase a stop at a zone is always rewarded). Scoring these sessions with the CUED
# rule instead - which is what this figure did until 2026-09-06 - pins them flat at chance,
# but that is an artifact of applying a contingency that did not exist yet, not behaviour.
#
# After the fork the cue determines which action is correct, and BehaviorTrialwise's own
# choice_R1_correct/choice_R2_correct is that measure, split by cue.
CORRECTNESS_COLUMN = {'R1': 'choice_R1_correct', 'R2': 'choice_R2_correct'}
CHOICE_COLUMN = {'R1': 'choice_R1', 'R2': 'choice_R2'}
STOP_LEVEL = 1                         # choice_R1/R2's own encoding: 1 = stopped at the zone
CUES = (1, 2)                          # cue's own raw encoding in the behavior tables
# Which cue makes STOPPING the correct action at that zone, once the cue means anything. The
# split is by cue, so this only names the cue's action in the legend.
STOP_IS_CORRECT_CUE = {'R1': 1, 'R2': 2}

PHASE_POOLED, PHASE_CUED = 'both_rewarded', 'cued'

# The fork lines are CUE subsets - each is "the trials whose cue made this action correct
# here" - so they are drawn in that cue's own colour (orange for cue1, purple for cue2, from
# plotlib.colors), the same colour that cue carries in the SVM and variance figures. Because
# STOP_IS_CORRECT_CUE flips between the zones, the orange line is the stop-correct one in the
# R1 figure and the skip-correct one in the R2 figure: the colour follows the cue across the
# poster, and the legend names which action it is.
# The pre-cue group is that reward zone's OWN trace, so it takes that zone's grey from the
# stage diagram (ZONE_SPAN_COLORS) - dark for R1, light for R2 - instead of one grey standing
# for both. So the grey points in the R1 panel are the same dark grey as the R1 zone the
# kinematics figures shade, and a reader carries one colour per zone across the whole poster.
# matplotlib's named 'darkgrey' (#A9A9A9) is not dark and was what these used before.
POOLED_COLORS = ZONE_SPAN_COLORS
POOLED_FALLBACK_COLOR = ZONE_SPAN_FALLBACK_COLOR
# ...and the light one needs the diagram's outline to read as a shape rather than a smudge on
# white, which is exactly why the zone patches carry one too.
POOLED_EDGE_COLOR = ZONE_SPAN_EDGE_COLOR
# Above this the fill is too pale to stand on its own. Set between orange (0.68, which must
# NOT get an outline - a ring on one cue colour and not the other reads as a distinction) and
# the light R2 grey (0.85, which must).
POOLED_EDGE_LUMINANCE = 0.75
CHANCE_LEVEL = 0.5
MARKER = 'o'
# Ink sizes, quoted at FONTSIZE and scaled from it by `ink_scale` - a dot and a line weight
# are as much part of the panel's physical design as the type is, so drawing the 55mm panel
# with the 250mm panel's 8pt dots would leave it looking like a different figure, not a
# smaller one.
MARKERSIZE = 8                         # legend handle; MARKER_AREA is the scatter's own size
MARKER_AREA = 70
SEQUENCE_LINEWIDTH = 2.5
CI_Z = 1.96                            # 95% Wald interval on a proportion, when show_ci
MIN_FIT_POINTS = 3                     # below this a "trend" is just the two endpoints
SESSION_TICK_STEP = 5                  # gridline every 5th session...
                                       # ...but a NUMBER only at the two ends: the sequence
                                       # needs its endpoints named and nothing in between.
FRACTION_TICKS = (0.0, CHANCE_LEVEL, 1.0)
FRACTION_TICK_LABELS = ('0.0', '', '1.0')   # 0.5 is drawn as the chance line, not a number
FRACTION_LABEL = 'Correct'             # which zone it is, is said by the panel's placement
LABEL_WEIGHT = 'bold'                  # poster type, at every size

# Each cue line is labelled in place instead of through a legend, and the wording names the
# ACTION in the choice palette's blues - blue means the animal's choice everywhere on this
# poster, so the words are blue while the line itself stays the cue's own orange/purple. The
# two encodings are on the same mark on purpose: the cue is which line, the action is what
# doing it right looked like.
ACTION_LABEL_COLORS = {'stop': modality_colors['choice_R1'][1],
                       'skip': modality_colors['choice_R1'][0]}
ACTION_LABEL_OFFSET = 0.13             # from the line, in the fraction axis' own 0-1 units
ACTION_LABEL_FLOOR = 0.04              # ...but never pushed off the axis
# ...and staggered ALONG the sessions, one per cue: the two labels are long, and set at the
# same midpoint they overlap each other wherever the two cue lines run close together.
ACTION_LABEL_ALONG = (0.34, 0.70)
# Direction of training, in axes fractions: down the sessions when they run down the page,
# along them when they run across it.
PROGRESS_ARROW = {'x': {'xytext': (0.06, 0.90), 'xy': (0.26, 0.90)},
                  'y': {'xytext': (0.13, 0.62), 'xy': (0.13, 0.36)}}
PROGRESS_ARROW_WIDTH = 5.0             # scaled by ink_scale, like every other ink size
PROGRESS_ARROW_HEAD = 34.0
# 'best' rather than a fixed corner: which corner is free differs across the four layouts
# (R1/R2 x upright/rotated) - the rotated R2 panel in particular has points in the upper left
# that a fixed 'upper left' legend sat on top of. matplotlib's own collision check is right
# more often than a hardcoded guess, and `legend_loc=` still overrides it per call.
SEQUENCE_LEGEND_LOC = 'best'


def cue_contingency_split(trialwise):
    """The last session of the pre-cue phase: the LAST one in which both reward zones were
    rewarded (`both_R1_R2_rewarded == 1`), i.e. the last session in which the cue did not
    yet determine which choice was correct.

    Derived from the data rather than hardcoded, because it is a property of the animal's
    training schedule (session 7 for rYL006/P1100, where sessions 0-7 have the flag set
    and 8 onwards do not). Falls back to one before the first session in the frame if the
    flag is never set, which makes the whole sequence the cued phase."""
    per_session = trialwise.groupby('s_id')['both_R1_R2_rewarded'].max()
    both_rewarded = per_session[per_session == 1]
    return int(both_rewarded.index.max()) if len(both_rewarded) else int(per_session.index.min()) - 1


def correctness_by_session(trialwise, reward='R1', s_ids=None, split_s_id=None):
    """Per session, the fraction of correct `reward` choices - scored by the rule that
    actually applied in that session (see the comment above), and split by cue once there
    is a cue contingency to split by.

    `trialwise` is a `load_trialwise` frame carrying an `s_id` column; `s_ids` restricts and
    orders the sessions (None = every session in the frame, ascending). `split_s_id` is the
    last pre-cue session; None derives it from the data. Returns a DataFrame indexed by s_id
    with columns phase, n/correct, and n_cue1/correct_cue1, n_cue2/correct_cue2 - the cue
    columns are NaN in the pre-cue phase, where the split would be meaningless. Each n counts
    only the trials its fraction was taken over."""
    correct_col, choice_col = CORRECTNESS_COLUMN[reward], CHOICE_COLUMN[reward]
    if split_s_id is None:
        split_s_id = cue_contingency_split(trialwise)

    rows = []
    for s_id, session_trials in trialwise.groupby('s_id'):
        if s_ids is not None and s_id not in list(s_ids):
            continue
        cued = int(s_id) > split_s_id
        if cued:
            scored = session_trials[correct_col]
        else:
            choice = session_trials[choice_col]
            # .where(notna) so a trial with no scored choice stays NaN rather than becoming
            # a False - it is unscored, not incorrect
            scored = (choice == STOP_LEVEL).where(choice.notna())
        row = {'s_id': int(s_id), 'phase': PHASE_CUED if cued else PHASE_POOLED,
               'n': int(scored.notna().sum()), 'correct': scored.mean()}
        for cue in CUES:
            cue_scored = scored[session_trials['cue'] == cue] if cued else scored.iloc[:0]
            row[f'n_cue{cue}'] = int(cue_scored.notna().sum())
            row[f'correct_cue{cue}'] = cue_scored.mean() if row[f'n_cue{cue}'] else np.nan
        rows.append(row)

    table = pd.DataFrame(rows).set_index('s_id').sort_index()
    if s_ids is not None:
        table = table.reindex([s for s in s_ids if s in table.index])
    return table


def _sequence_lines(table, reward):
    """The three point clouds the figure fits a line through, in draw order: the pre-cue
    phase pooled (grey, no cue to split by), then one per cue over the cued phase."""
    pooled = table[table['phase'] == PHASE_POOLED]
    cued = table[table['phase'] == PHASE_CUED]

    pooled_color = POOLED_COLORS.get(f'reward{reward[1:]}Zone', POOLED_FALLBACK_COLOR)
    lines = [{'key': 'pooled', 'cue': None, 'action': None, 'color': pooled_color,
              's_ids': pooled.index.to_numpy(float),
              'values': pooled['correct'].to_numpy(float),
              'counts': pooled['n'].to_numpy(float), 'label': 'both rewarded'}]
    for i, cue in enumerate(CUES):
        action = 'stop' if cue == STOP_IS_CORRECT_CUE[reward] else 'skip'
        lines.append({'key': f'cue{cue}', 'cue': cue, 'action': action,
                      'label_along': ACTION_LABEL_ALONG[i], 'color': group_color('cue', cue),
                      's_ids': cued.index.to_numpy(float),
                      'values': cued[f'correct_cue{cue}'].to_numpy(float),
                      'counts': cued[f'n_cue{cue}'].to_numpy(float),
                      'label': f'{group_label("cue", cue)} ({action})'})
    return lines


def fit_line(s_ids, values, min_points=MIN_FIT_POINTS):
    """(x, y) endpoints of the least-squares line through the finite points of
    (`s_ids`, `values`), or None if there are fewer than `min_points` of them.

    Drawn instead of connecting the points, which is the whole reason the sessions are a
    scatter here: joining them draws a line through session-to-session wiggle that the ~20-60
    trials behind each point cannot support, and reads as if the animal really went up and
    down that way. The fit is deliberately not annotated with a slope or a p value - it is
    there to show the direction of the trend, and a p value on 8 points would invite more
    weight than it can carry."""
    s_ids, values = np.asarray(s_ids, float), np.asarray(values, float)
    finite = np.isfinite(s_ids) & np.isfinite(values)
    if finite.sum() < min_points:
        return None
    slope, intercept = np.polyfit(s_ids[finite], values[finite], 1)
    endpoints = np.array([s_ids[finite].min(), s_ids[finite].max()])
    return endpoints, intercept + slope * endpoints


def _action_label_position(line, offset, floor=ACTION_LABEL_FLOOR):
    """(session, fraction) to write a cue line's action label at: beside its own points,
    `line['label_along']` of the way through its sessions and `offset` away from the line
    along the fraction axis (signed - negative sits below/left of it). Taken from the points
    rather than from the fitted line so a group with `omit_fit` is still labelled. None if it
    has no points at all."""
    finite = np.isfinite(line['values'])
    if not finite.any():
        return None
    sessions, values = line['s_ids'][finite], line['values'][finite]
    session = sessions.min() + line['label_along'] * (sessions.max() - sessions.min())
    # the line's own height where the label sits, not its overall median, so the label
    # tracks a sloped line instead of drifting away from one end of it
    value = float(np.interp(session, sessions, values)) + offset
    return float(session), float(np.clip(value, floor, 1 - floor))


def _edge_for(color):
    """The outline a marker in `color` needs, or None if it is dark enough to stand alone.

    Only the pale end of the palette gets one (the light R2 grey), for the same reason the
    zone patches are outlined in the stage diagram: #d9d9d9 on white is a smudge without it."""
    r, g, b = to_rgba(color)[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return POOLED_EDGE_COLOR if luminance > POOLED_EDGE_LUMINANCE else None


def ink_scale(fontsize):
    """How much to scale the ink sizes quoted at FONTSIZE, for a panel drawn at `fontsize`.
    Areas take the square of it, lengths take it directly."""
    return fontsize / FONTSIZE


def session_ticks(s_ids, step=SESSION_TICK_STEP):
    """Every `step`-th session, plus the first and last so the range is still readable."""
    s_ids = list(s_ids)
    return sorted({s_ids[0], s_ids[-1]} | {s for s in s_ids if s % step == 0})


def draw_correctness_sequence(ax, table, reward='R1', session_axis='x', colors=None,
                              show_chance=True, show_ci=False, show_fit=True, omit_fit=(),
                              show_action_labels=True, show_progress_arrow=True,
                              show_legend=False, show_axis_labels=True, legend_loc=None,
                              session_tick_step=SESSION_TICK_STEP, fontsize=FONTSIZE):
    """Draw the correctness sequence into `ax` as one scatter per phase/cue plus its fitted
    trend line. Returns the legend handles.

    `session_axis` is which axis the session number runs along: 'x' gives the usual
    performance-over-training layout, 'y' the rotated one (sessions down the page,
    performance to the right) - a poster-layout choice, so it only swaps the coordinate
    pairs and the axis decoration, never the numbers.

    `colors`: {'pooled': .., 'cue1': .., 'cue2': ..} to override the canonical colours -
    the same three group keys `omit_fit` takes.

    `show_action_labels` writes 'correct stop'/'correct skip' beside the two cue lines in
    place of a legend (`show_legend` is off by default because of it); `show_progress_arrow`
    draws the direction of training alongside them.

    `omit_fit`: group keys to scatter WITHOUT a trend line, e.g. `('cue2',)` on the R2 panel,
    where cue2 (stop-correct) sits near the floor for the whole cued phase - a line fitted
    through it would suggest a trend in what is really a flat failure to acquire the
    behaviour at all, and a fit is a claim, so it is left off rather than drawn and caveated.

    `show_ci` adds a 95% Wald interval on each proportion - off by default because it doubles
    the ink on a figure whose point is the trend, but the per-session n is only ~20-60 trials
    per cue, so it is the honest option when a single session is read on its own."""
    lines = _sequence_lines(table, reward)
    scale = ink_scale(fontsize)
    marker_area, linewidth = MARKER_AREA * scale ** 2, SEQUENCE_LINEWIDTH * scale
    if colors:
        for line in lines:
            line['color'] = colors.get(line['key'], line['color'])

    def xy(along_session, along_fraction):
        return ((along_session, along_fraction) if session_axis == 'x'
                else (along_fraction, along_session))

    handles = []
    for line in lines:
        edge = _edge_for(line['color'])
        ax.scatter(*xy(line['s_ids'], line['values']), color=line['color'], s=marker_area,
                   marker=MARKER, zorder=3, edgecolors=edge,
                   linewidths=ZONE_SPAN_LINEWIDTH * scale if edge else 0)
        fit = fit_line(line['s_ids'], line['values'])
        if show_fit and line['key'] not in omit_fit and fit is not None:
            ax.plot(*xy(*fit), color=line['color'], linewidth=linewidth, linestyle='--',
                    zorder=2)
        if show_ci:
            with np.errstate(invalid='ignore', divide='ignore'):
                half = CI_Z * np.sqrt(line['values'] * (1 - line['values']) / line['counts'])
            bars = {'yerr' if session_axis == 'x' else 'xerr': half}
            ax.errorbar(*xy(line['s_ids'], line['values']), fmt='none', ecolor=line['color'],
                        elinewidth=1.2 * scale, alpha=0.6, zorder=2, **bars)
        handles.append(Line2D([], [], color=line['color'], linestyle='none', marker=MARKER,
                              markersize=MARKERSIZE * scale, label=line['label'],
                              markeredgecolor=edge or line['color'],
                              markeredgewidth=ZONE_SPAN_LINEWIDTH * scale if edge else 0))

    s_ids = table.index.to_numpy()
    session_lim = (s_ids.min() - 0.5, s_ids.max() + 0.5)
    ticks = session_ticks(s_ids, step=session_tick_step)
    # a gridline at every tick, a number only at the first and last
    tick_labels = [str(t) if t in (ticks[0], ticks[-1]) else '' for t in ticks]

    if session_axis == 'x':
        ax.set_xlim(*session_lim)
        ax.set_ylim(0, 1)
        ax.set_xticks(ticks, labels=tick_labels)
        ax.set_yticks(FRACTION_TICKS, labels=FRACTION_TICK_LABELS)
        ax.grid(axis='x', color='lightgrey', linewidth=0.5, linestyle='--', alpha=0.8)
        if show_chance:
            ax.axhline(CHANCE_LEVEL, color='grey', linewidth=0.8, linestyle=':', zorder=1)
        ax.spines[['top', 'right']].set_visible(False)
        if show_axis_labels:
            ax.set_xlabel('Session', fontsize=fontsize, fontweight=LABEL_WEIGHT)
            ax.set_ylabel(FRACTION_LABEL, fontsize=fontsize, fontweight=LABEL_WEIGHT)
    else:
        ax.set_ylim(session_lim[1], session_lim[0])
        ax.set_xlim(0, 1)
        ax.set_yticks(ticks, labels=tick_labels)
        ax.set_xticks(FRACTION_TICKS, labels=FRACTION_TICK_LABELS)
        ax.grid(axis='y', color='lightgrey', linewidth=0.5, linestyle='--', alpha=0.8)
        if show_chance:
            ax.axvline(CHANCE_LEVEL, color='grey', linewidth=0.8, linestyle=':', zorder=1)
        # Fraction axis along the top, session axis reading down the right - the session
        # numbers end up beside the last points rather than a panel-height away from them.
        # The tick MARKS stay on the left spine (that is the drawn edge); only the numbers
        # and the word move across.
        ax.spines[['bottom', 'right']].set_visible(False)
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(axis='y', left=True, labelleft=False, right=False, labelright=True)
        ax.yaxis.set_label_position('right')
        if show_axis_labels:
            ax.set_ylabel('Session', fontsize=fontsize, fontweight=LABEL_WEIGHT)
            # The word sits AGAINST the spine with the two end values outside it, which
            # set_xlabel cannot do - it always places the label beyond the tick labels - so
            # the word is drawn directly and the numbers are padded out past it.
            ax.text(0.5, 1.0, FRACTION_LABEL, transform=ax.transAxes, ha='center',
                    va='bottom', fontsize=fontsize, fontweight=LABEL_WEIGHT)
            ax.tick_params(axis='x', pad=fontsize * 1.35)

    ax.tick_params(labelsize=fontsize)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight(LABEL_WEIGHT)
    ax.set_axisbelow(True)

    if show_action_labels:
        cue_lines = [line for line in lines if line['action'] is not None]
        # Rotated, both labels sit to the LEFT of their line, where the panel is empty.
        # Upright there is no such free side - offsetting both down drops the upper line's
        # label straight onto the lower line - so the upper line is labelled above itself and
        # the lower one below, decided from the data rather than from which action it is
        # (the stop/skip roles swap between the R1 and R2 panels).
        heights = [np.nanmedian(line['values']) if np.isfinite(line['values']).any() else -np.inf
                   for line in cue_lines]
        upper = cue_lines[int(np.argmax(heights))] if cue_lines else None
        for line in cue_lines:
            above = session_axis == 'x' and line is upper
            placed = _action_label_position(line, ACTION_LABEL_OFFSET * (1 if above else -1))
            if placed is None:
                continue
            session, fraction = placed
            ax.text(*xy(session, fraction), f'correct {line["action"]}',
                    color=ACTION_LABEL_COLORS[line['action']], fontsize=fontsize - 2,
                    fontweight=LABEL_WEIGHT, ha='center', va='center',
                    rotation=90 if session_axis == 'y' else 0, zorder=4)

    if show_progress_arrow:
        arrow = PROGRESS_ARROW[session_axis]
        ax.annotate('', xy=arrow['xy'], xytext=arrow['xytext'], xycoords='axes fraction',
                    arrowprops={'arrowstyle': '-|>', 'color': 'black', 'shrinkA': 0,
                                'shrinkB': 0, 'linewidth': PROGRESS_ARROW_WIDTH * scale,
                                'mutation_scale': PROGRESS_ARROW_HEAD * scale})

    if show_legend:
        ax.legend(handles=handles, fontsize=fontsize - 2, frameon=False,
                  loc=legend_loc or SEQUENCE_LEGEND_LOC, labelspacing=0.3,
                  handletextpad=0.4, borderaxespad=0.2)
    return handles


def plot_correctness_sequence(trialwise, reward='R1', s_ids=None, split_s_id=None,
                              session_axis='x', colors=None, show_chance=True, show_ci=False,
                              show_fit=True, omit_fit=(), show_action_labels=True,
                              show_progress_arrow=True, show_legend=False,
                              show_axis_labels=True,
                              legend_loc=None, session_tick_step=SESSION_TICK_STEP,
                              title=None, figsize=None, fontsize=None, save_dir=None,
                              stem=None, dpi=export.DPI):
    """The whole training sequence in one axes: correct-`reward`-choice fraction per session
    as a scatter, with a fitted trend line per phase - grey while both zones were rewarded
    (where stopping is what counts as correct), then one line per cue (see the comments above).

    `reward` is 'R1' or 'R2'. `s_ids` restricts/orders the sessions (None = all in the frame).
    `split_s_id` is the last pre-cue session; None derives it from the data
    (`cue_contingency_split`). `session_axis='y'` gives the rotated layout - which carries its
    own `figsize`/`fontsize` defaults, since it is drawn at its printed size on the poster.
    Returns
    {'fig', 'ax', 'table', 'split_s_id', 'paths'}.
    """
    if split_s_id is None:
        split_s_id = cue_contingency_split(trialwise)
    table = correctness_by_session(trialwise, reward=reward, s_ids=s_ids, split_s_id=split_s_id)
    if figsize is None:
        figsize = SEQUENCE_FIGSIZE[session_axis]
    if fontsize is None:
        fontsize = SEQUENCE_FONTSIZE[session_axis]

    fig, ax = plt.subplots(figsize=figsize)
    draw_correctness_sequence(ax, table, reward=reward, session_axis=session_axis,
                              colors=colors, show_chance=show_chance, show_ci=show_ci,
                              show_fit=show_fit, omit_fit=omit_fit,
                              show_action_labels=show_action_labels,
                              show_progress_arrow=show_progress_arrow, show_legend=show_legend,
                              show_axis_labels=show_axis_labels, legend_loc=legend_loc,
                              session_tick_step=session_tick_step, fontsize=fontsize)
    if title is None:
        title = ''      # nothing is left for a title to say that the axes do not
    if title:
        # the rotated layout puts the fraction axis label above the axes too, so the title
        # has to clear it rather than land on it
        pad = 1.6 * fontsize if session_axis == 'y' else None
        ax.set_title(title, fontsize=fontsize, loc='left', pad=pad)
    fig.tight_layout()

    if stem is None:
        stem = f'correctness_sequence_{reward}' + ('' if session_axis == 'x' else '_rotated')
    return {'fig': fig, 'ax': ax, 'table': table, 'split_s_id': split_s_id,
            'paths': export.save_figure(fig, save_dir, stem, dpi=dpi)}
