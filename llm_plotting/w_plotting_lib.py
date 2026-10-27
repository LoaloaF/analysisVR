"""Figures about a single fitted SVM weight vector - what a W *is* and what it does.

Third library of the SVM analysis line, next to `svm_plotting_lib` (one session's fit
sweep) and `svm_heatmap_plotting_lib` (all sessions at a glance). The unit of analysis
here is different from both: one already-selected W row (see `highlight_selections` /
`get_session_sequence_Ws`), asked four questions -

  * what does the weight profile look like across neurons        (plot_W_list)
  * what does the population do along it, trial by trial         (plot_w_projection_session)
  * how much of the variance does that direction explain         (plot_w_variance_explained)
  * does its activation track any kinematic variable             (plot_w_projection_kinematics)

Extracted 2026-08-11 from svm_analysis_2026.py, where all of this had accumulated inline.

KNOWN DEBT: the four figure-level entry points here still call `analytics.get_analytics`
internally to fetch that session's firing rate, unlike every other library in this tree,
which takes its data as arguments. They also used to read `ith_to_dt_session` off the
notebook's globals - that is now an explicit keyword argument. Converting the firing-rate
fetch to a passed-in argument is deliberately left for the trial-trace unification, which
reworks that data path anyway.
"""

import itertools
import os
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.colors import LinearSegmentedColormap, to_rgb, rgb_to_hsv, hsv_to_rgb

from analytics_processing import analytics  # see KNOWN DEBT above

from svm_plotting_lib import (
    ALL_Y_PREDICTED_NAMES, CORRECTNESS_SPECS, CORNER_FONTSIZE, HIGHLIGHT_COLOR, HIGHLIGHT_BOX_LW,
    SUPTITLE_KW,
    _covariate_color, _level_label, _interval_short_label, _smooth, _draw_text_rows,
)
from plotlib.trial_labels import COMBO_DASH, COMBO_LINEWIDTH
from plotlib.pca import pca_eigen, plot_pca_eigenvalues
from plotlib.export import save_figure
from plotlib.session_data import session_fr_for_modality
from plotlib.trial_traces import interval_x0_map
from plotlib import fit_samples as fitspl


# ---------------------------------------------------------------- W as an object
# The region gradient marks which cortical region a neuron ID falls in. These three
# constants describe THIS probe layout (mPFC, anterior->posterior); they are recording
# metadata living in a plotting module, and would need to become data-driven if a
# second probe geometry ever gets the same treatment.


def _desaturate(color, factor):
    """`color` with its HSV saturation scaled by `factor` (0 = gray, 1 = unchanged) -
    same hue/value, just softer. Used to tone down the region-gradient's raw
    red/green/yellow, which read as too garish/saturated at full strength."""
    h, s, v = rgb_to_hsv(to_rgb(color))
    return tuple(hsv_to_rgb((h, s * factor, v)))


REGION_LABELS = ('Anterior Cingulate', 'Prelimbic', 'Infralimbic')


REGION_COLORS = tuple(_desaturate(c, 0.55) for c in ('red', 'green', 'yellow'))


REGION_FRACTIONS = (1.0, 1.25, 0.85)


def _region_gradient_cmap(colors, fractions, transition_frac=0.3):
    """Build a colormap that holds each of `colors` solid over its own span (sized by
    `fractions`, normalized to sum to 1) with a short blend in between, plus the
    normalized cumulative boundaries (`edges`, length len(colors)+1) those spans sit at -
    `_plot_region_gradient` uses `edges` again to place each region's label/gridline at
    the exact same boundary the color transition happens at. Evenly-spaced stops (as a
    plain from_list(colors) would give) make each color instantaneous rather than a real
    band - the blend into the next color starts immediately at each stop - so this instead
    only blends within a `transition_frac` fraction of the *smaller* of the two adjacent
    spans (capped there so a narrow region still keeps a visible solid core rather than
    being entirely swallowed by the transition into its wider neighbor)."""
    fractions = np.asarray(fractions, dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(fractions) / fractions.sum()])
    stops = [(0.0, colors[0])]
    for j in range(1, len(colors)):
        half_trans = transition_frac / 2 * min(edges[j] - edges[j - 1], edges[j + 1] - edges[j])
        stops.append((edges[j] - half_trans, colors[j - 1]))
        stops.append((edges[j] + half_trans, colors[j]))
    stops.append((1.0, colors[-1]))
    return LinearSegmentedColormap.from_list('region_gradient', stops), edges


def _plot_region_gradient(ax, n_features, region_labels=REGION_LABELS, colors=REGION_COLORS,
                            fractions=REGION_FRACTIONS):
    """Thin extra column for plot_W_list: a smooth color gradient (red -> green -> yellow
    by default) spanning the full Neuron ID range, split into len(region_labels) zones
    sized by `fractions` (one label per zone) - meant to mark which cortical region each
    row's neuron ID falls in, assuming they're ordered anterior-to-posterior/superficial-
    to-deep matching `region_labels`' own order. Plotted as a single-column imshow rather
    than discrete per-zone rects so the transition between zones is continuous, not a hard
    color jump (see _region_gradient_cmap for how). `origin='lower'` + a plain
    (non-flipped) extent, rather than flipping extent's bottom/top to invert directly, is
    what makes the gradient's start (red) land at Neuron ID 0 once the shared
    invert_yaxis() call elsewhere in plot_W_list flips this axis along with every other
    panel - flipping the extent instead would fight that shared call and silently cancel
    back out to an uninverted axis."""
    cmap, edges = _region_gradient_cmap(colors, fractions)
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=(0, 1, -0.5, n_features - 0.5), origin='lower')
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.tick_params(left=False, labelleft=False)
    for i, label in enumerate(region_labels):
        y_center = (edges[i] + edges[i + 1]) / 2 * n_features
        ax.text(0.5, y_center, label, rotation=90, ha='center', va='center',
                fontsize=7, color='black', fontweight='bold')


# Geometry/typography of plot_W_list, fixed rather than exposed as kwargs (see the
# "geometry knobs become module constants" convention): one stem panel is W_PANEL_WIDTH
# wide, which is also what sizes the angle heatmap's cells (it spans exactly the stem
# panels, so one cell sits under one panel) and what the panel titles are wrapped to.
# Type scale for this library's figures. Applied via @plt.rc_context so it does not
# depend on - and does not touch - whatever the calling notebook set globally: lm4_plots
# runs at font.size 15 (right for a single full-width figure, far too big for a 4-inch
# stack of nine rows, where it swallowed the suptitle and the axis labels).
W_FIGURE_RC = {
    'font.size': 7,
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.titlesize': 9,
}

W_PANEL_WIDTH = 0.8
W_PANEL_WSPACE = 0.15
W_TITLE_FONTSIZE = 6
# characters that fit across one panel at W_TITLE_FONTSIZE (~0.62 * fontsize per char),
# measured on the drawn panel (i.e. minus the gap between panels) - a line wrapped to the
# nominal panel width still runs into its neighbor's title
W_TITLE_WRAP = max(8, int(W_PANEL_WIDTH / (1 + W_PANEL_WSPACE) * 72 / (0.62 * W_TITLE_FONTSIZE)))


def _stacked_label(text, width=W_TITLE_WRAP):
    """`text`'s underscore-separated parts, capitalized, packed into lines of at most
    `width` characters - 'cue_choice_overlap_2ndsquare' -> 'CueChoice\\nOverlap2ndsquare'.
    Two width savings in one: dropping the underscores and capitalizing costs nothing but
    makes the parts read as separate words anyway, and the part boundaries are where the
    line breaks go, so a title never breaks mid-word the way a blind textwrap does. Parts
    that already carry their own capitalization ('R1') are left alone. A single part
    longer than `width` gets its own line and is allowed to overflow - breaking inside it
    would be the mid-word break this exists to avoid."""
    parts = [p.capitalize() if p.islower() else p for p in str(text).split('_') if p]
    lines = ['']
    for part in parts:
        if lines[-1] and len(lines[-1]) + len(part) > width:
            lines.append('')
        lines[-1] += part
    return '\n'.join(lines)


def _w_label_fields(row, ith_to_dt_session=None):
    """(column, text) pairs describing one W row, in title order - `name` (the
    selection's own free-form label) first, then the fixed-shape fields. Every value is
    run through _stacked_label, which is what keeps a title inside its own panel;
    `interval_name` additionally goes through interval_short_label first (same
    transformation, plus dropping the '_interval' suffix: 'R1_entry_interval' ->
    'R1Entry'), and `session_id` prints as its session number ('2024-12-04_18-06' ->
    'S13') when ith_to_dt_session is given."""
    fields = []
    if 'name' in row and pd.notna(row['name']):
        fields.append(('name', _stacked_label(row['name'])))
    for col, fmt in (('predict_y_name', '{}'), ('conditioner', 'cond={}'),
                     ('train_stratum', 'ts={}'), ('interval_name', '{}'),
                     ('timebin', 't={}'), ('session_id', '{}')):
        if col not in row or pd.isna(row[col]):
            continue
        value = row[col]
        if col == 'interval_name':
            value = _interval_short_label(str(value))
        elif col == 'session_id' and ith_to_dt_session and value in ith_to_dt_session:
            value = f'S{ith_to_dt_session[value]}'
        # the 'cond='/'t='/... prefix sits on the value's first line, so it comes out of
        # that line's own character budget
        fields.append((col, fmt.format(_stacked_label(value, W_TITLE_WRAP - len(fmt.format(''))))))
    return fields


def _w_pairwise_angles(Ws):
    """Pairwise angle (degrees) between every pair of rows of `Ws` - the same
    arccos-of-cosine-similarity quantity svm_plotting_lib.plot_w_comparison's heatmap
    shows, on the same 0-180 scale, so the two figures read identically. An all-zero w
    has no direction and comes back NaN (drawn blank) rather than as a spurious 90."""
    norms = np.linalg.norm(Ws, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_sim = (Ws @ Ws.T) / np.outer(norms, norms)
    return np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))


@plt.rc_context(W_FIGURE_RC)
def plot_W_list(highlighted_rows, plot_type='stem', min_abs_w=0.4, show_region_gradient=True,
                 region_labels=REGION_LABELS, show_angle_heatmap=True, ith_to_dt_session=None,
                 title=None, save_path=None, save_path_svg=None):
    """Plot each row's `w` vector as its own thin panel, side by side, features listed
    top-to-bottom (feature 0 at top, y-axis shared across panels so they line up
    feature-for-feature) - meant for a small hand-picked set of W's (the DataFrame
    svm_plotting_lib.resolve_w_selections returns for a highlight_selections list), not a
    whole session's worth
    of timebins side by side. `plot_type='stem'` (default) draws a horizontal lollipop per
    feature (a thin line from 0 out to the value, red = positive, blue = negative) - less
    visually noisy than a filled bar once there are more than a few dozen features, while
    still showing each individual value (unlike `plot_type='line'`, which instead connects
    the same values with a single thin black line - worth it when only the *shape* of the
    profile across features is being compared, not each value); `plot_type='bar'` keeps
    the original filled horizontal bar. `min_abs_w` (stem only, default 0.4) skips drawing
    a stem/marker for any feature whose |w| falls below it - the near-zero weights add
    visual noise without carrying much signal, so leaving them blank (rather than a barely
    visible sliver) makes the features that actually stand out easier to spot. Only the first (leftmost) panel keeps y-tick
    labels/label - panels share the same y-axis, so repeating them on every panel would
    just be clutter - and panels are packed close together (small wspace) since there's
    nothing else between them worth the gap; the same first panel carries the (also
    shared) x-axis label, the weight-coefficient scale every panel is drawn on.

    Each panel's title carries every one of
    name/session_id/predict_y_name/conditioner/train_stratum/interval_name/timebin that
    is present on that row (see _w_label_fields, and _highlighted_rows for where the
    columns come from), one field per line. What keeps those lines inside a panel only
    W_PANEL_WIDTH wide is _stacked_label: each value's underscore-separated parts are
    capitalized and packed into short lines ('cue_choice_overlap_2ndsquare' ->
    'CueChoice' / 'Overlap2ndsquare'), so the underscores cost no width and the line
    breaks land on part boundaries instead of mid-word. `ith_to_dt_session`
    ({session_id: index}, as everywhere else in this library) shortens a session further,
    to 'S13' rather than its 16-char id. Each panel is also tagged W0, W1, ... - the same
    tags label the angle heatmap's rows, so a cell can be traced back to the profile
    above it.

    With show_region_gradient (default True), one extra narrow column is added after the W
    panels - see _plot_region_gradient - marking region_labels' evenly-sized zones along
    the same (shared) Neuron ID axis. With show_angle_heatmap (default True, and only
    meaningful for 2+ rows of equal dimensionality), a square pairwise-angle heatmap is
    added below, spanning exactly the stem panels' own width - one column per panel,
    directly underneath it, so it reads as a shared x-axis without actually being one
    (the panels' x-axis is the weight scale, the heatmap's is the W index). Values,
    colormap and 0-180 scale are the same as svm_plotting_lib.plot_w_comparison's
    cross-block heatmap; its colorbar sits under the region-gradient column. Note this
    one is deliberately *not* alpha-masked by chance band or metric the way that figure
    is - it is a handful of hand-picked W's being read cell by cell, with the angles
    printed in the cells."""
    if highlighted_rows.empty:
        print("plot_W_list: nothing to plot (highlighted_rows is empty)")
        return

    n = len(highlighted_rows)
    ws = [np.asarray(row['w'], dtype=float) for _, row in highlighted_rows.iterrows()]
    n_features = len(ws[0])
    # a pairwise angle is only defined between W's of the same dimensionality (same
    # X_modality) - see plot_w_comparison, which blanks such pairs rather than comparing
    # them; here the whole figure already assumes one shared feature axis, so a mismatch
    # just drops the heatmap instead of silently comparing incomparable rows.
    same_dim = all(len(w) == n_features for w in ws)
    if show_angle_heatmap and n > 1 and not same_dim:
        print("plot_W_list: rows have different feature counts - skipping the angle heatmap")
    draw_heatmap = show_angle_heatmap and n > 1 and same_dim

    # narrow per panel (nothing to show horizontally besides the one profile) and no
    # taller than needed for n_features rows, regardless of how many panels there are;
    # the region-gradient column (if any) is narrower still - just wide enough for its
    # rotated label text, not another full data panel. The heatmap row is as tall as the
    # stem panels are wide in total, which is what makes its n x n cells square.
    extra_col = show_region_gradient or draw_heatmap  # gradient above, colorbar below
    ncols = n + 1 if extra_col else n
    width_ratios = [1] * n + ([0.4] if extra_col else [])
    stems_width = max(2, W_PANEL_WIDTH * n)
    fig_width = stems_width + (0.6 if extra_col else 0)
    stems_height = 0.07 * n_features + 0.8
    heatmap_height = stems_width if draw_heatmap else 0
    height_ratios = [stems_height] + ([heatmap_height] if draw_heatmap else [])

    fig = plt.figure(figsize=(fig_width, stems_height + heatmap_height), dpi=150)
    gs = fig.add_gridspec(len(height_ratios), ncols, width_ratios=width_ratios,
                          height_ratios=height_ratios, wspace=W_PANEL_WSPACE, hspace=0.12)
    axes = []
    for i in range(ncols):
        axes.append(fig.add_subplot(gs[0, i], sharey=axes[0] if axes else None))

    row_fields = [_w_label_fields(row, ith_to_dt_session)
                  for _, row in highlighted_rows.iterrows()]

    for i, (ax, w, fields) in enumerate(zip(axes[:n], ws, row_fields)):
        feature_idx = np.arange(len(w))  # per row: rows of unequal length still draw
        if plot_type == 'bar':
            colors = np.where(w >= 0, '#d62728', '#1f77b4')
            ax.barh(feature_idx, w, color=colors, height=0.85)
        elif plot_type == 'stem':
            keep = np.abs(w) >= min_abs_w
            colors = np.where(w >= 0, '#d62728', '#1f77b4')
            ax.hlines(feature_idx[keep], 0, w[keep], color=colors[keep], linewidth=1, zorder=2)
            ax.scatter(w[keep], feature_idx[keep], color=colors[keep], s=5, zorder=3)
        else:
            ax.plot(w, feature_idx, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=0.6, alpha=0.5)
        title_lines = [f'W{i}'] if draw_heatmap else []
        title_lines += [text for _, text in fields]  # already stacked, see _stacked_label
        ax.set_title('\n'.join(title_lines), fontsize=W_TITLE_FONTSIZE)
        ax.tick_params(labelsize=W_TITLE_FONTSIZE)
        ax.set_xlim(-1, 1)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        if i:
            # every panel is drawn on the same fixed xlim, so only the first one needs
            # tick labels - and neighboring panels' own -5/5 labels would collide anyway
            ax.tick_params(left=False, labelleft=False, labelbottom=False)

    if show_region_gradient:
        _plot_region_gradient(axes[-1], n_features, region_labels=region_labels)
    elif extra_col:
        axes[-1].axis('off')

    axes[0].invert_yaxis()  # feature 0 at top
    axes[0].set_ylabel('Neuron ID')
    axes[0].set_xlabel('W coefficient', fontsize=W_TITLE_FONTSIZE)

    if draw_heatmap:
        angles = _w_pairwise_angles(np.vstack(ws))
        heat_ax = fig.add_subplot(gs[1, 0:n])
        cmap = plt.cm.bwr_r.with_extremes(bad='white')
        im = heat_ax.pcolormesh(np.arange(n), np.arange(n), np.ma.masked_invalid(angles),
                                shading='nearest', cmap=cmap, vmin=0, vmax=180)
        for i in range(n):
            for j in range(n):
                if np.isnan(angles[i, j]):
                    continue
                heat_ax.text(j, i, f'{angles[i, j]:.0f}', ha='center', va='center',
                             fontsize=W_TITLE_FONTSIZE, color='black')
        heat_ax.set_xticks(np.arange(n))  # ticks only - the panels above are the labels
        heat_ax.set_xticklabels([])
        heat_ax.set_yticks(np.arange(n))
        heat_ax.set_yticklabels([f'W{i}' for i in range(n)], fontsize=W_TITLE_FONTSIZE)
        heat_ax.set_xlim(-0.2, n - 0.2)
        heat_ax.set_ylim(n - 0.5, -0.5)  # W0 at top, matching the panel order above
        heat_ax.tick_params(length=2)
        # as a ylabel, not a title - a title would land in the same narrow gap as the
        # stem panels' own x-axis label
        heat_ax.set_ylabel('pairwise angle', fontsize=W_TITLE_FONTSIZE)

        # the extra column's own cell on this row is free (the gradient sits in the row
        # above) - park a thin colorbar at its left edge rather than letting fig.colorbar
        # steal width from the heatmap, which has to keep matching the stem panels exactly
        cbar_holder = fig.add_subplot(gs[1, -1])
        cbar_holder.axis('off')
        cax = cbar_holder.inset_axes([0.0, 0.05, 0.25, 0.9])
        cbar = fig.colorbar(im, cax=cax, ticks=[0, 90, 180])
        cbar.ax.set_title('deg', fontsize=W_TITLE_FONTSIZE)
        cbar.ax.tick_params(labelsize=W_TITLE_FONTSIZE, length=2)
        cbar.outline.set_visible(False)

    if title:
        # wrapped to the figure's own width - a longer line would (with the bbox='tight'
        # saves below) widen the canvas around it, leaving the plot in a narrow strip
        fig.suptitle(textwrap.fill(title, width=max(20, int(fig_width * 72 / (0.62 * 8)))),
                     fontsize=8, y=0.995)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if save_path_svg is not None:
        fig.savefig(save_path_svg, bbox_inches='tight')
    plt.show()


# --------------------------------------------------- projecting activity onto a W


def get_trial_w_projection(s_id, t0_events, fr_x, which_intervals, w,
                            normalize=fitspl.FIT_L2, b=0.0):
    """Per-trial projection of session `s_id`'s activity (already sliced to one
    X_modality - see _x_modality_fr) onto a single fitted decoder weight vector `w`, at
    native 40ms resolution. Same per-trial windowing as get_session_data's `all_pos`
    (t0_events' own full_intrvl per trial, same modal-length trim) - but unlike
    all_pos/create_subplot, this is NOT resampled down to the SVM's own timebin count:
    since the SVM's sliding window advances exactly one native bin per timebin (see
    timeline_decoding in postproc_mea1k_ephys.py), timebin index and native-bin index
    are the same sequence (create_subplot's x_0 == that interval's n_bins_left - see
    get_TrialWiseT0Events40ms), so plotting by raw native-bin index already lines up
    with the existing offset x-axis.

    `normalize` is the preprocessing applied to each trial-bin's feature vector before
    projecting - one of plotlib.fit_samples' MODES, or a bool (see its `resolve_mode`).
    The default, FIT_L2, is fit parity: the same 3-native-bin centered mean the fit
    averages over, then per-sample L2, in that order. Before 2026-08-20 this was a bool
    that L2-normalized *single* bins, which is not what the classifier saw (normalizing
    the mean is not the mean of the normalized bins) - pass 'l2_bin' to get that old
    trace back for comparison, or FIT_RAW to skip only the L2 (which is what the fit
    itself does while postproc's Normalizer step is commented out - see fit_samples).
    Callers holding a whole `w_row` should resolve this via fit_samples.mode_for_w_row so
    the preprocessing follows the W's own provenance rather than this default.
    Consequence of fit parity: the first and last bin of every interval have no full
    window and come back NaN (see fit_samples.window_mean) - native-bin indices stay
    valid, so every consumer here just has to be NaN-aware.

    `b` is the fitted intercept, added to the dot product so the trace is the decoder's
    actual decision function and y=0 is its boundary. It defaults to 0.0 because older
    cached analytics never stored one - see fit_samples.bias_for_w_row, which is what
    callers should pass here.

    Returns two dicts, both keyed by interval: {interval: (n_trials, n_bins) array} and
    {interval: trial_id array} (same row order as the first) - the second is what a
    caller needs to split by covariate afterward (see _trial_covariate_levels), since
    the modal-length trim below can drop trials."""
    w = np.asarray(w)
    projections, trial_ids_by_interval = {}, {}
    for which_interval in which_intervals:
        trial_ids, trial_windows = [], []
        for trial_id, full_intrvl in t0_events.loc[s_id, which_interval].dropna().items():
            trial_ids.append(trial_id)
            trial_windows.append(fr_x[fr_x.index.overlaps(full_intrvl)].values)

        # Same modal-length trim as get_session_data's all_pos - defensive, since these
        # windows are built off a regular time grid and shouldn't actually vary, but
        # kept for parity/safety with the established pattern.
        lengths = pd.Series([len(x) for x in trial_windows])
        correct_length = lengths.value_counts().index[0]
        keep = [len(x) == correct_length for x in trial_windows]
        trial_windows = np.stack([x for x, k in zip(trial_windows, keep) if k])
        trial_ids_by_interval[which_interval] = np.array([t for t, k in zip(trial_ids, keep) if k])

        trial_windows = fitspl.prepare(trial_windows, normalize)
        projections[which_interval] = trial_windows @ w + b
    return projections, trial_ids_by_interval


def get_trial_behavior_trace(s_id, t0_events, behavior, which_intervals, column):
    """Per-trial native-resolution trace of one `behavior` column - used below for
    'frame_raw_500msMedian' (forward velocity, see plot_trial_velocity), but generic
    over any `behavior` column - aligned the same way as get_session_data's `all_pos`
    (same full_intrvl per trial, same trial_id pre-filter, same modal-length trim) -
    kept as its own small function rather than generalizing all_pos itself, since that
    one additionally resamples down to the SVM's timebin count for create_subplot/row 0,
    and this needs to stay at native resolution to match get_trial_w_projection's
    x-axis (see there for why that's safe to share directly). Returns two dicts, both
    keyed by interval, exactly like get_trial_w_projection: {interval: (n_trials,
    n_bins) array} and {interval: trial_id array}."""
    traces, trial_ids_by_interval = {}, {}
    for which_interval in which_intervals:
        trial_ids, trial_traces = [], []
        for trial_id, full_intrvl in t0_events.loc[s_id, which_interval].dropna().items():
            trial_behavior = behavior.xs(s_id, level='session_id')
            trial_behavior = trial_behavior[trial_behavior['trial_id'] == trial_id]
            dat = trial_behavior.loc[trial_behavior.index.overlaps(full_intrvl)]
            trial_ids.append(trial_id)
            trial_traces.append(dat[column].values)
        lengths = pd.Series([len(x) for x in trial_traces])
        correct_length = lengths.value_counts().index[0]
        keep = [len(x) == correct_length for x in trial_traces]
        traces[which_interval] = np.stack([x for x, k in zip(trial_traces, keep) if k])
        trial_ids_by_interval[which_interval] = np.array([t for t, k in zip(trial_ids, keep) if k])
    return traces, trial_ids_by_interval


# _trial_covariate_levels: now imported from plotlib.trial_labels (it moved there when
# var_plotting_lib's condition-split variance curves needed the same lookup). Same
# implementation, same private name here, so every call site below is unchanged.
from plotlib.trial_labels import trial_covariate_levels as _trial_covariate_levels  # noqa: E402


def _interval_x0(interval_x0s, which_interval, offset):
    """t=0 native-bin position for one interval, at the current x-axis `offset`.

    `interval_x0s` comes from plotlib.trial_traces.interval_x0_map, i.e. measured from
    t0_events. This used to be hardcoded as "3 for cue_entry, else 10", which matched
    the *stale* t0 table it was written against and silently disagreed with both the
    current table (R1_entry is 4 bins, not 10) and with the 3 sessions where even the
    stale table used 15. Shared by row 0 (velocity) and row 1 (projection) so their
    dashed t0 lines land on the same x position."""
    if not interval_x0s or which_interval not in interval_x0s:
        raise ValueError(f"no t0 geometry for {which_interval!r}; build it with "
                         f"plotlib.trial_traces.interval_x0_map(t0_events, which_intervals)")
    return interval_x0s[which_interval] + offset


def _nan_col_stat(stat_fn, arr):
    """`stat_fn` (np.nanmean/np.nanstd) down the trial axis of an (n_trials, n_bins)
    array, without numpy's RuntimeWarning for the all-NaN columns.

    Every interval's first and last bin is all-NaN under fit-parity preprocessing (no
    full fit window there - see fit_samples.window_mean), which is a structural feature
    of these arrays rather than a data problem: nan is exactly the wanted answer, so the
    'Mean of empty slice' warning is pure noise, once per group per interval per panel."""
    with np.errstate(invalid='ignore'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            return stat_fn(arr, axis=0)


def _shared_ylim(traces_by_interval, pad=0.1):
    """Y-limits from the full data range across every interval/trial (not just the
    mean line, so no variance band ever clips) - used to give row 0, row 1, and every
    covariate panel below it a fixed, comparable scale instead of each Axes
    autoscaling independently (which made panels hard to compare against each other,
    and let a single outlier trial blow out one panel's range). `pad` adds a fractional
    margin on each side."""
    all_vals = np.concatenate([arr.ravel() for arr in traces_by_interval.values() if len(arr)])
    # nan-aware: fit-parity projections carry NaN at each interval's first/last bin (no
    # full fit window there - see fit_samples.window_mean), and a single NaN would
    # otherwise poison both limits into nan and silently blank the whole axis.
    lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    margin = (hi - lo) * pad if hi > lo else 1.0
    return lo - margin, hi + margin


def plot_trial_velocity(ax, velocity_traces, which_intervals, interval_x0s, split=None):
    """Row 0: single-trial velocity traces (thin, low-alpha) plus a dashed t0 line per
    interval, at native-bin resolution (no resampling to the SVM's timebin count - see
    get_trial_behavior_trace/get_trial_w_projection for why that's safe to share with
    plot_w_projection's x-axis below; call with sharex=True axes). `velocity_traces`:
    {interval: (n_trials, n_bins) array}, e.g. from
    get_trial_behavior_trace(..., column='frame_raw_500msMedian'). `split`, if given, is
    (covariate_name, {interval: level_array}) - see _trial_covariate_levels/
    plot_w_projection - colors each single-trial line by its own covariate level
    (_covariate_color) instead of uniform black. Returns the final offset."""
    offset = 0
    for which_interval in which_intervals:
        trial_traces = velocity_traces.get(which_interval)
        if trial_traces is None or len(trial_traces) == 0:
            continue
        n_bins = trial_traces.shape[1]
        x = np.arange(n_bins) + offset
        if split is not None:
            covariate, levels_by_interval = split
            colors = [_covariate_color(covariate, lvl) for lvl in levels_by_interval[which_interval]]
        else:
            colors = ['k'] * len(trial_traces)
        for trial_trace, trial_color in zip(trial_traces, colors):
            ax.plot(x, trial_trace, color=trial_color, alpha=0.08, linewidth=0.8, zorder=4)
        ax.axvline(x=_interval_x0(interval_x0s, which_interval, offset), color='k', linestyle='--', alpha=0.4)
        offset += n_bins
        if which_interval != which_intervals[-1]:
            offset += 1  # gap matching the heatmap's inserted NaN separator row
    ax.set_ylabel('forward velocity')
    ax.tick_params(labelbottom=False)
    for sp in ax.spines.values():  # same clean look as the main figure's rows
        sp.set_visible(False)
    return offset


def plot_w_projection(ax, projections, which_intervals, split, interval_x0s, show_variance=True,
                       show_xticklabels=True,
                       corner_text=None, smooth=None, highlight_range=None, highlight_var_pct=None):
    """Split W-projection traces across `which_intervals`, sharing the same
    per-interval offset/gap/t0-dashed-line scheme as create_subplot/_plot_f1_lines (see
    _interval_x0) - no resampling needed here, see get_trial_w_projection.
    `projections`: {interval: (n_trials, n_bins) array}, e.g. straight from
    get_trial_w_projection. `show_variance` shades +/-0.5 SD across trials around each
    group's mean line (trial-to-trial spread, not a bootstrap CI - there's no bootstrap
    here). `split` is (covariate_name, {interval: level_array}) - see
    _trial_covariate_levels - most usefully the W's own predict_y_name (the dimension
    the subspace was actually fit to distinguish - see plot_w_projection_session): only
    the per-level group lines/bands are drawn, no separate aggregate/black line.
    `show_xticklabels` draws the same $t_<interval>$/'1s' xticks row 2's heatmap uses in
    the main figure (_interval_short_label, 25 native bins = 1000ms) - only the
    bottom-most Axes of a stacked figure should set this True (see
    plot_w_projection_session). `corner_text`, if given, is drawn bottom-right (which
    W this is - session/interval/timebin - now that dropping the aggregate line also
    dropped its old legend label). `smooth`, if given (e.g. 3), is a rolling-mean window
    (see _smooth) applied to each group's mean line and variance band before plotting -
    same semantics as _plot_f1_lines's `smooth`, computed separately per interval so
    smoothing never blends values across an interval boundary. `highlight_range`, if
    given, is (interval_name, bin_start, bin_end) (both bin indices inclusive, in the
    same native-bin/timebin index get_trial_w_projection relies on being identical) -
    marks that sub-window of that one interval directly on top of the actual projection
    trace, in HIGHLIGHT_COLOR (the same dark green every other figure marks a selected W
    with - see svm_plotting_lib): a translucent axvspan for a real window, or, for the
    single-bin (start == end) case a W is actually selected at, one vertical line - so a
    specific window of interest
    (e.g. the same window plot_w_variance_explained was run on) can be seen in the
    context of the real projected activity instead of only as a standalone number/bar
    chart. `highlight_var_pct`, if also given, annotates that span with '{:.1f}% var' -
    see plot_w_projection_session, which computes this the same way
    plot_w_variance_explained does (get_interval_window_activity + _pca_eigen), just
    without building a separate figure for it. Returns the final offset."""
    offset = 0
    xticks, xtick_labels = [], []
    labels_drawn = set()
    covariate, levels_by_interval = split
    for i, which_interval in enumerate(which_intervals):
        trial_projs = projections.get(which_interval)
        if trial_projs is None or len(trial_projs) == 0:
            continue
        n_bins = trial_projs.shape[1]
        x = np.arange(n_bins) + offset

        if highlight_range is not None and highlight_range[0] == which_interval:
            _, hl_start, hl_end = highlight_range
            if hl_start == hl_end:
                # a W is selected at one (interval, timebin), so that is what the usual
                # case marks - a span of zero width has nothing to shade, so draw the
                # single bin as a line instead
                ax.axvline(hl_start + offset, color=HIGHLIGHT_COLOR, alpha=0.5,
                           linewidth=HIGHLIGHT_BOX_LW, zorder=0)
            else:
                ax.axvspan(hl_start + offset, hl_end + offset, color=HIGHLIGHT_COLOR, alpha=0.12, zorder=0)
            if highlight_var_pct is not None:
                # y in axes-fraction space (via get_xaxis_transform) so this stays pinned
                # near the top regardless of the ylim the caller sets afterward.
                ax.text((hl_start + hl_end) / 2 + offset, 0.95, f'W: {highlight_var_pct:.1f}% var',
                        transform=ax.get_xaxis_transform(), color=HIGHLIGHT_COLOR, fontsize=6,
                        ha='center', va='top', zorder=15,
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.75))

        levels = levels_by_interval[which_interval]
        groups = [(trial_projs[levels == lvl], _covariate_color(covariate, lvl), _level_label(covariate, lvl))
                  for lvl in (0, 1) if (levels == lvl).sum() > 0]

        for group_projs, group_color, group_label in groups:
            # nanmean/nanstd, not mean/std: fit-parity projections are NaN at each
            # interval's first/last bin (see fit_samples.window_mean), and those bins are
            # meant to read as a gap in the line, not to blank the whole trace. An
            # all-NaN bin stays NaN here (_nan_col_stat suppresses the empty-slice noise).
            mean_proj = _smooth(_nan_col_stat(np.nanmean, group_projs), smooth)
            if show_variance:
                half_std = _smooth(_nan_col_stat(np.nanstd, group_projs), smooth) / 2
                ax.fill_between(x, mean_proj - half_std, mean_proj + half_std, color=group_color, alpha=0.2,
                                 edgecolor=None, zorder=1)
            ax.plot(x, mean_proj, color=group_color, linewidth=1.5, alpha=0.9, zorder=10,
                    label=group_label if group_label not in labels_drawn else None)
            labels_drawn.add(group_label)

        x_0 = _interval_x0(interval_x0s, which_interval, offset)
        ax.axvline(x=x_0, color='k', linestyle='--', alpha=0.4)
        xticks.extend([x_0, x_0 + 25])
        xtick_labels.extend([f'$t_{{{_interval_short_label(which_interval)}}}$', '1s'])

        offset += n_bins
        if i < len(which_intervals) - 1:
            offset += 1  # gap matching the heatmap's inserted NaN separator row
    ax.axhline(0, color='k', linestyle=':', alpha=0.3, linewidth=0.7)
    ax.set_ylabel('W projection')
    if show_xticklabels:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, fontsize=7)
    else:
        ax.set_xticks(xticks)
        ax.tick_params(labelbottom=False)
    if labels_drawn:
        ax.legend(fontsize=7)
    if corner_text:
        ax.text(0.95, 0.05, corner_text, transform=ax.transAxes, fontsize=7,
                verticalalignment='bottom', horizontalalignment='right')
    for sp in ax.spines.values():
        sp.set_visible(False)
    return offset


_NESTED_SPLIT_PARTNER = {'choice_R1': 'cue', 'cue': 'choice_R1'}


def _covariate_panel_specs(predict_y_name, covariate_names=None):
    """(covariate, filter_by) pairs for plot_w_projection_covariates to draw, one panel
    each - covariate_names defaults to every ALL_Y_PREDICTED_NAMES entry except
    predict_y_name, plus the two CORRECTNESS_SPECS names (same defaulting as
    plot_covariate_f1), each paired with filter_by=None (no restriction, the plain
    split-by-that-covariate panel). Additionally, only for the cue/choice_R1 pairing
    (_NESTED_SPLIT_PARTNER): two extra nested-split panels - the partner covariate
    split within each level of predict_y_name itself (e.g. for a choice_R1 W: a cue
    split restricted to choice_R1=0 trials only, and again restricted to choice_R1=1
    trials only) - same color as the plain partner panel (plot_w_projection_covariate_single's
    `filter_by` just edits the corner label to say which subset is shown)."""
    if covariate_names is None:
        covariate_names = [y for y in ALL_Y_PREDICTED_NAMES if y != predict_y_name]
        covariate_names += [spec['name'] for spec in CORRECTNESS_SPECS]
    specs = [(cov, None) for cov in covariate_names]
    partner = _NESTED_SPLIT_PARTNER.get(predict_y_name)
    if partner is not None:
        specs += [(partner, (predict_y_name, lvl)) for lvl in (0, 1)]
    return specs


def plot_w_projection_covariate_single(ax, projections, trial_ids_by_interval, t0_events, s_id, covariate,
                                        which_intervals, interval_x0s, show_variance=True, filter_by=None,
                                        show_xticklabels=False, smooth=None):
    """One small panel: `covariate`'s own per-level split lines/bands (no separate
    aggregate/black line) - same idea as plot_covariate_f1_single (per-level
    comparison), but without its OOB-alignment machinery (_stack_covariate_true/
    _stack_by_timebin) - this works from raw per-trial data joined straight to
    t0_events via _trial_covariate_levels, not reconstructed from bootstrap-aggregated
    predictions. `filter_by`, if given, is (filter_covariate, filter_level) - restrict
    to only the trials matching that level before splitting by `covariate` (the
    nested/conditional split - see _covariate_panel_specs); the corner label then names
    both, e.g. 'cue | choice_R1=stop'. `show_xticklabels`: see plot_w_projection - only
    the bottom-most panel of the grid should set this True. `smooth`: see
    plot_w_projection - same rolling-mean window applied to each level's mean line and
    variance band."""
    offset = 0
    xticks, xtick_labels = [], []
    for which_interval in which_intervals:
        trial_projs = projections.get(which_interval)
        if trial_projs is None or len(trial_projs) == 0:
            continue
        n_bins = trial_projs.shape[1]
        x = np.arange(n_bins) + offset
        trial_ids = trial_ids_by_interval[which_interval]

        if filter_by is not None:
            filter_covariate, filter_level = filter_by
            filter_levels = _trial_covariate_levels(t0_events, s_id, trial_ids, filter_covariate)
            keep = filter_levels == filter_level
            trial_projs, trial_ids = trial_projs[keep], trial_ids[keep]

        if len(trial_projs) > 0:
            levels = _trial_covariate_levels(t0_events, s_id, trial_ids, covariate)
            for lvl in (0, 1):
                group_projs = trial_projs[levels == lvl]
                if len(group_projs) == 0:
                    continue
                # nan-aware for the same reason as plot_w_projection's own group means
                group_mean = _smooth(_nan_col_stat(np.nanmean, group_projs), smooth)
                lvl_color = _covariate_color(covariate, lvl)
                if show_variance:
                    half_std = _smooth(_nan_col_stat(np.nanstd, group_projs), smooth) / 2
                    ax.fill_between(x, group_mean - half_std, group_mean + half_std, color=lvl_color, alpha=0.15,
                                     edgecolor=None, zorder=1)
                ax.plot(x, group_mean, color=lvl_color, linewidth=1.2, alpha=0.9, zorder=10)

        x_0 = _interval_x0(interval_x0s, which_interval, offset)
        ax.axvline(x=x_0, color='k', linestyle='--', alpha=0.4)
        xticks.extend([x_0, x_0 + 25])
        xtick_labels.extend([f'$t_{{{_interval_short_label(which_interval)}}}$', '1s'])
        offset += n_bins
        if which_interval != which_intervals[-1]:
            offset += 1  # gap matching the heatmap's inserted NaN separator row
    ax.axhline(0, color='k', linestyle=':', alpha=0.3, linewidth=0.5)
    panel_label = covariate
    if filter_by is not None:
        panel_label += f" | {filter_by[0]}={_level_label(filter_by[0], filter_by[1])}"
    ax.text(0.95, 0.05, panel_label, transform=ax.transAxes, fontsize=7,
            verticalalignment='bottom', horizontalalignment='right')
    ax.set_xticks(xticks)
    if show_xticklabels:
        ax.set_xticklabels(xtick_labels, fontsize=7)
    else:
        ax.tick_params(labelbottom=False)
    ax.tick_params(labelleft=False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_w_projection_covariates(axes, projections, trial_ids_by_interval, t0_events, s_id, which_intervals,
                                  predict_y_name, interval_x0s, covariate_names=None, show_variance=True,
                                  smooth=None):
    """One small subplot per (covariate, filter_by) pair from _covariate_panel_specs,
    stacked below the main split row - `axes` is a sequence of Axes, one per pair. The
    last Axes gets show_xticklabels=True (see plot_w_projection_covariate_single) -
    it's the bottom-most row of the whole figure whenever this grid is shown. `smooth`:
    see plot_w_projection, passed through to every panel."""
    specs = _covariate_panel_specs(predict_y_name, covariate_names)
    for i, (ax, (covariate, filter_by)) in enumerate(zip(axes, specs)):
        plot_w_projection_covariate_single(ax, projections, trial_ids_by_interval, t0_events, s_id, covariate,
                                            which_intervals, interval_x0s, show_variance=show_variance,
                                            filter_by=filter_by,
                                            show_xticklabels=(i == len(specs) - 1), smooth=smooth)


@plt.rc_context(W_FIGURE_RC)
def plot_w_projection_session(s_id, w_row, t0_events, behavior, session_names, which_intervals,
                               covariate_names=None, show_covariate_plot=True, split_covariate=None,
                               projection_ylim=(-1.25, 1.25), normalize=fitspl.AUTO, smooth=None,
                               highlight_range=None, save_dir=None, save_suffix='',
                               ith_to_dt_session=None):
    """Full per-session W-projection figure projecting one fixed, hand-picked decoder
    (`w_row` - a single row from svm_plotting_lib.resolve_w_selections' frame,
    supplying X_modality/predict_y_name/w/interval_name/timebin) onto `s_id`'s own
    activity - `s_id` is deliberately independent of `w_row['session_id']` (the session
    that W happened to be fit on), since the whole point of this figure is to see how
    one fixed subspace's projection drifts/holds up across *other* sessions' activity,
    not to recompute each session's own self-matched W against itself. Loop this over
    every session the same way plot_session_analysis is in the main per-session loop
    above, keeping `w_row` constant across the loop, to see that one hand-picked W's
    projection drift/consistency across sessions. Row 0: single-trial velocity, colored
    by that trial's own cue level (_covariate_color) - the same behavioral grounding
    create_subplot's own row 0 gives the main figure. Row 1: the W's own predict_y_name
    split (see _trial_covariate_levels/plot_w_projection's `split`) - the dimension this
    specific subspace was actually fit to distinguish, so it's shown regardless of
    show_covariate_plot, not just as one of the "other" covariates below. Pass
    `split_covariate` to split row 1 by something else instead: projecting a fixed W into
    another session's activity (or over intervals it was never fit on) is just a
    timeseries, so the fit target stops being privileged - whichever covariate row 1
    ends up showing is the one the grid below excludes and nests its partner splits in. No plain
    aggregate/black line anywhere in this figure - every row shows only the per-group
    average (see plot_w_projection/plot_w_projection_covariate_single). Rows 2+, when
    show_covariate_plot (default): one small panel per remaining covariate, plus (only
    for a choice_R1 or cue target - see _NESTED_SPLIT_PARTNER) two extra nested-split
    panels, stacked below row 1 exactly like plot_session_analysis's
    show_covariate_plot - see plot_w_projection_covariates/_covariate_panel_specs for
    the defaulting scheme (predict_y_name is excluded there, since row 1 already
    covers it) and _trial_covariate_levels for how this differs from plot_covariate_f1
    (no OOB-alignment machinery needed, since this works from raw per-trial data joined
    straight to t0_events). The bottom-most Axes of the whole figure gets the
    $t_<interval>$/'1s' xtick labels (see plot_w_projection's show_xticklabels), the
    same annotation the main figure's row-2 heatmap carries. `normalize` (default 'auto'
    here: resolved off `w_row`'s own `normalizer` column via fit_samples.mode_for_w_row,
    so the trace follows how that W was actually fitted) applies the classifier's own
    sample preprocessing - the 3-native-bin mean and, when the fit used it, per-sample
    L2. The L2 part matters for this figure specifically: without it the projection's
    magnitude scales with each session's raw feature-vector norm (neuron count/yield,
    overall activity level), which is session-dependent and unrelated to decoding signal
    strength - this is why one session can look much larger than the others for no
    interpretable reason. Normalized samples fix that at the source, which is what makes the fixed
    `projection_ylim` ((-1.25, 1.25) by default, shared by row 1 and every covariate
    panel, or None for _shared_ylim's per-session empirical range) comparable from
    session to session in the first place (a FIT_RAW W, fitted without that step, will
    generally need its own limits). That default was (-0.8, 0.8) before fit parity: the
    fit window's mean averages out per-bin noise before the L2 step, so the same W's
    group-mean amplitude comes out 1.0-1.5x larger (measured across one animal's
    sessions on one choice_R1 W: peak group mean 0.42-0.86 -> 0.57-1.18), and 0.8 clipped
    the strongest sessions - it was in fact already clipping two of them beforehand.
    An exact +/-||w|| bound is also
    available via Cauchy-Schwarz once vectors are unit-norm, but deliberately not used
    for the axis limits - in this many dimensions a near-random unit vector's dot
    product with w concentrates far below ||w||, so that bound is essentially never
    approached and makes the actual line look flat on a mostly-empty axis. The fitted
    intercept is added to the projection too (fit_samples.bias_for_w_row), which is what
    makes the dotted y=0 line the decoder's actual boundary rather than an arbitrary
    offset - but only for analytics that store one: older tables have no `b` at all
    (postproc's `_bootstrap_and_aggregate` computed the per-bootstrap intercepts and
    then dropped them before returning), so they fall back to b=0 with a printed notice
    and their zero line stays uninterpretable as a threshold. Every comparison this
    figure actually draws (level vs level, time course, spread) is unaffected either
    way, since b is one constant offset shared by every trial and bin. `smooth`, if
    given (e.g. 3), is a rolling-mean window (see _smooth) applied to row 1 and every
    covariate panel's mean line/variance band - same semantics as the main figure's
    `smooth` (plot_session_analysis/_plot_f1_lines). Row 0's single-trial velocity
    lines are left unsmoothed, same as create_subplot's own row 0. If `w_row` carries a
    `name` (the highlight_selection's own optional short human label - see
    _highlighted_rows), it's prepended (wrapped, since it's free-form text) to both row
    1's corner_text and the figure suptitle, so a specific hand-picked W stays
    identifiable by name across every session it gets projected into. `highlight_range`,
    if given, is (interval_name, bin_start, bin_end) (both inclusive) - shaded on row 1
    via plot_w_projection's own `highlight_range` (see there), and additionally used
    here to compute what % of that window's own pooled-trial variance w itself explains
    (get_interval_window_activity + _pca_eigen, same computation
    plot_w_variance_explained does standalone) and annotate it directly on the shaded
    span - lets a specific window's W-variance-explained number be read in the context
    of the actual projected-activity trace, instead of only as a separate bar chart.
    Saves save_dir/session_{s_id}_S{s_num}_wproj{save_suffix}.{png,svg} if save_dir is
    given (see plotlib.export.save_figure); the PNG path is what's returned. Returns
    that path (or None). `save_suffix` matters as soon as a loop varies the W rather than
    the session - several W's projected into the same session all write the same filename
    without it, and only the last one survives."""
    x_modality, predict_y_name, w = w_row['X_modality'], w_row['predict_y_name'], w_row['w']
    session_name = next(sn for sn in session_names if sn.startswith(s_id))

    fr_z_raw = analytics.get_analytics('FiringRate40msZ', session_names=[session_name]).xs(s_id, level='session_id')
    fr_x = session_fr_for_modality(fr_z_raw, x_modality)

    # One resolution of the preprocessing per figure, shared by the projection rows and
    # the highlight's variance number below - the two disagreeing is exactly the bug this
    # replaced. `b` is 0.0 for any analytic cached before the intercept was stored.
    mode = fitspl.mode_for_w_row(w_row, normalize)
    b = fitspl.bias_for_w_row(w_row)

    highlight_var_pct = None
    if highlight_range is not None:
        hl_interval, hl_start, hl_end = highlight_range
        pooled = get_interval_window_activity(s_id, t0_events, fr_x, hl_interval, (hl_start, hl_end),
                                               normalize=mode)
        _, _, hl_total_var = pca_eigen(pooled)
        hl_centered = pooled - pooled.mean(axis=0, keepdims=True)
        hl_w_unit = np.asarray(w) / np.linalg.norm(w)
        highlight_var_pct = np.var(hl_centered @ hl_w_unit, ddof=1) / hl_total_var * 100

    projections, proj_trial_ids = get_trial_w_projection(s_id, t0_events, fr_x, which_intervals, w,
                                                           normalize=mode, b=b)
    velocity_traces, vel_trial_ids = get_trial_behavior_trace(s_id, t0_events, behavior, which_intervals,
                                                               'frame_raw_500msMedian')

    # what row 1 splits by: the W's own predict_y_name unless the caller names another
    # covariate - once a W is just a fixed direction being projected into (possibly
    # another session's) activity, the dimension it happened to be fit on is no longer
    # necessarily the one worth splitting the projection by. Whatever row 1 shows is
    # what the covariate grid below leaves out (and nests its partner splits within).
    row1_covariate = split_covariate or predict_y_name
    target_split = (row1_covariate, {iv: _trial_covariate_levels(t0_events, s_id, ids, row1_covariate)
                                      for iv, ids in proj_trial_ids.items()})
    velocity_split = ('cue', {iv: _trial_covariate_levels(t0_events, s_id, ids, 'cue')
                               for iv, ids in vel_trial_ids.items()})

    covariate_specs = _covariate_panel_specs(row1_covariate, covariate_names)
    n_covariates = len(covariate_specs) if show_covariate_plot else 0

    fig, axes = plt.subplots(2 + n_covariates, 1, figsize=(max(4, 1.4 * len(which_intervals) + 1),
                                                            3.6 + 0.8 * n_covariates),
                              height_ratios=[1, 1.3] + [0.8] * n_covariates, sharex=True, dpi=150)
    # measured from this session's own t0_events rows - see plotlib.trial_traces
    interval_x0s = interval_x0_map(t0_events, which_intervals, s_id=s_id, warn=False)
    plot_trial_velocity(axes[0], velocity_traces, which_intervals, interval_x0s, split=velocity_split)
    axes[0].set_ylim(_shared_ylim(velocity_traces))
    # origin_label names which session this W was actually fit on (w_row['session_id']),
    # distinct from s_id/s_label below (the session whose activity it's being projected
    # onto here) - the two now routinely differ, so the corner text has to carry both.
    origin_num = (ith_to_dt_session or {}).get(w_row['session_id'], "Unknown")
    origin_label = f'S{origin_num:02d}' if isinstance(origin_num, int) else f'S{origin_num}'
    # w_row['name']: the highlight_selection's own optional short human label (see
    # _highlighted_rows) - wrapped since, unlike the other corner-text fields, it's
    # free-form text that can run longer than fits on one line in the corner.
    w_name = w_row.get('name')
    name_line = f"{textwrap.fill(str(w_name), width=24)}\n" if isinstance(w_name, str) and w_name else ''
    plot_w_projection(axes[1], projections, which_intervals, target_split, interval_x0s,
                       show_xticklabels=(n_covariates == 0),
                       corner_text=f"{name_line}{predict_y_name} W @ {w_row['interval_name']} t={w_row['timebin']} "
                                   f"(from {origin_label})", smooth=smooth,
                       highlight_range=highlight_range, highlight_var_pct=highlight_var_pct)
    if show_covariate_plot:
        plot_w_projection_covariates(axes[2:], projections, proj_trial_ids, t0_events, s_id, which_intervals,
                                      row1_covariate, interval_x0s, covariate_names=covariate_names,
                                      smooth=smooth)
    # One fixed y-range for row 1 and every covariate panel: they all show the same
    # quantity, and a fixed range is also what makes one W's projection comparable across
    # the sessions it gets projected into (an autoscaled axis hides exactly the drift
    # this figure exists to show). projection_ylim=None falls back to the empirical range
    # of this session's own data - fine for a one-off, not for a cross-session loop.
    # (the L2 half of the preprocessing is what makes a fixed range viable at all - a
    # FIT_RAW W's projection scales with the session's own norms, see the docstring.)
    for ax in axes[1:]:
        ax.set_ylim(projection_ylim if projection_ylim is not None else _shared_ylim(projections))

    # same S<n> numbering as the main figure's title (plot_session_analysis) - see
    # the ith_to_dt_session mapping built by plotlib.session_data.get_session_names.
    s_num = (ith_to_dt_session or {}).get(s_id, "Unknown")
    s_label = f'S{s_num:02d}' if isinstance(s_num, int) else f'S{s_num}'
    title = f'{s_label}: activity projected onto {origin_label}\'s W ({x_modality})'
    if isinstance(w_name, str) and w_name:
        title = f"'{w_name}'\n{title}"
    fig.suptitle(title)
    plt.tight_layout()

    # Writes BOTH formats and returns the PNG path. This used to save only .svg while its
    # docstring promised .png, and callers feed the returned path straight to
    # build_canvas - which is PIL, i.e. PNG-only - so the canvas cell died with
    # UnidentifiedImageError on the .svg it got handed. The SVG is still written; it is
    # what gets hand-assembled in Inkscape.
    save_path = save_figure(fig, save_dir, f'session_{s_id}_S{s_num}_wproj{save_suffix}').get('png')
    plt.show()
    return save_path


# ------------------------------------ one W, seen through every trial split that matters


# Which trial splits the projection figure draws, one row each, top to bottom. A string is
# a plain two-level split (one line per level, in that label's own colour); a TUPLE is the
# JOINT split of its covariates - one line per cell of the 2x2, drawn bicolour-dashed in
# both covariates' colours (COMBO_DASH), the same encoding var_plotting_lib and
# neuro_plotting_lib give a joint condition, so a "C1 & skip" line is the same line in all
# three families of figures.
#
# The order is the argument the figure makes: the two labels the W could have been fit on,
# then their interaction (does the choice split hold within a cue?), then whether the
# animal was right - which is neither of the two labels but a function of both, and is the
# split that says whether the direction carries the task rather than one of its variables.
PROJECTION_SPLITS = ('cue', 'choice_R1', ('cue', 'choice_R1'), 'correct_R1')

PROJECTION_ROW_HEIGHT = 0.8      # inches per split row
PROJECTION_INTERVAL_WIDTH = 1.4  # inches per interval - the CCGP/alignment panels' width,
                                 #   so a projection panel tiles under them at one scale
PROJECTION_YLABEL_WIDTH = 1.0    # what the y label + tick numbers cost (see
                                 #   plot_session_analysis's own note on show_ylabel)


def _split_covariates(split):
    """The covariate name(s) of one PROJECTION_SPLITS entry, always as a tuple."""
    return (split,) if isinstance(split, str) else tuple(split)


def _split_groups(split):
    """The groups one split draws: [(levels, colors, label), ...], in drawing order.

    `levels` is one 0/1 level per covariate (a 1-tuple for a plain split, a 2-tuple for a
    joint one), `colors` the matching per-covariate colours - a single colour draws a plain
    line, two draw one bicolour-dashed line (see PROJECTION_SPLITS)."""
    covariates = _split_covariates(split)
    groups = []
    for levels in itertools.product(*[(0, 1)] * len(covariates)):
        colors = tuple(_covariate_color(cov, lvl) for cov, lvl in zip(covariates, levels))
        label = ' & '.join(_level_label(cov, lvl) for cov, lvl in zip(covariates, levels))
        groups.append((levels, colors, label))
    return groups


def _split_key_segments(split):
    """The split's row key as (text, textprops) segments for _draw_text_rows: each
    covariate's two class names in their own colours, joined by 'x' for a joint split -
    'C1/C2 x skip/stop'. That names both the split and its colours in one short row, which
    is what replaces a legend box on a panel this narrow (4 groups would need 4 entries)."""
    small = dict(fontsize=CORNER_FONTSIZE, color='black')
    bold = dict(fontsize=CORNER_FONTSIZE + 1, fontweight='bold')
    segments = []
    for i, covariate in enumerate(_split_covariates(split)):
        if i:
            segments.append((' x ', small))
        segments += [(_level_label(covariate, 0), dict(bold, color=_covariate_color(covariate, 0))),
                     ('/', small),
                     (_level_label(covariate, 1), dict(bold, color=_covariate_color(covariate, 1)))]
    return segments


def split_projection_curves(projections, trial_ids_by_interval, t0_events, s_id, split,
                            which_intervals, smooth=None, spread='sd'):
    """The per-group mean +/- band projection curves for one split - the numbers
    plot_w_projection_splits draws, computed without touching an Axes so the figure can
    take one pass to find a shared y range and a second to draw.

    `projections`/`trial_ids_by_interval`: straight from get_trial_w_projection. Returns
    one dict per group of _split_groups (same order), with keys `levels`, `colors`,
    `label`, `n` ({interval: n_trials}) and `curves` ({interval: (mean, band)}).
    nan-aware throughout: fit-parity projections are all-NaN at each interval's first and
    last bin (fit_samples.window_mean), which is a gap in the line, not a data problem.

    `spread` picks what the band is:

      * `'sd'` (default) - half the trial-to-trial SD, i.e. how spread the single trials
        are. What the W-projection panels show.
      * `'sem'` - the standard error of the group MEAN (SD / sqrt(n) per bin). Use it when
        the quantity is skewed and its SD is therefore not a sensible half-width: a squared
        z, for instance, has an SD that scales with its own mean, so a +/-SD band around it
        runs several times the mean and reaches below zero - a range the quantity cannot
        occupy - and one such group then sets the y range for every panel it shares.
      * `None` - no band (a zero-width one), for the mean lines alone.
    """
    groups = []
    for levels, colors, label in _split_groups(split):
        curves, counts = {}, {}
        for which_interval in which_intervals:
            trial_projs = projections.get(which_interval)
            if trial_projs is None or len(trial_projs) == 0:
                continue
            keep = np.ones(len(trial_projs), dtype=bool)
            for covariate, level in zip(_split_covariates(split), levels):
                trial_levels = _trial_covariate_levels(t0_events, s_id,
                                                       trial_ids_by_interval[which_interval], covariate)
                keep &= trial_levels == level
            counts[which_interval] = int(keep.sum())
            if not keep.any():
                continue
            group_projs = trial_projs[keep]
            group_sd = _nan_col_stat(np.nanstd, group_projs)
            if spread is None:
                band = np.zeros_like(group_sd)
            elif spread == 'sd':
                band = group_sd / 2
            elif spread == 'sem':
                # per bin, not per group: the all-NaN first/last bin has n=0, and clamping
                # it to 1 leaves the band there NaN-driven rather than dividing by zero
                n_bin = np.sum(np.isfinite(group_projs), axis=0)
                band = group_sd / np.sqrt(np.maximum(n_bin, 1))
            else:
                raise ValueError(f"spread must be 'sd', 'sem' or None, got {spread!r}")
            curves[which_interval] = (_smooth(_nan_col_stat(np.nanmean, group_projs), smooth),
                                      _smooth(band, smooth))
        groups.append(dict(levels=levels, colors=colors, label=label, n=counts, curves=curves))
    return groups


def _curves_ylim(groups_by_split, show_variance=True, pad=0.15):
    """Y-limits covering every group's drawn line (and band, with show_variance) across
    every split row - NOT the full single-trial range _shared_ylim takes.

    A raw (noNorm) W's single-trial projections spread several times wider than the group
    means they average into, so scaling to them flattens every line in the figure into the
    middle of an empty axis. What this figure compares is the group means, so those are
    what set the scale; one range shared by all rows keeps a split's separation readable
    against the others' instead of each row autoscaling its own noise up to full height."""
    los, his = [], []
    for groups in groups_by_split:
        for group in groups:
            for mean, half_std in group['curves'].values():
                band = half_std if show_variance else np.zeros_like(mean)
                los.append(np.nanmin(mean - band))
                his.append(np.nanmax(mean + band))
    if not los or not np.isfinite(los).any():
        return None
    lo, hi = float(np.nanmin(los)), float(np.nanmax(his))
    margin = (hi - lo) * pad if hi > lo else 1.0
    return lo - margin, hi + margin


def plot_split_projection(ax, groups, which_intervals, interval_x0s, show_variance=True,
                           show_xticklabels=False, split=None, show_key=True):
    """One split's row: every group of `groups` (from split_projection_curves) as a line
    over the same per-interval offset/gap/t0 scheme every other row in this library uses
    (see _interval_x0), a group with two colours drawn twice with complementary COMBO_DASH
    phases so it reads as one alternating bicolour line rather than two curves.

    `split`, if given, adds the coloured row key (_split_key_segments) in the upper left.
    `show_xticklabels`: only the bottom-most Axes of the stack should set this True (see
    plot_w_projection). Returns the final offset."""
    offset = 0
    xticks, xtick_labels = [], []
    for i, which_interval in enumerate(which_intervals):
        n_bins = None
        for group in groups:
            curve = group['curves'].get(which_interval)
            if curve is None:
                continue
            mean, half_std = curve
            n_bins = len(mean)
            x = np.arange(n_bins) + offset
            if show_variance:
                ax.fill_between(x, mean - half_std, mean + half_std, color=group['colors'][0],
                                alpha=0.15, edgecolor=None, zorder=1)
            if len(group['colors']) == 1:
                ax.plot(x, mean, color=group['colors'][0], linewidth=1.4, alpha=0.9, zorder=10)
            else:
                for color, dashes in zip(group['colors'], COMBO_DASH):
                    ax.plot(x, mean, color=color, linewidth=COMBO_LINEWIDTH, alpha=0.9,
                            zorder=10, linestyle=dashes)
        if n_bins is None:   # nothing drawn for this interval at all
            continue
        x_0 = _interval_x0(interval_x0s, which_interval, offset)
        ax.axvline(x=x_0, color='k', linestyle='--', alpha=0.4)
        xticks.append(x_0)
        xtick_labels.append(f'$t_{{{_interval_short_label(which_interval)}}}$')
        # ...and the '1s' mark only where it still lands well inside its own block. These
        # intervals are ~30 native bins with t0 near the start, so a non-final block's 1s
        # comes out within a bin or two of the NEXT block's own t0 label (one separator bin
        # apart) and the two overprint each other - the scale mark is worth less than the
        # collision costs, and the final block still carries one for the whole row.
        is_last = i == len(which_intervals) - 1
        if x_0 + 25 <= offset + n_bins - (0 if is_last else 6):
            xticks.append(x_0 + 25)
            xtick_labels.append('1s')
        offset += n_bins
        if i < len(which_intervals) - 1:
            offset += 1  # gap matching the heatmap's inserted NaN separator row

    ax.axhline(0, color='k', linestyle=':', alpha=0.3, linewidth=0.7)
    ax.set_xticks(xticks)
    if show_xticklabels:
        ax.set_xticklabels(xtick_labels, fontsize=7)
    else:
        # marks off too, not just the numbers: these rows draw no spines, so a tick mark
        # on a row that isn't the bottom one is a stray dash hanging under the curves
        ax.tick_params(bottom=False, labelbottom=False)
    if show_key and split is not None:
        # in_layout=False for the same reason the alignment panel's key needs it: a key
        # wider than a 1.4in panel otherwise makes tight_layout shrink the AXES to fit the
        # text rather than let it overflow (see _draw_text_rows).
        _draw_text_rows(ax, [_split_key_segments(split)], (0.02, 0.98), ha='left', va='top',
                        in_layout=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return offset


@plt.rc_context(W_FIGURE_RC)
def draw_split_rows(groups_by_split, splits, t0_events, s_id, which_intervals, ylim=None,
                     show_variance=True, show_ylabel=True, ylabel='W projection', subtitle=None,
                     show_key=True, ith_to_dt_session=None, save_dir=None, stem=None):
    """The figure half of plot_w_projection_splits: one `splits` row per entry of
    `groups_by_split` (each from split_projection_curves), stacked on one shared x axis and
    one shared y range, under the big bold S<n> header.

    Split out from the projection figure so anything else that can produce per-trial values
    on this same t0-aligned grid gets the identical panel for free - a single unit's rate,
    say - without either copying the layout or being forced through a `w_row` it does not
    have. `subtitle`, if given, is the small descriptive line above the top row (already
    wrapped by this function); `ylabel` names what the rows show. Returns
    ([png_path], [svg_path]), the nesting build_canvas expects.
    """
    fig_width = PROJECTION_INTERVAL_WIDTH * len(which_intervals) + (PROJECTION_YLABEL_WIDTH if show_ylabel else 0)
    fig, axes = plt.subplots(len(splits), 1, sharex=True, dpi=150,
                             figsize=(fig_width, PROJECTION_ROW_HEIGHT * len(splits) + 0.6))
                             # +0.6in: the ticks/labels at the bottom and the 20pt S<n>
                             # header at the top, neither of which is a row's own height
    axes = np.atleast_1d(axes)
    interval_x0s = interval_x0_map(t0_events, which_intervals, s_id=s_id, warn=False)
    for i, (ax, split, groups) in enumerate(zip(axes, splits, groups_by_split)):
        plot_split_projection(ax, groups, which_intervals, interval_x0s, show_variance=show_variance,
                              show_xticklabels=(i == len(splits) - 1), split=split, show_key=show_key)
        if ylim is not None:
            ax.set_ylim(ylim)
        if show_ylabel:
            ax.set_ylabel(ylabel if i == len(splits) // 2 else '')
        else:
            ax.set_yticks([])

    s_num = (ith_to_dt_session or {}).get(s_id, "Unknown")
    s_label = f'S{s_num:02d}' if isinstance(s_num, int) else f'S{s_num}'
    # The big bold S<n>, in the same SUPTITLE_KW every other per-session panel in this tree
    # uses and, like them, drawn unconditionally: on an exported panel it is the only thing
    # saying which session this column is, and it has to survive the poster reduction. It
    # names the session whose ACTIVITY this is, which for a fixed W projected across the
    # sequence is not the session the W came from - the subtitle below says that.
    fig.suptitle(s_label, y=.99, **SUPTITLE_KW)
    if subtitle:
        # ...as the top row's axes title rather than a second suptitle line, which would
        # have to share the 20pt size. No leading "S13:" - the header just said it.
        # Wrapped to the figure's own width (~0.62 * fontsize per char), the same measure
        # plot_W_list uses: this panel is 1.4in per interval, i.e. narrower than one line of
        # most of these sentences, and an unwrapped title is simply cut off at both ends
        # rather than widening the figure.
        wrap = max(20, int(fig_width * 72 / (0.62 * 7)))
        axes[0].set_title('\n'.join(textwrap.fill(line, width=wrap) for line in subtitle.split('\n')),
                          fontsize=7)
    plt.tight_layout()

    saved = save_figure(fig, save_dir, stem) if stem else {}
    plt.show()
    return ([saved['png']] if saved.get('png') else []), ([saved['svg']] if saved.get('svg') else [])


@plt.rc_context(W_FIGURE_RC)
def plot_w_projection_splits(s_id, w_row, t0_events, session_names, which_intervals,
                              splits=PROJECTION_SPLITS, normalize=fitspl.AUTO, smooth=None,
                              projection_ylim=None, show_variance=True, show_ylabel=True,
                              show_title=True, show_key=True, fr_x=None,
                              ith_to_dt_session=None, save_dir=None, save_suffix='',
                              verbose=True):
    """One hand-picked W's projection, one row per trial split - the poster form of
    plot_w_projection_session.

    Same quantity as that figure (session `s_id`'s per-trial activity dotted onto
    `w_row`'s weight vector at native 40ms resolution - see get_trial_w_projection), and
    the same x axis every other t0-aligned figure in this tree uses, but stripped to the
    comparison itself:

      * **no kinematics row.** The velocity traces are the grounding for a figure read on
        its own; on the poster that job is already done, once, by the condition panel.
      * **no covariate grid.** Rows 2+ of plot_w_projection_session are every remaining
        label, each nested split its own panel, which is a screen to go looking in. Here
        the rows are named up front (`splits`, default PROJECTION_SPLITS): cue, choice,
        their interaction, and correctness.
      * **no privileged fit target.** Which label the W happens to have been fit on gets
        no row of its own; it is simply one of the splits, and the corner text still says
        which it was.

    Every row is the same quantity on one shared y range (`projection_ylim`, default: the
    range of every group's own drawn line/band across every row - see _curves_ylim), so
    the rows are comparable by eye, which is the whole point of stacking them. The dotted
    y=0 line is the direction's neutral point; it is the classifier's decision boundary
    only for a table that stored an intercept (fit_samples.bias_for_w_row - the _noNorm
    CCGP tables do not, and say so once).

    `normalize` defaults to AUTO, i.e. the preprocessing follows this W's own `normalizer`
    column (fit_samples.mode_for_w_row). For a **noNorm** fit that resolves to FIT_RAW -
    the 3-native-bin fit window mean, no per-sample L2 - which is fit parity for those
    tables and is what makes these betas' own projection the thing being drawn. It also
    means the y scale carries the session's own feature-vector norm, so it is comparable
    across the ROWS of one panel (same trials, same W) but not across sessions the way an
    L2-fit W's fixed +/-1.25 range is; leave projection_ylim=None and read each session's
    panel on its own scale.

    The big bold **S<n>** header is always drawn (SUPTITLE_KW, as on every other
    per-session panel here) and names the session whose ACTIVITY this is. `show_title` is
    the separate, small descriptive line above the top row - which W this is and which
    session it came from - off by default because the poster states it once beside the
    panel; turn it on whenever a fixed W is being projected across sessions, since then
    the header and the W's origin are two different sessions.

    `fr_x`, if given, is this session's firing rate already sliced to `w_row`'s X_modality
    (session_fr_for_modality) - pass it to skip the per-figure NAS read when looping (see
    the module's KNOWN DEBT note). Saves save_dir/session_{s_id}_S{n}_wsplits{save_suffix}
    .{png,svg}; returns ([png_path], [svg_path]) - the nesting build_canvas expects, one
    panel per session."""
    x_modality, w = w_row['X_modality'], w_row['w']
    if fr_x is None:
        session_name = next(sn for sn in session_names if sn.startswith(s_id))
        fr_z_raw = analytics.get_analytics('FiringRate40msZ',
                                           session_names=[session_name]).xs(s_id, level='session_id')
        fr_x = session_fr_for_modality(fr_z_raw, x_modality)

    mode = fitspl.mode_for_w_row(w_row, normalize)
    b = fitspl.bias_for_w_row(w_row)
    projections, proj_trial_ids = get_trial_w_projection(s_id, t0_events, fr_x, which_intervals, w,
                                                          normalize=mode, b=b)

    # every row's numbers first, so the shared y range can be measured before anything is
    # drawn (an Axes-by-Axes autoscale is what made these rows incomparable)
    groups_by_split = [split_projection_curves(projections, proj_trial_ids, t0_events, s_id, split,
                                               which_intervals, smooth=smooth)
                       for split in splits]
    ylim = projection_ylim if projection_ylim is not None else _curves_ylim(groups_by_split, show_variance)

    origin_num = (ith_to_dt_session or {}).get(w_row.get('session_id'), "Unknown")
    origin_label = f'S{origin_num:02d}' if isinstance(origin_num, int) else f'S{origin_num}'
    w_name = w_row.get('name')
    subtitle = (f"activity projected onto {origin_label}'s "
                f"{w_row['predict_y_name']} W @ {_interval_short_label(str(w_row['interval_name']))} "
                f"t={w_row['timebin']} ({x_modality})")
    if isinstance(w_name, str) and w_name:
        subtitle = f"'{w_name}'\n{subtitle}"

    s_num = (ith_to_dt_session or {}).get(s_id, "Unknown")
    paths = draw_split_rows(groups_by_split, splits, t0_events, s_id, which_intervals,
                            ylim=ylim, show_variance=show_variance, show_ylabel=show_ylabel,
                            ylabel='W projection', subtitle=subtitle if show_title else None,
                            show_key=show_key, ith_to_dt_session=ith_to_dt_session,
                            save_dir=save_dir, stem=f'session_{s_id}_S{s_num}_wsplits{save_suffix}')

    if verbose:
        s_label = f'S{s_num:02d}' if isinstance(s_num, int) else f'S{s_num}'
        counts = ', '.join(f"{g['label']}={sum(g['n'].values())}" for g in groups_by_split[-1])
        print(f"{s_label} <- {origin_label} {w_name or w_row['predict_y_name']}: "
              f"{sum(len(ids) for ids in proj_trial_ids.values())} trial-windows, "
              f"{splits[-1] if isinstance(splits[-1], str) else ' x '.join(splits[-1])} {counts}")
    return paths



# ------------------------------------------- how much variance does that W explain


def get_interval_window_activity(s_id, t0_events, fr_x, interval_name, timebin_range,
                                  normalize=fitspl.FIT_L2):
    """Pooled (n_samples, n_features) activity matrix for one
    session/interval/timebin_range - same per-trial windowing as get_trial_w_projection
    (t0_events' own full_intrvl per trial, same modal-length trim), but instead of
    projecting onto a W, keeps the full feature vectors and only slices down to
    `timebin_range` (a (start, end) pair, both inclusive - the same native-bin/timebin
    index get_trial_w_projection relies on being identical, see its own docstring)
    before flattening trials and timebins together into one sample axis. Feeds
    plot_w_variance_explained's pooled PCA/variance-explained-by-W computation, which
    (per how this is called) treats trial-to-trial and within-window temporal
    variability as one combined source of variance rather than analyzing them
    separately per timebin.

    `normalize`: same preprocessing argument as get_trial_w_projection, applied *before*
    the window slice so each bin's fit window is built from its true neighbours rather
    than being cut off at the window's edge. Until 2026-08-20 this function did no
    preprocessing at all, which meant plot_w_variance_explained's '% var' - annotated
    straight onto the (normalized) projection row by plot_w_projection_session - was the
    one number in the figure computed on different samples than everything else.
    Rows left NaN by the fit-parity edge padding are dropped here (fit_samples.
    drop_nan_rows) rather than passed on: the consumers are PCA and a variance fraction,
    neither of which takes NaN, and n_samples is what shrinks, not n_features."""
    trial_windows = []
    for _, full_intrvl in t0_events.loc[s_id, interval_name].dropna().items():
        trial_windows.append(fr_x[fr_x.index.overlaps(full_intrvl)].values)
    lengths = pd.Series([len(x) for x in trial_windows])
    correct_length = lengths.value_counts().index[0]
    trial_windows = np.stack([x for x in trial_windows if len(x) == correct_length])

    trial_windows = fitspl.prepare(trial_windows, normalize)
    start, end = timebin_range
    trial_windows = trial_windows[:, start:end + 1, :]
    n_trials, n_bins, n_features = trial_windows.shape
    return fitspl.drop_nan_rows(trial_windows.reshape(n_trials * n_bins, n_features))


def plot_w_variance_explained(target_spec, w_row, t0_events, session_names, max_pcs=50, save_dir=None,
                               ith_to_dt_session=None, normalize=fitspl.AUTO):
    """Parts 2+3 of the W-variance-explained analysis (see plot_session_pca for part 1,
    the standalone whole-session baseline this is deliberately kept separate from).
    `target_spec`: a dict shaped like a highlight_selections entry, but naming an
    activity WINDOW to evaluate rather than a fitted decoder row - keys `session_id`,
    `interval`, `timebin_range` (a (start, end) pair, both inclusive - see
    get_interval_window_activity), and an optional `name`. `w_row`: a single row from
    svm_plotting_lib.resolve_w_selections' frame (as in plot_w_projection_session)
    - supplies X_modality/w/predict_y_name/interval_name/timebin (which decoder) and its
    own `session_id` (where it was fit, for the title) - deliberately independent of
    target_spec['session_id'] (where its variance is being evaluated), the same
    origin/target split plot_w_projection_session uses, so one fixed W can be checked
    against any session's activity, not just its own origin session. Pools every
    trial's activity across every timebin in target_spec['timebin_range'] into one
    sample set (get_interval_window_activity), computes that pooled set's PCA
    eigenvalue spectrum (% of total variance - _pca_eigen) and the variance
    w_row['w'] itself explains in the same pooled set (this direction's own variance /
    total variance - not literally one of the PCA eigenvalues, since w isn't generally a
    PCA eigenvector, but directly comparable in the same %-of-total-variance units),
    then plots both together (plot_pca_eigenvalues) so the W direction's variance can be
    judged against the ranked PC spectrum, in particular PC1 (the max direction).
    `normalize`: the sample preprocessing, resolved off `w_row`'s own provenance by
    default (fit_samples.mode_for_w_row) so this number lands on the same axis as
    var_plotting_lib's per-timebin curve and as the projection row it gets annotated
    onto - see get_interval_window_activity. Saves to save_dir/wvar_{name or
    session_interval_range}.png if save_dir is given. Returns
    (save_path, w_var_pct, eigenvalues_pct)."""
    x_modality, w = w_row['X_modality'], np.asarray(w_row['w'])
    mode = fitspl.mode_for_w_row(w_row, normalize)
    t_s_id = target_spec['session_id']
    interval_name, timebin_range = target_spec['interval'], target_spec['timebin_range']

    session_name = next(sn for sn in session_names if sn.startswith(t_s_id))
    fr_z_raw = analytics.get_analytics('FiringRate40msZ', session_names=[session_name]).xs(t_s_id, level='session_id')
    fr_x = session_fr_for_modality(fr_z_raw, x_modality)

    pooled = get_interval_window_activity(t_s_id, t0_events, fr_x, interval_name, timebin_range,
                                           normalize=mode)
    eigenvalues_pct, _, total_var = pca_eigen(pooled)

    centered = pooled - pooled.mean(axis=0, keepdims=True)
    w_unit = w / np.linalg.norm(w)
    w_var_pct = np.var(centered @ w_unit, ddof=1) / total_var * 100

    origin_num = (ith_to_dt_session or {}).get(w_row['session_id'], "Unknown")
    origin_label = f'S{origin_num:02d}' if isinstance(origin_num, int) else f'S{origin_num}'
    t_num = (ith_to_dt_session or {}).get(t_s_id, "Unknown")
    t_label = f'S{t_num:02d}' if isinstance(t_num, int) else f'S{t_num}'
    target_name = target_spec.get('name')
    w_name = w_row.get('name')
    w_label = w_name if isinstance(w_name, str) and w_name else f"{w_row['predict_y_name']} W"

    n_shown = min(max_pcs or len(eigenvalues_pct), len(eigenvalues_pct))
    fig, ax = plt.subplots(figsize=(4, max(3, 0.18 * n_shown)), dpi=150)
    plot_pca_eigenvalues(ax, eigenvalues_pct, w_var_pct=w_var_pct, w_label=w_label, max_pcs=max_pcs,
                          title=f'{t_label} {interval_name} t={timebin_range[0]}-{timebin_range[1]}'
                                f'\n{w_label} (from {origin_label})')
    plt.tight_layout()

    save_path = None
    if save_dir is not None:
        fname_tag = target_name or f'{t_s_id}_{interval_name}_{timebin_range[0]}-{timebin_range[1]}'
        save_path = os.path.join(save_dir, f'wvar_{fname_tag}.png')
        fig.savefig(save_path, dpi=150)
    plt.show()
    return save_path, w_var_pct, eigenvalues_pct


# ------------------------------------------------- does it track anything kinematic


def _window_trial_bin_values(data_by_interval, trial_ids_by_interval, interval_name, timebin_range, label_range=None):
    """Every (trial, bin) sample within `timebin_range` (inclusive, native-bin index -
    same convention as get_interval_window_activity/plot_w_projection's own
    `highlight_range`) of one interval's (n_trials, n_bins) array, pooled into one long
    Series - same pooling get_interval_window_activity uses for PCA (trial-to-trial and
    within-window temporal variability treated as one combined source of variation, NOT
    averaged away), just for a per-trial-per-bin scalar (a W projection, or one raw
    kinematic column) instead of a full feature vector. Works identically on
    get_trial_w_projection's and get_trial_behavior_trace's return shape, since both are
    {interval: (n_trials, n_bins) array} + {interval: trial_id array} pairs.
    Indexed by a (trial_id, timebin) MultiIndex, so two different windows/variables can
    be joined sample-for-sample even if they dropped different trials to their own
    modal-length trim. `label_range`, if given, overrides which timebin labels are used
    in that index (must be the same length as `timebin_range` spans) - lets a caller
    pull data from a *shifted* window (e.g. a lagged kinematic_offset) while its samples
    still line up against another series' unshifted (trial_id, timebin) labels."""
    start, end = timebin_range
    window = data_by_interval[interval_name][:, start:end + 1]
    trial_ids = trial_ids_by_interval[interval_name]
    bin_labels = list(label_range) if label_range is not None else list(range(start, end + 1))
    index = pd.MultiIndex.from_product([trial_ids, bin_labels], names=['trial_id', 'timebin'])
    return pd.Series(window.ravel(), index=index)


KINEMATIC_COLUMNS = ['frame_raw_500msMedian', 'frame_raw_abs_acc_500msMedian', 'head_angle', 'head_angle_vel',
                      'movement_energy_smooth5', 'frame_YawPitch_abs_vel_sum_500msMedian',
                      'frame_YawPitch_abs_acc_sum_500msMedian']


def plot_w_projection_kinematics(s_id, w_row, t0_events, behavior, session_names, interval_name, timebin_range,
                                  kinematic_columns=KINEMATIC_COLUMNS, kinematic_offset=0, color_by=None,
                                  normalize=fitspl.AUTO, save_dir=None, ith_to_dt_session=None):
    """One session, one fixed `timebin_range` (inclusive, native-bin index - same
    convention as get_interval_window_activity/plot_w_projection's own
    `highlight_range`) within `interval_name`: scatter of subspace activation (w_row's
    projection) against each of `kinematic_columns` (pulled from Behavior40msAligned via
    get_trial_behavior_trace) - one panel per kinematic variable - to check whether the
    projection tracks a kinematic variable rather than (or in addition to) the task
    covariate it was fit to distinguish. Every (trial, bin) sample inside the window is
    its own scatter point (see _window_trial_bin_values) - pooled the same way
    get_interval_window_activity pools trials and bins for PCA, NOT averaged down to one
    point per trial: an average would blur out exactly the within-window timing this is
    meant to probe, and would be meaningless for a circular variable like head_angle.
    `kinematic_offset` (native bins, default 0) shifts *only* the kinematic side's whole
    window relative to `timebin_range` - e.g. `kinematic_offset=5` checks whether
    kinematics 200ms after the projection's own window explain it, for a behavioral
    response that lags the neural signal rather than tracking it instantaneously.
    `color_by`, if given, is a covariate name (see _trial_covariate_levels - most usefully
    w_row['predict_y_name'], the dimension this subspace was actually fit to distinguish):
    colors every point by its own trial's level instead of uniform gray, so a kinematic
    correlation can be visually checked against whether it's really just the task
    covariate underneath (e.g. the animal simply runs faster on one choice).
    `normalize`: see get_trial_w_projection - 'auto' by default here, the same
    provenance-driven resolution plot_w_projection_session uses (the raw dot product
    otherwise scales with each session's own feature-vector norm, unrelated to signal).
    Samples with a NaN in either variable (e.g. head_angle can have missing camera-pose
    frames, and fit parity leaves each interval's first/last bin NaN) are dropped before
    computing each panel's Pearson r (scipy.stats.pearsonr),
    annotated in that panel's corner. Saves to save_dir/session_{s_id}_S{s_num}_
    {interval_name}_t{start}-{end}_kinematics.png if save_dir is given. Returns a
    {column: (r, p)} dict."""
    x_modality, w = w_row['X_modality'], w_row['w']
    session_name = next(sn for sn in session_names if sn.startswith(s_id))
    fr_z_raw = analytics.get_analytics('FiringRate40msZ', session_names=[session_name]).xs(s_id, level='session_id')
    fr_x = session_fr_for_modality(fr_z_raw, x_modality)

    projections, proj_trial_ids = get_trial_w_projection(s_id, t0_events, fr_x, [interval_name], w,
                                                           normalize=fitspl.mode_for_w_row(w_row, normalize),
                                                           b=fitspl.bias_for_w_row(w_row))
    proj_series = _window_trial_bin_values(projections, proj_trial_ids, interval_name, timebin_range)

    color_by_trial = None
    if color_by is not None:
        unique_trial_ids = proj_trial_ids[interval_name]
        levels = _trial_covariate_levels(t0_events, s_id, unique_trial_ids, color_by)
        color_by_trial = pd.Series([_covariate_color(color_by, lvl) for lvl in levels], index=unique_trial_ids)

    start, end = timebin_range
    kin_range = (start + kinematic_offset, end + kinematic_offset)

    fig, axes = plt.subplots(1, len(kinematic_columns), figsize=(3.2 * len(kinematic_columns), 3.2), dpi=150,
                              squeeze=False)
    axes = axes[0]

    correlations = {}
    for ax, column in zip(axes, kinematic_columns):
        traces, trial_ids = get_trial_behavior_trace(s_id, t0_events, behavior, [interval_name], column)
        kin_series = _window_trial_bin_values(traces, trial_ids, interval_name, kin_range,
                                               label_range=range(start, end + 1))

        shared = proj_series.index.intersection(kin_series.index)
        x, y = kin_series.loc[shared], proj_series.loc[shared]
        valid = x.notna() & y.notna()
        x, y = x[valid], y[valid]
        r, p = stats.pearsonr(x, y)
        correlations[column] = (r, p)

        dot_colors = 'k'
        if color_by_trial is not None:
            dot_colors = color_by_trial.reindex(x.index.get_level_values('trial_id')).values
        ax.scatter(x, y, c=dot_colors, s=8, alpha=0.3, edgecolor='none')
        ax.set_xlabel(column, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.text(0.95, 0.05, f'r={r:.2f}', transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                color='purple', bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.75))
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel('W projection', fontsize=8)

    s_num = (ith_to_dt_session or {}).get(s_id, "Unknown")
    s_label = f'S{s_num:02d}' if isinstance(s_num, int) else f'S{s_num}'
    title = f"{s_label}: {w_row['predict_y_name']} W @ {interval_name} t={start}-{end} vs. kinematics"
    if kinematic_offset:
        title += f' (kinematics @ t={kin_range[0]}-{kin_range[1]})'
    if color_by is not None:
        title += f'\ncolored by {color_by}'
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()

    save_path = None
    if save_dir is not None:
        save_path = os.path.join(save_dir, f'session_{s_id}_S{s_num}_{interval_name}_t{start}-{end}_kinematics.png')
        fig.savefig(save_path, dpi=150)
    plt.show()
    return correlations


def plot_session_projection_kinematics(s_id, w_row, behavior, session_names, kinematic_columns=KINEMATIC_COLUMNS,
                                        color_by=None, normalize=fitspl.AUTO, max_points=None, save_dir=None,
                                        ith_to_dt_session=None):
    """Whole-session counterpart to plot_w_projection_kinematics - same idea (subspace
    activation vs. kinematics scatter, one panel per kinematic column, Pearson r
    annotated), but over w_row's projection at *every* native 40ms bin of session
    `s_id`'s entire continuous recording, not just bins that fall inside a trial window.
    Answers a different question than the trial-windowed version: is a kinematic
    relationship specific to the task-relevant window(s) the windowed scatter looks at,
    or does it hold throughout the whole session, including inter-trial time? Kept as
    its own function rather than a mode flag on plot_w_projection_kinematics - the same
    "standalone whole-recording view, kept separate from the windowed one" split this
    file already uses for PCA (see plot_session_pca vs. plot_w_variance_explained).
    No t0_events/trial involvement at all here: the projection and each kinematic column
    are aligned directly on the shared 40ms ephys-bin grid both FiringRate40msZ and
    Behavior40msAligned are built from, matched by each bin's right/`to_ephys_timestamp`
    edge (both use the same closed='right' convention - see _x_modality_fr and the
    `behavior` IntervalIndex construction above).
    `color_by`, if given, must be a plain per-bin `behavior` column ('cue', 'choice_R1',
    'choice_R2', 'trial_outcome', ...) - NOT a CORRECTNESS_SPECS name ('correct_R1'/
    'correct_R2'), since those are only computable per-trial via t0_events, which this
    whole-session view deliberately doesn't touch. Raw 'cue' (1/2) and 'trial_outcome'
    are adjusted to 0/1 the same way _trial_covariate_levels does; bins outside any
    trial (ITI, where these columns are NaN/-1) are drawn in light gray rather than
    dropped, so the ITI bins stay visible as their own (uncolored) part of the cloud.
    `normalize`: see get_trial_w_projection/plot_w_projection_kinematics. `max_points`,
    if given, randomly subsamples the (typically tens of thousands of bins per session)
    aligned point set for plotting only, since a scatter that dense gets slow/heavy to
    render - Pearson r is always computed on the full aligned set first, before any
    subsampling for display. Saves to save_dir/session_{s_id}_S{s_num}_session_
    kinematics.png if save_dir is given. Returns a {column: (r, p)} dict."""
    x_modality, w = w_row['X_modality'], np.asarray(w_row['w'])
    session_name = next(sn for sn in session_names if sn.startswith(s_id))
    fr_z_raw = analytics.get_analytics('FiringRate40msZ', session_names=[session_name]).xs(s_id, level='session_id')
    fr_x = session_fr_for_modality(fr_z_raw, x_modality)

    # Whole-session path, same preprocessing as the trial-aligned one (and as
    # var_plotting_lib.session_samples): the rolling fit-window mean runs across the
    # whole recording here, inter-trial time included, so only the very first/last bin of
    # the session end up NaN - they drop out with the per-panel `valid` mask below.
    values = fitspl.prepare(fr_x.values, fitspl.mode_for_w_row(w_row, normalize))
    proj = pd.Series(values @ w + fitspl.bias_for_w_row(w_row), index=fr_x.index.right)

    behavior_s = behavior.xs(s_id, level='session_id')
    color_col = None
    if color_by is not None:
        color_col = behavior_s[color_by]
        if color_by == 'cue':
            color_col = (color_col - 1).clip(0, 1)
        elif color_by == 'trial_outcome':
            color_col = color_col.clip(0, 1)
        color_col = pd.Series(color_col.values, index=behavior_s.index.right)

    fig, axes = plt.subplots(1, len(kinematic_columns), figsize=(3.2 * len(kinematic_columns), 3.2), dpi=150,
                              squeeze=False)
    axes = axes[0]

    correlations = {}
    rng = np.random.default_rng(0)
    for ax, column in zip(axes, kinematic_columns):
        kin = pd.Series(behavior_s[column].values, index=behavior_s.index.right)
        shared = proj.index.intersection(kin.index)
        x, y = kin.loc[shared], proj.loc[shared]
        valid = x.notna() & y.notna()
        x, y = x[valid], y[valid]
        r, p = stats.pearsonr(x, y)
        correlations[column] = (r, p)

        plot_idx = x.index
        if max_points is not None and len(plot_idx) > max_points:
            plot_idx = rng.choice(plot_idx, size=max_points, replace=False)
        dot_colors = 'k'
        if color_col is not None:
            levels = color_col.reindex(plot_idx)
            dot_colors = [_covariate_color(color_by, lvl) if pd.notna(lvl) else '#cccccc' for lvl in levels]
        ax.scatter(x.loc[plot_idx], y.loc[plot_idx], c=dot_colors, s=6, alpha=0.15, edgecolor='none',
                   rasterized=True)
        ax.set_xlabel(column, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.text(0.95, 0.05, f'r={r:.2f}', transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                color='purple', bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.75))
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel('W projection\n(whole session)', fontsize=8)

    s_num = (ith_to_dt_session or {}).get(s_id, "Unknown")
    s_label = f'S{s_num:02d}' if isinstance(s_num, int) else f'S{s_num}'
    title = f"{s_label}: {w_row['predict_y_name']} W vs. kinematics (whole session)"
    if color_by is not None:
        title += f'\ncolored by {color_by} (gray = outside any trial)'
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()

    save_path = None
    if save_dir is not None:
        save_path = os.path.join(save_dir, f'session_{s_id}_S{s_num}_session_kinematics.png')
        fig.savefig(save_path, dpi=150)
    plt.show()
    return correlations
