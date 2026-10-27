#!/usr/bin/env python
# coding: utf-8

# %%
# to setup import paths add project root dirs to sys.path
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "..", ".."))
from baseVR.base_functionality import init_import_paths
init_import_paths()

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'widget')

from analytics_processing import analytics
import analytics_processing.analytics_constants as C
from CustomLogger import CustomLogger as Logger

from analytics_processing.sessions_from_nas_parsing import sessionlist_fullfnames_from_args
from analytics_processing.sessions_from_nas_parsing import fullfnames2snames

# Shared session loading + the HP/mPFC unit split (one definition, instead of the bare
# .iloc[:, 20:] that used to appear here, in svm_analysis_2026 and in the fit itself).
from plotlib import session_data as sd

# %%
nas_dir = C.device_paths()[0]
output_dir = os.path.join(nas_dir, 'Simon', 'mPFC_timealigned_PCA_dims',)
os.makedirs(output_dir, exist_ok=True)
Logger().init_logger(None, None, logging_level="INFO")

# %%
animal_ids = [6]
paradigm_ids = [1100]
session_ids = None

# %%
session_names, session_to_dt_session = sd.get_session_names(paradigm_ids, animal_ids, session_ids)
session_names

# %%
# ---- data for the first plot: single-trial position kinematics across t0 intervals ----
t0_events = sd.load_t0_events(session_names)   # defaults to the current (...40ms2) table

behavior = analytics.get_analytics(
    'Behavior40msAligned', session_names=session_names,
    columns=['trial_id', 'cue', 'trial_outcome', 'choice_R1', 'choice_R2',
             'to_ephys_timestamp', 'frame_position'])
behavior = behavior.reset_index().drop(columns=['paradigm_id', 'animal_id', 'entry_id'])
behavior.index = pd.IntervalIndex.from_arrays(
    left=behavior.to_ephys_timestamp - 40_000,
    right=behavior.to_ephys_timestamp,
    closed='right')
behavior.set_index('session_id', append=True, inplace=True)
t0_events.t0_event_name.unique()

# %%
# ---- ephys: population activity magnitude, aligned to the same t0 intervals ----
# Alternative A: raw (non-Z) firing rate, then Z-score per unit column pooled *across all
# sessions* instead of per-session (FiringRate40msZ's own approach - see get_FiringRate40msZ
# in postproc_mea1k_ephys.py). NB: 'UnitNNNN' is just that session's own sequential sorted-
# cluster index (see extract_jrc_spikes) - the same name in two different sessions is a
# different physical neuron, so this pools different neurons' distributions together under
# one column. Simon confirmed this tradeoff and wants it anyway, for a first pass.
fr_hz_all = analytics.get_analytics('FiringRate40msHz', session_names=session_names)
unit_cols = [c for c in fr_hz_all.columns if c.startswith('Unit')]

global_mean = fr_hz_all[unit_cols].mean()
global_std = fr_hz_all[unit_cols].std().replace(0, np.nan)  # avoid inf for an always-silent unit
fr_z_global = fr_hz_all.copy()
fr_z_global[unit_cols] = (fr_hz_all[unit_cols] - global_mean) / global_std

# %%
# Alternative B: the existing per-session Z-scored analytic, straight from disk
fr_z = analytics.get_analytics('FiringRate40msZ', session_names=session_names)

# %%
# ---- unit quality: the mean rate each unit's Z-scores were computed against --------------
# Needed for MIN_UNIT_RATE_HZ below. A unit firing ~0.005 Hz (one spike every few minutes) is
# z-scored against a std estimated over a session in which it is almost always zero, so its
# single spike arrives as z~80 and can own a whole timebin's covariance spectrum on its own -
# which is exactly what the discrete jumps in the variance-explained curve turned out to be.
# On animal 6: 5-9 of 57 mPFC units per session are below 0.1 Hz, 15-20 below 0.5 Hz.
# fr_hz_all is the table the Alternative A cell above already loaded - passing it in avoids a
# second full load of FiringRate40msHz, which is minutes of I/O for a column mean.
unit_rate_hz = sd.unit_mean_rates(fr_hz=fr_hz_all)
print(unit_rate_hz.notna().sum(1).rename('n_units').to_frame().assign(
    below_0p1=(unit_rate_hz < 0.1).sum(1), below_0p5=(unit_rate_hz < 0.5).sum(1),
    median_hz=unit_rate_hz.median(1).round(3)))

# %%
# pick which Z-scoring feeds the plot below (defaults to the plain per-session analytic -
# swap in fr_z_global to use the pooled-across-sessions version instead), then align it to
# behavior's own trial_id/IntervalIndex: fr_z/fr_z_global share the exact same 40ms ephys
# bin grid as behavior, row-for-row (confirmed on real data - both derive from the same
# FiringRate40msHz timestamp columns, see get_Behavior40msAligned), so behavior's trial_id
# can be copied across directly rather than needing its own trial_id column.
fr_z_plot = fr_z  # or fr_z_global
# fr_z_plot = fr_z_global

fr_z_plot = fr_z_plot.reset_index().drop(columns=['paradigm_id', 'animal_id', 'entry_id'])
fr_z_plot['trial_id'] = behavior['trial_id'].values
fr_z_plot.index = pd.IntervalIndex.from_arrays(
    left=fr_z_plot.to_ephys_timestamp - 40_000,
    right=fr_z_plot.to_ephys_timestamp,
    closed='right')
fr_z_plot.set_index('session_id', append=True, inplace=True)

# slice to mPFC units:
X_MODALITY = 'mPFC'
fr_z_plot = fr_z_plot[sd.unit_columns(fr_z_plot, X_MODALITY) + ['trial_id']]
# unit_cols must describe THIS frame, not the unsliced one it came from: the throwaway
# cross-session cell further down passes it straight to dim_plotting_lib and used to
# KeyError on the 20 HP names that no longer exist here.
unit_cols = sd.unit_columns(fr_z_plot, X_MODALITY)
fr_z_plot

# %%
import time
import dim_plotting_lib as dpl
from dim_plotting_lib import plot_session_kinematics_neural_and_pca

which_intervals = [
                #    'track_start_interval',
                   'cue_entry_interval', 
                   'cue_exit_interval',
                    'R1_entry_interval', 
                    # 'reward1_sound_interval',
                    # 'R2_entry_interval', 
                    # 'reward2_sound_interval',
                    'ITI_interval'
                    ]

# ---- trial grouping ---------------------------------------------------------------------
# The split now happens *inside* one figure rather than by rendering one figure per group
# (two separate figures were what made the comparison hard to read): GROUPBY draws row 1's own
# two lines - population RMS (solid) and, on the twin axis, population mean z (dotted) - once
# per group value, over the pooled black one, in the canonical per-label colors (cue1 orange /
# cue2 purple, exactly as in the SVM figures - plotlib/colors.py; an ordered grouping like
# session_time instead runs green->yellow off viridis). HIGHLIGHTS is the separate decision of
# which trials every row's pooled/heatmap content is built from.
GROUPBY = 'choice_R1'          # None | 'cue' | 'choice_R1' | 'choice_R2' | 'trial_outcome' | any other
                         #   per-trial behavior column | 'session_time' (trials in order, split
                         #   into GROUPBY_N_BINS equal-count groups: early/mid/late)
GROUPBY_N_BINS = 3       # 'session_time' only
GROUPBY_MATCH_N = True   # subsample every group to the smallest group's trial count - the group
                         #   lines are drawn band-free, so a group with half the trials just looks
                         #   noisier, which reads as a difference between groups rather than n
HIGHLIGHTS = ['all']     # ONE FIGURE PER ENTRY, stacked into this session's canvas column:
                         #   'all' = every trial feeds the pooled line and the heatmaps (the group
                         #   lines are then read against a pooled reference); or e.g. [1, 2] for
                         #   cue, ['early', 'mid', 'late'] for session_time, to restrict those to
                         #   one group at a time - the other groups' lines stay drawn either way
# ---- what the variance-explained curve actually measures ---------------------------------
# The three settings that decide whether this figure means anything (see
# dim_plotting_lib._pca_basis_for_n_pcs' `scaling` note for the measurements behind them).
PCA_SCALING = 'covariance'   # 'correlation' = standardize every unit at each timebin before the
                              #   SVD, so no single loud unit can raise variance-explained on its
                              #   own; 'covariance' = the old behavior, on which 8.6% of timebins
                              #   carried a spurious +0.1-0.2 jump that was ONE near-silent unit
                              #   firing ONCE in ONE trial (drop that unit or that trial and 96%
                              #   of the jump goes away)
PCA_NULL_REPS = 0             # chance band on the summary row's variance-explained curve
                              #   (each unit's trials permuted independently). OFF: it is a
                              #   chance level for the DIMENSIONALITY number, not a test of any
                              #   condition difference, and its answer here is a dull constant
                              #   ~2.5pp above chance everywhere. Set to 25 to re-check that on
                              #   new data. The shuffle that tests the trial split is
                              #   DRAW_SHUFFLE below
DRAW_SHUFFLE = True           # row 1's band becomes the GROUPBY permutation null instead of the
                              #   sem of the pooled mean: SHUFFLE_REPS random re-assignments of
                              #   the group labels across trials, 5-95% of the resulting
                              #   group-mean traces. A colored group line leaving the band is a
                              #   split effect a random split would rarely produce. Needs
                              #   GROUPBY set and highlight='all'. NB wider than the sem band it
                              #   replaces (+/-1.64 sd of the null vs +/-1 sem) - that is the
                              #   band changing meaning, not the data getting noisier
SHUFFLE_REPS = 200
MIN_UNIT_RATE_HZ = 0.1        # drop units firing below this (None = keep all). Removes the
                              #   near-silent clusters whose z-scores are single spikes over a
                              #   near-zero std; 0.5 is defensible for mPFC and drops ~1/3
SAVE_DPI = 200           # PNG export resolution (matplotlib default 100 - neuron ids unreadable)

saved_png_paths, saved_svg_paths = [], []
for i, s_id in enumerate(session_names[:20]):
    if (i < 7) or (i > 16):
        continue
    s_id = s_id[:16]  # strip off the paradigm/animal suffix
    if s_id not in t0_events.index.unique('session_id'):
        print(f"Session {s_id} has no t0 events, skipping")
        continue
    if s_id not in fr_z_plot.index.unique('session_id'):
        print(f"Session {s_id} has no ephys, skipping")
        continue
    session_pngs, session_svgs = [], []
    for highlight in HIGHLIGHTS:
        split_label = f"{GROUPBY}-{highlight}" if GROUPBY else "all"
        # every plot_session_kinematics_neural_and_pca kwarg, spelled out with its own default
        # value (see the function's docstring in dim_plotting_lib.py for what each one does) -
        # so tweaking one just means editing the value already sitting here, not looking up
        # what else there is to override. The fine-grained figure-geometry knobs (row heights,
        # gap_bins, colorbar sizing, ...) got cut from the function entirely - nothing here
        # ever changed them - so this list is now just the parameters that actually matter.
        fig, (ax0, ax1, ax2) = plot_session_kinematics_neural_and_pca(
            t0_events, behavior, fr_z_plot, s_id, which_intervals,
            unit_cols=None,
            groupby=GROUPBY,                       # None = one undivided figure, exactly as before
            highlight=highlight,                   # 'all' | one of GROUPBY's values: what the pooled
                                                   #   black line and every heatmap row are built from
            groupby_n_bins=GROUPBY_N_BINS,         # 'session_time' only
            groupby_match_n=GROUPBY_MATCH_N,       # equalize group trial counts before fitting
            smooth=3,                              # rows 0-1 (position/RMS) display smoothing
            pca_smooth=3,                          # smoothing applied before PCA. NB a centered
                                                   #   rolling mean: any feature narrower than
                                                   #   this many bins is a single-40ms-bin event
                                                   #   smeared into a boxcar, not a real plateau
            pca_n_pcs=6,                           # fixed number of top PCs kept at every timebin
            pca_scaling=PCA_SCALING,               # 'correlation' | 'covariance' - see above
            pca_null_reps=PCA_NULL_REPS,           # dimensionality chance band, summary row (0=off)
            draw_shuffle=DRAW_SHUFFLE,             # row 1's band = the GROUPBY permutation null
            shuffle_reps=SHUFFLE_REPS,             # permutations behind that band
            shuffle_seed=0,                        # fixed, so a re-render is identical
            unit_rate_hz=unit_rate_hz,             # per-session mean Hz, for the floor below
            min_unit_rate_hz=MIN_UNIT_RATE_HZ,     # unit quality floor (None = keep every unit)
            angle_range=(30, 70),                  # canonical-angle heatmap color scale (true range is 0-90)
            metric_range=(.7, 1.1),                # fixed y-lim for row 1 (population RMS)
            mean_z_range=(-.1, .1),                # None = no line; else draws a dotted population-mean line on a twin y-axis
            draw_colorbar=False,                    # master switch for every colorbar
            var_explained_display='scatter',           # 'scatter' (halo on the RMS line) | 'off'; the old dedicated 'row' is gone,
                                                   #   its curve is now the ceiling line of the summary row below the neuron heatmap
            var_explained_center=0.35,             # diverging colormap center (40% explained)
            var_explained_range=(0.1, 0.7),        # fixed color range, +/-0.3 around the center.
                                                   #   NB both were picked for 'covariance' values.
                                                   #   Under 'correlation' the curve only spans
                                                   #   ~0.28-0.37 across 4 measured sessions, so
                                                   #   this range renders the halo near-uniform -
                                                   #   narrow to e.g. (0.25, 0.45)/center 0.32 to
                                                   #   get its contrast back
            show_neuron_contrib=True,              # extra heatmap row: which neurons carry the subspace
            show_neuron_contrib_summary=True,      # thin line row below it: how much variance the
                                                   #   column above actually accounts for
            neuron_contrib_summary_line='drawn',   # ONE mass curve: 'drawn' (dark red, what the
                                                   #   neurons you can see sum to) | 'selected'
                                                   #   (orange, what the rule picked before the
                                                   #   row cap) | 'both' (the old two-curve row)
            neuron_contrib_select='per_pc',        # 'per_pc' (top-|loading| per eigenvector) | 'top_contrib' (largest contributors)
            neuron_contrib_top_per_pc=3,           # 'per_pc' only: neurons drawn per PC per timebin
            neuron_contrib_top_n=12,               # 'top_contrib' only: neurons drawn per timebin
            neuron_contrib_fill='dense',           # 'dense' (every shown neuron at every timebin) | 'selected' (only nominated cells)
            neuron_contrib_max_rows=12,            # row cap, kept by largest total contribution
            neuron_contrib_range='default',        # cell color scale: 'default' resolves from
                                                   #   pca_scaling (0-10% of total variance under
                                                   #   'covariance', 0-(1/n_units) - the per-cell
                                                   #   arithmetic ceiling - under 'correlation');
                                                   #   or a (min, max) fraction pair, or None to
                                                   #   autoscale to this session
        )
        fig.suptitle(f"Session {i:02d} - {s_id} - {split_label}")
        plt.show()

        png_path = os.path.join(output_dir, f"{i:02d}_{s_id}_mPFC_pca_{split_label}.png")
        svg_path = os.path.join(output_dir, f"{i:02d}_{s_id}_mPFC_pca_{split_label}.svg")
        fig.savefig(png_path, dpi=SAVE_DPI)
        fig.savefig(svg_path)
        session_pngs.append(png_path)
        session_svgs.append(svg_path)
    # one canvas column per session, its HIGHLIGHTS figures stacked top to botom
    # (build_canvas' own layout; a session that rendered nothing contributes no column rather
    # than an empty one, which build_canvas would choke on)
    if session_pngs:
        saved_png_paths.append(session_pngs)
        saved_svg_paths.append(session_svgs)
    # break

# %% [markdown]
"""
Done — the dedicated variance-explained row is gone, its curve is now labelled 6 PCs explained var (all units) in the summary row, and the color scale is a fixed parameter. Your figure is now: position → RMS → neuron heatmap → summary → angle heatmap. Passing var_explained_display='row' raises with a pointer instead of silently doing nothing; I set your call site to 'off' ('scatter' still available for the halo on the RMS line).

Red vs orange — yes, both depend on select, but differently:

Orange = the selection itself, summed at each timebin. With top_contrib: exactly the top_n=12 largest contributors. With per_pc: the union of each of the 6 eigenvectors' top-3 leaders — so ≤ 18, and your instinct is right that it's usually fewer, because eigenvectors share leaders. On real data it's median 13, range 8–18. That's exactly what the dashed blue count line shows in per_pc mode (and why there's no such line in top_contrib mode — it would be flat at 12).
Red = the neurons that have a row in the heatmap above, at every timebin (dense fill). Its dependence on select is only indirect: the row ranking sums each neuron's contribution over the bins where it was selected, so per_pc favours transient specialists and top_contrib steady carriers. Once the 35 rows are chosen, red is just their total.
That's why red can sit above orange: a shown neuron that wasn't in this timebin's top-12 still contributes, and dense fill counts it. Both are always ≤ the gray ceiling.

And yes, your reading of the color is exactly right. A cell is that neuron's share of the timebin's total across-trial variance — not of the subspace's variance. Same quantity and same units as the gray line, which is why summing a column's cells over all neurons reproduces the gray line exactly (verified to 1e-15). So a cell reading 8% and a ceiling of 40% means that one neuron accounts for a fifth of what the whole 6-PC subspace captures.

Scale now fixed by parameter: neuron_contrib_range=(0, 0.10) is the new default (fractions of total variance; None restores per-session autoscale). I measured before picking it — across 6 sessions the 98th percentile of drawn cell values sat between 7.9% and 11.0% under both select rules (pooled median 0.51%, p90 3.4%, max 73%), so 0–10% loses essentially no contrast versus autoscaling while making colors mean the same thing in every session. About 1.9% of cells clip to the darkest red.

Verified: all 24 flag combinations (var_explained_display × select × fill × summary × heatmap) render with nothing off-canvas, and the real-session render in your exact config is 10.1×19.0in with the lowest colorbar at the 0.05in buffer.
"""


# %%
# one PNG canvas (build_canvas, rasterized via PIL) and one SVG canvas (build_canvas_svg,
# stays vector/editable - see dim_plotting_lib.py) with every session above placed side by
# side - each session is a column of its HIGHLIGHTS figures stacked top to bottom, which is
# exactly the per-session list-of-paths those two take; ported from
# the same build_canvas/build_canvas_svg used for svm_analysis_2026.py's per-session posters.
if saved_png_paths:
    canvas_path = os.path.join(output_dir, 'mPFC_CHOICE_R1_pca_canvas.png')
    dpl.build_canvas(saved_png_paths, canvas_path)
    print(f"Saved combined PNG canvas of {len(saved_png_paths)} sessions to {canvas_path}")

    canvas_path_svg = os.path.join(output_dir, 'mPFC_CHOICE_R1_pca_canvas.svg')
    dpl.build_canvas_svg(saved_svg_paths, canvas_path_svg)
    print(f"Saved combined SVG canvas to {canvas_path_svg}")

# %%
# ---- levels, not dynamics: one column per (interval, condition), one row per session ------
# The same quantity as row 1 of the figures above (population RMS, or the population mean z
# that row draws on its twin axis - ACROSS_METRIC picks which), over the SAME which_intervals,
# but box-plot style: every condition of every interval gets its own narrow column, the
# conditions sitting next to each other rather than overlaid, and inside a column that
# interval's timebins are squeezed onto ~0.1in - so the time course is only a faint smear and
# what jumps out is the thick short horizontal line, that column's mean over the whole
# interval. Each row is one session, all on one shared y-axis. The question is only "do the
# conditions sit at different levels, in which interval, and does that hold session to
# session" - which the timeline figures, being one session each and spending their width on
# time, cannot answer.
from dim_plotting_lib import plot_metric_across_sessions

ACROSS_METRIC = 'rms'                 # 'rms' (population RMS) | 'mean_z' (population mean)
ACROSS_INTERVALS = which_intervals       # same blocks as the figures above, in the same order
ACROSS_POINT_LEVEL = 'bin'               # what one dot is: 'bin' (that timebin's mean across
                                         #   trials, i.e. the points row 1's line runs through)
                                         #   | 'trial' (one dot per trial, its own interval
                                         #   mean - shows the across-trial spread) | 'trial_bin'
                                         #   (every trial x bin value - the full raw cloud)
ACROSS_N_SESSIONS = 20                   # rows, in session order
ACROSS_METRIC_RANGE = {'rms': (.7, 1.1),     # fixed y-limits, per metric - the SAME limits row 1
                       'mean_z': (-.1, .1),  #   of the figures above uses (metric_range /
                       }[ACROSS_METRIC]      #   mean_z_range), so a level here reads at the same
                                             #   height as it does there. Shared by every row, so
                                             #   sessions are comparable; None autoscales instead
                                             #   (still across all rows, never per row)

across_session_ids = [sn[:16] for sn in session_names[:ACROSS_N_SESSIONS]]
across_labels = {s_id: f'{i:02d} {s_id}' for i, s_id in enumerate(across_session_ids)}

fig_across, axes_across = plot_metric_across_sessions(
    t0_events, behavior, fr_z_plot, across_session_ids, ACROSS_INTERVALS,
    unit_cols=unit_cols,
    metric=ACROSS_METRIC,
    groupby=GROUPBY,                       # None = one black column per interval
    groupby_n_bins=GROUPBY_N_BINS,         # 'session_time' only
    groupby_match_n=GROUPBY_MATCH_N,       # equalize group trial counts per session
    point_level=ACROSS_POINT_LEVEL,
    smooth=1,                              # per-trial smoothing before averaging; leave off
                                           #   here - it would make neighbouring dots
                                           #   correlated and the cloud tighter than the data
    unit_rate_hz=unit_rate_hz,             # per-session mean Hz, for the floor below
    min_unit_rate_hz=MIN_UNIT_RATE_HZ,     # same unit quality floor as the figures above
    metric_range=ACROSS_METRIC_RANGE,      # fixed y-limits (see above); None autoscales
    slot_width_in=0.10,                    # ONE condition's column - how much of the time
                                           #   course survives; this narrow, a column reads as
                                           #   a single level with a bit of scatter around it
    slot_gap_in=0.03,                      # between conditions of the same interval
    interval_gap_in=0.40,                  # between intervals - carries most of the panel's
                                           #   width now that the columns themselves are thin
    row_height_in=1.0,                     # one session row
    label_width_in=0.9,                    # left margin for the row labels/y ticks
    point_size=6,                          # the dots are the faint indicator, not the message
    point_alpha=0.35,
    mean_linewidth=3.0,                    # the message: one thick line per column
    show_pooled=False,                     # True adds a dashed black all-trials mean per block
    show_t0=True,                          # hairline t0 inside each column
    xtick_rotation=45,                     # interval labels; 0 only fits short names
    fontsize=8,                            # every text in the figure, and the margins for it
    session_labels=across_labels,
)
plt.show()

across_name = f'across_sessions_{ACROSS_METRIC}_{ACROSS_POINT_LEVEL}_{len(ACROSS_INTERVALS)}intervals'
fig_across.savefig(os.path.join(output_dir, f'{across_name}.png'), dpi=SAVE_DPI)
fig_across.savefig(os.path.join(output_dir, f'{across_name}.svg'))
print(f"Saved {across_name} to {output_dir}")

# %%
# ==========================================================================================
# EXPERIMENTAL - delete this cell (and the next) once done exploring. Quick, minimal-effort
# hack: concatenate every session's own PC-subspace-per-timebin sequence (see
# dim_plotting_lib._subspace_block_with_separators - same per-timebin PCA the single-session
# heatmap above uses) into ONE long sequence, insert a small blank-separator gap between
# sessions (same zero-basis trick used between intervals within a session), then run the
# same canonical-angle comparison over the whole thing - i.e. the single-session heatmap
# above, scaled up to "does session A's subspace at time t look like session B's at time t'".
# Reuses dim_plotting_lib's own private helpers directly rather than adding anything new to
# that file, since this is meant to be thrown away, not become a permanent library function.
#
# Cost is O(n^2) in total bins across all included sessions (~50us/pair on this machine,
# measured directly) - full 29 sessions (~6400 bins incl. gaps) would take ~15-20 min.
# N_SESSIONS below caps it to a fast, iterable subset; raise it once you know this is worth
# waiting longer for.
N_SESSIONS = 17

valid_session_ids = [s_id for s_id in (sn[:16] for sn in session_names)
                     if s_id in t0_events.index.unique('session_id')
                     and s_id in fr_z_plot.index.unique('session_id')]
pca_session_ids = valid_session_ids[:N_SESSIONS]

SESSION_GAP = 5  # blank separator slots between sessions, same idea as gap_bins between intervals

all_bases = []
session_bounds = []  # (start_idx, end_idx, s_id) per session, for boundary lines/labels
for s_id in pca_session_ids:
    x_coords_s, bases_s, _, _, _, _ = dpl._subspace_block_with_separators(
        t0_events, fr_z_plot, s_id, which_intervals, unit_cols, gap_bins=2, smooth=5, n_pcs=6,
        scaling=PCA_SCALING)
    if len(bases_s) == 0:
        print(f"Session {s_id}: no PCA bases, skipping")
        continue
    start = len(all_bases)
    all_bases.extend(bases_s)
    session_bounds.append((start, len(all_bases), s_id))
    all_bases.extend([np.zeros((0, len(unit_cols)))] * SESSION_GAP)

print(f"Total bins across {len(session_bounds)} sessions (incl. gaps): {len(all_bases)}")

# %%
n = len(all_bases)
angles = np.full((n, n), np.nan)
t_start = time.time()
for i in range(n):
    if all_bases[i].shape[0] == 0:
        continue
    if i % max(1, n // 10) == 0:
        print(f"  row {i}/{n} ({time.time() - t_start:.0f}s elapsed)")
    for j in range(i, n):
        if all_bases[j].shape[0] == 0:
            continue
        theta = dpl.subspace_angles(all_bases[i].T, all_bases[j].T)
        deg = np.mean(np.degrees(theta))
        angles[i, j] = angles[j, i] = deg
print(f"done in {time.time() - t_start:.0f}s")

fig, ax = plt.subplots(figsize=(12, 12))
cmap = plt.cm.viridis.with_extremes(bad='white')
mesh = ax.pcolormesh(np.ma.masked_invalid(angles), cmap=cmap, vmin=0, vmax=90)
for start, end, s_id in session_bounds:
    ax.axvline(x=start - 0.5, color='k', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.axhline(y=start - 0.5, color='k', linestyle=':', linewidth=0.8, alpha=0.6)
centers = [(start + end) / 2 for start, end, _ in session_bounds]
labels = [s_id for _, _, s_id in session_bounds]
ax.set_xticks(centers)
ax.set_xticklabels(labels, rotation=90, fontsize=7)
ax.set_yticks(centers)
ax.set_yticklabels(labels, fontsize=7)
ax.set_aspect('equal', adjustable='datalim')
fig.colorbar(mesh, ax=ax, label='Canonical angle (deg)', shrink=0.8)
ax.set_title(f'Cross-session PC-subspace canonical angle ({len(session_bounds)} sessions)')
fig.tight_layout()
plt.show()

# %%
# ---- the plain neuro view: one unit's z-scored rate, per condition, over the same x -------
# Everything above plots a quantity *derived* from the firing rates (a decoder's accuracy, a
# subspace's dimensionality, a direction's variance share). This is the rates themselves: one
# curve per cue x choice_R1 cell, on the same t0-aligned x with the same between-interval
# gaps - the curves simply break at a gap, so two timebins seconds apart are never joined.
# A 2x2 cell is one BICOLOUR DASHED line (cue1 & skip = orange interleaved with light blue),
# the same encoding var_plotting_lib's variance/activation rows use, and its opacity scales
# with its trial count. One figure per session, saved and canvassed like the PCA figures.
import neuro_plotting_lib as npl
which_intervals = ['cue_entry_interval', 'cue_exit_interval', 'R1_entry_interval', 'R2_entry_interval', ]
NEURO_UNIT = None            # None = mean over every unit in unit_cols (population mean z);
                             #   or one unit, as a column name ('Unit0046') or a bare id (46 -
                             #   NB ids are the *unsliced* table's, so mPFC starts around 41);
                             #   or a list of units, averaged together
NEURO_CONDITIONS = ('cue', 'choice_R1')   # the 2x2: C1&skip, C1&stop, C2&skip, C2&stop, one
                             #   curve each, ordered level 0 before level 1. A single name
                             #   ('cue') gives 2 plain solid curves; adding a third
                             #   ('trial_outcome') gives 8, most trial-poor - see MIN_TRIALS
NEURO_SMOOTH = 5             # centered rolling mean over bins, applied to each trial's own trace
                             #   BEFORE averaging across trials. A single unit's z at 40ms is
                             #   mostly spike-count noise; 3 = 120ms, 5 reads a lot calmer
NEURO_MIN_TRIALS = 3         # a condition with fewer trials than this in an interval leaves a
                             #   gap in its curve rather than a mean of two trials
NEURO_YLIM = -.2,.2            # THE knob for a session sequence: fixed (lo, hi), or a scalar read
                             #   as +-it, so every panel of the canvas means the same thing -
                             #   a per-session autoscale silently rescales y under the reader
                             #   between one session and the next. It CLIPS. Measured on
                             #   2024-12-10_17-20: a single mPFC unit ran -0.30..0.44 (so
                             #   +-0.5 is a reasonable start), the population mean only
                             #   -0.07..0.09 - one ylim cannot serve both, set it per NEURO_UNIT
NEURO_SHOW_SEM = True        # each curve's +-1 SEM as a faint band - worth keeping, a 2x2 cell
                             #   can be 9-13 trials and the mean curve alone doesn't say so
NEURO_ALPHA_FLOOR = 0.4      # opacity of the smallest cell relative to the largest (1.0)
NEURO_N_SESSIONS = 20

neuro_png_paths, neuro_svg_paths = [], []
for i, s_id in enumerate(sn[:16] for sn in session_names[:NEURO_N_SESSIONS]):
    if s_id not in t0_events.index.unique('session_id'):
        print(f"Session {s_id} has no t0 events, skipping")
        continue
    if s_id not in fr_z_plot.index.unique('session_id'):
        print(f"Session {s_id} has no ephys, skipping")
        continue
    result = npl.plot_condition_lines(
        t0_events, fr_z_plot, s_id, which_intervals,
        unit=NEURO_UNIT,
        unit_cols=unit_cols,                 # the mPFC slice this notebook already resolved
        conditions=NEURO_CONDITIONS,
        smooth=NEURO_SMOOTH,
        min_trials=NEURO_MIN_TRIALS,
        ylim=NEURO_YLIM,                     # fixed across sessions; clips
        show_sem=NEURO_SHOW_SEM,
        alpha_floor=NEURO_ALPHA_FLOOR,
        show_legend=True,                    # sits above the axes next to the title, with each
                                             #   curve's own n - which is per session, so it
                                             #   cannot move to a shared key the way the
                                             #   variance row's legend did
        save_dir=None,                       # saved by hand below, so the file naming stays in
                                             #   the same scheme as the other cells' exports
    )
    fig, block = result['fig'], result['block']
    # prepend the session index - the number every other figure in this notebook labels with
    fig.axes[0].set_title(f'{i:02d} {s_id} - {block["unit_label"]}', fontsize=9, loc='left',
                          pad=npl.legend_title_pad(len(block['row_keys'])))
    plt.show()

    unit_tag = str(NEURO_UNIT) if NEURO_UNIT is not None else 'popmean'
    stem = f'{i:02d}_{s_id}_mPFC_{unit_tag}_condition_lines'
    png_path = os.path.join(output_dir, f'{stem}.png')
    svg_path = os.path.join(output_dir, f'{stem}.svg')
    fig.savefig(png_path, dpi=SAVE_DPI)
    fig.savefig(svg_path)
    neuro_png_paths.append([png_path])
    neuro_svg_paths.append([svg_path])

# %%
# same side-by-side assembly the PCA figures get - one column per session
if neuro_png_paths:
    unit_tag = str(NEURO_UNIT) if NEURO_UNIT is not None else 'popmean'
    canvas_path = os.path.join(output_dir, f'mPFC_{unit_tag}_condition_lines_canvas.png')
    dpl.build_canvas(neuro_png_paths, canvas_path)
    print(f"Saved combined PNG canvas of {len(neuro_png_paths)} sessions to {canvas_path}")

    canvas_path_svg = os.path.join(output_dir, f'mPFC_{unit_tag}_condition_lines_canvas.svg')
    dpl.build_canvas_svg(neuro_svg_paths, canvas_path_svg)
    print(f"Saved combined SVG canvas to {canvas_path_svg}")

# %%
