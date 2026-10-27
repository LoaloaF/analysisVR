# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # poster_26 - the 2026 poster figures
#
# Starts with the **behaviour** results, which are the poster's first column:
#
#  1. bootstrap + imports
#  2. session list, output dir
#  3. trial table (`BehaviorTrialwise`) - shared by every cell below
#  4. **kinematics, skip vs stop**, one .svg/.png pair per session x metric
#  5. **correctness across the sequence**, one figure: a scatter per session with a fitted
#     trend line per group - grey over the pre-cue sessions, then one per cue
#  6. the same figure rotated (session down the y axis, S1 at the top), for a portrait
#     poster column
#
# Then the first **ephys** figure, which is the reader's grounding for every t0-aligned
# figure after it:
#
#  7. session list + the three 40ms tables the panel needs (t0 events, behaviour, rates)
#  8. **the condition panel**, one compressed two-row figure per session: the trials' own
#     position traces (with the cue zone marked) above, the population's z-scored firing
#     rate per cue below, aligned to cue entry and to the R1 choice
#  9. the same panels glued side by side into one .svg canvas
#
# Definitions live in `behavior_plotting_lib.py`, not here - this notebook is calls only,
# the convention `post_lm4.py`/`lm4_plots.py` follow (see `plotlib/__init__.py`'s layering
# note). The behaviour figures are matplotlib rewrites of the plotly dashboard plot
# `lm4_plots.py` used; see that library's module docstring for what changed and why.
#
# **Editing a library needs no kernel restart** - `%autoreload 2` (cell 2) patches the
# imported functions in place, so re-run only the cell you are working on.

# %%
import os
import sys

# to setup import paths add project root dirs to sys.path
sys.path.append(os.path.join(os.getcwd(), "..", ".."))
from baseVR.base_functionality import init_import_paths
init_import_paths()

# %%
# Load-bearing, and written as run_line_magic rather than as `# %autoreload 2`: in a
# jupytext percent file that comment form only becomes a real magic when opened through the
# pairing and stays an inert comment otherwise, which is how these libraries silently went
# un-reloaded and a changed signature kept raising TypeError until a kernel restart.
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
# %matplotlib widget

import analytics_processing.analytics_constants as C
from CustomLogger import CustomLogger as Logger

from plotlib import session_data as sd
from plotlib import export
import behavior_plotting_lib as bpl
import neuro_plotting_lib as npl

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

plt.rcParams.update({'svg.fonttype': 'none'})   # keep text as text in the .svg, so the
                                                # poster panels stay editable in Inkscape


def show(result):
    """Display one figure exactly once, then drop it.

    Every cell below goes through this rather than calling plt.show(). A figure that is
    still open when the cell ends gets rendered a SECOND time by the inline backend's
    end-of-cell flush - which is where the duplicated outputs came from - and 30 open
    figures also hold their memory for the rest of the kernel's life. Displaying
    explicitly and closing immediately leaves the flush nothing to find."""
    display(result['fig'])
    plt.close(result['fig'])

# %%
nas_dir = C.device_paths()[0]
output_dir = os.path.join(nas_dir, 'Simon', 'poster_26')
Logger().init_logger(None, None, logging_level="INFO")

session_names, ith_to_dt_session = sd.get_session_names(paradigm_ids=(1100,), animal_ids=(6,))

# S0..S15 of rYL006/P1100: the sessions the poster covers. S0 is left out of the figures
# below (the sequence starts at S1, as in the poster draft); widen this slice to include it.
BEHAVIOR_S_IDS = range(1, 16)
behavior_session_names = [session_names[s_id] for s_id in BEHAVIOR_S_IDS]
behavior_session_names

# %%
# The trial table - one row per trial, `s_id` column holding the chronological session
# number. Shared by the kinematics cell (it is the source of truth for choice_R1/R2, see
# load_trackwise's `choices_from`) and by both correctness figures.
trialwise = sd.load_trialwise(behavior_session_names, ith_to_dt_session)
trialwise.groupby('s_id')[['both_R1_R2_rewarded', 'cue', 'choice_R1', 'choice_R1_correct']].mean().round(2)

# %%
# ===========================================================================
# Kinematics, skip vs stop - one figure (svg + png) per session x metric.
# ===========================================================================
# Trackwise is loaded per session rather than in one call: it is ~30-40k rows per session
# and nothing here compares sessions, so this keeps the cell re-runnable without holding
# 600k rows. The track illustration the plotly version drew on top is gone; the two reward
# zones are shaded instead (from the data's own track_zone column).
# 15 sessions x 2 metrics x (svg + png) is 60 files, so they go in one subfolder per metric
# rather than into the poster's top-level directory - the panels for a metric are picked out
# as a set, and the top level stays a short list of the figures actually going on the poster.
plt.close('all')
KINEMATICS_METRICS = ['RawYawPitch Sum Velocity', 'BallForward Velocity']

for s_id in BEHAVIOR_S_IDS:
    trackwise_s = sd.load_trackwise([session_names[s_id]], choices_from=trialwise)
    for metric in KINEMATICS_METRICS:
        metric_dir = os.path.join(output_dir, 'kinematics',
                                  bpl.KINEMATICS_METRICS[metric]['slug'])
        result = bpl.plot_session_kinematics(trackwise_s, s_id=s_id, metric=metric,
                                             groupby='choice_R1', save_dir=metric_dir)
        print(f"S{s_id:<2} {metric:<26} "
              f"{ {g['label']: g['n_trials'] for g in result['traces']['groups']} } "
              f"-> {os.path.basename(result['paths']['svg'])}")
        show(result)

# %%
# ===========================================================================
# Correctness across the session sequence - the whole sequence in one axes.
# ===========================================================================
# `reward='R1'`/'R2' picks the reward zone; the last pre-cue session is derived from the data
# (the last with both_R1_R2_rewarded == 1) unless split_s_id is passed.
#
# No trend line on R2's cue2 points: those are the trials where stopping at R2 was correct,
# and the animal essentially never did it - the group sits on the floor for the whole cued
# phase. A fitted line there would read as a trend in what is a flat failure to acquire the
# behaviour, so the points are shown and left to speak for themselves.
OMIT_FIT = {'R2': ('cue2',)}

plt.close('all')
correctness_dir = os.path.join(output_dir, 'correctness')
for reward in ('R1', 'R2'):
    result = bpl.plot_correctness_sequence(trialwise, reward=reward, s_ids=BEHAVIOR_S_IDS,
                                           omit_fit=OMIT_FIT.get(reward, ()),
                                           save_dir=correctness_dir)
    print(f"{reward}: cue contingency from S{result['split_s_id'] + 1}")
    print(result['table'].round(3).to_string())
    show(result)

# %%
# The same figure rotated for a portrait poster column: session down the y axis (S1 at the
# TOP, so the sequence reads downward), correctness along x. Same numbers, only the layout
# differs - including the same omitted R2 cue2 fit.
plt.close('all')
for reward in ('R1', 'R2'):
    result = bpl.plot_correctness_sequence(trialwise, reward=reward, s_ids=BEHAVIOR_S_IDS,
                                           session_axis='y', omit_fit=OMIT_FIT.get(reward, ()),
                                           save_dir=correctness_dir)
    show(result)

# %%

# %% [markdown]
# ## The neural grounding panel
#
# The first ephys figure of the poster, and deliberately the plainest one: no decoder, no
# subspace, just the population's own z-scored firing rate, split by cue, on the t0-aligned
# x-axis every later figure uses. Two intervals - the cue entry and the R1 choice - with the
# trials' own position traces above them, so the reader learns to read that x-axis here,
# where nothing is derived, before meeting it again under a decoding curve.
#
# Definitions are in `neuro_plotting_lib.py` (`plot_condition_panel`), which is also where
# the fuller working version of the same curves lives (`plot_condition_lines`, four
# intervals and the cue x choice_R1 grid - what `dim_analysis.py` calls).

# %%
# The three 40ms tables the panel needs, for the selected sessions only: FiringRate40msZ is
# minutes of I/O over all 34 sessions and this figure only ever draws a handful, so widening
# NEURAL_S_IDS means re-running this cell too. `trial_id_from` copies behaviour's own trial
# ids onto the rate table positionally - the two share the exact same 40ms ephys-bin grid,
# and load_firing_rate checks that rather than assuming it.
NEURAL_S_IDS = [9,11,12,13,14,15]            # S9 is the panel that goes on the poster. Pass a list to render
                              #   several - e.g. [9, 11, 12] - and the canvas cell at the
                              #   bottom glues whatever this rendered into one .svg
X_MODALITY = 'mPFC'           # which probe's units the curves average over

neural_session_names = [session_names[s_id] for s_id in NEURAL_S_IDS]
t0_events = sd.load_t0_events(neural_session_names)
behavior40 = sd.load_behavior(
    neural_session_names,
    # cue_visible is what marks the cue zone on the position row (see npl.cue_zone_extent -
    # it is phase-gated, so a pre-cue session correctly gets no band); frame_position is the
    # row itself; the rest are the trial labels the curves split on.
    # track_zone is what the R1 reward band on that row is measured from (npl.track_zone_
    # extents), the same way the behaviour kinematics figure gets its zone spans.
    columns=['trial_id', 'cue', 'trial_outcome', 'choice_R1', 'choice_R2',
             'to_ephys_timestamp', 'frame_position', 'cue_visible', 'track_zone'])
fr_z = sd.load_firing_rate(neural_session_names, trial_id_from=behavior40)
unit_cols = sd.unit_columns(fr_z, X_MODALITY)
fr_z = fr_z[unit_cols + ['trial_id']]
print(f"{len(unit_cols)} {X_MODALITY} units, "
      f"{len(t0_events.index.unique('session_id'))} session(s), "
      f"intervals available: {[c for c in t0_events.columns if c.endswith('_interval')]}")

# %%
# One panel per session. The knobs that matter for a poster panel are the two ylims: both
# CLIP, and both are fixed here rather than autoscaled, because a per-session autoscale
# silently rescales the axis under the reader from one column of the canvas to the next.
plt.close('all')
NEURAL_YLIM = (-0.12, 0.12)       # z-scored rate. The across-neuron mean sits on a ~5x
                                  #   smaller scale than a single unit would - see
                                  #   npl.draw_condition_lines
NEURAL_POSITION_YLIM = (-170, 130)  # cm, the track from its start to past the R1 zone
NEURAL_INTERVALS = ['cue_entry_interval']   # which t0 windows the panel shows, left to right.
                                  #   npl.PANEL_INTERVALS is the pair
                                  #   ['cue_entry_interval', 'R1_entry_interval']; every column
                                  #   of t0_events ending in '_interval' is available (cue_exit,
                                  #   R2_entry, reward1_sound, ITI, track_start). The time-scale
                                  #   tick adapts: a 1s tick needs an interval at least that
                                  #   long, and cue entry's own window is 0.96s, so a
                                  #   cue-entry-only panel gets a 0.5s tick instead
NEURAL_CONDITIONS = 'cue'         # two solid curves, orange/purple, the poster's own cue
                                  #   colours. ('cue', 'choice_R1') gives the full 2x2 as
                                  #   bicolour dashed lines - too much for this panel
NEURAL_SMOOTH = 3                 # bins of centered rolling mean, applied to each TRIAL's own
                                  #   trace before the average - so 200ms, and yes, that is
                                  #   why these curves are smoother than the native 40ms rate.
                                  #   Pass 1 for the unsmoothed thing
NEURAL_SEM_Z = 1.96               # the shaded band: 1.96 x SEM = the 95% CI of each curve's
                                  #   mean (1.0 would be the plain +-1 SEM this library's
                                  #   working figure draws)
neural_dir = os.path.join(output_dir, 'neural')

neural_svg_paths, neural_png_paths = [], []
for i, s_id in enumerate(NEURAL_S_IDS):
    s_name = session_names[s_id][:16]      # the id every 40ms analytic indexes by
    if s_name not in t0_events.index.unique('session_id'):
        print(f"S{s_id} has no t0 events, skipping")
        continue
    if s_name not in fr_z.index.unique('session_id'):
        print(f"S{s_id} has no ephys, skipping")
        continue
    result = npl.plot_condition_panel(
        t0_events, behavior40, fr_z, s_name,
        which_intervals=NEURAL_INTERVALS,
        unit_cols=unit_cols,
        conditions=NEURAL_CONDITIONS,
        smooth=NEURAL_SMOOTH,
        sem_z=NEURAL_SEM_Z,
        ylim=NEURAL_YLIM,
        position_ylim=NEURAL_POSITION_YLIM,
        label=f'S{s_id}',
        # No legend and no y axis labels, on any panel: they are identical in every column of
        # the canvas, and the poster names the axes once beside the assembled figure. Both are
        # off in the library's own defaults - pass show_legend=True / show_ylabel=True here for
        # a panel meant to be read on its own. The y tick NUMBERS go on the leftmost panel
        # only, for the same reason; the space they take is still reserved in the others, so
        # every column keeps the same axes geometry.
        show_yticklabels=(i == 0),
        save_dir=neural_dir, stem=f'S{s_id:02d}_{s_name}_condition_panel')
    print(f"S{s_id:<2} n per cue: "
          f"{ {lbl: n for lbl, n in zip(result['block']['row_labels'], result['block']['row_n'])} }"
          f" cue zone {result['cue_zone']} zones {result['zones']}")
    neural_svg_paths.append([result['paths']['svg']])
    neural_png_paths.append([result['paths']['png']])
    show(result)

# %%
# Glue the panels above into ONE figure - one column per session, left to right in
# NEURAL_S_IDS order. The .svg is the deliverable (each panel keeps its own vector content
# inside a nested <svg>, so the whole thing stays editable in Inkscape); the .png is for
# looking at the sequence quickly.
if neural_svg_paths:
    canvas_svg = os.path.join(neural_dir, 'condition_panel_canvas.svg')
    export.build_canvas_svg(neural_svg_paths, canvas_svg)
    export.build_canvas(neural_png_paths, canvas_svg.replace('.svg', '.png'))
    print(f"{len(neural_svg_paths)} panels -> {canvas_svg}")


# %%

# %% [markdown]
# ## CCGP representations
#
# The decoding column of the poster, and the CCGP table rather than the naive one: each W
# here was fit on ONE level of the other variable and scored on the other level, so a curve
# above chance is the claim the poster actually wants to make - the code generalises across
# the untrained condition, i.e. it is a *representation*, not a decodable mixture.
#
# Two figures, one per target, each over the same short session sequence (CCGP_S_IDS =
# S9/S11/S13/S14), each session exported as its own panel and then glued into one canvas:
#
#  * **cue** over the cue-entry interval - where the cue is actually presented
#  * **choice_R1** over the R1-entry interval - where the choice is made
#
# Every panel is two rows: the decoder's score (balanced accuracy) with its bootstrap CI
# band, and below it that column's own W-vs-W angle heatmap (every timebin's weight vector
# against every other's - red = aligned, blue = anti-aligned, white = ~90 deg, i.e. two
# directions with nothing to do with each other).
# Everything else `plot_session_analysis` can draw - the kinematics row, the variance and
# activation rows, the cross-column W comparison - is off here; those live in `post_lm4.py`,
# which is the working notebook for them. See there for the full knob list.
#
# Panels are exported with a **transparent** background (CCGP_TRANSPARENT_RC below), so the
# assembled .svg tiles without each panel's white page rectangle covering its neighbour.

# %%
# The CCGP fit + its two supporting tables, for the four selected sessions only. Same
# pattern as post_lm4.py's data cell (that notebook is the up-to-date reference for these
# calls) - narrowed to CCGP_S_IDS, since each of these is a per-session NAS read.
from analytics_processing import analytics
from svm_plotting_lib import (plot_session_analysis, w_session_summary_values,
                              plot_w_session_summary)
from plotlib.trial_labels import level_label

CCGP_S_IDS = [9, 11, 13, 14, 15]
# The CCGP table: L1, normalizer='none', targets cue and choice_R1, each fit twice (one
# direction per train_stratum of the other variable). It has no `b` column - which costs
# nothing here, since the activation row (the only thing needing the intercept) is off.
CCGP_ANALYTIC = 'SVML1CueOutcomeChoicePred2CCGP_noNorm'
CCGP_X_MODALITY = 'mPFC'

# Passed by name rather than through sd.t0_analytic_for: the _noNorm tables are not in
# ANALYTIC_PAIRS, and this is the current geometry, the one they were cut against.
ccgp_session_names = [session_names[s_id] for s_id in CCGP_S_IDS]
ccgp_fits = analytics.get_analytics(CCGP_ANALYTIC, session_names=ccgp_session_names)
ccgp_t0_events = sd.load_t0_events(ccgp_session_names, analytic='TrialWiseT0Events40ms2')
ccgp_behavior = sd.load_behavior(
    ccgp_session_names,
    columns=['trial_id', 'cue', 'trial_outcome', 'choice_R1', 'choice_R2',
             'to_ephys_timestamp', 'frame_position'],
    choice_r1_as_bool=True)

# Resolved from what actually arrived in all three tables rather than from CCGP_S_IDS: a
# session can come back empty from get_analytics (no fit for that target yet), and
# plot_session_analysis raises on an s_id missing from a table's index.
ccgp_session_ids = sorted(
    set(ccgp_fits.index.unique('session_id'))
    & set(ccgp_t0_events.index.unique('session_id'))
    & set(ccgp_behavior.index.unique('session_id')),
    key=lambda s_id: pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M'))
print(f"{len(ccgp_session_ids)} CCGP sessions: "
      f"{[f'S{ith_to_dt_session[s]}' for s in ccgp_session_ids]}, "
      f"targets: {sorted(ccgp_fits['predict_y_name'].unique())}")

# %%
# ===========================================================================
# Shared settings + the call wrapper - the two caller cells below differ only in
# target / interval / output name.
# ===========================================================================
# CCGP only: pins which of a target's two fitted directions is "the" one - the star in the
# score row's corner annotation, and the W feeding the angle heatmap. Without it the
# fallback rule (larger stratum wins) flips between sessions, which is what once made one
# session show the other direction in everything at once.
CCGP_STRATUM = {'cue': 1, 'choice_R1': 1}

# The two hand-picked W's the poster follows past the heatmaps - both from S13, the session
# where the cue and choice representations actually overlap (see the alignment panel below;
# its boxed runs are where these two timebins were read off). A selection is matched
# against the fitted table by whichever keys it sets (session_id/y_name/conditioner/
# train_stratum/interval/timebin - see svm_plotting_lib._selection_matches), so a CCGP
# target needs `train_stratum` as well: `conditioner` alone names which variable the fit
# was conditioned on, not which of the two directions, and would match both.
#
# Passed into the panels above (they draw a dotted callout box around that timebin's row,
# which is what points the poster reader at the W) AND resolved to the actual weight
# vectors below - the plots hand no data back, resolve_w_selections is the only way to get
# a W out of a selection.
POSTER_W_SELECTIONS = [
    # the choice direction at the end of the R1 approach, where the choice is made - and
    # the last timebin of S13's final co-active run (48-54 on the alignment panel's own
    # concatenated axis, i.e. R1 bins 17-23)
    {'session_id': '2024-12-04_18-06', 'y_name': 'choice_R1', 'conditioner': 'cue',
     'train_stratum': 1, 'interval': 'R1_entry_interval', 'timebin': 23,
     'name': 'poster26_S13_choiceEnd'},
    # the cue direction mid cue zone, inside S13's first and longest co-active run
    # (8-22 on that same axis)
    {'session_id': '2024-12-04_18-06', 'y_name': 'cue', 'conditioner': 'choice_R1',
     'train_stratum': 1, 'interval': 'cue_entry_interval', 'timebin': 9 + 4 + 5,
     'name': 'poster26_S13_cueMiddle'},
]

# Angle-heatmap colour, the two knobs that do different jobs (see
# svm_plotting_lib._angle_colormapping):
#   white_sigma widens the NEUTRAL zone - each colour is blended toward white by
#     exp(-d^2/2 sigma^2) in d = angle - 90, so the map is untouched away from 90 and only
#     the middle whitens. This replaces mask_chance_band (a hard cut at the +-21 deg mPFC
#     band), which is why that is off below.
#   clim sets where FULL colour is reached: bwr_r ramps linearly to full red at vmin, so
#     on the full 0-180 range a 45 deg angle is only half saturated and everything reads
#     as pale pink. None = the full range.
CCGP_ANGLE_WHITE_SIGMA = 8
CCGP_ANGLE_CLIM = (0, 180)

# Transparent panels. matplotlib writes the figure's (and each axes') background as a real
# white <rect> into the SVG, which on the assembled canvas sits ON TOP of the panel to its
# left - so the tiles cannot be nudged into overlap in Inkscape without erasing each other.
# 'none' makes matplotlib emit no patch at all (verified in the scratchpad: zero
# `fill: #ffffff` rects in the saved .svg). Applied via rc_context inside the wrapper
# rather than globally, so it holds no matter which cells were re-run in what order.
# The .png canvas is unaffected - build_canvas still composites onto white, which is what
# you want for a quick look at the sequence.
CCGP_TRANSPARENT_RC = {'figure.facecolor': 'none', 'axes.facecolor': 'none',
                       'savefig.facecolor': 'none'}

ccgp_dir = os.path.join(output_dir, 'ccgp')


def plot_ccgp_panels(y_name, which_intervals, canvas_stem=None, session_ids=None,
                     **overrides):
    """One `plot_session_analysis` panel per session for one CCGP target, then the canvas.

    A call wrapper only - the caller cells below hold what actually differs between them
    (target, interval, output name), not two copies of one long argument list that drift
    apart. Anything in the kwargs dict is overridable per call via **overrides, which is
    how min_metric / mask_chance_band / the extra rows get turned on for a look without
    editing this cell: e.g.

        plot_ccgp_panels('cue', ['cue_entry_interval'], min_metric=0.55, show_shuffle=True)

    Returns (png_paths, svg_paths) - one single-element list per session, the nesting
    build_canvas/build_canvas_svg expect (inner list = stacked vertically, outer = side by
    side).
    """
    session_ids = ccgp_session_ids if session_ids is None else session_ids
    canvas_stem = canvas_stem or f'ccgp_{y_name}_{CCGP_X_MODALITY}'
    # One subfolder per target, because plot_session_analysis names its files after the
    # SESSION alone (session_<id>_S9_main.svg) - both caller cells would otherwise write
    # the same four filenames and the second call would silently overwrite the first
    # target's panels on disk. The assembled canvases stay at the top of ccgp_dir, so that
    # level is the short list of figures actually going on the poster.
    save_dir = os.path.join(ccgp_dir, y_name)
    os.makedirs(save_dir, exist_ok=True)

    kwargs = dict(
        x_modalities=CCGP_X_MODALITY,
        y_predicted_names=[y_name],
        which_intervals=which_intervals,
        ith_to_dt_session=ith_to_dt_session,
        save_dir=save_dir,
        metric='balanced_accuracy',      # or 'f1'
        smooth=2,                        # rolling mean over timebins, per interval; None = raw
        show_aggr_ci=True,               # the CI band around the score line - bootstrapped at
                                         #   plot time from this row's own predictions
        show_mean=False,                 # True adds mean-across-bootstraps + its own CI
        show_shuffle=False,              # True adds the dashed permutation null
        min_metric=None,                 # e.g. 0.55: dim heatmap cells whose own decoder
                                         #   scored below this (the angle of a W nobody
                                         #   could decode with is not evidence)
        mask_chance_band=False,          # OFF: CCGP_ANGLE_WHITE_SIGMA does this job, smoothly
        angle_white_sigma=CCGP_ANGLE_WHITE_SIGMA,
        angle_clim=CCGP_ANGLE_CLIM,
        ccgp_w_train_stratum=CCGP_STRATUM,
        # The corner text under the score row: 'sentence' states the design in one row -
        # "Train on skip trials, Test on stop:  C1 vs C2", the two classes bold and in their
        # own colours - which is what a poster reader needs from a CCGP panel. 'counts' is
        # the two-line Train/Test listing with each stratum's own class balance instead, for
        # when the numbers themselves are what is being checked.
        ccgp_annotation='sentence',
        # the dotted callout boxes around the two selected W's timebins; a selection
        # only draws on the session AND target it names, so each panel gets at most one
        highlight_selections=POSTER_W_SELECTIONS,
        show_ccgp_secondary=False,       # True draws BOTH train directions as equal lines
                                         #   instead of the one elected above
        # Everything below is a row this figure deliberately does not have. post_lm4.py is
        # where they are being worked on; turning one on here also makes the panel taller,
        # so do it for both caller cells or the canvas columns stop matching.
        show_trial_kinematics=False,     # row 0 (position traces) - hidden, and its grid
                                         #   slot shrinks to a sliver rather than reflowing
        show_covariate_plot=False,
        show_w_variance=False,
        show_w_activation=False,
        show_w_comparison=False,         # a single target/column has nothing to compare
                                         #   against anyway - see plot_session_analysis
        show_colorbar=False,             # the angle scale goes on the poster once, by hand
        # No y-axis label and no y tick NUMBERS on any panel - the score row's scale is the
        # same in every column (0.25/0.5/0.75 gridlines, chance in the middle) and the
        # poster states it once, beside the assembled canvas. This is also what makes the
        # panels pack: the label needs 1in of reserved margin, so dropping it takes a
        # one-interval panel from 2.4in wide down to 1.4in.
        # show_ylabel=True brings both back - for ALL panels, since a panel that has the
        # label ends up with a WIDER plotting area than one that doesn't (tight_layout
        # hands it the freed margin: measured 108.3pt against 69.2pt), and mixed columns
        # then no longer line up.
        show_ylabel=False,
        score_ylabel='Balanced accuracy',  # the text IF show_ylabel is turned back on -
                                           #   short, because the auto label ('Decode
                                           #   choice_R1 Balanced Accuracy') is longer than
                                           #   the row is tall and gets clipped
        show_title=False,                # per-column "<target> decoding from mPFC" - the
                                         #   poster names it once, beside the panel
    )
    kwargs.update(overrides)

    all_png, all_svg = [], []
    with plt.rc_context(CCGP_TRANSPARENT_RC):
        for s_id in session_ids:
            png_paths, svg_paths = plot_session_analysis(
                s_id, ccgp_t0_events, ccgp_behavior, ccgp_fits, **kwargs)
            print(f"S{ith_to_dt_session[s_id]:<2} {y_name:<10} -> "
                  f"{os.path.basename(png_paths[0]) if png_paths else '(not saved)'}")
            if png_paths:
                all_png.append(png_paths)
                all_svg.append(svg_paths)

    if all_png:
        # The .svg is the deliverable (vector, tileable in Inkscape); the .png is for
        # looking at the sequence quickly.
        export.build_canvas(all_png, os.path.join(ccgp_dir, f'{canvas_stem}.png'))
        export.build_canvas_svg(all_svg, os.path.join(ccgp_dir, f'{canvas_stem}.svg'))
        print(f"{len(all_svg)} panels -> {os.path.join(ccgp_dir, canvas_stem + '.svg')}")
    return all_png, all_svg


# %%
# ===========================================================================
# cue, at cue entry
# ===========================================================================
# Only the cue-entry interval: the cue is decodable where it is presented, and following it
# into the R1 approach mostly plots the post-cue decay. Add 'cue_exit_interval' /
# 'R1_entry_interval' to the list to see those too - the panel widens per interval, so both
# caller cells should get the same number of them if the two canvases are to be stacked.
plt.close('all')
cue_ccgp_png, cue_ccgp_svg = plot_ccgp_panels(
    'cue', ['cue_entry_interval', 'R1_entry_interval'],
    # --- the usual overrides; uncomment to try one
    # show_ylabel=True,             # y label + tick numbers back, on every panel
    min_metric=0.55,
    # mask_chance_band=True,
    show_shuffle=True,
    # show_ccgp_secondary=True,
    show_colorbar=True
)

# %%
# ===========================================================================
# choice_R1, at the R1 choice
# ===========================================================================
plt.close('all')
choice_ccgp_png, choice_ccgp_svg = plot_ccgp_panels(
    'choice_R1', ['cue_entry_interval', 'R1_entry_interval'],
    min_metric=0.55,
    # mask_chance_band=True,
    show_shuffle=True,
    # show_ccgp_secondary=True,
    # show_colorbar=True
)

# %%
# The two selected W's themselves, out of the same table the panels above were drawn from.
# The panels only DREW the callout boxes - a plot call and a data query are different
# questions, so nothing came back from them; this is the query. `ccgp_w_train_stratum` has
# to be the same election the figures used, or a selection that doesn't pin its own
# train_stratum would match both CCGP directions and resolve to nothing.
#
# One row per resolved selection, carrying that row's whole fit - `w` (the beta, from the
# _noNorm table, i.e. fitted without the per-sample L2), `acc_aggr`, `normalizer`,
# `interval_name`, `timebin` - which is what every W figure at the end of this notebook
# takes as its `w_row`.
from svm_plotting_lib import resolve_w_selections

selected_w_rows_df = resolve_w_selections(POSTER_W_SELECTIONS, ccgp_fits,
                                          x_modality=CCGP_X_MODALITY,
                                          ccgp_w_train_stratum=CCGP_STRATUM)
# defensively, by what is actually in the table: an older cached fit has no `normalizer`
# (and then fit_samples falls back to LEGACY_POSTPROC_MODE with a printed notice), and a
# naive-fit table has no train_stratum at all.
w_summary_cols = ['name', 'session_id', 'predict_y_name', 'conditioner', 'train_stratum',
                  'interval_name', 'timebin', 'acc_aggr', 'normalizer']
display(selected_w_rows_df[[c for c in w_summary_cols if c in selected_w_rows_df.columns]])

# %%

# %% [markdown]
# ### Cue vs choice: co-active, and aligned?
#
# Same sessions, same panel format, one question further on. The two panels above each show
# one representation on its own; this one puts them on one timeline and asks the two things
# that follow:
#
#  * **when** is each of them there at all - the `> min_metric` strip that runs along the
#    bottom of the score row above, here promoted to be the whole top row: cue in its own
#    orange/purple, choice_R1 in its two blues, one strip above the other. A run of at least
#    `co_active_min_bins` timebins where BOTH clear the threshold gets a box.
#  * **do they point the same way** where they overlap - the heatmap below is every cue
#    timebin's W (y) against every choice timebin's W (x), the same angle colours as the
#    per-target heatmaps above (red = aligned, blue = anti-aligned, white = orthogonal,
#    i.e. two directions with nothing to do with each other). The angle is meaningful
#    because both W's are vectors over the *same units* of the same mPFC population.
#
# The box from the top row is redrawn on the heatmap's **diagonal**, spanning that run on
# both axes: those cells are the two decoders compared at the same moment, so how red /
# blue / white that square is *is* the answer for the co-active window. Everything off the
# diagonal is the same comparison across time lags, and everything outside the boxes is a
# comparison at moments when at least one of the two wasn't there to compare.
#
# No score lines here - they are read off the panels above; two curves plus two strips in a
# row this short just reads as four curves.

# %%
# ===========================================================================
# cue vs choice_R1 alignment - one panel per session, then the canvas
# ===========================================================================
from svm_plotting_lib import plot_session_cue_choice_alignment

align_dir = os.path.join(ccgp_dir, 'alignment')


def plot_alignment_panels(which_intervals, canvas_stem=None, session_ids=None, **overrides):
    """One `plot_session_cue_choice_alignment` panel per session, then the canvas.

    Same shape as `plot_ccgp_panels` above (and the same transparent-background export, so
    the panels tile in Inkscape) - the shared settings live here, anything else goes in
    **overrides for a look without editing this cell, e.g.

        plot_alignment_panels(['R1_entry_interval'], min_metric=0.6, co_active_min_bins=5)
    """
    session_ids = ccgp_session_ids if session_ids is None else session_ids
    canvas_stem = canvas_stem or f'alignment_cue_choice_{CCGP_X_MODALITY}'
    os.makedirs(align_dir, exist_ok=True)

    kwargs = dict(
        x_modality=CCGP_X_MODALITY,
        which_intervals=which_intervals,
        # (top strip / heatmap y axis, bottom strip / heatmap x axis)
        y_names=('cue', 'choice_R1'),
        metric='balanced_accuracy',
        smooth=2,                        # same rolling mean the score rows above use, so a
                                         #   strip switches off where the drawn curve dips,
                                         #   not on single noisy timebins
        min_metric=0.55,                 # what both rows key off: the strips ARE this
                                         #   threshold, and the heatmap dims every cell whose
                                         #   own decoder fell below it
        co_active_min_bins=3,            # consecutive both-above timebins before a run gets
                                         #   a box - 3 x 40ms = 120ms; a single co-active bin
                                         #   is what two noisy curves produce all the time
        ccgp_w_train_stratum=CCGP_STRATUM,   # same elected direction per target as above
        mask_chance_band=False,          # OFF, same as the panels above: the white core does
                                         #   this job smoothly
        angle_white_sigma=CCGP_ANGLE_WHITE_SIGMA,
        angle_clim=CCGP_ANGLE_CLIM,
        ith_to_dt_session=ith_to_dt_session,
        save_dir=align_dir,
        show_ylabel=False,               # True names which target is on which heatmap axis
                                         #   (and costs 1in of width, so all-or-nothing
                                         #   across panels - see plot_ccgp_panels)
        show_annotation=True,            # the small "bal. acc > 0.55: C1/C2 skip/stop" key
                                         #   above the strips, in the strips' own colours
        show_boxes=True,                 # False = the raw comparison, no co-active boxes
        show_colorbar=False,
    )
    kwargs.update(overrides)

    all_png, all_svg = [], []
    with plt.rc_context(CCGP_TRANSPARENT_RC):
        for s_id in session_ids:
            png_paths, svg_paths = plot_session_cue_choice_alignment(
                s_id, ccgp_t0_events, ccgp_fits, **kwargs)
            if png_paths:
                all_png.append(png_paths)
                all_svg.append(svg_paths)

    if all_png:
        export.build_canvas(all_png, os.path.join(ccgp_dir, f'{canvas_stem}.png'))
        export.build_canvas_svg(all_svg, os.path.join(ccgp_dir, f'{canvas_stem}.svg'))
        print(f"{len(all_svg)} panels -> {os.path.join(ccgp_dir, canvas_stem + '.svg')}")
    return all_png, all_svg


# %%
# Both intervals, so the cue window and the choice window are on the same timeline - which
# is what makes a co-active run mean something. Drop to one interval for a narrower panel.
plt.close('all')
align_png, align_svg = plot_alignment_panels(
    ['cue_entry_interval', 'R1_entry_interval'],
    # --- the usual overrides; uncomment to try one
    # min_metric=0.6,
    co_active_min_bins=3,
    # show_ylabel=True,          # axis names back, on every panel
    # show_boxes=False,
    # show_colorbar=True,
)

# %% [markdown]
# ### The session summary - two panels over the whole sequence
#
# The same W's as the panels above, collapsed over time and read across sessions instead of
# within one: **how much of the population's variance each fitted W explains** on top, and
# **the score that same W's classifier reached** below, sharing one x axis of session ids.
#
# The two rows belong together on one axis because the top one is only interpretable
# through the bottom one - a direction explaining a lot of variance in a session whose
# decoder sat at chance is not a task axis, it is a random-ish direction that happens to
# lie in a big mode - and that is exactly the comparison people get wrong when the two are
# separate figures on separate scales.
#
# Each session contributes a cloud of its own timebins per series; the joined markers are
# the median of that cloud, so the trend line is visibly resting on something (or not).
# Definitions are in `svm_plotting_lib` (`w_session_summary_values` computes,
# `plot_w_session_summary` draws).

# %%
# Data for the summary, over a LONGER sequence than the four panel sessions: this figure is
# about a trend, and four points barely make one. The firing rate is the expensive part -
# one FiringRate40msZ read per session, minutes over the whole range - so it is loaded here,
# once, rather than inside the plot call, and SUMMARY_LOAD_FR=False turns it off to get the
# score panel alone at no I/O cost at all.
SUMMARY_S_IDS = list(range(8, 16))     # S8..S15; sessions with no CCGP fit are skipped
SUMMARY_LOAD_FR = True

summary_session_names = [session_names[s_id] for s_id in SUMMARY_S_IDS]
summary_fits = {CCGP_ANALYTIC: analytics.get_analytics(CCGP_ANALYTIC,
                                                       session_names=summary_session_names)}
summary_t0_events = sd.load_t0_events(summary_session_names, analytic='TrialWiseT0Events40ms2')
summary_session_ids = [sn[:16] for sn in summary_session_names
                       if sn[:16] in summary_fits[CCGP_ANALYTIC].index.unique('session_id')
                       and sn[:16] in summary_t0_events.index.unique('session_id')]

summary_fr = {}
if SUMMARY_LOAD_FR:
    for s_id in summary_session_ids:
        full_name = next(sn for sn in summary_session_names if sn.startswith(s_id))
        summary_fr[s_id] = sd.session_fr_for_modality(
            analytics.get_analytics('FiringRate40msZ', session_names=[full_name])
            .xs(s_id, level='session_id'), CCGP_X_MODALITY)
print(f"{len(summary_session_ids)} sessions: "
      f"{[f'S{ith_to_dt_session[s]:02d}' for s in summary_session_ids]}"
      f"{' (+ firing rate)' if summary_fr else ' (score panel only)'}")

# %%
# The conditioner of each CCGP target is the OTHER target in this table (a cue direction is
# fit within one choice level and vice versa) - needed only to name a stratum in a legend.
CCGP_CONDITIONER = {'cue': 'choice_R1', 'choice_R1': 'cue'}
SUMMARY_INTERVALS = ['cue_entry_interval']   # the cue zone by default; add
                                             #   'R1_entry_interval' for both


def plot_ccgp_summary(y_name, which_intervals=None, both_strata=False, **overrides):
    """The two-panel session summary for one CCGP target.

    `both_strata=True` adds the OTHER trained-on direction as a second series, its own
    column beside the first with a DASHED stat mark - the pair reads as one target seen
    from its two CCGP directions rather than as two different things, which colouring them
    differently would imply. That is also the honest way to look at an elected stratum: the
    election (CCGP_STRATUM) pins which direction every other figure in this notebook shows,
    and this says what the discarded one was doing.
    """
    which_intervals = SUMMARY_INTERVALS if which_intervals is None else which_intervals
    conditioner = CCGP_CONDITIONER[y_name]
    elected = CCGP_STRATUM[y_name]
    series = [dict(analytic=CCGP_ANALYTIC, y_name=y_name, train_stratum=elected,
                   label=f'train {level_label(conditioner, elected)}')]
    if both_strata:
        # its median alone, as a hollow circle, with no cloud of its own: the discarded
        # direction is a reference for the elected one, and drawing its every timebin next
        # to the same session's other cloud doubles the ink for a comparison of two levels
        series.append(dict(analytic=CCGP_ANALYTIC, y_name=y_name, train_stratum=1 - elected,
                           show_points=False, stat_marker='o',
                           linestyle='--', label=f'train {level_label(conditioner, 1 - elected)}'))

    kwargs = dict(
        stat='median',                 # 'mean' sits a few tenths higher - these clouds are
                                       #   right-skewed
        metric='balanced_accuracy',
        min_metric=None,               # e.g. 0.55: drop timebins whose own classifier never
                                       #   worked, so the variance panel is about decoder
                                       #   axes rather than about arbitrary directions
        title=y_name,
    )
    kwargs.update(overrides)
    value_keys = ('min_metric', 'metric')
    values = w_session_summary_values(
        summary_session_ids, series, which_intervals, summary_fits, summary_t0_events,
        fr_by_session=summary_fr or None, x_modality=CCGP_X_MODALITY,
        ith_to_dt_session=ith_to_dt_session,
        **{k: kwargs[k] for k in value_keys if k in kwargs})
    with plt.rc_context(CCGP_TRANSPARENT_RC):
        return plot_w_session_summary(
            values, series, which_intervals, save_dir=ccgp_dir,
            stem=f'sessionSummary_{y_name}_{CCGP_X_MODALITY}',
            **{k: v for k, v in kwargs.items() if k not in ('min_metric',)})


# %%
plt.close('all')
cue_summary = plot_ccgp_summary(
    'cue',
    # both_strata=True,              # the discarded CCGP direction too, dashed
    # min_metric=0.55,
    # which_intervals=['cue_entry_interval', 'R1_entry_interval'],
)
print(cue_summary['values'].groupby(['series', 'label'], observed=True)[['var_explained', 'metric']]
      .median().round(2))
show(cue_summary)

# %%
plt.close('all')
choice_summary = plot_ccgp_summary(
    'choice_R1',
    # both_strata=True,
    # min_metric=0.55,
)
show(choice_summary)

# %%

# %% [markdown]
# ## What those two W's are, and what the population does along them
#
# The panels above stop at "there is a direction that decodes this". These last cells follow
# the two W's picked out of S13 (`POSTER_W_SELECTIONS`, resolved into `selected_w_rows_df`
# right after the CCGP callers) one step further, and ask two different questions of them:
#
#  * **what the direction is** - `plot_W_list`: each W's weight profile over the neurons,
#    one thin stem panel each, plus the pairwise angle between them underneath. Two
#    hand-picked W's read cell by cell, which is why that little heatmap prints its angles
#    and is deliberately *not* alpha-masked the way the big comparison heatmaps are.
#  * **what the population does along it** - `plot_w_projection_splits`: the trials'
#    activity dotted onto that fixed W, over the same t0-aligned x axis as every panel
#    above, split four ways. This is the part that can say something the decoding panels
#    cannot: a direction fit to separate cue1 from cue2 either does or does not also
#    separate the choices, their interaction, and right from wrong trials.
#
# Both use the **beta from the _noNorm fit** (`CCGP_ANALYTIC`), and the projection follows
# that provenance automatically: `normalize='auto'` reads the row's own `normalizer` column
# and resolves to fit parity for these tables (the 3-native-bin fit-window mean, no
# per-sample L2 - see `plotlib/fit_samples.py`). The consequence to keep in mind is that
# the y scale then carries the session's own feature-vector norm, so it is comparable
# across the ROWS of a panel (same trials, same W) but not across sessions - each panel is
# read on its own scale. The same tables store no intercept, so the dotted y=0 is the
# direction's neutral point, not the classifier's boundary.

# %%
# ===========================================================================
# the W's themselves: profile + the angle between them
# ===========================================================================
from w_plotting_lib import plot_W_list, plot_w_projection_splits

w_dir = os.path.join(ccgp_dir, 'w')
os.makedirs(w_dir, exist_ok=True)

plt.close('all')
plot_W_list(
    selected_w_rows_df,
    plot_type='stem',        # 'line' for the profile's shape alone, 'bar' for filled bars
    min_abs_w=0.0,           # leave near-zero weights blank rather than drawing a sliver
                             #   (L1 fits are sparse, so this is most of them)
    ith_to_dt_session=ith_to_dt_session,
    save_path=os.path.join(w_dir, f'posterWs_{CCGP_X_MODALITY}.png'),
    save_path_svg=os.path.join(w_dir, f'posterWs_{CCGP_X_MODALITY}.svg'),
)

# %%
# ===========================================================================
# the projection: one fixed W, four splits
# ===========================================================================
# One panel per (W, session), written as CHRONOLOGICAL session numbers (the S<n> every
# figure in this notebook is labelled with) rather than 16-char session ids -
# `projection_session_ids` below takes either, so `[13, 14, 15]` here and
# `session_ids=[9, 11]` in a caller cell both work.
#
# The W stays fixed while the activity changes: S13 alone is "what did THIS session's
# population do along this direction", and adding the sessions after it is the drift
# question - does the same direction still mean anything once the animal has moved on. A
# session outside the CCGP set works too: nothing here needs that session's own fit, only
# its rates and its t0 windows, and both are loaded on demand below.
PROJECTION_S_IDS = [13, 14, 15]

PROJECTION_INTERVALS = ['cue_entry_interval', 'R1_entry_interval', 'ITI_interval']   # the same three intervals as the CCGP panels above, so the

proj_dir = os.path.join(ccgp_dir, 'projection')

# One FiringRate40msZ read per session, held here rather than inside the plot call: the
# library still fetches its own rates when not handed any (see w_plotting_lib's KNOWN
# DEBT), which for two W's into the same session means reading the same table twice.
projection_fr = {}

# t0 windows for whichever sessions get asked for. Starts as the CCGP set's own table (the
# five sessions the panels above were drawn from) and grows: projecting a fixed W into a
# session that was never CCGP-fit is a perfectly good question, and needs only that
# session's trial windows, not its fit.
projection_t0_events = ccgp_t0_events


def projection_session_ids(session_ids):
    """Chronological session numbers -> the 16-char session ids the libraries take.

    Accepts either form (and any mix), so a caller cell can say `[13, 14, 15]` - the S<n>
    the panels are labelled with - instead of pasting date strings. Raises on a number
    outside the loaded sequence rather than silently plotting the wrong session.
    """
    resolved = []
    for s in session_ids:
        if isinstance(s, str):
            resolved.append(s)
            continue
        if not 0 <= s < len(session_names):
            raise ValueError(f"S{s} is outside the loaded session list (S0..S{len(session_names) - 1}); "
                             f"widen sd.get_session_names' paradigm/animal filter to reach it")
        resolved.append(session_names[s][:16])
    return resolved


def projection_full_name(s_id):
    return next(sn for sn in session_names if sn.startswith(s_id))


def projection_fr_for(s_id):
    if s_id not in projection_fr:
        projection_fr[s_id] = sd.session_fr_for_modality(
            analytics.get_analytics('FiringRate40msZ', session_names=[projection_full_name(s_id)])
            .xs(s_id, level='session_id'), CCGP_X_MODALITY)
    return projection_fr[s_id]


def projection_t0_for(s_id):
    """The t0 table to hand the plot call, with `s_id`'s own rows in it - loaded and kept
    if this session isn't one of the five the CCGP cells already read."""
    global projection_t0_events
    if s_id not in projection_t0_events.index.unique('session_id'):
        projection_t0_events = pd.concat([
            projection_t0_events,
            sd.load_t0_events([projection_full_name(s_id)], analytic='TrialWiseT0Events40ms2')])
    return projection_t0_events


def plot_projection_panels(w_name, session_ids=None, which_intervals=None, canvas_stem=None,
                           **overrides):
    """One `plot_w_projection_splits` panel per session for ONE selected W, then the canvas.

    Same shape as `plot_ccgp_panels`/`plot_alignment_panels` above, including the
    transparent-background export, so these tile with them in Inkscape. `w_name` is a
    `name` from POSTER_W_SELECTIONS; `session_ids` is chronological session NUMBERS
    (or ids - see projection_session_ids), defaulting to PROJECTION_S_IDS.

    Note what is and isn't shared across the panels of one canvas: the W is (that is the
    point), the y scale is NOT - each panel scales to its own drawn lines, and these are
    noNorm betas whose projection carries each session's own feature-vector norm. Pass
    `projection_ylim=(lo, hi)` to read the sequence on one scale.
    """
    w_row = selected_w_rows_df[selected_w_rows_df['name'] == w_name].iloc[0]
    session_ids = projection_session_ids(PROJECTION_S_IDS if session_ids is None else session_ids)
    which_intervals = PROJECTION_INTERVALS if which_intervals is None else which_intervals
    save_dir = os.path.join(proj_dir, w_name)   # per W, or several W's into the same
    os.makedirs(save_dir, exist_ok=True)        #   session overwrite each other's files

    kwargs = dict(
        which_intervals=which_intervals,
        splits=('cue', 'choice_R1', ('cue', 'choice_R1'), 'correct_R1'),
                                         # one row each, in this order; a TUPLE is the
                                         #   joint split - one bicolour-dashed line per
                                         #   cell of the 2x2, the same encoding the
                                         #   condition panel gives a joint condition
        smooth=2,                        # same rolling mean over native bins as the score
                                         #   rows above
        show_variance=True,              # +/-0.5 SD across trials (spread, not a CI)
        projection_ylim=None,            # None = the range of the drawn group lines/bands,
                                         #   shared by all four rows; pass a pair to pin it
        show_ylabel=False,               # True adds the label + tick numbers, at 1in of
                                         #   width - all-or-nothing across panels, see
                                         #   plot_ccgp_panels
        show_title=False,                # the small "projected onto S13's cue W @ ..." line
                                         #   above the top row; the big bold S<n> header is
                                         #   always there. Worth turning ON for a canvas
                                         #   spanning sessions - the header is then the
                                         #   ACTIVITY's session, not the W's
        show_key=True,                   # the coloured 'C1/C2 x skip/stop' row keys
        ith_to_dt_session=ith_to_dt_session,
        save_dir=save_dir,
        save_suffix=f'_{w_name}',
    )
    kwargs.update(overrides)

    all_png, all_svg = [], []
    with plt.rc_context(CCGP_TRANSPARENT_RC):
        for s_id in session_ids:
            png_paths, svg_paths = plot_w_projection_splits(
                s_id, w_row, projection_t0_for(s_id), session_names,
                fr_x=projection_fr_for(s_id), **kwargs)
            if png_paths:
                all_png.append(png_paths)
                all_svg.append(svg_paths)

    if all_png and len(all_png) > 1:
        canvas_stem = canvas_stem or f'projection_{w_name}'
        export.build_canvas(all_png, os.path.join(ccgp_dir, f'{canvas_stem}.png'))
        export.build_canvas_svg(all_svg, os.path.join(ccgp_dir, f'{canvas_stem}.svg'))
        print(f"{len(all_svg)} panels -> {os.path.join(ccgp_dir, canvas_stem + '.svg')}")
    return all_png, all_svg


# %%
# The cue direction, mid cue zone - inside the co-active run the alignment panel boxes.
plt.close('all')
cue_proj_png, cue_proj_svg = plot_projection_panels(
    'poster26_S13_cueMiddle',
    # --- the usual overrides; uncomment to try one
    # show_ylabel=True,                      # y label + tick numbers back
    # show_title=True,                       # names the W and where it came from
    # projection_ylim=(-1.25, 1.25),         # one pinned scale for every panel - worth it
                                             #   as soon as the canvas spans sessions
    # splits=('cue', ('cue', 'choice_R1')),  # fewer rows
    # session_ids=[9, 11, 13, 14, 15],       # any sessions, by their S<n> number
)

# %%
# The choice direction, at the end of the R1 approach.
plt.close('all')
choice_proj_png, choice_proj_svg = plot_projection_panels(
    'poster26_S13_choiceEnd',
    # show_ylabel=True,
    # session_ids=[9, 11, 13, 14, 15],
)

# %%

# %% [markdown]
# ### EXPERIMENTAL: the same rows, one neuron at a time
#
# The projection above is a weighted sum over ~50 units, which is exactly what makes it a
# population claim and exactly what makes it uninterpretable at the level of "what is this
# cell doing". `plot_W_list` says which units carry the W (L1 leaves only a handful with
# |w| > 0.1); this draws those units' **own** activity through the identical split rows, on
# the identical t0-aligned grid, so a row here can be read straight against the same row of
# the projection panel.
#
# Two things to be aware of before reading it:
#
#  * **Squared z deletes the sign.** `z**2` is "how far this unit is from its own session
#    mean", up and down merged - and most of these weights are negative, so a unit that goes
#    DOWN on cue2 and one that goes UP look the same here. It also has no zero (`E[z²] =
#    var + mean²`), so a merely noisy unit reads as active, and a condition with fewer
#    trials reads as *more* active. Set `UNIT_TRANSFORM = 'z'` for the signed rate, which is
#    the one that answers "is this cell higher on cue2" - the two together are worth more
#    than either. (Same argument as for not squaring a supervised W's projection: the sign
#    is the interpretable part. Here z's sign is grounded too - z > 0 is above this unit's
#    own session average.)
#  * These are **single units against single-trial noise**, so the +/-0.5 SD bands are wide
#    and a difference has no test behind it. This is a look, not a result.
#
# The unit indices are positions in the W / columns of the mPFC slice - the same axis
# `plot_W_list` labels 'Neuron ID' - and each panel's subtitle carries that unit's own
# weight in the W, which is the link back to the population figure.

# %%
import warnings

from w_plotting_lib import (get_trial_w_projection, split_projection_curves, draw_split_rows,
                            _curves_ylim)
from plotlib import fit_samples as fitspl
from plotlib.trial_labels import trial_covariate_levels

UNIT_IDS = [1, 8, 12]        # the units carrying this W - read off plot_W_list above
UNIT_TRANSFORM = 'square'    # 'square' -> z**2 (magnitude only), 'z' -> the signed rate

unit_dir = os.path.join(ccgp_dir, 'units')


def plot_unit_split_panels(units=None, w_name='poster26_S13_cueMiddle', session_ids=None,
                           which_intervals=None, transform=None, splits=None, smooth=2,
                           spread=None, ylim=None, show_ylabel=True, show_title=True,
                           canvas_stem=None, save_dir=None):
    """One `draw_split_rows` panel per (unit, session), then the canvas - units across,
    sessions down.

    Reuses the projection figure's own machinery rather than a second implementation:
    projecting activity onto a ONE-HOT vector is that unit's own value, so
    `get_trial_w_projection` gives the unit's rate through the exact same windowing and
    preprocessing (`mode_for_w_row` off the named W, i.e. the 3-native-bin fit-window mean
    for these noNorm fits) as the population rows, on the same native-bin x axis.

    `w_name` is only used for the preprocessing mode and to look each unit's weight up for
    the subtitle - the traces themselves are that unit's activity, no W involved.
    `ylim=None` (default) gives each panel one range across its own four rows (rows showing
    the same quantity have to share a scale to be readable against each other), and
    `show_ylabel=True`
    (also the default, unlike the poster panels) then prints the numbers - three different
    neurons have genuinely different dynamic ranges, and one flat panel next to one tall one
    says nothing about either. `ylim='shared'` puts the whole call on one range instead,
    which is the comparison "how much does each unit move" - but be aware one unit can own
    that range: for [1, 8, 12] here, unit 8's ITI excursion (z**2 up to 2.2) against unit
    12's 0.25-0.47 leaves the other two panels visibly flat. A pair pins it by hand.

    `spread` defaults to `'sem'` for the squared transform and `'sd'` for the signed one:
    z**2's SD scales with its own mean, so the projection panels' +/-0.5 SD band runs
    several times the curve and dips below zero, and on a shared y range one noisy unit's
    band then flattens every other panel (measured here: a shared range of -3.1..15.6 for
    curves that live in 0.2..2.2). The standard error of the group mean is the band that
    actually matches what these rows compare.
    """
    units = UNIT_IDS if units is None else units
    transform = UNIT_TRANSFORM if transform is None else transform
    session_ids = projection_session_ids(PROJECTION_S_IDS if session_ids is None else session_ids)
    which_intervals = PROJECTION_INTERVALS if which_intervals is None else which_intervals
    splits = ('cue', 'choice_R1', ('cue', 'choice_R1'), 'correct_R1') if splits is None else splits
    spread = ('sem' if transform == 'square' else 'sd') if spread is None else spread
    save_dir = os.path.join(unit_dir, transform) if save_dir is None else save_dir
    os.makedirs(save_dir, exist_ok=True)

    w_row = selected_w_rows_df[selected_w_rows_df['name'] == w_name].iloc[0]
    w = np.asarray(w_row['w'], dtype=float)
    mode = fitspl.mode_for_w_row(w_row)
    bad = [u for u in units if not 0 <= u < len(w)]
    if bad:
        raise ValueError(f"unit(s) {bad} outside this W's {len(w)} features")

    # pass 1: every panel's curves, so the shared y range can be measured before drawing
    curves = {}
    for s_id in session_ids:
        t0_events = projection_t0_for(s_id)
        fr_x = projection_fr_for(s_id)
        for unit in units:
            onehot = np.zeros(len(w))
            onehot[unit] = 1.0
            values, trial_ids = get_trial_w_projection(s_id, t0_events, fr_x, which_intervals,
                                                       onehot, normalize=mode, b=0.0)
            if transform == 'square':
                values = {iv: arr ** 2 for iv, arr in values.items()}
            elif transform != 'z':
                raise ValueError(f"transform must be 'square' or 'z', got {transform!r}")
            curves[(s_id, unit)] = [split_projection_curves(values, trial_ids, t0_events, s_id,
                                                            split, which_intervals, smooth=smooth,
                                                            spread=spread)
                                    for split in splits]

    # 'shared' = one range over every panel of the call; None = one range per panel,
    # measured across that panel's own four rows (the same thing plot_w_projection_splits
    # means by projection_ylim=None - rows showing the same quantity have to share a scale
    # or the four are not readable against each other); a pair = pinned by hand.
    def panel_ylim_for(key):
        if ylim == 'shared':
            return _curves_ylim([g for groups in curves.values() for g in groups])
        return _curves_ylim(curves[key]) if ylim is None else ylim

    # pass 2: draw, columns = units (build_canvas' outer list), rows = sessions (inner)
    ylabel = 'z$^2$' if transform == 'square' else 'z-scored rate'
    band_label = {'sd': '+/-0.5 SD', 'sem': '+/-SEM', None: 'no band'}[spread]
    all_png, all_svg = [], []
    with plt.rc_context(CCGP_TRANSPARENT_RC):
        for unit in units:
            col_png, col_svg = [], []
            for s_id in session_ids:
                subtitle = (f"unit {unit} alone ({ylabel}, {band_label}), "
                            f"w={w[unit]:+.2f} in '{w_name}'" if show_title else None)
                png, svg = draw_split_rows(
                    curves[(s_id, unit)], splits, projection_t0_for(s_id), s_id, which_intervals,
                    ylim=panel_ylim_for((s_id, unit)), show_ylabel=show_ylabel, ylabel=ylabel,
                    subtitle=subtitle,
                    ith_to_dt_session=ith_to_dt_session, save_dir=save_dir,
                    stem=f'session_{s_id}_S{ith_to_dt_session[s_id]}_unit{unit}_{transform}')
                col_png += png
                col_svg += svg
            all_png.append(col_png)
            all_svg.append(col_svg)

    if any(all_png):
        canvas_stem = canvas_stem or f'units_{"-".join(map(str, units))}_{transform}'
        export.build_canvas(all_png, os.path.join(ccgp_dir, f'{canvas_stem}.png'))
        export.build_canvas_svg(all_svg, os.path.join(ccgp_dir, f'{canvas_stem}.svg'))
        print(f"{sum(len(c) for c in all_svg)} panels -> "
              f"{os.path.join(ccgp_dir, canvas_stem + '.svg')}")
    return all_png, all_svg


# %%
# The three units, S13 only. Add sessions with session_ids=[13, 14, 15] (they stack
# vertically under each unit's column).
plt.close('all')
unit_png, unit_svg = plot_unit_split_panels(
    [1, 8, 12], session_ids=[13,14,15],
    # --- the usual overrides; uncomment to try one
    # transform='z',                   # the SIGNED rate - the one that says up or down
    # w_name='poster26_S13_choiceEnd', # look the weights up in the choice W instead
    # spread='sd',                     # trial spread instead of the mean's error
    # ylim='shared',                   # one range for all three - see the docstring, unit 8
                                       #   owns it and flattens the other two
    # show_ylabel=False,               # drop the scale (only if the panels share a ylim)
)

# %%

# %% [markdown]
# ### EXPERIMENTAL: the group of neurons as one trace
#
# The single-unit panels above give interpretability but no summary; the projection gives a
# summary that needs a 57-element weight vector to explain. This is the middle: **one trace
# that is an average over the handful of units the W actually uses**, describable in a
# sentence and drawn through the same split rows as everything else.
#
# The question is what to do about these units pointing different ways - the three weights
# are **-0.16, +0.35, -0.26**, so a plain mean adds a cell that goes UP for cue2 to two that
# go DOWN and they cancel. There are two ways to stop that, and they are not the same thing:
#
#  * **flip the UNIT** (`'signed'`): multiply each unit by `sign(w_i)` - one fixed +1/-1 per
#    neuron, decided once and applied to every trial and every timebin. A unit with a
#    negative weight is one that *deactivates* for the pattern, so its going down is counted
#    as taking part. Nothing is rectified: the trace still moves up and down over time, and
#    a moment where that unit sits above its own mean still pulls the trace negative.
#  * **flip each SAMPLE** (`'squared'`): `z_i**2`, per trial per timebin. This also stops the
#    cancelling, but it drops the direction of *every measurement*, not just the unit's fixed
#    role - so cue1 at z = -1 and cue2 at z = +1 both become 1, and the two conditions the
#    figure exists to compare collapse onto each other wherever they sit symmetrically about
#    the unit's own mean.
#
# | `combine` | the trace | keeps of the W | keeps of z |
# |---|---|---|---|
# | `'mean'` | `mean(z_i)` | nothing, not even the sign | everything |
# | `'signed'` | `mean(sign(w_i) * z_i)` | the sign only | everything |
# | `'weighted'` | `sum(w_i z_i) / sum(abs(w_i))` | everything (the projection, restricted) | everything |
# | `'squared'` | `mean(z_i**2)` | nothing | magnitude only |
#
# Which of those costs what is measured rather than argued. The cell prints, per variant,
# the correlation with the **full** W projection over every trial and bin, and the peak
# |Cohen's d| that variant can still hold between the two cues and between the two choices -
# with the full projection's own d in the last row as the reference.

# %%
UNIT_GROUP = [1, 8, 12]
UNIT_COMBINE = 'signed'      # 'signed' | 'squared' | 'mean' | 'weighted' - see the table above


def unit_group_values(per_unit, w, units, combine, which_intervals):
    """Combine the per-unit traces into one, `combine` naming how.

    `per_unit`: {unit: {interval: (n_trials, n_bins)}}, each from a one-hot projection, so
    every variant here starts from the same windowing/preprocessing as the population rows.
    Three of the four are linear (a weighted sum, i.e. themselves projections onto a vector);
    `'squared'` is not, which is exactly why it has to be built from the per-unit stack
    rather than from a vector - and is a hint about what it does to the comparison.
    """
    weights = np.asarray([w[u] for u in units], dtype=float)
    out = {}
    for iv in which_intervals:
        stack = np.stack([per_unit[u][iv] for u in units])       # (n_units, n_trials, n_bins)
        wcol = weights[:, None, None]
        if combine == 'mean':
            out[iv] = stack.mean(axis=0)
        elif combine == 'signed':
            out[iv] = (np.sign(wcol) * stack).mean(axis=0)
        elif combine == 'weighted':
            out[iv] = (wcol * stack).sum(axis=0) / np.abs(weights).sum()
        elif combine == 'squared':
            out[iv] = (stack ** 2).mean(axis=0)
        else:
            raise ValueError(f"combine must be 'mean', 'signed', 'weighted' or 'squared', "
                             f"got {combine!r}")
    return out


def level_separation(values, trial_ids, t0_events, s_id, covariate, which_intervals):
    """Peak |Cohen's d| between `covariate`'s two levels, over every bin of every interval.

    The number the figure is actually for: whichever way the units are combined, can the
    resulting trace still tell the two conditions apart? Pooled-SD standardised so the four
    combinations - which live on different scales - are comparable at all."""
    best = 0.0
    for iv in which_intervals:
        levels = trial_covariate_levels(t0_events, s_id, trial_ids[iv], covariate)
        a, b = values[iv][levels == 0], values[iv][levels == 1]
        if len(a) < 2 or len(b) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)   # all-NaN edge bins
            pooled = np.sqrt((np.nanstd(a, axis=0, ddof=1) ** 2 + np.nanstd(b, axis=0, ddof=1) ** 2) / 2)
            d = np.abs(np.nanmean(b, axis=0) - np.nanmean(a, axis=0)) / np.where(pooled > 0, pooled, np.nan)
            best = max(best, float(np.nanmax(d)) if np.isfinite(d).any() else 0.0)
    return best


def plot_unit_group_panels(units=None, combine=None, w_name='poster26_S13_cueMiddle',
                           session_ids=None, which_intervals=None, splits=None, smooth=2,
                           spread='sd', ylim='shared', show_ylabel=True, show_title=True,
                           compare_variants=True, canvas_stem=None, save_dir=None):
    """One panel per session: the group trace through the same split rows as the projection.

    `ylim='shared'` is the default here (unlike the single-unit cell): these traces are means
    of z, so they mean the same thing in every session and a shared scale is honest.
    `compare_variants` prints the table that decides between the four combinations - each
    one's correlation with the full W projection, and how far apart it can still hold the
    two cue levels and the two choices (peak |d|).
    """
    units = UNIT_GROUP if units is None else list(units)
    combine = UNIT_COMBINE if combine is None else combine
    session_ids = projection_session_ids(PROJECTION_S_IDS if session_ids is None else session_ids)
    which_intervals = PROJECTION_INTERVALS if which_intervals is None else which_intervals
    splits = ('cue', 'choice_R1', ('cue', 'choice_R1'), 'correct_R1') if splits is None else splits
    save_dir = os.path.join(unit_dir, f'group_{combine}') if save_dir is None else save_dir
    os.makedirs(save_dir, exist_ok=True)

    w_row = selected_w_rows_df[selected_w_rows_df['name'] == w_name].iloc[0]
    w = np.asarray(w_row['w'], dtype=float)
    mode = fitspl.mode_for_w_row(w_row)
    bad = [u for u in units if not 0 <= u < len(w)]
    if bad:
        raise ValueError(f"unit(s) {bad} outside this W's {len(w)} features")

    curves, rows = {}, []
    for s_id in session_ids:
        t0_events, fr_x = projection_t0_for(s_id), projection_fr_for(s_id)
        # each unit on its own, once - every combination below is built from this same stack
        per_unit = {}
        for unit in units:
            onehot = np.zeros(len(w))
            onehot[unit] = 1.0
            per_unit[unit], trial_ids = get_trial_w_projection(s_id, t0_events, fr_x, which_intervals,
                                                               onehot, normalize=mode, b=0.0)
        values = unit_group_values(per_unit, w, units, combine, which_intervals)
        curves[s_id] = [split_projection_curves(values, trial_ids, t0_events, s_id, split,
                                                which_intervals, smooth=smooth, spread=spread)
                        for split in splits]

        if compare_variants:
            full, _ = get_trial_w_projection(s_id, t0_events, fr_x, which_intervals, w,
                                             normalize=mode, b=0.0)
            flat_full = np.concatenate([full[iv].ravel() for iv in which_intervals])
            for variant in ('mean', 'signed', 'weighted', 'squared'):
                vals = unit_group_values(per_unit, w, units, variant, which_intervals)
                flat = np.concatenate([vals[iv].ravel() for iv in which_intervals])
                ok = np.isfinite(flat) & np.isfinite(flat_full)
                rows.append({
                    'session': f'S{ith_to_dt_session[s_id]}', 'combine': variant,
                    'r_vs_W': np.corrcoef(flat[ok], flat_full[ok])[0, 1],
                    'd_cue': level_separation(vals, trial_ids, t0_events, s_id, 'cue', which_intervals),
                    'd_choice': level_separation(vals, trial_ids, t0_events, s_id, 'choice_R1', which_intervals),
                })
            full_row = {'session': f'S{ith_to_dt_session[s_id]}', 'combine': f'FULL W ({len(w)}u)',
                        'r_vs_W': 1.0,
                        'd_cue': level_separation(full, trial_ids, t0_events, s_id, 'cue', which_intervals),
                        'd_choice': level_separation(full, trial_ids, t0_events, s_id, 'choice_R1', which_intervals)}
            rows.append(full_row)

    if rows:
        print(f"units {units} vs the full {len(w)}-unit projection "
              f"(r over every trial and bin; d = peak |Cohen's d| between the two levels):")
        display(pd.DataFrame(rows).set_index(['session', 'combine']).round(3))

    panel_ylim = _curves_ylim([g for groups in curves.values() for g in groups],
                              show_variance=True) if ylim == 'shared' else ylim
    label = {'mean': 'mean z', 'signed': 'mean sign(w)*z', 'weighted': 'weighted z',
             'squared': 'mean z$^2$'}[combine]
    ylabel = f'{label} ({len(units)} units)'

    all_png, all_svg = [], []
    with plt.rc_context(CCGP_TRANSPARENT_RC):
        for s_id in session_ids:
            subtitle = (f"units {', '.join(map(str, units))} combined '{combine}' "
                        f"(w={', '.join(f'{w[u]:+.2f}' for u in units)})" if show_title else None)
            png, svg = draw_split_rows(
                curves[s_id], splits, projection_t0_for(s_id), s_id, which_intervals,
                ylim=_curves_ylim(curves[s_id]) if ylim is None else panel_ylim,
                show_ylabel=show_ylabel, ylabel=ylabel, subtitle=subtitle,
                ith_to_dt_session=ith_to_dt_session, save_dir=save_dir,
                stem=f'session_{s_id}_S{ith_to_dt_session[s_id]}_unitgroup_{combine}')
            all_png.append(png)
            all_svg.append(svg)

    if any(all_png):
        canvas_stem = canvas_stem or f'unitgroup_{"-".join(map(str, units))}_{combine}'
        export.build_canvas(all_png, os.path.join(ccgp_dir, f'{canvas_stem}.png'))
        export.build_canvas_svg(all_svg, os.path.join(ccgp_dir, f'{canvas_stem}.svg'))
        print(f"{len(all_svg)} panels -> {os.path.join(ccgp_dir, canvas_stem + '.svg')}")
    return all_png, all_svg


# %%
plt.close('all')
group_png, group_svg = plot_unit_group_panels(
    [1,8,12], session_ids=[13,14,15],
    # [12], session_ids=[13],
    # [8], session_ids=[13],
    # [1, ], session_ids=[13],
    # --- the usual overrides; uncomment to try one
    # combine='squared',               # mean z**2 - magnitude only, no direction
    # combine='mean',                  # the literal average - watch it cancel
    combine='weighted',              # the real projection, restricted to these units
    w_name='poster26_S13_choiceEnd', # the choice W's own units instead
    # w_name='poster26_S13_cueMiddle', # the choice W's own units instead
    # session_ids=[13, 14, 15],
    # ylim=None,                       # each session on its own scale
    # compare_variants=False,          # skip the comparison table (it costs 4 extra passes)
)

# %%
