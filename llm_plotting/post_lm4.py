# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # post_lm4 - the choice_R1 and cue W/SVM figures
#
# Split out of `lm4_plots.py` after LM4: those two `plot_session_analysis` cells are the
# ones still being iterated on, and they had accumulated enough of their own configuration
# (analytic pairing, W selections, and now the variance row) that sharing a notebook with
# the behaviour/ensemble figures meant re-running unrelated top-level cells to touch them.
#
# Definitions live in the libraries, not here - this notebook is calls only, the same
# convention `lm4_plots.py` follows (see `plotlib/__init__.py`'s layering note).
#
# **Editing a library needs no kernel restart** - `%autoreload 2` (cell 2) patches the
# imported functions in place, so re-run only the cell you are working on.
#
# Cells:
#  1. bootstrap + imports
#  2. session list
#  3. data load (SVM fits, t0 events, behaviour) - shared by both figures
#  4. the hand-picked W's
#  5. shared figure settings, including the variance row
#  6. **choice_R1** figure
#  7. **cue** figure

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
# %matplotlib widget

from analytics_processing import analytics
import analytics_processing.analytics_constants as C
from CustomLogger import CustomLogger as Logger

from analytics_processing.sessions_from_nas_parsing import sessionlist_fullfnames_from_args
from analytics_processing.sessions_from_nas_parsing import fullfnames2snames

# One definition of each of these, in the libraries - see svm_plotting_lib.py's module
# docstring for why nothing is redefined in a notebook any more. Safe to bind by name here
# despite the edit-and-re-run loop: autoreload patches these function objects in place.
from svm_plotting_lib import (METRIC_SPECS, _apply_ccgp_election, _check_one_row_per_timebin,
                              _filter_analysis,
                              plot_session_analysis, plot_session_cross_analytic_w_comparison,
                              plot_session_sequence_w_comparison,
                               build_canvas, build_canvas_svg)
from var_plotting_lib import (plot_w_variance_legend, get_interval_trial_stack,
                              timebin_samples)
from plotlib.variance import var_explained
from plotlib.trial_traces import interval_short_label
from plotlib.export import save_figure
from plotlib import fit_samples as fitspl
from plotlib import session_data as sd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# %%
data = {}
nas_dir = C.device_paths()[0]
output_dir = os.path.join(nas_dir, 'Simon', 'post_lm4')
Logger().init_logger(None, None, logging_level="INFO")

animal_ids = [6]
paradigm_ids = [1100]
session_ids = None

# %%
session_dirs, _ = sessionlist_fullfnames_from_args(paradigm_ids, animal_ids, session_ids)
session_names = fullfnames2snames(session_dirs)
session_to_dt_session = {sn[:16]: i for i, sn in enumerate(session_names)}
session_names

# %%
# ===========================================================================
# Data - loaded once, shared by both the choice_R1 and cue figures below.
# ===========================================================================
svm_session_names = session_names[7:16]

# The L2 fit. This choice is load-bearing for the variance row below, not just a
# preference: an L1 (sparse) W structurally cannot explain much of the total population
# variance, because that variance is spread over all ~54 units and a k-nonzero direction
# only reaches into k of them - so on an L1 analytic a low variance line would be a
# statement about the penalty rather than about mPFC, and comparing it to PC1 would be
# unfair by construction.
#
# The analytic's *name* records nothing about this: get_SVMCueOutcomeChoicePred's own
# default is svm_penalty='l2', only the CCGP branch passes 'l1' explicitly, and these
# cached tables have no compute branch left in analytics.py at all. Check the data:
#     svm_results['svm_penalty'].unique()
# SVM_ANALYTIC = 'SVML2CueOutcomeChoicePred2'
# SVM_ANALYTIC = 'SVML1CueOutcomeChoicePred2'
# SVM_ANALYTIC = 'SVML1CueOutcomeChoicePred2CCGP_noNorm'
# The NAIVE table, and the switch is load-bearing for the activation row below rather than
# a preference: only this one stores the fitted intercept `b` (verified 2026-08-22 - `b` is
# present and non-null on every row here, and the column does not exist in the CCGP table
# at all). Without it fit_samples.bias_for_w_row falls back to 0 and the row's zero line
# stops being the decision boundary. Targets in here: cue, choice_R1 and
# correctR1stopVScorrectR1skip - all naive, L1, normalizer='none'. The AND/XOR targets
# (cueANDstopR1/R2, choice_R2, baseline_vs_t) are in SVML1CueOutcomeChoicePred2, not here.
# SVM_ANALYTIC = 'SVML1CueOutcomeChoicePred2_noNorm'
# re-enabling this line costs the activation row its zero: no `b` in the CCGP table
SVM_ANALYTIC = 'SVML1CueOutcomeChoicePred2CCGP_noNorm'

svm_results = analytics.get_analytics(SVM_ANALYTIC, session_names=svm_session_names)

# Several generations of t0_events exist on the NAS with different interval geometry, and
# mixing them misaligns every figure silently - see plotlib.session_data.ANALYTIC_PAIRS,
# and check_geometry, which turns a mismatch into an error rather than a wrong plot.
t0_events = sd.load_t0_events(svm_session_names, analytic='TrialWiseT0Events40ms2')
# sd.check_geometry(t0_events, svm_results)

behavior = analytics.get_analytics(
    'Behavior40msAligned', session_names=svm_session_names,
    columns=['trial_id', 'cue', 'trial_outcome', 'choice_R1', 'choice_R2',
             'to_ephys_timestamp', 'frame_position'])
behavior = behavior.reset_index().drop(columns=['paradigm_id', 'animal_id', 'entry_id'])
behavior.index = pd.IntervalIndex.from_arrays(
    left=behavior.to_ephys_timestamp - 40_000,
    right=behavior.to_ephys_timestamp,
    closed='right')
behavior.set_index('session_id', append=True, inplace=True)
behavior['choice_R1'] = behavior['choice_R1'].astype(bool)

# Sessions resolved from what actually made it into all three tables (rather than from
# svm_session_names directly) - a session can come back empty from get_analytics, e.g. no
# SVM fit yet, and indexing get_session_data with an s_id absent from its index raises.
plot_session_ids = sorted(
    set(svm_results.index.unique('session_id')) &
    set(t0_events.index.unique('session_id')) &
    set(behavior.index.unique('session_id')),
    key=lambda s_id: pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M'))
print(f"{len(plot_session_ids)} sessions: {plot_session_ids[0]} .. {plot_session_ids[-1]}")

# %%
# ===========================================================================
# The hand-picked W's - one definition, used by both figures below.
# ===========================================================================
# Passing this list to a plot only draws a callout box (see HIGHLIGHT_COLOR) - the plots
# never hand data back. `w` starts None and is filled in place by
# svm_plotting_lib.resolve_w_selections, so this list doubles as the record of what was
# actually picked.
#
# A selection only constrains the keys it sets (see _selection_matches), so leaving out
# conditioner/train_stratum matches naive and CCGP fits alike - but pinning them means
# those entries can ONLY resolve against a CCGP analytic (the non-CCGP analytics have no
# CCGP rows at all). Note also that a session which was naive-fit-only for a given target
# will silently match nothing if the selection pins those keys.
W_SELECTIONS = [
    # # the shared choice-R1 axis, one W per session - the sequence to reference/compare
    # {'session_id': '2024-11-26_16-39', 'y_name': 'choice_R1',
    #  'interval': 'R1_entry_interval', 'timebin': 13, 'name': 'S07_choice_earlySquare_Aligned', 'w': None},

    # # smth else
    # {'session_id': '2024-12-04_18-06', 'y_name': 'choice_R1', 'conditioner': 'cue',
    #  'train_stratum': 1, 'interval': 'R1_entry_interval', 'timebin': 9 + 4 + 3,
    #  'name': 'S13_choice_2ndsquare_maxAligned', 'w': None},
]

# %%
# ===========================================================================
# Shared figure settings - the two figures below differ only in target/interval/save dir.
# ===========================================================================

# The variance row asks whether each timebin's W is a dominant direction in mPFC or a thin
# one: a black W curve read between a purple ceiling band (top PCs) and a gray floor band
# (random directions). Full account in var_plotting_lib.plot_w_variance_row.
svm_x_modality = 'mPFC'

# CCGP analytics only: pins which train_stratum is elected. Without it the fallback rule
# (larger stratum wins) flips between sessions - that is what made S14 show the other
# direction in its star, heatmap, covariate lines and W block.
SVM_CCGP_STRATUM = {'cue': 1, 'choice_R1': 1}

# The 2x2 the variance row is split by, in one place: the panel below and the key exported
# next to it have to name the same four cells in the same order, and the tuple (not a list -
# see resolve_variance_splits) is what makes it ONE panel of four combo curves.
W_VARIANCE_SPLIT = ('cue', 'choice_R1')

# The row's key, exported once - it needs ~2.6in and a column here is ~1.4in, so it cannot
# go in-panel. Place it on the assembled canvas by hand, like the shared y-axis. Wider and
# taller than the default now that it also carries the four combo swatches; their trial
# counts stay in each panel's own corner, being per session.
plot_w_variance_legend(save_dir=output_dir, n_pcs=3, show_session=False,
                        split=W_VARIANCE_SPLIT, figsize=(3.6, 1.1))


def plot_svm_sequence(y_predicted_names, which_intervals, save_subdir, canvas_stem,
                       show_w_variance=True, fits=None, session_ids=None, analytic=None,
                       **overrides):
    """Run plot_session_analysis over `session_ids` and assemble the canvases.

    A call wrapper only, so the cells below differ in what actually differs between them -
    target, interval, output name - rather than in copies of one long argument list that
    drift apart. Everything else is overridable per call via **overrides.

    `fits`/`session_ids` default to this notebook's own svm_results/plot_session_ids; the
    SVM_TaskSubspaces cell at the bottom passes its own, since that table has its own
    session coverage and names its targets as block specs rather than plain strings.

    `analytic` is the by-name form of `fits` - loaded here for this call only, so one cell
    can plot a different table (an L2 fit, the CCGP one) without touching the module-level
    SVM_ANALYTIC every other cell reads. It is consumed here rather than passed on:
    plot_session_analysis takes fitted rows, never an analytic name, and forwarding it
    through **overrides is a TypeError. Giving both is refused rather than silently
    picking one.
    """
    if analytic is not None:
        if fits is not None:
            raise ValueError("pass either analytic= or fits=, not both")
        fits = analytics.get_analytics(analytic, session_names=svm_session_names)
    fits = svm_results if fits is None else fits
    session_ids = plot_session_ids if session_ids is None else session_ids
    save_dir = os.path.join(output_dir, save_subdir)
    # plot_session_analysis savefig's straight into save_dir without creating it, so a new
    # save_subdir would otherwise fail per session after the figures were already drawn
    os.makedirs(save_dir, exist_ok=True)

    all_png, all_svg = [], []
    for s_id in session_ids:
        if s_id not in session_to_dt_session:
            print(f"Session {s_id} not found in session_to_dt_session mapping, skipping.")
            continue
        print(f"Session {s_id} -> S{session_to_dt_session[s_id]:02d}")

        # plot_session_analysis does no I/O of its own. get_analytics needs the FULL NAS
        # name to resolve a path but returns data keyed by the SHORT one - hence startswith
        # in, s_id out. Passed raw; it gets sliced per column by that column's X_modality.
        # Only fetched when the row is on: one FiringRate40msZ load per session, and
        # nothing else in this figure needs firing rate at all.
        fr_z_session = None
        if show_w_variance:
            full_name = next(sn for sn in session_names if sn.startswith(s_id))
            fr_z_session = analytics.get_analytics(
                'FiringRate40msZ', session_names=[full_name]).xs(s_id, level='session_id')

        kwargs = dict(
            x_modalities=svm_x_modality,
            y_predicted_names=y_predicted_names,
            which_intervals=which_intervals,
            ith_to_dt_session=session_to_dt_session,
            save_dir=save_dir,
            metric='balanced_accuracy',      # or 'f1'
            smooth=2,                        # rolling mean over timebins; None to disable
            min_metric=0.55,                 # dim heatmap cells whose decoder scored below this
            mask_chance_band=False,          # OFF: angle_white_sigma below does this job
            # Colour for the angle heatmaps - two knobs that do different jobs, and both
            # are needed to get a readable one:
            #   angle_white_sigma widens the NEUTRAL ZONE. Each colour is blended toward
            #     white by exp(-d^2/2 sigma^2) in d = angle - 90, so the base colormap is
            #     untouched away from 90 (14% blend at 2 sigma, 1% at 3 sigma) and only the
            #     middle whitens. ~20 covers the mPFC chance band (angle_2std_thr, +-21).
            #   angle_clim sets where FULL colour is reached: bwr_r ramps linearly to full
            #     red at vmin, so on 0-180 a 45-degree angle is only half-saturated. (45,135)
            #     makes 45 deg full red, which is what stops everything reading as pale pink.
            # See svm_plotting_lib._angle_colormapping.
            # angle_clim=(45, 135),
            angle_white_sigma=8,            # None -> plain bwr_r
            show_shuffle=True,               # dashed shuffle-control null on the score row
            show_mean=False,                 # True adds mean-across-bootstraps + its CI
            show_trial_kinematics=False,     # True brings back the position-trace row 0
            show_covariate_plot=False,       # True adds one small panel per other covariate
            show_w_comparison=True,          # second figure: every W block against every other
            show_ccgp_secondary=False,        # CCGP only: draw both train directions
            ccgp_w_train_stratum=SVM_CCGP_STRATUM,
            highlight_selections=W_SELECTIONS,
            show_ylabel=True,                # False for every session after the first
            show_title=True,                 # ditto - shared furniture, drawn once
            # --- the variance row
            show_w_variance=show_w_variance,
            fr_z_session=fr_z_session,
            w_variance_n_pcs=3,              # PCs the ceiling averages; 1 = PC1 alone
            # OFF: the black dashed curve here is the same W scored over the whole
            # recording - NOT a shuffle (the black dashed line one row up, in the score
            # row, is the shuffle null; same encoding, two meanings). Cut to make room for
            # the per-condition permutation bands, and it is the expensive half of the
            # computation anyway.
            w_variance_show_session=False,
            # Two-level splits only (four bands behind four bicolour curves is the pile-up
            # the joint panel exists to avoid), so this is inert while W_VARIANCE_SPLIT is
            # the 2x2 - left set for when a single-label split is put back.
            w_variance_split_perm_reps=200,   # per-condition permutation band; 0 = off
            # --- the activation row: where each cell SITS along the same W (signed
            # distance to the decision boundary), under the variance row's "how much does
            # it fluctuate". Same four cells, same colours, same dashed combo encoding.
            # Read the within-class contrast (does stop sit differently from skip inside
            # one cue) - the decoder's own two classes are forced apart by the fit.
            show_w_activation=True,
            w_activation_split=W_VARIANCE_SPLIT,
            w_activation_row_ratio=1.0,
            w_activation_ylim=2,           # scalar = +-0.3; fixed so panels compare,
                                             #   None autoscales. Clips on wide sessions.
            w_activation_show_sem=True,      # +-1 SEM band; a 2x2 cell can be 5-9 trials
            w_variance_row_ratio=1.4,        # row height vs 1.0 for the score row. One
                                             #   variance row now instead of four, so the
                                             #   figure is shorter overall even at 1.4
            w_variance_ylim=(0, 8),          # fixed % so panels compare; None autoscales
            # ONE panel, four curves: the four cue x choice_R1 cells, each drawn as a
            # bicolour dashed line (cue's colour + choice's colour, interleaved dashes) so a
            # combo reads as a combination rather than as a fifth condition. This replaces
            # the two mirrored `given` entries below, which cost four panels and drew each
            # of the four cells twice - once grouped by cue, once by choice.
            # A bare name is one panel splitting by it and POOLING over everything else;
            # {'split': X, 'given': Y} is one panel per level of Y; a LIST is a list of
            # panels while a TUPLE of two labels is this single joint panel.
            # w_variance_split=[{'split': 'choice_R1', 'given': 'cue'}, {'split': 'cue', 'given': 'choice_R1'}],
            # w_variance_split=W_VARIANCE_SPLIT,
            
            w_variance_show_pc_band=True,    # False if the PC ceiling band dominates the
                                             #   smaller conditioned panels
            show_colorbar=False,
        )
        kwargs.update(overrides)

        png_paths, svg_paths = plot_session_analysis(
            s_id, t0_events, behavior, fits, **kwargs)
        if png_paths:
            all_png.append(png_paths)
            all_svg.append(svg_paths)

    if all_png:
        png_canvas = os.path.join(save_dir, f'{canvas_stem}.png')
        build_canvas(all_png, png_canvas)
        print(f"Saved combined canvas of {len(all_png)} sessions to {png_canvas}")

        svg_canvas = os.path.join(save_dir, f'{canvas_stem}.svg')
        build_canvas_svg(all_svg, svg_canvas)
        print(f"Saved combined SVG canvas of {len(all_svg)} sessions to {svg_canvas}")
    return all_png, all_svg


# %%
# ===========================================================================
# choice_R1
# ===========================================================================
# One target and one interval list for BOTH calls below - the per-session figure and the
# cross-session W comparison have to be about the same fit or they cannot be read together.
CHOICE_TARGET = 'choice_R1'
# CHOICE_TARGET = 'cue'
# CHOICE_TARGET = 'correctR1stopVScorrectR1skip'
CHOICE_INTERVALS = ['cue_entry_interval', 'R1_entry_interval']

choice_png_paths, choice_svg_paths = plot_svm_sequence(
    analytic=SVM_ANALYTIC,
    y_predicted_names=[CHOICE_TARGET],
    # y_predicted_names=['cue', 'choice_R1'],
    which_intervals=CHOICE_INTERVALS,
    save_subdir='joined_vs_factor_code',
    canvas_stem='cue_factored_07to15',
)


# %%
CHOICE_TARGET = 'choice_R1'

choice_seq_fits = analytics.get_analytics(SVM_ANALYTIC, session_names=svm_session_names)
choice_seq_stem = f'sessionSequence_{CHOICE_TARGET}_{svm_x_modality}'
choice_seq_dir = os.path.join(output_dir, 'joined_vs_factor_code')
# returns None - it saves rather than handing paths back, unlike plot_session_analysis
plot_session_sequence_w_comparison(
    plot_session_ids, choice_seq_fits, t0_events, svm_x_modality, CHOICE_TARGET,
    CHOICE_INTERVALS,
    metric='balanced_accuracy',
    smooth=2,
    min_metric=None,                 # matched to the cell above (its min_metric is off)
    mask_chance_band=False,          # OFF: the white core does this job, same as above
    # angle_clim=(45, 135),
    angle_white_sigma=8,
    ccgp_w_train_stratum=SVM_CCGP_STRATUM,   # inert on a naive table
    highlight_selections=W_SELECTIONS,
    ith_to_dt_session=session_to_dt_session,
    save_path=os.path.join(choice_seq_dir, choice_seq_stem + '.png'),
    save_path_svg=os.path.join(choice_seq_dir, choice_seq_stem + '.svg'))
print(f"session-sequence W comparison -> {os.path.join(choice_seq_dir, choice_seq_stem + '.png')}")

# %%
# ===========================================================================
# cue
# ===========================================================================
# cue_entry_interval rather than R1_entry: the cue is only decodable where it is actually
# presented, so following it into the R1 approach mostly plots the post-cue decay. Add
# 'R1_entry_interval' to the list to see both side by side on the shared offset x-axis.
cue_png_paths, cue_svg_paths = plot_svm_sequence(
    analytic=SVM_ANALYTIC,
    y_predicted_names=['choice_R1', ], #'cue'],
    which_intervals=['cue_entry_interval', 'cue_exit_interval', 'R1_entry_interval'],
    save_subdir='joined_vs_factor_code',
    canvas_stem='cue_07to15',
)

# %%






def plot_var_explained_by_session(series, which_intervals, save_stem,
                                   ccgp_w_train_stratum=None, session_ids=None,
                                   x_modality=None, stat='mean', value='var_explained',
                                   ylim='auto', min_metric=None, metric='balanced_accuracy',
                                   session_gap=1.0, col_w=0.3, point_size=9,
                                   save_dir=None):
    """The variance row's black curve, collapsed over time: one cloud of points per
    (session, interval, series), with its mean, all in ONE axes.

    Same number the row draws - each timebin's own fitted W (`w`, straight off its table)
    scored on that timebin's own fit samples, as a % of that trial window's total
    population variance (see var_plotting_lib.w_variance_profile) - but read across
    sessions instead of within one. Computed here rather than through w_variance_profile
    because only the black curve is wanted: the PC bracket (cross-validated, 5 repeats per
    bin) and the random-direction band are that function's expensive parts and neither is
    drawn here.

    `series` is the comparison, one entry per line:

        [dict(analytic='SVML1...CCGP_noNorm', y_name='cue', label='cue (CCGP)'), ...]

    Each entry names its own `analytic` (loaded here, once per distinct name) and `y_name`,
    plus an optional `label` for the legend and an optional per-entry `analysis`/
    `ccgp_w_train_stratum` overriding the call-wide ones. Different tables AND different
    targets sit in one axes because the quantity is the same in all of them - a fraction of
    that session's own population variance in that trial window - so they share a
    denominator and are directly comparable. What they do NOT share is trial count: a CCGP
    fit sees one stratum's trials, so part of any offset between series can be an n effect
    rather than one axis genuinely being more dominant.

    `ccgp_w_train_stratum`/`analysis` pick WHICH fitted row per timebin, and must match what
    the figures were drawn with or they disagree silently: on a CCGP table a target has two
    rows per (interval, timebin), one per trained-on direction, so without an election this
    averages across directions (measured: 2.6/2.3/2.5% for stratum 0 against 2.9/2.9/3.0%
    for stratum 1 on S07/S09/S11). Both go through the same
    `_filter_analysis`/`_apply_ccgp_election` the row itself uses, and what survives is
    checked for the one-row-per-timebin invariant rather than assumed. Both are inert on a
    naive-only table, so one setting is right for a mixed comparison.

    `value` picks WHAT is plotted per timebin, with everything else - layout, series,
    threshold, statistic - unchanged:

    * `'var_explained'` (default) - the variance row's black curve, as described above.
    * `'metric'` - that timebin's own decoding score (`metric`, default balanced accuracy,
      the same number the score row draws), straight off the fit table. No firing rate is
      loaded at all in this mode, so it is much faster; it is also the reference the
      variance version is usually read against - a W can only be a meaningful direction
      where its classifier worked.

    `ylim='auto'` picks a fixed range per `value` ((0, 5) percent, or (0.4, 1.0) for a
    score); pass a pair to override or None to autoscale.

    `min_metric`, if given, keeps only timebins whose own classifier scored at least that
    (`metric`, default balanced accuracy, read from the fit's own aggr column) - the same
    threshold the figures use for dimming, applied here as an exclusion. Worth reaching for
    because the question is about *decoder axes*: a bin where the fit found nothing still
    has a W, and the variance it explains is then a statement about a random-ish direction.
    How many bins each series loses is printed, since a hard threshold can quietly empty a
    weak session.

    Layout: colour is the interval, marker is the series - the three series of one interval
    sit `col_w` apart so they read as one group, with a wider gap to the next interval and
    `session_gap` to the next session. A series can set `filled=True` to get a solid black
    marker instead of a white one. `stat` ('mean', or 'median') is drawn per column and
    joined across intervals within a session, so the clouds show whether that line rests on
    anything; these clouds are right-skewed, so a mean sits a few tenths above the median.

    No reference band, deliberately: the grey chance level is per timebin and per session
    (it depends on the trial count), so one horizontal line would be wrong for most columns.
    Read this figure for the comparison between series and the row itself for "is this above
    chance".
    """
    session_ids = plot_session_ids if session_ids is None else session_ids
    x_modality = svm_x_modality if x_modality is None else x_modality
    agg = {'mean': np.mean, 'median': np.median}[stat]
    metric_col = METRIC_SPECS[metric]['aggr_col']
    if value not in ('var_explained', 'metric'):
        raise ValueError(f"value must be 'var_explained' or 'metric', got {value!r}")
    # note this makes min_metric circular in 'metric' mode (thresholding what is plotted) -
    # still a legitimate view, just remember the floor of the cloud is the threshold
    if ylim == 'auto':
        ylim = (0, 5) if value == 'var_explained' else (0.4, 1.0)
    dropped = {}
    frames = {}
    for spec in series:
        frames.setdefault(spec['analytic'],
                          analytics.get_analytics(spec['analytic'],
                                                  session_names=svm_session_names))
    series_labels = [spec.get('label', f"{spec['y_name']} ({spec['analytic']})")
                     for spec in series]

    records = []
    for s_id in session_ids:
        # one firing-rate load and one trial stack per (session, interval), shared by every
        # series - the fits differ, the population activity they are scored on does not.
        # Only in 'var_explained' mode: a score comes off the fit table alone.
        fr_x = None
        if value == 'var_explained':
            full_name = next(sn for sn in session_names if sn.startswith(s_id))
            fr_x = sd.session_fr_for_modality(
                analytics.get_analytics('FiringRate40msZ', session_names=[full_name])
                .xs(s_id, level='session_id'), x_modality)
        stacks = {}
        for spec, series_label in zip(series, series_labels):
            fits, y_name = frames[spec['analytic']], spec['y_name']
            if s_id not in fits.index.unique('session_id'):
                print(f"S{session_to_dt_session[s_id]:02d}: not in {spec['analytic']}, skipped")
                continue
            rows = fits.xs(s_id, level='session_id')
            rows = rows[(rows.X_modality == x_modality) & (rows.predict_y_name == y_name)]
            rows = _apply_ccgp_election(
                _filter_analysis(rows, spec.get('analysis')),
                spec.get('ccgp_w_train_stratum', ccgp_w_train_stratum))
            if rows.empty:
                print(f"S{session_to_dt_session[s_id]:02d}: no {y_name} fit in "
                      f"{spec['analytic']}, skipped")
                continue
            _check_one_row_per_timebin(rows, f'{y_name} in {s_id} ({spec["analytic"]})')
            normalize = fitspl.mode_for_frame(rows)
            for interval in which_intervals:
                iv_rows = rows[rows.interval_name == interval].sort_values('timebin')
                if min_metric is not None and metric_col in iv_rows.columns:
                    # ~(x >= t), not (x < t): a NaN score (a collapsed fit) must fail the
                    # threshold - `<` is False on NaN and would keep exactly those bins
                    below = ~(iv_rows[metric_col] >= min_metric)
                    dropped[series_label] = dropped.get(series_label, 0) + int(below.sum())
                    iv_rows = iv_rows[~below]
                if iv_rows.empty:
                    continue
                if value == 'metric':
                    for tb, score in zip(iv_rows['timebin'], iv_rows[metric_col]):
                        if np.isfinite(score):
                            records.append(dict(s_id=s_id, label=f'S{session_to_dt_session[s_id]:02d}',
                                                series=series_label, interval=interval,
                                                timebin=int(tb), value=float(score)))
                    continue
                if interval not in stacks:
                    stacks[interval] = get_interval_trial_stack(s_id, t0_events, fr_x, interval)[0]
                stack = stacks[interval]
                for _, row in iv_rows.iterrows():
                    samples = timebin_samples(stack, int(row['timebin']), normalize=normalize)
                    w = np.asarray(row['w'], dtype=float).ravel()
                    if samples is None or w.size != stack.shape[2]:
                        continue
                    # nan for a zero-norm w (L1 shrank everything) - the same gaps the row has
                    ve = var_explained(samples, w)
                    if np.isfinite(ve):
                        records.append(dict(s_id=s_id, label=f'S{session_to_dt_session[s_id]:02d}',
                                            series=series_label, interval=interval,
                                            timebin=int(row['timebin']), value=100 * ve))
    for series_label, n in dropped.items():
        print(f"{series_label}: {n} timebins below {metric} {min_metric}, excluded")
    df = pd.DataFrame(records)
    if df.empty:
        print('nothing to plot')
        return {'values': df, 'paths': {}}

    labels = [f'S{session_to_dt_session[s]:02d}' for s in session_ids
              if f'S{session_to_dt_session[s]:02d}' in set(df.label)]
    drawn = [(spec, lab) for spec, lab in zip(series, series_labels) if lab in set(df.series)]
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(which_intervals)))
    markers, linestyles = ['o', 's', 'D', '^'], ['-', '--', ':', '-.']
    n_s = len(drawn)
    slot = n_s * col_w + 0.45        # one interval's group, plus the gap to the next
    span = len(which_intervals) * slot
    # floor scales with the series count, not just a constant: the two legends (intervals
    # left, series right) sit side by side and collided at the one-interval width
    fig, ax = plt.subplots(figsize=(max(5.0 + 0.9 * n_s,
                                        0.22 * len(labels) * len(which_intervals) * n_s), 3.4))
    centres, legended = [], set()
    for si, label in enumerate(labels):
        x0 = si * (span + session_gap)
        for k, (spec, series_label) in enumerate(drawn):
            face = 'black' if spec.get('filled') else 'white'
            xs, stats = [], []
            for ii, interval in enumerate(which_intervals):
                vals = df.loc[(df.label == label) & (df.interval == interval)
                              & (df.series == series_label), 'value'].to_numpy()
                if not vals.size:
                    continue
                x = x0 + ii * slot + k * col_w
                ax.scatter(np.full(vals.size, x), vals, s=point_size, color=colors[ii],
                           alpha=0.3, lw=0,
                           label=(interval_short_label(interval)
                                  if interval not in legended else None))
                legended.add(interval)
                xs.append(x)
                stats.append(agg(vals))
            ax.plot(xs, stats, linestyles[k % len(linestyles)], color='0.35', lw=1, zorder=4)
            ax.scatter(xs, stats, s=34, marker=markers[k % len(markers)], facecolor=face,
                       edgecolor='black', lw=1.2, zorder=5)
        centres.append(x0 + (span - 0.45 - col_w) / 2)

    # Session labels as annotations rather than major ticks: matplotlib hides a minor tick
    # label wherever a major tick sits, which silently swallowed labels at slot centres.
    ax.set_xticks([])
    for centre, label in zip(centres, labels):
        ax.annotate(label, xy=(centre, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -8), textcoords='offset points', ha='center', va='top',
                    rotation=45, annotation_clip=False)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel('% var. expl. by W' if value == 'var_explained'
                  else METRIC_SPECS[metric]['label'])
    title = (f'{x_modality}: variance explained by each fitted W ({stat} per interval)'
             if value == 'var_explained'
             else f'{x_modality}: decoding score ({stat} per interval)')
    ax.set_title(title + (f', {metric} >= {min_metric}' if min_metric is not None else ''))
    interval_legend = ax.legend(fontsize=8, frameon=False, ncol=len(which_intervals),
                                loc='upper left')
    ax.add_artist(interval_legend)
    handles = [plt.Line2D([], [], color='0.35', marker=markers[k % len(markers)],
                          linestyle=linestyles[k % len(linestyles)],
                          markerfacecolor='black' if spec.get('filled') else 'white',
                          markeredgecolor='black', label=lab)
               for k, (spec, lab) in enumerate(drawn)]
    # stacked under the interval legend rather than beside it: both are upper-left, and at
    # the one-interval width a side-by-side pair overlapped
    ax.legend(handles=handles, fontsize=7, frameon=False, loc='upper left',
              bbox_to_anchor=(0, 0.90))
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return {'values': df, 'stat': stat,
            'paths': save_figure(fig, save_dir or output_dir, save_stem)}




# %%

VAR_BY_SESSION_INTERVALS = ['cue_entry_interval', 'cue_exit_interval', 'R1_entry_interval']
VAR_BY_SESSION_INTERVALS = ['cue_entry_interval', ]

# The three things being compared, in one axes. Note the third: the joined
# correctR1stopVScorrectR1skip target is NOT in SVML1CueOutcomeChoicePred3CCGP_noNorm - that
# table holds cue and choice_R1 only (its compute call in analytics.py passes
# y_names=['choice_R1','cue'], ccgp=True, with the joined variant commented out), and it
# cannot hold it: the target is 1-vs-1 over a subset on which cue == choice_R1, so a
# conditioned fit of it is degenerate by construction and get_SVMCueOutcomeChoicePred
# raises on it under ccgp=True. Its only fit is the naive one in ...Pred2_noNorm. Repoint
# this entry if a Pred3 naive table is ever cached.
VAR_BY_SESSION_SERIES = [
    dict(analytic='SVML1CueOutcomeChoicePred2CCGP_noNorm', y_name='cue',
         label='cue (CCGP)'),
    # dict(analytic='SVML1CueOutcomeChoicePred3CCGP_noNorm', y_name='choice_R1',
    #      label='choice_R1 (CCGP)'),
    # dict(analytic='SVML1CueOutcomeChoicePred2_noNorm', y_name='correctR1stopVScorrectR1skip',
    #      label='correct-diagonal (naive)', filled=True),
]
SVM_CCGP_STRATUM = {'cue': 0, 'choice_R1': 0}

var_by_session = plot_var_explained_by_session(
    series=VAR_BY_SESSION_SERIES,
    which_intervals=VAR_BY_SESSION_INTERVALS,
    save_stem='varExplained_by_session_cue_choice_correctDiagonal',
    # matched to the figures above; inert on the naive series, load-bearing on the CCGP ones
    ccgp_w_train_stratum=SVM_CCGP_STRATUM,
    stat='median',
    # min_metric=0.55,                 # only W's whose own classifier cleared this; None = all
    save_dir=os.path.join(output_dir, 'joined_vs_factor_code'))
print(var_by_session['values'].groupby(['series', 'label'], observed=True)['value']
      .agg(['mean', 'count']).round(2))
print(var_by_session['paths'])

# The same figure for the score itself - the reference the one above is read against, since
# a W is only a meaningful direction where its classifier worked. No min_metric here: that
# would cut the cloud off at the threshold, which is the thing being looked at.
acc_by_session = plot_var_explained_by_session(
    series=VAR_BY_SESSION_SERIES,
    which_intervals=VAR_BY_SESSION_INTERVALS,
    save_stem='balancedAcc_by_session_cue_choice_correctDiagonal',
    value='metric',
    stat='median',
    ylim=(0.4, .8),                 # fixed to match the figure above
    ccgp_w_train_stratum=SVM_CCGP_STRATUM,
    save_dir=os.path.join(output_dir, 'joined_vs_factor_code'))

# [romt wjere savedd:
print(acc_by_session['paths']['svg'])

# %%
# ===========================================================================
# Naive AND CCGP as columns of ONE plot_session_analysis figure - scores and variance
# rows too, not just the W angles
# ===========================================================================
# Runs after the cell above and reuses its two loaded frames and its WCMP_* settings, so
# the two cells always ask the same question: naive/CCGP x cue/choice_R1, over
# WCMP_INTERVALS. What is added is everything the W-angle figure does not show - the score
# row with its shuffle null and CI, the W-variance row, each column's own
# temporal-stability heatmap - with the all-to-all W comparison arriving as
# plot_session_analysis' own second figure (show_w_comparison=True, one block per column).
#
# This was already possible for cue vs choice_R1 *within* one analytic - the only thing
# missing across analytics is that plot_session_analysis takes a single fit frame. Hence
# the concat: it never needs the two frames to be one fit, it needs one frame it can slice
# per column, and the column block spec (_as_column_spec) already carries the `analysis`
# key that separates unconditioned from conditioned rows - written for the v2 tables that
# hold both in one frame, and doing the same job here. Verified 2026-08-20 against the live
# cache that these two v1 frames concat into exactly that shape:
#   * same 4-level index (paradigm_id/animal_id/session_id/entry_id) and the same
#     predict_y_name/interval_name/X_modality/timebin columns, so every slice downstream
#     works unchanged.
#   * BOTH already carry an `analysis` column, so _filter_analysis takes its column branch
#     rather than the train_stratum fallback. Per (session, target, mPFC, three intervals):
#     analysis='naive' -> 79 rows, one per timebin, no (interval, timebin) duplicates;
#     analysis='ccgp' -> 158, i.e. both directions, which _split_ccgp_primary_secondary
#     separates downstream exactly as it does when the CCGP table is passed on its own.
#   * the non-shared columns are n_bins_left (naive only) and n_train*/n_test* (CCGP only),
#     NaN on the other side's rows. Harmless: nothing reads n_bins_left off a fit row any
#     more (interval geometry comes from t0_events - see
#     plotlib.trial_traces.interval_n_bins_left), and the n_train*/n_test* counts are only
#     read for the CCGP class-balance annotation, off CCGP rows.
#   * train_stratum/test_stratum/more_reliable are all-None object columns on the naive
#     side, so those merged columns are object rather than int64/bool. Only the pinned
#     election reads them here, and it compares by value.
# Both frames are L1 and the cell above already reported both against this notebook's
# t0_events generation - the two preconditions get_cross_analytic_Ws names, and the same
# two that make these columns comparable.
WCMP_MERGED_FITS = pd.concat([wcmp_naive_results, wcmp_ccgp_results])

# One column per (target, analytic), built target-major and then reordered by
# WCMP_COLUMN_ORDER below - so unlike WCMP_BLOCK_SPECS (which stays in the order its cell
# builds it), this figure's columns read in the hand-chosen order.
#
# `analysis` is not optional here, and this is the one thing the merge costs: on the merged
# frame a bare 'cue' selects the naive AND the ccgp rows together (237 rows, every
# (interval, timebin) duplicated) and _check_one_row_per_timebin raises rather than drawing
# three fits as one line. A guard, not a limitation - but it does mean plain-string targets
# cannot be mixed into this list.
# For the CCGP columns train_stratum both filters to the conditioned rows and ELECTS the
# direction that gets the star and whose W feeds the heatmap/variance row; the other
# direction is still drawn as a line by show_ccgp_secondary (which plot_svm_sequence has on).
WCMP_COLUMN_SPECS = [
    (dict(y_name=y_name, analysis='naive', label=f'{y_name} naive') if kind == 'naive'
     else dict(y_name=y_name, analysis='ccgp', train_stratum=WCMP_TRAIN_STRATUM[y_name],
                label=f'{y_name} CCGP'))
    for y_name in WCMP_TARGET_NAMES
    for kind in WCMP_ANALYSES
    if kind == 'naive' or y_name in WCMP_CCGP_FIT_TARGETS
]
# The hand-chosen left-to-right order, by column label rather than by position: the two
# CCGP columns on the outside, each one's naive twin just inside it, and the naive-only
# correct-diagonal column in the centre - so the figure reads outward from it and each
# target's naive/CCGP pair stays adjacent for comparison.
#
# By label, not the positional index-swap this used to be, because the two lists above are
# meant to be trimmed (see the size note) and every trim shifts those positions: a
# WCMP_ANALYSES of ['naive'] alone silently reordered unrelated columns, and dropping a
# target IndexError'd. A label that isn't in the list here keeps its target-major position,
# after the ordered ones (sorted is stable), so adding a target without touching this list
# appends it rather than raising. Tuple for immutability.
WCMP_COLUMN_ORDER = ['cue CCGP', 'cue naive', 'correctR1stopVScorrectR1skip naive',
                     'choice_R1 naive', 'choice_R1 CCGP']
WCMP_COLUMN_SPECS = tuple(sorted(
    WCMP_COLUMN_SPECS,
    key=lambda spec: (WCMP_COLUMN_ORDER.index(spec['label'])
                      if spec['label'] in WCMP_COLUMN_ORDER else len(WCMP_COLUMN_ORDER))))
print(f'{len(WCMP_COLUMN_SPECS)} columns: {[s["label"] for s in WCMP_COLUMN_SPECS]}')

# drop the naive columns
WCMP_COLUMN_SPECS = [spec for spec in WCMP_COLUMN_SPECS if spec['label'] not in ('choice_R1 naive', 'cue naive')]

# What these labels do NOT do is title the columns of the main figure: create_subplot draws
# the column title on row 0, which show_trial_kinematics=False hides (true of every cell
# above too - there the columns differ by target, which the y-axis label carries). So on the
# main figure a column is identified by its y-axis label (target only) plus the fact that a
# CCGP column draws its two direction lines in the conditioner's colours; the labels above
# do appear, in full, on the W-comparison figure's block ticks.

# Its own save_subdir: plot_session_analysis names its files by session alone
# (session_<id>_S<n>_main/_wcomp), so two configurations sharing a directory overwrite each
# other - the block tag has to go in the directory name, not the filename.
merged_png_paths, merged_svg_paths = plot_svm_sequence(
    y_predicted_names=WCMP_COLUMN_SPECS,
    which_intervals=WCMP_INTERVALS,
    save_subdir=os.path.join('joined_vs_factor_code', f'naiveCCGPcolumns_{WCMP_BLOCK_TAG}'),
    canvas_stem=f'naiveCCGPcolumns_{WCMP_BLOCK_TAG}',
    fits=WCMP_MERGED_FITS,
    session_ids=WCMP_SESSION_IDS,
    # the second figure per session: every column's elected W against every other's. With
    # more than one y_name each CCGP column contributes only its elected direction, so this
    # is the 4x4 block version of the cell above (see plot_session_analysis' single_y_name).
    show_w_comparison=True,
    show_ylabel=True, show_title=True,
    # min_metric=None, 
    mask_chance_band=False,  # show all angles, not just the good ones
    ccgp_w_train_stratum=WCMP_TRAIN_STRATUM,
)


# %%



# %%

















































































# %%
# ===========================================================================
# SVM_TaskSubspaces - first look at the new analytic, and what changed vs the old ones
# ===========================================================================
# One session at a time. Every output kind the new analytic produces is exercised here in
# the same patterns the cells above use: a single target, the XOR target, and both CCGP
# directions, plus the two W comparisons (directions against each other, and new against
# old). The old tables are loaded alongside so a changed number can be attributed rather
# than guessed at.
#
# What genuinely differs from the cells above, and why the numbers are not expected to match
# the old tables bin for bin (see svm_task_subspaces.py's docstring for the full account):
#   * the shuffle control is now drawn with the SAME estimator as the observation (same
#     bootstrap count, C re-selected per draw) and held fixed across a row group, so
#     show_shuffle draws a valid null - the old one sat systematically low
#   * a collapsed fit is NaN, not 0.500, so gaps in a line are now honest holes
#   * C comes from a 6-point grid over 0.1-100 with single-class candidates excluded
#   * the metric is balanced accuracy only - metric='f1' does not exist on this table
#   * naive and CCGP fits are ONE table now, so a target alone no longer names a single
#     line: every call below says which rows it means, as a block spec (see
#     svm_plotting_lib._as_column_spec). Plotting an unpinned target raises rather than
#     silently drawing three fits on top of each other.
# The v1/v2 column-name differences (n_pos/n_neg, the null lists) are resolved inside the
# plotting libs, so the table is passed exactly as it comes off the analytic.
import ephys_preprocessing.svm_task_subspaces as svmts

TS_SESSION = '2024-12-04_18-06'
TS_INTERVALS = ['cue_entry_interval', 'R1_entry_interval', 'R2_entry_interval']
TS_MODALITY = 'mPFC'
ts_full_name = next(sn for sn in session_names if sn.startswith(TS_SESSION))
ts_save_dir = os.path.join(output_dir, 'task_subspaces_check')
os.makedirs(ts_save_dir, exist_ok=True)

ts_raw = sd.load_task_subspaces([ts_full_name])

print("=== coverage: rows per (interval, target, analysis, train_stratum)")
print(ts_raw.groupby(['interval_name', 'predict_y_name', 'analysis', 'train_stratum'],
                     dropna=False).size().to_string())
print("\n=== geometry stamped on the rows (must match t0_events, or the figures shear)")
print(ts_raw.groupby('interval_name')[['n_bins_left', 'n_bins_right', 'window_bins']]
      .agg(['min', 'max']).to_string())
print(f"\ncheck_geometry vs the loaded t0_events: "
      f"{sd.check_geometry(t0_events, ts_raw, raise_on_mismatch=False) or 'consistent'}")
print(f"\ncollapsed fits (NaN, not 0.5): {int(ts_raw.fit_collapsed.sum())}/{len(ts_raw)}; "
      f"best_C picked: {sorted(ts_raw.best_C.dropna().unique())}")

# The statistic that belongs to a whole line, for every row group in one table: single
# targets, the XOR, and each CCGP direction separately (a direction is its own row group).
# obs_max is the line's highest point, thr_* the 95th percentile of the per-draw maxima, and
# p_* its rank among them - the per-row p_point_* answers the weaker per-timebin question.
print("\n=== row-group statistics (sequence_stats)")
ts_seq = svmts.sequence_stats(ts_raw)
print(ts_seq[ts_seq.X_modality == TS_MODALITY]
      .sort_values(['predict_y_name', 'interval_name', 'train_stratum'])
      .to_string(index=False))

print("\n=== same score, new table vs the two old ones (unconditioned rows only)")
ts_old_naive = analytics.get_analytics('SVML1CueOutcomeChoicePred2', session_names=[ts_full_name])
ts_old_ccgp = analytics.get_analytics(SVM_ANALYTIC, session_names=[ts_full_name])
for old_name, old_frame in (('SVML1CueOutcomeChoicePred2', ts_old_naive),
                            (SVM_ANALYTIC, ts_old_ccgp)):
    cmp = sd.compare_svm_tables(ts_raw, old_frame, label_new='new', label_old='old')
    print(f"\n--- vs {old_name}")
    print("(no overlapping unconditioned rows)" if cmp.empty
          else cmp[cmp.X_modality == TS_MODALITY].to_string(index=False))

# --- the figures, in the same patterns as the cells above -------------------------------
ts_kwargs = dict(
    x_modalities=TS_MODALITY, which_intervals=TS_INTERVALS,
    ith_to_dt_session=session_to_dt_session, save_dir=ts_save_dir,
    metric='balanced_accuracy',      # 'f1' is not computed by this analytic
    smooth=2, min_metric=0.55, mask_chance_band=True,
    show_shuffle=True,               # now a valid null (see the header of this cell)
    show_mean=False, show_trial_kinematics=True, show_covariate_plot=False,
    show_w_variance=False,           # on demand; needs the FiringRate40msZ load
    show_ylabel=True, show_title=True,
)

# 1. the plain targets, one per figure; 2. both CCGP directions of one target plus the W
#    comparison between them (a column that leaves train_stratum unpinned still contributes
#    its two directions as separate blocks, which is the single-y_name case
#    plot_w_comparison draws direction-vs-direction for); 3. the drift diagnostics, read
#    against the shift null rather than the permutation one - session_half IS the drift
#    axis, so permuting its labels is not the question worth asking about it.
for ts_targets, ts_extra in (
        ([{'y_name': 'cue', 'analysis': 'naive'}], dict()),
        ([{'y_name': 'cueXORchoice_R1', 'analysis': 'naive'}], dict()),
        ([{'y_name': 'choice_R1', 'analysis': 'ccgp'}],
         dict(show_w_comparison=True, show_ccgp_secondary=True,
              ccgp_w_train_stratum={'choice_R1': 1})),
        ([{'y_name': 'session_half', 'analysis': 'naive'},
          {'y_name': 'random_control', 'analysis': 'naive'}], dict(null_kind='shift')),
):
    print(f"\n=== plot_session_analysis: {ts_targets} {ts_extra}")
    plot_session_analysis(TS_SESSION, t0_events, behavior, ts_raw,
                          y_predicted_names=ts_targets, **{**ts_kwargs, **ts_extra})

# 4. the naive-vs-CCGP-vs-XOR figure this table was written for: three blocks, one frame,
#    one figure - with show_w_comparison=True their W's are compared against each other
#    below it. That comparison used to need plot_session_cross_analytic_w_comparison and
#    two separately-fit analytics whose C selection (and so, under L1, sparsity) differed;
#    here the three fits come out of the same run, so the angles mean what they say.
plot_session_analysis(
    TS_SESSION, t0_events, behavior, ts_raw,
    y_predicted_names=[{'y_name': 'choice_R1', 'analysis': 'naive'},
                       {'y_name': 'cueXORchoice_R1', 'analysis': 'naive'},
                       {'y_name': 'choice_R1', 'analysis': 'ccgp', 'train_stratum': 1,
                        'label': 'choice_R1 | cue=1'}],
    show_w_comparison=True, **ts_kwargs)

# 5. new vs old W, one block per analytic - the cross-analytic comparison from the cell
#    above, with the new table as two more blocks. The old naive side is the L1 table
#    because an L1 and an L2 W are not angle-comparable (this analytic is L2, so read the
#    new-vs-old rectangles as "does the axis move", not as a strict angle test).
ts_block_specs = [
    dict(label='TaskSubspaces', svm_results=ts_raw, y_name='choice_R1', analysis='naive'),
    dict(label='TaskSubspaces CCGP', svm_results=ts_raw, y_name='choice_R1', train_stratum=1),
    dict(label='old naive L1', svm_results=ts_old_naive, y_name='choice_R1'),
    dict(label='old CCGP L1', svm_results=ts_old_ccgp, y_name='choice_R1', train_stratum=1),
]
plot_session_cross_analytic_w_comparison(
    TS_SESSION, ts_block_specs, t0_events, TS_MODALITY, ['R1_entry_interval'],
    min_metric=0.55, mask_chance_band=True, smooth=2,
    ith_to_dt_session=session_to_dt_session, reference_label=None,
    save_path=os.path.join(ts_save_dir, f'wCrossAnalytic_new_vs_old_{TS_SESSION}.png'),
    save_path_svg=os.path.join(ts_save_dir, f'wCrossAnalytic_new_vs_old_{TS_SESSION}.svg'))

# %%













# ===========================================================================
# SVM_TaskSubspaces - the session sequence
# ===========================================================================
# The same per-session-panel sequence as the choice_R1/cue cells at the top, over the new
# analytic. Kept as its own cell (and its own frame/session list) rather than switching
# SVM_ANALYTIC above, so the old figures stay reproducible in the same kernel and the two
# can be diffed panel for panel.
#
# Everything the two generations spell differently is handled inside the plotting libs;
# the two things that are genuinely this cell's business:
#   * every target is a block spec, since naive and CCGP rows share this table
#   * the session list comes from THIS table (its own coverage), not from plot_session_ids,
#     which was resolved off the L2 frame in the data cell above
TS_SEQ_ANALYTIC = 'SVM_TaskSubspaces'
ts_seq_results = sd.load_task_subspaces(svm_session_names)
sd.check_geometry(t0_events, ts_seq_results)

ts_seq_session_ids = sorted(
    set(ts_seq_results.index.unique('session_id')) &
    set(t0_events.index.unique('session_id')) &
    set(behavior.index.unique('session_id')),
    key=lambda s_id: pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M'))
print(f"{len(ts_seq_session_ids)} sessions: {ts_seq_session_ids[0]} .. {ts_seq_session_ids[-1]}")

# %%
# choice_R1: the unconditioned fit, the XOR, and the cue=1-conditioned direction, side by
# side in every session's panel - which is the comparison the merged table exists for.
ts_seq_png, ts_seq_svg = plot_svm_sequence(
    y_predicted_names=[
                    #    {'y_name': 'choice_R1', 'analysis': 'naive'},
                    #    {'y_name': 'choice_R1', 'analysis': 'ccgp', 'train_stratum': 1,
                    #     'label': 'choice_R1 | cue=1'},
                    #    {'y_name': 'cueXORchoice_R1', 'analysis': 'naive'},
                       {'y_name': 'cue', 'analysis': 'ccgp', 'train_stratum': 1,
                                               'label': 'cue | choiceR1=1'},
                       {'y_name': 'cue', 'analysis': 'naive'},
                       ],
    which_intervals=['cue_entry_interval', 'cue_exit_interval', 'R1_entry_interval'],
    save_subdir='task_subspaces_new',
    canvas_stem='choiceR1_naive_xor_ccgp',
    fits=ts_seq_results, session_ids=ts_seq_session_ids,
    show_w_variance=True,            # gates the whole row AND the per-session FiringRate40msZ load
    show_ccgp_secondary=True,       # each spec is one direction already
    ccgp_w_train_stratum=None,       # ditto - the specs pin what they mean
    show_w_comparison=True,          # the three fits are compared against each other below the score row
    mask_chance_band=True, smooth=2, min_metric=.52,
    w_variance_show_session=True, w_variance_n_pcs=3, w_variance_row_ratio=0.6, w_variance_ylim=(0, 8),
    show_ylabel=True, 
    show_title=True
    
)

# %%
# cue, and the drift diagnostic every other curve should be read against: session_half
# decodes wherever the population drifts, so a target that tracks it is suspect. Read
# against the shift null (null_kind), which preserves the labels' own temporal structure.
ts_cue_png, ts_cue_svg = plot_svm_sequence(
    y_predicted_names=[
                        {'y_name': 'cueXORchoice_R1', 'analysis': 'naive'},
                        {'y_name': 'cue', 'analysis': 'ccgp', 'train_stratum': 1,
                        'label': 'cue | choiceR1=1'}],
    which_intervals=['cue_entry_interval', 'cue_exit_interval', 'R1_entry_interval'],
    save_subdir='task_subspaces_new',
    canvas_stem='cue_vs_drift',
    fits=ts_seq_results, session_ids=ts_seq_session_ids,
    null_kind='shift',
    show_w_variance=True,
    show_ccgp_secondary=False,
    ccgp_w_train_stratum=None,
    show_w_comparison=True,
    mask_chance_band=False, smooth=2, min_metric=0.52,
    w_variance_show_session=True, w_variance_n_pcs=3, w_variance_row_ratio=0.6, w_variance_ylim=(0, 8),
    show_ylabel=True, 
    show_title=True
)

# %%
