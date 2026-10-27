# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: analysisVR
#     language: python
#     name: python3
# ---

# %%
import os
import sys
import textwrap

# to setup import paths add project root dirs to sys.path
sys.path.append(os.path.join(os.getcwd(), "..", ".."))
from baseVR.base_functionality import init_import_paths
init_import_paths()

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # %matplotlib widget
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgb, rgb_to_hsv, hsv_to_rgb
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

from analytics_processing import analytics
import analytics_processing.analytics_constants as C
from CustomLogger import CustomLogger as Logger

from analytics_processing.sessions_from_nas_parsing import sessionlist_fullfnames_from_args
from analytics_processing.sessions_from_nas_parsing import fullfnames2snames

import scipy.stats as stats
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', 100)     # Show all rows
pd.set_option('display.width', 500)        # Wide display
pd.set_option('display.max_colwidth', 40) # Full width column content
pd.set_option('display.expand_frame_repr', False)  # Don't wrap to multiple lines

# %%
# Pure plotting library (constants + plot_session_analysis and its full dependency
# closure) - imported, not redefined here, so there's exactly one copy; see
# svm_plotting_lib.py's own module docstring. Edit definitions there, not inline in
# this notebook - re-run this cell (or restart the kernel) to pick up a change if
# autoreload isn't active.
from svm_plotting_lib import (
    modality_colors, ALL_Y_PREDICTED_NAMES, LEVEL_LABELS, CORRECTNESS_SPECS, METRIC_SPECS, angle_2std_thr,
    _covariate_color, _level_label, _interval_short_label, _resolve_column_specs, _stack_by_timebin,
    _stack_covariate_true, _bootstrap_aggr_ci, _df_to_markdown_table, _ordered_svm_res,
    _w_block_with_interval_separators, _split_ccgp_primary_secondary, _resolve_ccgp_w_stratum,
    _rows_for_stratum, _selection_matches, _highlighted_rows, resolve_w_selections,
    get_session_data, _resample_pos, _smooth,
    _ccgp_n_annotation, _plot_f1_lines, create_subplot, plot_covariate_f1_single, plot_correctness_f1_single,
    _covariate_specs_for_target, plot_covariate_f1, _write_wcomp_markdown, plot_w_comparison,
    _write_session_main_markdown, plot_session_analysis,
    # These two were redefined further down this notebook until 2026-08-11, shadowing
    # these (newer) library copies - see the note at that cell.
    get_session_sequence_Ws, plot_session_sequence_w_comparison,
)
# Multi-session canvas assembly (PNG raster / SVG vector merge) - one definition, shared
# with dim_plotting_lib; used to be a third copy inline in this notebook.
from plotlib.export import build_canvas, build_canvas_svg, save_figure
# Session loading + the t0_events/SVM analytic version pairing (several generations of
# both exist on the NAS with different interval geometry - mixing them misaligns figures
# silently, so check_geometry makes it an error).
from plotlib import session_data as sd
SVM_ANALYTIC = 'SVML1CueOutcomeChoicePred2'
# Everything about a single fitted W - its weight profile, what the population does along
# it, how much variance it explains, whether it tracks kinematics. All of this used to be
# defined inline further down this notebook (~900 lines).
from w_plotting_lib import (
    plot_W_list, plot_w_projection_session, plot_w_variance_explained,
    plot_w_projection_kinematics, plot_session_projection_kinematics,
)
# Whole-session PCA and cross-session subspace drift - population geometry, so it lives
# with the rest of it in dim_plotting_lib rather than in an SVM library.
from dim_plotting_lib import (
    plot_session_pca, plot_session_pca_similarity, plot_pca_reference_drift,
)
# Cross-session heatmap view of the same decoding metric - its own module (it's a different
# figure, not part of plot_session_analysis' closure), reusing this library's conventions.
from svm_heatmap_plotting_lib import plot_metric_heatmap_sessions, plot_metric_distribution


# %%
# for poster, use larger fonts

plt.rcParams.update({
    'font.size': 15,           # general font size
    'axes.titlesize': 15,      # subplot title
    'axes.labelsize': 15,      # x and y labels
    'xtick.labelsize': 15,     # x tick labels
    'ytick.labelsize': 15,     # y tick labels
    'legend.fontsize': 15,     # legend
    # thicker cue-zone hatch lines in create_subplot - set globally (not scoped via
    # rc_context) since hatches render lazily at savefig()/show() time, long after any
    # local `with` block around the draw call would already have reverted it
    'hatch.linewidth': 2.5,
})

# %%
ALL_X_MODALITIES = ['mPFC', 'HP+mPFC', ]#'HP']
ALL_INTERVALS = ['cue_entry_interval', 'R1_entry_interval', 'R2_entry_interval']

# %%
# first set the paths and logger. You can see all kind of outputs in this directory 
# already. Poster used plots from this directory.

data = {}
nas_dir = C.device_paths()[0]
output_dir = f"{nas_dir}/Simon/lab_meeting_2026/"
Logger().init_logger(None, None, logging_level="WARNING")

# %%
# analysis was intially done on all sessions from animal 6 with paradigm 1100
# But the poster focused on S8-15

animal_ids = [6]
paradigm_ids = [1100]
session_ids = None

# 7 sessions double stopping setting
to_s_id_06 = pd.to_datetime('2024-11-25_20-00', format='%Y-%m-%d_%H-%M')
to_s_id_07 = pd.to_datetime('2024-11-26_20-00', format='%Y-%m-%d_%H-%M')
to_s_id_14 = pd.to_datetime('2024-12-06_20-00', format='%Y-%m-%d_%H-%M')
to_s_id_15 = pd.to_datetime('2024-12-09_20-00', format='%Y-%m-%d_%H-%M')
to_s_id_20 = pd.to_datetime('2025-01-14_20-00', format='%Y-%m-%d_%H-%M')
to_s_id_26 = pd.to_datetime('2025-01-22_20-00', format='%Y-%m-%d_%H-%M')



# %%
session_dirs, _ = sessionlist_fullfnames_from_args(paradigm_ids, animal_ids, session_ids)
session_names = fullfnames2snames(session_dirs)
ith_to_dt_session = {sn[:16]:i for i,sn in enumerate(session_names)}
session_names

# %%
# this is a new analytic. It is a crucial component that is required by the SVM
# decoding analytic to be computed. It is also used here for plotting.
# Another input analytic that is used to calc TrialWiseT0Events40ms that I use 
# now a lot is Behavior40msAligned - this is similar to BehaviorTrackwise, but 
# instead of binning according to track position, we bin according to the 40ms 
# ephys bins. So usually an average over 2 or 3 unity frames.

# Each row is an interval - a start and a stop ephys time. Think of this as the 
# selecter of firingrates X for the SVM that we want to train with. There are 4 
# differnent intervals a row may define (and 2 more specifcal-case ones): 
# cue_entry_interval, cue_exit_interval, R1_entry_interval, R2_entry_interval
# (cue_exit_interval is not used, in reality it overlaps a lot with R1_entry_interval).
# 
# So in general, each row is an interval of one of these kinds. All other intervals 
# are NaN in that row (so each interval has it's own, sparse column)
# Columns: trial_id	cue	trial_outcome	choice_R1	choice_R2	t0_event_name	t0	x_position	x_alignment	pre_cue_interval	nextto_cue_interval	cue_entry_interval	R1_entry_interval	R2_entry_interval
# So you can see we have the labels, a name of the interval (t0_event_name), the t0 time, and importantly, the x_position 
# we tried to align to (x_alignment) and finally the actual x_position that the 
# unty frame had. So you can see the way it works is for every trial, we check a 
# set of x_alignments (e.g. cue position, R1 position, R2 position) and find the
# ephys time (t0) that corresponds to that x_position. Then we define intervals
# around that t0. The length of the intervals can differ. 

# There are also two special intervals, pre_cue_interval, and nextto_cue_interval 
# which are in the same row and short, 200ms,  or 5 ephys bins. The ones above have 
# usually around 30 ephys bins. These were used to train an SVM to decode pre- vs 
# within cue neural activity, first for cue1 and then for cue2 trials. Then I 
# tried to compare these two fits to check if they are orthogonal.

# all of this is calcuclated in analytics/integr_analytics.py

t0_events = analytics.get_analytics('TrialWiseT0Events40ms', session_names=session_names)
t0_events


# %%
# this is the new analyric I metioned Behavior40msAligned. I think we just need this
# to draw x positions at every timepoint in one of the plots

behavior = analytics.get_analytics('Behavior40msAligned', session_names=session_names)
fr = analytics.get_analytics('FiringRate40msHz', session_names=session_names)
behavior.iloc[100:200]


# %%
behavior.set_index('to_ephys_timestamp', append=True, inplace=True)
behavior.index = behavior.index.droplevel('entry_id')
behavior

# %%
# get the raw firing rates, i don't rmember why

fr = fr.set_index(["from_ephys_timestamp","to_ephys_timestamp"], append=True)
fr.reset_index(level=(0,1,3,4), inplace=True, drop=True)
fr

# %%
behavior

# %%



# %%
# the main data to plot. Calculated like 1000 other things in postproc_mea1k_ephys.py....
# columns w	predictions	n_positives	n_negatives	f1_aggr	f1_mean	f1_ci_lower	f1_ci_upper	acc_mean	acc_ci_lower	acc_ci_upper	X_modality	predict_y_name	interval_name	timebin	y_true
# so we have the weight vecter (avg over all bootstraps), the predictions, the number of positive and negative labels, 
# a bunch of performance metrics, X_madilty = the input data used (X), 
# predict_y_name the SVM that was fit - this is closely linked to the interval used.
# interval_name, the interval name used to select X
# timebin, very important, the specfic timebin used for fitting. This goes from 0 to eg 30. 
# Remeber that in one interval we do a fit for each timebin - we slide by one 
# timebin (40 ms ephys bin) The intervals from from T0Events analytic. This is 
# where we select one timepoint as t0 (eg cue visible) and say lets look at 10 
# timebins before that and 20 after that event. For each of those timepoints 
# we fit a SVM.

# in this case below, we slice to only the special case I described before: 
# pre vs within cue zone for cue1 and cue2 (didn't make it on to the poster, but
# I explained this to you before I remember) This one is not too important, don't
# need to focus on it

# data = analytics.get_analytics('SVMCueOutcomeChoicePred', session_names=session_names)
data = analytics.get_analytics('SVML1CueOutcomeChoicePred2', session_names=session_names)
#data = data[data.predict_y_name.isin(['pre_vs_in_cue1', 'pre_vs_in_cue2'])]
data

# %%
# fetch the data again, this time unfiltered

# svm_results = analytics.get_analytics('SVML1CueOutcomeChoicePredCCGP', session_names=session_names)
svm_results = analytics.get_analytics('SVML1CueOutcomeChoicePred2', session_names=session_names)
svm_results = analytics.get_analytics('SVML1CueOutcomeChoicePred2CCGP_noNorm', session_names=session_names)
# plt.xscale('log')
svm_results

# %%
# fetch data again, this time the t0 events

# TrialWiseT0Events40ms2, NOT the bare TrialWiseT0Events40ms this used to load: the bare
# one is stale and its interval windows no longer match the geometry these fits used
# (cue 40 vs 25 bins, R1 40/45 vs 34), which silently misaligned every position trace
# against its decoding curve. session_data.check_geometry below enforces the pairing.
t0_events = sd.load_t0_events(session_names, analytic=sd.t0_analytic_for(SVM_ANALYTIC))
# sd.check_geometry(t0_events, svm_results)
t0_events

# %%
# # the numbers above come from here.
# #
# # Null-distribution rehaul: rather than comparing actual fitted SVM weight vectors
# # (which conflates "chance baseline" with whatever structure the fit/bootstrap
# # procedure itself imparts, and pools W's across *different* sessions' - i.e.
# # different physical neurons' - decoders into one distribution), sample a bunch of
# # raw z-scored population-activity vectors (FiringRate40msZ, one 40ms native bin =
# # one vector) per session and compute pairwise cosine angles between those instead -
# # a more direct "what does chance look like for this dataset's own neurons" baseline.
# # Kept strictly within-session (vectors from different sessions are never compared to
# # each other - different sessions record different physical neurons even at the same
# # column index/X_modality slice, so a cross-session angle wouldn't mean anything);
# # only the resulting angle *values* are pooled across sessions into one combined
# # per-modality null. Column slicing must match get_SVMCueOutcomeChoicePred's X_full
# # construction (postproc_mea1k_ephys.py, fr_z.iloc[:, 20:] for 'mPFC') - same fragile,
# # not-yet-centralized split _x_modality_fr uses elsewhere in this file (see its
# # docstring) - so any future change to that split needs updating here too.
# plt.close("all")
# rng = np.random.default_rng(42)
# n_samples_per_session = 300

# # for x_modal in ['mPFC', 'HP', 'HP+mPFC']:
# for x_modal in ['mPFC', ]:
#     plt.figure(figsize=(5, 5))
#     all_degrees = []
#     for full_sname in session_names:
#         s_id = full_sname[:16]
#         fr_z_raw = analytics.get_analytics('FiringRate40msZ', session_names=[full_sname])
#         if fr_z_raw is None:
#             continue
#         fr_z_raw = fr_z_raw.xs(s_id, level='session_id').drop(
#             columns=['from_ephys_timestamp', 'to_ephys_timestamp'])
#         if x_modal == 'mPFC':
#             fr_x = fr_z_raw.iloc[:, 20:]
#         elif x_modal == 'HP':
#             fr_x = fr_z_raw.iloc[:, :20]
#         else:
#             fr_x = fr_z_raw
#         n = len(fr_x)
#         if n < 2:
#             continue
#         # subsample row *indices* first, then fillna/convert only that small slice -
#         # fr_x itself can be tens of thousands of rows (every 40ms bin of the session),
#         # no need to densify/fillna the whole thing just to use ~300 of its rows
#         sample_idx = rng.choice(n, size=min(n_samples_per_session, n), replace=False)
#         vecs = fr_x.iloc[sample_idx].fillna(0).values
#         norms = np.linalg.norm(vecs, axis=1)
#         cos_sim_matrix = (vecs @ vecs.T) / (norms[:, np.newaxis] * norms[np.newaxis, :])
#         degrees = (np.arccos(np.clip(cos_sim_matrix, -1.0, 1.0)) * (180 / np.pi)).flatten()
#         all_degrees.append(degrees)

#     degrees = np.concatenate(all_degrees)
#     ax = plt.gca()
#     hist = ax.hist(degrees, bins=80, color='gray', density=True)
#     # draw in 2 std dev lines
#     mean_angle = np.nanmean(degrees)
#     std_angle = np.nanstd(degrees)
#     lo, hi = mean_angle - 2*std_angle, mean_angle + 2*std_angle
#     print(hi, x_modal)
#     print(lo, x_modal)
#     ax.axvspan(lo, hi, color='lightblue', alpha=0.15, zorder=0, label='~95% of angles')
#     ax.axvline(hi, color='lightblue', linestyle='--')
#     ax.axvline(lo, color='lightblue', linestyle='--')
#     ax.annotate(f'{hi:.1f}°', (hi, ax.get_ylim()[1]), xytext=(3, -3),
#                 textcoords='offset points', va='top', color='steelblue')
#     ax.annotate(f'{lo:.1f}°', (lo, ax.get_ylim()[1]), xytext=(-3, -3),
#                 textcoords='offset points', va='top', ha='right', color='steelblue')
#     ax.set_xticks([0, 60, 90, 120, 180])
#     ax.set_xlim(0, 180)
#     ax.grid(axis='x', color='lightgray', linewidth=0.5, zorder=0)
#     ax.set_yticks([])
#     for spine in ax.spines.values():
#         spine.set_visible(False)
#     ax.set_title(f'Null angle distribution ({x_modal}, chance baseline)')
#     ax.set_xlabel('Angle (degrees)')
#     ax.set_ylabel('Density')
#     plt.tight_layout()
#     # save to output dir
#     plt.savefig(os.path.join(output_dir, f'null_angle_distribution_{x_modal}.svg'), dpi=300)
    
# plt.show()



# hardcoded 2 std boundaries of 2 random angles within a high dim space, matching dims of our data:
# so HP eg is 20 dims and is therfore more spread out around 90, versus mPFC with eg 54 dims
# calculcated later, but placed here for convenience - now from the raw-activity null above,
# not fitted SVM weights (see the rehaul comment above)




# %%
svm_results


# %%
behavior = analytics.get_analytics('Behavior40msAligned', session_names=session_names)
behavior = behavior.reset_index().drop(columns=['paradigm_id', 'animal_id', 'entry_id'])
behavior.index = pd.IntervalIndex.from_arrays(
    left=behavior.to_ephys_timestamp - 40_000,
    right=behavior.to_ephys_timestamp,
    closed='right'
)
behavior.set_index('session_id', append=True, inplace=True)
behavior['choice_R1'] = behavior['choice_R1'].astype(bool)
behavior['choice_R1']



# %%
# POSTER PLOTS
#
# Per-session SVM decoding plot. Layout per column: (0) single-trial position
# trajectories colored by decode F1, (1) F1 over time (f1_aggr = bold reference
# over all trials, f1_mean = thin, mean across bootstrap iterations), (2) cosine-angle
# heatmap between per-timebin SVM weight vectors, (3+, optional) one small panel per
# other behavioral covariate (cue / choice_R1 / choice_R2 / trial_outcome), each
# showing F1 split by that covariate's value against the same f1_aggr baseline.
#
# Pass a list for x_modalities and/or y_predicted_names to plot_session_analysis to
# get one column per value; pass lists for *both* to multiplex every combination
# into its own column (see _resolve_column_specs).






















# %%
t0_events.index.unique('session_id').to_list()  # check which sessions are available in the t0_events DataFrame
#  '2024-11-26_16-39'
#  '2024-11-28_17-41',
#  '2024-11-29_17-21',
#  '2024-12-02_16-09',
#  '2024-12-03_16-23',
#  '2024-12-04_18-06',
#  '2024-12-06_16-49',
#  '2024-12-09_17-45',
# %%
# Hand-picked W's to call out with a thin dotted green box wherever they appear - in
# plot_session_analysis's cross-column comparison below, and again further down in the
# cross-session "grand comparison". Only the keys you set are checked (see
# _selection_matches), so a dict can omit `conditioner` for a naive-fit selection, or
# omit `timebin` to mark every timebin of that interval - but if a CCGP target's *two*
# directions can both appear (single_y_name in plot_session_analysis, or a specific
# direction you want regardless of which one get_session_sequence_Ws elected), set
# `train_stratum` (0/1) too - `conditioner` alone only names which other target it's
# conditioned on, not which direction, so it can't disambiguate between the two W's on
# its own. Replace with your own real session_id/interval/timebin values - these four
# are just illustrative placeholders.
highlight_selections = [
    
    # {'session_id': '2024-11-26_16-39', 'y_name': 'choice_R1',
    #  'interval': 'cue_exit_interval', 'timebin': 22, 'name': 'predicted_shared_choice_07'},
    
    
    
    # # shared choice R1 decoder across sessions - the one to reference/ compare
    # # session07
    # {'session_id': '2024-11-26_16-39', 'y_name': 'choice_R1',
    #  'interval': 'R1_entry_interval', 'timebin': 13-5, 'name': 'shared_choice_07'},
    # # # session09
    
    # {'session_id': '2024-11-26_16-39', 'y_name': 'choice_R1',
    #  'interval': 'R1_entry_interval', 'timebin': 21, 'name': 'shared_choice_07'},
    
    
    # # session09
    # {'session_id': '2024-11-28_17-41', 'y_name': 'choice_R1',
    #  'interval': 'R1_entry_interval', 'timebin': 19, 'name': 'shared_choice_09'},
    # # session11
    # {'session_id': '2024-12-02_16-09', 'y_name': 'choice_R1', 'conditioner': 'cue',
    #  'train_stratum': 1, 'interval': 'R1_entry_interval', 'timebin': 15, 'name': 'shared_choice_11'},
    # # session12
    # {'session_id': '2024-12-03_16-23', 'y_name': 'choice_R1', 'conditioner': 'cue',
    #  'train_stratum': 1, 'interval': 'R1_entry_interval', 'timebin': 27, 'name': 'shared_choice_12'},
    # session13. --  the one to reference index = 4
    {'session_id': '2024-12-04_18-06', 'y_name': 'choice_R1', 'conditioner': 'cue',
     'train_stratum': 1, 'interval': 'R1_entry_interval', 'timebin': 23, 'name': 'poster26_S13_choiceEnd'},
    
    # # 14
    # {'session_id': '2024-12-06_16-49', 'y_name': 'choice_R1', 'conditioner': 'cue',
    #  'train_stratum': 1, 'interval': 'R1_entry_interval', 'timebin': 18, 'name': 'shared_choice_14'},
    # # 15
    # {'session_id': '2024-12-09_17-45', 'y_name': 'choice_R1', 'conditioner': 'cue',
    #  'train_stratum': 1, 'interval': 'R1_entry_interval', 'timebin': 18, 'name': 'shared_choice_15'},
    
    
    # # # 14 cue information first and second squares
    # {'session_id': '2024-12-06_16-49', 'y_name': 'cue', 'conditioner': 'choice_R1',
    #  'train_stratum': 1, 'interval': 'cue_entry_interval', 'timebin': 9, 'name': 'shared_cue_14_firstsquare'},
    # # # 14
    # {'session_id': '2024-12-06_16-49', 'y_name': 'cue', 'conditioner': 'choice_R1',
    #  'train_stratum': 1, 'interval': 'cue_entry_interval', 'timebin': 9+19, 'name': 'shared_cue_14_secondquare'},

    # the clearly joint cue choice overlap from session 13
    # session13. --  the one to reference
    {'session_id': '2024-12-04_18-06', 'y_name': 'cue', 'conditioner': 'choice_R1',
     'train_stratum': 1, 'interval': 'cue_entry_interval', 'timebin': 9+4+5, 'name': 'poster26_S13_cueMiddle'},
    
    # # session13. --  first square
    # {'session_id': '2024-12-04_18-06', 'y_name': 'cue', 'conditioner': 'choice_R1',
    #  'train_stratum': 1, 'interval': 'cue_entry_interval', 'timebin': 8, 'name': 'S13_Cue_cue_choice_overlap_1stsquare'},
    # {'session_id': '2024-12-04_18-06', 'y_name': 'cue', 'conditioner': 'choice_R1',
    #  'train_stratum': 1, 'interval': 'cue_entry_interval', 'timebin': 9, 'name': 'S13_Cue_cue_choice_overlap_1stsquare'},
    # {'session_id': '2024-12-04_18-06', 'y_name': 'cue', 'conditioner': 'choice_R1',
    #  'train_stratum': 1, 'interval': 'cue_entry_interval', 'timebin': 10, 'name': 'S13_Cue_cue_choice_overlap_1stsquare'},
    
    # {'session_id': '2024-12-04_18-06', 'y_name': 'choice_R1', 'conditioner': 'cue',
    #  'train_stratum': 1, 'interval': 'R1_entry_interval', 'timebin': 22+7, 'name': 'S13_Choice_cue_choice_overlap'},
    
    
    
    
]



plt.close("all")
y_predicted_names = ['cue', 'choice_R1', ]  # pick from ALL_Y_PREDICTED_NAMES; pass a list to add columns
# y_predicted_names = ['cue']  # pick from ALL_Y_PREDICTED_NAMES; pass a list to add columns
# y_predicted_names = ['cue',]  # pick from ALL_Y_PREDICTED_NAMES; pass a list to add columns
# y_predicted_names = ['cueANDstopR1']  # pick from ALL_Y_PREDICTED_NAMES; pass a list to add columns
x_modalities = ['mPFC']  # pick from ALL_MODALITIES; pass a list to add columns
which_intervals = ['cue_entry_interval', 'cue_exit_interval',  'R1_entry_interval' ]  # pick from ALL_INTERVALS; pass a list to add columns
# which_intervals = ['R1_entry_interval',  ]  # pick from ALL_INTERVALS; pass a list to add columns
# which_intervals = ['R1_entry_interval']  # pick from ALL_INTERVALS; pass a list to add columns

# svm_results = analytics.get_analytics('SVML1CueOutcomeChoicePredCCGP', session_names=session_names)
# svm_results = analytics.get_analytics('SVML1CueOutcomeChoicePred2', session_names=session_names)
# svm_results = analytics.get_analytics('SVML1CueOutcomeChoicePred2_noNorm', session_names=session_names)
svm_results = analytics.get_analytics('SVML1CueOutcomeChoicePred2CCGP_noNorm', session_names=session_names)

# Quick/hacky patch, not a real CCGP fit: this older analytic never split by
# conditioner/train_stratum at all, so highlight_selections entries that pin those (e.g.
# 'shared_cue_14_*', 'S13_*' - all of which use train_stratum=1) can't match a bare
# naive row. Rather than editing every selection dict, fake a single "stratum 1" CCGP
# direction on every row instead: conditioner = whichever OTHER target this one would've
# been conditioned on (matching what highlight_selections already assumes), train_stratum
# = 1 (every current selection uses 1, never 0). With train_stratum non-NaN,
# _split_ccgp_primary_secondary puts every row in `primary` and leaves `secondary` empty
# (no real second direction exists to fill it) - so _resolve_ccgp_w_stratum happily
# elects stratum 1 (it's the only one available) and _highlighted_rows' matching against
# highlight_selections goes through, while _plot_f1_lines' has_secondary stays False
# (secondary is empty) so it still draws the plain naive-style single line, not a fake
# two-direction CCGP comparison. `more_reliable` also has to be added (even though
# there's only ever one row per group, so its value never actually matters) -
# _split_ccgp_primary_secondary's own sort_values references it unconditionally once
# train_stratum is non-NaN, so it KeyErrors without it (caught by testing this patch
# before handing back). The other genuinely-missing columns (test_stratum, best_C,
# n_train*, n_test*, analysis) are only ever read on the has_secondary=True path, so
# they're left absent - not needed for this to run.
# _conditioner_for = {'cue': 'choice_R1', 'choice_R1': 'cue', 'choice_R2': 'cue'}
# svm_results['conditioner'] = svm_results['predict_y_name'].map(_conditioner_for)
# svm_results['train_stratum'] = 1
# svm_results['more_reliable'] = True


session_plots_dir = os.path.join(output_dir, 'part2_plots')
# session_plots_dir = os.path.join(output_dir, 'cue_analysis')
os.makedirs(session_plots_dir, exist_ok=True)

# Set to False once you've exported the first session's figure and just want the rest to
# match it without their own copy of the shared y-axis label/ticks.

# Main execution. Sessions are resolved and sorted up front (rather than filtered
# inline while looping) so next_session_delta_days below can look one session ahead -
# the day-gap arrow on session N's figure describes the gap to session N+1's, which
# requires knowing N+1 before N finishes plotting.
plot_session_ids = sorted(
    (s_id for s_id in svm_results.index.unique('session_id')
    #  if pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') < to_s_id_07),
    #  if pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') <= to_s_id_15),
    # bewteen the two
    # if pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') > to_s_id_07 and
    if pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') > to_s_id_06 and
       pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') < to_s_id_15),
    key=lambda s_id: pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M'))

SVM_CCGP_STRATUM = {'cue': 1, 'choice_R1': 1, }

all_saved_png_paths, all_saved_svg_paths = [], []
for i, s_id in enumerate(plot_session_ids):
    print(f"Plotting session {s_id}")
    next_session_delta_days = None
    
    fr_z_session = None
    full_name = next(sn for sn in session_names if sn.startswith(s_id))
    fr_z_session = analytics.get_analytics(
        'FiringRate40msZ', session_names=[full_name]).xs(s_id, level='session_id')


    if i < len(plot_session_ids) - 1:
        this_ts = pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M')
        next_ts = pd.to_datetime(plot_session_ids[i + 1], format='%Y-%m-%d_%H-%M')
        # ceil, not Timedelta.days (which truncates) - a 23h gap is still a full day apart
        # as far as the reader's concerned, not "0d"
        next_session_delta_days = int(np.ceil((next_ts - this_ts).total_seconds() / 86400))

    saved_png_paths, saved_svg_paths = plot_session_analysis(
        s_id, t0_events, behavior, svm_results, show_colorbar=False,
        ccgp_w_train_stratum=SVM_CCGP_STRATUM,
        # min_metric=.55,
        metric='balanced_accuracy', show_shuffle=True, smooth=2,
        show_w_comparison=True, save_dir=session_plots_dir, save_markdown=False,
        x_modalities=x_modalities, y_predicted_names=y_predicted_names, 
        show_covariate_plot=True,
        which_intervals=which_intervals, show_title=True, show_ccgp_secondary=True,
        show_ylabel=False, next_session_delta_days=next_session_delta_days,
        ith_to_dt_session=ith_to_dt_session,
        highlight_selections=highlight_selections,
        show_w_variance=True,
        fr_z_session=fr_z_session,
        
        )
    if saved_png_paths:
        all_saved_png_paths.append(saved_png_paths)
        all_saved_svg_paths.append(saved_svg_paths)
    # break

# highlight_selections above only drew the callout boxes - the plots hand no data back.
# One row per resolved selection, with each selection's own `w` key filled in place;
# inspect e.g. selected_w_rows_df.w or .acc_aggr directly.
selected_w_rows_df = resolve_w_selections(highlight_selections, svm_results,
                                           x_modality=x_modalities[0],
                                           ccgp_w_train_stratum=SVM_CCGP_STRATUM)
# First proper plot used in poster. Play around with the part above to select
# sessions, intervals, y_predicted_names, x_modalities.

if all_saved_png_paths:
    canvas_path = os.path.join(session_plots_dir, 'cueChoiceR1_L1_CCGP_HP.png')
    build_canvas(all_saved_png_paths, canvas_path)
    print(f"Saved combined canvas of {len(all_saved_png_paths)} sessions to {canvas_path}")

    # cconditioner = svm_res['conditioner']anvas_path_svg = os.path.join(session_plots_dir, 'cue_choiceR1_L1_CCGP.svg')
    # build_canvas_svg(all_saved_svg_paths, canvas_path_svg)
    # print(f"Saved combined SVG canvas of {len(all_saved_svg_paths)} sessions to {canvas_path_svg}")

# %%
selected_w_rows_df
# %%
# get_session_sequence_Ws / plot_session_sequence_w_comparison used to be redefined
# here, shadowing svm_plotting_lib's own copies for every cell run after this one.
# The library versions had meanwhile gained smooth/ith_to_dt_session/
# reference_session_id/as_column/mask_chance_band, so this notebook and lm4_plots.py
# were producing different figures from the same call. They are now imported above.
# %%
# Cross-session weight comparison: fixed X_modality and predict_y_name, sequence of
# sessions. Reuses the same session subset/filtering as the main execution loop above.
session_sequence = [s_id for s_id in svm_results.index.unique('session_id')
                    #  if to_s_id_07 < pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') < to_s_id_15]
                    # if pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') < to_s_id_07]
                     if pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') < to_s_id_15]

x_modality_for_sequence = 'mPFC'  # pick a single X_modality

# reuses `highlight_selections` defined earlier (before the main per-session loop) - the
# same hand-picked W's get called out here too.

sequence_saved_paths, sequence_saved_svg_paths = [], []
for y_predicted_name in y_predicted_names:  # single predict_y_name per plot; loop covers each
    save_path = os.path.join(session_plots_dir,
                              f'session_sequence_{y_predicted_name}_{x_modality_for_sequence}.png')
    save_path_svg = os.path.join(session_plots_dir,
                                  f'session_sequence_{y_predicted_name}_{x_modality_for_sequence}.svg')
    # highlight_selections draws the callout boxes only; selected_w_rows_df (resolved
    # right after the main per-session loop above, via resolve_w_selections) is where the
    # W's themselves come from
    plot_session_sequence_w_comparison(
        session_sequence, svm_results, t0_events, x_modality_for_sequence, y_predicted_name, which_intervals,
        ccgp_w_train_stratum={'cue': 1, 'choice_R1': 1, 'choice_R2': 1}, min_metric=.55,
    #    ccgp_w_train_stratum={'choice_R1':0,}, min_metric=.55,
        highlight_selections=highlight_selections,
        save_path=save_path, save_path_svg=save_path_svg)
    sequence_saved_paths.append([save_path])
    sequence_saved_svg_paths.append([save_path_svg])

if sequence_saved_paths:
    canvas_path = os.path.join(session_plots_dir, 'L1_CCGP_W_canvas_cue_trainedOnR1Stop.png')
    build_canvas(sequence_saved_paths, canvas_path)
    print(f"Saved combined session-sequence canvas of {len(sequence_saved_paths)} plots to {canvas_path}")

    # canvas_path_svg = os.path.join(session_plots_dir, 'cue_L1_upto7_W_canvas.svg')
    # build_canvas_svg(sequence_saved_svg_paths, canvas_path_svg)
    # print(f"Saved combined SVG session-sequence canvas of {len(sequence_saved_svg_paths)} plots to {canvas_path_svg}")

# %%
# The scalar summary of the W-similarity blocks above: cue axis vs choice axis, one point
# per session. There the cue-vs-choice question is a whole rectangle of (timebin, timebin)
# pairs; here it is only that rectangle's MATCHED-timebin diagonal - "at this moment, do
# the two decoders point the same way" - reduced to one line across sessions so a trend
# over learning is readable at a glance.
#
# Angles are FOLDED at 90 (a > 90 -> 180 - a), so 0 = collinear and 90 = orthogonal. Not
# cosmetic: an SVM's w has an arbitrary overall sign (swap which class is 1 and w flips),
# so a pair at 175 degrees is exactly as aligned as one at 5, and leaving them unfolded
# would average the two into a meaningless ~90. It does mean "aligned" here says nothing
# about whether cue-1 and stop point the same way or opposite ways.
#
# Still inline, per plotlib's promotion rule - this moves into svm_plotting_lib as soon as
# it is called twice, called in a session loop, or needs its own save_dir.
#
# Own font sizes, the same way w_plotting_lib's figures carry W_FIGURE_RC: this notebook
# sets 15pt poster fonts globally (cell 4), and on a panel this small that alone puts the
# title and the y-label off the canvas.
W_ANGLE_RC = {'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
              'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7}


@plt.rc_context(W_ANGLE_RC)
def plot_w_angle_over_sessions(session_ids, svm_results, t0_events, x_modality,
                                which_intervals, y_name_a='cue', y_name_b='choice_R1',
                                min_metric=0.55, metric='balanced_accuracy', smooth=None,
                                pairs='diagonal', ccgp_w_train_stratum=None,
                                ith_to_dt_session=None, show_chance_band=True,
                                point_size=None, save_dir=None, stem=None):
    """Angle between the `y_name_a` and `y_name_b` W's, per session, as a collapsed scatter.

    The cell above draws this comparison as a heatmap: a whole (timebin, timebin)
    rectangle per session. This collapses that rectangle to one column of small dots per
    session, plus the mean across them as a line, so a trend over sessions is readable
    instead of having to be eyeballed across nine matrices.

    `pairs` is WHICH cells of that rectangle get collapsed:
      'diagonal' (default) - only matched timebins, i.e. "at this moment, do the two
          decoders point the same way". One dot per timebin.
      'all' - every (timebin_a, timebin_b) cell, i.e. the whole rectangle. Answers a
          different question ("is the cue axis anywhere near the choice axis at any lag")
          and is dominated by far-apart pairs, since there are n^2 of those and only n
          matched ones - so the mean sits higher and is not comparable to 'diagonal'.

    `min_metric` is the subsetting knob: a timebin contributes its angle only if BOTH
    decoders scored at least this at that timebin - the same rule (and, via `smooth`, the
    same smoothed score) that dims a cell in plot_w_comparison's min_metric masking, so
    the two figures are reading the same subset. An angle between two W's that are not
    decoding anything is just noise around 90, and a handful of those drags a session's
    mean up. None keeps every fitted timebin. A NaN score - a collapsed fit, or the
    zero-row interval separators the blocks carry - never passes.

    `metric`/`smooth`/`ccgp_w_train_stratum`/`ith_to_dt_session` pass straight through to
    get_session_sequence_Ws, so this reads the same W's, the same elected CCGP direction
    and the same session labels as the sequence figure above.

    `point_size` overrides the dot size (default: 8 for 'diagonal', 1.5 for 'all', which
    puts thousands of dots in one column).

    Returns {'angles': long DataFrame (one row per contributing pair), 'per_session':
    the per-session mean/n table, 'paths': save_figure's {format: path} dict}.
    """
    blocks = {}
    for y_name in (y_name_a, y_name_b):
        block_Ws, block_labels, _, _, block_metrics, _ = get_session_sequence_Ws(
            session_ids, svm_results, t0_events, x_modality, y_name, which_intervals,
            metric=metric, ccgp_w_train_stratum=ccgp_w_train_stratum, smooth=smooth,
            ith_to_dt_session=ith_to_dt_session)
        blocks[y_name] = (block_Ws, block_labels, block_metrics)

    records = []
    for k, s_id in enumerate(session_ids):
        Wa, Wb = blocks[y_name_a][0][k], blocks[y_name_b][0][k]
        ma, mb = blocks[y_name_a][2][k], blocks[y_name_b][2][k]
        label = blocks[y_name_a][1][k].split(' ')[0]   # 'S7 (train:stop)' -> 'S7'
        # Both blocks are built from the same which_intervals and the same t0 geometry, so
        # equal shape means row k is the same (interval, timebin) on both sides - and an
        # unequal one means a target is missing an interval (or a whole session) here,
        # which is exactly when a positional diagonal would compare the wrong moments.
        if Wa.size == 0 or Wb.size == 0:
            missing = ' and '.join(y for y, W in ((y_name_a, Wa), (y_name_b, Wb)) if W.size == 0)
            print(f"{label} ({s_id}): no fitted W for {missing} in these intervals, skipped")
            continue
        if Wa.shape != Wb.shape:
            print(f"{label} ({s_id}): {y_name_a} block {Wa.shape} vs {y_name_b} {Wb.shape}"
                  f" - not row-aligned, skipped")
            continue
        norms_a, norms_b = np.linalg.norm(Wa, axis=1), np.linalg.norm(Wb, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            if pairs == 'diagonal':
                cos_sim = np.sum(Wa * Wb, axis=1) / (norms_a * norms_b)
                ok_a, ok_b = ma, mb          # one score per point, on each side
            else:
                cos_sim = (Wa @ Wb.T) / np.outer(norms_a, norms_b)
                ok_a, ok_b = ma[:, np.newaxis], mb[np.newaxis, :]   # broadcast over the grid
        # Fold at 90: an SVM's w has an arbitrary overall sign, so 175 degrees is as
        # aligned as 5 (see the note above the function).
        angles = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
        angles = np.where(angles > 90, 180 - angles, angles)
        keep = np.isfinite(angles)
        n_fitted = int(keep.sum())
        if min_metric is not None:
            # `>=`, so a NaN score fails - same reasoning as _angle_alpha_mask's ~(x >= t)
            keep &= (ok_a >= min_metric) & (ok_b >= min_metric)
        idx_a, idx_b = (np.nonzero(keep) if pairs != 'diagonal'
                        else (np.flatnonzero(keep),) * 2)
        for row_a, row_b in zip(idx_a, idx_b):
            records.append(dict(session_id=s_id, label=label,
                                row_a=int(row_a), row_b=int(row_b),
                                angle=float(angles[(row_a, row_b) if pairs != 'diagonal' else row_a]),
                                **{f'{metric}_{y_name_a}': float(ma[row_a]),
                                   f'{metric}_{y_name_b}': float(mb[row_b])}))
        if n_fitted and not keep.any():
            print(f"{label} ({s_id}): no timebin has both decoders >= {min_metric} "
                  f"({n_fitted} fitted) - no point drawn")

    angles_df = pd.DataFrame(records)
    if angles_df.empty:
        print("nothing to plot")
        return {'angles': angles_df, 'per_session': angles_df, 'paths': {}}
    # session order follows session_ids, not the groupby's sort order
    order = [blocks[y_name_a][1][k].split(' ')[0] for k, s_id in enumerate(session_ids)
             if s_id in set(angles_df.session_id)]
    per_session = (angles_df.groupby('label')['angle'].agg(['mean', 'std', 'count'])
                    .reindex(order))

    dot_size = point_size if point_size is not None else (8 if pairs == 'diagonal' else 1.5)
    fig, ax = plt.subplots(figsize=(max(4.5, min(16.0, 0.6 * len(order))), 3.0))
    if show_chance_band:
        # What a RANDOM pair of directions in this modality's dimensionality does, folded
        # the same way the data is (see angle_2std_thr): 95% of random pairs land in the
        # band, so only points BELOW it are more aligned than chance. The null belongs
        # here rather than on the sequence figure - this is the plot that compares two
        # decoders against each other.
        lower, upper = angle_2std_thr[x_modality]
        ax.axhspan(min(lower, 180 - upper), 90, color='gray', alpha=0.15, lw=0,
                   label='random pair (95%)')
    xs = np.arange(len(order))
    for x, label in zip(xs, order):
        vals = angles_df.loc[angles_df.label == label, 'angle'].to_numpy()
        ax.scatter(np.full(vals.size, x), vals, s=dot_size, color='black',
                   alpha=0.25 if pairs == 'diagonal' else 0.08, lw=0)
        ax.annotate(f'n={vals.size}', (x, 2), ha='center', va='bottom', fontsize=7,
                    color='gray')
    ax.plot(xs, per_session['mean'].to_numpy(), '-o', color='black', lw=2, ms=5,
            label=f"mean over {'timebins' if pairs == 'diagonal' else 'pairs'}")
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=45, ha='right')
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 30, 60, 90])
    ax.set_ylabel(f'angle({y_name_a}, {y_name_b}) [deg]\n0=collinear, 90=orthogonal')
    # two lines: on a 7-session panel the figure is ~4.5in wide and the one-line version
    # of this ran off both ends of the canvas
    ax.set_title(f'{x_modality}: W angle, {y_name_a} vs {y_name_b} ({pairs})'
                 + (f'\n{metric} >= {min_metric}' if min_metric is not None else ''))
    # lifted clear of the n= row along the bottom; the lower half of the axes is empty in
    # practice (a pair of fitted decoders is rarely more aligned than ~30 degrees)
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.12), frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    paths = save_figure(fig, save_dir,
                        stem or f'w_angle_{y_name_a}_vs_{y_name_b}_{x_modality}_{pairs}')
    return {'angles': angles_df, 'per_session': per_session, 'paths': paths}


w_angle_result = plot_w_angle_over_sessions(
    session_sequence, svm_results, t0_events, x_modality_for_sequence, which_intervals,
    y_name_a='cue', y_name_b='choice_R1',
    min_metric=.55,                 # None keeps every fitted timebin
    pairs='all',                # 'all' collapses the whole (timebin, timebin) grid
    smooth=2,                        # matched to the figures above
    ccgp_w_train_stratum={'cue': 1, 'choice_R1': 1, 'choice_R2': 1},
    ith_to_dt_session=ith_to_dt_session,
    save_dir=session_plots_dir)
print(w_angle_result['per_session'])

# %%














































# plot_W_list + the cortical-region gradient moved to w_plotting_lib.py
# %%
# Example: plot the W's pulled out via highlight_selections above - either the
# per-session (selected_w_rows_df) or cross-session (selected_w_rows_df)
# set, or both concatenated together.
plot_W_list(selected_w_rows_df, plot_type='stem', save_path_svg=os.path.join(nas_dir, 'Simon', 'lm4', 'highlighted_Ws_stem.svg'), min_abs_w=0.1, )
# plot_W_list(selected_w_rows_df, plot_type='line')




































    








# %%
# The W-projection machinery moved to w_plotting_lib.py; _x_modality_fr became
# plotlib.session_data.session_fr_for_modality.


# %%
# W-variance-explained moved to w_plotting_lib.py; _pca_eigen/plot_pca_eigenvalues
# became plotlib.pca.pca_eigen/plot_pca_eigenvalues (shared with dim_plotting_lib).

# %%
# The kinematics-vs-projection figures moved to w_plotting_lib.py

# %%
# Demo: does forward/rotational velocity or acceleration, head angle, or overall
# movement energy explain subspace activation over one session/interval/timebin_range -
# on top of (or instead of) the task covariate the W was fit to distinguish? Every
# (trial, bin) sample in the window is its own scatter point (see
# _window_trial_bin_values) - pooled the same way get_interval_window_activity pools
# trials and bins for PCA below, not averaged into one point per trial. Pick any w_row
# the same way the W-projection/W-variance demo cells above do; `kinematics_offset`
# (native bins) shifts the kinematics side's whole window later than the projection's -
# set to 0 to compare at the same window.
kinematics_w_row = selected_w_rows_df.iloc[0]
kinematics_w_row = selected_w_rows_df[selected_w_rows_df['name'] == 'poster26_S13_choiceEnd'].iloc[0]
kinematics_interval, kinematics_timebin_range, kinematics_offset = 'R1_entry_interval', (5, 15), 0

# Same "where does this W actually point" sanity checks used in the standalone
# W-variance-explained section further down - called here too, on the exact same
# window chosen above, before reading into any kinematic correlation below.
# 1) plot_w_variance_explained/plot_pca_eigenvalues: PC spectrum + purple dashed line
#    for how much of the pooled trial-to-trial + within-window variance in this window
#    the W direction itself explains. Line near 0 (well below even the smallest PCs
#    shown) means the decoder barely moves here - a kinematic correlation found below
#    is then worth interpreting cautiously, since there's little of "its own" variance
#    to explain.
# 2) plot_w_projection_session(..., highlight_range=...): the same window shaded
#    directly on the actual projection trace (pink span + 'W: x% var' - see
#    plot_w_projection's own highlight_range/highlight_var_pct), so the number from (1)
#    can be seen in context of what the trace is actually doing there.
kinematics_target_spec = {'session_id': kinematics_w_row['session_id'], 'interval': kinematics_interval,
                           'timebin_range': kinematics_timebin_range}
_ = plot_w_variance_explained(kinematics_target_spec, kinematics_w_row, t0_events, session_names,
                               ith_to_dt_session=ith_to_dt_session)
_ = plot_w_projection_session(
    kinematics_w_row['session_id'], kinematics_w_row, t0_events, behavior, session_names, [kinematics_interval],
    show_covariate_plot=False, smooth=2, ith_to_dt_session=ith_to_dt_session,
    highlight_range=(kinematics_interval, *kinematics_timebin_range))

kinematics_correlations = plot_w_projection_kinematics(
    kinematics_w_row['session_id'], kinematics_w_row, t0_events, behavior, session_names,
    kinematics_interval, kinematics_timebin_range, kinematic_offset=kinematics_offset,
    color_by=kinematics_w_row['predict_y_name'], ith_to_dt_session=ith_to_dt_session)
kinematics_correlations


# %%
# plot_session_projection_kinematics moved to w_plotting_lib.py

# %%
# Demo: same question as the trial-windowed demo above, but over the whole session's
# continuous activity - is a kinematic relationship specific to the task-relevant
# window(s), or does it hold throughout the whole session (including inter-trial time)?
# Same kinematics_w_row as above; max_points keeps the scatter responsive given tens of
# thousands of bins per session (set to None to plot every bin).
session_kinematics_correlations = plot_session_projection_kinematics(
    kinematics_w_row['session_id'], kinematics_w_row, behavior, session_names,
    color_by=kinematics_w_row['predict_y_name'], max_points=20000,
    ith_to_dt_session=ith_to_dt_session)
session_kinematics_correlations

# %%

# %%
selected_w_rows_df
# %%
# loop over every session and project one specific, fixed w - s_id (which session's
# activity to project into) is independent of w_row['session_id'] (which session that W
# was fit on), so w_row is picked once, outside the loop, and reused unchanged for every
# target session.
w_which_intervals = ['cue_entry_interval', 'R1_entry_interval']  # pick from ALL_INTERVALS


    # # # 14 cue information first and second squares
    # {'session_id': '2024-12-06_16-49', 'y_name': 'cue', 'conditioner': 'choice_R1',
    #  'train_stratum': 1, 'interval': 'cue_entry_interval', 'timebin': 9, 'name': 'shared_cue_14_firstsquare'},
    # # # 14
    # {'session_id': '2024-12-06_16-49', 'y_name': 'cue', 'conditioner': 'choice_R1',
    #  'train_stratum': 1, 'interval': 'cue_entry_interval', 'timebin': 9+14, 'name': 'shared_cue_14_secondquare'},

    # # the clearly joint cue choice overlap from session 13
    # # session13. --  the one to reference
    # {'session_id': '2024-12-04_18-06', 'y_name': 'cue', 'conditioner': 'choice_R1',
    #  'train_stratum': 1, 'interval': 'cue_entry_interval', 'timebin': 9+4, 'name': 'S13_Cue_cue_choice_overlap'},
    # {'session_id': '2024-12-04_18-06', 'y_name': 'choice_R1', 'conditioner': 'cue',
    #  'train_stratum': 1, 'interval': 'R1_entry_interval', 'timebin': 22+7, 'name': 'S13_Choice_cue_choice_overlap'},
    
    
# for shared choice space over sessions
# w_row = selected_w_rows_df[selected_w_rows_df['name'] == 'shared_choice_13'].iloc[0]  # the choice axis that is quite preserved
    
# for the conserverd first square activations    
# w_row = selected_w_rows_df[selected_w_rows_df['name'] == 'shared_cue_14_firstsquare'].iloc[0]  # the choice axis that is quite preserved

# the bigger square coming after, reactivating on error in Ssesion13
# w_row = selected_w_rows_df[selected_w_rows_df['name'] == 'shared_cue_14_secondquare'].iloc[0]  # the choice axis that is quite preserved




# the bigger square coming after, reactivating on error in Ssesion13
# w_row = selected_w_rows_df[selected_w_rows_df['name'] == 'S13_Cue_cue_choice_overlap_2ndsquare'].iloc[0]  # the choice axis that is quite preserved


# shared_choice_13
# S13_Cue_cue_choice_overlap_2ndsquare

# the only real alignment in session 13, from cue side
# w_row = selected_w_rows_df[selected_w_rows_df['name'] == 'S13_Cue_cue_choice_overlap'].iloc[0]  # the choice axis that is quite preserved
# # the only real alignment in session 13, from choice side

# w_row = selected_w_rows_df[selected_w_rows_df['name'] == 'S13_Cue_cue_choice_overlap_2ndsquare'].iloc[0]  # the choice axis that is quite preserved
w_row = selected_w_rows_df[selected_w_rows_df['name'] == 'poster26_S13_choiceEnd'].iloc[0]  # the choice axis that is quite preserved
w_row = selected_w_rows_df[selected_w_rows_df['name'] == 'poster26_S13_cueMiddle'].iloc[0]  # the choice axis that is quite preserved


# w_row = selected_w_rows_df.iloc[0]  # the first choice  squarey in S7

session_plots_dir = os.path.join(nas_dir, 'Simon', 'lm4', 'choice7')
os.makedirs(session_plots_dir, exist_ok=True)

wproj_saved_paths = []
for s_id in plot_session_ids:
# for s_id in ['2024-12-04_18-06']:
    saved_path = plot_w_projection_session(s_id, w_row, t0_events, behavior, session_names, w_which_intervals,
                                            save_dir=session_plots_dir, smooth=2,
                                            ith_to_dt_session=ith_to_dt_session)
    wproj_saved_paths.append(saved_path)

# if wproj_saved_paths:
#     # one image per session (not a stack), so wrap each path in its own single-element
#     # list - build_canvas expects one list of paths *per session* to stack vertically.
#     canvas_path = os.path.join(session_plots_dir, 'S13_error_replay.svg')
#     build_canvas([[p] for p in wproj_saved_paths], canvas_path)
#     print(f"Saved combined canvas of {len(wproj_saved_paths)} sessions to {canvas_path}")






























# %%
# Same fixed w_row as above, but restricted to just the sessions that have a matched W
# (see highlight_selections/selected_w_rows_df above) - one figure per session,
# same hand-picked target/direction each time, the same way the main per-session loop
# iterates plot_session_ids - so this W's own projection drift/consistency across
# sessions shows up the same way plot_session_sequence_w_comparison already shows it for
# the W's themselves. Uses its own w_which_intervals, not the shared `which_intervals`
# global other cells rely on. show_covariate_plot=True (default) adds one small panel per
# covariate below the aggregate row - set False for just the velocity+aggregate rows.
w_which_intervals = ['cue_entry_interval', 'R1_entry_interval']  # pick from ALL_INTERVALS

wproj_saved_paths = []
for i,s_id in enumerate(selected_w_rows_df['session_id']):
    w_row = selected_w_rows_df.iloc[i]  # pick the W that was actually fit on this session, will go from S07-S15
    saved_path = plot_w_projection_session(s_id, w_row, t0_events, behavior, session_names, w_which_intervals,
                                            save_dir=session_plots_dir, smooth=2,
                                            ith_to_dt_session=ith_to_dt_session)
    wproj_saved_paths.append(saved_path)

if wproj_saved_paths:
    canvas_path = os.path.join(session_plots_dir, 'wproj_canvas_matched_sessions.png')
    build_canvas([[p] for p in wproj_saved_paths], canvas_path)
    print(f"Saved combined canvas of {len(wproj_saved_paths)} sessions to {canvas_path}")


# %%
# Whole-session PCA / subspace drift moved to dim_plotting_lib.py (population
# geometry, not SVM); canonical angles are now plotlib.pca.canonical_angles_deg.

# %%
# Part 1: standalone whole-session PCA baseline, looped over every plotted session (not
# tied to any specific decoder) - see plot_session_pca. Aggregated into one canvas the
# same way the W-projection loops above are (see build_canvas). Eigenvectors/eigenvalues
# are kept around (not just the saved paths) to feed the cross-session canonical-angle
# similarity comparison right below (plot_session_pca_similarity).
pca_x_modality = 'mPFC'  # pick a single X_modality

session_pca_saved_paths, session_pca_eigvals, session_pca_eigvecs, session_pca_labels = [], [], [], []
for s_id in plot_session_ids:
    saved_path, eigenvalues_pct, eigenvectors = plot_session_pca(s_id, session_names, pca_x_modality,
                                                                  save_dir=session_plots_dir,
                                                                  ith_to_dt_session=ith_to_dt_session)
    session_pca_saved_paths.append(saved_path)
    session_pca_eigvals.append(eigenvalues_pct)
    session_pca_eigvecs.append(eigenvectors)
    s_num = ith_to_dt_session.get(s_id, "Unknown")
    session_pca_labels.append(f'S{s_num:02d}' if isinstance(s_num, int) else f'S{s_num}')

if session_pca_saved_paths:
    canvas_path = os.path.join(session_plots_dir, 'pca_canvas_all_sessions.png')
    build_canvas([[p] for p in session_pca_saved_paths], canvas_path)
    print(f"Saved combined canvas of {len(session_pca_saved_paths)} sessions to {canvas_path}")

# %%
# More intuitive than a canonical-angle heatmap (see plot_session_pca_similarity /
# _canonical_angles_deg below, kept but not used here): fix the *last* plotted session's
# whole-session PCA subspace as a reference, then ask - directly, in the same %-variance
# units used throughout this section - how much of every *earlier* session's own
# variance those same fixed directions still capture (see plot_pca_reference_drift). A
# bar noticeably below the dashed reference line means that session's activity has
# drifted out of the reference session's own principal subspace.
pca_drift_path = os.path.join(session_plots_dir, 'pca_reference_drift.png')
pca_drift_labels, pca_drift_pct = plot_pca_reference_drift(
    plot_session_ids[0], plot_session_ids[1:], session_names, pca_x_modality, save_path=pca_drift_path,
    ith_to_dt_session=ith_to_dt_session)

# %%
# Canonical-angle alternative (see plot_session_pca_similarity/_canonical_angles_deg
# above) - a symmetric all-pairs comparison, but in degrees rather than the more
# directly interpretable %-of-variance units plot_pca_reference_drift uses. Left here in
# case the full pairwise picture is still useful alongside the reference-drift view.
pca_sim_path = os.path.join(session_plots_dir, 'pca_canonical_angle_similarity.png')
pca_similarity = plot_session_pca_similarity(session_pca_eigvecs, session_pca_eigvals, session_pca_labels,
                                              save_path=pca_sim_path)

# %%
# Part 2+3: how much variance a specific, fixed W direction explains within one or more
# hand-picked (session, interval, timebin-range) windows - see plot_w_variance_explained.
# `w_row` is picked the same way as the W-projection cells above (any single row from
# selected_w_rows_df/selected_w_rows_df); w_variance_selections is a list
# shaped like highlight_selections, but naming activity windows (timebin_range, not a
# single fitted timebin) rather than fitted decoder rows to highlight.

# w_row = selected_w_rows_df[selected_w_rows_df['name'] == 'S13_Choice_cue_choice_overlap'].iloc[0]
# w_variance_selections = [
#     {'session_id': '2024-12-04_18-06', 'interval': 'R1_entry_interval', 'timebin_range': (20, 30),
#      'name': 'S13_cue_10-20'},
# ]

# DELETE-ME-EASILY: a small fallback, not a design decision. Picking a W by `name` here
# raised IndexError whenever that name matched nothing - the normal case for any selection
# pinning conditioner/train_stratum, since SVML1CueOutcomeChoicePred2 has no CCGP rows at
# all (only the stale ...CCGP analytic does); it also can't match if the sequence above was
# built over different intervals than the name refers to. Falls back to the first available
# W so the rest of the notebook still runs. Delete this helper and go back to a bare
# `[df['name'] == ...].iloc[0]` once the W-selection story is settled.
def _pick_w(df, name):
    match = df[df['name'] == name] if 'name' in df.columns else df.iloc[0:0]
    if not match.empty:
        return match.iloc[0]
    available = sorted(df['name'].dropna().unique()) if 'name' in df.columns else []
    print(f"WARNING: no highlighted W named {name!r} (available: {available}) - using the first instead")
    return df.iloc[0]


w_row = _pick_w(selected_w_rows_df, 'S13_Cue_cue_choice_overlap_2ndsquare')
w_variance_selections = [
    {'session_id': '2024-12-04_18-06', 'interval': 'cue_entry_interval', 'timebin_range': (20, 30), 'name': 'S14_cue_0-20'},
]

wvar_saved_paths = []
for target_spec in w_variance_selections:
    saved_path, w_var_pct, _ = plot_w_variance_explained(target_spec, w_row, t0_events, session_names,
                                                          save_dir=session_plots_dir,
                                                          ith_to_dt_session=ith_to_dt_session)
    print(f"{target_spec.get('name', target_spec['session_id'])}: W explains {w_var_pct:.1f}% of variance")
    wvar_saved_paths.append(saved_path)

if wvar_saved_paths:
    canvas_path = os.path.join(session_plots_dir, 'wvar_canvas.png')
    build_canvas([[p] for p in wvar_saved_paths], canvas_path)
    print(f"Saved combined canvas of {len(wvar_saved_paths)} selections to {canvas_path}")


# %%
# Same variance-explained number as above, but seen together with the actual projected
# activity trace it was computed from, rather than as its own standalone bar chart - the
# same (session, interval, timebin-range) window from w_variance_selections, reused here
# as plot_w_projection_session's `highlight_range` - shades that window directly on row
# 1 and annotates it with the same '% var' number plot_w_variance_explained computes.
target_spec = w_variance_selections[0]
wproj_highlight_saved_path = plot_w_projection_session(
    target_spec['session_id'], w_row, t0_events, behavior, session_names, w_which_intervals,
    highlight_range=(target_spec['interval'], *target_spec['timebin_range']),
    save_dir=session_plots_dir, smooth=2, ith_to_dt_session=ith_to_dt_session)

























































# %%
plot_session_ids = sorted(
    (s_id for s_id in svm_results.index.unique('session_id')
    #  if pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') < to_s_id_07),
    #  if pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') <= to_s_id_15),
    ),
    # bewteen the two
    # if pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') > to_s_id_07 and
    # if pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') > to_s_id_07 and
    #    pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') <= to_s_id_20),
    #    pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') <= to_s_id_15),
    key=lambda s_id: pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M'))
plot_session_ids

intervals = ["track_start_interval",
              "cue_entry_interval",
              "cue_exit_interval",
              "R1_entry_interval",
              "R2_entry_interval",
            #   "reward1_sound_interval",
            #   "reward2_sound_interval",
              "ITI_interval",
]

svm_results = analytics.get_analytics('SVML2CueOutcomeChoicePred2', session_names=session_names)
svm_results = analytics.get_analytics('SVM_TaskSubspaces', session_names=session_names)
svm_results.index.unique('session_id')

# %%
# Cross-session overview of the same score plot_session_analysis' row 1 draws one session
# at a time: one heatmap figure per y_name (one after the other, sharing one column layout
# and color scale), each row a session and each column a timebin - intervals concatenated
# left to right with the usual 1-column white separator and $t_{Interval}$/'1s' ticks.
# Either list can be None for "every one available" (resolved against the other's choice -
# so with the reward-sound intervals left out below, baseline_vs_t isn't offered either).
# save_path gets the y_name appended per figure; width_per_interval/row_height take bigger
# cells if these come out too small.

from svm_heatmap_plotting_lib import plot_metric_heatmap_sessions


session_plots_dir = os.path.join(output_dir, 'sessions_metric_heatmap')
os.makedirs(session_plots_dir, exist_ok=True)

heatmap_intervals = intervals
heatmap_y_names = None
heatmap_figs, heatmap_scores = plot_metric_heatmap_sessions(
    svm_results, t0_events, heatmap_intervals, heatmap_y_names, x_modality='mPFC',
    session_ids=plot_session_ids, metric='balanced_accuracy',
    ith_to_dt_session=ith_to_dt_session, smooth=2,
    cmap='afmhot_r', vmin=0.5, vmax=.85, #width_per_interval=0.5, row_height=0.5,
    ccgp_w_train_stratum={'cue': 1, 'choice_R1': 1},
    save_path=os.path.join(session_plots_dir, 'sessions_metric_heatmap.png'),
    save_path_svg=os.path.join(session_plots_dir, 'sessions_metric_heatmap.svg'),
)


# %%
# Averages + confidence-interval sizes for a whole collection of fits, on the same grid as
# the heatmap above: intervals left to right as columns, one row per y_name. Each cell
# collapses that (interval, target)'s time axis and sessions into a distribution, one violin
# per variant - so a better hyperparameter setting reads as higher in the score figure AND
# narrower in the CI-width figure (two figures, both on this grid). dist_summary holds the
# numbers per (target, interval, variant) - what to actually compare on, since a few
# hundredths' difference isn't readable off two violins. plot_type='hist' for step
# histograms instead.
#
# Imported here rather than relying on the import cell at the top, and reloaded, so this
# cell runs (and picks up lib edits) on its own without restarting the kernel.
import importlib
import svm_heatmap_plotting_lib
importlib.reload(svm_heatmap_plotting_lib)
from svm_heatmap_plotting_lib import plot_metric_distribution

dist_variants = {
    'L1': svm_results,
    'L1_noNorm': analytics.get_analytics('SVML1CueOutcomeChoicePred2_noNorm', session_names=session_names),
    # 'L2': analytics.get_analytics('SVML2CueOutcomeChoicePred2', session_names=session_names),
    # 'NewL2': analytics.get_analytics('SVM_TaskSubspaces', session_names=session_names),
    # 'L1_5bins': analytics.get_analytics('SVML1CueOutcomeChoicePred5', session_names=session_names),
    # 'L1_CCGP': analytics.get_analytics('SVML1CueOutcomeChoicePredCCGP', session_names=session_names),
    # 'L1_CCGP5': analytics.get_analytics('SVML1CueOutcomeChoicePredCCGP5', session_names=session_names),
}
dist_figs, dist_summary = plot_metric_distribution(
    dist_variants, which_intervals=intervals, y_predicted_names=None, x_modality='mPFC',
    session_ids=plot_session_ids, metric='balanced_accuracy',
    show_n=True,
    ccgp_w_train_stratum={'cue': 1, 'choice_R1': 1}, plot_type='violin',
    save_path=os.path.join(session_plots_dir, 'metric_distribution.png'),
)

# %%
analytics.get_analytics('SVML2CueOutcomeChoicePred', session_names=session_names)
# %%
