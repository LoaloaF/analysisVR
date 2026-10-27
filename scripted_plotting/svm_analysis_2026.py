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
from PIL import Image
import xml.etree.ElementTree as ET

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
    _rows_for_stratum, _selection_matches, _highlighted_rows, get_session_data, _resample_pos, _smooth,
    _ccgp_n_annotation, _plot_f1_lines, create_subplot, plot_covariate_f1_single, plot_correctness_f1_single,
    _covariate_specs_for_target, plot_covariate_f1, _write_wcomp_markdown, plot_w_comparison,
    _write_session_main_markdown, plot_session_analysis,
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
# plt.xscale('log')
svm_results

# %%
# fetch data again, this time the t0 events

t0_events = analytics.get_analytics('TrialWiseT0Events40ms', session_names=session_names)
t0_events.index = t0_events.index.droplevel(['entry_id', 'paradigm_id', 'animal_id'])
t0_events.set_index("trial_id", append=True, inplace=True)
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
    # shared choice R1 decoder across sessions - the one to reference/ compare
    # session07
    {'session_id': '2024-11-26_16-39', 'y_name': 'choice_R1',
     'interval': 'R1_entry_interval', 'timebin': 13, 'name': 'shared_choice_07'},
    # session09
    
    {'session_id': '2024-11-26_16-39', 'y_name': 'choice_R1',
     'interval': 'R1_entry_interval', 'timebin': 21, 'name': 'shared_choice_07'},
    # session09
    {'session_id': '2024-11-28_17-41', 'y_name': 'choice_R1',
     'interval': 'R1_entry_interval', 'timebin': 19, 'name': 'shared_choice_09'},
    # session11
    {'session_id': '2024-12-02_16-09', 'y_name': 'choice_R1', 'conditioner': 'cue',
     'train_stratum': 1, 'interval': 'R1_entry_interval', 'timebin': 15, 'name': 'shared_choice_11'},
    # session12
    {'session_id': '2024-12-03_16-23', 'y_name': 'choice_R1', 'conditioner': 'cue',
     'train_stratum': 1, 'interval': 'R1_entry_interval', 'timebin': 27, 'name': 'shared_choice_12'},
    # session13. --  the one to reference index = 4
    {'session_id': '2024-12-04_18-06', 'y_name': 'choice_R1', 'conditioner': 'cue',
     'train_stratum': 1, 'interval': 'R1_entry_interval', 'timebin': 22, 'name': 'shared_choice_13'},
    # 14
    {'session_id': '2024-12-06_16-49', 'y_name': 'choice_R1', 'conditioner': 'cue',
     'train_stratum': 1, 'interval': 'R1_entry_interval', 'timebin': 18, 'name': 'shared_choice_14'},
    # 15
    {'session_id': '2024-12-09_17-45', 'y_name': 'choice_R1', 'conditioner': 'cue',
     'train_stratum': 1, 'interval': 'R1_entry_interval', 'timebin': 18, 'name': 'shared_choice_15'},
    
    
    # # # 14 cue information first and second squares
    # {'session_id': '2024-12-06_16-49', 'y_name': 'cue', 'conditioner': 'choice_R1',
    #  'train_stratum': 1, 'interval': 'cue_entry_interval', 'timebin': 9, 'name': 'shared_cue_14_firstsquare'},
    # # # 14
    # {'session_id': '2024-12-06_16-49', 'y_name': 'cue', 'conditioner': 'choice_R1',
    #  'train_stratum': 1, 'interval': 'cue_entry_interval', 'timebin': 9+19, 'name': 'shared_cue_14_secondquare'},

    # the clearly joint cue choice overlap from session 13
    # session13. --  the one to reference
    {'session_id': '2024-12-04_18-06', 'y_name': 'cue', 'conditioner': 'choice_R1',
     'train_stratum': 1, 'interval': 'cue_entry_interval', 'timebin': 9+4+5, 'name': 'S13_Cue_cue_choice_overlap_2ndsquare'},
    
    # session13. --  first square
    {'session_id': '2024-12-04_18-06', 'y_name': 'cue', 'conditioner': 'choice_R1',
     'train_stratum': 1, 'interval': 'cue_entry_interval', 'timebin': 8, 'name': 'S13_Cue_cue_choice_overlap_1stsquare'},
    {'session_id': '2024-12-04_18-06', 'y_name': 'cue', 'conditioner': 'choice_R1',
     'train_stratum': 1, 'interval': 'cue_entry_interval', 'timebin': 9, 'name': 'S13_Cue_cue_choice_overlap_1stsquare'},
    {'session_id': '2024-12-04_18-06', 'y_name': 'cue', 'conditioner': 'choice_R1',
     'train_stratum': 1, 'interval': 'cue_entry_interval', 'timebin': 10, 'name': 'S13_Cue_cue_choice_overlap_1stsquare'},
    
    # {'session_id': '2024-12-04_18-06', 'y_name': 'choice_R1', 'conditioner': 'cue',
    #  'train_stratum': 1, 'interval': 'R1_entry_interval', 'timebin': 22+7, 'name': 'S13_Choice_cue_choice_overlap'},
    
]



plt.close("all")
# y_predicted_names = ['cue', 'choice_R1', ]  # pick from ALL_Y_PREDICTED_NAMES; pass a list to add columns
# y_predicted_names = ['cue']  # pick from ALL_Y_PREDICTED_NAMES; pass a list to add columns
y_predicted_names = ['choice_R1']  # pick from ALL_Y_PREDICTED_NAMES; pass a list to add columns
# y_predicted_names = ['cueANDstopR1']  # pick from ALL_Y_PREDICTED_NAMES; pass a list to add columns
x_modalities = ['mPFC']  # pick from ALL_MODALITIES; pass a list to add columns
which_intervals = ['R1_entry_interval',  ]  # pick from ALL_INTERVALS; pass a list to add columns
# which_intervals = ['R1_entry_interval',  ]  # pick from ALL_INTERVALS; pass a list to add columns
# which_intervals = ['R1_entry_interval']  # pick from ALL_INTERVALS; pass a list to add columns

# svm_results = analytics.get_analytics('SVML1CueOutcomeChoicePredCCGP', session_names=session_names)
svm_results = analytics.get_analytics('SVML1CueOutcomeChoicePred2', session_names=session_names)

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
     if pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') <= to_s_id_15),
    # bewteen the two
    # if pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') > to_s_id_07 and
    # if pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') > to_s_id_06 and
    #    pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') < to_s_id_15),
    key=lambda s_id: pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M'))

all_saved_png_paths, all_saved_svg_paths = [], []
all_highlighted_rows = []  # the actual selected W's across every session plotted below
for i, s_id in enumerate(plot_session_ids):
    print(f"Plotting session {s_id}")
    next_session_delta_days = None
    if i < len(plot_session_ids) - 1:
        this_ts = pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M')
        next_ts = pd.to_datetime(plot_session_ids[i + 1], format='%Y-%m-%d_%H-%M')
        # ceil, not Timedelta.days (which truncates) - a 23h gap is still a full day apart
        # as far as the reader's concerned, not "0d"
        next_session_delta_days = int(np.ceil((next_ts - this_ts).total_seconds() / 86400))

    saved_png_paths, saved_svg_paths, highlighted_rows = plot_session_analysis(
        s_id, t0_events, behavior, svm_results, show_colorbar=False,
        ccgp_w_train_stratum={'cue': 1, 'choice_R1': 1, },
        min_metric=.55,
        metric='balanced_accuracy', show_shuffle=True, smooth=2,
        show_w_comparison=True, save_dir=session_plots_dir, save_markdown=False,
        x_modalities=x_modalities, y_predicted_names=y_predicted_names, show_covariate_plot=True,
        which_intervals=which_intervals, show_title=True, show_ccgp_secondary=True,
        show_ylabel=False, next_session_delta_days=next_session_delta_days,
        ith_to_dt_session=ith_to_dt_session,
        highlight_selections=highlight_selections
        )
    if saved_png_paths:
        all_saved_png_paths.append(saved_png_paths)
        all_saved_svg_paths.append(saved_svg_paths)
    all_highlighted_rows.append(highlighted_rows)
    # break

# one row per matched highlight_selections entry, across every session plotted above -
# inspect e.g. all_highlighted_rows_df.w or .acc_aggr directly
all_highlighted_rows_df = pd.concat(all_highlighted_rows) if all_highlighted_rows else pd.DataFrame()
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
all_highlighted_rows_df.iloc[0].w
# %%
def get_session_sequence_Ws(session_ids, svm_results, x_modality, y_predicted_name, which_intervals,
                             metric='f1', ccgp_w_train_stratum=None, highlight_selections=None):
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
    just seeing a box drawn around them. Returns (block_Ws, block_labels,
    block_modalities, block_t0_positions, block_metrics, highlight_positions,
    highlighted_rows) - one entry per session, except highlight_positions/
    highlighted_rows, which have one entry per matched selection (highlighted_rows as a
    single DataFrame, empty if nothing matched)."""
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
        s_label = f'S{ith_to_dt_session.get(s_id, "?")}'

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
            ordered, which_intervals, metric_col=spec['aggr_col'])

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
                                        ccgp_w_train_stratum=None, min_metric=None, highlight_selections=None):
    """Cross-session SVM-weight similarity for a single (x_modality, y_predicted_name):
    one block per session (see plot_w_comparison for what a block means and
    get_session_sequence_Ws for how a CCGP session's single direction is chosen), in the
    given session order. Shows how stable/consistent a single decoder's weights are
    across sessions, the same way plot_w_comparison shows it across columns within one
    session. `metric`, `ccgp_w_train_stratum`, `min_metric`, and `highlight_selections`
    all pass straight through to get_session_sequence_Ws/plot_w_comparison - see there
    for what each does. Returns `highlighted_rows` (see get_session_sequence_Ws) - the
    actual matched data, not just the boxes drawn around it - empty if
    highlight_selections wasn't given or nothing matched."""
    block_Ws, block_labels, block_modalities, block_t0_positions, block_metrics, highlight_positions, \
        highlighted_rows = get_session_sequence_Ws(
            session_ids, svm_results, x_modality, y_predicted_name, which_intervals,
            metric=metric, ccgp_w_train_stratum=ccgp_w_train_stratum, highlight_selections=highlight_selections)
    title = f'{y_predicted_name} decoding from {x_modality} across sessions'
    fig_width = max(6, 1.2 * len(session_ids))
    plot_w_comparison(block_Ws, block_labels, block_modalities, title, fig_width,
                       block_t0_positions=block_t0_positions, save_path=save_path, save_path_svg=save_path_svg,
                       block_metrics=block_metrics, min_metric=min_metric, highlight_positions=highlight_positions)
    return highlighted_rows
    


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
sequence_highlighted_rows = []  # the actual selected W's - see highlight_selections above
for y_predicted_name in y_predicted_names:  # single predict_y_name per plot; loop covers each
    save_path = os.path.join(session_plots_dir,
                              f'session_sequence_{y_predicted_name}_{x_modality_for_sequence}.png')
    save_path_svg = os.path.join(session_plots_dir,
                                  f'session_sequence_{y_predicted_name}_{x_modality_for_sequence}.svg')
    highlighted_rows = plot_session_sequence_w_comparison(
        session_sequence, svm_results, x_modality_for_sequence, y_predicted_name, which_intervals,
        ccgp_w_train_stratum={'cue': 1, 'choice_R1': 1, 'choice_R2': 1}, min_metric=.55,
    #    ccgp_w_train_stratum={'choice_R1':0,}, min_metric=.55,
        highlight_selections=highlight_selections,
        save_path=save_path, save_path_svg=save_path_svg)
    sequence_saved_paths.append([save_path])
    sequence_saved_svg_paths.append([save_path_svg])
    sequence_highlighted_rows.append(highlighted_rows)

# one row per matched highlight_selections entry, across every y_predicted_name plotted
# above - inspect e.g. sequence_highlighted_rows_df.w or .acc_aggr directly
sequence_highlighted_rows_df = pd.concat(sequence_highlighted_rows) if sequence_highlighted_rows else pd.DataFrame()

if sequence_saved_paths:
    canvas_path = os.path.join(session_plots_dir, 'L1_CCGP_W_canvas_cue_trainedOnR1Stop.png')
    build_canvas(sequence_saved_paths, canvas_path)
    print(f"Saved combined session-sequence canvas of {len(sequence_saved_paths)} plots to {canvas_path}")

    # canvas_path_svg = os.path.join(session_plots_dir, 'cue_L1_upto7_W_canvas.svg')
    # build_canvas_svg(sequence_saved_svg_paths, canvas_path_svg)
    # print(f"Saved combined SVG session-sequence canvas of {len(sequence_saved_svg_paths)} plots to {canvas_path_svg}")

# %%














































"""W plotting utility, also has the region-gradient colormap used in plot_W_list's extra column to mark which"""






def _desaturate(color, factor):
    """`color` with its HSV saturation scaled by `factor` (0 = gray, 1 = unchanged) -
    same hue/value, just softer. Used to tone down the region-gradient's raw
    red/green/yellow, which read as too garish/saturated at full strength."""
    h, s, v = rgb_to_hsv(to_rgb(color))
    return tuple(hsv_to_rgb((h, s * factor, v)))


REGION_LABELS = ('Anterior Cingulate', 'Prelimbic', 'Infralimbic')
REGION_COLORS = tuple(_desaturate(c, 0.55) for c in ('red', 'green', 'yellow'))
# Relative sizes, same order as REGION_LABELS/REGION_COLORS - Prelimbic a bit bigger than
# Infralimbic (Anterior Cingulate left at the baseline weight). Sizes need not be equal;
# _region_gradient_cmap and _plot_region_gradient both key off these same fractions, so
# the color bands and the labels/boundaries always agree with each other.
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


def plot_W_list(highlighted_rows, plot_type='stem', min_abs_w=0.4, show_region_gradient=True,
                 region_labels=REGION_LABELS, title=None, save_path=None, save_path_svg=None):
    """Plot each row's `w` vector as its own thin panel, side by side, features listed
    top-to-bottom (feature 0 at top, y-axis shared across panels so they line up
    feature-for-feature) - meant for a small hand-picked set of W's (see
    get_session_sequence_Ws/plot_session_analysis's highlight_selections and the
    highlighted_rows/highlighted_Ws DataFrame they return), not a whole session's worth
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
    nothing else between them worth the gap. Each panel's title is built from whichever
    of name/session_id/predict_y_name/conditioner/train_stratum/interval_name/timebin
    columns are present on that row (see _highlighted_rows for where those come from) -
    `name` (the selection's own short human label, if it set one) leads, wrapped at 14
    chars (textwrap) since it's free-form text and can run longer than the other,
    fixed-shape fields. With show_region_gradient (default True), one extra narrow column
    is added after the W panels - see _plot_region_gradient - marking region_labels'
    evenly-sized zones along the same (shared) Neuron ID axis."""
    if highlighted_rows.empty:
        print("plot_W_list: nothing to plot (highlighted_rows is empty)")
        return

    n = len(highlighted_rows)
    n_features = len(highlighted_rows['w'].iloc[0])
    # narrow per panel (nothing to show horizontally besides the one profile) and no
    # taller than needed for n_features rows, regardless of how many panels there are;
    # the region-gradient column (if any) is narrower still - just wide enough for its
    # rotated label text, not another full data panel.
    ncols = n + 1 if show_region_gradient else n
    width_ratios = [1] * n + ([0.4] if show_region_gradient else [])
    fig_width = max(2, 0.8 * n) + (0.6 if show_region_gradient else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, 0.07 * n_features + 0.8),
                              sharey=True, dpi=150, gridspec_kw={'width_ratios': width_ratios})
    axes = np.atleast_1d(axes)
    feature_idx = np.arange(n_features)

    label_cols = [('predict_y_name', '{}'), ('conditioner', 'cond={}'), ('train_stratum', 'ts={}'),
                  ('interval_name', '{}'), ('timebin', 't={}'), ('session_id', '{}')]
    for ax, (_, row) in zip(axes[:n], highlighted_rows.iterrows()):
        w = np.asarray(row['w'])
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
        title_lines = []
        if 'name' in row and pd.notna(row['name']):
            title_lines.append(textwrap.fill(str(row['name']), width=14))
        title_lines += [fmt.format(row[col]) for col, fmt in label_cols if col in row and pd.notna(row[col])]
        ax.set_title('\n'.join(title_lines), fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_xlim(-5, 5)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        if ax is not axes[0]:
            ax.tick_params(left=False, labelleft=False)

    if show_region_gradient:
        _plot_region_gradient(axes[-1], n_features, region_labels=region_labels)

    axes[0].invert_yaxis()  # feature 0 at top
    axes[0].set_ylabel('Neuron ID')
    if title:
        fig.suptitle(title, y=1.02)
    plt.subplots_adjust(wspace=0.15)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if save_path_svg is not None:
        fig.savefig(save_path_svg)
    plt.show()

# %%
# Example: plot the W's pulled out via highlight_selections above - either the
# per-session (all_highlighted_rows_df) or cross-session (sequence_highlighted_rows_df)
# set, or both concatenated together.
plot_W_list(sequence_highlighted_rows_df, plot_type='stem', save_path_svg=os.path.join(nas_dir, 'Simon', 'lm4', 'highlighted_Ws_stem.svg'))
# plot_W_list(all_highlighted_rows_df, plot_type='line')













































# %%
def _x_modality_fr(fr_z_raw, x_modality):
    """Session-scoped firing-rate frame for one X_modality, built to match
    get_SVMCueOutcomeChoicePred's X_full construction exactly (postproc_mea1k_ephys.py,
    same column slice/order - fr_z.iloc[:, 20:] for 'mPFC') - needed so a dot product
    with a fitted `w` lines up feature-for-feature with what it was actually trained on.
    `fr_z_raw`: one session's raw get_analytics('FiringRate40msZ', ...) rows (still has
    from_ephys_timestamp/to_ephys_timestamp columns, no IntervalIndex yet)."""
    fr = fr_z_raw.copy()
    fr.index = pd.IntervalIndex.from_arrays(fr.pop('from_ephys_timestamp'), fr.pop('to_ephys_timestamp'),
                                             closed='right')
    if x_modality == 'mPFC':
        return fr.iloc[:, 20:].fillna(0)
    elif x_modality == 'HP':
        return fr.iloc[:, :20].fillna(0)
    elif x_modality == 'HP+mPFC':
        return fr.fillna(0)
    raise ValueError(f"Unknown x_modality {x_modality!r}")


def get_trial_w_projection(s_id, t0_events, fr_x, which_intervals, w, normalize=False):
    """Per-trial projection of session `s_id`'s activity (already sliced to one
    X_modality - see _x_modality_fr) onto a single fitted decoder weight vector `w`, at
    native 40ms resolution. Same per-trial windowing as get_session_data's `all_pos`
    (t0_events' own full_intrvl per trial, same modal-length trim) - but unlike
    all_pos/create_subplot, this is NOT resampled down to the SVM's own timebin count:
    since the SVM's sliding window advances exactly one native bin per timebin (see
    timeline_decoding in postproc_mea1k_ephys.py), timebin index and native-bin index
    are the same sequence (create_subplot's x_0 == that interval's n_bins_left - see
    get_TrialWiseT0Events40ms), so plotting by raw native-bin index already lines up
    with the existing offset x-axis - one bin more precisely than the resampled
    position trace, in fact, since there's no window-averaging blur at the edges.
    `normalize`: L2-normalize each trial-bin's feature vector before projecting,
    matching the classifier's own Normalizer(norm='l2') preprocessing (see
    get_SVMCueOutcomeChoicePred) - off by default, since the raw dot product is simpler
    to read as "projected activity"; flip on if the raw trace disagrees with the
    classifier's actual behavior. Returns two dicts, both keyed by interval:
    {interval: (n_trials, n_bins) array} and {interval: trial_id array} (same row order
    as the first) - the second is what a caller needs to split by covariate afterward
    (see _trial_covariate_levels), since the modal-length trim below can drop trials."""
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

        if normalize:
            norms = np.linalg.norm(trial_windows, axis=2, keepdims=True)
            trial_windows = trial_windows / np.where(norms == 0, 1, norms)

        projections[which_interval] = trial_windows @ w
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


def _trial_covariate_levels(t0_events, s_id, trial_ids, covariate):
    """0/1 level for `covariate` - one of ALL_Y_PREDICTED_NAMES/'trial_outcome', or a
    derived CORRECTNESS_SPECS name ('correct_R1'/'correct_R2') - for each of
    `trial_ids`, in the same order. t0_events has one row per (trial, t0_event_name),
    but every one of these is constant per trial (see select_t0_events's base_info in
    get_TrialWiseT0Events40ms), so groupby('trial_id').first() collapses that safely.
    Plain covariates: t0_events' own 'cue'/'trial_outcome' are still raw (1/2, and
    0/1/2/3...) rather than the 0/1 encoding _covariate_color/_level_label/
    LEVEL_LABELS expect - that conversion normally happens inside
    get_SVMCueOutcomeChoicePred's _process_y_values, a later stage this feature doesn't
    go through - so it's replicated here. choice_R1/choice_R2 are already native 0/1.
    Derived (correct_R1/correct_R2): computed the same way plot_correctness_f1_single
    does, from this same adjusted cue plus the raw (already native 0/1) choice column
    that spec names."""
    per_trial = t0_events.loc[s_id].groupby('trial_id').first()

    spec = next((s for s in CORRECTNESS_SPECS if s['name'] == covariate), None)
    if spec is not None:
        cue = (per_trial['cue'] - 1).clip(0, 1)
        values = spec['is_correct'](cue, per_trial[spec['choice_name']]).astype(int)
    else:
        values = per_trial[covariate]
        if covariate == 'cue':
            values = (values - 1).clip(0, 1)
        elif covariate == 'trial_outcome':
            values = values.clip(0, 1)

    return values.loc[trial_ids].values


def _interval_x0(which_interval, offset):
    """t=0 native-bin position for one interval, at the current x-axis `offset` - see
    get_TrialWiseT0Events40ms's n_bins_left (3 for cue_entry_interval, 10 for
    R1/R2_entry_interval - the only intervals this notebook actually uses; extend here
    if a differently-aligned interval is ever added). Shared by row 0 (position) and
    row 1 (projection) below so their dashed t0 lines land on the same x position."""
    return (3 if which_interval == 'cue_entry_interval' else 10) + offset


def _shared_ylim(traces_by_interval, pad=0.1):
    """Y-limits from the full data range across every interval/trial (not just the
    mean line, so no variance band ever clips) - used to give row 0, row 1, and every
    covariate panel below it a fixed, comparable scale instead of each Axes
    autoscaling independently (which made panels hard to compare against each other,
    and let a single outlier trial blow out one panel's range). `pad` adds a fractional
    margin on each side."""
    all_vals = np.concatenate([arr.ravel() for arr in traces_by_interval.values() if len(arr)])
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    margin = (hi - lo) * pad if hi > lo else 1.0
    return lo - margin, hi + margin


def plot_trial_velocity(ax, velocity_traces, which_intervals, split=None):
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
        ax.axvline(x=_interval_x0(which_interval, offset), color='k', linestyle='--', alpha=0.4)
        offset += n_bins
        if which_interval != which_intervals[-1]:
            offset += 1  # gap matching the heatmap's inserted NaN separator row
    ax.set_ylabel('forward velocity')
    ax.tick_params(labelbottom=False)
    for sp in ax.spines.values():  # same clean look as the main figure's rows
        sp.set_visible(False)
    return offset


def plot_w_projection(ax, projections, which_intervals, split, show_variance=True, show_xticklabels=True,
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
    shades that sub-window of that one interval with a translucent purple axvspan,
    directly on top of the actual projection trace, so a specific window of interest
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
            ax.axvspan(hl_start + offset, hl_end + offset, color='purple', alpha=0.12, zorder=0)
            if highlight_var_pct is not None:
                # y in axes-fraction space (via get_xaxis_transform) so this stays pinned
                # near the top regardless of the ylim the caller sets afterward.
                ax.text((hl_start + hl_end) / 2 + offset, 0.95, f'W: {highlight_var_pct:.1f}% var',
                        transform=ax.get_xaxis_transform(), color='purple', fontsize=6,
                        ha='center', va='top', zorder=15,
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.75))

        levels = levels_by_interval[which_interval]
        groups = [(trial_projs[levels == lvl], _covariate_color(covariate, lvl), _level_label(covariate, lvl))
                  for lvl in (0, 1) if (levels == lvl).sum() > 0]

        for group_projs, group_color, group_label in groups:
            mean_proj = _smooth(group_projs.mean(axis=0), smooth)
            if show_variance:
                half_std = _smooth(group_projs.std(axis=0), smooth) / 2
                ax.fill_between(x, mean_proj - half_std, mean_proj + half_std, color=group_color, alpha=0.2,
                                 edgecolor=None, zorder=1)
            ax.plot(x, mean_proj, color=group_color, linewidth=1.5, alpha=0.9, zorder=10,
                    label=group_label if group_label not in labels_drawn else None)
            labels_drawn.add(group_label)

        x_0 = _interval_x0(which_interval, offset)
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


# The only (target, partner) pairing Simon wants a nested/conditional split for - not a
# general permutation system over every covariate pair, just this one. See
# _covariate_panel_specs.
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
                                        which_intervals, show_variance=True, filter_by=None, show_xticklabels=False,
                                        smooth=None):
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
                group_mean = _smooth(group_projs.mean(axis=0), smooth)
                lvl_color = _covariate_color(covariate, lvl)
                if show_variance:
                    half_std = _smooth(group_projs.std(axis=0), smooth) / 2
                    ax.fill_between(x, group_mean - half_std, group_mean + half_std, color=lvl_color, alpha=0.15,
                                     edgecolor=None, zorder=1)
                ax.plot(x, group_mean, color=lvl_color, linewidth=1.2, alpha=0.9, zorder=10)

        x_0 = _interval_x0(which_interval, offset)
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
                                  predict_y_name, covariate_names=None, show_variance=True, smooth=None):
    """One small subplot per (covariate, filter_by) pair from _covariate_panel_specs,
    stacked below the main split row - `axes` is a sequence of Axes, one per pair. The
    last Axes gets show_xticklabels=True (see plot_w_projection_covariate_single) -
    it's the bottom-most row of the whole figure whenever this grid is shown. `smooth`:
    see plot_w_projection, passed through to every panel."""
    specs = _covariate_panel_specs(predict_y_name, covariate_names)
    for i, (ax, (covariate, filter_by)) in enumerate(zip(axes, specs)):
        plot_w_projection_covariate_single(ax, projections, trial_ids_by_interval, t0_events, s_id, covariate,
                                            which_intervals, show_variance=show_variance, filter_by=filter_by,
                                            show_xticklabels=(i == len(specs) - 1), smooth=smooth)


def plot_w_projection_session(s_id, w_row, t0_events, behavior, session_names, which_intervals,
                               covariate_names=None, show_covariate_plot=True, normalize=True,
                               smooth=None, highlight_range=None, save_dir=None):
    """Full per-session W-projection figure projecting one fixed, hand-picked decoder
    (`w_row` - a single row from all_highlighted_rows_df/sequence_highlighted_rows_df,
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
    show_covariate_plot, not just as one of the "other" covariates below. No plain
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
    same annotation the main figure's row-2 heatmap carries. `normalize` (default True
    here, unlike get_trial_w_projection's own bare default) L2-normalizes each
    trial-bin's feature vector before projecting, matching the classifier's own
    preprocessing - without it, the projection's magnitude scales with each session's
    raw feature-vector norm (neuron count/yield, overall activity level), which is
    session-dependent and unrelated to decoding signal strength - this is why one
    session can look much larger than the others for no interpretable reason.
    normalize=True fixes that at the source, so row 1 and every covariate panel can
    just use _shared_ylim's empirical data range; an exact +/-||w|| bound is also
    available via Cauchy-Schwarz once vectors are unit-norm, but deliberately not used
    for the axis limits - in this many dimensions a near-random unit vector's dot
    product with w concentrates far below ||w||, so that bound is essentially never
    approached and makes the actual line look flat on a mostly-empty axis. `smooth`, if
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
    Saves to save_dir/session_{s_id}_S{s_num}_wproj.png if save_dir is given. Returns
    that path (or None)."""
    x_modality, predict_y_name, w = w_row['X_modality'], w_row['predict_y_name'], w_row['w']
    session_name = next(sn for sn in session_names if sn.startswith(s_id))

    fr_z_raw = analytics.get_analytics('FiringRate40msZ', session_names=[session_name]).xs(s_id, level='session_id')
    fr_x = _x_modality_fr(fr_z_raw, x_modality)

    highlight_var_pct = None
    if highlight_range is not None:
        hl_interval, hl_start, hl_end = highlight_range
        pooled = get_interval_window_activity(s_id, t0_events, fr_x, hl_interval, (hl_start, hl_end))
        _, _, hl_total_var = _pca_eigen(pooled)
        hl_centered = pooled - pooled.mean(axis=0, keepdims=True)
        hl_w_unit = np.asarray(w) / np.linalg.norm(w)
        highlight_var_pct = np.var(hl_centered @ hl_w_unit, ddof=1) / hl_total_var * 100

    projections, proj_trial_ids = get_trial_w_projection(s_id, t0_events, fr_x, which_intervals, w,
                                                           normalize=normalize)
    velocity_traces, vel_trial_ids = get_trial_behavior_trace(s_id, t0_events, behavior, which_intervals,
                                                               'frame_raw_500msMedian')

    target_split = (predict_y_name, {iv: _trial_covariate_levels(t0_events, s_id, ids, predict_y_name)
                                      for iv, ids in proj_trial_ids.items()})
    velocity_split = ('cue', {iv: _trial_covariate_levels(t0_events, s_id, ids, 'cue')
                               for iv, ids in vel_trial_ids.items()})

    covariate_specs = _covariate_panel_specs(predict_y_name, covariate_names)
    n_covariates = len(covariate_specs) if show_covariate_plot else 0

    fig, axes = plt.subplots(2 + n_covariates, 1, figsize=(max(4, 1.4 * len(which_intervals) + 1),
                                                            3.6 + 0.8 * n_covariates),
                              height_ratios=[1, 1.3] + [0.8] * n_covariates, sharex=True, dpi=150)
    plot_trial_velocity(axes[0], velocity_traces, which_intervals, split=velocity_split)
    axes[0].set_ylim(_shared_ylim(velocity_traces))
    # origin_label names which session this W was actually fit on (w_row['session_id']),
    # distinct from s_id/s_label below (the session whose activity it's being projected
    # onto here) - the two now routinely differ, so the corner text has to carry both.
    origin_num = ith_to_dt_session.get(w_row['session_id'], "Unknown")
    origin_label = f'S{origin_num:02d}' if isinstance(origin_num, int) else f'S{origin_num}'
    # w_row['name']: the highlight_selection's own optional short human label (see
    # _highlighted_rows) - wrapped since, unlike the other corner-text fields, it's
    # free-form text that can run longer than fits on one line in the corner.
    w_name = w_row.get('name')
    name_line = f"{textwrap.fill(str(w_name), width=24)}\n" if isinstance(w_name, str) and w_name else ''
    plot_w_projection(axes[1], projections, which_intervals, split=target_split,
                       show_xticklabels=(n_covariates == 0),
                       corner_text=f"{name_line}{predict_y_name} W @ {w_row['interval_name']} t={w_row['timebin']} "
                                   f"(from {origin_label})", smooth=smooth,
                       highlight_range=highlight_range, highlight_var_pct=highlight_var_pct)
    if show_covariate_plot:
        plot_w_projection_covariates(axes[2:], projections, proj_trial_ids, t0_events, s_id, which_intervals,
                                      predict_y_name, covariate_names=covariate_names, smooth=smooth)
    # normalize=True already fixes the actual per-session scale problem (see docstring),
    # so the empirical range is consistent enough across sessions - just use it
    # directly, rather than the much looser (and session-independent, so not actually
    # helping comparability) +/-||w|| Cauchy-Schwarz bound.
    # projection_ylim = _shared_ylim(projections)
    projection_ylim = (-.3,.3)
    for ax in axes[1:]:
        ax.set_ylim(projection_ylim)

    # same S<n> numbering as the main figure's title (plot_session_analysis) - see
    # ith_to_dt_session near the top of this notebook.
    s_num = ith_to_dt_session.get(s_id, "Unknown")
    s_label = f'S{s_num:02d}' if isinstance(s_num, int) else f'S{s_num}'
    title = f'{s_label}: activity projected onto {origin_label}\'s W ({x_modality})'
    if isinstance(w_name, str) and w_name:
        title = f"'{w_name}'\n{title}"
    fig.suptitle(title)
    plt.tight_layout()

    save_path = None
    if save_dir is not None:
        save_path = os.path.join(save_dir, f'session_{s_id}_S{s_num}_wproj.svg')
        fig.savefig(save_path, dpi=150)
    plt.show()
    return save_path


# %%
def _pca_eigen(data):
    """Eigen-decomposition of `data`'s (n_samples, n_features) covariance matrix. `data`
    is centered internally (per-feature mean subtracted) before computing the
    covariance - necessary even for already globally z-scored FiringRate40msZ data,
    since a small window's/session's own local mean generally isn't exactly zero.
    Negative eigenvalues from floating-point noise (a valid covariance matrix is PSD, so
    these should only ever be ~0) are clipped to 0 before normalizing. Returns
    (eigenvalues_pct, eigenvectors, total_var): eigenvalues_pct is % of total variance
    (sum of eigenvalues == trace(cov)), sorted descending; eigenvectors is
    (n_features, n_components), columns in the same descending order (orthonormal,
    straight from eigh) - feeds plot_session_pca_similarity's canonical-angle subspace
    comparison, which needs the actual PC directions, not just their variances;
    total_var is kept for callers that need the same denominator for a non-PCA
    direction's own variance (see plot_w_variance_explained)."""
    centered = data - data.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]  # descending, columns stay matched to eigvals
    eigvals = np.clip(eigvals, 0, None)
    total_var = eigvals.sum()
    return eigvals / total_var * 100, eigvecs, total_var


def plot_pca_eigenvalues(ax, eigenvalues_pct, w_var_pct=None, w_label=None, max_pcs=30, title=None):
    """Horizontal bar plot of PCA eigenvalues (already a % of total variance - see
    _pca_eigen), ordered by magnitude with PC1 at the top. `max_pcs`, if given
    (default 30), only shows the top N components - a session/interval can have dozens
    of neurons, and past the first few tens the tail is uninformative and just wastes
    vertical space. `w_var_pct`, if given, draws a vertical dashed purple line at that
    x-value across every bar - the % of variance a specific W direction explains in this
    same data (see plot_w_variance_explained; unrelated to _pca_eigen, which this value
    isn't derived from) - for a direct visual comparison against
    the ranked PC spectrum, in particular PC1 (the max direction): if the line reaches
    close to/past PC1's bar, that decoder direction captures about as much variance as
    the single largest source of variability in the data; if it's far short of even the
    smaller PCs shown, it's a minor contributor. `w_label` (e.g. a highlight_selection's
    own `name`) annotates that line; defaults to plain 'W'."""
    if max_pcs is not None:
        eigenvalues_pct = eigenvalues_pct[:max_pcs]
    pc_idx = np.arange(len(eigenvalues_pct))
    ax.barh(pc_idx, eigenvalues_pct, color='#555555', height=0.8, zorder=5)
    ax.invert_yaxis()  # PC1 at top
    ax.set_yticks(pc_idx)
    ax.set_yticklabels([f'PC{i + 1}' for i in pc_idx], fontsize=6)
    ax.set_xlabel('% of total variance', fontsize=8)
    ax.tick_params(labelsize=7)
    if w_var_pct is not None:
        ax.axvline(w_var_pct, color='purple', linestyle='--', linewidth=1.5, zorder=10)
        # Anchored in axes-fraction space (top-right corner), not at the line's own data
        # x-position - keeps the label off the title above regardless of where the line
        # falls (near 0% or near 100%) and regardless of how long w_label is (wrapped +
        # given a translucent background so it stays legible over a long PC1 bar).
        label_text = textwrap.fill(f'{w_label or "W"}: {w_var_pct:.1f}%', width=20)
        ax.text(0.98, 0.98, label_text, transform=ax.transAxes, color='purple', fontsize=7,
                ha='right', va='top', zorder=15,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.75))
    if title:
        ax.set_title(title, fontsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)


def get_interval_window_activity(s_id, t0_events, fr_x, interval_name, timebin_range):
    """Pooled (n_trials * n_bins_in_range, n_features) activity matrix for one
    session/interval/timebin_range - same per-trial windowing as get_trial_w_projection
    (t0_events' own full_intrvl per trial, same modal-length trim), but instead of
    projecting onto a W, keeps the full feature vectors and only slices down to
    `timebin_range` (a (start, end) pair, both inclusive - the same native-bin/timebin
    index get_trial_w_projection relies on being identical, see its own docstring)
    before flattening trials and timebins together into one sample axis. Feeds
    plot_w_variance_explained's pooled PCA/variance-explained-by-W computation, which
    (per how this is called) treats trial-to-trial and within-window temporal
    variability as one combined source of variance rather than analyzing them
    separately per timebin."""
    trial_windows = []
    for _, full_intrvl in t0_events.loc[s_id, interval_name].dropna().items():
        trial_windows.append(fr_x[fr_x.index.overlaps(full_intrvl)].values)
    lengths = pd.Series([len(x) for x in trial_windows])
    correct_length = lengths.value_counts().index[0]
    trial_windows = np.stack([x for x in trial_windows if len(x) == correct_length])

    start, end = timebin_range
    trial_windows = trial_windows[:, start:end + 1, :]
    n_trials, n_bins, n_features = trial_windows.shape
    return trial_windows.reshape(n_trials * n_bins, n_features)


def plot_w_variance_explained(target_spec, w_row, t0_events, session_names, max_pcs=50, save_dir=None):
    """Parts 2+3 of the W-variance-explained analysis (see plot_session_pca for part 1,
    the standalone whole-session baseline this is deliberately kept separate from).
    `target_spec`: a dict shaped like a highlight_selections entry, but naming an
    activity WINDOW to evaluate rather than a fitted decoder row - keys `session_id`,
    `interval`, `timebin_range` (a (start, end) pair, both inclusive - see
    get_interval_window_activity), and an optional `name`. `w_row`: a single row from
    all_highlighted_rows_df/sequence_highlighted_rows_df (as in plot_w_projection_session)
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
    judged against the ranked PC spectrum, in particular PC1 (the max direction). Saves
    to save_dir/wvar_{name or session_interval_range}.png if save_dir is given. Returns
    (save_path, w_var_pct, eigenvalues_pct)."""
    x_modality, w = w_row['X_modality'], np.asarray(w_row['w'])
    t_s_id = target_spec['session_id']
    interval_name, timebin_range = target_spec['interval'], target_spec['timebin_range']

    session_name = next(sn for sn in session_names if sn.startswith(t_s_id))
    fr_z_raw = analytics.get_analytics('FiringRate40msZ', session_names=[session_name]).xs(t_s_id, level='session_id')
    fr_x = _x_modality_fr(fr_z_raw, x_modality)

    pooled = get_interval_window_activity(t_s_id, t0_events, fr_x, interval_name, timebin_range)
    eigenvalues_pct, _, total_var = _pca_eigen(pooled)

    centered = pooled - pooled.mean(axis=0, keepdims=True)
    w_unit = w / np.linalg.norm(w)
    w_var_pct = np.var(centered @ w_unit, ddof=1) / total_var * 100

    origin_num = ith_to_dt_session.get(w_row['session_id'], "Unknown")
    origin_label = f'S{origin_num:02d}' if isinstance(origin_num, int) else f'S{origin_num}'
    t_num = ith_to_dt_session.get(t_s_id, "Unknown")
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


# %%
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


# Kinematic Behavior40msAligned columns worth checking against subspace activation (see
# that analytic's column list in analytics_processing/analytics.py) - forward velocity/
# acceleration, head angle/angular velocity, overall movement energy, rotational
# velocity/acceleration. Not exhaustive - any other Behavior40msAligned column works
# too via plot_w_projection_kinematics' own `kinematic_columns` argument, this is just a
# sensible default set.
KINEMATIC_COLUMNS = ['frame_raw_500msMedian', 'frame_raw_abs_acc_500msMedian', 'head_angle', 'head_angle_vel',
                      'movement_energy_smooth5', 'frame_YawPitch_abs_vel_sum_500msMedian',
                      'frame_YawPitch_abs_acc_sum_500msMedian']


def plot_w_projection_kinematics(s_id, w_row, t0_events, behavior, session_names, interval_name, timebin_range,
                                  kinematic_columns=KINEMATIC_COLUMNS, kinematic_offset=0, color_by=None,
                                  normalize=True, save_dir=None):
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
    `normalize`: see get_trial_w_projection - True by default here, matching
    plot_w_projection_session's own opinionated default (the raw dot product otherwise
    scales with each session's own feature-vector norm, unrelated to signal).
    Samples with a NaN in either variable (e.g. head_angle can have missing camera-pose
    frames) are dropped before computing each panel's Pearson r (scipy.stats.pearsonr),
    annotated in that panel's corner. Saves to save_dir/session_{s_id}_S{s_num}_
    {interval_name}_t{start}-{end}_kinematics.png if save_dir is given. Returns a
    {column: (r, p)} dict."""
    x_modality, w = w_row['X_modality'], w_row['w']
    session_name = next(sn for sn in session_names if sn.startswith(s_id))
    fr_z_raw = analytics.get_analytics('FiringRate40msZ', session_names=[session_name]).xs(s_id, level='session_id')
    fr_x = _x_modality_fr(fr_z_raw, x_modality)

    projections, proj_trial_ids = get_trial_w_projection(s_id, t0_events, fr_x, [interval_name], w,
                                                           normalize=normalize)
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

    s_num = ith_to_dt_session.get(s_id, "Unknown")
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
kinematics_w_row = sequence_highlighted_rows_df.iloc[0]
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
_ = plot_w_variance_explained(kinematics_target_spec, kinematics_w_row, t0_events, session_names)
_ = plot_w_projection_session(
    kinematics_w_row['session_id'], kinematics_w_row, t0_events, behavior, session_names, [kinematics_interval],
    show_covariate_plot=False, smooth=2,
    highlight_range=(kinematics_interval, *kinematics_timebin_range))

kinematics_correlations = plot_w_projection_kinematics(
    kinematics_w_row['session_id'], kinematics_w_row, t0_events, behavior, session_names,
    kinematics_interval, kinematics_timebin_range, kinematic_offset=kinematics_offset,
    color_by=kinematics_w_row['predict_y_name'])
kinematics_correlations


# %%
def plot_session_projection_kinematics(s_id, w_row, behavior, session_names, kinematic_columns=KINEMATIC_COLUMNS,
                                        color_by=None, normalize=True, max_points=None, save_dir=None):
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
    fr_x = _x_modality_fr(fr_z_raw, x_modality)

    values = fr_x.values
    if normalize:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values = values / np.where(norms == 0, 1, norms)
    proj = pd.Series(values @ w, index=fr_x.index.right)

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

    s_num = ith_to_dt_session.get(s_id, "Unknown")
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


# %%
# Demo: same question as the trial-windowed demo above, but over the whole session's
# continuous activity - is a kinematic relationship specific to the task-relevant
# window(s), or does it hold throughout the whole session (including inter-trial time)?
# Same kinematics_w_row as above; max_points keeps the scatter responsive given tens of
# thousands of bins per session (set to None to plot every bin).
session_kinematics_correlations = plot_session_projection_kinematics(
    kinematics_w_row['session_id'], kinematics_w_row, behavior, session_names,
    color_by=kinematics_w_row['predict_y_name'], max_points=20000)
session_kinematics_correlations

# %%

# %%
plot_session_ids
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
# w_row = sequence_highlighted_rows_df[sequence_highlighted_rows_df['name'] == 'shared_choice_13'].iloc[0]  # the choice axis that is quite preserved
    
# for the conserverd first square activations    
# w_row = sequence_highlighted_rows_df[sequence_highlighted_rows_df['name'] == 'shared_cue_14_firstsquare'].iloc[0]  # the choice axis that is quite preserved

# the bigger square coming after, reactivating on error in Ssesion13
# w_row = sequence_highlighted_rows_df[sequence_highlighted_rows_df['name'] == 'shared_cue_14_secondquare'].iloc[0]  # the choice axis that is quite preserved

# the bigger square coming after, reactivating on error in Ssesion13
# w_row = sequence_highlighted_rows_df[sequence_highlighted_rows_df['name'] == 'S13_Cue_cue_choice_overlap_2ndsquare'].iloc[0]  # the choice axis that is quite preserved


# the only real alignment in session 13, from cue side
# w_row = sequence_highlighted_rows_df[sequence_highlighted_rows_df['name'] == 'S13_Cue_cue_choice_overlap'].iloc[0]  # the choice axis that is quite preserved
# # the only real alignment in session 13, from choice side
# w_row = sequence_highlighted_rows_df[sequence_highlighted_rows_df['name'] == 'S13_Choice_cue_choice_overlap'].iloc[0]  # the choice axis that is quite preserved


w_row = sequence_highlighted_rows_df.iloc[0]  # the first choice  squarey in S7

session_plots_dir = os.path.join(nas_dir, 'Simon', 'lm4', 'choice7')
os.makedirs(session_plots_dir, exist_ok=True)

wproj_saved_paths = []
for s_id in plot_session_ids:
# for s_id in ['2024-12-04_18-06']:
    saved_path = plot_w_projection_session(s_id, w_row, t0_events, behavior, session_names, w_which_intervals,
                                            save_dir=session_plots_dir, smooth=2)
    wproj_saved_paths.append(saved_path)

# if wproj_saved_paths:
#     # one image per session (not a stack), so wrap each path in its own single-element
#     # list - build_canvas expects one list of paths *per session* to stack vertically.
#     canvas_path = os.path.join(session_plots_dir, 'S13_error_replay.svg')
#     build_canvas([[p] for p in wproj_saved_paths], canvas_path)
#     print(f"Saved combined canvas of {len(wproj_saved_paths)} sessions to {canvas_path}")

# %%
# Same fixed w_row as above, but restricted to just the sessions that have a matched W
# (see highlight_selections/sequence_highlighted_rows_df above) - one figure per session,
# same hand-picked target/direction each time, the same way the main per-session loop
# iterates plot_session_ids - so this W's own projection drift/consistency across
# sessions shows up the same way plot_session_sequence_w_comparison already shows it for
# the W's themselves. Uses its own w_which_intervals, not the shared `which_intervals`
# global other cells rely on. show_covariate_plot=True (default) adds one small panel per
# covariate below the aggregate row - set False for just the velocity+aggregate rows.
w_which_intervals = ['cue_entry_interval', 'R1_entry_interval']  # pick from ALL_INTERVALS

wproj_saved_paths = []
for i,s_id in enumerate(sequence_highlighted_rows_df['session_id']):
    w_row = sequence_highlighted_rows_df.iloc[i]  # pick the W that was actually fit on this session, will go from S07-S15
    saved_path = plot_w_projection_session(s_id, w_row, t0_events, behavior, session_names, w_which_intervals,
                                            save_dir=session_plots_dir, smooth=2)
    wproj_saved_paths.append(saved_path)

if wproj_saved_paths:
    canvas_path = os.path.join(session_plots_dir, 'wproj_canvas_matched_sessions.png')
    build_canvas([[p] for p in wproj_saved_paths], canvas_path)
    print(f"Saved combined canvas of {len(wproj_saved_paths)} sessions to {canvas_path}")


# %%
def plot_session_pca(s_id, session_names, x_modality, max_pcs=50, save_dir=None):
    """Standalone baseline for the W-variance-explained analysis (part 1 - see
    plot_w_variance_explained for part 2/3, which compares a specific decoder direction
    against this same kind of spectrum but restricted to one interval/timebin-range and
    pooled across trials): PCA across session `s_id`'s *entire* continuous recording -
    every native 40ms bin of FiringRate40msZ, not just trial windows - for one
    X_modality, not tied to any specific decoder. Plots the eigenvalue spectrum (%
    of total variance - see _pca_eigen/plot_pca_eigenvalues) as a horizontal
    bar chart ordered by magnitude. Saves to save_dir/session_{s_id}_S{s_num}_pca.png if
    save_dir is given. Returns (save_path, eigenvalues_pct, eigenvectors) - eigenvectors
    (see _pca_eigen) feeds plot_session_pca_similarity's cross-session canonical-angle
    comparison, called once per session the same way this function is."""
    session_name = next(sn for sn in session_names if sn.startswith(s_id))
    fr_z_raw = analytics.get_analytics('FiringRate40msZ', session_names=[session_name]).xs(s_id, level='session_id')
    fr_x = _x_modality_fr(fr_z_raw, x_modality)
    eigenvalues_pct, eigenvectors, _ = _pca_eigen(fr_x.values)

    s_num = ith_to_dt_session.get(s_id, "Unknown")
    s_label = f'S{s_num:02d}' if isinstance(s_num, int) else f'S{s_num}'
    n_shown = min(max_pcs or len(eigenvalues_pct), len(eigenvalues_pct))
    fig, ax = plt.subplots(figsize=(4, max(3, 0.18 * n_shown)), dpi=150)
    plot_pca_eigenvalues(ax, eigenvalues_pct, max_pcs=max_pcs,
                          title=f'{s_label}: whole-session PCA ({x_modality})')
    plt.tight_layout()

    save_path = None
    if save_dir is not None:
        save_path = os.path.join(save_dir, f'session_{s_id}_S{s_num}_pca.png')
        fig.savefig(save_path, dpi=150)
    plt.show()
    return save_path, eigenvalues_pct, eigenvectors


def _canonical_angles_deg(subspace_a, subspace_b):
    """Canonical (principal) angles, in degrees, between the subspaces spanned by
    `subspace_a`/`subspace_b`'s own columns - each an orthonormal basis, e.g. a
    contiguous slice of _pca_eigen's eigenvectors, which are orthonormal by
    construction. Same method already used elsewhere in this codebase for comparing PCA
    subspaces across sessions (get_SessionPCs40msCAs in
    ephys_preprocessing/postproc_mea1k_ephys.py; plot_EvolvingPCSubspace.render_plot in
    dashsrc/plot_components/plots/), reused here rather than reinvented: SVD of the
    cross-subspace Gram matrix gives singular values that are exactly the cosines of the
    principal angles (Bjorck-Golub). Returns one angle per min(k_a, k_b) - 0 degrees
    means that pair of directions is identical, 90 means orthogonal."""
    M = subspace_a.T @ subspace_b
    singular_values = np.linalg.svd(M, compute_uv=False)
    return np.rad2deg(np.arccos(np.clip(singular_values, -1, 1)))


def plot_session_pca_similarity(session_eigvecs, session_eigvals_pct, session_labels, var_thresh=0.8,
                                 agg='mean', vmax=90, save_path=None):
    """Cross-session similarity of the whole-session PCA subspaces from plot_session_pca
    (looped over sessions), via canonical/principal angles - see _canonical_angles_deg
    for the method, already established elsewhere in this codebase
    (get_SessionPCs40msCAs/plot_EvolvingPCSubspace) for exactly this kind of
    cross-session PCA-subspace comparison, ported here as a static matplotlib heatmap
    for this notebook's own whole-session PCA. For each session, only the top-k
    eigenvectors needed to reach `var_thresh` cumulative variance (default 0.8,
    matching that same established convention) are used as its subspace - a
    low-variance session contributes fewer dimensions than a high-variance one, same as
    get_SessionPCs40msCAs (guarded to at least 1 component, in case a single PC alone
    already exceeds var_thresh). Each pair of sessions gets min(k_i, k_j) canonical
    angles; `agg` ('mean' by default, or 'median') collapses that set to one scalar per
    pair for the heatmap - same aggregation choice plot_EvolvingPCSubspace's
    render_plot offers. Diagonal entries are always 0 (a subspace compared to itself).
    Lower = more similar (aligned) subspaces. Saves to save_path if given. Returns the
    (n_sessions, n_sessions) similarity matrix."""
    n = len(session_labels)
    agg_fn = np.mean if agg == 'mean' else np.median
    thresh_pct = var_thresh * 100
    n_components = [max(1, int((eigvals_pct.cumsum() < thresh_pct).sum())) for eigvals_pct in session_eigvals_pct]

    sim = np.zeros((n, n))
    for i in range(n):
        subspace_i = session_eigvecs[i][:, :n_components[i]]
        for j in range(n):
            subspace_j = session_eigvecs[j][:, :n_components[j]]
            sim[i, j] = agg_fn(_canonical_angles_deg(subspace_i, subspace_j))

    fig, ax = plt.subplots(figsize=(max(4, 0.45 * n), max(3.5, 0.45 * n)), dpi=150)
    im = ax.imshow(sim, cmap='viridis', vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(session_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(session_labels, fontsize=8)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f'{agg.capitalize()} canonical angle (degrees)', fontsize=8)
    ax.set_title(f'Whole-session PCA subspace similarity\n(top-k @ {int(var_thresh * 100)}% variance per session)',
                 fontsize=9)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()
    return sim


def plot_pca_reference_drift(ref_s_id, session_ids, session_names, x_modality, var_thresh=0.8, save_path=None):
    """More intuitive alternative to plot_session_pca_similarity's canonical-angle
    comparison (canonical angles in degrees aren't a very directly readable quantity) -
    fixes one reference session's whole-session PCA subspace (its top-k eigenvectors, k
    chosen the same var_thresh-cumulative-variance way as plot_session_pca_similarity)
    and asks one directly interpretable question for every session in `session_ids`
    (pass these in chronological order, typically every session up to and including
    ref_s_id - e.g. plot_session_ids[:-1] with ref_s_id = plot_session_ids[-1], the
    "does the most recent session's PCA basis still explain earlier sessions'
    variance" framing): "if I keep using ref_s_id's own principal directions, how much
    of *this* session's own variance do they still capture?" - i.e. project each
    session's own (centered) activity onto the reference's fixed top-k subspace and
    divide by that session's own total variance (trace of its own covariance) - no
    per-session PCA needed at all beyond the reference's own, unlike the canonical-angle
    comparison, which needs every session's own eigenvectors too. Plotted as one bar per
    session (in the order given), with a dashed reference line at var_thresh*100% - the
    value the reference session would trivially hit against itself (k was chosen to
    reach exactly that threshold there) - so any bar noticeably below the line directly
    shows how much that session's activity has drifted out of the reference's subspace,
    in the same %-of-variance units used throughout this section. Saves to save_path if
    given. Returns (labels, pct_explained)."""
    ref_session_name = next(sn for sn in session_names if sn.startswith(ref_s_id))
    ref_fr_z_raw = analytics.get_analytics('FiringRate40msZ',
                                            session_names=[ref_session_name]).xs(ref_s_id, level='session_id')
    ref_fr_x = _x_modality_fr(ref_fr_z_raw, x_modality)
    ref_eigvals_pct, ref_eigvecs, _ = _pca_eigen(ref_fr_x.values)
    k = max(1, int((ref_eigvals_pct.cumsum() < var_thresh * 100).sum()))
    ref_subspace = ref_eigvecs[:, :k]

    labels, pct_explained = [], []
    for s_id in session_ids:
        session_name = next(sn for sn in session_names if sn.startswith(s_id))
        fr_z_raw = analytics.get_analytics('FiringRate40msZ', session_names=[session_name]).xs(s_id, level='session_id')
        data = _x_modality_fr(fr_z_raw, x_modality).values
        centered = data - data.mean(axis=0, keepdims=True)
        total_var = np.var(centered, axis=0, ddof=1).sum()  # trace(cov) - basis-invariant, no eigendecomposition needed
        projected_var = np.sum((centered @ ref_subspace) ** 2) / (len(centered) - 1)  # trace(P^T cov P), P orthonormal
        pct_explained.append(projected_var / total_var * 100)

        s_num = ith_to_dt_session.get(s_id, "Unknown")
        labels.append(f'S{s_num:02d}' if isinstance(s_num, int) else f'S{s_num}')

    ref_num = ith_to_dt_session.get(ref_s_id, "Unknown")
    ref_label = f'S{ref_num:02d}' if isinstance(ref_num, int) else f'S{ref_num}'

    fig, ax = plt.subplots(figsize=(max(4, 0.6 * len(session_ids)), 3.5), dpi=150)
    ax.bar(np.arange(len(session_ids)), pct_explained, color='#555555')
    ax.axhline(var_thresh * 100, color='purple', linestyle='--', linewidth=1.2,
               label=f"{ref_label}'s own {int(var_thresh * 100)}% (k={k} PCs)")
    ax.set_xticks(np.arange(len(session_ids)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("% of session's own variance\nexplained by reference PCs", fontsize=8)
    ax.set_title(f"Variance explained by {ref_label}'s PCA subspace, across sessions ({x_modality})", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    for sp in ax.spines.values():
        sp.set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.show()
    return labels, pct_explained


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
                                                                  save_dir=session_plots_dir)
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
    plot_session_ids[0], plot_session_ids[1:], session_names, pca_x_modality, save_path=pca_drift_path)

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
# sequence_highlighted_rows_df/all_highlighted_rows_df); w_variance_selections is a list
# shaped like highlight_selections, but naming activity windows (timebin_range, not a
# single fitted timebin) rather than fitted decoder rows to highlight.

# w_row = sequence_highlighted_rows_df[sequence_highlighted_rows_df['name'] == 'S13_Choice_cue_choice_overlap'].iloc[0]
# w_variance_selections = [
#     {'session_id': '2024-12-04_18-06', 'interval': 'R1_entry_interval', 'timebin_range': (20, 30),
#      'name': 'S13_cue_10-20'},
# ]

w_row = sequence_highlighted_rows_df[sequence_highlighted_rows_df['name'] == 'S13_Cue_cue_choice_overlap_2ndsquare'].iloc[0]
w_variance_selections = [
    {'session_id': '2024-12-04_18-06', 'interval': 'cue_entry_interval', 'timebin_range': (20, 30), 'name': 'S14_cue_0-20'},
]

wvar_saved_paths = []
for target_spec in w_variance_selections:
    saved_path, w_var_pct, _ = plot_w_variance_explained(target_spec, w_row, t0_events, session_names,
                                                          save_dir=session_plots_dir)
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
    save_dir=session_plots_dir, smooth=2)

























































# %%
plot_session_ids = sorted(
    (s_id for s_id in svm_results.index.unique('session_id')
    #  if pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') < to_s_id_07),
    #  if pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') <= to_s_id_15),
    ),
    # bewteen the two
    # if pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') > to_s_id_07 and
    # if pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') > to_s_id_06 and
    #    pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M') < to_s_id_15),
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
    svm_results, heatmap_intervals, heatmap_y_names, x_modality='mPFC',
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
    'L2': analytics.get_analytics('SVMCueOutcomeChoicePred', session_names=session_names),
}
dist_figs, dist_summary = plot_metric_distribution(
    dist_variants, which_intervals=intervals, y_predicted_names=None, x_modality='mPFC',
    session_ids=plot_session_ids, metric='balanced_accuracy',
    show_n=True,
    ccgp_w_train_stratum={'cue': 1, 'choice_R1': 1}, plot_type='violin',
    save_path=os.path.join(session_plots_dir, 'metric_distribution.png'),
)

# %%
analytics.get_analytics('SVMCueOutcomeChoicePred', session_names=session_names)
# %%
