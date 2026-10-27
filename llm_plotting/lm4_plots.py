#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

# to setup import paths add project root dirs to sys.path
sys.path.append(os.path.join(os.getcwd(), "..", ".."))
from baseVR.base_functionality import init_import_paths
init_import_paths()


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from dashsrc.plot_components.plots import plot_EnsambleEncondings
from dashsrc.plot_components.plots import plot_EnsembleChoiceEncoding
from dashsrc.plot_components.plots import plot_SessionKinematics


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib qt
get_ipython().run_line_magic('matplotlib', 'widget')

from analytics_processing import analytics
import analytics_processing.analytics_constants as C
from CustomLogger import CustomLogger as Logger

from dashsrc.plot_components.plot_wrappers.data_selection import group_filter_data

from analytics_processing.modality_loading import session_modality_from_nas
from analytics_processing.sessions_from_nas_parsing import sessionlist_fullfnames_from_args
from analytics_processing.sessions_from_nas_parsing import fullfnames2snames

# import GridSpecFromSubplotSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec
import scipy.stats as stats
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap, Normalize



# In[4]:


data = {}
nas_dir = C.device_paths()[0]
output_dir = os.path.join(nas_dir, 'Simon', 'lm4')
Logger().init_logger(None, None, logging_level="INFO")


# In[5]:


animal_ids = [6]
paradigm_ids = [1100]
session_ids = None


# In[6]:


session_dirs, _ = sessionlist_fullfnames_from_args(paradigm_ids, animal_ids, session_ids)
session_names = fullfnames2snames(session_dirs)
session_to_dt_session = {sn[:16]:i for i,sn in enumerate(session_names)}
session_names


# In[ ]:





# In[7]:


trialw_beh = analytics.get_analytics('BehaviorTrialwise', session_names=session_names[:8])

trialw_beh.index = trialw_beh.index.droplevel(('animal_id', 'paradigm_id', 'entry_id'))
trialw_beh.index = trialw_beh.index.map(lambda x: session_to_dt_session[x[:16]] if x[:16] in session_to_dt_session else x)
trialw_beh


# In[ ]:





# In[8]:


plt.rcParams.update({
    'font.size': 15,           # general font size
    'axes.titlesize': 15,      # subplot title
    'axes.labelsize': 15,      # x and y labels
    'xtick.labelsize': 15,     # x tick labels
    'ytick.labelsize': 15,     # y tick labels
    'legend.fontsize': 15,     # legend
})


# In[9]:


plt.close('all')
fig, axes = plt.subplots(figsize=(3.5,5), nrows=2, sharex=True)
x, y_loc1, y_loc2 = [], [], []
for s_id in trialw_beh.index.unique('session_id'):
    # print(s_id)
    a_trialw_beh = trialw_beh.loc[s_id]
    n_R_1st_location = a_trialw_beh.trial_outcome %5
    a_trialw_beh.loc[:,'n_R_1st_location'] = n_R_1st_location.values
    n_R_2nd_location = a_trialw_beh.trial_outcome //10
    a_trialw_beh.loc[:,'n_R_2nd_location'] = n_R_2nd_location.values
    n = len(a_trialw_beh)
    # print(a_trialw_beh[['trial_outcome', 'n_R_1st_location', 'n_R_2nd_location']].to_string())
    print(f"Session {s_id}: n_R1: {(n_R_1st_location>=1).sum()}/{n}, n_R2: {(n_R_2nd_location>=1).sum()}/{n}")
    
    # print(f"Session {s_id}: 1st loc R proportion: {(n_R_1st_location>1).sum()/n:.2f}, 2nd loc R proportion: {(n_R_2nd_location>1).sum()/n:.2f}")
    x.append(s_id)
    y_loc1.append((n_R_1st_location>=1).sum()/n)
    # y_loc1
    y_loc2.append((n_R_2nd_location>=1).sum()/n)

    axes[0].scatter([s_id,], [y_loc1[-1]], label='1st location R' if s_id == 0 else None, color='lightgrey')
    axes[1].scatter([s_id,], [y_loc2[-1]], label='2nd location R' if s_id == 0 else None, color='darkgrey')

# linear fit
x_num = np.arange(len(x))
slope, intercept, r_value, p_value, std_err = stats.linregress(x_num, y_loc1)
axes[0].plot(x, intercept + slope * x_num, '--', label=f'Fit: slope={slope:.3f}, p={p_value:.3f}', color='lightgrey', linewidth=2)
axes[0].set_title(f'R1 (1st location reward)\np={p_value:.3f}')

slope, intercept, r_value, p_value, std_err = stats.linregress(x_num, y_loc2)
axes[1].plot(x, intercept + slope * x_num, '--', label=f'Fit: slope={slope:.3f}, p={p_value:.3f}', color='darkgrey', linewidth=2)
axes[1].set_title(f'R2 (2nd location reward)\np={p_value:.3f}')

axes[1].set_xlabel('Session ID')
axes[0].set_ylabel('Proportion of trials')
axes[1].set_ylabel('Proportion of trials')
axes[0].set_ylim(0, .6)
axes[1].set_ylim(0, .6)
axes[1].set_xticks(x)
axes[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
# gridlines
axes[0].grid(axis='x', alpha=0.5, linestyle='--')
axes[1].grid(axis='x', alpha=0.5, linestyle='--')
# removee all spines
[ax.spines[['top', 'right']].set_visible(False) for ax in axes]



fig.tight_layout(h_pad=2.5)
plt.show()

fname = os.path.join(output_dir, 'double_stop_performance.svg')
fig.savefig(fname, format='svg', dpi=300)


# In[ ]:


# session-7 movement kinematics, grouped by R1 choice (stop vs skip)
for s_id in range(7,16):
    # if s_id not in [7]:
    #     continue
    s_session_name = session_names[s_id]
    trackw_beh_s = analytics.get_analytics('BehaviorTrackwise', session_names=[s_session_name])
    sess_meta_s = analytics.get_analytics('SessionMetadata', session_names=[s_session_name])

    # NB: BehaviorTrackwise's own 'choice_R1' disagrees with BehaviorTrialwise's for ~1/3 of
    # trials (verified against trial_outcome, which matches perfectly across both sources) -
    # trialwise is the internally-consistent one, so overwrite trackwise's copy with it before
    # grouping/coloring by R1 choice.
    trialw_s = analytics.get_analytics('BehaviorTrialwise', session_names=[s_session_name])
    correct_choice_R1 = trialw_s.reset_index().set_index('trial_id')['choice_R1']
    trackw_beh_s['choice_R1'] = trackw_beh_s['trial_id'].map(correct_choice_R1)

    group_by = 'R1 choice'
    n_trials = int(trackw_beh_s['trial_id'].dropna().max())
    # NB: group_filter_data's own cue_filter default (['Early R', 'Late R']) doesn't match
    # what it checks internally ('Cue1 trials'/'Cue2 trials'), so it must be passed explicitly.
    filtered_trackw_s, group_by_values = group_filter_data(trackw_beh_s,
                                                            cue_filter=['Cue1 trials', 'Cue2 trials'],
                                                            group_by=group_by)

    fig = plot_SessionKinematics.render_plot(filtered_trackw_s, sess_meta_s, n_trials,
                                            group_by, group_by_values,
                                            metric='RawYawPitch Sum Velocity', metric_max=220,
                                            smooth_data=True, var_viz='80th percent.', events=[],
                                            width=900, height=500)
    fig.show()

    fname = os.path.join(output_dir, f'session{s_id}_kinematics_choiceR1_RYP_sum.svg')
    fig.write_image(fname)
    fig.write_image(fname.replace('.svg', '.png'), scale=2)

    fig = plot_SessionKinematics.render_plot(filtered_trackw_s, sess_meta_s, n_trials,
                                            group_by, group_by_values,
                                            metric='BallForward Velocity', metric_max=220,
                                            smooth_data=True, var_viz='80th percent.', events=[],
                                            width=900, height=500)
    # update subtitle
    print(f"Session {s_id}")
    fig.show()

    fname = os.path.join(output_dir, f'session{s_id}_kinematics_choiceR1_BallForward.svg')
    fig.write_image(fname)
    fig.write_image(fname.replace('.svg', '.png'), scale=2)
    print('\n\n\n\n\n')


# %%

# In[ ]:


# 2x2 (or 1x2, cue collapsed) matrix of trial counts split by choice_R1 (cols) and cue (rows)
# color swatches double as the row/col labels, matching the choice_R1/cue color scheme in
# dashvis_constants.R1_CHOICE_CMAP / svm_analysis_2026.modality_colors
CHOICE_R1_MATRIX_COLORS = {0: '#3094db', 1: '#0044ff'}  # skip, stop
CUE_MATRIX_COLORS = {1: '#e98800', 2: '#bc00e9'}        # cue 1, cue 2

def plot_choiceR1_cue_matrix(trial_data, collapse_cue=False, normalize=False, ax=None, title=None):
    choice_vals = [0, 1]
    choice_labels = {0: 'skip', 1: 'stop'}

    if collapse_cue:
        vc = trial_data['choice_R1'].value_counts().reindex(choice_vals, fill_value=0)
        matrix = vc.values.reshape(1, 2).astype(float)
        cue_vals = []
    else:
        cue_vals = sorted(trial_data['cue'].dropna().unique())
        matrix = np.zeros((len(cue_vals), 2))
        for i, cv in enumerate(cue_vals):
            sub = trial_data[trial_data['cue'] == cv]
            vc = sub['choice_R1'].value_counts().reindex(choice_vals, fill_value=0)
            matrix[i] = vc.values

    total_n = int(matrix.sum())
    n_rows = matrix.shape[0]

    header = 0.5   # total header size (row/col label area)
    strip = 0.2    # thickness of the color strip within the header, against the matrix
    gap = 0.05     # negative space between all rectangles
    x_off = header if not collapse_cue else 0

    if ax is None:
        fig_w = 1.1 * 2 + x_off + 0.3
        fig_h = 1.1 * n_rows + header + 0.3
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    else:
        fig = ax.figure

    ax.set_xlim(0, x_off + 2)
    ax.set_ylim(0, n_rows + header)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.axis('off')

    def add_rect(x, y, w, h, **kwargs):
        ax.add_patch(plt.Rectangle((x + gap / 2, y + gap / 2), w - gap, h - gap, **kwargs))

    # cell border alpha gives a slight indication of magnitude (fraction of total + a
    # constant floor, so even a 0% cell isn't borderless); face stays plain white
    alpha_const = 0.2

    # column strips (choice_R1: skip/stop), a thin band just above the data cells; the
    # label sits in the blank space above the strip, not on top of the color itself
    for j, cv in enumerate(choice_vals):
        add_rect(x_off + j, header - strip, 1, strip, facecolor=CHOICE_R1_MATRIX_COLORS[cv],
                edgecolor='none')
        ax.text(x_off + j + 0.5, (header - strip) / 2, choice_labels[cv], ha='center', va='center',
                color='black', fontsize=7)

    # row strips (cue), a thin band just left of the data cells, unless collapsed; same
    # blank-space-next-to-the-strip placement as the column labels
    if not collapse_cue:
        for i, cv in enumerate(cue_vals):
            add_rect(header - strip, header + i, strip, 1, facecolor=CUE_MATRIX_COLORS.get(cv, 'grey'),
                    edgecolor='none')
            ax.text((header - strip) / 2, header + i + 0.5, f'cue {int(cv)}', ha='center', va='center',
                    color='black', fontsize=7, rotation=90)

    # corner: total n_trials (only a real cell when row labels are present, else folded into the title below)
    if not collapse_cue:
        ax.text(header / 2, header / 2, f'n_trials={total_n}', ha='center', va='center', fontsize=8)

    # data cells: always white, border alpha gives a slight indication of magnitude
    for i in range(n_rows):
        for j in range(2):
            val = matrix[i, j]
            frac = val / total_n if total_n else 0
            edge_alpha = min(1.0, frac + alpha_const)
            add_rect(x_off + j, header + i, 1, 1, facecolor='white',
                    edgecolor=(0, 0, 0, edge_alpha), linewidth=1.5)
            label = f'n={int(val)}'
            if normalize:
                label += f'\n({frac*100:.0f}%)'
            ax.text(x_off + j + 0.5, header + i + 0.5, label, ha='center', va='center',
                    fontsize=8, color='black')

    # title reflects whether the cue dimension is actually shown; when collapsed there's
    # no corner cell for total n, so fold it into the title instead
    suffix = 'choice_R1' if collapse_cue else 'choice_R1 x cue'
    full_title = f'{title}: {suffix}' if title else suffix
    if collapse_cue:
        full_title += f' (n_trials={total_n})'
    ax.set_title(full_title, fontsize=10, pad=6)

    fig.tight_layout()
    return fig, ax


# %%

# In[68]:



# reload independently rather than reusing the kinematics cell's trackw_beh_s/s_id -
# BehaviorTrialwise alone has everything this needs (trial-level cue + choice_R1) and is
# the internally-consistent source (see the choice_R1 note in the kinematics cell above)
matrix_s_id = 7
matrix_s_session_name = session_names[matrix_s_id]
trialw_matrix = analytics.get_analytics('BehaviorTrialwise', session_names=[matrix_s_session_name])
trial_level_s = trialw_matrix.reset_index()[['trial_id', 'cue', 'choice_R1']].set_index('trial_id')

fig, ax = plot_choiceR1_cue_matrix(trial_level_s, collapse_cue=True, normalize=True,
                                   title=f'Session {matrix_s_id}')
plt.show()

fname = os.path.join(output_dir, f'session{matrix_s_id}_choiceR1_cue_matrix.svg')
fig.savefig(fname, format='svg', dpi=300)
fig.savefig(fname.replace('.svg', '.png'), dpi=300)



# %%
# the numbers above come from here.
#
# Null-distribution rehaul: rather than comparing actual fitted SVM weight vectors
# (which conflates "chance baseline" with whatever structure the fit/bootstrap
# procedure itself imparts, and pools W's across *different* sessions' - i.e.
# different physical neurons' - decoders into one distribution), sample a bunch of
# raw z-scored population-activity vectors (FiringRate40msZ, one 40ms native bin =
# one vector) per session and compute pairwise cosine angles between those instead -
# a more direct "what does chance look like for this dataset's own neurons" baseline.
# Kept strictly within-session (vectors from different sessions are never compared to
# each other - different sessions record different physical neurons even at the same
# column index/X_modality slice, so a cross-session angle wouldn't mean anything);
# only the resulting angle *values* are pooled across sessions into one combined
# per-modality null. Column slicing must match get_SVMCueOutcomeChoicePred's X_full
# construction (postproc_mea1k_ephys.py, fr_z.iloc[:, 20:] for 'mPFC') - same fragile,
# not-yet-centralized split _x_modality_fr uses elsewhere in this file (see its
# docstring) - so any future change to that split needs updating here too.
plt.close("all")
rng = np.random.default_rng(42)
n_samples_per_session = 300

# for x_modal in ['mPFC', 'HP', 'HP+mPFC']:
for x_modal in ['mPFC', ]:
    plt.figure(figsize=(5, 5))
    all_degrees = []
    for full_sname in session_names:
        s_id = full_sname[:16]
        fr_z_raw = analytics.get_analytics('FiringRate40msZ', session_names=[full_sname])
        if fr_z_raw is None:
            continue
        fr_z_raw = fr_z_raw.xs(s_id, level='session_id').drop(
            columns=['from_ephys_timestamp', 'to_ephys_timestamp'])
        if x_modal == 'mPFC':
            fr_x = fr_z_raw.iloc[:, 20:]
        elif x_modal == 'HP':
            fr_x = fr_z_raw.iloc[:, :20]
        else:
            fr_x = fr_z_raw
        n = len(fr_x)
        if n < 2:
            continue
        # subsample row *indices* first, then fillna/convert only that small slice -
        # fr_x itself can be tens of thousands of rows (every 40ms bin of the session),
        # no need to densify/fillna the whole thing just to use ~300 of its rows
        sample_idx = rng.choice(n, size=min(n_samples_per_session, n), replace=False)
        vecs = fr_x.iloc[sample_idx].fillna(0).values
        norms = np.linalg.norm(vecs, axis=1)
        cos_sim_matrix = (vecs @ vecs.T) / (norms[:, np.newaxis] * norms[np.newaxis, :])
        degrees = (np.arccos(np.clip(cos_sim_matrix, -1.0, 1.0)) * (180 / np.pi)).flatten()
        all_degrees.append(degrees)

    degrees = np.concatenate(all_degrees)
    ax = plt.gca()
    hist = ax.hist(degrees, bins=80, color='gray', density=True)
    # draw in 2 std dev lines
    mean_angle = np.nanmean(degrees)
    std_angle = np.nanstd(degrees)
    lo, hi = mean_angle - 2*std_angle, mean_angle + 2*std_angle
    print(hi, x_modal)
    print(lo, x_modal)
    ax.axvspan(lo, hi, color='lightblue', alpha=0.15, zorder=0, label='~95% of angles')
    ax.axvline(hi, color='lightblue', linestyle='--')
    ax.axvline(lo, color='lightblue', linestyle='--')
    ax.annotate(f'{hi:.1f}°', (hi, ax.get_ylim()[1]), xytext=(3, -3),
                textcoords='offset points', va='top', color='steelblue')
    ax.annotate(f'{lo:.1f}°', (lo, ax.get_ylim()[1]), xytext=(-3, -3),
                textcoords='offset points', va='top', ha='right', color='steelblue')
    ax.set_xticks([0, 60, 90, 120, 180])
    ax.set_xlim(0, 180)
    ax.grid(axis='x', color='lightgray', linewidth=0.5, zorder=0)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f'Null angle distribution ({x_modal}, chance baseline)')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Density')
    plt.tight_layout()
    # save to output dir
    plt.savefig(os.path.join(output_dir, f'null_angle_distribution_{x_modal}.svg'), dpi=300)
    
plt.show()
# %%
# ===========================================================================
# The hand-picked W's - one definition, used by every cell below.
# ===========================================================================
# Its own cell, above the first use: passing this list to a plot only draws a callout box
# (see HIGHLIGHT_COLOR) - the plots never hand data back. `w` starts None and is filled in
# place by svm_plotting_lib.resolve_w_selections (the resolver cell near the bottom), so
# this list doubles as the record of what was actually picked.
#
# A selection only constrains the keys it sets (see _selection_matches), so leaving out
# conditioner/train_stratum matches naive and CCGP fits alike - but pinning them, as the
# S13_cue_* entries do, means those can ONLY resolve against a CCGP analytic
# (SVML1CueOutcomeChoicePred2 has no CCGP rows at all).
W_SELECTIONS = [
    # # the shared choice-R1 axis, one W per session - the sequence to reference/compare
    {'session_id': '2024-11-26_16-39', 'y_name': 'choice_R1',
     'interval': 'R1_entry_interval', 'timebin': 8, 'name': 'S07_choice_earlySquare', 'w': None},
    
    # {'session_id': '2024-11-26_16-39', 'y_name': 'choice_R1',
    #  'interval': 'R1_entry_interval', 'timebin': 13-3, 'name': 'S07_choice_early2ndSquare', 'w': None},
    

    # # smth else
    # {'session_id': '2024-12-04_18-06', 'y_name': 'choice_R1', 'conditioner': 'cue',
    #  'train_stratum': 1, 'interval': 'R1_entry_interval', 'timebin': 9 + 4 + 3,
    #  'name': 'S13_choice_2ndsquare_maxAligned', 'w': None},
]

# %%

# Test plot for svm_plotting_lib.plot_session_analysis (extracted from svm_analysis_2026.py
# so it can be imported here without re-running that notebook's own top-level cells - see
# svm_plotting_lib.py's module docstring). Session/target/interval selection is local to
# this cell and independent of the rest of this notebook.

from svm_plotting_lib import plot_session_analysis, build_canvas, build_canvas_svg
from plotlib import session_data as sd

svm_x_modality = 'mPFC'
svm_y_predicted_name = ['choice_R1',]  # or 'cue' / 'choice_R1' / 'choice_R2' / a list of these
# svm_y_predicted_name = ['cueANDstopR1', 'cue']  # or 'cue' / 'choice_R1' / 'choice_R2' / a list of these
svm_which_intervals = [ 'R1_entry_interval']
svm_which_intervals = ['cue_entry_interval', 'cue_exit_interval', 'R1_entry_interval',]# 'R2_entry_interval', 'ITI_interval']
svm_test_session_names = session_names[7:8]  # one session, just to test the plot works

# Was 'SVML1CueOutcomeChoicePred' - an orphaned generation whose 42-bin fit windows match
# NO t0_events table on the NAS (the stale one stores 40/45, the current one 25/34), so
# every position trace it drew was misaligned with its own decoding curve and nothing
# could fix it. This is a plain choice_R1 figure, not CCGP, so it simply moves to the
# current analytic, which is self-consistent and covers 28 sessions instead of 15.
SVM_ANALYTIC = 'SVML1CueOutcomeChoicePred2'
# SVM_ANALYTIC = 'SVML2CueOutcomeChoicePred2'
# SVM_ANALYTIC = 'SVML1CueOutcomeChoicePredCCGP'
svm_results_test = analytics.get_analytics(SVM_ANALYTIC, session_names=svm_test_session_names)

# Deliberately the STALE t0 table: it is the one these two older analytics were fit
# against (see plotlib.session_data.ANALYTIC_PAIRS). The current-geometry fit
# SVML1CueOutcomeChoicePred2 has no CCGP rows, so the CCGP figures below stay on this
# pairing until that analytic is recomputed. check_geometry confirms the pairing holds.
t0_events_test = sd.load_t0_events(svm_test_session_names, analytic='TrialWiseT0Events40ms2')
# sd.check_geometry(t0_events_test, svm_results_test)

behavior_test = analytics.get_analytics(
    'Behavior40msAligned', session_names=svm_test_session_names,
    columns=['trial_id', 'cue', 'trial_outcome', 'choice_R1', 'choice_R2',
             'to_ephys_timestamp', 'frame_position'])
behavior_test = behavior_test.reset_index().drop(columns=['paradigm_id', 'animal_id', 'entry_id'])
behavior_test.index = pd.IntervalIndex.from_arrays(
    left=behavior_test.to_ephys_timestamp - 40_000,
    right=behavior_test.to_ephys_timestamp,
    closed='right')
behavior_test.set_index('session_id', append=True, inplace=True)
behavior_test['choice_R1'] = behavior_test['choice_R1'].astype(bool)

# Sessions are resolved from what actually made it into the 3 analytics tables above
# (rather than from svm_test_session_names directly) - a session sliced in up top can
# still come back empty from analytics.get_analytics (e.g. no SVM fit yet), and indexing
# get_session_data with an s_id absent from its index is what raised the KeyError here.
plot_session_ids = sorted(
    set(svm_results_test.index.unique('session_id')) &
    set(t0_events_test.index.unique('session_id')) &
    set(behavior_test.index.unique('session_id')),
    key=lambda s_id: pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M'))

fr_z_session = None
full_name = next(sn for sn in session_names if sn.startswith(s_id))
fr_z_session = analytics.get_analytics(
    'FiringRate40msZ', session_names=[full_name]).xs(s_id, level='session_id')


all_saved_png_paths, all_saved_svg_paths = [], []
for s_id in plot_session_ids:
    if s_id not in session_to_dt_session:
        print(f"Session {s_id} not found in session_to_dt_session mapping, skipping.")
        continue
    dt_session = session_to_dt_session[s_id]
    print(f"Session {s_id} mapped to dt_session {dt_session}.")
    png_paths, svg_paths = plot_session_analysis(
        s_id, t0_events_test, behavior_test, svm_results_test,
        x_modalities=svm_x_modality, y_predicted_names=svm_y_predicted_name,
        which_intervals=svm_which_intervals, 
        ith_to_dt_session=session_to_dt_session, 
        show_shuffle=True, 
        show_mean=False,
        min_metric=0.55,
        mask_chance_band=True,
        smooth=2,
        show_trial_kinematics=False,
        show_w_comparison=True,
        show_covariate_plot=False,
        # show_colorbar=True,
        show_ylabel=False,
        show_title=False,
        show_ccgp_secondary=True,
        show_w_variance=True,
        fr_z_session=fr_z_session,
        # Must be set for any CCGP analytic. Without it the election falls back to the
        # automatic more_reliable rule (whichever direction trained on the LARGER
        # stratum), which is per-session by construction and therefore inconsistent
        # across the sequence: S14 trains 48 (C1) vs 47 (C2), so it picks C1, while S15
        # trains 36 (C1) vs 47 (C2) and picks C2. That flip is what made S14's star, row
        # 2 heatmap, covariate lines and W block all show the other direction.
        ccgp_w_train_stratum={'cue': 1, 'choice_R1': 1, }, #'choice_R2': 1},
        save_dir=os.path.join(output_dir, 'svm_analysis_choiceR1'),
        
        highlight_selections=W_SELECTIONS,
    )
    if png_paths:
        all_saved_png_paths.append(png_paths)
        all_saved_svg_paths.append(svg_paths)


if all_saved_png_paths:
    canvas_path = os.path.join(output_dir, 'svm_analysis_choiceR1', 'choiceR1_0to15.png')
    build_canvas(all_saved_png_paths, canvas_path)
    print(f"Saved combined canvas of {len(all_saved_png_paths)} sessions to {canvas_path}")

    canvas_path = os.path.join(output_dir, 'svm_analysis_choiceR1', 'choiceR1_0to15.svg')
    build_canvas_svg(all_saved_svg_paths, canvas_path)
    print(f"Saved combined SVG canvas of {len(all_saved_svg_paths)} sessions to {canvas_path}")
# %%
# Cross-session W comparison - same (x_modality, predict_y_name) fit, compared across
# the session sequence resolved above, instead of across columns within one session (see
# plot_session_analysis's own show_w_comparison for that). Now lives in
# svm_plotting_lib.py (get_session_sequence_Ws/plot_session_sequence_w_comparison) rather
# than being redefined here - this cell just imports and calls it, mirroring
# svm_analysis_2026.py's own "Cross-session weight comparison" cell, but reusing this
# notebook's already-fetched svm_results_test/plot_session_ids/session_to_dt_session.
from svm_plotting_lib import plot_session_sequence_w_comparison

sequence_save_dir = os.path.join(output_dir, 'svm_analysis_choiceR1', )
os.makedirs(sequence_save_dir, exist_ok=True)

for y_predicted_name in svm_y_predicted_name:  # single predict_y_name per plot; loop covers each
    save_path = os.path.join(sequence_save_dir, f'session_sequence_{y_predicted_name}_{svm_x_modality}.png')
    save_path_svg = os.path.join(sequence_save_dir, f'session_sequence_{y_predicted_name}_{svm_x_modality}.svg')
    # highlight_selections draws the callout boxes only - resolve_w_selections is what
    # gets the W's themselves (see the resolver cell further down)
    plot_session_sequence_w_comparison(
        plot_session_ids, svm_results_test, t0_events_test, svm_x_modality, y_predicted_name,
        svm_which_intervals,
        mask_chance_band=True, min_metric=0.55, ith_to_dt_session=session_to_dt_session,
        # Must match the per-session call above. Left off, S14 elects C1 (more_reliable,
        # 48 vs 47 train trials) - and C1's decoder only ever reaches 0.561 balanced
        # accuracy, so min_metric=0.55 dims 95% of its cells and the whole block renders
        # flat. C2 reaches 0.703 and keeps ~half its cells. The two directions' *angles*
        # are near-identical (mean |angle-90| 18.9 vs 16.4 deg) - what you see is the
        # alpha mask, not the geometry.
        ccgp_w_train_stratum={'cue': 1, 'choice_R1': 1, }, #'choice_R2': 1},
        smooth=2,
        # at 13
        # reference_session_id=next(sid for sid, idx in session_to_dt_session.items() if idx == 13),
        # at 7
        reference_session_id=next(sid for sid, idx in session_to_dt_session.items() if idx == 7),
        highlight_selections=W_SELECTIONS,
        save_path=save_path, save_path_svg=save_path_svg)

# saved at
print(f"Saved cross-session W comparison plots to {sequence_save_dir}")













# %%


"""Getting the W's for the hand-picked selections above, so they can be plotted in a callout box"""

from svm_plotting_lib import resolve_w_selections, plot_w_comparison

# W_SEL_ANALYTIC = 'SVML1CueOutcomeChoicePredCCGP'
W_SEL_ANALYTIC = 'SVML1CueOutcomeChoicePred2'
W_SEL_X_MODALITY = 'mPFC'
W_SEL_CCGP_STRATUM = {'cue': 1, 'choice_R1': 1, 'choice_R2': 1}

w_sel_svm_results = analytics.get_analytics(W_SEL_ANALYTIC, session_names=session_names)

selected_w_rows_df = resolve_w_selections(
    W_SELECTIONS, w_sel_svm_results, x_modality=W_SEL_X_MODALITY,
    ccgp_w_train_stratum=W_SEL_CCGP_STRATUM)

selected_w_rows_df[['session_id', 'name', 'predict_y_name', 'interval_name', 'timebin',
                    'X_modality', 'acc_aggr']]



# %%
# First W figure: each hand-picked selection's weight vector as its own thin panel,
# features (neurons) top-to-bottom on a shared y-axis, plus the region-gradient column.
# Runs straight off selected_w_rows_df from the cell above (resolve_w_selections filled
# each row's `w`), so it needs no analytic of its own. plot_W_list does not create its
# save dir - hence the makedirs, same dir the CCGP figures above write into.
from w_plotting_lib import plot_W_list

w_list_save_dir = os.path.join(output_dir, 'svm_analysis_cue_CCGP')
os.makedirs(w_list_save_dir, exist_ok=True)

plot_W_list(
    selected_w_rows_df, plot_type='stem',
    ith_to_dt_session=session_to_dt_session,  # session_id -> S<n> in the titles
    save_path=os.path.join(w_list_save_dir, 'highlighted_Ws_stem.png'),
    save_path_svg=os.path.join(w_list_save_dir, 'highlighted_Ws_stem.svg'))


# %%
# ===========================================================================
# W projection - shared setup for both options below
# ===========================================================================
# Data for the projection figures, fetched here rather than reused from the CCGP cells
# above: those narrowed both the session range (session_names[8:16]) and behavior's
# columns, and the projection needs neither restriction - its velocity row reads
# frame_raw_500msMedian, which that narrowed fetch does not include. The t0 table is
# pinned to W_SEL_ANALYTIC's own pairing (see plotlib.session_data.ANALYTIC_PAIRS), i.e.
# the same geometry the W's in selected_w_rows_df were fit under.
from w_plotting_lib import plot_w_projection_session

WPROJ_INTERVALS = ['cue_entry_interval', 'R1_entry_interval']  # free choice, any of ALL_INTERVALS
WPROJ_SPLIT_COVARIATE = None  # None -> split row 1 by each W's own predict_y_name;
                              # or name one ('cue' / 'choice_R1' / ...) to split by that
                              # instead, in which case it drops out of the grid below.
WPROJ_YLIM = (-0.3, 0.3)  # fixed across sessions on purpose - autoscaling per session
                          # would hide the very drift option 2 is looking for; None
                          # falls back to each session's own empirical range.

wproj_session_names = session_names[7:8]  # one session, just to test the plot works
wproj_t0_events = sd.load_t0_events(wproj_session_names, analytic='TrialWiseT0Events40ms2')
wproj_behavior = sd.load_behavior(wproj_session_names)

# %%



# %%

# w_row is picked once, outside the loop, and the *session*
# varies - so this shows how one hand-picked subspace holds up (or drifts) in sessions it
# was never fit on. Its own session list, deliberately defined here rather than reusing
# the CCGP cells' plot_session_ids: which sessions are worth projecting into is this
# figure's own question, and the answer is usually "more of them than were plotted above".
WPROJ_FIXED_W_NAME = None  # None -> the first selection; or a W_SELECTIONS 'name' to pin one
# WPROJ_FIXED_W_NAME = 'S07_choice_earlyEarly'  # None -> the first selection; or a W_SELECTIONS 'name' to pin one

wproj_fixed_w_row = (selected_w_rows_df.iloc[0] if WPROJ_FIXED_W_NAME is None else
                     selected_w_rows_df[selected_w_rows_df['name'] == WPROJ_FIXED_W_NAME].iloc[0])

wproj_plot_session_ids = sorted(
    set(wproj_t0_events.index.unique('session_id')) & set(wproj_behavior.index.unique('session_id')),
    key=lambda s_id: pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M'))
# wproj_plot_session_ids = wproj_plot_session_ids[8:16]  # narrow the sweep here

wproj_fixed_tag = (wproj_fixed_w_row['name'] if pd.notna(wproj_fixed_w_row.get('name'))
                   else f"{wproj_fixed_w_row['interval_name']}_t{wproj_fixed_w_row['timebin']}")
wproj_fixed_dir = os.path.join(output_dir, 'w_projection', f'fixed_{wproj_fixed_tag}')
os.makedirs(wproj_fixed_dir, exist_ok=True)

wproj_fixed_paths = []
for s_id in wproj_plot_session_ids:
    saved_path = plot_w_projection_session(
        s_id, wproj_fixed_w_row, wproj_t0_events, wproj_behavior, wproj_session_names,
        WPROJ_INTERVALS, split_covariate=WPROJ_SPLIT_COVARIATE, projection_ylim=WPROJ_YLIM,
        smooth=2, highlight_range=(wproj_fixed_w_row['interval_name'],
                         wproj_fixed_w_row['timebin'], wproj_fixed_w_row['timebin']),
        ith_to_dt_session=session_to_dt_session, save_dir=wproj_fixed_dir)
    wproj_fixed_paths.append(saved_path)

if wproj_fixed_paths:
    canvas_path = os.path.join(wproj_fixed_dir, 'wproj_fixed_canvas.png')
    build_canvas([[p] for p in wproj_fixed_paths], canvas_path)
    print(f"Saved combined canvas of {len(wproj_fixed_paths)} sessions to {canvas_path}")
    
# %%
# Does this W's activation track a kinematic variable rather than (or on top of) the task
# covariate it was fit to distinguish? Two views of the same question, both on the same
# wproj_fixed_w_row / wproj_* data as the projection cell above.
from w_plotting_lib import (plot_w_variance_explained, plot_w_projection_kinematics,
                            plot_session_projection_kinematics)

# Any session the wproj_* data covers - not necessarily the W's own session, same as the
# fixed-W loop above (which projects one hand-picked W into sessions it was never fit on).
kin_s_id = wproj_plot_session_ids[0]
kin_interval, kin_timebin_range = 'R1_entry_interval', (5, 15)
kin_offset = 0  # native bins; shifts *only* the kinematics side's window later, to test
                # whether kinematics N bins after the projection's window explain it
kin_color_by = wproj_fixed_w_row['predict_y_name']  # the whole-session view (2) needs a plain
                                                    # per-bin behavior column ('cue'/'choice_R1'/
                                                    # 'choice_R2'/'trial_outcome'); correct_R1/R2
                                                    # are trial-only -> set None for those

# 1) Windowed: one scatter point per (trial, bin) sample inside kin_timebin_range. Sanity
#    checks first - the purple dashed line in plot_w_variance_explained is how much of that
#    window's pooled variance the W direction itself explains (near 0 -> the decoder barely
#    moves here, so read any correlation below cautiously), and plot_w_projection_session
#    shades the same window on the actual projection trace for context.
_ = plot_w_projection_session(
    kin_s_id, wproj_fixed_w_row, wproj_t0_events, wproj_behavior, wproj_session_names,
    [kin_interval], show_covariate_plot=False, smooth=2,
    ith_to_dt_session=session_to_dt_session,
    highlight_range=(kin_interval, *kin_timebin_range))

kin_windowed_correlations = plot_w_projection_kinematics(
    kin_s_id, wproj_fixed_w_row, wproj_t0_events, wproj_behavior, wproj_session_names,
    kin_interval, kin_timebin_range, kinematic_offset=kin_offset, color_by=kin_color_by,
    ith_to_dt_session=session_to_dt_session)

# 2) Whole session: the same scatter over *every* native 40ms bin of the continuous
#    recording, inter-trial time included - no t0_events/interval/timebin involvement at
#    all. Answers whether a correlation from (1) is specific to the task window or just
#    holds throughout the session. max_points subsamples for rendering only (tens of
#    thousands of bins per session); Pearson r is computed on the full aligned set first.
kin_session_correlations = plot_session_projection_kinematics(
    kin_s_id, wproj_fixed_w_row, wproj_behavior, wproj_session_names,
    color_by=kin_color_by, max_points=20000, ith_to_dt_session=session_to_dt_session)

kin_windowed_correlations, kin_session_correlations




# %%

















"""MOVING ON TO CUE"""





# reload independently rather than reusing the kinematics cell's trackw_beh_s/s_id -
# BehaviorTrialwise alone has everything this needs (trial-level cue + choice_R1) and is
# the internally-consistent source (see the choice_R1 note in the kinematics cell above)
matrix_s_id = 13
matrix_s_session_name = session_names[matrix_s_id]
trialw_matrix = analytics.get_analytics('BehaviorTrialwise', session_names=[matrix_s_session_name])
trial_level_s = trialw_matrix.reset_index()[['trial_id', 'cue', 'choice_R1']].set_index('trial_id')

fig, ax = plot_choiceR1_cue_matrix(trial_level_s, collapse_cue=True, normalize=True,
                                   title=f'Session {matrix_s_id}')
plt.show()

fname = os.path.join(output_dir, f'session{matrix_s_id}_choiceR1_cue_matrix.svg')
fig.savefig(fname, format='svg', dpi=300)
fig.savefig(fname.replace('.svg', '.png'), dpi=300)



# %%

# Test plot for svm_plotting_lib.plot_session_analysis (extracted from svm_analysis_2026.py
# so it can be imported here without re-running that notebook's own top-level cells - see
# svm_plotting_lib.py's module docstring). Session/target/interval selection is local to
# this cell and independent of the rest of this notebook.

from svm_plotting_lib import plot_session_analysis, build_canvas, build_canvas_svg

# the clearly joint cue choice overlap from session 13
# session13. --  the one to reference
W_SELECTIONS = [
    {'session_id': '2024-12-04_18-06', 'y_name': 'cue', 'conditioner': 'choice_R1', 'train_stratum': 1, 
     'interval': 'cue_entry_interval', 'timebin': 9+4+5, 'name': 'S13_Cue_cue_choice_overlap_2ndsquare', 'w': None},
    # session13. --  first square
    # {'session_id': '2024-12-04_18-06', 'y_name': 'cue', 'conditioner': 'choice_R1',    'train_stratum': 1, 
    #  'interval': 'cue_entry_interval', 'timebin': 8, 'name': 'S13_Cue_cue_choice_overlap_1stsquare', 'w': None},
    {'session_id': '2024-12-04_18-06', 'y_name': 'cue', 'conditioner': 'choice_R1',  'train_stratum': 1, 
     'interval': 'cue_entry_interval', 'timebin': 9, 'name': 'S13_Cue_cue_choice_overlap_1stsquare', 'w': None},
    # {'session_id': '2024-12-04_18-06', 'y_name': 'cue', 'conditioner': 'choice_R1', 'train_stratum': 1, 
    #  'interval': 'cue_entry_interval', 'timebin': 10, 'name': 'S13_Cue_cue_choice_overlap_1stsquare', 'w': None},
]


    # {'session_id': '2024-12-04_18-06', 'y_name': 'choice_R1', 'conditioner': 'cue',
    #  'train_stratum': 1, 'interval': 'R1_entry_interval', 'timebin': 9 + 4 + 3,
    #  'name': 'S13_choice_2ndsquare_maxAligned', 'w': None},

svm_x_modality = 'mPFC'
svm_y_predicted_name = ['cue']  # or 'cue' / 'choice_R1' / 'choice_R2' / a list of these
# svm_which_intervals = ['cue_entry_interval', ]
svm_which_intervals = ['cue_entry_interval', 'R1_entry_interval']
svm_test_session_names = session_names[8:16]  # one session, just to test the plot works

SVM_ANALYTIC = 'SVML1CueOutcomeChoicePredCCGP'
SVM_ANALYTIC = 'SVML1CueOutcomeChoicePred2'
svm_results_test = analytics.get_analytics(SVM_ANALYTIC, session_names=svm_test_session_names)

# Deliberately the STALE t0 table: it is the one these two older analytics were fit
# against (see plotlib.session_data.ANALYTIC_PAIRS). The current-geometry fit
# SVML1CueOutcomeChoicePred2 has no CCGP rows, so the CCGP figures below stay on this
# pairing until that analytic is recomputed. check_geometry confirms the pairing holds.
t0_events_test = sd.load_t0_events(svm_test_session_names, analytic='TrialWiseT0Events40ms2')
sd.check_geometry(t0_events_test, svm_results_test)

behavior_test = analytics.get_analytics(
    'Behavior40msAligned', session_names=svm_test_session_names,
    columns=['trial_id', 'cue', 'trial_outcome', 'choice_R1', 'choice_R2',
             'to_ephys_timestamp', 'frame_position'])
behavior_test = behavior_test.reset_index().drop(columns=['paradigm_id', 'animal_id', 'entry_id'])
behavior_test.index = pd.IntervalIndex.from_arrays(
    left=behavior_test.to_ephys_timestamp - 40_000,
    right=behavior_test.to_ephys_timestamp,
    closed='right')
behavior_test.set_index('session_id', append=True, inplace=True)
behavior_test['choice_R1'] = behavior_test['choice_R1'].astype(bool)

# Sessions are resolved from what actually made it into the 3 analytics tables above
# (rather than from svm_test_session_names directly) - a session sliced in up top can
# still come back empty from analytics.get_analytics (e.g. no SVM fit yet), and indexing
# get_session_data with an s_id absent from its index is what raised the KeyError here.
plot_session_ids = sorted(
    set(svm_results_test.index.unique('session_id')) &
    set(t0_events_test.index.unique('session_id')) &
    set(behavior_test.index.unique('session_id')),
    key=lambda s_id: pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M'))

all_saved_png_paths, all_saved_svg_paths = [], []
for s_id in plot_session_ids:
    if s_id not in session_to_dt_session:
        print(f"Session {s_id} not found in session_to_dt_session mapping, skipping.")
        continue
    dt_session = session_to_dt_session[s_id]
    print(f"Session {s_id} mapped to dt_session {dt_session}.")
    png_paths, svg_paths = plot_session_analysis(
        s_id, t0_events_test, behavior_test, svm_results_test,
        x_modalities=svm_x_modality, y_predicted_names=svm_y_predicted_name,
        which_intervals=svm_which_intervals, 
        ith_to_dt_session=session_to_dt_session, 
        show_shuffle=True, 
        show_mean=False,
        show_ccgp_secondary=False,
        min_metric=0.55,
        mask_chance_band=False,
        smooth=2,
        show_trial_kinematics=False,
        show_w_comparison=False,
        # show_covariate_plot=False,
        # show_colorbar=True,
        show_ylabel=False,
        # highlight_selections=W_SELECTIONS,
        show_title=False,
        ccgp_w_train_stratum={'cue': 1, 'choice_R1': 1, }, #'choice_R2': 1},
        save_dir=os.path.join(output_dir, 'svm_analysis_cue_CCGP'),
    )
    if png_paths:
        all_saved_png_paths.append(png_paths)
        all_saved_svg_paths.append(svg_paths)
    # break

if all_saved_png_paths:
    canvas_path = os.path.join(output_dir, 'svm_analysis_cue_CCGP', 'cue_7to15.png')
    build_canvas(all_saved_png_paths, canvas_path)
    print(f"Saved combined canvas of {len(all_saved_png_paths)} sessions to {canvas_path}")

    canvas_path = os.path.join(output_dir, 'svm_analysis_cue_CCGP', 'cue_7to15.svg')
    build_canvas_svg(all_saved_svg_paths, canvas_path)
    print(f"Saved combined SVG canvas of {len(all_saved_svg_paths)} sessions to {canvas_path}")
    
# %%
# %%
# Cross-session W comparison - same (x_modality, predict_y_name) fit, compared across
# the session sequence resolved above, instead of across columns within one session (see
# plot_session_analysis's own show_w_comparison for that). Now lives in
# svm_plotting_lib.py (get_session_sequence_Ws/plot_session_sequence_w_comparison) rather
# than being redefined here - this cell just imports and calls it, mirroring
# svm_analysis_2026.py's own "Cross-session weight comparison" cell, but reusing this
# notebook's already-fetched svm_results_test/plot_session_ids/session_to_dt_session.
from svm_plotting_lib import plot_session_sequence_w_comparison

sequence_save_dir = os.path.join(output_dir, 'svm_analysis_cue_CCGP', )
os.makedirs(sequence_save_dir, exist_ok=True)

for y_predicted_name in svm_y_predicted_name:  # single predict_y_name per plot; loop covers each
    save_path = os.path.join(sequence_save_dir, f'session_sequence_{y_predicted_name}_{svm_x_modality}.png')
    save_path_svg = os.path.join(sequence_save_dir, f'session_sequence_{y_predicted_name}_{svm_x_modality}.svg')
    plot_session_sequence_w_comparison(
        plot_session_ids, svm_results_test, t0_events_test, svm_x_modality, y_predicted_name,
        svm_which_intervals,
        mask_chance_band=True,
        min_metric=0.55,
        smooth=2,
        ith_to_dt_session=session_to_dt_session,
        ccgp_w_train_stratum={'cue': 1, 'choice_R1': 1, }, #'choice_R2': 1},
        reference_session_id=next(sid for sid, idx in session_to_dt_session.items() if idx == 13),
        highlight_selections=W_SELECTIONS,
        save_path=save_path, save_path_svg=save_path_svg)

# saved at
print(f"Saved cross-session W comparison plots to {sequence_save_dir}")


# %%



# %%


"""Getting the W's for the hand-picked selections above, so they can be plotted in a callout box"""

from svm_plotting_lib import resolve_w_selections, plot_w_comparison

# W_SEL_ANALYTIC = 'SVML1CueOutcomeChoicePredCCGP'
W_SEL_ANALYTIC = 'SVML1CueOutcomeChoicePred2'
W_SEL_X_MODALITY = 'mPFC'
W_SEL_CCGP_STRATUM = {'cue': 1, 'choice_R1': 1, 'choice_R2': 1}

w_sel_svm_results = analytics.get_analytics(W_SEL_ANALYTIC, session_names=session_names)

selected_w_rows_df = resolve_w_selections(
    W_SELECTIONS, w_sel_svm_results, x_modality=W_SEL_X_MODALITY,
    ccgp_w_train_stratum=W_SEL_CCGP_STRATUM)

selected_w_rows_df[['session_id', 'name', 'predict_y_name', 'interval_name', 'timebin',
                    'X_modality', 'acc_aggr']]



# %%
# First W figure: each hand-picked selection's weight vector as its own thin panel,
# features (neurons) top-to-bottom on a shared y-axis, plus the region-gradient column.
# Runs straight off selected_w_rows_df from the cell above (resolve_w_selections filled
# each row's `w`), so it needs no analytic of its own. plot_W_list does not create its
# save dir - hence the makedirs, same dir the CCGP figures above write into.
from w_plotting_lib import plot_W_list

w_list_save_dir = os.path.join(output_dir, 'svm_analysis_cue_CCGP')
os.makedirs(w_list_save_dir, exist_ok=True)

plot_W_list(
    selected_w_rows_df, plot_type='stem',
    ith_to_dt_session=session_to_dt_session,  # session_id -> S<n> in the titles
    save_path=os.path.join(w_list_save_dir, 'highlighted_Ws_stem.png'),
    save_path_svg=os.path.join(w_list_save_dir, 'highlighted_Ws_stem.svg'))


# %%
# ===========================================================================
# W projection - shared setup for both options below
# ===========================================================================
# Data for the projection figures, fetched here rather than reused from the CCGP cells
# above: those narrowed both the session range (session_names[8:16]) and behavior's
# columns, and the projection needs neither restriction - its velocity row reads
# frame_raw_500msMedian, which that narrowed fetch does not include. The t0 table is
# pinned to W_SEL_ANALYTIC's own pairing (see plotlib.session_data.ANALYTIC_PAIRS), i.e.
# the same geometry the W's in selected_w_rows_df were fit under.
from w_plotting_lib import plot_w_projection_session

WPROJ_INTERVALS = ['cue_entry_interval', 'R1_entry_interval']  # free choice, any of ALL_INTERVALS
WPROJ_SPLIT_COVARIATE = None  # None -> split row 1 by each W's own predict_y_name;
                              # or name one ('cue' / 'choice_R1' / ...) to split by that
                              # instead, in which case it drops out of the grid below.
WPROJ_YLIM = (-1.3, 1.3)  # fixed across sessions on purpose - autoscaling per session
                          # would hide the very drift option 2 is looking for; None
                          # falls back to each session's own empirical range.

wproj_session_names = session_names[13:14]  # one session, just to test the plot works
wproj_t0_events = sd.load_t0_events(wproj_session_names, analytic='TrialWiseT0Events40ms2')
wproj_behavior = sd.load_behavior(wproj_session_names)

# %%



# %%

# w_row is picked once, outside the loop, and the *session*
# varies - so this shows how one hand-picked subspace holds up (or drifts) in sessions it
# was never fit on. Its own session list, deliberately defined here rather than reusing
# the CCGP cells' plot_session_ids: which sessions are worth projecting into is this
# figure's own question, and the answer is usually "more of them than were plotted above".
WPROJ_FIXED_W_NAME = None  # None -> the first selection; or a W_SELECTIONS 'name' to pin one
WPROJ_FIXED_W_NAME = 'S13_Cue_cue_choice_overlap_2ndsquare'  # None -> the first selection; or a W_SELECTIONS 'name' to pin one
# WPROJ_FIXED_W_NAME = 'S13_Cue_cue_choice_overlap_1stsquare'  # None -> the first selection; or a W_SELECTIONS 'name' to pin one

wproj_fixed_w_row = (selected_w_rows_df.iloc[0] if WPROJ_FIXED_W_NAME is None else
                     selected_w_rows_df[selected_w_rows_df['name'] == WPROJ_FIXED_W_NAME].iloc[0])

wproj_plot_session_ids = sorted(
    set(wproj_t0_events.index.unique('session_id')) & set(wproj_behavior.index.unique('session_id')),
    key=lambda s_id: pd.to_datetime(s_id, format='%Y-%m-%d_%H-%M'))
# wproj_plot_session_ids = wproj_plot_session_ids[8:16]  # narrow the sweep here

wproj_fixed_tag = (wproj_fixed_w_row['name'] if pd.notna(wproj_fixed_w_row.get('name'))
                   else f"{wproj_fixed_w_row['interval_name']}_t{wproj_fixed_w_row['timebin']}")
wproj_fixed_dir = os.path.join(output_dir, 'w_projection', f'fixed_{wproj_fixed_tag}')
os.makedirs(wproj_fixed_dir, exist_ok=True)

wproj_fixed_paths = []
for s_id in wproj_plot_session_ids:
    saved_path = plot_w_projection_session(
        s_id, wproj_fixed_w_row, wproj_t0_events, wproj_behavior, wproj_session_names,
        WPROJ_INTERVALS, split_covariate=WPROJ_SPLIT_COVARIATE, projection_ylim=WPROJ_YLIM,
        smooth=2, highlight_range=(wproj_fixed_w_row['interval_name'],
                         wproj_fixed_w_row['timebin'], wproj_fixed_w_row['timebin']),
        ith_to_dt_session=session_to_dt_session, save_dir=wproj_fixed_dir)
    wproj_fixed_paths.append(saved_path)

if wproj_fixed_paths:
    canvas_path = os.path.join(wproj_fixed_dir, 'wproj_fixed_canvas.png')
    build_canvas([[p] for p in wproj_fixed_paths], canvas_path)
    print(f"Saved combined canvas of {len(wproj_fixed_paths)} sessions to {canvas_path}")
    
# %%
# Does this W's activation track a kinematic variable rather than (or on top of) the task
# covariate it was fit to distinguish? Two views of the same question, both on the same
# wproj_fixed_w_row / wproj_* data as the projection cell above.
from w_plotting_lib import (plot_w_variance_explained, plot_w_projection_kinematics,
                            plot_session_projection_kinematics)

# Any session the wproj_* data covers - not necessarily the W's own session, same as the
# fixed-W loop above (which projects one hand-picked W into sessions it was never fit on).
kin_s_id = wproj_plot_session_ids[0]
kin_interval, kin_timebin_range = 'R1_entry_interval', (25, 32)
kin_offset = 0  # native bins; shifts *only* the kinematics side's window later, to test
                # whether kinematics N bins after the projection's window explain it
kin_color_by = wproj_fixed_w_row['predict_y_name']  # the whole-session view (2) needs a plain
                                                    # per-bin behavior column ('cue'/'choice_R1'/
                                                    # 'choice_R2'/'trial_outcome'); correct_R1/R2
                                                    # are trial-only -> set None for those

# 1) Windowed: one scatter point per (trial, bin) sample inside kin_timebin_range. Sanity
#    checks first - the purple dashed line in plot_w_variance_explained is how much of that
#    window's pooled variance the W direction itself explains (near 0 -> the decoder barely
#    moves here, so read any correlation below cautiously), and plot_w_projection_session
#    shades the same window on the actual projection trace for context.
_ = plot_w_projection_session(
    kin_s_id, wproj_fixed_w_row, wproj_t0_events, wproj_behavior, wproj_session_names,
    [kin_interval], show_covariate_plot=False, smooth=2,
    ith_to_dt_session=session_to_dt_session,
    highlight_range=(kin_interval, *kin_timebin_range))

kin_windowed_correlations = plot_w_projection_kinematics(
    kin_s_id, wproj_fixed_w_row, wproj_t0_events, wproj_behavior, wproj_session_names,
    kin_interval, kin_timebin_range, kinematic_offset=kin_offset, color_by=kin_color_by,
    ith_to_dt_session=session_to_dt_session)

# 2) Whole session: the same scatter over *every* native 40ms bin of the continuous
#    recording, inter-trial time included - no t0_events/interval/timebin involvement at
#    all. Answers whether a correlation from (1) is specific to the task window or just
#    holds throughout the session. max_points subsamples for rendering only (tens of
#    thousands of bins per session); Pearson r is computed on the full aligned set first.
kin_session_correlations = plot_session_projection_kinematics(
    kin_s_id, wproj_fixed_w_row, wproj_behavior, wproj_session_names,
    color_by=kin_color_by, max_points=20000, ith_to_dt_session=session_to_dt_session)

kin_windowed_correlations, kin_session_correlations


# %%
# 


# %%
# Generalized version of the double_stop_performance cell above: same 2-panel (R1 top,
# R2 bottom) layout, but plots choice_R1_correct/choice_R2_correct (cue-dependent
# correctness, computed in modality_transformations.calc_trialwise_metrics) instead of
# decoding trial_outcome's %5///10 digits - lm4_plots.py and plot_StayPerformance.py
# disagree on which digit is R1 vs R2, so trial_outcome's mapping can't be trusted here.
# split_by_cue=True reproduces the old res_cue1/res_cue2 behavior (one line per cue,
# both panels); split_by_cue=False collapses each panel to a single line across all cues.
CUE_PERF_COLORS = {None: 'darkgrey', 1: 'orange', 2: 'purple'}

def plot_stop_correctness(session_names, split_by_cue=False, session_range=None,
                           output_dir=None, fname=None):
    # Always fetch fresh (BehaviorTrialwise loads fast) instead of taking a pre-loaded
    # trialw_beh - reusing whatever happened to be in scope caused stale/mismatched
    # session ranges depending on cell execution order.
    trialw_beh = analytics.get_analytics('BehaviorTrialwise', session_names=session_names)
    trialw_beh.index = trialw_beh.index.droplevel(('animal_id', 'paradigm_id', 'entry_id'))
    trialw_beh.index = trialw_beh.index.map(
        lambda x: session_to_dt_session[x[:16]] if x[:16] in session_to_dt_session else x)

    session_ids = [s_id for s_id in trialw_beh.index.unique('session_id')
                   if session_range is None or (session_range[0] <= s_id <= session_range[1])]
    if not session_ids:
        available = sorted(trialw_beh.index.unique('session_id'))
        raise ValueError(f'session_range={session_range} has no overlap with trialw_beh '
                          f'(available session_ids: {available}) - check that trialw_beh '
                          f'was loaded/reloaded with the sessions you expect.')
    cue_vals = [1, 2] if split_by_cue else [None]

    plt.close('all')
    fig, axes = plt.subplots(figsize=(3.5, 5), nrows=2, sharex=True)

    for ax, location in zip(axes, ('R1', 'R2')):
        correct_col = f'choice_{location}_correct'
        for i, cue_val in enumerate(cue_vals):
            color = CUE_PERF_COLORS[cue_val]
            x, y = [], []
            for s_id in session_ids:
                s_trialw_beh = trialw_beh.loc[s_id]
                if cue_val is not None:
                    s_trialw_beh = s_trialw_beh[s_trialw_beh['cue'] == cue_val]
                if len(s_trialw_beh) == 0:
                    continue
                x.append(s_id)
                y.append((s_trialw_beh[correct_col] == 1).mean())
            x, y = np.array(x), np.array(y)

            label = f'cue {cue_val}' if cue_val is not None else f'{location} correct'
            ax.scatter(x, y, color=color, label=label if location == 'R1' else None)

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            ax.plot(x, intercept + slope * x, '--', color=color, linewidth=2)
            ax.text(0.95, 0.9 - i * 0.15, f'p={p_value:.3f}', color=color, fontsize=9,
                    transform=ax.transAxes, va='top', ha='right')

        ax.set_title(f'{location} correct' + (', by cue' if split_by_cue else ''))

    axes[1].set_xlabel('Session ID')
    axes[0].set_ylabel('Proportion correct')
    axes[1].set_ylabel('Proportion correct')
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(session_ids)
    axes[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axes[0].grid(axis='x', alpha=0.5, linestyle='--')
    axes[1].grid(axis='x', alpha=0.5, linestyle='--')
    [ax.spines[['top', 'right']].set_visible(False) for ax in axes]
    if split_by_cue:
        axes[0].legend()

    fig.tight_layout(h_pad=2.5)
    plt.show()

    if output_dir and fname:
        fig.savefig(os.path.join(output_dir, fname), format='svg', dpi=300)

    return fig


fig = plot_stop_correctness(session_names[:], split_by_cue=True, session_range=(8, 15),
                            #  output_dir=output_dir, fname='stop_correctness_by_cue_S8-15.svg')
                             output_dir=output_dir, fname='stop_correctness_S8-15.svg')


# %%
# ===========================================================================
# W_SELECTIONS -> the actual W's. Selection only, no figure.
# ===========================================================================
# The counterpart to passing W_SELECTIONS (defined once, way above) to the plots: there it
# only draws a callout box, here it resolves to data. resolve_w_selections fills each
# selection's `w` in place - so `W_SELECTIONS[0]['w']` is the vector after this cell runs -
# and returns the full matched rows as a frame, which is what the W figures downstream want
# as their `w_row`.
#
# Self-contained on purpose: loads its own svm_results rather than reusing whichever SVM
# cell above happened to run last. The two knobs below MUST match what the figures were
# drawn with, or a selection resolves to a different W than the box was drawn around:
#   - W_SEL_ANALYTIC: a selection pinning conditioner/train_stratum can only ever resolve
#     against a CCGP analytic; SVML1CueOutcomeChoicePred2 has no CCGP rows at all.
#   - W_SEL_CCGP_STRATUM: same election as the plots (see the S14/S15 note above) -
#     without it a selection that doesn't pin train_stratum itself matches both CCGP
#     directions and resolves to nothing.
from svm_plotting_lib import resolve_w_selections, plot_w_comparison

W_SEL_ANALYTIC = 'SVML1CueOutcomeChoicePredCCGP'
W_SEL_X_MODALITY = 'mPFC'
W_SEL_CCGP_STRATUM = {'cue': 1, 'choice_R1': 1, 'choice_R2': 1}

w_sel_svm_results = analytics.get_analytics(W_SEL_ANALYTIC, session_names=session_names)

selected_w_rows_df = resolve_w_selections(
    W_SELECTIONS, w_sel_svm_results, x_modality=W_SEL_X_MODALITY,
    ccgp_w_train_stratum=W_SEL_CCGP_STRATUM)

selected_w_rows_df[['session_id', 'name', 'predict_y_name', 'interval_name', 'timebin',
                    'X_modality', 'acc_aggr']]



# %%
# %%

# %%

# %%

# %%

# %%