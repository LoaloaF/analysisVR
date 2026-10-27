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
s_id = 15
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
fig.show()

fname = os.path.join(output_dir, f'session{s_id}_kinematics_choiceR1_BallForward.svg')
fig.write_image(fname)
fig.write_image(fname.replace('.svg', '.png'), scale=2)


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

# Test plot for svm_plotting_lib.plot_session_analysis (extracted from svm_analysis_2026.py
# so it can be imported here without re-running that notebook's own top-level cells - see
# svm_plotting_lib.py's module docstring). Session/target/interval selection is local to
# this cell and independent of the rest of this notebook.

from svm_plotting_lib import plot_session_analysis, build_canvas, build_canvas_svg

svm_x_modality = 'mPFC'
svm_y_predicted_name = ['choice_R1']  # or 'cue' / 'choice_R1' / 'choice_R2' / a list of these
# svm_which_intervals = ['cue_entry_interval', 'R1_entry_interval']
svm_which_intervals = [ 'R1_entry_interval']
svm_test_session_names = session_names[:16]  # one session, just to test the plot works

# svm_results_test = analytics.get_analytics('SVML1CueOutcomeChoicePred',
svm_results_test = analytics.get_analytics('SVML1CueOutcomeChoicePredCCGP',
                                            session_names=svm_test_session_names)

t0_events_test = analytics.get_analytics('TrialWiseT0Events40ms', session_names=svm_test_session_names)
t0_events_test.index = t0_events_test.index.droplevel(['entry_id', 'paradigm_id', 'animal_id'])
t0_events_test.set_index('trial_id', append=True, inplace=True)

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
    png_paths, svg_paths, highlighted_rows = plot_session_analysis(
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
        show_covariate_plot=True,
        # show_colorbar=True,
        show_ylabel=False,
        show_title=False,
        show_ccgp_secondary=True,
        # ccgp_w_train_stratum={'cue': 1, 'choice_R1': 1, }, #'choice_R2': 1},
        save_dir=os.path.join(output_dir, 'svm_analysis_choiceR1'),
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

sequence_highlighted_rows = []
for y_predicted_name in svm_y_predicted_name:  # single predict_y_name per plot; loop covers each
    save_path = os.path.join(sequence_save_dir, f'session_sequence_{y_predicted_name}_{svm_x_modality}.png')
    save_path_svg = os.path.join(sequence_save_dir, f'session_sequence_{y_predicted_name}_{svm_x_modality}.svg')
    highlighted_rows = plot_session_sequence_w_comparison(
        plot_session_ids, svm_results_test, svm_x_modality, y_predicted_name, svm_which_intervals,
        mask_chance_band=True, min_metric=0.55, ith_to_dt_session=session_to_dt_session,
        reference_session_id=next(sid for sid, idx in session_to_dt_session.items() if idx == 7),
        save_path=save_path, save_path_svg=save_path_svg)
    sequence_highlighted_rows.append(highlighted_rows)

sequence_highlighted_rows_df = pd.concat(sequence_highlighted_rows) if sequence_highlighted_rows else pd.DataFrame()

# saved at
print(f"Saved cross-session W comparison plots to {sequence_save_dir}")



# %%


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

svm_x_modality = 'mPFC'
svm_y_predicted_name = ['cue']  # or 'cue' / 'choice_R1' / 'choice_R2' / a list of these
# svm_which_intervals = ['cue_entry_interval', ]
svm_which_intervals = ['cue_entry_interval', 'R1_entry_interval']
svm_test_session_names = session_names[8:16]  # one session, just to test the plot works

svm_results_test = analytics.get_analytics('SVML1CueOutcomeChoicePredCCGP',
                                            session_names=svm_test_session_names)

t0_events_test = analytics.get_analytics('TrialWiseT0Events40ms', session_names=svm_test_session_names)
t0_events_test.index = t0_events_test.index.droplevel(['entry_id', 'paradigm_id', 'animal_id'])
t0_events_test.set_index('trial_id', append=True, inplace=True)

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
    png_paths, svg_paths, highlighted_rows = plot_session_analysis(
        s_id, t0_events_test, behavior_test, svm_results_test,
        x_modalities=svm_x_modality, y_predicted_names=svm_y_predicted_name,
        which_intervals=svm_which_intervals, 
        ith_to_dt_session=session_to_dt_session, 
        show_shuffle=True, 
        show_mean=False,
        show_ccgp_secondary=True,
        min_metric=0.55,
        # mask_chance_band=False,
        smooth=2,
        show_trial_kinematics=False,
        show_w_comparison=False,
        show_covariate_plot=True,
        # show_colorbar=True,
        show_ylabel=False,
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

sequence_highlighted_rows = []
for y_predicted_name in svm_y_predicted_name:  # single predict_y_name per plot; loop covers each
    save_path = os.path.join(sequence_save_dir, f'session_sequence_{y_predicted_name}_{svm_x_modality}.png')
    save_path_svg = os.path.join(sequence_save_dir, f'session_sequence_{y_predicted_name}_{svm_x_modality}.svg')
    highlighted_rows = plot_session_sequence_w_comparison(
        plot_session_ids, svm_results_test, svm_x_modality, y_predicted_name, svm_which_intervals,
        mask_chance_band=True, 
        min_metric=0.55, 
        smooth=2,
        ith_to_dt_session=session_to_dt_session,
        ccgp_w_train_stratum={'cue': 1, 'choice_R1': 1, }, #'choice_R2': 1},
        # reference_session_id=next(sid for sid, idx in session_to_dt_session.items() if idx == 13),
        save_path=save_path, save_path_svg=save_path_svg)
    sequence_highlighted_rows.append(highlighted_rows)

sequence_highlighted_rows_df = pd.concat(sequence_highlighted_rows) if sequence_highlighted_rows else pd.DataFrame()

# saved at
print(f"Saved cross-session W comparison plots to {sequence_save_dir}")

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


# %%

# %%

# %%

# %%

# %%