import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashsrc.components.dashvis_constants import *
# from .plot_utils import make_discr_trial_cmap

_LEGACY_COL_ALIASES = {
    'staytime_correct_r': 'correctR_staytime',
    'staytime_incorrect_r': 'incorrectR_staytime',
    'staytime_reward1': 'trackzone_reward1_staytime',
    'staytime_reward2': 'trackzone_reward2_staytime',
    'stay_time': 'velocity_threshold_at_R1',
    'baseline_velocity': 'trackzone_before_reward1_avg_velocity',
}


def _duration_s_from_ephys(all_data, enter_col, exit_col):
    if enter_col not in all_data.columns or exit_col not in all_data.columns:
        return None
    enter_t = pd.to_numeric(all_data[enter_col], errors='coerce')
    exit_t = pd.to_numeric(all_data[exit_col], errors='coerce')
    duration_s = (exit_t - enter_t) / 1_000_000
    duration_s[duration_s < 0] = np.nan
    return duration_s


def _ensure_legacy_columns(all_data):
    # 1) Direct aliasing from renamed columns.
    for legacy_col, new_col in _LEGACY_COL_ALIASES.items():
        if legacy_col not in all_data.columns and new_col in all_data.columns:
            all_data[legacy_col] = all_data[new_col]

    # 2) New schema fallback: derive reward-zone staytimes from zone enter/exit timestamps.
    if 'staytime_reward1' not in all_data.columns:
        reward1_st = _duration_s_from_ephys(all_data, 'reward1Zone_enter_ephys_t', 'reward1Zone_exit_ephys_t')
        if reward1_st is not None:
            all_data['staytime_reward1'] = reward1_st
    if 'staytime_reward2' not in all_data.columns:
        reward2_st = _duration_s_from_ephys(all_data, 'reward2Zone_enter_ephys_t', 'reward2Zone_exit_ephys_t')
        if reward2_st is not None:
            all_data['staytime_reward2'] = reward2_st

    # 3) Derive correct/incorrect staytimes from cue and optional flip mapping.
    has_reward_cols = all(c in all_data.columns for c in ('staytime_reward1', 'staytime_reward2'))
    if has_reward_cols and (
        'staytime_correct_r' not in all_data.columns or
        'staytime_incorrect_r' not in all_data.columns
    ):
        cue = pd.to_numeric(all_data['cue'], errors='coerce') if 'cue' in all_data.columns else pd.Series(np.nan, index=all_data.index)
        if 'flip_Cue1R1_Cue2R2' in all_data.columns:
            flip = all_data['flip_Cue1R1_Cue2R2'].astype('boolean')
            cue1_is_r1 = ~flip.fillna(False)
        else:
            cue1_is_r1 = pd.Series(True, index=all_data.index)

        correct_is_r1 = ((cue == 1) & cue1_is_r1) | ((cue == 2) & (~cue1_is_r1))
        stay_r1 = pd.to_numeric(all_data['staytime_reward1'], errors='coerce')
        stay_r2 = pd.to_numeric(all_data['staytime_reward2'], errors='coerce')
        all_data['staytime_correct_r'] = np.where(correct_is_r1, stay_r1, stay_r2)
        all_data['staytime_incorrect_r'] = np.where(correct_is_r1, stay_r2, stay_r1)

    # Keep plotting code resilient across schema versions.
    for col in (
        'staytime_correct_r', 'staytime_incorrect_r',
        'staytime_reward1', 'staytime_reward2',
        'stay_time', 'baseline_velocity',
    ):
        if col not in all_data.columns:
            all_data[col] = np.nan
    return all_data

def _draw_success_rate(fig, all_data, double_reward_filter=None):
    print("INNNN")
    print(double_reward_filter)
    if double_reward_filter == [] or double_reward_filter is None or sorted(double_reward_filter) == ['Early R', 'Late R']:
        print("in double")
        all_data['success'] = all_data.trial_outcome >= 1
    elif "Early R" in double_reward_filter:
        print("in early r")
        all_data['success'] = all_data.trial_outcome_r1 >= 1
    elif "Late R" in double_reward_filter:
        print("in late r")
        all_data['success'] = all_data.trial_outcome_r2 >= 1
    print(all_data['success'])
    
    # sucessrate
    session_ids = all_data.index.unique(level='session_id')
    success = all_data.groupby('session_id')['success'].sum() / all_data.groupby('session_id')['success'].count()
    success = success.reindex(session_ids)
    print(success)
    fig.add_trace(go.Scatter(
        x=np.arange(len(session_ids)),
        y=success,
        mode='lines',
        fill='tozeroy',  # Fill to the x-axis
        line=dict(color=OUTCOME_COL_MAP[1]),
        fillcolor=OUTCOME_COL_MAP[1],
        name='Success',
        showlegend=True,
        stackgroup='one'
    ), row=2, col=1)  
    
    fig.add_trace(go.Scatter(
        x=np.arange(len(session_ids)),
        y=1-success,
        mode='lines',
        line=dict(color=OUTCOME_COL_MAP[0]),  # Red color for the line
        fillcolor=OUTCOME_COL_MAP[0],
        name='Failure',
        showlegend=True,
        stackgroup='one'
    ), row=2, col=1)
        
        
def _draw_staytime_average(fig, all_data):
    # iterate sessions
    averages = []
    for i,session_id in enumerate(all_data.index.unique(level='session_id')):
        data = all_data.loc[pd.IndexSlice[:,:,session_id:session_id,:]]
        
        avg_staytime = (40 /data.loc[:, 'baseline_velocity']).mean()
        avg_reward_staytimes = data.loc[:, ['staytime_correct_r', 'staytime_incorrect_r']].mean().mean()
        averages.append([avg_staytime, avg_reward_staytimes])
    averages = np.array(averages)
    
    # draw two traces
    fig.add_trace(go.Scatter(
        x=np.arange(len(all_data.index.unique(level='session_id'))),
        y=averages[:,0],
        mode='markers+lines',
        marker=dict(color='rgba(0,0,0,.1)'),  # Red color for the line
        line=dict(color='rgba(0,0,0,.1)', width=8),
        name='Baseline',
        showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.arange(len(all_data.index.unique(level='session_id'))),
        y=averages[:,1],
        mode='markers+lines',
        marker=dict(color='rgba(0,0,0,.6)', size=2),
        line=dict(color='rgba(0,0,0,.6)', width=2),
        name='Rewardzone',
        showlegend=True,
    ), row=1, col=1)
    

    
def _draw_staytime_distr(fig, all_data, double_reward_filter=None):
    # iterate sessions
    for i,session_id in enumerate(all_data.index.unique(level='session_id')):
        data = all_data.loc[pd.IndexSlice[:,:,session_id:session_id,:]]
        
        # draw horizontal bar with width equal to the std 
        avg_staytime = (40 /data.loc[:, 'baseline_velocity']).mean()
        std_staytime = (40 /data.loc[:, 'baseline_velocity']).std()
        fig.add_trace(go.Bar(
            x=[i],
            y=[std_staytime],
            base=[avg_staytime-std_staytime/2],
            marker_color='rgba(0,0,0,.15)',
            name='Baseline',
            legendgroup='Baseline',
            showlegend=i==0,
            width=.7,
        ))

        if double_reward_filter == [] or double_reward_filter is None:
            print("in double")
            # draw a bocplot for staytimes in correct location
            fig.add_trace(go.Scatter(
                x=np.linspace(i-.15, i+.15, len(data)),
                y=data.staytime_correct_r,
                mode='markers',
                legendgroup='Rewardzone',
                name='Rewardzone',
                marker_color=data.outcome_color,
                marker=dict(symbol='circle', size=3,),
                showlegend=i==0,
            ))
            
            
        elif "Early R" in double_reward_filter:
            print("in early r")
            fig.add_trace(go.Scatter(
                x=np.linspace(i-.15, i+.15, len(data)),
                y=data.staytime_reward1,
                mode='markers',
                legendgroup='Rewardzone',
                name='Rewardzone',
                # marker_line_width=1,
                marker_color=data.outcome_color_r1,
                marker=dict(symbol='circle', size=3),
                showlegend=i==0,
            ))
        if "Late R" in double_reward_filter:
            print("in late r")
            fig.add_trace(go.Scatter(
                x=np.linspace(i-.15, i+.15, len(data)),
                y=data.staytime_reward2,
                mode='markers',
                legendgroup='Rewardzone',
                name='Rewardzone',
                # marker_line_width=1,
                marker_line_color=REWARD_LOCATION_COL_MAP[2],
                marker_color=data.outcome_color_r2, 
                marker=dict(symbol='circle', size=3),
                showlegend=i==0,
            ))
        
        # # draw a bocplot for staytimes in correct location
        # fig.add_trace(go.Scatter(
        #     x=np.linspace(i-.15, i+.15, len(data)),
        #     y=data.staytime_correct_r,
        #     mode='markers',
        #     legendgroup='Rewardzone',
        #     name='Rewardzone',
        #     marker=dict(marker_line_color=data.outcome_color, marker_color="lightskyblue",
        #         color=data.outcome_color, symbol='circle', size=3, 
        #                 edgecolor='black'),
        #     showlegend=i==0,
        # ))
        
        # Threshold
        fig.add_trace(go.Scatter(
            x=np.linspace(i-.4, i+.4, len(data)),
            y=data.stay_time/1.15,
            # y=data.stay_time/2,
            line=dict(color='rgba(0,0,0,.8)', width=1), # green
            mode='lines',
            legendgroup='threshold',
            visible='legendonly',
            showlegend=i==0,
            name='Staytime\nThreshold',
            zorder=20
        ))
    return 
            
def _configure_axis(fig, all_data, metric_max):
    session_ids = all_data.index.unique("session_id")
    session_labels = []
    for s_id in session_ids:
        s_id = str(s_id)
        if '_' in s_id:
            date_part, time_part = s_id.split('_', 1)
            session_labels.append(f"{date_part}<br>{time_part}")
        else:
            session_labels.append(s_id)
    fig.update_xaxes(
        tickvals=np.arange(session_ids.size),
        ticktext=session_labels,
        ticks='outside',
        zeroline=False,
        # rotate ticklables
        tickangle=-45,
        gridcolor='rgba(128,128,128,0.4)',  # Customize grid color
        title_text='Session Date',
        row=2, col=1
    )
    fig.update_yaxes(
        row=1, col=1,
        range=[-0.03, metric_max],
        showgrid=True,
        zeroline=False,
        gridcolor='rgba(128,128,128,0.2)',  # Customize grid color
        title_text='Staytime [s]',
    )
    fig.update_yaxes(
        row=2, col=1,
        range=[0, 1],
        tickvals=[0, 1],
        ticktext=['0%', '100%'],
        title_text='Succsess',
    )
    
    # make backgorund white
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=20, t=50, b=50)  # Adjust margins as needed
    )    
    return fig

def render_plot(all_data, metric_max, var_vis, double_reward_filter=None, sort_by=['session_id'], ):
    print(all_data)
    print(all_data.columns)
    print("================")
    all_data = _ensure_legacy_columns(all_data)

    # Data transformations    
    # sort_by = ['session_id', 'cue', 'staytime_correct_r']
    # sort_by = ['session_id', 'cue']
    # if sort_by != ['session_id']: # already sorted anyway, mixes things up
    #     all_data = all_data.sort_values(by=sort_by)
    
    # Convert legacy microseconds data to seconds; skip columns that are already in seconds.
    for col in ['staytime_correct_r', 'staytime_incorrect_r', 'staytime_reward1', 'staytime_reward2']:
        numeric_col = pd.to_numeric(all_data[col], errors='coerce')
        if numeric_col.dropna().abs().median() > 1e4:
            all_data[col] = numeric_col / 1_000_000
    # reset trial_id to set of sessions, each is unique
    all_data = all_data.droplevel('entry_id')
    all_data['entry_id'] = np.arange(len(all_data))
    all_data.set_index('entry_id', append=True, inplace=True)
    
    # do the plotting
    fig = make_subplots(rows=2, cols=1, row_heights=(.9,.1), shared_xaxes=True,
                        vertical_spacing=0.03)
    
    # add outcome color and  cue color columns for later
    cmap_transparent = {k: v.replace("rgb","rgba")[:-1]+f', {MULTI_MARKERS_ALPHA})' 
                        for k,v in OUTCOME_COL_MAP.items()}
    
    # TODO handle this properly
    # deal with double outcomes
    
    # deal with double outcomes
    all_data['trial_outcome_r1'] = all_data['trial_outcome'] // 10 
    print(['trial_outcome_r1'])
    print(all_data['trial_outcome_r1'])
    all_data['trial_outcome_r2'] = all_data['trial_outcome'] % 5
    print(['trial_outcome_r2'])
    print(all_data['trial_outcome_r2'])
    
    outcome_r1 = all_data['trial_outcome'] // 10 
    outcome_r2 = all_data['trial_outcome'] % 10

    all_data['outcome_color_r1'] = outcome_r1.map(cmap_transparent)
    all_data['outcome_color_r2'] = outcome_r2.map(cmap_transparent)

    outcome_r1.loc[:] = np.max([outcome_r1, outcome_r2], axis=0)
    all_data['outcome_color'] = outcome_r1.map(cmap_transparent)
    
    # all_data['outcome_color'] = all_data['trial_outcome'].map(cmap_transparent)
    # all_data['cue_color'] = all_data['cue'].map(CUE_COL_MAP)
    
    if var_vis == 'Average':
        # compare just baseline to staytimes in rewardzone averages
        _draw_staytime_average(fig, all_data, )
    elif var_vis == 'Distribution':    
        _draw_staytime_distr(fig, all_data, double_reward_filter)
        # _draw_treshold_and_cue_indicator(fig, all_data, sort_by, include_threshold=True)
    
    _draw_success_rate(fig, all_data, double_reward_filter)
    
    # _draw_treshold_and_cue_indicator(fig, all_data, sort_by, include_threshold=True)
    # _draw_single_trials(fig, all_data)

    fig = _configure_axis(fig, all_data, metric_max)
    return fig
