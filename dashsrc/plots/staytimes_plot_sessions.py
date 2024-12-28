import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from .plot_constants import *
from .plot_utils import make_discr_trial_cmap

def _draw_success_rate(fig, all_data):
        all_data['success'] = all_data.trial_outcome >= 1
        # sucessrate
        success = all_data.groupby('session_id')['success'].sum() / all_data.groupby('session_id')['success'].count()
        fig.add_trace(go.Scatter(
            x=np.arange(len(all_data.index.unique(level='session_id'))),
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
            x=np.arange(len(all_data.index.unique(level='session_id'))),
            y=1-success,
            mode='lines',
            line=dict(color=OUTCOME_COL_MAP[0]),  # Red color for the line
            fillcolor=OUTCOME_COL_MAP[0],
            name='Failure',
            showlegend=True,
            stackgroup='one'
        ), row=2, col=1)  
    
def _draw_staytime_distr(fig, all_data):
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
            marker_color='rgba(0,0,0,.1)',
            name='Baseline',
            legendgroup='Baseline',
            showlegend=i==0,
            width=.5,
        ))

        # draw a bocplot for staytimes in correct location
        fig.add_trace(go.Scatter(
            x=np.linspace(i-.15, i+.15, len(data)),
            y=data.staytime_correct_r,
            mode='markers',
            legendgroup='Rewardzone',
            name='Rewardzone',
            marker=dict(color=data.outcome_color, symbol='circle', size=8),
            showlegend=i==0,
        ))
        
        # Threshold
        fig.add_trace(go.Scatter(
            x=np.linspace(i-.2, i+.2, len(data)),
            y=data.stay_time/2,
            line=dict(color='rgba(0,0,0,.8)', width=3), # green
            mode='lines',
            legendgroup='threshold',
            showlegend=i==0,
            name='Staytime\nThreshold',
            zorder=20
        ))
    return 
            
def _configure_axis(fig, all_data, metric_max):
    fig.update_xaxes(
        tickvals=np.arange(all_data.index.unique("session_id").size),
        ticks='outside',
        zeroline=False,
        gridcolor='rgba(128,128,128,0.4)',  # Customize grid color
        title_text='Session ID',
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
        title_text='Succsess %',
    )
    
    # make backgorund white
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=20, t=50, b=50)  # Adjust margins as needed
    )    
    return fig

def render_plot(all_data, metric_max, sort_by=['session_id']):
    print(all_data)
    print("================")

    # Data transformations    
    # sort_by = ['session_id', 'cue', 'staytime_correct_r']
    # sort_by = ['session_id', 'cue']
    # if sort_by != ['session_id']: # already sorted anyway, mixes things up
    #     all_data = all_data.sort_values(by=sort_by)
    
    all_data.loc[:, ['staytime_correct_r', 'staytime_incorrect_r']] /= 1_000_000  # us to s
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
    all_data['outcome_color'] = all_data['trial_outcome'].map(cmap_transparent)
    # all_data['cue_color'] = all_data['cue'].map(CUE_COL_MAP)
    
    
    _draw_staytime_distr(fig, all_data, )
    # _draw_treshold_and_cue_indicator(fig, all_data, sort_by, include_threshold=True)
    
    _draw_success_rate(fig, all_data)
    
    # _draw_treshold_and_cue_indicator(fig, all_data, sort_by, include_threshold=True)
    # _draw_single_trials(fig, all_data)

    fig = _configure_axis(fig, all_data, metric_max)
    return fig