import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashsrc.components.dashvis_constants import *
# from .plot_utils import make_discr_trial_cmap

def _draw_single_trials(fig, all_data):
    # draw single staytimes scatter plots
    for cue, cue_data in all_data.groupby('cue'):
        if cue == 1:
            marker_correct_r = 'circle-dot'
            marker_incorrect_r = 'star-triangle-up'
        else:
            marker_correct_r = 'star-triangle-up'
            marker_incorrect_r = 'circle-dot'

        # Correct location
        fig.add_trace(go.Scatter(
            x=cue_data.x_indices,
            y=cue_data.staytime_correct_r,
            marker=dict(color=cue_data.outcome_color, symbol=marker_correct_r, size=15),
            mode='markers',
        ))
        # Incorrect location
        fig.add_trace(go.Scatter(
            x=cue_data.x_indices,
            y=cue_data.staytime_incorrect_r,
            marker=dict(color='rgba(128,128,128,.6)', symbol=marker_incorrect_r, size=10),
            mode='markers',
        ))
    

def _draw_treshold_and_cue_indicator(fig, all_data, sort_by, include_threshold=False):
    # iterate sessions
    for i,session_id in enumerate(all_data.index.unique(level='session_id')):
        data = all_data.loc[pd.IndexSlice[:,:,session_id,:]]
        
        # If cue sorted, just draw two long lines
        if 'cue' in sort_by:
            grouped_data = [group for _, group in data.groupby('cue')]
        # If not cue sorted, draw a single line for each trial
        else:
            grouped_data = [data.iloc[i:i+2] for i in range(len(data))]

        for j, d in enumerate(grouped_data):
            if include_threshold:
                # Threshold
                fig.add_trace(go.Scatter(
                    x=d.x_indices,
                    y=d.stay_time / 2,
                    line=dict(color='rgb(28,28,28)', width=4),
                    mode='lines',
                    legendgroup='threshold',
                    showlegend=True if i == j == 0 else False,
                    name='Threshold',
                    zorder=20
                ))
            # Cue indication
            fig.add_trace(go.Scatter(
                x=d.x_indices,
                y=[.2] * len(d),
                mode='markers',
                marker=dict(color=d.cue_color.iloc[0], size=5, symbol='square'),
                legendgroup='cue',
                name='Trial type',
                showlegend=True if i == j == 0 else False,
                
            ))
    return fig

def _draw_violin_plots(fig, all_data, metric_max, indicate_outcome):
    # iterate sessions
    showlegend = True
    for i,session_id in enumerate(all_data.index.unique(level='session_id')):
        data = all_data.loc[pd.IndexSlice[:,:,session_id,:]]
        
        for j, (cue, d) in enumerate(data.groupby('cue')):
            if cue not in (1,2):
                continue
            d.loc[:, ['staytime_correct_r', 'staytime_incorrect_r']] = d.loc[:, ['staytime_correct_r', 'staytime_incorrect_r']].clip(0, metric_max) 
            
            # cue == 1 case, correct == early location
            cor_color = EARLY_REWARD_LOCATION_COLOR
            incor_color = LATE_REWARD_LOCATION_COLOR
            cor_side='negative'
            incor_side='positive'

            # for cue == 2, correct == late location, flip here
            if cue == 2:
                cor_color, incor_color = incor_color, cor_color
                cor_side, incor_side = incor_side, cor_side
            
            violin_x = d.x_indices.iloc[len(d)//2]
            cor_side_point_pos_x = violin_x +8 if cor_side == 'positive' else violin_x -8 
            incor_side_point_pos_x = violin_x +8 if incor_side == 'positive' else violin_x -8
            
            violin_kwargs = dict(bandwidth=.15, x=[violin_x] *len(d),
                                 spanmode='manual', points=False,
                                 span=[0.25, metric_max], width=len(d),
                                 line_color=d.cue_color.iloc[0])
            
            #correct location staytimes violin plot
            fig.add_trace(go.Violin(
                y=d.staytime_correct_r,
                pointpos=0.5 if cor_side == 'positive' else -0.5,
                side=cor_side,
                fillcolor=cor_color,
                name='Correct location distr.',
                legendgroup='Correct location distr.',
                showlegend=showlegend,
                **violin_kwargs
            ))

            # single points for each trial
            if not indicate_outcome:
                marker = dict(color='rgba(0,0,0,0)', symbol='circle', size=4, 
                              line=dict(color=cor_color, width=3))
            else:
                marker = dict(color=d.outcome_color, symbol='circle', size=6)
                
            fig.add_trace(go.Scatter(
                x=[cor_side_point_pos_x] * len(d),
                y=d.staytime_correct_r,
                mode='markers',
                showlegend=showlegend,
                name='Correct location points',
                legendgroup='Correct location points',
                marker=marker
            ))
            
            
            # incorrect location staytimes violin plot
            fig.add_trace(go.Violin(
                y=d.staytime_incorrect_r,
                pointpos=0.5 if incor_side == 'positive' else -0.5,
                side=incor_side,
                fillcolor=incor_color,
                name='Incorrect location distr.',
                legendgroup='Incorrect location distr.',
                showlegend=showlegend,
                **violin_kwargs
            ))
            # single points for each trial
            fig.add_trace(go.Scatter(
                x=[incor_side_point_pos_x] * len(d),
                y=d.staytime_incorrect_r,
                mode='markers',
                marker=dict(color="rgba(0,0,0,0)", symbol='circle', size=4, 
                            line=dict(color=incor_color, width=3)),
                name='Incorrect location points',
                legendgroup='Incorrect location points',
                showlegend=showlegend,
            ))
            
            
            # shift line x positions slightly
            xoffset = np.array([-2,2]) if cor_side == 'positive' else np.array([2,-2])
            for k, (_, row) in enumerate(d.iterrows()):
                # draw a line between correct and incorrect staytime
                fig.add_trace(go.Scatter(
                    y=row.loc[['staytime_correct_r', 'staytime_incorrect_r']],
                    x=[cor_side_point_pos_x, incor_side_point_pos_x]+xoffset,
                    mode='lines',
                    line=dict(color='rgba(0,0,0,.1)'),
                    name='Withintrial staytimes',
                    legendgroup='Withintrial staytimes',
                    showlegend=showlegend,
                ))
                showlegend = False
            
            
def _calc_x_positions(all_data, sort_by):
    x_indices = []
    i = 0
    for identif, row in all_data.iterrows():
        if row['cue'] not in (1,2):
            # very rare edge case in early sessions
            x_indices.append(i)
            continue
        
        if i == 0: # init   
            cur_session_id = identif[2]
            cur_trialtype = row['cue']
        
        # next session, add space
        if cur_session_id != identif[2]:
            i += 15
            cur_session_id = identif[2]
        
        # next cue, add space
        if all(('cue' in sort_by, 'staytime_correct_r' not in sort_by,
                    cur_trialtype != row['cue'])):
            i += 2
            cur_trialtype = row['cue']
        x_indices.append(i)
        i += 1
    return x_indices

def _configure_axis(fig, all_data, metric_max):
    # make backgorund white
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=20, t=50, b=50)  # Adjust margins as needed
    )    
    
    # make empty list of tick (step=5),
    # for ticklabels replace every 5th with the trial_id (-> step=25)
    ticklabels = [''] *all_data.x_indices[::5].shape[0]
    ticklabels[::5] = all_data.index.get_level_values("entry_id")[::5*5]
    fig.update_xaxes(
        tickvals=all_data.x_indices[::5],
        ticktext=ticklabels,
        gridwidth=1,
        gridcolor='rgba(128,128,128,.14)',
        title_text='Trial ID',
    )
    
    fig.update_yaxes(
        range=[0, metric_max],
        zeroline=False,
        showgrid=False,
        ticks='outside',
        title_text="Time in reward zone [s]",
    )
    return fig

def render_plot(all_data, metric_max, sort_by=['session_id']):
    print(all_data)
    print("================")

    # Data transformations    
    # sort_by = ['session_id', 'cue', 'staytime_correct_r']
    sort_by = ['session_id', 'cue']
    if sort_by != ['session_id']: # already sorted anyway, mixes things up
        all_data = all_data.sort_values(by=sort_by)
    
    all_data.loc[:, ['staytime_correct_r', 'staytime_incorrect_r']] /= 1_000_000  # us to s
    # reset trial_id to set of sessions, each is unique
    all_data = all_data.droplevel('entry_id')
    all_data['entry_id'] = np.arange(len(all_data))
    all_data.set_index('entry_id', append=True, inplace=True)
    
    # do the plotting
    fig = make_subplots(rows=1, cols=1)
    
    all_data['x_indices'] = _calc_x_positions(all_data, sort_by)

    # add outcome color and  cue color columns for later    
    cmap_transparent = {k: v.replace("rgb","rgba")[:-1]+f', {MULTI_MARKERS_ALPHA})' 
                        for k,v in OUTCOME_COL_MAP.items()}
    all_data['outcome_color'] = all_data['trial_outcome'].map(cmap_transparent)
    all_data['cue_color'] = all_data['cue'].map(CUE_COL_MAP)
    
    
    _draw_violin_plots(fig, all_data, metric_max, indicate_outcome=True)
    _draw_treshold_and_cue_indicator(fig, all_data, sort_by, include_threshold=True)
    
    # _draw_treshold_and_cue_indicator(fig, all_data, sort_by, include_threshold=True)
    # _draw_single_trials(fig, all_data)

    _configure_axis(fig, all_data, metric_max)
    return fig