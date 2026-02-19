import pandas as pd
import json

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .general_plot_helpers import draw_track_illustration
from .general_plot_helpers import make_kinematics_figure
from .general_plot_helpers import make_discr_trial_cmap

from dashsrc.components.dashvis_constants import *

def _parse_args(n_trials, group_by, metric, metric_max, smooth_data):
    # parse arguemnts and set defaults    
    if metric == 'Velocity':
        metric_col = 'posbin_velocity'
        y_axis_label = 'Velocity [cm/s]'
        y_axis_range = 0, metric_max
    elif metric == 'Acceleration':
        metric_col = 'posbin_acc'
        y_axis_label = 'Acceleration [cm/s^2]'
        y_axis_range = -metric_max, metric_max
    elif metric == 'Lick':
        metric_col = 'lick_detected'
        y_axis_label = 'Lick count'
        y_axis_range = 0, metric_max
        
    elif metric == 'RawYawPitch Sum Velocity':
        metric_col = 'posbin_RawYawPitch_abs_vel_sum'
        y_axis_label = 'RawYawPitch Sum Velocity [cm/s]'
        y_axis_range = 0, metric_max
    
    elif metric == 'BallForward Velocity':
        metric_col = 'posbin_raw'
        y_axis_label = 'BallForward Velocity [cm/s]'
        y_axis_range = 0, metric_max
    
    elif metric == 'BallSide Velocity':
        metric_col = 'posbin_pitch'
        y_axis_label = 'BallSide Velocity [cm/s]'
        y_axis_range = -metric_max/2, metric_max/2
    
    elif metric == 'BallRotation Velocity':
        metric_col = 'posbin_yaw'
        y_axis_label = 'BallRotation Velocity [cm/s]'
        y_axis_range = -metric_max/2, metric_max/2
        
    if smooth_data and metric not in ['Lick', 'Velocity', 'Acceleration']:
        metric_col = metric_col + '_500msMedian'
    
    # Determine the color column and color mapping
    if group_by == 'Outcome':
        group_col, cmap = 'trial_outcome', OUTCOME_COL_MAP
    elif group_by == 'Cue':
        group_col, cmap = 'cue', CUE_COL_MAP
    elif group_by == 'Part of session':
        group_col = 'trial_id'
        cmap =  make_discr_trial_cmap(n_trials, TRIAL_COL_MAP)
    elif group_by == "None":
        group_col = "cue" # can be anything
        cmap = dict.fromkeys([1,2], "rgb(128,128,128)")
    # add transparency to the colors
    cmap_transparent = {k: v.replace("rgb","rgba")[:-1]+f', {MULTI_TRACES_ALPHA})' 
                        for k,v in cmap.items()}
    return metric_col, y_axis_label, y_axis_range, group_col, cmap, cmap_transparent

# def _make_figure():
    
#     # Create subplots with a slim top axis
#     fig = make_subplots(
#         rows=2, cols=1,
#         row_heights=[0.07, 0.93],  # Slim top axis
#         shared_xaxes=True,
#         vertical_spacing=0.02
#     )
#     return fig

def _configure_axis(fig, height, width, y_axis_range, y_axis_label):
    # Update layout for the main axis
    kwargs = {}
    if height != -1:
        kwargs['height'] = height
    if width != -1:
        kwargs['width'] = width
        
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        autosize=True,
        margin=dict(l=0, r=0, t=0, b=0),  # Adjust margins as needed
        **kwargs,
    )
    
    # Update layout for the main axis
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0),  # Adjust margins as needed
        # width=800, height=400,
    )
    # fig.update_layout(
    #     plot_bgcolor='white',
    #     paper_bgcolor='white',
    #     autosize=True,
    #     **kwargs,
    # )
    
    fig.update_xaxes(
        showgrid=False,  # No x grid lines
        zeroline=False,
        showticklabels=True,
        title_text='Position [cm]',
        row=2, col=1
    )
    
    fig.update_yaxes(
        range=y_axis_range,
        showgrid=True,  # y grid lines
        gridwidth=1,
        gridcolor='LightGray',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='LightGray',
        title_text=y_axis_label,
        row=2, col=1
    )
    return fig

def _draw_all_single_trials(fig, data, metric_col, cmap_transparent, group_col,
                            events):
    # Create the main plot with line grouping by trial_id and color based on group_by
    for trial_id, trial_data in data.groupby('trial_id'):
        trace = go.Scatter(x=trial_data['from_position_bin'], 
                        y=trial_data[metric_col], mode='lines',
                        line=dict(color=cmap_transparent[trial_data[group_col].iloc[0]]),
                        name=f'Tr. {trial_id}')
        fig.add_trace(trace, row=2, col=1)
        
        # draw event markers if selected
        for event in events:
            event_data = trial_data[trial_data[event] >= 1]
            r1_thr = trial_data['velocity_threshold_at_R1'].iloc[0] *60
            r2_thr = trial_data['velocity_threshold_at_R2'].iloc[0] *60
            
            labels = {'reward-sound_count': {
                        'label': f'Sound, R1,R2 thr. {r1_thr:.1f},{r2_thr:.1f}',
                        'color': 'green',
                        'size': 8,
                        'marker': 'triangle-right'},
                      'reward-valve-open_count': {
                          'label': f'Reward, R1,R2 thr. {r1_thr:.1f},{r2_thr:.1f}',
                            'color': 'green',
                            'size': 8,
                            'marker': 'circle'},
                      'lick_count': {
                          'label': 'Lick',
                          'color': 'black',
                          'size': 8,
                          'marker': 'line-ns'},
                      'reward-removed_count': {
                          'label': 'Reward removed',
                          'color': 'red',
                          'size': 8,
                          'marker': 'x'} }
            
            
            print(event_data[metric_col])
            fig.add_trace(go.Scatter(
                x=event_data['from_position_bin'],
                y=event_data[metric_col],
                mode='markers',
                marker=dict(
                    color=labels[event]['color'],
                    size=labels[event]['size'],
                    symbol=labels[event]['marker'],
                ),
                name=f'Tr.{trial_id}, {labels[event]["label"]}',
            ), row=2, col=1)
            
            

def _draw_percentile_area_plot(fig, upper_perc, lower_perc, metric_col, transp_color):
    # draw an area plot for the 80th percentile
    # draw essentially 2 lines and fill the area between them
    # print(upper_perc, lower_perc)
    fig.add_trace(go.Scatter(
        x=upper_perc['from_position_bin'].tolist() + lower_perc['from_position_bin'].tolist()[::-1],
        y=upper_perc[metric_col].tolist() + lower_perc[metric_col].tolist()[::-1],
        fill='toself',
        fillcolor=transp_color,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        name='80th perc.',
    ), row=2, col=1)
    
def render_plot(all_data, metadata, n_trials, group_by, group_by_values, metric, 
                metric_max, smooth_data, var_viz, events, width=-1, height=-1):
    print("================")
        
    fig = make_kinematics_figure(height=height)
    # parse the arguments
    _ = _parse_args(n_trials, group_by, metric, metric_max, smooth_data)
    metric_col, y_axis_label, y_axis_range, group_col, cmap, cmap_transparent = _
    
    for session_id, data in all_data.groupby(level='session_id'):
        # handle the data
        # Smooth the data with exponential moving average
        # if smooth_data:
        #     data[metric_col] = data[metric_col].rolling(window=30, center=True, min_periods=1).median()
        
        # # last spatial bins can ba NaN, remove them
        # min_max_pos_bin = data['from_position_bin'].min(), data['from_position_bin'].max()
        # data = data[(data['from_position_bin'] > min_max_pos_bin[0]) &
        #             (data['from_position_bin'] < min_max_pos_bin[1])]
            
        # # TODO handle this properly
        # # deal with double outcomes
        # outcome_r1 = all_data['trial_outcome'] // 10 
        # outcome_r2 = all_data['trial_outcome'] % 10
        # print(all_data['trial_outcome'])
        # print(outcome_r1)
        # print(outcome_r2)
        # outcome_r1.loc[:] = np.max([outcome_r1, outcome_r2], axis=0)
        # print(outcome_r1)
        
        #TODO if P1100: choice_str='Stop', if DR == 1 draw_cues=[]
        # print()
        # print(metadata.iloc[0].values)
        # print(data.columns)
        
        draw_cues = [2]
        flipped = False
        double_rewards = False
        if data['both_R1_R2_rewarded'].dropna().iloc[0] == 1:
            draw_cues = []
            double_rewards = True
        if data['flip_Cue1R1_Cue2R2'].dropna().any():# or session_id >= 27:
            flipped = True
        
        min_track, max_track = data['from_position_bin'].min(), data['from_position_bin'].max()
        draw_track_illustration(fig, row=1, col=1,  track_details=json.loads(metadata.iloc[0]['track_details']), 
                                min_track=min_track, max_track=max_track, choice_str='Stop', 
                                draw_cues=draw_cues, double_rewards=double_rewards,
                                flipped=flipped,)


        if var_viz == 'Single trials':
            _draw_all_single_trials(fig, data, metric_col, cmap_transparent, 
                                    group_col, events)
            
        if group_by == "None":
            med_values = data.groupby('from_position_bin')[metric_col].mean().reset_index()
            # print(med_values.isna().sum())
            # print(med_values[med_values.isna()])
            # Add the mean trace to the main plot
            mean_trace = go.Scatter(
                x=med_values['from_position_bin'],
                y=med_values[metric_col],
                mode='lines',
                line=dict(color='black', width=3),
                name='Median'
            )
            fig.add_trace(mean_trace, row=2, col=1)
            
            if var_viz == '80th percent.':
                upper_perc = data.groupby('from_position_bin')[metric_col].quantile(0.9).reset_index().dropna()
                lower_perc = data.groupby('from_position_bin')[metric_col].quantile(0.1).reset_index().dropna()
                _draw_percentile_area_plot(fig, upper_perc, lower_perc, metric_col, 'rgba(128,128,128,0.3)')
        
        else:
            for group_lbl, group_values in group_by_values.items():
                group_data = data[data[group_col].isin(group_values)]
                
                groupwise_med_values = group_data.groupby(['from_position_bin', group_col])[metric_col].median().reset_index()
                # groupwise_med_values = groupwise_med_values[groupwise_med_values[group_col].isin(group_values)]
                # when group_values has more than one value, this dim needs to collapse to one value
                groupwise_med_values = groupwise_med_values.groupby('from_position_bin').median().reset_index()
                if group_by == 'Part of session':
                    # we only draw one line for a set of trials (eg first 3rd of trials),
                    # for line color, use the trial_id in the middle of the set (used for cmap below)
                    color = cmap[group_values[len(group_values)//2]]
                    transp_color = cmap_transparent[group_values[len(group_values)//2]]
                else:
                    color = cmap[group_values[0]]
                    transp_color = cmap_transparent[group_values[0]]
                
                # draw the group median line
                trace = go.Scatter(
                    x=groupwise_med_values['from_position_bin'],
                    y=groupwise_med_values[metric_col],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f'{group_lbl}'
                )
                fig.add_trace(trace, row=2, col=1)
                
                # draw the 80th percentile area plot
                if var_viz == '80th percent.':
                    upper_perc = group_data.groupby('from_position_bin')[metric_col].quantile(0.9).reset_index().dropna()
                    lower_perc = group_data.groupby('from_position_bin')[metric_col].quantile(0.1).reset_index().dropna()
                    _draw_percentile_area_plot(fig, upper_perc, lower_perc, metric_col, transp_color)
                
    fig = _configure_axis(fig, height, width, y_axis_range, y_axis_label)
    
    return fig