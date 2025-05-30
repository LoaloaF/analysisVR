import pandas as pd
import json
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .general_plot_helpers import draw_track_illustration
from .general_plot_helpers import make_track_tuning_figure
from .general_plot_helpers import make_discr_trial_cmap

from dashsrc.components.dashvis_constants import *

def _parse_args(n_trials, group_by, metric, metric_max):
    # parse arguemnts and set defaults    
    if metric == 'Velocity':
        metric_col = 'posbin_z_velocity'
        y_axis_label = 'Velocity [cm/s]'
        y_axis_range = 0, metric_max
    elif metric == 'Acceleration':
        metric_col = 'posbin_z_acceleration'
        y_axis_label = 'Acceleration [cm/s^2]'
        y_axis_range = -metric_max, metric_max
    elif metric == 'Lick':
        metric_col = 'L_count'
        y_axis_label = 'Lick count'
        y_axis_range = 0, metric_max
    
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

def _draw_all_single_trials(fig, data, metric_col, cmap_transparent, group_col):
    # Create the main plot with line grouping by trial_id and color based on group_by
    for trial_id, trial_data in data.groupby('trial_id'):
        trace = go.Scatter(x=trial_data['from_z_position_bin'], 
                        y=trial_data[metric_col], mode='lines',
                        line=dict(color=cmap_transparent[trial_data[group_col].iloc[0]]),
                        name=f'Tr. {trial_id}')
        fig.add_trace(trace, row=2, col=1)

def _draw_percentile_area_plot(fig, upper_perc, lower_perc, metric_col, transp_color):
    # draw an area plot for the 80th percentile
    # draw essentially 2 lines and fill the area between them
    # print(upper_perc, lower_perc)
    fig.add_trace(go.Scatter(
        x=upper_perc['from_z_position_bin'].tolist() + lower_perc['from_z_position_bin'].tolist()[::-1],
        y=upper_perc[metric_col].tolist() + lower_perc[metric_col].tolist()[::-1],
        fill='toself',
        fillcolor=transp_color,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        name='80th perc.',
    ), row=2, col=1)
    
def render_plot(track_data, fr, metadata, spike_metadata, metric, n_sessions,
                metric_max, smooth_data, width=-1, height=-1):
    fr = fr.set_index(['trial_id', 'from_z_position_bin', 'cue'], append=True, )
    fr.drop(columns=['trial_outcome','bin_length'], inplace=True)
    fr.columns = fr.columns.astype(int)
    fr = fr.reindex(columns=sorted(fr.columns))
    
    print(fr)
    fr = fr.groupby(['from_z_position_bin', 'session_id']).mean().sort_index(level=['session_id', 'from_z_position_bin']).fillna(0)
    # print(fr)
    
    # fr.columns = fr.columns.astype(int)
    # fr = fr.reindex(columns=sorted(fr.columns))
    print(track_data)
    print(track_data.iloc[0])
    # print(metadata)
    # print(spike_metadata)
    print("================")
    # reorder to sorted columns
    
    cmap =  make_discr_trial_cmap(n_sessions, TRIAL_COL_MAP)
    
    unit_ids = fr.columns
    # unit_ids = (4,6,8,9,21,23,24,26,29)
    
    
    fig, height = make_track_tuning_figure(height=height, n_units=len(unit_ids))
    fig.update_layout(
        height=height,
    )
    
    session_ids = fr.index.unique('session_id')
    for i, cluster_id in enumerate(unit_ids):
    # for i, cluster_id in enumerate(fr.columns):
        print("\ncluster_id ", cluster_id)
        # if cluster_id not in (4,6,8,9,21,23,24,26,29): 
        #     continue
        
        # handle the data
        track_visrow = i*4 +1
        event_row = i*4 +2
        tuning_row = i*4 +3
        fig.update_yaxes(title_text=f"Neuron {i}", row=tuning_row, col=1)
        
        #TODO if P1100: choice_str='Stop', if DR == 1 draw_cues=[]
        min_track, max_track = track_data['from_z_position_bin'].min(), track_data['from_z_position_bin'].max()
        draw_track_illustration(fig, row=track_visrow, col=1,  track_details=json.loads(metadata.iloc[12]['track_details']), 
                                min_track=min_track, max_track=max_track, choice_str='Stop', draw_cues=[2], double_rewards=False)

        print()
        print()
        print()
        neuron_i_fr = fr.iloc[:, i].unstack(level='session_id').fillna(0).copy()
        print(neuron_i_fr)
        # neuron_i_fr.drop(columns=[10,24,25], inplace=True)
        neuron_i_fr.index = np.sort(track_data['from_z_position_bin'].unique())
        # if cluster_id == 24:
        #     print(neuron_i_fr)
        # neuron_i_fr = neuron_i_fr.drop(columns=[10, 25])
                    
        # do a heatmap instead
        fig.add_trace(
            go.Heatmap(
                z=neuron_i_fr.T.values,
                x=neuron_i_fr.T.columns,
                y=neuron_i_fr.T.index,
                colorscale="amp",  # Color scale
                zmin=0,
                zmax=metric_max,
                showscale=False,
            ), row=tuning_row, col=1,
        )
        unit_metad = spike_metadata[spike_metadata['cluster_id'] == int(cluster_id)].iloc[0]
        
        # for j, session_id in enumerate(session_ids):
        #     # print("session_id ", session_id)
        #     # handle the data
        #     # Smooth the data with exponential moving average
        #     if smooth_data:
        #         neuron_i_fr[session_id] = neuron_i_fr[session_id].rolling(window=10, center=True, min_periods=1).mean()
            
        #     if i > 3:
        #         continue
            # print(neuron_i_fr[session_id])
            
            
            # session_allunits_metad = spike_metadata.loc[pd.IndexSlice[:,:,session_id]]
            # unit_metad = spike_metadata[session_allunits_metad['cluster_id'] == int(cluster_id)].iloc[0]
            # print(unit_metad)
            
            # Add the mean trace to the main plot
            # mean_trace = go.Scatter(
            #     x=neuron_i_fr.index,
            #     y=neuron_i_fr[session_id],
            #     mode='lines',
            #     opacity=0.7,
            #     line=dict(color=cmap[j], width=1),
            #     name=f'Session {session_id}'
            # )
            # fig.add_trace(mean_trace, row=tuning_row, col=1)

        ann = f"Neuron {unit_metad['cluster_id']:03d}" 
        ann += ", HP" if unit_metad['shank_id'] == 2 else ", mPFC"
        fig.update_yaxes(
            title_text=ann, 
            tickvals=neuron_i_fr.columns,
            autorange='reversed',
            ticktext=[f"S{s_id:02d}" for s_id in neuron_i_fr.columns],
            row=tuning_row, col=1)
        fig.update_xaxes(
            range=(min_track, max_track),
            row=tuning_row, col=1
        )
        
        
        
        
        # event plot
        event_cnt = track_data.loc[:, ['from_z_position_bin', 'V_count', 'L_count', 'R_count', #'S_count', 
                                      ]].groupby('from_z_position_bin').sum().astype(float).T
        # print(event_cnt)
        event_cnt /= event_cnt.max(axis=1).values[:, np.newaxis]
        # print(event_cnt)
        
        # small heatmap
        fig.add_trace(
            go.Heatmap(
                z=event_cnt.values,
                x=event_cnt.columns,
                y=event_cnt.index,
                colorscale="Greys",  # Color scale
                showscale=False,
            ), row=event_row, col=1,
        )
        
    
    return fig
    # parse the arguments
    _ = _parse_args(n_trials, group_by, metric, metric_max)
    metric_col, y_axis_label, y_axis_range, group_col, cmap, cmap_transparent = _
    
    for session_id, data in all_data.groupby(level='session_id'):
        # handle the data
        # Smooth the data with exponential moving average
        if smooth_data:
            data[metric_col] = data[metric_col].rolling(window=10, center=True, min_periods=1).mean()
        
        # # last spatial bins can ba NaN, remove them
        # min_max_pos_bin = data['from_z_position_bin'].min(), data['from_z_position_bin'].max()
        # data = data[(data['from_z_position_bin'] > min_max_pos_bin[0]) &
        #             (data['from_z_position_bin'] < min_max_pos_bin[1])]
            
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
        min_track, max_track = data['from_z_position_bin'].min(), data['from_z_position_bin'].max()
        draw_track_illustration(fig, row=1, col=1,  track_details=json.loads(metadata.iloc[0]['track_details']), 
                                min_track=min_track, max_track=max_track, choice_str='Stop', draw_cues=[], double_rewards=True)


        if var_viz == 'Single trials':
            _draw_all_single_trials(fig, data, metric_col, cmap_transparent, group_col)
            
        if group_by == "None":
            med_values = data.groupby('from_z_position_bin')[metric_col].mean().reset_index()
            # print(med_values.isna().sum())
            # print(med_values[med_values.isna()])
            # Add the mean trace to the main plot
            mean_trace = go.Scatter(
                x=med_values['from_z_position_bin'],
                y=med_values[metric_col],
                mode='lines',
                line=dict(color='black', width=3),
                name='Median'
            )
            fig.add_trace(mean_trace, row=2, col=1)
            
            if var_viz == '80th percent.':
                upper_perc = data.groupby('from_z_position_bin')[metric_col].quantile(0.9).reset_index().dropna()
                lower_perc = data.groupby('from_z_position_bin')[metric_col].quantile(0.1).reset_index().dropna()
                _draw_percentile_area_plot(fig, upper_perc, lower_perc, metric_col, 'rgba(128,128,128,0.3)')
        
        else:
            for group_lbl, group_values in group_by_values.items():
                group_data = data[data[group_col].isin(group_values)]
                
                groupwise_med_values = group_data.groupby(['from_z_position_bin', group_col])[metric_col].median().reset_index()
                # groupwise_med_values = groupwise_med_values[groupwise_med_values[group_col].isin(group_values)]
                # when group_values has more than one value, this dim needs to collapse to one value
                groupwise_med_values = groupwise_med_values.groupby('from_z_position_bin').median().reset_index()
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
                    x=groupwise_med_values['from_z_position_bin'],
                    y=groupwise_med_values[metric_col],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f'{group_lbl}'
                )
                fig.add_trace(trace, row=2, col=1)
                
                # draw the 80th percentile area plot
                if var_viz == '80th percent.':
                    upper_perc = group_data.groupby('from_z_position_bin')[metric_col].quantile(0.9).reset_index().dropna()
                    lower_perc = group_data.groupby('from_z_position_bin')[metric_col].quantile(0.1).reset_index().dropna()
                    _draw_percentile_area_plot(fig, upper_perc, lower_perc, metric_col, transp_color)
                
    fig = _configure_axis(fig, height, width, y_axis_range, y_axis_label)
    
    return fig