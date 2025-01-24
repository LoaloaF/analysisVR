import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json

from dashsrc.components.dashvis_constants import *
# from .plot_utils import make_discr_trial_cmap

def _parse_args(metric, metric_max):
    # parse arguemnts and set defaults    
    if metric == 'Velocity':
        metric_col = 'posbin_z_velocity'
        y_axis_label = 'Velocity [cm/s]'
        z_axis_range = 0, metric_max
        cmap = px.colors.sequential.Plotly3_r
    elif metric == 'Acceleration':
        metric_col = 'posbin_z_acceleration'
        y_axis_label = 'Acceleration [cm/s^2]'
        z_axis_range = -metric_max, metric_max
        cmap = px.colors.diverging.Tropic_r
    elif metric == 'Lick':
        metric_col = 'L_count'
        y_axis_label = 'Lick count'
        z_axis_range = 0, metric_max
        cmap = px.colors.sequential.Blues
    return metric_col, y_axis_label, z_axis_range, cmap

def make_kinematics_figure(height):
    if height == -1:
        height = KINEMATICS_HEATMAP_DEFAULT_HEIGHT
    # absolute px for top t2o axes, track illustration
    track_height_prop = TRACK_VISUALIZATION_HEIGHT/height
    print("prop: ", track_height_prop)
    hm_label_height_prop = KINEMATICS_HEATMAP_XLABELSIZE_HEIGHT/height
    hm_height = 1 - track_height_prop*2 - hm_label_height_prop
    
    # Create subplots with a slim top axis
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[track_height_prop*2,hm_height,hm_label_height_prop],
        shared_xaxes=True,
        vertical_spacing=0,
    )
    print("prop2: ", track_height_prop*2,hm_height,hm_label_height_prop)
  
    return fig

def _configure_axis(fig, session_ids, width, height):
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
    
    # move legend to the bottom of plot
    fig.update_layout(legend=dict(
        orientation='h',
        yanchor='bottom',
        xanchor='right',
        x=1
    ))
    
    fig.update_xaxes(
        # showgrid=False,  # No x grid lines
        zeroline=False,
        showticklabels=True,
        ticks='outside',
        title_text='Position [cm]',
        row=2, col=1
    )
    fig.update_yaxes(
        range=[session_ids[-1]+.5, session_ids[0]-.55],
        tickvals=session_ids,
        zeroline=False,
        ticks='outside',
        title_text="Session ID",
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
    print(upper_perc, lower_perc)
    fig.add_trace(go.Scatter(
        x=upper_perc['from_z_position_bin'].tolist() + lower_perc['from_z_position_bin'].tolist()[::-1],
        y=upper_perc[metric_col].tolist() + lower_perc[metric_col].tolist()[::-1],
        fill='toself',
        fillcolor=transp_color,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        name='80th perc.',
    ), row=2, col=1)

def draw_track_illustration(fig, row, col, track_details, min_track, max_track, 
                            draw_cues=[2], choice_str='Stay', double_rewards=False):
    track_details = [pd.Series(details, name=zone) for zone, details in track_details.items()]
    track_details = pd.concat(track_details, axis=1).T
    
    # draw the bars indicating track zones/ wall textures
    # iterate the two different type of tracks
    for track_type in (1, 2):
        fig.update_yaxes(row=1, col=1, range=(2,0), ticks='', showticklabels=False,)
        fig.update_xaxes(range=(min_track, max_track), row=1, col=1, ticks='outside', 
                         showticklabels=False)
        
        # iterate early and late cues
        for i,cue in enumerate(draw_cues):
            cuezone_width = track_details.loc[f'cue{cue}', 'end_pos'] - track_details.loc[f'cue{cue}', 'start_pos']
            cuezone_center = track_details.loc[f'cue{cue}', 'start_pos'] +cuezone_width/2
            
            # Cue annotation white text, once per track type
            if track_type == 1:
                fig.add_trace(go.Scatter(
                    text=f'Cue', 
                    x=[cuezone_center], y=[1], mode='text',
                    textposition='middle center', textfont=dict(size=12, weight='bold', color='white'),
                    showlegend=False, zorder=30,
                ), row=1, col=1)

            # cue bars
            yloc, yloc_offset = track_type-1, .17
            color, width = CUE_COL_MAP[track_type], 10
            fig.add_trace(go.Scatter(
                x=track_details.loc[f'cue{cue}', ['start_pos', 'end_pos', 'start_pos', 'start_pos', 'end_pos']],
                y=[yloc+yloc_offset, yloc+yloc_offset, None, yloc+1-yloc_offset, yloc+1-yloc_offset],
                line=dict(color=color, width=width),
                marker=dict(color=color, size=width, symbol='circle'),
                showlegend=False, mode='lines+markers',
            ), row=row, col=col)
        
        # iterate reward locations
        for r in (1, 2):
            # draw the reward zones
            yloc, yloc_offset = track_type-1, .12
            color, width = REWARD_LOCATION_COL_MAP[r], 7
            fig.add_trace(go.Scatter(
                x=track_details.loc[f'reward{r}', ['start_pos', 'end_pos', 'start_pos', 'start_pos', 'end_pos']],
                y=[yloc+yloc_offset, yloc+yloc_offset, None, yloc+1-yloc_offset, yloc+1-yloc_offset],
                line=dict(color=color, width=width),
                showlegend=False, mode='lines',
            ), row=row, col=col)
            
            
            if (track_type == r) or double_rewards:

                # reward location annotation `Stop -> R`
                r_width = track_details.loc[f'reward{r}', 'end_pos'] - track_details.loc[f'reward{r}', 'start_pos']
                r_center = track_details.loc[f'reward{r}', 'start_pos'] + r_width/2
                print("---")
                annotations = [
                    ([r_center-8], [track_type-1+.42], choice_str, {}),
                    ([r_center-8], [track_type-1+.65], '→', {'size': 20}),
                    ([r_center-8], [track_type-1+.54], '        R', {'size': 16, 'weight': 'bold', 
                                                            'color': OUTCOME_COL_MAP[1]}),
                ]
                print(annotations)
                text_args = {'mode': 'text', 'textposition': 'middle center', 'showlegend': False}
                for x, y, text, font_dict in annotations:
                    fig.add_trace(go.Scatter(
                        x=x, y=y, text=text, **text_args, textfont=font_dict,
                    ), row=row, col=col)
        
    # trial type annotation at start of track
    fig.add_trace(go.Scatter(
        x=[min_track], y=[.98],
        text=f'50% of<br>trials',
        mode='text', textposition='middle right',
        showlegend=False,
    ), row=row, col=col)
    
    # draw track borders
    x = [min_track, max_track, None, min_track, max_track, None, min_track, max_track]
    y = [0, 0, None, 1, 1, None, 2, 2]
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines', line=dict(color='black', width=1),
        showlegend=False, zorder=20,
    ), row=row, col=col)


        
def render_plot(all_data, metadata, metric, metric_max, smooth_data, width, height):
    print(metadata)
    print("================")
    
    session_ids = all_data.index.unique('session_id')
    fig = make_kinematics_figure(height=height)
    # parse the arguments
    metric_col, y_axis_label, z_axis_range, cmap = _parse_args(metric, metric_max)
    
    all_median_values = []
    for session_id, data in all_data.groupby(level='session_id'):
        # Smooth the data with exponential moving average
        if smooth_data:
            data[metric_col] = data[metric_col].rolling(window=10, center=True, min_periods=1).mean()
        
        # last spatial bins can ba NaN, remove them
        min_max_pos_bin = data['from_z_position_bin'].min(), data['from_z_position_bin'].max()
        data = data[(data['from_z_position_bin'] > min_max_pos_bin[0]) &
                    (data['from_z_position_bin'] < min_max_pos_bin[1])]

        # Here we use mean instead of median (otherwise it will all be 0 for lick)
        med_values = data.groupby('from_z_position_bin')[metric_col].mean().reset_index()
        med_values = med_values.set_index('from_z_position_bin').iloc[:,0]
        med_values.name = session_id
        all_median_values.append(med_values)

    heatmap_data = pd.concat(all_median_values, axis=1).T
    heatmap_data.index.name = 'session_id'
    # print(heatmap_data)
    
    min_track, max_track = heatmap_data.columns.min(), heatmap_data.columns.max()
    draw_track_illustration(fig, row=1, col=1,  track_details=json.loads(metadata.iloc[0]['track_details']), 
                            min_track=min_track, max_track=max_track)
    
    # Draw a heatmap of the median values
    heatmap = go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=cmap,
        zmin=z_axis_range[0],
        zmax=z_axis_range[1],
        showscale=True,
        colorbar=dict(title=y_axis_label, titleside='right')  # Add colorbar title

    )
    fig.add_trace(heatmap, row=2, col=1)
    
    
    fig = _configure_axis(fig, session_ids, width, height)
    return fig