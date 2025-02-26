import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json

from dashsrc.components.dashvis_constants import *
# from .plot_utils import make_discr_trial_cmap

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
        rows=2, cols=1,
        row_heights=[track_height_prop*2, hm_height],
        shared_xaxes=True,
        vertical_spacing=0,
    )
    print("prop2: ", track_height_prop*2,hm_height,hm_label_height_prop)
  
    return fig

def _configure_axis(fig, n_trials, width, height):
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
        range=[n_trials, -50],
        # tickvals=session_ids,
        zeroline=False,
        ticks='outside',
        title_text="Trials",
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
    print(track_details)
    
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


        
def render_plot(all_data, metadata, width, height):
    print(metadata)
    print("================")
    
    metric_col = 'L_count'
    
    session_ids = all_data.index.unique('session_id')
    fig = make_kinematics_figure(height=height)
    # parse the arguments
    # metric_col, y_axis_label, z_axis_range, cmap = _parse_args(metric, metric_max)
    
    lickdata = []
    rewarddata = []
    suctiondata = []
    n = 0
    for session_id, data in all_data.groupby(level='session_id'):
        # Smooth the data with exponential moving average
        # if smooth_data:
        #     data[metric_col] = data[metric_col].rolling(window=10, center=True, min_periods=1).mean()
        d = (data.loc[:, ['from_z_position_bin', "trial_id", metric_col, "R_count", "V_count"]]).set_index(['trial_id', 'from_z_position_bin'])
        d = d.unstack('from_z_position_bin').reindex(np.arange(data['trial_id'].max())).fillna(0)
        lickdata.append(d[metric_col])
        rewarddata.append(d["R_count"])
        suctiondata.append(d["V_count"])
        
        min_track, max_track = d.columns.min()[1], d.columns.max()[1]
        print(min_track, max_track, n)
        n += d.shape[0]
        # fig.add_scatter(x=[min_track, max_track], y=[n, n], mode='lines', 
        #                 line=dict(color='gray', width=.5),
        #                 zorder = 20,
        #                 row=2, col=1,
        #                 showlegend=False)
        
    r_scatter = pd.concat(rewarddata, axis=0).reset_index(drop=True).stack()
    r_scatter = r_scatter[r_scatter > 0]
    # draw the reward locations
    fig.add_trace(go.Scatter(
        x=r_scatter.index.get_level_values(1),
        y=r_scatter.index.get_level_values(0),
        mode='markers',
        marker=dict(color='green', size=3, symbol='circle'),
        showlegend=False,
        zorder=10,
    ), row=2, col=1)
    
    v_scatter = pd.concat(suctiondata, axis=0).reset_index(drop=True).stack()
    v_scatter = v_scatter[v_scatter > 0]
    fig.add_trace(go.Scatter(
        x=v_scatter.index.get_level_values(1),
        y=v_scatter.index.get_level_values(0),
        mode='markers',
        marker=dict(color='red', size=3, symbol='circle'),
        showlegend=False,
        zorder=10,
    ), row=2, col=1)
    
    
    heatmap_data = pd.concat(lickdata, axis=0).reset_index(drop=True)
    print()
    print(heatmap_data)
    print()
    
    min_track, max_track = heatmap_data.columns.min(), heatmap_data.columns.max()
    draw_track_illustration(fig, row=1, col=1,  track_details=json.loads(metadata.iloc[0]['track_details']), 
                            min_track=min_track, max_track=max_track)
    # Draw a heatmap of the median values
    heatmap = go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=px.colors.sequential.gray_r,
        zmin=0,
        zmax=1,
        zorder=1,
        showscale=False,
        colorbar=dict(title="Licks", titleside='right')  # Add colorbar title

    )
    fig.add_trace(heatmap, row=2, col=1)
    
    n_trials = heatmap_data.shape[0]
    fig = _configure_axis(fig, n_trials, width, height)
    return fig