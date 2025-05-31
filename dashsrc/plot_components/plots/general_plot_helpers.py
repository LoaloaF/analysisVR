import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from dashsrc.components.dashvis_constants import *

def draw_track_illustration(fig, row, col, track_details, min_track, max_track, 
                            draw_cues=[2], choice_str='Stay', double_rewards=False):
    track_details = [pd.Series(details, name=zone) for zone, details in track_details.items()]
    track_details = pd.concat(track_details, axis=1).T
    track_details.loc['cue2', 'start_pos'] = -80 # wrong metadat, update manually
    
    # draw the bars indicating track zones/ wall textures
    # iterate the two different type of tracks
    for track_type in (1, 2):
        fig.update_yaxes(row=row, col=col, range=(2,0), ticks='', showticklabels=False,)
        fig.update_xaxes(range=(min_track, max_track), row=row, col=col, ticks='outside', 
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
                ), row=row, col=1)

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
                # print("---")
                annotations = [
                    ([r_center-8], [track_type-1+.42], choice_str, {}),
                    ([r_center-8], [track_type-1+.65], '→', {'size': 20}),
                    ([r_center-8], [track_type-1+.54], '        R', {'size': 16, 'weight': 'bold', 
                                                            'color': OUTCOME_COL_MAP[1]}),
                ]
                # print(annotations)
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

def make_track_tuning_figure(height, n_units):
    if height == -1:
        height = (TRACK_VISUALIZATION_HEIGHT + EVENT_VISUALIZATION_HEIGHT + 
                  TRACK_TUNING_PLOT_DEFAULT_HEIGHT + TRACK_TUNING_PLOT_DEFAULT_HSPACE) *n_units
    # absolute px for top t2o axes, track illustration
    trackvis_height_prop = TRACK_VISUALIZATION_HEIGHT/height
    event_height = EVENT_VISUALIZATION_HEIGHT/height
    tuning_height = TRACK_TUNING_PLOT_DEFAULT_HEIGHT/height
    spacer_height = TRACK_TUNING_PLOT_DEFAULT_HSPACE/height
    all_height_props = [pr for _ in range(n_units) for pr in [trackvis_height_prop, event_height, tuning_height, spacer_height]]
    
    # Create subplots with a slim top axis
    fig = make_subplots(
        rows=4*n_units, cols=1,
        row_heights= all_height_props,
        shared_xaxes=True,
        vertical_spacing=.0002,
    )
    return fig, height

import plotly.express as px
import plotly.colors as pc
import numpy as np

def make_discr_trial_cmap(n_trials, px_seq_cmap):
    discrete_colors = pc.sample_colorscale(px_seq_cmap, samplepoints=n_trials+1)
    return dict(zip(range(n_trials+1),discrete_colors))

    # rgb = px.colors.convert_colors_to_same_type(px_seq_cmap)[0]
    # print()
    # print(px.colors.convert_colors_to_same_type(px_seq_cmap))
    # print(rgb)

    # colorscale = {}
    # # n_steps = 4  # Control the number of colors in the final colorscale
    # for i in range(len(rgb) - 1):
    #     for trial_i, step in zip(range(n_trials), np.linspace(0, 1, n_trials)):
    #     # for step in np.linspace(0, 1, n_trials):
    #         col = px.colors.find_intermediate_color(rgb[i], rgb[i + 1], step, 
    #                                                 colortype='rgb') 
    #         colorscale[trial_i] = col
    # print(colorscale)
    # return colorscale