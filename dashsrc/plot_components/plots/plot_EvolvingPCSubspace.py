import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashsrc.components.dashvis_constants import *
from .general_plot_helpers import make_discr_trial_cmap
from .general_plot_helpers import draw_track_illustration
from scipy.linalg import subspace_angles

import json



def render_plot(data, CCs, metadata, selected_predictor, selected_session_focus, 
                selected_trackzone, max_metric, avg_over_n_CCs, height, width):    
    
    session_t = pd.to_datetime(metadata.start_time, format='%Y-%m-%d_%H-%M')
    session_t.index = session_t.index.get_level_values('session_id')
    
    
    sessions = CCs.index.get_level_values('session_id').unique().to_list()
    
    if selected_predictor == 'HP':
        nfeatures = 20
    elif selected_predictor == 'mPFC':
        nfeatures = 57
    elif selected_predictor == 'HP-mPFC':
        nfeatures = 77
    elif selected_predictor == 'behavior':
        nfeatures = 6
        
    zones = {
        'beforeCueZone': (-168, -100),
        'cueZone': (-80, 25),
        'afterCueZone': (25, 50),
        'reward1Zone': (50, 110),
        'betweenRewardsZone': (110, 170),
        'reward2Zone': (170, 230),
        'postRewardZone': (230, 260),
        'wholeTrack': (-168, 260),
    }
        
    fig = make_subplots(rows=4, cols=1, row_heights=(.125, .125, .05, .7), vertical_spacing=0.01,)
    
    if selected_session_focus is not None:
        idx_selected_session_focus = sessions.index(selected_session_focus)
        
        if idx_selected_session_focus < len(sessions)-1:
            nxt_selected_session_focus = sessions[idx_selected_session_focus + 1]
            
            min_track, max_track = -169, 260
            draw_track_illustration(fig, row=3, col=1, track_details=json.loads(metadata.iloc[0]['track_details']), 
                                    min_track=min_track, max_track=max_track, choice_str='Stop', draw_cues=[2], 
                                    double_rewards=False)
        
            print('===============')
            print(data)
            print('Selected Session Focus:', selected_session_focus)
            print('Next Selected Session Focus:', nxt_selected_session_focus)
            print(CCs.index.get_level_values('session_id').unique())
            print(CCs.index.get_level_values('comp_session_id').unique())
            print(CCs)
            CCs_arr = CCs.loc[(selected_session_focus, nxt_selected_session_focus), :].dropna(axis=1, how='all').values
            CCs_arr = np.abs(CCs_arr.reshape(CCs_arr.shape[0], nfeatures, -1))
            print(CCs_arr)
            print("CCs_arr.shape:")
            print(CCs_arr.shape)
            # print(PCs_range)
            print('---------------')
            
            fig.add_trace(
                go.Heatmap(
                    z=CCs_arr[:avg_over_n_CCs].mean(axis=0),  # Average across CC_i
                    x=np.arange(*zones[selected_trackzone]),  # Use feature indices as x-axis
                    # y=np.arange(CCs_arr.shape[2]),  # Use CC_i indices as y-axis
                    # y=mean_angles.index.get_level_values('comp_session_id').astype(str),  # Use session_id as y-axis
                    colorscale='Greys',
                ), row=1, col=1,
            )
            fig.add_trace(
                go.Heatmap(
                    z=CCs_arr[-avg_over_n_CCs:].mean(axis=0),  # Average across CC_i
                    x=np.arange(*zones[selected_trackzone]),  # Use feature indices as x-axis
                    # y=np.arange(CCs_arr.shape[2]),  # Use CC_i indices as y-axis
                    # x=mean_angles.columns.astype(str),  # Use session_id as x-axis
                    # y=mean_angles.index.get_level_values('comp_session_id').astype(str),  # Use session_id as y-axis
                    colorscale='Greys',
                ), row=2, col=1,
            )
            
            fig.update_xaxes(
                range=[min_track, max_track],
                row=2, col=1
            )
            fig.update_yaxes(
                title_text='last CCs',
                range=(0, nfeatures),
                row=2, col=1
            )
            fig.update_xaxes(
                range=[min_track, max_track],
                row=1, col=1
            )
            fig.update_yaxes(
                title_text='first CCs',
                range=(0, nfeatures),
                row=1, col=1
            )
            
            
    
    # selected_session_focus = [selected_session_focus, selected_session_focus + 1]
    # selected_session_focus = filter(lambda x: x in session_slice, selected_session_focus)

    mean_angles = data.iloc[:, :].mean(axis=1).unstack(level=0)
    mean_angles = np.cos(mean_angles)  # Convert radian angles to cosine similarity
    print(mean_angles)
    
    if height == -1:
        height = 2000
   
    
    # draw_track_illustration(fig, row, col, track_details, min_track, max_track, 
    #                         draw_cues=[2], choice_str='Stay', double_rewards=False):
    
    
    fig.add_trace(
        go.Heatmap(
            z=mean_angles.values,
            x=mean_angles.columns.astype(str),  # Use session_id as x-axis
            y=mean_angles.index.get_level_values('comp_session_id').astype(str),  # Use session_id as y-axis
            colorscale='Viridis',
            colorbar=dict(title='Subspace Angle (cosine similarity)'),
            zmin=0, zmax=max_metric,  # Adjust based on expected range of angles
        ), row=4, col=1,
    )
    
    # Draw BOX around selected session, sel session+1
    if selected_session_focus is not None and selected_session_focus+1 in mean_angles.columns:
        # Find the x-position (index) of the selected session
        position = list(mean_angles.columns.astype(str)).index(str(selected_session_focus))
        fig.add_shape(
            type='rect',
            x0=position-0.5,
            y0=position-0.5,
            x1=position+1.5,
            y1=position+1.5,
            fillcolor='rgba(255,255,255,0.05)',
            line=dict(color='red', width=2, dash='dash'),
            xref='x',
            yref='y',
            row=4, col=1
        )
        
        print(CCs)
        # Add text annotation for the selected session
        
    fig.update_yaxes(
        title_text='Session ID',
        scaleanchor="x",
        scaleratio=1,
        row=4, col=1
    )
    
    # set background to white
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        title=f'Subspace Angles for {selected_predictor} {selected_trackzone}',
        # autosize=True,
        
        height=height,
        
        
        # margin=dict(l=100, r=20, t=50, b=50)  # Adjust margins as needed
    )
    print("Done rednering")
    return fig