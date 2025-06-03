import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashsrc.components.dashvis_constants import *
from .general_plot_helpers import make_discr_trial_cmap
from scipy.linalg import subspace_angles


def render_plot(data, metadata, selected_predictor, selected_trackzone, max_metric, height, width):    
    session_t = pd.to_datetime(metadata.start_time, format='%Y-%m-%d_%H-%M')
    session_t.index = session_t.index.get_level_values('session_id')

    mean_angles = data.iloc[:, :].mean(axis=1).unstack(level=0)
    mean_angles = np.cos(np.deg2rad(mean_angles))  # Convert angles to cosine similarity
    print(data)
    print(mean_angles)
    
    if height == -1:
        height = 1000
   
    fig = make_subplots(rows=1, cols=1) #, row_heights=(.9, .1), shared_xaxes=True,)
    
    
    fig.add_trace(
        go.Heatmap(
            z=mean_angles.values,
            x=mean_angles.columns.astype(str),  # Use session_id as x-axis
            y=mean_angles.index.get_level_values('comp_session_id').astype(str),  # Use session_id as y-axis
            colorscale='Viridis',
            colorbar=dict(title='Subspace Angle (cosine similarity)'),
            zmin=0, zmax=max_metric,  # Adjust based on expected range of angles
            
        ), row=1, col=1,
    )
    
    # # get the x axis for the fig
    # fig.add_annotation(
    #     xref='x', yref='y',
    #     x=10,
    #     y=10,
    #     text="balbla",
    #     showarrow=False,
    #     font=dict(size=10),
    #     align='center',
    #     xanchor='center',
    #     yanchor='bottom',
    #     row=2, col=1,
    # )
    
    
    
    # set background to white
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_title='Session ID',
        yaxis_title='Session ID',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        title=f'Subspace Angles for {selected_predictor} {selected_trackzone}',
        # autosize=True,
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        height=height,
        
        margin=dict(l=100, r=20, t=50, b=50)  # Adjust margins as needed
    )
    
    return fig