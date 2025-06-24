import plotly.graph_objects as go
import json

from scipy.ndimage import gaussian_filter

from sklearn.metrics import f1_score, balanced_accuracy_score
import pandas as pd
import numpy as np
from dash import dcc, html
from plotly.subplots import make_subplots

from .general_plot_helpers import draw_track_illustration

def render_plot(data, metadata, width, height, smooth):
    fig = make_subplots(rows=2, cols=2, row_heights=[.1, .9], column_widths=[.1, .9],
                        vertical_spacing=0.01, horizontal_spacing=0.01,
                        )
    
    min_track, max_track = -169, 260
    draw_track_illustration(fig, row=1, col=2, track_details=json.loads(metadata.iloc[0]['track_details']),
                            min_track=min_track, max_track=max_track, choice_str='Stop', draw_cues=[2],
                            double_rewards=False, track_types=(2,))
    draw_track_illustration(fig, row=2, col=1, track_details=json.loads(metadata.iloc[0]['track_details']),
                            min_track=min_track, max_track=max_track, choice_str='Stop', draw_cues=[2],
                            double_rewards=False, track_types=(1,), vertical=True, draw_trial_p_annotation=False)
    
    predictor = data.pop("predictor").iloc[0]
    print(data)
    # data = data.iloc[::-1]
    # print(data)
    cue1_posbins = data.pop("cue1_from_position_bin").values
    cue2_posbins = data.columns.values.astype(int)
    
    # gaussian filter the cross-correlation matrix
    if smooth:
        data.values[:] = gaussian_filter(data.values, sigma=1)
        
    print("Cue 1 Position Bins:", cue1_posbins)
    print("Cue 2 Position Bins:", cue1_posbins.dtype)
    print("Cue 1 Position Bins:", cue2_posbins)
    print("Cue 2 Position Bins:", cue2_posbins.dtype)
    
    # heatmap
    fig.add_trace(
        go.Heatmap(
            z=data.values,
            y=cue1_posbins,
            x=cue2_posbins,
            zmin=.6, zmax=1,
            colorbar=dict(title='Pearson r'),
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(
        # scaleanchor="y",
        # scaleratio=1,
        range=[min_track, max_track],
        title_text='Position [cm]',
        row=2, col=2,
    )
    
    fig.update_yaxes(
        # scaleanchor="x",
        # scaleratio=1,
        showticklabels=False,
        ticks='',
        range=[max_track,min_track],
        row=2, col=2,
    )
    
    fig.update_layout(
        title=f'population vector correlation between cue1 vs cue2 {predictor}',
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white',
        
        # Force aspect ratio to be equal
        # yaxis2=dict(
        #     scaleanchor='x2',
        #     scaleratio=1,
        # ),
        # scaleratio=1,
        
        width=width,
        height=height,
    )
    
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    
    
    
    return fig