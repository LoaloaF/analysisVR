import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plot_constants import *
from .plot_utils import make_discr_trial_cmap

def render_plot(data, n_trials, trial_color, metric):
    # Filter data based on the selected animal, session, and trial IDs
    print("-----------------")
    
    if metric == 'Velocity':
        metric_col = 'posbin_z_velocity'
    elif metric == 'Acceleration':
        metric_col = 'posbin_z_acceleration'
    
    # Determine the color column and color mapping
    if trial_color == 'Outcome':
        color_col, cmap = 'trial_outcome', OUTCOME_COL_MAP
    elif trial_color == 'Cue':
        color_col, cmap = 'cue', CUE_COL_MAP
    elif trial_color == 'Trial ID':
        color_col = 'trial_id'
        cmap =  make_discr_trial_cmap(n_trials, TRIAL_COL_MAP)
    
    # add transparency to the colors
    cmap = {k: v.replace("rgb","rgba")[:-1]+f', {MULTI_TRACES_ALPHA})' 
            for k,v in cmap.items()}
    
    # Create the main plot with line grouping by trial_id and color based on trial_color
    fig_main = px.line(
        data, 
        x='from_z_position_bin', 
        y=metric_col, 
        line_group="trial_id",
        color=color_col,
        color_discrete_map=cmap
    )
    # # fig_main.update_traces(line=dict(width=2, color='rgba(0,0,0,0.3)'))  # Example with 30% opacity (alpha = 0.3)
    # for i, trace in enumerate(fig_main.data):
    #     trace.line.color = trace.line.color[:-1] + '0.3)'  # Change the alpha to 0.3 for each trace

    
    # Create subplots with a slim top axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.1, 0.9],  # Slim top axis
        shared_xaxes=True,
        vertical_spacing=0.02
    )
    
    # Add the main plot to the subplots
    for trace in fig_main['data']:
        fig.add_trace(trace, row=2, col=1)
        
    trace2 = go.Line(
        y=data.groupby('from_z_position_bin')[metric_col].mean(),
        x=data["from_z_position_bin"],
        mode='lines',
        line=dict(color='black', width=2),
        name='Average'
        )
    fig.add_trace(trace2, row=2, col=1)
    
    # Update layout for the top axis
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        row=1, col=1
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        row=1, col=1
    )
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update layout for the main axis
    fig.update_layout(
        xaxis_title='Position (Z)',
        yaxis_title=metric,
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="Black"
        )
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='LightGray',
        row=2, col=1
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='LightGray',
        row=2, col=1
    )
    
    return fig