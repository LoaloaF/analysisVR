import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc

from plotly.subplots import make_subplots

from .plot_constants import *

def _make_discr_trial_cmap(n_trials, px_seq_cmap):
    discrete_colors = pc.sample_colorscale(px_seq_cmap, samplepoints=n_trials+1)
    return dict(zip(range(n_trials+1),discrete_colors))

def render_plot(data, trial_color, metric, metric_max):
    # Filter data based on the selected animal, session, and trial IDs
    print("-----------------")
    
    if metric == 'Velocity':
        metric_col = 'posbin_z_velocity'
        y_axis_label = 'Velocity [cm/s]'
        y_axis_range = 0, metric_max
    elif metric == 'Acceleration':
        metric_col = 'posbin_z_acceleration'
        y_axis_label = 'Acceleration [cm/s^2]'
        y_axis_range = -metric_max, metric_max
    
    # Determine the color column and color mapping
    if trial_color == 'Outcome':
        color_col, cmap = 'trial_outcome', OUTCOME_COL_MAP
    elif trial_color == 'Cue':
        color_col, cmap = 'cue', CUE_COL_MAP
    elif trial_color == 'Session ID':
        # color_col = 'trial_id'
        max_n_sessions = data.index.unique('session_id').shape[0]
        print("max_n_sessions ", max_n_sessions)
        cmap =  _make_discr_trial_cmap(max_n_sessions, TRIAL_COL_MAP)
    
    # add transparency to the colors
    cmap = {k: v.replace("rgb","rgba")[:-1]+f', {MULTI_TRACES_ALPHA})' 
            for k,v in cmap.items()}
    
    # Create subplots with a slim top axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.07, 0.93],  # Slim top axis
        shared_xaxes=True,
        vertical_spacing=0.02
    )
    
    # Create the main plot with line grouping by trial_id and color based on trial_color
    for anim_session_id, session_data in data.groupby(level=('animal_id', 'session_id')):
        print("========")
        # print(session_data)
        # print(session_data.groupby("trial_id")[metric_col].mean())
        mean_values = session_data.groupby('from_z_position_bin')[metric_col].mean().reset_index()
        # print(mean_values)
        # print(anim_session_id)
        
        
        trace = go.Scatter(
            x=mean_values['from_z_position_bin'],
            y=mean_values[metric_col],
            mode='lines',
            line=dict(color=cmap[anim_session_id[1]]),
            name=f'Animal/Session {anim_session_id}'
        )
        fig.add_trace(trace, row=2, col=1)
    
    # Calculate the mean values for the metric
    mean_values = data.groupby('from_z_position_bin')[metric_col].mean().reset_index()

    # Add the mean trace to the main plot
    mean_trace = go.Scatter(
        x=mean_values['from_z_position_bin'],
        y=mean_values[metric_col],
        mode='lines',
        line=dict(color='black', width=3),
        name='Mean'
    )
    fig.add_trace(mean_trace, row=2, col=1)
        
    # Update layout for the main axis
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(
        showgrid=False,  # No x grid lines
        zeroline=False,
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