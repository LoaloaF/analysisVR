import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from .plot_constants import *
from .plot_utils import make_discr_trial_cmap

def _parse_args(metric, metric_max):
    # parse arguemnts and set defaults    
    if metric == 'Velocity':
        metric_col = 'posbin_z_velocity'
        y_axis_label = 'Velocity [cm/s]'
        y_axis_range = 0, metric_max
        cmap = px.colors.sequential.Plotly3
    elif metric == 'Acceleration':
        metric_col = 'posbin_z_acceleration'
        y_axis_label = 'Acceleration [cm/s^2]'
        y_axis_range = -metric_max, metric_max
        cmap = px.colors.diverging.Tropic_r
    return metric_col, y_axis_label, y_axis_range, cmap

def _make_figure():
    # Create subplots with a slim top axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.07, 0.93],  # Slim top axis
        shared_xaxes=True,
        vertical_spacing=0.02
    )
  
    return fig

def _configure_axis(fig, y_axis_range):
    # Update layout for the main axis
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        # width=800, height=400,
        margin=dict(l=50, r=20, t=50, b=50)  # Adjust margins as needed
    )
    
    fig.update_xaxes(
        showgrid=False,  # No x grid lines
        zeroline=False,
        title_text='Position [cm]',
        row=2, col=1
    )
    
    fig.update_yaxes(
        range=y_axis_range,
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
    
def render_plot(all_data, metric, metric_max, smooth_data):
    print(all_data)
    print("================")
    
    fig = _make_figure()
    # parse the arguments
    metric_col, y_axis_label, y_axis_range, cmap = _parse_args(metric, metric_max)
    
    all_median_values = []
    for session_id, data in all_data.groupby(level='session_id'):
        # Smooth the data with exponential moving average
        if smooth_data:
            data[metric_col] = data[metric_col].rolling(window=10, center=True, min_periods=1).mean()
        
        # last spatial bins can ba NaN, remove them
        min_max_pos_bin = data['from_z_position_bin'].min(), data['from_z_position_bin'].max()
        data = data[(data['from_z_position_bin'] > min_max_pos_bin[0]) &
                    (data['from_z_position_bin'] < min_max_pos_bin[1])]

        med_values = data.groupby('from_z_position_bin')[metric_col].median().reset_index()
        med_values = med_values.set_index('from_z_position_bin').iloc[:,0]
        med_values.name = session_id
        all_median_values.append(med_values)

        # print(med_values)
        # # Add the mean trace to the main plot
        # mean_trace = go.Scatter(
        #     x=med_values['from_z_position_bin'],
        #     y=med_values[metric_col],
        #     mode='lines',
        #     line=dict(color='black', width=3),
        #     name='Median'
        # )
        # fig.add_trace(mean_trace, row=2, col=1)
    heatmap_data = pd.concat(all_median_values, axis=1).T
    heatmap_data.index.name = 'session_id'
    print(heatmap_data)
    
    # Draw a heatmap of the median values
    heatmap = go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=cmap,
        zmin=y_axis_range[0],
        zmax=y_axis_range[1],
        showscale=False,
    )
    fig.add_trace(heatmap, row=2, col=1)
    
    fig = _configure_axis(fig, y_axis_range=(max(heatmap_data.index), min(heatmap_data.index)))
    return fig