import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plot_constants import *
from .plot_utils import make_discr_trial_cmap

def render_plot(data, n_trials, group_by, group_by_values, metric, metric_max, 
                 smooth_data, var_viz):
    # parse arguemnts and set defaults    
    if len(var_viz) == 1:
        var_viz = var_viz[0]
    if len(smooth_data) == 1:
        smooth_data = True
    
    if metric == 'Velocity':
        metric_col = 'posbin_z_velocity'
        y_axis_label = 'Velocity [cm/s]'
        y_axis_range = 0, metric_max
    elif metric == 'Acceleration':
        metric_col = 'posbin_z_acceleration'
        y_axis_label = 'Acceleration [cm/s^2]'
        y_axis_range = -metric_max, metric_max
    
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
        cmap = dict.fromkeys(data[group_col].unique(), "rgb(128,128,128)")
    # add transparency to the colors
    cmap_transparent = {k: v.replace("rgb","rgba")[:-1]+f', {MULTI_TRACES_ALPHA})' 
                        for k,v in cmap.items()}

    # Smooth the data with exponential moving average
    if smooth_data:
        data[metric_col] = data[metric_col].rolling(window=10, center=True, min_periods=1).mean()
    # last spatial bins can ba NaN, remove them
    min_max_pos_bin = data['from_z_position_bin'].min(), data['from_z_position_bin'].max()
    data = data[(data['from_z_position_bin'] > min_max_pos_bin[0]) &
                (data['from_z_position_bin'] < min_max_pos_bin[1])]





    
    # Create subplots with a slim top axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.07, 0.93],  # Slim top axis
        shared_xaxes=True,
        vertical_spacing=0.02
    )
    
    if var_viz == 'Single trials':
        # Create the main plot with line grouping by trial_id and color based on group_by
        for trial_id, trial_data in data.groupby('trial_id'):
            trace = go.Scatter(x=trial_data['from_z_position_bin'], 
                               y=trial_data[metric_col], mode='lines',
                               line=dict(color=cmap_transparent[trial_data[group_col].iloc[0]]),
                               name=f'Tr. {trial_id}')
            fig.add_trace(trace, row=2, col=1)
    
    if group_by == "None":
        med_values = data.groupby('from_z_position_bin')[metric_col].median().reset_index()
        # Add the mean trace to the main plot
        mean_trace = go.Scatter(
            x=med_values['from_z_position_bin'],
            y=med_values[metric_col],
            mode='lines',
            line=dict(color='black', width=3),
            name='Median'
        )
        fig.add_trace(mean_trace, row=2, col=1)
    
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
            
            
            if var_viz == '80th percent.':
                upper_perc = group_data.groupby('from_z_position_bin')[metric_col].quantile(0.9).reset_index()
                lower_perc = group_data.groupby('from_z_position_bin')[metric_col].quantile(0.1).reset_index()
                # draw an area plot for the 80th percentile
                fig.add_trace(go.Scatter(
                    x=upper_perc['from_z_position_bin'],
                    y=upper_perc[metric_col],
                    fill=None,
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False
                ), row=2, col=1)
                # draw essentially 2 lines and fill the area between them
                fig.add_trace(go.Scatter(
                    x=upper_perc['from_z_position_bin'].tolist() + lower_perc['from_z_position_bin'].tolist()[::-1],
                    y=upper_perc[metric_col].tolist() + lower_perc[metric_col].tolist()[::-1],
                    fill='toself',
                    fillcolor=transp_color,
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                ), row=2, col=1)
                
            trace = go.Scatter(
                x=groupwise_med_values['from_z_position_bin'],
                y=groupwise_med_values[metric_col],
                mode='lines',
                line=dict(color=color, width=2),
                name=f'{group_lbl}'
            )
            fig.add_trace(trace, row=2, col=1)
            
        
        
        
        
        
    # Update layout for the main axis
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        # width=800, height=400,
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