from dash import html, dcc, Input, Output, State, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
# from data_loading.animal_loading import get_animal_modality
# from analysis_core import get_default_nas_dir
from dashsrc.plots import kinematics_plot
from .constants import *
from .plot_ui_components import (
    get_animal_dropdown_component,
    get_paradigm_dropdown_component,
    get_metric_radioitems_component,
    get_groupby_radioitems_component,
    get_filter_checklist_component,
    get_display_options_checklist_component,
    # get_trial_range_slider_component,
    register_animal_dropdown_callback,
    register_paradigm_dropdown_callback,
    register_trial_slider_callback
)
from .plot_components import get_track_graph_component

def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
    # Register the callbacks
    register_paradigm_dropdown_callback(app, global_data, vis_name)
    register_animal_dropdown_callback(app, global_data, vis_name)
    # register_session_dropdown_callback(app, global_data, vis_name)
    # register_trial_slider_callback(app, global_data, vis_name)

    @app.callback(
        Output(f'figure-{vis_name}', 'figure'),
        Input(f'animal-dropdown-{vis_name}', 'value'),
        # Input(f'trial-range-slider-{vis_name}', 'value'),
        Input(f'group-by-{vis_name}', 'value'),
        Input(f'metric-{vis_name}', 'value'),
        Input(f'max-metric-value-{vis_name}', 'value'),
        Input(f'smooth-data-{vis_name}', 'value'),
        Input(f'outcome-group-filter-{vis_name}', 'value'),
        Input(f'cue-group-filter-{vis_name}', 'value'),
        Input(f'trial-group-filter-{vis_name}', 'value'),
    )
    def update_plot(selected_animals, group_by, 
                    metric, metric_max, smooth_data,
                    outcome_filter, cue_filter, trial_filter):
        
        if not all((selected_animals, metric_max)):
            return {}
        
        # animal and session filtering
        data = global_data['UnityTrackwise'].loc[pd.IndexSlice[:,selected_animals, :,:]]
        print(data)
        
        # trial slider filtering
        # n_trials = data['trial_id'].max().item()
        # data = data[data.trial_id.isin(np.arange(int(trial_range[0]), int(trial_range[1]) + 1))]
        
        # list to single value
        if len(smooth_data) == 1:
            smooth_data = True

        fig = kinematics_plot.render_plot(data, metric, metric_max, 
                                                     smooth_data)
        return fig
    
    return html.Div([
        dcc.Store(id='data-loaded', data=False),  # Store to track data loading state
        dbc.Row([
            # Left side for plots
            dbc.Col([
                get_track_graph_component(vis_name),
            ], width=7),

            # Right side for UI controls
            dbc.Col([
                # three rows, header, dropdown/tickpoxes, range slider
                dbc.Row([
                    html.H5(f"Data Selection for {vis_name}", style={"marginTop": 20}),
                ]),                                
                
                dbc.Row([
                    dbc.Col([
                        # Dropdown for paradigm selection
                        *get_paradigm_dropdown_component(vis_name, global_data),
                        # Dropdown for animal selection
                        *get_animal_dropdown_component(vis_name, global_data),
                        # Radioitems for metric selection
                        *get_metric_radioitems_component(vis_name),
                    ], width=4),
                    
                    # Other options in middle column
                    dbc.Col([
                        # Radioitems for group by selection
                        *get_groupby_radioitems_component(vis_name),
                    ], width=4),
                    
                    # Other options in right column
                    dbc.Col([
                        # Filter checklist
                        *get_filter_checklist_component(vis_name),
                        html.Hr(),
                        # Display options checklist
                        *get_display_options_checklist_component(vis_name),
                    ], width=4),
                ]),
                
                # Range slider for trial selection
                # *get_trial_range_slider_component(vis_name)
            ], width=5)
        ]),
        html.Hr()
    ], id=f"{vis_name}-container")  # Initial state is hidden