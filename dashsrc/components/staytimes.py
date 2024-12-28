from dash import html, dcc, Input, Output, State, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
# from data_loading.animal_loading import get_animal_modality
# from analysis_core import get_default_nas_dir
from dashsrc.plots import staytimes_plot
from .constants import *
from .plot_ui_components import (
    get_animal_dropdown_component,
    get_paradigm_dropdown_component,
    get_metric_radioitems_component,
    get_groupby_radioitems_component,
    get_filter_checklist_component,
    get_display_options_checklist_component,
    get_session_range_slider_component,
    # get_trial_range_slider_component,
    register_animal_dropdown_callback,
    register_paradigm_dropdown_callback,
    register_session_slider_callback
)
from .plot_components import get_track_graph_component

def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
    # Register the callbacks
    register_paradigm_dropdown_callback(app, global_data, 'UnityTrialwiseMetrics', vis_name)
    register_animal_dropdown_callback(app, global_data, 'UnityTrialwiseMetrics', vis_name)
    register_session_slider_callback(app, global_data, 'UnityTrialwiseMetrics', vis_name)

    @app.callback(
        Output(f'figure-{vis_name}', 'figure'),
        Input(f'animal-dropdown-{vis_name}', 'value'),
        Input(f'session-range-slider-{vis_name}', 'value'),
        # Input(f'group-by-{vis_name}', 'value'),
        # Input(f'metric-{vis_name}', 'value'),
        Input(f'max-metric-value-{vis_name}', 'value'),
        # Input(f'smooth-data-{vis_name}', 'value'),
        Input(f'outcome-group-filter-{vis_name}', 'value'),
        Input(f'cue-group-filter-{vis_name}', 'value'),
    )
    def update_plot(selected_animals, session_range, # group_by, metric,
                    metric_max, outcome_filter, cue_filter):
        
        if not all((selected_animals, metric_max)):
            return {}
        
        selected_sessions = np.arange(int(session_range[0]), int(session_range[1]) + 1)
        # TODO fix later 
        selected_sessions = [s for s in selected_sessions if s in global_data['UnityTrialwiseMetrics'].index.unique('session_id')]
        # animal and session filtering
        data = global_data['UnityTrialwiseMetrics'].loc[pd.IndexSlice[:,selected_animals,selected_sessions,:]]
        
        # list to single value
        # if len(smooth_data) == 1:
        #     smooth_data = True

        fig = staytimes_plot.render_plot(data, metric_max, )
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
                        *get_paradigm_dropdown_component(vis_name, global_data, 'UnityTrialwiseMetrics'),
                        # Dropdown for animal selection
                        *get_animal_dropdown_component(vis_name, global_data, 'UnityTrialwiseMetrics'),
                        # Radioitems for metric selection
                        # *get_metric_radioitems_component(vis_name),
                    ], width=4),
                    
                    # Other options in middle column
                    dbc.Col([
                        # Radioitems for group by selection
                        *get_groupby_radioitems_component(vis_name),
                    ], width=4),
                    
                    # Other options in right column
                    dbc.Col([
                        # Filter checklist
                        *get_filter_checklist_component(vis_name, with_trial_group=False),
                        html.Hr(),
                        # Display options checklist, smooth and max metric
                        *get_display_options_checklist_component(vis_name, with_smooth=False,
                                                                 initial_value=8),
                    ], width=4),
                ]),
                
                # Range slider for trial selection
                *get_session_range_slider_component(vis_name),
            ], width=5)
        ]),
        html.Hr()
    ], id=f"{vis_name}-container")  # Initial state is hidden