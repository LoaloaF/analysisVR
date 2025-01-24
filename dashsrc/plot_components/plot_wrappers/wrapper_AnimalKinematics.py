from dash import html, dcc, Input, Output, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from .. .components.dcc_graphs import get_general_graph_component
from .data_selection_components import (
    paradigm_dropdown_component,
    animal_dropdown_component,
    session_range_slider_component,
    metric_radioitems_component,
    outcome_group_filter_component,
    cue_group_filter_component,
    trial_group_filter_component,
    max_metric_input_component,
    smooth_checklist_component,
    figure_width_input_component,
    figure_height_input_component,
    
    register_animal_dropdown_callback,
    register_session_slider_callback,
    register_paradigm_dropdown_callback,
)
from .data_selection import group_filter_data
from ..plots import plot_AnimalKinematics

import dashsrc.components.dashvis_constants as C

def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
    analytic = 'UnityTrackwise'
    # components with data depedency need these arguments
    comp_args = vis_name, global_data, analytic
    
    # Register the callbacks
    register_paradigm_dropdown_callback(app, *comp_args)
    register_animal_dropdown_callback(app, *comp_args)
    register_session_slider_callback(app, *comp_args)
    
    # create the html components to have their IDs (needed for the callbacks)
    paradigm_dropd, PARADIGM_DROPD_ID = paradigm_dropdown_component(*comp_args)
    animal_dropd, ANIMAL_DROPD_ID = animal_dropdown_component(*comp_args)
    # these don't need data to be initialized    
    session_slider, SESSION_SLIDER_ID = session_range_slider_component(vis_name)
    metrics_radioi, METRICS_RADIOI_ID = metric_radioitems_component(vis_name)
    maxmetric_inp, MAXMETRIC_INP_ID = max_metric_input_component(vis_name, initial_value=80)
    smooth_checkl, SMOOTH_CHECKL_ID = smooth_checklist_component(vis_name)

    outcome_filter, OUTCOME_FILTER_ID = outcome_group_filter_component(vis_name)
    cue_filter, CUE_FILTER_ID = cue_group_filter_component(vis_name)
    trial_filter, TRIAL_FILTER_ID = trial_group_filter_component(vis_name)
    height_input, HEIGHT_INP_ID = figure_width_input_component(vis_name)
    width_input, WIDTH_INP_ID = figure_height_input_component(vis_name)

    graph, GRAPH_ID = get_general_graph_component(vis_name)
    
    
    @app.callback(
        Output(GRAPH_ID, 'figure'),
        Input(PARADIGM_DROPD_ID, 'value'),
        Input(ANIMAL_DROPD_ID, 'value'),
        Input(SESSION_SLIDER_ID, 'value'),

        Input(METRICS_RADIOI_ID, 'value'),
        Input(MAXMETRIC_INP_ID, 'value'),
        Input(SMOOTH_CHECKL_ID, 'value'),

        Input(OUTCOME_FILTER_ID, 'value'),
        Input(CUE_FILTER_ID, 'value'),
        Input(TRIAL_FILTER_ID, 'value'),
        Input(WIDTH_INP_ID, 'value'),
        Input(HEIGHT_INP_ID, 'value'),
        )
    def update_plot(selected_paradigm, selected_animal, session_range,
                    metric, metric_max, smooth_data,
                    outcome_filter, cue_filter, trial_filter,
                    width, height):
    
        if not all((selected_paradigm, selected_animal, metric_max)):
            return {}
        
        paradigm_slice = slice(selected_paradigm, selected_paradigm)
        animal_slice = slice(selected_animal, selected_animal)
        session_slice = [sid for sid in np.arange(session_range[0], session_range[1] + 1)
                         if sid in global_data['UnityTrackwise'].index.unique('session_id')]
        
        # paradigm, animal and session filtering
        data = global_data[analytic].loc[pd.IndexSlice[paradigm_slice, animal_slice, 
                                                       session_slice, :]]
        # filter the data based on the group by values
        data, _ = group_filter_data(data, outcome_filter, cue_filter, 
                                    trial_filter)
        
        # list to single value
        if len(smooth_data) == 1:
            smooth_data = True
            
        fig = plot_AnimalKinematics.render_plot(data, global_data['SessionMetadata'], 
                                                metric, metric_max, 
                                                smooth_data, width, height)
        return fig
    
    return html.Div([
        dcc.Store(id=C.DATA_LOADED_Kinematics_ID, data=False),  # Store to track data loading state
        dbc.Row([
            # Left side for plots
            dbc.Col([
                graph,
            ], width=8),
            # Right side for UI controls
            dbc.Col([
                # three rows, header, dropdown/tickpoxes, range slider
                dbc.Row([
                    html.H5(f"Data Selection for {vis_name}", style={"marginTop": 20}),
                ]),                                
                
                dbc.Row([
                    dbc.Col([
                        # Dropdown for paradigm selection, animal selection
                        *paradigm_dropd, *animal_dropd, *metrics_radioi,
                    ], width=4),
                    
                    # Other options in right column
                    dbc.Col([
                        *outcome_filter, *cue_filter, *trial_filter,
                        html.Hr(),
                        *maxmetric_inp, *smooth_checkl,
                    ], width=4),

                    # Other options in right column
                    dbc.Col([
                        *width_input, *height_input,
                    ], width=4),
                ]),
                # Range slider for session selection
                *session_slider
            ], width=4)
        ]),
        html.Hr()
    ], id=f"{vis_name}-container")  # Initial state is hidden