from dash import html, dcc, Input, Output, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

import dashsrc.components.dashvis_constants as C

from .. .components.dcc_graphs import get_general_graph_component
from .data_selection_components import (
    animal_dropdown_component,
    session_dropdown_component,
    metric_radioitems_component,
    groupby_radioitems_component,
    variance_radioitems_component,
    outcome_group_filter_component,
    cue_group_filter_component,
    trial_group_filter_component,
    max_metric_input_component,
    smooth_checklist_component,
    
    register_animal_dropdown_callback,
    register_session_dropdown_callback,
    register_paradigm_dropdown_callback,
)
from .data_selection import group_filter_data
from ..plots import plot_SessionKinematics

def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
    analytic = 'UnityTrackwise'
    # components with data depedency need these arguments
    comp_args = vis_name, global_data, analytic
    
    # Register the callbacks
    register_animal_dropdown_callback(app, *comp_args)
    register_session_dropdown_callback(app, *comp_args)
    
    # create the html components to have their IDs (needed for the callbacks)
    animal_dropd, ANIMAL_DROPD_ID = animal_dropdown_component(*comp_args)
    session_dropd, SESSION_DROPD_ID = session_dropdown_component(*comp_args)
    # these don't need data to be initialized    
    groupby_radioi, GROUPBY_RADIOI_ID = groupby_radioitems_component(vis_name)
    metrics_radioi, METRICS_RADIOI_ID = metric_radioitems_component(vis_name)
    maxmetric_inp, MAXMETRIC_INP_ID = max_metric_input_component(vis_name, initial_value=80)
    smooth_checkl, SMOOTH_CHECKL_ID = smooth_checklist_component(vis_name)
    varianve_radioi, VARIANCE_RADIOI_ID = variance_radioitems_component(vis_name)
    outcome_filter, OUTCOME_FILTER_ID = outcome_group_filter_component(vis_name)
    cue_filter, CUE_FILTER_ID = cue_group_filter_component(vis_name)
    trial_filter, TRIAL_FILTER_ID = trial_group_filter_component(vis_name)

    graph, GRAPH_ID = get_general_graph_component(vis_name)
    
    @app.callback(
        Output(GRAPH_ID, 'figure'),
        Input(ANIMAL_DROPD_ID, 'value'),
        Input(SESSION_DROPD_ID, 'value'),
        Input(GROUPBY_RADIOI_ID, 'value'),
        Input(METRICS_RADIOI_ID, 'value'),
        Input(MAXMETRIC_INP_ID, 'value'),
        Input(SMOOTH_CHECKL_ID, 'value'),
        Input(VARIANCE_RADIOI_ID, 'value'),
        Input(OUTCOME_FILTER_ID, 'value'),
        Input(CUE_FILTER_ID, 'value'),
        Input(TRIAL_FILTER_ID, 'value'),
    )
    def update_plot(selected_animal, selected_session, group_by, 
                    metric, metric_max, smooth_data, var_viz,
                    outcome_filter, cue_filter, trial_filter):
        
        # only render the plot if all the necessary data selections are made   
        if not all((selected_animal, (selected_session is not None), metric_max)):
            return {}
        
        paradigm_slice = slice(None)
        animal_slice = slice(selected_animal, selected_animal)
        session_slice = slice(selected_session, selected_session)
        
        # paradigm, animal and session filtering
        data = global_data[analytic].loc[pd.IndexSlice[paradigm_slice, animal_slice, 
                                                       session_slice, :]]
        
        # print("----")
        # print(data)
        
        # check max trial id for the session, before potential filtering
        n_trials = data['trial_id'].max().item()
        # filter the data based on the group by values
        data, group_by_values = group_filter_data(data, outcome_filter, cue_filter, 
                                                  trial_filter, group_by=group_by)
        
        print("----")
        print(group_by_values)
        print("----")
                
        # selected_sessions = np.arange(int(session_range[0]), int(session_range[1]) + 1)
        # # TODO fix later
        # selected_sessions = [s for s in selected_sessions if s in global_data['UnityTrackwise'].index.unique('session_id')]
        
        # list to single value
        if len(var_viz) == 1:
            var_viz = var_viz[0]
        if len(smooth_data) == 1:
            smooth_data = True
        
        fig = plot_SessionKinematics.render_plot(data, global_data['SessionMetadata'],
                                                 n_trials, group_by, group_by_values,
                                                 metric, metric_max, 
                                                 smooth_data, var_viz)
        return fig
    
    return html.Div([
        dcc.Store(id=C.get_vis_name_data_loaded_id(vis_name), data=False),  # Store to track data loading state
        dbc.Row([
            # Left side for plots
            dbc.Col([
                graph,
            ], width=7),
            # Right side for UI controls
            dbc.Col([
                # three rows, header, dropdown/tickpoxes, range slider
                dbc.Row([
                    html.H5(f"Data Selection for {vis_name}", style={"marginTop": 20}),
                ]),                                
                
                dbc.Row([
                    dbc.Col([
                        # Dropdown for animal session and metric selection
                        *animal_dropd, *session_dropd, *metrics_radioi,
                    ], width=4),
                    
                    # Other options in middle column
                    dbc.Col([
                        # Radioitems for group by selection and variance visualization
                        *groupby_radioi, *varianve_radioi,
                    ], width=4),
                    
                    # Other options in right column
                    dbc.Col([
                        # Filter checklist, max metric and smooth data
                        *outcome_filter, *cue_filter, *trial_filter,
                        html.Hr(),
                        *maxmetric_inp, *smooth_checkl,
                    ], width=4),
                ]),
            ], width=5)
        ]),
        html.Hr()
    ], id=f"{vis_name}-container")  # Initial state is hidden
    
    
    