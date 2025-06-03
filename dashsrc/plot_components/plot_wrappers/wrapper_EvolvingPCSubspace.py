from dash import html, dcc, Input, Output, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from .. .components.dcc_graphs import get_general_graph_component
from .data_selection_components import (
    paradigm_dropdown_component,
    animal_dropdown_component,
    session_range_slider_component,
    figure_width_input_component,
    figure_height_input_component,
    predictor_dropdown_component,
    zone_dropdown_component,
    digit_input_component,
    PCs_slider_component,
    
    register_animal_dropdown_callback,
    register_session_slider_callback,
    register_paradigm_dropdown_callback,
    register_predictor_dropdown_callback,
    register_zone_dropdown_callback,
    register_PCs_slider_callback,
)
from ..plots import plot_EvolvingPCSubspace

import dashsrc.components.dashvis_constants as C

def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
    analytic = 'PCsSubspaceAngles'
    # components with data depedency need these arguments
    comp_args = vis_name, global_data, analytic
    
    # Register the callbacks
    register_paradigm_dropdown_callback(app, *comp_args)
    register_animal_dropdown_callback(app, *comp_args)
    register_session_slider_callback(app, *comp_args, default_select_sessions=(7, 26)) 
    register_predictor_dropdown_callback(app, *comp_args)
    register_zone_dropdown_callback(app, *comp_args)
    register_PCs_slider_callback(app, *comp_args)
    
    # create the html components to have their IDs (needed for the callbacks)
    paradigm_dropd, PARADIGM_DROPD_ID = paradigm_dropdown_component(*comp_args)
    animal_dropd, ANIMAL_DROPD_ID = animal_dropdown_component(*comp_args)
    
    predictor_dropd, PREDICTOR_DROPD_ID = predictor_dropdown_component(*comp_args)
    zone_dropd, ZONE_DROPD_ID = zone_dropdown_component(*comp_args)
    
    # these don't need data to be initialized    
    session_slider, SESSION_SLIDER_ID = session_range_slider_component(vis_name)
    PCs_slider, PCS_SLIDER_ID = PCs_slider_component(vis_name)
    
    max_metric, MAX_METRIC_ID = digit_input_component(vis_name, "Max angle", 
                                                      initial_value=.6, step=0.01,)
    height_input, HEIGHT_INP_ID = figure_height_input_component(vis_name)
    width_input, WIDTH_INP_ID = figure_width_input_component(vis_name)

    graph, GRAPH_ID = get_general_graph_component(vis_name, )
    
    @app.callback(
        Output(GRAPH_ID, 'figure'),
        Input(PARADIGM_DROPD_ID, 'value'),
        Input(ANIMAL_DROPD_ID, 'value'),
        Input(SESSION_SLIDER_ID, 'value'),
        Input(PREDICTOR_DROPD_ID, 'value'),
        Input(ZONE_DROPD_ID, 'value'),
        Input(PCS_SLIDER_ID, 'value'),
        Input(MAX_METRIC_ID, 'value'),
        Input(HEIGHT_INP_ID, 'value'),
        Input(WIDTH_INP_ID, 'value'),
        )
    def update_plot(selected_paradigm, selected_animal, session_range,
                    selected_predictor, selected_zone, PCs_range,
                    max_metric, height, width):
    
        if not all((selected_paradigm, selected_animal, selected_predictor, selected_zone)):
            return {}
        
        paradigm_slice = slice(selected_paradigm, selected_paradigm)
        animal_slice = slice(selected_animal, selected_animal)
        session_slice = [sid for sid in np.arange(session_range[0], session_range[1] + 1)
                         if sid in global_data[analytic].index.unique('session_id')]
        
        # paradigm, animal and session filtering
        data = global_data[analytic].loc[pd.IndexSlice[paradigm_slice, animal_slice, 
                                                       session_slice, :]]
        
        # metadata
        metadata = global_data['SessionMetadata'].loc[pd.IndexSlice[paradigm_slice, 
                                                                    animal_slice, 
                                                                    session_slice]]
        
        # reshape the data to have a multi-index with predictor and track_zone
        
        data = data[np.isin(data['comp_session_id'], session_slice)]
        data.index = data.index.droplevel((0, 1, 3))
        data.set_index(["comp_session_id", 'predictor', 'track_zone'], 
                       inplace=True, append=True)
        data = data.loc[pd.IndexSlice[:, :, [selected_predictor], [selected_zone]]]
        
        # slice the PCs
        data = data.iloc[:, PCs_range[0]:PCs_range[1]]
        
        fig = plot_EvolvingPCSubspace.render_plot(data, metadata, selected_predictor, 
                                                  selected_zone, max_metric, height, width)
        return fig
    
    return html.Div([
        dcc.Store(id=C.get_vis_name_data_loaded_id(vis_name), data=False),  # Store to track data loading state
        dbc.Row([
            # Left side for plots
            dbc.Col([
                graph,
            ], width=10),
            # Right side for UI controls
            dbc.Col([
                # three rows, header, dropdown/tickpoxes, range slider
                dbc.Row([
                    html.H5(f"Data Selection for {vis_name}", style={"marginTop": 20}),
                ]),                                
                
                dbc.Row([
                    dbc.Col([
                        # Dropdown for paradigm selection, animal selection
                        *paradigm_dropd, *animal_dropd,
                    ], width=4),
                    
                    # Other options in right column
                    dbc.Col([
                        *predictor_dropd, *zone_dropd,
                    ], width=4),

                    # Other options in right column
                    dbc.Col([
                        *width_input, *height_input, *max_metric,
                    ], width=4),
                ]),
                # Range slider for session selection
                *session_slider,
                *PCs_slider
            ], width=2)
        ]),
        html.Hr()
    ], id=f"{vis_name}-container")  # Initial state is hidden