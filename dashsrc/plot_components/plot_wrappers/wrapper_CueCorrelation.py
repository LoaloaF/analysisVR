from dash import html, dcc, Input, Output, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

import dashsrc.components.dashvis_constants as C

from .. .components.dcc_graphs import get_general_graph_component
from .data_selection_components import (
    animal_dropdown_component,
    session_dropdown_component,
    predictor_dropdown_component,
    paradigm_dropdown_component,
    figure_width_input_component,
    figure_height_input_component,
    smooth_checklist_component,
    
    register_animal_dropdown_callback,
    register_session_dropdown_callback,
    register_predictor_dropdown_callback,
    register_paradigm_dropdown_callback,
)
from .data_selection import group_filter_data
from ..plots import plot_CueCorrelation

def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
    analytic = 'PVCueCorr'
    # components with data depedency need these arguments
    comp_args = vis_name, global_data, analytic
    
    # Register the callbacks
    register_paradigm_dropdown_callback(app, *comp_args)
    register_animal_dropdown_callback(app, *comp_args)
    register_session_dropdown_callback(app, *comp_args)
    register_predictor_dropdown_callback(app, *comp_args)
    
    # create the html components to have their IDs (needed for the callbacks)
    paradigm_dropd, PARADIGM_DROPD_ID = paradigm_dropdown_component(*comp_args)
    animal_dropd, ANIMAL_DROPD_ID = animal_dropdown_component(*comp_args)
    session_dropd, SESSION_DROPD_ID = session_dropdown_component(*comp_args)
    predictor_dropd, PREDICTOR_DROPD_ID = predictor_dropdown_component(*comp_args)
    smooth_checklist, SMOOTH_CHECKLIST_ID = smooth_checklist_component(vis_name)
    width_input, WIDTH_INP_ID = figure_width_input_component(vis_name, 850)
    height_input, HEIGHT_INP_ID = figure_height_input_component(vis_name, 700)

    graph, GRAPH_ID = get_general_graph_component(vis_name)
    
    @app.callback(
        Output(GRAPH_ID, 'figure'),
        Input(PARADIGM_DROPD_ID, 'value'),
        Input(ANIMAL_DROPD_ID, 'value'),
        Input(SESSION_DROPD_ID, 'value'),
        Input(PREDICTOR_DROPD_ID, 'value'),
        Input(SMOOTH_CHECKLIST_ID, 'value'),
        Input(WIDTH_INP_ID, 'value'),
        Input(HEIGHT_INP_ID, 'value'),
    )
    def update_plot(paradigm_dropd,  selected_animal, selected_session, predictor,
                    smooth, width, height):
        
        # only render the plot if all the necessary data selections are made   
        if not all((selected_animal, (selected_session is not None), predictor)):
            return {}
        
        paradigm_slice = slice(paradigm_dropd, paradigm_dropd)
        animal_slice = slice(selected_animal, selected_animal)
        session_slice = slice(selected_session, selected_session)
        
        # paradigm, animal and session filtering
        data = global_data[analytic].loc[pd.IndexSlice[paradigm_slice, animal_slice, 
                                                       session_slice, :]]
        data = data[data.predictor == predictor]
        print("----")
        print(data)
        fig = plot_CueCorrelation.render_plot(data, global_data['SessionMetadata'],
                                                width=width, height=height, smooth=smooth,
                                              )
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
                        # Dropdown for animal session and metric selection
                        *paradigm_dropd, *animal_dropd, *smooth_checklist,
                    ], width=6),
                    
                    # Other options in middle column
                    dbc.Col([
                        # Radioitems for group by selection and variance visualization
                        *session_dropd, *predictor_dropd, * width_input, *height_input,
                    ], width=6),
                    
                    
                ]),
            ], width=2)
        ]),
        html.Hr()
    ], id=f"{vis_name}-container")  # Initial state is hidden
    
    
    