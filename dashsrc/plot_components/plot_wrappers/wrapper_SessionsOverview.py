from dash import html, dcc, Input, Output, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from .. .components.dcc_graphs import get_general_graph_component
from .data_selection_components import (
    date_range_slider_component,
    from_hour_input_component,
    to_hour_input_component,
    plot_type_selecter_component,
    animal_dropdown_component,
    
    register_animal_dropdown_callback,
)
from .data_selection import group_filter_data
from ..plots import plot_SessionsOverview

import dashsrc.components.dashvis_constants as C

def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
    analytic = 'SessionMetadata'
    data_loaded_id = C.get_vis_name_data_loaded_id(vis_name)
    
    # Register the callbacks
    register_animal_dropdown_callback(app, vis_name, global_data, analytic)
    
    # create the html components to have their IDs (needed for the callbacks)
    from_hour_input, FROM_HOUR_INP_ID = from_hour_input_component(vis_name)
    to_hour_input, TO_HOUR_INP_ID = to_hour_input_component(vis_name)
    date_range_slider, DATE_RANGE_SLIDER_ID = date_range_slider_component(vis_name)
    animal_dropdown, ANIMAL_DROPD_ID = animal_dropdown_component(vis_name, global_data, analytic)
    plot_type_selecter, PLOT_TYPE_SELECT_ID = plot_type_selecter_component(vis_name)
    
    graph, GRAPH_ID = get_general_graph_component(vis_name, fixed_height=800)
    
    @app.callback(
        Output(GRAPH_ID, 'figure'),
        Input(data_loaded_id, 'data'),
        Input(FROM_HOUR_INP_ID, 'value'),
        Input(TO_HOUR_INP_ID, 'value'),
        Input(DATE_RANGE_SLIDER_ID, 'start_date'),
        Input(DATE_RANGE_SLIDER_ID, 'end_date'),
        Input(ANIMAL_DROPD_ID, 'value'),
        Input(PLOT_TYPE_SELECT_ID, 'value'),
        )
    def update_plot(data_loaded, from_hour, to_hour, start_date, end_date, 
                    selected_animal, plot_type):
        if not data_loaded:
            return {}
        
        animal_slice = slice(None)
        if plot_type == 'Modalities':
            animal_slice = slice(selected_animal, selected_animal)
        data = global_data[analytic].loc[pd.IndexSlice[:, animal_slice, :, :]]
        
        fig = plot_SessionsOverview.render_plot(data, from_hour, 
                                                to_hour, start_date, end_date,
                                                plot_type)
        return fig
    
    return html.Div([
        dcc.Store(id=data_loaded_id, data=False),  # Store to track data loading state
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
                        html.Label("From, to hour", style={"marginTop": 15}),
                        *from_hour_input, *to_hour_input
                    ]),
                ]),
                
                dbc.Row([
                    *date_range_slider
                ]),
                dbc.Row([
                    *plot_type_selecter,
                ]),
                dbc.Row([
                    *animal_dropdown,
                ]),
            ], width=2),
        ]),
    ], id=f"{vis_name}-container")  # Initial state is hidden
    