from dash import html, dcc, Input, Output, State, Dash
import dash_bootstrap_components as dbc

import pandas as pd

# from data_loading.animal_loading import get_animal_modality
from analysis_core import NAS_DIR
from .constants import *

from analytics_processing import analytics

def render(app: Dash, data: dict) -> html.Div:
    
    @app.callback(
        Output(D2M_LOAD_DATA_BUTTON_ID, 'style'),
        Output('data-loaded', 'data'),
        Input(D2M_LOAD_DATA_BUTTON_ID, 'n_clicks'),
        State(D2M_MODALITY_DROPDOWN_ID, 'value'),
    )
    def load_data(n_clicks, selected_modalities):
        if n_clicks and selected_modalities:
            _load_all_data(selected_modalities, data)
            return {"marginTop": 15, "backgroundColor": "green"}, True
        return {"marginTop": 15, "backgroundColor": "blue"}, False
    
    return dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id=D2M_MODALITY_DROPDOWN_ID,
                    options= list(data.keys()),
                    multi=True,
                    value=list(data.keys())[0:1],
                    placeholder="Select one or more analytics",
                    style={"marginTop": 15}
                ),
            ], width=8, ),  # Light blue background for debugging
            dbc.Col([
                dbc.Button("Load Data", id=D2M_LOAD_DATA_BUTTON_ID, color="primary", 
                           style={"marginTop": 15}),
            ], width=4, )  # Light red background for debugging
        ])
    
def _load_all_data(selected_modalities, data):
    for analytic in selected_modalities:
        print(f"Loading {analytic}...", end=" ")
        data[analytic] = analytics.get_analytics(analytic, mode='set',  
                                                 animal_ids=[1,2,3], session_ids=None, 
                                                 paradigm_ids=[800])
                                                #  animal_ids=[9,], session_ids=[10,5,4], paradigm_ids=[1100])
        print("Done.")