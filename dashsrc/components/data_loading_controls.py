from dash import html, dcc, Input, Output, State, Dash
import dash_bootstrap_components as dbc

import pandas as pd

# from data_loading.animal_loading import get_animal_modality
from analysis_core import NAS_DIR
# from .constants import *
from . import constants as C

from analytics_processing import analytics

def render(app: Dash, data: dict) -> html.Div:
    @app.callback(
        Output(C.D2M_LOAD_DATA_BUTTON_ID, 'style'),
        Output('data-loaded', 'data'),
        Input(C.D2M_LOAD_DATA_BUTTON_ID, 'n_clicks'),
        State(C.D2M_ANALYTICS_DROPDOWN_ID, 'value'),
        State(C.D2M_PARADIGMS_DROPDOWN_ID, 'value'),
        State(C.D2M_ANIMALS_DROPDOWN_ID, 'value'),
    )
    def load_data(n_clicks, selected_analytics, selected_paradigms, selected_animals):
        if n_clicks and selected_analytics:
            _load_all_data(selected_analytics, data, selected_paradigms, selected_animals)
            return {"marginTop": 15, "backgroundColor": "green"}, True
        return {"marginTop": 15, "backgroundColor": "blue"}, False
    
    default_animals = [1]
    default_paradigms = [800]
    # default_analytics = ['UnityTrialwiseMetrics', 'SesssionMetadata']
    default_analytics = ['UnityTrackwise', 'SessionMetadata']
    # default_animals = C.ANIMALS[-2:]
    # default_paradigms = C.PARADIGMS
    # default_analytics = list(data.keys())[0:1]
    
    return dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id=C.D2M_ANALYTICS_DROPDOWN_ID,
                        options= list(data.keys()),
                        multi=True,
                        value=default_analytics,
                        placeholder="Select one or more analytics",
                        style={"marginTop": 15}
                    ),
                ], width=4),

                dbc.Col([
                    dcc.Dropdown(
                        id=C.D2M_PARADIGMS_DROPDOWN_ID,
                        options=C.PARADIGMS,
                        multi=True,
                        value=default_paradigms,
                        placeholder="Select paradigms",
                        style={"marginTop": 15}
                    ),
                ], width=3),
       
                dbc.Col([
                    dcc.Dropdown(
                        id=C.D2M_ANIMALS_DROPDOWN_ID,
                        options=C.ANIMALS,
                        multi=True,
                        value=default_animals,
                        placeholder="Select animals",
                        style={"marginTop": 15}
                    ),
                ], width=3),
                
                dbc.Col([
                    dbc.Button("Load Data", id=C.D2M_LOAD_DATA_BUTTON_ID, color="primary", 
                            style={"marginTop": 15}),
                ], width=2, )  # Light red background for debugging
        ])
    
def _load_all_data(selected_analytics, data, selected_paradigms, selected_animals):
    for analytic in selected_analytics:
        print(f"Loading {analytic}...", end=" ")
        if len(selected_paradigms) == 0:
            selected_paradigms = None
        if len(selected_animals) == 0:
            selected_animals = None
        data[analytic] = analytics.get_analytics(analytic, mode='set', session_ids=None,
                                                 paradigm_ids=selected_paradigms,
                                                 animal_ids=selected_animals)
        print(data['UnityTrialwiseMetrics'])
        print("Done.")  