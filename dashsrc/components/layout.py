from dash import Dash, html
import dash_bootstrap_components as dbc

import pandas as pd

from . import analysis_buttons
from . import data_loading_controls

import dashsrc.components.trial_wise_kinematics as trial_wise_kinematics
import dashsrc.components.constants as C

def create_analysis_containers(app: Dash, global_data: dict):
    analysis_row_containers = []
    for i, analysis in enumerate(C.ROW1_ANALYSISS):
        match analysis:
            case 'Trial-Wise-Kinematics':
                analysis_div = trial_wise_kinematics.render(app, global_data, vis_name=analysis)
            case _:
                analysis_div = html.Div([html.H5(analysis)], id=f'{analysis}-container', style={'display': 'none', 'backgroundColor': 'gray'})
                
        analysis_row_containers.append(analysis_div)
    return html.Div(analysis_row_containers, id='analysis-container')

def create_layout(app: Dash, global_data: dict) -> html.Div:
    return dbc.Container([
        # header row
        dbc.Row([
            dbc.Col([
                html.H3("VR Sesssion Wise", style={"textAlign": "center"}),
            ], width=2),
            dbc.Col([
                analysis_buttons.render(app),
            ], width=7),
            dbc.Col([
                data_loading_controls.render(app, global_data)
            ], width=3),
        ], align="center"),
        
        html.Hr(style={"borderTop": "2px solid #bbb", "marginBottom": "20px"}),
        
        create_analysis_containers(app, global_data),
        
    ], fluid=True)