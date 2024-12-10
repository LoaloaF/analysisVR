from dash import Dash, html
import dash_bootstrap_components as dbc

import pandas as pd

from . import session_wise_vis_buttons
from . import multisession_vis_buttons
from . import data_loading_controls

import dashsrc.components.trial_wise_kinematics as trial_wise_kinematics
import dashsrc.components.kinematics as kinematics
import dashsrc.components.constants as C

def create_sessionwise_vis_containers(app: Dash, global_data: dict):
    viss_row_container = []
    for i, vis_name in enumerate(C.SESSION_WISE_VISS):
        match vis_name:
            case 'Trial-Wise-Kinematics':
                analysis_div = trial_wise_kinematics.render(app, global_data, vis_name=vis_name)
            case _:
                analysis_div = html.Div([html.H5(vis_name)], id=f'{vis_name}-container', 
                                        style={'display': True, 'backgroundColor': 'gray'})
                
        viss_row_container.append(analysis_div)
    return html.Div(viss_row_container, id='vis_name-container')

def create_multisession_vis_containers(app: Dash, global_data: dict):
    viss_row_container = []
    for i, vis_name in enumerate(C.MULTISESSION_VISS):
        match vis_name:
            case "Kinematics":
                analysis_div = kinematics.render(app, global_data, vis_name=vis_name)
            case _:
                analysis_div = html.Div([html.H5(vis_name)], id=f'{vis_name}-container', 
                                        style={'display': 'none', 'backgroundColor': 'gray'})
        
        viss_row_container.append(analysis_div)
    return html.Div(viss_row_container, id='vis_name-container')

def create_layout(app: Dash, global_data: dict) -> html.Div:
    return dbc.Container([
        # header row
        dbc.Row([
            dbc.Col([           
                html.H3("VR Sesssion Wise", style={"textAlign": "center"}),
            ], width=2),
            dbc.Col([
                session_wise_vis_buttons.render(app),
            ], width=7),
            dbc.Col([
                data_loading_controls.render(app, global_data)
            ], width=3),
        ], align="center"),
        html.Hr(style={"borderTop": "2px solid #bbb", "marginBottom": "20px"}),
        create_sessionwise_vis_containers(app, global_data),
        
        # header row
        dbc.Row([
            dbc.Col([           
                html.H3("VR Multi-Session", style={"textAlign": "center"}),
            ], width=2),
            dbc.Col([
                multisession_vis_buttons.render(app),
            ], width=7),
        ], align="center"),
        html.Hr(style={"borderTop": "2px solid #bbb", "marginBottom": "20px"}),
        create_multisession_vis_containers(app, global_data),
    ], fluid=True)