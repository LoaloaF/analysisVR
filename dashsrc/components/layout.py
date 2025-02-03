from dash import Dash, html
import dash_bootstrap_components as dbc

from . import dashvis_constants as C
from . import session_wise_vis_buttons
from . import multisession_vis_buttons
from . import data_loading_controls

from ..plot_components.plot_wrappers import wrapper_SessionKinematics
from ..plot_components.plot_wrappers import wrapper_AnimalKinematics
from ..plot_components.plot_wrappers import wrapper_StayPerformance
from ..plot_components.plot_wrappers import wrapper_SessionsOverview
from ..plot_components.plot_wrappers import wrapper_StayRatio

def create_sessionwise_vis_containers(app: Dash, global_data: dict):
    viss_row_container = []
    for i, vis_name in enumerate(C.SESSION_WISE_VISS):
        match vis_name:
            case 'SessionKinematics':
                analysis_div = wrapper_SessionKinematics.render(app, global_data, 
                                                                vis_name=vis_name)
            case _:
                analysis_div = html.Div([html.H5(vis_name)], id=f'{vis_name}-container', 
                                        style={'display': 'none', 
                                               'backgroundColor': 'gray'})
                
        viss_row_container.append(analysis_div)
    return html.Div(viss_row_container)

def create_multisession_vis_containers(app: Dash, global_data: dict):
    viss_row_container = []
    for i, vis_name in enumerate(C.ANIMAL_WISE_VISS):
        match vis_name:
            case "Kinematics":
                analysis_div = wrapper_AnimalKinematics.render(app, global_data, 
                                                               vis_name=vis_name)
            case "StayRatio":
                analysis_div = wrapper_StayRatio.render(app, global_data, 
                                                        vis_name=vis_name)
            case "StayPerformance":
                analysis_div = wrapper_StayPerformance.render(app, global_data, 
                                                              vis_name=vis_name)
            case "SessionsOverview":
                analysis_div = wrapper_SessionsOverview.render(app, global_data, 
                                                              vis_name=vis_name)
            case _:
                analysis_div = html.Div([html.H5(vis_name)], id=f'{vis_name}-container', 
                                        style={'display': 'none', 'backgroundColor': 'gray'})
        
        viss_row_container.append(analysis_div)
    return html.Div(viss_row_container)

def create_layout(app: Dash, global_data: dict) -> html.Div:
    return dbc.Container([
        # header row
        dbc.Row([
            dbc.Col([           
                html.H3("VR Sesssion Wise", style={"textAlign": "center"}),
            ], width=2),
            dbc.Col([
                session_wise_vis_buttons.render(app),
            ], width=5),
            dbc.Col([
                data_loading_controls.render(app, global_data)
            ], width=5),
        ], align="center"),
        html.Hr(style={"borderTop": "2px solid #bbb", "marginBottom": "20px"}),
        create_sessionwise_vis_containers(app, global_data),
        
        # header row
        dbc.Row([
            dbc.Col([           
                html.H3("VR Animal Wise", style={"textAlign": "center"}),
            ], width=2),
            dbc.Col([
                multisession_vis_buttons.render(app),
            ], width=7),
        ], align="center"),
        html.Hr(style={"borderTop": "2px solid #bbb", "marginBottom": "20px"}),
        create_multisession_vis_containers(app, global_data),
    ], fluid=True)