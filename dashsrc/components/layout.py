from dash import Dash, html, dcc
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
from ..plot_components.plot_wrappers import wrapper_EvolvingStayTime
from ..plot_components.plot_wrappers import wrapper_EvolvingStayDecision
from ..plot_components.plot_wrappers import wrapper_RawSpikes
from ..plot_components.plot_wrappers import wrapper_TrackFiringRate
from ..plot_components.plot_wrappers import wrapper_SVMPredictions
from ..plot_components.plot_wrappers import wrapper_EvolvingPCSubspace
from ..plot_components.plot_wrappers import wrapper_CueCorrelation

def create_sessionwise_vis_containers(app: Dash, loaded_analytics: dict, loaded_raw_traces: dict):
    viss_row_container = []
    for i, vis_name in enumerate(C.SESSION_WISE_VISS):
        match vis_name:
            case 'SessionKinematics':
                analysis_div = wrapper_SessionKinematics.render(app, loaded_analytics, 
                                                                vis_name=vis_name)
            case "RawSpikes":
                analysis_div = wrapper_RawSpikes.render(app, loaded_analytics, 
                                                        loaded_raw_traces,
                                                        vis_name=vis_name)
            case "SVMPredictions":
                analysis_div = wrapper_SVMPredictions.render(app, loaded_analytics, 
                                                             vis_name=vis_name)
            case "CueCorrelation":
                analysis_div = wrapper_CueCorrelation.render(app, loaded_analytics, 
                                                             vis_name=vis_name)
            case _:
                analysis_div = html.Div([html.H5(vis_name)], id=f'{vis_name}-container', 
                                        style={'display': 'none', 
                                               'backgroundColor': 'gray'})
                
        viss_row_container.append(analysis_div)
    return html.Div(viss_row_container)

def create_multisession_vis_containers(app: Dash, loaded_analytics: dict):
    viss_row_container = []
    for i, vis_name in enumerate(C.ANIMAL_WISE_VISS):
        match vis_name:
            case "Kinematics":
                analysis_div = wrapper_AnimalKinematics.render(app, loaded_analytics, 
                                                               vis_name=vis_name)
            case "StayRatio":
                analysis_div = wrapper_StayRatio.render(app, loaded_analytics, 
                                                        vis_name=vis_name)
            case "StayPerformance":
                analysis_div = wrapper_StayPerformance.render(app, loaded_analytics, 
                                                              vis_name=vis_name)
            case "SessionsOverview":
                analysis_div = wrapper_SessionsOverview.render(app, loaded_analytics, 
                                                              vis_name=vis_name)
            case "EvolvingStayTime":
                analysis_div = wrapper_EvolvingStayTime.render(app, loaded_analytics, 
                                                              vis_name=vis_name)
            case "EvolvingStayDecision":
                analysis_div = wrapper_EvolvingStayDecision.render(app, loaded_analytics, 
                                                              vis_name=vis_name)
            case "TrackFiringRate":
                analysis_div = wrapper_TrackFiringRate.render(app, loaded_analytics,
                                                              vis_name=vis_name)
            case "EvolvingPCSubspace":
                analysis_div = wrapper_EvolvingPCSubspace.render(app, loaded_analytics,
                                                              vis_name=vis_name)
            
            case _:
                analysis_div = html.Div([html.H5(vis_name)], id=f'{vis_name}-container', 
                                        style={'display': 'none', 'backgroundColor': 'gray'})
        
        viss_row_container.append(analysis_div)
    return html.Div(viss_row_container)

def create_layout(app: Dash, loaded_analytics: dict, loaded_raw_traces: dict) -> html.Div:
    return dbc.Container([
        # header row
        dbc.Row([
            dbc.Col([           
                html.H3("VR Sesssion Wise", style={"textAlign": "center"}),
            ], width=2),
            dbc.Col([
                session_wise_vis_buttons.render(app),
            ], width=4),
            dbc.Col([
                data_loading_controls.render(app, loaded_analytics, loaded_raw_traces)
            ], width=6),
        ], align="center"),
        html.Hr(style={"borderTop": "2px solid #bbb", "marginBottom": "20px"}),
        create_sessionwise_vis_containers(app, loaded_analytics, loaded_raw_traces),
        
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
        create_multisession_vis_containers(app, loaded_analytics),
    ], fluid=True)