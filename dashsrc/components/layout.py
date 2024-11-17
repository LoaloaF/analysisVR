from dash import Dash, html
import dash_bootstrap_components as dbc

import pandas as pd

# from .analysis_buttons import analysis_buttons
from . import analysis_buttons
from . import data_loading_controls
from .analysis_rows import create_analysis_containers

def create_layout(app: Dash, data: pd.DataFrame) -> html.Div:
    return dbc.Container([
        # header row
        dbc.Row([
            analysis_buttons.render(app, data),
            data_loading_controls.render(app, data),
        ], align="center"),
        
        html.Hr(style={"borderTop": "2px solid #bbb", "marginBottom": "20px"}),
        
        create_analysis_containers(app, data),
        
    ], fluid=True)