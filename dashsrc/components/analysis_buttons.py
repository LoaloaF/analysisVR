from dash import html, Input, Output, Dash
import dash_bootstrap_components as dbc

import pandas as pd

from .constants import *

def render(app: Dash) -> html.Div:
    @app.callback(
        [(Output(f'{analysis}-container', 'style'), Output(f'{analysis}-button', 'active')) for i, analysis in enumerate(ROW1_ANALYSISS)],
        [Input(f'{analysis}-button', 'n_clicks') for analysis in ROW1_ANALYSISS]
    )
    def toggle_rows(*args):
        outputs = []
        for i, n_clicks in enumerate(args):
            if n_clicks and n_clicks % 2 != 0:  # Odd clicks = row visible, button active
                outputs.append(({'display': 'block'}, True))
            else:  # Even clicks = row hidden, button inactive
                outputs.append(({'display': 'none'}, False))
        return outputs
    
    return dbc.ButtonGroup([
            dbc.Button(analysis_name, id=analysis_name+'-button', color="primary", outline=True) 
            for analysis_name in ROW1_ANALYSISS
        ], vertical=False, style={"marginTop": "10px", "marginBottom": "5px"})