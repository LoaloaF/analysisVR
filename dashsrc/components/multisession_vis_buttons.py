from dash import html, Input, Output, Dash
import dash_bootstrap_components as dbc

import pandas as pd

from . import constants as C

def render(app: Dash) -> html.Div:
    @app.callback(
        [(Output(f'{vis_name}-container', 'style'), Output(f'{vis_name}-button', 'active')) 
         for i, vis_name in enumerate(C.MULTISESSION_VISS)],
        [Input(f'{vis_name}-button', 'n_clicks') for vis_name in C.MULTISESSION_VISS]
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
            for analysis_name in C.MULTISESSION_VISS
        ], vertical=False, style={"marginTop": "10px", "marginBottom": "5px"})