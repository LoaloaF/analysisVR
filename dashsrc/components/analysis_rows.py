from dash import html, Dash

import pandas as pd

from dashsrc.components.constants import ROW1_ANALYSISS, ROW2_ANALYSISS
import dashsrc.components.trial_wise_kinematics as trial_wise_kinematics

def create_analysis_containers(app: Dash, data: pd.DataFrame):
    analysis_row_containers = []
    for i, analysis in enumerate(ROW1_ANALYSISS+ROW2_ANALYSISS):
        match analysis:
            case 'Trial-Wise-Kinematics':
                analysis_div = trial_wise_kinematics.render(app, data, analysis_name=analysis)
            case _:
                analysis_div = html.Div([html.H5(analysis)], id=f'{analysis}-container', style={'display': 'none', 'backgroundColor': 'gray'})
                
        analysis_row_containers.append(analysis_div)
    return html.Div(analysis_row_containers, id='analysis-container')