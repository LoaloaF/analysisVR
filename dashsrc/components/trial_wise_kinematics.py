from dash import html, dcc, Input, Output, State, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from data_loading.animal_loading import get_animal_modality
# from analysis_core import get_default_nas_dir
from dashsrc.plots import trial_wise_kinematics_plot

from .constants import *

def render(app: Dash, data: pd.DataFrame, analysis_name: str) -> html.Div:
    modality = 'unity_frame'

    @app.callback(
        Output(f'animal-dropdown-{analysis_name}', 'options'),
        Input('data-loaded', 'data'),
    )
    def update_animal_options(data_loaded):
        if data_loaded:
            animal_ids = data[modality].index.unique("animal_id")
            return [{'label': f'rYL{i:03}', 'value': i} for i in animal_ids]
        return []

    @app.callback(
        Output(f'session-dropdown-{analysis_name}', 'options'),
        Input(f'animal-dropdown-{analysis_name}', 'value')
    )
    def update_session_dropdown(selected_animal):
        if not selected_animal:
            return []
        session_ids = [{'label': f'Session {i}', 'value': i} 
                       for i in data[modality].loc[selected_animal].index.unique('session_id')]
        return session_ids
    
    @app.callback(
        Output(f'trial-range-slider-{analysis_name}', 'min'),
        Output(f'trial-range-slider-{analysis_name}', 'max'),
        Input(f'animal-dropdown-{analysis_name}', 'value'),
        Input(f'session-dropdown-{analysis_name}', 'value')
    )
    def update_trial_slider(selected_animal, selected_session):
        if not selected_animal or not selected_session:
            return 0, 100
        last_trial_id = int(data[modality].loc[(selected_animal, selected_session)].index.get_level_values('trial_id').max())
        return 1, last_trial_id
    
    @app.callback(
        Output(f'{analysis_name}-plot', 'figure'),
        Input(f'animal-dropdown-{analysis_name}', 'value'),
        Input(f'session-dropdown-{analysis_name}', 'value'),
        Input(f'trial-range-slider-{analysis_name}', 'value'),
        Input(f'trial-coloring-{analysis_name}', 'value'),
        Input(f'metric-{analysis_name}', 'value'),
    )
    def update_plot(selected_animal, selected_session, trial_range, trial_color, metric):
        if not (selected_animal and selected_session and trial_range and trial_range[0] != 0):
            return {}
        trial_ids = np.arange(int(trial_range[0]), int(trial_range[1]) + 1)
        fig = trial_wise_kinematics_plot.render_plot(data[modality], selected_animal, 
                                                     selected_session, trial_ids, 
                                                     trial_color, metric)
        return fig
    
    return html.Div([
        dcc.Store(id='data-loaded', data=False),  # Store to track data loading state
        dbc.Row([
            # Left side for plots
            dbc.Col([
                dcc.Graph(
                    id=f'{analysis_name}-plot',
                    config={
                        'toImageButtonOptions': {
                            'format': 'svg',
                            'filename': f'custom_svg_image_{analysis_name}',
                            'height': 600,
                            'width': 800,
                            'scale': 1
                        },
                        'displaylogo': False,
                        'modeBarButtonsToAdd': ['toImage'],
                        'modeBarButtonsToRemove': ['autoScale2d', 'zoomIn2d', 'zoomOut2d']
                    }
                ),
            ], width=9),

            # Right side for UI controls
            dbc.Col([
                # three rows, header, dropdown/tickpoxes, range slider
                dbc.Row([
                    html.H5(f"Data Selection for {analysis_name}", style={"marginTop": 20}),
                ]),                                
                
                dbc.Row([
                    dbc.Col([
                        # Dropdown for animal selection
                        html.Label("Select an animal", style={"marginTop": 15}),
                        dcc.Dropdown(
                            id=f'animal-dropdown-{analysis_name}',
                            options=[],
                            # multi=True,
                            placeholder="Animal ID"
                        ),
                        # Dropdown for session selection
                        html.Label("Select a session", style={"marginTop": 15}),
                        dcc.Dropdown(
                            id=f'session-dropdown-{analysis_name}',
                            # multi=True,
                            placeholder="Session ID"
                        ),
                    ], width=4),
                    
                    # Other options in middle column
                    dbc.Col([
                        html.Label("Trial Color", style={"marginTop": 15}),
                        dcc.RadioItems(['Outcome', 'Cue', 'Trial ID'], 
                                       inputStyle={"margin-right": "5px"},
                                       style={"marginLeft": 5},
                                       value='Outcome',
                                       id=f'trial-coloring-{analysis_name}')
                    ], width=4),
                    
                    # Other options in right column
                    dbc.Col([
                        html.Label("Metric", style={"marginTop": 15}),
                        dcc.RadioItems(['Velocity', 'Acceleration'], 
                                       inputStyle={"margin-right": "5px"},
                                       style={"marginLeft": 5},
                                       value='Velocity',
                                       id=f'metric-{analysis_name}')
                    ], width=4),
                ]),
                
                # Range slider for trial selection
                dbc.Row([
                    html.Label("Select a range of trials", style={"marginTop": 15}),
                    dcc.RangeSlider(0, 100, value=[0, 100], id=f'trial-range-slider-{analysis_name}')
                ])
            ], width=3)
        ]),
        html.Hr()
    ], id=f"{analysis_name}-container")  # Initial state is hidden

# # Example usage
# app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# data = pd.DataFrame({
#     'animal_id': ['A', 'A', 'B', 'B'],
#     'session_id': [1, 2, 1, 2],
#     'trial_id': [1, 2, 1, 2],
#     'modality': ['unity_frame', 'unity_frame', 'unity_frame', 'unity_frame']
# }).set_index(['animal_id', 'session_id', 'trial_id'])
# app.layout = render(app, data, "Trial-Wise-Kinematics")

# if __name__ == '__main__':
#     app.run_server(debug=True)