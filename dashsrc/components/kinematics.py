from dash import html, dcc, Input, Output, State, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
# from data_loading.animal_loading import get_animal_modality
# from analysis_core import get_default_nas_dir
from dashsrc.plots import kinematics_plot

from .constants import *

def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:

    @app.callback(
        Output(f'animal-dropdown-{vis_name}', 'options'),
        Input('data-loaded', 'data'),
    )
    def update_animal_options(data_loaded):
        if data_loaded:
            animal_ids = global_data['UnityTrackwise'].index.unique("animal_id")
            return [{'label': f'Animal {i:02}', 'value': i} for i in animal_ids]
        return []

    @app.callback(
        Output(f'session-dropdown-{vis_name}', 'options'),
        Input(f'animal-dropdown-{vis_name}', 'value')
    )
    def update_session_dropdown(selected_animals):
        if not selected_animals:
            return []
        sessions = global_data['UnityTrackwise'].loc[pd.IndexSlice[:,selected_animals,:,:]].index.unique('session_id')
        session_ids = [{'label': f'Session {i}', 'value': i} for i in sessions]
        return session_ids
    
    @app.callback(
        Output(f'trial-range-slider-{vis_name}', 'min'),
        Output(f'trial-range-slider-{vis_name}', 'max'),
        Output(f'trial-range-slider-{vis_name}', 'value'),
        Input(f'animal-dropdown-{vis_name}', 'value'),
        Input(f'session-dropdown-{vis_name}', 'value')
    )
    def update_trial_slider(selected_animals, selected_session):
        if not selected_animals or not selected_session:
            return 0, 100, (0,100)
        
        last_trial_id = global_data['UnityTrackwise'].loc[pd.IndexSlice[:,selected_animals,selected_session,:]]['trial_id'].max()
        return 1, last_trial_id, (1, last_trial_id)
    
    @app.callback(
        Output(f'{vis_name}-plot', 'figure'),
        Input(f'animal-dropdown-{vis_name}', 'value'),
        Input(f'session-dropdown-{vis_name}', 'value'),
        Input(f'trial-range-slider-{vis_name}', 'value'),
        Input(f'trial-coloring-{vis_name}', 'value'),
        Input(f'metric-{vis_name}', 'value'),
        Input(f'max-metric-value-{vis_name}', 'value'),
    )
    def update_plot(selected_animals, selected_session, trial_range, trial_color, metric, metric_max):
        # if not all((selected_animals, (selected_session is not None), trial_range, metric_max)):
        #     return {}
        if not all((selected_animals, )):
            return {}
        print(selected_animals)
        data = global_data['UnityTrackwise'].loc[pd.IndexSlice[:,selected_animals,:,:]]
        # n_trials = data['trial_id'].max().item()
        # data = data[data.trial_id.isin(np.arange(int(trial_range[0]), int(trial_range[1]) + 1))]
        print(data)
        
        fig = kinematics_plot.render_plot(data, trial_color, 
                                          metric, metric_max)
        return fig
    
    return html.Div([
        dcc.Store(id='data-loaded', data=False),  # Store to track data loading state
        dbc.Row([
            # Left side for plots
            dbc.Col([
                dcc.Graph(
                    id=f'{vis_name}-plot',
                    config={
                        'toImageButtonOptions': {
                            'format': 'svg',
                            'filename': f'custom_svg_image_{vis_name}',
                            'height': 600,
                            'width': 800,
                            'scale': 1
                        },
                        'displaylogo': False,
                        'modeBarButtonsToAdd': ['toImage'],
                        'modeBarButtonsToRemove': ['autoScale2d', 'zoomIn2d', 'zoomOut2d']
                    },
                style={"height": 350}),
            ], width=9),

            # Right side for UI controls
            dbc.Col([
                # three rows, header, dropdown/tickpoxes, range slider
                dbc.Row([
                    html.H5(f"Data Selection for {vis_name}", style={"marginTop": 20}),
                ]),                                
                
                dbc.Row([
                    dbc.Col([
                        # Dropdown for animal selection
                        html.Label("Select animals", style={"marginTop": 15}),
                        dcc.Dropdown(
                            id=f'animal-dropdown-{vis_name}',
                            options= [] if global_data['UnityTrackwise'] is None else global_data['UnityTrackwise'].index.unique("animal_id"),
                            multi=True,
                            placeholder="Animal ID"
                        ),
                        # Dropdown for session selection
                        html.Label("Select session", style={"marginTop": 15}),
                        dcc.Dropdown(
                            id=f'session-dropdown-{vis_name}',
                            placeholder="Session ID"
                        ),
                    ], width=4),
                    
                    # Other options in middle column
                    dbc.Col([
                        html.Label("Session Color", style={"marginTop": 15}),
                        dcc.RadioItems(['Outcome', 'Cue', 'Session ID'], 
                                       inputStyle={"margin-right": "5px"},
                                       style={"marginLeft": 5},
                                       value='Outcome',
                                       id=f'trial-coloring-{vis_name}'),
                    ], width=4),
                    
                    # Other options in right column
                    dbc.Col([
                        html.Label("Metric", style={"marginTop": 15}),
                        dcc.RadioItems(['Velocity', 'Acceleration'], 
                                       inputStyle={"margin-right": "5px"},
                                       style={"marginLeft": 5},
                                       value='Velocity',
                                       id=f'metric-{vis_name}'),
                        html.Label("Max Metric", style={"marginTop": 10}),
                        dcc.Input(id=f'max-metric-value-{vis_name}', 
                                  type='number', value=60, style={"width": "60%"}),
                    ], width=4),
                ]),
                
                # Range slider for trial selection
                dbc.Row([
                    html.Label("Select a range of trials", style={"marginTop": 15}),
                    dcc.RangeSlider(0, 100, value=[0, 100], id=f'trial-range-slider-{vis_name}')
                ])
            ], width=3)
        ]),
        html.Hr()
    ], id=f"{vis_name}-container")  # Initial state is hidden

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