from dash import html, dcc, Input, Output, State, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
# from data_loading.animal_loading import get_animal_modality
# from analysis_core import get_default_nas_dir
from dashsrc.plots import trial_wise_kinematics_plot

from .constants import *

def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:

    @app.callback(
        Output(f'animal-dropdown-{vis_name}', 'options'),
        Input('data-loaded', 'data'),
    )
    def update_animal_options(data_loaded):
        if data_loaded:
            animal_ids = global_data['unity_trackwise'].index.unique("animal_id")
            return [{'label': f'Animal {i:02}', 'value': i} for i in animal_ids]
        return []

    @app.callback(
        Output(f'session-dropdown-{vis_name}', 'options'),
        Input(f'animal-dropdown-{vis_name}', 'value')
    )
    def update_session_dropdown(selected_animal):
        if not selected_animal:
            return []
        sessions = global_data['unity_trackwise'].loc[pd.IndexSlice[:,selected_animal,:,:]].index.unique('session_id')
        session_ids = [{'label': f'Session {i}', 'value': i} for i in sessions]
        return session_ids
    
    @app.callback(
        Output(f'trial-range-slider-{vis_name}', 'min'),
        Output(f'trial-range-slider-{vis_name}', 'max'),
        Output(f'trial-range-slider-{vis_name}', 'value'),
        Input(f'animal-dropdown-{vis_name}', 'value'),
        Input(f'session-dropdown-{vis_name}', 'value')
    )
    def update_trial_slider(selected_animal, selected_session):
        if not selected_animal or not selected_session:
            return 0, 100, (0,100)
        
        last_trial_id = global_data['unity_trackwise'].loc[pd.IndexSlice[:,selected_animal,selected_session,:]]['trial_id'].max()
        return 1, last_trial_id, (1, last_trial_id)
    
    @app.callback(
        Output(f'{vis_name}-plot', 'figure'),
        Input(f'animal-dropdown-{vis_name}', 'value'),
        Input(f'session-dropdown-{vis_name}', 'value'),
        Input(f'trial-range-slider-{vis_name}', 'value'),
        Input(f'trial-coloring-{vis_name}', 'value'),
        Input(f'metric-{vis_name}', 'value'),
    )
    def update_plot(selected_animal, selected_session, trial_range, trial_color, metric):
        if not (selected_animal and selected_session and trial_range and trial_range[0] != 0):
            return {}
        
        data = global_data['unity_trackwise'].loc[pd.IndexSlice[:,selected_animal,selected_session,:]]
        n_trials = data['trial_id'].max().item()
        data = data[data.trial_id.isin(np.arange(int(trial_range[0]), int(trial_range[1]) + 1))]
        print(data)
        
        trial_ids = np.arange(int(trial_range[0]), int(trial_range[1]) + 1)
        fig = trial_wise_kinematics_plot.render_plot(data, n_trials, 
                                                     trial_color, metric)
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
                    }
                ),
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
                        html.Label("Select an animal", style={"marginTop": 15}),
                        dcc.Dropdown(
                            id=f'animal-dropdown-{vis_name}',
                            options= [] if global_data['unity_trackwise'] is None else global_data['unity_trackwise'].index.unique("animal_id"),
                            # multi=True,
                            placeholder="Animal ID"
                        ),
                        # Dropdown for session selection
                        html.Label("Select a session", style={"marginTop": 15}),
                        dcc.Dropdown(
                            id=f'session-dropdown-{vis_name}',
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
                                       id=f'trial-coloring-{vis_name}')
                    ], width=4),
                    
                    # Other options in right column
                    dbc.Col([
                        html.Label("Metric", style={"marginTop": 15}),
                        dcc.RadioItems(['Velocity', 'Acceleration'], 
                                       inputStyle={"margin-right": "5px"},
                                       style={"marginLeft": 5},
                                       value='Velocity',
                                       id=f'metric-{vis_name}')
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