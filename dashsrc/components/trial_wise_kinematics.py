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
            animal_ids = global_data['UnityTrackwise'].index.unique("animal_id")
            return [{'label': f'Animal {i:02}', 'value': i} for i in animal_ids]
        return []

    @app.callback(
        Output(f'session-dropdown-{vis_name}', 'options'),
        Input(f'animal-dropdown-{vis_name}', 'value')
    )
    def update_session_dropdown(selected_animal):
        if not selected_animal:
            return []
        sessions = global_data['UnityTrackwise'].loc[pd.IndexSlice[:,selected_animal,:,:]].index.unique('session_id')
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
        if selected_animal is None or selected_session is None:
            return 0, 100, (0,100)
        print("updated trial slider")
        
        last_trial_id = global_data['UnityTrackwise'].loc[pd.IndexSlice[:,selected_animal,selected_session,:]]['trial_id'].max()
        return 1, last_trial_id, (1, last_trial_id)

    @app.callback(
        Output(f'{vis_name}-plot', 'figure'),
        Input(f'animal-dropdown-{vis_name}', 'value'),
        Input(f'session-dropdown-{vis_name}', 'value'),
        Input(f'trial-range-slider-{vis_name}', 'value'),
        Input(f'group-by-{vis_name}', 'value'),
        Input(f'metric-{vis_name}', 'value'),
        Input(f'max-metric-value-{vis_name}', 'value'),
        Input(f'smooth-data-{vis_name}', 'value'),
        Input(f'variance-vis-{vis_name}', 'value'),
        Input(f'outcome-group-filter-{vis_name}', 'value'),
        Input(f'cue-group-filter-{vis_name}', 'value'),
        Input(f'trial-group-filter-{vis_name}', 'value'),
    )
    def update_plot(selected_animal, selected_session, trial_range, group_by, 
                    metric, metric_max, smooth_data, var_viz,
                    outcome_filter, cue_filter, trial_filter):
        if not all((selected_animal, (selected_session is not None), trial_range, metric_max)):
            return {}
        # animal and session filtering
        data = global_data['UnityTrackwise'].loc[pd.IndexSlice[:,selected_animal,selected_session,:]]
        
        # trial slider filtering
        n_trials = data['trial_id'].max().item()
        data = data[data.trial_id.isin(np.arange(int(trial_range[0]), int(trial_range[1]) + 1))]
        
        group_values = {}
        # outcome filtering
        if '1 R' in outcome_filter:
            group_values['1 R'] = [1]
        if '1+ R' in outcome_filter:
            group_values['1+ R'] = [2,3,4,5]
        if 'no R' in outcome_filter:
            group_values['no R'] = [0]
        data = data[data['trial_outcome'].isin(np.concatenate(list(group_values.values())))]
        if group_by == 'Outcome':
            group_by_values = group_values
        # print(data)
        
        # cue filtering
        group_values = {}
        if 'Early R' in cue_filter:
            group_values['Early R'] = [1]
        if 'Late R' in cue_filter:
            group_values['Late R'] = [2]
        data = data[data['cue'].isin(np.concatenate(list(group_values.values())))]
        if group_by == 'Cue':
            group_by_values = group_values
        # print(data)
            
            
        # trial filtering
        group_values = {}
        # get the 1st, 2nd, 3rd proportion of trials/ split in thirds
        trial_groups = np.array_split(data['trial_id'].unique(), 3)
        if "1/3" in trial_filter:
            group_values["1/3"] = trial_groups[0]
        if "2/3" in trial_filter:
            group_values["2/3"] = trial_groups[1]
        if "3/3" in trial_filter:
            group_values["3/3"] = trial_groups[2]
        incl_trials = np.concatenate([tg for tg in group_values.values()])
        data = data[data['trial_id'].isin(incl_trials)]
        if group_by == 'Part of session':
            group_by_values = group_values
            
        if group_by == 'None':
            group_by_values = None
        
        fig = trial_wise_kinematics_plot.render_plot(data, n_trials, group_by, group_by_values,
                                                     metric, metric_max, 
                                                     smooth_data, var_viz)
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
            ], width=7),

            # Right side for UI controls
            dbc.Col([
                # three rows, header, dropdown/tickpoxes, range slider
                dbc.Row([
                    html.H5(f"Data Selection for {vis_name}", style={"marginTop": 20}),
                ]),                                
                
                dbc.Row([
                    dbc.Col([
                        # Dropdown for animal selection
                        html.Label("Select animal", style={"marginTop": 15}),
                        dcc.Dropdown(
                            id=f'animal-dropdown-{vis_name}',
                            options= [] if global_data['UnityTrackwise'] is None else global_data['UnityTrackwise'].index.unique("animal_id"),
                            placeholder="Animal ID"
                        ),
                        # Dropdown for session selection
                        html.Label("Select session", style={"marginTop": 15}),
                        dcc.Dropdown(
                            id=f'session-dropdown-{vis_name}',
                            placeholder="Session ID"
                        ),
                        
                        html.Label("Metric", style={"marginTop": 15}),
                        dcc.RadioItems(['Velocity', 'Acceleration'], 
                                       inputStyle={"margin-right": "5px"},
                                       style={"marginLeft": 5},
                                       value='Velocity',
                                       id=f'metric-{vis_name}'),
                    ], width=4),
                    
                    # Other options in middle column
                    dbc.Col([
                        html.Label("Group by", style={"marginTop": 15}),
                        dcc.RadioItems(['Outcome', 'Cue', 'Part of session', 'None'], 
                                       inputStyle={"margin-right": "5px"},
                                       style={"marginLeft": 5},
                                       value='None',
                                       id=f'group-by-{vis_name}'),
                        
                        html.Label("Variance vis.", style={"marginTop": 15}),
                        dcc.RadioItems(['Single trials', '80th percent.', "None"], 
                                       inputStyle={"margin-right": "5px"},
                                       style={"marginLeft": 5},
                                       value='Single trials',
                                       id=f'variance-vis-{vis_name}'),
                    ], width=4),
                    
                    # Other options in right column
                    dbc.Col([
                        html.Label("Filter", style={"marginTop": 15}),
                        dcc.Checklist(id=f'outcome-group-filter-{vis_name}',
                                        options=['1 R', '1+ R','no R',],
                                        value=['1 R', "1+ R", 'no R'],
                                        inline=True,
                                        inputStyle={"margin-right": "7px", "margin-left": "3px"},
                                        ),
                        dcc.Checklist(id=f'cue-group-filter-{vis_name}',
                                        options=['Early R', 'Late R'],
                                        value=['Early R', 'Late R'],
                                        inline=True,
                                        inputStyle={"margin-right": "7px", "margin-left": "3px"},
                                        ),
                        dcc.Checklist(id=f'trial-group-filter-{vis_name}',
                                        options=['1/3', '2/3', '3/3'],
                                        value=['1/3', '2/3', '3/3'],
                                        inline=True,
                                        inputStyle={"margin-right": "7px", "margin-left": "3px"},
                                        ),

                        html.Label("Display options", style={"marginTop": 15}),
                        
                        dcc.Checklist(['Smooth'], 
                            inputStyle={"margin-right": "5px"},
                            style={"marginLeft": 5, "marginTop": 5},
                            value=[],
                            id=f'smooth-data-{vis_name}'),
                        
                        html.Label("Max Metric", style={"marginTop": 10}),
                        dcc.Input(id=f'max-metric-value-{vis_name}', 
                                  type='number', value=80, style={"width": "60%"}),
                                                
                    ], width=4),
                ]),
                
                # Range slider for trial selection
                dbc.Row([
                    html.Label("Select a range of trials", style={"marginTop": 15}),
                    dcc.RangeSlider(0, 100, value=[0, 100], id=f'trial-range-slider-{vis_name}')
                ])
            ], width=5)
        ]),
        html.Hr()
    ], id=f"{vis_name}-container")  # Initial state is hidden