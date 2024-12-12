# FILE: plot_ui_components.py
import pandas as pd
from dash import html, dcc, Input, Output

def get_paradigm_dropdown_component(vis_name, global_data):
    return [
        html.Label("Select paradigm", style={"marginTop": 15}),
        dcc.Dropdown(
            id=f'paradigm-dropdown-{vis_name}',
            options=[] if global_data['UnityTrackwise'] is None else [{'label': f'Paradigm {i:02}', 'value': i} for i in global_data['UnityTrackwise'].index.unique("paradigm_id")],
            placeholder="Paradigm ID"
        )
    ]
def get_animal_dropdown_component(vis_name, global_data, multi=False):
    return [
        html.Label("Select animal", style={"marginTop": 15}),
        dcc.Dropdown(
            id=f'animal-dropdown-{vis_name}',
            options=[] if global_data['UnityTrackwise'] is None else [{'label': f'Animal {i:02}', 'value': i} for i in global_data['UnityTrackwise'].index.unique("animal_id")],
            placeholder="Animal ID",
            multi=multi,
        )
    ]

def get_session_dropdown_component(vis_name):
    return [
        html.Label("Select session", style={"marginTop": 15}),
        dcc.Dropdown(
            id=f'session-dropdown-{vis_name}',
            placeholder="Session ID"
        )
    ]

def get_metric_radioitems_component(vis_name):
    return [
        html.Label("Metric", style={"marginTop": 15}),
        dcc.RadioItems(
            ['Velocity', 'Acceleration'],
            inputStyle={"margin-right": "5px"},
            style={"marginLeft": 5},
            value='Velocity',
            id=f'metric-{vis_name}'
        )
    ]

def get_groupby_radioitems_component(vis_name):
    return [
        html.Label("Group by", style={"marginTop": 15}),
        dcc.RadioItems(
            ['Outcome', 'Cue', 'Part of session', 'None'],
            inputStyle={"margin-right": "5px"},
            style={"marginLeft": 5},
            value='None',
            id=f'group-by-{vis_name}'
        )
    ]

def get_variance_radioitems_component(vis_name):
    return [
        html.Label("Variance vis.", style={"marginTop": 15}),
        dcc.RadioItems(
            ['Single trials', '80th percent.', "None"],
            inputStyle={"margin-right": "5px"},
            style={"marginLeft": 5},
            value='Single trials',
            id=f'variance-vis-{vis_name}'
        )
    ]

def get_filter_checklist_component(vis_name):
    return [
        html.Label("Filter", style={"marginTop": 15}),
        dcc.Checklist(
            id=f'outcome-group-filter-{vis_name}',
            options=['1 R', '1+ R','no R'],
            value=['1 R', "1+ R", 'no R'],
            inline=True,
            inputStyle={"margin-right": "7px", "margin-left": "3px"}
        ),
        dcc.Checklist(
            id=f'cue-group-filter-{vis_name}',
            options=['Early R', 'Late R'],
            value=['Early R', 'Late R'],
            inline=True,
            inputStyle={"margin-right": "7px", "margin-left": "3px"}
        ),
        dcc.Checklist(
            id=f'trial-group-filter-{vis_name}',
            options=['1/3', '2/3', '3/3'],
            value=['1/3', '2/3', '3/3'],
            inline=True,
            inputStyle={"margin-right": "7px", "margin-left": "3px"}
        )
    ]

def get_display_options_checklist_component(vis_name):
    return [
        html.Label("Display options", style={"marginTop": 0}),
        dcc.Checklist(
            ['Smooth'],
            inputStyle={"margin-right": "5px"},
            style={"marginLeft": 5, "marginTop": 5},
            value=[],
            id=f'smooth-data-{vis_name}'
        ),
        html.Label("Max Metric", style={"marginTop": 10}),
        dcc.Input(
            id=f'max-metric-value-{vis_name}',
            type='number',
            value=80,
            style={"width": "60%"}
        )
    ]

def get_trial_range_slider_component(vis_name):
    return [
        html.Label("Select a range of trials", style={"marginTop": 15}),
        dcc.RangeSlider(
            0, 100,
            value=[0, 100],
            id=f'trial-range-slider-{vis_name}'
        )
    ]

def register_animal_dropdown_callback(app, global_data, vis_name):
    @app.callback(
        Output(f'animal-dropdown-{vis_name}', 'options'),
        Input('data-loaded', 'data'),
    )
    def update_animal_options(data_loaded):
        if data_loaded:
            animal_ids = global_data['UnityTrackwise'].index.unique("animal_id")
            return [{'label': f'Animal {i:02}', 'value': i} for i in animal_ids]
        return []

def register_session_dropdown_callback(app, global_data, vis_name):
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

def register_trial_slider_callback(app, global_data, vis_name):
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

def register_paradigm_dropdown_callback(app, global_data, vis_name):
    @app.callback(
        Output(f'paradigm-dropdown-{vis_name}', 'options'),
        Input('data-loaded', 'data'),
    )
    def update_paradigm_options(data_loaded):
        if data_loaded:
            paradigm_ids = global_data['UnityTrackwise'].index.unique("paradigm_id")
            return [{'label': f'Paradigm {i:04}', 'value': i} for i in paradigm_ids]
        return []