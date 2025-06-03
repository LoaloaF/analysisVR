from datetime import date

from dash import dcc
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

import dashsrc.components.dashvis_constants as C
import pandas as pd
    
def paradigm_dropdown_component(vis_name, global_data, analytic):
    data = global_data[analytic]
    component_id = f'paradigm-dropdown-{vis_name}'
    return [
        html.Label("Select paradigm", style={"marginTop": 15}),
        dcc.Dropdown(
            id=component_id,
            options=[] if data is None else [{'label': f'Paradigm {i:02}', 'value': i} 
                                             for i in data.index.unique("paradigm_id")],
            placeholder="Paradigm ID"
        )
    ], component_id

def animal_dropdown_component(vis_name, global_data, analytic, multi=False):
    data = global_data[analytic]
    component_id = f'animal-dropdown-{vis_name}'
    return [
        html.Label("Select animal", style={"marginTop": 15}),
        dcc.Dropdown(
            id=component_id,
            options=[] if data is None else [{'label': f'Animal {i:02}', 'value': i} 
                                             for i in data.index.unique("animal_id")],
            placeholder="Animal ID",
            multi=multi,
        )
    ], component_id

def session_dropdown_component(vis_name, global_data, analytic):
    data = global_data[analytic]
    component_id = f'session-dropdown-{vis_name}'
    return [
        html.Label("Select session", style={"marginTop": 15}),
        dcc.Dropdown(
            id=component_id,
            options=[] if data is None else [{'label': f'Session {i:02}', 'value': i} 
                                             for i in data.index.unique("session_id")],
            placeholder="Session ID"
        )
    ], component_id

def metric_radioitems_component(vis_name):
    component_id = f'metric-{vis_name}'
    return [
        html.Label("Metric", style={"marginTop": 15}),
        dcc.RadioItems(
            ['Velocity', 'Acceleration', 'Lick'],
            inputStyle={"margin-right": "5px"},
            style={"marginLeft": 5},
            value='Velocity',
            id=component_id
        )
    ], component_id
    
def models_radioitems_component(vis_name):
    component_id = f'models-{vis_name}'
    return [
        html.Label("Model", style={"marginTop": 15}),
        dcc.RadioItems(
            [{'label':'linear-SVM', 'value':'linear'},
             {'label': 'kernel-SVM', 'value': 'rbf'}],
            
            inputStyle={"margin-right": "5px"},
            style={"marginLeft": 5},
            value='linear',
            id=component_id
        )
    ], component_id

def groupby_radioitems_component(vis_name):
    component_id = f'group-by-{vis_name}'
    return [
        html.Label("Group by", style={"marginTop": 15}),
        dcc.RadioItems(
            ['Outcome', 'Cue', 'Part of session', 'None'],
            inputStyle={"margin-right": "5px"},
            style={"marginLeft": 5},
            value='None',
            id=component_id
        )
    ], component_id

def variance_radioitems_component(vis_name):
    component_id = f'variance-vis-{vis_name}'
    return [
        html.Label("Variance vis.", style={"marginTop": 15}),
        dcc.RadioItems(
            ['Single trials', '80th percent.', "None"],
            inputStyle={"margin-right": "5px"},
            style={"marginLeft": 5},
            value='Single trials',
            id=component_id
        )
    ], component_id

def metrics_radioitems_component(vis_name):
    component_id = f'metrics-vis-{vis_name}'
    return [
        html.Label("Model metric", style={"marginTop": 15}),
        dcc.RadioItems(
            [{'label': 'macro f1', 'value': 'f1'},
             {'label': 'accur.', 'value': 'acc'}],
            inputStyle={"margin-right": "5px"},
            style={"marginLeft": 5},
            value='acc',
            id=component_id
        )
    ], component_id

def outcome_group_filter_component(vis_name):
    component_id = f'outcome-group-filter-{vis_name}'
    return [
        html.Label("Outcome subset", style={"marginTop": 15}),
        dcc.Checklist(
            id=component_id,
            options=['1 R', '1+ R', 'no R'],
            value=['1 R', '1+ R', 'no R'],
            inline=True,
            inputStyle={"margin-right": "7px", "margin-left": "3px"}
        )
    ], component_id

def cue_group_filter_component(vis_name):
    component_id = f'cue-group-filter-{vis_name}'
    return [
        html.Label("Cue subset", style={"marginTop": 15}),
        dcc.Checklist(
            id=component_id,
            options=['Cue1 trials', 'Cue2 trials'],
            value=['Cue1 trials', 'Cue2 trials'],
            inline=True,
            inputStyle={"margin-right": "7px", "margin-left": "3px"}
        )
    ], component_id

def trial_group_filter_component(vis_name):
    component_id = f'trial-group-filter-{vis_name}'
    return [
        html.Label("Trial subset", style={"marginTop": 15}),
        dcc.Checklist(
            id=component_id,
            options=['1/3', '2/3', '3/3'],
            value=['1/3', '2/3', '3/3'],
            inline=True,
            inputStyle={"margin-right": "7px", "margin-left": "3px"}
        )
    ], component_id

def smooth_checklist_component(vis_name):
    component_id = f'smooth-data-{vis_name}'
    return [
        html.Label("Display options", style={"marginTop": 0}),
        dcc.Checklist(
            ['Smooth'],
            inputStyle={"margin-right": "5px"},
            style={"marginLeft": 5, "marginTop": 5},
            value=[],
            id=component_id
        )
    ], component_id

def use_subset_component(vis_name):
    component_id = f'use-subset-{vis_name}'
    return [
        html.Label("Display options", style={"marginTop": 0}),
        dcc.Checklist(
            ['use subset'],
            inputStyle={"margin-right": "5px"},
            style={"marginLeft": 5, "marginTop": 5},
            value=[],
            id=component_id
        )
    ], component_id
    
def digit_input_component(vis_name, label='Input', initial_value=0, width=.5, 
                          step=1, debounce=True):
    component_id = f'{label}-{vis_name}'
    return [
        html.Label(label, style={"marginTop": 10,}),
        dcc.Input(
            id=component_id,
            type='number',
            step=step,
            value=initial_value,
            debounce=debounce,
            style={"width": f"{width*100}%", "marginLeft": "5px", 'marginRight': "5px"},
        )
    ], component_id

def shank_input_component(vis_name, initial_value=[1,2]):
    component_id = f'shanks-{vis_name}'
    return [
        html.Label("Shanks", style={"marginTop": 10}),
        dcc.Dropdown(
            id=component_id,
            options=[1,2,3,4],
            placeholder="Select shanks",
            multi=True,
            value=initial_value,
        )
    ], component_id
        
def max_metric_input_component(vis_name, initial_value):
    component_id = f'max-metric-value-{vis_name}'
    return [
        html.Label("Max Metric", style={"marginTop": 10}),
        dcc.Input(
            id=component_id,
            type='number',
            value=initial_value,
            style={"width": "60%"}
        )
    ], component_id
    

def trial_range_slider_component(vis_name):
    component_id = f'trial-range-slider-{vis_name}'
    return [
        html.Label("Select a range of trials", style={"marginTop": 15}),
        dcc.RangeSlider(
            0, 100,
            step=1,
            value=[0, 100],
            id=component_id
        )
    ], component_id
    
def session_time_slider_component(vis_name):
    component_id = f'session-time-slider-{vis_name}'
    return [
        html.Label("Select a session timestamp", style={"marginTop": 15}),
        dcc.Slider(
            0, 5*60,
            step=1,
            value=3*60,
            marks={t: f'{t//60}min' for t in range(0, 60*5, 60)},
            id=component_id,
            included=False
        )
    ], component_id

def session_range_slider_component(vis_name):
    component_id = f'session-range-slider-{vis_name}'
    return [
        html.Label("Select a range of sessions", style={"marginTop": 15}),
        dcc.RangeSlider(
            0, 10,
            value=[0, 10],
            step=1,
            id=component_id
        )
    ], component_id
    
    
def figure_width_input_component(vis_name):
    component_id = f'width-input-{vis_name}'
    return [
        html.Label("Width", style={"marginTop": 15}),
        dcc.Input(
            id=component_id,
            type='number',
            value=-1,
            style={"width": "40%", "marginLeft": "5px"},
            debounce=True,
        )
    ], component_id

def figure_height_input_component(vis_name):
    component_id = f'height-input-{vis_name}'
    return [
        html.Label("Height", style={"marginTop": 15}),
        dcc.Input(
            id=component_id,
            type='number',
            value=-1,
            style={"width": "40%", "marginLeft": "5px"},
            debounce=True,
        )
    ], component_id
    
def plot_type_component(vis_name):
    component_id = f'var-vis-{vis_name}'
    return [
        html.Label("Plot type", style={"marginTop": 15}),
        dcc.RadioItems(
            ['Distribution', 'Average'],
            inputStyle={"margin-right": "5px"},
            style={"marginLeft": 5},
            value='Average',
            id=component_id
        )
    ], component_id

def p1100_double_rewards_filter_component(vis_name):
    component_id = f'double_r-filter-{vis_name}'
    return [
        html.Label("Double Rewards", style={"marginTop": 15}),
        dcc.Checklist(
            id=component_id,
            options=['Early R', 'Late R'],
            value=['Early R', 'Late R'],
            inline=True,
            inputStyle={"margin-right": "7px", "margin-left": "3px"}
        )
    ], component_id

def from_hour_input_component(vis_name):
    component_id = f'from-hour-input-{vis_name}'
    return [dcc.Input(
                id=component_id,
                type='number',
                value=11,
                min=0,
                max=23,
                style={"width": "15%", "marginLeft": "5px", 'marginRight': "5px"},
            ),
    ], component_id

def to_hour_input_component(vis_name):
    component_id = f'to-hour-input-{vis_name}'
    return [dcc.Input(
                id=component_id,
                type='number',
                value=20,
                min=1,
                max=24,
                style={"width": "15%", "marginLeft": "5px", 'marginRight': "5px"},
            ),
    ], component_id
    
def date_range_slider_component(vis_name):
    component_id = f'date-range-slider-{vis_name}'
    from_date = date(C.MIN_DATE_YEAR, C.MIN_DATE_MONTH, C.MIN_DATE_DAY)
    to_date = date(C.MAX_DATE_YEAR, C.MAX_DATE_MONTH, C.MAX_DATE_DAY)
    return [
        html.Label("Select a date range of sessions", style={"marginTop": 15}),
        dcc.DatePickerRange(
            id=component_id,
            min_date_allowed=from_date,
            max_date_allowed=to_date,
            initial_visible_month=to_date,
            start_date=from_date,
            end_date=to_date,
        )
    ], component_id
    
def plot_type_selecter_component(vis_name):
    component_id = f'plot-type-{vis_name}'
    return [
        html.Label("Plot type", style={"marginTop": 15}),
        dcc.RadioItems(
            ['Timeseries', 'Modalities', "Analytics"],
            inputStyle={"margin-right": "3px"},
            style={"marginLeft": 10},
            value='Timeseries',
            inline=True,
            id=component_id
        )
    ], component_id
        
def time_step_button_component(vis_name, direction='forward', disabled=False):
    component_id = f'time-step-button-{direction}-{vis_name}'
    return [
        dbc.Button(
            ">>" if direction == 'forward' else "<<",
            id=component_id,
            n_clicks=0,
            style={"marginTop": 6, "marginLeft": 5, "width": "40px"},
            color="secondary",
            size="sm",
            disabled=disabled,
        )
    ], component_id

def trace_group_selecter_component(vis_name):
    component_id = f'plot-type-{vis_name}'
    return [
        html.Label("Draw traces", style={"marginTop": 15}),
        dcc.Checklist(
            ["non_curated", "curated", "spike_traces"],
            id=component_id,
            inputStyle={"margin-right": "3px", "margin-left": "3px"},
            style={"marginLeft": 10},
            value=['spike_traces'],
            inline=True,
        )
    ], component_id

def predictor_dropdown_component(vis_name, global_data, analytic):
    data = global_data[analytic]
    component_id = f'predictor-dropdown-{vis_name}'
    return [
        html.Label("Select predictor", style={"marginTop": 15}),
        dcc.Dropdown(
            id=component_id,
            options=[] if data is None else [data['predictor'].unique()],
            placeholder="Predictor",
        )
    ], component_id

def zone_dropdown_component(vis_name, global_data, analytic):
    data = global_data[analytic]
    component_id = f'zone-dropdown-{vis_name}'
    return [
        html.Label("Select track zone", style={"marginTop": 15}),
        dcc.Dropdown(
            id=component_id,
            options=[] if data is None else [data['zone'].unique()],
            placeholder="Zone",
        )
    ], component_id

def register_paradigm_dropdown_callback(app, vis_name, global_data, analytic):
    # html not used, just ensure that callcack is linked to correct component
    _, paradigm_dropd_comp_id = paradigm_dropdown_component(vis_name, global_data, analytic)
    data_loaded_id = C.get_vis_name_data_loaded_id(vis_name)
    
    @app.callback(
        Output(paradigm_dropd_comp_id, 'options'),
        Input(data_loaded_id, 'data'),
    )
    def update_paradigm_options(data_loaded_id):
        data = global_data[analytic]
        if data_loaded_id and data is not None:
            paradigm_ids = data.index.unique("paradigm_id")
            return [{'label': f'Paradigm {i:04}', 'value': i} for i in paradigm_ids]
        return []
    
def register_animal_dropdown_callback(app, vis_name, global_data, analytic):
    data_loaded_id = C.get_vis_name_data_loaded_id(vis_name)
    # html not used, just ensure that callcack is linked to correct component
    _, animal_dropd_comp_id = animal_dropdown_component(vis_name, global_data, analytic)
    
    @app.callback(
        Output(animal_dropd_comp_id, 'options'),
        Input(data_loaded_id, 'data'),
    )
    def update_animal_options(data_loaded):
        data = global_data[analytic]
        if data_loaded and data is not None:
            animal_ids = data.index.unique("animal_id")
            return [{'label': f'Animal {i:02}', 'value': i} for i in animal_ids]
        return []

def register_session_dropdown_callback(app, vis_name, global_data, analytic):
    # html not used, just ensure that callcack is linked to correct component
    _, session_dropd_comp_id = session_dropdown_component(vis_name, global_data, analytic)
    _, animal_dropd_comp_id = animal_dropdown_component(vis_name, global_data, analytic)
    @app.callback(
        Output(session_dropd_comp_id, 'options'),
        Input(animal_dropd_comp_id, 'value')
    )
    def update_session_dropdown(selected_animal):
        data = global_data[analytic]
        if not selected_animal or data is None:
            return []
        sessions = data.loc[pd.IndexSlice[:,selected_animal,:,:]].index.unique('session_id')
        session_ids = [{'label': f'Session {i}', 'value': i} for i in sessions]
        return session_ids

def register_session_slider_callback(app, vis_name, global_data, analytic, 
                                     default_select_sessions=None):
    # html not used, just ensure that callcack is linked to correct component
    _, session_slider_comp_id = session_range_slider_component(vis_name)
    _, animal_dropd_comp_id = animal_dropdown_component(vis_name, global_data, analytic)
    @app.callback(
        Output(session_slider_comp_id, 'min'),
        Output(session_slider_comp_id, 'max'),
        Output(session_slider_comp_id, 'value'),
        Input(animal_dropd_comp_id, 'value'),
    )
    def update_session_slider(selected_animal):
        data = global_data[analytic]
        if selected_animal is None or data is None:
            return 0, 10, (0,10), 
        
        last_session_id = data.index.unique("session_id").max()
        if default_select_sessions is not None:
            from_sel, to_sel = default_select_sessions
        else:
            from_sel, to_sel = 0, last_session_id
        return 0, last_session_id, (from_sel, to_sel)

def register_session_time_slider_callback(app, vis_name, global_data, analytic, loaded_raw_traces):
    # html not used, just ensure that callcack is linked to correct component
    _, session_time_slider_comp_id = session_time_slider_component(vis_name)
    _, paradigm_dropd_comp_id = paradigm_dropdown_component(vis_name, global_data, analytic)
    _, animal_dropd_comp_id = animal_dropdown_component(vis_name, global_data, analytic)
    _, session_dropd_comp_id = session_dropdown_component(vis_name, global_data, analytic)
    
    @app.callback(
        Output(session_time_slider_comp_id, 'min'),
        Output(session_time_slider_comp_id, 'max'),
        Output(session_time_slider_comp_id, 'value'),
        Output(session_time_slider_comp_id, 'marks'),
        Input(paradigm_dropd_comp_id, 'value'),
        Input(animal_dropd_comp_id, 'value'),
        Input(session_dropd_comp_id, 'value')
    )
    def update_session_time_slider(selected_paradigm, selected_animal, selected_session):
        # data = global_data['ephys_traces']
        if not all((selected_paradigm, selected_animal, (selected_session is not None),
                   len(loaded_raw_traces.keys()))):
            return 0, 5*60, 3*60, {t: f'{t//60}min' for t in range(0, 60*5, 60)},
        
        traces, _ = loaded_raw_traces[str((selected_paradigm, selected_animal, selected_session))]
        max_time_sec = traces.shape[1] *50 /1_000_000
        
        # marks at 1 min, 10, 20, 30, 40, 50
        marks = {t: f'{t//60}min' for t in range(60, int(max_time_sec) + 1, 60*10)}
        return 0, max_time_sec, 0, marks

def register_session_time_step_callback(app, vis_name, direction='forward'):
    # html not used, just ensure that callcack is linked to correct component
    _, session_time_step_comp_id = time_step_button_component(vis_name, direction)
    _, session_time_slider_component_id = session_time_slider_component(vis_name)
    _, int_input_component_id = digit_input_component(vis_name, "Interval [ms]")

    @app.callback(
    Output(session_time_slider_component_id, 'value', allow_duplicate=True),  # Allow duplicate output
    Input(session_time_slider_component_id, 'value'),
    Input(session_time_step_comp_id, 'n_clicks'),
    Input(int_input_component_id, 'value'),
    prevent_initial_call=True  # Prevent the callback from firing on app load
    )   
    def update_session_time_step(current_selected_time, n_clicks, interval):
        if n_clicks == 0:
            return current_selected_time
        else:
            new_time = current_selected_time*1_000 + interval if direction == 'forward' else -interval*2
            new_time = max(0, new_time)
            return new_time/1_000 

def register_trial_slider_callback(app, vis_name, global_data, analytic):
    # html not used, just ensure that callcack is linked to correct component
    _, trial_slider_comp_id = trial_range_slider_component(vis_name)
    _, animal_dropd_comp_id = animal_dropdown_component(vis_name, global_data, analytic)
    _, session_dropd_comp_id = session_dropdown_component(vis_name, global_data, analytic)
    
    @app.callback(
        Output(trial_slider_comp_id, 'min'),
        Output(trial_slider_comp_id, 'max'),
        Output(trial_slider_comp_id, 'value'),
        Input(animal_dropd_comp_id, 'value'),
        Input(session_dropd_comp_id, 'value')
    )
    def update_trial_slider(selected_animal, selected_session):
        data = global_data[analytic]
        if selected_animal is None or selected_session is None or data is None:
            return 0, 100, (0,100)
        
        last_trial_id = data.loc[pd.IndexSlice[:,selected_animal,selected_session,:]]['trial_id'].max()
        return 1, last_trial_id, (1, last_trial_id)
    
def register_predictor_dropdown_callback(app, vis_name, global_data, analytic):
    data_loaded_id = C.get_vis_name_data_loaded_id(vis_name)
    # html not used, just ensure that callcack is linked to correct component
    _, paradigm_dropd_comp_id = paradigm_dropdown_component(vis_name, global_data, analytic)
    _, animal_dropd_comp_id = animal_dropdown_component(vis_name, global_data, analytic)
    
    _, predictor_dropd_comp_id = predictor_dropdown_component(vis_name, global_data, analytic)
    
    @app.callback(
        Output(predictor_dropd_comp_id, 'options'),
        Input(data_loaded_id, 'data'),
        Input(paradigm_dropd_comp_id, 'value'),
        Input(animal_dropd_comp_id, 'value'),
    )
    def update_predictor_options(data_loaded, selected_paradigm, selected_animal):
        data = global_data[analytic]
        if data_loaded and data is not None:
            if selected_paradigm is not None and selected_animal is not None:
                preds = data.loc[(selected_paradigm, selected_animal), 'predictor'].unique()
                return preds
        return []
    
def register_zone_dropdown_callback(app, vis_name, global_data, analytic):
    data_loaded_id = C.get_vis_name_data_loaded_id(vis_name)
    # html not used, just ensure that callcack is linked to correct component
    _, paradigm_dropd_comp_id = paradigm_dropdown_component(vis_name, global_data, analytic)
    _, animal_dropd_comp_id = animal_dropdown_component(vis_name, global_data, analytic)
    
    _, zone_dropd_comp_id = zone_dropdown_component(vis_name, global_data, analytic)
    
    @app.callback(
        Output(zone_dropd_comp_id, 'options'),
        Input(data_loaded_id, 'data'),
        Input(paradigm_dropd_comp_id, 'value'),
        Input(animal_dropd_comp_id, 'value'),
    )
    def update_zone_options(data_loaded, selected_paradigm, selected_animal):
        data = global_data[analytic]
        if data_loaded and data is not None:
            if selected_paradigm is not None and selected_animal is not None:
                print(data)
                zones = data.loc[(selected_paradigm, selected_animal), 'track_zone'].unique()
                return zones
        return []


def PCs_slider_component(vis_name):
    component_id = f'PCs-slider-{vis_name}'
    return [
        html.Label("Average over PCs", style={"marginTop": 15}),
        dcc.RangeSlider(
            1, 15,
            step=1,
            value=[1, 15],
            id=component_id
        )
    ], component_id
    
def register_PCs_slider_callback(app, vis_name, global_data, analytic):
    # html not used, just ensure that callcack is linked to correct component
    _, paradigm_dropd_comp_id = paradigm_dropdown_component(vis_name, global_data, analytic)
    _, animal_dropd_comp_id = animal_dropdown_component(vis_name, global_data, analytic)
    _, pred_dropd_comp_id = predictor_dropdown_component(vis_name, global_data, analytic)
    _, zone_dropd_comp_id = zone_dropdown_component(vis_name, global_data, analytic)
    
    _, PCs_slider_comp_id = PCs_slider_component(vis_name)
    
    @app.callback(
        Output(PCs_slider_comp_id, 'min'),
        Output(PCs_slider_comp_id, 'max'),
        Output(PCs_slider_comp_id, 'value'),
        Input(paradigm_dropd_comp_id, 'value'),
        Input(animal_dropd_comp_id, 'value'),
        Input(pred_dropd_comp_id, 'value'),
        Input(zone_dropd_comp_id, 'value')
    )
    def update_PCs_slider(selected_paradigm, selected_animal, selected_predictor, selected_zone):
        data = global_data[analytic]
        if not all((selected_paradigm, selected_animal, selected_predictor, selected_zone)) or data is None:
            return 0, 15, (0,15)
        
        # get the PCs for the given paradigm, animal and predictor
        data_sl = data.loc[(selected_paradigm, selected_animal)]
        data_sl = data_sl[(data_sl.predictor == selected_predictor) &
                          (data_sl.track_zone == selected_zone)]
        nPCs = data_sl.shape[1] -2 # -2 because of paradigm_id and animal_id columns
    
        return 1, nPCs, (4, nPCs)
    

def register_trial_slider_callback(app, vis_name, global_data, analytic):
    # html not used, just ensure that callcack is linked to correct component
    _, trial_slider_comp_id = trial_range_slider_component(vis_name)
    _, animal_dropd_comp_id = animal_dropdown_component(vis_name, global_data, analytic)
    _, session_dropd_comp_id = session_dropdown_component(vis_name, global_data, analytic)
    
    @app.callback(
        Output(trial_slider_comp_id, 'min'),
        Output(trial_slider_comp_id, 'max'),
        Output(trial_slider_comp_id, 'value'),
        Input(animal_dropd_comp_id, 'value'),
        Input(session_dropd_comp_id, 'value')
    )
    def update_trial_slider(selected_animal, selected_session):
        data = global_data[analytic]
        if selected_animal is None or selected_session is None or data is None:
            return 0, 100, (0,100)
        
        last_trial_id = data.loc[pd.IndexSlice[:,selected_animal,selected_session,:]]['trial_id'].max()
        return 1, last_trial_id, (1, last_trial_id)