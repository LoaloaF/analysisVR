from dash import html, dcc, Input, Output, State, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
# from data_loading.animal_loading import get_animal_modality
# from analysis_core import get_default_nas_dir
from dashsrc.plots import kinematics_plot
from .constants import *

from .plot_ui_components import (
    get_animal_dropdown_component,
    get_paradigm_dropdown_component,
    get_metric_radioitems_component,
    # get_groupby_radioitems_component,
    get_filter_checklist_component,
    get_display_options_checklist_component,
    get_session_range_slider_component,
    get_width_input,
    get_height_input,
    
    register_animal_dropdown_callback,
    register_paradigm_dropdown_callback,
    register_session_slider_callback
)
from .plot_components import get_track_graph_component

def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
    # Register the callbacks
    register_paradigm_dropdown_callback(app, global_data, 'UnityTrackwise', vis_name)
    register_animal_dropdown_callback(app, global_data, 'UnityTrackwise', vis_name)
    register_session_slider_callback(app, global_data, 'UnityTrackwise', vis_name)
    
    # register_session_dropdown_callback(app, global_data, vis_name)
    # register_trial_slider_callback(app, global_data, vis_name)

    @app.callback(
        Output(f'figure-{vis_name}', 'figure'),
        Input(f'animal-dropdown-{vis_name}', 'value'),
        Input(f'session-range-slider-{vis_name}', 'value'),
        
        # Input(f'trial-range-slider-{vis_name}', 'value'),
        Input(f'metric-{vis_name}', 'value'),
        Input(f'max-metric-value-{vis_name}', 'value'),
        Input(f'smooth-data-{vis_name}', 'value'),
        Input(f'outcome-group-filter-{vis_name}', 'value'),
        Input(f'cue-group-filter-{vis_name}', 'value'),
        Input(f'trial-group-filter-{vis_name}', 'value'),
        
        Input(f'width-input-{vis_name}', 'value'),
        Input(f'height-input-{vis_name}', 'value'),
    )
    def update_plot(selected_animals, session_range,
                    metric, metric_max, smooth_data,
                    outcome_filter, cue_filter, trial_filter,
                    width, height):
        
        if not all((selected_animals, metric_max)):
            return {}
        
        selected_sessions = np.arange(int(session_range[0]), int(session_range[1]) + 1)
        
        
        # animal and session filtering
        data = global_data['UnityTrackwise'].loc[pd.IndexSlice[:,selected_animals,selected_sessions,:]]
        print(data)
        
        data, group_by_values = group_filter_data(data, outcome_filter, cue_filter, trial_filter)
        
        # trial slider filtering
        # n_trials = data['trial_id'].max().item()
        # data = data[data.trial_id.isin(np.arange(int(trial_range[0]), int(trial_range[1]) + 1))]
        
        
        
        # list to single value
        if len(smooth_data) == 1:
            smooth_data = True
            
        fig = kinematics_plot.render_plot(data, global_data['SessionMetadata'], 
                                          metric, metric_max, 
                                          smooth_data, width, height)
        return fig
    
    return html.Div([
        dcc.Store(id='data-loaded', data=False),  # Store to track data loading state
        dbc.Row([
            # Left side for plots
            dbc.Col([
                get_track_graph_component(vis_name, fixed_width=None, fixed_height=None),
            ], width=8),

            # Right side for UI controls
            dbc.Col([
                # three rows, header, dropdown/tickpoxes, range slider
                dbc.Row([
                    html.H5(f"Data Selection for {vis_name}", style={"marginTop": 20}),
                ]),                                
                
                dbc.Row([
                    dbc.Col([
                        # Dropdown for paradigm selection
                        *get_paradigm_dropdown_component(vis_name, global_data, 'UnityTrackwise'),
                        # Dropdown for animal selection
                        *get_animal_dropdown_component(vis_name, global_data, 'UnityTrackwise'),
                        # Radioitems for metric selection
                        *get_metric_radioitems_component(vis_name),
                    ], width=4),
                    
                    # Other options in right column
                    dbc.Col([
                        # Filter checklist
                        *get_filter_checklist_component(vis_name),
                        html.Hr(),
                        # Display options checklist
                        *get_display_options_checklist_component(vis_name,
                                                                 initial_value=80),
                    ], width=4),
                    
                    # Other options in right column
                    dbc.Col([
                        # Filter checklist
                        *get_width_input(vis_name),
                        # Display options checklist
                        *get_height_input(vis_name),
                    ], width=4),
                ]),
                
                # Range slider for trial selection
                # *get_trial_range_slider_component(vis_name)
                # Range slider for trial selection
                *get_session_range_slider_component(vis_name),
            ], width=4)
        ]),
        html.Hr()
    ], id=f"{vis_name}-container")  # Initial state is hidden
    
def group_filter_data(data, outcome_filter, cue_filter, trial_filter, group_by="None"):
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
    
    # cue filtering
    group_values = {}
    if 'Early R' in cue_filter:
        group_values['Early R'] = [1]
    if 'Late R' in cue_filter:
        group_values['Late R'] = [2]
    data = data[data['cue'].isin(np.concatenate(list(group_values.values())))]
    if group_by == 'Cue':
        group_by_values = group_values
        
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
    
    return data, group_by_values