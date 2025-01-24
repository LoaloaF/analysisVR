from dash import html, dcc, Input, Output, State, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
# from data_loading.animal_loading import get_animal_modality

from dashsrc.plot_components import staytimes_plot_sessions as staytimes_plot_session
from dashsrc.components.dashvis_constants import *
import dashsrc.components.dashvis_constants as C

from .plot_ui_components import (
    get_animal_dropdown_component,
    get_paradigm_dropdown_component,
    get_metric_radioitems_component,
    get_groupby_radioitems_component,
    get_filter_checklist_component,
    get_display_options_checklist_component,
    get_session_range_slider_component,
    # get_trial_range_slider_component,
    register_animal_dropdown_callback,
    register_paradigm_dropdown_callback,
    register_session_slider_callback
)
from .dcc_graphs import get_general_graph_component

def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
    # Register the callbacks
    register_paradigm_dropdown_callback(app, global_data, 'UnityTrialwiseMetrics', vis_name)
    register_animal_dropdown_callback(app, global_data, 'UnityTrialwiseMetrics', vis_name)
    register_session_slider_callback(app, global_data, 'UnityTrialwiseMetrics', vis_name)

    @app.callback(
        Output(f'figure-{vis_name}', 'figure'),
        Input(f'animal-dropdown-{vis_name}', 'value'),
        Input(f'session-range-slider-{vis_name}', 'value'),
        # Input(f'group-by-{vis_name}', 'value'),
        # Input(f'metric-{vis_name}', 'value'),
        Input(f'max-metric-value-{vis_name}', 'value'),
        # Input(f'smooth-data-{vis_name}', 'value'),
        Input(f'outcome-group-filter-{vis_name}', 'value'),
        Input(f'cue-group-filter-{vis_name}', 'value'),
        Input(f'trial-group-filter-{vis_name}', 'value'),
        Input(f'var-vis-{vis_name}', 'value'),
        Input(f'dr-filter-{vis_name}', 'value'),
        
    )
    def update_plot(selected_animals, session_range, # group_by, metric,
                    metric_max, outcome_filter, cue_filter, trial_filter, var_vis,
                    double_reward_filter):
        
        if not all((selected_animals, metric_max)):
            return {}
        
        selected_sessions = np.arange(int(session_range[0]), int(session_range[1]) + 1)
        # TODO fix later 
        selected_sessions = [s for s in selected_sessions if s in global_data['UnityTrialwiseMetrics'].index.unique('session_id')]
        # animal and session filtering
        data = global_data['UnityTrialwiseMetrics'].loc[pd.IndexSlice[:,selected_animals,selected_sessions,:]]
        
        filtered_data = []
        for s_id in selected_sessions:
            d, group_by_values = group_filter_data(data.loc[pd.IndexSlice[:,:,s_id:s_id,:]], 
                                                   outcome_filter, cue_filter, trial_filter)
            filtered_data.append(d)
        data = pd.concat(filtered_data)
        
        # list to single value
        # if len(smooth_data) == 1:
        #     smooth_data = True

        fig = staytimes_plot_session.render_plot(data, metric_max, var_vis,
                                                 double_reward_filter=double_reward_filter)
        return fig
    
    return html.Div([
        dcc.Store(id=C.DATA_LOADED_StayPerformance_ID, data=False),  # Store to track data loading state
        dbc.Row([
            # Left side for plots
            dbc.Col([
                get_general_graph_component(vis_name)[1],
            ], width=7),

            # Right side for UI controls
            dbc.Col([
                # three rows, header, dropdown/tickpoxes, range slider
                dbc.Row([
                    html.H5(f"Data Selection for {vis_name}", style={"marginTop": 20}),
                ]),                                
                
                dbc.Row([
                    dbc.Col([
                        # Dropdown for paradigm selection
                        *get_paradigm_dropdown_component(vis_name, global_data, 'UnityTrialwiseMetrics'),
                        # Dropdown for animal selection
                        *get_animal_dropdown_component(vis_name, global_data, 'UnityTrialwiseMetrics'),
                        # Radioitems for metric selection
                        # *get_metric_radioitems_component(vis_name),
                    ], width=4),
                    
                    # Other options in middle column
                    dbc.Col([
                        # Radioitems for group by selection
                        # *get_groupby_radioitems_component(vis_name),
                        # html.Label("Group by", style={"marginTop": 15}),
                        dcc.RadioItems(
                            ['Distribution', 'Average', ],
                            inputStyle={"margin-right": "5px"},
                            style={"marginLeft": 5},
                            value='Average',
                            id=f'var-vis-{vis_name}'
                        ),
                        
                        html.Label("Double Rewards", style={"marginTop": 15}),
                        dcc.Checklist(
                            id=f'dr-filter-{vis_name}',
                            options=['Early R', 'Late R'],
                            value=['Early R', 'Late R'],
                            inline=True,
                            inputStyle={"margin-right": "7px", "margin-left": "3px"}
                        ),
                        
                        
                    ], width=4),
                    
                    # Other options in right column
                    dbc.Col([
                        # Filter checklist
                        *get_filter_checklist_component(vis_name, with_trial_group=True),
                        html.Hr(),
                        # Display options checklist, smooth and max metric
                        *get_display_options_checklist_component(vis_name, with_smooth=False,
                                                                 initial_value=8),
                    ], width=4),
                ]),
                
                # Range slider for trial selection
                *get_session_range_slider_component(vis_name),
            ], width=5)
        ]),
        html.Hr()
    ], id=f"{vis_name}-container")  # Initial state is hidden
    
    
def group_filter_data(data, outcome_filter, cue_filter, trial_filter, group_by="None"):
    group_values = {}
    # outcome filtering
    one_r_outcomes = [1,11,21,31,41,51,10,20,30,40,50]
    if '1 R' in outcome_filter:
        # group_values['1 R'] = [1]
        group_values['1 R'] = one_r_outcomes
    if '1+ R' in outcome_filter:
        group_values['1+ R'] = [i for i in range(1,56) if i not in one_r_outcomes]
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