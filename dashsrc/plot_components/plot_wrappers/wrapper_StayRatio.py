from dash import html, dcc, Input, Output, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from .. .components.dcc_graphs import get_general_graph_component
from .data_selection_components import (
    paradigm_dropdown_component,
    animal_dropdown_component,
    session_range_slider_component,
    outcome_group_filter_component,
    cue_group_filter_component,
    trial_group_filter_component,
    max_metric_input_component,
    figure_width_input_component,
    figure_height_input_component,
    
    register_animal_dropdown_callback,
    register_session_slider_callback,
    register_paradigm_dropdown_callback,
)
from .data_selection import group_filter_data
from ..plots import plot_StayRatio

import dashsrc.components.dashvis_constants as C

def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
    analytic = 'UnityTrialwiseMetrics'
    # components with data depedency need these arguments
    comp_args = vis_name, global_data, analytic
    
    # Register the callbacks
    register_paradigm_dropdown_callback(app, *comp_args)
    register_animal_dropdown_callback(app, *comp_args)
    register_session_slider_callback(app, *comp_args)
    
    # create the html components to have their IDs (needed for the callbacks)
    paradigm_dropd, PARADIGM_DROPD_ID = paradigm_dropdown_component(*comp_args)
    animal_dropd, ANIMAL_DROPD_ID = animal_dropdown_component(*comp_args)
    # these don't need data to be initialized    
    session_slider, SESSION_SLIDER_ID = session_range_slider_component(vis_name)
    maxmetric_inp, MAXMETRIC_INP_ID = max_metric_input_component(vis_name, initial_value=8)

    outcome_filter, OUTCOME_FILTER_ID = outcome_group_filter_component(vis_name)
    cue_filter, CUE_FILTER_ID = cue_group_filter_component(vis_name)
    trial_filter, TRIAL_FILTER_ID = trial_group_filter_component(vis_name)
    height_input, HEIGHT_INP_ID = figure_width_input_component(vis_name)
    width_input, WIDTH_INP_ID = figure_height_input_component(vis_name)

    graph, GRAPH_ID = get_general_graph_component(vis_name)
    
    @app.callback(
        Output(GRAPH_ID, 'figure'),
        Input(PARADIGM_DROPD_ID, 'value'),
        Input(ANIMAL_DROPD_ID, 'value'),
        Input(SESSION_SLIDER_ID, 'value'),
        Input(MAXMETRIC_INP_ID, 'value'),
        Input(OUTCOME_FILTER_ID, 'value'),
        Input(CUE_FILTER_ID, 'value'),
        Input(TRIAL_FILTER_ID, 'value'),
        Input(WIDTH_INP_ID, 'value'),
        Input(HEIGHT_INP_ID, 'value'),
        )
    def update_plot(selected_paradigm, selected_animal, session_range,
                    metric_max, outcome_filter, cue_filter, trial_filter,
                    width, height):
    
        if not all((selected_paradigm, selected_animal, metric_max)):
            return {}
        
        paradigm_slice = slice(selected_paradigm, selected_paradigm)
        animal_slice = slice(selected_animal, selected_animal)
        session_slice = [sid for sid in np.arange(session_range[0], session_range[1] + 1)
                         if sid in global_data[analytic].index.unique('session_id')]
        
        # paradigm, animal and session filtering
        data = global_data[analytic].loc[pd.IndexSlice[paradigm_slice, animal_slice, 
                                                       session_slice, :]]
        # filter the data based on the group by values
        data, _ = group_filter_data(data, outcome_filter, cue_filter, 
                                    trial_filter)
        
        sorted_by = 'session_id'
        # sorted_by = 'cues'
        print(data)
        fig = plot_StayRatio.render_plot(data, metric_max, sort_by=sorted_by)
        return fig
    
    return html.Div([
        dcc.Store(id=C.DATA_LOADED_StayRatio_ID, data=False),  # Store to track data loading state
        dbc.Row([
            # Left side for plots
            dbc.Col([
                graph,
            ], width=8),
            # Right side for UI controls
            dbc.Col([
                # three rows, header, dropdown/tickpoxes, range slider
                dbc.Row([
                    html.H5(f"Data Selection for {vis_name}", style={"marginTop": 20}),
                ]),                                
                
                dbc.Row([
                    dbc.Col([
                        # Dropdown for paradigm selection, animal selection
                        *paradigm_dropd, *animal_dropd,
                    ], width=4),
                    
                    # Other options in right column
                    dbc.Col([
                        *outcome_filter, *cue_filter, *trial_filter,
                        html.Hr(),
                        *maxmetric_inp,
                    ], width=4),

                    # Other options in right column
                    dbc.Col([
                        *width_input, *height_input,
                    ], width=4),
                ]),
                # Range slider for session selection
                *session_slider
            ], width=4)
        ]),
        html.Hr()
    ], id=f"{vis_name}-container")  # Initial state is hidden



















# def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
#     # Register the callbacks
#     register_paradigm_dropdown_callback(app, global_data, 'UnityTrialwiseMetrics', vis_name)
#     register_animal_dropdown_callback(app, global_data, 'UnityTrialwiseMetrics', vis_name)
#     register_session_slider_callback(app, global_data, 'UnityTrialwiseMetrics', vis_name)

#     @app.callback(
#         Output(f'figure-{vis_name}', 'figure'),
#         Input(f'animal-dropdown-{vis_name}', 'value'),
#         Input(f'session-range-slider-{vis_name}', 'value'),
#         # Input(f'group-by-{vis_name}', 'value'),
#         # Input(f'metric-{vis_name}', 'value'),
#         Input(f'max-metric-value-{vis_name}', 'value'),
#         # Input(f'smooth-data-{vis_name}', 'value'),
#         Input(f'outcome-group-filter-{vis_name}', 'value'),
#         Input(f'cue-group-filter-{vis_name}', 'value'),
#         Input(f'trial-group-filter-{vis_name}', 'value'),
        
#     )
#     def update_plot(selected_animals, session_range, # group_by, metric,
#                     metric_max, outcome_filter, cue_filter, trial_filter):
        
#         if not all((selected_animals, metric_max)):
#             return {}
        
#         selected_sessions = np.arange(int(session_range[0]), int(session_range[1]) + 1)
#         # TODO fix later 
#         selected_sessions = [s for s in selected_sessions if s in global_data['UnityTrialwiseMetrics'].index.unique('session_id')]
#         # animal and session filtering
#         data = global_data['UnityTrialwiseMetrics'].loc[pd.IndexSlice[:,selected_animals,selected_sessions,:]]
        
#         data, group_by_values = group_filter_data(data, outcome_filter, cue_filter, trial_filter)
        
        
#         # list to single value
#         # if len(smooth_data) == 1:
#         #     smooth_data = True

#         fig = staytimes_plot.render_plot(data, metric_max, )
#         return fig
    
#     return html.Div([
#         dcc.Store(id=C.DATA_LOADED_StayRatio_ID, data=False),  # Store to track data loading state
#         dbc.Row([
#             # Left side for plots
#             dbc.Col([
#                 get_general_graph_component(vis_name)[1],
#             ], width=7),

#             # Right side for UI controls
#             dbc.Col([
#                 # three rows, header, dropdown/tickpoxes, range slider
#                 dbc.Row([
#                     html.H5(f"Data Selection for {vis_name}", style={"marginTop": 20}),
#                 ]),                                
                
#                 dbc.Row([
#                     dbc.Col([
#                         # Dropdown for paradigm selection
#                         *get_paradigm_dropdown_component(vis_name, global_data, 'UnityTrialwiseMetrics'),
#                         # Dropdown for animal selection
#                         *get_animal_dropdown_component(vis_name, global_data, 'UnityTrialwiseMetrics'),
#                         # Radioitems for metric selection
#                         # *get_metric_radioitems_component(vis_name),
#                     ], width=4),
                    
#                     # Other options in middle column
#                     dbc.Col([
#                         # Radioitems for group by selection
#                         *get_groupby_radioitems_component(vis_name),
#                     ], width=4),
                    
#                     # Other options in right column
#                     dbc.Col([
#                         # Filter checklist
#                         *get_filter_checklist_component(vis_name, with_trial_group=True),
#                         html.Hr(),
#                         # Display options checklist, smooth and max metric
#                         *get_display_options_checklist_component(vis_name, with_smooth=False,
#                                                                  initial_value=8),
#                     ], width=4),
#                 ]),
                
#                 # Range slider for trial selection
#                 *get_session_range_slider_component(vis_name),
#             ], width=5)
#         ]),
#         html.Hr()
#     ], id=f"{vis_name}-container")  # Initial state is hidden
    
    
# def group_filter_data(data, outcome_filter, cue_filter, trial_filter, group_by="None"):
#     group_values = {}
#     # outcome filtering
#     one_r_outcomes = [1,11,21,31,41,51,10,20,30,40,50]
#     if '1 R' in outcome_filter:
#         # group_values['1 R'] = [1]
#         group_values['1 R'] = one_r_outcomes
#     if '1+ R' in outcome_filter:
#         group_values['1+ R'] = [i for i in range(1,56) if i not in one_r_outcomes]
#     if 'no R' in outcome_filter:
#         group_values['no R'] = [0]
#     data = data[data['trial_outcome'].isin(np.concatenate(list(group_values.values())))]
#     if group_by == 'Outcome':
#         group_by_values = group_values
    
#     # cue filtering
#     group_values = {}
#     if 'Early R' in cue_filter:
#         group_values['Early R'] = [1]
#     if 'Late R' in cue_filter:
#         group_values['Late R'] = [2]
#     data = data[data['cue'].isin(np.concatenate(list(group_values.values())))]
#     if group_by == 'Cue':
#         group_by_values = group_values
        
#     # trial filtering
#     group_values = {}
#     # get the 1st, 2nd, 3rd proportion of trials/ split in thirds
#     trial_groups = np.array_split(data['trial_id'].unique(), 3)
#     if "1/3" in trial_filter:
#         group_values["1/3"] = trial_groups[0]
#     if "2/3" in trial_filter:
#         group_values["2/3"] = trial_groups[1]
#     if "3/3" in trial_filter:
#         group_values["3/3"] = trial_groups[2]
#     incl_trials = np.concatenate([tg for tg in group_values.values()])
#     data = data[data['trial_id'].isin(incl_trials)]
#     if group_by == 'Part of session':
#         group_by_values = group_values
        
#     if group_by == 'None':
#         group_by_values = None
    
#     return data, group_by_values