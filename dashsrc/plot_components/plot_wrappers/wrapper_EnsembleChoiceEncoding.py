from dash import html, dcc, Input, Output, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

import dashsrc.components.dashvis_constants as C
from ..plots import plot_EnsembleChoiceEncoding

from .. .components.dcc_graphs import get_general_graph_component
from .data_selection_components import (
    R1_choice_filter_component,
    R2_choice_filter_component,
    animal_dropdown_component,
    ensemble_dropdown_component,
    event_dropdown_component,
    session_range_slider_component,
    
    # metric_radioitems_component,
    groupby_radioitems_component,
    # variance_radioitems_component,
    outcome_group_filter_component,
    cue_group_filter_component,
    trial_group_filter_component,
    # max_metric_input_component,
    # smooth_checklist_component,
    

    register_animal_dropdown_callback,
    register_session_slider_callback,
    register_ensemble_dropdown_callback,
    register_event_dropdown_callback,
    
    # register_session_dropdown_callback,
    # register_paradigm_dropdown_callback,
)
from .data_selection import group_filter_data

def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
    analytic = 'Ensemble40msProjEventAligned'  # the global_data to be used for the plot
    # components with global_data depedency need these arguments
    comp_args = vis_name, global_data, analytic
    
    # Register the callbacks
    register_animal_dropdown_callback(app, vis_name, global_data, 'SessionMetadata')
    register_session_slider_callback(app, *comp_args)
    register_ensemble_dropdown_callback(app, *comp_args)
    register_event_dropdown_callback(app, *comp_args)
    
    # create the html components to have their IDs (needed for the callbacks)
    animal_dropd, ANIMAL_DROPD_ID = animal_dropdown_component(*comp_args)
    ensemble_dropd, ENSEMBLE_DROPD_ID = ensemble_dropdown_component(*comp_args)
    event_dropd, EVENT_DROPD_ID = event_dropdown_component(*comp_args)
    session_slider, SESSION_SLIDER_ID = session_range_slider_component(vis_name)

    # session_dropd, SESSION_DROPD_ID = session_dropdown_component(*comp_args)
    # these don't need global_data to be initialized    
    groupby_radioi, GROUPBY_RADIOI_ID = groupby_radioitems_component(vis_name)
    # metrics_radioi, METRICS_RADIOI_ID = metric_radioitems_component(vis_name)
    # maxmetric_inp, MAXMETRIC_INP_ID = max_metric_input_component(vis_name, initial_value=80)
    # smooth_checkl, SMOOTH_CHECKL_ID = smooth_checklist_component(vis_name)
    # varianve_radioi, VARIANCE_RADIOI_ID = variance_radioitems_component(vis_name)
    outcome_filter, OUTCOME_FILTER_ID = outcome_group_filter_component(vis_name)
    cue_filter, CUE_FILTER_ID = cue_group_filter_component(vis_name)
    trial_filter, TRIAL_FILTER_ID = trial_group_filter_component(vis_name)
    R1_choice_filter, R1_CHOICE_FILTER_ID = R1_choice_filter_component(vis_name)
    R2_choice_filter, R2_CHOICE_FILTER_ID = R2_choice_filter_component(vis_name)
    graph, GRAPH_ID = get_general_graph_component(vis_name)
    
    @app.callback(
        Output(GRAPH_ID, 'figure'),
        Input(ANIMAL_DROPD_ID, 'value'),
        Input(ENSEMBLE_DROPD_ID, 'value'),
        Input(EVENT_DROPD_ID, 'value'),
        Input(SESSION_SLIDER_ID, 'value'),
        Input(GROUPBY_RADIOI_ID, 'value'),
        # Input(METRICS_RADIOI_ID, 'value'),
        # Input(MAXMETRIC_INP_ID, 'value'),
        # Input(SMOOTH_CHECKL_ID, 'value'),
        # Input(VARIANCE_RADIOI_ID, 'value'),
        Input(OUTCOME_FILTER_ID, 'value'),
        Input(CUE_FILTER_ID, 'value'),
        Input(TRIAL_FILTER_ID, 'value'),
        Input(R1_CHOICE_FILTER_ID, 'value'),
        Input(R2_CHOICE_FILTER_ID, 'value'),
    )
    def update_plot(selected_animal, ens_selection, event_selection, session_slider, group_by, 
                    # metric, metric_max, smooth_data, var_viz,
                    outcome_filter, cue_filter, trial_filter,
                    R1_choice_filter, R2_choice_filter):

        # only render the plot if all the necessary global_data selections are made
        if not all((selected_animal, ens_selection, event_selection, session_slider)):
            return {}
        
        
        
        # animal_ids = [6]
        # session_ids = None
        # paradigm_ids = [1100, 0]
        # excl_session_names = ['2024-11-29_17-21_rYL006_P1100_LinearTrackStop_28min', 
        #                       '2025-01-21_18-49_rYL006_P1100_LinearTrackStop_30min',
        #                       '2024-12-02_16-09_rYL006_P1100_LinearTrackStop_28min']
        # cols = ["session_spike_count", "session_nsamples", "cluster_id", "unit_snr", "unit_Vpp"]
        # width = 700
        # height = 700
        # analytic = 'Ensemble40msProjEventAligned'
        # global_data[analytic] = analytics.get_analytics(analytic, mode='set',
        #                                         paradigm_ids=None,
        #                                         animal_ids=animal_ids,
        #                                         #  excl_session_names=excl_session_names,
        #                                         session_ids=session_ids)
        # global_data[analytic].set_index(['session_id', 't0', 'interval_t'], inplace=True,)
        
        invalid_session_ids = 10, 24, 25
        data = global_data[analytic][~global_data[analytic].index.get_level_values('session_id').isin(invalid_session_ids)].copy()

        # ens_selection = 'Assembly012'
        # event_selection = ['enter_afterCueZone',]
        # session_slider = [4, 19]
        valid_sessions = [sid for sid in range(session_slider[0], session_slider[1] + 1) 
                        if sid in data.index.unique('session_id')]

        data = data.loc[valid_sessions,]
        data = data[data.t0_event_name.isin(event_selection)]
        drp_ens_cols = [c for c in data.columns if c.startswith('Assembly') and c != ens_selection]
        data.drop(columns=drp_ens_cols, inplace=True)

        # simplification, 0 or one instead of 1, 1+ 0 rewards
        data.loc[:, 'trial_outcome'] = data.loc[:, 'trial_outcome'].astype(bool).astype(int)

        # outcome_filter = ['1 R', '1+ R', 'no R']
        # cue_filter = ['Cue1 trials', 'Cue2 trials']
        # part_of_session_filter = ['1/3', '2/3', '3/3']
        # r1_choice_filter = ['stop', 'skip']
        # r2_choice_filter = ['stop', 'skip']
        # group_by = 'Cue' #'R1 choice' #, 'R1 choice', 'R2 choice'
        data, group_by_values = group_filter_data(data, outcome_filter=outcome_filter,
                                                            cue_filter=cue_filter,
                                                            trial_filter=trial_filter,
                                                            r1_choice_filter=R1_choice_filter,
                                                            r2_choice_filter=R2_choice_filter,
                                                            group_by=group_by)
        print(group_by_values)
        # print(global_data[analytic])

        fig = plot_EnsembleChoiceEncoding.render_plot(data, ens_selection=ens_selection,)
        # fullfname = f'{output_dir}/cue1vs2_delayperiod_ens12.svg'
        # fig.write_image(fullfname, width=fig.layout.width, height=fig.layout.height, scale=1)
        # execute = f"code {fullfname}"
        # os.system(execute)
        # fig.show()
        return fig
    
    return html.Div([
        dcc.Store(id=C.get_vis_name_data_loaded_id(vis_name), data=False),  # Store to track global_data loading state
        dbc.Row([
            # Left side for plots
            dbc.Col([
                graph,
            ], width=9),
            # Right side for UI controls
            dbc.Col([
                # three rows, header, dropdown/tickpoxes, range slider
                dbc.Row([
                    html.H5(f"Data Selection for {vis_name}", style={"marginTop": 20}),
                ]),                                
                
                dbc.Row([
                    dbc.Col([
                        *animal_dropd,
                        *ensemble_dropd,
                        *event_dropd,
                        # *metrics_radioi,
                    ], width=4),
                    
                    # Other options in middle column
                    dbc.Col([
                        # Radioitems for group by selection and variance visualization
                        *groupby_radioi, # *varianve_radioi,
                    ], width=4),
                    
                    # Other options in right column
                    dbc.Col([
                        # Filter checklist, max metric and smooth global_data
                        *outcome_filter, *cue_filter, *trial_filter, *R1_choice_filter, *R2_choice_filter,
                        html.Hr(),
                        # *maxmetric_inp, *smooth_checkl,
                    ], width=4),
                ]),
                dbc.Row([
                    # Range slider for session selection
                    *session_slider,
                ]),
            ], width=3)
        ]),
        html.Hr()
    ], id=f"{vis_name}-container")  # Initial state is hidden
    
    
    