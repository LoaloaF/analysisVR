from dash import html, dcc, Input, Output, Dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

import dashsrc.components.dashvis_constants as C

from .. .components.dcc_graphs import get_general_graph_component
from .data_selection_components import (
    paradigm_dropdown_component,
    animal_dropdown_component,
    session_dropdown_component,
    trial_group_filter_component,
    max_metric_input_component,
    session_time_slider_component,
    digit_input_component,
    shank_input_component,
    trace_group_selecter_component,
    
    time_step_button_component,
    
    register_animal_dropdown_callback,
    register_session_dropdown_callback,
    register_paradigm_dropdown_callback,
    register_session_time_slider_callback,
    register_session_time_step_callback,
)
from .data_selection import group_filter_data
from ..plots import plot_RawSpikes

from analytics_processing.modality_loading import session_modality_from_nas
from analytics_processing.sessions_from_nas_parsing import sessionlist_fullfnames_from_args


def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
    analytic = 'SessionMetadata'
    # components with data depedency need these arguments
    comp_args = vis_name, global_data, analytic
    
    # Register the callbacks
    register_paradigm_dropdown_callback(app, *comp_args)
    register_animal_dropdown_callback(app, *comp_args)
    register_session_dropdown_callback(app, *comp_args)
    
    # timestamp related callbacks
    register_session_time_slider_callback(app, vis_name, global_data, 'ephys_traces')
    register_session_time_step_callback(app, vis_name, direction='forward')
    # register_session_time_step_callback(app, vis_name, direction='backward')
    
    # create the html components to have their IDs (needed for the callbacks)
    paradigm_dropd, PARADIGM_DROPD_ID = paradigm_dropdown_component(*comp_args)
    animal_dropd, ANIMAL_DROPD_ID = animal_dropdown_component(*comp_args)
    session_dropd, SESSION_DROPD_ID = session_dropdown_component(*comp_args)
    
    session_t_slider, SESSION_T_SLIDER_ID = session_time_slider_component(vis_name)
    interval_input, INTERVAL_INPUT_ID = digit_input_component(vis_name, "Interval [ms]", initial_value=300,
                                                              width=.3)
    time_step_forw_btn, _ = time_step_button_component(vis_name, direction='forward')
    time_step_backw_btn, _ = time_step_button_component(vis_name, direction='backward', disabled=True)
    shanks_input, SHANKS_INPUT_ID = shank_input_component(vis_name)
    trace_group_input, TRACE_GROUP_INPUT_ID = trace_group_selecter_component(vis_name)
    
    um_per_px_input, UM_PER_PX_INPUT_ID = digit_input_component(vis_name, "um-per-px", width=.8, 
                                                                initial_value=.25, step=.05)
    uV_per_um_input, UV_PER_UM_INPUT_ID = digit_input_component(vis_name, "um-per-uV", width=.8, 
                                                                initial_value=1, step=.05)
    

    graph, GRAPH_ID = get_general_graph_component(vis_name)
    
    @app.callback(
        Output(GRAPH_ID, 'figure'),
        Input(PARADIGM_DROPD_ID, 'value'),
        Input(ANIMAL_DROPD_ID, 'value'),
        Input(SESSION_DROPD_ID, 'value'),
        Input(SESSION_T_SLIDER_ID, 'value'),
        Input(INTERVAL_INPUT_ID, 'value'),
        Input(SHANKS_INPUT_ID, 'value'),
        Input(TRACE_GROUP_INPUT_ID, 'value'),
        Input(UM_PER_PX_INPUT_ID, 'value'),
        Input(UV_PER_UM_INPUT_ID, 'value'),
    )
    def update_plot(selected_paradigm, selected_animal, selected_session,
                    selected_start_time, interval, shanks, trace_groups,
                    um_per_px, uV_per_um):
        
        # only render the plot if all the necessary data selections are made   
        if not all((selected_paradigm, selected_animal, (selected_session is not None))):
            return {}
        
        paradigm_slice = slice(selected_paradigm)
        animal_slice = slice(selected_animal, selected_animal)
        session_slice = slice(selected_session, selected_session)
        
        # paradigm, animal and session filtering
        spikes = global_data['spikes'].loc[pd.IndexSlice[paradigm_slice, animal_slice, 
                                                         session_slice, :]]
        
        # print(selected_session)
        [print(i, type(global_data['ephys_traces'][i])) for i in range(len(global_data['ephys_traces']))]
        # print(global_data['ephys_traces'][selected_session])
        # print(global_data['implant_mapping'][selected_session])
        
        # print(selected_start_time) # in seconds
        # print(interval) # in milliseconds
        # convert to samples
        from_smpl = selected_start_time * 1_000_000 /50  # to us, then sample ID
        to_smpl = from_smpl + interval * 1_000 /50  # to us, then sample ID
        # print(f"from smpl: {from_smpl}, to smpl: {to_smpl}")
        # print(f"um_per_px: {um_per_px}, uV_per_um: {uV_per_um}")
                
        fig = plot_RawSpikes.render_plot(ephys_traces=global_data['ephys_traces'][selected_session],
                                        implant_mapping=global_data['implant_mapping'][selected_session],
                                        spikes=spikes,
                                        from_smpl=int(from_smpl),
                                        to_smpl=int(to_smpl),
                                        shanks=shanks,
                                        um_per_px=um_per_px,
                                        um_per_uV=uV_per_um,
                                        trace_groups=trace_groups,
        )
        return fig
    
    return html.Div([
        dcc.Store(id=C.get_vis_name_data_loaded_id(vis_name), data=False),  # Store to track data loading state
        dbc.Row([
            # Left side for plots
            dbc.Col([
                graph,
            ], width=10),
            # Right side for UI controls
            dbc.Col([
                # three rows, header, dropdown/tickpoxes, range slider
                dbc.Row([
                    html.H5(f"Data Selection for {vis_name}", style={"marginTop": 20}),
                ]),
                
                # Dropdown for animal session and metric selection
                *paradigm_dropd, *animal_dropd, *session_dropd,
                # seperator
                html.Hr(),
                *session_t_slider,
                # row with 2 columns
                dbc.Row([
                    # Other options in right column
                    dbc.Col([
                        *interval_input, 
                    ], width=7),
                    # Other options in right column
                    dbc.Col([
                        *time_step_backw_btn,
                    ], width=2),
                    # Other options in right column
                    dbc.Col([
                        *time_step_forw_btn,
                    ], width=2),
                ]),
                html.Hr(),
                *shanks_input,
                *trace_group_input,
                
                html.Hr(),
                dbc.Row([
                    # Other options in right column
                    dbc.Col([
                        *um_per_px_input,
                    ], width=6),
                    # Other options in right column
                    dbc.Col([
                        *uV_per_um_input,
                    ], width=6),
                ]),
                
                
            ], width=2)
        ]),
        html.Hr()
    ], id=f"{vis_name}-container")  # Initial state is hidden
    
    
    