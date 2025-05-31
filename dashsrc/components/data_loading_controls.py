import os
from dash import html, dcc, Input, Output, State, Dash, no_update
import dash_bootstrap_components as dbc
from dash import ctx  # Dash >= 2.9

import numpy as np

from CustomLogger import CustomLogger as Logger

import dashsrc.components.dashvis_constants as C
from analytics_processing import analytics

from analytics_processing.sessions_from_nas_parsing import sessionlist_fullfnames_from_args
from analytics_processing.modality_loading import session_modality_from_nas


def render(app: Dash, loaded_analytics: dict, loaded_raw_traces: dict) -> html.Div:
    L = Logger()
    # component specific html ids
    D2M_ANIMALS_DROPDOWN_ID = 'd2m-animals-dropdown'
    D2M_PARADIGMS_DROPDOWN_ID = 'd2m-paradigm-dropdown'
    D2M_ANALYTICS_DROPDOWN_ID = 'd2m-analytics-dropdown'
    D2M_LOAD_DATA_BUTTON_ID = 'd2m-load-data-button'
    LOADING_OUTPUT_ID = 'loading-output'

    all_vis_names = C.SESSION_WISE_VISS + C.ANIMAL_WISE_VISS


    @app.callback(
        Output(D2M_ANALYTICS_DROPDOWN_ID, 'value'),
        [
            Input('d2m-auto-select-analytics', 'value'),
            *[Input(f'{vis_name}-button', 'n_clicks') for vis_name in all_vis_names]
        ],
        State(D2M_ANALYTICS_DROPDOWN_ID, 'value')
    )
    def auto_select_analytics(auto_select, *args_and_cur_value):
        *args, cur_value = args_and_cur_value

        # Only update if "auto_select" is checked
        if auto_select and 'auto_select' in auto_select:
            selected_viss = []
            for i, n_clicks in enumerate(args):
                if n_clicks is not None and n_clicks % 2 != 0:
                    selected_viss.append(all_vis_names[i])

            selected_analytics = list(set([
                analytic
                for vis_name in selected_viss
                for analytic in C.get_vis_name_req_data(vis_name)
            ]))

            print("Auto-select triggered:", selected_viss, selected_analytics)
            return selected_analytics

        # No update (preserve user's manual selection)
        print("Auto-select not triggered, preserving:", cur_value)
        return no_update



    # outputs specific for plot-wise data selection
    @app.callback(
        Output(D2M_LOAD_DATA_BUTTON_ID, 'style'),
        Output(LOADING_OUTPUT_ID, 'children'),
        *[Output(data_loaded_id, 'data') for data_loaded_id in C.get_all_data_loaded_ids()],
        Input(D2M_LOAD_DATA_BUTTON_ID, 'n_clicks'),
        State(D2M_ANALYTICS_DROPDOWN_ID, 'value'),
        State(D2M_PARADIGMS_DROPDOWN_ID, 'value'),
        State(D2M_ANIMALS_DROPDOWN_ID, 'value'),
    )
    def load_data(n_clicks, selected_analytics, selected_paradigms, selected_animals):
        n_plots = len(C.SESSION_WISE_VISS) + len(C.ANIMAL_WISE_VISS)
        if n_clicks and selected_analytics:
            # Show loading message
            loading_message = "Loading data, please wait..."
            
            _load_all_data(selected_analytics, loaded_analytics, loaded_raw_traces, 
                           selected_paradigms, selected_animals)
            
            data_exists = [np.all(np.isin(req_d, selected_analytics)).item()
                           for req_d in C.get_all_viss_req_data()]
            L.logger.debug(L.fmtmsg((dict(zip(C.get_all_data_loaded_ids(), data_exists)))))
                
            return {"marginTop": 15, "backgroundColor": "green"}, "", *data_exists
        return {"marginTop": 15, "backgroundColor": "blue"}, no_update, *([False] * n_plots)

    default_animals = [6]
    default_paradigms = [1100]
    default_analytics = ['SessionMetadata',]

    return dbc.Row([
        dbc.Col([
            dcc.Checklist(
                id='d2m-auto-select-analytics',
                options=[
                    {'label': 'match-vis', 'value': 'auto_select'},
                ],
                value=['auto_select'],
                inline=True,
                style={"marginTop": 15},
                inputStyle={"marginRight": 10}
            ),
        ], width=2),
        
        dbc.Col([
            dcc.Dropdown(
                id=D2M_ANALYTICS_DROPDOWN_ID,
                options=[*loaded_analytics.keys(), 'raw_traces'],
                multi=True,
                value=default_analytics,
                placeholder="Select one or more analytics",
                style={"marginTop": 15}
            ),

        ], width=3),

        dbc.Col([
            dcc.Dropdown(
                id=D2M_PARADIGMS_DROPDOWN_ID,
                options=C.PARADIGMS,
                multi=True,
                value=default_paradigms,
                placeholder="Select paradigms",
                style={"marginTop": 15}
            ),
        ], width=2),

        dbc.Col([
            dcc.Dropdown(
                id=D2M_ANIMALS_DROPDOWN_ID,
                options=C.ANIMALS,
                multi=True,
                value=default_animals,
                placeholder="Select animals",
                style={"marginTop": 15}
            ),
        ], width=2),

        dbc.Col([
                dbc.Button("Load Data", id=D2M_LOAD_DATA_BUTTON_ID, color="primary", 
                        style={"marginTop": 15}),
        ], width=2),
        
        dbc.Col([
            dcc.Loading(
                id="loading-spinner",
                type="circle",
                children=[
                    html.Div(id=LOADING_OUTPUT_ID, style={"marginTop": 35})
                ])
        ], width=1),
    ])
    
def _load_all_data(selected_analytics, loaded_analytics, loaded_raw_traces, selected_paradigms, selected_animals):
    L = Logger()
    
    selected_analytics_filt = [a for a in selected_analytics if a != 'raw_traces']
    for analytic in selected_analytics_filt:
        if len(selected_paradigms) == 0:
            selected_paradigms = None
        if len(selected_animals) == 0:
            selected_animals = None
        
        dat = analytics.get_analytics(analytic, mode='set', session_ids=None,
                                      paradigm_ids=selected_paradigms,
                                      animal_ids=selected_animals,)
                                    #   session_ids=np.arange(1,3))
        loaded_analytics[analytic] = dat
        
        # check if also raw_traces was in the passed selected_analytics list
        if len(selected_analytics) != len(selected_analytics_filt):
            session_ffnames, identifs = sessionlist_fullfnames_from_args(paradigm_ids=selected_paradigms, 
                                                                         animal_ids=selected_animals)
            for session_ffname, identf in zip(session_ffnames, identifs):
                raw_data_mmap, mapping = session_modality_from_nas(session_ffname, 'ephys_traces')
                loaded_raw_traces[str(identf)] = (raw_data_mmap, mapping)
                
    # raw traces
    log_msg = {identif: "Not loaded" if loaded_raw_traces[identif][0] is None 
               else str(loaded_raw_traces[identif][0].shape)
               for identif in loaded_raw_traces.keys()}
    L.logger.debug(L.fmtmsg(log_msg))
    L.logger.info("All data loaded.")
    # analytics
    log_msg = {analytic: "Not loaded" if loaded_analytics[analytic] is None 
               else f"Loaded ({loaded_analytics[analytic].shape})" 
               for analytic in loaded_analytics.keys()}
    L.logger.debug(L.fmtmsg(log_msg))