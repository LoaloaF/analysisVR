from dash import html, dcc, Input, Output, State, Dash
import dash_bootstrap_components as dbc

from CustomLogger import CustomLogger as Logger

import dashsrc.components.dashvis_constants as C
from analytics_processing import analytics

def render(app: Dash, global_data: dict) -> html.Div:
    # component specific html ids
    D2M_ANIMALS_DROPDOWN_ID = 'd2m-animals-dropdown'
    D2M_PARADIGMS_DROPDOWN_ID = 'd2m-paradigm-dropdown'
    D2M_ANALYTICS_DROPDOWN_ID = 'd2m-analytics-dropdown'
    D2M_LOAD_DATA_BUTTON_ID = 'd2m-load-data-button'

    # outputs specifc for plot wise data selection
    data_loaded_plot_outputs = [Output(C.get_vis_name_data_loaded_id(vis_name), 'data') 
                                for vis_name in [*C.SESSION_WISE_VISS, *C.ANIMAL_WISE_VISS]]
    n_plots = len(data_loaded_plot_outputs)

    @app.callback(
        Output(D2M_LOAD_DATA_BUTTON_ID, 'style'),
        *data_loaded_plot_outputs,
        Input(D2M_LOAD_DATA_BUTTON_ID, 'n_clicks'),
        State(D2M_ANALYTICS_DROPDOWN_ID, 'value'),
        State(D2M_PARADIGMS_DROPDOWN_ID, 'value'),
        State(D2M_ANIMALS_DROPDOWN_ID, 'value'),
    )
    def load_data(n_clicks, selected_analytics, selected_paradigms, selected_animals):
        if n_clicks and selected_analytics:
            _load_all_data(selected_analytics, global_data, selected_paradigms, selected_animals)
            return {"marginTop": 15, "backgroundColor": "green"}, *([True]*n_plots)
        return {"marginTop": 15, "backgroundColor": "blue"}, *([False]*n_plots)
    
    default_animals = [6]
    default_paradigms = [1100]
    default_analytics = ['SessionMetadata', 'UnityTrialwiseMetrics']
    
    return dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id=D2M_ANALYTICS_DROPDOWN_ID,
                        options= list(global_data.keys()),
                        multi=True,
                        value=default_analytics,
                        placeholder="Select one or more analytics",
                        style={"marginTop": 15}
                    ),
                ], width=4),

                dbc.Col([
                    dcc.Dropdown(
                        id=D2M_PARADIGMS_DROPDOWN_ID,
                        options=C.PARADIGMS,
                        multi=True,
                        value=default_paradigms,
                        placeholder="Select paradigms",
                        style={"marginTop": 15}
                    ),
                ], width=3),
       
                dbc.Col([
                    dcc.Dropdown(
                        id=D2M_ANIMALS_DROPDOWN_ID,
                        options=C.ANIMALS,
                        multi=True,
                        value=default_animals,
                        placeholder="Select animals",
                        style={"marginTop": 15}
                    ),
                ], width=3),
                
                dbc.Col([
                    dbc.Button("Load Data", id=D2M_LOAD_DATA_BUTTON_ID, color="primary", 
                            style={"marginTop": 15}),
                ], width=2, )  # Light red background for debugging
        ])
    
def _load_all_data(selected_analytics, global_data, selected_paradigms, selected_animals):
    L = Logger()
    
    for analytic in selected_analytics:
        if len(selected_paradigms) == 0:
            selected_paradigms = None
        if len(selected_animals) == 0:
            selected_animals = None
        dat = analytics.get_analytics(analytic, mode='set', session_ids=None,
                                      paradigm_ids=selected_paradigms,
                                      animal_ids=selected_animals)
        global_data[analytic] = dat
        log_msg = {analytic: "Not loaded" if global_data[analytic] is None 
                   else f"Loaded ({global_data[analytic].shape})" 
                   for analytic in global_data.keys()}
        L.logger.debug(L.fmtmsg(log_msg))
    L.logger.info(L.fmtmsg(log_msg))