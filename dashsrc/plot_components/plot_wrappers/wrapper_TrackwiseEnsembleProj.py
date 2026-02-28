from dash import html, dcc, Input, Output, State, Dash, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from ...components.dcc_graphs import get_general_graph_component
from .data_selection_components import (
    paradigm_dropdown_component,
    animal_dropdown_component,
    session_range_slider_component,
    metric_radioitems_component,
    outcome_group_filter_component,
    cue_group_filter_component,
    trial_group_filter_component,
    max_metric_input_component,
    smooth_checklist_component,
    normalize_checklist_component,
    
    register_animal_dropdown_callback,
    register_session_slider_callback,
    register_paradigm_dropdown_callback,
    get_session_slice_from_range,
)
from .data_selection import group_filter_data
from ..plots import plot_TrackFiringRate
import dashsrc.components.dashvis_constants as C

def calculate_figure(selected_paradigm, selected_animal, session_range,
                    metric, metric_max, smooth_data, normalize_data,
                    outcome_filter, cue_filter, trial_filter,
                    colormap,
                    global_data, prim_analytic, sec_analytic):
    
    if not all((selected_paradigm, selected_animal, metric_max)):
        return {}
    
    paradigm_slice = slice(selected_paradigm, selected_paradigm)
    animal_slice = slice(selected_animal, selected_animal)

    # try:
    #     global_data[sec_analytic]['paradigm_id'] = selected_paradigm
    #     global_data[sec_analytic]['animal_id'] = selected_animal
    #     global_data[sec_analytic] = global_data[sec_analytic].set_index(['paradigm_id', 'animal_id','session_id']).sort_index()  
    # except KeyError:
    #     pass
    # Safety check for session_range
    if not session_range: 
        return {}
        
    session_slice = get_session_slice_from_range(
        global_data[prim_analytic],
        selected_animal,
        session_range,
        selected_paradigm=selected_paradigm,
    )
    if len(session_slice) == 0:
        return {}
    
    # paradigm, animal and session filtering
    prim_data = global_data[prim_analytic].loc[pd.IndexSlice[paradigm_slice, animal_slice, 
                                                    session_slice, :]]
    sec_data = global_data[sec_analytic]
    session_slice_str = [str(s) for s in session_slice]
    if isinstance(sec_data.index, pd.MultiIndex) and 'session_id' in sec_data.index.names:
        sec_mask = sec_data.index.get_level_values('session_id').astype(str).isin(session_slice_str)
        sec_data = sec_data[sec_mask]
    elif 'session_id' in sec_data.columns:
        sec_data = sec_data[sec_data['session_id'].astype(str).isin(session_slice_str)].copy()
        sec_data = sec_data.set_index('session_id')
    else:
        sec_data = sec_data.loc[session_slice, :]
    n_sessions = len(session_slice)
    
    # filter the data based on the group by values
    prim_data, _ = group_filter_data(prim_data, outcome_filter, cue_filter, 
                                trial_filter)
    sec_data, _ = group_filter_data(sec_data, outcome_filter, cue_filter, 
                                trial_filter)
    
    # list to single value conversion
    if smooth_data and len(smooth_data) >= 1:
        smooth_data = True
    else:
        smooth_data = False

    if normalize_data and len(normalize_data) >= 1:
        normalize_data = True
    else:
        normalize_data = False
        
    fig = plot_TrackFiringRate.render_plot(prim_data, sec_data, global_data['SessionMetadata'], 
                                            global_data['SpikeClusterMetadata'],
                                            n_sessions, metric_max, smooth_data, normalize_data,
                                            colormap=colormap)
    return fig


def colormap_dropdown_component(instance_vis_name):
    colormap_dropd_id = f"{instance_vis_name}-colormap-dropdown"
    colormap_dropd = [
        html.Label("Color map"),
        dcc.Dropdown(
            id=colormap_dropd_id,
            options=[
                {"label": "Default", "value": "oxy"},
                {"label": "Viridis", "value": "viridis"},
                {"label": "Solar", "value": "solar"},
            ],
            value="oxy",
            clearable=False,
        ),
    ]
    return colormap_dropd, colormap_dropd_id


# Creates one instance of the plot + controls
def create_view_instance(app, global_data, base_vis_name, suffix_id, prim_analytic, sec_analytic):
    # Create a unique name for this instance to handle data processing
    instance_vis_name = f"{base_vis_name}{suffix_id}"
    
    comp_args = instance_vis_name, global_data, prim_analytic

    # Register the callbacks
    register_paradigm_dropdown_callback(app, *comp_args)
    register_animal_dropdown_callback(app, *comp_args)
    register_session_slider_callback(app, *comp_args)

    # create the html components to have their IDs (needed for the callbacks)
    paradigm_dropd, PARADIGM_DROPD_ID = paradigm_dropdown_component(*comp_args)
    animal_dropd, ANIMAL_DROPD_ID = animal_dropdown_component(*comp_args)
    session_slider, SESSION_SLIDER_ID = session_range_slider_component(instance_vis_name)
    metrics_radioi, METRICS_RADIOI_ID = metric_radioitems_component(instance_vis_name)
    maxmetric_inp, MAXMETRIC_INP_ID = max_metric_input_component(instance_vis_name, initial_value=80)
    smooth_checkl, SMOOTH_CHECKL_ID = smooth_checklist_component(instance_vis_name)
    normalize_checkl, NORMALIZE_CHECKL_ID = normalize_checklist_component(instance_vis_name)
    colormap_dropd, COLORMAP_DROPD_ID = colormap_dropdown_component(instance_vis_name)

    outcome_filter, OUTCOME_FILTER_ID = outcome_group_filter_component(instance_vis_name)
    cue_filter, CUE_FILTER_ID = cue_group_filter_component(instance_vis_name)
    trial_filter, TRIAL_FILTER_ID = trial_group_filter_component(instance_vis_name)
    
    graph, GRAPH_ID = get_general_graph_component(instance_vis_name)

    # Create the UI Layout for this instance
    layout = html.Div([
        dbc.Row([
            dbc.Col([graph], width=10),
            dbc.Col([
                dbc.Row([html.H5(f"Data Selection", style={"marginTop": 20})]),                                
                dbc.Row([
                    dbc.Col([*paradigm_dropd, *animal_dropd, *metrics_radioi], width=12),
                    dbc.Col([*outcome_filter, *cue_filter, *trial_filter, html.Hr(), *maxmetric_inp, *smooth_checkl, *normalize_checkl, *colormap_dropd], width=12),
                ]),
                *session_slider
            ], width=2)
        ])
    ])

    # Register the Main Update Callback for this instance
    @app.callback(
        Output(GRAPH_ID, 'figure'),
        Input(PARADIGM_DROPD_ID, 'value'),
        Input(ANIMAL_DROPD_ID, 'value'),
        Input(SESSION_SLIDER_ID, 'value'),
        Input(METRICS_RADIOI_ID, 'value'),
        Input(MAXMETRIC_INP_ID, 'value'),
        Input(SMOOTH_CHECKL_ID, 'value'),
        Input(NORMALIZE_CHECKL_ID, 'value'),
        Input(OUTCOME_FILTER_ID, 'value'),
        Input(CUE_FILTER_ID, 'value'),
        Input(TRIAL_FILTER_ID, 'value'),
        Input(COLORMAP_DROPD_ID, 'value'),
        # Input(WIDTH_INP_ID, 'value'),
        # Input(HEIGHT_INP_ID, 'value'),
    )
    def update_plot_wrapper(selected_paradigm, selected_animal, session_range,
                            metric, metric_max, smooth_data, normalize_data,
                            outcome_filter, cue_filter, trial_filter, colormap):
                            # width, height
        
        return calculate_figure(
            selected_paradigm, selected_animal, session_range,
            metric, metric_max, smooth_data, normalize_data,
            outcome_filter, cue_filter, trial_filter, colormap,
            global_data, prim_analytic, sec_analytic
        )

    return layout



# main render function
def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
    prim_analytic = 'BehaviorTrackwise'
    sec_analytic = 'TrackwiseEnsembleProj'
    
    # IDs for the container and button
    CONTAINER_ID = f"{vis_name}-main-container"
    SPLIT_BTN_ID = f"{vis_name}-split-btn"
    LEFT_VIEW_WRAPPER = f"{vis_name}-left-wrapper"
    RIGHT_VIEW_WRAPPER = f"{vis_name}-right-wrapper"

    # 2 instances: one hidden in beginning
    left_view = create_view_instance(app, global_data, vis_name, "_L", prim_analytic, sec_analytic)
    right_view = create_view_instance(app, global_data, vis_name, "_R", prim_analytic, sec_analytic)

    # Callback to handle the Split Action
    @app.callback(
        Output(RIGHT_VIEW_WRAPPER, 'style'),
        Output(LEFT_VIEW_WRAPPER, 'width'),
        Output(RIGHT_VIEW_WRAPPER, 'width'),
        Output(SPLIT_BTN_ID, 'children'),
        Input(SPLIT_BTN_ID, 'n_clicks'),
        prevent_initial_call=True
    )
    def toggle_split(n_clicks):
        # If odd clicks split is active
        is_split = n_clicks is not None and n_clicks % 2 == 1
        if is_split:
            return {'display': 'block'}, 6, 6, "Merge View"
        else:
            return {'display': 'none'}, 12, 6, "Split View"

    return html.Div([
        dcc.Store(id=C.get_vis_name_data_loaded_id(vis_name), data=False),
        
        # Header with Split Button
        dbc.Row([
            dbc.Col(html.H4(f"{vis_name}"), width=10),
            dbc.Col(dbc.Button("Split View", id=SPLIT_BTN_ID, color="primary", outline=True, size="sm"), width=2, style={'textAlign': 'right'})
        ], className="mb-3"),

        html.Hr(),
        
        # Container holding both views
        dbc.Row([
            # Left View Container
            dbc.Col(left_view, id=LEFT_VIEW_WRAPPER, width=12, style={'transition': 'all 0.3s'}),
            
            # Right View Container (Hidden by default)
            dbc.Col(right_view, id=RIGHT_VIEW_WRAPPER, width=6, style={'display': 'none', 'transition': 'all 0.3s', 'borderLeft': '2px solid #ccc'}),
        ], id=CONTAINER_ID)

    ], id=f"{vis_name}-container")
