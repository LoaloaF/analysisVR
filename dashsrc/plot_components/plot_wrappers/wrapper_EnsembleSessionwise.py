from dash import html, dcc, Input, Output, Dash
import dash_bootstrap_components as dbc
import pandas as pd

import dashsrc.components.dashvis_constants as C
from ..plots import plot_EnsembleSessionwise

from .. .components.dcc_graphs import get_general_graph_component
from .data_selection_components import (
    R1_choice_filter_component,
    R2_choice_filter_component,
    animal_dropdown_component,
    ensemble_dropdown_component,
    session_range_slider_component,
    groupby_radioitems_component,
    outcome_group_filter_component,
    cue_group_filter_component,
    trial_group_filter_component,
    register_animal_dropdown_callback,
    register_session_slider_callback,
    register_ensemble_dropdown_callback,
    get_session_slice_from_range,
)
from .data_selection import group_filter_data


PREFERRED_INTERVAL_ORDER = [
    "pre_cue_interval",
    "cue_entry_interval",
    "nextto_cue_interval",
    "cue_exit_interval",
    "R1_entry_interval",
    "R1_exit_interval",
    "R2_entry_interval",
    "R2_exit_interval",
]


def _pick_event_column(df, selected_events):
    if "interval_name" in df.columns and "t0_event_name" in df.columns:
        selected = set(selected_events)
        interval_overlap = len(selected.intersection(set(df["interval_name"].dropna().astype(str).unique())))
        t0_overlap = len(selected.intersection(set(df["t0_event_name"].dropna().astype(str).unique())))
        return "interval_name" if interval_overlap >= t0_overlap else "t0_event_name"
    if "interval_name" in df.columns:
        return "interval_name"
    if "t0_event_name" in df.columns:
        return "t0_event_name"
    return None


def _keep_required_columns(data, ens_selection):
    required_cols = {
        ens_selection,
        "animal_id",
        "session_id",
        "interval_name",
        "t0_event_name",
        "from_ephys_timestamp",
        "t0",
        "trial_outcome",
        "cue",
        "choice_R1",
        "choice_R2",
        "trial_id",
        "behavior_trial_id",
        "trial",
        "trial_index",
        "entry_id",
    }
    keep_cols = [c for c in data.columns if c in required_cols]
    return data.loc[:, keep_cols].copy()


def _filter_by_animal(data, selected_animal):
    if isinstance(data.index, pd.MultiIndex) and "animal_id" in data.index.names:
        animal_vals = data.index.get_level_values("animal_id")
        mask = animal_vals == selected_animal
        if not mask.any():
            mask = animal_vals.astype(str) == str(selected_animal)
        return data[mask].copy()
    if "animal_id" in data.columns:
        mask = data["animal_id"] == selected_animal
        if not mask.any():
            mask = data["animal_id"].astype(str) == str(selected_animal)
        return data[mask].copy()
    return data


def _get_ordered_event_values(data, selected_animal):
    if data is None or selected_animal is None:
        return []
    data = _filter_by_animal(data, selected_animal)
    if data is None or data.empty:
        return []

    source = data if ("interval_name" in data.columns or "t0_event_name" in data.columns) else data.reset_index()
    if "interval_name" in source.columns:
        events = source["interval_name"].dropna().astype(str).unique().tolist()
        ordered = [name for name in PREFERRED_INTERVAL_ORDER if name in events]
        ordered += [name for name in sorted(events) if name not in ordered]
        return ordered
    if "t0_event_name" in source.columns:
        return sorted(source["t0_event_name"].dropna().astype(str).unique().tolist())
    return []


def _filter_by_sessions(data, valid_sessions):
    valid_sessions_str = {str(s) for s in valid_sessions}
    if isinstance(data.index, pd.MultiIndex) and "session_id" in data.index.names:
        sess_vals = data.index.get_level_values("session_id").astype(str)
        return data[sess_vals.isin(valid_sessions_str)].copy()
    if "session_id" in data.columns:
        return data[data["session_id"].astype(str).isin(valid_sessions_str)].copy()
    return data


def _apply_event_filter(data, event_selection):
    if isinstance(event_selection, str):
        event_selection = [event_selection]
    event_selection = [str(ev) for ev in (event_selection or []) if ev is not None]

    source = data if ("interval_name" in data.columns or "t0_event_name" in data.columns) else data.reset_index()
    event_col = _pick_event_column(source, event_selection)
    if event_col is None:
        return data.iloc[0:0]

    if event_col in data.columns:
        return data[data[event_col].astype(str).isin(event_selection)].copy()

    tmp = data.reset_index()
    tmp = tmp[tmp[event_col].astype(str).isin(event_selection)]
    if tmp.empty:
        return tmp
    idx_cols = [c for c in data.index.names if c in tmp.columns]
    return tmp.set_index(idx_cols) if len(idx_cols) else tmp


def _ensure_session_index(data):
    if isinstance(data.index, pd.MultiIndex) and "session_id" in data.index.names:
        return data
    if isinstance(data.index, pd.MultiIndex):
        return data.set_index("session_id", append=True)
    return data.set_index("session_id")


def _prepare_filtered_data(
    raw_data,
    selected_animal,
    valid_sessions,
    ens_selection,
    event_selection,
    outcome_filter,
    cue_filter,
    trial_filter,
    r1_choice_filter,
    r2_choice_filter,
    group_by="None",
):
    data = _filter_by_animal(raw_data, selected_animal)

    invalid_session_ids = {"10", "24", "25"}
    if isinstance(data.index, pd.MultiIndex) and "session_id" in data.index.names:
        keep_mask = ~data.index.get_level_values("session_id").astype(str).isin(invalid_session_ids)
        data = data[keep_mask].copy()
    elif "session_id" in data.columns:
        data = data[~data["session_id"].astype(str).isin(invalid_session_ids)].copy()

    data = _filter_by_sessions(data, valid_sessions)
    if data.empty:
        return None, None

    data = _keep_required_columns(data, ens_selection)
    data = _apply_event_filter(data, event_selection)
    if data.empty:
        return None, None

    if "trial_outcome" in data.columns:
        data.loc[:, "trial_outcome"] = data.loc[:, "trial_outcome"].astype(bool).astype(int)

    data = _ensure_session_index(data)

    data, group_by_values = group_filter_data(
        data,
        outcome_filter=outcome_filter,
        cue_filter=cue_filter,
        trial_filter=trial_filter,
        r1_choice_filter=r1_choice_filter,
        r2_choice_filter=r2_choice_filter,
        group_by=group_by,
    )
    if data.empty:
        return None, None

    return data, group_by_values


def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
    analytic = "EnsembleT0Projection"
    comp_args = vis_name, global_data, analytic

    register_animal_dropdown_callback(app, vis_name, global_data, "SessionMetadata")
    register_session_slider_callback(app, vis_name, global_data, "SessionMetadata")
    register_ensemble_dropdown_callback(app, *comp_args)

    animal_dropd, ANIMAL_DROPD_ID = animal_dropdown_component(*comp_args)
    ensemble_dropd, ENSEMBLE_DROPD_ID = ensemble_dropdown_component(*comp_args)
    EVENT_DROPD_ID = f"event-dropdown-{vis_name}"
    event_dropd = [
        html.Div(
            [
                html.Label("Select event"),
                dcc.Dropdown(
                    id=EVENT_DROPD_ID,
                    options=[],
                    placeholder="Event ID",
                    multi=True,
                    maxHeight=180,
                    optionHeight=28,
                    className="compact-event-dropdown",
                    style={"fontSize": "10px", "lineHeight": "16px"},
                ),
            ],
            style={"marginTop": 15, "position": "relative", "zIndex": 3},
        )
    ]
    session_slider, SESSION_SLIDER_ID = session_range_slider_component(vis_name)

    groupby_radioi, GROUPBY_RADIOI_ID = groupby_radioitems_component(vis_name)
    outcome_filter, OUTCOME_FILTER_ID = outcome_group_filter_component(vis_name)
    cue_filter, CUE_FILTER_ID = cue_group_filter_component(vis_name)
    trial_filter, TRIAL_FILTER_ID = trial_group_filter_component(vis_name)
    R1_choice_filter, R1_CHOICE_FILTER_ID = R1_choice_filter_component(vis_name)
    R2_choice_filter, R2_CHOICE_FILTER_ID = R2_choice_filter_component(vis_name)
    amplitude_slider_id = f"amplitude-slider-{vis_name}"
    amplitude_slider = [
        html.Label("Amplitude scale", style={"marginTop": 15}),
        dcc.Slider(
            id=amplitude_slider_id,
            min=0.25,
            max=3.0,
            step=0.05,
            value=1.0,
            marks={0.5: "0.5", 1.0: "1", 2.0: "2", 3.0: "3"},
            tooltip={"always_visible": False, "placement": "bottom"},
        ),
    ]
    show_single_trials_id = f"show-single-trials-{vis_name}"
    show_single_trials_toggle = [
        html.Label("Display", style={"marginTop": 15}),
        dcc.Checklist(
            id=show_single_trials_id,
            options=[{"label": "Show single trials", "value": "show_single_trials"}],
            value=[],
            inputStyle={"margin-right": "7px", "margin-left": "3px"},
        ),
    ]
    graph, GRAPH_ID = get_general_graph_component(vis_name)
    sessions_graph, SESSIONS_GRAPH_ID = get_general_graph_component(
        f"{vis_name}-session-summary", fixed_height=220
    )

    @app.callback(
        Output(EVENT_DROPD_ID, "options"),
        Input(C.get_vis_name_data_loaded_id(vis_name), "data"),
        Input(ANIMAL_DROPD_ID, "value"),
        Input(ENSEMBLE_DROPD_ID, "value"),
    )
    def update_event_options(data_loaded, selected_animal, ens_selection):
        data = global_data[analytic]
        if not data_loaded or data is None or len(data) == 0:
            return []
        if selected_animal is None or ens_selection is None:
            return []

        events = _get_ordered_event_values(data, selected_animal)
        if not events:
            return []
        return [{"label": ev, "value": ev} for ev in events]

    @app.callback(
        Output(EVENT_DROPD_ID, "value"),
        Input(ANIMAL_DROPD_ID, "value"),
        Input(ENSEMBLE_DROPD_ID, "value"),
        Input(EVENT_DROPD_ID, "options"),
    )
    def default_select_all_events(selected_animal, ens_selection, event_options):
        if not all((selected_animal, ens_selection)):
            return []
        if not event_options:
            return []
        return [opt["value"] for opt in event_options]

    @app.callback(
        Output(SESSIONS_GRAPH_ID, "figure"),
        Output(GRAPH_ID, "figure"),
        Input(ANIMAL_DROPD_ID, "value"),
        Input(ENSEMBLE_DROPD_ID, "value"),
        Input(EVENT_DROPD_ID, "value"),
        Input(SESSION_SLIDER_ID, "value"),
        Input(GROUPBY_RADIOI_ID, "value"),
        Input(amplitude_slider_id, "value"),
        Input(show_single_trials_id, "value"),
        Input(OUTCOME_FILTER_ID, "value"),
        Input(CUE_FILTER_ID, "value"),
        Input(TRIAL_FILTER_ID, "value"),
        Input(R1_CHOICE_FILTER_ID, "value"),
        Input(R2_CHOICE_FILTER_ID, "value"),
    )
    def update_plot(
        selected_animal,
        ens_selection,
        event_selection,
        session_slider,
        group_by,
        amplitude_scale,
        show_single_trials_value,
        outcome_filter,
        cue_filter,
        trial_filter,
        r1_choice_filter,
        r2_choice_filter,
    ):
        if not all((selected_animal, ens_selection, event_selection, session_slider)):
            return {}, {}

        data = global_data[analytic]
        if data is None or len(data) == 0:
            return {}, {}

        valid_sessions = get_session_slice_from_range(
            global_data["SessionMetadata"],
            selected_animal,
            session_slider,
        )
        if len(valid_sessions) == 0:
            return {}, {}

        data, group_by_values = _prepare_filtered_data(
            data,
            selected_animal=selected_animal,
            valid_sessions=valid_sessions,
            ens_selection=ens_selection,
            event_selection=event_selection,
            outcome_filter=outcome_filter,
            cue_filter=cue_filter,
            trial_filter=trial_filter,
            r1_choice_filter=r1_choice_filter,
            r2_choice_filter=r2_choice_filter,
            group_by=group_by,
        )
        if data is None or data.empty:
            return {}, {}

        sessions_fig = plot_EnsembleSessionwise.render_session_summary_plot(
            data,
            ens_selection=ens_selection,
            event_selection=event_selection,
            group_by=group_by,
            group_by_values=group_by_values,
        )

        return sessions_fig, plot_EnsembleSessionwise.render_plot(
            data,
            ens_selection=ens_selection,
            event_selection=event_selection,
            group_by=group_by,
            group_by_values=group_by_values,
            amplitude_scale=amplitude_scale,
            show_single_trials="show_single_trials" in (show_single_trials_value or []),
        )

    return html.Div(
        [
            dcc.Store(id=C.get_vis_name_data_loaded_id(vis_name), data=False),
            dbc.Row(
                [
                    dbc.Col([graph], width=9),
                    dbc.Col(
                        [
                            dbc.Row([html.H5(f"Data Selection for {vis_name}", style={"marginTop": 20})]),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            *animal_dropd,
                                            *ensemble_dropd,
                                        ],
                                        width=4,
                                    ),
                                    dbc.Col([*groupby_radioi, *amplitude_slider, *show_single_trials_toggle], width=4),
                                    dbc.Col(
                                        [
                                            *outcome_filter,
                                            *cue_filter,
                                            *trial_filter,
                                            *R1_choice_filter,
                                            *R2_choice_filter,
                                            html.Hr(),
                                        ],
                                        width=4,
                                    ),
                                ]
                            ),
                            dbc.Row([dbc.Col([*event_dropd], width=12)], style={"marginTop": 8, "marginBottom": 18}),
                            dbc.Row([*session_slider]),
                            dbc.Row([dbc.Col([sessions_graph], width=12)], style={"marginTop": 76}),
                        ],
                        width=3,
                    ),
                ]
            ),
            html.Hr(),
        ],
        id=f"{vis_name}-container",
    )
