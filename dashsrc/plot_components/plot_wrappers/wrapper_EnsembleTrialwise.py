from dash import html, dcc, Input, Output, Dash
import dash_bootstrap_components as dbc
import pandas as pd

import dashsrc.components.dashvis_constants as C
from ..plots import plot_EnsembleTrialwise

from ...components.dcc_graphs import get_general_graph_component
from .data_selection_components import (
    R1_choice_filter_component,
    R2_choice_filter_component,
    animal_dropdown_component,
    ensemble_dropdown_component,
    event_dropdown_component,
    groupby_radioitems_component,
    outcome_group_filter_component,
    cue_group_filter_component,
    trial_group_filter_component,
    trial_range_slider_component,
    session_dropdown_component,
    register_animal_dropdown_callback,
    register_session_dropdown_callback,
    register_ensemble_dropdown_callback,
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


def _resolve_trial_column(df):
    trial_col_candidates = ["trial_id", "behavior_trial_id", "trial", "trial_index", "entry_id"]
    return next((c for c in trial_col_candidates if c in df.columns), None)


def _keep_required_columns(data, ens_selection):
    if data is None or data.empty:
        return data

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


def _filter_by_animal_session(data, selected_animal=None, selected_session=None):
    if data is None or len(data) == 0:
        return data.iloc[0:0] if data is not None else None

    if isinstance(data.index, pd.MultiIndex):
        mask = pd.Series(True, index=data.index)
        if "animal_id" in data.index.names and selected_animal is not None:
            animal_vals = data.index.get_level_values("animal_id")
            animal_mask = animal_vals == selected_animal
            if not animal_mask.any():
                animal_mask = animal_vals.astype(str) == str(selected_animal)
            mask &= animal_mask

        if "session_id" in data.index.names and selected_session is not None:
            sess_vals = data.index.get_level_values("session_id")
            sess_mask = sess_vals == selected_session
            if not sess_mask.any():
                sess_mask = sess_vals.astype(str) == str(selected_session)
            mask &= sess_mask
        return data[mask].copy()

    out = data.copy()
    if "animal_id" in out.columns and selected_animal is not None:
        amask = out["animal_id"] == selected_animal
        if not amask.any():
            amask = out["animal_id"].astype(str) == str(selected_animal)
        out = out[amask]
    if "session_id" in out.columns and selected_session is not None:
        smask = out["session_id"] == selected_session
        if not smask.any():
            smask = out["session_id"].astype(str) == str(selected_session)
        out = out[smask]
    return out


def _apply_event_filter(data, event_selection):
    if data is None or data.empty:
        return data

    if isinstance(event_selection, str):
        event_selection = [event_selection]
    event_selection = [str(ev) for ev in (event_selection or []) if ev is not None]
    if len(event_selection) == 0:
        return data.iloc[0:0]

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
    if data is None or data.empty:
        return data
    if isinstance(data.index, pd.MultiIndex) and "session_id" in data.index.names:
        return data
    if "session_id" not in data.columns:
        return data.iloc[0:0]
    if isinstance(data.index, pd.MultiIndex):
        return data.set_index("session_id", append=True)
    return data.set_index("session_id")


def _apply_trial_range_filter(data, trial_slider):
    if data is None or data.empty or trial_slider is None:
        return data
    source = data if "trial_id" in data.columns else data.reset_index()
    trial_col = _resolve_trial_column(source)
    if trial_col is None:
        return data.iloc[0:0]

    trial_min = pd.to_numeric(pd.Series([trial_slider[0]]), errors="coerce").iloc[0]
    trial_max = pd.to_numeric(pd.Series([trial_slider[1]]), errors="coerce").iloc[0]
    if pd.isna(trial_min) or pd.isna(trial_max):
        return data.iloc[0:0]
    if trial_min > trial_max:
        trial_min, trial_max = trial_max, trial_min

    if trial_col in data.columns:
        trial_vals = pd.to_numeric(data[trial_col], errors="coerce")
        return data[trial_vals.between(trial_min, trial_max, inclusive="both")].copy()

    tmp = data.reset_index()
    trial_vals = pd.to_numeric(tmp[trial_col], errors="coerce")
    tmp = tmp[trial_vals.between(trial_min, trial_max, inclusive="both")]
    if tmp.empty:
        return tmp
    idx_cols = [c for c in data.index.names if c in tmp.columns]
    return tmp.set_index(idx_cols) if len(idx_cols) else tmp


def _prepare_base_filtered_data(
    raw_data,
    selected_animal,
    ens_selection,
    event_selection,
    trial_slider=None,
):
    if raw_data is None or len(raw_data) == 0:
        return None
    data = raw_data

    invalid_session_ids = {"10", "24", "25"}

    data = _filter_by_animal_session(data, selected_animal)
    if data is None or data.empty:
        return None

    if isinstance(data.index, pd.MultiIndex) and "session_id" in data.index.names:
        keep_mask = ~data.index.get_level_values("session_id").astype(str).isin(invalid_session_ids)
        data = data[keep_mask].copy()
    elif "session_id" in data.columns:
        data = data[~data["session_id"].astype(str).isin(invalid_session_ids)].copy()
    if data is None or data.empty:
        return None

    data = _keep_required_columns(data, ens_selection)
    if data is None or data.empty:
        return None

    data = _apply_event_filter(data, event_selection)
    if data is None or data.empty:
        return None

    data = _apply_trial_range_filter(data, trial_slider)
    if data is None or data.empty:
        return None

    return data


def _apply_group_filters(
    data,
    outcome_filter,
    cue_filter,
    trial_filter,
    r1_choice_filter,
    r2_choice_filter,
    group_by="None",
):
    if data is None or data.empty:
        return None, None

    if "trial_outcome" in data.columns:
        data.loc[:, "trial_outcome"] = data.loc[:, "trial_outcome"].astype(bool).astype(int)

    data = _ensure_session_index(data)
    if data is None or data.empty:
        return None, None

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
    register_session_dropdown_callback(app, vis_name, global_data, "SessionMetadata")
    register_ensemble_dropdown_callback(app, *comp_args)

    animal_dropd, ANIMAL_DROPD_ID = animal_dropdown_component(vis_name, global_data, "SessionMetadata")
    session_dropd, SESSION_DROPD_ID = session_dropdown_component(vis_name, global_data, "SessionMetadata")
    ensemble_dropd, ENSEMBLE_DROPD_ID = ensemble_dropdown_component(*comp_args)
    event_dropd, EVENT_DROPD_ID = event_dropdown_component(*comp_args)
    trial_slider, TRIAL_SLIDER_ID = trial_range_slider_component(vis_name)

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

    graph, GRAPH_ID = get_general_graph_component(vis_name)
    sessions_graph, SESSIONS_GRAPH_ID = get_general_graph_component(
        f"{vis_name}-session-summary", fixed_height=320
    )

    @app.callback(
        Output(EVENT_DROPD_ID, "options"),
        Input(C.get_vis_name_data_loaded_id(vis_name), "data"),
    )
    def update_event_options(data_loaded):
        data = global_data[analytic]
        if not data_loaded or data is None or len(data) == 0:
            return []

        source = data if ("interval_name" in data.columns or "t0_event_name" in data.columns) else data.reset_index()
        if "interval_name" in source.columns:
            events = source["interval_name"].dropna().astype(str).unique().tolist()
            ordered = [name for name in PREFERRED_INTERVAL_ORDER if name in events]
            ordered += [name for name in sorted(events) if name not in ordered]
            return [{"label": ev, "value": ev} for ev in ordered]

        if "t0_event_name" in source.columns:
            events = sorted(source["t0_event_name"].dropna().astype(str).unique().tolist())
            return [{"label": ev, "value": ev} for ev in events]
        return []

    @app.callback(
        Output(EVENT_DROPD_ID, "value"),
        Input(ANIMAL_DROPD_ID, "value"),
        Input(SESSION_DROPD_ID, "value"),
        Input(ENSEMBLE_DROPD_ID, "value"),
        Input(EVENT_DROPD_ID, "options"),
    )
    def default_select_all_events(selected_animal, selected_session, ens_selection, event_options):
        if not all((selected_animal, (selected_session is not None), ens_selection)):
            return []
        if not event_options:
            return []
        return [opt["value"] for opt in event_options]

    @app.callback(
        Output(TRIAL_SLIDER_ID, "min"),
        Output(TRIAL_SLIDER_ID, "max"),
        Output(TRIAL_SLIDER_ID, "value"),
        Output(TRIAL_SLIDER_ID, "marks"),
        Input(ANIMAL_DROPD_ID, "value"),
        Input(SESSION_DROPD_ID, "value"),
        Input(EVENT_DROPD_ID, "value"),
    )
    def update_trial_slider(selected_animal, selected_session, event_selection):
        data = global_data[analytic]
        if selected_animal is None or selected_session is None or data is None or len(data) == 0:
            return 0, 0, (0, 0), {0: "no trials"}

        session_data = _filter_by_animal_session(data, selected_animal, selected_session)
        if session_data is None or session_data.empty:
            return 0, 0, (0, 0), {0: "no trials"}

        session_data = _apply_event_filter(session_data, event_selection)
        if session_data is None or session_data.empty:
            return 0, 0, (0, 0), {0: "no trials"}

        source = session_data if "trial_id" in session_data.columns else session_data.reset_index()
        trial_col = _resolve_trial_column(source)
        if trial_col is None:
            return 0, 0, (0, 0), {0: "no trials"}

        trials = pd.to_numeric(source[trial_col], errors="coerce").dropna().astype(int)
        if trials.empty:
            return 0, 0, (0, 0), {0: "no trials"}

        trial_min = int(trials.min())
        trial_max = int(trials.max())
        span = trial_max - trial_min + 1
        if span <= 20:
            marks = {t: str(t) for t in range(trial_min, trial_max + 1)}
        else:
            step = max(1, span // 10)
            marks = {
                trial_min: str(trial_min),
                trial_max: str(trial_max),
            }
            for t in range(trial_min, trial_max + 1, step):
                marks[t] = str(t)

        return trial_min, trial_max, (trial_min, trial_max), marks

    @app.callback(
        Output(SESSIONS_GRAPH_ID, "figure"),
        Output(GRAPH_ID, "figure"),
        Input(ANIMAL_DROPD_ID, "value"),
        Input(SESSION_DROPD_ID, "value"),
        Input(ENSEMBLE_DROPD_ID, "value"),
        Input(EVENT_DROPD_ID, "value"),
        Input(TRIAL_SLIDER_ID, "value"),
        Input(GROUPBY_RADIOI_ID, "value"),
        Input(amplitude_slider_id, "value"),
        Input(OUTCOME_FILTER_ID, "value"),
        Input(CUE_FILTER_ID, "value"),
        Input(TRIAL_FILTER_ID, "value"),
        Input(R1_CHOICE_FILTER_ID, "value"),
        Input(R2_CHOICE_FILTER_ID, "value"),
    )
    def update_plot(
        selected_animal,
        selected_session,
        ens_selection,
        event_selection,
        trial_slider,
        group_by,
        amplitude_scale,
        outcome_filter,
        cue_filter,
        trial_filter,
        r1_choice_filter,
        r2_choice_filter,
    ):
        if not all((selected_animal, (selected_session is not None), ens_selection, event_selection, trial_slider)):
            return {}, {}
        if any(v is None or len(v) == 0 for v in [outcome_filter, cue_filter, trial_filter, r1_choice_filter, r2_choice_filter]):
            return {}, {}

        base_data = _prepare_base_filtered_data(
            global_data[analytic],
            selected_animal=selected_animal,
            ens_selection=ens_selection,
            event_selection=event_selection,
            trial_slider=trial_slider,
        )
        if base_data is None or base_data.empty:
            return {}, {}

        data, group_by_values = _apply_group_filters(
            base_data,
            outcome_filter=outcome_filter,
            cue_filter=cue_filter,
            trial_filter=trial_filter,
            r1_choice_filter=r1_choice_filter,
            r2_choice_filter=r2_choice_filter,
            group_by=group_by,
        )
        if data is None or data.empty:
            return {}, {}

        sessions_fig = plot_EnsembleTrialwise.render_all_sessions_plot(
            data,
            ens_selection=ens_selection,
            event_selection=event_selection,
            selected_session=selected_session,
        )

        session_data = _filter_by_animal_session(data, selected_session=selected_session)
        if session_data is None or session_data.empty:
            return sessions_fig, {}

        return sessions_fig, plot_EnsembleTrialwise.render_plot(
            session_data,
            ens_selection=ens_selection,
            event_selection=event_selection,
            selected_session=selected_session,
            group_by=group_by,
            group_by_values=group_by_values,
            amplitude_scale=amplitude_scale,
        )

    return html.Div(
        [
            dcc.Store(id=C.get_vis_name_data_loaded_id(vis_name), data=False),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row([dbc.Col([graph], width=12)], align="start")
                        ],
                        width=9,
                    ),
                    dbc.Col(
                        [
                            dbc.Row([html.H5(f"Data Selection for {vis_name}", style={"marginTop": 20})]),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            *animal_dropd,
                                            *session_dropd,
                                            *ensemble_dropd,
                                            *event_dropd,
                                        ],
                                        width=4,
                                    ),
                                    dbc.Col([*groupby_radioi, *amplitude_slider], width=4),
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
                            dbc.Row([*trial_slider]),
                            dbc.Row([dbc.Col([sessions_graph], width=12)], style={"marginTop": 10}),
                        ],
                        width=3,
                    ),
                ]
            ),
            html.Hr(),
        ],
        id=f"{vis_name}-container",
    )
