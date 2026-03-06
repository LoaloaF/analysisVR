from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd

import dashsrc.components.dashvis_constants as C

from ...components.dcc_graphs import get_general_graph_component
from .data_selection_components import (
    animal_dropdown_component,
    metrics_radioitems_component,
    models_radioitems_component,
    register_animal_dropdown_callback,
    register_session_dropdown_callback,
    session_dropdown_component,
)
from ..plots import plot_mPFCRepresentation


def _slice_single_session(data, selected_animal, selected_session):
    if data is None or len(data) == 0:
        return None

    if isinstance(data.index, pd.MultiIndex):
        out = data
        if "animal_id" in out.index.names and selected_animal is not None:
            a_vals = out.index.get_level_values("animal_id")
            a_mask = a_vals == selected_animal
            if not a_mask.any():
                a_mask = a_vals.astype(str) == str(selected_animal)
            out = out[a_mask]

        if "session_id" in out.index.names and selected_session is not None:
            s_vals = out.index.get_level_values("session_id")
            s_mask = s_vals == selected_session
            if not s_mask.any():
                s_mask = s_vals.astype(str) == str(selected_session)
            out = out[s_mask]
        return out.copy()

    out = data.copy()
    if "animal_id" in out.columns and selected_animal is not None:
        a_mask = out["animal_id"] == selected_animal
        if not a_mask.any():
            a_mask = out["animal_id"].astype(str) == str(selected_animal)
        out = out[a_mask]

    if "session_id" in out.columns and selected_session is not None:
        s_mask = out["session_id"] == selected_session
        if not s_mask.any():
            s_mask = out["session_id"].astype(str) == str(selected_session)
        out = out[s_mask]

    return out


def _checklist(label, component_id, options, value):
    return [
        html.Label(label, style={"marginTop": 10}),
        dcc.Checklist(
            id=component_id,
            options=options,
            value=value,
            inline=True,
            inputStyle={"margin-right": "7px", "margin-left": "3px"},
        ),
    ]


def _group_controls(vis_name, group_key, title):
    prefix = f"split-{group_key}-{vis_name}"
    return (
        html.Div(
            [
                html.H6(title, style={"marginTop": 8}),
                *_checklist(
                    "Outcome",
                    f"{prefix}-outcome",
                    ["1 R", "1+ R", "no R"],
                    ["1 R", "1+ R", "no R"],
                ),
                *_checklist(
                    "Cue",
                    f"{prefix}-cue",
                    ["Cue1 trials", "Cue2 trials"],
                    ["Cue1 trials", "Cue2 trials"],
                ),
                *_checklist(
                    "Session Part",
                    f"{prefix}-trial",
                    ["1/3", "2/3", "3/3"],
                    ["1/3", "2/3", "3/3"],
                ),
                *_checklist(
                    "R1 Choice",
                    f"{prefix}-r1",
                    ["stop", "skip"],
                    ["stop", "skip"],
                ),
                *_checklist(
                    "R2 Choice",
                    f"{prefix}-r2",
                    ["stop", "skip"],
                    ["stop", "skip"],
                ),
            ],
            style={"border": "1px solid #ddd", "padding": "8px", "borderRadius": "6px", "marginBottom": "8px"},
        ),
        {
            "outcome": f"{prefix}-outcome",
            "cue": f"{prefix}-cue",
            "trial": f"{prefix}-trial",
            "r1": f"{prefix}-r1",
            "r2": f"{prefix}-r2",
        },
    )


def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
    analytic = "SVMCueOutcomeChoicePred"

    register_animal_dropdown_callback(app, vis_name, global_data, analytic)
    register_session_dropdown_callback(app, vis_name, global_data, analytic)

    animal_dropd, ANIMAL_DROPD_ID = animal_dropdown_component(vis_name, global_data, analytic)
    session_dropd, SESSION_DROPD_ID = session_dropdown_component(vis_name, global_data, analytic)
    models_radioi, MODELS_RADIOI_ID = models_radioitems_component(vis_name)
    metrics_radioi, METRICS_RADIOI_ID = metrics_radioitems_component(vis_name)

    xmod_dropdown_id = f"xmodality-dropdown-{vis_name}"
    ypred_dropdown_id = f"ypred-dropdown-{vis_name}"
    interval_dropdown_id = f"interval-dropdown-{vis_name}"

    group1_div, g1_ids = _group_controls(vis_name, "g1", "Group 1")
    group2_div, g2_ids = _group_controls(vis_name, "g2", "Group 2")

    graph, GRAPH_ID = get_general_graph_component(vis_name)

    @app.callback(
        Output(xmod_dropdown_id, "options"),
        Output(xmod_dropdown_id, "value"),
        Output(ypred_dropdown_id, "options"),
        Output(ypred_dropdown_id, "value"),
        Output(interval_dropdown_id, "options"),
        Output(interval_dropdown_id, "value"),
        Input(ANIMAL_DROPD_ID, "value"),
        Input(SESSION_DROPD_ID, "value"),
    )
    def update_selector_options(selected_animal, selected_session):
        if selected_animal is None or selected_session is None:
            return [], None, [], None, [], []

        d = _slice_single_session(global_data[analytic], selected_animal, selected_session)
        if d is None or len(d) == 0:
            return [], None, [], None, [], []

        src = d.reset_index() if isinstance(d.index, pd.MultiIndex) else d.copy()

        xmods = []
        if "X_modality" in src.columns:
            xmods = [str(v) for v in pd.unique(src["X_modality"].dropna())]

        ypreds = []
        if "predict_y_name" in src.columns:
            ypreds = [str(v) for v in pd.unique(src["predict_y_name"].dropna())]

        intervals = []
        if "interval_name" in src.columns:
            intervals = [str(v) for v in pd.unique(src["interval_name"].dropna())]

        xmod_options = [{"label": x, "value": x} for x in xmods]
        ypred_options = [{"label": y, "value": y} for y in ypreds]
        interval_options = [{"label": i, "value": i} for i in intervals]

        xmod_default = xmods[0] if len(xmods) else None
        ypred_default = ypreds[0] if len(ypreds) else None
        interval_default = intervals[:2] if len(intervals) > 1 else intervals

        return xmod_options, xmod_default, ypred_options, ypred_default, interval_options, interval_default

    @app.callback(
        Output(GRAPH_ID, "figure"),
        Input(ANIMAL_DROPD_ID, "value"),
        Input(SESSION_DROPD_ID, "value"),
        Input(MODELS_RADIOI_ID, "value"),
        Input(METRICS_RADIOI_ID, "value"),
        Input(xmod_dropdown_id, "value"),
        Input(ypred_dropdown_id, "value"),
        Input(interval_dropdown_id, "value"),
        Input(g1_ids["outcome"], "value"),
        Input(g1_ids["cue"], "value"),
        Input(g1_ids["trial"], "value"),
        Input(g1_ids["r1"], "value"),
        Input(g1_ids["r2"], "value"),
        Input(g2_ids["outcome"], "value"),
        Input(g2_ids["cue"], "value"),
        Input(g2_ids["trial"], "value"),
        Input(g2_ids["r1"], "value"),
        Input(g2_ids["r2"], "value"),
    )
    def update_plot(
        selected_animal,
        selected_session,
        model,
        which_metric,
        x_modality,
        predict_y_name,
        selected_intervals,
        g1_outcome,
        g1_cue,
        g1_trial,
        g1_r1,
        g1_r2,
        g2_outcome,
        g2_cue,
        g2_trial,
        g2_r1,
        g2_r2,
    ):
        _ = which_metric  # split plot currently recomputes Macro F1 by design
        if (
            selected_animal is None
            or selected_session is None
            or x_modality is None
            or predict_y_name is None
        ):
            return {}

        svm_data = _slice_single_session(global_data[analytic], selected_animal, selected_session)
        behavior = _slice_single_session(global_data.get("BehaviorTrialwise"), selected_animal, selected_session)
        t0_events = _slice_single_session(global_data.get("TrialWiseT0Events40ms"), selected_animal, selected_session)
        behavior_aligned = _slice_single_session(global_data.get("Behavior40msAligned"), selected_animal, selected_session)

        group1_filters = {
            "outcome": g1_outcome or [],
            "cue": g1_cue or [],
            "trial": g1_trial or [],
            "r1": g1_r1 or [],
            "r2": g1_r2 or [],
        }
        group2_filters = {
            "outcome": g2_outcome or [],
            "cue": g2_cue or [],
            "trial": g2_trial or [],
            "r1": g2_r1 or [],
            "r2": g2_r2 or [],
        }

        fig = plot_mPFCRepresentation.render_trial_split_plot(
            svm_data=svm_data,
            behavior_trialwise=behavior,
            t0_events=t0_events,
            behavior_aligned=behavior_aligned,
            x_modality=x_modality,
            predict_y_name=predict_y_name,
            selected_intervals=selected_intervals or [],
            model=model,
            group1_filters=group1_filters,
            group2_filters=group2_filters,
            group1_label="Group 1",
            group2_label="Group 2",
        )
        return fig

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
                                            *session_dropd,
                                            *models_radioi,
                                            *metrics_radioi,
                                            html.Label("Brain area (X_modality)", style={"marginTop": 15}),
                                            dcc.Dropdown(id=xmod_dropdown_id, options=[], value=None),
                                            html.Label("Decoded target (predict_y_name)", style={"marginTop": 15}),
                                            dcc.Dropdown(id=ypred_dropdown_id, options=[], value=None),
                                            html.Label("Intervals", style={"marginTop": 15}),
                                            dcc.Dropdown(id=interval_dropdown_id, options=[], value=[], multi=True),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                            html.Hr(),
                            group1_div,
                            group2_div,
                        ],
                        width=3,
                    ),
                ]
            ),
            html.Hr(),
        ],
        id=f"{vis_name}-container",
    )
