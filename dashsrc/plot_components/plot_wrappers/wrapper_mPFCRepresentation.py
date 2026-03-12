from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc
import numpy as np
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
from ..plots.plot_mPFCRepresentation import (
    _apply_group_filters,
    _extract_trial_table,
    _get_interval_trial_ids,
)


MAX_INTERVALS_PER_UNIT = 3


def _compute_group_n(behavior, t0_events, selected_animal, selected_session, selected_intervals, filters):
    """Return the number of trials that match *filters* and appear in the selected intervals."""
    b = _slice_single_session(behavior, selected_animal, selected_session)
    t0 = _slice_single_session(t0_events, selected_animal, selected_session)
    trial_meta = _extract_trial_table(b)
    all_ids: set = set()
    for iv in (selected_intervals or [])[:MAX_INTERVALS_PER_UNIT]:
        all_ids.update(_get_interval_trial_ids(t0, iv).tolist())
    if not all_ids:
        return 0
    ids_arr = np.array(sorted(all_ids), dtype=int)
    mask = _apply_group_filters(trial_meta, ids_arr, filters)
    return int(mask.sum())


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
                html.Div(
                    id=f"{prefix}-header-label",
                    children=html.H4(title, style={"marginTop": 8, "fontWeight": "bold"}),
                ),
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
            "header_label": f"{prefix}-header-label",
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
    xmod_compare_dropdown_id = f"xmodality-compare-dropdown-{vis_name}"
    ypred_dropdown_id = f"ypred-dropdown-{vis_name}"
    interval_dropdown_g1_id = f"interval-dropdown-g1-{vis_name}"
    interval_dropdown_g2_id = f"interval-dropdown-g2-{vis_name}"
    compare_mode_id = f"compare-mode-{vis_name}"
    layout_mode_id = f"layout-mode-{vis_name}"
    g2_panel_id = f"g2-panel-{vis_name}"
    interval_g2_row_id = f"interval-g2-row-{vis_name}"

    group1_div, g1_ids = _group_controls(vis_name, "g1", "Group 1")
    group2_div, g2_ids = _group_controls(vis_name, "g2", "Group 2")

    graph, GRAPH_ID = get_general_graph_component(vis_name)

    @app.callback(
        Output(xmod_dropdown_id, "options"),
        Output(xmod_dropdown_id, "value"),
        Output(xmod_compare_dropdown_id, "options"),
        Output(xmod_compare_dropdown_id, "value"),
        Output(ypred_dropdown_id, "options"),
        Output(ypred_dropdown_id, "value"),
        Output(interval_dropdown_g1_id, "options"),
        Output(interval_dropdown_g1_id, "value"),
        Output(interval_dropdown_g2_id, "options"),
        Output(interval_dropdown_g2_id, "value"),
        Input(ANIMAL_DROPD_ID, "value"),
        Input(SESSION_DROPD_ID, "value"),
    )
    def update_selector_options(selected_animal, selected_session):
        if selected_animal is None or selected_session is None:
            return [], None, [], None, [], None, [], [], [], []

        d = _slice_single_session(global_data[analytic], selected_animal, selected_session)
        if d is None or len(d) == 0:
            return [], None, [], None, [], None, [], [], [], []

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
        xmod_compare_options = [{"label": x, "value": x} for x in xmods]
        ypred_options = [{"label": y, "value": y} for y in ypreds]
        interval_options = [{"label": i, "value": i} for i in intervals]

        xmod_default = xmods[0] if len(xmods) else None
        xmod_compare_default = xmods[1] if len(xmods) > 1 else xmod_default
        ypred_default = ypreds[0] if len(ypreds) else None
        interval_default_g1 = intervals[:MAX_INTERVALS_PER_UNIT]
        interval_default_g2 = intervals[:MAX_INTERVALS_PER_UNIT]

        return (
            xmod_options,
            xmod_default,
            xmod_compare_options,
            xmod_compare_default,
            ypred_options,
            ypred_default,
            interval_options,
            interval_default_g1,
            interval_options,
            interval_default_g2,
        )

    # ── Dynamic group-header labels (n = X trials) ─────────────────────────
    @app.callback(
        Output(g1_ids["header_label"], "children"),
        Output(g2_ids["header_label"], "children"),
        Input(ANIMAL_DROPD_ID, "value"),
        Input(SESSION_DROPD_ID, "value"),
        Input(compare_mode_id, "value"),
        Input(interval_dropdown_g1_id, "value"),
        Input(interval_dropdown_g2_id, "value"),
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
    def update_group_headers(
        selected_animal,
        selected_session,
        compare_mode,
        selected_intervals_g1,
        selected_intervals_g2,
        g1o, g1c, g1t, g1r1, g1r2,
        g2o, g2c, g2t, g2r1, g2r2,
    ):
        _ALL = {
            "outcome": ["1 R", "1+ R", "no R"],
            "cue": ["Cue1 trials", "Cue2 trials"],
            "trial": ["1/3", "2/3", "3/3"],
            "r1": ["stop", "skip"],
            "r2": ["stop", "skip"],
        }
        if compare_mode == "brain_area":
            g1_filters = g2_filters = _ALL
            label1, label2 = "Brain Area 1", "Brain Area 2"
            desc1 = desc2 = "all trials"
        elif compare_mode == "win_lose":
            g1_filters = {**_ALL, "outcome": ["1 R", "1+ R"]}
            g2_filters = {**_ALL, "outcome": ["no R"]}
            label1, label2 = "Win  (rewarded)", "Lose  (no reward)"
            desc1 = desc2 = None
        else:
            g1_filters = {"outcome": g1o or [], "cue": g1c or [], "trial": g1t or [], "r1": g1r1 or [], "r2": g1r2 or []}
            g2_filters = {"outcome": g2o or [], "cue": g2c or [], "trial": g2t or [], "r1": g2r1 or [], "r2": g2r2 or []}
            label1, label2 = "Group 1", "Group 2"
            desc1 = desc2 = None

        n1 = _compute_group_n(
            global_data.get("BehaviorTrialwise"),
            global_data.get("TrialWiseT0Events40ms"),
            selected_animal, selected_session, selected_intervals_g1, g1_filters,
        )
        n2 = _compute_group_n(
            global_data.get("BehaviorTrialwise"),
            global_data.get("TrialWiseT0Events40ms"),
            selected_animal, selected_session, selected_intervals_g2, g2_filters,
        )

        def _make_header(label, n, desc, color):
            n_str = f"n = {n} trials" if desc is None else f"{desc}  |  n = {n} trials"
            return [
                html.H4(label, style={"marginTop": 8, "fontWeight": "bold", "color": color, "marginBottom": 2}),
                html.Small(n_str, style={"color": "#666", "fontStyle": "italic"}),
            ]

        return (
            _make_header(label1, n1, desc1, "#1f77b4"),
            _make_header(label2, n2, desc2, "#ff7f0e"),
        )

    @app.callback(
        Output(g2_panel_id, "style"),
        Output(interval_g2_row_id, "style"),
        Input(layout_mode_id, "value"),
    )
    def toggle_g2_visibility(layout_mode):
        if layout_mode == "two":
            return {}, {}
        return {"display": "none"}, {"display": "none"}

    @app.callback(
        Output(GRAPH_ID, "figure"),
        Input(ANIMAL_DROPD_ID, "value"),
        Input(SESSION_DROPD_ID, "value"),
        Input(MODELS_RADIOI_ID, "value"),
        Input(METRICS_RADIOI_ID, "value"),
        Input(layout_mode_id, "value"),
        Input(compare_mode_id, "value"),
        Input(xmod_dropdown_id, "value"),
        Input(xmod_compare_dropdown_id, "value"),
        Input(ypred_dropdown_id, "value"),
        Input(interval_dropdown_g1_id, "value"),
        Input(interval_dropdown_g2_id, "value"),
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
        layout_mode,
        compare_mode,
        x_modality,
        x_modality_compare,
        predict_y_name,
        selected_intervals_g1,
        selected_intervals_g2,
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

        selected_intervals_g1 = (selected_intervals_g1 or [])[:MAX_INTERVALS_PER_UNIT]
        selected_intervals_g2 = (selected_intervals_g2 or [])[:MAX_INTERVALS_PER_UNIT]

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

        group1_label = "Group 1"
        group2_label = "Group 2"
        group1_modality = x_modality
        group2_modality = x_modality

        if compare_mode == "brain_area":
            group2_modality = x_modality_compare or x_modality
            group1_label = str(group1_modality)
            group2_label = str(group2_modality)
            # In brain-area mode, each column represents all trials for that area.
            group1_filters = {
                "outcome": ["1 R", "1+ R", "no R"],
                "cue": ["Cue1 trials", "Cue2 trials"],
                "trial": ["1/3", "2/3", "3/3"],
                "r1": ["stop", "skip"],
                "r2": ["stop", "skip"],
            }
            group2_filters = group1_filters.copy()
        elif compare_mode == "win_lose":
            group1_label = "Win"
            group2_label = "Lose"
            group1_filters = {
                "outcome": ["1 R", "1+ R"],
                "cue": ["Cue1 trials", "Cue2 trials"],
                "trial": ["1/3", "2/3", "3/3"],
                "r1": ["stop", "skip"],
                "r2": ["stop", "skip"],
            }
            group2_filters = {
                "outcome": ["no R"],
                "cue": ["Cue1 trials", "Cue2 trials"],
                "trial": ["1/3", "2/3", "3/3"],
                "r1": ["stop", "skip"],
                "r2": ["stop", "skip"],
            }

        if layout_mode == "two":
            fig = plot_mPFCRepresentation.render_two_group_columns(
                svm_data=svm_data,
                behavior_trialwise=behavior,
                t0_events=t0_events,
                behavior_aligned=behavior_aligned,
                predict_y_name=predict_y_name,
                model=model,
                group1_modality=group1_modality,
                group2_modality=group2_modality,
                group1_intervals=selected_intervals_g1,
                group2_intervals=selected_intervals_g2,
                group1_filters=group1_filters,
                group2_filters=group2_filters,
                group1_label=group1_label,
                group2_label=group2_label,
            )
        else:
            plot_mode = "brain_area" if compare_mode == "brain_area" else "group_custom"
            fig = plot_mPFCRepresentation.render_trial_split_plot(
                svm_data=svm_data,
                behavior_trialwise=behavior,
                t0_events=t0_events,
                behavior_aligned=behavior_aligned,
                x_modality=group1_modality,
                x_modality_compare=group2_modality,
                predict_y_name=predict_y_name,
                selected_intervals=selected_intervals_g1,
                model=model,
                compare_mode=plot_mode,
                group1_filters=group1_filters,
                group2_filters=group2_filters,
                group1_label=group1_label,
                group2_label=group2_label,
            )
        return fig

    return html.Div(
        [
            dcc.Store(id=C.get_vis_name_data_loaded_id(vis_name), data=False),
            # ── Section description ──────────────────────────────────────────
            html.Div(
                [
                    html.H2(
                        "mPFC / HP Neural Representation — SVM Decoding",
                        style={"marginTop": 12, "marginBottom": 4},
                    ),
                    html.P(
                        [
                            "An SVM decoder is trained on neural population activity (spike counts) "
                            "for a chosen ",
                            html.Strong("brain area"),
                            ", ",
                            html.Strong("decoded target"),
                            " (e.g. cue identity, trial outcome, R1/R2 choice), and "
                            "one or more ",
                            html.Strong("time intervals"),
                            ". "
                            "The macro-F1 score is then re-evaluated "
                            "\u2014 ",
                            html.Em("without re-training"),
                            " \u2014 "
                            "separately on two user-defined trial subsets, revealing how well the "
                            "population code generalises across different behavioural conditions. "
                            "Select a ",
                            html.Strong("comparison mode"),
                            " to compare:",
                        ],
                        style={"color": "#444", "fontSize": "0.92rem", "marginBottom": 4},
                    ),
                    html.Ul(
                        [
                            html.Li([html.Strong("Custom groups"), " — freely filter trials by outcome, cue, session third, and R1/R2 choice."]),
                            html.Li([html.Strong("Two brain areas"), " — same trial set decoded by Brain Area 1 vs Brain Area 2 (e.g. HP vs mPFC)."]),
                            html.Li([html.Strong("Win vs Lose"), " — rewarded trials (1 R or 1+ R) vs unrewarded trials (no R)."]),
                        ],
                        style={"color": "#444", "fontSize": "0.92rem", "marginTop": 0, "marginBottom": 8},
                    ),
                    html.Hr(style={"marginBottom": 6}),
                ],
                style={"paddingLeft": 12, "paddingRight": 12},
            ),
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
                                            html.Label("Layout", style={"marginTop": 15}),
                                            dcc.RadioItems(
                                                id=layout_mode_id,
                                                options=[
                                                    {"label": "Single column", "value": "single"},
                                                    {"label": "Two columns", "value": "two"},
                                                ],
                                                value="single",
                                                inline=True,
                                                inputStyle={"margin-right": "5px", "margin-left": "10px"},
                                            ),
                                            html.Label("Comparison mode", style={"marginTop": 15}),
                                            dcc.Dropdown(
                                                id=compare_mode_id,
                                                options=[
                                                    {"label": "Custom groups", "value": "custom"},
                                                    {"label": "Two brain areas", "value": "brain_area"},
                                                    {"label": "Win vs Lose", "value": "win_lose"},
                                                ],
                                                value="custom",
                                                clearable=False,
                                            ),
                                            html.Label("Brain area (X_modality)", style={"marginTop": 15}),
                                            dcc.Dropdown(id=xmod_dropdown_id, options=[], value=None),
                                            html.Label("Brain area 2 (for Two brain areas mode)", style={"marginTop": 15}),
                                            dcc.Dropdown(id=xmod_compare_dropdown_id, options=[], value=None),
                                            html.Label("Decoded target (predict_y_name)", style={"marginTop": 15}),
                                            dcc.Dropdown(id=ypred_dropdown_id, options=[], value=None),
                                            html.Label("Intervals Group 1 (max 3)", style={"marginTop": 15}),
                                            dcc.Dropdown(id=interval_dropdown_g1_id, options=[], value=[], multi=True),
                                            html.Div(
                                                id=interval_g2_row_id,
                                                children=[
                                                    html.Label("Intervals Group 2 (max 3)", style={"marginTop": 15}),
                                                    dcc.Dropdown(id=interval_dropdown_g2_id, options=[], value=[], multi=True),
                                                ],
                                                style={"display": "none"},
                                            ),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                            html.Hr(),
                            group1_div,
                            html.Div(id=g2_panel_id, children=[group2_div], style={"display": "none"}),
                        ],
                        width=3,
                    ),
                ]
            ),
            html.Hr(),
        ],
        id=f"{vis_name}-container",
    )
