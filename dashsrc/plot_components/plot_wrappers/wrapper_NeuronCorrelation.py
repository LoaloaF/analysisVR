from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

import dashsrc.components.dashvis_constants as C
from ...components.dcc_graphs import get_general_graph_component
from .data_selection import group_filter_data
from .data_selection_components import (
    R1_choice_filter_component,
    R2_choice_filter_component,
    animal_dropdown_component,
    ensemble_dropdown_component,
    outcome_group_filter_component,
    cue_group_filter_component,
    trial_group_filter_component,
    paradigm_dropdown_component,
    register_animal_dropdown_callback,
    register_paradigm_dropdown_callback,
    register_session_slider_callback,
    session_range_slider_component,
    get_session_slice_from_range,
)
from ..plots import plot_NeuronCorrelation


ZONE_CUE = "#1f77b4"
ZONE_R1 = "#ff8c00"
ZONE_R2 = "#7b2cbf"
ZONE_DEFAULT = "#444444"


def _zone_color(position_bin):
    if -80 <= position_bin <= 25:
        return ZONE_CUE
    if 50 <= position_bin <= 110:
        return ZONE_R1
    if 170 <= position_bin <= 230:
        return ZONE_R2
    return ZONE_DEFAULT


def _build_position_marks(min_bin, max_bin):
    ticks = {
        int(min_bin),
        int(max_bin),
        -80,
        25,
        50,
        110,
        170,
        230,
    }
    for t in range(int(min_bin), int(max_bin) + 1, 20):
        ticks.add(int(t))

    marks = {}
    for t in sorted(x for x in ticks if min_bin <= x <= max_bin):
        marks[t] = {
            "label": str(t),
            "style": {
                "color": _zone_color(t),
                "fontSize": "10px",
            },
        }
    return marks


def _top_neurons_input_component(vis_name, default_value=20):
    comp_id = f"top-neurons-input-{vis_name}"
    return [
        html.Label("Top n neurons", style={"marginTop": 12}),
        dcc.Input(
            id=comp_id,
            type="number",
            min=2,
            max=256,
            step=1,
            value=default_value,
            debounce=True,
            style={"width": "70%"},
        ),
    ], comp_id


def _position_range_slider_component(vis_name):
    comp_id = f"position-range-slider-{vis_name}"
    return [
        html.Label("from_position_bin range", style={"marginTop": 12}),
        dcc.RangeSlider(
            id=comp_id,
            min=-80,
            max=230,
            value=[-80, 230],
            step=1,
            marks=_build_position_marks(-80, 230),
            allowCross=False,
            tooltip={"always_visible": False, "placement": "bottom"},
        ),
    ], comp_id


def _select_track_data(data, selected_paradigm, selected_animal, session_slice):
    if data is None or len(data) == 0:
        return data

    if not isinstance(data.index, pd.MultiIndex):
        return data.iloc[0:0]

    try:
        return data.loc[pd.IndexSlice[selected_paradigm, selected_animal, session_slice, :]].copy()
    except Exception:
        idx = data.index
        mask = pd.Series(True, index=idx)

        if "paradigm_id" in idx.names:
            mask &= idx.get_level_values("paradigm_id").astype(str) == str(selected_paradigm)
        if "animal_id" in idx.names:
            mask &= idx.get_level_values("animal_id").astype(str) == str(selected_animal)
        if "session_id" in idx.names:
            session_str = {str(s) for s in session_slice}
            mask &= idx.get_level_values("session_id").astype(str).isin(session_str)

        return data[mask.to_numpy()].copy()


def _sanitize_top_n(top_n):
    if top_n is None:
        return 20
    try:
        value = int(top_n)
    except Exception:
        return 20
    return max(2, value)


def render(app: Dash, global_data: dict, vis_name: str) -> html.Div:
    analytic = "FiringRateTrackwiseHz"
    ensemble_analytic = "ConcatenatedEnsambles40ms"
    comp_args = vis_name, global_data, analytic

    register_paradigm_dropdown_callback(app, *comp_args)
    register_animal_dropdown_callback(app, *comp_args)
    register_session_slider_callback(app, *comp_args)

    paradigm_dropd, PARADIGM_DROPD_ID = paradigm_dropdown_component(*comp_args)
    animal_dropd, ANIMAL_DROPD_ID = animal_dropdown_component(*comp_args)
    session_slider, SESSION_SLIDER_ID = session_range_slider_component(vis_name)
    ensemble_dropd, ENSEMBLE_DROPD_ID = ensemble_dropdown_component(
        vis_name, global_data, ensemble_analytic
    )
    top_n_inp, TOP_N_INP_ID = _top_neurons_input_component(vis_name, default_value=20)
    pos_slider, POSITION_SLIDER_ID = _position_range_slider_component(vis_name)

    outcome_filter, OUTCOME_FILTER_ID = outcome_group_filter_component(vis_name)
    cue_filter, CUE_FILTER_ID = cue_group_filter_component(vis_name)
    trial_filter, TRIAL_FILTER_ID = trial_group_filter_component(vis_name)
    r1_choice_filter, R1_CHOICE_FILTER_ID = R1_choice_filter_component(vis_name)
    r2_choice_filter, R2_CHOICE_FILTER_ID = R2_choice_filter_component(vis_name)

    graph, GRAPH_ID = get_general_graph_component(vis_name)

    @app.callback(
        Output(ENSEMBLE_DROPD_ID, "options"),
        Input(C.get_vis_name_data_loaded_id(vis_name), "data"),
    )
    def update_ensemble_options(data_loaded):
        default_option = [{"label": "None", "value": "__none__"}]
        data = global_data.get(ensemble_analytic)
        if not data_loaded or data is None:
            return default_option

        ensembles = [c for c in data.columns if isinstance(c, str) and c.startswith("Assembly")]
        ensemble_options = [{"label": int(ens[-3:]), "value": ens} for ens in ensembles]
        return default_option + ensemble_options

    @app.callback(
        Output(POSITION_SLIDER_ID, "min"),
        Output(POSITION_SLIDER_ID, "max"),
        Output(POSITION_SLIDER_ID, "value"),
        Output(POSITION_SLIDER_ID, "marks"),
        Input(PARADIGM_DROPD_ID, "value"),
        Input(ANIMAL_DROPD_ID, "value"),
        Input(SESSION_SLIDER_ID, "value"),
    )
    def update_position_slider(selected_paradigm, selected_animal, session_range):
        default_min, default_max = -80, 230
        if not all((selected_paradigm, selected_animal, session_range)):
            return (
                default_min,
                default_max,
                (default_min, default_max),
                _build_position_marks(default_min, default_max),
            )

        session_slice = get_session_slice_from_range(
            global_data[analytic],
            selected_animal,
            session_range,
            selected_paradigm=selected_paradigm,
        )
        if len(session_slice) == 0:
            return (
                default_min,
                default_max,
                (default_min, default_max),
                _build_position_marks(default_min, default_max),
            )

        track_data = _select_track_data(
            global_data[analytic], selected_paradigm, selected_animal, session_slice
        )
        if track_data is None or track_data.empty or "from_position_bin" not in track_data.columns:
            return (
                default_min,
                default_max,
                (default_min, default_max),
                _build_position_marks(default_min, default_max),
            )

        bins = pd.to_numeric(track_data["from_position_bin"], errors="coerce").dropna()
        if bins.empty:
            return (
                default_min,
                default_max,
                (default_min, default_max),
                _build_position_marks(default_min, default_max),
            )

        min_bin = int(np.floor(bins.min()))
        max_bin = int(np.ceil(bins.max()))
        if min_bin == max_bin:
            max_bin = min_bin + 1
        marks = _build_position_marks(min_bin, max_bin)
        return min_bin, max_bin, (min_bin, max_bin), marks

    @app.callback(
        Output(GRAPH_ID, "figure"),
        Input(PARADIGM_DROPD_ID, "value"),
        Input(ANIMAL_DROPD_ID, "value"),
        Input(SESSION_SLIDER_ID, "value"),
        Input(ENSEMBLE_DROPD_ID, "value"),
        Input(TOP_N_INP_ID, "value"),
        Input(POSITION_SLIDER_ID, "value"),
        Input(OUTCOME_FILTER_ID, "value"),
        Input(CUE_FILTER_ID, "value"),
        Input(TRIAL_FILTER_ID, "value"),
        Input(R1_CHOICE_FILTER_ID, "value"),
        Input(R2_CHOICE_FILTER_ID, "value"),
    )
    def update_plot(
        selected_paradigm,
        selected_animal,
        session_range,
        selected_ensemble,
        top_n,
        position_bin_range,
        outcome_values,
        cue_values,
        trial_values,
        r1_values,
        r2_values,
    ):
        if not all((selected_paradigm, selected_animal, session_range)):
            return {}

        if any(
            values is None or len(values) == 0
            for values in [outcome_values, cue_values, trial_values, r1_values, r2_values]
        ):
            return {}

        session_slice = get_session_slice_from_range(
            global_data[analytic],
            selected_animal,
            session_range,
            selected_paradigm=selected_paradigm,
        )
        if len(session_slice) == 0:
            return {}

        track_data = _select_track_data(
            global_data[analytic], selected_paradigm, selected_animal, session_slice
        )
        if track_data is None or track_data.empty:
            return {}

        unit_cols = [c for c in track_data.columns if isinstance(c, str) and c.startswith("Unit")]
        req_cols = ["trial_outcome", "cue", "trial_id", "choice_R1", "choice_R2", "from_position_bin"]
        keep_cols = [c for c in req_cols if c in track_data.columns] + unit_cols
        track_data = track_data[keep_cols].copy()
        if len(unit_cols) < 2:
            return {}

        if position_bin_range and "from_position_bin" in track_data.columns:
            lo, hi = sorted(position_bin_range)
            pos = pd.to_numeric(track_data["from_position_bin"], errors="coerce")
            track_data = track_data[pos.between(lo, hi, inclusive="both")]
            if track_data.empty:
                return {}

        track_data, _ = group_filter_data(
            track_data,
            outcome_filter=outcome_values,
            cue_filter=cue_values,
            trial_filter=trial_values,
            r1_choice_filter=r1_values,
            r2_choice_filter=r2_values,
            group_by="None",
        )
        if track_data.empty:
            return {}

        if selected_ensemble in ("None", "", "__none__"):
            selected_ensemble = None

        return plot_NeuronCorrelation.render_plot(
            data=track_data,
            session_order=session_slice,
            top_n=_sanitize_top_n(top_n),
            ensemble_name=selected_ensemble,
            ens_data=global_data.get(ensemble_analytic),
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
                                    dbc.Col([*paradigm_dropd, *animal_dropd, *ensemble_dropd, *top_n_inp], width=12),
                                    dbc.Col(
                                        [
                                            *outcome_filter,
                                            *cue_filter,
                                            *trial_filter,
                                            *r1_choice_filter,
                                            *r2_choice_filter,
                                            html.Hr(),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                            dbc.Row([*pos_slider]),
                            dbc.Row([*session_slider]),
                        ],
                        width=3,
                    ),
                ]
            ),
            html.Hr(),
        ],
        id=f"{vis_name}-container",
    )
