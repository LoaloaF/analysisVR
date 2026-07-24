import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _get_unit_columns(data):
    return [c for c in data.columns if isinstance(c, str) and c.startswith("Unit")]


def _format_unit_label(unit_name):
    if isinstance(unit_name, str) and unit_name.startswith("Unit") and unit_name[4:].isdigit():
        return str(int(unit_name[4:]))
    return str(unit_name)


def _infer_weight_unit_names(ens_data):
    if ens_data is None or len(ens_data) == 0:
        return []

    if "cluster_id_str" in ens_data.columns:
        return ens_data["cluster_id_str"].astype(str).tolist()

    index = ens_data.index
    if isinstance(index, pd.MultiIndex):
        if "cluster_id_str" in index.names:
            raw = index.get_level_values("cluster_id_str")
        else:
            raw = index.get_level_values(-1)
    else:
        raw = index

    values = list(raw)
    if len(values) == 0:
        return []

    str_values = [str(v) for v in values]
    if all(v.startswith("Unit") for v in str_values):
        return str_values

    def _is_int_like(v):
        try:
            int(v)
            return True
        except Exception:
            return False

    if all(_is_int_like(v) for v in values):
        return [f"Unit{int(v) + 1:04d}" for v in values]

    return [f"Unit{i + 1:04d}" for i in range(len(values))]


def _get_ensemble_weight_series(ens_data, ensemble_name):
    if (
        ens_data is None
        or len(ens_data) == 0
        or ensemble_name is None
        or ensemble_name not in ens_data.columns
    ):
        return None

    weights = pd.to_numeric(ens_data[ensemble_name], errors="coerce")
    if weights.isna().all():
        return None

    unit_names = _infer_weight_unit_names(ens_data)
    if len(unit_names) != len(weights):
        unit_names = [f"Unit{i + 1:04d}" for i in range(len(weights))]

    series = pd.Series(weights.values, index=unit_names).dropna()
    if series.empty:
        return None

    # Keep one weight per unit id if duplicates exist.
    return series.groupby(level=0).mean()


def _select_ordered_units(data, top_n, ensemble_name=None, ens_data=None):
    unit_cols = _get_unit_columns(data)
    if len(unit_cols) == 0:
        return [], "no Unit columns available"

    top_n = max(2, min(int(top_n), len(unit_cols)))

    if ensemble_name:
        weights = _get_ensemble_weight_series(ens_data, ensemble_name)
        if weights is not None:
            weighted = weights.reindex(unit_cols).dropna()
            if not weighted.empty:
                ordered = weighted.abs().sort_values(ascending=False).index.tolist()
                return ordered[:top_n], f"{ensemble_name} absolute weights"

    corr = data[unit_cols].corr().abs()
    if corr.empty:
        return [], "correlation fallback unavailable"

    np.fill_diagonal(corr.values, np.nan)
    scores = corr.mean(axis=1, skipna=True).sort_values(ascending=False)
    ordered = scores.index.tolist()[:top_n]
    return ordered, "mean absolute Pearson correlation"


def _session_groups_by_name(data):
    if not isinstance(data.index, pd.MultiIndex) or "session_id" not in data.index.names:
        return {}
    return {str(session_id): sess_data for session_id, sess_data in data.groupby(level="session_id")}


def _session_trial_count(session_data):
    if "trial_id" in session_data.columns:
        trial_ids = pd.to_numeric(session_data["trial_id"], errors="coerce")
        return int(trial_ids.dropna().nunique())

    if isinstance(session_data.index, pd.MultiIndex) and "trial_id" in session_data.index.names:
        trial_ids = pd.to_numeric(
            session_data.index.get_level_values("trial_id"),
            errors="coerce",
        )
        return int(pd.Series(trial_ids).dropna().nunique())

    return 0


def render_plot(data, session_order, top_n=20, ensemble_name=None, ens_data=None):
    if data is None or len(data) == 0:
        return {}

    ordered_units, ordering_label = _select_ordered_units(
        data,
        top_n=top_n,
        ensemble_name=ensemble_name,
        ens_data=ens_data,
    )
    if len(ordered_units) < 2:
        return {}

    session_groups = _session_groups_by_name(data)
    session_keys = [str(s) for s in (session_order or [])]
    valid_sessions = [s for s in session_keys if s in session_groups]

    if len(valid_sessions) == 0:
        return {}

    n_sessions = len(valid_sessions)
    n_cols = min(4, n_sessions)
    n_rows = int(math.ceil(n_sessions / n_cols))
    n_slots = n_rows * n_cols

    subplot_titles = [
        f"S{s.split('_')[0]} (n={_session_trial_count(session_groups[s])})"
        for s in valid_sessions
    ] + [""] * (n_slots - n_sessions)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.03,
        vertical_spacing=0.06,
    )

    unit_labels = [_format_unit_label(u) for u in ordered_units]

    shown = 0
    for i, session_id in enumerate(valid_sessions):
        session_data = session_groups[session_id]
        if session_data.shape[0] < 2:
            continue

        corr = session_data[ordered_units].corr()
        corr = corr.reindex(index=ordered_units, columns=ordered_units).fillna(0.0)

        row = i // n_cols + 1
        col = i % n_cols + 1
        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=unit_labels,
                y=unit_labels,
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                showscale=(shown == 0),
                colorbar={"title": "Pearson r", "x": 1.02} if shown == 0 else None,
                hovertemplate="X: %{x}<br>Y: %{y}<br>r: %{z:.3f}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        shown += 1

    if shown == 0:
        return {}

    fig.update_xaxes(showticklabels=True, tickangle=90, tickfont={"size": 8})
    fig.update_yaxes(showticklabels=True, tickfont={"size": 8}, autorange="reversed")

    start_date = valid_sessions[0].split("_")[0]
    end_date = valid_sessions[-1].split("_")[0]
    fig.update_layout(
        title_text=(
            f"Top {len(ordered_units)} Neurons: Pearson Correlation Matrices Across Sessions "
            f"{start_date} to {end_date}<br>(ordered by {ordering_label})"
        ),
        title_font_size=16,
        width=300 * n_cols,
        height=300 * n_rows,
        showlegend=False,
        template="plotly_white",
        margin={"l": 30, "r": 20, "t": 95, "b": 20},
    )
    return fig
