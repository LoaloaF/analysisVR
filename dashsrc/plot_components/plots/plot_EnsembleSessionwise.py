import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dashsrc.components.dashvis_constants as C


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


def _clean_interval_label(interval_name):
    return str(interval_name).replace("_interval", "").replace("_", " ")


def _pick_event_column(df, selected_events):
    if "interval_name" in df.columns and "t0_event_name" in df.columns:
        selected = set(selected_events)
        interval_overlap = len(selected.intersection(set(df["interval_name"].dropna().unique())))
        t0_overlap = len(selected.intersection(set(df["t0_event_name"].dropna().unique())))
        return "interval_name" if interval_overlap >= t0_overlap else "t0_event_name"
    if "interval_name" in df.columns:
        return "interval_name"
    if "t0_event_name" in df.columns:
        return "t0_event_name"
    return None


def _resolve_group_specs(group_by, group_by_values):
    if group_by == "None" or not group_by_values:
        return [{"name": "All", "column": None, "values": None, "color": "rgba(120,120,120,0.9)"}]

    if group_by == "Outcome":
        column = "trial_outcome"
        cmap = C.OUTCOME_COL_MAP
    elif group_by == "Cue":
        column = "cue"
        cmap = C.CUE_COL_MAP
    elif group_by == "Part of session":
        column = "trial_id"
        cmap = None
    elif group_by == "R1 choice":
        column = "choice_R1"
        cmap = C.R1_CHOICE_CMAP
    elif group_by == "R2 choice":
        column = "choice_R2"
        cmap = C.R2_CHOICE_CMAP
    else:
        return [{"name": "All", "column": None, "values": None, "color": "rgba(120,120,120,0.9)"}]

    specs = []
    fallback_cols = pc.qualitative.Set2 + pc.qualitative.Set3
    for i, (group_name, group_vals) in enumerate(group_by_values.items()):
        group_vals = list(group_vals)
        if len(group_vals) == 0:
            continue
        if cmap is None:
            color = fallback_cols[i % len(fallback_cols)]
        else:
            key = group_vals[0]
            color = cmap.get(key, fallback_cols[i % len(fallback_cols)])
        specs.append(
            {
                "name": str(group_name),
                "column": column,
                "values": group_vals,
                "color": color,
            }
        )
    return specs if specs else [{"name": "All", "column": None, "values": None, "color": "rgba(120,120,120,0.9)"}]


def _resolve_trial_column(df):
    trial_col_candidates = ["trial_id", "behavior_trial_id", "trial", "trial_index", "entry_id"]
    return next((c for c in trial_col_candidates if c in df.columns), None)


def _sort_session_ids(session_ids):
    session_ids = [str(sid) for sid in session_ids]
    sess_dt = pd.to_datetime(pd.Series(session_ids), format="%Y-%m-%d_%H-%M", errors="coerce")
    return (
        pd.DataFrame({"session_id": session_ids, "_dt": sess_dt})
        .drop_duplicates()
        .sort_values(["_dt", "session_id"])
        ["session_id"]
        .tolist()
    )


def render_session_summary_plot(data, ens_selection, event_selection, group_by="None", group_by_values=None):
    fig = go.Figure()
    if data is None or len(data) == 0 or ens_selection is None:
        return fig

    if isinstance(data.index, pd.MultiIndex):
        df = data.reset_index()
    elif data.index.name is not None:
        df = data.reset_index()
    else:
        df = data.copy()

    if ens_selection not in df.columns or "session_id" not in df.columns:
        return fig

    if isinstance(event_selection, str):
        event_selection = [event_selection]
    event_selection = [str(ev) for ev in (event_selection or []) if ev is not None]
    if len(event_selection) == 0:
        return fig

    event_col = _pick_event_column(df, event_selection)
    if "interval_name" not in df.columns:
        if event_col is None:
            return fig
        df = df.copy()
        df["interval_name"] = df[event_col].astype(str)
    else:
        df = df.copy()
        if event_col is not None and event_col in df.columns:
            df["interval_name"] = df["interval_name"].fillna(df[event_col]).astype(str)
        else:
            df["interval_name"] = df["interval_name"].astype(str)

    df[ens_selection] = pd.to_numeric(df[ens_selection], errors="coerce")
    df["session_id"] = df["session_id"].astype(str)
    df = df.dropna(subset=["session_id", "interval_name", ens_selection]).copy()
    if df.empty:
        return fig

    present_intervals = set(df["interval_name"].astype(str).unique())
    interval_order = [name for name in PREFERRED_INTERVAL_ORDER if name in present_intervals]
    interval_order += [name for name in event_selection if name in present_intervals and name not in interval_order]
    interval_order += [name for name in sorted(present_intervals) if name not in interval_order]
    if len(interval_order) == 0:
        return fig

    session_order = _sort_session_ids(df["session_id"].unique().tolist())
    if len(session_order) == 0:
        return fig

    group_specs = _resolve_group_specs(group_by, group_by_values)
    interval_groups = {
        interval_name: grp for interval_name, grp in df.groupby("interval_name", sort=False)
    }
    session_positions = np.arange(len(session_order), dtype=float)
    session_date_labels = [sid.split("_")[0] for sid in session_order]
    interval_labels = {
        name: _clean_interval_label(name)
        for name in interval_order
    }
    interval_palette = pc.qualitative.Plotly + pc.qualitative.Set2 + pc.qualitative.Set3
    interval_colormap = {
        name: interval_palette[i % len(interval_palette)]
        for i, name in enumerate(interval_order)
    }
    interval_dashes = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]

    for i_interval, interval in enumerate(interval_order):
        interval_df = interval_groups.get(interval)
        if interval_df is None or interval_df.empty:
            continue

        interval_label = interval_labels[interval]
        if group_by == "None":
            sess_mean = interval_df.groupby("session_id")[ens_selection].mean()
            y_vals = np.array([sess_mean.get(sid, np.nan) for sid in session_order], dtype=float)
            if not np.isfinite(y_vals).any():
                continue
            customdata = np.column_stack(
                [
                    np.array(session_order, dtype=object),
                    np.full(len(session_order), interval_label, dtype=object),
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x=session_positions,
                    y=y_vals,
                    mode="lines+markers",
                    name=f"{interval_label} summary",
                    legendgroup=f"summary__{interval}",
                    showlegend=False,
                    line=dict(color=interval_colormap[interval], width=2.2),
                    marker=dict(size=5, color=interval_colormap[interval]),
                    customdata=customdata,
                    hovertemplate=(
                        "session=%{customdata[0]}<br>"
                        "interval=%{customdata[1]}<br>"
                        "mean assembly=%{y:.3f}<extra></extra>"
                    ),
                )
            )
        else:
            dash_style = interval_dashes[i_interval % len(interval_dashes)]
            for spec in group_specs:
                if spec["column"] is None:
                    grp_interval_df = interval_df
                elif spec["column"] in interval_df.columns:
                    grp_interval_df = interval_df[interval_df[spec["column"]].isin(spec["values"])]
                else:
                    continue
                if grp_interval_df.empty:
                    continue

                sess_mean = grp_interval_df.groupby("session_id")[ens_selection].mean()
                y_vals = np.array([sess_mean.get(sid, np.nan) for sid in session_order], dtype=float)
                if not np.isfinite(y_vals).any():
                    continue
                customdata = np.column_stack(
                    [
                        np.array(session_order, dtype=object),
                        np.full(len(session_order), interval_label, dtype=object),
                        np.full(len(session_order), spec["name"], dtype=object),
                    ]
                )
                fig.add_trace(
                    go.Scatter(
                        x=session_positions,
                        y=y_vals,
                        mode="lines+markers",
                        name=f"{spec['name']} | {interval_label}",
                        legendgroup=f"summary__{spec['name']}__{interval}",
                        showlegend=False,
                        line=dict(color=spec["color"], width=2.0, dash=dash_style),
                        marker=dict(size=4.5, color=spec["color"]),
                        customdata=customdata,
                        hovertemplate=(
                            "session=%{customdata[0]}<br>"
                            "interval=%{customdata[1]}<br>"
                            "group=%{customdata[2]}<br>"
                            "mean assembly=%{y:.3f}<extra></extra>"
                        ),
                    )
                )

    fig.update_layout(
        template="plotly_white",
        title=dict(text="Session Interval Means", font=dict(size=11)),
        height=220,
        margin=dict(l=45, r=10, t=28, b=58),
        hovermode="closest",
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=session_positions.tolist(),
        ticktext=session_date_labels,
        tickangle=60,
        tickfont=dict(size=7),
        title_text="Sessions",
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        title_text=f"Mean {ens_selection}",
        showgrid=True,
        gridcolor="rgba(120,120,120,0.15)",
        zeroline=False,
    )
    return fig


def render_plot(
    data,
    ens_selection,
    event_selection,
    group_by="None",
    group_by_values=None,
    amplitude_scale=1.0,
    show_single_trials=False,
):
    fig = go.Figure()
    if data is None or len(data) == 0 or ens_selection is None:
        return fig

    try:
        amplitude_scale = float(amplitude_scale)
    except (TypeError, ValueError):
        amplitude_scale = 1.0
    if not np.isfinite(amplitude_scale):
        amplitude_scale = 1.0
    amplitude_scale = max(0.25, min(3.0, amplitude_scale))

    if isinstance(data.index, pd.MultiIndex):
        df = data.reset_index()
    elif data.index.name is not None:
        df = data.reset_index()
    else:
        df = data.copy()

    if ens_selection not in df.columns:
        return fig

    if isinstance(event_selection, str):
        event_selection = [event_selection]
    event_selection = [str(ev) for ev in (event_selection or []) if ev is not None]
    if len(event_selection) == 0:
        return fig

    event_col = _pick_event_column(df, event_selection)

    if "interval_name" not in df.columns:
        if event_col is None:
            return fig
        df = df.copy()
        df["interval_name"] = df[event_col].astype(str)
    else:
        df = df.copy()
        if event_col is not None and event_col in df.columns:
            df["interval_name"] = df["interval_name"].fillna(df[event_col]).astype(str)
        else:
            df["interval_name"] = df["interval_name"].astype(str)

    required_cols = ["session_id", "from_ephys_timestamp", "t0", ens_selection]
    for col in required_cols:
        if col not in df.columns:
            return fig
    trial_col = _resolve_trial_column(df)
    if trial_col is None:
        trial_col = "__trial_fallback"
        df[trial_col] = df.groupby(["session_id", "t0"], dropna=False).ngroup()

    for col in ["from_ephys_timestamp", "t0", ens_selection]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["session_id", trial_col, "from_ephys_timestamp", "t0", ens_selection]).copy()
    if df.empty:
        return fig

    df["session_id"] = df["session_id"].astype(str)
    df["rel_t_s"] = ((df["from_ephys_timestamp"] - df["t0"]) / 1_000_000.0).round(6)
    df = df.dropna(subset=["rel_t_s"]).copy()
    if df.empty:
        return fig

    present_intervals = set(df["interval_name"].astype(str).unique())
    interval_order = [name for name in PREFERRED_INTERVAL_ORDER if name in present_intervals]
    interval_order += [name for name in event_selection if name in present_intervals and name not in interval_order]
    interval_order += [name for name in sorted(present_intervals) if name not in interval_order]
    if len(interval_order) == 0:
        return fig

    # Build separated x-segments per interval.
    interval_windows = {}
    widths = []
    for interval in interval_order:
        vals = df.loc[df["interval_name"] == interval, "rel_t_s"].dropna()
        if vals.empty:
            continue
        lo, hi = float(vals.min()), float(vals.max())
        span = hi - lo
        pad = 0.03 * span if span > 0 else 0.15
        lo, hi = lo - pad, hi + pad
        if hi <= lo:
            hi = lo + 0.2
        interval_windows[interval] = (lo, hi)
        widths.append(hi - lo)
    interval_order = [name for name in interval_order if name in interval_windows]
    if len(interval_order) == 0:
        return fig

    segment_gap = max(0.08 * float(np.median(widths)) if len(widths) else 0.0, 0.08)
    interval_shift = {}
    interval_bounds = {}
    interval_centers = {}
    interval_zero_x = {}
    cursor = 0.0
    for i, interval in enumerate(interval_order):
        lo, hi = interval_windows[interval]
        if i > 0:
            cursor += segment_gap
        width = hi - lo
        x0, x1 = cursor, cursor + width
        shift = x0 - lo
        interval_shift[interval] = shift
        interval_bounds[interval] = (x0, x1)
        interval_centers[interval] = 0.5 * (x0 + x1)
        interval_zero_x[interval] = shift
        cursor = x1
    total_x_span = max(cursor, 1.0)
    x_range = [-0.01 * total_x_span, cursor + 0.01 * total_x_span]

    session_order = _sort_session_ids(df["session_id"].unique().tolist())
    if len(session_order) == 0:
        return fig
    top_panel_height = 180
    bottom_panel_height = max(42 * len(session_order) + 520, 900)
    total_plot_height = top_panel_height + bottom_panel_height

    act_vals = pd.to_numeric(df[ens_selection], errors="coerce").dropna()
    if len(act_vals) == 0:
        return fig
    activation_cmin = float(act_vals.quantile(0.05))
    activation_cmax = float(act_vals.quantile(0.95))
    if not np.isfinite(activation_cmin) or not np.isfinite(activation_cmax) or activation_cmax <= activation_cmin:
        activation_cmin = float(act_vals.min())
        activation_cmax = float(act_vals.max()) if float(act_vals.max()) > float(act_vals.min()) else float(act_vals.min()) + 1.0
    activation_center = float(act_vals.median())
    activation_span = float(act_vals.max() - act_vals.min())
    if not np.isfinite(activation_span) or activation_span <= 0:
        activation_span = 1.0

    ridge_scale = 6.0 * amplitude_scale
    session_stride_base = max(0.40 * activation_span, 0.36)
    session_stride = max(0.28, session_stride_base * (0.94 + 0.06 * amplitude_scale))
    session_centers = {
        sess_id: (len(session_order) - 1 - i) * session_stride
        for i, sess_id in enumerate(session_order)
    }
    t0_line_half = max(0.18 * activation_span * max(amplitude_scale, 0.5), 0.08 + 0.04 * amplitude_scale)

    # Estimate split threshold for discontinuous chunks.
    _tmp = df.sort_values(["session_id", trial_col, "interval_name", "rel_t_s"]).copy()
    _tmp["dt_s"] = _tmp.groupby(["session_id", trial_col, "interval_name"])["rel_t_s"].diff()
    dt_pos = _tmp["dt_s"][(_tmp["dt_s"] > 0) & _tmp["dt_s"].notna()]
    step_s = float(dt_pos.median()) if len(dt_pos) else 0.04
    gap_threshold_s = max(1.5 * step_s, 0.06)

    group_specs = _resolve_group_specs(group_by, group_by_values)
    interval_groups = {
        interval_name: grp for interval_name, grp in df.groupby("interval_name", sort=False)
    }
    session_interval_groups = {
        key: grp.sort_values("rel_t_s")
        for key, grp in df.groupby(["session_id", "interval_name"], sort=False)
    }
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.002,
        row_heights=[top_panel_height / total_plot_height, bottom_panel_height / total_plot_height],
    )

    # Top panel: average trajectory across all selected sessions.
    top_default_color = "rgba(95,95,95,0.95)"
    for interval in interval_order:
        interval_df = interval_groups.get(interval)
        if interval_df is None or interval_df.empty:
            continue

        if group_by == "None":
            top_groups = [("All", interval_df, top_default_color)]
        else:
            top_groups = []
            for spec in group_specs:
                if spec["column"] is None:
                    grp_df = interval_df
                elif spec["column"] in interval_df.columns:
                    grp_df = interval_df[interval_df[spec["column"]].isin(spec["values"])]
                else:
                    continue
                if grp_df.empty:
                    continue
                top_groups.append((spec["name"], grp_df, spec["color"]))

        for group_name, grp_df, line_color in top_groups:
            session_mean_df = (
                grp_df.groupby(["session_id", "rel_t_s"], as_index=False)
                .agg(session_mean=(ens_selection, "mean"))
                .sort_values("rel_t_s")
            )
            if session_mean_df.empty:
                continue

            mean_df = (
                session_mean_df.groupby("rel_t_s", as_index=False)
                .agg(mean_activation=("session_mean", "mean"), n_support=("session_id", "nunique"))
                .sort_values("rel_t_s")
            )
            if len(mean_df) < 2:
                continue

            mean_rel = mean_df["rel_t_s"].to_numpy(dtype=float)
            mean_x = mean_rel + interval_shift[interval]
            mean_act = mean_df["mean_activation"].to_numpy(dtype=float)

            if group_by == "None":
                customdata = np.column_stack([mean_rel, mean_df["n_support"].to_numpy(dtype=float)])
                hovertemplate = (
                    f"interval={interval}<br>"
                    "t_rel_interval=%{customdata[0]:.3f} s<br>"
                    "mean assembly=%{y:.3f}<br>"
                    "sessions=%{customdata[1]:.0f}<extra></extra>"
                )
            else:
                customdata = np.column_stack(
                    [mean_rel, mean_df["n_support"].to_numpy(dtype=float), np.full(len(mean_df), group_name)]
                )
                hovertemplate = (
                    f"interval={interval}<br>"
                    "group=%{customdata[2]}<br>"
                    "t_rel_interval=%{customdata[0]:.3f} s<br>"
                    "mean assembly=%{y:.3f}<br>"
                    "sessions=%{customdata[1]:.0f}<extra></extra>"
                )

            fig.add_trace(
                go.Scatter(
                    x=mean_x,
                    y=mean_act,
                    mode="lines",
                    line=dict(width=2.2, color=line_color),
                    name=f"{_clean_interval_label(interval)} | {group_name}",
                    showlegend=False,
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                ),
                row=1,
                col=1,
            )

    for interval in interval_order:
        fig.add_vline(
            x=interval_zero_x[interval],
            line_width=1.0,
            line_dash="dash",
            line_color="rgba(90,90,90,0.9)",
            row=1,
            col=1,
        )

    # Interval backgrounds on session ridgeline, matching Trialwise styling.
    cue_interval_color = "rgba(175,175,175,0.10)"
    reward_interval_color = "rgba(175,175,175,0.12)"
    cue_entry_intervals = {"cue_entry_interval", "nextto_cue_interval"}
    cue_exit_intervals = {"cue_exit_interval"}
    reward_intervals = {"R1_entry_interval", "R2_entry_interval"}
    reward_exit_intervals = {"R1_exit_interval", "R2_exit_interval"}

    session_band_half = 0.47 * session_stride
    ridgeline_shapes = []
    for sess_id in session_order:
        base = session_centers[sess_id]

        for interval in interval_order:
            fill_col = None
            if interval in reward_intervals or interval in reward_exit_intervals:
                fill_col = reward_interval_color
            elif interval in cue_entry_intervals or interval in cue_exit_intervals:
                fill_col = cue_interval_color
            if fill_col is None:
                continue

            if interval in reward_exit_intervals or interval in cue_exit_intervals:
                x0 = interval_bounds[interval][0]
                x1 = interval_zero_x[interval]
            else:
                x0 = interval_zero_x[interval]
                x1 = interval_bounds[interval][1]

            ridgeline_shapes.append(
                dict(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    xref="x2",
                    y0=base - session_band_half,
                    y1=base + session_band_half,
                    yref="y2",
                    line=dict(width=0),
                    fillcolor=fill_col,
                    layer="below",
                )
            )

    activation_colorscale = "Plotly3"
    trial_alpha = 0.035 if group_by != "None" else 0.025
    baseline_x = []
    baseline_y = []
    t0_x = []
    t0_y = []
    trial_payloads = {
        spec["name"]: {
            "color": spec["color"],
            "x": [],
            "y": [],
            "customdata": [],
        }
        for spec in group_specs
    }
    mean_line_payloads = {
        spec["name"]: {
            "color": spec["color"],
            "x": [],
            "y": [],
            "customdata": [],
        }
        for spec in group_specs
    }
    mean_marker_x = []
    mean_marker_y = []
    mean_marker_color = []

    for sess_id in session_order:
        for interval in interval_order:
            interval_df = session_interval_groups.get((sess_id, interval))
            if interval_df is None or interval_df.empty:
                continue

            base = session_centers[sess_id]
            x_shift = interval_shift[interval]
            seg_x0, seg_x1 = interval_bounds[interval]
            baseline_x.extend([seg_x0, seg_x1, None])
            baseline_y.extend([base, base, None])

            for spec in group_specs:
                if spec["column"] is None:
                    grp_df = interval_df
                else:
                    grp_df = interval_df[interval_df[spec["column"]].isin(spec["values"])]
                if grp_df.empty:
                    continue

                if show_single_trials:
                    trial_payload = trial_payloads[spec["name"]]
                    for (_, _), ev_df in grp_df.groupby([trial_col, "t0"], sort=False):
                        ev_df = ev_df.sort_values("rel_t_s")
                        seg_id = (ev_df["rel_t_s"].diff().fillna(0) > gap_threshold_s).cumsum()
                        for _, seg in ev_df.groupby(seg_id, sort=False):
                            if len(seg) < 2:
                                continue
                            trial_val = str(seg[trial_col].iloc[0])
                            x_vals = (seg["rel_t_s"].to_numpy(dtype=float) + x_shift).tolist()
                            y_vals = (
                                base + (seg[ens_selection].to_numpy(dtype=float) - activation_center) * ridge_scale
                            ).tolist()
                            trial_payload["x"].extend(x_vals)
                            trial_payload["x"].append(None)
                            trial_payload["y"].extend(y_vals)
                            trial_payload["y"].append(None)
                            trial_payload["customdata"].extend(
                                np.column_stack(
                                    [
                                        np.full(len(seg), sess_id, dtype=object),
                                        np.full(len(seg), interval, dtype=object),
                                        np.full(len(seg), trial_val, dtype=object),
                                        seg["rel_t_s"].to_numpy(dtype=float),
                                        seg[ens_selection].to_numpy(dtype=float),
                                    ]
                                ).tolist()
                            )
                            trial_payload["customdata"].append([sess_id, interval, trial_val, np.nan, np.nan])

                # Mean trajectory for group.
                mean_df = (
                    grp_df.groupby("rel_t_s", as_index=False)
                    .agg(mean_activation=(ens_selection, "mean"), n_support=(ens_selection, "count"))
                    .sort_values("rel_t_s")
                )
                if mean_df.empty:
                    continue
                min_support = 2 if mean_df["n_support"].max() >= 2 else 1
                mean_df = mean_df[mean_df["n_support"] >= min_support]
                if len(mean_df) < 2:
                    continue

                mean_rel = mean_df["rel_t_s"].to_numpy(dtype=float)
                mean_act = mean_df["mean_activation"].to_numpy(dtype=float)
                mean_x = mean_rel + x_shift
                mean_y = base + (mean_act - activation_center) * ridge_scale

                mean_line_payload = mean_line_payloads[spec["name"]]
                mean_line_payload["x"].extend(mean_x.tolist())
                mean_line_payload["x"].append(None)
                mean_line_payload["y"].extend(mean_y.tolist())
                mean_line_payload["y"].append(None)
                mean_line_payload["customdata"].extend(
                    np.column_stack(
                        [
                            np.full(len(mean_df), sess_id, dtype=object),
                            np.full(len(mean_df), interval, dtype=object),
                            mean_rel,
                            mean_act,
                            mean_df["n_support"].to_numpy(dtype=float),
                        ]
                    ).tolist()
                )
                mean_line_payload["customdata"].append([sess_id, interval, np.nan, np.nan, np.nan])
                mean_marker_x.extend(mean_x.tolist())
                mean_marker_y.extend(mean_y.tolist())
                mean_marker_color.extend(mean_act.tolist())

            x0 = interval_zero_x[interval]
            t0_x.extend([x0, x0, None])
            t0_y.extend([base - t0_line_half, base + t0_line_half, None])

    if baseline_x:
        fig.add_trace(
            go.Scatter(
                x=baseline_x,
                y=baseline_y,
                mode="lines",
                line=dict(color="rgba(130,130,130,0.35)", width=0.6, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

    for spec in group_specs:
        trial_payload = trial_payloads[spec["name"]]
        if len(trial_payload["x"]):
            fig.add_trace(
                go.Scattergl(
                    x=trial_payload["x"],
                    y=trial_payload["y"],
                    mode="lines",
                    name=f"{spec['name']} trials",
                    legendgroup=f"{spec['name']}__trials",
                    showlegend=True,
                    line=dict(color=trial_payload["color"], width=0.45),
                    opacity=trial_alpha,
                    customdata=np.asarray(trial_payload["customdata"], dtype=object),
                    hovertemplate=(
                        "session=%{customdata[0]}<br>"
                        "interval=%{customdata[1]}<br>"
                        "trial=%{customdata[2]}<br>"
                        "t_rel_interval=%{customdata[3]:.3f} s<br>"
                        "assembly=%{customdata[4]:.3f}<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

    for spec in group_specs:
        mean_line_payload = mean_line_payloads[spec["name"]]
        if len(mean_line_payload["x"]):
            fig.add_trace(
                go.Scattergl(
                    x=mean_line_payload["x"],
                    y=mean_line_payload["y"],
                    mode="lines",
                    name=f"{spec['name']} mean",
                    legendgroup=f"{spec['name']}__mean",
                    showlegend=True,
                    line=dict(color=mean_line_payload["color"], width=2.5),
                    customdata=np.asarray(mean_line_payload["customdata"], dtype=object),
                    hovertemplate=(
                        "session=%{customdata[0]}<br>"
                        "interval=%{customdata[1]}<br>"
                        "t_rel_interval=%{customdata[2]:.3f} s<br>"
                        "mean assembly=%{customdata[3]:.3f}<br>"
                        "n=%{customdata[4]:.0f}<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

    if mean_marker_x:
        fig.add_trace(
            go.Scattergl(
                x=mean_marker_x,
                y=mean_marker_y,
                mode="markers",
                marker=dict(size=3.0, color=mean_marker_color, coloraxis="coloraxis", line=dict(width=0)),
                opacity=0.62,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

    if t0_x:
        fig.add_trace(
            go.Scatter(
                x=t0_x,
                y=t0_y,
                mode="lines",
                line=dict(color="rgba(90,90,90,0.9)", width=1.0, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

    # Vertical separators between interval segments.
    for i in range(len(interval_order) - 1):
        _, left_end = interval_bounds[interval_order[i]]
        right_start, _ = interval_bounds[interval_order[i + 1]]
        x_sep = 0.5 * (left_end + right_start)
        ridgeline_shapes.append(
            dict(
                type="line",
                x0=x_sep,
                x1=x_sep,
                xref="x2",
                y0=min(session_centers.values()) - 2 * t0_line_half,
                y1=max(session_centers.values()) + 2 * t0_line_half,
                yref="y2",
                line=dict(color="rgba(120,120,120,0.15)", width=1),
                layer="below",
            )
        )
        fig.add_vline(
            x=x_sep,
            line_width=1,
            line_color="rgba(120,120,120,0.12)",
            row=1,
            col=1,
        )

    if ridgeline_shapes:
        fig.update_layout(shapes=list(fig.layout.shapes or []) + ridgeline_shapes)

    # Tight y-range from rendered traces. This keeps the ridgeline plot large and
    # readable while still leaving a small amount of breathing room.
    y_arrays = []
    for tr in fig.data:
        if getattr(tr, "yaxis", None) != "y2":
            continue
        legendgroup = str(getattr(tr, "legendgroup", "") or "")
        if legendgroup.endswith("__trials"):
            continue
        y = getattr(tr, "y", None)
        if y is None:
            continue
        arr = np.asarray(y, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            y_arrays.append(arr)
    if y_arrays:
        y_min = min(float(a.min()) for a in y_arrays)
        y_max = max(float(a.max()) for a in y_arrays)
        y_span = y_max - y_min
        y_pad = 0.02 * y_span if y_span > 0 else 0.05 * activation_span
        y_range = [y_min - y_pad, y_max + y_pad]
    else:
        y_range = None

    fig.update_layout(
        title=dict(
            text=(
                "EnsembleSessionwise trajectories "
                f"(ensemble={ens_selection}, group by={group_by})"
            ),
            y=0.995,
            yanchor="top",
            pad=dict(b=8),
        ),
        template="plotly_white",
        coloraxis=dict(
            colorscale=activation_colorscale,
            cmin=activation_cmin,
            cmax=activation_cmax,
            colorbar=dict(
                title=f"Mean {ens_selection}",
                len=0.66,
                y=0.31,
                yanchor="middle",
                x=1.03,
                xanchor="left",
                thickness=18,
            ),
        ),
        height=total_plot_height,
        width=1280,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.10,
            groupclick="togglegroup",
            bgcolor="rgba(255,255,255,0.85)",
        ),
        margin=dict(l=170, r=340, t=80, b=45),
        hovermode="closest",
    )

    fig.update_xaxes(
        row=1,
        col=1,
        range=x_range,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
    )
    fig.update_yaxes(
        row=1,
        col=1,
        title_text=f"Mean {ens_selection}",
        showgrid=True,
        gridcolor="rgba(120,120,120,0.15)",
        zeroline=False,
    )

    fig.update_xaxes(
        row=2,
        col=1,
        title_text="Interval-local time segments (t0 = 0 in each interval)",
        range=x_range,
        tickmode="array",
        tickvals=[interval_centers[name] for name in interval_order],
        ticktext=[_clean_interval_label(name).replace(" ", "<br>") for name in interval_order],
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        row=2,
        col=1,
        title_text="Sessions",
        tickmode="array",
        tickvals=[session_centers[sid] for sid in session_order],
        ticktext=[sid.split("_")[0] for sid in session_order],
        range=y_range,
        showgrid=False,
        zeroline=False,
    )

    if y_range is not None:
        for interval in interval_order:
            fig.add_annotation(
                x=interval_zero_x[interval],
                y=y_range[0],
                xanchor="center",
                yanchor="top",
                yshift=-2,
                text="0 s",
                showarrow=False,
                font=dict(size=10, color="rgba(80,80,80,0.9)"),
                row=2,
                col=1,
            )

    return fig
