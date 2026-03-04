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


def render_plot(data, ens_selection, event_selection, group_by="None", group_by_values=None, amplitude_scale=1.0):
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

    df = data.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif df.index.name is not None:
        df = df.reset_index()

    if ens_selection not in df.columns:
        return fig

    if isinstance(event_selection, str):
        event_selection = [event_selection]
    event_selection = [ev for ev in (event_selection or []) if ev is not None]
    if len(event_selection) == 0:
        return fig

    event_col = _pick_event_column(df, event_selection)
    if event_col is None:
        return fig

    df = df[df[event_col].isin(event_selection)].copy()
    if df.empty:
        return fig

    if "interval_name" not in df.columns:
        df["interval_name"] = df[event_col].astype(str)
    else:
        df["interval_name"] = df["interval_name"].fillna(df[event_col]).astype(str)

    required_cols = ["session_id", "from_ephys_timestamp", "t0", ens_selection]
    for col in required_cols:
        if col not in df.columns:
            return fig
    trial_col_candidates = ["trial_id", "behavior_trial_id", "trial", "trial_index", "entry_id"]
    trial_col = next((c for c in trial_col_candidates if c in df.columns), None)
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

    # Session order by parsed timestamp then lexical fallback.
    _sess_dt = pd.to_datetime(df["session_id"], format="%Y-%m-%d_%H-%M", errors="coerce")
    session_order = (
        pd.DataFrame({"session_id": df["session_id"], "_dt": _sess_dt})
        .drop_duplicates()
        .sort_values(["_dt", "session_id"])
        ["session_id"]
        .tolist()
    )
    if len(session_order) == 0:
        return fig

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
    interval_offsets = {interval: 0.0 for interval in interval_order}
    session_stride_base = max(0.50 * activation_span, 0.45)
    session_stride = max(0.35, session_stride_base * (0.55 + 0.45 * amplitude_scale))
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
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.05,
        row_heights=[0.14, 0.86],
    )

    # Top panel: session-wise interval means after the same filter stack as ridgeline.
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
        interval_df = df[df["interval_name"] == interval]
        if interval_df.empty:
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
                    showlegend=True,
                    line=dict(color=interval_colormap[interval], width=2.2),
                    marker=dict(size=5, color=interval_colormap[interval]),
                    customdata=customdata,
                    hovertemplate=(
                        "session=%{customdata[0]}<br>"
                        "interval=%{customdata[1]}<br>"
                        "mean assembly=%{y:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
        else:
            dash_style = interval_dashes[i_interval % len(interval_dashes)]
            for spec in group_specs:
                if spec["column"] is None:
                    grp_interval_df = interval_df
                else:
                    grp_interval_df = interval_df[interval_df[spec["column"]].isin(spec["values"])]
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
                        showlegend=True,
                        line=dict(color=spec["color"], width=2.0, dash=dash_style),
                        marker=dict(size=4.5, color=spec["color"]),
                        customdata=customdata,
                        hovertemplate=(
                            "session=%{customdata[0]}<br>"
                            "interval=%{customdata[1]}<br>"
                            "group=%{customdata[2]}<br>"
                            "mean assembly=%{y:.3f}<extra></extra>"
                        ),
                    ),
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

            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=base - session_band_half,
                y1=base + session_band_half,
                line=dict(width=0),
                fillcolor=fill_col,
                layer="below",
                row=2,
                col=1,
            )

    trial_legend_shown = set()
    mean_legend_shown = set()
    activation_colorscale = "Plotly3"
    trial_alpha = 0.08 if group_by != "None" else 0.05

    for sess_id in session_order:
        sess_df = df[df["session_id"] == sess_id]
        if sess_df.empty:
            continue

        for interval in interval_order:
            interval_df = sess_df[sess_df["interval_name"] == interval]
            if interval_df.empty:
                continue

            base = session_centers[sess_id] + interval_offsets[interval]
            x_shift = interval_shift[interval]
            seg_x0, seg_x1 = interval_bounds[interval]

            fig.add_trace(
                go.Scatter(
                    x=[seg_x0, seg_x1],
                    y=[base, base],
                    mode="lines",
                    line=dict(color="rgba(130,130,130,0.35)", width=0.6, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )

            for spec in group_specs:
                if spec["column"] is None:
                    grp_df = interval_df
                else:
                    grp_df = interval_df[interval_df[spec["column"]].isin(spec["values"])]
                if grp_df.empty:
                    continue

                # Single-trial traces.
                for (_, _), ev_df in grp_df.groupby([trial_col, "t0"], sort=False):
                    ev_df = ev_df.sort_values("rel_t_s")
                    seg_id = (ev_df["rel_t_s"].diff().fillna(0) > gap_threshold_s).cumsum()
                    for _, seg in ev_df.groupby(seg_id, sort=False):
                        if len(seg) < 2:
                            continue
                        trial_val = str(seg[trial_col].iloc[0])
                        show_trial_legend = spec["name"] not in trial_legend_shown
                        fig.add_trace(
                            go.Scatter(
                                x=seg["rel_t_s"] + x_shift,
                                y=base + (seg[ens_selection].to_numpy(dtype=float) - activation_center) * ridge_scale,
                                mode="lines",
                                name=f"{spec['name']} trials",
                                legendgroup=f"{spec['name']}__trials",
                                showlegend=show_trial_legend,
                                line=dict(color=spec["color"], width=0.8),
                                opacity=trial_alpha,
                                customdata=np.column_stack(
                                    [
                                        seg["rel_t_s"].to_numpy(dtype=float),
                                        seg[ens_selection].to_numpy(dtype=float),
                                        np.full(len(seg), trial_val, dtype=object),
                                    ]
                                ),
                                hovertemplate=(
                                    f"session={sess_id}<br>interval={interval}<br>"
                                    "trial=%{customdata[2]}<br>"
                                    "t_rel_interval=%{customdata[0]:.3f} s<br>"
                                    "assembly=%{customdata[1]:.3f}<extra></extra>"
                                ),
                            ),
                            row=2,
                            col=1,
                        )
                        if show_trial_legend:
                            trial_legend_shown.add(spec["name"])

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

                show_mean_legend = spec["name"] not in mean_legend_shown
                fig.add_trace(
                    go.Scatter(
                        x=mean_x,
                        y=mean_y,
                        mode="lines",
                        name=f"{spec['name']} mean",
                        legendgroup=f"{spec['name']}__mean",
                        showlegend=show_mean_legend,
                        line=dict(color=spec["color"], width=2.5),
                        customdata=np.column_stack([mean_rel, mean_act, mean_df["n_support"].to_numpy(dtype=float)]),
                        hovertemplate=(
                            f"session={sess_id}<br>interval={interval}<br>"
                            "t_rel_interval=%{customdata[0]:.3f} s<br>"
                            "mean assembly=%{customdata[1]:.3f}<br>"
                            "n=%{customdata[2]:.0f}<extra></extra>"
                        ),
                    ),
                    row=2,
                    col=1,
                )
                if show_mean_legend:
                    mean_legend_shown.add(spec["name"])

                fig.add_trace(
                    go.Scatter(
                        x=mean_x,
                        y=mean_y,
                        mode="markers",
                        marker=dict(size=4.5, color=mean_act, coloraxis="coloraxis", line=dict(width=0)),
                        showlegend=False,
                        legendgroup=f"{spec['name']}__mean",
                        hoverinfo="skip",
                    ),
                    row=2,
                    col=1,
                )

            # Reset t0 line for this interval segment.
            x0 = interval_zero_x[interval]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x0],
                    y=[base - t0_line_half, base + t0_line_half],
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
        fig.add_shape(
            type="line",
            x0=x_sep,
            x1=x_sep,
            y0=min(session_centers.values()) - 2 * t0_line_half,
            y1=max(session_centers.values()) + 2 * t0_line_half,
            line=dict(color="rgba(120,120,120,0.15)", width=1),
            layer="below",
            row=2,
            col=1,
        )

    # Tight y-range from rendered traces.
    y_arrays = []
    for tr in fig.data:
        if getattr(tr, "yaxis", None) != "y2":
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
        y_pad = 0.001 * y_span if y_span > 0 else 0.05 * activation_span
        y_range = [y_min - y_pad, y_max + y_pad]
    else:
        y_range = None

    ridge_height = max(42 * len(session_order) + 520, 900)
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
        height=ridge_height + 120,
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
        title_text="Sessions",
        tickmode="array",
        tickvals=session_positions.tolist(),
        ticktext=session_date_labels,
        tickangle=45,
        showgrid=False,
        zeroline=False,
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
