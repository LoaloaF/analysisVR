from datetime import datetime

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


def _value_in_group(value, group_vals):
    if group_vals is None:
        return True
    if value in group_vals:
        return True
    return str(value) in {str(v) for v in group_vals}


def _clean_interval_label(interval_name):
    return str(interval_name).replace("_interval", "").replace("_", " ")


def _sort_session_ids(session_ids):
    def _key(s):
        ss = str(s)
        try:
            dt = datetime.strptime(ss, "%Y-%m-%d_%H-%M")
            return (0, dt, ss)
        except ValueError:
            pass
        return (1, ss, ss)

    return sorted(session_ids, key=_key)


def _build_interval_segments(df, interval_order):
    interval_windows = {}
    widths = []
    for interval in interval_order:
        vals = df.loc[df["interval_name"] == interval, "rel_t_s"].dropna()
        if vals.empty:
            continue
        lo, hi = float(vals.min()), float(vals.max())
        span = hi - lo
        pad = 0.03 * span if span > 0 else 0.12
        lo, hi = lo - pad, hi + pad
        if hi <= lo:
            hi = lo + 0.2
        interval_windows[interval] = (lo, hi)
        widths.append(hi - lo)

    interval_order = [name for name in interval_order if name in interval_windows]
    if len(interval_order) == 0:
        return None

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
    return interval_order, interval_shift, interval_bounds, interval_centers, interval_zero_x, x_range


def render_all_sessions_plot(data, ens_selection, event_selection, selected_session):
    fig = go.Figure()
    if data is None or len(data) == 0 or ens_selection is None:
        return fig

    df = data.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif df.index.name is not None:
        df = df.reset_index()

    if ens_selection not in df.columns:
        return fig
    if "session_id" not in df.columns:
        return fig

    if isinstance(event_selection, str):
        event_selection = [event_selection]
    event_selection = [str(ev) for ev in (event_selection or []) if ev is not None]
    if len(event_selection) == 0:
        return fig

    event_col = _pick_event_column(df, event_selection)
    if event_col is None:
        return fig
    df[event_col] = df[event_col].astype(str)
    df = df[df[event_col].isin(event_selection)].copy()
    if df.empty:
        return fig

    if "interval_name" not in df.columns:
        df["interval_name"] = df[event_col].astype(str)
    else:
        df["interval_name"] = df["interval_name"].fillna(df[event_col]).astype(str)

    df[ens_selection] = pd.to_numeric(df[ens_selection], errors="coerce")
    df = df.dropna(subset=[ens_selection]).copy()
    if df.empty:
        return fig

    present_intervals = set(df["interval_name"].astype(str).unique())
    interval_order = [name for name in PREFERRED_INTERVAL_ORDER if name in present_intervals]
    interval_order += [name for name in event_selection if name in present_intervals and name not in interval_order]
    interval_order += [name for name in sorted(present_intervals) if name not in interval_order]
    if len(interval_order) == 0:
        return fig

    session_order = _sort_session_ids(df["session_id"].astype(str).unique().tolist())
    if len(session_order) == 0:
        return fig

    means = (
        df.groupby(["session_id", "interval_name"], as_index=False)[ens_selection]
        .mean()
        .copy()
    )
    means["session_id"] = means["session_id"].astype(str)

    session_x = {sess: i for i, sess in enumerate(session_order)}
    n_intervals = max(1, len(interval_order))
    offsets = np.linspace(-0.22, 0.22, n_intervals) if n_intervals > 1 else np.array([0.0])
    interval_palette = pc.qualitative.Plotly + pc.qualitative.Set2 + pc.qualitative.Set3

    for i, interval in enumerate(interval_order):
        sub = means[means["interval_name"] == interval].copy()
        if sub.empty:
            continue
        sub["x"] = sub["session_id"].map(session_x).astype(float) + float(offsets[i])
        sub = sub.sort_values("session_id")
        fig.add_trace(
            go.Scatter(
                x=sub["x"],
                y=sub[ens_selection],
                mode="lines+markers",
                line=dict(width=0.9, color=interval_palette[i % len(interval_palette)]),
                marker=dict(size=6, color=interval_palette[i % len(interval_palette)]),
                name=_clean_interval_label(interval),
                customdata=np.column_stack([sub["session_id"].astype(str), sub["interval_name"].astype(str)]),
                hovertemplate=(
                    "session=%{customdata[0]}<br>"
                    "interval=%{customdata[1]}<br>"
                    f"mean {ens_selection}=%{{y:.3f}}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    grand = means.groupby("session_id", as_index=False)[ens_selection].mean()
    grand["x"] = grand["session_id"].map(session_x).astype(float)
    grand = grand.sort_values("x")
    if not grand.empty:
        fig.add_trace(
            go.Scatter(
                x=grand["x"],
                y=grand[ens_selection],
                mode="lines+markers",
                marker=dict(size=5, color="rgba(20,20,20,0.95)"),
                line=dict(width=3.2, color="rgba(20,20,20,0.95)"),
                name="Grand average",
                hovertemplate=f"session=%{{x}}<br>grand avg {ens_selection}=%{{y:.3f}}<extra></extra>",
                showlegend=False,
            )
        )

    if selected_session is not None:
        selected_session = str(selected_session)
        if selected_session in session_x:
            x_sel = session_x[selected_session]
            fig.add_vline(
                x=x_sel,
                line_width=1,
                line_dash="solid",
                line_color="rgba(90,90,90,0.8)",
            )

    fig.update_layout(
        template="plotly_white",
        title=dict(text="Session Interval Means", font=dict(size=12)),
        height=320,
        margin=dict(l=45, r=12, t=32, b=62),
        hovermode="closest",
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(session_order))),
        ticktext=session_order,
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
    selected_session=None,
    group_by="None",
    group_by_values=None,
    amplitude_scale=1.0,
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

    df = data.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif df.index.name is not None:
        df = df.reset_index()

    if ens_selection not in df.columns:
        return fig

    if isinstance(event_selection, str):
        event_selection = [event_selection]
    event_selection = [str(ev) for ev in (event_selection or []) if ev is not None]
    if len(event_selection) == 0:
        return fig

    event_col = _pick_event_column(df, event_selection)
    if event_col is None:
        return fig

    df[event_col] = df[event_col].astype(str)
    df = df[df[event_col].isin(event_selection)].copy()
    if df.empty:
        return fig

    if "interval_name" not in df.columns:
        df["interval_name"] = df[event_col].astype(str)
    else:
        df["interval_name"] = df["interval_name"].fillna(df[event_col]).astype(str)

    trial_col = _resolve_trial_column(df)
    if trial_col is None:
        trial_col = "__trial_fallback"
        if "session_id" in df.columns and "t0" in df.columns:
            df[trial_col] = df.groupby(["session_id", "t0"], dropna=False).ngroup() + 1
        elif "t0" in df.columns:
            df[trial_col] = df.groupby(["t0"], dropna=False).ngroup() + 1
        else:
            df[trial_col] = np.arange(df.shape[0]) + 1

    required_cols = ["from_ephys_timestamp", "t0", ens_selection, trial_col]
    for col in required_cols:
        if col not in df.columns:
            return fig

    for col in ["from_ephys_timestamp", "t0", ens_selection]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required_cols).copy()
    if df.empty:
        return fig

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

    interval_pack = _build_interval_segments(df, interval_order)
    if interval_pack is None:
        return fig
    (
        interval_order,
        interval_shift,
        interval_bounds,
        interval_centers,
        interval_zero_x,
        x_range,
    ) = interval_pack

    trial_numeric = pd.to_numeric(df[trial_col], errors="coerce")
    if trial_numeric.notna().all():
        trial_vals = sorted(trial_numeric.unique().tolist())
        df["__trial_value"] = trial_numeric
    else:
        trial_vals = sorted(df[trial_col].astype(str).unique().tolist(), key=lambda v: str(v))
        df["__trial_value"] = df[trial_col].astype(str)

    if len(trial_vals) == 0:
        return fig

    act_vals = pd.to_numeric(df[ens_selection], errors="coerce").dropna()
    if len(act_vals) == 0:
        return fig

    q05 = float(act_vals.quantile(0.05))
    q95 = float(act_vals.quantile(0.95))
    activation_center = float(act_vals.median())
    activation_span = q95 - q05
    if not np.isfinite(activation_span) or activation_span <= 0:
        activation_span = float(act_vals.max() - act_vals.min())
    if not np.isfinite(activation_span) or activation_span <= 0:
        activation_span = 1.0

    ridge_scale = 3.2 * amplitude_scale
    trial_stride_base = max(0.85 * activation_span * 3.2, 0.70)
    trial_stride = max(0.35, trial_stride_base * (0.55 + 0.45 * amplitude_scale))
    trial_centers = {
        trial_val: (len(trial_vals) - 1 - i) * trial_stride
        for i, trial_val in enumerate(trial_vals)
    }
    t0_line_half = max(0.18 * activation_span * max(ridge_scale, 0.5), 0.08 + 0.06 * amplitude_scale)

    tmp_dt = df.sort_values(["__trial_value", "interval_name", "rel_t_s"]).copy()
    tmp_dt["dt_s"] = tmp_dt.groupby(["__trial_value", "interval_name"])["rel_t_s"].diff()
    dt_pos = tmp_dt["dt_s"][(tmp_dt["dt_s"] > 0) & tmp_dt["dt_s"].notna()]
    step_s = float(dt_pos.median()) if len(dt_pos) else 0.04
    gap_threshold_s = max(1.5 * step_s, 0.06)

    interval_palette = pc.qualitative.Plotly + pc.qualitative.Set2 + pc.qualitative.Set3
    interval_colormap = {
        name: interval_palette[i % len(interval_palette)]
        for i, name in enumerate(interval_order)
    }

    group_specs = _resolve_group_specs(group_by, group_by_values)
    default_spec = group_specs[0]
    trial_group_spec = {}
    for trial_val in trial_vals:
        t_df = df[df["__trial_value"] == trial_val]
        if t_df.empty:
            trial_group_spec[trial_val] = default_spec
            continue
        t_row = t_df.iloc[0]
        matched = default_spec
        for spec in group_specs:
            if spec["column"] is None:
                matched = spec
                break
            if spec["column"] not in t_row.index:
                continue
            if _value_in_group(t_row[spec["column"]], spec["values"]):
                matched = spec
                break
        trial_group_spec[trial_val] = matched

    if selected_session is None and "session_id" in df.columns and df["session_id"].nunique() == 1:
        selected_session = str(df["session_id"].iloc[0])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.002,
        row_heights=[0.05, 0.95],
    )

    # Top panel: average activation per interval over all displayed trials.
    top_default_color = "rgba(95,95,95,0.95)"
    for interval in interval_order:
        interval_df = df[df["interval_name"] == interval]
        if interval_df.empty:
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
            mean_df = (
                grp_df.groupby("rel_t_s", as_index=False)
                .agg(mean_activation=(ens_selection, "mean"), n_support=(ens_selection, "count"))
                .sort_values("rel_t_s")
            )
            if mean_df.empty:
                continue
            mean_df = mean_df[mean_df["n_support"] >= 1]
            if len(mean_df) < 2:
                continue
            mean_rel = mean_df["rel_t_s"].to_numpy(dtype=float)
            mean_x = mean_rel + interval_shift[interval]
            mean_act = mean_df["mean_activation"].to_numpy(dtype=float)

            if group_by == "None":
                customdata = np.column_stack([mean_rel, mean_df["n_support"].to_numpy(dtype=float)])
                hovertemplate = (
                    f"session={selected_session}<br>"
                    f"interval={interval}<br>"
                    "t_rel_interval=%{customdata[0]:.3f} s<br>"
                    "mean assembly=%{y:.3f}<br>"
                    "n=%{customdata[1]:.0f}<extra></extra>"
                )
            else:
                customdata = np.column_stack(
                    [mean_rel, mean_df["n_support"].to_numpy(dtype=float), np.full(len(mean_df), group_name)]
                )
                hovertemplate = (
                    f"session={selected_session}<br>"
                    f"interval={interval}<br>"
                    "group=%{customdata[2]}<br>"
                    "t_rel_interval=%{customdata[0]:.3f} s<br>"
                    "mean assembly=%{y:.3f}<br>"
                    "n=%{customdata[1]:.0f}<extra></extra>"
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

    # Interval backgrounds on trial ridgeline.
    cue_color_map = {
        1: "rgba(233,136,0,0.10)",
        2: "rgba(188,0,233,0.07)",
    }
    reward_interval_color = "rgba(175,175,175,0.12)"
    cue_entry_intervals = {"cue_entry_interval", "nextto_cue_interval"}
    cue_exit_intervals = {"cue_exit_interval"}
    reward_intervals = {"R1_entry_interval", "R2_entry_interval"}
    reward_exit_intervals = {"R1_exit_interval", "R2_exit_interval"}
    trial_meta = df.groupby("__trial_value", as_index=True).agg(cue=("cue", "first")) if "cue" in df.columns else None

    for trial_val in trial_vals:
        base = trial_centers[trial_val]
        cue_val = None
        if trial_meta is not None and trial_val in trial_meta.index:
            cue_val = trial_meta.loc[trial_val, "cue"]

        for interval in interval_order:
            fill_col = None
            if interval in reward_intervals or interval in reward_exit_intervals:
                fill_col = reward_interval_color
            elif interval in cue_entry_intervals or interval in cue_exit_intervals:
                cue_key = pd.to_numeric(pd.Series([cue_val]), errors="coerce").iloc[0]
                cue_key = int(cue_key) if pd.notna(cue_key) else 1
                fill_col = cue_color_map.get(cue_key, cue_color_map[1])
            if fill_col is None:
                continue

            if interval in reward_exit_intervals or interval in cue_exit_intervals:
                x0 = interval_bounds[interval][0]
                x1 = interval_zero_x[interval]
            else:
                x0 = interval_zero_x[interval]
                x1 = interval_bounds[interval][1]
            y0 = base - 0.47 * trial_stride
            y1 = base + 0.47 * trial_stride
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                line=dict(width=0),
                fillcolor=fill_col,
                layer="below",
                row=2,
                col=1,
            )

    # Trial-wise ridgeline traces.
    legend_shown = set()
    for trial_val in trial_vals:
        trial_df = df[df["__trial_value"] == trial_val]
        if trial_df.empty:
            continue

        base = trial_centers[trial_val]
        style = trial_group_spec.get(trial_val, default_spec)

        for interval in interval_order:
            interval_df = trial_df[trial_df["interval_name"] == interval]
            if interval_df.empty:
                continue

            x_shift = interval_shift[interval]
            seg_x0, seg_x1 = interval_bounds[interval]
            fig.add_trace(
                go.Scatter(
                    x=[seg_x0, seg_x1],
                    y=[base, base],
                    mode="lines",
                    line=dict(color="rgba(130,130,130,0.30)", width=0.6, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )

            interval_df = interval_df.sort_values("rel_t_s")
            seg_id = (interval_df["rel_t_s"].diff().fillna(0) > gap_threshold_s).cumsum()
            for _, seg in interval_df.groupby(seg_id, sort=False):
                if len(seg) < 2:
                    continue
                show_legend = style["name"] not in legend_shown
                fig.add_trace(
                    go.Scatter(
                        x=seg["rel_t_s"] + x_shift,
                        y=base + (seg[ens_selection].to_numpy(dtype=float) - activation_center) * ridge_scale,
                        mode="lines",
                        name=style["name"],
                        legendgroup=style["name"],
                        showlegend=show_legend,
                        line=dict(color=style["color"], width=1.2),
                        customdata=np.column_stack(
                            [
                                np.full(len(seg), str(trial_val), dtype=object),
                                np.full(len(seg), interval, dtype=object),
                                seg["rel_t_s"].to_numpy(dtype=float),
                                seg[ens_selection].to_numpy(dtype=float),
                            ]
                        ),
                        hovertemplate=(
                            f"session={selected_session}<br>"
                            "trial=%{customdata[0]}<br>"
                            "interval=%{customdata[1]}<br>"
                            "t_rel_interval=%{customdata[2]:.3f} s<br>"
                            "assembly=%{customdata[3]:.3f}<extra></extra>"
                        ),
                    ),
                    row=2,
                    col=1,
                )
                if show_legend:
                    legend_shown.add(style["name"])

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

    if len(interval_order) > 1:
        y0 = min(trial_centers.values()) - 2 * t0_line_half
        y1 = max(trial_centers.values()) + 2 * t0_line_half
        for i in range(len(interval_order) - 1):
            _, left_end = interval_bounds[interval_order[i]]
            right_start, _ = interval_bounds[interval_order[i + 1]]
            x_sep = 0.5 * (left_end + right_start)
            fig.add_shape(
                type="line",
                x0=x_sep,
                x1=x_sep,
                y0=y0,
                y1=y1,
                line=dict(color="rgba(120,120,120,0.15)", width=1),
                layer="below",
                row=2,
                col=1,
            )
            fig.add_vline(
                x=x_sep,
                line_width=1,
                line_color="rgba(120,120,120,0.12)",
                row=1,
                col=1,
            )

    y_min = min(trial_centers.values()) - 2 * t0_line_half
    y_max = max(trial_centers.values()) + 2 * t0_line_half
    y_range = [y_min, y_max]

    trial_outcome_map = {}
    if "trial_outcome" in df.columns:
        trial_outcome_map = (
            df.groupby("__trial_value", as_index=True)["trial_outcome"]
            .first()
            .to_dict()
        )

    def _fmt_trial_label(trial_val):
        if isinstance(trial_val, (float, np.floating)) and np.isfinite(trial_val):
            if abs(trial_val - round(trial_val)) < 1e-9:
                return str(int(round(trial_val)))
        return str(trial_val)

    trial_ticktext = []
    for t in trial_vals:
        label = _fmt_trial_label(t)
        if t in trial_outcome_map:
            outc = pd.to_numeric(pd.Series([trial_outcome_map[t]]), errors="coerce").iloc[0]
            if pd.notna(outc):
                col = "#2ca02c" if float(outc) > 0 else "#d62728"
                trial_ticktext.append(f"<span style='color:{col}'>{label}</span>")
                continue
        trial_ticktext.append(label)

    fig.update_layout(
        title=dict(
            text=(
                "Trial-wise ensemble ridgeline "
                f"(session={selected_session}, ensemble={ens_selection}, group by={group_by})"
            ),
            y=0.98,
            yanchor="top",
        ),
        template="plotly_white",
        height=max(620, 30 * len(trial_vals) + 360),
        width=1180,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        margin=dict(l=140, r=220, t=85, b=62),
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
        title_text="Trial ID",
        tickmode="array",
        tickvals=[trial_centers[t] for t in trial_vals],
        ticktext=trial_ticktext,
        range=y_range,
        showgrid=False,
        zeroline=False,
    )

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
