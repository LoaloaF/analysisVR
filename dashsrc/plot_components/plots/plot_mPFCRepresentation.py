import ast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import f1_score


_ONE_R_OUTCOMES = {1, 11, 21, 31, 41, 51, 10, 20, 30, 40, 50}


def _to_1d_array(value):
    if value is None:
        return np.array([])
    if isinstance(value, np.ndarray):
        return value.reshape(-1)
    if isinstance(value, (list, tuple, pd.Series)):
        return np.asarray(value).reshape(-1)
    if isinstance(value, str):
        txt = value.strip()
        if txt.startswith("[") and txt.endswith("]"):
            try:
                return np.asarray(ast.literal_eval(txt)).reshape(-1)
            except (SyntaxError, ValueError):
                return np.array([])
        return np.array([])
    if pd.isna(value):
        return np.array([])
    return np.asarray([value]).reshape(-1)


def _coerce_binary(arr):
    vals = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy()
    if vals.size == 0:
        return vals
    return vals.astype(float)


def _extract_trial_table(behavior_trialwise):
    if behavior_trialwise is None or len(behavior_trialwise) == 0:
        return pd.DataFrame(columns=["cue", "trial_outcome", "choice_R1", "choice_R2"])

    beh = behavior_trialwise.reset_index() if isinstance(behavior_trialwise.index, pd.MultiIndex) else behavior_trialwise.copy()
    required_cols = ["trial_id", "cue", "trial_outcome", "choice_R1", "choice_R2"]
    for col in required_cols:
        if col not in beh.columns:
            beh[col] = np.nan

    beh = beh[required_cols].copy()
    beh = beh.dropna(subset=["trial_id"])\
             .drop_duplicates(subset=["trial_id"], keep="first")\
             .sort_values("trial_id")

    beh["trial_id"] = pd.to_numeric(beh["trial_id"], errors="coerce")
    beh = beh.dropna(subset=["trial_id"])
    beh["trial_id"] = beh["trial_id"].astype(int)

    # Keep choices as 0/1 floats to tolerate occasional NA rows.
    beh["choice_R1"] = pd.to_numeric(beh["choice_R1"], errors="coerce")
    beh["choice_R2"] = pd.to_numeric(beh["choice_R2"], errors="coerce")
    beh["cue"] = pd.to_numeric(beh["cue"], errors="coerce")
    beh["trial_outcome"] = pd.to_numeric(beh["trial_outcome"], errors="coerce")

    return beh.set_index("trial_id")


def _get_interval_trial_ids(t0_events, interval_name):
    if t0_events is None or len(t0_events) == 0:
        return np.array([], dtype=int)

    if interval_name not in t0_events.columns:
        return np.array([], dtype=int)

    src = t0_events.reset_index() if isinstance(t0_events.index, pd.MultiIndex) else t0_events.copy()
    if "trial_id" not in src.columns:
        return np.array([], dtype=int)

    sub = src.loc[src[interval_name].notna(), ["trial_id"]].copy()
    if sub.empty:
        return np.array([], dtype=int)

    vals = pd.to_numeric(sub["trial_id"], errors="coerce").dropna().astype(int)
    return vals.to_numpy()


def _apply_group_filters(trial_meta, trial_ids, filters):
    if len(trial_ids) == 0:
        return np.zeros(0, dtype=bool)

    rows = trial_meta.reindex(trial_ids)
    valid_mask = rows.index.to_series().notna().to_numpy()

    mask = np.ones(len(rows), dtype=bool)

    outcome_filter = filters.get("outcome", [])
    if len(outcome_filter) == 0:
        return np.zeros(len(rows), dtype=bool)
    allowed_outcomes = set()
    if "1 R" in outcome_filter:
        allowed_outcomes.update(_ONE_R_OUTCOMES)
    if "1+ R" in outcome_filter:
        allowed_outcomes.update([i for i in range(1, 56) if i not in _ONE_R_OUTCOMES])
    if "no R" in outcome_filter:
        allowed_outcomes.add(0)
    mask &= rows["trial_outcome"].isin(allowed_outcomes).to_numpy()

    cue_filter = filters.get("cue", [])
    if len(cue_filter) == 0:
        return np.zeros(len(rows), dtype=bool)
    allowed_cues = []
    if "Cue1 trials" in cue_filter:
        allowed_cues.append(1)
    if "Cue2 trials" in cue_filter:
        allowed_cues.append(2)
    mask &= rows["cue"].isin(allowed_cues).to_numpy()

    r1_filter = filters.get("r1", [])
    if len(r1_filter) == 0:
        return np.zeros(len(rows), dtype=bool)
    allowed_r1 = []
    if "stop" in r1_filter:
        allowed_r1.append(1)
    if "skip" in r1_filter:
        allowed_r1.append(0)
    mask &= rows["choice_R1"].isin(allowed_r1).to_numpy()

    r2_filter = filters.get("r2", [])
    if len(r2_filter) == 0:
        return np.zeros(len(rows), dtype=bool)
    allowed_r2 = []
    if "stop" in r2_filter:
        allowed_r2.append(1)
    if "skip" in r2_filter:
        allowed_r2.append(0)
    mask &= rows["choice_R2"].isin(allowed_r2).to_numpy()

    trial_filter = filters.get("trial", [])
    if len(trial_filter) == 0:
        return np.zeros(len(rows), dtype=bool)
    third_masks = np.zeros(len(rows), dtype=bool)
    thirds = np.array_split(np.arange(len(rows)), 3)
    if "1/3" in trial_filter and len(thirds) > 0:
        third_masks[thirds[0]] = True
    if "2/3" in trial_filter and len(thirds) > 1:
        third_masks[thirds[1]] = True
    if "3/3" in trial_filter and len(thirds) > 2:
        third_masks[thirds[2]] = True
    mask &= third_masks

    mask &= valid_mask
    mask &= rows["cue"].notna().to_numpy()
    mask &= rows["trial_outcome"].notna().to_numpy()
    mask &= rows["choice_R1"].notna().to_numpy()
    mask &= rows["choice_R2"].notna().to_numpy()
    return mask


def _compute_f1_for_group(interval_df, subgroup_mask):
    if interval_df.empty:
        return pd.DataFrame(columns=["timebin", "f1", "n_used"])

    out = []
    for _, row in interval_df.sort_values("timebin").iterrows():
        y_true = _coerce_binary(_to_1d_array(row.get("y_true")))
        pred = _coerce_binary(_to_1d_array(row.get("predictions")))
        n = min(len(y_true), len(pred), len(subgroup_mask))
        if n == 0:
            out.append({"timebin": row.get("timebin", np.nan), "f1": np.nan, "n_used": 0})
            continue

        yt = y_true[:n]
        yp = pred[:n]
        sm = subgroup_mask[:n]

        finite_mask = np.isfinite(yt) & np.isfinite(yp) & sm
        if finite_mask.sum() < 2:
            out.append({"timebin": row.get("timebin", np.nan), "f1": np.nan, "n_used": int(finite_mask.sum())})
            continue

        yt = yt[finite_mask].astype(int)
        yp = yp[finite_mask].astype(int)
        # Macro F1 is not meaningful with a single class in truth labels.
        if np.unique(yt).size < 2:
            out.append({"timebin": row.get("timebin", np.nan), "f1": np.nan, "n_used": int(finite_mask.sum())})
            continue

        score = f1_score(yt, yp, average="macro", zero_division=0)
        out.append({"timebin": row.get("timebin", np.nan), "f1": float(score), "n_used": int(finite_mask.sum())})

    return pd.DataFrame(out)


def _ordered_intervals(df, selected_intervals):
    if selected_intervals:
        return [i for i in selected_intervals if i in set(df["interval_name"].astype(str).unique())]

    return list(pd.unique(df["interval_name"].astype(str)))


def _interval_rows_with_trial_ids(t0_events, interval_name):
    if t0_events is None or len(t0_events) == 0:
        return pd.DataFrame(columns=["trial_id", "interval"])

    if interval_name not in t0_events.columns:
        return pd.DataFrame(columns=["trial_id", "interval"])

    src = t0_events.reset_index() if isinstance(t0_events.index, pd.MultiIndex) else t0_events.copy()
    if "trial_id" not in src.columns:
        return pd.DataFrame(columns=["trial_id", "interval"])

    sub = src.loc[src[interval_name].notna(), ["trial_id", interval_name]].copy()
    if sub.empty:
        return pd.DataFrame(columns=["trial_id", "interval"])

    sub["trial_id"] = pd.to_numeric(sub["trial_id"], errors="coerce")
    sub = sub.dropna(subset=["trial_id"])
    sub["trial_id"] = sub["trial_id"].astype(int)
    sub = sub.rename(columns={interval_name: "interval"})
    return sub


def _extract_interval_position_trajectories(behavior_aligned, t0_events, interval_name, max_trials=120):
    if behavior_aligned is None or len(behavior_aligned) == 0:
        return None

    beh = behavior_aligned.reset_index() if isinstance(behavior_aligned.index, pd.MultiIndex) else behavior_aligned.copy()
    required = {"trial_id", "frame_position"}
    if not required.issubset(set(beh.columns)):
        return None

    ts_col = None
    for candidate in ("to_ephys_timestamp", "from_ephys_timestamp"):
        if candidate in beh.columns:
            ts_col = candidate
            break
    if ts_col is None:
        return None

    beh["trial_id"] = pd.to_numeric(beh["trial_id"], errors="coerce")
    beh = beh.dropna(subset=["trial_id"])
    if beh.empty:
        return None
    beh["trial_id"] = beh["trial_id"].astype(int)

    beh[ts_col] = pd.to_numeric(beh[ts_col], errors="coerce")
    beh["frame_position"] = pd.to_numeric(beh["frame_position"], errors="coerce")
    beh = beh.dropna(subset=[ts_col, "frame_position"])
    if beh.empty:
        return None

    by_trial = {tid: g.sort_values(ts_col) for tid, g in beh.groupby("trial_id")}
    intervals = _interval_rows_with_trial_ids(t0_events, interval_name)
    if intervals.empty:
        return None

    trajectories = []
    for _, row in intervals.iterrows():
        tid = int(row["trial_id"])
        intrvl = row["interval"]
        if not isinstance(intrvl, pd.Interval):
            continue
        trial_df = by_trial.get(tid)
        if trial_df is None or trial_df.empty:
            continue

        left = intrvl.left
        right = intrvl.right
        if pd.isna(left) or pd.isna(right):
            continue

        mask = (trial_df[ts_col] >= left) & (trial_df[ts_col] <= right)
        vals = trial_df.loc[mask, "frame_position"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size >= 3:
            trajectories.append(vals)
        if len(trajectories) >= max_trials:
            break

    if len(trajectories) == 0:
        return None

    lengths = pd.Series([len(v) for v in trajectories])
    common_len = int(lengths.mode().iloc[0])
    keep = [v for v in trajectories if len(v) == common_len]
    if len(keep) == 0:
        return None

    return np.vstack(keep)


def _compute_angle_matrix(plane_df):
    rows = []
    for _, row in plane_df.iterrows():
        w = _to_1d_array(row.get("w"))
        if w.size == 0:
            continue
        w = pd.to_numeric(pd.Series(w), errors="coerce").to_numpy(dtype=float)
        if np.isfinite(w).sum() == 0:
            continue
        rows.append(w)

    if len(rows) < 2:
        return None

    min_dim = min(len(r) for r in rows)
    if min_dim == 0:
        return None

    W = np.vstack([r[:min_dim] for r in rows])
    norms = np.linalg.norm(W, axis=1)
    valid = norms > 1e-12
    if valid.sum() < 2:
        return None

    W = W[valid]
    norms = norms[valid]
    sim = (W @ W.T) / np.outer(norms, norms)
    sim = np.clip(sim, -1.0, 1.0)
    return np.degrees(np.arccos(sim))


def _build_axis_ticks(interval_tick_data, max_ticks_per_interval=8):
    """Build tickvals/ticktext so each interval's start is labelled t=0."""
    tick_vals, tick_texts = [], []
    for interval_start, x_vals_shifted in interval_tick_data:
        n = len(x_vals_shifted)
        if n == 0:
            continue
        stride = max(1, n // max_ticks_per_interval)
        indices = list(range(0, n, stride))
        for i in indices:
            tick_vals.append(float(x_vals_shifted[i] + interval_start))
            tick_texts.append(str(int(round(float(x_vals_shifted[i])))))
    return tick_vals, tick_texts


def render_trial_split_plot(
    svm_data,
    behavior_trialwise,
    t0_events,
    behavior_aligned,
    x_modality,
    x_modality_compare,
    predict_y_name,
    selected_intervals,
    model,
    compare_mode,
    group1_filters,
    group2_filters,
    group1_label="Group 1",
    group2_label="Group 2",
):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.42, 0.24, 0.34],
        subplot_titles=(
            "Position vs Timebin (marker color = balanced macro-F1)",
            f"F1 over timepoints ({group1_label} vs {group2_label})",
            "Angle between fitted SVM planes (degrees)",
        ),
    )

    if svm_data is None or len(svm_data) == 0:
        fig.add_annotation(text="No SVM data available.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

    src = svm_data.reset_index() if isinstance(svm_data.index, pd.MultiIndex) else svm_data.copy()
    required = {"X_modality", "predict_y_name", "interval_name", "timebin"}
    if not required.issubset(set(src.columns)):
        fig.add_annotation(text="Missing required SVM columns for split plot.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

    base_df = src.copy()
    base_df = base_df[base_df["predict_y_name"].astype(str) == str(predict_y_name)]
    if "model" in base_df.columns and model is not None:
        base_df = base_df[base_df["model"].astype(str) == str(model)]

    df = base_df[base_df["X_modality"].astype(str) == str(x_modality)].copy()
    if df.empty:
        fig.add_annotation(text="No rows for selected modality/target/model.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

    df_compare = pd.DataFrame()
    if compare_mode == "brain_area" and x_modality_compare is not None:
        df_compare = base_df[base_df["X_modality"].astype(str) == str(x_modality_compare)].copy()
        if df_compare.empty:
            fig.add_annotation(
                text="No rows for secondary brain area under the current target/model.",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
            )
            return fig

    intervals = _ordered_intervals(df, selected_intervals)
    if compare_mode == "brain_area" and not df_compare.empty:
        compare_intervals = set(df_compare["interval_name"].astype(str).unique())
        intervals = [i for i in intervals if i in compare_intervals]
    if len(intervals) == 0:
        fig.add_annotation(text="No intervals selected.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

    intervals = intervals[:3]

    trial_meta = _extract_trial_table(behavior_trialwise)

    offset = 0.0
    shown_legend = {"g1": False, "g2": False, "all": False, "pos": False}
    top_has_data = False
    plane_rows = []
    plane_x_coords = []
    interval_x_ranges = []
    interval_tick_data = []

    for interval_name in intervals:
        interval_df = df[df["interval_name"].astype(str) == str(interval_name)].sort_values("timebin")
        if interval_df.empty:
            continue

        trial_ids = _get_interval_trial_ids(t0_events, interval_name)
        if len(trial_ids) == 0:
            # Fallback when interval trial mapping is unavailable.
            trial_ids = trial_meta.index.to_numpy(dtype=int) if len(trial_meta) else np.array([], dtype=int)

        if compare_mode == "brain_area" and not df_compare.empty:
            interval_df_compare = df_compare[df_compare["interval_name"].astype(str) == str(interval_name)].sort_values("timebin")
            compare_trial_ids = _get_interval_trial_ids(t0_events, interval_name)
            if len(compare_trial_ids) == 0:
                compare_trial_ids = trial_meta.index.to_numpy(dtype=int) if len(trial_meta) else np.array([], dtype=int)

            all_mask_primary = np.ones(len(trial_ids), dtype=bool)
            all_mask_secondary = np.ones(len(compare_trial_ids), dtype=bool)
            g1 = _compute_f1_for_group(interval_df, all_mask_primary)
            g2 = _compute_f1_for_group(interval_df_compare, all_mask_secondary)
        else:
            mask1 = _apply_group_filters(trial_meta, trial_ids, group1_filters)
            mask2 = _apply_group_filters(trial_meta, trial_ids, group2_filters)
            g1 = _compute_f1_for_group(interval_df, mask1)
            g2 = _compute_f1_for_group(interval_df, mask2)

        if "f1_mean" in interval_df.columns:
            all_trace = interval_df[["timebin", "f1_mean"]].copy().sort_values("timebin")
        else:
            all_mask = np.ones(len(trial_ids), dtype=bool)
            all_trace = _compute_f1_for_group(interval_df, all_mask).rename(columns={"f1": "f1_mean"})

        plane_rows.append(interval_df.copy())

        x_vals = pd.to_numeric(interval_df["timebin"], errors="coerce").to_numpy()
        x_vals = x_vals[np.isfinite(x_vals)]
        if len(x_vals) == 0:
            continue

        interval_x0 = float(np.nanmin(x_vals))
        x_vals_shifted = x_vals - interval_x0

        interval_start = float(offset)
        interval_end = float(np.nanmax(x_vals_shifted) + offset)
        interval_x_ranges.append((interval_start, interval_end))
        plane_x_coords.extend((x_vals_shifted + offset).tolist())
        interval_tick_data.append((interval_start, x_vals_shifted))

        pos_traj = _extract_interval_position_trajectories(behavior_aligned, t0_events, interval_name)
        if pos_traj is not None:
            n_time = min(pos_traj.shape[1], len(all_trace), len(x_vals))
            if n_time >= 2:
                x_plot = x_vals_shifted[:n_time] + offset
                y_traj = pos_traj[:, :n_time]
                f1_color = pd.to_numeric(all_trace["f1_mean"], errors="coerce").to_numpy()[:n_time]

                for tr in y_traj[:40]:
                    fig.add_trace(
                        go.Scatter(
                            x=x_plot,
                            y=tr,
                            mode="lines",
                            line=dict(color="rgba(70,70,70,0.12)", width=1),
                            hoverinfo="skip",
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )

                fig.add_trace(
                    go.Scatter(
                        x=x_plot,
                        y=np.nanmean(y_traj, axis=0),
                        mode="markers+lines",
                        marker=dict(
                            size=8,
                            color=f1_color,
                            cmin=0.2,
                            cmax=0.9,
                            colorscale="Viridis",
                            colorbar=dict(title="Balanced F1"),
                        ),
                        line=dict(color="rgba(0,0,0,0.35)", width=1),
                        name="Mean position",
                        legendgroup="pos",
                        showlegend=not shown_legend["pos"],
                    ),
                    row=1,
                    col=1,
                )
                shown_legend["pos"] = True
                top_has_data = True

        if compare_mode != "brain_area":
            all_x = pd.to_numeric(all_trace["timebin"], errors="coerce") - interval_x0 + offset
            fig.add_trace(
                go.Scatter(
                    x=all_x,
                    y=all_trace["f1_mean"],
                    mode="lines",
                    line=dict(color="rgba(120,120,120,0.8)", dash="dot"),
                    name="All trials",
                    legendgroup="all",
                    showlegend=not shown_legend["all"],
                ),
                row=2,
                col=1,
            )
            shown_legend["all"] = True

        g1_x = pd.to_numeric(g1["timebin"], errors="coerce") - interval_x0 + offset
        fig.add_trace(
            go.Scatter(
                x=g1_x,
                y=g1["f1"],
                mode="lines+markers",
                marker=dict(size=5),
                line=dict(color="#2a9d8f", width=2),
                name=group1_label,
                legendgroup="g1",
                showlegend=not shown_legend["g1"],
            ),
            row=2,
            col=1,
        )
        shown_legend["g1"] = True

        g2_x = pd.to_numeric(g2["timebin"], errors="coerce") - interval_x0 + offset
        fig.add_trace(
            go.Scatter(
                x=g2_x,
                y=g2["f1"],
                mode="lines+markers",
                marker=dict(size=5),
                line=dict(color="#e76f51", width=2),
                name=group2_label,
                legendgroup="g2",
                showlegend=not shown_legend["g2"],
            ),
            row=2,
            col=1,
        )
        shown_legend["g2"] = True

        fig.add_vrect(
            x0=interval_start,
            x1=interval_end,
            fillcolor="rgba(0,0,0,0.03)",
            line_width=0,
            row=1,
            col=1,
        )
        fig.add_vrect(
            x0=interval_start,
            x1=interval_end,
            fillcolor="rgba(0,0,0,0.03)",
            line_width=0,
            row=2,
            col=1,
        )

        fig.add_annotation(
            text=str(interval_name).replace("_interval", "").replace("_", " "),
            x=(interval_start + interval_end) / 2,
            y=0.98,
            xref="x",
            yref="paper",
            showarrow=False,
            font=dict(size=10, color="gray"),
        )

        offset += float(np.nanmax(x_vals_shifted) + 1)

    if not top_has_data:
        fig.add_annotation(
            text="No Behavior40msAligned trajectory data available for selected intervals.",
            x=0.5,
            y=0.82,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=11, color="gray"),
        )

    if len(plane_rows) > 0:
        plane_df = pd.concat(plane_rows, axis=0, ignore_index=True)
        angle_matrix = _compute_angle_matrix(plane_df)
        if angle_matrix is not None:
            n = angle_matrix.shape[0]
            if len(plane_x_coords) >= n:
                heatmap_axis = np.asarray(plane_x_coords[:n], dtype=float)
            else:
                # Fallback keeps behavior stable when coordinate extraction is incomplete.
                heatmap_axis = np.arange(n, dtype=float)

            fig.add_trace(
                go.Heatmap(
                    z=angle_matrix,
                    x=heatmap_axis,
                    y=heatmap_axis,
                    colorscale="RdBu_r",
                    zmin=0,
                    zmax=180,
                    colorbar=dict(title="Angle (deg)"),
                ),
                row=3,
                col=1,
            )

            for i in range(len(interval_x_ranges) - 1):
                boundary = (interval_x_ranges[i][1] + interval_x_ranges[i + 1][0]) / 2
                fig.add_hline(y=boundary, line_color="rgba(0,0,0,0.25)", line_dash="dash", row=3, col=1)

            fig.update_xaxes(title_text="Fitted planes (interval/timebin index)", row=3, col=1)
            fig.update_yaxes(title_text="Fitted planes", row=3, col=1)
        else:
            fig.add_annotation(
                text="Not enough valid SVM weight vectors to compute plane angles.",
                x=0.5,
                y=0.16,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=11, color="gray"),
            )

    fig.add_hline(y=0.5, line_dash="dash", line_color="black", line_width=1, row=2, col=1)
    fig.update_yaxes(title_text="Position (cm)", row=1, col=1)
    fig.update_yaxes(range=[0.0, 1.0], title_text="Macro F1", row=2, col=1)

    fig.update_xaxes(title_text="Timebin (concatenated across intervals)", showticklabels=True, row=1, col=1)
    fig.update_xaxes(title_text="Timebin (concatenated across intervals)", showticklabels=True, row=2, col=1)
    fig.update_xaxes(title_text="Timebin (concatenated across intervals)", showticklabels=True, row=3, col=1)
    fig.update_xaxes(matches="x", row=2, col=1)
    fig.update_xaxes(matches="x", row=3, col=1)
    if len(interval_x_ranges) > 0:
        # Keep shared x-axis flush with real data bounds (start exactly at x=0).
        x_min = 0.0
        x_max = float(interval_x_ranges[-1][1])
        for r in (1, 2, 3):
            fig.update_xaxes(
                autorange=False,
                range=[x_min, x_max],
                constrain="domain",
                row=r,
                col=1,
            )
        # Dashed line at the start of each non-first interval (marks t=0 for that interval)
        for i in range(1, len(interval_x_ranges)):
            iv_start = interval_x_ranges[i][0]
            fig.add_vline(x=iv_start, line_color="rgba(0,0,0,0.45)", line_dash="dash", row=1, col=1)
            fig.add_vline(x=iv_start, line_color="rgba(0,0,0,0.45)", line_dash="dash", row=2, col=1)
            fig.add_vline(x=iv_start, line_color="rgba(0,0,0,0.45)", line_dash="dash", row=3, col=1)
    if interval_tick_data:
        tv, tt = _build_axis_ticks(interval_tick_data)
        for r in (1, 2, 3):
            fig.update_xaxes(tickvals=tv, ticktext=tt, row=r, col=1)
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=30),
        height=1150,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
    )
    return fig


def render_two_group_columns(
    svm_data,
    behavior_trialwise,
    t0_events,
    behavior_aligned,
    predict_y_name,
    model,
    group1_modality,
    group2_modality,
    group1_intervals,
    group2_intervals,
    group1_filters,
    group2_filters,
    group1_label="Group 1",
    group2_label="Group 2",
):
    fig = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=False,
        vertical_spacing=0.08,
        horizontal_spacing=0.07,
        row_heights=[0.42, 0.24, 0.34],
        subplot_titles=(
            f"{group1_label}: Position vs Timebin",
            f"{group2_label}: Position vs Timebin",
            f"{group1_label}: Macro F1 over time",
            f"{group2_label}: Macro F1 over time",
            f"{group1_label}: Angle between fitted SVM planes (deg)",
            f"{group2_label}: Angle between fitted SVM planes (deg)",
        ),
    )

    if svm_data is None or len(svm_data) == 0:
        fig.add_annotation(text="No SVM data available.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

    src = svm_data.reset_index() if isinstance(svm_data.index, pd.MultiIndex) else svm_data.copy()
    required = {"X_modality", "predict_y_name", "interval_name", "timebin"}
    if not required.issubset(set(src.columns)):
        fig.add_annotation(text="Missing required SVM columns for split plot.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

    base_df = src.copy()
    base_df = base_df[base_df["predict_y_name"].astype(str) == str(predict_y_name)]
    if "model" in base_df.columns and model is not None:
        base_df = base_df[base_df["model"].astype(str) == str(model)]

    trial_meta = _extract_trial_table(behavior_trialwise)

    def _render_group_column(col, x_modality, selected_intervals, filters, label, color, show_pos_colorbar=False):
        df = base_df[base_df["X_modality"].astype(str) == str(x_modality)].copy()
        if df.empty:
            fig.add_annotation(
                text=f"No rows for {label} ({x_modality}).",
                x=0.23 if col == 1 else 0.77,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=11, color="gray"),
            )
            return

        intervals = _ordered_intervals(df, selected_intervals)
        if len(intervals) == 0:
            fig.add_annotation(
                text=f"No intervals selected for {label}.",
                x=0.23 if col == 1 else 0.77,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=11, color="gray"),
            )
            return
        intervals = intervals[:3]

        offset = 0.0
        plane_rows = []
        plane_x_coords = []
        interval_x_ranges = []
        interval_tick_data = []
        top_has_data = False
        shown_legend = {"group": False, "all": False, "pos": False}

        for interval_name in intervals:
            interval_df = df[df["interval_name"].astype(str) == str(interval_name)].sort_values("timebin")
            if interval_df.empty:
                continue

            trial_ids = _get_interval_trial_ids(t0_events, interval_name)
            if len(trial_ids) == 0:
                trial_ids = trial_meta.index.to_numpy(dtype=int) if len(trial_meta) else np.array([], dtype=int)

            mask = _apply_group_filters(trial_meta, trial_ids, filters)
            group_trace = _compute_f1_for_group(interval_df, mask)

            if "f1_mean" in interval_df.columns:
                all_trace = interval_df[["timebin", "f1_mean"]].copy().sort_values("timebin")
            else:
                all_mask = np.ones(len(trial_ids), dtype=bool)
                all_trace = _compute_f1_for_group(interval_df, all_mask).rename(columns={"f1": "f1_mean"})

            plane_rows.append(interval_df.copy())

            x_vals = pd.to_numeric(interval_df["timebin"], errors="coerce").to_numpy()
            x_vals = x_vals[np.isfinite(x_vals)]
            if len(x_vals) == 0:
                continue

            interval_x0 = float(np.nanmin(x_vals))
            x_vals_shifted = x_vals - interval_x0

            interval_start = float(offset)
            interval_end = float(np.nanmax(x_vals_shifted) + offset)
            interval_x_ranges.append((interval_start, interval_end))
            plane_x_coords.extend((x_vals_shifted + offset).tolist())
            interval_tick_data.append((interval_start, x_vals_shifted))

            pos_traj = _extract_interval_position_trajectories(behavior_aligned, t0_events, interval_name)
            if pos_traj is not None:
                n_time = min(pos_traj.shape[1], len(all_trace), len(x_vals))
                if n_time >= 2:
                    x_plot = x_vals_shifted[:n_time] + offset
                    y_traj = pos_traj[:, :n_time]
                    f1_color = pd.to_numeric(all_trace["f1_mean"], errors="coerce").to_numpy()[:n_time]

                    for tr in y_traj[:40]:
                        fig.add_trace(
                            go.Scatter(
                                x=x_plot,
                                y=tr,
                                mode="lines",
                                line=dict(color="rgba(70,70,70,0.12)", width=1),
                                hoverinfo="skip",
                                showlegend=False,
                            ),
                            row=1,
                            col=col,
                        )

                    fig.add_trace(
                        go.Scatter(
                            x=x_plot,
                            y=np.nanmean(y_traj, axis=0),
                            mode="markers+lines",
                            marker=dict(
                                size=8,
                                color=f1_color,
                                cmin=0.2,
                                cmax=0.9,
                                colorscale="Viridis",
                                showscale=show_pos_colorbar,
                                colorbar=dict(title="Balanced F1"),
                            ),
                            line=dict(color="rgba(0,0,0,0.35)", width=1),
                            name=f"Mean position ({label})",
                            legendgroup=f"pos-{col}",
                            showlegend=not shown_legend["pos"],
                        ),
                        row=1,
                        col=col,
                    )
                    shown_legend["pos"] = True
                    top_has_data = True

            all_x = pd.to_numeric(all_trace["timebin"], errors="coerce") - interval_x0 + offset
            fig.add_trace(
                go.Scatter(
                    x=all_x,
                    y=all_trace["f1_mean"],
                    mode="lines",
                    line=dict(color="rgba(120,120,120,0.8)", dash="dot"),
                    name="All trials",
                    legendgroup=f"all-{col}",
                    showlegend=not shown_legend["all"],
                ),
                row=2,
                col=col,
            )
            shown_legend["all"] = True

            gx = pd.to_numeric(group_trace["timebin"], errors="coerce") - interval_x0 + offset
            fig.add_trace(
                go.Scatter(
                    x=gx,
                    y=group_trace["f1"],
                    mode="lines+markers",
                    marker=dict(size=5),
                    line=dict(color=color, width=2),
                    name=label,
                    legendgroup=f"group-{col}",
                    showlegend=not shown_legend["group"],
                ),
                row=2,
                col=col,
            )
            shown_legend["group"] = True

            fig.add_vrect(
                x0=interval_start,
                x1=interval_end,
                fillcolor="rgba(0,0,0,0.03)",
                line_width=0,
                row=1,
                col=col,
            )
            fig.add_vrect(
                x0=interval_start,
                x1=interval_end,
                fillcolor="rgba(0,0,0,0.03)",
                line_width=0,
                row=2,
                col=col,
            )

            offset += float(np.nanmax(x_vals_shifted) + 1)

        if not top_has_data:
            fig.add_annotation(
                text=f"No trajectory data available for {label} intervals.",
                x=0.23 if col == 1 else 0.77,
                y=0.88,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=10, color="gray"),
            )

        if len(plane_rows) > 0:
            plane_df = pd.concat(plane_rows, axis=0, ignore_index=True)
            angle_matrix = _compute_angle_matrix(plane_df)
            if angle_matrix is not None:
                n = angle_matrix.shape[0]
                if len(plane_x_coords) >= n:
                    heatmap_axis = np.asarray(plane_x_coords[:n], dtype=float)
                else:
                    heatmap_axis = np.arange(n, dtype=float)

                fig.add_trace(
                    go.Heatmap(
                        z=angle_matrix,
                        x=heatmap_axis,
                        y=heatmap_axis,
                        colorscale="RdBu_r",
                        zmin=0,
                        zmax=180,
                        showscale=(col == 2),
                        colorbar=dict(title="Angle (deg)"),
                    ),
                    row=3,
                    col=col,
                )

                for i in range(len(interval_x_ranges) - 1):
                    boundary = (interval_x_ranges[i][1] + interval_x_ranges[i + 1][0]) / 2
                    fig.add_hline(y=boundary, line_color="rgba(0,0,0,0.25)", line_dash="dash", row=3, col=col)
            else:
                fig.add_annotation(
                    text=f"Not enough valid SVM vectors for {label} angle matrix.",
                    x=0.23 if col == 1 else 0.77,
                    y=0.2,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                )

        fig.add_hline(y=0.5, line_dash="dash", line_color="black", line_width=1, row=2, col=col)
        fig.update_yaxes(title_text="Position (cm)", row=1, col=col)
        fig.update_yaxes(range=[0.0, 1.0], title_text="Macro F1", row=2, col=col)
        fig.update_xaxes(title_text="Timebin (concatenated across intervals)", showticklabels=True, row=1, col=col)
        fig.update_xaxes(title_text="Timebin (concatenated across intervals)", showticklabels=True, row=2, col=col)
        fig.update_xaxes(title_text="Timebin (concatenated across intervals)", showticklabels=True, row=3, col=col)
        fig.update_yaxes(title_text="Fitted planes", row=3, col=col)
        if len(interval_x_ranges) > 0:
            x_max = float(interval_x_ranges[-1][1])
            for r in (1, 2, 3):
                fig.update_xaxes(
                    autorange=False,
                    range=[0.0, x_max],
                    constrain="domain",
                    row=r,
                    col=col,
                )
            # Dashed line at the start of each non-first interval (marks t=0 for that interval)
            for i in range(1, len(interval_x_ranges)):
                iv_start = interval_x_ranges[i][0]
                fig.add_vline(x=iv_start, line_color="rgba(0,0,0,0.45)", line_dash="dash", row=1, col=col)
                fig.add_vline(x=iv_start, line_color="rgba(0,0,0,0.45)", line_dash="dash", row=2, col=col)
                fig.add_vline(x=iv_start, line_color="rgba(0,0,0,0.45)", line_dash="dash", row=3, col=col)
        if interval_tick_data:
            tv, tt = _build_axis_ticks(interval_tick_data)
            for r in (1, 2, 3):
                fig.update_xaxes(tickvals=tv, ticktext=tt, row=r, col=col)

    _render_group_column(
        col=1,
        x_modality=group1_modality,
        selected_intervals=group1_intervals,
        filters=group1_filters,
        label=group1_label,
        color="#2a9d8f",
        show_pos_colorbar=False,
    )
    _render_group_column(
        col=2,
        x_modality=group2_modality,
        selected_intervals=group2_intervals,
        filters=group2_filters,
        label=group2_label,
        color="#e76f51",
        show_pos_colorbar=True,
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=65, b=30),
        height=1150,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
    )
    return fig
