import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash import dcc, html
from plotly.subplots import make_subplots
from .general_plot_helpers import make_discr_trial_cmap
from dashsrc.components.dashvis_constants import *


def draw_group(fig, dat, i, j, event_name, session_id, var_viz, group_name=None,
               color='#000000', transp_color='rgba(50, 50, 50, 0.3)'):
    """Draw one ensemble group as a line with optional percentile fill."""
    m = dat.mean(axis=0)
    fig.add_trace(
        go.Scatter(
            x=dat.columns,
            y=m,
            mode='lines',
            name=group_name,
            legendgroup=group_name,
            showlegend=True if i == 0 and j == 0 else False,
            line=dict(color=color, width=2),
        ),
        row=i + 1, col=j + 1
    )

    if var_viz == '50th perc.':
        # much faster than DataFrame.quantile
        arr = dat.to_numpy()
        upper_perc = np.nanpercentile(arr, 75, axis=0)
        lower_perc = np.nanpercentile(arr, 25, axis=0)
        x_vals = np.concatenate([dat.columns, dat.columns[::-1]])
        y_vals = np.concatenate([upper_perc, lower_perc[::-1]])

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            fill='toself',
            fillcolor=transp_color,
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=True if i == 0 and j == 0 else False,
            name='50th perc.',
            legendgroup='50th perc.',
        ), row=i + 1, col=j + 1)


def render_plot(encoding_data, group_by, group_by_values, ens_selection):
    print(encoding_data)

    # determine grouping column and color map
    if group_by == 'Cue':
        group_by_col = 'cue'
        cmap = CUE_COL_MAP
    elif group_by == 'Outcome':
        group_by_col = 'trial_outcome'
        cmap = OUTCOME_COL_MAP
    elif group_by == 'Part of session':
        group_by_col = 'trial_id'
        cmap = make_discr_trial_cmap(encoding_data['trial_id'].nunique(), TRIAL_COL_MAP)
    elif group_by == 'R1 choice':
        group_by_col = 'choice_R1'
        cmap = R1_CHOICE_CMAP
    elif group_by == 'R2 choice':
        group_by_col = 'choice_R2'
        cmap = R2_CHOICE_CMAP
    else:
        group_by_col = None
        print("No group_by selected, using 'None'")
        print(group_by)

    print('--------------------------')
    print(f"Group by: {group_by} ({group_by_col}) ")
    print('--------------------------')

    var_viz = '50th perc.'
    smooth = 6
    yrange = [-0.4, 3]
    event_names = encoding_data['t0_event_name'].unique()
    session_ids = encoding_data.index.unique('session_id')

    # subplot titles
    subplot_titles = []
    for i, event_name in enumerate(event_names):
        for j, session_id in enumerate(session_ids):
            if j == 0:
                title = event_name.replace('_', ' ').capitalize().replace("zone", '') + ' over Sessions'
                subplot_titles.append(title)
            else:
                subplot_titles.append(None)

    fig = make_subplots(
        rows=event_names.size, cols=session_ids.size,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=.1,
        horizontal_spacing=.1 / session_ids.size,
        subplot_titles=subplot_titles,
    )

    fig.update_layout(
        height=350 * event_names.size,
        template="plotly_white",
        margin=dict(t=50, b=20, l=100, r=20),
        title_font=dict(size=28, color='black'),
    )

    # precompute smoothed data
    precomputed_data = {}
    for ev in event_names:
        dat_event = encoding_data[encoding_data['t0_event_name'] == ev]
        for s in session_ids:
            sub = dat_event.loc[s, ens_selection].unstack(level='interval_t')
            if smooth is not None:
                sub = sub.T.rolling(window=smooth, center=True,
                                    min_periods=smooth).median().dropna(how='all').T
            precomputed_data[(ev, s)] = sub

    shapes = []
    annotations = []

    with fig.batch_update():  # defer expensive layout rebuilds
        # iterate rows
        for i, event_name in enumerate(event_names):
            fig.update_yaxes(
                title_text="Ensemble Activation [a.u.]",
                row=i + 1, col=1,
                title_standoff=10,
                title_font=dict(size=18, color='black'),
                tickfont=dict(size=18, color='black'),
                tickvals=[0],
                showgrid=True, gridcolor='grey',
                zeroline=True,
                zerolinecolor='lightgrey',
                zerolinewidth=1,
                range=yrange,
            )

            # iterate columns
            for j, session_id in enumerate(session_ids):
                ens_dat = precomputed_data[(event_name, session_id)]

                fig.update_xaxes(
                    title_text="entry time (ms)",
                    title_font=dict(size=14, color='black'),
                    tickfont=dict(size=14, color='black'),
                    tickvals=[12, 25, 38],
                    ticktext=['-500', '0', '500'],
                    showticklabels=True,
                    row=i + 1, col=j + 1,
                )

                # collect shapes instead of adding directly
                shapes.append(dict(
                    type='line', x0=25, x1=25, y0=yrange[0], y1=yrange[1],
                    line=dict(color='black', width=2, dash='dash'),
                    xref=f"x{j+1}", yref=f"y{i+1}"
                ))
                shapes.append(dict(
                    type='line', x0=0, x1=100, y0=0, y1=0,
                    line=dict(color='lightgray', width=2),
                    xref=f"x{j+1}", yref=f"y{i+1}"
                ))

                # collect annotations
                annotations.append(dict(
                    x=35, y=1.45, text=f"S{session_id:02}", showarrow=False,
                    font=dict(size=16, color='black'),
                    xref=f"x{j+1}", yref=f"y{i+1}"
                ))

                # plot traces
                if group_by == 'None':
                    draw_group(fig, ens_dat, i, j, event_name, session_id, var_viz)
                else:
                    dat = encoding_data[encoding_data['t0_event_name'] == event_name]
                    for group_name, group_vals in group_by_values.items():
                        group_dat_vals = dat.loc[session_id, group_by_col].unstack(level='interval_t').iloc[:, 0]
                        mask = group_dat_vals.isin(group_vals)
                        if mask.any():
                            group_dat = ens_dat[mask]
                            color = cmap[group_vals[0]]
                            cmap_transparent = {
                                k: v.replace("rgb", "rgba")[:-1] + f', {MULTI_TRACES_ALPHA})'
                                for k, v in cmap.items()
                            }
                            transp_color = cmap_transparent[group_vals[0]]
                            draw_group(fig, group_dat, i, j, event_name, session_id,
                                       var_viz, group_name, color=color, transp_color=transp_color)

    # apply all shapes and annotations at once
    fig.update_layout(shapes=shapes, annotations=annotations)
    return fig
