import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json

from dashsrc.components.dashvis_constants import *
        
def render_plot(ephys_traces, implant_mapping, spikes, from_smpl=0, to_smpl=10_000, 
                shanks=[1,2], um_per_px=0.15, um_per_uV=1,
                trace_groups=("non_curated", "curated", "spike_traces")):
    print(spikes)
    print(implant_mapping)
    print(ephys_traces.shape)
    print(ephys_traces[:1000, :1000])
    print("================")
    
    # from_smpl, to_smpl = (0, 10_000)
    if spikes is not None:
        spikes = spikes[(spikes['sample_id'] >= from_smpl) & (spikes['sample_id'] <= to_smpl)]
    
    implant_mapping = implant_mapping[implant_mapping['shank_id'].isin(shanks)]
    
    # height settings
    # um_per_px = 0.15
    yaxis_gap_px = 50
    shank_depths = implant_mapping.groupby('shank_id')['depth'].max().values
    # between shank y spacing in pixels 50
    height = (len(shanks)-1)*50 + sum([d for d in shank_depths])*um_per_px
    height_ratios = [d/sum(shank_depths) for d in shank_depths]
    # um_per_uV = 1
    
    
    # width = 2000
    # xaxis_gap_px = 5
    fig = make_subplots(rows=len(shanks), cols=2, row_heights=height_ratios, 
                        vertical_spacing=yaxis_gap_px/height,
                        horizontal_spacing=.001,
                        shared_yaxes=True,
                        shared_xaxes=True, column_widths=[0.04, 0.96],)
    
    _draw_shank_gemoetries(fig, implant_mapping, shanks, shank_depths)
    
    _draw_traces(fig, ephys_traces, implant_mapping, um_per_uV, shanks, from_smpl, to_smpl, 
                 trace_groups, spikes)
    
    fig.update_layout(
        height=height,
        # width=width,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            yanchor="top",  # Anchor the legend to the top
            y=0,            # Position the legend at the top (1 = top of the plot)
            xanchor="center",  # Center the legend horizontally
            x=0.5,          # Position the legend in the center horizontally
            # orientation="h",  # Horizontal orientation
            font=dict(size=12),
        )
    )
    print("Rendered plot")
    return fig

def _draw_shank_gemoetries(fig, implant_mapping, shanks, shank_depths):
    # Draw shank geometry
    for i, shank in enumerate(shanks):
        shank_mapping = implant_mapping[implant_mapping['shank_id'] == shank]
        
        # Draw fibers (lines connecting electrode pads)
        shank_mapping_2metalsfiber = shank_mapping.groupby('el_pair').filter(lambda x: len(x) > 1)
        # Prepare x and y coordinates for the lines
        x_coords = []
        y_coords = []
        for el_pair, group in shank_mapping_2metalsfiber.groupby('el_pair'):
            if len(group) > 1:
                x_coords.extend(group.shank_side.map(lambda x: -1 if x == 'left' else 1).values)
                y_coords.extend(group.depth.values)
                x_coords.append(None)  # Add None to break the line
                y_coords.append(None)  # Add None to break the line
        # Draw lines
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(color='lightgray', width=7),
                showlegend=False,
            ),
            row=i+1, col=1
        )
        
        # Create hover text for electrodes
        cols = ['amplifier_id', 'mea1k_el', 'pad_id', 'depth', 'shank_id', 'metal',
       'shank_name', 'el_pair', 'shank_side', 'mea1k_connectivity', 
       'connectivity_order','curated_trace']
        shank_mapping.loc[:,"mea1k_connectivity"] = shank_mapping.mea1k_connectivity.round(2)
        hover_text = shank_mapping.apply(
            lambda row: "<br>".join([f"{col}: {row[col]}" for col in cols]),
            axis=1
        )
        
        # Draw the electrodes
        fig.add_trace(
            go.Scatter(
                x=shank_mapping.shank_side.map(lambda x: -1 if x == 'left' else 1),
                y=shank_mapping.depth,
                mode='markers',
                marker=dict(size=10, symbol='square', color=shank_mapping.metal.map(lambda x: 'green' if x == 1 else 'purple')),
                hovertext=hover_text,  # Add hover text
                hoverinfo='text',  # Display hover text
                showlegend=False,
            ),
            row=i+1, col=1
        )
        
        # Draw the electrode ticks
        y_coords = np.stack((shank_mapping.depth, shank_mapping.depth), axis=1)
        y_coords = np.append(y_coords, np.full((y_coords.shape[0], 1), np.nan), axis=1)
        x_coords = np.tile(np.array([2, 2.5, np.nan]), (y_coords.shape[0], 1))

        fig.add_trace(
            go.Scatter(
                x=x_coords.flatten(),
                y=y_coords.flatten(),
                mode='lines',
                line=dict(color='black', width=.5),
                showlegend=False,
            ),
            row=i+1, col=1
        )
        
        # Set x-axis limits
        fig.update_xaxes(
            range=[-2.5, 2.5],
            row=i+1, col=1,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        )
        
        ylbl = shank_mapping.shank_name.values[0].replace('_', ' ').capitalize()
        ylbl += ", depth [um]"
        fig.update_yaxes(
            title=ylbl,
            title_font=dict(size=15),
            row=i+1, col=1,
            title_standoff=13,
            title_font_color='black',
            range=[shank_depths[i]+200, -600],
            autorange='reversed',
        )
    
def _draw_traces(fig, ephys_traces, implant_mapping, um_per_uV, shanks, from_smpl, 
                 to_smpl, trace_groups, spikes=None, ):
    for i, shank in enumerate(shanks):
        row = i+1
        _draw_scale_indicator(fig, i+1, um_per_uV, to_smpl)
        
        for trace_group in trace_groups:
        # for trace_group in ("non_curated", "curated", "spike_traces"):
            mask = (implant_mapping['shank_id'] == shank)
            if trace_group == "non_curated":
                mask = mask & ~implant_mapping['curated_trace']
            
            elif trace_group == "curated":
                if spikes is not None:
                    spike_trace_mask = implant_mapping.index.isin(spikes.channel)
                else:
                    spike_trace_mask = pd.Series(False, index=implant_mapping.index)
                mask = mask & implant_mapping['curated_trace'] & ~spike_trace_mask
            
            elif trace_group == "spike_traces" and spikes is not None:
                spike_trace_mask = implant_mapping.index.isin(spikes.channel)
                mask = mask & spike_trace_mask
            
            if not mask.any():
                continue
            
            # get the traces via iloc that match the trace_group mask
            ilocs = implant_mapping.index[mask].values
            shank_mapping = implant_mapping[mask]
            trace_group_name = shank_mapping.shank_name.values[0].replace('_', ' ').capitalize() + " - " + trace_group
            
            # load shank traces to memory
            shank_traces_y = ephys_traces[ilocs, from_smpl:to_smpl].astype(np.float64)
            print(shank_traces_y)
            print(shank_traces_y.shape  )
            # rescale uV to um / pixels
            shank_traces_y *= -1 /um_per_uV
            # add el depth offset
            shank_traces_y += shank_mapping.depth.values[:, np.newaxis]
            # append a nan column at the end, for line breaks
            shank_traces_y = np.append(shank_traces_y, np.full((shank_traces_y.shape[0], 1), np.nan), axis=1)
            
            # create x coordinates matching y coordinates
            shank_traces_x = np.tile(np.arange(from_smpl, to_smpl), (shank_traces_y.shape[0], 1))
            shank_traces_x = np.append(shank_traces_x, np.full((shank_traces_x.shape[0], 1), np.nan), axis=1)
            
            # draw in one go        
            fig.add_trace(
                # go.Scattergl(
                go.Scatter(
                    x = shank_traces_x.flatten(),
                    y = shank_traces_y.flatten(),
                    mode='lines',
                    opacity=0.5,
                    hoverinfo='none',
                    # visible=True if trace_group == 'spike_traces' else 'legendonly',
                    line=dict(color='black', width=1),
                    name=trace_group_name,
                ), row=row, col=2
            )

            if spikes is not None:
                # won't draw for 'curated' and also 'non_curated' (if manual curation is properly done) 
                _draw_spikes_on_traces(fig, spikes, shank_mapping, shank_traces_y, 
                                    from_smpl, trace_group_name, row)
        
            # set x range for both top and bottom x-axes
            ticktext = [f"Sample {s:,d}, t={(s*50/1_000_000) //60}min, {(s*50/1_000_000) %60:.2f}s" 
                        for s in [from_smpl, to_smpl]]
            fig.update_xaxes(
                range=[from_smpl, to_smpl],
                tickvals=[from_smpl, to_smpl],
                ticktext=ticktext,
                showgrid=False,
                zeroline=False,
                showticklabels=True,  # Enable tick labels
                ticks="outside",  # Place ticks outside
                side="bottom",  # Bottom x-axis
                row=row, col=2,
            )
            
        
def _draw_scale_indicator(fig, row, um_per_uV, to_smpl, origin=None, 
                          vertical_length_uV=500, horizontal_length_ms=20):
    """
    Draws a scale indicator on the plot.

    Args:
        fig: The Plotly figure object.
        row: The row in the subplot where the scale indicator will be drawn.
        um_per_uV: Conversion factor from microvolts to micrometers.
        to_smpl: The upper sample index for the x-axis.
        origin: A tuple (x_origin, y_origin) specifying the origin of the scale indicator.
                Defaults to (to_smpl-40, 200/um_per_uV).
        vertical_length_uV: Length of the vertical bar in microvolts (default: 500 uV).
        horizontal_length_ms: Length of the horizontal bar in milliseconds (default: 20 ms).
    """
    # Set default origin if not provided
    if origin is None:
        origin = (to_smpl - 40, -450)

    x_origin, y_origin = origin

    # Convert horizontal length from ms to samples
    horizontal_length_samples = horizontal_length_ms * 20

    # Draw vertical bar
    fig.add_trace(
        go.Scatter(
            x=[x_origin, x_origin],
            y=[y_origin, y_origin + vertical_length_uV / um_per_uV],
            mode='lines',
            line=dict(color='gray', width=4),
            showlegend=False,
        ),
        row=row, col=2
    )

    # Draw horizontal bar
    fig.add_trace(
        go.Scatter(
            x=[x_origin, x_origin - horizontal_length_samples],
            y=[y_origin, y_origin],
            mode='lines',
            line=dict(color='gray', width=4),
            showlegend=False,
        ),
        row=row, col=2
    )

    # Annotate vertical scale
    fig.add_annotation(
        x=x_origin + 10,
        y=y_origin + vertical_length_uV / (2 * um_per_uV),
        xanchor='left',
        yanchor='middle',
        text=f"{vertical_length_uV:.0f} uV",
        showarrow=False,
        font=dict(size=15, color='black'),
        row=row, col=2,
    )

    # Annotate horizontal scale
    fig.add_annotation(
        x=x_origin - horizontal_length_samples / 2,
        y=y_origin + 10,
        xanchor='center',
        yanchor='top',
        text=f"{horizontal_length_ms:.0f} ms",
        showarrow=False,
        font=dict(size=15, color='black'),
        row=row, col=2,
    )
    
def _draw_spikes_on_traces(fig, spikes, shank_mapping, shank_traces_y, from_smpl, 
                           legendgroup, row):
    count = 0
    for _, spike_info in spikes[spikes.channel.isin(shank_mapping.index)].iterrows():
        # Convert from the general iloc to the shank_wise iloc
        iloc = np.where(shank_mapping.index == spike_info.channel)[0][0]
        # wf_sample_idx = np.arange(spike_info.sample_id - 10, spike_info.sample_id + 10)
        # offset the timeslice according to the interval we look at
        wf_sample_idx = np.arange(spike_info.sample_id - 10 - from_smpl, 
                                  spike_info.sample_id + 10 - from_smpl,)
        # the _10 for the spike waveform can be larger than the raw trace arr
        wf_sample_idx = wf_sample_idx[wf_sample_idx <= shank_traces_y.shape[1]-1]
        raw_trace = shank_traces_y[iloc, wf_sample_idx]

        # Add a single trace for each waveform one by one
        fig.add_trace(
            go.Scatter(
                x=wf_sample_idx + from_smpl,
                y=raw_trace,
                mode='lines',
                opacity=0.5,
                # line=dict(color=spike_info.spike_color, width=4),
                line=dict(color=spike_info.cluster_color, width=4),
                showlegend=False,
                hoverinfo='text',
                hovertext=f"Cluster{spike_info.cluster_id:03d}",
                name=legendgroup,
                legendgroup=legendgroup,
            ),
            row=row, col=2
        )
        count += 1
    print(f"Drew {count} spikes for {legendgroup}")