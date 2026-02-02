import plotly.graph_objects as go
import json
import re
from scipy.stats import zscore
        
import pandas as pd
import numpy as np
from dash import dcc, html
from plotly.subplots import make_subplots
import plotly.express as px

from analytics_processing.analytics_constants import PARADIGM_NAMES

def render_plot(spike_metadata, width, height):
    clusters = spike_metadata.cluster_id.unique()
    session_ids = spike_metadata.index.unique('session_id').unique()
    
    # Parse session_ids (format: "YYYY-MM-DD_HH-MM") to datetime and calculate session durations
    session_start_times = {s_id: pd.to_datetime(s_id, format="%Y-%m-%d_%H-%M") for s_id in session_ids}
    
    # Calculate session end times using session_nsamples / 20_000 Hz
    session_end_times = {}
    for s_id in session_ids:
        session_data = spike_metadata.xs(s_id, level='session_id')
        session_nsamples = session_data['session_nsamples'].iloc[0]
        session_duration_seconds = session_nsamples / 20_000
        session_end_times[s_id] = session_start_times[s_id] + pd.Timedelta(seconds=session_duration_seconds)
    
    tile_height = 70
    tile_width = 40
    
    left_margin = 100
    right_margin = 50
    top_margin = 50
    bottom_margin = 50
    
    fig = make_subplots(rows=1, cols=1,)
    
    x0s = np.arange(len(session_ids)) * tile_width
    y0s = np.arange(len(clusters)) * tile_height

    # Add gray background for paradigm_id 0 sessions
    for i, s_id in enumerate(session_ids):
        paradigm_id = spike_metadata.xs(s_id, level='session_id').index.get_level_values('paradigm_id')[0]
        if paradigm_id == 0:
            fig.add_shape(
                type="rect",
                x0=x0s[i],
                y0=0,
                x1=x0s[i] + tile_width,
                y1=len(clusters) * tile_height,
                fillcolor="lightgray",
                opacity=0.25,
                layer="below",
                line_width=0,
            )

    # draw grid of tiles
    for i in range(len(clusters)+1):
        fig.add_trace(
            # horizontal lines
            go.Scatter(
                x=[0, len(session_ids) * tile_width],
                y=[tile_height * i, tile_height * i],
                mode='lines',
                opacity=0.2,
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip',  
            )
            
        )
    for i in range(len(session_ids)+1):
        fig.add_trace(
            go.Scatter(
                x=[tile_width * i, tile_width * i],
                y=[0, len(clusters) * tile_height],
                mode='lines',
                opacity=0.2,
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip',  
            )
        )

    # reverse y axis
    fig.update_yaxes(
        range=[len(clusters)*tile_height +bottom_margin, 0-top_margin],
        showticklabels=False,
        showgrid=False,
    )
    # revers x axis
    fig.update_xaxes(
        range=[0-left_margin, len(session_ids)*tile_width +right_margin],
        showgrid=False,
    )

    # Update layout for better visualization
    fig.update_layout(
        # plot_bgcolor='white',
        # paper_bgcolor='white',
        # title=f"Firing Rate Heatmap",
        # xaxis_title="Track Bins",
        # yaxis_title="Neurons",
        template="plotly_white",
        height=tile_height * len(clusters) +top_margin + bottom_margin,
        width=tile_width * len(session_ids) +left_margin + right_margin,
        margin=dict(l=0, r=0, t=0, b=0),
        
        
    )
    
    def format_timedelta(td):
        """Format timedelta as Xd, Xh, or Xmin depending on size."""
        total_minutes = td.total_seconds() / 60
        if total_minutes >= 20 * 60:  # >20h round up to 1 day
            days = round(total_minutes / (60 * 24))
            return f"{days}d"
        elif total_minutes >= 90:  # 1.5 hours
            hours = round(total_minutes / 60)
            return f"{hours}h"
        else:
            minutes = round(total_minutes)
            return f"{minutes}min"
        
    # add annotations for session_ids and clusters
    prv_session_end_time = None
    for i, s_id in enumerate(session_ids):
        # Get paradigm_id from spike_metadata for this session
        paradigm_id = spike_metadata.xs(s_id, level='session_id').index.get_level_values('paradigm_id')[0]
        paradigm_name = PARADIGM_NAMES.get(paradigm_id, f"P{paradigm_id}")
        
        fig.add_annotation(
            x=x0s[i] + tile_width / 2,
            y=-4,
            text=paradigm_name,
            showarrow=False,
            font=dict(size=7.5),
            xanchor="center",
            yanchor="bottom",
        )
        if prv_session_end_time is not None:
            # Calculate time delta between end of previous session and start of current session
            timedelta = session_start_times[s_id] - prv_session_end_time
            fig.add_annotation(
                x=x0s[i],
                y=-30,
                text=f"{format_timedelta(timedelta)}",
                showarrow=False,
                font=dict(size=9),
                xanchor="center",
                yanchor="bottom",
            )
        prv_session_end_time = session_end_times[s_id]
        
        
        
    for i, cluster in enumerate(clusters):
        fig.add_annotation(
            x=-10,
            y=y0s[i] + tile_height / 2,
            text=f"Cluster {cluster}",
            showarrow=False,
            font=dict(size=12),
            xanchor="right",
            yanchor="middle",
        )
        region = spike_metadata[spike_metadata.cluster_id == cluster].gross_brain_area.unique()
        fig.add_annotation(
            x=-10,
            y=y0s[i] + tile_height/2 +12,
            text=f"{region[0]}",
            showarrow=False,
            font=dict(size=9),
            xanchor="right",
            yanchor="middle",
        )
    
    # return fig
    
    def draw_scalebars(fig, origin=(0, 0), uV=50, us=1000):
        """
        Draws scale bars for amplitude (uV) and time (us) using the waveform scaling functions.
        - uV: vertical scale bar length in microvolts
        - us: horizontal scale bar length in microseconds
        """
        # Calculate pixel lengths using the scaling lambdas
        # us to samples: samples = us * 20 / 1000 (since 20,000 Hz = 20 samples per ms)
        n_samples = int(us * 20 / 1000)
        x0, y0 = origin
        x1 = x0 + X_smpls2px(n_samples)
        y1 = y0 - Y_uV2px(uV)

        # Draw horizontal (time) bar
        # Draw vertical (amplitude) bar
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1],
                y=[y0, y0, y1],
                mode='lines',
                line=dict(color="black", width=3),
                showlegend=False,
                hoverinfo='skip',
            )
        )

        # Add text annotations
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=y0,
            text=f"{us//1000}ms",
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=12),
        )
        fig.add_annotation(
            x=x1 + 5,
            y=(y0 + y1) / 2,
            text=f"{uV}μV",
            xanchor="left",
            yanchor="middle",
            showarrow=False,
            font=dict(size=12),
        )
        
    # def draw_scalebars(fig, origin=(0, 0), uV=50, us=250):
        # # Draw the scale bars for each subplot
        # fig.add_trace(
        #     go.Scatter(
        #         x=[origin[0], origin[0] + 20, origin[0] + 20],
        #         y=[origin[1], origin[1], origin[1] + 50],
        #         mode='lines',
        #         line=dict(color="black", width=3),
        #         showlegend=False,
        #         hoverinfo='skip',
        #     ), row=i + 1, col=j + 1
        # )
        # for x,y,text in zip([17, 23], [10, -25], ["250us", "50uV"]):
        #     fig.add_annotation(
        #         x=x,
        #         y=y,
        #         text=text,
        #         xanchor="left",
        #         yanchor="bottom",
        #         showarrow=False,
        #         row=i + 1, col=j + 1
        #     )
    
    def draw_waveform(mean_wf, std_wf, x, annotation, col): 
        # Draw the mean waveform
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean_wf,
                mode='lines',
                name=annotation,
                line=dict(color=col, width=1),
                showlegend=False,
                hovertemplate=f"{annotation}<extra></extra>",  
                hoverlabel=dict(bgcolor="lightgray"),  
            ),
        )
    
    
    def extract_sessionwise_wfs():
        sessionw_wfs_counts = []
        sessionw_wfs = []
        for j, session in enumerate(session_ids):
            sess_spike_metadata = clst_metdadata.loc[pd.IndexSlice[:, :, session]].iloc[0]
            n_spikes = sess_spike_metadata.session_spike_count
            if n_spikes == 0:
                continue

            # iterate over channels for session+cluster and aggreate 
            # how often each channel has a waveform (1st, 2nd, 3rd hieghest)
            all_chnl_counts = []
            for chnl_count_col, chnl_count_col_annot in zip(['channel_wf_count', 'channel_2nd_wf_count', 'channel_3rd_wf_count'], 
                                                            ('1st-Chnl', '2nd-Chnl', '3rd-Chnl')):
                if chnl_count_col not in sess_spike_metadata: # often 3rd wf
                    continue
                chnl_wf_counts = {int(chnl):int(cnt/sess_spike_metadata.session_spike_count*100) 
                                for chnl,cnt in sess_spike_metadata[chnl_count_col].items() 
                                if cnt is not None}
                all_chnl_counts.append(pd.Series(chnl_wf_counts, name=(session,chnl_count_col_annot)))
            all_chnl_counts = pd.concat(all_chnl_counts, axis=1)
            # print(all_chnl_counts)
            sessionw_wfs_counts.append(all_chnl_counts)
            
            # iterate over channels for session+cluster and aggreate
            # average and std waveforms for each channel, irrespective of 1st, 2nd, 3rd highest
            avg_wfs = {int(chnl):wf for chnl,wf in sess_spike_metadata['averge_wf'].items() 
                       if int(chnl) in all_chnl_counts.index}
            std_wfs = {int(chnl):wf for chnl,wf in sess_spike_metadata['std_wf'].items() 
                       if int(chnl) in all_chnl_counts.index}
            peak_uV = {int(chnl):wf.min().item() *-1 
                       for chnl,wf in sess_spike_metadata['averge_wf'].items() if wf is not None}
            sessionw_wfs.append(pd.Series(avg_wfs, name=(session,'avg_wf')))
            sessionw_wfs.append(pd.Series(peak_uV, name=(session,'peak_uV')))
            sessionw_wfs.append(pd.Series(std_wfs, name=(session,'std_wf')))
        
        # concat counts over session_ids
        sessionw_wfs_counts = pd.concat(sessionw_wfs_counts, axis=1)
        # rank channels by max count over all session_ids, devide which to draw
        all_chnls_ranked = sessionw_wfs_counts.groupby(level=0).max()
        all_chnls_ranked = all_chnls_ranked.reindex(all_chnls_ranked.sum(axis=1).sort_values(ascending=False).index, axis=0)
        selected_chnls = all_chnls_ranked.index[:n_channels]
        # the non selected channels (sum over all of them)
        sessionw_wfs_other_counts = sessionw_wfs_counts.drop(selected_chnls).sum(axis=0).T.rename(-1).to_frame().T
        sessionw_wfs_counts = sessionw_wfs_counts.loc[selected_chnls].fillna(0).astype(int)
        # insert the non selected channels as a single channel
        sessionw_wfs_counts = pd.concat([sessionw_wfs_counts, sessionw_wfs_other_counts], axis=0)
            
        # aggregate waveforms for selected channels, select only those
        # selected with high counts caluclated above
        sessionw_wfs = pd.concat(sessionw_wfs, axis=1)
        sessionw_wfs = sessionw_wfs.loc[selected_chnls]
        # remove channels with low counts, smaller than 20% of spikes, won't be ddrawn
        for s_id in session_ids:
            for chnl in sessionw_wfs.index:
                if s_id not in sessionw_wfs_counts.columns:
                    continue
                if ((sessionw_wfs_counts.loc[chnl, s_id] > min_spikes_perc).any()):
                    continue
                sessionw_wfs.loc[chnl, s_id] = np.nan
                
        # now use the counts to create annotations for each session 
        sessionw_wfs_annots = {}
        for s_id in sessionw_wfs_counts.columns.levels[0]:
            info = sessionw_wfs_counts[s_id].astype(int).astype(str)
            info['peak ampl.'] = sessionw_wfs.loc[:, (s_id, 'peak_uV')].astype(object)
            info.loc[:,'peak ampl.'] = info['peak ampl.'].astype(str) + " uV"
            
            info.index = [f"C{idx if idx != -1 else '___'}:" for idx in info.index]
            info.loc[:,['1st-Chnl','2nd-Chnl']] = info.loc[:, ['1st-Chnl','2nd-Chnl']].astype(str) + " %"
            sessionw_wfs_annots[s_id] = info.to_string().replace("\n", "<br>")
        return sessionw_wfs, sessionw_wfs_annots

    def highlight_channel_row(annotation, channel_label, color="#ff6600"):
        # This will wrap the row starting with channel_label in a span
        
        # Escape for regex
        channel_label_escaped = re.escape(channel_label)
        # Replace only the first occurrence of the row starting with channel_label
        return re.sub(
            rf'({channel_label_escaped}.*?)(<br>|$)',
            rf'<span style="color:{color}">\1</span>\2',
            annotation,
            count=1
        )
        
    def calc_normed_avg_frate(clst_metdadata):
        # calculate the normalized average firing rate for the cluster
        # for all sessions, used for plotting
        sess_spk_cnt = clst_metdadata['session_spike_count']
        sess_seconds = clst_metdadata['session_nsamples']/20_000
        normed_avg_frate = (sess_spk_cnt /sess_seconds).to_frame().rename(columns={0: 'avg_frate'})
        normed_avg_frate.index = normed_avg_frate.index.droplevel((0,1,3))
        normed_avg_frate['normed_avg_frate'] = normed_avg_frate.values /normed_avg_frate.values.max()
        normed_avg_frate['n_spikes'] = sess_spk_cnt.values
        normed_avg_frate['sess_seconds'] = sess_seconds.values
        return normed_avg_frate

    def draw_avg_frate(normed_avg_frate, cluster_idx, s):
        # Get the per-session values
        avg_frate = normed_avg_frate.loc[:, 'normed_avg_frate']
        # Build a list of annotations for each session
        hover_texts = [
            f"Avg f. rate: {sess_vals.avg_frate:.2f}Hz<br>n={int(sess_vals.n_spikes):,d}"
            f" spikes, {int(sess_vals.sess_seconds/60):.2f}min<br>"
            f"Normed: {sess_vals.normed_avg_frate:.2f}, S{sess}"
            for sess, sess_vals in normed_avg_frate.iterrows()
        ]
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(session_ids)) * tile_width + tile_width * (2/3),
                y=np.full(len(session_ids), cluster_idx * tile_height + tile_height * (5/6)),
                mode='markers', hoverinfo='skip', showlegend=False,
                marker=dict(size=s,symbol='circle-open', color='lightgray', showscale=False),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(session_ids)) * tile_width + tile_width * (2/3),
                y=np.full(len(session_ids), cluster_idx * tile_height + tile_height * (5/6)),
                mode='markers',
                marker=dict(
                    size=s-1,
                    symbol='circle',
                    color=avg_frate,
                    colorscale=normed_fr_cscale,
                    showscale=False,
                    cmin=avg_frate.min(),
                    cmax=avg_frate.max(),
                ),
                showlegend=False,
                text=hover_texts,  # <-- Per-point annotation
                hovertemplate="%{text}<extra></extra>",  # <-- Use text for hover
                name=f"Cluster {cluster_idx} Avg Firing Rate",
            )
        )
        
    normed_fr_cscale = [[0,'#ffffff'], [.1,'#eaeaea'], [.2,'#d9d9d9'], 
                        [.3,'#c9c9c9'], [.4,'#b7b7b7',], 
                        [.5,'#9b9b9b'], [.6,'#818181'], [.7,'#716468'], 
                        [.8,'#70525c'], [.9,'#7d3d54'], [1,'#752541']]
        
    # wf_colors = ['#04724D', '#56876D', '#8DB38B', '#7D8060']
    # use plotly discerte color scale
    wf_colors = px.colors.qualitative.Plotly

    n_channels = 3
    
    min_spikes_perc = 10 # minimum percentage of spikes for a channel to be drawn
    min_firing_rate = 0.1  # Hz - minimum firing rate to draw waveforms
    markersize = 10
    
    waveform_width_scaler = .4
    uV2px_scaler = -.6
    
    X_smpls2px = lambda smpls: np.array(smpls) * (tile_width/25) *waveform_width_scaler
    Y_uV2px = lambda uV: np.array(uV) * uV2px_scaler
    
    intial_y_offset = 10
    intial_x_offset = 4
    chnl_y_spacing = 15
    
    draw_frate_markers = False
    draw_waveforms = True

    y_offset = intial_y_offset
    x_offset = intial_x_offset
    # Loop through each cluster and session to create subplots
    for i, cluster_id in enumerate((spike_metadata.cluster_id.unique())):
        print(f"Processing cluster {cluster_id} ({i+1}/{len(clusters)})")
        clst_metdadata = spike_metadata[spike_metadata.cluster_id == cluster_id]
        
        # get the spikes for the current cluster, unpack convoluted dicts
        normed_avg_frate = calc_normed_avg_frate(clst_metdadata)
        
        # draw scatter markers for average firing rate
        if draw_frate_markers:
            draw_avg_frate(normed_avg_frate, i, s=markersize)
        
        # draw the waveforms for each session
        if draw_waveforms:
            sessionw_wfs, sessionw_wfs_annots = extract_sessionwise_wfs()
            has_any_waveform = False  # Track if cluster has any waveforms drawn
            
            for j, s_id in enumerate(session_ids):
                # Check firing rate threshold
                if s_id not in normed_avg_frate.index:
                    x_offset += tile_width
                    continue
                    
                firing_rate = normed_avg_frate.loc[s_id, 'avg_frate']
                if firing_rate < min_firing_rate:
                    x_offset += tile_width
                    continue
                
                if i == 5 and j == 0:
                    # draw the scale bars only once
                    draw_scalebars(fig, origin=(tile_width*(1/4), tile_height*i +intial_y_offset))
                
                if s_id not in sessionw_wfs.columns.levels[0]: # no spike for this cluster+session
                    # shift to next box 
                    x_offset += tile_width
                    continue
                
                # draw avg firing rate as backgound color
                base_annot = sessionw_wfs_annots[s_id]
                
                for k, chnl in enumerate(sessionw_wfs.index):
                    # convert from data values to pixels
                    mean_wf = Y_uV2px(sessionw_wfs.loc[chnl, (s_id, 'avg_wf')]) +y_offset +chnl_y_spacing*k
                    std_wf = Y_uV2px(sessionw_wfs.loc[chnl, (s_id, 'std_wf')]) 
                    if not isinstance(mean_wf, np.ndarray): # is NA
                        continue
                    
                    has_any_waveform = True  # Found at least one waveform to draw
                    # print(base_annot)
                    annot = highlight_channel_row(base_annot, f"C{chnl}:", color=wf_colors[k])
                    # print(normed_avg_frate)
                    annot += (f"<br>" + f"{normed_avg_frate.loc[s_id,'avg_frate']:.2f}Hz,"
                              f" n={normed_avg_frate.loc[s_id,'n_spikes']:,d} spikes")
                    x = X_smpls2px(np.arange(len(mean_wf))) +x_offset
                    draw_waveform(mean_wf, std_wf, x, annotation=annot, col=wf_colors[k])
                
                # shift to next box 
                x_offset += tile_width
            
            # If no waveforms were drawn for this cluster, skip the y_offset increment
            if not has_any_waveform:
                print(f"  Cluster {cluster_id} has no waveforms above {min_firing_rate}Hz threshold")
                # Still reset x_offset but don't increment y_offset
                x_offset = intial_x_offset
                continue
            
        # reset x_offset for next cluster    
        x_offset = intial_x_offset
        # shift down for next cluster
        y_offset += tile_height
    return fig