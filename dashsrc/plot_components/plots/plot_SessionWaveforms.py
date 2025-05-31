import plotly.graph_objects as go

import pandas as pd
import numpy as np
from dash import dcc, html
from plotly.subplots import make_subplots

def render_plot(spikes, spike_metadata, width, height):
    print(spikes)
    print(spike_metadata)
    print("==========================")
    
    clusters = spike_metadata.cluster_id.unique()
    sessions = spike_metadata.index.unique('session_id').unique()
    print(clusters)
    print(sessions)
    
    
    tile_height = 160
    tile_width = 40
    
    fig = make_subplots(
        rows=len(clusters), cols=len(sessions),
        shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.1 / len(clusters),
        horizontal_spacing=0.1 / len(sessions),
    )
    
        
    def draw_waveform(waveform, chnl, site_k, first_session, sess_spike_metadata,
                      gnrl_annotation):
        waveform -= site_k*60
        mean_wf = np.mean(waveform, axis=0)
        std_wf = np.std(waveform, axis=0)
        print(waveform.shape)
        # print(sess_spike_metadata)
        # Draw the standard deviation as a filled area around the mean waveform
        upper_bound = mean_wf + std_wf
        lower_bound = mean_wf - std_wf

        # draw the legend
        if site_k == 0 and first_session:
            fig.add_trace(
                go.Scatter(
                    x=[18, 23, 23],
                    y=[10, 10, -40],
                    mode='lines',
                    line=dict(color="black", width=3),
                    showlegend=False,
                ),row=i + 1, col=j + 1
            )
            for x,y,text in zip([17, 23], [10, -25], ["250us", "50uV"]):
                fig.add_annotation(
                    x=x,
                    y=y,
                    text=text,
                    xanchor="left",
                    yanchor="bottom",
                    showarrow=False,
                    row=i + 1, col=j + 1
                )
        
        # general annotation
        if site_k == 0:
            fig.add_annotation(
                x=23, y=-55,
                text=gnrl_annotation,
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=13),
                row=i + 1, col=j + 1
                
            )

        # draw the std area waveform
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(mean_wf)).tolist() + np.arange(len(mean_wf))[::-1].tolist(),
                y=upper_bound.tolist() + lower_bound[::-1].tolist(),
                fill='toself',
                fillcolor='rgba(0, 0, 0, 0.08)',
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),  # Invisible boundary lines
                name=f"Std Area {cluster}",
                showlegend=False,
            ),
            row=i + 1, col=j + 1
        )
        # Draw the mean waveform
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(std_wf)),
                y=mean_wf,
                mode='lines',
                name=f"Std Waveform {cluster}",
                line=dict(color="black", width=1),
                showlegend=False,
            ),
            row=i + 1, col=j + 1
        )
        # annotate channel
        fig.add_annotation(
            x=0, y=mean_wf[0],
            text=f"Ch{chnl:03d}",
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=12),
            row=i + 1, col=j + 1
        )
        
    
    # Update layout for better visualization
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        # title=f"Firing Rate Heatmap",
        # xaxis_title="Track Bins",
        # yaxis_title="Neurons",
        # template="plotly_white",
        height=tile_height * len(clusters) +200,
        width=tile_width * len(sessions) +200,
        # yaxis_type="category",
    )
    
    # Loop through each cluster and session to create subplots
    for j, session in enumerate(sessions):
        sess_spikes = spikes.loc[pd.IndexSlice[:, :, session]]
        sess_spike_metadata = spike_metadata.loc[pd.IndexSlice[:, :, session]]

        for i, cluster in enumerate(clusters):
            spikes_cluster = sess_spikes[sess_spikes['cluster_id'] == cluster].reset_index(drop=True)
            
            if spikes_cluster.empty:
                continue
            
            # new tile
            fig.update_yaxes(tickvals=[], ticktext=[], row=i + 1, col=j + 1)
            
            for k, site_col in enumerate(['channel', 'channel_2nd', 'channel_3rd']):
                print(spikes_cluster)
                print(sess_spike_metadata.session_nsamples)
                avg_fr = spikes_cluster.shape[0] / sess_spike_metadata.session_nsamples.iloc[0] *20_000
                gnrl_annotation = f"{avg_fr:.2f}Hz"
                
                chnl = spikes_cluster[site_col].iloc[0]
                wfs = spikes_cluster[f'{site_col}_hpf_wf']
                if wfs.isnull().all():
                    continue
                
                # wfs = np.stack(wfs.apply(lambda x: np.fromstring(x.strip("[]"), sep=" ", 
                #                                                  dtype=np.int32)))
                wfs = np.stack(wfs[:1000].apply(lambda x:x))
                print(f"Channel {site_col} {chnl}: ", wfs.shape)
                
                # wfs = wfs[:, 5:-10]
                
                draw_waveform(wfs, chnl, k, j==0, sess_spike_metadata,
                              gnrl_annotation)

            # return fig
            # # prim_chn_wfs = spikes_cluster['channel_hpf_wf']
            # # print(prim_chn_wfs)
            # # parsed_arrays = prim_chn_wfs.apply(lambda x: print(np.fromstring(x.strip("[]"), sep=" ")))
            # parsed_arrays = np.stack(prim_chn_wfs.apply(lambda x: np.fromstring(x.strip("[]"), sep=" ", dtype=np.int32)))
            # parsed_arrays = parsed_arrays[:10_000, 10:-5]
            # # parsed_arrays -= parsed_arrays[:, 9:10]
            # # for row in parsed_arrays[:10_000, ]:
            # #     fig.add_trace(
            # #             go.Scatter(
            # #                 x=np.arange(len(row)),
            # #                 y=row,
            # #                 mode='lines',
            # #                 name=f"Waveform {cluster}",
            # #                 opacity=0.4,
            # #                 line=dict(color="blue", width=1),
            # #             ),
            # #             row=i + 1, col=j + 1
            # #         )
            
            
            
        #     break
        # break
    

    
    return fig