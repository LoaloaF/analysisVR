from plotly.subplots import make_subplots
import plotly.graph_objects as go

import numpy as np

def render_plot(spikes, behavior_events):
    print(spikes.ephys_timestamp)
    print(behavior_events.reset_index().event_ephys_timestamp)
    exit()

    n_eventtypes = 5
    subplot_titles=["SingleUnits", "RewardTone", "Reward", "Lick", "ZoneEntry"]
    fig = make_subplots(rows=5, cols=1, row_heights=[.6, .01, .01, .01, .01],
                        shared_xaxes=True,
                        vertical_spacing=0.02,)
    
    for i in range(n_eventtypes):
        print(i)
        y_label = subplot_titles[i]
        fig.update_yaxes(title_text=y_label, row=i+1, col=1)
        
        if i == 0:
            fig.add_scatter(
                x=spikes['ephys_timestamp'],
                y=spikes['cluster_id'],
                mode='markers',
                marker=dict(
                    size=3,
                    symbol='line-ns-open',
                    color='black',
                    opacity=1,
                    # line=dict(width=0)
                ),
                name='Spikes',
                row=i+1, col=1
            )
        
        elif i == 1:
            sounds = behavior_events[behavior_events.event_name == 'S']
            fig.add_scatter(
                x=sounds['event_ephys_timestamp'],
                y=np.ones(len(sounds)) * 0.5,
                mode='markers',
                marker=dict(
                    size=2,
                    color='blue',
                    opacity=1,
                    symbol='circle-open',
                    line=dict(width=2)
                ),
                name='Lick',
                row=i+1, col=1
            )
        
        elif i == 2:
            reward = behavior_events[behavior_events.event_name == 'R']
            fig.add_scatter(
                x=reward['event_ephys_timestamp'],
                y=np.ones(len(reward)) * 0.5,
                mode='markers',
                marker=dict(
                    size=2,
                    color='green',
                    opacity=1,
                    symbol='circle-open',
                    line=dict(width=2)
                ),
                name='Reward',
                row=i+1, col=1
            )
            
            # draw horizontal line across all subplots
            for t in reward['event_ephys_timestamp']: 
                fig.add_shape(
                    type="line",
                    x0=t,
                    y0=1,
                    x1=t,
                    y1=0,
                    xref="x",
                    yref="paper",
                    line=dict(color="green", width=1),
                )
            
        elif i == 3:
            licks = behavior_events[behavior_events.event_name == 'L']
            fig.add_scatter(
                x=licks['event_ephys_timestamp'],
                y=np.ones(len(licks)) * 0.5,
                mode='markers',
                marker=dict(
                    size=2,
                    color='black',
                    opacity=1,
                    symbol='circle-open',
                    line=dict(width=2)
                ),
                name='Reward Tone',
                row=i+1, col=1
            )
            
        
        
    
    
    
    # Set all background colors to white
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        width=1400,
        title_text="Raw Spikes and Events",
        title_y=0.95,
        title_x=0.5
    )
    
    # Hide x-axis tick labels for all rows except the last one
    for row in range(1, 5):  # Rows 1 to 4
        fig.update_xaxes(showticklabels=False, row=row, col=1)
    
    # Show x-axis tick labels only for the bottom row (row 5)
    fig.update_xaxes(showticklabels=True, row=5, col=1)
    
    return fig