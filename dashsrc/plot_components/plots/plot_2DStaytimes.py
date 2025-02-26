import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashsrc.components.dashvis_constants import *

def render_plot(data, metadata, width, height):

    # Data transformations    
    data.loc[:, ['staytime_correct_r', 'staytime_incorrect_r']] /= 1_000_000  # us to s
    # reset trial_id to set of sessions, each is unique
    data = data.droplevel('entry_id')
    data['entry_id'] = np.arange(len(data))
    data.set_index('entry_id', append=True, inplace=True)
    
    print(data)
    print(metadata)
    print("================")
    
    # # do the plotting
    fig = make_subplots(rows=1, cols=1)
    
    cue1_stayratios = []
    cue2_stayratios = []
    drew_dr_legend = False
    drew_rf_legend = False
    for s_id in data.index.unique(level='session_id'):
        print(s_id)
        s_data = data.loc[pd.IndexSlice[:,:,s_id,:]]
        print("=====")
        
        
        # session_type = data.loc[pd.IndexSlice[:,:,s_id,:], ["both_R1_R2_rewarded", "flip_Cue1R1_Cue2R2"]]
        session_type = data.loc[pd.IndexSlice[:,:,s_id,:], ["DR", "RF"]].fillna(0.0).astype(bool).all(axis=0)
        if session_type['DR']:
            fig.add_scatter(x=[s_id], y=[-.666], mode='markers', 
                            marker=dict(color='red'),
                            name='both R1 and R2 rewarded',
                            showlegend=not drew_dr_legend)
            drew_dr_legend = True
        if session_type['RF']:
            fig.add_scatter(x=[s_id], y=[-.666], mode='markers', 
                            marker=dict(color='blue'),
                            name='flip Cue1->R1 and Cue2->R2',
                            showlegend=not drew_rf_legend)
            drew_rf_legend = True
        
        stayratio = (s_data.staytime_reward1 - s_data.staytime_reward2) / (s_data.staytime_reward1 + s_data.staytime_reward2)
        cue1_stayratio = stayratio[s_data.cue == 1].mean()
        cue2_stayratio = stayratio[s_data.cue == 2].mean()
        cue1_stayratios.append(cue1_stayratio)
        cue2_stayratios.append(cue2_stayratio)
        
    # plot
    fig.add_trace(go.Scatter(x=np.arange(len(cue1_stayratios)), 
                             y=cue1_stayratios, mode='lines', name='Cue 1',
                             line=dict(color=CUE_COL_MAP[1])))
    fig.add_trace(go.Scatter(x=np.arange(len(cue2_stayratios)),
                                y=cue2_stayratios, mode='lines', name='Cue 2',
                                line=dict(color=CUE_COL_MAP[2])))
        
    # set range from 0.5 to -0.5, add y labels with x2 longer at R1 vs R2
    fig.update_yaxes(range=[-0.7, 0.7], 
                     title_text='Longer at R2             Longer at R1',
                     titlefont=dict(size=12),
                     tickvals=[-4/6, -3/5, -2/4, -1/3, 0, 1/3, 2/4, 3/5, 4/6],
                     ticktext=['5x','4x','3x', '2x','0','2x','3x','4x','5x'],
                     showgrid=True,
                     zeroline=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,.14)',
    )
    
    # add manual zeroline
    fig.add_shape(type="line",
        x0=0, y0=0, x1=len(cue1_stayratios), y1=0,
        line=dict(color="gray",width=.5),
    )
    
    # add annotations
    # fig.add_annotation(x=0, y=0.5, text="longer at R1", showarrow=False, align='right')
    # fig.add_annotation(x=0, y=-0.5, text="longer at R2", showarrow=False)
    fig.update_xaxes(title_text='Session ID')
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=20, t=50, b=50)  # Adjust margins as needed
    )
    
    # data['x_indices'] = _calc_x_positions(data)

    # # add outcome color and  cue color columns for later    
    # cmap_transparent = {k: v.replace("rgb","rgba")[:-1]+f', {MULTI_MARKERS_ALPHA})' 
    #                     for k,v in OUTCOME_COL_MAP.items()}
    # data['outcome_color'] = data['trial_outcome'].map(cmap_transparent)
    # data['cue_color'] = data['cue'].map(CUE_COL_MAP)
    
    
    # _draw_violin_plots(fig, data, metric_max, indicate_outcome=True)
    # _draw_treshold_and_cue_indicator(fig, data, sort_by, include_threshold=True)
    
    # _draw_treshold_and_cue_indicator(fig, data, sort_by, include_threshold=True)
    # _draw_single_trials(fig, data)

    # _configure_axis(fig, data, metric_max)
    return fig