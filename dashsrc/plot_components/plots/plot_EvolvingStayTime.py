import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashsrc.components.dashvis_constants import *
from .general_plot_helpers import make_discr_trial_cmap

def render_plot(data, metric_max, width, height):

    # Data transformations    
    data.loc[:, ['staytime_reward1', 'staytime_reward2']] /= 1_000_000  # us to s
    # clip to 0, 10 seconds
    data.loc[:, ['staytime_reward1', 'staytime_reward2']] = data.loc[:, ['staytime_reward1', 'staytime_reward2']].clip(0, metric_max)
    
    # reset trial_id to set of sessions, each is unique
    data = data.droplevel('entry_id')
    data['entry_id'] = np.arange(len(data))
    data.set_index('entry_id', append=True, inplace=True)
    
    # add outcome color and  cue color columns for later    
    cmap_transparent = {k: v.replace("rgb","rgba")[:-1]+f', {MULTI_MARKERS_ALPHA})' 
                        for k,v in OUTCOME_COL_MAP.items()}
    data['outcome_color'] = data['trial_outcome'].map(cmap_transparent)
    data['cue_color'] = data['cue'].map(CUE_COL_MAP)
    
    
    
    
    print(data)
    # print(metadata)
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
        print(s_data)
        print("=====")
        
        print(s_data['outcome_color'])
        
        n_trials = len(s_data)
        cmap =  make_discr_trial_cmap(n_trials, TRIAL_COL_MAP)
        cmap_transparent = {k: v.replace("rgb","rgba")[:-1]+f', {MULTI_TRACES_ALPHA})' 
                            for k,v in cmap.items()}
        s_data['trial_color'] = s_data['trial_id'].map(cmap_transparent).fillna('rgba(128,128,128,.14)')
        print(s_data['trial_color'])
        
        # session_type = data.loc[pd.IndexSlice[:,:,s_id,:], ["both_R1_R2_rewarded", "flip_Cue1R1_Cue2R2"]]
        # session_type = data.loc[pd.IndexSlice[:,:,s_id,:], ["DR", "RF"]].fillna(0.0).astype(bool).all(axis=0)
        # if session_type['DR']:
        fig.add_scatter3d(
                        z=s_data.staytime_reward1, 
                        y=s_data.staytime_reward2,
                        x=s_data.index.get_level_values('entry_id'),
                        mode='markers',
                        name='Trial Staytimes',
                        line=dict(color='rgba(128,128,128,.14)'),
                        marker=dict(color=s_data['outcome_color'], size=6),
                        legendgroup='Trial Staytimes',
                        showlegend=not drew_dr_legend)
        drew_dr_legend = True
        
        # add a line with smoothing of the same data as the scatter above
        smoothed_data = s_data[['staytime_reward1', 'staytime_reward2']].ewm(span=20, adjust=False).mean()
        fig.add_scatter3d(z=smoothed_data.staytime_reward1,
                        y=smoothed_data.staytime_reward2,
                        x=smoothed_data.index.get_level_values('entry_id'),
                        mode='lines',
                        line=dict(color='rgba(128,128,128,.14)', width=5),
                        showlegend=False)
        # draw 2d projection of the line for each plane
        fig.add_scatter3d(z=smoothed_data.staytime_reward1,
                        y=np.zeros_like(smoothed_data.staytime_reward2),
                        x=smoothed_data.index.get_level_values('entry_id'),
                        mode='lines',
                        legendgroup='Close (R1) Staytime',
                        line=dict(color='rgba(128,128,128,.14)', width=10,),
                        name='Close (R1) Staytime',
                        showlegend=not drew_rf_legend,
                        )
        fig.add_scatter3d(z=np.zeros_like(smoothed_data.staytime_reward1),
                        y=smoothed_data.staytime_reward2,
                        x=smoothed_data.index.get_level_values('entry_id'),
                        mode='lines',
                        line=dict(color='rgba(128,128,128,.14)', width=10,),
                        legendgroup='Far (R2) Staytime',
                        name='Far (R2) Staytime',
                        showlegend=not drew_rf_legend,
                        )
        
        last_id = s_data.index.get_level_values('entry_id').max()
        fig.add_trace(go.Mesh3d(
            x=[last_id+.001, last_id+.001, last_id, last_id],
            y=[0, 10, 10, 0],
            z=[0, 0, 10, 10],
            color='rgba(228, 228, 228, .8)',
            opacity=.8,
            name='Session End',
            legendgroup='Session End',
            showlegend=not drew_rf_legend,
        ))
        
        # add line one by one with seperate color of s_data['trial_color']
        line_lengths =[2]
        for i in range(len(s_data)):
            trial_data = s_data.iloc[i:i+2]
            line_length = (np.sqrt(np.diff(trial_data[['staytime_reward1', 'staytime_reward2']], axis=0).sum()**2))
            alpha = np.clip([(line_length/5)],0,1)[0]
            if pd.isna(alpha):
                alpha = .1
            fig.add_scatter3d(z=trial_data.staytime_reward1, 
                            y=trial_data.staytime_reward2,
                            x=trial_data.index.get_level_values('entry_id'),
                            mode='lines',
                            # line=dict(color=trial_data['trial_color'].iloc[0], width=5),
                            opacity=alpha,
                            line=dict(color=f'rgba(50,50,50,{alpha:.1f})', width=5),
                            legendgroup='Trial changes',
                            name="Trial changes",
                            showlegend=not drew_rf_legend,)
            drew_rf_legend = True
            
            
        
        # label z axis as trial id
        fig.update_layout(scene=dict(xaxis_title='Trial ID', 
                                     yaxis_title='Stay time at R2 (s)', 
                                     zaxis_title='Stay time at R1 (s)',
                                    zaxis_range=[0, metric_max],
                                    yaxis_range=[0, metric_max],
                                     camera = {
                                        "projection": {
                                            "type": "orthographic"
                                        },
                                    },
                            aspectmode='manual',
                            aspectratio=dict(x=data.shape[0]/30, 
                                             y=data.shape[0]/(30*10), 
                                             z=data.shape[0]/(30*10)),
                            xaxis_showbackground=True,
                            yaxis_showbackground=True,
                            zaxis_showbackground=True,
                            xaxis_backgroundcolor='white',
                            yaxis_backgroundcolor=REWARD_LOCATION_COL_MAP[1],
                            zaxis_backgroundcolor=REWARD_LOCATION_COL_MAP[2],
                            
            )
            )
        

    
    # fig.update_xaxes(range=[0, metric_max], title_text='Stay time at R1 (s)')
    # fig.update_yaxes(range=[0, metric_max], title_text='Stay time at R2 (s)')
    # fig.update_layout(scene=dict(xaxis=dict(range=[len(s_data),0], title='Trial ID')))

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=20, t=50, b=50)  # Adjust margins as needed
    )
    
    return fig