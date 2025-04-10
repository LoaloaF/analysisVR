import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashsrc.components.dashvis_constants import *
from .general_plot_helpers import make_discr_trial_cmap

def render_plot(data, mode_span, width, height):

    # Data transformations    
    data.loc[:, ['staytime_reward1', 'staytime_reward2']] /= 1_000_000  # us to s
    # clip to 0, 10 seconds
    data.loc[:, ['chose_reward1', 'chose_reward2']] = (data.loc[:, ['staytime_reward1', 'staytime_reward2']] > 2).values
    print(data)
    # exit()
    
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
    trials = data.index.get_level_values('entry_id')
    fig = make_subplots(rows=1, cols=1)
    
    
    data['chose_only_R1'] = data['chose_reward1'] & ~data['chose_reward2']
    data['chose_only_R2'] = ~data['chose_reward1'] & data['chose_reward2']
    data['chose_both'] = data['chose_reward1'] & data['chose_reward2']
    data['chose_neither'] = ~data['chose_reward1'] & ~data['chose_reward2']
    
    y_ticks = [],[]
    for i in [0,1]:
        cue, other_cue = (1, 2) if i == 0 else (2, 1)
        cue_data = data[data['cue'] == cue].copy()
        # sucess case
        cue_data[f'chose_only_R{cue}'] = cue_data[f'chose_only_R{cue}'] *2
        cue_data[f'chose_only_R{other_cue}'] = cue_data[f'chose_only_R{other_cue}'] * -1
        cue_data['chose_both'] = cue_data['chose_both'] * 1
        cue_data['chose_neither'] = cue_data['chose_neither'] * 0
        print(cue_data.iloc[:, -4:])
        cue_data['choice'] = cue_data.iloc[:, -4:].sum(axis=1)
        print(cue_data['choice'])
        
        cue_data['smoothed_choice'] = cue_data['choice'].rolling(mode_span).apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan, raw=False)
        
        y_ticks[0].extend(list(np.array([-1, 0, 1, 2]) +6*i))
        y_ticks[1].extend([f'only R{other_cue}', 'Neither', 'Both', f'only R{cue}'])
        
        
        fig.add_trace(go.Scatter(
            x=cue_data.index.get_level_values('entry_id'),
            y=cue_data['choice'] + 6*i,
            mode='markers',
            name=f'Cue {cue} trials',
            marker=dict(color=cue_data['outcome_color'], size=10),
            legendgroup=f'Cue {cue} trials',
            showlegend=True,
        ))
        
        # Add trace for the black line (Average strategy)
        fig.add_trace(go.Scatter(
            x=cue_data.index.get_level_values('entry_id'),
            y=cue_data['smoothed_choice'] + (6 * i),
            mode='lines',
            name='Average strategy',
            opacity=.4,
            line=dict(color='black', width=5),
            legendgroup=f'Cue {cue} trials',
            showlegend=False,
        ))

        # Add trace for the colored line (Cue-specific Average strategy)
        fig.add_trace(go.Scatter(
            x=cue_data.index.get_level_values('entry_id'),
            y=cue_data['smoothed_choice'] + 13,
            mode='lines',
            name='Average strategy',
            opacity=.4,
            line=dict(color=CUE_COL_MAP[cue], width=5),
            legendgroup='Average strategy',
            showlegend=True,
        ))
        
        y_ticks[0].extend(list(np.array([-1, 0, 1, 2]) +13))
        y_ticks[1].extend([f'only incorrect R', 'Neither', 'Both', f'only correct R'])
        
        
        y_bounds = -1.5, 2.5
        fig.add_shape(
            type='rect',
            xref='x',
            yref='y',
            x0=trials.min()-len(trials)*.005,
            y0=y_bounds[0] +6*i,
            x1=trials.min()-len(trials)*.01,
            y1=y_bounds[1] +6*i,
            fillcolor=CUE_COL_MAP[cue],
            line=dict(width=0),
        )
        
        # add session seperators lines
        for s_id in data.index.unique(level='session_id'):
            s_data = data.loc[pd.IndexSlice[:,:,s_id,:]]
            last_id = s_data.index.get_level_values('entry_id').max()
            
            for j in range(2):
                fig.add_shape(
                    type='line',
                    x0=last_id+.001, 
                    y0=y_bounds[0] +6*i if not j else y_bounds[0] +13,
                    x1=last_id+.001,
                    y1=y_bounds[1] +6*i if not j else y_bounds[1] +13,
                    line=dict(color='rgba(128,128,128,.8)', width=2, dash='dash'),
                )
            # with annotation aligned to the right
            if i == 0:
                fig.add_annotation(
                    x=last_id, 
                    y=y_bounds[0] +6*i,
                    text=f'S{s_id:02d}',
                    showarrow=False,
                    font=dict(size=12, color='rgba(128,128,128,.8)'),
                    xanchor='right',  # Align text horizontally to the right
                )
                
        
        fig.add_annotation(
            x=-.028,  # Position relative to the paper (0 is the left edge, 1 is the right edge)
            y=0.5 + 6 * i,  # Adjust the y position relative to the data
            text='Choice',  # Use HTML tags for bold text
            showarrow=False,
            font=dict(size=12),
            xanchor='right',  # Align text horizontally to the right
            textangle=-90,  # Rotate the text vertically
            xref='paper',  # Position relative to the paper, not the data
            yref='y',  # Position relative to the y-axis
        )
        
        # cue    
        fig.add_annotation(
            x = 0,
            y = 2.8 + 6*i,
            text = f'<b>Cue{cue}</b> trials',
            showarrow = False,
            font = dict(size=12, color=CUE_COL_MAP[cue]),
            xanchor = 'center',
        )
    
        
                
            
        
    # label y axis choice
    
    fig.update_yaxes(tickvals=y_ticks[0], ticktext=y_ticks[1], )#range=[-1.5, 13])
    fig.update_xaxes(title_text='Trial ID',
                    showgrid=True,
                    range=[trials.min()-len(trials)*.01, trials.max()+len(trials)*.01]
    )
        
    # Choice label
    fig.add_annotation(
        x=-.028,  # Position relative to the paper (0 is the left edge, 1 is the right edge)
        y=0.5 + 6 * i,  # Adjust the y position relative to the data
        text='Choice',  # Use HTML tags for bold text
        showarrow=False,
        font=dict(size=12),
        xanchor='right',  # Align text horizontally to the right
        textangle=-90,  # Rotate the text vertically
        xref='paper',  # Position relative to the paper, not the data
        yref='y',  # Position relative to the y-axis
    )
    
    # set background to white
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, zeroline=False),
        autosize=True,
        margin=dict(l=100, r=20, t=50, b=50)  # Adjust margins as needed
    )
    
    # cue1_stayratios = []
    # cue2_stayratios = []
    # drew_dr_legend = False
    # drew_rf_legend = False
    # for s_id in data.index.unique(level='session_id'):
    #     print(s_id)
    #     s_data = data.loc[pd.IndexSlice[:,:,s_id,:]]
    #     print(s_data)
    #     print("=====")
        
    #     # print(s_data['outcome_color'])
        
    #     n_trials = len(s_data)
    #     cmap =  make_discr_trial_cmap(n_trials, TRIAL_COL_MAP)
    #     cmap_transparent = {k: v.replace("rgb","rgba")[:-1]+f', {MULTI_TRACES_ALPHA})' 
    #                         for k,v in cmap.items()}
    #     s_data['trial_color'] = s_data['trial_id'].map(cmap_transparent).fillna('rgba(128,128,128,.14)')
    #     # print(s_data['trial_color'])
        
    #     # session_type = data.loc[pd.IndexSlice[:,:,s_id,:], ["both_R1_R2_rewarded", "flip_Cue1R1_Cue2R2"]]
    #     # session_type = data.loc[pd.IndexSlice[:,:,s_id,:], ["DR", "RF"]].fillna(0.0).astype(bool).all(axis=0)
    #     # if session_type['DR']:
    #     fig.add_scatter3d(
    #                     z=s_data.staytime_reward1, 
    #                     y=s_data.staytime_reward2,
    #                     x=s_data.index.get_level_values('entry_id'),
    #                     mode='markers',
    #                     name='Trial Staytimes',
    #                     line=dict(color='rgba(128,128,128,.14)'),
    #                     marker=dict(color=s_data['outcome_color'], size=6),
    #                     legendgroup='Trial Staytimes',
    #                     showlegend=not drew_dr_legend)
    #     drew_dr_legend = True
        
    #     # add a line with smoothing of the same data as the scatter above
    #     smoothed_data = s_data[['staytime_reward1', 'staytime_reward2']].ewm(span=20, adjust=False).mean()
    #     fig.add_scatter3d(z=smoothed_data.staytime_reward1,
    #                     y=smoothed_data.staytime_reward2,
    #                     x=smoothed_data.index.get_level_values('entry_id'),
    #                     mode='lines',
    #                     line=dict(color='rgba(128,128,128,.14)', width=5),
    #                     showlegend=False)
    #     # draw 2d projection of the line for each plane
    #     fig.add_scatter3d(z=smoothed_data.staytime_reward1,
    #                     y=np.zeros_like(smoothed_data.staytime_reward2),
    #                     x=smoothed_data.index.get_level_values('entry_id'),
    #                     mode='lines',
    #                     legendgroup='Close (R1) Staytime',
    #                     line=dict(color='rgba(128,128,128,.14)', width=10,),
    #                     name='Close (R1) Staytime',
    #                     showlegend=not drew_rf_legend,
    #                     )
    #     fig.add_scatter3d(z=np.zeros_like(smoothed_data.staytime_reward1),
    #                     y=smoothed_data.staytime_reward2,
    #                     x=smoothed_data.index.get_level_values('entry_id'),
    #                     mode='lines',
    #                     line=dict(color='rgba(128,128,128,.14)', width=10,),
    #                     legendgroup='Far (R2) Staytime',
    #                     name='Far (R2) Staytime',
    #                     showlegend=not drew_rf_legend,
    #                     )
        
    #     last_id = s_data.index.get_level_values('entry_id').max()
    #     fig.add_trace(go.Mesh3d(
    #         x=[last_id+.001, last_id+.001, last_id, last_id],
    #         y=[0, 10, 10, 0],
    #         z=[0, 0, 10, 10],
    #         color='rgba(228, 228, 228, .8)',
    #         opacity=.8,
    #         name='Session End',
    #         legendgroup='Session End',
    #         showlegend=not drew_rf_legend,
    #     ))
        
    #     # add line one by one with seperate color of s_data['trial_color']
    #     line_lengths =[2]
    #     for i in range(len(s_data)):
    #         trial_data = s_data.iloc[i:i+2]
    #         line_length = (np.sqrt(np.diff(trial_data[['staytime_reward1', 'staytime_reward2']], axis=0).sum()**2))
    #         alpha = np.clip([(line_length/5)],0,1)[0]
    #         if pd.isna(alpha):
    #             alpha = .1
    #         fig.add_scatter3d(z=trial_data.staytime_reward1, 
    #                         y=trial_data.staytime_reward2,
    #                         x=trial_data.index.get_level_values('entry_id'),
    #                         mode='lines',
    #                         # line=dict(color=trial_data['trial_color'].iloc[0], width=5),
    #                         opacity=alpha,
    #                         line=dict(color=f'rgba(50,50,50,{alpha:.1f})', width=5),
    #                         legendgroup='Trial changes',
    #                         name="Trial changes",
    #                         showlegend=not drew_rf_legend,)
    #         drew_rf_legend = True
            
            
        
    #     # label z axis as trial id
    #     fig.update_layout(scene=dict(xaxis_title='Trial ID', 
    #                                  yaxis_title='Stay time at R2 (s)', 
    #                                  zaxis_title='Stay time at R1 (s)',
    #                                 zaxis_range=[0, metric_max],
    #                                 yaxis_range=[0, metric_max],
    #                                  camera = {
    #                                     "projection": {
    #                                         "type": "orthographic"
    #                                     },
    #                                 },
    #                         aspectmode='manual',
    #                         aspectratio=dict(x=data.shape[0]/30, 
    #                                          y=data.shape[0]/(30*10), 
    #                                          z=data.shape[0]/(30*10)),
    #                         xaxis_showbackground=True,
    #                         yaxis_showbackground=True,
    #                         zaxis_showbackground=True,
    #                         xaxis_backgroundcolor='white',
    #                         yaxis_backgroundcolor=REWARD_LOCATION_COL_MAP[1],
    #                         zaxis_backgroundcolor=REWARD_LOCATION_COL_MAP[2],
                            
    #         )
    #         )
        

    
    # # fig.update_xaxes(range=[0, metric_max], title_text='Stay time at R1 (s)')
    # # fig.update_yaxes(range=[0, metric_max], title_text='Stay time at R2 (s)')
    # # fig.update_layout(scene=dict(xaxis=dict(range=[len(s_data),0], title='Trial ID')))

    # fig.update_layout(
    #     plot_bgcolor='white',
    #     paper_bgcolor='white',
    #     margin=dict(l=50, r=20, t=50, b=50)  # Adjust margins as needed
    # )
    
    return fig