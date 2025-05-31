import plotly.graph_objects as go
import json

from sklearn.metrics import f1_score, balanced_accuracy_score
import pandas as pd
import numpy as np
from dash import dcc, html
from plotly.subplots import make_subplots

from .general_plot_helpers import draw_track_illustration

def render_plot(svm_output, metadata, model, use_subset, which_metric):
    predictor_cmap = {
        'HP': 'orange',
        'mPFC': 'red',
        'behavior': 'lightblue',
        'HP-mPFC': 'yellow',
    }
    #TODO remove this hardcoded
    zones = {
        'beforeCueZone': (-168, -100),
        'cueZone': (-80, 25),
        'afterCueZone': (25, 50),
        'reward1Zone': (50, 110),
        'betweenRewardsZone': (110, 170),
        'reward2Zone': (170, 230),
        'postRewardZone': (230, 260),
    }
    
    if use_subset:
        # recalcuclcate the f1 and balanced accuracy
        print(svm_output)
        svm_output.drop(columns=['f1', 'acc'], inplace=True)
        new_metrics = svm_output.groupby(['predicting', 'predictor', 'model', 'track_zone']).apply(
            lambda x: pd.Series({
                    "f1": f1_score(x.y, x.y_true, average='macro'), 
                    "acc": balanced_accuracy_score(x.y, x.y_true),
                    
            })
        )
        
        svm_output.groupby(['predicting', 'predictor', 'model', 'track_zone']).apply(
            lambda x: print((x.y == x.y_true).values.sum()/ x.y.shape[0])
        )
        
        print(new_metrics)
        svm_output = svm_output.merge(new_metrics, on=['predicting', 'predictor', 'model', 'track_zone'])
        print(svm_output)
        print(model)
    

    fig = make_subplots(rows=8, cols=1, row_heights=[.3, .14, .3, .14, .3, .14, .3, .14],
                        shared_xaxes=True,
                        vertical_spacing=0.01, )
    
    min_track, max_track = -169, 260
    for row in (2,4,6,8):
        draw_track_illustration(fig, row=row, col=1,  track_details=json.loads(metadata.iloc[0]['track_details']), 
                                min_track=min_track, max_track=max_track, choice_str='Stop', draw_cues=[2], 
                                double_rewards=False)
        
    for i, predicting in enumerate(svm_output.predicting.unique()):
        row = i*2 + 1
        svm_outp = svm_output[svm_output['predicting'] == predicting]
        # print(svm_outp)
        
        # line at 0.5
        fig.add_trace(go.Scatter(
            x=[min_track, max_track],
            y=[0.5, 0.5],
            mode='lines',
            line=dict(color='black', width=1, dash='dash'),
            name='0.5',
            showlegend=False, 
        ), row=row, col=1)
        
        for predictor in svm_outp['predictor'].unique():
            print(predictor)
            # every row is a trial now, model was fit on this set
            # all model metrics are in the same every row
            pred = svm_outp[(svm_outp['predictor'] == predictor) & 
                            (svm_outp['model'] == model)]
            if pred.empty:
                print(f"Warning: No predictions for {predictor} in {predicting}!")
                continue
                
            zone_accs, zone_stds, zone_x = [], [], []
            for zone, (start, end) in zones.items():        
                zone_x.append(start + (end-start)/2)
                
                # draw zone indicator 4x, TODO fix
                fig.add_trace(go.Scatter(
                    x=[start+3, end-3],
                    y=[1, 1],
                    mode='lines',
                    opacity=0.1,
                    line=dict(color='black', width=10),
                    name=f'Prediction {i+1}',
                    showlegend=False, 
                ), row=row, col=1)
        
                zone_accs.append(pred[pred.track_zone==zone][which_metric].values[0])
                zone_stds.append(pred[pred.track_zone==zone].acc_std.values[0])
            

            # draw the SVM prediction acc + - std
            fig.add_trace(go.Scatter(
                x=zone_x,
                y=zone_accs,
                mode='markers+lines',
                marker=dict(color=predictor_cmap[predictor], size=10),
                line=dict(color=predictor_cmap[predictor], width=2),
                name=f'{predictor}',
                legendgroup=f'{predictor}',
                showlegend=True,
            ), row=row, col=1)
            
            # std
            for m, std, x in zip(zone_accs, zone_stds, zone_x):
                fig.add_trace(go.Scatter(
                    x=[x, x],
                    y=[m-std, m+std],
                    mode='lines',
                    line=dict(color=predictor_cmap[predictor], width=.5),
                    name=f'{predictor} std',
                    legendgroup=f'{predictor}',
                    showlegend=False,
                ), row=row, col=1)
        
        
            
        fig.update_yaxes(title_text=f"accur. predict {predicting}", row=row, col=1,
                         range=(0.35, 1), showticklabels=True, ticks="outside",)
    
    
    fig.layout.update(
        height=800,
        width=800,
        margin=dict(l=0, r=0, t=0, b=0),
        # showlegend=False,
        
    )
    return fig