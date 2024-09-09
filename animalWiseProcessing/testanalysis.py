import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../')) # project dir
sys.path.insert(1, os.path.join(sys.path[0], '../../CoreRatVR')) # project dir
sys.path.insert(1, os.path.join(sys.path[0], '../SessionWiseProcessing')) # project dir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from CustomLogger import CustomLogger as Logger
from Parameters import Parameters

from sessionWiseProcessing.sessionLoading import get_session_modality
from sessionWiseProcessing.sessionTransformations import calc_staytimes


def plot(data):
    i = 0
    prv_session_id = 0
    thresholds = []
    plt.figure()
    plt.gca().set_ylim([0, 35])
    for (trial_idx, trial_data) in data.iterrows():
        cue = trial_data.cue
        # correct_region_staytime = trial_data.staytime_r1 if cue==1 else trial_data.staytime_r2
        # incorrect_region_staytime = trial_data.staytime_r1 if cue==2 else trial_data.staytime_r2
        
        if cue == 1:
            correct_region_col = "staytime_r1"
            incorrect_region_col = "staytime_r2"
        elif cue == 2:
            correct_region_col = "staytime_r2"
            incorrect_region_col = "staytime_r1"
        else:
            print("Cue with other then 1,2! :", cue)
            continue
        correct_region_staytime = trial_data.loc[correct_region_col]
        incorrect_region_staytime = trial_data.loc[incorrect_region_col]
        
            
        # correct location
        correct_region_col = 'green' if correct_region_staytime>trial_data.stay_time else 'red'
        plt.scatter(i, correct_region_staytime, color=correct_region_col, s=10, marker='o' if cue == 1 else "v")
        
        # incorrect location
        incorrect_region_col = 'gray'
        plt.scatter(i, incorrect_region_staytime, color=incorrect_region_col, s=10, marker='o' if cue == 1 else "v")
        
        if trial_idx[0] != prv_session_id:
            # Separate x and y values
            x_values, y_values = zip(*thresholds)
            plt.plot(x_values, y_values, linestyle='-', color='k', alpha=.5)
            thresholds.clear()
            i += 10
        
        prv_session_id = trial_idx[0]
        i += 1
        thresholds.append((i,trial_data.stay_time))
    x_values, y_values = zip(*thresholds)
    plt.plot(x_values, y_values, linestyle='-', color='k', alpha=.5)
    
    #average
    mean_staytimes = data.groupby('session_id')[['staytime_r1', 'staytime_r2']].median()
    if (data.loc[:,'cue'] == 1).all():
        x = [12, 40, 70, 105, 135, 168]
        plt.plot(x, mean_staytimes.staytime_r1*1, linestyle='-', color='k')        
        plt.plot(x, mean_staytimes.staytime_r2*1, linestyle='-', color='k', alpha=.5, linewidth=1)
    # else:
    elif (data.loc[:,'cue'] == 2).all():
        x = [5, 30, 55, 85, 110, 133]
        plt.plot(x, mean_staytimes.staytime_r2*1, linestyle='-', color='k')        
        plt.plot(x, mean_staytimes.staytime_r1*1, linestyle='-', color='k', alpha=.5, linewidth=1)
    
    print(mean_staytimes)
    



nas_dir = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning"
animal_id = 8

paradigm_id = 800
paradigm_dir = f"RUN_rYL00{animal_id}/rYL{animal_id:03}_P{paradigm_id:04d}"
data = []
session_dirs = [sd for sd in os.listdir(os.path.join(nas_dir, paradigm_dir)) if sd.endswith("min")]

# for i, session_name in enumerate(sorted(session_dirs)):
#     print(session_name)
#     if not session_name.endswith("min"):
#         continue
    
#     session_dir = os.path.join(paradigm_dir, session_name)
#     trial_data = get_session_modality(from_nas=(nas_dir, session_dir, session_name), 
#                                       modality="unity_trial", complement_trial_data=True,
#                                       rename2oldkeys=True, to_deltaT_from_session_start=True,
#                                       us2s=True)
#     frames = get_session_modality(from_nas=(nas_dir, session_dir, session_name), 
#                                   modality="unity_frame", rename2oldkeys=True, us2s=True)
#     staytimes = calc_staytimes(trial_data, frames)
    
#     session_data = pd.concat((trial_data, staytimes), axis=1)
#     session_data.index = pd.MultiIndex.from_tuples([(i, ID_) for ID_ in session_data.ID]).set_names(["session_id", "trial_id"])
#     session_data = session_data.drop(columns=["ID"])
#     data.append(session_data)
    
# data = pd.concat(data, axis=0)
# data.to_pickle(f"staytimes_rYL{animal_id:03}_P{paradigm_id:04d}.pkl")
data = pd.read_pickle(f"staytimes_rYL{animal_id:03}_P{paradigm_id:04d}.pkl")
# sessionid = 0
# data = data.loc[0:1]
# print(data.iloc[-1])
plot(data.loc[data['cue']==1])
plt.show()
plot(data.loc[data['cue']==2])
plt.show()
plot(data)
plt.show()
    

