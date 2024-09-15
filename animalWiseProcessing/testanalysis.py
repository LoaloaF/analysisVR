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

from sessionWiseProcessing.session_loading import get_session_modality
from sessionWiseProcessing.session_loading import get_session_metadata

from sessionWiseProcessing.session_transformations import calc_staytimes
from sessionWiseProcessing.session_transformations import calc_timeintervals_around_lick
from sessionWiseProcessing.session_transformations import calc_unity_velocity

def trial_wise_staytime(data, sort=True, with_average=False, max_staytime=25, fname_postfix=""):
    session_ids = data.index.get_level_values("session_id").unique()
    plt.figure()
    
    xloc_session_centers = []
    bewteen_session_centers = 20
    xloc = 0
    for session_id in session_ids:
        session_data = data.loc[session_id]
        cue1_only = (session_data['cue'] == 1).all()
        cue2_only = (session_data['cue'] == 2).all()
        
        if cue1_only:
            correct_label = "Time in Early Goal"
            incorrect_label = "Time in Late Goal"
            marker = 'o'
            plt.title("Early Goal Trials")
            
        elif cue2_only:
            correct_label = "Time in Late Goal"
            incorrect_label = "Time in Early Goal"
            marker = 'v'
            plt.title("Late Goal Trials")
            
        else:
            correct_label = ""
            incorrect_label = ""
        
        if sort:
            session_data = session_data.sort_values(by="staytime_correct_r", ascending=False)
            
        markers = session_data['cue'].apply(lambda x: 'o' if x == 1 else 'v').values
        colors = session_data["trial_outcome"].apply(lambda x: 'green' if x > 0 else 'red').values
        # colors = session_data.T.apply(lambda t: "green" if t.staytime_correct_r > t.stay_time else "red" ).values

        x = np.arange(xloc, xloc + len(session_data))
        xloc_session_centers.append(x.mean())
        
        
        for xi, yi, color, marker in zip(x, session_data['staytime_correct_r'], colors, markers):
            plt.scatter(xi, yi/1e6, color=color, s=10, marker=marker, alpha=.5)
        for xi, yi, marker in zip(x, session_data['staytime_incorrect_r'], markers):
            plt.scatter(xi, yi/1e6, color='gray', s=10, marker=marker, alpha=.5)
    
        xloc += len(session_data) + bewteen_session_centers
    
    #legend
    if cue1_only or cue2_only:
        plt.scatter([], [], marker=marker, color='green', label=correct_label)
        plt.scatter([], [], marker=marker, color='red', label=correct_label)
        plt.scatter([], [], marker=marker, color='gray', label=incorrect_label)
        plt.legend()



    if with_average:
        # average time in correct and incorrect goal region
        mean_staytimes = data.groupby('session_id')[['staytime_correct_r', 'staytime_incorrect_r']].median()
        
        plt.plot(xloc_session_centers, mean_staytimes.staytime_correct_r, color='k', alpha=1, linewidth=2, linestyle="-", label=correct_label)
        plt.plot(xloc_session_centers, mean_staytimes.staytime_incorrect_r, color='gray', alpha=1, linewidth=2, linestyle="--", label=incorrect_label)
        plt.legend()
        # alterantive average time in r1, r2    
        # mean_staytimes = data.groupby('session_id')[['staytime_r2', 'staytime_r1']].median()
        # plt.plot(xloc_session_centers, mean_staytimes.staytime_r1, color='k', alpha=1, linewidth=2, linestyle="-")
        # plt.plot(xloc_session_centers, mean_staytimes.staytime_r2, color='gray', alpha=1, linewidth=2, linestyle="--")

    ax = plt.gca()
    ax.set_xticks(xloc_session_centers)
    ax.set_xticklabels([f"Session{session_id+1:02d}" if session_id in [0,5] else "" for session_id in session_ids])
    
    ax.set_ylim([0, max_staytime])
    ax.set_ylabel("Time in goal region [s]")
    
    # Adjust the spines
    ax.spines['left'].set_position(('outward', 10))  # Move left spine outward by 10 points
    ax.spines['bottom'].set_position(('outward', 10))  # Move bottom spine outward by 10 points
    ax.spines['top'].set_visible(False)  # Hide the top spine
    ax.spines['right'].set_visible(False)  # Hide the right spine

    
    

    plt.savefig(f"outputs/animal8_LM/singletrial_staytimes{fname_postfix}.svg")
    


def successrate_plot(data):
    plt.subplots(figsize=(4.5, 5))
    # percentage of correct trials
    for cue in [1, 2]:
        results = data.loc[data['cue'] == cue].groupby('session_id')[["trial_outcome"]].apply(lambda x: (x>0).sum()/len(x))
        plt.plot(results.values, color='#003c00', alpha=.3, 
                 marker='o' if cue == 1 else 'v', 
                 label=f"Eearly Goal Trials" if cue == 1 else "Late Goal Trials")
    results = data.groupby('session_id')[["trial_outcome"]].apply(lambda x: (x>0).sum()/len(x))
    print(results)
    plt.plot(results.values, color='green', )
    plt.legend()
    plt.ylim(0,1)
    
    ax = plt.gca()
    # Adjust the spines
    ax.spines['left'].set_position(('outward', 10))  # Move left spine outward by 10 points
    ax.spines['bottom'].set_position(('outward', 10))  # Move bottom spine outward by 10 points
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_xticklabels(f"Session{xt:02d}" if xt in (results.index[0]+1,results.index[-1]) else "" for xt in ax.get_xticks().astype(int)+1 )
    ax.hlines(0.8, -1, results.index[-1]+1, color='k', linestyle='--')    
    ax.set_xlim(results.index[0]-1, results.index[-1]+1)
    ax.set_ylabel("Success rate")
    
    ax.text(-.6, results.iloc[0], f"{results.iloc[0].item()*100:.0f}%", fontsize=8)
    ax.text(results.index[-1], results.iloc[-1], f"{results.iloc[-1].item()*100:.0f}%", fontsize=8)
    
    plt.savefig("outputs/animal8_LM/successrate_v1.svg")


def compare_lick_velocity(lick_velocities, average_velocity):
    session_ids = lick_velocities.index.get_level_values("session_id").unique()
    plt.figure()

    xloc_session_centers = []
    bewteen_session_centers = 2
    xloc = 0
    for session_id in session_ids:
        lick_vel_label = "velocity during licking" if xloc == 0 else ""
        avg_vel_label = "average velocity" if xloc == 0 else ""
        
        # lick velocities
        lick_velocities_session = lick_velocities.loc[session_id]
        lick_vel_session_avg = lick_velocities_session.groupby("lickgroup_id").median()
        # lick_vel_session_std = lick_velocities_session.groupby("lickgroup_id").std()
        plt.scatter([xloc-.15]*len(lick_vel_session_avg), lick_vel_session_avg.z_velocity, color='k', s=5, alpha=.3)
        plt.scatter(xloc-.15, lick_vel_session_avg.z_velocity.mean(), color='k', s=40, alpha=.8, marker='o', label=lick_vel_label) 
        
        # compare with average velocity
        plt.scatter(xloc+.15, average_velocity.loc[session_id,"average_velocity"], 
                    color='#091bc3', s=40, alpha=1, marker='o', label=avg_vel_label)
        std = average_velocity.loc[session_id,"std_velocity"]
        plt.errorbar(xloc+.15, average_velocity.loc[session_id,"average_velocity"], yerr=std, color='#091bc3', alpha=1)
        
        xloc += 2
    plt.legend()
    
    ax = plt.gca()
    ax.set_ylim([-1, 100])
    
    ax.set_xticklabels([f"Session{session_id:02d}" if session_id in [0,5] else "" for session_id in session_ids])
    ax.set_ylabel("Track Velocity [cm/s]")
    
    ax.spines['left'].set_position(('outward', 10))  # Move left spine outward by 10 points
    ax.spines['bottom'].set_position(('outward', 10))  # Move bottom spine outward by 10 points
    ax.spines['top'].set_visible(False)  # Hide the top spine
    ax.spines['right'].set_visible(False)  # Hide the right spine
    
    plt.savefig("outputs/animal8_LM/velocity_comparison.svg")
    

nas_dir = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning"
animal_id = 8

paradigm_id = 800
paradigm_dir = f"RUN_rYL00{animal_id}/rYL{animal_id:03}_P{paradigm_id:04d}"
data = []
session_dirs = [sd for sd in os.listdir(os.path.join(nas_dir, paradigm_dir)) if sd.endswith("min")]
data2 = []
for i, session_name in enumerate(sorted(session_dirs)):
    print(session_name)
    if not session_name.endswith("min"):
        continue
    
    session_dir = os.path.join(paradigm_dir, session_name)
    
    # # lick calculation, move later TODO
    # lick_data = get_session_modality(from_nas=(nas_dir, session_dir, session_name), 
    #                                   modality="event", us2s=True, event_subset="L")
    # frames = get_session_modality(from_nas=(nas_dir, session_dir, session_name), 
    #                               modality="unity_frame", us2s=True, pct_as_index=True, 
    #                               complement_data=True)
    # frames = frames.loc[frames.cue == 1]
    # intervals = calc_timeintervals_around_lick(lick_data, interval=.4)
    
    # all_lick_vels = []
    # for lickgroup, interval in enumerate(intervals):
    #     lick_velocity = frames.loc[interval.left:interval.right, ["z_velocity"]]
    #     lick_velocity.index = pd.MultiIndex.from_tuples([(i, lickgroup, j) for j in lick_velocity.index], 
    #                                                     names=["session_id","lickgroup_id", "frame_id"])
    #     all_lick_vels.append(lick_velocity)
    # lick_velocities = pd.concat(all_lick_vels)
    # data.append(lick_velocities)
    # data2.append(pd.Series((frames.z_velocity.median(), frames.z_velocity.std()), 
    #                        index=("average_velocity", 'std_velocity'),
    #                        name=i))
    
    #   stay time calculation, TODO move later    
    trial_data = get_session_modality(from_nas=(nas_dir, session_dir, session_name), 
                                      modality="unity_trial", complement_data=True,
                                       to_deltaT_from_session_start=True,
                                      )
    get_session_metadata(from_nas=(nas_dir, session_dir, session_name))
    exit()
    # print(trial_data)
    # print(trial_data.columns)
    # exit()
    # frames = get_session_modality(from_nas=(nas_dir, session_dir, session_name), 
    #                               modality="unity_frame", us2s=True)
    # staytimes = calc_staytimes(trial_data, frames)
    
    # session_data = pd.concat((trial_data, staytimes), axis=1)
    session_data = trial_data
    session_data.index = pd.MultiIndex.from_tuples([(i, ID_) for ID_ in session_data.trial_id]).set_names(["session_id", "trial_id"])
    session_data = session_data.drop(columns=["trial_id"])
    data.append(session_data)

    
# data = pd.concat(data, axis=0)
# # print(data)
# data2 = pd.concat(data2, axis=1).T
# print(data2)
# compare_lick_velocity(data, data2)

data = pd.concat(data, axis=0)
data.to_pickle(f"staytimes_rYL{animal_id:03}_P{paradigm_id:04d}.pkl")
# data = pd.read_pickle(f"staytimes_rYL{animal_id:03}_P{paradigm_id:04d}.pkl")

# trial_wise_staytime(data.loc[data['cue']==1].loc[1:2], fname_postfix="_cue1_S1-2")
# trial_wise_staytime(data.loc[data['cue']==2].loc[1:2], fname_postfix="_cue2_S1-2")
trial_wise_staytime(data.loc[data['cue']==1], with_average=True, fname_postfix="_cue1")
trial_wise_staytime(data.loc[data['cue']==2], with_average=True, fname_postfix="_cue2")
# trial_wise_staytime(data)
successrate_plot(data)
plt.show()
    
