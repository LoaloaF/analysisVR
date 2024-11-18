# import json
# import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import analysis_core
# from CustomLogger import CustomLogger as Logger

# from Parameters import Parameters
# from sessionWiseProcessing.session_loading import get_session_modality

from data_loading.session_loading import get_session_modality
from data_loading.animal_loading import get_animal_modality
from data_loading.paradigm_loading import get_paradigm_modality

# from sessionWiseProcessing.session_loading import get_session_metadata

# from sessionWiseProcessing.session_transformations import calc_staytimes
# from sessionWiseProcessing.session_transformations import calc_timeintervals_around_lick
# from sessionWiseProcessing.session_transformations import calc_unity_velocity

def trial_wise_staytime(data, sort=True, with_average=False, max_staytime=25, fname_postfix=""):
    session_ids = data.index.get_level_values("session_id").unique()
    plt.figure()

    xloc_session_centers = []
    bewteen_session_centers = 20
    xloc = 0
    for session_id in session_ids:
        # session_data = data.loc[session_id]
        session_data = data.xs(session_id, level="session_id")

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
    print(data.loc[pd.IndexSlice[:,1:1,:]])
    # exit()
    # for cue in [1, 2]:
    #     print(f"\nProcessing cue {cue}:")
    #     cue_data = data.loc[data['cue'] == cue]
    #     print("Session IDs for this cue:", cue_data.index.unique('session_id'))
        
    #     # cue_results = cue_data.groupby('session_id')[["trial_outcome"]].apply(lambda x: (x > 0).sum() / len(x))
    #     cue_results = cue_data.groupby('session_id')[["staytime_correct_r"]].median()
    #     othercue_results = cue_data.groupby('session_id')[["staytime_incorrect_r"]].median()
    #     print("Results for this cue:")
    #     print(cue_results)
    for cue in [1, 2]:
        # print(data.loc[data['cue'] == cue])
        # data.loc[data['cue'] == cue].groupby('session_id')[["trial_outcome"]].apply(lambda x: print(x))
        # data.loc[data['cue'] == cue].groupby('session_id')[["trial_outcome"]].apply(lambda x: print(x))
        # data.loc[data['cue'] == cue].groupby('session_id')[["trial_outcome"]].apply(lambda x: print((x.index.get_level_values("session_id"))))
        # data.loc[data['cue'] == cue].groupby('session_id')[["trial_outcome"]].apply(lambda x: print((x.index.get_level_values("session_id"))))
        results = data.loc[data['cue'] == cue].groupby('session_id')[["trial_outcome"]].apply(lambda x: (x>0).sum()/len(x))
        # results = data.loc[data['cue'] == cue].groupby('session_id')[["staytime_correct_r"]].median()
        print(results)
        # offresults = data.loc[data['cue'] == cue].groupby('session_id')[["staytime_incorrect_r"]].median()
        plt.plot(results.values, color='#003c00', alpha=.1, 
                 marker='o' if cue == 1 else 'v', 
                 label=f"Eearly Goal Trials" if cue == 1 else "Late Goal Trials")
        # plt.plot(offresults.values, color='red', alpha=.1, 
        #          marker='o' if cue == 1 else 'v', 
        #          label=f"Eearly Goal Trials" if cue == 1 else "Late Goal Trials")
    results = data.groupby('session_id')[["trial_outcome"]].apply(lambda x: (x>0).sum()/len(x))
    # results = data.groupby('session_id')[["staytime_correct_r"]].median()
    plt.plot(results.values, color='green', )
    results = data.groupby('session_id')[["staytime_incorrect_r"]].median()
    plt.plot(results.values, color='red', )
    plt.legend()
    # plt.ylim(0,1)
    
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

def trial_length_plot(data):
    pass

def corr_plot(data):
    plt.figure()
    # Calculate the correlation matrix between columns
    
    
    # Replace NaN values with zeros
    # corr = corr.fillna(0)

    idx = ['trial_id', 'trial_outcome', 'trial_pc_duration',
            'staytime_cue1',
            'staytime_cue1_passed', 
            'staytime_between_cues', 
            'staytime_cue2_visible', 
            'staytime_cue2', 
            'staytime_cue2_passed',
            'staytime_before_reward1', 
            'staytime_reward1', 
            'staytime_before_reward2',
            'staytime_reward2',
            'staytime_post_reward',
            'staytime_incorrect_r', 
            'staytime_correct_r', 
            ]
    data = data.loc[:, idx]
    corr = data.corr()
        
    corr = corr.reindex(idx, axis=0)
    corr = corr.reindex(idx, axis=1)
    
    # Plot the correlation matrix
    plt.imshow(corr.values, cmap='coolwarm', interpolation='none', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Correlation Matrix')
    plt.gca().set_yticks(np.arange(len(corr.columns)))
    plt.gca().set_xticks(np.arange(len(corr.columns)))
    plt.gca().set_yticklabels(corr.columns, fontsize=8)
    plt.gca().set_xticklabels(corr.index, rotation=80, fontsize=8)
    # plt.show()
    





























analysis_core.init_analysis("DEBUG")

# args: skip_animals, from_date, to_date
paradigm_parsing_kwargs={"skip_animals":[1,2,3,4,5,6,7,8]}
# args: skip_sessions, from_date, to_date
animal_parsing_kwargs={"skip_sessions":["2024-07-29_16-58_rYL001_P0800_LinearTrack_41min"], # bell vel doesn't match up with unity frames
                       }#"from_date": "2024-07-29", "to_date": "2024-08-03",}
# args: to_deltaT_from_session_start pct_as_index us2s event_subset na2null
# complement_data position_bin_index rename2oldkeys
session_parsing_kwargs={"complement_data":True, "to_deltaT_from_session_start":True, 
                        "position_bin_index": True}
# args: where start stop columns
modality_parsing_kwargs={"columns": ['trial_id', 'cue', 'trial_outcome', 'trial_pc_duration']}

data = get_paradigm_modality(paradigm_id=1100, modality='unity_trial', cache='to', 
                             **paradigm_parsing_kwargs,
                             animal_parsing_kwargs=animal_parsing_kwargs,
                             session_parsing_kwargs=session_parsing_kwargs,
                           )

trial_wise_staytime(data.loc[data['cue']==1], with_average=True, fname_postfix="_cue1")
trial_wise_staytime(data.loc[data['cue']==2], with_average=True, fname_postfix="_cue2")
successrate_plot(data)

trial_length_plot(data)
corr_plot(data.loc[data['cue']==1])
# corr_plot(data.loc[data['cue']==1])
# corr_plot(data.loc[data['cue']==2])
plt.show()