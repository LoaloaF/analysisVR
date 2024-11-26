import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../')) # project dir
sys.path.insert(1, os.path.join(sys.path[0], '../../CoreRatVR')) # project dir
sys.path.insert(1, os.path.join(sys.path[0], '../SessionWiseProcessing')) # project dir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from CustomLogger import CustomLogger as Logger

# from sessionWiseProcessing.session_loading import get_session_modality
import analysis_core
from data_loading.animal_loading import get_animal_modality
from data_loading.paradigm_loading import get_paradigm_modality

def plot_track_velocity(data):
   # print(data)
   # exit()
   # spatial_averages = []   
   for trial_id in data.index.get_level_values("trial_id").unique():
      # print(trial_id)
      print(data)
      trial_data = data.loc[pd.IndexSlice[:, :, trial_id]]#.rolling(window=5).mean()
      # trial_data.loc[:,'posbin_pc_timestamp'] -= trial_data.loc[:,'posbin_pc_timestamp'].iloc[0]
      # print(trial_data)
      # plt.plot(trial_data['posbin_pc_timestamp'], trial_data['posbin_z_position'])
      # print(data2[data2.trial_id==trial_id])
      col = 'r' if trial_data['trial_outcome'].iloc[0] < 1 else 'g'
      plt.plot(trial_data["posbin_z_position"], trial_data["posbin_z_velocity"].rolling(5).mean(), alpha=0.1, linewidth=1, color=col)
      
      # d = gaussian_filter(trial_data['z_velocity'], sigma=20, mode='reflect')
      # plt.plot(trial_data["posbin_pc_timestamp"], trial_data['z_velocity'], alpha=0.2, linewidth=1, color=col)
      # trial_data['z_acceleration'] = np.gradient(trial_data['z_velocity'], trial_data['posbin_pc_timestamp'])
      
      r1_start = trial_data[trial_data['zone']=='reward1'].iloc[0]
      r1_stop = trial_data[trial_data['zone']=='reward1'].iloc[-1]
      r2_start = trial_data[trial_data['zone']=='reward2'].iloc[0]
      r2_stop = trial_data[trial_data['zone']=='reward2'].iloc[-1]
      cue2_start = trial_data[trial_data['zone']=='cue2'].iloc[0]
      cue2_stop = trial_data[trial_data['zone']=='cue2'].iloc[-1]
      if trial_id == 1:
         plt.axvline(x=r1_start['posbin_z_position'], color='r', linestyle='--')
         plt.axvline(x=r1_stop['posbin_z_position'], color='r', linestyle='--')
         plt.axvline(x=r2_start['posbin_z_position'], color='g', linestyle='--')
         plt.axvline(x=r2_stop['posbin_z_position'], color='g', linestyle='--')
         plt.axvline(x=cue2_start['posbin_z_position'], color='gray', linestyle='--')
         plt.axvline(x=cue2_stop['posbin_z_position'], color='gray', linestyle='--')
         plt.axhline(y=0)
         
         # print(r1_start, r1_stop)
         # trial_data['zone'].iloc[0]
         
         # print(trial_data)

      # trial_data['z_acceleration'] -= r1_start['z_acceleration'] 

      
      # d = trial_data['z_velocity']#.rolling(40).mean()
      # d2 = np.gradient(d, trial_data['posbin_pc_timestamp'])
      # d2
      # d -= r1_start['z_velocity']
      # plt.plot(trial_data["posbin_z_position"], d, alpha=0.17, linewidth=1, color='b')
      # print(trial_data)
      # spatial_average = (trial_data[['binned_pos', 'z_velocity', 'z_acceleration']].groupby('binned_pos').mean())
      # plt.plot(trial_data["posbin_z_position"], trial_data["z_velocity"], alpha=0.17, linewidth=1, color='b')
      # plt.plot(spatial_average.index.mid, spatial_average['z_velocity'], alpha=0.17, linewidth=1, color='r')
      # spatial_averages.append(spatial_average)
      
      # if trial_id > 5:
      #    break

   # remove category columns
   print(data)
   data = data.drop(columns=['zone', 'cue',])
   print(data)
   fail_trial_data = data.loc[data.trial_outcome < 1].groupby("binned_pos").mean()
   success_trial_data = data.loc[data.trial_outcome >= 1].groupby("binned_pos").mean()
   print(fail_trial_data)
   # print(fail_trial_data)
   plt.plot(fail_trial_data["posbin_z_position"], fail_trial_data["posbin_z_velocity"].rolling(20).mean(), alpha=0.7, linewidth=1, color='r')
   plt.plot(success_trial_data["posbin_z_position"], success_trial_data["posbin_z_velocity"].rolling(20).mean(), alpha=0.7, linewidth=1, color='g')
   
   # plt.plot(trial_data["posbin_z_position"], trial_data["posbin_z_velocity"].rolling(20).mean(), alpha=0.7, linewidth=1, color=col)
      
   # spatial_averages = pd.concat(spatial_averages, axis=0)
   # # print(spatial_averages)
   # mean_data = spatial_averages.groupby("binned_pos").mean()      
   # # mean_data = data[['z_velocity','binned_pos', 'z_velocity', 'cue']].groupby("binned_pos").mean()
   # # mean_data = data[['z_velocity','binned_pos', 'z_velocity', 'cue']].groupby("trial_id").mean()
   # print(mean_data)
   # plt.plot(mean_data.index.mid, mean_data['z_acceleration'].rolling(20).mean(), color='r', linewidth=1, alpha=1,)
         
   # mean_data = data[['z_velocity','binned_pos', 'cue']].groupby("binned_pos").mean()
   
   # counts = data["binned_pos"].value_counts()
   # index = counts.index.sort_values()
   # counts = counts.loc[index]
   # print(index)
   # print(counts)
   # plt.plot(index.mid, counts)
   
   # mean_data_cue1 = mean_data.loc[mean_data['cue'] == 1]   
   # mean_data_cue2 = mean_data.loc[mean_data['cue'] == 2]   
   # plt.plot(mean_data_cue1.index.mid, mean_data_cue1['z_velocity'], color='red', linewidth=2, alpha=.5, linestyle='--')
   # plt.plot(mean_data_cue2.index.mid, mean_data_cue2['z_velocity'], color='green', linewidth=2, alpha=.5, linestyle='--')

def plot_track_licks(data):
   sessions = data.index.get_level_values("session_id").unique()
   print(data)
   # print(sessions)
   
   for session in sessions:
      if session<15:
         continue
      # print(session)
      session_data = data.loc[pd.IndexSlice[:, session, :], :]
      # continue
      
      for cue in (1,2):
         cum = session_data[session_data.cue == cue]
         c = 'r' if cue == 1 else 'g'
      
         cum = cum.loc[:,['posbin_z_position', 'posbin_lick', ]].groupby("posbin_z_position").sum()
         
         # bin the data from acm intervals to 10cm intervals
         cum = cum.groupby(pd.cut(cum.index, np.arange(cum.index[0], cum.index[-1], 5))).sum()
         
         plt.plot(cum.index.categories.mid, cum['posbin_lick'], alpha=min(0.1*session,1), label=f"session {session}", color=c)
   plt.legend()

def plot_lick_vel(data):
   for session_id in data.index.get_level_values("session_id").unique():
      session_data = data.loc[pd.IndexSlice[:, session_id, :], :]
      vels = session_data[session_data.posbin_lick.astype(bool)].posbin_raw.values
      plt.boxplot(vels, positions=[session_id])
   






















analysis_core.init_analysis("DEBUG")

# args: skip_animals, from_date, to_date
paradigm_parsing_kwargs={}#"skip_animals":[1,2,4,5,6,7,8]}
# args: skip_sessions, from_date, to_date
animal_parsing_kwargs={"skip_sessions":["2024-07-29_16-58_rYL001_P0800_LinearTrack_41min", # bell vel doesn't match up with unity frames
                                        "2024-07-26_14-57_rYL003_P0800_LinearTrack_27min", # trial_id 1-5 duplicated
                                        "2024-08-09_18-20_rYL003_P0800_LinearTrack_21min", # has only 1 trial
]}
# args: to_deltaT_from_session_start pct_as_index us2s event_subset na2null
# complement_data position_bin_index rename2oldkeys
session_parsing_kwargs={"complement_data":True, "to_deltaT_from_session_start":True, 
                        "position_bin_index": True}
# args: where start stop columns
# modality_parsing_kwargs={"columns": ['trial_id', 'cue', 'trial_outcome', 'trial_pc_duration']}

data = get_paradigm_modality(paradigm_id=800, modality='unity_frame', cache='to', 
                             **paradigm_parsing_kwargs,
                             animal_parsing_kwargs=animal_parsing_kwargs,
                             session_parsing_kwargs=session_parsing_kwargs,
                           )

data = data.loc[800]

# plot_track_velocity(data)
# plot_lick_vel(data)
plt.show()