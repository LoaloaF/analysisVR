# import numpy as np
import pandas as pd

from CustomLogger import CustomLogger as Logger
import analytics_processing.analytics_constants as C

def transform_to_position_bin_index(data):
    Logger().logger.debug(f"Transforming {data.shape[0]} unity frames to 1cm "
                          "position bin-index...")
    def proc_trial_data(trial_data):
        trial_id = trial_data['trial_id'].iloc[0]
        if trial_id == -1:
            return
        
        # collapse the unity frames corresponding to the same spatial bin, 1cm in size
        def proc_pos_bin(pos_bin_data):
            # average over all frames in the spatial bin
            mean_cols = ['frame_z_position', 'frame_z_velocity', 'frame_z_acceleration']
            agg_bin_data = pos_bin_data[mean_cols].mean()
            
            # number of frames in the spatial bin
            agg_bin_data['nframes_in_bin'] = pos_bin_data.shape[0]
            
            # if any of the frames in the bin had a reward or lick event
            all_none_cols = ['frame_reward', 'frame_lick']
            # agg_bin_data = pd.concat([agg_bin_data,pos_bin_data[all_none_cols].any()])
            
            # sum the ball velocity components
            ball_vel_cols = ['frame_raw', 'frame_yaw', 'frame_pitch']
            # agg_bin_data = pd.concat([agg_bin_data, pos_bin_data[ball_vel_cols].sum()])
            
            # get the rows corresponding to the start and end of the spatial bin
            end_interval_columns = ['frame_pc_timestamp', 'frame_id', 'frame_ephys_timestamp']
            end_interval_agg = pos_bin_data[end_interval_columns].iloc[-1]
            end_interval_agg.rename({"frame_pc_timestamp": "posbin_to_pc_timestamp",
                                     "frame_ephys_timestamp": "posbin_to_ephys_timestamp",
                                     "frame_id": "posbin_to_frame_id"}, inplace=True)
            agg_bin_data = pd.concat([agg_bin_data, end_interval_agg])
            
            # duplicated for convient access
            identical_cols = [col for col in pos_bin_data.columns 
                              if col not in agg_bin_data.index and
                              col not in C.POSITION_BIN_TABLE_EXCLUDE]
            agg_bin_data = pd.concat([agg_bin_data, pos_bin_data[identical_cols].iloc[0]])
            
            renamer = {col: col.replace('frame_', 'posbin_') for col in agg_bin_data.index}
            renamer.update({'frame_pc_timestamp': 'posbin_from_pc_timestamp',
                            'frame_ephys_timestamp': 'posbin_from_ephys_timestamp',
                            'frame_id': 'posbin_from_frame_id',
                            'posbin_to_frame_id': 'posbin_to_frame_id',})
            agg_bin_data.rename(renamer, inplace=True)
            return agg_bin_data
        # on trial level, group by spatial bin
        return trial_data.groupby(['from_z_position_bin','to_z_position_bin'], 
                                  observed=True).apply(proc_pos_bin)
    # first, group into trials
    posbin_data = data.groupby('trial_id').apply(proc_trial_data)
    
    # posbin_data = posbin_data.astype({'posbin_reward': bool, 'posbin_lick': bool})
    posbin_data.index = posbin_data.index.droplevel(0)
    return posbin_data    

