import pandas as pd
import numpy as np
from CustomLogger import CustomLogger as Logger

import analytics_processing.analytics_constants as C

def get_UnityTrackwise(unity_tramewise):
    return transform_to_position_bin_index(unity_tramewise)

def get_UnityTrialwiseMetrics(unity_framewise):
    return pd.DataFrame([])
    # raise NotImplementedError
    # return unity_framewise.groupby('trial_id').first()

def merge_behavior_events_with_frames(unity_framewise, behavior_events):
    # exclude the ball velocity events
    behavior_events = behavior_events[behavior_events["event_name"] != "B"]
    
    # decide which timestamp to use
    if unity_framewise["frame_ephys_timestamp"].isna().any():
        frame_time_col = "frame_pc_timestamp"
        behavior_time_col = "event_pc_timestamp"
    else:
        frame_time_col = "frame_ephys_timestamp"
        behavior_time_col = "event_ephys_timestamp"
    
    # get the unique event types
    event_types = behavior_events["event_name"].unique()
    
    for each_event in event_types:
        certain_event = behavior_events[behavior_events["event_name"] == each_event]
        frame_timestamps = unity_framewise[frame_time_col].values
        event_timestamps = certain_event[behavior_time_col].values

        # get the closest frame to each event
        abs_diff_event = np.abs(frame_timestamps[:, np.newaxis] - event_timestamps)
        closest_unity_indices = np.argmin(abs_diff_event, axis=0)
        counts = np.bincount(closest_unity_indices)

        unity_framewise[f'{each_event}_count'] = 0
        for index, count in enumerate(counts):
            if index < len(unity_framewise):
                unity_framewise.at[index, f'{each_event}_count'] = count

    return unity_framewise

def transform_to_position_bin_index(data):
    Logger().logger.debug(f"Transforming {data.shape[0]} unity frames to 1cm "
                          "position bin-index...")
    def interp_missing_pos_bins(data):
        # reindex to the full range of spatial bins, then fill NaNs
        from_bin, to_bin = data.index[0][0], data.index[-1][1]
        from_z_position_bin = np.arange(from_bin, to_bin)
        to_z_position_bin = np.arange(from_bin+1, to_bin+1)
        midx = pd.MultiIndex.from_arrays([from_z_position_bin, to_z_position_bin],
                                         names=['from_z_position_bin', 'to_z_position_bin'])
        data = data.reindex(midx)
        
        # convert to pandas <NA> type
        data = data.convert_dtypes()
        # what identifies this is an interpolated observation, 0 frames in bin
        data.loc[:, "nframes_in_bin"] = data.loc[:, "nframes_in_bin"].fillna(0)
        
        # fill the index columns
        data['from_z_position_bin'] = from_z_position_bin
        data['to_z_position_bin'] = to_z_position_bin

        # Interpolate numeric columns, kinematics 
        kinematic_cols = ["posbin_z_position", "posbin_z_velocity", "posbin_z_acceleration",]
        data.loc[:,kinematic_cols] = data.loc[:,kinematic_cols].interpolate(method='linear', 
                                                                            limit_direction='both')
        # and timestamps
        time_cols = ["posbin_to_pc_timestamp", "posbin_from_pc_timestamp",
            "posbin_from_ephys_timestamp", "posbin_to_ephys_timestamp"]
        for col in time_cols:
            if not data.loc[:, col].isna().all():
                # integers may be cast to string if not converted to datetime
                dt_time = pd.to_datetime(data.loc[:,col], unit='us')
                data.loc[:,col] = dt_time.interpolate(method='linear', 
                                                      limit_direction='both').astype('int64')
        
        const_cols = [
            # these columns are constant within a trial
            "maximum_reward_number", "stay_time", "cue", "lick_reward", "trial_id",
            "trial_start_frame", "trial_start_pc_timestamp", "trial_end_frame",
            "trial_end_pc_timestamp", "trial_pc_duration", "trial_outcome",
            "trial_start_ephys_timestamp", "trial_end_ephys_timestamp",
            # these columns change slowly and can be forward filled (although not 100% to be trusted)
            "zone", "posbin_state",  
        ]
        data.loc[:,const_cols] = data.loc[:,const_cols].ffill().bfill()
        
        # fill the events with zeros
        event_cols = ["L_count", "R_count", "S_count", "V_count"]
        data.loc[:,event_cols] = data.loc[:,event_cols].fillna(0)
        return data
        
    def proc_trial_data(trial_data):
        trial_id = trial_data['trial_id'].iloc[0]
        if trial_id == -1:
            return

        # collapse the unity frames corresponding to the same spatial bin, 1cm in size
        def proc_pos_bin(pos_bin_data):
            # average over all frames in the spatial bin
            mean_cols = ['frame_z_position', 'frame_z_velocity', 'frame_z_acceleration',
                         'L_count', 'R_count', 'S_count', 'V_count']
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
        bin_trial_data = trial_data.groupby(['from_z_position_bin','to_z_position_bin'], 
                                            observed=True).apply(proc_pos_bin)        
        # print(bin_trial_data)
        bin_trial_data = interp_missing_pos_bins(bin_trial_data)
        return bin_trial_data

    # first, group into trials
    posbin_data = data.groupby('trial_id').apply(proc_trial_data)
    
    # posbin_data = posbin_data.astype({'posbin_reward': bool, 'posbin_lick': bool})
    posbin_data.index = posbin_data.index.droplevel(0)
    return posbin_data

