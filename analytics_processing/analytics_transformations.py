import pandas as pd
import numpy as np

from CustomLogger import CustomLogger as Logger
import analytics_processing.analytics_constants as C

def async_events_to_framewise(events, which_t_col, track_kinematics):
    # assign each event to a unity frame
    frame_intervals = pd.IntervalIndex.from_arrays(
        track_kinematics['frame'+which_t_col][:-1].values, 
        track_kinematics['frame'+which_t_col][1:].values,
        closed='left')
    # use the intervals to assign events to frames
    assignm = frame_intervals.get_indexer(events['event'+which_t_col].values)
    assigned_frame_ids = track_kinematics['frame_id'].iloc[assignm]
    assigned_frame_ids[assignm == -1] = np.nan  # events outside unity frames NA

    if (assignm == -1).any():
        Logger().logger.warning(f"{(assignm == -1).sum()} events are not "
                                f"within unity frame intervals")
    events['frame_id'] = assigned_frame_ids.values
    
    # count the events per frame, and their mean value within it
    fr_wise_events = events.groupby(['frame_id', 'event_name_full']).apply(
                                    lambda x: pd.Series({f'{x.event_name_full.iloc[0]}_count': x.shape[0],
                                                         f'{x.event_name_full.iloc[0]}_mean_value': x['event_value'].mean(),}))
    fr_wise_events.index = fr_wise_events.index.droplevel(1)  # drop the event_name_full level

    fr_wise_events = fr_wise_events.unstack(level=1) # index from groupby output
    # put the frame_id from index back to column
    fr_wise_events.reset_index(inplace=True)
    return fr_wise_events

def transform_to_position_bin_index(data):
    Logger().logger.debug(f"Transforming {data.shape[0]} unity frames to 1cm "
                          "position bin-index...")
    def interp_missing_pos_bins(data):
        # reindex to the full range of spatial bins, then fill NaNs
        from_bin, to_bin = data.index[0][0], data.index[-1][1]
        from_position_bin = np.arange(from_bin, to_bin)
        to_position_bin = np.arange(from_bin+1, to_bin+1)
        midx = pd.MultiIndex.from_arrays([from_position_bin, to_position_bin],
                                         names=['from_position_bin', 'to_position_bin'])
        data = data.reindex(midx)
        
        # convert to pandas <NA> type
        data = data.convert_dtypes()
        # what identifies this is an interpolated observation, 0 frames in bin
        data.loc[:, "nframes_in_bin"] = data.loc[:, "nframes_in_bin"].fillna(0)
        
        # fill the index columns
        data['from_position_bin'] = from_position_bin
        data['to_position_bin'] = to_position_bin

        # Interpolate numeric columns, kinematics 
        kinematic_cols = ["posbin_position", "posbin_velocity", "posbin_acceleration",]
        data.loc[:,kinematic_cols] = data.loc[:,kinematic_cols].interpolate(method='linear', 
                                                                            limit_direction='both')
        
        # Interpolate facecam pose columns
        facecam_pose_cols = [col for col in data.columns if col.startswith('facecam_pose_')]
        data.loc[:,facecam_pose_cols] = data.loc[:,facecam_pose_cols].interpolate(method='linear', 
                                                                                  limit_direction='both')

        # and timestamps
        time_cols = ["posbin_to_pc_timestamp", "posbin_from_pc_timestamp",
                     "posbin_from_ephys_timestamp", "posbin_to_ephys_timestamp", ]
        for col in time_cols:
            if not data.loc[:, col].isna().all():
                # integers may be cast to string if not converted to datetime
                dt_time = pd.to_datetime(data.loc[:,col], unit='us')
                data.loc[:,col] = dt_time.interpolate(method='linear', 
                                                      limit_direction='both').astype('int64')
        
        # fill the events with zeros
        count_cols = np.array(["lick_count", "reward-valve-open_count", 
                                      "reward-sound_count", "reward-removed_count"])
        # TODO track_kinematics should have always all event columns
        # in case an event is not present in the entire ssesion 
        count_cols = count_cols[np.isin(count_cols, data.columns)]
        data.loc[:,count_cols] = data.loc[:,count_cols].fillna(0)
        
        # const_cols = [c if c not in [] for c in data.columns]
        
        # const_cols = [
        #     # these columns are constant within a trial
        #     "maximum_reward_number", "stay_time", "cue", "lick_reward", "trial_id",
        #     "trial_start_frame", "trial_start_pc_timestamp", "trial_end_frame",
        #     "trial_end_pc_timestamp", "trial_pc_duration", "trial_outcome",
        #     "trial_start_ephys_timestamp", "trial_end_ephys_timestamp",
        #     # these columns change slowly and can be forward filled (although not 100% to be trusted)
        #     "track_zone", "posbin_state",  
        # ]
        data.loc[:,:] = data.loc[:,:].ffill().bfill()
        
        return data
        
    def proc_trial_data(trial_data):
        trial_id = trial_data['trial_id'].iloc[0]
        print(f"cm-binning trial {trial_id} with {trial_data.shape[0]} frames...", end='\r')
        if trial_id == -1:
            return

        # collapse the unity frames corresponding to the same spatial bin, 1cm in size
        def proc_pos_bin(pos_bin_data):
            
            #TODO are they being renamed to posbin_?
            # average over all frames in the spatial bin
            mean_cols = ['frame_position', 'frame_velocity', 'frame_acceleration',
                         'frame_raw', 'frame_yaw', 'frame_pitch']
            
            # add all the facecam pose columns
            facecam_pose_cols = [col for col in pos_bin_data.columns 
                                 if col.startswith('facecam_pose_')]
            mean_cols.extend(facecam_pose_cols)

            agg_bin_data = pos_bin_data[mean_cols].mean()
            
            # number of frames in the spatial bin
            agg_bin_data['nframes_in_bin'] = pos_bin_data.shape[0]
            
            # if any of the frames in the bin had a reward or lick event
            all_none_cols = ['frame_reward', 'frame_lick']
            # agg_bin_data = pd.concat([agg_bin_data,pos_bin_data[all_none_cols].any()])
            
            # sum the count columns
            count_cols = np.array(["lick_count", "reward-valve-open_count", 
                                      "reward-sound_count", "reward-removed_count"])
            # TODO track_kinematics should have always all event columns
            # in case an event is not present in the entire ssesion 
            count_cols = count_cols[np.isin(count_cols, pos_bin_data.columns)]
            agg_bin_data = pd.concat([agg_bin_data, pos_bin_data[count_cols].sum()])
            
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
        bin_trial_data = trial_data.groupby(['from_cm_position_bin','to_cm_position_bin'], 
                                            observed=True).apply(proc_pos_bin)        
        
        bin_trial_data = interp_missing_pos_bins(bin_trial_data)
        return bin_trial_data

    # first, group into trials
    print("Processing position bin:")
    posbin_data = data.groupby('trial_id').apply(proc_trial_data)
    
    # posbin_data = posbin_data.astype({'posbin_reward': bool, 'posbin_lick': bool})
    posbin_data.index = posbin_data.index.droplevel(0)
    return posbin_data

