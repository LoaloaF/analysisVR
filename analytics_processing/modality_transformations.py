import numpy as np
import pandas as pd

from CustomLogger import CustomLogger as Logger

# =============================================================================
# all-modalities transformations
# =============================================================================

def data_modality_na2null(data):
    # Convert float64 columns to object type
    for col in data.select_dtypes(include=['float64']).columns:
        #TODO do this conversion only if there are na values in that column
        data[col] = data[col].astype(object)
    # Fill NaN values with "null"
    data.fillna("null", inplace=True)
    return data

def data_modality_pct_as_index(data):
    pct_col = [col for col in data.columns if col.endswith("_pc_timestamp")][0]
    if data[pct_col].is_unique:
        data.set_index(data[pct_col], inplace=True, drop=True)
    else:
        n = data.shape[0]
        data = data.drop_duplicates(subset=[pct_col])
        Logger().logger.warning(f"Non-unique index values found in the "
                        f"timestamp column. Before {n} rows, after "
                        f"{data.shape[0]}, diff {n-data.shape[0]}")
    data.set_index(pd.to_datetime(data[pct_col], unit='us'), 
                    inplace=True, drop=True)
    return data

def data_modality_rename2oldkeys(data, key):
    #TODO doesn't have all modalities
    if key == "unity_frame":
        rename_dict = {
            
            'frame_id': 'ID',
            'frame_pc_timestamp': 'PCT',
            'frame_x_position': 'X',
            'frame_z_position': 'Z',
            'frame_angle': 'A',
            'frame_state': 'S' ,
            'frame_blinker': 'FB',
            'ballvelocity_first_package': 'BFP',
            'ballvelocity_last_package': 'BLP',
        }
        data['N'] = "U" # add back the name "U", indicating Unity frame
        
    elif key == 'paradigm_variable':
        rename_dict = {
            'trial_id': 'ID',
            'cue': 'C',
            'maximum_reward_number': 'MRN',
            'stay_time': 'ST',
            'lick_reward': 'LR',
        }
        
    elif key == "unity_trial":
        rename_dict = {
            "trial_id": "ID", 
            "trial_start_frame": "SFID", 
            "trial_start_pc_timestamp": "SPCT", 
            "trial_end_frame": "EFID", 
            "trial_end_pc_timestamp": "EPCT",
            "trial_pc_duration": "TD", 
            "trial_outcome": "O"}
    
    elif key.endswith("cam_packages"):
        cam_name = key.split("_")[0]
        rename_dict = {
            f'{cam_name}_image_id': "ID", 
            f'{cam_name}_image_pc_timestamp': "PCT",
            f'{cam_name}_image_ephys_timestamp': "ET",
        }
        
    elif key in ["event", "ballvelocity"]:
        rename_dict = {
            f'{key}_package_id': 'ID',
            f'{key}_portenta_timestamp': 'T',
            f'{key}_pc_timestamp': 'PCT',
            f'{key}_value': 'V',
            f'{key}_name': 'N',
        }
        if key == "ballvelocity":
            rename_dict.update({
                'ballvelocity_raw': "raw",
                'ballvelocity_yaw': "yaw",
                'ballvelocity_pitch': "pitch",
                })
    else:
        rename_dict = {}
    
    Logger().logger.debug(f"Renaming columns to old keys: {data.columns}")
    data = data.rename(columns=rename_dict)
    return data

def data_modality_deltaT_from_session_start(data):
    # frame timestamps, trial timestamps PC time    
    if 'frame_start_pc_timestamp' in data.columns:
        data["frame_start_pc_timestamp"] -= data["frame_start_pc_timestamp"].iloc[0]
        data["frame_end_pc_timestamp"] -= data["frame_start_pc_timestamp"].iloc[0]
    if 'trial_start_pc_timestamp' in data.columns:
        data["trial_start_pc_timestamp"] -= data["trial_start_pc_timestamp"].iloc[0]
        data["trial_end_pc_timestamp"] -= data["trial_start_pc_timestamp"].iloc[0]
    # ephys timestamps, frame and trials
    if 'frame_start_ephys_timestamp' in data.columns:
        data["frame_start_ephys_timestamp"] -= data["frame_start_ephys_timestamp"].iloc[0]
        data["frame_end_ephys_timestamp"] -= data["frame_start_ephys_timestamp"].iloc[0]
    if 'trial_start_ephys_timestamp' in data.columns:
        data["trial_start_ephys_timestamp"] -= data["trial_start_ephys_timestamp"].iloc[0]
        data["trial_end_ephys_timestamp"] -= data["trial_start_ephys_timestamp"].iloc[0]
    return data

def data_modality_us2s(data):
    us_columns = [c for c in data.columns if "pc_timestamp" in c or "pc_duration" in c]
    data[us_columns] = data[us_columns].astype(float)
    data[us_columns] /= 1e6
    return data

# =============================================================================
# portenta modality transformations
# =============================================================================

# the licksensor saves events as the time when the lick stopped + a ngative 
# interval of lick duration. Convert here to start time and duration
def event_modality_stoplick2startlick(eventdata):
    lick_events_mask = eventdata["event_name"] == "L"
    lick_length = eventdata.loc[lick_events_mask, "event_value"] * -1
    lick_length = round(lick_length/50) * 50 # round the ephys sampling rate (20KHz)
    eventdata.loc[lick_events_mask, "event_value"] = lick_length
    # lick length is not positive, subtract it from the stop time to make start stime
    eventdata.loc[lick_events_mask, "event_pc_timestamp"] -= lick_length
    if eventdata["event_ephys_timestamp"].notna().all():
        # ephys data exists
        eventdata.loc[lick_events_mask, "event_ephys_timestamp"] += lick_length
    return eventdata

def ballvel_modality_split_ball_velocity(balldata):
    # split the ballvelocity into 3 different tables, merge them with events later
    yaw_data = balldata.copy().drop(columns=['ballvelocity_raw', 'ballvelocity_pitch'])
    raw_data = balldata.copy().drop(columns=['ballvelocity_yaw', 'ballvelocity_pitch'])
    pitch_data = balldata.copy().drop(columns=['ballvelocity_raw', 'ballvelocity_yaw'])
    # name becomes indicative for raw yaw pitch
    yaw_data['ballvelocity_name'] = "B_forward"
    raw_data['ballvelocity_name'] = "B_rotate"
    pitch_data['ballvelocity_name'] = "B_sideway"
    # the value is the current sensor value/ ball velocity
    yaw_data.rename(columns={'ballvelocity_yaw': 'ballvelocity_value'}, inplace=True)
    raw_data.rename(columns={'ballvelocity_raw': 'ballvelocity_value'}, inplace=True)
    pitch_data.rename(columns={'ballvelocity_pitch': 'ballvelocity_value'}, inplace=True)
    return yaw_data, raw_data, pitch_data

def event_modality_calc_timeintervals_around_lick(lickdata, interval):
    # check if data is in seconds or microseconds
    unit = 'us' if lickdata["event_pc_timestamp"].iloc[0] > 1e10 else "s"
    t = pd.to_datetime(lickdata["event_pc_timestamp"], unit=unit)
    
    # Calculate the start and end times for each interval
    start_times = t - pd.Timedelta(seconds=interval)
    end_times = t + pd.Timedelta(seconds=interval)
    intervals = pd.DataFrame({'start': start_times, 'end': end_times})
        
    ## stackoverflow magic by John Smith <3
    ## This line compares if start of present row is greater than largest end in previous 
    ## rows ("shift" shifts up end by one row). The value of expression before
    ## cumsum will be True if interval breaks (i.e. cannot be merged), so
    ## cumsum will increment group value when interval breaks (cum sum treats True=1, False=0)
    intervals["group"]=(intervals["start"]>intervals["end"].shift().cummax()).cumsum()
    ## this returns min value of "start" column from a group and max value fro m "end"
    result=intervals.groupby("group").agg({"start":"min", "end": "max"})
    interval_index = pd.IntervalIndex.from_arrays(result['start'], result['end'], closed='both')
    return interval_index

# =============================================================================
# unity frame modality transformations
# =============================================================================
    
def unity_modality_track_spatial_bins(frames):
    z = frames['frame_z_position'].astype(int)
    bin_edges = np.arange(z.min(), z.max()+1, 1)
    binned_pos = pd.cut(z, bins=bin_edges, right=False)
    from_z_position = binned_pos.values.map(lambda interval: interval.left, 
                                            na_action='ignore')
    to_z_position = binned_pos.values.map(lambda interval: interval.right, 
                                            na_action='ignore')
    return from_z_position, to_z_position

def unity_modality_track_zones(frames, track_details):
    # Extract start and end positions from the dictionary
    intervals = [(details['start_pos'], details['end_pos']) for details in track_details.values()]

    # Create an IntervalIndex from the list of tuples
    interval_index = pd.IntervalIndex.from_tuples(intervals, closed='left')
    binned_pos = pd.cut(frames['frame_z_position'], bins=interval_index)
    # Map the intervals to their corresponding zone names
    interval_to_zone = {interval: zone for interval, zone in zip(interval_index, 
                                                                 track_details.keys())}
    return binned_pos.map(interval_to_zone)

def unity_modality_track_kinematics(frames):
    positions = frames["frame_z_position"].copy()
    positions[frames["trial_id"]==-1] = np.nan
    
    # data in microseconds, convert to seconds
    scaler = 1e6
    tstamps = frames["frame_pc_timestamp"] /scaler
    velocity = pd.Series(np.gradient(positions,tstamps,), index=frames.index, 
                         name='frame_z_velocity')
    acceleration = pd.Series(np.gradient(velocity, velocity.index), index=frames.index, 
                             name='frame_z_acceleration')
    return velocity, acceleration # in cm/s, cm/s^2


def fix_missing_paradigm_variable_names(data):
    if data is None:
        return None
    renamer = {
        # P800
        "LR": "lick_triggers_reward",
        
        # P1100
        "ST_2": "velocity_threshold_at_R2",
        "SR": 'multi_reward_requires_stop',
        "DR": 'both_R1_R2_rewarded',
        "RF": 'flip_Cue1R1_Cue2R2',
        "NP": 'prob_cue1_trial',
        "GF": 'movement_gain_scaler',
    }
    if all([True if c in data.columns else False for c in ["ST_2","SR","DR","RF","NP","GF"]]):
        # paradigm P1100 hacky fix to correct an old label from P0800...
        renamer['stay_time'] = 'velocity_threshold_at_R1'
        renamer['stop_threshold'] = 'velocity_threshold_at_R1'
    
    data = data.rename(columns=renamer)
    return data
    
def fix_ephys_timestamps_offset(data):
    if not isinstance(data, pd.DataFrame):
        return data
    ephys_t_col = [col for col in data.columns if "ephys_timestamp" in col]
    if len(ephys_t_col) == 0:
        return data

    # offset to 0
    data[ephys_t_col] -= data[ephys_t_col].iloc[0]
    if ephys_t_col[0] == "frame_ephys_timestamp":
        data[ephys_t_col[0]] += 1.82 *1e6 # on average, other sessions have first unity frame after
    elif ephys_t_col[0] == "ballvelocity_ephys_timestamp":
        data[ephys_t_col[0]] += 0.82 *1e6 # on average, other sessions have first ball vel pack after
    elif ephys_t_col[0] == "event_ephys_timestamp":
        data[ephys_t_col[0]] += 8.2 *1e6 # on average, super uncertain
    elif ephys_t_col[0] == "trial_start_ephys_timestamp":
        data[ephys_t_col[0]] = pd.NA # hope it's never used
    elif ephys_t_col[0] == "trial_end_ephys_timestamp":
        data[ephys_t_col[0]] = pd.NA # hope it's never used
        
         
    # camera missing, but not relevant for now
    return data
    






























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
            mean_cols = ['frame_z_position', 'z_velocity', 'z_acceleration']
            agg_bin_data = pos_bin_data[mean_cols].mean()
            
            # if any of the frames in the bin had a reward or lick event
            all_none_cols = ['frame_reward', 'frame_lick']
            agg_bin_data = pd.concat([agg_bin_data,pos_bin_data[all_none_cols].any()])
            
            # sum the ball velocity components
            ball_vel_cols = ['frame_raw', 'frame_yaw', 'frame_pitch']
            agg_bin_data = pd.concat([agg_bin_data, pos_bin_data[ball_vel_cols].sum()])
            
            # these columns are identical for all frames in the bin
            idential_cols = ['trial_id', 'cue', 'trial_outcome', 'zone',
                            'frame_pc_timestamp', 'binned_pos',]
            idential_cols = [col for col in idential_cols if col in pos_bin_data.columns]
            agg_bin_data = pd.concat([agg_bin_data, pos_bin_data[idential_cols].iloc[0]])
            
            renamer = {'frame_z_position': 'posbin_z_position',
                    'z_velocity': 'posbin_z_velocity',
                    'z_acceleration': 'posbin_z_acceleration',
                    'frame_pc_timestamp': 'posbin_start_pc_timestamp',
                    'frame_id': 'posbin_start_frame_id',
                    'frame_reward': 'posbin_reward',
                    'frame_lick': 'posbin_lick',
                    'frame_raw': 'posbin_raw',
                    'frame_yaw': 'posbin_yaw',
                    'frame_pitch': 'posbin_pitch',
                    }
            agg_bin_data.rename(renamer, inplace=True)
            agg_bin_data['nframes_in_bin'] = pos_bin_data.shape[0]
            agg_bin_data.name = agg_bin_data['binned_pos']
            return agg_bin_data
        # on trial level, group by spatial bin
        return trial_data.groupby('binned_pos', observed=True).apply(proc_pos_bin)
    # first, group into trials
    posbin_data = data.groupby('trial_id').apply(proc_trial_data)
    
    posbin_data = posbin_data.astype({'posbin_reward': bool, 'posbin_lick': bool})
    posbin_data.index = posbin_data.index.droplevel(0)
    return posbin_data    



# trial complementation using frames 
def calc_staytimes(trials, frames, track_details):
    def _calc_trial_staytimes(trial_frames):
        trial_id = trial_frames["trial_id"].iloc[0]
        if trial_id == -1:
            return pd.Series(dtype='int64')

        staytimes = {}
        outside_R_avg_velocties = []
        for zone, zone_details in track_details.items():
            zone_frames = trial_frames.loc[(trial_frames["frame_z_position"] >= zone_details["start_pos"]) & 
                                           (trial_frames["frame_z_position"] < zone_details["end_pos"])]
            if zone_frames.empty:
                zone_staytime = 0
            else:
                zone_staytime = zone_frames["frame_pc_timestamp"].iloc[-1] - zone_frames["frame_pc_timestamp"].iloc[0]
            
            # check if the zone is a reward zone and if the trial was correct or incorrect
            cue = trials[trials["trial_id"] == trial_id]["cue"].item()
            if (zone == "reward1" and cue == 1) or (zone == "reward2" and cue == 2): 
                staytimes["staytime_correct_r"] = zone_staytime
            elif (zone == "reward2" and cue == 1) or (zone == "reward1" and cue == 2):
                staytimes["staytime_incorrect_r"] = zone_staytime
            else:
                if zone_staytime != 0:
                    avg_vel = (zone_details['end_pos']-zone_details['start_pos']) /zone_staytime/1e-6
                    outside_R_avg_velocties.append(avg_vel)
                
            # add the time of entry into the reward zone
            if zone == 'reward1' or zone == 'reward2':
                # print("Rewardzone size: ", zone_details['end_pos']-zone_details['start_pos'])
                if zone_frames.empty:
                    staytimes[f"enter_{zone}"] = np.nan
                    Logger().logger.warning(f"Trial {trial_id} has no frames in {zone}")
                else:
                    staytimes[f"enter_{zone}"] = zone_frames["frame_pc_timestamp"].iloc[0]
                    staytimes[f"staytime_{zone}"] = zone_staytime
        staytimes['baseline_velocity'] = np.median(outside_R_avg_velocties)

        return pd.Series(staytimes)
    result = frames.groupby("trial_id").apply(_calc_trial_staytimes).unstack().reset_index(drop=True)
    return result


# join frames and events and aggregate events to frame resolution
def frame_wise_events(frames, events):
    # Create intervals for each frame
    frame_intervals = pd.IntervalIndex.from_arrays(
        frames['frame_pc_timestamp'],
        frames['frame_pc_timestamp'].shift(-1, fill_value=frames['frame_pc_timestamp'].iloc[-1] + 16_666),
        closed='left'
    )
    
    # final event postporcessing columns
    # licking: not, licks, bites, consumes 
    # reward_available: False, True
    # sound: S event +250ms
    
    # assign each event to a frame interval, initial and final events can be outside of frame intervals
    events['frame_bin'] = pd.cut(events['event_pc_timestamp'], bins=frame_intervals)
    events.set_index('frame_bin', inplace=True)
    # keep only one event per frame
    events = events[~events.index.duplicated(keep='first')]

    #TODO: add sound events, vacuum, reward available, etc.
    #TODO lick has length, calc backwards using value field
    events = events.reindex(frame_intervals).reset_index(drop=True)
    events['reward'] = (events['event_name'] == 'R').fillna(False)
    events['lick'] = (events['event_name'] == 'L').fillna(False)
    return events['reward'], events['lick']

    # trial_i = merged_df[merged_df['trial_id'] == 3].reset_index()
    # import matplotlib.pyplot as plt
    # # plt.plot(trial_i['frame_z_position'], label='position')
    # plt.plot(trial_i['lick'], alpha=.5, label='lick')
    # plt.plot(trial_i['reward'], alpha=.5, label='reward')
    # plt.legend()
    # plt.show()
    # exit()


# join frames and ball velocity packages and aggregate ball velocity to frame resolution
def frame_wise_ball_velocity(frames, ball_vel):
    ball_vel.set_index("ballvelocity_package_id", inplace=True)
    # Create IntervalIndex with both sides closed, first frame does not have a previous frame
    first_pckg_ids = frames.iloc[1:]['ballvelocity_first_package'].values
    last_pckg_ids = frames.iloc[:-1]['ballvelocity_last_package'].values

    # do a first check if the difference between package ids is 1
    f2f_diff = first_pckg_ids-last_pckg_ids
    invalid_diff = f2f_diff != 1
    if invalid_diff.any():
        Logger().logger.warning(f"Frame to frame ballvell package id difference "
                                f"is not 1 for all, {f2f_diff[invalid_diff]}")
        # check if the invalid frames are outside of ITI
        invalid_frames = frames.iloc[1:][invalid_diff]
        if (invalid_frames['trial_id'] != -1).any(): 
            # raise ValueError(f"Irregularities outside ITI: \n{invalid_frames}")
            Logger().logger.warning(f"Irregularities outside ITI: \n{invalid_frames}")
    
    # create intervals for each frame, except the first which can't be validated to not be overlapping
    intervals = pd.IntervalIndex.from_arrays(frames.iloc[1:].loc[~invalid_diff, 'ballvelocity_first_package'],
                                             frames.iloc[1:].loc[~invalid_diff, 'ballvelocity_last_package'],
                                             closed='both')
    
    # Assign collection of ball velocity packages to its frame interval
    Logger().logger.debug(f"Assigning ball velocity packages to frame intervals...")
    ball_vel['frame_bin'] = pd.cut(ball_vel.index, bins=intervals)
    frame_wise_ryp = ball_vel.loc[:, ['frame_bin','ballvelocity_raw',
                                      'ballvelocity_yaw','ballvelocity_pitch']]
    
    frame_wise_ryp = frame_wise_ryp.groupby('frame_bin', observed='False').sum().rename(
        {'ballvelocity_raw': 'frame_raw',
         'ballvelocity_yaw': 'frame_yaw',
         'ballvelocity_pitch': 'frame_pitch'}, axis=1)
    # instead of bin indices, go back to frame indices
    frame_wise_ryp.index = frames.iloc[1:].loc[~invalid_diff].index
    return frame_wise_ryp

def _validate_ballvell_package_ids():
    pass