import numpy as np
import pandas as pd

from CustomLogger import CustomLogger as Logger

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
            f'{cam_name}_image_pc_timestamp': "frame_pc_timestamp",
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
    not_time_columns = [c for c in data.columns if not c.endswith("_timestamp")]
    t0 = data.iloc[0].copy()
    t0.loc[not_time_columns] = 0 
    data -= t0
    return data

def data_modality_us2s(data):
    us_columns = [c for c in data.columns if "pc_timestamp" in c or "pc_duration" in c]
    data[us_columns] = data[us_columns].astype(float)
    data[us_columns] /= 1e6
    return data

def metadata_complement_P0800(env_metadata):
    n_pillar_types = len(env_metadata["pillar_details"])
    pillars_posY = {}
    for i in range(1, n_pillar_types):
        pillar_pos = [val["y"] for val in env_metadata["pillars"].values() if val['id'] == i][0]
        # transform excel coordinates to unity coordinates
        pillars_posY[f"pillar{i}"] = env_metadata["envX_size"]/2 - pillar_pos
    
    # the most up-to-date version of the paradigm
    if n_pillar_types == 17:
        # gget the start and stop unity cooredinates for each region
        zone_positions = {
            # 'start_zone': (-169, pillars_posY["pillar5"]),
            'start_zone': (pillars_posY["pillar5"]-40, pillars_posY["pillar5"]),
            'cue1_visible': (pillars_posY["pillar5"], pillars_posY["pillar6"]),
            'cue1': (pillars_posY["pillar6"], pillars_posY["pillar7"]),
            'cue1_passed': (pillars_posY["pillar7"], pillars_posY["pillar8"]),
            'between_cues': (pillars_posY["pillar8"], pillars_posY["pillar9"]),
            'cue2_visible': (pillars_posY["pillar9"], pillars_posY["pillar10"]),
            'cue2': (pillars_posY["pillar10"], pillars_posY["pillar11"]),
            'cue2_passed': (pillars_posY["pillar11"], pillars_posY["pillar12"]),
            'before_reward1': (pillars_posY["pillar12"], pillars_posY["pillar13"]),
            'reward1': (pillars_posY["pillar13"], pillars_posY["pillar14"]),
            'before_reward2': (pillars_posY["pillar14"], pillars_posY["pillar15"]),
            'reward2': (pillars_posY["pillar15"], pillars_posY["pillar16"]),
            'post_reward': (pillars_posY["pillar16"], env_metadata["envX_size"]/2),
        }
        
    # initial version of paradigm
    elif n_pillar_types == 11:
        if env_metadata["envX_size"] == 480:
            # early case when we start at 0 and cue at first position
            start_pos = -env_metadata["envX_size"]/2
        elif env_metadata["envX_size"] == 540:
            # later case when we start at -169 and cue at second position
            start_pos = -169
        zone_positions = {
            'start_zone': (start_pos, pillars_posY["pillar5"]),
            'cue1_visible': (pillars_posY["pillar5"], pillars_posY["pillar5"]+40),
            'cue1': (pillars_posY["pillar5"]+40, pillars_posY["pillar5"]+80),
            'cue1_passed': (pillars_posY["pillar5"]+80, pillars_posY["pillar6"]),
            'between_cues': (pillars_posY["pillar6"], pillars_posY["pillar7"]),
            'cue2_visible': (pillars_posY["pillar7"], pillars_posY["pillar7"]+40),
            'cue2': (pillars_posY["pillar7"]+40, pillars_posY["pillar7"]+80),
            'cue2_passed': (pillars_posY["pillar7"]+80, pillars_posY["pillar8"]),
            'before_reward1': (pillars_posY["pillar8"], pillars_posY["pillar8"]+40),
            'reward1': (pillars_posY["pillar8"]+40, pillars_posY["pillar9"]),
            'before_reward2': (pillars_posY["pillar9"], pillars_posY["pillar9"]+40),
            'reward2': (pillars_posY["pillar9"]+40, pillars_posY["pillar10"]),
            'post_reward': (pillars_posY["pillar10"], env_metadata["envX_size"]/2),
        }
    else:
        raise ValueError(f"Unknown number of pillar types {n_pillar_types}, not 11 or 17")
    
    # these are the regions that have relevant pillar details (actual pillars)
    special_zones_pillar_indices = {"cue1": '1', "cue2": '2', "reward1": '3', 
                                    "reward2": '4'}
    
    P0800_pillar_details = {}
    for zone in zone_positions.keys():
        P0800_zone_details = {"start_pos": zone_positions[zone][0], 
                              "end_pos": zone_positions[zone][1]}
        if zone in special_zones_pillar_indices:
            pillar_idx = special_zones_pillar_indices[zone]
            P0800_zone_details["radius"] = env_metadata["pillar_details"][pillar_idx]["pillarRadius"]
            P0800_zone_details["height"] = env_metadata["pillar_details"][pillar_idx]["pillarHeight"]
            P0800_zone_details["z_pos"] = env_metadata["pillar_details"][pillar_idx]["pillarZposition"]
            P0800_zone_details["texture"] = env_metadata["pillar_details"][pillar_idx]["pillarTexture"]
            P0800_zone_details["transparency"] = env_metadata["pillar_details"][pillar_idx]["pillarTransparency"]
            P0800_zone_details["reward_radius"] = env_metadata["pillar_details"][pillar_idx]["pillarRewardRadius"]
            P0800_zone_details["show_ground"] = env_metadata["pillar_details"][pillar_idx]["pillarShowGround"]
        P0800_pillar_details[zone] = P0800_zone_details
    
    return P0800_pillar_details

def calc_timeintervals_around_lick(lickdata, interval):
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

def calc_unity_kinematics(frames):
    positions = frames["frame_z_position"].copy()
    positions[frames["trial_id"]==-1] = np.nan
    
    # data in seconds or microseconds?
    scaler = 1e6 if frames["frame_pc_timestamp"].iloc[0] > 1e10 else 1
    tstamps = frames["frame_pc_timestamp"] / scaler
    velocity = pd.Series(np.gradient(positions,tstamps,), index=frames.index, name='z_velocity')
    acceleration = pd.Series(np.gradient(velocity, velocity.index), index=frames.index, name='z_acceleration')
    return velocity, acceleration # in cm/s, cm/s^2
    
def calc_staytimes(trials, frames, P0800_pillar_details):
    def _calc_trial_staytimes(trial_frames):
        trial_id = trial_frames["trial_id"].iloc[0]
        if trial_id == -1:
            return pd.Series(dtype='int64')

        staytimes = {}
        for zone, zone_details in P0800_pillar_details.items():
            zone_frames = trial_frames.loc[(trial_frames["frame_z_position"] >= zone_details["start_pos"]) & 
                                           (trial_frames["frame_z_position"] < zone_details["end_pos"])]
            if zone_frames.empty:
                zone_staytime = 0
            else:
                zone_staytime = zone_frames["frame_pc_timestamp"].iloc[-1] - zone_frames["frame_pc_timestamp"].iloc[0]
            staytimes["staytime_"+zone] = zone_staytime
            
            # check if the zone is a reward zone and if the trial was correct or incorrect
            cue = trials[trials["trial_id"] == trial_id]["cue"].item()
            if (zone == "reward1" and cue == 1) or (zone == "reward2" and cue == 2): 
                staytimes["staytime_correct_r"] = zone_staytime
            elif (zone == "reward2" and cue == 1) or (zone == "reward1" and cue == 2):
                staytimes["staytime_incorrect_r"] = zone_staytime
                
            # add the time of entry into the reward zone
            if zone == 'reward1' or zone == 'reward2':
                staytimes[f"enter_{zone}"] = zone_frames["frame_pc_timestamp"].iloc[0]
        return pd.Series(staytimes)
    result = frames.groupby("trial_id").apply(_calc_trial_staytimes).unstack().reset_index(drop=True)
    return result

def calc_track_bins(frames):
    z = frames['frame_z_position']
    bin_edges = np.arange(z.min(), z.max(), 1)
    binned_pos = pd.cut(z, bins=bin_edges)
    return binned_pos

def calc_zone(frames, P0800_pillar_details):
    # Extract start and end positions from the dictionary
    intervals = [(details['start_pos'], details['end_pos']) for details in P0800_pillar_details.values()]

    # Create an IntervalIndex from the list of tuples
    interval_index = pd.IntervalIndex.from_tuples(intervals, closed='left')
    binned_pos = pd.cut(frames['frame_z_position'], bins=interval_index)
    # Map the intervals to their corresponding zone names
    interval_to_zone = {interval: zone for interval, zone in zip(interval_index, 
                                                                 P0800_pillar_details.keys())}
    return binned_pos.map(interval_to_zone)

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