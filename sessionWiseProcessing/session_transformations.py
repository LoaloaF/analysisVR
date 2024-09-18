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
    not_time_columns = [c for c in data.columns if not c.endswith("pc_timestamp")]
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
            'start_zone': (-169, pillars_posY["pillar5"]),
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
        if env_metadata["envX_size"] == 440:
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

def calc_unity_velocity(frames):
    positions = frames["frame_z_position"].copy()
    positions[frames["trial_id"]==-1] = np.nan
    
    # data in seconds or microseconds?
    scaler = 1e6 if frames["frame_pc_timestamp"].iloc[0] > 1e10 else 1
    tstamps = frames["frame_pc_timestamp"] / scaler
    velocity = pd.Series(np.gradient(positions,tstamps), index=frames.index, name='z_velocity')
    return velocity # in cm/s
    
def calc_staytimes(trials, frames, metadata):
    def _calc_trial_staytimes(trial_frames):
        trial_id = trial_frames["trial_id"].iloc[0]
        if trial_id == -1:
            return pd.Series(dtype='int64')
        
        staytimes = {}
        for zone, zone_details in metadata["P0800_pillar_details"].items():
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
    return frames.groupby("trial_id").apply(_calc_trial_staytimes).unstack().reset_index(drop=True)