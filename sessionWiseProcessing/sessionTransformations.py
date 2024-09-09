import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../../CoreRatVR')) # project dir

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
        L.logger.warning(f"Non-unique index values found in the "
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
        # add back the name "U", indicating Unity frame
        data['N'] = "U"
    elif key == "unity_trial":
        rename_dict = {
            "trial_id": "ID", 
            "trial_start_frame": "SFID", 
            "trial_start_pc_timestamp": "SPCT", 
            "trial_end_frame": "EFID", 
            "trial_end_pc_timestamp": "EPCT",
            "trial_pc_duration": "TD", 
            "trial_outcome": "O"}
    elif key == "paradigmVariable_data":
        rename_dict = {}
        
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
    
def calc_staytimes(trials, frames):
    # frames = access_session_data("unity_frame", na2null=True, rename2oldkeys=True)
    #TODO don't need trials, use only frames, concat in higher order function
    staytimes = []
    for t, trial in trials.iterrows():
        trial_frames = frames.loc[frames["trial_id"] == trial["ID"]]
        within_cue1_frames = trial_frames.loc[trial_frames["S"] == 804]
        within_cue1_us = within_cue1_frames["PCT"].iloc[-1] - within_cue1_frames["PCT"].iloc[0]
        within_cue2_frames = trial_frames.loc[trial_frames["S"] == 808]
        within_cue2_us = within_cue2_frames["PCT"].iloc[-1] - within_cue2_frames["PCT"].iloc[0]

        within_r1_frames = trial_frames.loc[trial_frames["S"] == 811]
        within_r1_us = within_r1_frames["PCT"].iloc[-1] - within_r1_frames["PCT"].iloc[0]
        within_r2_frames = trial_frames.loc[trial_frames["S"] == 813]
        within_r2_us = within_r2_frames["PCT"].iloc[-1] - within_r2_frames["PCT"].iloc[0]
        
        staytimes.append(pd.Series(name=t, data={
            "staytime_cue1": within_cue1_us,
            "PCT_enter_cue1": within_cue1_frames["PCT"].iloc[0],
            "staytime_cue2": within_cue2_us,
            "PCT_enter_cue2": within_cue2_frames["PCT"].iloc[0],
            "staytime_r1": within_r1_us,
            "PCT_enter_r1": within_r1_frames["PCT"].iloc[0],
            "staytime_r2": within_r2_us,
            "PCT_enter_r2": within_r2_frames["PCT"].iloc[0],
        }))
    # trials = pd.concat([trials, pd.concat(staytimes, axis=1).T], axis=1)
    return pd.concat(staytimes, axis=1).T