import json
import os

import h5py
import pandas as pd

from CustomLogger import CustomLogger as Logger

import analytics_processing.modality_transformations as mT
from analytics_processing import metadata_loading
# import analytics_processing.analytics_constants as C

def _load_cam_frames(session_file, key):
    #TODO implement this
    raise NotImplementedError("Loading camera frames from session data is not implemented yet")

def session_modality_from_nas(session_fullfname, key, where=None, start=None, 
                               stop=None, columns=None):
    L = Logger()
    # session_fullfname = _make_session_filename(nas_base_dir, pardigm_subdir, session_name)
    
    # special key for camera frames    
    if key.startswith("cam") and key.endswith("frames"):
        session_file = load_session_hdf5(session_fullfname)
        return _load_cam_frames(session_file, key)
    
    # pandas based data
    try:
        with pd.HDFStore(session_fullfname, mode='r') as store:
            if store.get_storer(key).is_table:
                args = {keystr: key for keystr, key in zip(("start", "stop", "where", "columns"),
                                                           (start, stop, where, columns)) 
                        if key is not None}
                argsmsg = "with "+L.fmtmsg(args) if args else ""
                L.logger.debug(f"Accessing {key} from table {argsmsg}")
                data = store.select(key, start=start, stop=stop, where=where, 
                                    columns=columns)
            # very old data (non-patched) may not be in table format, can't read with select
            else:
                L.logger.debug(f"Accessing {key}...")
                if columns is not None:
                    L.logger.warning(f"Data not in table format, `columns` will be ignored")
                data = store.select(key)
            L.logger.debug(f"Got data with shape ({data.shape[0]:,}, {data.shape[1]:,})")
        # data = pd.read_hdf(session_fullfname, key=key, mode='r', start=start, 
        #                    stop=stop, where=where, columns=columns)
        L.logger.debug(f"Successfully accessed {key}")
        L.logger.debug(f"Data: \n{data}")
        L.logger.debug(f"Data: \n{data.iloc[:2].T}")
    except KeyError:
        L.logger.error(f"Key {key} not found in session data")
        data = None
    except FileNotFoundError:
        L.logger.error(f"Session file not found:\n{session_fullfname}")
        data = None
    
    if key == 'metadata':
        session_name = os.path.basename(session_fullfname).replace(".hdf5", "")
        if data is not None:
            # metadata is the only key that will never return None
            data = metadata_loading.extract_metadata(data, session_name)
        else:
            data = metadata_loading.minimal_metad_from_session_name(session_name)
    elif key == 'paradigm_variable':
        data = mT.fix_missing_paradigm_variable_names(data)
    return data

def load_session_hdf5(session_fullfname):
    try:
        session_file = h5py.File(session_fullfname, 'r')
    except OSError:
        Logger().logger.spacer()
        Logger().logger.error(f"Session file {session_fullfname} not found")
        session_file = None
    return session_file

# def get_complemented_session_modality(session_fullfname, modality, modality_parsing_kwargs={}, 
#                          **session_parsing_kwargs):
#     L = Logger()
#     data = session_modality_from_nas(session_fullfname, modality, 
#                                       **modality_parsing_kwargs)
#     if data is None:
#         Logger().logger.error(f"Failed to load {modality} data")
#         return 
    
#     if modality == 'metadata':
#         if session_parsing_kwargs.get("dict2pandas"):
#             # drop nested json-like fields
#             data = {k:v for k,v in data.items() 
#                     if k not in ["env_metadata","fsm_metadata","configuration","log_file_content"]}
#             data = pd.Series(data, name=0).to_frame().T
        
#         elif session_parsing_kwargs.get("dict2json"):
#             data = json.dumps(data, indent='  ').replace("NaN", "null")
            
#         elif session_parsing_kwargs.get("only_get_track_details"):
#             data = metadata_loading.env_metadata2track_details(data['env_metadata'])
    
#     elif modality == 'event':
#         pass
    
#         if session_parsing_kwargs.get("complement_data"):
#             lickdata = data.loc[data['event_name'] == "L"]
#             licks = mT.event_modality_calc_timeintervals_around_lick(lickdata, C.LICK_MERGE_INTERVAL)
            
            
#             #TODO process licks poerperly, make reward timeline with vacuum
#             #TODO add pose estimation
#             #  events = session_modality_from_nas(*session_dir_tuple, "event")
#                 # reward, lick = sT.frame_wise_events(data, events)
#                 # data['frame_reward'] = reward
#                 # data['frame_lick'] = lick
#                 # insert ball velocity
#                 # ball_vel = session_modality_from_nas(*session_dir_tuple, "ballvelocity")
#                 # raw_yaw_pitch = sT.frame_wise_ball_velocity(data, ball_vel)
#                 # data = pd.concat([data, raw_yaw_pitch], axis=1)






#     elif modality == "paradigm_variable":
#         if session_parsing_kwargs.get("fix_missing_names"):
#             data = mT.fix_missing_paradigm_variable_names(data)
    
#     elif modality == 'unity_frame' or modality == 'unity_trial':
#         paradigm_id = get_complemented_session_modality(session_fullfname, "metadata",
#                                            {"columns": ["paradigm_name"]})['paradigm_id']
#         if paradigm_id in (800, 1100):
#             track_details = get_complemented_session_modality(session_fullfname, "metadata",
#                                                  {"columns": ["env_metadata"]},
#                                                  only_get_track_details=True)
#             if modality == 'unity_frame':
#                 if session_parsing_kwargs.get("to_deltaT_from_session_start"):
#                     data = mT.data_modality_deltaT_from_session_start(data)
                
#                 if session_parsing_kwargs.get("complement_data"):
#                     # insert z position bin (1cm)
#                     from_z_position, to_z_position = mT.unity_modality_track_spatial_bins(data)
#                     data['from_z_position_bin'] = from_z_position
#                     data['to_z_position_bin'] = to_z_position

#                     # insert column with string which zone in at this frame                
#                     zone = mT.unity_modality_track_zones(data, track_details)
#                     data['zone'] = zone 
                    
#                     # insert cue/ trial type, outcome etc, denormalize 
#                     trials_variable = get_complemented_session_modality(session_fullfname, 
#                                                            "paradigm_variable",
#                                                            fix_missing_names=True)
#                     data = pd.merge(data, trials_variable, on='trial_id', how='left')
                    
#                     # insert velocity and acceleration
#                     vel, acc = mT.unity_modality_track_kinematics(data)
#                     data = pd.concat([data, vel, acc], axis=1)
            
#             elif modality == 'unity_trial':
#                 if session_parsing_kwargs.get("to_deltaT_from_session_start"):
#                     data = mT.data_modality_deltaT_from_session_start(data)
                
#                 trials_variable = get_complemented_session_modality(session_fullfname, 
#                                                        "paradigm_variable",
#                                                        fix_missing_names=True)
#                 trials_variable.drop(columns=['trial_id'], inplace=True) # double
#                 data = pd.concat([data, trials_variable], axis=1)
                
#                 cols = ('trial_id', "frame_z_position", 'frame_pc_timestamp')
#                 unity_frames = get_complemented_session_modality(session_fullfname, 
#                                                     "unity_frame", columns=cols)
                
#                 staytimes = mT.calc_staytimes(data, unity_frames, track_details)
#                 data = pd.concat([data, staytimes], axis=1)
#     return data