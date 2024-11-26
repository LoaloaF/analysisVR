import json
import os

import h5py
import pandas as pd

from CustomLogger import CustomLogger as Logger

import session_transformations as sT
import metadata_loading

def _make_session_filename(nas_base_dir, pardigm_subdir, session_name):
    L = Logger()
    session_fullfname = os.path.join(nas_base_dir, pardigm_subdir, session_name+".hdf5")
    if not os.path.exists(session_fullfname):
        L.logger.error(f"Session file not found:\n{session_fullfname}, trying "
                       f"with `behavior_` prefix...")
        new_sessionfname = os.path.basename(session_fullfname).replace("2024-", "behavior_2024-")
        session_fullfname = os.path.join(os.path.dirname(session_fullfname), new_sessionfname)
    return session_fullfname

def _load_cam_frames(session_file, key):
    # if session_file is None: return None
    # try:
    #     data = session_file[key]
    # except KeyError:
    #     Logger().logger.error(f"Key {key} not found in session data")
    #     return None
    #TODO implement this
    raise NotImplementedError("Loading camera frames from session data is not implemented yet")

def _session_modality_from_nas(nas_base_dir, pardigm_subdir, session_name, key, 
                               where=None, start=None, stop=None, columns=None):
    L = Logger()
    session_fullfname = _make_session_filename(nas_base_dir, pardigm_subdir, session_name)
    
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
    except KeyError:
        L.logger.error(f"Key {key} not found in session data")
        data = None
    except FileNotFoundError:
        L.logger.error(f"Session file not found:\n{session_fullfname}")
        data = None
    
    if key == 'metadata':
        if data is not None:
            # metadata is the only key that will never return None
            data = metadata_loading.extract_metadata(data, session_name)
        else:
            data = metadata_loading.minimal_metad_from_session_name(session_name)
    return data

# def _metadata_from_session_name(session_name):

# def _parse_metadata(metadata, session_name):
    
def load_session_hdf5(session_fullfname):
    try:
        session_file = h5py.File(session_fullfname, 'r')
    except OSError:
        Logger().logger.spacer()
        Logger().logger.error(f"Session file {session_fullfname} not found")
        session_file = None
    return session_file

# def get_session_metadata(session_dir_tuple, modality_parsing_kwargs={}):
#     if session_dir_tuple is not None:
#         metadata = _session_modality_from_nas(*session_dir_tuple, "metadata", **modality_parsing_kwargs)
#     elif from_db is not None:
#         metadata = _session_modality_from_db(*from_db, "metadata")
#     else:
#         raise ValueError("Either session_dir_tuple or from_db must be provided")
    
#     metadata = _parse_metadata(metadata)
#     if metadata["paradigm_id"] in (800, 1100):
#         P0800_pillar_details = sT.metadata_complement_P0800(metadata["env_metadata"])
#         metadata["P0800_pillar_details"] = P0800_pillar_details
#     return metadata
    
def get_session_modality(modality, session_dir_tuple, modality_parsing_kwargs={}, 
                         **session_parsing_kwargs):
    L = Logger()
    data = _session_modality_from_nas(*session_dir_tuple, modality, 
                                      **modality_parsing_kwargs)
    if data is None:
        Logger().logger.error(f"Failed to load {modality} data")
        return 
    
    if modality == 'metadata':
        if session_parsing_kwargs.get("dict2pandas"):
            # drop nested json-like fields
            data = {k:v for k,v in data.items() 
                    if k not in ["env_metadata","fsm_metadata","configuration","log_file_content"]}
            data = pd.Series(data, name=0).to_frame().T
            
        if session_parsing_kwargs.get("only_get_track_details"):
            data = metadata_loading.env_metadata2track_details(data['env_metadata'])
            
            
    
    if session_parsing_kwargs.get("to_deltaT_from_session_start"):
        data = sT.data_modality_deltaT_from_session_start(data)
    
    if session_parsing_kwargs.get("pct_as_index"):
        data = sT.data_modality_pct_as_index(data)
    
    if session_parsing_kwargs.get("us2s"):
        data = sT.data_modality_us2s(data)
    
    if modality == 'event' and (event_name := session_parsing_kwargs.get("event_subset")):
        data = data.loc[data['event_name'] == event_name]
    
    if session_parsing_kwargs.get("na2null"):
        data = sT.data_modality_na2null(data)
                
    if session_parsing_kwargs.get("complement_data"):
        # try:
            paradigm_id = get_session_modality("metadata", session_dir_tuple, 
                                               {"columns": ["paradigm_name"]})['paradigm_id']
            if paradigm_id in (800, 1100):
                track_details = get_session_modality("metadata", session_dir_tuple, {"columns": ["env_metadata"]},
                                                     only_get_track_details=True)
            if modality == "unity_trial":
                L.logger.debug(f"Complementing {modality} modality with paradigmVariable_data and unity_frame")
                
                #TODO rename them with metadata
                trials_variable = _session_modality_from_nas(*session_dir_tuple, "paradigm_variable")
                # cols = ["P0800_pillar_details", 'trial_id']
                
                # metadata =  get_session_metadata(session_dir_tuple, {"columns": cols})
                # metadata = _session_modality_from_nas(*session_dir_tuple, "metadata", columns=cols)
                
                cols = ('trial_id', "frame_z_position", 'frame_pc_timestamp')
                unity_frames = _session_modality_from_nas(*session_dir_tuple, "unity_frame", columns=cols)

                trials_variable.drop(columns=['trial_id'], inplace=True) # double
                data = pd.concat([data, trials_variable], axis=1)
                
                if paradigm_id in (800,1100):
                    staytimes = sT.calc_staytimes(data, unity_frames, track_details)
                    data = pd.concat([data, staytimes], axis=1)

            if modality == 'unity_frame':
                # insert z position bin (1cm)
                binned_pos = sT.calc_track_bins(data)
                data['binned_pos'] = binned_pos
                
                # metadata =  get_session_metadata(session_dir_tuple)
                # metadata = _session_modality_from_nas(*session_dir_tuple, "metadata")
                
                if paradigm_id in (800,1100):
                    # insert current zone 
                    # cols = ["P0800_pillar_details", 'trial_id']
                    # metadata =  get_session_metadata(session_dir_tuple, {"columns": cols})
                    # metadata = _session_modality_from_nas(*session_dir_tuple, "metadata", columns=cols)
                    zone = sT.calc_zone(data, track_details)
                    data['zone'] = zone 
                
                    # insert cue/ trial type, and outcome
                    trials_variable = _session_modality_from_nas(*session_dir_tuple, "paradigm_variable")
                    data['cue'] = data['trial_id'].map(trials_variable.set_index('trial_id')['cue'])
                    trial_data = _session_modality_from_nas(*session_dir_tuple, "unity_trial")
                    data['trial_outcome'] = data['trial_id'].map(trial_data.set_index('trial_id')['trial_outcome'])
                    
                # insert velocity and acceleration
                vel, acc = sT.calc_unity_kinematics(data)
                data = pd.concat([data, vel, acc], axis=1)
                # insert events
                events = _session_modality_from_nas(*session_dir_tuple, "event")
                reward, lick = sT.frame_wise_events(data, events)
                data['frame_reward'] = reward
                data['frame_lick'] = lick
                # insert ball velocity
                ball_vel = _session_modality_from_nas(*session_dir_tuple, "ballvelocity")
                raw_yaw_pitch = sT.frame_wise_ball_velocity(data, ball_vel)
                data = pd.concat([data, raw_yaw_pitch], axis=1)

        # except Exception as e:
        #     L.spacer()
        #     L.logger.error(f"Failed to complement {modality} data, {e}")
        #     return None
    
    if modality=='unity_frame' and session_parsing_kwargs.get("position_bin_index"):
        data = sT.transform_to_position_bin_index(data)
        
    if session_parsing_kwargs.get("rename2oldkeys"):
        data = sT.data_modality_rename2oldkeys(data, modality,)
    
    if isinstance(data, pd.DataFrame): # only not true for metadata in dict format
        L.logger.debug(f"Returning {modality} modality, {data.shape}, "
                    f"first row:\n{data.iloc[0]}\nlast row:\n{data.iloc[-1]}")
    else:
        L.logger.debug(f"Returning metadata with keys {list(data.keys())}")
    return data