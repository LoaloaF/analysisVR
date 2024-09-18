import json
import os

import h5py
import pandas as pd

from CustomLogger import CustomLogger as Logger

import session_transformations as sT

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
        L.logger.debug(f"Accessing {key}...")
        with pd.HDFStore(session_fullfname, mode='r') as store:
            if store.get_storer(key).is_table:
                # Table format, use the where clause
                data = store.select(key, start=start, stop=stop, where=where, 
                                    columns=columns)
            else:
                data = store.select(key)
        data = pd.read_hdf(session_fullfname, key=key, mode='r', start=start, 
                           stop=stop, where=where, columns=columns)
        L.logger.debug(f"Successfully accessed {key}")
    except KeyError:
        L.logger.error(f"Key {key} not found in session data")
        data = None
    except FileNotFoundError:
        L.logger.error(f"Session file not found:\n{session_fullfname}")
        data = None
    return data
        
def _session_modality_from_db(db_fullfname, key):
    #TODO implement this
    raise NotImplementedError("Loading session data from database is not implemented yet")

def _parse_metadata(metadata):
    metadata_parsed = {}
    
    paradigm_name = metadata.get("paradigm_name")
    if paradigm_name is not None and paradigm_name.shape[0]:
        paradigm_name = paradigm_name.item()
        metadata_parsed["paradigm_id"] = int(paradigm_name[1:5])

    animal = metadata.get("animal_name")
    if animal is not None and animal.shape[0]: 
        metadata_parsed["animal"] = animal.item()

    animal_weight = metadata.get("animal_weight")
    if animal_weight is not None and animal_weight.shape[0]: 
        metadata_parsed["animal_weight"] = animal_weight.item()

    start_time = metadata.get("start_time")
    if start_time is not None and start_time.shape[0]: 
        metadata_parsed["start_time"] = start_time.item()

    stop_time = metadata.get("stop_time")
    if stop_time is not None and stop_time.shape[0]: 
        metadata_parsed["stop_time"] = stop_time.item()

    duration = metadata.get("duration")
    if duration is not None and duration.shape[0]: 
        metadata_parsed["duration"] = duration.item()

    notes = metadata.get("notes")
    if notes is not None and notes.shape[0]: 
        metadata_parsed["notes"] = notes.item()
        
    env_metadata = metadata.get("env_metadata")
    if env_metadata is not None and env_metadata.shape[0]:
        metadata_parsed['env_metadata'] = json.loads(env_metadata.item())

    fsm_metadata = metadata.get("fsm_metadata")
    if fsm_metadata is not None and fsm_metadata.shape[0]:
        metadata_parsed['fsm_metadata'] = json.loads(fsm_metadata.item())
        
    log_file_content = metadata.get("log_file_content")
    if log_file_content is not None and log_file_content.shape[0]:
        metadata_parsed['log_file_content'] = json.loads(log_file_content.item())

    # old metadata format
    if (nested_metadata := metadata.get("metadata")) is not None: 
        nested_metadata = json.loads(nested_metadata.item())
        metadata_parsed.update({'env_metadata': {
            "pillars": nested_metadata.get("pillars"),
            "pillar_details": nested_metadata.get("pillar_details"),
            "envX_size": nested_metadata.get("envX_size"),
            "envY_size": nested_metadata.get("envY_size"),
            "base_length": nested_metadata.get("base_length"),
            "wallzone_size": nested_metadata.get("wallzone_size"),
            "wallzone_collider_size": nested_metadata.get("wallzone_collider_size"),}})
            
        metadata_parsed.update({'fsm_metadata': {
            "paradigms_states": nested_metadata.get("paradigms_states"),
            "paradigms_transitions": nested_metadata.get("paradigms_transitions"),
            "paradigms_decisions": nested_metadata.get("paradigms_decisions"),
            "paradigms_actions": nested_metadata.get("paradigms_actions"),
        }})
        
        metadata_parsed.update({'log_file_content': {
            "log_file_content": nested_metadata.get("log_files"),
        }})
    return metadata_parsed

def load_session_hdf5(session_fullfname):
    try:
        session_file = h5py.File(session_fullfname, 'r')
    except OSError:
        Logger().logger.spacer()
        Logger().logger.error(f"Session file {session_fullfname} not found")
        session_file = None
    return session_file

def get_session_metadata(from_nas=None, from_db=None):
    if from_nas is not None:
        metadata = _session_modality_from_nas(*from_nas, "metadata")
    elif from_db is not None:
        metadata = _session_modality_from_db(*from_db, "metadata")
    else:
        raise ValueError("Either from_nas or from_db must be provided")
    
    metadata = _parse_metadata(metadata)
    if metadata["paradigm_id"] == 800:
        P0800_pillar_details = sT.metadata_complement_P0800(metadata["env_metadata"])
        metadata["P0800_pillar_details"] = P0800_pillar_details
    return metadata
    
def get_session_modality(modality, from_nas=None, from_db=None, modality_kwargs={}, **kwargs):
    if from_nas is not None:
        data = _session_modality_from_nas(*from_nas, modality, **modality_kwargs)
    elif from_db is not None:
        data = _session_modality_from_db(*from_db, modality)
    else:
        raise ValueError("Either from_nas or from_db must be provided")

    if data is None:
        Logger().logger.error(f"Failed to load {modality} data")
        return None
    
    if kwargs.get("na2null"):
        data = sT.data_modality_na2null(data)
    
    if kwargs.get("to_deltaT_from_session_start"):
        data = sT.data_modality_deltaT_from_session_start(data)
    
    if kwargs.get("pct_as_index"):
        data = sT.data_modality_pct_as_index(data)
    
    if kwargs.get("us2s"):
        data = sT.data_modality_us2s(data)
    
    if modality == 'event' and (event_name := kwargs.get("event_subset")):
        data = data.loc[data['event_name'] == event_name]
                
    if kwargs.get("complement_data"):
        if modality == "unity_trial":
            Logger().logger.debug(f"Complementing {modality} modality with paradigmVariable_data and unity_frame")
            if from_nas is not None:
                trials_variable = _session_modality_from_nas(*from_nas, "paradigm_variable")
                unity_frames = _session_modality_from_nas(*from_nas, "unity_frame")
                metadata =  get_session_metadata(from_nas)
            elif from_db is not None:
                trials_variable = _session_modality_from_db(*from_db, "paradigm_variable")
                unity_frames = _session_modality_from_nas(*from_db, "unity_frame")
                metadata = get_session_metadata(from_db)
            
            trials_variable.drop(columns=['trial_id'], inplace=True) # double
            data = pd.concat([data, trials_variable], axis=1)
            
            if metadata['paradigm_id'] == 800:
                staytimes = sT.calc_staytimes(data, unity_frames, metadata)
                data = pd.concat([data, staytimes], axis=1)
            else:
                Logger().logger.info(f"Staytimes not calculated for paradigm {metadata['paradigm_id']}")

        if modality == 'unity_frame':
            vel = sT.calc_unity_velocity(data)
            data = pd.concat([data, vel], axis=1)
            
    if kwargs.get("rename2oldkeys"):
        data = sT.data_modality_rename2oldkeys(data, modality,)
    
    Logger().logger.debug(f"Returning {modality} modality, {data.shape}, first row:\n{data.iloc[0]}")
    return data