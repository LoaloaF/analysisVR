import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../../CoreRatVR'))


import h5py
import pandas as pd

from CustomLogger import CustomLogger as Logger

import sessionTransformations as sT

def _make_session_filename(nas_base_dir, pardigm_subdir, session_name):
    return os.path.join(nas_base_dir, pardigm_subdir, session_name+".hdf5")

def _load_cam_frames(session_file, key):
    # if session_file is None: return None
    # try:
    #     data = session_file[key]
    # except KeyError:
    #     Logger().logger.error(f"Key {key} not found in session data")
    #     return None
    #TODO implement this
    raise NotImplementedError("Loading camera frames from session data is not implemented yet")

def _session_modality_from_nas(nas_base_dir, pardigm_subdir, session_name, key):
    L = Logger()
    session_fullfname = _make_session_filename(nas_base_dir, pardigm_subdir, session_name)
    
    # special key for camera frames    
    if key.startswith("cam") and key.endswith("frames"):
        session_file = load_session_hdf5(session_fullfname)
        return _load_cam_frames(session_file, key)
        
    # pandas based data
    try:
        data = pd.read_hdf(session_fullfname, key=key)
        L.logger.debug(f"Successfully accessed {key} from session data:\n{data}")
    except KeyError:
        L.logger.error(f"Key {key} not found in session data")
        data = None
    except OSError:
        L.logger.error(f"Error reading {key} from session data")
        data = None
    return data
        
def _session_modality_from_db(db_fullfname, key):
    #TODO implement this
    raise NotImplementedError("Loading session data from database is not implemented yet")








def load_session_hdf5(session_fullfname):
    try:
        session_file = h5py.File(session_fullfname, 'r')
    except OSError:
        Logger().logger.spacer()
        Logger().logger.error(f"Session file {session_fullfname} not found")
        session_file = None
    return session_file
    
def get_session_modality(from_nas=None, from_db=None, modality=None, **kwargs):
    if from_nas is not None:
        data = _session_modality_from_nas(*from_nas, modality)
    elif from_db is not None:
        data = _session_modality_from_db(*from_db, modality)
    else:
        raise ValueError("Either from_nas or from_db must be provided")
    
    if kwargs.get("na2null"):
        data = sT.data_modality_na2null(data)
    
    if kwargs.get("to_deltaT_from_session_start"):
        data = sT.data_modality_deltaT_from_session_start(data)
    
    if kwargs.get("us2s"):
        data = sT.data_modality_us2s(data)
      
    if kwargs.get("pct_as_index"):
        data = sT.data_modality_pct_as_index(data)
    
    if kwargs.get("rename2oldkeys"):
        data = sT.data_modality_rename2oldkeys(data, modality)
        
    if modality == "unity_trial" and kwargs.get("complement_trial_data"):
        if from_nas is not None:
            trials_variable = _session_modality_from_nas(*from_nas, "paradigm_variable")
        elif from_db is not None:
            trials_variable = _session_modality_from_db(*from_db, "paradigm_variable")
        data = pd.concat([data, trials_variable], axis=1)
                    
    return data