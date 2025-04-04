import os
import cv2

import h5py
import pandas as pd
import numpy as np
from CustomLogger import CustomLogger as Logger

import analytics_processing.modality_transformations as mT
from analytics_processing import metadata_loading
import analytics_processing.analytics_constants as C
        
def session_modality_from_nas(session_fullfname, key, where=None, start=None, 
                               stop=None, columns=None):
    # special key for MEA1K h5 file
    if key == 'ephys_traces':
        data = _handle_ephys_from_nas(session_fullfname, start, stop, columns)
        
    # special key for camera frames
    elif key.endswith("cam_frames"):
        session_file = _load_session_hdf5(session_fullfname)
        data = _load_cam_frames(session_file, key, columns)
        session_file.close()
    
    else:
        data = _pandas_based_from_nas(session_fullfname, key, where, start, 
                                      stop, columns)
    
        # metadata is specical in that it's slightly `postprocessed` here already
        if key == 'metadata':
            session_name = os.path.basename(session_fullfname).replace(".hdf5", "")
            if data is not None:
                data = metadata_loading.extract_metadata(data, session_name, session_fullfname)
            else:
                # metadata is the only key that will never return None
                data = metadata_loading.minimal_metad_from_session_name(session_name)
        if key == 'paradigm_variable':
            # this table has many naming issues, fix them
            data = mT.fix_missing_paradigm_variable_names(data)
    return data

def _handle_ephys_from_nas(session_fullfname, start, stop, columns):
    L = Logger()
    session_name = os.path.basename(session_fullfname).replace(".hdf5", "")
    session_dir = os.path.dirname(session_fullfname)
    
    ephys_fname = [f for f in os.listdir(session_dir) 
                        if f.startswith(session_name) and f.endswith("ephys_traces.dat")]
    if len(ephys_fname) == 1:
        ephys_map_fname = ephys_fname[0].replace(".dat", "_mapping.csv")
        mapping = pd.read_csv(os.path.join(session_dir,ephys_map_fname), index_col=0)
        data = np.memmap(os.path.join(session_dir,ephys_fname[0]), dtype=np.int16, 
                         mode='r').reshape(len(mapping), -1, order='F')
        data = data[slice(start, stop)]
        if columns is not None:
            # load to memory if columns are specified (timeslice)
            data = np.array(data[:, slice(*columns)])
        return data, mapping.iloc[slice(start, stop), :]
    L.logger.info(f"No de-compressed ephys traces found for {session_name}")
    return None, None
            
def _pandas_based_from_nas(session_fullfname, key, where=None, start=None,
                            stop=None, columns=None):
    L = Logger()
    try:
        with pd.HDFStore(session_fullfname, mode='r') as store:
            if store.get_storer(key).is_table:
                args = {keystr: key for keystr, key in zip(("start", "stop", "where", "columns"),
                                                        (start, stop, where, columns)) 
                        if key is not None}
                argsmsg = "with "+L.fmtmsg(args) if args else ""
                L.logger.debug(f"Accessing {key} from table {argsmsg}")

                # check if columns are valid keys, skip otw
                if columns is not None:
                    columns = list(columns) # if tuple is passed
                    valid_cols = store.select(key, start=0, stop=0).columns
                    for i in range(len(columns)):
                        if columns[i] not in valid_cols:
                            L.logger.warning(f"Column `{columns[i]}` not found in {key}, skipping")
                            columns.pop(i)
                
                data = store.select(key, start=start, stop=stop, where=where, 
                                    columns=columns)
            # very old data (non-patched) may not be in table format, can't read with start stop
            else:
                # L.logger.error(f"Data not in table format; deprecated format")
                # raise KeyError(f"Data not in table format")
                
                L.logger.debug(f"Accessing {key}...")
                if columns is not None:
                    L.logger.warning(f"Data not in table format, `columns` will be ignored")
                data = store.select(key)
            L.logger.debug(f"Got data with shape ({data.shape[0]:,}, {data.shape[1]:,})")
        # data = pd.read_hdf(session_fullfname, key=key, mode='r', start=start, 
        #                    stop=stop, where=where, columns=columns)
        mid_iloc = data.shape[0] // 2
        L.logger.debug(f"Successfully accessed {key}")
        L.logger.debug(f"Data:\n{data}")
        L.logger.debug(f"Rows {mid_iloc}, {mid_iloc}+1:\n{data.iloc[mid_iloc:mid_iloc+2].T}")
    except KeyError:
        L.logger.error(f"Key {key} not found in session data")
        data = None
    except FileNotFoundError:
        L.logger.error(f"Session file not found:\n{session_fullfname}")
        data = None 
    return data
    
def get_modality_summary(session_fullfname):
    L = Logger()
    aggr = []
    for modality_key in C.MODALITY_KEYS:
        modality_summary = {}
        L.logger.debug(f"Creating summary of {modality_key} modality...")
        data = session_modality_from_nas(session_fullfname, modality_key)
                    
        if modality_key == "metadata":
            # to pandas 
            data = pd.Series(data).to_frame().T
        
        elif data is None or data is (None,None):
            L.logger.warning(f"Modality {modality_key} missing.")
            continue
        
        elif modality_key == 'ephys_traces':
            data, mapping = data # unpack tuple
            modality_summary['ephys_traces_n_columns'] = data.shape[1]
            modality_summary['ephys_traces_n_rows'] = data.shape[0]
            modality_summary['ephys_traces_columns'] = np.nan
            # modality_summary['ephys_traces_n_connected'] = np.nan
            aggr.append(pd.Series(modality_summary, name=modality_key))
            continue
            
        elif modality_key.endswith("cam_packages"):
            session_file = _load_session_hdf5(session_fullfname)
            raw_images_key = modality_key.replace("_packages", "_frames")
            L.logger.debug(f"Accessing `{raw_images_key}` keys to check for missing frames...")
            all_frame_keys = list(session_file[raw_images_key].keys())
            session_file.close()
            all_frame_keys = np.array([int(k.replace("frame_","")) for k in all_frame_keys])
            # assume that gaps are exactly 1 frame
            missing_frame_keys = all_frame_keys[:-1][np.diff(all_frame_keys) != 1]+1
            modality_summary['missing_frame_keys'] = missing_frame_keys.tolist()
            modality_summary['no_missing_frame_keys'] = missing_frame_keys.size == 0
        
        # get general shape of data
        modality_summary['n_columns'] = data.shape[1]
        modality_summary['n_rows'] = data.shape[0]
        
        # check for missing values regarding ephys timestamps, patching
        col_basename = modality_key
        if modality_key == "unity_frame":
            col_basename = "frame"
        elif modality_key == "unity_trial":
            col_basename = "trial_start"
        elif modality_key.endswith("cam_packages"):
            col_basename = modality_key.replace("_packages", "_image")
        
        ephys_t_col = f"{col_basename}_ephys_timestamp"
        et_patched_col = f"{col_basename}_ephys_patched"
        if ephys_t_col in data.columns:
            modality_summary['n_nan_ephys_timestamp'] = data[ephys_t_col].isna().sum()
            modality_summary['no_nan_ephys_timestamps'] = data[ephys_t_col].isna().sum() == 0
            if et_patched_col in data.columns:
                modality_summary['n_patched_ephys_timestamp'] = data[et_patched_col].isna().sum()
                modality_summary['no_patched_ephys_timestamps'] = data[et_patched_col].isna().sum() == 0
        
        modality_summary = pd.Series(modality_summary, name=modality_key)
        modality_summary.index = modality_key+"_"+modality_summary.index
        L.logger.debug(f"Summary of {modality_key}:\n{modality_summary}")
        aggr.append(modality_summary)
    aggr = pd.concat(aggr, axis=0)
    L.logger.debug(f"Summary of all modalities:\n{aggr}")
    return aggr

def _load_session_hdf5(session_fullfname):
    try:
        session_file = h5py.File(session_fullfname, 'r')
    except OSError:
        Logger().logger.spacer()
        Logger().logger.error(f"Session file {session_fullfname} not found")
        session_file = None
    return session_file

def _load_cam_frames(session_file, key, columns):
    L = Logger()
    frames_stack = []
    if columns is None:
        L.logger.debug(f"Loading entire sequence of {key}...")
        columns = list(session_file[key].keys())
        
    L.logger.debug(f"Loading {len(columns):,} frames...")
    for frame_key in columns:
        jpg_frame = np.array(session_file[key][frame_key][()])
        frame = cv2.imdecode(np.frombuffer(jpg_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
        frames_stack.append(frame)
    return np.stack(frames_stack)