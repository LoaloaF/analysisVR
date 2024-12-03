from datetime import datetime
import sys
import os
import argparse

# when executed as a process add parent project dir to path
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pandas as pd
import numpy as np

import analytics_processing.analytics_constants as C

from CustomLogger import CustomLogger as Logger
from analytics_processing.modality_loading import get_complemented_session_modality
from analysis_utils import device_paths
from analytics_processing.agg_modalities2analytic import transform_to_position_bin_index

import analytics_processing.agg_modalities2analytic as m2a

def _parse_paradigms_from_nas(nas_dir):
    # get list of something like "RUN_rYL001", in nas bas dir
    run_animal_names = [f for f in os.listdir(nas_dir) 
                        if f.startswith("RUN_") and str.isdigit(f[-3:])]
    paradigms = []
    for r_animal_name in run_animal_names:
        # extract the P part from something like "rYL001_P0100"
        paradigms.extend([int(f[-4:]) for f in os.listdir(os.path.join(nas_dir, r_animal_name)) 
                          if str.isdigit(f[-4:])])
    return sorted(list(set(paradigms)))
    
def _parse_paradigm_animals_from_nas(paradigm_id, nas_dir):
    # get list of something like "RUN_rYL001", in nas bas dir
    run_animal_names = [f for f in os.listdir(nas_dir) 
                        if f.startswith("RUN_") and str.isdigit(f[-3:])]
    run_animal_names.sort()
    # filter out animals that did not do the passed paradigm
    filtered_animals = [r_animal_name for r_animal_name in run_animal_names if 
                        f"{r_animal_name[4:]}_P{paradigm_id:04}" # eg rYL001_P0100
                        in os.listdir(os.path.join(nas_dir, r_animal_name))]
    return sorted([int(f[-3:]) for f in filtered_animals])

def _get_sessionlist_fullfnames(paradigm_ids, animal_ids, session_ids=None,
                                from_date=None, to_date=None):
    nas_dir, _, _ = device_paths()
    sessionlist_fullfnames = []
    from_date = datetime.strptime(from_date, "%Y-%m-%d") if from_date is not None else None
    to_date = datetime.strptime(to_date, "%Y-%m-%d") if to_date is not None else None

    identifier = []
    # get all paradigms if not specified
    paradigm_ids = paradigm_ids if paradigm_ids is not None else _parse_paradigms_from_nas(nas_dir)
    for p_id in paradigm_ids:
        # get all animals if not specified
        animal_ids_ = animal_ids if animal_ids is not None else _parse_paradigm_animals_from_nas(p_id, nas_dir)
        for animal_id in animal_ids_:
            # get all sessions for the paradigm+animal combo
            parad_animal_subdir = os.path.join(nas_dir, f"RUN_rYL{animal_id:03}", 
                                               f"rYL{animal_id:03}_P{p_id:04d}")
            if not os.path.exists(parad_animal_subdir):
                continue # not every combination exists
            
            # get all the session dirs in the animal+paradigm subdir, should end with min
            parad_animal_session_dirs = [sd for sd in os.listdir(parad_animal_subdir) 
                                         if sd.endswith("min")]
            for s_id, session_dir in enumerate(sorted(parad_animal_session_dirs)):
                date = datetime.strptime(session_dir[:10], "%Y-%m-%d")
                if from_date is not None and date < from_date:
                    continue
                if to_date is not None and date > to_date:
                    continue
                if session_ids is not None and s_id not in session_ids:
                    continue
                
                # get the session behavior data h5 file
                session_fname = [fname for fname in os.listdir(os.path.join(parad_animal_subdir, session_dir))
                                 if fname.endswith("min.hdf5")]
                if len(session_fname) != 1:
                    print(f"Expected 1 session file for {session_dir}, found "
                          f"{len(session_fname)}, {session_fname}")
                    continue
                session_fname = session_fname[0]
                
                fullfname = os.path.join(parad_animal_subdir, session_dir, session_fname)
                sessionlist_fullfnames.append(fullfname)
                identifier.append((p_id, animal_id, s_id))
                
    # print(f"For paradigms {paradigm_ids}, animals {animal_ids}, found\n{np.array(identifier)}")
    return sessionlist_fullfnames, identifier

def _get_analytics_fname(session_dir, analysis_name):
    full_path = os.path.join(session_dir, "session_analytics")
    if not os.path.exists(full_path):
        print("Creating analytics directory for session ", os.path.basename(session_dir))
        os.makedirs(full_path)
    fullfname = os.path.join(full_path, analysis_name+".parquet")
    return fullfname

def _compute_analytic(analytic, session_fullfname):
    if analytic == "metadata":
        data = get_complemented_session_modality(session_fullfname, "metadata", dict2pandas=True)
        
    if analytic == "behavior_event":
        data = get_complemented_session_modality(session_fullfname, "event",
                                                 complement_data=True)
        print(data)
        exit()
        data = data.reindex(columns=C.BEHAVIOR_EVENT_TABLE.keys())
        data = data.astype(C.BEHAVIOR_EVENT_TABLE)
    
    elif analytic == "unity_framewise":
        data = get_complemented_session_modality(session_fullfname, "unity_frame",
                                    position_bin_index=True, complement_data=True)
        data = data.reindex(columns=C.UNITY_FAMEWISE_TABLE.keys())
        data = data.astype(C.UNITY_FAMEWISE_TABLE)
    
    elif analytic == "unity_trialwise":
        data = get_complemented_session_modality(session_fullfname, "unity_trial",
                                    position_bin_index=True, complement_data=True)
        # TODO doesn't convert poroperly NaN
        # data = data.reindex(columns=C.UNITY_TRIALWISE_TABLE.keys())
        # data = data.astype(C.UNITY_TRIALWISE_TABLE)

    elif analytic == "unity_trackwise":
        # data = get_analytic(os.path.dirname(session_fullfname), "unity_framewise")
        data = get_analytics(analytic="unity_framewise", sessionlist_fullfnames=[session_fullfname])
        data.reset_index(inplace=True, drop=True)
        data = transform_to_position_bin_index(data)
        data = data.reindex(columns=C.UNITY_TRACKWISE_TABLE.keys())
        data = data.astype(C.UNITY_TRACKWISE_TABLE)
    return data

def _extract_id_from_sessionname(session_name):
    _, _, anim_name, parad_name, _, _ = session_name.split("_")
    return int(anim_name[-3:]), int(parad_name[1:]), 0

def get_analytics(analytic, mode="set", paradigm_ids=None, animal_ids=None, 
                  session_ids=None, sessionlist_fullfnames=None, 
                  from_date=None, to_date=None, columns=None,):
    L = Logger()
    
    if sessionlist_fullfnames is None:
        sessionlist_fullfnames, ids = _get_sessionlist_fullfnames(paradigm_ids, 
                                                                animal_ids, session_ids,
                                                                from_date, to_date)
    else:
        ids = [_extract_id_from_sessionname(os.path.basename(s))
               for s in sessionlist_fullfnames]

    aggr = []
    for session_fullfname, identif in zip(sessionlist_fullfnames, ids):
        analytics_fname = _get_analytics_fname(os.path.dirname(session_fullfname),
                                               analysis_name=analytic)
        
        if mode.endswith('compute'):
            if os.path.exists(analytics_fname) and mode != "recompute":
                print(f"Output exists, skipping.")
                continue
            data = _compute_analytic(analytic, session_fullfname)
            data.to_parquet(analytics_fname, index=False)
        
        elif mode == "set":
            if not os.path.exists(analytics_fname):
                print(f"Analytic not does not exist for {identif}, compute first.")
                continue
            data = pd.read_parquet(analytics_fname, columns=columns)
            midx = [(*identif, i) for i in range(data.shape[0])]
            names = ["paradigm_id", "animal_id", "session_id", "entry_id"]
            data.index = pd.MultiIndex.from_tuples(midx, names=names)
            aggr.append(data)
            
        elif mode == "available":
            print(analytics_fname)
            if os.path.exists(analytics_fname):
                aggr.append(identif)
        
        else:
            raise ValueError(f"Unknown mode {mode}")
    
    if mode == "set":
        return pd.concat(aggr)
    elif mode == "available":
        return np.array(aggr)
            
            
        

        
    


















# def get_session_analytic(analytic, paradigm_id, animal_id, session_id, columns=None):
#     nas_dir, _, _ = device_paths()
#     subdir = os.path.join(nas_dir, f"RUN_rYL{animal_id:03}", f"rYL{animal_id:03}_P{paradigm_id:04d}")
#     session_dir = [sd for sd in os.listdir(subdir) if sd.endswith("min")][session_id]
#     full_path = os.path.join(subdir, session_dir, "session_analytics")
#     full_fname = os.path.join(full_path, analytic+".parquet")
#     if not os.path.exists(full_fname):
#         raise FileNotFoundError(f"Analytics file {full_fname} not found. "
#                                 f"Generate first with lower level pipeline")
#     return pd.read_parquet(full_fname, columns=columns)
    
# def get_available_analytics(analysis_name):
#     nas_dir, _, _ = device_paths()
#     sessionlist_fullfnames, ids = _get_sessionlist_fullfnames(nas_dir, 
#                                                               C.ALL_PARADIGM_IDS, 
#                                                               C.ALL_ANIMAL_IDS)
#     # exit()
#     analytics_set = []
#     for session_fullfname, s_identifier in zip(sessionlist_fullfnames, ids):
#         print(s_identifier, session_fullfname)
#         analytics_fname = _get_analytics_fname(os.path.dirname(session_fullfname),
#                                                analysis_name=analysis_name)
#         # print((s_identifier, analytics_fname), end="\n")
#         if os.path.exists(analytics_fname):
#             # print(s_identifier)
#             analytics_set.append(s_identifier)
#     return np.array(analytics_set)

# def get_analytic(session_dir, analysis_name):
#     fullfname = _get_analytics_fname(session_dir, analysis_name)
#     if not os.path.exists(fullfname):
#         raise FileNotFoundError(f"Analytics file {fullfname} not found. "
#                                 f"Generate first with lower level pipeline")
#     return pd.read_parquet(fullfname)
    
# def run_pipeline(which_pipeline, paradigm_ids, animal_ids, sessionlist_fullfnames, 
#                  recompute, from_date, to_date, nas_dir, ):
#     L = Logger()
    
#     # get the NAS mount point if not passed as an argument
#     if nas_dir is None:
#         nas_dir, _, _ = device_paths()
#     if sessionlist_fullfnames is None:
#         sessionlist_fullfnames, _ = _get_sessionlist_fullfnames(nas_dir, paradigm_ids, 
#                                                                 animal_ids, from_date, 
#                                                                 to_date)
    
#     for session_fullfname in sessionlist_fullfnames:
#         analytics_fname = _get_analytics_fname(os.path.dirname(session_fullfname),
#                                                analysis_name=which_pipeline)
#         # print(os.path.basename(session_fullfname))
        
#         if os.path.exists(analytics_fname) and not recompute:
#             print(f"Output exists, skipping.")
#             continue
        
#         if which_pipeline == "metadata":
#             data = get_complemented_session_modality(session_fullfname, "metadata", dict2pandas=True)
        
#         elif which_pipeline == "unity_framewise":
#             data = get_complemented_session_modality(session_fullfname, "unity_frame",
#                                         to_deltaT_from_session_start=True,
#                                         position_bin_index=True, complement_data=True)
#             data = data.reindex(columns=C.UNITY_FAMEWISE_TABLE.keys())
#             data = data.astype(C.UNITY_FAMEWISE_TABLE)

#         elif which_pipeline == "unity_trackwise":
#             data = get_analytic(os.path.dirname(session_fullfname), "unity_framewise")
#             data = transform_to_position_bin_index(data)
#             data = data.reindex(columns=C.UNITY_TRACKWISE_TABLE.keys())
#             data = data.astype(C.UNITY_TRACKWISE_TABLE)

#         data.to_parquet(analytics_fname, index=False)
        
    
    
    
# def get_session_analytic(analytic, paradigm_id, animal_id, session_id, columns=None):
#     nas_dir, _, _ = device_paths()
#     subdir = os.path.join(nas_dir, f"RUN_rYL{animal_id:03}", f"rYL{animal_id:03}_P{paradigm_id:04d}")
#     session_dir = [sd for sd in os.listdir(subdir) if sd.endswith("min")][session_id]
#     full_path = os.path.join(subdir, session_dir, "session_analytics")
#     full_fname = os.path.join(full_path, analytic+".parquet")
#     if not os.path.exists(full_fname):
#         raise FileNotFoundError(f"Analytics file {full_fname} not found. "
#                                 f"Generate first with lower level pipeline")
#     return pd.read_parquet(full_fname, columns=columns)
    
    
    
    
    
    
# def get_analytics_set(paradigm_ids, animal_ids, analysis_name, columns=None,
#                       from_date=None, to_date=None, nas_dir=None, **kwargs):
#     L = Logger()
    
#     # get the NAS mount point if not passed as an argument
#     if nas_dir is None:
#         nas_dir, _, _ = device_paths()
#         sessionlist_fullfnames, ids = _get_sessionlist_fullfnames(nas_dir, paradigm_ids, 
#                                                                   animal_ids, from_date, 
#                                                                   to_date)
#     # print(sessionlist_fullfnames)
    
#     analytics_set = []
#     for session_fullfname, s_identifier in zip(sessionlist_fullfnames, ids):
#         analytics_fname = _get_analytics_fname(os.path.dirname(session_fullfname),
#                                                analysis_name=analysis_name)
#         # print(analytics_fname)
        
#         if not os.path.exists(analytics_fname):
#             # print(f"Output does not exist, skipping.")
#             continue
        
#         data = pd.read_parquet(analytics_fname, columns=columns)
#         midx = [(*s_identifier, i) for i in range(data.shape[0])]
#         data.index = pd.MultiIndex.from_tuples(midx, names=["paradigm_id", "animal_id", "session_id", "entry_id"])
#         analytics_set.append(data)
#     data = pd.concat(analytics_set)
#     # print("data.index")
#     # print(data)
#     return data
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    argParser = argparse.ArgumentParser("Parse sessions from NAS and save denormalized, integrated data")
    argParser.add_argument("--which_pipeline")
    argParser.add_argument("--paradigm_ids", nargs='+', default=None, type=int)
    argParser.add_argument("--animal_ids", nargs='+', default=None, type=int)
    argParser.add_argument("--sessionlist_fullfnames", nargs='+', default=None)
    argParser.add_argument("--recompute", action="store_true", default=False)
    argParser.add_argument("--from_date", default=None)
    argParser.add_argument("--to_date", default=None)
    argParser.add_argument("--logging_level", default="INFO")
    argParser.add_argument("--nas_dir", default=None)
    kwargs = vars(argParser.parse_args())
    
    L = Logger()
    Logger().init_logger(None, None, kwargs.pop("logging_level"))
    L.logger.info("Running pipeline")
    L.logger.info(L.fmtmsg(kwargs))
    L.spacer()
    
    # get_analytics(**kwargs)
    d = get_analytics("unity_framewise", mode="set", animal_ids=[1], 
                      paradigm_ids=[800])
    # d = get_analytics("behavior_event", mode="compute", animal_ids=[1], 
    #                   paradigm_ids=[800])
    print(d)
    print(d.columns)