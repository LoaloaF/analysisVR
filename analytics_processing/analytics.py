from datetime import datetime
import json
from time import sleep
import sys
import os
import argparse
from collections import OrderedDict

# when executed as a process add parent project dir to path
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# sys.path.insert(1, os.path.join(sys.path[0], '..', '..', 'ephysVR'))

import pandas as pd
import numpy as np

import analytics_processing.analytics_constants as C

from CustomLogger import CustomLogger as Logger
# from analytics_processing.modality_loading import get_complemented_session_modality
from analysis_utils import device_paths
# from analytics_processing.agg_modalities2analytic import transform_to_position_bin_index

import analytics_processing.agg_modalities2analytic as m2a
import analytics_processing.integr_analytics as integr_analytics

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
    L = Logger()
    L.logger.debug("Searching NAS for applicable sesseions...")
    
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
                    L.logger.warning(f"Expected 1 session file for {session_dir}, found "
                          f"{len(session_fname)}, {session_fname} in "
                          f"{os.path.join(parad_animal_subdir, session_dir)}")
                    continue
                session_fname = session_fname[0]
                
                fullfname = os.path.join(parad_animal_subdir, session_dir, session_fname)
                sessionlist_fullfnames.append(fullfname)
                identifier.append((p_id, animal_id, s_id))
    
    unique_animals = np.unique([i[1] for i in identifier])            
    L.logger.debug(f"For paradigms {paradigm_ids}, animals {unique_animals}, "
                   f"found {len(sessionlist_fullfnames)} sessions.")
    return sessionlist_fullfnames, identifier

def _get_analytics_fname(session_dir, analysis_name):
    full_path = os.path.join(session_dir, "session_analytics")
    if not os.path.exists(full_path):
        print("Creating analytics directory for session ", os.path.basename(session_dir))
        os.makedirs(full_path)
    fullfname = os.path.join(full_path, analysis_name+".parquet")
    return fullfname

# def _compute_analytic(analytic, session_fullfname):
#     if analytic == "metadata":
#         data = get_complemented_session_modality(session_fullfname, "metadata", dict2pandas=True)
        
#     if analytic == "behavior_event":
#         data = get_complemented_session_modality(session_fullfname, "event",
#                                                  complement_data=True)
#         print(data)
#         exit()
#         data = data.reindex(columns=C.BEHAVIOR_EVENT_TABLE.keys())
#         data = data.astype(C.BEHAVIOR_EVENT_TABLE)
    
#     elif analytic == "unity_framewise":
#         data = get_complemented_session_modality(session_fullfname, "unity_frame",
#                                     position_bin_index=True, complement_data=True)
#         data = data.reindex(columns=C.UNITY_FAMEWISE_TABLE.keys())
#         data = data.astype(C.UNITY_FAMEWISE_TABLE)
    
#     elif analytic == "unity_trialwise":
#         data = get_complemented_session_modality(session_fullfname, "unity_trial",
#                                     position_bin_index=True, complement_data=True)
#         # TODO doesn't convert poroperly NaN
#         # data = data.reindex(columns=C.UNITY_TRIALWISE_TABLE.keys())
#         # data = data.astype(C.UNITY_TRIALWISE_TABLE)

#     elif analytic == "unity_trackwise":
#         # data = get_analytic(os.path.dirname(session_fullfname), "unity_framewise")
#         data = get_analytics(analytic="unity_framewise", sessionlist_fullfnames=[session_fullfname])
#         data.reset_index(inplace=True, drop=True)
#         data = transform_to_position_bin_index(data)
#         data = data.reindex(columns=C.UNITY_TRACKWISE_TABLE.keys())
#         data = data.astype(C.UNITY_TRACKWISE_TABLE)
#     return data

def _compute_analytic(analytic, session_fullfname):
    print(f"Computing {analytic} for {os.path.basename(session_fullfname)}")
    if analytic == "SessionMetadata":
        data = m2a.get_SesssionMetadata(session_fullfname)
        data_table = C.SESSION_METADATA_TABLE
    
    elif analytic == "Portenta":
        data = m2a.get_Portenta(session_fullfname)
        data_table = C.BEHAVIOR_EVENT_TABLE
    
    elif analytic == "BehaviorEvents":
        data = m2a.get_BehaviorEvents(session_fullfname)
        data_table = C.BEHAVIOR_EVENT_TABLE
        
    elif analytic == "FacecamPoses":
        data = m2a.get_FacecamPoses(session_fullfname)
        int_columns = ['facecam_image_pc_timestamp', 'facecam_image_ephys_timestamp']
        data_table = OrderedDict((col, pd.Int64Dtype() if col in int_columns 
                                  else pd.Float32Dtype()) for col in data.columns)

    elif analytic == "UnityFramewise":
        data = m2a.get_UnityFramewise(session_fullfname)
        behavior_event_data = m2a.get_BehaviorEvents(session_fullfname)
        pose_data = m2a.get_FacecamPoses(session_fullfname)

        # integrate the behavior events and facecam poses
        data = integr_analytics.merge_behavior_events_with_frames(data, behavior_event_data)
        data = integr_analytics.merge_facecam_poses_with_frames(data, pose_data)

        data_table = C.UNITY_FAMEWISE_TABLE
        # update the data_table with the facecam pose columns
        for col in data.columns:
            if col.startswith("facecam_pose_"):
                data_table[col] = pd.Float32Dtype()

    elif analytic == "UnityTrackwise":
        unity_framewise = get_analytics(analytic="UnityFramewise", 
                                        sessionlist_fullfnames=[session_fullfname])
        data = integr_analytics.get_UnityTrackwise(unity_framewise)
        
        data_table = C.UNITY_TRACKWISE_TABLE
        # update the data_table with the facecam pose columns
        for col in data.columns:
            if col.startswith("facecam_pose_"):
                data_table[col] = pd.Float32Dtype()
    
    elif analytic == "UnityTrialwiseMetrics":
        data = m2a.get_UnityTrialwiseMetrics(session_fullfname)
        print(data['trial_outcome'])
        print()
        print()
        print()
        print()
        # exit()
        # unity_framewise = get_analytics(analytic="UnityFramewise", 
        #                                 sessionlist_fullfnames=[session_fullfname])
        # data = integr_analytics.get_UnityTrialwiseMetrics(unity_framewise)
        # data_table = C.UNITY_TRIALWISE_METRICS_TABLE
    
    elif analytic == "Spikes":
        data = m2a.get_Spikes(session_fullfname)
        data_table = C.SPIKE_TABLE

    #TODO fix later
    if analytic != "UnityTrialwiseMetrics":
        data = data.reindex(columns=data_table.keys())
        data = data.astype(data_table)        
    return data

def _extract_id_from_sessionname(session_name):
    session_name_split = session_name.split("_")
    anim_name, parad_name = session_name_split[2], session_name_split[3]
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
    L.logger.debug(f"Requested analytics: {analytic}, mode: {mode}, "
                   f"Paradigm_ids: {paradigm_ids}, animal_ids: {animal_ids}, "
                   f"session_ids: {session_ids}, from_date: {from_date}, "
                   f"to_date: {to_date}\n\t"
                   f"Processing {len(sessionlist_fullfnames)} sessions\n")

    aggr = []
    for session_fullfname, identif in zip(sessionlist_fullfnames, ids):
        L.logger.debug(f"Processing {identif} {os.path.basename(session_fullfname)}"
                       f"\n{os.path.dirname(session_fullfname)}")
        analytics_fname = _get_analytics_fname(os.path.dirname(session_fullfname),
                                               analysis_name=analytic)
        
        if mode.endswith('compute'):
            if os.path.exists(analytics_fname) and mode != "recompute":
                L.logger.info(f"Output exists, skipping.")
                continue
            data = _compute_analytic(analytic, session_fullfname)
            data.to_parquet(analytics_fname, index=False, engine='pyarrow')
        
        elif mode == "set":
            if not os.path.exists(analytics_fname):
                L.logger.info(f"Analytic `{analytic}` not does not exist for"
                              f" {identif}, compute first.")
                continue
            data = pd.read_parquet(analytics_fname, columns=columns)
            midx = [(*identif, i) for i in range(data.shape[0])]
            names = ["paradigm_id", "animal_id", "session_id", "entry_id"]
            data.index = pd.MultiIndex.from_tuples(midx, names=names)
            aggr.append(data)
            
        elif mode == "available":
            if os.path.exists(analytics_fname):
                aggr.append(identif)
        
        elif mode == 'clear':
            if os.path.exists(analytics_fname):
                L.logger.warning(f"Permantly DELETING {analytics_fname}in 3s !")
                sleep(3)
                os.remove(analytics_fname)
            else:
                L.logger.warning(f"File {analytics_fname} does not exist, skipping.")
        
        else:
            raise ValueError(f"Unknown mode {mode}")
    
    if mode == "set":
        aggr = pd.concat(aggr)
        
        session_ids = aggr.index.get_level_values("session_id").tolist()
        paradigm_ids = aggr.index.unique("paradigm_id").tolist()
        animal_ids = aggr.index.unique("animal_id").tolist()
        mid_iloc = aggr.shape[0] // 2
        L.spacer("debug")
        L.logger.info(f"Returning {analytic} for {len(session_ids)} sessions.")
        L.logger.debug(f"Paradigm_ids: {paradigm_ids}, Animal_ids: {animal_ids}"
                       f"\n{aggr}\n{aggr.iloc[mid_iloc:mid_iloc+1].T}")
        return aggr
    elif mode == "available":
        return np.array(aggr)
            
            
def main():
    argParser = argparse.ArgumentParser("Run pipeline to calculate analytics")
    argParser.add_argument("analytic", help="which analytic to compute", type=str)
    argParser.add_argument("--paradigm_ids", nargs='+', default=None, type=int)
    argParser.add_argument("--animal_ids", nargs='+', default=None, type=int)
    argParser.add_argument("--sessionlist_fullfnames", nargs='+', default=None)
    argParser.add_argument("--recompute", action="store_true", default=False)
    argParser.add_argument("--from_date", default=None)
    argParser.add_argument("--to_date", default=None)
    argParser.add_argument("--logging_level", default="DEBUG")
    kwargs = vars(argParser.parse_args())
    
    L = Logger()
    Logger().init_logger(None, None, kwargs.pop("logging_level"))
    L.logger.info(f"Running pipeline for analytic `{kwargs['analytic']}`")
    L.logger.debug(L.fmtmsg(kwargs))
    L.spacer()
    
    if not kwargs.pop("recompute"):
        kwargs['mode'] = 'set'
    else:
        kwargs['mode'] = 'recompute'
    get_analytics(**kwargs)

if __name__ == '__main__':
    main()