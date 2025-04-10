import os
from time import sleep
from collections import OrderedDict

import pandas as pd
import numpy as np

from CustomLogger import CustomLogger as Logger

import analytics_processing.analytics_constants as C
import analytics_processing.agg_modalities2analytic as m2a
import analytics_processing.integr_analytics as integr_analytics
import analytics_processing.sessions_from_nas_parsing as sp

def _get_analytics_fname(session_dir, analysis_name):
    full_path = os.path.join(session_dir, "session_analytics")
    if not os.path.exists(full_path):
        print("Creating analytics directory for session ", os.path.basename(session_dir))
        os.makedirs(full_path)
    fullfname = os.path.join(full_path, analysis_name+".parquet")
    return fullfname

def _compute_analytic(analytic, session_fullfname):
    print(f"Computing {analytic} for {os.path.basename(session_fullfname)}")
    if analytic == "SessionMetadata":
        data = m2a.get_SesssionMetadata(session_fullfname)
        data_table = C.SESSION_METADATA_TABLE
    
    elif analytic == "Portenta":
        data = m2a.get_Portenta(session_fullfname)
        data_table = C.BEHAVIOR_EVENT_TABLE
    
    elif analytic == "Unity":
        data = m2a.get_Unity(session_fullfname)
        data_table = C.UNITY_TABLE
    
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
    
    elif analytic == "MultiUnits":
        data = m2a.get_MultiUnits(session_fullfname)
        data_table = C.MULTI_UNITS_TABLE
        
    elif analytic == "Spikes":
        data = m2a.get_Spikes(session_fullfname)
        data_table = C.MULTI_UNITS_TABLE

    #TODO fix later
    if analytic != "UnityTrialwiseMetrics" and data is not None:
        data = data.reindex(columns=data_table.keys())
        data = data.astype(data_table)        
    return data

# def _extract_id_from_sessionname(session_name):
#     session_name_split = session_name.split("_")
#     anim_name, parad_name = session_name_split[2], session_name_split[3]
#     return int(anim_name[-3:]), int(parad_name[1:]), 0

def get_analytics(analytic, mode="set", paradigm_ids=None, animal_ids=None, 
                  session_ids=None, session_names=None, 
                  from_date=None, to_date=None, columns=None,):
    L = Logger()
    
    sessionlist_fullfnames, ids = sp.sessionlist_fullfnames_from_args(paradigm_ids, animal_ids, 
                                                                      session_ids, session_names, 
                                                                      from_date, to_date)
    
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
            if data is not None:
                data.to_parquet(analytics_fname, index=False, engine='pyarrow')
            else:
                L.logger.warning(f"Failed to compute {analytic} for {identif}")
            L.spacer("debug")
            
        
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
        aggr = np.array(aggr)
        L.logger.debug(f"Returning {analytic}:\n{aggr}")
        return aggr